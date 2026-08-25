"""
lie_detector.opencv_backend — OpenCV 后端 (CPU)

直接复用本地 lie-detector/scripts/ 项目的检测函数 (已稳定, 不复制代码):
- detect_lie_detector_window (auto_bbox.py / multiframe_bbox.py / white_silhouette_detector.py)
- detect_white_target (auto_bbox.py, 多阈值 180/200/220/240)
- 模板匹配兜底 (white_silhouette_detector.track_white_silhouette)

每次 update(frame) 走完整流程: 窗口检测 → 窗口内多阈值白块 → 全帧回退。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

from .state import (
    LieBackend, LieDetectResult, LiePhase,
    compute_confidence, inflate_bbox,
)

log = get_logger("lie_detector.opencv")


# 多阈值检测: 命中几个阈值 / 4 = confidence
THRESHOLD_LEVELS = (180, 200, 220, 240)


class OpenCVBackend:
    """OpenCV 后端: 纯 CPU, 每帧独立检测 (无需 video 序列上下文)。

    适配 bot 实时场景: 视觉线程每帧拿到最新帧 → detect_white_target → 输出 bbox+center。
    """

    def __init__(self, detector_repo_path: str | Path):
        self._repo_path = Path(detector_repo_path)
        self._detect_window = None
        self._detect_target = None
        self._track_template = None
        self._imported = False
        self._import_error: Optional[Exception] = None
        self._try_import()

    def _try_import(self) -> None:
        """注入 lie-detector/scripts 到 sys.path, 导入检测函数。

        失败不抛错 (bot 启动时不应崩) — 由调用方通过 .ready 检查。
        """
        scripts_dir = self._repo_path / "scripts"
        if not scripts_dir.is_dir():
            self._import_error = FileNotFoundError(f"lie-detector scripts not found: {scripts_dir}")
            log.warning(f"[opencv] {self._import_error}")
            return
        sys.path.insert(0, str(scripts_dir))
        try:
            # 优先 auto_bbox (有窗口检测 + 多阈值目标检测)
            from auto_bbox import detect_lie_detector_window, detect_white_target  # type: ignore
            self._detect_window = detect_lie_detector_window
            self._detect_target = detect_white_target
            # 兜底: 模板匹配 (形状无关跨帧跟踪)
            try:
                from white_silhouette_detector import track_white_silhouette  # type: ignore
                self._track_template = track_white_silhouette
            except Exception:
                pass  # 模板匹配非必需
            self._imported = True
            log.info(f"[opencv] 检测函数已加载 (scripts={scripts_dir})")
        except Exception as e:
            self._import_error = e
            log.warning(f"[opencv] 导入失败: {e}")

    @property
    def ready(self) -> bool:
        return self._imported

    @property
    def import_error(self) -> Optional[Exception]:
        return self._import_error

    def detect(self, frame: np.ndarray, scale: float = 1.0) -> LieDetectResult:
        """每帧调用: 输出 LieDetectResult (active=False 表示未触发, 决策层只看 active=True 的帧)。

        Args:
            frame: 待检测帧 (BGR)。
            scale: <1.0 表示 frame 已是全分辨率帧的缩小版 (服务端降采样检测提速)。
                   绝对阈值 (面积/最小尺寸) 同步按 scale 缩放, 返回的 bbox/center 缩回全分辨率坐标。
        Returns:
            LieDetectResult(active=False) — 未触发 / 导入失败
            LieDetectResult(active=True, phase=..., target_center=(cx,cy), target_bbox=(x1,y1,x2,y2),
                            confidence=0.0~1.0, brightness=0~255, backend=OPENCV)
        """
        if not self._imported:
            return LieDetectResult(active=False)

        H, W = frame.shape[:2]

        # 1. 找测谎仪窗口 (沙色矩形) — 失败回退全帧
        window_bbox = None
        try:
            window_bbox = self._detect_window(frame)
        except Exception as e:
            log.debug(f"[opencv] window detect failed: {e}")

        # 2. 在窗口内 (或全帧) 找白色目标 — 多阈值天然给 confidence
        bbox = None
        matched = 0
        try:
            # 自己跑多阈值, 计数得到 confidence; detect_white_target 内部已用同样四个阈值
            # 但它只返回最大的 bbox, 不返回每个阈值命中数。这里复刻一遍拿命中数。
            bbox, matched = self._detect_with_confidence(frame, window_bbox, scale)
        except Exception as e:
            log.debug(f"[opencv] target detect failed: {e}")

        if bbox is None:
            return LieDetectResult(active=False, backend=LieBackend.OPENCV)

        # 3. bbox 膨胀 (亮核 → 完整星形); 膨胀比乘性, 降采样空间膨胀后缩回等价全分辨率膨胀
        x1, y1, x2, y2 = inflate_bbox(bbox, ratio=1.6)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # 4. 亮度 (目标 ROI 平均; mean 亮度尺度不变, 降采样 ROI 足够)
        roi = frame[y1:y2, x1:x2]
        brightness = float(roi.mean()) if roi.size else 0.0

        # 5. 中心 (降采样空间)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 5b. 缩回全分辨率坐标 (scale<1 时放大 1/scale)
        if scale != 1.0 and scale > 0:
            inv = 1.0 / scale
            x1, y1 = int(round(x1 * inv)), int(round(y1 * inv))
            x2, y2 = int(round(x2 * inv)), int(round(y2 * inv))
            cx, cy = int(round(cx * inv)), int(round(cy * inv))

        # 6. phase: 简化判定 — 目标稳定 = COUNTDOWN; bbox 大小变化 = TRACKING
        # 这里给个保守值, 让调用方根据帧间位移再精修
        phase = LiePhase.COUNTDOWN if matched >= 2 else LiePhase.TRACKING

        return LieDetectResult(
            active=True,
            phase=phase,
            target_center=(cx, cy),
            target_bbox=(x1, y1, x2, y2),
            confidence=compute_confidence(matched, len(THRESHOLD_LEVELS)),
            brightness=brightness,
            backend=LieBackend.OPENCV,
        )

    def _detect_with_confidence(
        self, frame: np.ndarray, window_bbox: Optional[Tuple[int, int, int, int]],
        scale: float = 1.0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
        """复刻 detect_white_target 的多阈值逻辑, 同时返回命中数 (=confidence)。

        scale < 1.0: 帧已降采样, 绝对阈值 (面积≥200, 最小边≥15) 同步缩放, 保住降采样后的小目标。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if window_bbox is not None:
            x1, y1, x2, y2 = window_bbox
            roi = gray[y1:y2, x1:x2]
            roi_offset = (x1, y1)
        else:
            roi = gray
            roi_offset = (0, 0)

        H_roi, W_roi = roi.shape[:2]
        s = max(scale, 1e-3)
        min_area = int(200 * s * s)      # 全分辨率面积阈 200px² → 按 scale² 缩放
        min_dim = max(1, int(15 * s))    # 全分辨率最小边 15px → 按 scale 缩放
        best_per_thresh = {}  # thresh → max_area_bbox
        for thresh in THRESHOLD_LEVELS:
            _, bright = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area < min_area or area > 0.05 * H_roi * W_roi:
                    continue
                if max(w, h) / max(1, min(w, h)) > 1.5:
                    continue
                if min(w, h) < min_dim:
                    continue
                ax1, ay1 = x + roi_offset[0], y + roi_offset[1]
                if thresh not in best_per_thresh or area > best_per_thresh[thresh][4]:
                    best_per_thresh[thresh] = (ax1, ay1, ax1 + w, ay1 + h, area)

        if not best_per_thresh:
            return (None, 0)
        # 跨阈值 IoU 投票: 同一个位置被多少阈值同时命中 = matched
        matched = self._vote_iou(best_per_thresh)
        # 选面积最大的 bbox 作为结果
        best_bbox = max(best_per_thresh.values(), key=lambda b: b[4])[:4]
        return (best_bbox, matched)

    @staticmethod
    def _vote_iou(best_per_thresh: dict) -> int:
        """跨阈值 IoU 投票: 同一个空间被多少阈值命中 = confidence numerator。

        实现: 任取 220 阈值的 bbox 作为参考, 其它阈值 bbox 与它 IoU > 0.3 算命中。
        """
        if 220 not in best_per_thresh:
            # 没有 220 阈值命中 → 直接按阈值命中数 (退化)
            return len(best_per_thresh)
        ref = best_per_thresh[220][:4]
        rx1, ry1, rx2, ry2 = ref
        ref_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        matched = 0
        for thresh, (x1, y1, x2, y2, _) in best_per_thresh.items():
            ix1, iy1 = max(rx1, x1), max(ry1, y1)
            ix2, iy2 = min(rx2, x2), min(ry2, y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = ref_area + (x2 - x1) * (y2 - y1) - inter
            iou = inter / union if union > 0 else 0
            if iou > 0.3:
                matched += 1
        return max(matched, 1)
