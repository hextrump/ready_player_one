"""
lie_detector.adaptive_bg — 自适应背景模型 (背景差分层)

解决"透明/变色/渐隐目标"的检测: 目标叠在背景上时, 只要背景模型学得准,
|当前帧 - 背景| 就有强残差, 与目标当前颜色无关 (目标变红变绿变透明都有残差)。

关键设计 (实证于 01.mp4):
- 背景不是静止的 (画面中部缓慢动画, 采样点灰度 range 110~216) → 不能
  "存一张无目标背景永久减", 必须做**运行式背景模型**: EMA 只在非目标像素
  更新, 跟随背景漂移, 目标永不吸收进背景。
- 目标"瞬移+停顿" → 冻结区域 (freeze_bbox) 内像素不更新: 目标停顿时
  该处背景是旧的 → 残差强 → 中心可出。
- 预热: 前 N 帧用"剔除目标区域后的均值"建初始背景 → 即使目标在预热期
  已出现, 也不会污染背景。

用法 (配合 HybridBackend):
    bg = AdaptiveBackgroundModel(...)
    res = bg.update_and_detect(gray, freeze_bbox=blob_bbox)  # 每帧
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("lie_detector.adaptive_bg")


@dataclass
class ResidualResult:
    """背景残差检测结果 (中心/bbox/置信)。"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    confidence: float                    # [0,1] = 残差强度归一
    area: int
    mean_diff: float                     # 分量内平均残差 [0,255]


class AdaptiveBackgroundModel:
    """运行式背景模型: 预热(中值/均值) + EMA 增量更新 (仅在冻结区外)。

    线程安全: HybridBackend 视觉线程单线程调用, 无需加锁。
    """

    def __init__(
        self,
        alpha: float = 0.05,
        warmup_frames: int = 10,
        residual_thresh: int = 20,
        min_area: int = 200,
        max_area_ratio: float = 0.05,
        max_aspect: float = 1.5,
        min_dim: int = 15,
    ):
        """
        Args:
            alpha: EMA 更新率 (只在冻结区外像素生效; 背景缓慢动画用 0.05 可跟随)
            warmup_frames: 预热帧数 (剔目标区均值建初始背景)
            residual_thresh: 残差二值阈值 (亮目标残差 ~200, 淡目标 ~50)
            min_area / max_area_ratio / max_aspect / min_dim: 连通域过滤 (与白块一致)
        """
        self._alpha = alpha
        self._warmup_frames = max(1, int(warmup_frames))
        self._residual_thresh = residual_thresh
        self._min_area = min_area
        self._max_area_ratio = max_area_ratio
        self._max_aspect = max_aspect
        self._min_dim = min_dim
        self.reset()

    # ── 状态 ──

    def reset(self) -> None:
        """清空背景模型 (测谎退出/重新进入时调用)。"""
        self._bg: Optional[np.ndarray] = None          # uint8 灰度背景
        self._warmup_sum: Optional[np.ndarray] = None  # float32 累加
        self._warmup_count: Optional[np.ndarray] = None  # uint16 有效帧数
        self._warmup_seen: int = 0
        self._shape: Optional[Tuple[int, int]] = None
        log.debug("[adaptive_bg] 已重置")

    @property
    def ready(self) -> bool:
        return self._bg is not None

    # ── 主入口 ──

    def update_and_detect(
        self, gray: np.ndarray, freeze_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[ResidualResult]:
        """每帧调用: 更新背景模型 → 残差检测 → 返回最大分量。

        Args:
            gray: 单通道 uint8 灰度帧
            freeze_bbox: (x1,y1,x2,y2) 目标所在区域 → 该区像素冻结不更新背景
                         (目标停顿时残差仍强)。None = 全帧更新 (重新学背景)。

        Returns:
            ResidualResult | None — 无目标/背景未就绪时 None
        """
        if gray.ndim != 2 or gray.dtype != np.uint8:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.ndim == 3 else gray.astype(np.uint8)

        if self._bg is None:
            # 预热期: 剔除冻结区后累加, 攒够帧数用均值建背景
            self._warmup(gray, freeze_bbox)
            if self._bg is None:
                return None  # 背景还没建好, 由上层靠白块兜底
        else:
            self._update_ema(gray, freeze_bbox)

        return self._detect_residual(gray)

    # ── 内部 ──

    def _warmup(self, gray: np.ndarray, freeze_bbox: Optional[Tuple[int, int, int, int]]) -> None:
        if self._warmup_sum is None:
            self._warmup_sum = np.zeros(gray.shape, np.float32)
            self._warmup_count = np.zeros(gray.shape, np.uint16)
            self._shape = gray.shape

        valid = np.ones(gray.shape, dtype=bool)
        bbox = self._clip_bbox(freeze_bbox)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            valid[y1:y2, x1:x2] = False  # 目标像素不参与背景
        self._warmup_sum[valid] += gray[valid].astype(np.float32)
        self._warmup_count[valid] += 1
        self._warmup_seen += 1

        if self._warmup_seen >= self._warmup_frames:
            denom = np.maximum(self._warmup_count, 1)
            self._bg = (self._warmup_sum / denom).astype(np.uint8)
            self._warmup_sum = None
            self._warmup_count = None
            self._warmup_seen = 0
            log.info(f"[adaptive_bg] 背景已初始化 (warmup={self._warmup_frames}帧, "
                     f"alpha={self._alpha})")

    def _update_ema(self, gray: np.ndarray, freeze_bbox: Optional[Tuple[int, int, int, int]]) -> None:
        """EMA 更新; 冻结区内像素用旧背景覆盖回去 (目标永不吸收)。"""
        updated = cv2.addWeighted(self._bg, 1.0 - self._alpha, gray, self._alpha, 0)
        bbox = self._clip_bbox(freeze_bbox)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            updated[y1:y2, x1:x2] = self._bg[y1:y2, x1:x2]
        self._bg = updated

    def _detect_residual(self, gray: np.ndarray) -> Optional[ResidualResult]:
        if self._bg is None:
            return None
        diff = cv2.absdiff(gray, self._bg)
        _, binary = cv2.threshold(diff, self._residual_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        H, W = gray.shape

        best: Optional[tuple] = None
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < self._min_area or area > self._max_area_ratio * H * W:
                continue
            if max(w, h) / max(1, min(w, h)) > self._max_aspect:
                continue
            if min(w, h) < self._min_dim:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)

        if best is None:
            return None
        x, y, w, h, area = best
        bbox = (x, y, x + w, y + h)
        cx, cy = (2 * x + w) // 2, (2 * y + h) // 2
        mean_diff = float(diff[y:y + h, x:x + w].mean())
        # 残差强度 → 置信: 目标残差 ~200 满置信, 淡目标 ~50 给 0.5
        confidence = min(1.0, mean_diff / 100.0)
        return ResidualResult(
            center=(cx, cy), bbox=bbox,
            confidence=confidence, area=int(area), mean_diff=mean_diff,
        )

    def _clip_bbox(
        self, bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """越界裁剪 + 合法性检查 (None/非法 → None)。"""
        if bbox is None:
            return None
        if self._shape is None:
            return None
        H, W = self._shape
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
