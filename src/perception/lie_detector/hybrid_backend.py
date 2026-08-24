"""
lie_detector.hybrid_backend — 组合后端 (白块 + 自适应背景 + 时序差分 + Kalman + 可选 SOT)

融合公式 (用户拍板): S = 0.5·S_SOT + 0.3·S_背景残差 + 0.2·S_运动
  - SOT (UETrack) 未就绪时, 白块 confidence 顶替 SOT 位 (0.5)。
  - Kalman 只做**先验加权 + 输出平滑**, 不做硬搜索窗 — 目标实测会瞬移
    (01.mp4 帧间跳 262~824px), 硬收窗会漏掉, 搜索范围用弹窗/全帧。
  - 倒计时最清晰时刻 (峰值后下降) 才 init SOT 模板, 不在渐隐后 init。

实现同 OpenCVBackend 契约: detect(frame) -> LieDetectResult,
供 LieDetectorModel 直接换用, combat_brain / mouse_tracker 零改动。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

from .adaptive_bg import AdaptiveBackgroundModel, ResidualResult
from .kalman import TinyKalman
from .opencv_backend import OpenCVBackend
from .state import LieBackend, LieDetectResult, LiePhase

log = get_logger("lie_detector.hybrid")


class HybridConfig:
    """hybrid 后端参数 (从 bot config 的 lie_detector.hybrid 读)。"""

    def __init__(self, d: Optional[dict] = None):
        d = d or {}
        self.bg_warmup_frames: int = int(d.get("bg_warmup_frames", 10))
        self.bg_alpha: float = float(d.get("bg_alpha", 0.05))
        self.residual_thresh: int = int(d.get("residual_thresh", 20))
        self.search_scale: float = float(d.get("search_scale", 12))   # Kalman 先验 σ 参考
        self.kalman_enabled: bool = bool(d.get("kalman_enabled", True))
        fw = d.get("fusion_weights", [0.5, 0.3, 0.2])                 # SOT / 背景 / 运动
        self.fusion_weights: Tuple[float, float, float] = (
            float(fw[0]), float(fw[1]), float(fw[2]))
        self.min_conf: float = float(d.get("min_conf", 0.30))
        self.template_min_conf: float = float(d.get("template_min_conf", 0.5))
        # 候选远离 Kalman 预测 → 判定瞬移: reset 到观测直接输出, 不靠常速外推
        self.teleport_dist: float = float(d.get("teleport_dist", 120))
        # 锚点 re-init: SOT 中心与 blob 分歧超过此值且 blob 强检测 → 重灌 SOT 模板
        self.sot_reinit_dist: float = float(d.get("sot_reinit_dist", 60))
        # UETrack 隔 N 帧跑一次 (默认 1 = 每帧; >1 省 CPU, 间隔帧用 blob/残差顶 SOT 位)
        self.sot_track_every_n: int = int(d.get("sot_track_every_n", 1))
        # UETrack init 失败后的重试冷却 (避免 13s 级构建反复重试)
        self.sot_init_retry_frames: int = int(d.get("sot_init_retry_frames", 30))


class HybridBackend:
    """组合后端: 白块激活 + 自适应背景 + 时序差分 + Kalman 平滑 + 可选 SOT。

    每帧 detect(frame) 输出融合后的 LieDetectResult。stateful:
    bg 模型 / kalman / SOT 模板 / 帧历史 跨帧保持。
    """

    def __init__(
        self,
        detector_repo_path: str | Path,
        config: Optional[dict] = None,
        uetrack: object | None = None,
    ):
        """
        Args:
            detector_repo_path: lie-detector 项目路径 (交给白块后端)
            config: HybridConfig 用键 (bg_warmup_frames 等)
            uetrack: 可选 SOT 后端 (UETrackBackend), 须实现
                     ready / init_template(frame,bbox)->bool / track(frame)->(center,score)
        """
        self._cfg = HybridConfig(config)
        self._opencv = OpenCVBackend(detector_repo_path)
        self._bg = AdaptiveBackgroundModel(
            alpha=self._cfg.bg_alpha,
            warmup_frames=self._cfg.bg_warmup_frames,
            residual_thresh=self._cfg.residual_thresh,
        )
        self._kalman = TinyKalman()
        # SOT 后端: 优先外部注入 (测试用 mock); 否则按 config 的 hybrid.uetrack 构建
        if uetrack is not None:
            self._uetrack = uetrack
        else:
            self._uetrack = self._build_uetrack_from_config(config)
        self._sot_ready = bool(self._uetrack is not None and getattr(self._uetrack, "ready", False))
        self._sot_inited = False
        self._last_sot_init_attempt = -10**9   # init 失败重试冷却
        self._last_sot_track_frame = -1        # track_every_n 隔帧

        self._gray_prev2: Optional[np.ndarray] = None  # 时序差分历史 (t-2)
        self._last_res_bbox: Optional[Tuple[int, int, int, int]] = None
        # 最近一次有框结果的 bbox (尺寸参考): SOT/运动赢且无源 bbox 时, 用它在融合中心合成框
        self._last_box: Optional[Tuple[int, int, int, int]] = None
        self._frame_count = 0

        # SOT 模板 init 状态机: 峰值跟踪 → 峰值后下降 ≥2 帧 → 用最清晰帧+bbox init
        self._peak_conf = 0.0
        self._peak_bbox: Optional[Tuple[int, int, int, int]] = None
        self._peak_img: Optional[np.ndarray] = None
        self._peak_frame = -1
        self._reinit_allowed = True

        log.info(f"[hybrid] 就绪: warmup={self._cfg.bg_warmup_frames} "
                 f"alpha={self._cfg.bg_alpha} thresh={self._cfg.residual_thresh} "
                 f"kalman={self._cfg.kalman_enabled} fusion={self._cfg.fusion_weights} "
                 f"sot={'在' if self._sot_ready else '无(白块顶位)'}")

    def _build_uetrack_from_config(self, config: Optional[dict]) -> object | None:
        """按 config.lie_detector.hybrid.uetrack 构建 UETrackBackend (enabled 才建)。"""
        uetrack_cfg = (config or {}).get("uetrack")
        if not uetrack_cfg or not uetrack_cfg.get("enabled", False):
            return None
        try:
            from .uetrack_backend import UETrackBackend
            backend = UETrackBackend(
                repo_path=uetrack_cfg.get("repo_path", ""),
                ckpt_path=uetrack_cfg.get("ckpt_path", ""),
            )
            if not backend.ready:
                log.warning(f"[hybrid] UETrack 不可用: {backend.import_error}")
                return None
            return backend
        except Exception as e:
            log.warning(f"[hybrid] UETrack 构造失败: {e}")
            return None

    # ── 公共 ──

    @property
    def ready(self) -> bool:
        return self._opencv.ready

    @property
    def import_error(self):
        return self._opencv.import_error

    @property
    def sot_ready(self) -> bool:
        return self._sot_ready

    @property
    def sot_inited(self) -> bool:
        return self._sot_inited

    def warm(self) -> bool:
        """启动预热 UETrack (13s 一次性构建)。没有 SOT 后端时是 no-op。"""
        if self._sot_ready and hasattr(self._uetrack, "warm"):
            return bool(self._uetrack.warm())
        return False

    def reset(self) -> None:
        """清空全部跨帧状态 (测谎退出/重新进入时调用)。"""
        self._bg.reset()
        self._kalman = TinyKalman()
        self._sot_inited = False
        self._last_sot_init_attempt = -10**9
        self._last_sot_track_frame = -1
        self._reinit_allowed = True
        self._gray_prev2 = None
        self._last_res_bbox = None
        self._last_box = None
        self._peak_conf = 0.0
        self._peak_bbox = None
        self._peak_img = None
        self._peak_frame = -1
        # 不复用不卸模型: UETrack 构建 13s, 每弹窗重建会错过下一次。reset 只清模板状态
        # (_sot_inited=False + _reinit_allowed=True → 下个弹窗首次 init 重灌模板), 模型保持已构建。
        # 彻底释放走 stop()。
        log.info("[hybrid] 状态已重置 (SOT 模型保留已构建)")

    # ── 主入口 ──

    def detect(self, frame: np.ndarray) -> LieDetectResult:
        """每帧: 白块 → 背景残差 → 时序差分 → (SOT) → 融合 → Kalman → 结果。"""
        self._frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) 白块 (激活 + 亮期目标 + 模板源)
        blob = self._opencv.detect(frame)
        freeze_bbox = self._pick_freeze_bbox(blob)

        # 2) 背景残差 (冻结区 = 目标当前所在, 永不吸收进背景)
        res = self._bg.update_and_detect(gray, freeze_bbox)
        self._last_res_bbox = res.bbox if res is not None else None

        # 3) 时序差分 (瞬移确认; 停顿帧近零, 只做补充)
        motion = self._detect_motion(gray)

        # 4) SOT (已 init 才有; 模板 init 状态机在下面)
        sot = None
        if self._sot_ready and self._sot_inited:
            if self._frame_count - self._last_sot_track_frame >= self._cfg.sot_track_every_n:
                sot = self._uetrack.track(frame)   # (center, score) | None
                self._last_sot_track_frame = self._frame_count
                # 锚点 re-init 只看"本帧真跑过"的 SOT; 隔帧跳过帧 sot 保持 None (白块顶位)
                sot = self._sot_anchor_guard(frame, blob, sot)

        # 5) 候选 + 融合 (Kalman 预测只算一次, _fuse 打分与输出平滑共用)
        pred = None
        if self._cfg.kalman_enabled and self._kalman.initialized:
            pred = self._kalman.predict()
        fused = self._fuse(blob, res, motion, sot, pred)

        # 6) 模板 init 状态机 (峰值后下降 → 用最清晰帧 init, 不在渐隐后 init)
        if self._sot_ready and not self._sot_inited:
            self._maybe_init_sot(blob, frame)

        if fused is None:
            # 全部候选不足 → 不激活 (但白块 active 本身算激活, 交给 model 去抖)
            if blob.active:
                return blob
            return LieDetectResult(active=False, backend=LieBackend.OPENCV)

        center, conf, bbox, phase = fused

        # 6b) 赢家无 bbox (SOT/运动) 但确认跟踪 → 用最近已知框尺寸在融合中心合成框。
        #     (目标渐隐时白块缺席, SOT 仍持跟踪; 合成框让用户能看到跟踪位置)
        if bbox is None and conf >= self._cfg.min_conf and self._last_box is not None:
            w = self._last_box[2] - self._last_box[0]
            h = self._last_box[3] - self._last_box[1]
            cx, cy = center
            bbox = (max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
                    min(frame.shape[1], int(cx + w / 2)), min(frame.shape[0], int(cy + h / 2)))

        # 7) Kalman: 瞬移帧 reset 直接输出观测 (常速外推会偏), 停顿帧平滑
        out_center = center
        if self._cfg.kalman_enabled:
            if pred is not None:
                dist = math.hypot(center[0] - pred[0], center[1] - pred[1])
            else:
                dist = 0.0
            if dist > self._cfg.teleport_dist:
                # 目标瞬移: 重新起手在观测处, velocity 归零, 输出观测
                self._kalman.reset(center[0], center[1])
            else:
                out_center = self._kalman.correct(center[0], center[1], conf)
                out_center = (int(round(out_center[0])), int(round(out_center[1])))

        # 8) 亮度 (目标 ROI 平均)
        brightness = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            brightness = float(roi.mean()) if roi.size else 0.0

        if bbox is not None:
            self._last_box = bbox   # 记住最近框尺寸, 供渐隐期合成框用

        return LieDetectResult(
            # 激活 = 白块层命中 (opencv 基线语义: 有 bbox 即 active, 渐隐期置信低也跟)
            #        或 融合置信达标 (blob 缺席时残差/运动兜底)
            active=(blob.active or conf >= self._cfg.min_conf),
            phase=phase,
            target_center=out_center,
            target_bbox=bbox,
            confidence=conf,
            brightness=brightness,
            backend=LieBackend.OPENCV,
        )

    # ── 各层 ──

    def _pick_freeze_bbox(self, blob: LieDetectResult) -> Optional[Tuple[int, int, int, int]]:
        """冻结区: 亮期用白块 bbox; 渐隐期白块缩了用上帧残差 bbox; 都没有 → 全帧更新。"""
        if blob.active and blob.target_bbox is not None:
            return blob.target_bbox
        return self._last_res_bbox

    def _detect_motion(self, gray: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """|gray_t - gray_{t-2}| → 最大分量中心 + 置信 (瞬移确认)。"""
        if self._gray_prev2 is None:
            self._gray_prev2 = gray
            return None
        diff = cv2.absdiff(gray, self._gray_prev2)
        self._gray_prev2 = gray
        _, binary = cv2.threshold(diff, self._cfg.residual_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        H, W = gray.shape
        best = None
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 200 or area > 0.05 * H * W:
                continue
            if max(w, h) / max(1, min(w, h)) > 1.5:
                continue
            if min(w, h) < 15:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
        if best is None:
            return None
        x, y, w, h, area = best
        cx, cy = (2 * x + w) // 2, (2 * y + h) // 2
        conf = min(1.0, float(diff[y:y + h, x:x + w].mean()) / 100.0)
        return ((cx, cy), conf)

    # ── 融合 ──

    def _fuse(
        self,
        blob: LieDetectResult,
        res: Optional[ResidualResult],
        motion: Optional[Tuple[Tuple[int, int], float]],
        sot: Optional[Tuple[Tuple[int, int], float]],
        pred: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[int, int], float, Optional[Tuple[int, int, int, int]], LiePhase]]:
        """加权选最佳候选 (SOT/背景/运动; SOT 缺席时白块顶位)。

        Kalman 先验: 靠近预测点的候选加权 0.6~1.0 倍 (软先验, 不硬排除瞬移)。
        pred 由 detect() 算好传入 (同一帧只 predict 一次)。
        返回 (center, fused_conf, bbox, phase) 或 None。
        """
        # 源 → (center, conf, bbox, weight); phase 取主导源
        srcs = []
        phase = LiePhase.TRACKING
        if sot is not None and self._sot_inited:
            c, s = sot
            srcs.append((c, s, None, self._cfg.fusion_weights[0]))
            phase = LiePhase.TRACKING
        elif blob.active:
            c = blob.target_center or (0, 0)
            srcs.append((c, blob.confidence, blob.target_bbox, self._cfg.fusion_weights[0]))
            phase = blob.phase
        if res is not None:
            srcs.append((res.center, res.confidence, res.bbox, self._cfg.fusion_weights[1]))
        if motion is not None:
            srcs.append((motion[0], motion[1], None, self._cfg.fusion_weights[2]))

        if not srcs:
            return None

        # Kalman 先验 σ: 弹窗范围级别 (瞬移可达 800px, 不收死)
        sigma = 500.0 * self._cfg.search_scale / 12.0
        px, py = (pred if pred is not None else (None, None))

        best = None
        total_w = 0.0
        fused_conf = 0.0
        for c, conf, bbox, w in srcs:
            total_w += w
            fused_conf += w * conf
            score = w * conf
            if px is not None:
                dist = math.hypot(c[0] - px, c[1] - py)
                prior = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
                score *= 0.6 + 0.4 * prior
            if best is None or score > best[0]:
                best = (score, c, bbox)

        fused_conf = fused_conf / total_w if total_w > 0 else 0.0
        return (best[1], fused_conf, best[2], phase)

    # ── SOT 模板 init 状态机 ──

    def _maybe_init_sot(self, blob: LieDetectResult, frame: np.ndarray) -> None:
        """峰值跟踪 (首次 init): 倒计时最清晰 (conf 峰值后下降 ≥2 帧) → 用峰值帧+bbox init。

        只在 SOT 从未 init 时运行; 之后的锚点 re-init 交给 _sot_anchor_guard。
        """
        if not self._reinit_allowed:
            return
        if self._frame_count - self._last_sot_init_attempt < self._cfg.sot_init_retry_frames:
            return   # init 失败冷却中 (13s 级构建不反复重试)
        if blob.active and blob.confidence >= self._peak_conf:
            self._peak_conf = blob.confidence
            self._peak_bbox = blob.target_bbox
            self._peak_img = frame.copy()   # 快照最清晰帧 (目标可能随后瞬移走)
            self._peak_frame = self._frame_count
        elif (self._peak_conf >= self._cfg.template_min_conf
              and self._peak_bbox is not None
              and self._peak_img is not None
              and self._frame_count - self._peak_frame >= 2):
            ok = self._uetrack.init_template(self._peak_img, self._peak_bbox)
            self._sot_inited = ok
            if ok:
                self._reinit_allowed = False   # 首次 init 成功 → 后续全走锚点 re-init
                log.info(f"[hybrid] SOT 模板已 init (conf={self._peak_conf:.2f} "
                         f"bbox={self._peak_bbox})")
            else:
                # 失败 → 冷却后才重试 (13s 级构建不反复卡); 成功不占用冷却
                self._last_sot_init_attempt = self._frame_count
                log.warning(f"[hybrid] SOT 模板 init 失败 (conf={self._peak_conf:.2f}); "
                            f"冷却 {self._cfg.sot_init_retry_frames} 帧重试")

    def _sot_anchor_guard(self, frame, blob, sot) -> Optional[Tuple[Tuple[int, int], float]]:
        """锚点 re-init: blob 权威 (conf ≥ template_min_conf) 且 SOT 与其分歧 → 重灌模板。

        实测 (01.mp4): 目标每瞬移+变外观, SOT 卡死在旧锚点 (分歧常 >60px)。
        blob 高置信处重灌模板 → SOT 立刻对准 blob → 停顿帧误差 ~10px。
        渐隐期 blob 弱 → 不动 SOT (SOT 是渐隐期主心骨, 靠外观记忆撑住)。
        """
        if not (self._sot_ready and self._sot_inited):
            return sot   # 首次 init 走 _maybe_init_sot
        blob_confident = blob.active and blob.confidence >= self._cfg.template_min_conf
        if not blob_confident:
            return sot   # 渐隐/被遮挡: 保持 SOT 原样
        bc = blob.target_center
        if bc is None:
            return sot
        divergent = (sot is not None
                     and math.hypot(sot[0][0] - bc[0], sot[0][1] - bc[1])
                     > self._cfg.sot_reinit_dist)
        lost = sot is None   # inited 但本帧没跟出结果 → 同样重灌
        if not (divergent or lost):
            return sot
        if self._frame_count - self._last_sot_init_attempt < self._cfg.sot_init_retry_frames:
            return sot   # 上次 re-init 失败, 冷却中
        ok = self._uetrack.init_template(frame, blob.target_bbox)
        self._sot_inited = ok
        if ok:
            self._last_sot_track_frame = self._frame_count
            log.info(f"[hybrid] SOT 锚点 re-init @ f{self._frame_count} (blob={blob.target_bbox})")
            return (bc, blob.confidence)   # re-init 后 SOT 项 = blob 位置
        # 失败 → 冷却后才重试 (13s 级构建不反复卡)
        self._last_sot_init_attempt = self._frame_count
        log.warning(f"[hybrid] SOT 锚点 re-init 失败 @ f{self._frame_count}")
        return sot

    def stop(self) -> None:
        if self._uetrack is not None and hasattr(self._uetrack, "stop"):
            self._uetrack.stop()
