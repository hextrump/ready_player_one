"""
lie_detector.state — 数据类型 + 去抖逻辑

供 LieDetectorModel / OpenCVBackend / SamuraiBackend 共用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class LiePhase(str, Enum):
    """测谎仪当前阶段。"""
    IDLE = "idle"               # 未触发
    COUNTDOWN = "countdown"     # ~7s 倒计时 (白色图形静止)
    TRACKING = "tracking"       # 鼠标追踪阶段 (目标移动 + 渐隐)


class LieBackend(str, Enum):
    """跟踪后端选择。"""
    OPENCV = "opencv"           # CPU, 多阈值 + 模板跟踪兜底
    SAMURAI = "samurai"         # GPU, OpenCV 初始定位 + SAM2.1 propagate
    HYBRID = "hybrid"           # CPU, 白块 + 自适应背景 + 时序差分 + Kalman + (UETrack SOT)
    REMOTE = "remote"           # 全远程: opencv+samurai 检测都在 hhh 服务端, 本机只发帧收结果


@dataclass
class LieDetectResult:
    """单帧检测结果 (perception/decision 层的统一返回值)。"""
    active: bool                # 是否激活 (经过去抖后的稳定判定)
    phase: LiePhase = LiePhase.IDLE
    target_center: Optional[Tuple[int, int]] = None   # (cx, cy) in letterboxed 帧坐标
    target_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 0.0     # [0, 1] = 命中阈值数 / 总阈值数 (多阈值天然信号)
    brightness: float = 0.0     # 目标 ROI 平均亮度 [0, 255]
    backend: LieBackend = LieBackend.OPENCV


@dataclass
class _DebounceState:
    """去抖状态 (内部用, 不暴露给外部)。"""
    candidate_active: bool = False   # 当前帧是否"看起来激活"
    active: bool = False             # 经过去抖后的稳定激活
    hit_streak: int = 0              # 连续命中帧数
    miss_streak: int = 0             # 连续丢失帧数
    activate_after: int = 2          # 连续 K 帧命中才激活
    deactivate_after: int = 6        # 连续 M 帧丢失才解除
    activated_at: float = 0.0        # 最近一次进入激活态的时间戳


@dataclass
class _LieConfig:
    """去抖 + bbox 膨胀参数 (从 bot config 读)。"""
    activate_after_frames: int = 2
    deactivate_after_frames: int = 6
    bbox_inflate_ratio: float = 1.6  # 亮核 bbox → 完整星形膨胀比 (解决 SAMURAI 初始框过紧的跟丢)
    min_bbox_size: int = 20          # 最小目标尺寸 (像素²), 太小的当作噪声


def inflate_bbox(bbox: Tuple[int, int, int, int], ratio: float = 1.6) -> Tuple[int, int, int, int]:
    """bbox 中心不变, 各边放大 ratio 倍 (解决星形亮核小于整体的问题)。"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * ratio
    h = (y2 - y1) * ratio
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def compute_confidence(matched_thresholds: int, total_thresholds: int = 4) -> float:
    """多阈值检测天然信号: 命中几个阈值 / 总阈值数。"""
    if total_thresholds <= 0:
        return 0.0
    return max(0.0, min(1.0, matched_thresholds / total_thresholds))


def update_debounce(state: _DebounceState, candidate_active: bool, now: float) -> bool:
    """每帧调一次, 返回当前是否稳定激活。

    激活门槛低 (默认 2 帧) → 测谎出现快速响应;
    解除门槛高 (默认 6 帧) → 防止单帧漏检误解除, 避免战斗中频繁切换。
    """
    if candidate_active:
        state.hit_streak += 1
        state.miss_streak = 0
        if not state.active and state.hit_streak >= state.activate_after:
            state.active = True
            state.activated_at = now
    else:
        state.miss_streak += 1
        state.hit_streak = 0
        if state.active and state.miss_streak >= state.deactivate_after:
            state.active = False
            state.activated_at = 0.0
    return state.active


def make_default_config() -> _LieConfig:
    """bot 默认参数 (被 LieDetectorModel(cfg=...) 覆盖)。"""
    return _LieConfig()
