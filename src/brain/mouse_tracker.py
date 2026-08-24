"""
mouse_tracker — 测谎仪激活期间的鼠标跟随线程 (设计文档 §3 + §9)

独立线程: 测谎仪激活期间以自适应 Hz 读最新目标中心, 用 human_mouse.move_to 走拟人轨迹。
解除/超时自动停止。

自适应速度 (§9):
  confidence 高 → 慢 (低 Hz + 小步长, 节流)
  confidence 低 → 快 (高 Hz + 大步长, 跟上渐隐目标)
  EMA 平滑切换, 防止 Hz 在阈值附近抖动。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import win32api
import win32gui

from src.utils.logger import get_logger

from .human_mouse import HumanMouseConfig, HumanMouseController

log = get_logger("mouse_tracker")


@dataclass
class AdaptiveSpeedConfig:
    """§9 自适应速度参数 (config.yaml lie_detector.adaptive_speed.*)。"""
    enabled: bool = True
    hz_low: int = 20
    hz_high: int = 60
    step_low: int = 8
    step_high: int = 24
    conf_threshold: float = 0.5
    ema_alpha: float = 0.3

    @classmethod
    def from_dict(cls, d: dict | None) -> "AdaptiveSpeedConfig":
        if not d:
            return cls()
        c = cls()
        for k in (
            "enabled", "hz_low", "hz_high",
            "step_low", "step_high", "conf_threshold", "ema_alpha",
        ):
            if k in d:
                setattr(c, k, d[k])
        return c


@dataclass
class _TrackerTarget:
    """视觉线程 → 鼠标线程 的共享状态 (原子读, 用 threading.Lock 守护)。"""
    cx: int = 0
    cy: int = 0
    confidence: float = 0.0
    brightness: float = 0.0
    present: bool = False            # 是否有有效目标 (视觉线程这一帧没找到 → False)
    letterbox_scale: float = 1.0     # 最近一次 WindowCapture.last_letterbox[0]
    letterbox_pad_left: int = 0
    letterbox_pad_top: int = 0
    hwnd: int = 0                    # 截图窗口句柄 (ClientToScreen 用)


class MouseTracker:
    """测谎仪鼠标跟随线程。

    启动: MouseTracker(...).start()
    喂入目标: call .update_target(cx, cy, confidence, ...)  (视觉线程每帧调)
    停止:    call .stop() 或等到外部 .active = False

    线程安全: 视觉线程与鼠标线程通过 _TrackerTarget + Lock 通信。
    """

    def __init__(
        self,
        hwnd: int,
        human_cfg: HumanMouseConfig | None = None,
        speed_cfg: AdaptiveSpeedConfig | None = None,
    ):
        self._hwnd = hwnd
        self._human_cfg = human_cfg or HumanMouseConfig()
        self._speed_cfg = speed_cfg or AdaptiveSpeedConfig()
        self._human = HumanMouseController(self._human_cfg)

        self._target = _TrackerTarget(hwnd=hwnd)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 自适应速度 EMA 状态
        self._hz_smoothed: float = float(self._speed_cfg.hz_low)
        self._step_smoothed: float = float(self._speed_cfg.step_low)

        # 上次执行 move_to 时的"当前光标"(避免每次都 GetCursorPos)
        self._last_cursor: Optional[Tuple[int, int]] = None
        self._last_move_t: float = 0.0
        self._active_hits: int = 0      # 累计激活期间的目标更新次数 (诊断)

    # ── 公共接口 ──

    def start(self) -> None:
        """启动跟随线程 (幂等: 已在跑则跳过)。"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True, name="mouse_tracker")
            self._thread.start()
        log.info(f"[mouse_tracker] 启动 (hz={self._speed_cfg.hz_low}~{self._speed_cfg.hz_high} "
                 f"step={self._speed_cfg.step_low}~{self._speed_cfg.step_high})")

    def stop(self) -> None:
        """停止跟随线程。"""
        with self._lock:
            self._running = False
        log.info(f"[mouse_tracker] 停止 (累计目标更新 {self._active_hits} 次)")

    def update_target(
        self,
        cx: int, cy: int,
        confidence: float = 1.0,
        brightness: float = 255.0,
        letterbox_scale: float = 1.0,
        letterbox_pad_left: int = 0,
        letterbox_pad_top: int = 0,
        hwnd: Optional[int] = None,
    ) -> None:
        """视觉线程每帧调: 写入最新目标 + letterbox 参数。

        Args:
            cx, cy: 目标中心 in letterbox 帧坐标
            confidence / brightness: 来自 LieDetectResult
            letterbox_*: 来自 WindowCapture.last_letterbox
            hwnd: 可选覆盖 (e.g. 窗口变化时)
        """
        with self._lock:
            self._target.cx = cx
            self._target.cy = cy
            self._target.confidence = confidence
            self._target.brightness = brightness
            self._target.letterbox_scale = letterbox_scale
            self._target.letterbox_pad_left = letterbox_pad_left
            self._target.letterbox_pad_top = letterbox_pad_top
            self._target.present = True
            self._target.hwnd = hwnd if hwnd is not None else self._target.hwnd
            self._active_hits += 1

    def clear_target(self) -> None:
        """视觉线程检测失败时调: 让鼠标线程不要硬走轨迹。"""
        with self._lock:
            self._target.present = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # ── 主循环 ──

    def _loop(self) -> None:
        """线程主循环: 按当前 Hz 间隔检查目标, 若有则触发 move_to。"""
        while True:
            with self._lock:
                if not self._running:
                    return
                # 计算本帧 Hz / step (按 confidence EMA)
                tgt = self._target
                present = tgt.present
                conf = tgt.confidence
                # 自适应: 在低/高之间线性插值, EMA 平滑
                if self._speed_cfg.enabled:
                    alpha = self._speed_cfg.ema_alpha
                    target_hz = self._interp(conf, self._speed_cfg.hz_low, self._speed_cfg.hz_high)
                    target_step = self._interp(conf, self._speed_cfg.step_low, self._speed_cfg.step_high)
                    self._hz_smoothed = (1 - alpha) * self._hz_smoothed + alpha * target_hz
                    self._step_smoothed = (1 - alpha) * self._step_smoothed + alpha * target_step
                else:
                    self._hz_smoothed = float(self._speed_cfg.hz_low)
                    self._step_smoothed = float(self._speed_cfg.step_low)
                hz = max(1.0, self._hz_smoothed)
                step = max(1.0, self._step_smoothed)

            if present:
                try:
                    self._move_one(tgt, step)
                except Exception as e:
                    log.debug(f"[mouse_tracker] move 异常: {e}")

            # 按 Hz 间隔 sleep
            time.sleep(1.0 / hz)

    def _move_one(self, tgt: _TrackerTarget, max_step: float) -> None:
        """走一次轨迹: 把目标 letterbox 坐标 → 屏幕坐标, 调 human_mouse.move_to。

        max_step: 单帧最大位移上限 (屏幕 px); 当前光标到屏幕目标的距离 > max_step
                  时让 human_mouse 用更长 duration, 避免瞬移。
        """
        # 1. letterbox → client
        if tgt.letterbox_scale <= 0:
            return
        client_x = (tgt.cx - tgt.letterbox_pad_left) / tgt.letterbox_scale
        client_y = (tgt.cy - tgt.letterbox_pad_top) / tgt.letterbox_scale

        # 2. client → screen
        hwnd = tgt.hwnd or self._hwnd
        if not hwnd:
            return
        try:
            client_origin = win32gui.ClientToScreen(hwnd, (0, 0))
        except Exception:
            return
        screen_x = int(round(client_origin[0] + client_x))
        screen_y = int(round(client_origin[1] + client_y))

        # 3. 取当前光标位置 (只在第一次或距离远时 GetCursorPos, 否则用 self._last_cursor)
        if self._last_cursor is None:
            try:
                cur = win32api.GetCursorPos()
            except Exception:
                cur = (screen_x, screen_y)
        else:
            cur = self._last_cursor

        dist = ((screen_x - cur[0]) ** 2 + (screen_y - cur[1]) ** 2) ** 0.5
        # 距离 < 单步 → 不动 (避免抖动)
        if dist < max_step * 0.5:
            return
        # 距离 > 单步 → 截断到 max_step 步长内 (mouse_tracker Hz 不够时不让它一步跨太远)
        # 但 human_mouse 自己的 duration 已经按距离插值, 这里直接传目标点
        # 防止单步跨度太大的保护: 如果本帧 Hz 间隔内 target 已经移走 > 2*max_step,
        # 说明目标快速移动, 此时不截断 (human_mouse duration 已自适应)。
        self._human.move_to(cur, (screen_x, screen_y), duration_ms=None)
        self._last_cursor = (screen_x, screen_y)
        self._last_move_t = time.time()

    @staticmethod
    def _interp(confidence: float, lo: float, hi: float) -> float:
        """confidence 高 → lo (慢), 低 → hi (快)。threshold=0.5 时线性插值。"""
        # 高 confidence (>=1.0) → lo, 低 (0) → hi
        # conf=0.5 → 中点
        t = max(0.0, min(1.0, 1.0 - confidence))   # conf 高 → t=0 → lo
        return lo + t * (hi - lo)
