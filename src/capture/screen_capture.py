"""
高帧率屏幕截图模块 — 使用 mss 进行低延迟屏幕采集。

mss 比 Pillow 快约 10x，适合 30~60 FPS 的实时游戏画面捕获。
输出格式为 numpy ndarray (BGR)，可直接供 YOLO / OpenCV 使用。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import mss
import numpy as np

from src.utils.logger import get_logger

log = get_logger("capture")


class ScreenCapture:
    """
    高性能屏幕截图器。

    用法:
        cap = ScreenCapture(region=(0, 0, 1366, 768), target_fps=30)
        cap.start()
        frame = cap.grab()       # numpy BGR array
        cap.stop()
    """

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,
        target_fps: int = 30,
    ):
        """
        Args:
            region: 截图区域 (x, y, width, height)。None 表示全屏。
            target_fps: 目标帧率，用于限制截图频率。
        """
        self.region = region
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._sct: Optional[mss.mss] = None
        self._monitor: dict = {}
        self._last_grab_time: float = 0.0
        self._frame_count: int = 0

    def start(self) -> None:
        """初始化截图上下文。"""
        self._sct = mss.mss()
        if self.region:
            x, y, w, h = self.region
            self._monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            # 主显示器
            self._monitor = self._sct.monitors[1]
        self._frame_count = 0
        log.info(
            f"ScreenCapture 已启动 | 区域={self._monitor} | 目标FPS={self.target_fps}"
        )

    def stop(self) -> None:
        """释放截图上下文。"""
        if self._sct:
            self._sct.close()
            self._sct = None
        log.info(f"ScreenCapture 已停止 | 总帧数={self._frame_count}")

    def grab(self) -> np.ndarray:
        """
        截取一帧画面。

        Returns:
            BGR 格式的 numpy 数组，形状为 (height, width, 3)。
        """
        if self._sct is None:
            raise RuntimeError("ScreenCapture 未启动，请先调用 start()")

        # 帧率限制
        now = time.perf_counter()
        elapsed = now - self._last_grab_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        screenshot = self._sct.grab(self._monitor)
        # mss 返回 BGRA，转为 BGR
        frame = np.array(screenshot)[:, :, :3]
        frame = np.ascontiguousarray(frame)

        self._last_grab_time = time.perf_counter()
        self._frame_count += 1
        return frame

    def grab_region(self, region: tuple[int, int, int, int]) -> np.ndarray:
        """
        截取指定子区域。用于对聊天窗、状态栏等固定区域做 OCR/OpenCV。

        Args:
            region: (x, y, width, height)
        Returns:
            BGR numpy 数组。
        """
        if self._sct is None:
            raise RuntimeError("ScreenCapture 未启动，请先调用 start()")

        x, y, w, h = region
        monitor = {"left": x, "top": y, "width": w, "height": h}
        screenshot = self._sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]
        return np.ascontiguousarray(frame)

    @property
    def actual_fps(self) -> float:
        """返回当前实际帧率（近似）。"""
        if self._last_grab_time == 0:
            return 0.0
        return self._frame_count / max(
            time.perf_counter() - (self._last_grab_time - self._frame_count * self._frame_interval),
            0.001,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
