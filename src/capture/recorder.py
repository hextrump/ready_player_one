"""
训练数据录制器 — 用于采集 YOLO 训练图片。

支持两种模式:
1. 手动模式：按快捷键保存当前帧
2. 连续模式：每隔 N 帧自动保存

保存的图片会存入 data/raw/ 目录，按时间戳命名。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from src.capture.screen_capture import ScreenCapture
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("recorder")


class Recorder:
    """
    训练数据录制器。

    用法:
        recorder = Recorder(capture=cap, save_dir="data/raw")
        recorder.start_continuous(interval_frames=10)
        # 或
        recorder.save_current_frame(frame)
    """

    def __init__(
        self,
        capture: ScreenCapture,
        save_dir: str | Path = "data/raw",
        image_format: str = "png",
    ):
        """
        Args:
            capture: 屏幕截图器实例
            save_dir: 保存目录（相对于项目根目录，或绝对路径）
            image_format: 图片格式 (png / jpg)
        """
        self.capture = capture
        self.image_format = image_format
        self._total_saved = 0

        # 解析保存目录
        save_path = Path(save_dir)
        if not save_path.is_absolute():
            save_path = PROJECT_ROOT / save_path
        self.save_dir = save_path
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, frame: np.ndarray, prefix: str = "frame") -> Path:
        """
        保存单帧图片。

        Args:
            frame: BGR numpy 数组
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.{self.image_format}"
        filepath = self.save_dir / filename
        cv2.imwrite(str(filepath), frame)
        self._total_saved += 1
        log.debug(f"帧已保存: {filepath}")
        return filepath

    def record_batch(
        self,
        num_frames: int = 100,
        interval_frames: int = 5,
        prefix: str = "frame",
    ) -> list[Path]:
        """
        批量录制：每隔 interval_frames 帧保存一帧。

        Args:
            num_frames: 总共保存的帧数
            interval_frames: 每隔多少帧保存一次
            prefix: 文件名前缀

        Returns:
            所有保存的文件路径列表
        """
        saved_paths: list[Path] = []
        frame_counter = 0

        log.info(
            f"开始批量录制: 目标 {num_frames} 帧, 间隔 {interval_frames} 帧"
        )

        while len(saved_paths) < num_frames:
            frame = self.capture.grab()
            frame_counter += 1

            if frame_counter % interval_frames == 0:
                path = self.save_frame(frame, prefix)
                saved_paths.append(path)

                if len(saved_paths) % 10 == 0:
                    log.info(f"录制进度: {len(saved_paths)}/{num_frames}")

        log.info(f"批量录制完成: 共保存 {len(saved_paths)} 帧到 {self.save_dir}")
        return saved_paths

    def record_interactive(self) -> None:
        """
        交互式录制模式。

        按键说明:
            S     - 保存当前帧
            Q     - 退出录制
            空格  - 连续录制开关
        """
        continuous = False
        frame_count = 0
        continuous_interval = 10  # 连续模式每 10 帧保存一次

        log.info("交互式录制模式已启动 | S=保存 | 空格=连续录制 | Q=退出")
        print("\n=== 交互式录制模式 ===")
        print("S     → 保存当前帧")
        print("空格  → 切换连续录制")
        print("Q     → 退出")
        print("======================\n")

        while True:
            frame = self.capture.grab()
            frame_count += 1

            # 连续录制
            if continuous and frame_count % continuous_interval == 0:
                self.save_frame(frame, "auto")

            # 显示预览（缩小到 50%）
            preview = cv2.resize(frame, None, fx=0.5, fy=0.5)

            # 在预览上叠加状态
            status = f"帧: {frame_count} | 已保存: {self._total_saved}"
            if continuous:
                status += " | [连续录制中]"
            cv2.putText(
                preview, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Recorder Preview", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                path = self.save_frame(frame, "manual")
                print(f"  ✓ 已保存: {path.name}")
            elif key == ord(" "):
                continuous = not continuous
                mode_str = "开启" if continuous else "关闭"
                print(f"  ◆ 连续录制: {mode_str}")

        cv2.destroyAllWindows()
        log.info(f"交互式录制结束 | 总保存: {self._total_saved}")

    @property
    def total_saved(self) -> int:
        return self._total_saved
