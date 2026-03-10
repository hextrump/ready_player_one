"""
屏幕截图模块测试。
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestScreenCapture:
    """ScreenCapture 单元测试。"""

    def test_import(self):
        """验证模块可以被导入。"""
        from src.capture.screen_capture import ScreenCapture
        assert ScreenCapture is not None

    def test_init_default(self):
        """验证默认初始化。"""
        from src.capture.screen_capture import ScreenCapture
        cap = ScreenCapture()
        assert cap.target_fps == 30
        assert cap.region is None

    def test_init_with_region(self):
        """验证带区域的初始化。"""
        from src.capture.screen_capture import ScreenCapture
        cap = ScreenCapture(region=(0, 0, 800, 600), target_fps=60)
        assert cap.target_fps == 60
        assert cap.region == (0, 0, 800, 600)

    def test_grab_returns_numpy(self):
        """验证截图返回 numpy 数组。"""
        from src.capture.screen_capture import ScreenCapture
        cap = ScreenCapture(region=(0, 0, 100, 100))
        cap.start()
        try:
            frame = cap.grab()
            assert isinstance(frame, np.ndarray)
            assert frame.shape[2] == 3  # BGR
            assert frame.shape[0] == 100
            assert frame.shape[1] == 100
        finally:
            cap.stop()

    def test_context_manager(self):
        """验证上下文管理器。"""
        from src.capture.screen_capture import ScreenCapture
        with ScreenCapture(region=(0, 0, 50, 50)) as cap:
            frame = cap.grab()
            assert frame is not None


class TestRecorder:
    """Recorder 单元测试。"""

    def test_import(self):
        from src.capture.recorder import Recorder
        assert Recorder is not None

    def test_save_frame(self, tmp_path):
        """验证帧保存。"""
        from src.capture.screen_capture import ScreenCapture
        from src.capture.recorder import Recorder

        cap = ScreenCapture(region=(0, 0, 50, 50))
        cap.start()
        try:
            recorder = Recorder(capture=cap, save_dir=str(tmp_path))
            frame = cap.grab()
            path = recorder.save_frame(frame)
            assert path.exists()
            assert path.suffix == ".png"
            assert recorder.total_saved == 1
        finally:
            cap.stop()
