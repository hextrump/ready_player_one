"""
感知模块测试。
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestEvents:
    """事件系统测试。"""

    def test_event_types(self):
        from src.state.events import EventType
        assert EventType.MONSTER_DETECTED.value == "monster_detected"
        assert EventType.CAPTCHA_DETECTED.value == "captcha_detected"

    def test_bbox(self):
        from src.state.events import BBox
        bbox = BBox(x=10, y=20, w=100, h=50)
        assert bbox.center == (60, 45)
        assert bbox.area == 5000

    def test_detection(self):
        from src.state.events import Detection, BBox
        det = Detection(
            class_name="monster",
            bbox=BBox(x=0, y=0, w=50, h=50),
            confidence=0.9
        )
        assert det.class_name == "monster"
        assert det.center == (25, 25)

    def test_game_event_expiry(self):
        import time
        from src.state.events import GameEvent, EventType

        event = GameEvent(
            event_type=EventType.MONSTER_DETECTED,
            timestamp=time.time() - 1.0,  # 1秒前
        )
        assert event.is_expired(ttl_ms=500)  # 500ms TTL

        recent = GameEvent(
            event_type=EventType.MONSTER_DETECTED,
            timestamp=time.time(),
        )
        assert not recent.is_expired(ttl_ms=500)

    def test_platform(self):
        from src.state.events import Platform
        p = Platform(x=100, y=200, width=300)
        assert p.center == (250, 200)

    def test_rope(self):
        from src.state.events import Rope
        r = Rope(x=50, y_top=100, y_bottom=400)
        assert r.height == 300


class TestYOLODetector:
    """YOLO 检测器测试（不加载实际模型）。"""

    def test_import(self):
        from src.perception.yolo_detector import YOLODetector
        assert YOLODetector is not None

    def test_init(self):
        from src.perception.yolo_detector import YOLODetector
        detector = YOLODetector(
            model_path="dummy.pt",
            confidence=0.5,
            device="cpu"
        )
        assert detector.confidence == 0.5
        assert detector.device == "cpu"


class TestOpenCVAnalyzer:
    """OpenCV 分析器测试。"""

    def test_import(self):
        from src.perception.opencv_analyzer import OpenCVAnalyzer
        assert OpenCVAnalyzer is not None

    def test_build_nav_grid(self):
        from src.perception.opencv_analyzer import OpenCVAnalyzer
        from src.state.events import Platform, Rope

        analyzer = OpenCVAnalyzer(templates_dir="data/templates")

        platforms = [Platform(x=100, y=400, width=200)]
        ropes = [Rope(x=200, y_top=200, y_bottom=400)]

        grid = analyzer.build_nav_grid(
            platforms, ropes,
            map_size=(400, 600),
            grid_cell=20
        )

        assert grid.shape == (30, 20)  # 600/20, 400/20
        assert grid.sum() > 0  # 有可通行区域


class TestOCRReader:
    """OCR 读取器测试（不加载模型）。"""

    def test_import(self):
        from src.perception.ocr_reader import OCRReader
        assert OCRReader is not None

    def test_init(self):
        from src.perception.ocr_reader import OCRReader
        reader = OCRReader(lang=["ch", "en"])
        assert reader.engine_name == "paddleocr"
