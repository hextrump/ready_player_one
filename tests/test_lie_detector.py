"""
测谎仪 (LIE DETECTOR) 鼠标追踪 — 回归测试。

覆盖: 原生白块检测 (不依赖外部仓库) / 去抖状态机 / vendored 后端就绪 / 全链路 facade。
依赖 vendored 仓库 (models/lie_detector/) 的用例在仓库缺席时自动 skip。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.lie_detector.model import LieDetectorModel
from src.perception.lie_detector.opencv_backend import OpenCVBackend
from src.perception.lie_detector.state import (
    LieBackend, _DebounceState, update_debounce,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDORED_REPO = PROJECT_ROOT / "models" / "lie_detector"


def _vendored_repo_or_skip() -> Path:
    """vendored 仓库不在 → skip 依赖它的用例。"""
    if not (VENDORED_REPO / "scripts" / "auto_bbox.py").is_file():
        pytest.skip("models/lie_detector 未落地")
    return VENDORED_REPO


def _white_square_frame(box=(270, 130, 370, 230), size=(360, 640)) -> np.ndarray:
    """暗底 + 中央白色方块 (模拟测谎白色图形), 中心 (320, 180)。"""
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
    return frame


# ── 原生白块检测 (OpenCVBackend._detect_with_confidence, 不依赖外部仓库) ──


def test_detect_with_confidence_white_target():
    """暗底中央白色方块 → bbox 中心应精准命中方块中心 (多阈值 + IoU 投票)。"""
    be = OpenCVBackend("__nonexistent_repo__")  # 仓库缺席也照样能跑原生检测
    bbox, matched = be._detect_with_confidence(_white_square_frame(), None)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert ((x1 + x2) // 2, (y1 + y2) // 2) == (320, 180)
    assert matched >= 1


def test_detect_with_confidence_blank_frame():
    """纯暗帧 → 无目标, 返回 None (不误报)。"""
    be = OpenCVBackend("__nonexistent_repo__")
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    bbox, matched = be._detect_with_confidence(blank, None)
    assert bbox is None
    assert matched == 0


# ── 去抖状态机 (state.update_debounce) ──


def test_update_debounce_activate_then_clear():
    """连续 K 帧命中才激活, 连续 M 帧丢失才解除 (激活快、解除慢的防抖设计)。"""
    st = _DebounceState(activate_after=2, deactivate_after=2)
    assert update_debounce(st, True, 1.0) is False   # hit 1: 未达门槛
    assert update_debounce(st, True, 1.0) is True    # hit 2: 激活
    assert st.active is True
    assert update_debounce(st, False, 2.0) is True   # miss 1: 单帧漏检不解除
    assert update_debounce(st, False, 2.0) is False  # miss 2: 解除
    assert st.active is False


# ── vendored 后端就绪 (依赖 models/lie_detector, 缺席 skip) ──


def test_opencv_backend_ready_with_vendored_repo():
    """vendored 仓库在 → opencv 后端能 import 检测函数, ready=True。"""
    repo = _vendored_repo_or_skip()
    be = OpenCVBackend(str(repo))
    assert be.ready is True
    assert be.import_error is None


def test_model_init_without_repo_does_not_crash():
    """仓库路径不存在 → 模型初始化不崩, 只是不可用 (bot 启动不应失败)。"""
    model = LieDetectorModel("__nonexistent_repo__")
    assert model.opencv_ready is False
    assert model.active is False


# ── 全链路 facade (vendored 仓库在才跑) ──


def test_model_update_detects_white_target():
    """model.update 全链路: 2 帧命中 → active, 中心精准; 2 帧空白 → 解除。"""
    repo = _vendored_repo_or_skip()
    model = LieDetectorModel(
        repo, backend=LieBackend.OPENCV,
        config={"activate_after_frames": 2, "deactivate_after_frames": 2},
    )
    assert model.opencv_ready

    model.update(_white_square_frame())            # hit 1
    r = model.update(_white_square_frame())        # hit 2 → active
    assert r.active is True
    assert r.target_center == (320, 180)
    assert r.confidence > 0.9
    assert model.active is True

    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    model.update(blank)                            # miss 1
    model.update(blank)                            # miss 2 → 解除
    assert model.active is False
