"""
测谎仪 SAMURAI 流式会话 — 无 GPU 门禁回归。

本机是 CPU (torch 2.10.0+cpu), 无法真跑 SAM2 推理, 只测:
- 缺仓库 / 无 GPU → ready=False 且不崩 (import_error 记录原因);
- inference_state 构造 (SamuraiStream._build_state) 键与 sam2_video_predictor.init_state 对齐,
  且 images 可增长 / num_frames 联动 — 流式推进的结构基础。
真 GPU 推理由 hhh 服务端 --spike 门禁覆盖 (联调阶段)。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.lie_detector.samurai_stream import SamuraiStream

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDORED_REPO = PROJECT_ROOT / "models" / "lie_detector"

# sam2_video_predictor.py init_state 用到的全部键 (从源码 60-111 行核对)
SAM2_STATE_KEYS = {
    "images", "num_frames", "offload_video_to_cpu", "offload_state_to_cpu",
    "video_height", "video_width", "device", "storage_device",
    "point_inputs_per_obj", "mask_inputs_per_obj", "cached_features",
    "constants", "obj_id_to_idx", "obj_idx_to_id", "obj_ids",
    "output_dict", "output_dict_per_obj", "temp_output_dict_per_obj",
    "consolidated_frame_inds", "tracking_has_started", "frames_already_tracked",
}
OUTPUT_DICT_KEYS = {"cond_frame_outputs", "non_cond_frame_outputs"}
CONSOLIDATED_KEYS = {"cond_frame_outputs", "non_cond_frame_outputs"}


# ── 就绪门禁 (不崩) ──


def test_stream_nonexistent_repo_ready_false():
    """仓库路径不存在 → ready=False, import_error 有值, 不抛异常。"""
    ss = SamuraiStream("__nonexistent_repo__")
    assert ss.ready is False
    assert ss.import_error is not None
    assert ss.model_ready is False
    assert ss.session_active is False
    assert ss.build_error is None


def test_stream_cpu_no_gpu_ready_false():
    """vendored 仓库在 (本机 CPU) → 过目录检查, 卡在 CUDA 门禁: ready=False。"""
    if not (VENDORED_REPO / "samurai_repo" / "sam2").is_dir():
        pytest.skip("models/lie_detector 未落地")
    ss = SamuraiStream(VENDORED_REPO)
    assert ss.ready is False
    assert ss.import_error is not None
    assert "GPU" in str(ss.import_error) or "torch" in str(ss.import_error).lower()
    # warm 不应崩, 返回 False
    assert ss.warm() is False


# ── inference_state 构造 (流式结构) ──


def test_build_state_keys_complete():
    """_build_state 输出的 dict 键与 SAM2 init_state 对齐 (缺键必崩于推理)。"""
    st = SamuraiStream._build_state(540, 960)
    assert set(st.keys()) == SAM2_STATE_KEYS, f"缺键: {SAM2_STATE_KEYS - set(st.keys())}"
    assert set(st["output_dict"].keys()) == OUTPUT_DICT_KEYS
    assert set(st["consolidated_frame_inds"].keys()) == CONSOLIDATED_KEYS
    assert st["video_height"] == 540 and st["video_width"] == 960
    assert st["num_frames"] == 0
    assert st["images"] == []
    assert st["offload_video_to_cpu"] is True
    assert st["tracking_has_started"] is False


def test_build_state_with_first_image_sets_num_frames():
    """带首帧构造 (start 路径) → images=[1], num_frames=1 (SAM2 锁死 len(images))。"""
    dummy_img = object()   # 只测结构, 不真跑推理
    st = SamuraiStream._build_state(360, 640, device=None, images=[dummy_img])
    assert len(st["images"]) == 1
    assert st["num_frames"] == 1


def test_build_state_growable_for_streaming():
    """流式推进的结构基础: images 可 append + num_frames 手动递增 (SAM2 序列语义)。"""
    st = SamuraiStream._build_state(360, 640)
    for i in range(5):
        st["images"].append(object())
        st["num_frames"] += 1
        assert len(st["images"]) == st["num_frames"] == i + 1
    # output_dict 承载逐帧结果
    st["output_dict"]["non_cond_frame_outputs"][3] = {"dummy": True}
    assert 3 in st["output_dict"]["non_cond_frame_outputs"]
