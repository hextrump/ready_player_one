"""
lie_detector.samurai_stream — SAMURAI 流式会话 (GPU, 实时逐帧推进)

标准 propagate_in_video 生成器在 init_state 时把 num_frames 锁死
(sam2_video_predictor.py:62), 无法逐帧流式喂。这里手动构造 inference_state
(images 用可增长 list), 每帧 append + 更新 num_frames, 再调
_run_single_frame_inference 推进一步 — SAM2 记忆库正常续跟。

用法 (服务端, 每个测谎事件一个会话):
    ss = SamuraiStream(repo_path)          # repo = lie-detector 项目 (samurai_repo 在其中)
    if ss.ready: ss.warm()                 # 启动一次性构建 predictor (~10-20s)
    ss.start(frame_bgr, bbox)              # 首帧 + opencv bbox → init SAM2 state
    center, conf = ss.step(frame_bgr)      # 后续每帧推进 → ((cx,cy), confidence)
    ss.stop()                              # 结束会话, 释放 state/显存 (predictor 复用)

config 路径: 权重在 `samurai_repo/sam2/checkpoints/` (相对 samurai_repo 多一层 sam2);
config_name 相对 hydra 搜索根 (= sam2 包目录 `samurai_repo/sam2/sam2`),
用 SAMURAI 的 `configs/samurai/sam2.1_hiera_<size>.yaml`
(与官方 `configs/sam2.1/` 配置内容相同, demo.py 昨天实测跑通)。
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger("lie_detector.samurai_stream")

# ImageNet 归一化 (SAM2 仓库同款, 见 sam2/utils/misc.py load_video_frames)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class SamuraiStream:
    """SAMURAI 实时流式会话: start(首帧+bbox) → step(每帧) → stop()。

    单模型实例, 会话间复用 predictor, 只重置 inference_state。
    """

    DEFAULT_MODEL_SIZE = "base_plus"

    # 模型 size → 权重路径 (相对 samurai_repo, 实际在 samurai_repo/sam2/checkpoints/)
    MODEL_PATHS = {
        "tiny":      "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "small":     "sam2/checkpoints/sam2.1_hiera_s.pt",
        "base_plus": "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        "large":     "sam2/checkpoints/sam2.1_hiera_large.pt",
    }
    # config_name 相对 hydra 搜索根 = sam2 包目录 (samurai_repo/sam2/sam2);
    # 用 SAMURAI 的 configs/samurai/ 配置 (与 configs/sam2.1/ 官方配置同内容, 昨天实测跑通)
    CONFIG_PATHS = {
        "tiny":      "configs/samurai/sam2.1_hiera_t.yaml",
        "small":     "configs/samurai/sam2.1_hiera_s.yaml",
        "base_plus": "configs/samurai/sam2.1_hiera_b+.yaml",
        "large":     "configs/samurai/sam2.1_hiera_l.yaml",
    }

    def __init__(self, detector_repo_path: str | Path, model_size: str = DEFAULT_MODEL_SIZE,
                 image_size: int | None = None):
        self._repo_path = Path(detector_repo_path)
        self._model_size = model_size
        self._requested_image_size = image_size  # None = 用配置默认 (1024); 512 可省 ~4x 算力
        self._imported = False
        self._import_error: Optional[Exception] = None
        self._build_error: Optional[Exception] = None
        self._predictor = None
        self._state: Optional[dict] = None
        self._next_idx = 0
        self._device = None
        self._image_size = 1024
        self._img_mean = None
        self._img_std = None
        self._try_import()

    # ── 就绪检查 ──

    def _try_import(self) -> None:
        """注入 sam2 包到 sys.path, 检查 CUDA + 权重/config 存在。失败不崩 (ready=False)。"""
        samurai_repo = self._repo_path / "samurai_repo"
        sam2_pkg = samurai_repo / "sam2"
        if not (samurai_repo.is_dir() and sam2_pkg.is_dir()):
            self._import_error = FileNotFoundError(f"samurai_repo/sam2 not found: {samurai_repo}")
            log.warning(f"[samurai_stream] {self._import_error}")
            return
        sys.path.insert(0, str(sam2_pkg))
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                self._import_error = RuntimeError("torch.cuda.is_available()==False; SAMURAI 需要 GPU")
                log.warning(f"[samurai_stream] {self._import_error}")
                return
        except ImportError as e:
            self._import_error = e
            log.warning(f"[samurai_stream] torch 未安装: {e}")
            return

        if self._model_size not in self.MODEL_PATHS:
            self._import_error = ValueError(f"未知 model_size={self._model_size}; 可选 {list(self.MODEL_PATHS)}")
            log.warning(f"[samurai_stream] {self._import_error}")
            return
        model_abs = samurai_repo / self.MODEL_PATHS[self._model_size]
        if not model_abs.is_file():
            self._import_error = FileNotFoundError(f"模型权重缺失: {model_abs}")
            log.warning(f"[samurai_stream] {self._import_error}")
            return
        # config 是 hydra 相对名; 文件在 sam2 包目录下 samurai_repo/sam2/sam2/configs/...
        config_abs = samurai_repo / "sam2" / "sam2" / self.CONFIG_PATHS[self._model_size]
        if not config_abs.is_file():
            self._import_error = FileNotFoundError(f"sam2 config 缺失: {config_abs}")
            log.warning(f"[samurai_stream] {self._import_error}")
            return

        self._imported = True
        log.info(f"[samurai_stream] 后端就绪 (model={self._model_size}, ckpt={model_abs.name})")

    @property
    def ready(self) -> bool:
        return self._imported

    @property
    def import_error(self) -> Optional[Exception]:
        return self._import_error

    @property
    def model_ready(self) -> bool:
        """predictor 是否已构建 (warm 完成)。"""
        return self._predictor is not None

    @property
    def build_error(self) -> Optional[Exception]:
        return self._build_error

    @property
    def session_active(self) -> bool:
        return self._state is not None

    # ── 模型构建 (一次性, 后台/启动时) ──

    def warm(self) -> bool:
        """构建 SAM2 predictor (首次 ~10-20s)。跨会话复用。"""
        if not self._imported:
            return False
        if self._predictor is not None:
            return True
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore

            samurai_repo = self._repo_path / "samurai_repo"
            # hydra 已由 import sam2 初始化为包目录搜索根 → 传相对 config_name (非绝对路径)
            config_name = self.CONFIG_PATHS[self._model_size]
            model_abs = str(samurai_repo / self.MODEL_PATHS[self._model_size])
            extra = []
            if self._requested_image_size:
                # config 的 image_size 是标量 (如 1024), 保持标量 (list 会让 //16 崩溃)
                extra.append(f"++model.image_size={self._requested_image_size}")
            self._predictor = build_sam2_video_predictor(
                config_name, model_abs, device="cuda:0", hydra_overrides_extra=extra,
            )
            self._device = torch.device("cuda:0")
            size = self._predictor.image_size
            self._image_size = int(size[0]) if isinstance(size, (tuple, list)) else int(size)
            self._img_mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None].to(self._device)
            self._img_std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None].to(self._device)
            self._build_error = None
            log.info(f"[samurai_stream] predictor 构建完成 ({self._model_size}, image_size={self._image_size})")
            return True
        except Exception as e:
            self._build_error = e
            log.warning(f"[samurai_stream] warm 失败: {e}", exc_info=True)
            return False

    # ── 会话生命周期 ──

    def start(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """开启新会话: 首帧 + opencv bbox → init SAM2 state (cond frame 0)。"""
        if not self._imported or self._predictor is None:
            log.warning("[samurai_stream] start 被拒: 未 ready / 未 warm")
            return False
        try:
            import torch
            H, W = frame_bgr.shape[:2]
            t0 = self._preprocess(frame_bgr)          # CHW, CPU (offload)
            self._state = self._build_state(H, W, self._device, images=[t0])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # 同 init_state: 预热 frame 0 特征 → 喂 box → 收拢 cond 帧
                self._predictor._get_image_feature(self._state, frame_idx=0, batch_size=1)
                self._predictor.add_new_points_or_box(self._state, box=bbox, frame_idx=0, obj_id=0)
                self._predictor.propagate_in_video_preflight(self._state)
            self._next_idx = 1
            log.info(f"[samurai_stream] 会话开启 bbox={bbox} frame={H}x{W}")
            return True
        except Exception as e:
            log.warning(f"[samurai_stream] start 失败: {e}", exc_info=True)
            self._state = None
            return False

    def step(self, frame_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """推进一帧, 返回 ((cx, cy), confidence) 或 None (跟丢/异常)。

        center 在发送帧坐标空间 (mask 从 model-resize 回原分辨率)。
        """
        if not self._imported or self._predictor is None or self._state is None:
            return None
        try:
            import torch
            frame_idx = self._next_idx
            self._state["images"].append(self._preprocess(frame_bgr))
            self._state["num_frames"] += 1
            output_dict = self._state["output_dict"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                current_out, pred_masks = self._predictor._run_single_frame_inference(
                    inference_state=self._state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                self._predictor._add_output_per_object(
                    self._state, frame_idx, current_out, "non_cond_frame_outputs"
                )
                self._state["frames_already_tracked"][frame_idx] = {"reverse": False}
                _, video_res_masks = self._predictor._get_orig_video_res_output(
                    self._state, pred_masks
                )
            # video_res_masks: (num_obj, 1, H, W) logits @ 原分辨率; >0 = sigmoid>0.5
            mask = video_res_masks[0, 0].cpu().numpy()
            ys, xs = np.where(mask > 0.0)
            if len(xs) == 0:
                return None
            cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))
            obj_score = float(torch.sigmoid(current_out["object_score_logits"]).flatten()[0])
            self._next_idx = frame_idx + 1
            return (cx, cy), obj_score
        except Exception as e:
            log.warning(f"[samurai_stream] step 失败: {e}", exc_info=True)
            return None

    def stop(self) -> None:
        """结束会话: 丢弃 state, 释放显存。predictor 保留复用。"""
        self._state = None
        self._next_idx = 0
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.clear_autocast_cache()
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("[samurai_stream] 会话已结束 (显存已释放)")

    # ── inference_state 构造 (纯数据, 可无 GPU 单测) ──

    @staticmethod
    def _build_state(H: int, W: int, device=None, images=None) -> dict:
        """手动构造 SAM2 inference_state dict (键对齐 sam2_video_predictor.py init_state)。

        images 用可增长 list (逐帧 append 实现流式); offload_video_to_cpu=True 存 CPU。
        device 为空 (未 warm) 时只测结构, 不真跑推理。
        """
        import torch
        images = list(images) if images is not None else []
        return {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
            "video_height": H,
            "video_width": W,
            "device": device,
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }

    # ── 帧预处理 (与 load_video_frames 一致: RGB CHW /255 + ImageNet 归一化) ──

    def _preprocess(self, frame_bgr: np.ndarray):
        """BGR → RGB, resize 到 image_size, CHW float /255 归一化。返回 CPU tensor。"""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._image_size, self._image_size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        if self._img_mean is not None and self._img_std is not None:
            t = (t - self._img_mean.cpu()) / self._img_std.cpu()
        return t
