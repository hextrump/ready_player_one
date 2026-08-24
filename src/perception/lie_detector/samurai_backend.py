"""
lie_detector.samurai_backend — SAMURAI 后端 (GPU, 薄包装)

复用本地 lie-detector/samurai_track.py + samurai_repo/sam2 已有代码, 不重写 SAM2 包装。

流程:
  1. OpenCV 在第一帧检测 bbox (call OpenCVBackend.detect)
  2. 注入 sam2 sys.path, build_sam2_video_predictor
  3. propagate_in_video(state) 生成器 → per-frame (mask, bbox, center)
  4. mask 连续 N 帧空 + OpenCV 重检出 → add_new_points_or_box re-init (state 复用)

GPU 现实: torch.cuda.is_available() == False → 加载失败, ready=False。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

from .state import LieBackend, LieDetectResult, LiePhase

log = get_logger("lie_detector.samurai")


class SamuraiBackend:
    """SAMURAI 后端: GPU 跟踪, 视频序列上下文 (init_with_bbox + propagate 流)。

    使用方式:
        backend = SamuraiBackend(repo_path, model_size="base_plus")
        if backend.ready:
            backend.init_with_bbox(video_frames_iter, bbox=(x,y,x2,y2))
            for frame_idx, result in backend.propagate():
                # result = LieDetectResult(...)
    """

    # 默认模型: base_plus (速度和精度平衡); 大模型需要更多显存
    DEFAULT_MODEL_SIZE = "base_plus"

    # 模型 size → checkpoint 路径模板
    MODEL_PATHS = {
        "tiny":      "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "small":     "sam2/checkpoints/sam2.1_hiera_s.pt",
        "base_plus": "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        "large":     "sam2/checkpoints/sam2.1_hiera_large.pt",
    }

    CONFIG_PATHS = {
        "tiny":      "configs/samurai/sam2.1_hiera_t.yaml",
        "small":     "configs/samurai/sam2.1_hiera_s.yaml",
        "base_plus": "configs/samurai/sam2.1_hiera_b+.yaml",
        "large":     "configs/samurai/sam2.1_hiera_l.yaml",
    }

    def __init__(self, detector_repo_path: str | Path, model_size: str = DEFAULT_MODEL_SIZE):
        self._repo_path = Path(detector_repo_path)
        self._model_size = model_size
        self._predictor = None
        self._state = None
        self._imported = False
        self._import_error: Optional[Exception] = None
        self._try_import()

    def _try_import(self) -> None:
        """注入 sam2 + samurai scripts 到 sys.path; 检查 torch.cuda。

        注意 sys.path 顺序: sam2 包优先 (避免被 shadowing)。
        """
        samurai_repo = self._repo_path / "samurai_repo"
        sam2_pkg = samurai_repo / "sam2"
        samurai_scripts = samurai_repo / "scripts"
        if not (samurai_repo.is_dir() and sam2_pkg.is_dir()):
            self._import_error = FileNotFoundError(f"samurai_repo/sam2 not found: {samurai_repo}")
            log.warning(f"[samurai] {self._import_error}")
            return

        # 关键: sam2 包必须先于 repo 根入 path (避免 shadowing)
        sys.path.insert(0, str(sam2_pkg))
        sys.path.insert(0, str(samurai_scripts))

        # 检查 CUDA
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                self._import_error = RuntimeError("torch.cuda.is_available() == False; SAMURAI 需要 GPU")
                log.warning(f"[samurai] {self._import_error}")
                return
        except ImportError as e:
            self._import_error = e
            log.warning(f"[samurai] torch 未安装: {e}")
            return

        # 检查模型权重文件存在
        model_rel = self.MODEL_PATHS.get(self._model_size)
        if model_rel is None:
            self._import_error = ValueError(f"未知 model_size={self._model_size}; 可选 {list(self.MODEL_PATHS)}")
            log.warning(f"[samurai] {self._import_error}")
            return
        model_abs = samurai_repo / model_rel
        if not model_abs.is_file():
            self._import_error = FileNotFoundError(f"模型权重缺失: {model_abs}")
            log.warning(f"[samurai] {self._import_error}")
            return

        self._imported = True
        log.info(f"[samurai] 后端就绪 (model={self._model_size}, ckpt={model_abs.name})")

    @property
    def ready(self) -> bool:
        return self._imported

    @property
    def import_error(self) -> Optional[Exception]:
        return self._import_error

    def init_with_bbox(
        self,
        frames_iter,
        bbox: Tuple[int, int, int, int],
        obj_id: int = 0,
    ) -> bool:
        """初始化 SAM2 predictor + state。

        Args:
            frames_iter: 视频帧迭代器 (np.ndarray BGR)。sam2 自己读 video 或接 frames_iter。
                          这里用 frames_iter 形式让 bot 把窗口抓帧喂进来。
            bbox: (x1, y1, x2, y2) 第一帧的目标 bbox
            obj_id: 跟踪对象 id (单目标 = 0)

        Returns:
            True = 初始化成功, False = 失败 (查看 import_error / 日志)
        """
        if not self._imported:
            return False
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore

            config_rel = self.CONFIG_PATHS[self._model_size]
            config_abs = str(self._repo_path / "samurai_repo" / "sam2" / config_rel)
            model_abs = str(self._repo_path / "samurai_repo" / self.MODEL_PATHS[self._model_size])

            self._predictor = build_sam2_video_predictor(
                config_abs, model_abs, device="cuda:0",
            )
            log.info(f"[samurai] predictor 加载完成: {self._model_size}")

            import torch  # 已导入
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # offload_video_to_cpu=True: 视频帧太大时降低显存占用
                self._state = self._predictor.init_state(
                    video_frames=frames_iter, offload_video_to_cpu=True,
                )
                _, _, _ = self._predictor.add_new_points_or_box(
                    self._state, box=bbox, frame_idx=0, obj_id=obj_id,
                )
            log.info(f"[samurai] state 初始化完成 (bbox={bbox})")
            return True
        except Exception as e:
            log.warning(f"[samurai] init 失败: {e}", exc_info=True)
            return False

    def propagate(self, max_empty_reinit: int = 5) -> Iterator[Tuple[int, LieDetectResult]]:
        """生成器: 逐帧 yield (frame_idx, LieDetectResult)。

        mask 连续空 → yield active=False; 调用方决定是否 re-init (用 OpenCV 重检出 bbox)。
        """
        if not self._imported or self._predictor is None or self._state is None:
            return
        try:
            import torch
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for frame_idx, object_ids, masks in self._predictor.propagate_in_video(self._state):
                    yield from self._emit_frame(frame_idx, object_ids, masks)
        except Exception as e:
            log.warning(f"[samurai] propagate 异常: {e}", exc_info=True)
            return
        finally:
            self._cleanup()

    def _emit_frame(
        self, frame_idx: int, object_ids, masks,
    ) -> Iterator[Tuple[int, LieDetectResult]]:
        """把 sam2 输出转成 LieDetectResult (单个 obj_id=0)。"""
        for obj_id, mask in zip(object_ids, masks):
            if obj_id != 0:
                continue
            mask_np = mask[0].cpu().numpy() > 0.0
            ys, xs = np.where(mask_np)
            if len(xs) == 0:
                # mask 空 — 调用方应触发 re-init (此帧 active=False 但保留 backend=SAMURAI)
                yield (frame_idx, LieDetectResult(
                    active=False, phase=LiePhase.TRACKING, backend=LieBackend.SAMURAI,
                ))
                return
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            # 亮度由 mask 区域均值估 (mask 内 = 跟踪到目标, 应是亮的; 这里 mask 来自 sam2 不是亮度,
            # 所以 brightness 只能从原图取 — propagate 流没有原帧, 暂给 1.0 占位)
            yield (frame_idx, LieDetectResult(
                active=True,
                phase=LiePhase.TRACKING,
                target_center=(cx, cy),
                target_bbox=(x_min, y_min, x_max, y_max),
                confidence=1.0,           # SAMURAI 一旦有 mask 即视为高 confidence
                brightness=255.0,         # 占位 (propagate 流拿不到原帧)
                backend=LieBackend.SAMURAI,
            ))
            return

    def reinit_with_bbox(self, bbox: Tuple[int, int, int, int], obj_id: int = 0) -> bool:
        """不重建 predictor, 直接用新 bbox 喂给现有 state (用于 mask 连续空时)。"""
        if not self._imported or self._predictor is None or self._state is None:
            return False
        try:
            # 找到当前 propagate 进度对应的 frame_idx (state["frame_id"] 或类似)
            frame_idx = int(getattr(self._state, "frame_id", 0))
            _, _, _ = self._predictor.add_new_points_or_box(
                self._state, box=bbox, frame_idx=frame_idx, obj_id=obj_id,
            )
            log.info(f"[samurai] re-init bbox={bbox} @ frame={frame_idx}")
            return True
        except Exception as e:
            log.warning(f"[samurai] re-init 失败: {e}")
            return False

    def stop(self) -> None:
        """停止跟踪并清理 GPU 显存。"""
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            del self._predictor, self._state
        except Exception:
            pass
        self._predictor = None
        self._state = None
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.clear_autocast_cache()
                torch.cuda.empty_cache()
        except Exception:
            pass
