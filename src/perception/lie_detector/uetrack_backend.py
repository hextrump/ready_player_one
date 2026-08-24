"""
lie_detector.uetrack_backend — UETrack-T SOT 后端 (CPU, 轻量包装)

把 tools/uetrack_cpu_spike.py 验证过的 CPU 推理路径封装成 HybridBackend 可接的契约:
    ready / import_error / init_template(frame, bbox)->bool
    track(frame) -> (center, score) | None / stop()

实测结论 (tools/01.mp4 帧 40~199, 目标瞬移+变外观):
  - UETrack 一次 init 从头跟注定卡死 (中心误差 mean 127px): 瞬移后外观不匹配,
    Hann 窗把输出吸回搜索中心 = 钉在旧锚点。
  - 每个锚点重新 init 模板 (blob 强检测且分歧 >60px 时) → 停顿帧误差 ~10px。
  ⇒ 本 backend 允许反复 init_template; text_src (CLIP 语言 token, 与模板无关) 只算一次,
    re-init 只重灌视觉模板, 不重算 CLIP (否则每次 ~156ms)。
  - 模型构建 ~13s (含 CLIP ViT-L 加载), 懒加载: 构造后端只验路径/权重, 首次 init 才 build。
  - 输入为 BGR (与 bot 抓帧一致; spike 实测 BGR 优于 RGB — 白目标对通道序不敏感)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger("lie_detector.uetrack")


class UETrackBackend:
    """UETrack-T 单目标跟踪后端 (CPU)。

    用法 (供 HybridBackend):
        uet = UETrackBackend(repo_path=..., ckpt_path=...)
        if uet.ready:
            uet.init_template(frame, bbox)      # (x1,y1,x2,y2) → 内部转 (x,y,w,h)
            center, score = uet.track(frame)    # 每帧一次
    """

    def __init__(self, repo_path: str | Path, ckpt_path: str | Path):
        from src.utils.config import PROJECT_ROOT

        self._repo = Path(repo_path)
        self._ckpt = Path(ckpt_path)
        if not self._repo.is_absolute():
            self._repo = PROJECT_ROOT / self._repo
        if not self._ckpt.is_absolute():
            self._ckpt = PROJECT_ROOT / self._ckpt

        self._built = False
        self._inited = False
        self._import_error: Optional[Exception] = None
        self._imported = False
        self._try_import()

        self._network = None
        self._text_src = None
        self._state = None
        self._frame_id = 0

    # ── 状态 ──

    def _try_import(self) -> None:
        """只验路径 + torch 可用 (不 build, build 在首次 init_template 懒加载)。"""
        if not (self._repo / "lib").is_dir():
            self._import_error = FileNotFoundError(f"UETrack 仓库缺 lib/: {self._repo}")
            log.warning(f"[uetrack] {self._import_error}")
            return
        if not self._ckpt.is_file():
            self._import_error = FileNotFoundError(f"UETrack 权重缺失: {self._ckpt}")
            log.warning(f"[uetrack] {self._import_error}")
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            self._import_error = e
            log.warning(f"[uetrack] torch 未安装: {e}")
            return
        self._imported = True

    @property
    def ready(self) -> bool:
        """可构建 (路径/权重/torch 齐)。注意真实 build 在首次 init 才发生。"""
        return self._imported

    @property
    def import_error(self) -> Optional[Exception]:
        return self._import_error

    @property
    def inited(self) -> bool:
        return self._built and self._inited

    def warm(self) -> bool:
        """启动预热: 预构建网络 (13s 一次性), 避免首次弹窗现场阻塞 13s。

        测谎倒计时仅 ~7s, 现场构建会整段错过第一次。机器人启动时调一次,
        之后 init_template / track 都是毫秒级。
        """
        ok = self._ensure_built()
        log.info(f"[uetrack] 预热{'完成' if ok else '失败'} (ready={self.ready})")
        return ok

    # ── 构建 (懒) ──

    def _ensure_built(self) -> bool:
        """首次 init 时构建网络 + 载权重 + 预计算 text_src (CLIP 只算一次)。"""
        if self._built:
            return True
        if not self._imported:
            return False
        try:
            if str(self._repo) not in sys.path:
                sys.path.insert(0, str(self._repo))

            from lib.config.uetrack.config import cfg, update_config_from_file

            yaml_path = self._repo / "experiments" / "uetrack" / "uetrack_tiny.yaml"
            update_config_from_file(str(yaml_path))
            self._cfg = cfg

            # CPU 适配: 仓库用 is_main_process() 当 pretrained 标志, 单机 rank0 返回 True
            # → 尝试从空 PRETRAIN_TYPE 路径加载骨干 → crash; 权重在 tar 里, 这里强制 False。
            import lib.models.uetrack.encoder as _enc
            _enc.is_main_process = lambda: False

            from lib.models.uetrack import build_uetrack_inference
            net = build_uetrack_inference(cfg)
            net_state = load_net_state(str(self._ckpt))
            net.load_state_dict(net_state, strict=False)
            self._network = net.eval()

            # CPU preprocessor / hann / task_index
            from lib.test.tracker.utils import sample_target, transform_image_to_crop
            from lib.test.utils.hann import hann2d
            from .cpu_preprocessor import CPUPreprocessor

            self._sample_target = sample_target
            self._transform_image_to_crop = transform_image_to_crop
            self._preprocessor = CPUPreprocessor()
            fx_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
            self._output_window = hann2d(
                torch.tensor([fx_sz, fx_sz]).long(), centered=True)
            self._num_template = int(cfg.TEST.NUM_TEMPLATES)
            self._task_index_batch = torch.tensor([0])  # RGB 任务组 (LASOT/GOT10K)

            self._template_factor = cfg.TEST.TEMPLATE_FACTOR
            self._template_size = cfg.TEST.TEMPLATE_SIZE
            self._search_factor = cfg.TEST.SEARCH_FACTOR
            self._search_size = cfg.TEST.SEARCH_SIZE

            # 语言 token: use_nlp=False → 全零 token, 与模板无关 → 只算一次
            text_data = torch.zeros(1, 77, dtype=torch.long)
            with torch.no_grad():
                self._text_src = self._network.forward_textencoder_inference(text_data)

            self._built = True
            log.info(f"[uetrack] 模型构建+权重加载完成 ({self._ckpt.name})")
            return True
        except Exception as e:
            self._import_error = e
            log.warning(f"[uetrack] 构建失败: {e}", exc_info=True)
            return False

    # ── 模板 (可反复 init — 每个锚点 re-init) ──

    def init_template(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """以 bbox(x1,y1,x2,y2) 在 frame 上建立 SOT 模板。可反复调用 (锚点 re-init)。

        re-init 只重灌视觉模板 (text_src 已缓存), 单次 ~10-40ms。
        """
        if not self._ensure_built():
            return False
        try:
            x1, y1, x2, y2 = bbox
            init_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            z_patch_arr, rz = self._sample_target(
                frame, init_bbox, self._template_factor, output_sz=self._template_size)
            template = self._preprocessor.process(z_patch_arr)
            if template.size(1) == 3:
                template = torch.cat((template, template), dim=1)
            self._template_list = [template] * self._num_template
            self._state = init_bbox
            prev_crop = self._transform_image_to_crop(
                torch.tensor(init_bbox), torch.tensor(init_bbox), rz,
                torch.tensor([self._template_size, self._template_size]),
                normalize=True)
            self._template_anno_list = [prev_crop.unsqueeze(0)]
            self._frame_id = 0
            self._inited = True
            return True
        except Exception as e:
            log.warning(f"[uetrack] init_template 失败: {e}")
            self._inited = False
            return False

    # ── 跟踪 ──

    def track(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """每帧跟踪, 返回 (center, best_score); 未 init/异常返回 None。"""
        if not (self._built and self._inited):
            return None
        try:
            from lib.utils.box_ops import clip_box

            H, W, _ = frame.shape
            self._frame_id += 1
            x_patch_arr, rz = self._sample_target(
                frame, self._state, self._search_factor, output_sz=self._search_size)
            search = self._preprocessor.process(x_patch_arr)
            if search.size(1) == 3:
                search = torch.cat((search, search), dim=1)

            with torch.no_grad():
                enc_opt, _, _ = self._network.forward_encoder(
                    self._template_list, [search], self._template_anno_list,
                    self._text_src, self._task_index_batch)
                out_dict = self._network.forward_decoder(feature=enc_opt)

            pred_score_map = out_dict["score_map"]
            response = self._output_window * pred_score_map
            pred_boxes, conf_score = self._network.decoder.cal_bbox(
                response, out_dict["size_map"], out_dict["offset_map"], return_score=True)
            pred_boxes = pred_boxes.view(-1, 4)
            pred_box = (pred_boxes.mean(dim=0) * self._search_size / rz).tolist()

            # map_box_back: 相对 search 中心 → 原图坐标
            cx_prev = self._state[0] + 0.5 * self._state[2]
            cy_prev = self._state[1] + 0.5 * self._state[3]
            cx, cy, w, h = pred_box
            half = 0.5 * self._search_size / rz
            cx_real = cx + (cx_prev - half)
            cy_real = cy + (cy_prev - half)
            self._state = clip_box(
                [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], H, W, margin=10)
            x, y, w, h = self._state
            return ((int(x + w / 2), int(y + h / 2)), float(conf_score))
        except Exception as e:
            log.warning(f"[uetrack] track 异常: {e}")
            return None

    def stop(self) -> None:
        """释放模型 (仅清引用; 进程内复用无需重建)。"""
        self._network = None
        self._text_src = None
        self._built = False
        self._inited = False


def load_net_state(ckpt_path: str) -> dict:
    """从 uetrack_tiny.tar 抽出 'net' state dict (纯 tensor)。

    tar 是训练期完整 checkpoint, 顶层含 optimizer/stats/tensorboard 等对象,
    引用了 lib.train.admin.* 类 (发布仓库无对应文件)。用 stub unpickler 让这些
    全局返回哑对象 — 我们只要 'net' 的 tensor, 其余对象不实例化。
    """
    import pickle

    import torch as _t

    class _Dummy:
        def __init__(self, *a, **k):
            pass

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("lib.train") or module.startswith("lib.test"):
                return _Dummy
            return super().find_class(module, name)

    _orig = pickle.Unpickler
    pickle.Unpickler = _StubUnpickler
    try:
        sd = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    finally:
        pickle.Unpickler = _orig
    assert "net" in sd, f"checkpoint 缺 'net' 键: {list(sd.keys())}"
    return sd["net"]
