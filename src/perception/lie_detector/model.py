"""
lie_detector.model — 统一 facade

LieDetectorModel 是其他代码 (bot 视觉线程、离线脚本、其它机器集成) 唯一要 import 的类。

用法 (bot 实时):
    from src.perception.lie_detector import LieDetectorModel, LieBackend

    model = LieDetectorModel(
        detector_repo_path="C:/Users/heyas/Documents/code/lie-detector",
        backend=LieBackend.OPENCV,
        config={"activate_after_frames": 2, "deactivate_after_frames": 6},
    )
    while True:
        frame = capture.grab()              # letterboxed BGR
        result = model.update(frame)
        if result.active:
            mouse.move_to(result.target_center, result.confidence)

用法 (离线脚本/其它机器整合):
    model = LieDetectorModel(...)           # 同样接口
    for frame in load_video(path):
        result = model.update(frame)
        ...

设计思想 (设计文档 §既有资产 → §方案):
- bot 不复制检测代码, 而是注入 lie-detector/ 项目路径, import 它的函数;
- OpenCV 后端默认 (CPU, 无依赖);
- SAMURAI 后端可选 (需要 GPU + torch + samurai_repo);
- facade 统一处理 backend 切换 + 去抖 + bbox 膨胀, 调用方只看 LieDetectResult。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.logger import get_logger

from .opencv_backend import OpenCVBackend
from .samurai_backend import SamuraiBackend
from .state import (
    LieBackend, LieDetectResult, LiePhase,
    _DebounceState, _LieConfig, make_default_config,
    update_debounce,
)

log = get_logger("lie_detector.model")


class LieDetectorModel:
    """测谎仪统一 facade: OpenCV / SAMURAI 后端切换 + 去抖 + bbox 膨胀。

    单例状态机: 每个 bot 实例化一个, 视觉线程每帧调 update(frame)。
    """

    def __init__(
        self,
        detector_repo_path: str | Path,
        backend: str | LieBackend = LieBackend.OPENCV,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            detector_repo_path: lie-detector/ 项目绝对路径 (含 scripts/, samurai_repo/, models/)
            backend: "opencv" | "samurai"  (LieBackend 枚举或字符串)
            config: dict 覆盖默认参数, 支持的键:
                activate_after_frames (int): 连续 K 帧命中才激活 (默认 2)
                deactivate_after_frames (int): 连续 M 帧丢失才解除 (默认 6)
                bbox_inflate_ratio (float): 亮核→星形膨胀比 (默认 1.6)
        """
        self._repo_path = Path(detector_repo_path)
        self._backend_name = LieBackend(backend) if isinstance(backend, str) else backend

        # 配置: 合并默认 + 用户覆盖
        cfg = make_default_config()
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        self._cfg = cfg
        self._debounce = _DebounceState(
            activate_after=cfg.activate_after_frames,
            deactivate_after=cfg.deactivate_after_frames,
        )

        # 后端: 默认 OpenCV 必须成功; SAMURAI 失败时自动降级 OpenCV
        self._opencv = OpenCVBackend(self._repo_path)
        self._samurai: Optional[SamuraiBackend] = None
        if self._backend_name is LieBackend.SAMURAI:
            self._samurai = SamuraiBackend(self._repo_path)
            if not self._samurai.ready:
                log.warning(f"[model] SAMURAI 不可用, 降级 OpenCV: {self._samurai.import_error}")
                self._backend_name = LieBackend.OPENCV

        # 计时: 用于 timeout_sec 强制解除 (防检测卡住永远不释放)
        self._last_active_at: float = 0.0
        self._timeout_sec: float = 30.0  # 默认 30s, 可被 config 覆盖
        if config and "timeout_sec" in config:
            self._timeout_sec = float(config["timeout_sec"])

        log.info(
            f"[model] LieDetectorModel 就绪: backend={self._backend_name.value} "
            f"repo={self._repo_path} cfg=act{cfg.activate_after_frames}/"
            f"deact{cfg.deactivate_after_frames}/inflate{cfg.bbox_inflate_ratio:.2f} "
            f"timeout={self._timeout_sec:.0f}s"
        )

    # ── 公共属性 ──

    @property
    def backend(self) -> LieBackend:
        return self._backend_name

    @property
    def active(self) -> bool:
        """当前是否稳定激活 (经过去抖)。"""
        return self._debounce.active

    @property
    def opencv_ready(self) -> bool:
        return self._opencv.ready

    @property
    def samurai_ready(self) -> bool:
        return self._samurai is not None and self._samurai.ready

    # ── 主入口: 每帧调用 ──

    def update(self, frame: np.ndarray) -> LieDetectResult:
        """视觉线程每帧调一次, 返回当前帧检测结果。

        Args:
            frame: BGR numpy 数组, letterbox 后尺寸 (与 WindowCapture.grab() 一致)

        Returns:
            LieDetectResult — active 字段 = 经过去抖后的稳定判定;
            未触发时 active=False, target_center=None;
            触发时 active=True, target_center=(cx, cy), confidence/brightness 可用。
        """
        # OpenCV 后端: 每帧独立检测
        raw = self._opencv.detect(frame)

        # 阶段判定启发式: confidence 高 + 连续多帧中心稳定 = COUNTDOWN;
        # 否则 = TRACKING (目标在动)。这里用简单判定, 复杂逻辑留给调用方。
        if raw.active and raw.phase is LiePhase.COUNTDOWN:
            # bbox 面积大 + confidence 高 → 倒计时 (目标静止)
            if raw.confidence < 0.5 or raw.brightness < 150:
                raw.phase = LiePhase.TRACKING

        candidate_active = raw.active
        now = time.time()
        is_active = update_debounce(self._debounce, candidate_active, now)

        # 超时强制解除 (防检测卡住永远不释放)
        if is_active:
            if self._last_active_at == 0.0:
                self._last_active_at = now
            elif self._timeout_sec > 0 and (now - self._last_active_at) > self._timeout_sec:
                log.warning(f"[model] 测谎仪激活超时 ({self._timeout_sec:.0f}s), 强制解除")
                self._debounce.active = False
                self._debounce.miss_streak = self._debounce.deactivate_after
                self._last_active_at = 0.0
                return LieDetectResult(active=False)
        else:
            self._last_active_at = 0.0

        # 输出: active 取决于去抖后; 未激活时清掉 target 字段避免误用
        if is_active:
            return raw
        return LieDetectResult(active=False, backend=raw.backend)

    def reset(self) -> None:
        """重置去抖状态 (e.g. 测谎意外中途退出后想立刻重新进入)。"""
        self._debounce.active = False
        self._debounce.candidate_active = False
        self._debounce.hit_streak = 0
        self._debounce.miss_streak = 0
        self._debounce.activated_at = 0.0
        self._last_active_at = 0.0
        log.info("[model] 状态已重置")
