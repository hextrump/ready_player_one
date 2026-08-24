"""
lie_detector — MapleStory 测谎仪 (LIE DETECTOR) 鼠标追踪模型包

把 OpenCV 检测 + SAMURAI 跟踪打包成统一 facade (LieDetectorModel),
方便 ready_player_one bot 和其他机器的整合代码 import 即用。

公开 API:
    LieDetectorModel  — 统一入口 (视觉线程每帧 update)
    LieDetectResult   — 单帧结果 (active / target_center / target_bbox / confidence / brightness / backend)
    LiePhase          — 阶段枚举 (IDLE / COUNTDOWN / TRACKING)
    LieBackend        — 后端枚举 (OPENCV / SAMURAI)
    OpenCVBackend     — OpenCV 后端 (CPU, 默认)
    SamuraiBackend    — SAMURAI 后端 (GPU, 可选)

最小用法:
    from src.perception.lie_detector import LieDetectorModel

    model = LieDetectorModel(
        detector_repo_path="C:/Users/heyas/Documents/code/lie-detector",
        backend="opencv",  # 或 "samurai" (需 GPU)
    )
    while True:
        frame = capture.grab()           # BGR numpy
        result = model.update(frame)
        if result.active:
            # 鼠标跟随: result.target_center = (cx, cy) in letterbox coords
            ...
"""
from .model import LieDetectorModel
from .state import LieBackend, LieDetectResult, LiePhase
from .opencv_backend import OpenCVBackend
from .samurai_backend import SamuraiBackend

__all__ = [
    "LieDetectorModel",
    "LieDetectResult",
    "LiePhase",
    "LieBackend",
    "OpenCVBackend",
    "SamuraiBackend",
]
