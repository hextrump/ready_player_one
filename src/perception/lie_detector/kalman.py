"""
lie_detector.kalman — 极简恒速 Kalman (中心点跟踪)

4 态 [x, y, vx, vy] 常速度模型, 用于:
- 预测下一帧中心 (停顿帧: 预测目标仍在原处附近 → 门控候选)
- 修正带噪观测 (把各层融合后的中心平滑化, 防单帧抖跳)

设计说明 (实证于 01.mp4):
- 目标"瞬移+停顿", 帧间可跳几百 px → 这里**不做**"预测点周围小窗硬搜索"
  (那会漏掉瞬移); Kalman 只用于:
    a) 多候选打分时的先验权重 (靠近预测点得分更高)
    b) 输出平滑 (结果不瞬跳)
  搜索范围由上层给"弹窗/全帧", 不在本类收死。
- 测量噪声按层置信度给: 置信越高噪声越小 → 平滑越信任观测。
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("lie_detector.kalman")


class TinyKalman:
    """4 态恒速 Kalman (cv2.KalmanFilter), 线程单调用 (视觉线程)。"""

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measure_noise: float = 1e-1,
        measurement_scale: float = 1.0,   # 像素单位 → 状态单位 (状态用像素, 恒 1)
    ):
        self._dt = dt
        self._kf = cv2.KalmanFilter(4, 2)
        # F: [x; y; vx; vy] ← dt 常速
        self._kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # H: 观测 = 位置
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self, x: int, y: int) -> None:
        """以 (x,y) 重新起手 (测谎激活/目标瞬移重定位时调用)。"""
        self._kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self._initialized = True
        log.debug(f"[kalman] reset @ ({x},{y})")

    def predict(self) -> tuple[float, float]:
        """返回预测中心 (未初始化时返回 (0,0))。"""
        if not self._initialized:
            return (0.0, 0.0)
        pred = self._kf.predict()          # cv2 返回 (4,1) 列向量
        return (float(pred[0, 0]), float(pred[1, 0]))

    def correct(self, x: float, y: float, confidence: float = 1.0) -> tuple[float, float]:
        """用带置信的观测修正, 返回修正后中心。

        confidence ∈ [0,1]: 高置信 → 低测量噪声 → 平滑信任观测;
        低置信 → 高测量噪声 → 平滑更信预测 (停顿/模糊帧)。
        """
        if not self._initialized:
            self.reset(int(x), int(y))
            return (float(x), float(y))
        noise = 0.05 + (1.0 - max(0.0, min(1.0, confidence))) * 5.0
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise
        meas = np.array([[x], [y]], dtype=np.float32)
        corr = self._kf.correct(meas)      # (4,1) 列向量
        return (float(corr[0, 0]), float(corr[1, 0]))
