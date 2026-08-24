"""
human_mouse — 拟人化鼠标轨迹 (设计文档 §8)

替代原 move_mouse_smooth (线性缓动), 用贝塞尔曲线 + sigmoid 速度曲线 + 微抖 + 随机停顿,
让鼠标轨迹接近人手移动模式, 降低反作弊识别率。

核心函数: HumanMouseController.move_to(current_xy, target_xy, duration_ms, cfg) -> None
  - 阻塞执行完轨迹 (mouse_tracker 在独立线程里调, 不阻塞视觉线程)
  - 内部按贝塞尔等弧长采样 N 个中间点, 每步 win32api.SetCursorPos
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Tuple

import win32api

from src.utils.logger import get_logger

log = get_logger("human_mouse")


@dataclass
class HumanMouseConfig:
    """可调参数 (设计文档 §8 + config.yaml lie_detector.human_mouse.*)。"""
    bezier_jitter: float = 2.0       # 控制点随机偏移 ±px
    jitter_amplitude: float = 0.5    # 轨迹微抖 ±px
    pause_prob: float = 0.05         # 中段随机停顿概率
    pause_min_ms: int = 30
    pause_max_ms: int = 80
    duration_min_ms: int = 60        # 轨迹总时长下限
    duration_max_ms: int = 180       # 轨迹总时长上限 (按距离插值)
    min_steps: int = 6               # 最少步数 (避免极短轨迹只走 1~2 步)
    step_interval_ms: int = 8        # 步间隔基线 (ms); 实际按 sigmoid 算

    @classmethod
    def from_dict(cls, d: dict | None) -> "HumanMouseConfig":
        if not d:
            return cls()
        c = cls()
        for k in (
            "bezier_jitter", "jitter_amplitude", "pause_prob",
            "pause_min_ms", "pause_max_ms",
            "duration_min_ms", "duration_max_ms",
        ):
            if k in d:
                setattr(c, k, d[k])
        return c


class HumanMouseController:
    """拟人化鼠标控制器。

    设计: 不缓存任何状态, move_to() 接受当前光标位置 + 目标 + 时长, 独立完成一条轨迹。
    mouse_tracker 每次目标更新都生成一条新轨迹 (而不是单步插值), 让相邻轨迹之间有自然的
    速度断点, 不像"线性插值"那样一眼是机械的。
    """

    def __init__(self, config: HumanMouseConfig | None = None):
        self.cfg = config or HumanMouseConfig()

    def move_to(
        self,
        current_xy: Tuple[int, int],
        target_xy: Tuple[int, int],
        duration_ms: int | None = None,
    ) -> None:
        """从 current 到 target 走一条拟人轨迹 (阻塞 ~duration_ms)。

        Args:
            current_xy: 当前光标屏幕坐标 (x, y)
            target_xy:  目标屏幕坐标 (x, y)
            duration_ms: 轨迹总时长 (ms); None = 按距离在 cfg.duration_min/max_ms 间插值
        """
        cx, cy = current_xy
        tx, ty = target_xy
        dist = math.hypot(tx - cx, ty - cy)

        # 距离 < 2px 直接到位 (避免无意义的贝塞尔生成)
        if dist < 2:
            win32api.SetCursorPos((tx, ty))
            return

        # 时长: 按距离在 [duration_min_ms, duration_max_ms] 间线性插值
        if duration_ms is None:
            # 归一化距离: 50px = 短, 500px = 长
            t = min(1.0, max(0.0, (dist - 50.0) / 450.0))
            duration_ms = int(self.cfg.duration_min_ms
                              + t * (self.cfg.duration_max_ms - self.cfg.duration_min_ms))

        # 步数: 至少 min_steps, 多则按 ~10ms/步算
        n_steps = max(self.cfg.min_steps, duration_ms // max(1, self.cfg.step_interval_ms))
        n_steps = min(n_steps, 50)   # 上限防止卡顿

        # 1. 2 段贝塞尔: start → mid+δ → end, 控制点抖动
        mid_x = (cx + tx) / 2
        mid_y = (cy + ty) / 2
        # 法线方向 (垂直于 start-end) 用于偏移控制点
        dx, dy = tx - cx, ty - cy
        length = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / length, dx / length
        jitter = self.cfg.bezier_jitter
        offset = random.uniform(-jitter, jitter)
        ctrl1 = (mid_x + nx * offset, mid_y + ny * offset)
        # 第二段中点再偏一次 (方向同向, 模拟手抖轨迹方向稳定)
        offset2 = random.uniform(-jitter, jitter) * 0.6
        ctrl2 = (mid_x + dx * 0.25 + nx * offset2,
                 mid_y + dy * 0.25 + ny * offset2)

        # 2. 按等弧长采样 (用固定步数近似)
        points = self._sample_bezier(
            start=(float(cx), float(cy)),
            c1=ctrl1,
            c2=ctrl2,
            end=(float(tx), float(ty)),
            n_steps=n_steps,
        )

        # 3. sigmoid 速度曲线 → 时间戳
        timestamps = self._sigmoid_timestamps(duration_ms, n_steps)

        # 4. 微抖 + 随机停顿
        pause_at = -1
        if random.random() < self.cfg.pause_prob and 2 < n_steps - 3:
            pause_at = random.randint(int(n_steps * 0.3), int(n_steps * 0.7))

        # 5. 执行
        start_t = time.perf_counter()
        for i, (px, py) in enumerate(points):
            # 微抖
            jx = random.uniform(-self.cfg.jitter_amplitude, self.cfg.jitter_amplitude)
            jy = random.uniform(-self.cfg.jitter_amplitude, self.cfg.jitter_amplitude)
            # 终点不抖 (落点要准)
            if i == len(points) - 1:
                jx, jy = 0.0, 0.0
            x = int(round(px + jx))
            y = int(round(py + jy))
            try:
                win32api.SetCursorPos((x, y))
            except Exception as e:
                log.debug(f"[human_mouse] SetCursorPos 失败: {e}")
                return

            # 等到本步时间戳
            target_t = start_t + timestamps[i] / 1000.0
            now = time.perf_counter()
            sleep_s = target_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)

            # 中段停顿
            if i == pause_at:
                pause_ms = random.randint(self.cfg.pause_min_ms, self.cfg.pause_max_ms)
                time.sleep(pause_ms / 1000.0)

    # ── 内部 ──

    @staticmethod
    def _sample_bezier(start, c1, c2, end, n_steps: int):
        """3 阶贝塞尔 (2 控制点), 等参数采样 n_steps+1 个点 (含端点)。"""
        points = []
        for i in range(n_steps + 1):
            t = i / n_steps
            # De Casteljau
            p0 = _lerp(start, c1, t)
            p1 = _lerp(c1, c2, t)
            p2 = _lerp(c2, end, t)
            q0 = _lerp(p0, p1, t)
            q1 = _lerp(p1, p2, t)
            r = _lerp(q0, q1, t)
            points.append(r)
        return points

    @staticmethod
    def _sigmoid_timestamps(duration_ms: int, n_steps: int) -> list[float]:
        """sigmoid 速度曲线: 启动慢 → 中段快 → 收尾慢。

        返回每个采样点的累计时间戳 (ms)。
        """
        # 累积距离用 sigmoid 积分近似 (k=8 给出明显的启动/收尾缓速)
        k = 8.0
        ts = []
        cum_t = 0.0
        prev_v = 0.0
        for i in range(n_steps + 1):
            x = i / n_steps
            # sigmoid(2π * x) 周期为 1, 在 x=0.5 处过 0.5
            v = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
            cum_t += (v + prev_v) / 2.0   # 梯形积分
            prev_v = v
            ts.append(cum_t)
        # 归一化到 duration_ms
        total = ts[-1] if ts[-1] > 0 else 1.0
        return [t * duration_ms / total for t in ts]


def _lerp(a, b, t: float):
    """2 元组线性插值。"""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
