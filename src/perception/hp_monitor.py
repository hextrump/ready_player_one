"""
HP/MP 识别 — 参考 MapleStoryAutoLevelUp (HealthMonitor + get_bar_percent)
================================================================================
关键思想 (借鉴参考项目):
  1. 中心行读数: 取条中央水平线, 填充=彩色像素, 空余=浅灰 (R≈G≈B) 像素
     填充比 = 彩色 / (彩色 + 浅灰)  → 任意血量都准, 无需校准满值
  2. 条的全宽由"彩色填充 + 浅灰空余"的连续段确定, 不依赖固定宽度

适配: 参考项目用白边框条, 我们是"彩色填充 + 浅灰空余"的条, 结构一致故可直接套用。
     喝药只依赖 HP/MP, 校准只要求这两条 (EXP 尽力检测不阻塞)。

接口保持与旧版一致 (VitalStats / read / calibrate / hp_bbox / mp_bbox / last_hp_mask / last_mp_mask),
AutoHealer 与 viz 无需改动。
"""
import time
import os
from dataclasses import dataclass
import numpy as np
import cv2

from src.utils.logger import get_logger

log = get_logger("hp_monitor")

@dataclass
class VitalStats:
    hp_percent: float
    mp_percent: float
    hp_critical: bool
    mp_critical: bool

    @property
    def hp_display(self) -> str:
        return f"{self.hp_percent*100:.2f}%"

    @property
    def mp_display(self) -> str:
        return f"{self.mp_percent*100:.2f}%"


class HPMonitor:
    def __init__(
        self,
        hp_critical_threshold: float = 0.5,
        mp_critical_threshold: float = 0.3,
        **kwargs
    ):
        self.hp_threshold = hp_critical_threshold
        self.mp_threshold = mp_critical_threshold

        self.is_calibrated = False
        self.hp_bbox = (0, 0, 0, 0)
        self.mp_bbox = (0, 0, 0, 0)
        self.exp_bbox = (0, 0, 0, 0)

        # 可视化用 (保持旧接口)
        self.last_hp_mask = None
        self.last_mp_mask = None

    # ── 颜色掩膜: 各条的专属填充色 (HP红 / MP蓝 / EXP黄) ──
    def _get_color_mask(self, img_hsv, bar_type):
        if bar_type == 'HP':
            m1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            m2 = cv2.inRange(img_hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
            return m1 + m2
        elif bar_type == 'MP':
            return cv2.inRange(img_hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        elif bar_type == 'EXP':
            return cv2.inRange(img_hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))
        return None

    def _is_fill_pixel(self, bar_type, b, g, r):
        """是否该条的填充色 (彩色填充)。"""
        if bar_type == 'HP':
            return r > 120 and g < 100 and b < 100          # 红
        if bar_type == 'MP':
            return b > 120 and r < 100 and g < 180          # 蓝
        if bar_type == 'EXP':
            return r > 120 and g > 100 and b < 120          # 黄
        return False

    def _is_gray_bar(self, b, g, r):
        """是否条的空余部分 (浅灰, R≈G≈B 且较亮, 参考项目 tolerance 逻辑)。"""
        return abs(r - g) <= 15 and abs(r - b) <= 15 and r >= 120

    def _find_fill_bbox(self, hsv, bar_type, roi_top):
        """找某条的最大彩色填充矩形。"""
        mask = self._get_color_mask(hsv, bar_type)
        mask[:roi_top, :] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > 30 and ch > 2 and cw / ch > 2.0:
                valid.append((x, y, cw, ch))
        if not valid:
            return None
        return max(valid, key=lambda b: b[2] * b[3])

    def _expand_to_full_bar(self, frame, bar_type, fill_bbox):
        """从填充 bbox 出发, 沿中心行向两侧扩展, 得到条的全宽 (填充色 + 浅灰空余)。"""
        x, y, w, h = fill_bbox
        fh, fw = frame.shape[:2]
        cy = min(fh - 1, y + h // 2)
        cx = min(fw - 1, x + w // 2)

        def is_bar(px):
            b, g, r = px.astype(int)
            return self._is_fill_pixel(bar_type, b, g, r) or self._is_gray_bar(b, g, r)

        left = cx
        while left > 0 and is_bar(frame[cy, left]):
            left -= 1
        right = cx
        while right < fw - 1 and is_bar(frame[cy, right]):
            right += 1
        return (left + 1, y, max(1, right - left - 1), h)

    def calibrate(self, frame: np.ndarray) -> bool:
        """找 HP/MP 条并扩展到全宽。只要求这两条 (喝药只依赖它们)。"""
        h_img = frame.shape[0]
        roi_top = int(h_img * 0.88)  # 条在底部 UI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hp_fill = self._find_fill_bbox(hsv, 'HP', roi_top)
        mp_fill = self._find_fill_bbox(hsv, 'MP', roi_top)
        exp_fill = self._find_fill_bbox(hsv, 'EXP', roi_top)

        layout_ok = (
            hp_fill and mp_fill
            and hp_fill[0] < mp_fill[0]                    # HP 在 MP 左
            and abs(hp_fill[1] - mp_fill[1]) < 20          # 同一行
        )

        if layout_ok:
            self.hp_bbox = self._expand_to_full_bar(frame, 'HP', hp_fill)
            self.mp_bbox = self._expand_to_full_bar(frame, 'MP', mp_fill)
            self.exp_bbox = self._expand_to_full_bar(frame, 'EXP', exp_fill) if exp_fill else (0, 0, 0, 0)
            self.is_calibrated = True
            log.info(f"HP/MP 校准成功: HP全宽={self.hp_bbox} MP全宽={self.mp_bbox}")
        else:
            self.is_calibrated = False
            log.warning(f"HP/MP 校准失败 (需底部UI两条条可见且 HP在MP左): HP={hp_fill} MP={mp_fill}")
        return self.is_calibrated

    def _read_bar(self, frame: np.ndarray, bbox: tuple, bar_type: str) -> float:
        """中心行读数: 填充比 = 彩色 / (彩色 + 浅灰空余), 参考项目 get_bar_percent。"""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        if w <= 2 or h <= 2 or y >= fh or x >= fw:
            return 1.0
        cy = min(fh - 1, y + h // 2)
        x1 = min(fw, x + w)
        line = frame[cy, x:x1]

        filled = 0
        total = 0
        for (b, g, r) in line:
            b, g, r = int(b), int(g), int(r)
            if self._is_fill_pixel(bar_type, b, g, r):
                filled += 1
                total += 1
            elif self._is_gray_bar(b, g, r):
                total += 1
        return (filled / total) if total > 0 else 1.0

    def read(self, frame: np.ndarray) -> VitalStats:
        """读取 HP/MP 百分比。未校准或校准失败 → 返回全满 (不误喝药)。"""
        if not self.is_calibrated:
            self.calibrate(frame)
            if not self.is_calibrated:
                return VitalStats(hp_percent=1.0, mp_percent=1.0,
                                  hp_critical=False, mp_critical=False)

        hp_pct = self._read_bar(frame, self.hp_bbox, 'HP')
        mp_pct = self._read_bar(frame, self.mp_bbox, 'MP')

        hp_pct = max(0.0, min(1.0, hp_pct))
        mp_pct = max(0.0, min(1.0, mp_pct))

        return VitalStats(
            hp_percent=hp_pct,
            mp_percent=mp_pct,
            hp_critical=(hp_pct < self.hp_threshold),
            mp_critical=(mp_pct < self.mp_threshold)
        )
