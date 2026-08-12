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

    def _find_fill_bbox(self, hsv, bar_type, roi_top, right_limit=None, min_aspect=1.0):
        """找某条的最大彩色填充矩形。
        right_limit: 只搜该 x 左侧 (HP 固定在 MP 左侧, 过滤右侧技能图标)。
        min_aspect: 最低宽高比 (MP 满条用 2.5 过滤方形图标; HP 低血量细条用 1.0 不放宽)。"""
        mask = self._get_color_mask(hsv, bar_type)
        mask[:roi_top, :] = 0
        if right_limit is not None:
            mask[:, right_limit:] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > 8 and ch >= 3 and cw / ch >= min_aspect:
                valid.append((x, y, cw, ch))
        if not valid:
            return None
        return max(valid, key=lambda b: b[2] * b[3])

    def _expand_to_full_bar(self, frame, bar_type, fill_bbox):
        """沿填充纵向范围逐行向两侧扩展, 取最宽的一行作为条的全宽 (填充色 + 浅灰空余)。
        最宽不足 10px → 非条 (图标/误检), 返回 None。"""
        x, y, w, h = fill_bbox
        fh, fw = frame.shape[:2]
        best_w = 0
        best_left = 0
        for ry in range(max(0, y), min(fh - 1, y + h)):
            row = frame[ry]

            def is_bar(px):
                b, g, r = px.astype(int)
                return self._is_fill_pixel(bar_type, b, g, r) or self._is_gray_bar(b, g, r)

            # 在该行内找包含填充的最宽"条像素"段
            cx = min(fw - 1, x + w // 2)
            if not is_bar(row[cx]):
                continue
            left = cx
            while left > 0 and is_bar(row[left]):
                left -= 1
            right = cx
            while right < fw - 1 and is_bar(row[right]):
                right += 1
            ww = right - left - 1
            if ww > best_w:
                best_w = ww
                best_left = left + 1
        if best_w < 10:
            return None
        return (best_left, y, best_w, h)

    def calibrate(self, frame: np.ndarray) -> bool:
        """找 HP/MP 条并扩展到全宽。以 MP 蓝条为锚, HP 红条只在 MP 左侧同行找。
        放宽到低血量小填充 (细条也能命中); 扩展后过窄 → 视为非条, 失败待下帧重试。"""
        h_img = frame.shape[0]
        roi_top = int(h_img * 0.85)  # 条在底部 UI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # MP 先找 (满条宽扁, 用高宽高比过滤方形技能图标)
        mp_fill = self._find_fill_bbox(hsv, 'MP', roi_top, min_aspect=2.5)
        # HP 固定在 MP 左侧同行, 不限宽高比 (低血量时是细条)
        hp_fill = self._find_fill_bbox(hsv, 'HP', roi_top, right_limit=mp_fill[0]) if mp_fill else None

        layout_ok = (
            mp_fill is not None and hp_fill is not None
            and hp_fill[0] < mp_fill[0]                    # HP 在 MP 左
            and abs(hp_fill[1] - mp_fill[1]) < 25          # 同一行
        )

        if layout_ok:
            self.hp_bbox = self._expand_to_full_bar(frame, 'HP', hp_fill)
            self.mp_bbox = self._expand_to_full_bar(frame, 'MP', mp_fill)
            self.exp_bbox = (0, 0, 0, 0)
            if self.hp_bbox is None or self.mp_bbox is None:
                self.is_calibrated = False
                log.warning(f"HP/MP 校准失败 (条扩展异常): HP={self.hp_bbox} MP={self.mp_bbox}")
            else:
                self.is_calibrated = True
                log.info(f"HP/MP 校准成功: HP全宽={self.hp_bbox} MP全宽={self.mp_bbox}")
        else:
            self.is_calibrated = False
            log.warning(f"HP/MP 校准失败 (需底部UI两条条可见且 HP在MP左): HP={hp_fill} MP={mp_fill}")
        return self.is_calibrated

    def _read_bar(self, frame: np.ndarray, bbox: tuple, bar_type: str) -> float:
        """整条区域读数: 填充比 = 彩色 / (彩色 + 浅灰空余), 对中心行被遮挡/空心更鲁棒。"""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        if w <= 2 or y >= fh or x >= fw:
            return 1.0
        y1 = min(fh - 1, y + max(1, h))
        x1 = min(fw, x + w)

        filled = 0
        total = 0
        for yy in range(y, y1):
            for (b, g, r) in frame[yy, x:x1]:
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
