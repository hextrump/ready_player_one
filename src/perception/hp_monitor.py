"""
Precise HSV Color-Masking HP/MP Monitor — V5.1
Pure OpenCV approach: contour-based calibration + pixel-level percentage reading.
No YOLO dependency for UI detection.
"""

import time
import os
from dataclasses import dataclass
import numpy as np
import cv2

from src.utils.logger import get_logger

log = get_logger("hp_monitor_v5")

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
        
        self.hp_template_path = 'data/ui/hp_image.png'
        self.mp_template_path = 'data/ui/mp_image.png'

        self.is_calibrated = False

        # Baseline max values from 100% templates
        self.hp_max_pixels = 0
        self.hp_max_width = 0
        self.mp_max_pixels = 0
        self.mp_max_width = 0
        
        # Absolute screen regions for UI bars
        self.hp_bbox = (0, 0, 0, 0)
        self.mp_bbox = (0, 0, 0, 0)
        
        # We auto-calibrate on init if templates exist
        self._init_calibration()

    def _get_mask(self, img_hsv, bar_type):
        if bar_type == 'HP':
            # Red color ranges (wrap around in HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            return mask1 + mask2
        elif bar_type == 'MP':
            # Blue color ranges
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            return cv2.inRange(img_hsv, lower_blue, upper_blue)
        return None

    def _init_calibration(self):
        """Calibrate maximum pixels/width from the local 100% template images"""
        # HP Calib
        if os.path.exists(self.hp_template_path):
            img = cv2.imread(self.hp_template_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = self._get_mask(hsv, 'HP')
            self.hp_max_pixels = cv2.countNonZero(mask)
            _, _, self.hp_max_width, _ = cv2.boundingRect(mask)
            log.info(f"Loaded HP Template: Max Pix={self.hp_max_pixels}, Max W={self.hp_max_width}")
            
        # MP Calib
        if os.path.exists(self.mp_template_path):
            img = cv2.imread(self.mp_template_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = self._get_mask(hsv, 'MP')
            self.mp_max_pixels = cv2.countNonZero(mask)
            _, _, self.mp_max_width, _ = cv2.boundingRect(mask)
            log.info(f"Loaded MP Template: Max Pix={self.mp_max_pixels}, Max W={self.mp_max_width}")

    def calibrate(self, frame: np.ndarray):
        """
        Dynamically find the bar bounding boxes using HSV color contours.
        Filters by aspect ratio (bars are wide and thin) and only checks 
        the bottom 40% of the screen to avoid locking onto UI buttons.
        """
        h_img, w_img = frame.shape[:2]
        roi_top = int(h_img * 0.6)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # ---------------- HP Bar ----------------
        hp_mask = self._get_mask(hsv, 'HP')
        hp_mask[:roi_top, :] = 0
        
        contours, _ = cv2.findContours(hp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_hp = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch > 0 and (cw / ch) > 3.0 and cw > 50:
                valid_hp.append(c)
                
        if valid_hp:
            largest = max(valid_hp, key=cv2.contourArea)
            self.hp_bbox = cv2.boundingRect(largest)
            log.info(f"Calibrated HP BBox: {self.hp_bbox} (Ratio: {self.hp_bbox[2]/self.hp_bbox[3]:.1f})")
        else:
            log.warning("Failed to find HP bar!")

        # ---------------- MP Bar ----------------
        mp_mask = self._get_mask(hsv, 'MP')
        mp_mask[:roi_top, :] = 0
        
        valid_mp = []
        contours, _ = cv2.findContours(mp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch > 0 and (cw / ch) > 3.0 and cw > 50:
                valid_mp.append(c)
                
        if valid_mp:
            largest = max(valid_mp, key=cv2.contourArea)
            self.mp_bbox = cv2.boundingRect(largest)
            log.info(f"Calibrated MP BBox: {self.mp_bbox} (Ratio: {self.mp_bbox[2]/self.mp_bbox[3]:.1f})")
        else:
            log.warning("Failed to find MP bar!")

        self.is_calibrated = True

    def _read_bar_percentage(self, frame: np.ndarray, bbox: tuple, bar_type: str, max_w: int, max_p: int) -> float:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return 1.0
            
        pad = 5
        y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
        bar_img = frame[y1:y2, x1:x2]
        
        if bar_img.size == 0:
            return 1.0

        hsv = cv2.cvtColor(bar_img, cv2.COLOR_BGR2HSV)
        mask = self._get_mask(hsv, bar_type)
        
        # Save mask globally for visualization overlay
        full_screen_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_screen_mask[y1:y2, x1:x2] = mask
        if bar_type == 'HP':
            self.last_hp_mask = full_screen_mask
        else:
            self.last_mp_mask = full_screen_mask
        
        _, _, current_w, _ = cv2.boundingRect(mask)
        
        if max_w > 0:
            return float(current_w) / float(max_w)
            
        if max_p > 0:
            return float(cv2.countNonZero(mask)) / float(max_p)
            
        return 1.0

    def read(self, frame: np.ndarray) -> VitalStats:
        """
        Extremely fast and precise percentage extraction on CPU.
        """
        if not self.is_calibrated:
            self.calibrate(frame)
            
        hp_pct = self._read_bar_percentage(frame, self.hp_bbox, 'HP', self.hp_max_width, self.hp_max_pixels)
        mp_pct = self._read_bar_percentage(frame, self.mp_bbox, 'MP', self.mp_max_width, self.mp_max_pixels)

        hp_pct = max(0.0, min(1.0, hp_pct))
        mp_pct = max(0.0, min(1.0, mp_pct))

        return VitalStats(
            hp_percent=hp_pct,
            mp_percent=mp_pct,
            hp_critical=(hp_pct < self.hp_threshold),
            mp_critical=(mp_pct < self.mp_threshold)
        )

