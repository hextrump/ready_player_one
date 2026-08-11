"""
玩家名牌 HSV 定位器 (B 方案)
=============================

替代 NametagLocator (模板匹配) 的轻量版。优点:
- 无需预先截图采集模板
- 多人场景自动选最靠下 (主角一般在地面层)
- 单帧 26ms (CPU), GPU YOLO 更快但需训练 + 域迁移风险
- 不依赖 capture_nametag.py 流程

检测逻辑:
1. HSV mask 蓝徽章 "新手冒险家勋章" (H 88-118, S 40-255, V 150-230)
2. HSV mask 白名牌 "叮咚大狗叫" (H 0-40, S 0-100, V 200-255)
3. 形态学膨胀 + connectedComponents 找候选
4. 配对 (白名在上, 蓝徽在下), 未配对单名视为虚拟徽章
6. 取最靠下 (y 最大) = 主角

接口 (与 NametagLocator 完全一致, 方便替换):
  locate(frame, last_player_pos=(0,0)) -> (px, py, score, ok)
  available: True (无模板依赖)
"""
import cv2
import numpy as np


# ===== 阈值常量 (与 scripts/find_player_nametag.py 同源) =====
BADGE_LOWER = np.array([88, 40, 150])
BADGE_UPPER = np.array([118, 255, 230])
NAME_LOWER = np.array([0, 0, 200])
NAME_UPPER = np.array([40, 100, 255])

ASPECT_MIN, ASPECT_MAX = 3.0, 12.0
HEIGHT_MIN, HEIGHT_MAX = 15, 60
Y_MIN_BOX, Y_MAX_BOX = 200, 720

NAME_GAP_MIN, NAME_GAP_MAX = 5, 30
X_OFFSET_MAX = 30
WIDTH_TOLERANCE = 50

# 名牌 → 角色中心的偏移 (实测: 名牌在头顶偏上方, 角色在名牌下方约 30px)
HEAD_OFFSET_Y = 30

# 占位 score (模板匹配的 0.0~0.1 越低越好, 这里返回 0.05 表示命中, 1.0 表示未命中)
SCORE_OK = 0.05
SCORE_MISS = 1.0
# HUD 判定阈值 (score < 此值算 OK; SCORE_OK=0.05 严格小于 0.5)
NAMETAG_SCORE_OK_THRESHOLD = 0.5


def _find_candidates(mask: np.ndarray) -> list:
    """连通块 + bbox 过滤,返回 [(area, x, y, w, h), ...]"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    ret, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, ret):
        x, y, w, h, area = stats[i]
        if area < 500:
            continue
        ar = w / max(h, 1)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX and HEIGHT_MIN <= h <= HEIGHT_MAX):
            continue
        if not (Y_MIN_BOX <= y <= Y_MAX_BOX):
            continue
        cands.append((area, x, y, w, h))
    return cands


class NametagHSVLocator:
    """HSV 颜色过滤版名牌定位器 (替代 NametagLocator 模板匹配)"""

    def __init__(self):
        self.available = True  # 无模板依赖, 始终可用
        self.last_match_rect = None  # (x, y, w, h)
        self.last_player_pos = None  # (px, py)
        self.last_score = SCORE_MISS

    def locate(self, frame_bgr: np.ndarray, last_player_pos=(0, 0)) -> tuple:
        """
        返回 (player_x, player_y, score, ok).
        ok=False 时 player_x/player_y 仍返回上一次有效位置或 last_player_pos.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_score = SCORE_MISS
            return (last_player_pos[0], last_player_pos[1], SCORE_MISS, False)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        badge_mask = cv2.inRange(hsv, BADGE_LOWER, BADGE_UPPER)
        name_mask = cv2.inRange(hsv, NAME_LOWER, NAME_UPPER)

        badges = _find_candidates(badge_mask)
        names = _find_candidates(name_mask)

        if not badges and not names:
            self.last_match_rect = None
            self.last_score = SCORE_MISS
            return (last_player_pos[0], last_player_pos[1], SCORE_MISS, False)

        # 收集候选 (每个候选含底部 y, 用于选最靠下)
        candidates = []  # [(bottom_y, x, y, w, h), ...]

        # 1) 所有蓝徽章候选
        used_names_idx = set()
        for b_area, bx, by, bw, bh in badges:
            bcx = bx + bw // 2
            paired_idx = None
            for i, (n_area, nx, ny, nw, nh) in enumerate(names):
                if i in used_names_idx: continue
                if ny >= by: continue
                gap = by - (ny + nh)
                if not (NAME_GAP_MIN <= gap <= NAME_GAP_MAX): continue
                ncx = nx + nw // 2
                if abs(bcx - ncx) > X_OFFSET_MAX: continue
                if abs(bw - nw) > WIDTH_TOLERANCE: continue
                paired_idx = i
                break
            candidates.append((by + bh, bx, by, bw, bh))
            if paired_idx is not None:
                used_names_idx.add(paired_idx)

        # 2) 未配对的白名牌 → 虚拟徽章 (玩家被遮但名牌还在)
        for i, (n_area, nx, ny, nw, nh) in enumerate(names):
            if i in used_names_idx: continue
            virtual_y = ny + nh + NAME_GAP_MIN
            candidates.append((virtual_y + 20, nx, virtual_y, nw, 20))

        if not candidates:
            self.last_match_rect = None
            self.last_score = SCORE_MISS
            return (last_player_pos[0], last_player_pos[1], SCORE_MISS, False)

        # 取最靠下 (y 最大) = 主角
        candidates.sort(key=lambda t: -t[0])
        _, x, y, w, h = candidates[0]
        self.last_match_rect = (x, y, w, h)

        # 角色中心 = 徽章底部中点 + (0, HEAD_OFFSET_Y)
        px = x + w // 2
        py = y + h + HEAD_OFFSET_Y
        self.last_player_pos = (px, py)
        self.last_score = SCORE_OK
        return (px, py, SCORE_OK, True)