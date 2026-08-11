"""
名牌硬比对定位器 — 第二玩家位置传感器 (仅视觉线程调用)
======================================================
用玩家头顶静态名牌做模板匹配, 命中后按记录偏移换算玩家坐标。
参考: zhoufanglu/maplestory-auto-ds  test_title_locator.py (find_pattern_sqdiff + 偏移)

v2 多尺度匹配: 游戏窗口分辨率不稳定 (曾出现 1272x700 / 1366x768 / 2049x1152),
单尺度模板对不上缩放后的名牌。改为按 NAMETAG_SCALES 缩放模板匹配取最优,
分辨率怎么变都能命中。offset 随匹配尺度一起缩放。

注: 早先试过 alen20000 的分片匹配, 但名牌板近乎纯色时单片会在同色区滑动带偏位置,
    故仍用整板匹配 (无歧义); 重度遮挡时分数超阈自然失败, 由 YOLO 兜底。
"""
import json
import os

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("nametag_locator")

# ===== 名牌定位参数 =====
NAMETAG_TEMPLATE_PATH = "models/nametag/nametag.png"
NAMETAG_OFFSET_PATH = "models/nametag/nametag_offset.json"
NAMETAG_MATCH_THRESHOLD = 0.12  # TM_SQDIFF_NORMED 越小越好; 真实命中 0.03~0.09(随背景波动), 假匹配 ≥0.13
# 模板缩放候选 (覆盖窗口从 ~60% 到 ~200% 的大小变化)
NAMETAG_SCALES = (0.5, 0.62, 0.75, 0.85, 1.0, 1.1, 1.25, 1.45, 1.6, 1.8, 2.0)
NAMETAG_LOCAL_RADIUS = 220      # 局部搜索窗口半径 (围绕上帧玩家位置, 覆盖玩家移动+偏移尺度差)
NAMETAG_GLOBAL_RETRY_FRAMES = 10  # 名牌持续缺席时, 全局兜底最多重试几帧 (省算力)


class NametagLocator:
    """名牌模板匹配定位器 (多尺度)。

    locate() 返回 (玩家x, 玩家y, score, ok), ok=True 表示本轮给出一帧可信命中。
    仅由视觉线程调用, 内部状态无需加锁。模板缺失时 available=False 自动回退纯 YOLO。
    """

    def __init__(self, template_path: str = NAMETAG_TEMPLATE_PATH,
                 offset_path: str = NAMETAG_OFFSET_PATH):
        self.available = False
        self.template = None       # 灰度名牌模板
        self.template_w = 0
        self.template_h = 0
        self.offset_x = 0          # 名牌左上角 → 角色中心偏移 (由采集工具记录, 模板原始尺度)
        self.offset_y = 0
        self.last_score = 1.0
        self.last_scale = 1.0      # 上次命中的模板缩放 (尺度缓存, 加速连续帧)
        self.last_match_rect = None  # (x, y, w, h) 最近一次命中框 (调试用)
        self._miss_streak = 0      # 连续未命中帧计数 (用于全局兜底重试节流)
        self._load(template_path, offset_path)

    def _load(self, template_path: str, offset_path: str):
        if not os.path.exists(template_path):
            log.warning(f"名牌模板不存在: {template_path}")
            return
        if not os.path.exists(offset_path):
            log.warning(f"名牌偏移文件不存在: {offset_path}")
            return
        try:
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if self.template is None or self.template.size == 0:
                log.warning(f"名牌模板读取失败: {template_path}")
                return
            with open(offset_path, "r", encoding="utf-8") as f:
                offset = json.load(f)
            self.offset_x = int(offset.get("offset_x", 0))
            self.offset_y = int(offset.get("offset_y", 0))
            self.template_h, self.template_w = self.template.shape[:2]
            self.available = True
            log.info(f"名牌定位器已加载: {template_path} "
                     f"({self.template_w}x{self.template_h}), offset=({self.offset_x},{self.offset_y})")
        except Exception as e:
            log.error(f"名牌定位器加载失败: {e}")
            self.available = False

    def _match_at_scale(self, gray, s, fh, fw, roi=None):
        """按尺度 s 缩放模板做一次匹配, 返回 (score, 左上角, s) 或 None。
        roi = (裁剪图, x0, y0), 给局部搜索用; None 则全图匹配。"""
        tw_s = max(8, int(round(self.template_w * s)))
        th_s = max(8, int(round(self.template_h * s)))
        if fh < th_s or fw < tw_s:
            return None
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        tpl = cv2.resize(self.template, (tw_s, th_s), interpolation=interp)

        if roi is None:
            src, ox, oy = gray, 0, 0
        else:
            src, ox, oy = roi[0], roi[1], roi[2]
        if src.shape[0] < th_s or src.shape[1] < tw_s:
            return None

        res = cv2.matchTemplate(src, tpl, cv2.TM_SQDIFF_NORMED)
        res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
        mn, _, ml, _ = cv2.minMaxLoc(res)
        return mn, (ml[0] + ox, ml[1] + oy), s

    def locate(self, frame, last_player_pos=(0, 0)):
        """在 frame 中多尺度硬比对名牌模板, 返回 (玩家x, 玩家y, score, ok)。

        - score 为 TM_SQDIFF_NORMED 结果, 越小越好;
        - ok=True 表示命中; 未命中时 ok=False 且返回 last_player_pos, 由调用方兜底。
        """
        if not self.available or self.template is None:
            return last_player_pos[0], last_player_pos[1], 1.0, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        # 局部搜索 ROI: 围绕上帧玩家位置, 覆盖玩家移动 + offset*scale 的偏移
        roi = None
        if last_player_pos is not None:
            r = NAMETAG_LOCAL_RADIUS
            x0 = max(0, int(last_player_pos[0]) - r)
            y0 = max(0, int(last_player_pos[1]) - r)
            x1 = min(fw, int(last_player_pos[0]) + r)
            y1 = min(fh, int(last_player_pos[1]) + r)
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                roi = (gray[y0:y1, x0:x1], x0, y0)

        # 窗口已固定尺寸 → 尺度几乎不变: 先只试上次命中尺度的邻域 (局部), 命中即返回 (最快路径)
        near = [self.last_scale * 0.9, self.last_scale, self.last_scale * 1.1]
        for s in near:
            r = self._match_at_scale(gray, s, fh, fw, roi)
            if r and r[0] <= NAMETAG_MATCH_THRESHOLD:
                return self._commit(r)

        # 邻域没中 → 全尺度局部 (仍便宜, ~11ms)
        best = None
        for s in NAMETAG_SCALES:
            r = self._match_at_scale(gray, s, fh, fw, roi)
            if r and (best is None or r[0] < best[0]):
                best = r

        # 局部全没中 → 全局兜底 (慢, 罕见; 只在漏检初期重试, 且只搜局部最优附近的 3 个尺度)
        if (best is None or best[0] > NAMETAG_MATCH_THRESHOLD) \
                and self._miss_streak < NAMETAG_GLOBAL_RETRY_FRAMES:
            probe = [best[2] * 0.9, best[2], best[2] * 1.1] if best is not None else near
            for s in probe:
                r = self._match_at_scale(gray, s, fh, fw, None)
                if r and (best is None or r[0] < best[0]):
                    best = r

        if best is None or best[0] > NAMETAG_MATCH_THRESHOLD:
            self._miss_streak += 1
            self.last_score = best[0] if best is not None else 1.0
            return last_player_pos[0], last_player_pos[1], self.last_score, False

        return self._commit(best)

    def _commit(self, best):
        """命中后的收尾: 更新 last_scale/score/rect, 返回玩家坐标。"""
        self._miss_streak = 0
        score, top_left, s = best
        px = top_left[0] + self.offset_x * s
        py = top_left[1] + self.offset_y * s
        self.last_scale = s
        self.last_score = score
        self.last_match_rect = (top_left[0], top_left[1],
                                int(self.template_w * s), int(self.template_h * s))
        return int(px), int(py), score, True

    def adjust_offset(self, dx: int, dy: int):
        """微调 名牌→玩家 偏移 (可视化校准时用, 主线程调用, 下一帧生效)。"""
        self.offset_x += dx
        self.offset_y += dy
        log.info(f"[NAMETAG] offset 微调 → ({self.offset_x},{self.offset_y})")

    def save_offset(self, path: str = NAMETAG_OFFSET_PATH) -> bool:
        """把当前偏移写回 offset.json (可视化校准后保存)。"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"offset_x": self.offset_x, "offset_y": self.offset_y}, f, indent=2)
            log.info(f"[NAMETAG] offset 已保存 → {path} ({self.offset_x},{self.offset_y})")
            return True
        except Exception as e:
            log.error(f"[NAMETAG] 保存 offset 失败: {e}")
            return False
