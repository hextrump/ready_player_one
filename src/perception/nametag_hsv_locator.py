"""
玩家名牌 HSV 定位器 (B 方案)
=============================

定位"自己"的名牌 → 换算出玩家位置 (名牌顶部 = 脚底)。

三条路径, 按可靠度递减:
1. 参考图 (ref): 本模板自带的名牌裁剪图, 在下半屏做尺度自标定的模板匹配。
   这是唯一能回答"这是不是**我**"的路径 —— 其余两条只能回答"这里有个玩家"。
2. 身体几何 (body): v13 Player 框 + 正下方最近的名牌候选 ("脚底 = 名牌顶部")。
   注意 player_body 必须来自独立来源, 不能是当前估计位置 (会自证循环)。
3. 徽章配对 (badge): 蓝徽章 (H 88-118) + 上方白名 (H 0-60) 成对。
   多候选时按锚点就近选; 无锚点才退回"最靠下" (主角一般在画面中下)。

代价 (1600x900, CPU): 有锚点 ~20ms/帧, 丢失后全屏重捕获 ~42ms, 尺度标定 ~0.4s (只在启动/换窗口)。

身份 (identity) 约定 —— 重要:
  参考名牌图**属于玩家模板** (PlayerProfile.identity.nametag_path), 不再读全局
  models/nametag/nametag.png。全局文件会让"换角色忘了换名牌"静默地把别人当成自己。
  定位器只产出**观察 (observation)**, 不做跨帧收敛; 收敛(连续性/速度门控/两帧确认)
  由 PlayerState.observe() 负责 —— 身份是实体的属性, 不是每帧从上下文重新猜的结果。

接口:
  observe(frame, anchor=None, anchor_valid=False, player_body=None) -> NametagObservation
  locate(...) -> (px, py, score, ok)     # 旧四元组, 兼容保留
  available: 是否加载到本模板的参考名牌 (False = 降级到徽章配对/几何反查)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("nametag_hsv_locator")


# ===== 阈值常量 (与 scripts/find_player_nametag.py 同源) =====
BADGE_LOWER = np.array([88, 40, 150])
BADGE_UPPER = np.array([118, 255, 230])
# 名牌文字 HSV 范围 (2026-08 射手村实测: 战士名牌文字是橙棕 H≈43 S≈119, 火枪手是橙黄)
# 原 [0,0,200] / [40,100,255] 只覆盖"白字"; H>40 直接漏 → 玩家定位失败 → bot 不动
# 放宽: H 0-60 覆盖红/橙/黄/棕 (含所有职业名牌), S 0-200 覆盖白+职业色, V 180-255 排除暗背景
NAME_LOWER = np.array([0, 0, 180])
NAME_UPPER = np.array([60, 200, 255])

# 宠物黄框 mask (例 "花蘑菇仔" 黄色矩形) - 用于在白名 mask 里挖掉宠物区
# 防止: 1) 宠物遮挡玩家名 → aspect ratio 异常; 2) 宠物被误识别成白名候选
# 关键修正 (2026-08 射手村实测): 满屏小黄花(树叶/花瓣)用 15x35 全局膨胀会把玩家名牌也吃掉
# → 改成: 只对面积 >= PET_SPRITE_MIN_AREA 的"真宠物精灵"局部膨胀
PET_LOWER = np.array([15, 100, 150])
PET_UPPER = np.array([40, 255, 255])
PET_LOCAL_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 40))  # 单精灵周围膨胀 (覆盖下方名牌)
PET_SPRITE_MIN_AREA = 800   # 真宠物精灵最小面积 (滤树叶/花瓣小杂点; 真"花蘑菇仔"~2500+)

ASPECT_MIN, ASPECT_MAX = 2.5, 12.0   # 原 3.0-12.0; 放宽到 2.5 容忍边缘截断
HEIGHT_MIN, HEIGHT_MAX = 12, 60     # 原 15-60; 放宽到 12 容忍边缘截断
Y_MIN_BOX, Y_MAX_BOX = 200, 600     # 原 720; 收紧到 600 排除底部 HP/MP/EXP 条/聊天区白字

NAME_GAP_MIN, NAME_GAP_MAX = 5, 40   # 原 5, 30; 放宽到 40 容忍小字号/窄名牌 (射手村图2实测 gap≈35)
X_OFFSET_MAX = 30
WIDTH_TOLERANCE = 50

# 宠物名牌二次过滤 (兜底): 任何白名候选如果位于宠物精灵**正下方**且 x 对齐,
# 很可能是宠物名牌 (例 "花蘑菇仔"), 从玩家配对候选里剔除
PET_NAME_DY_MAX = 50      # 宠物名距宠物精灵底部不超过 50px
PET_NAME_DX_MAX = 50      # x 对齐容差

# 名牌 → 玩家脚底的关系 (实测: 名牌挂在角色脚底下方, 脚底正好在名牌上侧边缘)
# 玩家中心 = 名牌顶部 - FEET_TO_CENTER (中心在脚底上方 ~35px, 与 patrol_mover.PLAYER_FOOT_OFFSET 一致)
FEET_TO_CENTER = 35

# 占位 score (模板匹配的 0.0~0.1 越低越好, 这里返回 0.05 表示命中, 1.0 表示未命中)
SCORE_OK = 0.05
SCORE_MISS = 1.0
# HUD 判定阈值 (score < 此值算 OK; SCORE_OK=0.05 严格小于 0.5)
NAMETAG_SCORE_OK_THRESHOLD = 0.5

# ===== 参考图模板匹配 (主路径: 认"自己的名牌", 不依赖徽章) =====
# 新号无徽章 / 蓝地图徽章 mask 噪声大时, 徽章+白名配对不可靠;
# 用玩家自己名牌的裁剪图做模板, 在下半屏多尺度模板匹配, 峰值附近要有白名候选才算命中。
# 参考裁剪由**模板**提供 (data/player/<template>/nametag.png), 顶部=名牌上边≈脚底,
# 由 tools/capture_nametag.py 采集。绝不回落到全局 models/nametag/nametag.png。
REF_MATCH_MIN_SCORE = 0.60   # 真名牌 ~0.60~0.68; 原 0.45 太松, 战士模板(纯黄绿渐变)被树/猪/花瓣误锁
# 尺度不再写死: 它是"这张名牌图 vs 这个窗口"的关系, 启动时全帧扫一次测出来 (见 _calibrate_scale)。
# 写死的常数会在重新采集名牌图后静默失效 —— 实测过一次: 常数 0.85, 真实 1.00, 参考图路径全灭。
REF_CALIB_MIN = 0.55         # 标定搜索的尺度下界
REF_CALIB_MAX = 1.60         # 标定搜索的尺度上界 (1600x900 窗口 + 768 高的参考裁剪 ≈ 1.17)
REF_CALIB_STEPS = 22         # 标定档数 (~0.05 一档, 全帧扫一次约 0.4s, 只在启动/换窗口时做)
REF_RECALIB_AFTER_MISSES = 90  # 连续多少帧参考图不命中就重新标定 (~13s @7fps; 防窗口缩放后失准)
REF_MAX_PEAKS = 6            # 一帧最多取几个"像自己"的峰值 (多人同屏)
# 名牌位置先验: 真玩家名牌通常在画面中下, 远端上方/边界名牌是其他玩家或 UI
REF_PLAYER_Y_MIN = 0.50      # 名牌顶 y 须 ≥ 画面高的 50% (排除树顶/远端/远景误锁; 真玩家永远在中下)
# 【重要】上界: 排除底部 UI (HP/MP/经验条 + 聊天栏)。
# 名牌参考图是"深色横条 + 亮字", 经验条/血条也是深色横条 —— TM_CCOEFF_NORMED 在这种低细节
# 图案上极易在 UI 上打出更高的峰。2026-08-19 实测: 某帧全帧前 6 个峰全在 UI (经验条 0.664 /
# 聊天栏 0.564), 真名牌反而落选; 更糟的是**尺度标定**也被 UI 带偏 (标成 0.65, 真值 1.00),
# 一旦标错, 之后每帧都往经验条上锁 → HUD 上绿框乱飞、玩家十字被拉到屏幕底部。
# 0.78 与 HSV 路径的 Y_MAX_BOX=600 (600/768) 同口径, 两条路径的位置先验保持一致。
REF_PLAYER_Y_MAX = 0.78      # 名牌顶 y 须 ≤ 画面高的 78% (排除 HP/MP/EXP 条与聊天区)
REF_PLAYER_X_MIN = 0.05      # 名牌中心 x 须 ≥ 画面宽 5% (排除最左边缘 UI)
REF_PLAYER_X_MAX = 0.95      # 名牌中心 x 须 ≤ 画面宽 95% (排除最右边缘 UI)
# 多个"像自己"的候选时的取舍: 有可信锚点(上一帧已确认的自己)就选离锚点最近的,
# 没有锚点才退回"最靠下" —— 屏幕里同时站 3 个人时, "最靠下" 每帧都可能换人。
REF_ANCHOR_RADIUS = 220      # 锚点吸附半径 (px): 超出此半径的高分候选不认为是同一个人
SELF_EXCLUDE_RADIUS = 45     # locate_all 剔除"自己"的半径 (px)
REF_VIABLE_RATIO = 0.85      # 候选峰值不低于最高分的此比例才参与取舍


@dataclass
class NametagObservation:
    """一帧的名牌观察 (不含跨帧记忆; 收敛交给 PlayerState.observe)。

    ok:     本帧是否定位到"自己"的名牌
    x, y:   玩家中心屏幕坐标 (名牌顶部 = 脚底, 中心在脚底上方 FEET_TO_CENTER)
    source: 命中路径 "ref"(参考图) / "body"(v13 身体几何反查) / "badge"(徽章配对) / "none"
    score:  旧语义占位分 (0.05=命中, 1.0=未命中), 供 HUD 兼容
    match:  参考图模板匹配峰值 (仅 source="ref" 有意义, 越高越像自己)
    rect:   名牌框 (x, y, w, h), 画 HUD 用
    """
    ok: bool = False
    x: int = 0
    y: int = 0
    source: str = "none"
    score: float = SCORE_MISS
    match: float = 0.0
    rect: tuple | None = None

    # 来源可信度排序 (决定 PlayerState 是否允许大位移直接提交)
    @property
    def trust(self) -> int:
        return {"ref": 3, "body": 2, "badge": 1}.get(self.source, 0)


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


def _find_pet_sprites(pet_mask_raw: np.ndarray) -> list:
    """找宠物精灵 bbox [(x, y, w, h), ...] (黄色连通块, 面积 >= PET_SPRITE_MIN_AREA)。
    用于二次过滤: 宠物名牌在精灵**下方**, 按几何剔除。
    """
    ret, _, stats, _ = cv2.connectedComponentsWithStats(pet_mask_raw, 8)
    out = []
    for i in range(1, ret):
        x, y, w, h, area = stats[i]
        if area < PET_SPRITE_MIN_AREA:
            continue
        out.append((x, y, w, h))
    return out


class NametagHSVLocator:
    """HSV 颜色过滤版名牌定位器 (替代 NametagLocator 模板匹配)"""

    def __init__(self, ref_path: str | None = None, template_name: str = ""):
        """
        Args:
            ref_path: 参考名牌图路径。None → 从当前激活的 PlayerProfile.identity 取
                      (身份属于模板; 绝不回落全局 models/nametag/nametag.png)。
            template_name: 模板名, 仅用于日志。
        """
        self.last_match_rect = None  # (x, y, w, h)
        self.last_player_pos = None  # (px, py)
        self.last_score = SCORE_MISS
        self.last_badge_count = 0   # 本帧检出多少蓝徽章 (0=玩家没蓝徽章, 名牌不可靠)
        self.last_ref_confident = False  # 本帧是否通过参考图模板匹配定位到自己的名牌
        self._last_candidates = []  # 本帧所有名牌候选 [(bottom_y, x, y, w, h)], 供 locate_all 复用
        self._self_rect = None      # 本帧被判定为"自己"的名牌框 (画 HUD 用)
        self._self_pos = None       # 本帧被判定为"自己"的身体位置 (locate_all 用它排除自己)
        self._scale = None          # 参考名牌在当前窗口下的显示尺度 (自动标定)
        self._scale_key = None      # 标定时的帧尺寸 (窗口变了就重标)
        self._ref_misses = 0        # 参考图连续未命中帧数 (到阈值触发重标定)

        if ref_path is None or not template_name:
            try:
                from src.utils.player_profile import get_profile
                p = get_profile()
                ref_path = ref_path or p.identity.nametag_path
                template_name = template_name or p.template
            except Exception as e:
                log.error(f"[NAMETAG] 读取玩家模板身份失败: {e}")

        self.template_name = template_name or "(unknown)"
        self.ref_path = ref_path
        self._ref = None
        if ref_path:
            try:
                img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size:
                    self._ref = img
                    log.info(f"[NAMETAG] 身份名牌已绑定 → 模板 {self.template_name}: "
                             f"{ref_path} ({img.shape[1]}x{img.shape[0]})")
                else:
                    log.error(f"[NAMETAG] 名牌图读取为空: {ref_path}")
            except Exception as e:
                log.error(f"[NAMETAG] 名牌图加载失败 {ref_path}: {e}")
        else:
            log.warning(f"[NAMETAG] ⚠ 模板 {self.template_name} 未绑定名牌图, "
                        f"参考图路径不可用 → 降级为 徽章配对/几何反查")
        # available 语义修正: 定位器本身总能跑 (HSV 路径无模板依赖), 但
        # "认得出自己" 依赖参考图。两者分开, 决策层才能知道自己处于降级模式。
        self.available = True
        self.identity_bound = self._ref is not None

    def _filter_pet_names(self, names: list, pet_mask_raw: np.ndarray) -> list:
        """兜底剔除宠物名牌 (扩张 mask 没覆盖到的白名, 按精灵几何二次过滤)。

        规则: 白名候选如果位于某个宠物精灵**正下方** (y 在精灵底部 0~PET_NAME_DY_MAX 内)
        且 x 中心对齐 (|dx| <= PET_NAME_DX_MAX) → 视为宠物名牌, 剔除。
        """
        sprites = _find_pet_sprites(pet_mask_raw)
        if not sprites:
            return names
        kept = []
        for c in names:
            _, nx, ny, nw, nh = c
            nc_cx = nx + nw // 2
            nc_top = ny
            nc_bot = ny + nh
            is_pet_name = False
            for sx, sy, sw, sh in sprites:
                sprite_bot = sy + sh
                # 名牌顶部必须 > 精灵底部 (名牌在精灵下方)
                if nc_top <= sprite_bot:
                    continue
                # 名牌顶部距精灵底部 ≤ PET_NAME_DY_MAX (不能离太远)
                if nc_top - sprite_bot > PET_NAME_DY_MAX:
                    continue
                # x 中心对齐
                sprite_cx = sx + sw // 2
                if abs(nc_cx - sprite_cx) > PET_NAME_DX_MAX:
                    continue
                is_pet_name = True
                break
            if not is_pet_name:
                kept.append(c)
        return kept

    def _build_local_pet_mask(self, pet_mask_raw: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """只对面积 >= PET_SPRITE_MIN_AREA 的"真宠物精灵"局部膨胀, 避免树上小花瓣误覆盖玩家名牌。

        返回全零矩阵 (无真宠物) 或由"精灵 bbox + 周边 25x40 局部膨胀"组成的稀疏 mask。
        """
        h, w = frame_shape[:2]
        ret, _, stats, _ = cv2.connectedComponentsWithStats(pet_mask_raw, 8)
        if ret <= 1:
            return np.zeros((h, w), dtype=np.uint8)
        local = np.zeros((h, w), dtype=np.uint8)
        big_count = 0
        for i in range(1, ret):
            x, y, bw, bh, area = stats[i]
            if area < PET_SPRITE_MIN_AREA:
                continue
            big_count += 1
            # 局部膨胀: 在精灵 bbox 上下左右各扩一些 (25 宽, 40 高, 下方多一些以覆盖名牌)
            x0 = max(0, x - 5)
            y0 = max(0, y - 10)
            x1 = min(w, x + bw + 20)
            y1 = min(h, y + bh + 35)
            local[y0:y1, x0:x1] = 255
        if big_count == 0:
            return np.zeros((h, w), dtype=np.uint8)
        # 一次小膨胀平滑边缘
        local = cv2.dilate(local, PET_LOCAL_KERNEL, iterations=1)
        return local

    def _calibrate_scale(self, gray: np.ndarray) -> float | None:
        """标定参考名牌在当前窗口下的显示尺度 (全帧多尺度扫一次, 取峰值最高的尺度)。

        为什么要自动标定: 尺度是**参考图与当前窗口的关系**, 是个可以测出来的事实。
        原来它是代码里的常数 (REF_SCALE_BASE=0.85, 按某张旧名牌图标定的) —— 用
        tools/capture_nametag.py 重新采一张名牌后, 这个常数就悄悄失效了:
        实测真实尺度是 1.00 (峰值 0.676), 用 0.85 去匹配只有 0.562, 低于 0.60 阈值,
        于是参考图路径**一次都不命中**, 身份判定静默退化成"选屏幕上最靠下的名牌"
        —— 谁站得低谁就是我。这正是"player 和名牌对应不一致"的根。
        """
        # 只在"名牌可能出现的竖直带"内标定 —— 全帧标定会被底部 UI (经验条/聊天栏) 带偏,
        # 标错尺度会毒化之后每一帧的匹配 (实测: 标成 0.65, 真值 1.00)。
        h = gray.shape[0]
        band = gray[int(h * REF_PLAYER_Y_MIN):int(h * REF_PLAYER_Y_MAX), :]
        best_score, best_scale = 0.0, None
        for i in range(REF_CALIB_STEPS):
            scale = REF_CALIB_MIN + i * (REF_CALIB_MAX - REF_CALIB_MIN) / (REF_CALIB_STEPS - 1)
            t = cv2.resize(self._ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if t.shape[0] >= band.shape[0] or t.shape[1] >= band.shape[1]:
                continue
            mx = float(cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED).max())
            if mx > best_score:
                best_score, best_scale = mx, scale
        if best_scale is None or best_score < REF_MATCH_MIN_SCORE:
            log.warning(f"[NAMETAG] 尺度标定失败 (最佳峰值 {best_score:.3f} < {REF_MATCH_MIN_SCORE}); "
                        f"名牌图可能不是这个角色/这个窗口尺寸的。"
                        f" 重采: python tools/capture_nametag.py --player {self.template_name}")
            return None
        log.info(f"[NAMETAG] 尺度已标定: scale={best_scale:.2f} 峰值={best_score:.3f} "
                 f"(模板 {int(self._ref.shape[1]*best_scale)}x{int(self._ref.shape[0]*best_scale)})")
        return best_scale

    def _scale_for(self, gray: np.ndarray) -> float | None:
        """取当前窗口尺寸对应的模板尺度 (按窗口尺寸缓存; 连续 miss 到阈值就重标定)。"""
        key = gray.shape[:2]
        if key != self._scale_key or (self._ref_misses >= REF_RECALIB_AFTER_MISSES
                                      and self._scale is not None):
            if key != self._scale_key:
                log.info(f"[NAMETAG] 窗口尺寸变化 {self._scale_key} → {key}, 重新标定名牌尺度")
            else:
                log.warning(f"[NAMETAG] 参考图连续 {self._ref_misses} 帧未命中, 重新标定尺度")
            self._scale_key = key
            self._scale = self._calibrate_scale(gray)
            self._ref_misses = 0
        return self._scale

    def _locate_by_reference(self, frame_bgr: np.ndarray, names: list,
                             anchor: tuple | None = None) -> tuple | None:
        """参考图模板匹配定位自己的名牌 (在下半屏整块 ROI 上做一次相关, 取所有峰值)。

        为什么不再以"白名候选"为锚开小窗: 实测名牌文字的 HSV 连通块经常糊成一大片
        (一个候选宽 368px, 把名字/血条/背景连在一起), 真名牌位置附近**根本没有**候选,
        于是小窗永远开在错的地方, 参考图路径 0 命中。整块 ROI 一次相关只要 ~15ms,
        比"开一堆开错地方的小窗"更快也更对。

        返回 (score, px, py, top_y, tw, th) 或 None:
          score   模板匹配峰值 (TM_CCOEFF_NORMED)
          px, py  玩家中心 (名牌顶部=脚底, 中心在脚底上方 FEET_TO_CENTER)
          top_y   名牌顶部 y (屏幕坐标, 画框用)
          tw, th  命中模板尺寸 (画框用)
        """
        if self._ref is None:
            return None
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = self._scale_for(gray)
        if scale is None:
            return None

        t = cv2.resize(self._ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        th, tw = t.shape[:2]
        if th >= h or tw >= w:
            return None

        # 搜索范围: 有锚点先搜锚点附近的小窗 (命中率高且便宜), 没命中再退回下半屏全搜。
        # 相机跟着主角走, 名牌基本在上一帧位置附近 —— 小窗把每帧相关的开销砍掉 ~4/5。
        y_lo = max(0, int(h * REF_PLAYER_Y_MIN) - th)
        y_hi = min(h, int(h * REF_PLAYER_Y_MAX) + th)   # 上界排除底部 UI (经验条/聊天栏)
        boxes = []
        if anchor is not None:
            ax, ay = anchor
            bx1 = max(0, int(ax - REF_ANCHOR_RADIUS - tw))
            bx2 = min(w, int(ax + REF_ANCHOR_RADIUS + tw))
            by1 = max(y_lo, int(ay - REF_ANCHOR_RADIUS))
            by2 = min(y_hi, int(ay + REF_ANCHOR_RADIUS + th))
            boxes.append((bx1, by1, bx2, by2))
        boxes.append((0, y_lo, w, y_hi))     # 兜底: 中下带全搜 (不含底部 UI)

        res = x0 = y0 = None
        for bx1, by1, bx2, by2 in boxes:
            roi = gray[by1:by2, bx1:bx2]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            if float(r.max()) >= REF_MATCH_MIN_SCORE:
                res, x0, y0 = r, bx1, by1
                break
        if res is None:
            self._ref_misses += 1
            return None

        # 取所有过阈值的峰值 (贪心非极大抑制): 屏幕上可能有多个"像我的名牌"的位置
        scored = []  # [(score, nametag_bottom_y, px, py_player, top_y, tw, th)]
        work = res.copy()
        for _ in range(REF_MAX_PEAKS):
            _, mx, _, ml = cv2.minMaxLoc(work)
            if mx < REF_MATCH_MIN_SCORE:
                break
            px = x0 + ml[0] + tw // 2
            top_y = y0 + ml[1]
            # 抑制该峰周围, 再找下一个
            sx1, sy1 = max(0, ml[0] - tw // 2), max(0, ml[1] - th // 2)
            sx2, sy2 = min(work.shape[1], ml[0] + tw // 2), min(work.shape[0], ml[1] + th // 2)
            work[sy1:sy2, sx1:sx2] = -1.0
            if not (REF_PLAYER_X_MIN * w <= px <= REF_PLAYER_X_MAX * w
                    and h * REF_PLAYER_Y_MIN <= top_y <= h * REF_PLAYER_Y_MAX):
                continue
            scored.append((mx, top_y + th, px, top_y - FEET_TO_CENTER, top_y, tw, th))

        if not scored:
            self._ref_misses += 1
            return None
        self._ref_misses = 0
        # 高分候选取舍 (身份连续性优先):
        #   有锚点 (上一帧已确认的"自己") → 选离锚点最近的高分候选。
        #   无锚点 (刚启动/刚丢失)       → 退回"最靠下" (主角一般在画面中下)。
        # 只按"最靠下"选会在多人同屏时每帧换人 —— 那是把身份交给上下文, 不是状态。
        top_score = max(s[0] for s in scored)
        viable = [s for s in scored if s[0] >= top_score * REF_VIABLE_RATIO]
        pick = None
        if anchor is not None:
            near = [s for s in viable
                    if math.hypot(s[2] - anchor[0], s[3] - anchor[1]) <= REF_ANCHOR_RADIUS]
            if near:
                pick = min(near, key=lambda s: math.hypot(s[2] - anchor[0], s[3] - anchor[1]))
        if pick is None:
            pick = max(viable, key=lambda s: s[1])  # ny+nh 大者更靠下
        score, _, px, py, top_y, tw, th = pick
        return (score, px, py, top_y, tw, th)

    def locate(self, frame_bgr: np.ndarray, last_player_pos=(0, 0), player_body: tuple | None = None) -> tuple:
        """旧四元组接口 (兼容): 返回 (player_x, player_y, score, ok)。新代码请用 observe()。"""
        obs = self.observe(frame_bgr, anchor=None, player_body=player_body)
        if not obs.ok:
            return (last_player_pos[0], last_player_pos[1], obs.score, False)
        return (obs.x, obs.y, obs.score, True)

    def observe(self, frame_bgr: np.ndarray, anchor: tuple | None = None,
                player_body: tuple | None = None) -> NametagObservation:
        """产出本帧的名牌观察 (不做跨帧收敛)。

        决策链 (按可靠度递减):
        1. 参考图模板匹配 (自己的名牌 = 自己; 多候选时用 anchor 保持身份连续)
        2. v13/Player 角色框 + 正下方最近名牌候选 (走"脚底 = 名牌顶部"几何)
        3. 徽章+白名成对检测 (须真徽章)
        都没命中 → ok=False (不硬猜)。

        Args:
            anchor: 上一帧**已确认**的玩家中心 (px, py)。仅用于在多个同分候选中保持
                    身份连续性; 传 None 表示当前身份已丢失, 允许重新捕获。
            player_body: 玩家身体中心 (px, py)。**必须来自独立来源** (v13 Player 检测),
                    不能传"当前估计位置" —— 那会自证循环: 估计漂到哪就在哪确认一个名牌,
                    位置永远回不来 (2026-08 的老 bug)。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_score = SCORE_MISS
            self.last_ref_confident = False
            self._self_rect = None
            self._self_pos = None
            return NametagObservation()

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        name_mask = cv2.inRange(hsv, NAME_LOWER, NAME_UPPER)  # 不原地改 — 多路径复用
        names_raw = _find_candidates(name_mask)
        self._last_candidates = [(ny + nh, nx, ny, nw, nh) for _, nx, ny, nw, nh in names_raw]

        # ── 主路径: 参考图模板匹配 (只有绑定了本模板的名牌图才走这条) ──
        # 宠物 mask / 徽章连通块只有兜底路径要用, 放到 ref 未命中之后再算 (每帧省 ~30ms):
        # 正常情况下参考图直接命中, 那些慢路径根本不需要跑。
        ref_hit = self._locate_by_reference(frame_bgr, names_raw, anchor) if self._ref is not None else None
        if ref_hit is not None:
            match, px, py, top_y, tw, th = ref_hit
            self.last_ref_confident = True
            rect = (px - tw // 2, top_y, tw, th)
            self.last_match_rect = rect
            self._self_rect = rect
            self._self_pos = (px, py)
            self.last_player_pos = (px, py)
            self.last_score = SCORE_OK
            return NametagObservation(True, px, py, "ref", SCORE_OK, match, rect)
        self.last_ref_confident = False

        # ── 主路径 2: 玩家身体位置 + 正下方最近白名 (无徽章/参考图不准时的核心方案) ──
        # 名牌顶部 ≈ 玩家脚底 (y_body + body_radius); 玩家中心 = 脚底上方 FEET_TO_CENTER
        # body_radius: 实测玩家身体宽高约 70px, 半高 ~35, 加上脚底到名牌的间距 ~15
        if player_body is not None and names_raw:
            bx, by = player_body
            foot_y = by + 35
            best = None
            for _, nx, ny, nw, nh in names_raw:
                # 名牌顶部 ≈ ny, 脚底到名牌顶应 ≤ 35px (正常站姿)
                top_gap = abs(ny - foot_y)
                if top_gap > 50:  # 太远 → 不是我的名牌
                    continue
                # 横向也要对齐
                if abs((nx + nw // 2) - bx) > 60:
                    continue
                if best is None or top_gap < best[0]:
                    best = (top_gap, nx, ny, nw, nh)
            if best is not None:
                _, nx, ny, nw, nh = best
                px = nx + nw // 2
                py = ny - FEET_TO_CENTER
                rect = (nx, ny, nw, nh)
                self.last_match_rect = rect
                self._self_rect = rect
                self._self_pos = (px, py)
                self.last_player_pos = (px, py)
                self.last_score = SCORE_OK
                return NametagObservation(True, px, py, "body", SCORE_OK, 0.0, rect)

        # ── 兜底: 徽章+白名成对 (慢路径, 到这里才算宠物/徽章 mask) ──
        badge_mask = cv2.inRange(hsv, BADGE_LOWER, BADGE_UPPER)
        pet_mask_raw = cv2.inRange(hsv, PET_LOWER, PET_UPPER)
        # 关键: 全局膨胀会把树叶/花瓣黄噪点也铺满 → 改为只对面积 >= PET_SPRITE_MIN_AREA 的
        # "真宠物精灵" bbox 局部膨胀 (覆盖其下方名牌)
        pet_mask = self._build_local_pet_mask(pet_mask_raw, frame_bgr.shape[:2])
        badges = _find_candidates(badge_mask)
        self.last_badge_count = len(badges)
        # 宠物名牌二次过滤: 先用扩张 mask 去宠物区, 再按精灵 bbox 几何剔除下方名牌
        names_pet = _find_candidates(name_mask & ~pet_mask)
        names_pet = self._filter_pet_names(names_pet, pet_mask_raw)

        candidates = []
        used_names_idx = set()
        for b_area, bx, by, bw, bh in badges:
            bcx = bx + bw // 2
            for i, (n_area, nx, ny, nw, nh) in enumerate(names_pet):
                if i in used_names_idx:
                    continue
                if ny >= by:
                    continue
                gap = by - (ny + nh)
                if not (NAME_GAP_MIN <= gap <= NAME_GAP_MAX):
                    continue
                ncx = nx + nw // 2
                if abs(bcx - ncx) > X_OFFSET_MAX:
                    continue
                if abs(bw - nw) > WIDTH_TOLERANCE:
                    continue
                candidates.append((by + bh, bx, by, bw, bh))
                used_names_idx.add(i)
                break

        # ── 兜底2: 徽章上方局部找名字 (战士橙棕名牌碎片化, 全局候选不出来) ──
        # 既然徽章检到了, 玩家名牌必须在徽章正上方 (y < by, x 对齐) — 在该 ROI 内做
        # 横向亮像素扫描, 不依赖碎片连通块, 直接算名字 bbox
        if not candidates and badges:
            for b_area, bx, by, bw, bh in badges:
                bcx = bx + bw // 2
                # 扫描徽章上方 35px 高度, 横向 +/- 100px (战士名牌实测比徽章宽 ~80px)
                y_top = max(0, by - 35)
                y_bot = by
                x_left = max(0, bx - 80)
                x_right = min(frame_bgr.shape[1], bx + bw + 100)
                roi_name = name_mask[y_top:y_bot, x_left:x_right]
                if roi_name.sum() < 200:  # 太暗/空
                    continue
                # 找 ROI 内每行最左/最右亮像素 → 计算文字左右边界
                rows_with_text = []
                for ry in range(roi_name.shape[0]):
                    row = roi_name[ry]
                    xs = np.where(row > 0)[0]
                    if len(xs) >= 3:  # 一行至少 3 个亮像素才算文字行
                        rows_with_text.append((y_top + ry, x_left + xs[0], x_left + xs[-1]))
                if len(rows_with_text) < 3:
                    continue
                # 文字 y 范围
                y_min = rows_with_text[0][0]
                y_max = rows_with_text[-1][0]
                # 文字 x 范围 (取所有行的最左/最右)
                x_min = min(r[1] for r in rows_with_text)
                x_max = max(r[2] for r in rows_with_text)
                name_w = x_max - x_min
                name_h = y_max - y_min + 1
                if name_w < 20 or name_h < 8:
                    continue
                # x 中心对齐徽章
                name_cx = (x_min + x_max) // 2
                if abs(name_cx - bcx) > X_OFFSET_MAX * 2:
                    continue
                candidates.append((by + bh, x_min, y_min, name_w, name_h))
                break

        if not candidates:
            self.last_match_rect = None
            self.last_score = SCORE_MISS
            self._self_rect = None
            self._self_pos = None
            return NametagObservation()

        # 身份连续性: 有锚点先选离锚点最近的徽章候选, 无锚点才退回"最靠下"
        if anchor is not None:
            near = [c for c in candidates
                    if math.hypot((c[1] + c[3] // 2) - anchor[0],
                                  (c[2] - FEET_TO_CENTER) - anchor[1]) <= REF_ANCHOR_RADIUS]
            if near:
                candidates = near
        candidates.sort(key=lambda t: -t[0])
        _, x, y, w, h = candidates[0]
        rect = (x, y, w, h)
        self.last_match_rect = rect
        self._self_rect = rect

        # 名牌顶部 = 玩家脚底; 角色中心 = 脚底上方 FEET_TO_CENTER
        px = x + w // 2
        py = y - FEET_TO_CENTER
        self._self_pos = (px, py)
        self.last_player_pos = (px, py)
        self.last_score = SCORE_OK
        return NametagObservation(True, px, py, "badge", SCORE_OK, 0.0, rect)

    def locate_all(self, exclude_self: bool = True) -> list:
        """返回本帧**其他玩家**的身体位置, 复用 observe 缓存的候选。

        用于滤掉被怪模型误检成怪的玩家: 怪框中心贴近任一其他玩家身体 → 那是人, 不是怪。
        exclude_self=True 时按"本帧被判定为自己的名牌框"精确剔除自己 —— 原来靠
        "距自己 > 67px" 的半径法在名牌抖动时会把自己也算成别人, 于是自己脚边的猪被误滤,
        战士就永远打不到贴身的怪。
        """
        bodies = []
        me = self._self_pos if exclude_self else None
        for _, x, y, w, h in self._last_candidates:
            bx, by = x + w // 2, y - FEET_TO_CENTER
            if me is not None and math.hypot(bx - me[0], by - me[1]) <= SELF_EXCLUDE_RADIUS:
                continue   # 这就是自己的名牌 (三条路径的命中位置口径一致, 按位置剔除最稳)
            bodies.append((bx, by))
        return bodies