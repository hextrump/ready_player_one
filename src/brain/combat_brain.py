"""
V8.0 Combat Brain — v19 认怪 + v13 认地形/玩家 + PatrolMover 三层移动
====================================================================

架构 (决策结构参考 MapleStoryAutoLevelUp, 检测用更强的 YOLO V19):

1. 后台视觉线程 (眼手分离): 双模型 ~5.5fps 写共享缓存:
   - v19 (monster_v19.pt) 单类 Monster, 认怪 (含猪, 猪训练)。
   - v13 (super_brain_v13_merged.pt) 提供平台/梯子 (地形) + Player 兜底。
2. 主循环 (决策, 三层优先, 由 PatrolMover 驱动移动):
   - 怪在攻击范围 → 打 (burst 连打 / 跳发补刀)
   - 有"直接可及"的怪 (同面/登台跳/下跳) → 简单接近
   - 否则 → 地形巡逻 (沿平台走, 到边缘爬梯/换向), 怪进范围即打
3. 玩家位置: 名牌定位器锚定 → v13 Player 兜底 → 画面中心猜测 (漏检衰减)。
   v19 玩家误检由重叠面积过滤。
"""
import time
import threading
import math
import random
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

from ultralytics import YOLO

from src.capture.window_capture import WindowCapture
from src.brain.game_controller import GameController, Direction
from src.brain.patrol_mover import PatrolMover
from src.brain.action_executor import ActionExecutor
from src.perception.hp_monitor import HPMonitor
from src.brain.data_collector import DataCollector
from src.perception.nametag_hsv_locator import NametagHSVLocator, NAMETAG_SCORE_OK_THRESHOLD as NAMETAG_MATCH_THRESHOLD
from src.utils.logger import get_logger

log = get_logger("combat_brain")


# ===== 角色位置估计 =====
# 冒险岛角色始终在画面中间偏下的位置
# 基于 1600x900 分辨率, 角色大约在 (800, 520) 附近 (名牌定位器命中前的初值)
PLAYER_X = 800
PLAYER_Y = 520

# ===== 玩家定位参数 (名牌定位器为主, 此处为合理性门控与漏检衰减) =====
PLAYER_MAX_MOVE_PX = 260.0    # 名牌结果距上一位置的最大可信位移 (防误匹配远处其它玩家)
PLAYER_COMMIT_DIST = 40.0     # 距已确认位置 ≤40px → 直接提交 (正常走动/微移)
PLAYER_CONTINUITY_DIST = 80.0 # 大位移需两帧连续; 走动时每帧约 35px, 80 足够跟上又过滤瞬移
NAMETAG_CONFIDENT_SCORE = 0.08  # 名牌得分 ≤ 该值才可信 (真实命中 0.03~0.09); 平庸得分大位移=疑似别人名牌
PLAYER_MISS_DECAY_FRAMES = 30 # 名牌连续漏检多少帧后开始向画面中心衰减
PLAYER_MISS_DECAY_STEP = 20.0 # 衰减期每帧向中心移动的像素数

ATTACK_RANGE_X = 120  # 水平攻击距离
ATTACK_RANGE_Y = 30   # 垂直容差
JUMP_ATTACK_RANGE_Y_UP = 120 # 跳发最高打击距离

# Burst 连打参数 (拟人化 + 贴合冒险岛真实战斗节奏)
# 冒险岛按一下技能键, 攻击动画 ~0.5~1.5s, 实际命中 1~2 下/秒。
# burst 不应比游戏动画更快 — 更快也无伤害, 还会被 GameGuard 抓节奏特征。
BURST_INTERVAL = 0.65       # 基线 ~1.5 下/秒 (贴近真实战斗节奏)
BURST_JITTER   = 0.20       # 高斯抖动 ±200ms (0.45~0.85s 随机间隔)
BURST_HOLD_MIN = 0.025      # 按键最小保持 25ms
BURST_HOLD_MAX = 0.055      # 按键最大保持 55ms (按稍久更像人)
BURST_PAUSE_EVERY = 5       # 每 ~5 下插入一次"换气停顿"
BURST_PAUSE_MIN  = 0.7      # 停顿最短 0.7s (像技能动画未结束)
BURST_PAUSE_MAX  = 1.5      # 停顿最长 1.5s (像走神/换技能)
BURST_RECHECK = 0.40        # 每 400ms 检查目标是否还活着
BURST_TIMEOUT = 8.0         # 单轮 burst 硬上限 8s (慢节奏下要留够击杀时间)

# 平面地图模式: True = 关闭跳发攻击, 直线平推 (移动层跳跃/爬梯在 PatrolMover 独立控制)
# 非平地(如射手村打猪猪)必须 False → 保留跳跃启发式
FLAT_MODE = False

# 兜底 HP 药水: 每 10 分钟主动按一次 a (防止 auto_healer 漏触发)
HP_POTION_INTERVAL = 600.0  # 10 分钟 = 600 秒

# 定时喂宠物: 每 10 分钟按一次 j
PET_FEED_INTERVAL = 600.0   # 10 分钟 = 600 秒

# 固定游戏窗口客户区尺寸 (参考 MapleStoryAutoLevelUp 的 auto_resize)
# 防窗口过大/出屏导致 "视频超出"; 也让 1600x900 坐标常量重新生效。改这里即可换尺寸。
RESIZE_TARGET_CLIENT = (1600, 900)

# ===== 双模型感知 (v19 认怪 + v13 认地形/玩家) =====
MODEL_MONSTER = "models/monster_v19.pt"
MODEL_TERRAIN = "models/super_brain_v13_merged.pt"
IMGSZ_MONSTER = 640   # v19 训练尺寸
IMGSZ_TERRAIN = 640   # v13 地形用 640 已足够 (平台/梯子检出与 960 一致), 省算力保帧率
MONSTER_MIN_SIZE = 30 # 怪物框最小宽/高 (原 20; 打猪时地面掉落物/小杂物误识别多, 调大过滤)
TERRAIN_EVERY = 3     # 地形(平台/梯子)每 N 帧跑一次 (地形基本不动, 中间帧用旧结果省算力)

# 地形过滤阈值 (离线实测校准: 滤掉顶部 UI 噪声与低置信度碎块)
PLATFORM_CONF = 0.4        # 平台最低置信度
PLATFORM_MIN_W = 100       # 平台最小宽度 (px)
PLATFORM_UI_CUT = 60       # 顶部 UI 裁切 (平台 y 下限)
ROPE_CONF = 0.35           # 梯子最低置信度
ROPE_MIN_H = 80            # 梯子最小高度 (px)
ROPE_DEDUP_DX = 12         # 同一梯子 x 聚类去重容差

# v13 Player 兜底 (名牌失效时; 640 下真实玩家 conf≈0.41, 故取 0.4, 用尺寸+距离门限防误锁)
V13_PLAYER_CONF = 0.4
V13_PLAYER_MIN_SIZE = 80   # 玩家框尺寸合理性下限
V13_PLAYER_MAX_SIZE = 200  # 玩家框尺寸合理性上限


class BrainState(Enum):
    STANDBY = "standby"         # 待命状态（仅提供视觉检查）
    SCANNING = "scanning"       # 扫描画面找怪
    APPROACHING = "approaching" # 朝目标移动
    ATTACKING = "attacking"     # 发动攻击
    PATROLLING = "patrolling"   # 无怪时巡逻


@dataclass
class Target:
    name: str
    cx: int       # 怪物中心 X
    cy: int       # 怪物中心 Y
    w: int
    h: int
    conf: float
    dist: float   # 与角色的距离


class CombatBrain:
    def __init__(self):
        # 核心模型: v19 认怪 (猪训练) + v13 认地形/玩家
        self.monster_model = None
        self.terrain_model = None
        try:
            self.monster_model = YOLO(MODEL_MONSTER)
            log.info("成功激活怪模型: " + MODEL_MONSTER)
        except Exception as e:
            log.error(f"怪模型加载失败: {e}")
        try:
            self.terrain_model = YOLO(MODEL_TERRAIN)
            log.info("成功激活地形模型: " + MODEL_TERRAIN)
        except Exception as e:
            log.error(f"地形模型加载失败: {e}")

        # 地形结果缓存 (错帧跳过时复用, 平台/梯子基本不动)
        self._cached_platforms = []
        self._cached_ropes = []
        self._frame_idx = 0  # 感知帧计数 (地形错帧用)

        # 定时心跳截图 (数据收集, 仅常规样本)
        self.data_collector = DataCollector()

        # 多线程视觉缓存 (眼手分离)
        self._vision_lock = threading.Lock()
        self._latest_frame = None
        self._latest_perception = {
            "targets": [],
            "player_x": PLAYER_X,
            "player_y": PLAYER_Y,
            "platforms": [],   # (y, x_left, x_right) 行走面
            "ropes": [],       # (x, y_top, y_bottom) 攀爬
            "fps": 0.0
        }
        self._v13_fallback_first_log = True  # v13 Player 兜底首次接管时打一次日志
        self._prev_gray = None               # 帧间运动检测用上一帧灰度
        self._player_reliable = False  # 本帧玩家位置是否来自可信来源 (名牌/v13), 决定脱困跳是否可信
        self._running = False
        self.state = BrainState.STANDBY
        self.kill_count = 0
        self.active_hunting = False

        # 移动: PatrolMover (打→接近→巡逻 三层)
        self.mover = PatrolMover()

        # 眼手分离: 主循环决策 → 动作执行器 (独立线程) 执行按键
        self.controller = None        # run() 时注入
        self.executor = None          # run() 时创建

        # 兜底定时器节流时间戳
        self._last_hp_potion_time = time.time()  # HP 药水兜底节流时间戳
        self._last_pet_feed_time = time.time()   # 喂宠物节流时间戳

        # 玩家位置视觉惯性缓存
        self._cached_player_pos = (PLAYER_X, PLAYER_Y)
        self._player_miss_frames = 0  # 玩家连续漏检帧计数 (用于位置衰减)
        self._player_pending = None   # 名牌大位移候选 (等两帧确认, 防锁到其它玩家名牌)

        # ── 感知加固: 名牌 HSV 定位器 (无模板, 多人场景选最靠下) ──
        self.nametag_locator = NametagHSVLocator()
        if not self.nametag_locator.available:
            log.warning("[PLAYER] 名牌定位器不可用! 玩家位置将固定在初始猜测")

    def _perception_loop(self, capture: WindowCapture):
        """后台视觉线程：维持一秒看3-5次的高度警觉"""
        log.info("[VISION] 后台视觉线程已启动 (锁定 3-5 FPS)")

        while self._running:
            t0 = time.time()
            try:
                frame = capture.grab()
                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                # 运行核心检测 (双模型: v19 怪每帧 + v13 地形/玩家错帧)
                self._frame_idx += 1
                run_terrain = (self._frame_idx % TERRAIN_EVERY == 0)
                targets, px, py, raw_results, platforms, ropes = self.find_targets(frame, run_terrain)

                # 帧间运动量 (卡住检测): 玩家走动/相机滚动时画面一直变; 卡住时画面静止
                gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray_small, (160, 90))
                if self._prev_gray is not None:
                    motion = float(cv2.absdiff(gray_small, self._prev_gray).mean())
                else:
                    motion = 1.0
                self._prev_gray = gray_small

                # 更新共享缓存
                with self._vision_lock:
                    self._latest_frame = frame.copy()
                    self._latest_perception = {
                        "targets": targets,
                        "player_x": px,
                        "player_y": py,
                        "platforms": platforms,
                        "ropes": ropes,
                        "motion": motion,
                        "fps": 1.0 / (time.time() - t0 + 0.001)
                    }

                # 定时心跳截图 (仅常规样本, 每 save_interval_seconds 秒一帧)
                # 只挂机中采集; 按 F 停止 (active_hunting=False) 后不再自动截图
                if self.active_hunting and raw_results is not None:
                    self.data_collector.maybe_save_heartbeat(frame, raw_results)

                # 控制频率: 去掉 5fps 硬上限, 按单帧实际工作量跑 (~7fps) → 感知更新更快, 攻击反应更快
                elapsed = time.time() - t0
                time.sleep(max(0.005, 0.05 - elapsed))
            except Exception as e:
                # 单帧异常不杀死视觉线程 (否则 bot 变瞎); 记日志并短暂降频重试
                log.error(f"[VISION] 感知线程单帧异常: {e}", exc_info=True)
                time.sleep(0.5)

    def find_targets(self, frame, run_terrain: bool = True) -> tuple[List[Target], int, int, Optional[object], list, list]:
        """双模型感知: 玩家位置名牌锚定(v13 Player 兜底), 怪由 v19 检测,
        地形(平台/梯子)由 v13 提供。返回 (targets, px, py, raw_results, platforms, ropes)。
        run_terrain=False: 跳过地形模型 (错帧), 用缓存的地形结果。"""
        player_x, player_y = self._cached_player_pos
        raw_results = None
        platforms = []     # (y, x_left, x_right) 行走面
        ropes = []         # (x, y_top, y_bottom) 攀爬
        player_cand = None # v13 Player 兜底候选 (cx, cy)

        # ── 玩家位置: 名牌硬比对优先 (锚定静态名牌) ──
        # 防误匹配加固: 距已确认位置的小位移直接提交; 大位移必须连续两帧同位置才提交,
        # 否则会瞬时锁到附近其它玩家的名牌 → 玩家位置错乱 → bot 撞墙卡死。
        matched = False
        pending = False
        if self.nametag_locator.available:
            npx, npy, nscore, nok = self.nametag_locator.locate(frame, self._cached_player_pos)
            if nok:
                dist_c = math.hypot(npx - player_x, npy - player_y)
                if dist_c > PLAYER_MAX_MOVE_PX:
                    pass  # 瞬移太远 → 拒绝 (本轮视为无匹配)
                elif dist_c <= PLAYER_COMMIT_DIST:
                    # 小位移 → 直接确认 (正常走动/静止)
                    player_x, player_y = int(npx), int(npy)
                    self._cached_player_pos = (player_x, player_y)
                    self._player_pending = None
                    self._player_miss_frames = 0
                    matched = True
                elif nscore <= NAMETAG_CONFIDENT_SCORE:
                    # 大位移 + 得分可信 → 两帧连续同位置才确认 (玩家确实走到了这里)
                    if (self._player_pending is not None
                            and math.hypot(npx - self._player_pending[0], npy - self._player_pending[1]) <= PLAYER_CONTINUITY_DIST):
                        player_x, player_y = int(npx), int(npy)
                        self._cached_player_pos = (player_x, player_y)
                        self._player_pending = None
                        self._player_miss_frames = 0
                        matched = True
                    else:
                        # 第一帧 → 进入候选, 等下一帧确认
                        self._player_pending = (int(npx), int(npy))
                        pending = True
                else:
                    # 大位移但得分平庸 → 很可能是其它玩家的名牌, 不锁它 (走漏检衰减兜底)
                    pass

        # ── 地形 + 玩家兜底: v13 模型 (错帧: 每隔 TERRAIN_EVERY 帧跑一次, 中间用缓存) ──
        if self.terrain_model:
            if run_terrain:
                tres = self.terrain_model(frame, conf=0.15, imgsz=IMGSZ_TERRAIN, verbose=False)[0]
                for box in tres.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = tres.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1
                    if name == "Platform" and conf >= PLATFORM_CONF and w >= PLATFORM_MIN_W and y1 >= PLATFORM_UI_CUT:
                        platforms.append(((y1 + y2) // 2, x1, x2))
                    elif name == "Rope" and conf >= ROPE_CONF and h >= ROPE_MIN_H:
                        ropes.append(((x1 + x2) // 2, y1, y2))
                    elif name == "Player" and conf >= V13_PLAYER_CONF:
                        if V13_PLAYER_MIN_SIZE <= w <= V13_PLAYER_MAX_SIZE and V13_PLAYER_MIN_SIZE <= h <= V13_PLAYER_MAX_SIZE:
                            player_cand = ((x1 + x2) // 2, (y1 + y2) // 2)
                # 梯子按 x 聚类去重 (同一梯子 x 抖动 ±12px)
                ropes.sort(key=lambda r: r[0])
                dedup = []
                for r in ropes:
                    if dedup and abs(r[0] - dedup[-1][0]) <= ROPE_DEDUP_DX:
                        continue
                    dedup.append(r)
                ropes = dedup
                self._cached_platforms = platforms
                self._cached_ropes = ropes
            else:
                # 错帧: 复用上次地形结果 (平台/梯子基本不动, 0.3s 内足够移动用)
                platforms = self._cached_platforms
                ropes = self._cached_ropes

        # ── 玩家位置: 名牌 miss → v13 Player 兜底 → 画面中心衰减 ──
        used_v13 = False
        if not matched and not pending:
            if player_cand is not None:
                dist_c = math.hypot(player_cand[0] - player_x, player_cand[1] - player_y)
                if dist_c <= PLAYER_MAX_MOVE_PX:
                    player_x, player_y = player_cand
                    self._cached_player_pos = player_cand
                    self._player_miss_frames = 0
                    self._player_pending = None
                    used_v13 = True
                    if self._v13_fallback_first_log:
                        log.info("[PLAYER] 名牌 miss, 使用 v13 Player 兜底")
                        self._v13_fallback_first_log = False
            if not used_v13:
                # 连续漏检 → 缓慢向画面中心衰减, 避免冻结在陈旧位置
                self._player_miss_frames += 1
                if self._player_miss_frames == PLAYER_MISS_DECAY_FRAMES:
                    log.info("[PLAYER] 名牌连续漏检, 玩家位置开始向画面中心衰减")
                if self._player_miss_frames >= PLAYER_MISS_DECAY_FRAMES:
                    h, w = frame.shape[:2]
                    center = (w // 2, int(h * 0.58))
                    dx, dy = center[0] - player_x, center[1] - player_y
                    dist = math.hypot(dx, dy)
                    if dist > PLAYER_MISS_DECAY_STEP:
                        step = PLAYER_MISS_DECAY_STEP / dist
                        self._cached_player_pos = (
                            int(round(player_x + dx * step)),
                            int(round(player_y + dy * step))
                        )
                    player_x, player_y = self._cached_player_pos

        # 玩家位置可信度: 名牌命中 或 v13 兜底 → 可信; 中心猜测/衰减 → 不可信 (移动层据此关掉假脱困跳)
        self._player_reliable = matched or used_v13

        # 玩家排除区域 (名牌锚定): v19 常把玩家自己误检成 Monster, 用重叠面积过滤
        player_excl = (player_x - 45, player_y - 60, player_x + 45, player_y + 60)

        # ── 怪物检测: v19 专用单类 Monster (猪训练, 召回高) ──
        targets = []
        if self.monster_model:
            raw_results = self.monster_model(frame, conf=0.15, imgsz=IMGSZ_MONSTER, verbose=False)[0]
            for box in raw_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = raw_results.names[int(box.cls[0])]
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1

                if name != "Monster":
                    continue
                # 门槛: 信心度 + 尺寸 (防把地上掉落物当怪)
                if conf < 0.2 or w < MONSTER_MIN_SIZE or h < MONSTER_MIN_SIZE:
                    continue
                # 冲突过滤: 怪框与玩家区域重叠 >30% → 视为玩家自己被误检
                ix1, iy1 = max(player_excl[0], x1), max(player_excl[1], y1)
                ix2, iy2 = min(player_excl[2], x2), min(player_excl[3], y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > 0.3 * (w * h):
                    continue

                targets.append(Target(
                    name=name, cx=cx, cy=cy, w=w, h=h,
                    conf=conf, dist=0.0
                ))

        # 计算距离
        for t in targets:
            t.dist = math.hypot(t.cx - player_x, t.cy - player_y)

        return targets, player_x, player_y, raw_results, platforms, ropes

    def select_target(self, targets: List[Target], player_x: int, player_y: int) -> Optional[Target]:
        """选择最优目标: 优先打已在攻击范围内(含跳发范围)的, 否则最近优先。"""
        if not targets:
            return None

        # 已在攻击范围 → 先杀身边的
        for t in targets:
            if self.is_in_attack_range(t, player_x, player_y):
                return t

        # 否则最近优先
        return min(targets, key=lambda t: t.dist)

    def get_direction_to_target(self, target: Target, player_x: int) -> Direction:
        """判断目标在角色的哪个方向"""
        return Direction.LEFT if target.cx < player_x else Direction.RIGHT

    def is_in_attack_range(self, target: Target, player_x: int, player_y: int, buffer_x: int = 0) -> bool:
        """
        判断目标是否在攻击范围内 (包含跳发攻击判定)。
        buffer_x: 攻击距离缓冲 (正数为增加范围，负数为缩减范围用于防抖)
        """
        dx = abs(target.cx - player_x)
        dy = player_y - target.cy # 正值表示怪在上方

        # 1. 地面普通攻击范围
        if dx <= (ATTACK_RANGE_X + buffer_x) and abs(dy) <= ATTACK_RANGE_Y:
            return True

        # 2. 跳发攻击范围 (怪在头顶上方但稍微出头一点，或者就在上层平台边缘)
        if dx <= (ATTACK_RANGE_X + buffer_x) and 60 < dy <= JUMP_ATTACK_RANGE_Y_UP:
            return True

        return False

    def _any_target_in_range(self, buffer_x: int = 0) -> bool:
        """读视觉缓存, 判断当前画面是否有任何怪进入攻击范围 (替换原 run() 内两处重复闭包)。"""
        with self._vision_lock:
            perc = self._latest_perception
            tgs = perc["targets"]
            px = perc["player_x"]
            py = perc["player_y"]
        return any(self.is_in_attack_range(t, px, py, buffer_x=buffer_x) for t in tgs)

    # ---- PatrolMover 感知接口 (duck-typed) ----

    def any_target_in_range(self, buffer_x: int = -10) -> bool:
        """是否有怪进入攻击范围 (略缩范围防抖)。"""
        return self._any_target_in_range(buffer_x=buffer_x)

    def player_pos(self) -> tuple:
        """当前玩家位置。"""
        with self._vision_lock:
            return (self._latest_perception["player_x"], self._latest_perception["player_y"])

    def nearest_target(self) -> Optional[Target]:
        """最近怪 (巡逻方向偏置用)。"""
        with self._vision_lock:
            perc = self._latest_perception
            tgs = perc["targets"]
            px = perc["player_x"]
            py = perc["player_y"]
        if not tgs:
            return None
        return min(tgs, key=lambda t: math.hypot(t.cx - px, t.cy - py))

    def any_target_near(self, dist: float = 220.0) -> bool:
        """附近 dist 像素内是否有怪 (巡逻时判断要不要普攻, 避免空挥)。"""
        with self._vision_lock:
            perc = self._latest_perception
            tgs = perc["targets"]
            px = perc["player_x"]
            py = perc["player_y"]
        for t in tgs:
            if math.hypot(t.cx - px, t.cy - py) <= dist:
                return True
        return False

    def world_moving(self, threshold: float = 3.0) -> bool:
        """画面是否在变化 (玩家走动/相机滚动/怪移动)。卡在边缘时画面静止 → False。
        threshold 对应 src/brain/patrol_mover.MOTION_MOVING_THRESHOLD, 调低=更易判卡住。"""
        with self._vision_lock:
            m = self._latest_perception.get("motion", 1.0)
        return m > threshold

    def player_reliable(self) -> bool:
        """本帧玩家位置是否可信 (名牌/v13 命中)。不可信时移动层应关闭脱困跳, 避免假卡顿乱跳。"""
        with self._vision_lock:
            return self._player_reliable

    def _attack(self, controller: GameController, target: Target, px: int, py: int,
                cancel=None) -> int:
        """发动攻击: 区分地面 burst 连打和空中跳发补刀。
        cancel: 可选回调, 返回 True 表示决策已变更, 应中止当前攻击。"""
        direction = self.get_direction_to_target(target, px)
        dy = py - target.cy

        if dy > 60 and not FLAT_MODE:
            # 怪物在头顶上方: 跳发补刀 (一次做完)
            log.info(f"↑ 跳跃攻击 -> {target.name}")
            controller.jump_attack(direction)
            hit_count = 1
        else:
            # ===== Burst 连打循环 (提速核心) =====
            # 锁住怪位置用于存活判定, 每 BURST_RECHECK 重新扫一次
            controller.key_down(direction.value)
            target_lock = (target.cx, target.cy)
            start_t = time.time()
            last_attack_t = 0.0
            last_check_t = 0.0
            hit_count = 0

            try:
                # 拟人化: 每次循环重算下一次间隔 (高斯抖动 + 偶发停顿)
                next_interval = max(0.06, random.gauss(BURST_INTERVAL, BURST_JITTER))
                while time.time() - start_t < BURST_TIMEOUT:
                    now = time.time()

                    # 决策已变更 (新目标/换行动) → 中止当前 burst
                    if cancel and cancel():
                        log.info(f"[ATTACK] 决策变更,中止 burst (共 {hit_count} 下)")
                        break

                    # 每 BURST_RECHECK 重新从视觉线程拉最新目标,死了就走
                    if now - last_check_t >= BURST_RECHECK:
                        last_check_t = now
                        with self._vision_lock:
                            cur = self._latest_perception
                            cur_tgs = cur["targets"]
                            cur_px = cur["player_x"]
                            cur_py = cur["player_y"]
                        target_alive = False
                        for t in cur_tgs:
                            if self.is_in_attack_range(t, cur_px, cur_py):
                                # 距锁定位置 100px 内算同一只
                                if abs(t.cx - target_lock[0]) < 100 and abs(t.cy - target_lock[1]) < 80:
                                    target_alive = True
                                    break
                        if not target_alive:
                            log.info(f"[ATTACK] 目标消失/移出范围,提前结束 (共 {hit_count} 下)")
                            break

                    # 按键节奏: 间隔随机化 + 按下保持时长随机化
                    if now - last_attack_t >= next_interval:
                        hold = random.uniform(BURST_HOLD_MIN, BURST_HOLD_MAX)
                        controller.tap_key("x", post_action=False, hold=hold)
                        hit_count += 1
                        last_attack_t = now
                        # 重算下次间隔
                        next_interval = max(0.06, random.gauss(BURST_INTERVAL, BURST_JITTER))
                        # 周期性"思考停顿" (拟人节奏断点)
                        if hit_count % BURST_PAUSE_EVERY == 0:
                            time.sleep(random.uniform(BURST_PAUSE_MIN, BURST_PAUSE_MAX))
                            # 停顿后再来一击, 间隔稍长 (像换气)
                            next_interval = random.uniform(BURST_INTERVAL * 1.4,
                                                           BURST_INTERVAL * 2.2)

                    time.sleep(0.005)
            finally:
                controller.key_up(direction.value)

        self.kill_count += 1
        log.info(f"[ATTACK] {target.name} × {hit_count} 击 @ ({target.cx},{target.cy})")
        return hit_count

    def _decide(self, perc: dict) -> tuple:
        """三层优先决策 (不执行动作, 只返回命令): 打 → 接近(直接可及) → 地形巡逻。"""
        targets = perc["targets"]
        px, py = perc["player_x"], perc["player_y"]
        platforms = perc.get("platforms", [])
        if not self.active_hunting:
            self.state = BrainState.STANDBY
            return ("none", None)
        if targets:
            best = self.select_target(targets, px, py)
            if best and self.is_in_attack_range(best, px, py):
                self.state = BrainState.ATTACKING
                return ("attack", best)
            if best and self.mover.is_reachable(best, px, py, platforms):
                self.state = BrainState.APPROACHING
                return ("approach", best)
        self.state = BrainState.PATROLLING
        return ("patrol", None)

    def _action_worker(self, action_id: int, action: tuple) -> None:
        """动作执行 (ActionExecutor 线程内): 读最新感知, 分发到具体动作; 支持 cancel 中断。"""
        atype, payload = action
        controller = self.controller
        if controller is None:
            return
        cancel = lambda: self.executor.is_cancelled(action_id)
        # 动作开始时读最新感知 (动作内部如需最新数据会自行再读)
        with self._vision_lock:
            perc = self._latest_perception
            px, py = perc.get("player_x", PLAYER_X), perc.get("player_y", PLAYER_Y)
            platforms = perc.get("platforms", [])
            ropes = perc.get("ropes", [])
        try:
            if atype == "attack" and payload is not None:
                self._attack(controller, payload, px, py, cancel)
            elif atype == "approach" and payload is not None:
                self.mover.approach(controller, payload, px, py, platforms, self, cancel)
            elif atype == "patrol":
                self.mover.patrol(controller, px, py, platforms, ropes, self, cancel)
            # "none" 待机: 无需动作
        except Exception as e:
            log.error(f"[EXEC] 动作执行异常: {e}", exc_info=True)

    def run(self, capture: WindowCapture, controller: GameController, hp_monitor: Optional[HPMonitor] = None, show_vision: bool = True):
        """主循环 (眼手分离): 每帧感知 → 决策 → 推送命令 → 画 viz, 不再被动作阻塞 (~10fps)。"""
        self._running = True
        self.controller = controller
        self.capture = capture
        log.info("=== Combat Brain V8 (v19 认怪 + v13 地形 + 地形感知移动) ONLINE ===")
        log.info(f"State: {self.state.value}")

        # 固定窗口客户区尺寸 (参考 MapleStoryAutoLevelUp auto_resize), 防窗口过大/出屏 → "视频超出"
        capture.resize_window(*RESIZE_TARGET_CLIENT)

        if show_vision:
            cv2.namedWindow("Agent V7 Vision", cv2.WINDOW_NORMAL)

        # 启动后台视觉线程 + 动作执行线程 (眼手分离)
        threading.Thread(target=self._perception_loop, args=(capture,), daemon=True).start()
        self.executor = ActionExecutor(self._action_worker)

        while self._running:
            t0 = time.time() # 用于画面 FPS 显示统计
            # 获取最新感知数据 (不阻塞)
            with self._vision_lock:
                perc = self._latest_perception.copy()
                frame = self._latest_frame.copy() if self._latest_frame is not None else None

            if frame is None:
                time.sleep(0.1)
                continue

            targets = perc["targets"]
            px, py = perc["player_x"], perc["player_y"]

            # ===== 兜底 HP 药水: 每 10 分钟主动按一次 a (防止 auto_healer 漏触发) =====
            if self.active_hunting and time.time() - self._last_hp_potion_time >= HP_POTION_INTERVAL:
                log.info("[HP POTION] 兜底触发 10 分钟 HP 药水")
                controller.use_hp_potion()
                self._last_hp_potion_time = time.time()

            # ===== 定时喂宠物: 每 10 分钟按一次 j =====
            if self.active_hunting and time.time() - self._last_pet_feed_time >= PET_FEED_INTERVAL:
                log.info("[PET] 定时喂宠物 (j)")
                controller.tap_key("j")
                self._last_pet_feed_time = time.time()

            # 决策 → 推送动作 (ActionExecutor 内部按粗网格键去重, 动作不重复执行不被打断)
            action = self._decide(perc)
            self.executor.set_action(action)

            # 渲染可视化界面 (每帧, 不再被动作阻塞)
            if show_vision:
                key = self._draw_vision(frame, targets, px, py, t0, hp_monitor)
                if key is not None and (key & 0xFF) == ord('q'):
                    self._running = False
                    break
                self._handle_vision_key(key)

            # 限频 ~10fps (决策循环稳定跑, 感知变更立刻反映到决策)
            elapsed = time.time() - t0
            time.sleep(max(0.005, 0.1 - elapsed))

    def _draw_vision(self, frame, targets, px, py, t0, hp_monitor) -> Optional[int]:
        """绘制 Agent V7 可视化 HUD, 返回 cv2.waitKey(1) 按键值 (供 run() 处理 q/名牌校准)。"""
        vis_frame = frame.copy()

        # 绘制玩家位置 (红色十字, 校准确认用)
        cv2.drawMarker(vis_frame, (px, py), (255, 50, 50),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis_frame, "Player", (px + 15, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)

        # 绘制名牌框 (绿框, 校准确认用)
        nl = self.nametag_locator
        if nl.available and nl.last_match_rect:
            nx, ny, nw, nh = nl.last_match_rect
            cv2.rectangle(vis_frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"NAMETAG {nl.last_score:.2f}", (nx, ny - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制怪物
        for t in targets:
            tx1, ty1 = t.cx - t.w // 2, t.cy - t.h // 2
            tx2, ty2 = t.cx + t.w // 2, t.cy + t.h // 2
            color = (0, 165, 255)
            cv2.rectangle(vis_frame, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(vis_frame, f"{t.name} {t.dist:.0f}px", (tx1, ty1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制地形 (平台橙条 / 梯子黄条, 来自 v13)
        with self._vision_lock:
            plat = self._latest_perception.get("platforms", [])
            rope = self._latest_perception.get("ropes", [])
        for p in plat:
            cv2.rectangle(vis_frame, (p[1], p[0] - 3), (p[2], p[0] + 3), (0, 165, 255), -1)
            cv2.putText(vis_frame, "PLAT", (p[1], p[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        for r in rope:
            cv2.rectangle(vis_frame, (r[0] - 5, r[1]), (r[0] + 5, r[2]), (0, 255, 255), 3)
            cv2.putText(vis_frame, "ROPE", (r[0] - 14, r[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 绘制状态和帧率
        fps = 1.0 / max(0.001, time.time() - t0)
        status_color = (0, 0, 255) if not self.active_hunting else (0, 255, 0)
        status_text = f"State: {self.state.value} | FPS: {fps:.0f} | Kills: {self.kill_count}"
        cv2.putText(vis_frame, "ACTIVE" if self.active_hunting else "STANDBY (Press F1 to Start, F to Stop)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis_frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 名牌校准 HUD
        if nl.available:
            ok_str = "OK" if nl.last_score < NAMETAG_MATCH_THRESHOLD else "MISS"
            cv2.putText(vis_frame, f"NAMETAG {ok_str} score={nl.last_score:.2f}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # HSV 版本无 offset (无模板, 偏移硬编码在 HEAD_OFFSET_Y)
            cv2.putText(vis_frame, "HSV locator (no template needed)",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 处理 HP/MP 显示
        if hp_monitor:
            stats = hp_monitor.read(frame)
            hp, mp = stats.hp_percent * 100, stats.mp_percent * 100
            hp_text = f"HP: {hp:.1f}%" if hp > 0 else "HP: ???%"
            mp_text = f"MP: {mp:.1f}%" if mp > 0 else "MP: ???%"
            cv2.putText(vis_frame, hp_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis_frame, mp_text, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # 在画面上画出监控遮罩，让用户检查
            if hp_monitor.is_calibrated:
                hx, hy, hw, hh = hp_monitor.hp_bbox
                mx, my, mw, mh = hp_monitor.mp_bbox
                cv2.rectangle(vis_frame, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
                cv2.putText(vis_frame, "HP BOX", (hx, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(vis_frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
                cv2.putText(vis_frame, "MP BOX", (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 【响应用户】：视频直接遮罩 (Video direct mask overlay)
                if hasattr(hp_monitor, 'last_hp_mask') and hp_monitor.last_hp_mask is not None:
                    vh, vw = vis_frame.shape[:2]
                    colored_mask = np.zeros_like(vis_frame)
                    hp_m = hp_monitor.last_hp_mask
                    # 安全检查：mask 和 frame 尺寸可能因窗口抖动不一致
                    if hp_m.shape[0] == vh and hp_m.shape[1] == vw:
                        colored_mask[hp_m > 0] = [0, 0, 255]
                        if hasattr(hp_monitor, 'last_mp_mask') and hp_monitor.last_mp_mask is not None:
                            mp_m = hp_monitor.last_mp_mask
                            if mp_m.shape[0] == vh and mp_m.shape[1] == vw:
                                colored_mask[mp_m > 0] = [255, 0, 0]
                        vis_frame = cv2.addWeighted(vis_frame, 0.7, colored_mask, 0.6, 0)

        display = cv2.resize(vis_frame, (1280, 720))
        cv2.imshow("Agent V7 Vision", display)
        return cv2.waitKey(1)

    def _handle_vision_key(self, key: Optional[int]) -> None:
        """HSV 版本无 offset 调节 (无模板依赖), 此方法留空避免战斗中误改。"""
        # 模板匹配版的快捷键已废弃 (改用 HSV 检测, 偏移硬编码在 HEAD_OFFSET_Y)
        return

    def stop(self):
        self._running = False
        if self.executor is not None:
            self.executor.stop()  # 触发当前动作中断
        # 兜底松开可能被按住的键
        if self.controller is not None:
            try:
                self.controller.release_all_key()
            except Exception:
                pass
        # 停止 DXGI 抓帧线程 (释放资源)
        if getattr(self, 'capture', None) is not None:
            try:
                self.capture.stop()
            except Exception:
                pass
        log.info(f"Combat Brain stopped. Total attacks: {self.kill_count}")
        cv2.destroyAllWindows()
