"""
V8.0 Combat Brain — 010001010 认怪 + v13 认地形/玩家 + PatrolMover 三层移动
====================================================================

架构 (决策结构参考 MapleStoryAutoLevelUp, 检测用更强的 YOLO):

1. 后台视觉线程 (眼手分离): 双模型 ~7fps 写世界状态 (WorldState):
   - 怪模型 (模板指定: 战士 v19 单类 / 火枪手 010001010 多类) → MonsterTracker (带 id 的实体)
   - v13 (super_brain_v13_merged.pt) → TerrainTracker (平台/梯子, 世界坐标) + Player 兜底
   - 名牌定位器 → PlayerState (身份连续性收敛)
2. 主循环 (决策, 三层优先, 由 PatrolMover 驱动移动):
   - 怪在 engage 射程 → 打 (burst 连打 / 跳发补刀)
   - 有"直接可及"的怪 (同面/登台跳/下跳) → 接近到 engage 距离
   - 否则 → 地形巡逻 (沿平台走, 到边缘爬梯/换向), 怪进范围即打

状态总线 (设计见 design/状态总线.md):
- 每轮决策先 refresh_snapshot() 冻结世界为一个 WorldSnapshot, 决策/执行/移动层共用它。
  同一 seq 内所有判断自洽 —— 杜绝"决策说打得到、执行说够不着"的 0 击空转。
- 射程判定唯一入口 is_in_attack_range(hold=False|True): engage 进入 / hold 保持。
- 状态转换全部走 transition_to(): 合法性表 + 最小驻留 + 账本 (BrainLedger → SQLite)。
"""
import time
import threading
import math
import random
from pathlib import Path
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

from ultralytics import YOLO

from src.capture.window_capture import WindowCapture
from src.brain.game_controller import GameController, Direction
from src.brain.patrol_mover import PatrolMover, MoveKind
from src.brain.action_executor import ActionExecutor
from src.brain.entity_tracker import (WorldState, MonsterTracker, PlayerState, TerrainTracker,
                                      PlayerConfidence, WorldSnapshot)
from src.perception.hp_monitor import HPMonitor
from src.brain.data_collector import DataCollector
from src.perception.nametag_hsv_locator import NametagHSVLocator, NAMETAG_SCORE_OK_THRESHOLD as NAMETAG_MATCH_THRESHOLD
from src.state.ledger import BrainLedger
from src.utils.logger import get_logger
from src.utils.player_profile import get_profile

log = get_logger("combat_brain")


# ===== 角色位置估计 =====
# 冒险岛角色始终在画面中间偏下的位置
# 基于 1600x900 分辨率, 角色大约在 (800, 520) 附近 (名牌定位器命中前的初值)
PLAYER_X = 800
PLAYER_Y = 520

# ===== 玩家定位参数 =====
# 位置收敛的门控 (COMMIT/CONTINUITY/MAX_JUMP) 已下沉到 PlayerState —— 身份属于实体,
# 不再由 combat_brain 用一串散落的 if 现场判定。这里只留感知侧的两个参数。
PLAYER_BODY_RADIUS = 45       # 玩家身体半径: 怪框中心贴近任一"其他玩家身体" → 那是玩家, 滤掉
PLAYER_MISS_DECAY_FRAMES = PlayerState.LOST_FRAMES  # 降级 LOST 后开始向画面中心衰减
PLAYER_MISS_DECAY_STEP = 20.0 # 衰减期每帧向中心移动的像素数

# 攻击范围/burst 节奏/flat_mode/药水间隔 全部由玩家模板 (PlayerProfile) 驱动,
# 见 self.engage_range_x / self.attack_range_* / self.burst_* —— 此处不再放同名常量,
# 否则同一个事实有两个真源, 改了一处另一处静默失效。

# 定时喂宠物: 每 10 分钟按一次 j
PET_FEED_INTERVAL = 600.0   # 10 分钟 = 600 秒

# ===== 接近的时间预算 =====
# "接近该给多久"不是一个常数, 是 距离 ÷ 速度 —— 写死 4 秒等于宣称"所有目标都在 500px 内"。
# 实测远处的高怪要走 451px(中位)~1085px(最大) 才到起跳点, 需要 3.6~8.7s;
# 4 秒看门狗会在半路把它判成"走不到"退回巡逻 —— 于是远处高台上的怪永远打不到。
WALK_SPEED_PX_S = 125.0     # 角色行走速度 (px/s, 1600x900 下实测量级)
APPROACH_BUDGET_BASE = 2.5  # 起步/跳跃/感知延迟的固定开销
APPROACH_BUDGET_SLACK = 1.8 # 路上会被怪打断/绕路, 给的余量
APPROACH_BUDGET_MAX = 15.0  # 上限 (再远也不该无限耗着)

# ===== 相机位移估计 (相位相关; 让地形位姿每帧跟上镜头, 不必每帧跑地形模型) =====
CAM_CROP_TOP = 0.12        # 去掉顶部小地图/任务栏 (UI 不随镜头动, 会把位移拉向 0)
CAM_CROP_BOTTOM = 0.78     # 去掉底部 HP/MP/EXP 条与聊天区 (与名牌 REF_PLAYER_Y_MAX 同口径)
CAM_PC_W, CAM_PC_H = 320, 180   # 相位相关用的缩略图尺寸 (~1ms/帧, 精度约 ±3px 换算回原尺度)
CAM_PC_MIN_RESPONSE = 0.05      # 相关响应低于此 = 画面变化太杂 (换图/大特效), 本帧不猜位移
CAM_MAX_SHIFT_PX = 400          # 单帧位移上限 (超过多半是换图, 交给地形模型绝对匹配纠正)

# 固定游戏窗口客户区尺寸 (参考 MapleStoryAutoLevelUp 的 auto_resize)
# 防窗口过大/出屏导致 "视频超出"; 也让 1600x900 坐标常量重新生效。改这里即可换尺寸。
RESIZE_TARGET_CLIENT = (1600, 900)

# ===== 感知模型 (怪模型由模板指定; v13 固定提供地形/玩家) =====
MODEL_TERRAIN = "models/super_brain_v13_merged.pt"
IMGSZ_TERRAIN = 640   # v13 地形用 640 已足够 (平台/梯子检出与 960 一致), 省算力保帧率
TERRAIN_EVERY = 3     # 地形(平台/梯子)每 N 帧跑一次 (地形基本不动, 中间帧用旧结果省算力)

# 地形过滤阈值 (离线实测校准: 滤掉顶部 UI 噪声与低置信度碎块)
PLATFORM_CONF = 0.4        # 平台最低置信度
PLATFORM_MIN_W = 100       # 平台最小宽度 (px)
PLATFORM_UI_CUT = 60       # 顶部 UI 裁切 (平台 y 下限)
ROPE_CONF = 0.35           # 梯子最低置信度
ROPE_MIN_H = 80            # 梯子最小高度 (px)
ROPE_DEDUP_DX = 12         # 同一梯子 x 聚类去重容差

# v13 Player 兜底 (名牌失效时; 640 下真实玩家 conf≈0.41, 故取 0.4, 用尺寸+位置门限防误锁)
V13_PLAYER_CONF = 0.4
V13_PLAYER_MIN_SIZE = 80   # 玩家框尺寸合理性下限
V13_PLAYER_MAX_SIZE = 200  # 玩家框尺寸合理性上限
V13_PLAYER_Y_MIN = 0.30    # 玩家框中心 y 须 ≥ 画面高 30% (排除顶部 UI/工会/头像误检)
V13_PLAYER_Y_MAX = 0.95    # 玩家框中心 y 须 ≤ 画面高 95% (排除底部 UI 误检)


class BrainState(Enum):
    STANDBY = "standby"         # 待命状态（仅提供视觉检查）
    SCANNING = "scanning"       # 扫描画面找怪 (决策转换的中间态, 强制走查)
    APPROACHING = "approaching" # 朝目标移动
    ATTACKING = "attacking"     # 发动攻击
    PATROLLING = "patrolling"   # 无怪时巡逻


# ===== 状态机约束 (设计思想: 状态机不能被上下文碰瓷) =====
# 显式定义合法转换, 拒绝非法转换 (例如 STANDBY 直接 → ATTACKING 跳过 SCANNING)。
# STANDBY 是**全局 override** (用户按 F / 无药看门狗), 任何状态都能直接进 —— 原来它不在
# ATTACKING 的允许集里, 于是按 F 停手会打出 "非法转换" 告警再绕一圈 SCANNING 才停。
ALLOWED_TRANSITIONS: dict[BrainState, set[BrainState]] = {
    BrainState.STANDBY:      {BrainState.SCANNING, BrainState.PATROLLING},
    BrainState.SCANNING:     {BrainState.APPROACHING, BrainState.ATTACKING,
                              BrainState.PATROLLING},
    BrainState.APPROACHING:  {BrainState.ATTACKING, BrainState.SCANNING,
                              BrainState.PATROLLING},
    BrainState.ATTACKING:    {BrainState.SCANNING, BrainState.APPROACHING,
                              BrainState.PATROLLING},
    BrainState.PATROLLING:   {BrainState.SCANNING, BrainState.APPROACHING,
                              BrainState.ATTACKING},
}

# 各状态超时 (秒): 超过 watchdog 升级到更激进的状态 (防止死循环)
STATE_TIMEOUT_SEC: dict[BrainState, float] = {
    BrainState.SCANNING:     2.0,    # 扫描太久 = 视觉卡了 → 回 STANDBY
    BrainState.APPROACHING:  4.0,    # 接近太久 = 走不到 → 放弃, 重新 PATROL
    BrainState.ATTACKING:    12.0,   # 单只打 12s 还杀不死 → 强制结束 (目标可能漏检)
    BrainState.PATROLLING:   30.0,   # 巡逻 30s 没怪 → 可能走错了, 换向
}

# 最小驻留 (秒): 进入某状态后至少待这么久才允许"降级"离开。
# 决策循环 10Hz, 感知只有 ~7Hz —— 不设下限的话, 一只在射程边界的怪能让
# ATTACKING↔APPROACHING 每 100ms 互翻一次 (实测日志里 30 秒转换 300+ 次), 动作全被打断,
# 表现就是"人一直在原地抽搐"。升级 (→ATTACKING) 不受限, 打怪永远优先。
MIN_DWELL_SEC: dict[BrainState, float] = {
    BrainState.APPROACHING:  0.35,
    BrainState.ATTACKING:    0.40,
    BrainState.PATROLLING:   0.30,
}

# 状态"降级"顺序: 数字越大越激进。只有降级受 MIN_DWELL_SEC 约束。
STATE_RANK: dict[BrainState, int] = {
    BrainState.STANDBY: 0,
    BrainState.SCANNING: 1,
    BrainState.PATROLLING: 2,
    BrainState.APPROACHING: 3,
    BrainState.ATTACKING: 4,
}


@dataclass
class BrainStateCtx:
    """状态机上下文 (设计思想: 账本化, 转换可审计)"""
    state: BrainState
    entered_at: float = 0.0
    attempt_count: int = 0       # 当前 state 内连续失败尝试 (决策未推进) 次数
    last_target_id: int | None = None
    transition_count: int = 0    # 累计转换次数 (调试用)
    last_log_at: float = 0.0     # 上次转换日志时间 (去抖)
    blocked_count: int = 0       # 被最小驻留挡下的降级次数 (诊断抖动强度)
    budget: float = 0.0          # 本次状态的专属超时预算 (0=用 STATE_TIMEOUT_SEC 表)


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
        # ── 玩家模板 (按 config.yaml active_template 加载; 战士/火枪手等可调换) ──
        self.profile = get_profile()
        p = self.profile
        # 攻击范围/按键/节奏 (实例属性, 模板驱动; 默认值 = 旧硬编码常量)
        self.attack_range_x_bullet = p.attack_range_x_bullet      # 远程 b 距离 (melee 模板 = 0 → 决策层只走 melee)
        self.attack_range_x_melee = p.attack_range_x_melee        # 近战 x 距离
        self.engage_range_x = p.engage_range_x                    # 【唯一真源】能打到怪的最大水平距离
        self.attack_range_y = p.combat.attack_range_y             # 垂直容差
        self.attack_range_buffer = p.combat.attack_range_buffer   # 滞回缓冲 (只用于"保持攻击", 不用于"进入攻击")
        self.jump_attack_range_y_up = p.combat.jump_attack_range_y_up
        self.flat_mode = p.combat.flat_mode                       # True=关闭跳发/爬梯启发式
        self.primary_key = p.combat.primary_key                   # 主攻击键 (warrior=x, gunner=b)
        # Burst 节奏 (拟人化 + 贴合冒险岛战斗节奏)
        self.burst_interval = p.combat.burst_interval
        self.burst_jitter = p.combat.burst_jitter
        self.burst_hold_min = p.combat.burst_hold_min
        self.burst_hold_max = p.combat.burst_hold_max
        self.burst_pause_every = p.combat.burst_pause_every
        self.burst_pause_min = p.combat.burst_pause_min
        self.burst_pause_max = p.combat.burst_pause_max
        self.burst_recheck = p.combat.burst_recheck
        self.burst_timeout = p.combat.burst_timeout
        # 兜底药水 (HP 自动喝 + 10 分钟兜底)
        self.hp_potion_interval = p.combat.hp_potion_interval_sec
        # 目标怪类 (模板指定; None=全部 {1..7})
        self.monster_class_ids = p.monster_class_ids
        # 怪识别模型 (模板驱动: 战士→v19 单类, 火枪手→010001010 多类)
        self.monster_model_path = p.combat.monster_model_path
        self.monster_imgsz = p.combat.monster_model_imgsz
        # 模型调用/二次过滤阈值 (v19 单类基线 conf 可能偏低, 模板放宽)
        self.monster_conf = p.combat.monster_conf_threshold   # 模型推理 conf (原始)
        self.monster_filter_conf = p.combat.monster_filter_conf      # 起批门槛 (新目标要这么高)
        self.monster_maintain_conf = p.combat.monster_maintain_conf  # 维持门槛 (已确认目标可以低)
        self.monster_min_size = p.combat.monster_min_size
        self.monster_aspect_min = p.combat.monster_aspect_min
        self.monster_aspect_max = p.combat.monster_aspect_max
        log.info(f"[BRAIN] 已装备模板 {p.template} ({p.char_class}) key={self.primary_key} "
                 f"bullet={self.attack_range_x_bullet} melee={self.attack_range_x_melee} "
                 f"engage={self.engage_range_x} hold={self.engage_range_x + self.attack_range_buffer} "
                 f"flat={self.flat_mode} targets={sorted(self.monster_class_ids)} "
                 f"monster_model={self.monster_model_path} imgsz={self.monster_imgsz}")

        # 核心模型: 怪模型 (模板指定: v19 单类 或 010001010 多类) + v13 认地形/玩家
        self.monster_model = None
        self.terrain_model = None
        try:
            # 把相对路径解析为项目根绝对路径, 模板里写 "models/monster_v19.pt" 也能跑
            from src.utils.config import PROJECT_ROOT
            mm_path = self.monster_model_path
            if not Path(mm_path).is_absolute():
                mm_path = str(PROJECT_ROOT / mm_path)
            self.monster_model = YOLO(mm_path)
            log.info(f"成功激活怪模型: {mm_path} (imgsz={self.monster_imgsz})")
        except Exception as e:
            log.error(f"怪模型加载失败: {e}")
        try:
            self.terrain_model = YOLO(MODEL_TERRAIN)
            log.info("成功激活地形模型: " + MODEL_TERRAIN)
        except Exception as e:
            log.error(f"地形模型加载失败: {e}")

        # 地形持久化: 平台/梯子跨帧融合成状态 (身份+去抖+滤误检), 替换 3 帧缓存
        self.terrain_tracker = TerrainTracker()
        self._frame_idx = 0  # 感知帧计数 (地形错帧用)

        # 定时心跳截图 (数据收集, 仅常规样本)
        self.data_collector = DataCollector()

        # 多线程世界状态 (眼手分离): 感知线程更新 self.world, 决策线程只读它
        self._vision_lock = threading.Lock()
        self._latest_frame = None
        self.world = WorldState(
            player=PlayerState(x=PLAYER_X, y=PLAYER_Y),
            monsters=MonsterTracker(),
        )
        self._v13_fallback_first_log = True  # v13 Player 兜底首次接管时打一次日志
        self._prev_gray = None               # 帧间运动检测用上一帧灰度
        self._prev_pan_gray = None           # 相机位移估计用上一帧缩略灰度
        self._cam_window = cv2.createHanningWindow((CAM_PC_W, CAM_PC_H), cv2.CV_32F)
        self._last_action = None             # 上一帧决策 (滞回用: 锁定目标防反复换)
        self._state_log_count = 0            # 状态机观察日志计数 (每 ~30 帧输出一次)
        self._running = False
        self.state = BrainState.STANDBY
        self.state_ctx = BrainStateCtx(state=BrainState.STANDBY, entered_at=time.time())
        self.kill_count = 0        # 真实击杀 (burst 中目标消失才算)
        self.attack_count = 0      # 攻击轮次 (原来 kill_count 记的其实是这个, 数字虚高)
        self.active_hunting = False
        # 决策/执行共享的最新快照 (单一世界视图; 决策线程写, 执行线程读)
        self._snap: WorldSnapshot | None = None
        self._identity_lost_at = 0.0   # 身份丢失起始时间 (恢复时记账用)

        # 账本: 状态转换/身份丢失/击杀 异步落 SQLite (决策不被 IO 阻塞)
        self.ledger = BrainLedger()

        # 移动: PatrolMover (打→接近→巡逻 三层)
        self.mover = PatrolMover()

        # 眼手分离: 主循环决策 → 动作执行器 (独立线程) 执行按键
        self.controller = None        # run() 时注入
        self.executor = None          # run() 时创建

        # 兜底定时器节流时间戳
        self._last_hp_potion_time = time.time()  # HP 药水兜底节流时间戳
        self._last_pet_feed_time = time.time()   # 喂宠物节流时间戳

        # ── 感知加固: 名牌 HSV 定位器 (参考名牌图由**本模板**提供, 不读全局文件) ──
        self.nametag_locator = NametagHSVLocator(
            ref_path=p.identity.nametag_path, template_name=p.template)
        if not self.nametag_locator.identity_bound:
            log.warning(f"[PLAYER] ⚠ 模板 {p.template} 没有绑定名牌图 → 无法用参考图认出'自己', "
                        f"只能靠 v13 身体几何/徽章配对 (多人同屏时容易认错人)。"
                        f" 采集: python tools/capture_nametag.py --player {p.template}")

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

                # 相机位移 (每帧): 地形模型每 3 帧才跑一次, 位姿却必须每帧跟上镜头,
                # 否则中间帧的平台坐标最多偏 80px (实测), 走路时"到边缘/抓梯子"全判错。
                cam_dx, cam_dy = self._estimate_camera_shift(frame)

                # 提交世界状态 (感知线程单写; 决策线程只通过 snapshot() 读, 锁内一次提交)
                with self._vision_lock:
                    self._latest_frame = frame.copy()
                    # 先把上一帧的实体平移到本帧镜头下, 再拿本帧检测去匹配 (怪和地形同理)
                    self.world.monsters.apply_camera_shift(cam_dx, cam_dy)
                    self.world.monsters.update(targets, strong_conf=self.monster_filter_conf)
                    self.terrain_tracker.apply_camera_shift(cam_dx, cam_dy)
                    self.terrain_tracker.update(platforms, ropes)   # 地形跨帧融合
                    w = self.world
                    for t in w.targets:   # 只暴露本帧观察到的怪 (ghost 不参与决策, 不打空位)
                        t.dist = math.hypot(t.cx - px, t.cy - py)
                    w.platforms = self.terrain_tracker.platforms
                    w.ropes = self.terrain_tracker.ropes
                    w.world_offset = self.terrain_tracker.world_offset
                    w.motion = motion
                    w.fps = 1.0 / (time.time() - t0 + 0.001)
                    w.seq = self._frame_idx

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

    def _estimate_camera_shift(self, frame) -> tuple:
        """用相位相关估算本帧相机位移 (屏幕像素), 返回 (dx, dy)。

        只取"游戏画面区" (去掉顶部小地图/任务栏与底部 HP/EXP/聊天 UI) —— UI 在屏幕上不动,
        算进去会把位移拉向 0。缩略图 320x180 上算, 单帧 ~1ms, 精度约 ±3px (按 5x 放大回原尺度)。
        返回 (0,0) 表示本帧无可信位移 (首帧 / 相关性太低 / 位移超出合理范围)。
        """
        h, w = frame.shape[:2]
        crop = frame[int(h * CAM_CROP_TOP):int(h * CAM_CROP_BOTTOM), :]
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (CAM_PC_W, CAM_PC_H))
        small = small.astype(np.float32)
        prev = self._prev_pan_gray
        self._prev_pan_gray = small
        if prev is None:
            return (0.0, 0.0)
        try:
            (sdx, sdy), response = cv2.phaseCorrelate(prev, small, self._cam_window)
        except cv2.error:
            return (0.0, 0.0)
        if response < CAM_PC_MIN_RESPONSE:
            return (0.0, 0.0)   # 画面变化太杂 (换图/大量特效), 不猜
        dx = sdx * (w / CAM_PC_W)
        dy = sdy * ((h * (CAM_CROP_BOTTOM - CAM_CROP_TOP)) / CAM_PC_H)
        # 合理性: 单帧位移超过这个值多半是换图/闪烁, 交给地形模型的绝对匹配去纠正
        if abs(dx) > CAM_MAX_SHIFT_PX or abs(dy) > CAM_MAX_SHIFT_PX:
            return (0.0, 0.0)
        return (dx, dy)

    def find_targets(self, frame, run_terrain: bool = True) -> tuple[List[Target], int, int, Optional[object], list, list]:
        """双模型感知: 玩家位置名牌锚定(v13 Player 兜底), 怪由 010001010 检测,
        地形(平台/梯子)由 v13 提供。返回 (targets, px, py, raw_results, platforms, ropes)。
        run_terrain=False: 跳过地形模型 (错帧), 用缓存的地形结果。"""
        player = self.world.player
        raw_results = None
        platforms = []     # (y, x_left, x_right) 行走面
        ropes = []         # (x, y_top, y_bottom) 攀爬
        player_cand = None # v13 Player 检测候选 (cx, cy) — 独立于名牌的第二个来源
        v13_players = []   # 本帧所有合格的 v13 Player 框 (多人同屏会有好几个)

        # ── 顺序很重要: 先跑 v13 拿到独立的身体位置, 再用它去几何反查名牌 ──
        # 原来的顺序是反的: 名牌路径拿到的 player_body 恒为 None → 退化成"用当前估计位置
        # 反查名牌", 也就是估计漂到哪就在哪确认一个名牌 → 自证循环, 位置永远回不来。
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
                            cx_v13 = (x1 + x2) // 2
                            cy_v13 = (y1 + y2) // 2
                            # 位置门限: 真实玩家在画面中下, 顶部 UI/工会图标/系统头像误检会被过滤
                            fh = frame.shape[0]
                            if V13_PLAYER_Y_MIN * fh <= cy_v13 <= V13_PLAYER_Y_MAX * fh:
                                v13_players.append((cx_v13, cy_v13))
                # 梯子按 x 聚类去重 (同一梯子 x 抖动 ±12px)
                ropes.sort(key=lambda r: r[0])
                dedup = []
                for r in ropes:
                    if dedup and abs(r[0] - dedup[-1][0]) <= ROPE_DEDUP_DX:
                        continue
                    dedup.append(r)
                ropes = dedup
            # 错帧时 platforms/ropes 保持空, 持久地形由 terrain_tracker 跨帧融合提供

        # ── 玩家位置: 名牌观察 → PlayerState 收敛 (连续性/两帧确认由实体自己把关) ──
        # anchor: 只有身份还在 (非 LOST) 时才给锚点; 丢失后允许自由重新捕获。
        was_lost = player.confidence is PlayerConfidence.LOST
        anchor = None if was_lost else (player.x, player.y)
        # 多人同屏时 v13 会给出好几个 Player 框: 有身份就挑离自己最近的那个,
        # 没身份才退回"最靠下" (原来的写法是"循环里最后一个", 等于随机认一个人)。
        if v13_players:
            if anchor is not None:
                player_cand = min(v13_players,
                                  key=lambda c: math.hypot(c[0] - anchor[0], c[1] - anchor[1]))
            else:
                player_cand = max(v13_players, key=lambda c: c[1])
        obs = self.nametag_locator.observe(frame, anchor=anchor, player_body=player_cand)
        accepted = player.observe(obs.ok, obs.x, obs.y, obs.source)

        # 名牌没认出来 → v13 Player 检测兜底 (独立来源, 同样走实体的连续性门控)
        if not accepted and player_cand is not None:
            accepted = player.observe(True, player_cand[0], player_cand[1], "v13")
            if accepted and self._v13_fallback_first_log:
                log.info("[PLAYER] 名牌 miss, 使用 v13 Player 兜底")
                self._v13_fallback_first_log = False

        if not accepted:
            # 长期漏检 → 缓慢向画面中心衰减, 避免冻结在陈旧位置
            if player.miss_frames == PLAYER_MISS_DECAY_FRAMES:
                log.warning(f"[PLAYER] 身份丢失 (连续 {player.miss_frames} 帧无可信名牌, "
                            f"越界拒绝 {player.rejects} 次) → 位置降级为猜测")
                self._identity_lost_at = time.time()
                self.ledger.identity_lost((player.x, player.y), player.miss_frames, player.rejects)
            if player.miss_frames >= PLAYER_MISS_DECAY_FRAMES:
                h, w = frame.shape[:2]
                player.decay((w // 2, int(h * 0.58)), PLAYER_MISS_DECAY_STEP)
        elif was_lost:
            log.info(f"[PLAYER] 身份重新捕获 @ ({player.x},{player.y}) 来源={player.source}")
            self.ledger.identity_recaptured((player.x, player.y), player.source,
                                            time.time() - (self._identity_lost_at or time.time()))

        player_x, player_y = player.x, player.y

        # 玩家排除区域 (名牌锚定): 怪模型常把玩家自己误检成 Monster, 用重叠面积过滤。
        # 身份丢失时这个坐标只是"画面中心猜测" —— 拿它去挖洞会把中心的真怪一起挖掉,
        # 于是 bot 站在猪群里说"没有怪"。所以只在位置可信时才启用自排除。
        player_excl = None
        if player.confidence is not PlayerConfidence.LOST:
            player_excl = (player_x - 45, player_y - 60, player_x + 45, player_y + 60)

        # 其他玩家身体位置 (本帧名牌候选下方, 已由定位器按"本帧判定为自己的那块"精确排除自己)。
        # 用于滤掉被怪模型误检成怪的"其他玩家"; 主角自己脚边的猪是正当目标, 不能被顺手滤掉。
        player_bodies = self.nametag_locator.locate_all(exclude_self=True)

        # ── 怪物检测: 010001010 多类怪模型 (7 种怪, 类 1-7; Player/Platform/Rope 交给名牌/v13) ──
        targets = []
        if self.monster_model:
            raw_results = self.monster_model(frame, conf=self.monster_conf, imgsz=self.monster_imgsz, verbose=False)[0]
            for box in raw_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_id = int(box.cls[0])
                name = raw_results.names[cls_id]
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1

                if cls_id not in self.monster_class_ids:
                    continue
                # 门槛: 信心度 + 尺寸 (防把地上掉落物当怪) — 模板可调 (战士 v19 单类 conf 可能偏低 → 放宽)
                # 这里用**维持门槛** (低): 起批(新目标)用高门槛, 由 MonsterTracker 把关。
                # 好处: 贴着阈值抖动的怪不会一帧被滤一帧通过, 决策层不再打打停停。
                if conf < self.monster_maintain_conf or w < self.monster_min_size or h < self.monster_min_size:
                    continue
                # 宽高比: 滤窄长条(宠物) 与 过宽(群怪/杂物堆) — 模板可调
                aspect = w / h
                if aspect < self.monster_aspect_min or aspect > self.monster_aspect_max:
                    continue
                # 冲突过滤: 怪框与玩家区域重叠 >30% → 视为玩家自己被误检
                if player_excl is not None:
                    ix1, iy1 = max(player_excl[0], x1), max(player_excl[1], y1)
                    ix2, iy2 = min(player_excl[2], x2), min(player_excl[3], y2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > 0.3 * (w * h):
                        continue
                # 名牌过滤: 怪框中心贴近其他玩家身体 → 那是玩家, 不是怪 (人有名牌, 猪没有)
                if player_bodies:
                    if any(math.hypot(cx - bx, cy - by) <= PLAYER_BODY_RADIUS for bx, by in player_bodies):
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
        """选择最优目标: 已在攻击范围内的取最近的, 否则全场最近的。

        注意: 这里用 **engage** 范围 (不加 buffer)。buffer 是"已经在打的怪掉出射程边缘时
        别急着松手"的滞回, 拿它来挑新目标 = 选一个其实打不到的怪 → 执行层打 0 下 →
        决策再选它 → 空转 (2026-08 日志里 "Monster × 0 击" 刷屏就是这么来的)。
        """
        if not targets:
            return None
        in_range = [t for t in targets if self.is_in_attack_range(t, player_x, player_y)]
        if in_range:
            return min(in_range, key=lambda t: t.dist)
        return min(targets, key=lambda t: t.dist)

    def get_direction_to_target(self, target: Target, player_x: int) -> Direction:
        """判断目标在角色的哪个方向"""
        return Direction.LEFT if target.cx < player_x else Direction.RIGHT

    def is_in_attack_range(self, target: Target, player_x: int, player_y: int,
                           hold: bool = False) -> bool:
        """【攻击范围判定的唯一入口】目标现在打不打得到。

        hold=False (engage): 进入攻击的门槛 —— 决策层选目标、执行层起手都用它。
        hold=True  (hold):   保持攻击的门槛 = engage + attack_range_buffer, 只用于
                             "已经锁定并正在打的这只怪", 防止射程边界抖动打断 burst。

        射程按模板的 engage_range_x (= max(近战, 远程)) 算, 不再由调用方传 key 猜 ——
        以前 key 默认 "bullet", 战士的 bullet 射程是 0, 所以所有忘了传 key 的调用
        (滞回锁定、巡逻的 any_target_in_range) 对战士恒为 False, 静默失效。
        """
        dx = abs(target.cx - player_x)
        dy = player_y - target.cy  # 正值表示怪在上方
        limit = self.engage_range_x + (self.attack_range_buffer if hold else 0)

        # 1. 地面普通攻击范围
        if dx <= limit and abs(dy) <= self.attack_range_y:
            return True

        # 2. 跳发攻击范围 (怪在头顶上方/上层平台边缘) — 只有近战键能跳发
        if (not self.flat_mode) and self.attack_range_x_melee > 0:
            melee_limit = self.attack_range_x_melee + (self.attack_range_buffer if hold else 0)
            if dx <= melee_limit and 60 < dy <= self.jump_attack_range_y_up:
                return True

        return False

    # ---- 世界快照 (决策/执行/移动层的唯一世界视图) ----

    def snapshot(self, max_age_ms: float = 200.0) -> WorldSnapshot:
        """取当前世界快照。max_age_ms 内复用上一份, 避免一次动作里反复取到不同世界。"""
        snap = self._snap
        if snap is not None and snap.age_ms() <= max_age_ms:
            return snap
        return self.refresh_snapshot()

    def refresh_snapshot(self) -> WorldSnapshot:
        """强制重新冻结世界 (决策循环每轮开头、burst 定期 recheck 时调用)。"""
        with self._vision_lock:
            snap = self.world.snapshot()
        self._snap = snap
        return snap

    # ---- PatrolMover 感知接口 (duck-typed; 全部走快照, 不再各自加锁散读) ----

    def any_target_in_range(self, hold: bool = False) -> bool:
        """是否有怪进入攻击范围 (走路途中判断要不要停下来打)。"""
        s = self.snapshot(max_age_ms=120.0)
        return any(self.is_in_attack_range(t, s.px, s.py, hold=hold) for t in s.targets)

    def player_pos(self) -> tuple:
        """当前玩家位置 (屏幕坐标)。"""
        s = self.snapshot(max_age_ms=120.0)
        return (s.px, s.py)

    def player_world(self) -> tuple:
        """当前玩家位置 (**世界坐标**)。

        相机跟着玩家走, 屏幕坐标里玩家几乎不动 —— 巡逻要表达"走到那边平台的尽头"
        这种跨屏目标, 必须用世界坐标, 否则目标点会随镜头一起漂, 永远走不到。
        """
        return self.snapshot(max_age_ms=120.0).player_world

    def world_offset(self) -> tuple:
        """当前相机位姿 (world = screen + offset)。"""
        return self.snapshot(max_age_ms=120.0).world_offset

    def nearest_target(self) -> Optional[Target]:
        """最近怪 (巡逻方向偏置用)。"""
        s = self.snapshot(max_age_ms=120.0)
        if not s.targets:
            return None
        return min(s.targets, key=lambda t: math.hypot(t.cx - s.px, t.cy - s.py))

    def any_target_near(self, dist: float = 360.0) -> bool:
        """附近 dist 像素内是否有怪 (巡逻时判断要不要普攻, 避免空挥)。"""
        s = self.snapshot(max_age_ms=120.0)
        return any(math.hypot(t.cx - s.px, t.cy - s.py) <= dist for t in s.targets)

    def world_moving(self, threshold: float = 3.0) -> bool:
        """画面是否在变化 (玩家走动/相机滚动/怪移动)。卡在边缘时画面静止 → False。
        threshold 对应 src/brain/patrol_mover.MOTION_MOVING_THRESHOLD, 调低=更易判卡住。"""
        return self.snapshot(max_age_ms=120.0).motion > threshold

    def player_reliable(self) -> bool:
        """玩家位置是否可信 (CONFIRMED)。不可信时移动层应关闭脱困跳, 避免假卡顿乱跳。"""
        return self.snapshot(max_age_ms=200.0).player_reliable

    def _attack(self, controller: GameController, target: Target, px: int, py: int,
                cancel=None) -> int:
        """发动攻击: 区分地面远程(b)/近战(x)/跳发(x)补刀。
        cancel: 可选回调, 返回 True 表示决策已变更, 应中止当前攻击。"""
        direction = self.get_direction_to_target(target, px)
        dy = py - target.cy
        dx = abs(target.cx - px)

        # 起手守卫: 决策与执行用同一个 engage 判定, 这里再确认一次。够不着就直接交还,
        # 让决策改走 approach —— 绝不能"够不着也按一下", 那会变成原地空挥的死循环。
        if not self.is_in_attack_range(target, px, py, hold=True):
            log.debug(f"[ATTACK] 起手时目标已出射程 (dx={dx:.0f} > {self.engage_range_x}), 交还决策")
            return 0

        # 选键: 头顶上方 → x 跳发; 远端 → b 远程; 近端 → x 近战
        if dy > 60 and not self.flat_mode:
            attack_key = "x"
            log.info(f"↑ 跳跃攻击 ({attack_key}) -> {target.name} (dx={dx:.0f})")
        elif dx <= self.attack_range_x_melee + self.attack_range_buffer:
            attack_key = self.primary_key  # 近战: 用模板主键 (warrior=x)
            log.info(f"近战 ({attack_key}) -> {target.name} (dx={dx:.0f})")
        else:
            attack_key = self.profile.combat.ranged_key or self.primary_key  # 远程 (warrior 无远程键)
            log.info(f"远程 ({attack_key}) -> {target.name} (dx={dx:.0f})")

        start_all = time.time()
        killed = False
        if dy > 60 and not self.flat_mode:
            # 怪物在头顶上方: 跳发补刀 (一次做完)
            controller.jump_attack(direction)
            hit_count = 1
        else:
            # ===== Burst 连打循环 (提速核心) =====
            # 锁住怪身份用于存活判定 (实体 id, 跨帧稳定; 无 id 回退位置), 每 BURST_RECHECK 重新扫一次
            controller.key_down(direction.value)
            target_id = getattr(target, "id", None)
            target_lock = (target.cx, target.cy)  # 无实体 id 时回退像素锁定
            start_t = time.time()
            last_attack_t = 0.0
            # 首次 recheck 推迟一个周期: 起手守卫刚确认过射程, 立刻再查一次只会在
            # "怪正好走到边缘"时打出 0 击就返回, 又变成空转。保证每轮 burst 至少挥一下。
            last_check_t = start_t
            hit_count = 0

            try:
                # 拟人化: 每次循环重算下一次间隔 (高斯抖动 + 偶发停顿)
                next_interval = max(0.06, random.gauss(self.burst_interval, self.burst_jitter))
                while time.time() - start_t < self.burst_timeout:
                    now = time.time()

                    # 决策已变更 → 中止 burst。区分"目标死了"(正常) vs"真抖动换目标"(问题):
                    # 目标已死 = 决策自然前进, 不算异常 (看门狗不报); 真抖动才报。
                    if cancel and cancel():
                        snap = self.refresh_snapshot()
                        alive = self._find_locked(snap, target_id, target_lock) is not None
                        if alive:
                            log.info(f"[ATTACK] 决策变更,中止 burst (共 {hit_count} 下)")
                        else:
                            log.info(f"[ATTACK] 目标已死/消失,提前结束 (共 {hit_count} 下)")
                            killed = hit_count > 0
                        break

                    # 每 BURST_RECHECK 重新冻结世界拉最新目标, 死了/跑远了就走。
                    # 用 hold 范围 (engage + buffer): 打的过程中怪会走动, 用 engage 严判会
                    # 在边界反复中断 burst; 但"进入攻击"仍然用 engage (见 select_target)。
                    if now - last_check_t >= self.burst_recheck:
                        last_check_t = now
                        snap = self.refresh_snapshot()
                        cur = self._find_locked(snap, target_id, target_lock)
                        if cur is None:
                            log.info(f"[ATTACK] 目标消失 (疑似击杀),结束 (共 {hit_count} 下)")
                            killed = hit_count > 0
                            break
                        if not self.is_in_attack_range(cur, snap.px, snap.py, hold=True):
                            log.info(f"[ATTACK] 目标移出射程,结束 (共 {hit_count} 下)")
                            break
                        target_lock = (cur.cx, cur.cy)   # 跟随怪的位移刷新像素锁

                    # 按键节奏: 间隔随机化 + 按下保持时长随机化
                    if now - last_attack_t >= next_interval:
                        hold = random.uniform(self.burst_hold_min, self.burst_hold_max)
                        controller.tap_key(attack_key, post_action=False, hold=hold)
                        hit_count += 1
                        last_attack_t = now
                        # 重算下次间隔
                        next_interval = max(0.06, random.gauss(self.burst_interval, self.burst_jitter))
                        # 周期性"思考停顿" (拟人节奏断点)
                        if hit_count % self.burst_pause_every == 0:
                            time.sleep(random.uniform(self.burst_pause_min, self.burst_pause_max))
                            # 停顿后再来一击, 间隔稍长 (像换气)
                            next_interval = random.uniform(self.burst_interval * 1.4,
                                                           self.burst_interval * 2.2)

                    time.sleep(0.005)
            finally:
                controller.key_up(direction.value)

        # 计数口径 (账本要说实话): kill_count 只在"burst 中目标消失且我们确实打过"时 +1;
        # 原来每次 _attack 调用都 +1, 包括 0 击空转 —— 于是 HUD 上"击杀 113"其实大半是空挥。
        self.attack_count += 1
        if killed:
            self.kill_count += 1
            self.ledger.kill(target.name, hit_count, time.time() - start_all, self.kill_count)
        log.info(f"[ATTACK] {target.name} × {hit_count} 击 @ ({target.cx},{target.cy})"
                 f"{' [KILL]' if killed else ''}")
        return hit_count

    @staticmethod
    def _find_locked(snap: WorldSnapshot, target_id, target_lock) -> Optional[Target]:
        """在快照里找回"正在打的那只怪": 优先按实体 id (跨帧稳定身份), 无 id 回退像素邻近。"""
        if target_id is not None:
            return snap.target_by_id(target_id)
        for t in snap.targets:
            if abs(t.cx - target_lock[0]) < 100 and abs(t.cy - target_lock[1]) < 80:
                return t
        return None

    def _decide(self, snap: WorldSnapshot) -> tuple:
        """三层优先决策 (只读快照, 不执行动作): 打 → 接近(直接可及) → 地形巡逻。

        输入是**一个快照**: 同一轮决策里所有判断看到的是同一个世界, 不会出现
        "选目标时怪在射程内、执行时又不在" 的自相矛盾。

        承诺 (commitment): 已锁定并正在打的怪, 只要还在 hold 射程内就继续打, 不重新选目标;
        这既是滞回, 也是"状态机不被每帧上下文推翻"的体现。
        """
        if not self.active_hunting:
            self.transition_to(BrainState.STANDBY, reason="用户停手/看门狗")
            self._last_action = None
            return ("none", None)

        # ── Watchdog: 状态超时强制升级 ──
        self._state_watchdog()

        targets = snap.targets
        px, py = snap.px, snap.py

        if targets:
            # 1. 承诺优先: 上一轮在打的那只怪还在 hold 射程内 → 继续打它 (按身份 id 找回)
            locked = None
            last = self._last_action
            if last is not None and last[0] == "attack" and last[1] is not None:
                cand = snap.target_by_id(getattr(last[1], "id", None))
                if cand is not None and self.is_in_attack_range(cand, px, py, hold=True):
                    locked = cand
            if locked is not None:
                self.transition_to(BrainState.ATTACKING, reason="继续打已锁定目标")
                self.state_ctx.last_target_id = locked.id
                self._last_action = ("attack", locked)
                return ("attack", locked)

            # 2. 选新目标: 用 engage 判定 (打得到才叫在范围内)
            best = self.select_target(targets, px, py)
            if best is not None and self.is_in_attack_range(best, px, py):
                self.transition_to(BrainState.ATTACKING, reason="目标进入射程")
                self.state_ctx.last_target_id = getattr(best, "id", None)
                self._last_action = ("attack", best)
                return ("attack", best)

            # 3. 怪在可及范围 (同面/一跳可上/下落/爬梯) → 接近
            # 只算一次 plan: 结论(能不能去)和路程(要走多远)都从它来, 不再分头再推一遍。
            if best is not None:
                plan = self.mover.plan_move(best, px, py,
                                            list(snap.platforms), list(snap.ropes))
                if plan.kind is not MoveKind.UNREACHABLE:
                    tid = getattr(best, "id", None)
                    new_target = (self.state_ctx.last_target_id != tid)
                    self.transition_to(BrainState.APPROACHING, reason=plan.reason)
                    if new_target or self.state_ctx.budget <= 0:
                        # 换目标 = 重新计时并重新给预算 (远的目标本来就该给更久)
                        self.state_ctx.budget = self._approach_budget(plan.travel_px)
                        self.state_ctx.entered_at = time.time()
                        log.info(f"[APPROACH] {plan.reason}; 需走 {plan.travel_px:.0f}px, "
                                 f"预算 {self.state_ctx.budget:.1f}s")
                    self.state_ctx.last_target_id = tid
                    self._last_action = ("approach", best)
                    return ("approach", best)

            # 有怪但都不可及 → 交给巡逻重新规划路线
            self.transition_to(BrainState.PATROLLING, reason="有怪但不可及")
            self._last_action = ("patrol", None)
            return ("patrol", None)

        # 无怪 → 巡逻
        self.transition_to(BrainState.PATROLLING, reason="视野内无怪")
        self._last_action = ("patrol", None)
        return ("patrol", None)

    def transition_to(self, new_state: BrainState, reason: str = "", force: bool = False) -> bool:
        """显式状态转换 (设计思想: 状态机账本化, 拒绝非法转换)。

        三道闸:
        1. STANDBY 是全局 override (用户按 F / 无药看门狗) → 任何状态都可直接进。
        2. 最小驻留: "降级" (打→接近→巡逻→扫描) 必须等够 MIN_DWELL_SEC, 否则被挡下并计数。
           升级 (→ATTACKING) 不受限 —— 有怪能打永远立刻响应。
        3. ALLOWED_TRANSITIONS 表: 不在表里 = 状态图有洞, 记警告并走 SCANNING 兜底。

        返回 True=状态已变更, False=被挡下 (保持原状态)。
        """
        old = self.state_ctx.state
        if old == new_state:
            return False  # 同状态: 无事可做 (不刷新 entered_at, 否则超时 watchdog 永远不触发)

        now = time.time()

        # 1. STANDBY override
        if new_state is BrainState.STANDBY:
            force = True

        # 2. 最小驻留 (只约束降级)
        if not force and STATE_RANK.get(new_state, 0) < STATE_RANK.get(old, 0):
            dwell = MIN_DWELL_SEC.get(old, 0.0)
            if now - self.state_ctx.entered_at < dwell:
                self.state_ctx.blocked_count += 1
                return False

        # 3. 合法性
        if not force:
            allowed = ALLOWED_TRANSITIONS.get(old, set())
            if new_state not in allowed:
                log.warning(f"[STATE] 非法转换 {old.value} → {new_state.value}; 走 SCANNING 兜底")
                new_state = BrainState.SCANNING
                if new_state == old:
                    return False

        dwell = now - self.state_ctx.entered_at
        self.state_ctx.state = new_state
        self.state_ctx.entered_at = now
        self.state_ctx.attempt_count = 0
        self.state_ctx.budget = 0.0        # 新状态默认用超时表; 需要的话由调用方另行设定
        self.state_ctx.transition_count += 1
        self.state = new_state
        # 日志去抖: 高频状态对 (attack↔approach) 1 秒内只打一条, 但账本每条都记
        if now - self.state_ctx.last_log_at > 1.0:
            log.info(f"[STATE] {old.value} → {new_state.value}"
                     f"{' (' + reason + ')' if reason else ''} 驻留={dwell:.1f}s")
            self.state_ctx.last_log_at = now
        self.ledger.state_changed(old.value, new_state.value, reason, dwell,
                                  self._snap.seq if self._snap else 0)
        return True

    @staticmethod
    def _approach_budget(travel_px: float) -> float:
        """接近一个目标该给多少秒 = 固定开销 + 路程/速度 × 余量 (上限封顶)。"""
        return min(APPROACH_BUDGET_MAX,
                   APPROACH_BUDGET_BASE + (travel_px / WALK_SPEED_PX_S) * APPROACH_BUDGET_SLACK)

    def _state_watchdog(self) -> None:
        """状态机超时 watchdog: 卡在 APPROACHING/ATTACKING/PATROLLING 太久 → 升级处理。
        防止单一状态死循环 (设计思想: 账本超时审计)。"""
        ctx = self.state_ctx
        # 有专属预算就用预算 (接近远处目标需要的时间是按距离算的, 不是常数)
        timeout = ctx.budget if ctx.budget > 0 else STATE_TIMEOUT_SEC.get(ctx.state, 0)
        if timeout <= 0:
            return
        elapsed = time.time() - ctx.entered_at
        if elapsed <= timeout:
            return
        # 升级: 不同状态不同处理 (force=True 绕过最小驻留 —— 超时本身就是"待够了"的证明)
        if ctx.state == BrainState.SCANNING:
            log.warning(f"[STATE] SCANNING 超时 {elapsed:.1f}s → PATROLLING (没找到可打的, 换个地方)")
            self.ledger.stuck(ctx.state.value, elapsed, "patrolling")
            self.transition_to(BrainState.PATROLLING, reason="扫描超时", force=True)
        elif ctx.state == BrainState.APPROACHING:
            log.warning(f"[STATE] APPROACHING 超时 {elapsed:.1f}s → PATROLLING (走不到)")
            ctx.attempt_count += 1
            self.ledger.stuck(ctx.state.value, elapsed, "patrolling")
            self.transition_to(BrainState.PATROLLING, reason="接近超时", force=True)
        elif ctx.state == BrainState.ATTACKING:
            log.warning(f"[STATE] ATTACKING 超时 {elapsed:.1f}s → SCANNING (目标可能漏检)")
            ctx.attempt_count += 1
            self._last_action = None   # 解除目标承诺, 否则会立刻锁回同一只打不动的怪
            self.ledger.stuck(ctx.state.value, elapsed, "scanning")
            self.transition_to(BrainState.SCANNING, reason="攻击超时", force=True)
        elif ctx.state == BrainState.PATROLLING:
            log.info(f"[STATE] PATROLLING {elapsed:.1f}s, 强制换向")
            self.mover.flip()  # 巡逻换向
            ctx.attempt_count += 1
            self.ledger.stuck(ctx.state.value, elapsed, "flip+scanning")
            self.transition_to(BrainState.SCANNING, reason="巡逻超时换向", force=True)

    def _action_worker(self, action_id: int, action: tuple) -> None:
        """动作执行 (ActionExecutor 线程内): 读最新感知, 分发到具体动作; 支持 cancel 中断。"""
        atype, payload = action
        controller = self.controller
        if controller is None:
            return
        cancel = lambda: self.executor.is_cancelled(action_id)
        # 动作开始时取一份新鲜快照 (与决策看到的世界同源; 动作内部通过 brain.* 接口再取)
        snap = self.refresh_snapshot()
        px, py = snap.px, snap.py
        platforms = list(snap.platforms)
        ropes = list(snap.ropes)
        t_start = time.time()
        try:
            if atype == "attack" and payload is not None:
                # 用快照里的同一只怪 (身份查找), payload 可能已是上一帧的旧坐标
                cur = snap.target_by_id(getattr(payload, "id", None)) or payload
                self._attack(controller, cur, px, py, cancel)
            elif atype == "approach" and payload is not None:
                cur = snap.target_by_id(getattr(payload, "id", None)) or payload
                self.mover.approach(controller, cur, px, py, platforms, self, cancel, ropes=ropes)
            elif atype == "patrol":
                # world_offset 与 platforms 同源 (同一份快照), 否则世界坐标换算会错位
                self.mover.patrol(controller, px, py, platforms, ropes, self, cancel,
                                  world_offset=snap.world_offset)
            # "none" 待机: 无需动作
        except Exception as e:
            log.error(f"[EXEC] 动作执行异常: {e}", exc_info=True)
        finally:
            # 空转刹车: 动作瞬间返回 (够不着/目标没了) 时不要让 executor 以 10Hz 空跑,
            # 那会把 CPU 和按键队列都塞满。给一帧的休息, 等感知刷新再决策。
            if time.time() - t_start < 0.05 and atype != "none":
                time.sleep(0.06)

    def run(self, capture: WindowCapture, controller: GameController, hp_monitor: Optional[HPMonitor] = None, show_vision: bool = True):
        """主循环 (眼手分离): 每帧感知 → 决策 → 推送命令 → 画 viz, 不再被动作阻塞 (~10fps)。"""
        self._running = True
        self.controller = controller
        self.capture = capture
        log.info("=== Combat Brain V8 (010001010 认怪 + v13 地形 + 地形感知移动) ONLINE ===")
        log.info(f"State: {self.state.value}")

        # 固定窗口客户区尺寸 (参考 MapleStoryAutoLevelUp auto_resize), 防窗口过大/出屏 → "视频超出"
        capture.resize_window(*RESIZE_TARGET_CLIENT)

        if show_vision:
            cv2.namedWindow("Agent V7 Vision", cv2.WINDOW_NORMAL)

        # 启动后台视觉线程 + 动作执行线程 (眼手分离)
        threading.Thread(target=self._perception_loop, args=(capture,), daemon=True).start()
        self.executor = ActionExecutor(self._action_worker)

        last_loop_t = time.time()  # 上一轮循环起点 (算真实 fps, 含限频 sleep)
        while self._running:
            t0 = time.time() # 用于画面 FPS 显示统计
            loop_fps = 1.0 / max(0.001, t0 - last_loop_t)  # 完整循环周期 (含 sleep) → ~10fps
            last_loop_t = t0
            # 冻结世界: 本轮决策 + HUD 都看这一份 (不阻塞)
            snap = self.refresh_snapshot()
            with self._vision_lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
            targets = snap.targets
            px, py = snap.px, snap.py

            if frame is None:
                time.sleep(0.1)
                continue

            # ===== 兜底 HP 药水: 每 N 分钟主动按一次 a (防止 auto_healer 漏触发; N 由模板 hp_potion_interval_sec) =====
            if self.active_hunting and time.time() - self._last_hp_potion_time >= self.hp_potion_interval:
                log.info(f"[HP POTION] 兜底触发 {int(self.hp_potion_interval//60)} 分钟 HP 药水")
                controller.use_hp_potion()
                self._last_hp_potion_time = time.time()

            # ===== 定时喂宠物: 每 10 分钟按一次 j =====
            if self.active_hunting and time.time() - self._last_pet_feed_time >= PET_FEED_INTERVAL:
                log.info("[PET] 定时喂宠物 (j)")
                controller.tap_key("j")
                self._last_pet_feed_time = time.time()

            # 决策 → 推送动作 (ActionExecutor 内部按稳定身份去重, 动作不重复执行不被打断)
            action = self._decide(snap)
            self.executor.set_action(action)

            # 状态机观察日志: 每 ~3 秒输出一次 (供看门狗/人观察状态机是否正确)
            self._state_log_count += 1
            if self._state_log_count % 30 == 0:
                sup = self.mover.support(px, snap.foot_y, list(snap.platforms))
                nearest = min((t.dist for t in targets), default=-1)
                log.info(f"[STATE] 玩家=({px},{py}) 身份={snap.confidence.value}/{snap.player_source} "
                         f"表面={'平台y=%d' % sup[0] if sup else '地面'} "
                         f"平台={len(snap.platforms)} 怪={len(targets)} 最近={nearest:.0f}px "
                         f"射程={self.engage_range_x} 状态={self.state.value} "
                         f"已驻={time.time()-self.state_ctx.entered_at:.1f}s "
                         f"转换#{self.state_ctx.transition_count} 抖动挡下={self.state_ctx.blocked_count} "
                         f"击杀={self.kill_count}/{self.attack_count}轮 动作={action[0]}")

            # 渲染可视化界面 (每帧, 不再被动作阻塞)
            if show_vision:
                key = self._draw_vision(frame, snap, loop_fps, hp_monitor)
                if key is not None and (key & 0xFF) == ord('q'):
                    self._running = False
                    break
                self._handle_vision_key(key)

            # 限频 ~10fps (决策循环稳定跑, 感知变更立刻反映到决策)
            elapsed = time.time() - t0
            time.sleep(max(0.005, 0.1 - elapsed))

    def _draw_vision(self, frame, snap: WorldSnapshot, fps, hp_monitor) -> Optional[int]:
        """绘制 Agent V7 可视化 HUD, 返回 cv2.waitKey(1) 按键值 (供 run() 处理 q/名牌校准)。
        snap: 本轮决策用的世界快照 (HUD 与决策看同一个世界, 否则画面骗人)。
        fps: 主循环真实帧率 (含限频 sleep, ~10fps)。"""
        vis_frame = frame.copy()
        targets, px, py = snap.targets, snap.px, snap.py

        # 绘制玩家位置 (十字颜色 = 身份可信度: 绿=确认 / 黄=短暂丢失 / 红=已丢失, 只是猜测)
        pcolor = {PlayerConfidence.CONFIRMED: (80, 220, 80),
                  PlayerConfidence.STALE: (0, 200, 255)}.get(snap.confidence, (60, 60, 255))
        cv2.drawMarker(vis_frame, (px, py), pcolor, cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis_frame, f"Player[{snap.confidence.value}:{snap.player_source}]",
                    (px + 15, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pcolor, 2)
        # 攻击射程可视化 (engage 实线 / hold 虚线): 一眼看出"够不够得着"
        cv2.rectangle(vis_frame,
                      (px - self.engage_range_x, py - self.attack_range_y),
                      (px + self.engage_range_x, py + self.attack_range_y), (90, 200, 90), 1)

        # 绘制名牌框 (绿框, 校准确认用)
        nl = self.nametag_locator
        if nl.last_match_rect:
            nx, ny, nw, nh = nl.last_match_rect
            cv2.rectangle(vis_frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"NAMETAG {nl.last_score:.2f}", (nx, ny - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制怪物 (在射程内 = 绿框, 够不着 = 橙框; 锁定的目标加粗)
        locked_id = self.state_ctx.last_target_id if self.state is BrainState.ATTACKING else None
        for t in targets:
            tx1, ty1 = t.cx - t.w // 2, t.cy - t.h // 2
            tx2, ty2 = t.cx + t.w // 2, t.cy + t.h // 2
            hit = self.is_in_attack_range(t, px, py)
            color = (80, 220, 80) if hit else (0, 165, 255)
            thick = 3 if t.id == locked_id else 2
            cv2.rectangle(vis_frame, (tx1, ty1), (tx2, ty2), color, thick)
            cv2.putText(vis_frame, f"#{t.id} {t.name} {t.dist:.0f}px", (tx1, ty1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制地形 (平台橙条 / 梯子黄条, 来自同一份快照)
        plat, rope = snap.platforms, snap.ropes
        for p in plat:
            cv2.rectangle(vis_frame, (p[1], p[0] - 3), (p[2], p[0] + 3), (0, 165, 255), -1)
        for r in rope:
            cv2.rectangle(vis_frame, (r[0] - 5, r[1]), (r[0] + 5, r[2]), (0, 255, 255), 3)

        # 绘制状态和帧率 (fps 由主循环按完整周期传入, ~10)
        status_color = (0, 0, 255) if not self.active_hunting else (0, 255, 0)
        status_text = (f"State: {self.state.value} | FPS: {fps:.0f} | "
                       f"Kills: {self.kill_count}/{self.attack_count} | Seq: {snap.seq}")
        cv2.putText(vis_frame, "ACTIVE" if self.active_hunting else "STANDBY (Press F1 to Start, F to Stop)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis_frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 名牌/身份 HUD: 显示绑定的是哪个模板的名牌 (换角色忘换名牌时一眼看见)
        id_str = f"ID:{self.profile.template}" if nl.identity_bound else f"ID:{self.profile.template}(NO NAMETAG)"
        ok_str = "OK" if nl.last_score < NAMETAG_MATCH_THRESHOLD else "MISS"
        cv2.putText(vis_frame, f"{id_str} nametag={ok_str}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if nl.identity_bound else (0, 140, 255), 2)

        # 处理 HP/MP 显示
        if hp_monitor:
            stats = hp_monitor.read(frame)
            hp, mp = stats.hp_percent * 100, stats.mp_percent * 100
            hp_text = f"HP: {hp:.1f}%" if hp > 0 else "HP: ???%"
            mp_text = f"MP: {mp:.1f}%" if mp > 0 else "MP: ???%"
            cv2.putText(vis_frame, hp_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis_frame, mp_text, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # 在画面上画出监控遮罩，让用户检查 (只画框线, 不标文字)
            if hp_monitor.is_calibrated:
                hx, hy, hw, hh = hp_monitor.hp_bbox
                mx, my, mw, mh = hp_monitor.mp_bbox
                cv2.rectangle(vis_frame, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
                cv2.rectangle(vis_frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

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
        # 玩家位置由 HSV 名牌定位器按"名牌顶部=脚底"计算 (见 nametag_hsv_locator)
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
        try:
            self.ledger.close()
        except Exception:
            pass
        log.info(f"Combat Brain stopped. 击杀 {self.kill_count} / 攻击轮次 {self.attack_count} "
                 f"| 状态转换 {self.state_ctx.transition_count} (抖动挡下 {self.state_ctx.blocked_count})")
        cv2.destroyAllWindows()
