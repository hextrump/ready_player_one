"""
PatrolMover — 地形感知移动 (从零重写, 极简版)
============================================
三层优先级 (由 CombatBrain.run 驱动):
  1. 打:   怪在攻击范围 → burst (CombatBrain._attack 负责)。
  2. 接近: 有"直接可及"的怪 → 简单走/跳过去 (不做多步路线)。
  3. 巡逻: 都没有 → 用实时地形(平台/梯子)生成巡逻路线, 沿路找怪。

核心思想: 不逐怪规划路线, 用"地形巡逻"覆盖全图; 怪进范围就打。
强项 = 实时看到平台/梯子/怪 (v13/v19), 所以巡逻知道往哪走、到边缘会爬梯/跳台。

感知通过 brain 对象回调读取 (duck-typed):
  - brain.any_target_in_range()  -> bool  是否有怪进入攻击范围
  - brain.player_pos()           -> (x, y) 当前玩家位置
  - brain.nearest_target()       -> Target|None 最近怪 (巡逻方向偏置)
"""
import math
import random
import time
from dataclasses import dataclass
from enum import Enum

from src.brain.game_controller import GameController, Direction
from src.brain.entity_tracker import PLAYER_FOOT_OFFSET
from src.brain.terrain_graph import TerrainGraph, MIN_DROP_DY
from src.utils.logger import get_logger
from src.utils.player_profile import get_profile

log = get_logger("patrol_mover")

# ===== 移动参数 =====
APPROACH_MAX_SEC = 3.0   # 单次水平走最长秒数 (超时交给外层重决策)
APPROACH_POLL = 0.04     # 行走循环轮询间隔
MOVE_PX_EPS = 3          # 判定"在移动"的像素阈值 (感知 ~5.5fps)
STUCK_JUMP_TIMEOUT = 1.0 # 移动无进度多久判定卡住, 触发反向脱困跳 (感知~5.5fps, 需容忍几帧)

WALK_EDGE_MARGIN = 25    # 巡逻到平台边缘多少 px 内处理 (爬梯/换向)

# ===== 巡逻: 目标驱动 (看地形 → 去最远的地方) =====
# 之前两版都是"定时换向": 每 3s / 每 6~15s 翻一次。问题不在幅度而在**没有目标** ——
# 相机跟着人走, 屏幕坐标里人几乎不动, bot 既不知道自己在哪也不知道走了多远。
# 现在: 从地形里选一个最远的落脚点, 存**世界坐标**, 走到了才换下一个。
GOAL_ARRIVE_PX = 70          # 距目标多近算到达
GOAL_WALK_CHUNK = 3.0        # 单次朝目标连续走多久 (秒), 到点交还决策层重新评估
GOAL_HEADING_BONUS = 1.35    # 同方向候选的得分加权 (防"到了就掉头"来回蹭)
GOAL_VISIT_MEMORY = 8        # 记住最近去过的几个点
GOAL_VISIT_RADIUS = 220      # 这个半径内算"刚去过"
GOAL_VISIT_PENALTY = 0.45    # 刚去过的地方得分打折 (别在一处反复刷)
GOAL_VISIT_FORGET_SEC = 90.0 # 多久之后"去过"就忘掉 (怪会重刷, 该回去了)

PATROL_PAUSE_CHANCE = 0.10   # 走完一段后站着发呆的概率 (像人在看屏幕)
PATROL_PAUSE_MIN = 1.0
PATROL_PAUSE_MAX = 3.0
PATROL_ATTACK_RANGE = 220  # 巡逻时距怪多少 px 内才普攻 (防空挥)
PATROL_CHASE_RANGE = 450   # 巡逻方向只朝该距离内的怪偏置 (防止被远处/异层怪锁方向)
PATROL_STUCK_TIMEOUT = 1.0  # 画面静止多久判定卡住 (名牌失效也能用: 相机不动=玩家没动)
MOTION_MOVING_THRESHOLD = 3.0  # 帧间运动量高于此才算"画面在动" (调低=更容易判定卡住)

SURFACE_TOL_Y = 30       # 点是否落在平台上的 y 容差 (实测怪框/玩家脚底与平台 y 偏移 23-25px)
# PLAYER_FOOT_OFFSET 从 entity_tracker 导入 (全局唯一真源, 名牌定位器的 FEET_TO_CENTER 与之对齐)

EDGE_STEP_OVER = 25      # 下落时走出平台边缘多少 px (确保真的离开平台, 不是停在边上)
TAKEOFF_TOL = 40         # 距起跳点多近算"站定了, 可以跳"
TAKEOFF_INSET = 40       # 起跳点从目标平台两端各内缩多少 (别贴边跳, 容易擦边掉下去)
# 单跳高度 (edge_jump_max_dy) 是对**我们自己能力**的估计, 地形/梯子检测又不可靠。
# 所以在"肯定跳得上"和"肯定跳不上"之间留一条**试一试**的带子:
# 落差在 [单跳高度, 单跳高度 × TRY_JUMP_FACTOR] 之间时先去试跳, 跳不上由失败计数学到,
# 而不是从"没检测到梯子"这种不可靠的否定证据里推出"够不着"。
TRY_JUMP_FACTOR = 2.0


class MoveKind(Enum):
    """从玩家所在行走面到目标所在行走面的关系 (类型化的结论, 不是一堆 dx/dy)。"""
    WALK = "walk"               # 同一层, 直接走
    JUMP_UP = "jump_up"         # 上一层: 先走到目标平台正下方, 再跳
    DROP = "drop"               # 下一层: 走出边缘自然落下
    CLIMB = "climb"             # 落差超出单跳: 走到梯子再爬
    UNREACHABLE = "unreachable" # 够不着 —— 决策层据此改去巡逻别处


@dataclass(frozen=True)
class MovePlan:
    """怎么过去。plan_move() 是唯一生产者, is_reachable / approach 都消费它。

    kind:       动作类型
    takeoff_x:  起跳/抓梯点 (屏幕 x); WALK/DROP 为 None
    surface_dy: 表面落差 (正 = 目标在上方), 诊断用
    reason:     为什么是这个结论 —— 账本化, 日志里能直接看懂它在想什么
    """
    kind: MoveKind
    takeoff_x: int | None
    surface_dy: float
    reason: str
    travel_px: float = 0.0   # 到起跳点/目标还要走多远 (决策层用它算超时预算)

JUMP_UP_DY = 120         # 目标高于此且水平近 → 登台跳
JUMP_TO_UPPER_DX = 150   # 登台跳最大水平距离
JUMP_DOWN_DY = 30        # 目标低于此且水平近 → 下跳
JUMP_DOWN_DX = 80        # 下跳最大水平距离

ROPE_GRAB_DX = 30        # 平台边缘附近多少 px 内算"有梯子可爬"
CLIMB_BURST = 0.6        # 巡逻爬梯单次脉冲秒数
CLIMB_MAX_BURSTS = 8     # 单次爬梯最多脉冲数 (防卡死)

JUMP_FAIL_LIMIT = 3      # 登台跳连跳几次仍不可打 → 临时放弃该目标 (防跳-loop 卡住)
BLOCK_COOLDOWN = 4.0     # 放弃目标后的冷却秒数 (冷却期内不再尝试)

# 边缘上跳 (从低平台跳到相邻高平台, 解决高低平台无梯子连不上导致的卡住)
EDGE_JUMP_MAX_DY = 140   # 单次上跳能上的最大高度差 (游戏跳高 ~150)
EDGE_JUMP_MAX_GAP = 60   # 高平台起点到当前边缘的最大水平缝隙 (单向平台可从下方跳上)
EDGE_JUMP_COOLDOWN = 1.5 # 边缘上跳冷却 (防跳不上时热循环)

FLAT_MODE = False        # True=关闭跳跃/爬梯, 纯平面推图


class PatrolMover:
    def __init__(self):
        # ── 玩家模板驱动 (巡逻/卡住/边缘跳参数全部按模板读) ──
        profile = get_profile()
        self.flat_mode = profile.combat.flat_mode
        self.approach_max_sec = profile.combat.approach_max_sec
        self.stuck_jump_timeout = profile.combat.stuck_jump_timeout
        self.patrol_attack_range = profile.combat.patrol_attack_range
        self.patrol_chase_range = profile.combat.patrol_chase_range
        self.patrol_stuck_timeout = profile.combat.patrol_stuck_timeout
        self.edge_jump_max_dy = profile.combat.edge_jump_max_dy
        self.edge_jump_max_gap = profile.combat.edge_jump_max_gap
        self.edge_jump_cooldown = profile.combat.edge_jump_cooldown
        self.jump_up_dy = profile.combat.jump_up_dy
        self.jump_to_upper_dx = profile.combat.jump_to_upper_dx
        self.jump_down_dy = profile.combat.jump_down_dy
        self.jump_down_dx = profile.combat.jump_down_dx
        # 巡逻: 目标驱动 (模板可覆盖)
        self.goal_timeout = profile.combat.patrol_goal_timeout
        self.goal_min_gain = profile.combat.patrol_goal_min_gain
        self.goal_explore_push = profile.combat.patrol_goal_explore_push
        self.patrol_pause_chance = profile.combat.patrol_pause_chance
        self.patrol_pause_min = profile.combat.patrol_pause_min
        self.patrol_pause_max = profile.combat.patrol_pause_max
        # 默认不用 ↓+Alt 下跳, 改成走出平台边缘自然落下 (见 _descend 的说明)
        self.allow_jump_down = profile.combat.allow_jump_down

        self.patrol_direction = Direction.RIGHT
        self._goal = None                   # 当前巡逻目标 (世界坐标 wx, wy)
        self._goal_kind = ""                # 目标类型 (平台端/梯子/探索/怪), 日志用
        self._goal_t = 0.0                  # 目标设定时间 (超时用)
        self._visited = []                  # 最近去过的点 [(wx, wy, t)], 防在一处反复刷
        self._last_motion_t = time.time()   # 最近一次画面有动静的时间 (卡住检测)
        self._jump_count = {}               # 目标粗网格key -> 连续登台跳次数
        self._blocked = {}                  # 目标粗网格key -> 冷却截止时间 (防跳-loop)
        self._last_edge_jump_t = 0.0        # 最近一次边缘上跳时间 (防热循环)
        self._last_patrol_x = None          # 上次巡逻时的玩家位置 (卡住检测主信号)
        self._last_patrol_y = None
        self._last_platforms = None         # 最近构建图的平台 (变化才重建)
        self.terrain_graph = None           # 平台可达图 (跳上/跳下, 算一次缓存)

    @staticmethod
    def _tkey(target):
        """目标身份 key (容忍微动/抖动)。优先用实体 id (跨帧稳定), 无 id 回退粗网格。"""
        tid = getattr(target, "id", None)
        if tid is not None:
            return ("id", tid)
        return ("grid", target.cx // 60, target.cy // 60)

    def _is_blocked(self, key) -> bool:
        if key in self._blocked:
            if time.time() >= self._blocked[key]:
                self._blocked.pop(key, None)
                return False
            return True
        return False

    def _find_upper_platform(self, sup, edge_x, is_right, platforms):
        """从 sup 平台边缘朝 is_right 侧找一块可单跳上去的更高平台 (y 更小), 无则 None。
        冒险岛平台是单向的: 可从下方跳上, 故只需高平台 x 范围覆盖边缘附近即可。"""
        sup_y = sup[0]
        for p in platforms:
            if p[0] >= sup_y - 40:              # 不是足够高的平台 (y 更大 = 更低)
                continue
            if sup_y - p[0] > self.edge_jump_max_dy: # 高度差超过单跳范围
                continue
            if is_right:
                if p[1] <= edge_x + self.edge_jump_max_gap and p[2] >= edge_x:
                    return p
            else:
                if p[2] >= edge_x - self.edge_jump_max_gap and p[1] <= edge_x:
                    return p
        return None

    def _ensure_graph(self, platforms: list, px=None, pfeet=None):
        """平台变化时重建可达图 (含玩家表面); 返回玩家当前表面。
        玩家脚底不在任何检测平台 → 隐含地面节点 (无限宽, 玩家脚底所在层):
        v13 常漏检地面层, 但玩家脚下一定是某个表面, 没有它就跳不起来。"""
        graph_platforms = list(platforms)
        player_surface = (self.support(px, pfeet, platforms)
                          if (px is not None and pfeet is not None) else None)
        if player_surface is None and px is not None:
            player_surface = (pfeet, -10**6, 10**6)   # 隐含地面
            graph_platforms.append(player_surface)
        key = tuple(graph_platforms)
        if key != self._last_platforms:
            self._last_platforms = key
            self.terrain_graph = TerrainGraph(graph_platforms)
        return player_surface

    # ===== 可及性判断 =====

    def support(self, x: int, feet_y: float, platforms: list):
        """返回 (x, feet_y) 所在平台 (y, x_left, x_right); 不落任何平台则 None (地面)。"""
        for p in platforms:
            if abs(p[0] - feet_y) <= SURFACE_TOL_Y and p[1] <= x <= p[2]:
                return p
        return None

    # ===== 可及性: 收敛成一个类型化的 MovePlan (单一生产者, 两个消费者) =====
    #
    # 设计思想 (design/设计思想.md): "上下文不能碰瓷状态"。
    # 原来"我够不够得到那只怪"是在两个地方各用一堆 dx/dy 魔法数现推一遍 ——
    # is_reachable 判一次(决策层), approach 又判一次(执行层), 阈值还不一样。
    # 那是把 evidence 直接当 state 用。现在把它**收敛为一个确定的数据结构**:
    #   plan_move() 是唯一生产者, 输出 MovePlan(kind, takeoff_x, surface_dy, reason)
    #   is_reachable = plan.kind is not UNREACHABLE      (决策层: 我能不能)
    #   approach     = 按 plan 执行                       (执行层: 具体怎么做)
    # 于是"决策放行的, 执行一定做得了"成为一条可测试的不变量 (与射程的 engage/hold 同构)。

    def plan_move(self, target, px: int, py: int, platforms: list, ropes: list) -> MovePlan:
        """算出"从我脚下这个面, 到目标脚下那个面"该怎么走。唯一生产者。"""
        pfeet = py + PLAYER_FOOT_OFFSET
        tfeet = target.cy + target.h / 2
        surface_dy = pfeet - tfeet          # 正 = 目标在上方

        p_surf = self.support(px, pfeet, platforms)
        t_surf = self.support(target.cx, tfeet, platforms)

        # 同层 (含两边都没检测到平台但脚底高度相近)
        if abs(surface_dy) <= SURFACE_TOL_Y:
            return MovePlan(MoveKind.WALK, None, surface_dy, "同一行走面",
                            travel_px=abs(target.cx - px))

        if self.flat_mode:
            return MovePlan(MoveKind.UNREACHABLE, None, surface_dy, "平地模式, 不做垂直移动")

        # ── 目标在上方 ──
        if surface_dy > 0:
            if self._is_blocked(self._tkey(target)):
                return MovePlan(MoveKind.UNREACHABLE, None, surface_dy, "该目标跳失败过, 冷却中")
            # 起跳点 = **怪的正下方**。怪站在那儿, 就说明那儿有个能站的地方 ——
            # 这个推断不依赖地形检测 (v13 常漏检平台, 实测 60 例上方怪一个平台都没检出来)。
            # 检测到平台时只拿它做**微调** (别贴着边跳, 容易擦边掉下去), 而不是当前提。
            takeoff = int(target.cx)
            if t_surf is not None and (t_surf[2] - t_surf[1]) < 1e5:
                lo, hi = t_surf[1] + TAKEOFF_INSET, t_surf[2] - TAKEOFF_INSET
                if lo <= hi:
                    takeoff = int(min(max(takeoff, lo), hi))

            if surface_dy <= self.edge_jump_max_dy:
                return MovePlan(MoveKind.JUMP_UP, takeoff, surface_dy,
                                f"上方 {surface_dy:.0f}px, 单跳可及 (起跳点 x={takeoff})",
                                travel_px=abs(takeoff - px))

            # 超出单跳估计。有梯子就走梯子 (检测到梯子是**正证据**, 可信)
            rope = self._rope_spanning(ropes, pfeet, tfeet)
            if rope is not None:
                return MovePlan(MoveKind.CLIMB, int(rope[0]), surface_dy,
                                f"上方 {surface_dy:.0f}px 超出单跳, 走梯子 x={rope[0]}",
                                travel_px=abs(rope[0] - px))

            # 没检测到梯子 —— 但"没检测到"不等于"没有"。地形检测不可靠, 从否定证据推结论
            # 是最容易错的一步。怪在上面就说明那儿有路, 所以先去试跳; 跳不上去由
            # JUMP_FAIL_LIMIT 次失败学到 (然后 BLOCK_COOLDOWN 冷却), 代价只有几次跳。
            if surface_dy <= self.edge_jump_max_dy * TRY_JUMP_FACTOR:
                return MovePlan(MoveKind.JUMP_UP, takeoff, surface_dy,
                                f"上方 {surface_dy:.0f}px 超出单跳估计, 但怪在上面说明有路 → 先试跳",
                                travel_px=abs(takeoff - px))
            return MovePlan(MoveKind.UNREACHABLE, None, surface_dy,
                            f"上方 {surface_dy:.0f}px 高出单跳 {TRY_JUMP_FACTOR:.0f} 倍, 试跳也上不去")

        # ── 目标在下方 ──
        if -surface_dy >= MIN_DROP_DY:
            # 走出边缘自然落下 (起跳点由 _descend 按当前平台算; 这里只标明意图)
            return MovePlan(MoveKind.DROP, None, surface_dy,
                            f"下方 {-surface_dy:.0f}px, 走出边缘落下",
                            travel_px=abs(target.cx - px))
        return MovePlan(MoveKind.WALK, None, surface_dy, "落差不足一层, 当同层走",
                        travel_px=abs(target.cx - px))

    def _rope_spanning(self, ropes: list, pfeet: float, tfeet: float):
        """找一根竖直跨度覆盖 [tfeet, pfeet] 的梯子 (能把这两层连起来)。"""
        top, bot = min(pfeet, tfeet), max(pfeet, tfeet)
        for r in ropes:
            rx, rtop, rbot = r
            if rtop <= top + SURFACE_TOL_Y and rbot >= bot - SURFACE_TOL_Y:
                return r
        return None

    def is_reachable(self, target, px: int, py: int, platforms: list, ropes: list = ()) -> bool:
        """决策层消费: 够不够得着 (与执行层同一个判定, 不再各推一遍)。"""
        return self.plan_move(target, px, py, platforms, list(ropes)).kind is not MoveKind.UNREACHABLE

    def _same_surface(self, px: int, pfeet: float, tx: int, tfeet: float, platforms: list) -> bool:
        pp = self.support(px, pfeet, platforms)
        tp = self.support(tx, tfeet, platforms)
        if pp is not None and tp is not None:
            return pp[0] == tp[0]
        if pp is None and tp is None:
            return abs(pfeet - tfeet) <= SURFACE_TOL_Y
        return False

    # ===== 接近 (执行 MovePlan) =====

    def approach(self, controller: GameController, target, px: int, py: int,
                 platforms: list, brain, cancel=None, ropes: list = ()) -> None:
        """执行层消费: 按 plan_move 的结论行动。决策放行什么, 这里就做什么。"""
        plan = self.plan_move(target, px, py, platforms, list(ropes))

        if plan.kind is MoveKind.UNREACHABLE:
            log.debug(f"[APPROACH] 够不着 ({plan.reason}), 交还决策")
            return

        if plan.kind is MoveKind.DROP:
            self._descend(controller, target, px, py, platforms, brain, cancel)
            return

        if plan.kind is MoveKind.CLIMB:
            if abs(px - plan.takeoff_x) > ROPE_GRAB_DX:
                self._walk_toward(controller, plan.takeoff_x, px, brain, cancel)
                return
            rope = self._rope_spanning(list(ropes), py + PLAYER_FOOT_OFFSET,
                                       target.cy + target.h / 2)
            if rope is not None:
                log.info(f"🧗 爬梯上去打 -> {target.name}")
                self._climb(controller, rope, True, brain, cancel)
            return

        if plan.kind is MoveKind.JUMP_UP:
            # 不再要求先走到精确的起跳点才跳 —— 地形/起跳点本来就是估出来的
            # (takeoff_x 只在检测到平台时做过内缩微调, 平台经常漏检), 走位等对齐
            # 只是在拿不可靠的坐标死磕。没对齐就朝目标方向边走边跳 (edge_jump_up
            # 空中会飘一段, 本身就是在缩小水平差), 对齐了就原地上跳; 跳不上去的
            # 代价一直是 JUMP_FAIL_LIMIT 次尝试, 而不是先耗时间走位。
            key = self._tkey(target)
            cnt = self._jump_count.get(key, 0) + 1
            self._jump_count[key] = cnt
            if cnt >= JUMP_FAIL_LIMIT:
                log.info(f"!! 上跳 {cnt} 次仍上不去, 冷却该目标, 改巡逻 !!")
                self._blocked[key] = time.time() + BLOCK_COOLDOWN
                self._jump_count.pop(key, None)
                return
            if abs(px - plan.takeoff_x) > TAKEOFF_TOL:
                direction = Direction.RIGHT if plan.takeoff_x >= px else Direction.LEFT
                log.info(f"↑ 边跳边靠近({cnt}/{JUMP_FAIL_LIMIT}) -> {target.name} "
                         f"(距起跳点 {abs(px - plan.takeoff_x):.0f}px, {plan.reason})")
                controller.edge_jump_up(direction)
            else:
                log.info(f"↑ 原地上跳({cnt}/{JUMP_FAIL_LIMIT}) -> {target.name} ({plan.reason})")
                controller.jump()      # 站在单向平台正下方原地跳即可穿上去
            return

        # WALK: 走到"打得到"的位置就停, 不是走到怪身上
        engage = getattr(brain, "engage_range_x", 0) or 0
        stop_margin = int(engage * 0.6)
        dx = target.cx - px
        goal_x = target.cx - stop_margin if dx >= 0 else target.cx + stop_margin
        self._walk_toward(controller, goal_x, px, brain, cancel)

    def _descend(self, controller: GameController, target, px: int, py: int,
                 platforms: list, brain, cancel=None) -> None:
        """下到更低的平台: **走出平台边缘自然落下**, 不用 ↓+Alt 下跳。

        为什么不用下跳 (2026-08-19 实测):
          地形层把"同一层地面被融合出 4px 高度差"也当成了"下面还有一层"
          (玩家踩隐含地面节点时它无限宽, 在 x 上与每块平台都重叠, 于是任何略低的平台
          都成了可下跳目标)。而站在最底层按 ↓+Alt 是**空动作** —— 表现就是原地无故抽搐。

        走出边缘在任何情况下都安全: 下面真有平台就落下去, 没有就只是往前走了两步。
        代价是穿不过"正上方就是单向平台"的场景 —— 那种情况把模板的
        combat.allow_jump_down 打开即可 (默认关)。
        """
        if self.allow_jump_down:
            log.info(f"↓ 下跳 -> {target.name}")
            controller.jump_down()
            return

        goal = target.cx
        sup = self.support(px, py + PLAYER_FOOT_OFFSET, platforms)
        # 站在有限宽的平台上, 且目标就在这块平台的跨度内 (正下方) → 走不出去,
        # 得先走到较近的一侧边缘并迈出去一点点。
        if sup is not None and (sup[2] - sup[1]) < 1e5 and sup[1] <= target.cx <= sup[2]:
            to_left = px - sup[1]
            to_right = sup[2] - px
            goal = (sup[1] - EDGE_STEP_OVER) if to_left <= to_right else (sup[2] + EDGE_STEP_OVER)
            log.info(f"↓ 走出平台边缘下落 (x={goal}) -> {target.name}")
        else:
            log.info(f"↓ 朝目标走并自然落下 -> {target.name}")
        self._walk_toward(controller, goal, px, brain, cancel)

    def _walk_toward(self, controller: GameController, goal_x: int, px: int, brain,
                     cancel=None) -> None:
        """水平走向 goal_x: 进范围/到达/超时 即停。
        仅在玩家位置可信时做脱困跳 (中心猜测时位置冻结, 会误判卡住狂跳)。

        注意 brain.any_target_in_range() 现在按模板 engage 射程判定 —— 它以前默认按"远程 b"
        算, 而战士的远程射程是 0, 于是这个"进范围就停"的分支对战士恒为 False:
        战士只能一路走到 3 秒超时或撞在怪身上, 这是"走过头 / 贴脸抽搐"的直接原因。"""
        direction = Direction.RIGHT if goal_x >= px else Direction.LEFT
        reliable = brain.player_reliable()
        start = time.time()
        last_px = px
        last_move = time.time()
        controller.key_down(direction.value)
        try:
            while time.time() - start < self.approach_max_sec:
                if cancel and cancel():
                    break
                if brain.any_target_in_range():
                    break
                cur_px = brain.player_pos()[0]
                if (direction == Direction.RIGHT and cur_px >= goal_x - MOVE_PX_EPS) or \
                   (direction == Direction.LEFT and cur_px <= goal_x + MOVE_PX_EPS):
                    break
                # 玩家位置不可信 → 只盲走到超时/进范围, 不做卡顿跳 (位置冻结会假触发)
                if not reliable:
                    time.sleep(APPROACH_POLL)
                    continue
                if abs(cur_px - last_px) >= MOVE_PX_EPS:
                    last_px = cur_px
                    last_move = time.time()
                elif (not self.flat_mode) and time.time() - last_move > self.stuck_jump_timeout:
                    log.info("!! 走向目标被卡, 反向脱困跳 !!")
                    controller.key_up(direction.value)
                    back = Direction.LEFT if direction == Direction.RIGHT else Direction.RIGHT
                    controller.diagonal_jump(back)
                    controller.key_down(direction.value)
                    last_move = time.time()
                time.sleep(APPROACH_POLL)
        finally:
            controller.key_up(direction.value)

    # ===== 巡逻 (地形实时生成) =====

    def patrol(self, controller: GameController, px: int, py: int,
               platforms: list, ropes: list, brain, cancel=None,
               world_offset: tuple | None = None) -> None:
        """目标驱动巡逻: **看地形 → 选最远的落脚点 → 一路走过去**。

        为什么不是"定时换向":
          相机跟着玩家走, 屏幕坐标里玩家几乎不动, 所以按时间翻向的巡逻既不知道自己在哪,
          也不知道走了多远 —— 表现就是原地左摇右晃。
          现在目标点存**世界坐标** (screen + 相机位姿), 镜头怎么滚它都钉在原处;
          走到了才换下一个, 于是一次能横穿大半张图。

        选点规则 (像人扫图):
          候选 = 各平台的左右端点 + 梯子上下端 (这就是"看地形")
          得分 = 距离; 同方向的加权 (防到了就掉头); 刚去过的打折 (防在一处反复刷)
          选不出来 (地形太少/都去过) → 朝当前方向推一个远点, 继续探索
        """
        if cancel and cancel():
            return

        # ── 卡住检测 (保持原逻辑: 位置在动 OR 画面在动 → 没卡住) ──
        cur_px, cur_py = brain.player_pos()
        moved = (self._last_patrol_x is not None
                 and (abs(cur_px - self._last_patrol_x) >= MOVE_PX_EPS
                      or abs(cur_py - self._last_patrol_y) >= MOVE_PX_EPS))
        self._last_patrol_x, self._last_patrol_y = cur_px, cur_py
        if moved or brain.world_moving():
            self._last_motion_t = time.time()
        elif time.time() - self._last_motion_t > self.patrol_stuck_timeout:
            log.info("!! 巡逻卡住(位置/画面都不动), 反向脱困走+跳 + 换目标 !!")
            back = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
            controller.move_direction(back, duration=0.5)
            controller.diagonal_jump(back)
            self._abandon_goal("卡住")
            self._last_motion_t = time.time()
            return

        # 相机位姿必须和 platforms 同源: platforms 来自动作开始时的快照, 现取的 offset 是
        # 另一个时刻的 —— 混用两个时刻的坐标正是之前踩过的坑, 所以由调用方一起传进来。
        if world_offset is None:
            world_offset = brain.world_offset()
        pwx, pwy = brain.player_world()
        now = time.time()

        # ── 附近有怪 → 目标直接设成它 (比"方向偏置"更直接, 也不会被左右两只怪拽成钟摆) ──
        # 只追**够得着**的怪。追一只够不着的(比如在上层平台却没梯子)会走到它正下方,
        # 而"到达"只比 x —— 于是到了就换目标、怪还在又设回来, 在下层平台原地来回蹭。
        nt = brain.nearest_target()
        if nt is not None and math.hypot(nt.cx - px, nt.cy - py) <= self.patrol_chase_range:
            plan = self.plan_move(nt, px, py, platforms, ropes)
            if plan.kind is not MoveKind.UNREACHABLE:
                ox, oy = world_offset
                self._goal = (nt.cx + ox, nt.cy + oy)
                self._goal_kind = "monster"
                self._goal_t = now

        # ── 目标生命周期: 到了 / 超时 / 没有 → 重新选 ──
        if self._goal is not None:
            if abs(self._goal[0] - pwx) <= GOAL_ARRIVE_PX:
                self._mark_visited(self._goal)
                log.info(f"[PATROL] 到达目标 ({self._goal_kind}), 选下一个")
                self._goal = None
            elif now - self._goal_t > self.goal_timeout:
                log.info(f"[PATROL] 目标超时 {self.goal_timeout:.0f}s, 放弃重选")
                self._abandon_goal("超时")

        if self._goal is None:
            self._pick_goal(pwx, pwy, platforms, ropes, brain, world_offset)

        # ── 朝目标的方向 ──
        self.patrol_direction = Direction.RIGHT if self._goal[0] >= pwx else Direction.LEFT

        # ── 挡路的地形: 到平台边缘就爬梯/上跳/绕 (这也是"看地形") ──
        sup = self.support(px, py + PLAYER_FOOT_OFFSET, platforms)
        if sup is not None:
            if self.patrol_direction == Direction.RIGHT and px >= sup[2] - WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, True, brain, platforms, cancel)
                return
            if self.patrol_direction == Direction.LEFT and px <= sup[1] + WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, False, brain, platforms, cancel)
                return

        # ── 朝目标连续走一段 (不再是 0.5s 一抽一抽) ──
        self._walk_to_world(controller, self._goal[0], brain, cancel)

        if cancel and cancel():
            return
        if brain.any_target_near(self.patrol_attack_range):
            controller.attack_single()
            return
        # 偶尔站一会儿 (像人在看屏幕); 折返已删除 —— 那本身就是"左摇右晃"
        if random.random() < self.patrol_pause_chance:
            self._idle_pause(brain, cancel)

    # ===== 目标选择 (看地形) =====

    def _pick_goal(self, pwx: float, pwy: float, platforms: list, ropes: list, brain,
                   world_offset: tuple | None = None) -> None:
        """从地形里挑一个"最远且值得去"的落脚点 (世界坐标), 写入 self._goal。

        world_offset 必须和 platforms 来自同一份快照 (同一时刻的镜头位姿)。
        """
        ox, oy = world_offset if world_offset is not None else brain.world_offset()
        cands = []
        for (y, xl, xr) in platforms:
            if xr - xl > 1e5:          # 隐含地面 (无限宽), 端点没有意义
                continue
            cands.append((xl + ox, y + oy, "平台左端"))
            cands.append((xr + ox, y + oy, "平台右端"))
        for (x, yt, yb) in ropes:
            cands.append((x + ox, yt + oy, "梯子顶"))
            cands.append((x + ox, yb + oy, "梯子底"))

        heading = 1 if self.patrol_direction == Direction.RIGHT else -1
        best, best_score = None, 0.0
        for (wx, wy, kind) in cands:
            d = abs(wx - pwx)
            if d < self.goal_min_gain:       # 太近, 不值得当"远处目标"
                continue
            score = d
            if (1 if wx >= pwx else -1) == heading:
                score *= GOAL_HEADING_BONUS  # 同方向优先, 防到达即掉头
            if self._recently_visited((wx, wy)):
                score *= GOAL_VISIT_PENALTY  # 刚去过的地方降权 (别在一处反复刷)
            if score > best_score:
                best, best_score = (wx, wy, kind), score

        if best is not None:
            self._goal = (best[0], best[1])
            self._goal_kind = best[2]
        else:
            # 地形太少 / 都去过 → 朝当前方向推一个远点继续探索 (人也是这么干的)
            self._goal = (pwx + heading * self.goal_explore_push, pwy)
            self._goal_kind = "探索"
        self._goal_t = time.time()
        log.info(f"[PATROL] 新目标 {self._goal_kind} 距离 {abs(self._goal[0]-pwx):.0f}px "
                 f"方向 {'右' if self._goal[0] >= pwx else '左'}")

    def _abandon_goal(self, why: str = "") -> None:
        """放弃当前目标 (并记为去过, 免得立刻又选它)。"""
        if self._goal is not None:
            self._mark_visited(self._goal)
        self._goal = None
        self._goal_kind = ""

    def _mark_visited(self, wpt: tuple) -> None:
        self._visited.append((wpt[0], wpt[1], time.time()))
        if len(self._visited) > GOAL_VISIT_MEMORY:
            self._visited.pop(0)

    def _recently_visited(self, wpt: tuple) -> bool:
        now = time.time()
        for (vx, vy, t) in self._visited:
            if now - t > GOAL_VISIT_FORGET_SEC:
                continue
            if math.hypot(wpt[0] - vx, wpt[1] - vy) <= GOAL_VISIT_RADIUS:
                return True
        return False

    def _walk_to_world(self, controller: GameController, goal_wx: float, brain,
                       cancel=None) -> None:
        """朝世界坐标 goal_wx 连续走一段 (最多 GOAL_WALK_CHUNK 秒)。

        和 _walk_toward 的区别: 目标是**世界坐标**, 镜头滚动不会让目标跑掉,
        所以可以放心地一次走好几秒 —— 这是"大幅度变换位置"的关键。
        """
        pwx = brain.player_world()[0]
        direction = Direction.RIGHT if goal_wx >= pwx else Direction.LEFT
        reliable = brain.player_reliable()
        start = time.time()
        last_wx = pwx
        last_move = start
        controller.key_down(direction.value)
        try:
            while time.time() - start < GOAL_WALK_CHUNK:
                if cancel and cancel():
                    break
                if brain.any_target_in_range():
                    break
                cur_wx = brain.player_world()[0]
                reached = (cur_wx >= goal_wx - GOAL_ARRIVE_PX if direction == Direction.RIGHT
                           else cur_wx <= goal_wx + GOAL_ARRIVE_PX)
                if reached:
                    break
                if not reliable:
                    time.sleep(APPROACH_POLL)
                    continue
                if abs(cur_wx - last_wx) >= MOVE_PX_EPS:
                    last_wx = cur_wx
                    last_move = time.time()
                elif (not self.flat_mode) and time.time() - last_move > self.stuck_jump_timeout:
                    log.info("!! 走向目标被卡, 原地跳一下 !!")
                    controller.diagonal_jump(direction)
                    last_move = time.time()
                time.sleep(APPROACH_POLL)
        finally:
            controller.key_up(direction.value)

    def _idle_pause(self, brain, cancel=None) -> None:
        """站着发会儿呆 (像人在看屏幕/挂机)。分片睡, 有怪进范围或决策变更就立刻醒。"""
        dur = random.uniform(self.patrol_pause_min, self.patrol_pause_max)
        log.debug(f"[PATROL] 停顿 {dur:.1f}s")
        end = time.time() + dur
        while time.time() < end:
            if cancel and cancel():
                return
            if brain.any_target_in_range() or brain.any_target_near(self.patrol_attack_range):
                return
            time.sleep(0.05)

    def _at_edge(self, controller: GameController, px: int, py: int, ropes: list,
                 sup: tuple, is_right: bool, brain, platforms, cancel=None) -> None:
        """到平台边缘: 有梯子 → 爬梯换层; 无梯子但相邻有可跳上的高平台 → 边缘上跳; 否则换向。"""
        edge_x = sup[2] if is_right else sup[1]
        rope = self._rope_near(ropes, edge_x)
        if rope is not None:
            up = self._climb_bias(brain, py)
            log.info(f"🧗 巡逻爬梯 {'上' if up else '下'} (x={rope[0]})")
            self._climb(controller, rope, up, brain, cancel)
            return
        # 无梯子: 尝试边缘上跳 (从低平台跳到相邻高平台), 冷却期内/无高平台才换向
        if cancel and cancel():
            return
        if time.time() - self._last_edge_jump_t >= self.edge_jump_cooldown:
            up = self._find_upper_platform(sup, edge_x, is_right, platforms)
            if up is not None:
                self._last_edge_jump_t = time.time()
                log.info(f"↑ 边缘上跳 -> 高平台 y={up[0]} @ x=[{up[1]},{up[2]}]")
                controller.edge_jump_up(Direction.RIGHT if is_right else Direction.LEFT)
                return
        self.flip()

    def _rope_near(self, ropes: list, x: int):
        """平台边缘 x 附近是否有一根梯子。"""
        for r in ropes:
            if abs(r[0] - x) <= ROPE_GRAB_DX:
                return r
        return None

    def _climb_bias(self, brain, py: int) -> bool:
        """最近怪在上 → 上爬; 在下 → 下爬; 无怪 → 默认上。"""
        nt = brain.nearest_target()
        if nt is not None:
            if nt.cy < py - 30:
                return True
            if nt.cy > py + 30:
                return False
        return True

    def _climb(self, controller: GameController, rope: tuple, up: bool, brain,
               cancel=None) -> None:
        """在梯子处短脉冲攀爬, 进范围或到顶/底或超时停。"""
        rx, rtop, rbot = rope
        for _ in range(CLIMB_MAX_BURSTS):
            if cancel and cancel():
                return
            if brain.any_target_in_range():
                return
            cur_py = brain.player_pos()[1]
            if up and cur_py <= rtop + 30:
                break
            if (not up) and cur_py >= rbot - 30:
                break
            if up:
                controller.climb_up(CLIMB_BURST)
            else:
                controller.climb_down(CLIMB_BURST)

    def flip(self) -> None:
        """强制换向: 放弃当前目标, 下次 patrol 会在反方向重新选点。
        combat_brain 的 PATROLLING 超时看门狗会调它 (公开接口)。"""
        self.patrol_direction = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
        self._abandon_goal("强制换向")
        log.info(f"[PATROL] 换向 --> {self.patrol_direction.name}, 重新选目标")

    # 旧私有名保留一轮, 避免外部调用点漏改后静默 AttributeError
    _flip = flip
