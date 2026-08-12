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
import time

from src.brain.game_controller import GameController, Direction
from src.utils.logger import get_logger

log = get_logger("patrol_mover")

# ===== 移动参数 =====
APPROACH_MAX_SEC = 3.0   # 单次水平走最长秒数 (超时交给外层重决策)
APPROACH_POLL = 0.04     # 行走循环轮询间隔
MOVE_PX_EPS = 3          # 判定"在移动"的像素阈值 (感知 ~5.5fps)
STUCK_JUMP_TIMEOUT = 1.0 # 移动无进度多久判定卡住, 触发反向脱困跳 (感知~5.5fps, 需容忍几帧)

WALK_EDGE_MARGIN = 25    # 巡逻到平台边缘多少 px 内处理 (爬梯/换向)
PATROL_DURATION = 3.0    # 无怪时定时换向间隔
PATROL_STEP = 0.5        # 巡逻单步行走时长
PATROL_ATTACK_RANGE = 220  # 巡逻时距怪多少 px 内才普攻 (防空挥)

SURFACE_TOL_Y = 30       # 点是否落在平台上的 y 容差 (实测怪框/玩家脚底与平台 y 偏移 23-25px)
PLAYER_FOOT_OFFSET = 35  # 玩家中心 → 脚底

JUMP_UP_DY = 120         # 目标高于此且水平近 → 登台跳
JUMP_TO_UPPER_DX = 150   # 登台跳最大水平距离
JUMP_DOWN_DY = 30        # 目标低于此且水平近 → 下跳
JUMP_DOWN_DX = 80        # 下跳最大水平距离

ROPE_GRAB_DX = 30        # 平台边缘附近多少 px 内算"有梯子可爬"
CLIMB_BURST = 0.6        # 巡逻爬梯单次脉冲秒数
CLIMB_MAX_BURSTS = 8     # 单次爬梯最多脉冲数 (防卡死)

FLAT_MODE = False        # True=关闭跳跃/爬梯, 纯平面推图


class PatrolMover:
    def __init__(self):
        self.patrol_direction = Direction.RIGHT
        self._patrol_start_time = time.time()

    # ===== 可及性判断 =====

    def support(self, x: int, feet_y: float, platforms: list):
        """返回 (x, feet_y) 所在平台 (y, x_left, x_right); 不落任何平台则 None (地面)。"""
        for p in platforms:
            if abs(p[0] - feet_y) <= SURFACE_TOL_Y and p[1] <= x <= p[2]:
                return p
        return None

    def is_reachable(self, target, px: int, py: int, platforms: list) -> bool:
        """直接可及: 同一行走面, 或跳发范围内 (上登台/下下跳)。"""
        dx = abs(target.cx - px)
        dy = py - target.cy
        pfeet = py + PLAYER_FOOT_OFFSET
        tfeet = target.cy + target.h / 2
        if self._same_surface(px, pfeet, target.cx, tfeet, platforms):
            return True
        if (not FLAT_MODE) and dy > JUMP_UP_DY and dx <= JUMP_TO_UPPER_DX:
            return True
        if (not FLAT_MODE) and dy < -JUMP_DOWN_DY and dx <= JUMP_DOWN_DX:
            return True
        return False

    def _same_surface(self, px: int, pfeet: float, tx: int, tfeet: float, platforms: list) -> bool:
        pp = self.support(px, pfeet, platforms)
        tp = self.support(tx, tfeet, platforms)
        if pp is not None and tp is not None:
            return pp[0] == tp[0]
        if pp is None and tp is None:
            return abs(pfeet - tfeet) <= 30
        return False

    # ===== 接近 (只处理直接可及) =====

    def approach(self, controller: GameController, target, px: int, py: int,
                 platforms: list, brain) -> None:
        """简单接近: 同面 → 水平走; 上近 → 登台跳; 下近 → 下跳。做完交给外层重决策。"""
        dx = target.cx - px
        dy = py - target.cy
        direction = Direction.RIGHT if target.cx >= px else Direction.LEFT
        if (not FLAT_MODE) and dy > JUMP_UP_DY and abs(dx) <= JUMP_TO_UPPER_DX:
            log.info(f"↑ 登台跳 -> {target.name}")
            controller.diagonal_jump(direction)
            return
        if (not FLAT_MODE) and dy < -JUMP_DOWN_DY and abs(dx) <= JUMP_DOWN_DX:
            log.info(f"↓ 下跳 -> {target.name}")
            controller.jump_down()
            return
        self._walk_toward(controller, target.cx, px, brain)

    def _walk_toward(self, controller: GameController, goal_x: int, px: int, brain) -> None:
        """水平走向 goal_x: 进范围/到达/超时 即停。
        仅在玩家位置可信时做脱困跳 (中心猜测时位置冻结, 会误判卡住狂跳)。"""
        direction = Direction.RIGHT if goal_x >= px else Direction.LEFT
        reliable = brain.player_reliable()
        start = time.time()
        last_px = px
        last_move = time.time()
        controller.key_down(direction.value)
        try:
            while time.time() - start < APPROACH_MAX_SEC:
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
                elif (not FLAT_MODE) and time.time() - last_move > STUCK_JUMP_TIMEOUT:
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
               platforms: list, ropes: list, brain) -> None:
        """地形巡逻: 沿平台走, 到边缘 爬梯/换向; 方向朝最近怪偏置。"""
        nt = brain.nearest_target()
        if nt is not None:
            # 朝最近怪方向走 (覆盖可及性不足时, 巡逻主动靠近)
            self.patrol_direction = Direction.RIGHT if nt.cx >= px else Direction.LEFT
            self._patrol_start_time = time.time()
        elif time.time() - self._patrol_start_time > PATROL_DURATION:
            self._flip()

        sup = self.support(px, py + PLAYER_FOOT_OFFSET, platforms)
        if sup is not None:
            if self.patrol_direction == Direction.RIGHT and px >= sup[2] - WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, True, brain)
                return
            if self.patrol_direction == Direction.LEFT and px <= sup[1] + WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, False, brain)
                return

        controller.move_direction(self.patrol_direction, duration=PATROL_STEP)

        # 附近有怪才普攻 (否则纯走路, 防空挥)
        if brain.any_target_near(PATROL_ATTACK_RANGE):
            controller.attack_single()

    def _at_edge(self, controller: GameController, px: int, py: int, ropes: list,
                 sup: tuple, is_right: bool, brain) -> None:
        """到平台边缘: 有梯子 → 爬梯换层 (方向偏置); 否则换向。"""
        edge_x = sup[2] if is_right else sup[1]
        rope = self._rope_near(ropes, edge_x)
        if rope is not None:
            up = self._climb_bias(brain, py)
            log.info(f"🧗 巡逻爬梯 {'上' if up else '下'} (x={rope[0]})")
            self._climb(controller, rope, up, brain)
            return
        self._flip()

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

    def _climb(self, controller: GameController, rope: tuple, up: bool, brain) -> None:
        """在梯子处短脉冲攀爬, 进范围或到顶/底或超时停。"""
        rx, rtop, rbot = rope
        for _ in range(CLIMB_MAX_BURSTS):
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

    def _flip(self) -> None:
        self.patrol_direction = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
        self._patrol_start_time = time.time()
        log.info(f"[PATROL] 换向 --> {self.patrol_direction.name}")
