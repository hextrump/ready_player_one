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
PATROL_CHASE_RANGE = 450   # 巡逻方向只朝该距离内的怪偏置 (防止被远处/异层怪锁方向)
PATROL_STUCK_TIMEOUT = 1.0  # 画面静止多久判定卡住 (名牌失效也能用: 相机不动=玩家没动)
MOTION_MOVING_THRESHOLD = 3.0  # 帧间运动量高于此才算"画面在动" (调低=更容易判定卡住)

SURFACE_TOL_Y = 30       # 点是否落在平台上的 y 容差 (实测怪框/玩家脚底与平台 y 偏移 23-25px)
PLAYER_FOOT_OFFSET = 35  # 玩家中心 → 脚底

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
        self.patrol_direction = Direction.RIGHT
        self._patrol_start_time = time.time()
        self._last_motion_t = time.time()   # 最近一次画面有动静的时间 (卡住检测)
        self._jump_count = {}               # 目标粗网格key -> 连续登台跳次数
        self._blocked = {}                  # 目标粗网格key -> 冷却截止时间 (防跳-loop)
        self._last_edge_jump_t = 0.0        # 最近一次边缘上跳时间 (防热循环)
        self._last_patrol_x = None          # 上次巡逻时的玩家位置 (卡住检测主信号)
        self._last_patrol_y = None

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
            if sup_y - p[0] > EDGE_JUMP_MAX_DY: # 高度差超过单跳范围
                continue
            if is_right:
                if p[1] <= edge_x + EDGE_JUMP_MAX_GAP and p[2] >= edge_x:
                    return p
            else:
                if p[2] >= edge_x - EDGE_JUMP_MAX_GAP and p[1] <= edge_x:
                    return p
        return None

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
            return not self._is_blocked(self._tkey(target))   # 跳不上去冷却期内视为不可及
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
                 platforms: list, brain, cancel=None) -> None:
        """简单接近: 同面 → 水平走; 上近 → 登台跳 (先贴近平台边缘再跳); 下近 → 下跳。做完交给外层重决策。
        cancel: 可选回调, True=决策已变更, 中止走动。"""
        dx = target.cx - px
        dy = py - target.cy
        direction = Direction.RIGHT if target.cx >= px else Direction.LEFT
        if (not FLAT_MODE) and dy > JUMP_UP_DY and abs(dx) <= JUMP_TO_UPPER_DX:
            key = self._tkey(target)
            if self._is_blocked(key):
                return  # 冷却期内不再跳, 交给巡逻 (用移动/地形找别的路)
            # 若该侧存在可跳上的高平台, 且离平台边缘还远 → 先走向边缘, 贴近了再跳
            # (单次斜跳水平位移有限, 从中间起跳上不去, 是之前卡住的根因)
            sup = self.support(px, py + PLAYER_FOOT_OFFSET, platforms)
            is_right = direction == Direction.RIGHT
            if sup is not None:
                edge_x = sup[2] if is_right else sup[1]
                up = self._find_upper_platform(sup, edge_x, is_right, platforms)
                if up is not None and abs(px - edge_x) > EDGE_JUMP_MAX_GAP + 20:
                    self._walk_toward(controller, edge_x, px, brain, cancel)
                    return
            cnt = self._jump_count.get(key, 0) + 1
            self._jump_count[key] = cnt
            if cnt >= JUMP_FAIL_LIMIT:
                log.info(f"!! 登台跳 {cnt} 次仍不可打, 冷却该目标, 改巡逻 !!")
                self._blocked[key] = time.time() + BLOCK_COOLDOWN
                self._jump_count.pop(key, None)
                return
            log.info(f"↑ 登台跳({cnt}/{JUMP_FAIL_LIMIT}) -> {target.name}")
            controller.diagonal_jump(direction)
            return
        if (not FLAT_MODE) and dy < -JUMP_DOWN_DY and abs(dx) <= JUMP_DOWN_DX:
            log.info(f"↓ 下跳 -> {target.name}")
            controller.jump_down()
            return
        self._walk_toward(controller, target.cx, px, brain, cancel)

    def _walk_toward(self, controller: GameController, goal_x: int, px: int, brain,
                     cancel=None) -> None:
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
               platforms: list, ropes: list, brain, cancel=None) -> None:
        """地形巡逻: 沿平台走, 到边缘 爬梯/换向; 方向朝最近怪偏置。
        卡住检测: 名牌失效时玩家坐标不更新, 用"画面是否静止"判断玩家是否真的在动。"""
        if cancel and cancel():
            return
        # 卡住检测: 玩家位置在动 OR 画面在动 → 没卡住。
        # 主信号用玩家位置 (名牌) — 静态相机里走动画面不动, 用画面运动会误判卡住;
        # 画面运动做名牌失效时的兜底。
        cur_px, cur_py = brain.player_pos()
        moved = (self._last_patrol_x is not None
                 and (abs(cur_px - self._last_patrol_x) >= MOVE_PX_EPS
                      or abs(cur_py - self._last_patrol_y) >= MOVE_PX_EPS))
        self._last_patrol_x, self._last_patrol_y = cur_px, cur_py
        if moved or brain.world_moving():
            self._last_motion_t = time.time()
        elif time.time() - self._last_motion_t > PATROL_STUCK_TIMEOUT:
            log.info("!! 巡逻卡住(位置/画面都不动), 反向脱困跳 + 换向 !!")
            back = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
            controller.diagonal_jump(back)
            self._flip()
            self._last_motion_t = time.time()
            return

        nt = brain.nearest_target()
        chase = False
        if nt is not None:
            d = math.hypot(nt.cx - px, nt.cy - py)
            if d <= PATROL_CHASE_RANGE:
                # 附近怪才朝其方向走 (远处/异层怪不锁方向, 防被带偏一直走)
                self.patrol_direction = Direction.RIGHT if nt.cx >= px else Direction.LEFT
                self._patrol_start_time = time.time()
                chase = True
        if not chase and time.time() - self._patrol_start_time > PATROL_DURATION:
            self._flip()

        sup = self.support(px, py + PLAYER_FOOT_OFFSET, platforms)
        if sup is not None:
            if self.patrol_direction == Direction.RIGHT and px >= sup[2] - WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, True, brain, platforms, cancel)
                return
            if self.patrol_direction == Direction.LEFT and px <= sup[1] + WALK_EDGE_MARGIN:
                self._at_edge(controller, px, py, ropes, sup, False, brain, platforms, cancel)
                return

        controller.move_direction(self.patrol_direction, duration=PATROL_STEP)

        # 附近有怪才普攻 (否则纯走路, 防空挥)
        if cancel and cancel():
            return
        if brain.any_target_near(PATROL_ATTACK_RANGE):
            controller.attack_single()

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
        if time.time() - self._last_edge_jump_t >= EDGE_JUMP_COOLDOWN:
            up = self._find_upper_platform(sup, edge_x, is_right, platforms)
            if up is not None:
                self._last_edge_jump_t = time.time()
                log.info(f"↑ 边缘上跳 -> 高平台 y={up[0]} @ x=[{up[1]},{up[2]}]")
                controller.edge_jump_up(Direction.RIGHT if is_right else Direction.LEFT)
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

    def _flip(self) -> None:
        self.patrol_direction = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
        self._patrol_start_time = time.time()
        log.info(f"[PATROL] 换向 --> {self.patrol_direction.name}")
