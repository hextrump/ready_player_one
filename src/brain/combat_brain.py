"""
V6.0 Combat Brain - YOLO-Powered Auto-Hunting System
=====================================================

【核心架构准则：单一模型原则 (Single Model Principle)】
1. 性能优化：拒绝使用多个 YOLO 模型并行推理。所有识别任务（玩家、怪物、血条、UI）全部整合
   入单一的 "Super Brain" 模型。这能极大节省显存，降低 GPU 延迟，消除进程间卡顿。
2. 冲突消除：由单模型统一进行 NMS（非极大值抑制），从根本上解决了“玩家被误识别为怪物”
   的逻辑冲突。
3. 逻辑简化：通过 find_targets 一次性吐出所有感知结果，保持“眼手分离”的高效同步。

核心循环：
1. YOLO 扫描 → 定位所有怪物
2. 选择最近的怪物作为目标
3. 判断怪物在角色的左边还是右边
4. 朝目标方向移动 + 攻击
5. 无怪时自动巡逻（左右来回走）
6. 击杀后捡取掉落物
"""
import time
import threading
import math
import cv2
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

from ultralytics import YOLO

from src.capture.window_capture import WindowCapture
from src.brain.game_controller import GameController, Direction
from src.brain.auto_healer import AutoHealer
from src.perception.hp_monitor import HPMonitor
from src.navigation.nav_mesh import NavMeshBuilder
from src.navigation.pathfinder import PathFinder
from src.navigation.action_translator import ActionTranslator
from src.utils.logger import get_logger

log = get_logger("combat_brain")


# ===== 角色位置估计 =====
# 冒险岛角色始终在画面中间偏下的位置
# 基于 1600x900 分辨率，角色大约在 (800, 520) 附近
PLAYER_X = 800
PLAYER_Y = 520

ATTACK_RANGE_X = 120  # 水平攻击距离
ATTACK_RANGE_Y = 60   # 垂直容差
JUMP_ATTACK_RANGE_Y_UP = 160 # 跳发最高打击距离

# 巡逻参数 (阔步流星版)
PATROL_DURATION = 3.0  # 每3秒豪迈换向一次
PATROL_SPEED_SEC = 1.5 # 每次大步跨越 1.5秒 (约400px)
LOOT_DELAY = 1.0       # 击杀后等待掉落的时间

# 屏幕有效区域（排除 UI）
GAME_LEFT = 0
GAME_RIGHT = 1070  # 右侧是物品栏
GAME_TOP = 80
GAME_BOTTOM = 750


class BrainState(Enum):
    STANDBY = "standby"         # 待命状态（仅提供视觉检查）
    SCANNING = "scanning"       # 扫描画面找怪
    APPROACHING = "approaching" # 朝目标移动
    ATTACKING = "attacking"     # 发动攻击
    LOOTING = "looting"         # 捡取掉落物
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
        # 1. 核心综合大模型 Super Brain V11 (高清纠偏版)
        self.super_model = None
        # 优先使用 models/ 文件夹下的统一路径 (方便从 GitHub 拉取代码后直接运行)
        possible_paths = [
            "models/super_brain_v11.pt",
            "runs/detect/super_brain_v11_highres/weights/best.pt",
            "runs/detect/runs/detect/super_brain_v11_highres/weights/best.pt"
        ]
        
        for path in possible_paths:
            try:
                self.super_model = YOLO(path)
                log.info(f"成功激活超级大脑: {path}")
                break
            except:
                continue
                
        if self.super_model is None:
            log.error("无法加载 Super Brain V11，尝试加载本地备份...")
            try:
                # 同样优先 check models
                self.super_model = YOLO("models/monster_v19.pt")
                log.info("已切换至 models/monster_v19 备份模型")
            except:
                try:
                    self.super_model = YOLO("runs/detect/runs/detect/monster_v19_pig/weights/best.pt")
                    log.info("已切换至 runs/detect 旧版本地路径备份")
                except:
                    log.error("核心识别模型完全不可用！")

        # 2. 子系统模型 (地形识别)
        self.terrain_model = None
        try:
            # 优先使用 models 路径
            terrain_path = "models/terrain_v6.pt" if os.path.exists("models/terrain_v6.pt") else "runs/detect/runs/detect/terrain_v6_henesys/weights/best.pt"
            self.terrain_model = YOLO(terrain_path)
            log.info(f"成功激活地形模型: {terrain_path}")
        except:
            log.warning("地形模型加载失败")
            
        # 4. 多线程视觉缓存 (眼手分离)
        self._vision_lock = threading.Lock()
        self._latest_frame = None
        self._latest_perception = {
            "targets": [],
            "player_x": PLAYER_X,
            "player_y": PLAYER_Y,
            "hp_box": None,
            "mp_box": None,
            "fps": 0.0
        }
        self._running = False
        self.state = BrainState.STANDBY
        self.patrol_direction = Direction.RIGHT
        self.kill_count = 0
        self.active_hunting = False

        # 安全区间
        self._last_target_time = time.time()
        self._patrol_start_time = time.time()
        
        # 性能优化：路径冷却与帧率控制
        self._last_path_time = 0
        self._last_path_pos = (0, 0) # (target_x, target_y)
        self._target_fps = 10        # 目标脑波频率 10Hz (够快了)
        self._last_tick_time = time.time()
    
    def _perception_loop(self, capture: WindowCapture):
        """后台视觉线程：维持一秒看3-5次的高度警觉"""
        log.info("[VISION] 后台视觉线程已启动 (锁定 3-5 FPS)")
        last_time = time.time()
        
        while self._running:
            t0 = time.time()
            frame = capture.grab()
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue
            
            # 运行核心检测
            targets, px, py, hp, mp = self.find_targets(frame)
            
            # 更新共享缓存
            with self._vision_lock:
                self._latest_frame = frame.copy()
                self._latest_perception = {
                    "targets": targets,
                    "player_x": px,
                    "player_y": py,
                    "hp_box": hp,
                    "mp_box": mp,
                    "fps": 1.0 / (time.time() - t0 + 0.001)
                }
            
            # 控制频率，一秒看5次左右 (200ms)
            elapsed = time.time() - t0
            time.sleep(max(0.01, 0.2 - elapsed))

    def find_targets(self, frame) -> tuple[List[Target], int, int, Optional[tuple], Optional[tuple]]:
        """单模型全要素扫描：利用 Super Brain V10 实现高精度识别与冲突处理"""
        targets = []
        player_x, player_y = 800, 520
        player_rect = None
        hp_box = None
        mp_box = None
        
        if self.super_model:
            # 使用 imgsz=1280 以匹配训练分辨率，彻底解决定位“歪”的问题
            results = self.super_model(frame, conf=0.5, imgsz=1280, verbose=False)[0]
            
            # --- 第一步：先确定 Player 位置 ---
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_id = int(box.cls[0])
                name = results.names[cls_id]
                
                if name == "Player":
                    player_x, player_y = cx, cy
                    player_rect = (x1, y1, x2, y2)
                    break # 找到主玩家即可
            else:
                if time.time() % 2 < 0.1: # 仅每2秒记录一次，避免刷屏
                    log.warning("Vision: 未检测到 Player，使用预测/默认位置 (800, 520)")
            
            # --- 第二步：收集怪物和其他实体，并过滤掉遮挡玩家的误报 ---
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_id = int(box.cls[0])
                name = results.names[cls_id]
                conf = float(box.conf[0])
                
                if name in ["Monster", "Wild Boar", "Dark Axe Stump", "Pig", "Ribbon Pig"]:
                    # ⚠️ 核心冲突处理：如果怪物框与玩家框重合度过高，判定为 Player 误报成 Monster
                    if player_rect:
                        # 计算交集区域
                        ix1 = max(x1, player_rect[0])
                        iy1 = max(y1, player_rect[1])
                        ix2 = min(x2, player_rect[2])
                        iy2 = min(y2, player_rect[3])
                        
                        if ix2 > ix1 and iy2 > iy1:
                            # 简单的重叠判定：如果怪物中心在玩家框内，直接忽略
                            if player_rect[0] < cx < player_rect[2] and player_rect[1] < cy < player_rect[3]:
                                continue
                                
                    targets.append(Target(
                        name="Monster", cx=cx, cy=cy, w=x2-x1, h=y2-y1, 
                        conf=conf, dist=0.0
                    ))
                elif name == "HP":
                    hp_box = (x1, y1, x2, y2)
                elif name == "MP":
                    mp_box = (x1, y1, x2, y2)
            
        # 计算距离
        for t in targets:
            t.dist = math.hypot(t.cx - player_x, t.cy - player_y)
        
        return targets, player_x, player_y, hp_box, mp_box
    
    def select_target(self, targets: List[Target], player_x: int, player_y: int) -> Optional[Target]:
        """选择最优目标：优先寻找怪物密集的层和区域（如：身边一堆怪），其次考虑距离"""
        if not targets:
            return None

        # --- 密度计算阶段 ---
        # 对于每个怪物，统计半径 R 内（优先同平台）的同类怪数量，作为它的 "密度得分"
        # 玩具城地图平层较多，我们设定：
        # - Y 轴差距 < 80 认为在同层附近
        # - X 轴差距 < 350 认为是 "比较扎堆可以顺手清"
        for t in targets:
            density_score = 0
            for other in targets:
                if t is other: continue
                dx = abs(t.cx - other.cx)
                dy = abs(t.cy - other.cy)
                if dx < 350 and dy < 80:
                    density_score += 1
            # 将密度分挂载到 target 实例上（临时属性）
            t.density_score = density_score
            
        # --- 综合评分打分阶段 ---
        # 评分公式： 密度得分 (权重极高) - 距离惩罚 + 同层奖励
        best_target = None
        best_score = -float('inf')

        for t in targets:
            # 基础奖励：密度，每多一个怪 +1000 分，强制机器人倾向于去怪多的地方
            score = t.density_score * 1000 
            
            # ⚠️反呆滞补丁：如果这个怪物已经在我们身边可以直接打到，无视“远方这群怪”的诱惑，直接设为首选！
            if self.is_in_attack_range(t, player_x, player_y, buffer_x=-10):
                score += 1000000
                
            # 距离惩罚：距离越远扣分越多，保证如果是同样密集的几堆怪，去最近的一堆
            score -= t.dist
            
            # 同层奖励：优先打在同一高度的（防止盲目上下跳跃）
            dy_player = abs(t.cy - player_y)
            if dy_player <= ATTACK_RANGE_Y * 2:
                score += 500  # 给予很大优势，优先平推当前视线层
                
            if score > best_score:
                best_score = score
                best_target = t
                
        return best_target
    
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
    
    def run(self, capture: WindowCapture, controller: GameController, hp_monitor: Optional[HPMonitor] = None, show_vision: bool = True):
        """主战斗循环"""
        self._running = True
        log.info("=== Combat Brain V6.0 (Unified Entity & NavMesh) ONLINE ===")
        log.info(f"State: {self.state.value}")
        
        # 初始化导航组件
        nav_builder = NavMeshBuilder()
        pathfinder = PathFinder(nav_builder)
        translator = ActionTranslator(controller)
        
        # 保存上一帧导航画面的全局变量（仅为了可视化）
        last_nav_path = None
        last_nav_start = None
        self._last_path_pos = (0, 0) # (target_x, target_y)
        self._target_fps = 10.0       # 恢复灵敏脑波：10Hz (支持大步流星，灵敏反应)
        self._last_tick_time = time.time()
    
        if show_vision:
            cv2.namedWindow("Agent V5.1 Vision", cv2.WINDOW_NORMAL)
            
        # 启动后台视觉线程
        vision_thread = threading.Thread(target=self._perception_loop, args=(capture,), daemon=True)
        vision_thread.start()

        # 定义中断检查回调：如果在走路的过程中，眼睛（后台线程）看到了近处的怪，就立刻停下
        def check_nearby_monster():
            with self._vision_lock:
                perc = self._latest_perception
                tgs = perc["targets"]
                px = perc["player_x"]
                py = perc["player_y"]
                for t in tgs:
                    # 如果有任何怪进了攻击范围（不论是地面还是跳发范围）
                    if self.is_in_attack_range(t, px, py):
                        return True
            return False
        
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
            player_x = perc["player_x"]
            player_y = perc["player_y"]
            hp_box = perc["hp_box"]
            mp_box = perc["mp_box"]
            player_box = None # 初始化
            
            # 使用识别到的 HP / MP 框估算当前比例
            if hp_box:
                hx1, hy1, hx2, hy2 = hp_box
                hp_w = hx2 - hx1
                # 假设满血长度大约为 190 像素（根据之前校准得知）
                cur_hp_pct = min(1.0, max(0.0, hp_w / 190.0))
                # 将血量信息传递给 auto_healer
                if hasattr(hp_monitor, 'current_hp_pct'):
                    hp_monitor.current_hp_pct = cur_hp_pct
                    
            if mp_box:
                mx1, my1, mx2, my2 = mp_box
                mp_w = mx2 - mx1
                cur_mp_pct = min(1.0, max(0.0, mp_w / 190.0))
                if hasattr(hp_monitor, 'current_mp_pct'):
                    hp_monitor.current_mp_pct = cur_mp_pct
            
            # 记录玩家包围盒以便绘制
            player_box = None
            if self.super_model is not None:
                player_box = (player_x - 20, player_y - 40, player_x + 20, player_y + 40)
            
            if targets:
                self._last_target_time = time.time()
                self._patrol_stuck_count = 0
                
                # 选择目标
                best = self.select_target(targets, player_x, player_y)
                
                # ==== 只有在 AUTO HUNTING 激活时才执行动作 ====
                if not self.active_hunting:
                    self.state = BrainState.STANDBY
                else:
                    if best and self.is_in_attack_range(best, player_x, player_y):
                        # 在攻击范围内 → 发动攻击！
                        self.state = BrainState.ATTACKING
                        direction = self.get_direction_to_target(best, player_x)
                        
                        log.info(f"[ATTACK] {best.name} @ ({best.cx},{best.cy}) "
                                 f"dist={best.dist:.0f} dy={player_y - best.cy:.0f} conf={best.conf:.2f}")
                        
                        direction = self.get_direction_to_target(best, player_x)
                        
                        # 核心逻辑：区分地面攻击和空中跳发
                        dy = player_y - best.cy
                        if dy > 60:
                            # 怪物在头顶上方：执行豪迈的跳发补刀任务一次做完
                            log.info(f"↑ 执行跳跃攻击 -> {best.name}")
                            controller.jump_attack(direction)
                        else:
                            # 怪物在面前：连续爆裂攻击 (增加攻击次数至 10 次，确保彻底清理)
                            controller.key_down(direction.value)
                            time.sleep(0.02)
                            for _ in range(10):
                                controller.tap_key("z")
                                time.sleep(0.01)
                            controller.key_up(direction.value)
                        
                        self.kill_count += 1
                        # 💡 战术停顿：打完怪停 200ms 观察四周，防止立刻起步导致错过身边新刷的怪
                        time.sleep(0.2)
                    elif best:
                        # ==== 智能多层寻路 ====
                        self.state = BrainState.APPROACHING
                        path = None
                        # 3. 寻路冷却逻辑：对于蹦蹦跳跳的蓝水灵，增加容差 (150px)
                        target_moved = math.hypot(best.cx - self._last_path_pos[0], best.cy - self._last_path_pos[1])
                        path_cooldown = (time.time() - self._last_path_time) < 1.5

                        if self.terrain_model is not None and (target_moved > 150 or not path_cooldown):
                            # 只有目标大幅瞬移或距离上次算路已久，才重新扫描地形和算路
                            res = self.terrain_model(frame, conf=0.4, verbose=False)[0]
                            plats, ropes = [], []
                            for box in res.boxes:
                                c = int(box.cls[0])
                                n = res.names[c]
                                px1, py1, px2, py2 = map(int, box.xyxy[0])
                                if n == 'Platform': plats.append((px1, px2, (py1+py2)//2))
                                elif n == 'Rope': ropes.append(((px1+px2)//2, py1, py2))
                            
                            nav_builder.build_graph(plats, ropes)
                            path = pathfinder.get_path(player_x, player_y, best.cx, best.cy)
                            
                            if path:
                                self._last_path_time = time.time()
                                self._last_path_pos = (best.cx, best.cy)
                        
                        # ── 寻路中断逻辑: 只有怪物离得足够近才停下，且保持大步流星的冲劲 ──
                        def check_nearby_monster():
                            with self._vision_lock:
                                perception_targets = self._latest_perception["targets"]
                                px = self._latest_perception["player_x"]
                                py = self._latest_perception["player_y"]
                            if not perception_targets: return False
                            # 稍微放宽中断判定范围 (buffer_x=-20)，比刚才更敏感一点
                            for t in perception_targets:
                                if self.is_in_attack_range(t, px, py, buffer_x=-20):
                                    return True
                            return False

                        if path:
                            log.info(f"[APPROACH] 执行导航路径 -> {best.name}, 共 {len(path)} 步 (支持动态拦截)")
                            translator.execute_path(path, nav_builder.nodes, check_interrupt=check_nearby_monster)
                        else:
                            # ==== 直线阔步分支：大步流星，果敢行进 ====
                            direction = self.get_direction_to_target(best, player_x)
                            
                            # 策略：如果离得近 (140~300px)，迈半步 (0.5s)；如果很远，迈阔步 (1.5s)
                            move_duration = 1.5 if best.dist > 300 else 0.5
                            log.info(f"[MOVE-FLAT] {best.name} @ {best.dist:.0f}px, 豪迈迈步 {move_duration}s")
                            
                            controller.key_down(direction.value)
                            start_t = time.time()
                            interrupted = False
                            while time.time() - start_t < move_duration:
                                # 💡 阔步冲劲：前 0.4s 绝不回头，保持“大步流星”的惯性
                                if (time.time() - start_t > 0.4) and check_nearby_monster():
                                    log.info(f"!! 发现截击机会，停步开火 !! (进度: {time.time()-start_t:.1f}s)")
                                    interrupted = True
                                    break
                                time.sleep(0.04)
                            controller.key_up(direction.value)
                            
                            if not interrupted:
                                time.sleep(0.05) # 短暂硬直，准备下一击
            else:
                if not self.active_hunting:
                    self.state = BrainState.STANDBY
                else:
                    # 没有怪物 → 巡逻或捡东西
                    idle_time = time.time() - self._last_target_time
                    
                    if idle_time < LOOT_DELAY:
                        # 刚打完怪，等掉落物+捡取
                        self.state = BrainState.LOOTING
                        controller.loot_sweep(width=0.4)
                        time.sleep(0.2)
                    else:
                        # 真的没怪了，阔步巡逻
                        self.state = BrainState.PATROLLING
                        if time.time() - self._patrol_start_time > PATROL_DURATION:
                            self.patrol_direction = Direction.LEFT if self.patrol_direction == Direction.RIGHT else Direction.RIGHT
                            self._patrol_start_time = time.time()
                            log.info(f"[PATROL] 豪迈换向 --> {self.patrol_direction.name}")
                        
                        # 走大步：执行一次长久的阔步 (1.5秒)
                        controller.move_direction(self.patrol_direction, duration=PATROL_SPEED_SEC)
                        # 巡逻中也攻击（可能有视野外的怪被走到了）
                        controller.attack_single()
            
            # 渲染可视化界面
            if show_vision:
                vis_frame = frame.copy()
                
                # 绘制玩家位置
                cv2.circle(vis_frame, (player_x, player_y), 5, (255, 50, 50), -1)
                cv2.putText(vis_frame, "Player", (player_x - 20, player_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
                
                # 绘制怪物
                for t in targets:
                    tx1, ty1 = t.cx - t.w // 2, t.cy - t.h // 2
                    tx2, ty2 = t.cx + t.w // 2, t.cy + t.h // 2
                    color = (0, 165, 255)
                    cv2.rectangle(vis_frame, (tx1, ty1), (tx2, ty2), color, 2)
                    cv2.putText(vis_frame, f"{t.name} {t.dist:.0f}px", (tx1, ty1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                # 绘制 Player
                if player_box:
                    px1, py1, px2, py2 = player_box
                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(vis_frame, "Player", (px1, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                # 绘制 HP / MP
                if hp_box:
                    hx1, hy1, hx2, hy2 = hp_box
                    cv2.rectangle(vis_frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                    cv2.putText(vis_frame, "HP Bar", (hx1, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if mp_box:
                    mx1, my1, mx2, my2 = mp_box
                    cv2.rectangle(vis_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                    cv2.putText(vis_frame, "MP Bar", (mx1, my1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                
                # 把上一帧算出的 NavMesh 路径画上去（如果有）
                if last_nav_path and self.state == BrainState.APPROACHING:
                    try:
                        vis_frame = pathfinder.draw_path(vis_frame, last_nav_path, last_nav_start, last_nav_target)
                    except:
                        pass
                
                # 绘制状态和帧率
                fps = 1.0 / max(0.001, time.time() - t0)
                status_color = (0, 0, 255) if not self.active_hunting else (0, 255, 0)
                status_text = f"State: {self.state.value} | FPS: {fps:.0f} | Kills: {self.kill_count}"
                cv2.putText(vis_frame, "ACTIVE" if self.active_hunting else "STANDBY (Press F1 to Start, F to Stop)", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(vis_frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 处理 HP/MP 显示
                if hp_monitor:
                    stats = hp_monitor.read(frame)
                    hp, mp = stats.hp_percent * 100, stats.mp_percent * 100
                    hp_text = f"HP: {hp:.1f}%" if hp > 0 else "HP: ???%"
                    mp_text = f"MP: {mp:.1f}%" if mp > 0 else "MP: ???%"
                    cv2.putText(vis_frame, hp_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(vis_frame, mp_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    # 在画面上画出监控遮罩，让用户检查
                    if hp_monitor.is_calibrated:
                        hx, hy, hw, hh = hp_monitor.hp_bbox
                        mx, my, mw, mh = hp_monitor.mp_bbox
                        cv2.rectangle(vis_frame, (hx, hy), (hx+hw, hy+hh), (0, 0, 255), 2)
                        cv2.putText(vis_frame, "HP BOX", (hx, hy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(vis_frame, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
                        cv2.putText(vis_frame, "MP BOX", (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # 【响应用户】：视频直接遮罩 (Video direct mask overlay)
                        if hasattr(hp_monitor, 'last_hp_mask') and hp_monitor.last_hp_mask is not None:
                            import numpy as np
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
                cv2.imshow("Agent V5.1 Vision", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False
                    break
            else:
                # 短暂等待避免操作过于频繁
                time.sleep(0.08)
    
    def stop(self):
        self._running = False
        log.info(f"Combat Brain stopped. Total attacks: {self.kill_count}")
        cv2.destroyAllWindows()
