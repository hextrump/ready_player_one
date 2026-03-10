"""
Action Translator — 动作翻译器
负责将 NavMesh + A* 寻路生成的 Edge（边）动作序列，
翻译并执行为真实的键盘操作 (通过 GameController)。
"""

import time
import math
from typing import List
from src.navigation.nav_mesh import Edge, Node
from src.brain.game_controller import GameController, Direction
from src.utils.logger import get_logger

log = get_logger("action_translator")

class ActionTranslator:
    def __init__(self, controller: GameController):
        self.ctrl = controller
        
        # 物理常数 (根据冒险岛角色的移动速度进行微调)
        # 假设角色基础移动速度约 250 像素/秒 (大步流星，更快转换)
        self.walk_speed_px_per_sec = 250.0  
        self.climb_speed_px_per_sec = 140.0 

    def execute_path(self, path: List[Edge], nodes: List[Node], check_interrupt=None):
        """
        按顺序执行整条寻路路径。
        优化：合并连续的 WALK 动作，且支持在移动中随时被“发现怪物”打断。
        """
        if not path:
            return

        log.info(f"开始执行导航路径, 共 {len(path)} 步 (支持实时中断...)")
        
        i = 0
        while i < len(path):
            # 每一小步开始前，先看一眼是否需要中断 (打近处的怪)
            if check_interrupt and check_interrupt():
                log.info("!! 发现近处怪物，中断远程寻路 !!")
                break

            edge = path[i]
            n1 = nodes[edge.n1]
            n2 = nodes[edge.n2]
            
            # 1. 尝试合并连续的 WALK
            if edge.action == "WALK":
                current_dir_is_right = n2.x > n1.x
                total_dist = abs(n2.x - n1.x)
                
                # 向后看，合并相同方向的 WALK
                j = i + 1
                while j < len(path):
                    next_edge = path[j]
                    next_n1 = nodes[next_edge.n1]
                    next_n2 = nodes[next_edge.n2]
                    next_dir_is_right = next_n2.x > next_n1.x
                    
                    if next_edge.action == "WALK" and next_dir_is_right == current_dir_is_right:
                        total_dist += abs(next_n2.x - next_n1.x)
                        j += 1
                    else:
                        break
                
                # 执行合并后的阔步走
                duration = total_dist / self.walk_speed_px_per_sec
                direction = Direction.RIGHT if current_dir_is_right else Direction.LEFT
                log.info(f"==> 合并阔步 {direction.value} {duration:.2f}秒")
                
                interrupted = self._sleep_interruptible(duration, direction.value, check_interrupt)
                i = j 
                if interrupted: break
                continue
            
            # 2. 其他非 WALK 动作
            is_right = n2.x > n1.x
            if edge.action == "JUMP":
                self._do_jump(n1, n2, is_right)
            elif edge.action == "DOWN_JUMP":
                self._do_down_jump()
            elif edge.action == "CLIMB_UP":
                self._do_climb(n1, n2, is_up=True, check_interrupt=check_interrupt)
            elif edge.action == "CLIMB_DOWN":
                self._do_climb(n1, n2, is_up=False, check_interrupt=check_interrupt)
            elif edge.action == "JUMP_CLIMB":
                self._do_jump_climb(n1, n2, is_up=True)
            
            time.sleep(0.01)
            i += 1
            
        # 释放按键
        for k in ['left', 'right', 'up', 'down']:
            self.ctrl.key_up(k)

    def _sleep_interruptible(self, duration: float, key: str, check_interrupt) -> bool:
        """带中断检查的按键持有，返回是否被中断"""
        self.ctrl.key_down(key)
        start_t = time.time()
        interrupted = False
        
        while time.time() - start_t < duration:
            if check_interrupt and check_interrupt():
                interrupted = True
                break
            time.sleep(0.05) # 50ms 检查一次环境
            
        self.ctrl.key_up(key)
        return interrupted

    def _do_jump(self, n1: Node, n2: Node, is_right: bool):
        # 带方向的跳跃 (跨越平台)
        direction_str = "right" if is_right else "left"
        log.info(f"↗ 跳跃 {direction_str}")
        
        # 冒险岛经典操作：先按方向，紧接着按跳跃，然后空中持续按住方向
        self.ctrl.key_down(direction_str)
        time.sleep(0.05)
        self.ctrl.key_down('alt')
        time.sleep(0.1)
        self.ctrl.key_up('alt')
        
        # 空中滞空时间，根据距离稍微调整
        dist = abs(n2.x - n1.x)
        flight_time = min(0.6, max(0.3, dist / self.walk_speed_px_per_sec))
        time.sleep(flight_time)
        self.ctrl.key_up(direction_str)

    def _do_down_jump(self):
        # 下跳 (穿透下落)
        log.info("↓ 下跳 (Down + Alt)")
        self.ctrl.key_down('down')
        time.sleep(0.05)
        self.ctrl.key_down('alt')
        time.sleep(0.1)
        self.ctrl.key_up('alt')
        time.sleep(0.4) # 等待掉落
        self.ctrl.key_up('down')

    def _do_climb(self, n1: Node, n2: Node, is_up: bool, check_interrupt=None):
        dist = abs(n2.y - n1.y)
        # 💡 安全冗余：加上 1.0 秒的额外时间，确保绝对能走完绳子/梯子落到实处
        duration = (dist / self.climb_speed_px_per_sec) + 1.0
        key = 'up' if is_up else 'down'
        log.info(f"↕ 攀爬 {key} {duration:.2f}秒 (包含安全冗余)")
        # 💡 绳子上容易卡死，所以强制关闭打怪打断 (传 None)，先老老实实爬完再说
        self._sleep_interruptible(duration, key, None)

    def _do_jump_climb(self, n1: Node, n2: Node, is_up: bool):
        # 悬空绳子：极其重要的操作，跳跃并准确捕捉绳子
        # 冒险岛逻辑：跳起后（甚至起跳瞬间）按住 [上] 就能在空中碰到绳子时自动吸附
        log.info("↑ 强力跳跃抓绳 (Alt -> Up Buffer)")
        
        self.ctrl.key_down('alt') # 起跳
        time.sleep(0.05)
        self.ctrl.key_down('up')  # 立即进入预备抓取状态
        time.sleep(0.1)
        self.ctrl.key_up('alt')
        
        # 持续按住 [上]，直到理论上到达抓取高度 (Kerning City 梯子较高)
        grab_hold_time = 0.7 
        time.sleep(grab_hold_time)
        
        # 抓稳后，如果还需要往上爬，计算余下的垂直行程
        total_dy = abs(n2.y - n1.y)
        already_jumped = 100 # 估计起跳能覆盖的高度
        remaining_dy = total_dy - already_jumped
        
        if remaining_dy > 0:
            # 同样加上安全冗余时间
            duration = (remaining_dy / self.climb_speed_px_per_sec) + 0.8
            log.info(f"↕ 继续攀爬 {duration:.2f}秒 (剩余:{remaining_dy}px + 0.8s安全冗余)")
            time.sleep(duration)
        else:
            time.sleep(0.5) # 即使计算出不需要接着爬，也强行按住上爬一会儿保证上台阶
            
        self.ctrl.key_up('up')
        time.sleep(0.1)

