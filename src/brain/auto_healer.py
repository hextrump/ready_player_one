"""
后台自动恢复系统 — 监控 HP/MP 并在阈值下自动喝药。

独立线程运行，不阻塞主 AI 逻辑。

按键绑定 (根据用户配置):
- A: 加血 (HP)
- S: 加蓝 (MP)
"""

import threading
import time
from typing import Optional

from src.capture.window_capture import WindowCapture
from src.perception.hp_monitor import HPMonitor
from src.brain.game_controller import GameController
from src.utils.logger import get_logger

log = get_logger("auto_heal")


class AutoHealer:
    def __init__(
        self,
        window_capture: WindowCapture,
        game_controller: GameController,
        hp_monitor: Optional[HPMonitor] = None, # 支持外部传入
        hp_threshold: float = 0.5,
        mp_threshold: float = 0.3,
        check_interval: float = 0.2, 
    ):
        self.wc = window_capture
        self.ctrl = game_controller
        # 如果外部传了就用外部的，否则自己建（保持兼容）
        self.hp_monitor = hp_monitor if hp_monitor else HPMonitor()
        
        self.hp_threshold = hp_threshold
        self.mp_threshold = mp_threshold
        self.check_interval = check_interval
        self.active_hunting = False
        
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """启动后台加血线程"""
        if self._running:
            return
        
        log.info("大模型医疗兵启动前建档标定中... (将调用Gemini寻找精准的血条位置)")
        frame = self.wc.grab()
        self.hp_monitor.calibrate(frame)
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info(f"大模型建档完毕，高频医疗兵运行中 (HP<{self.hp_threshold*100}%, MP<{self.mp_threshold*100}%)，极速监控频率: {self.check_interval}s")

    def stop(self):
        """停止后台加血线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("大模型自动恢复系统已停止")

    def _run_loop(self):
        while self._running:
            try:
                if not self.active_hunting:
                    time.sleep(self.check_interval)
                    continue

                frame = self.wc.grab()
                vitals = self.hp_monitor.read(frame)

                # 检查 HP
                if vitals.hp_critical:
                    log.warning(f"检测血量极低 ({vitals.hp_display})，自动使用 HP 药水 [A]")
                    self.ctrl.tap_key("a")
                    time.sleep(0.5) # 喝药 CD
                
                # 检查 MP
                if vitals.mp_critical:
                    log.warning(f"检测蓝量极低 ({vitals.mp_display})，自动使用 MP 药水 [S]")
                    self.ctrl.tap_key("s")
                    time.sleep(0.5)

            except Exception as e:
                log.error(f"AutoHealer 异常: {e}")
                
            time.sleep(self.check_interval)
