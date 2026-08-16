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

# 无药看门狗: HP 持续低于阈值多少秒判定"血药用尽/药水无效" (防干等死)
NO_POTION_TIMEOUT = 12.0
# 回城键 (配置; 空 = 只停止狩猎, 不按回城)
RETURN_HOME_KEY = ""


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
        self.stop_hunting_cb = None    # 无药看门狗触发时回调 (Agent 接线停战斗brain)
        self.return_home_key = RETURN_HOME_KEY  # 回城键 (可被 Agent 从 config 覆盖; 空=只停不打)

        self._hp_low_start = None     # HP 开始低于阈值的时间 (无药看门狗)
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
                    # 无药看门狗: HP 持续低于阈值 → 药水没作用/血药用尽
                    if self._hp_low_start is None:
                        self._hp_low_start = time.time()
                    if time.time() - self._hp_low_start > NO_POTION_TIMEOUT:
                        log.warning(f"!!! HP 持续低于阈值 {NO_POTION_TIMEOUT:.0f}s (血药用尽/药水无效), 停止狩猎防死亡 !!"
                                    f" (最后读数 {vitals.hp_display})")
                        if self.return_home_key:
                            self.ctrl.tap_key(self.return_home_key)
                        self.active_hunting = False
                        if self.stop_hunting_cb:
                            try:
                                self.stop_hunting_cb()
                            except Exception:
                                pass
                        self._hp_low_start = None  # 防连报 (等用户手动处理)
                        time.sleep(5)  # 停止后冷静一下
                        continue
                    log.warning(f"检测血量极低 ({vitals.hp_display})，自动使用 HP 药水 [A]")
                    self.ctrl.tap_key("a")
                    time.sleep(0.5) # 喝药 CD
                else:
                    self._hp_low_start = None  # HP 恢复 → 重置看门狗

                # 检查 MP
                if vitals.mp_critical:
                    log.warning(f"检测蓝量极低 ({vitals.mp_display})，自动使用 MP 药水 [S]")
                    self.ctrl.tap_key("s")
                    time.sleep(0.5)

            except Exception as e:
                log.error(f"AutoHealer 异常: {e}")
                
            time.sleep(self.check_interval)
