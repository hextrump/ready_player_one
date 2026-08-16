"""
Ready Player One V7.0 - Full Auto-Hunting Agent (V19 怪 + 名牌玩家定位)
================================================

See  → WindowCapture (background screen grab)
Sense → YOLOv8 (real-time monster detection, RTX 2080 accelerated)
Think → CombatBrain (greedy: attack-in-range → approach-nearest → patrol)
Heal → AutoHealer (independent HP/MP monitoring thread)
Act  → GameController (background DirectInput injection)
"""
import ctypes
import os
import sys
import time
import threading
import argparse

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.capture.window_capture import WindowCapture
from src.perception.hp_monitor import HPMonitor
from src.brain.game_controller import GameController
from src.brain.auto_healer import AutoHealer
from src.brain.combat_brain import CombatBrain
from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("agent_v5")


class AgentV7:
    """
    V7.0 Full Auto-Hunting Agent

    Architecture:
    - See:   WindowCapture grabs 1600x900 frames in background
    - Sense: YOLOv8n (V19 single-class) detects monsters on RTX 2080
    - Think: CombatBrain greedy decision loop drives hunting behavior
    - Heal:  AutoHealer monitors HP/MP independently
    - Act:   GameController injects low-level keyboard scancodes
    """

    def __init__(self, process_name: str = None):
        # 默认进程名从 config.yaml 读取 (game.process_name), 与配置保持一致
        if process_name is None:
            process_name = load_config().get("game", {}).get("process_name", "msw.exe")
        # 1. See
        self.capture = WindowCapture(process_name=process_name)
        if not self.capture.find_window():
            log.error("Game window not found!")
            sys.exit(1)
        log.info(f"Game window found: HWND={self.capture.hwnd}")
        
        # 2. Act
        self.controller = GameController(hwnd=self.capture.hwnd)
        
        # 3. Heal
        self.hp_monitor = HPMonitor()
        self.auto_healer = AutoHealer(
            window_capture=self.capture,
            game_controller=self.controller,
            hp_monitor=self.hp_monitor, # 直接传入同一个引用
            hp_threshold=0.5,
            mp_threshold=0.3,
            check_interval=0.4  # 0.4s 喝药足够响应, 降低和视觉线程抢抓帧锁
        )
        
        # 4. Think + Sense (YOLO)
        self.combat_brain = CombatBrain()

        # 无药看门狗接线: 血药用尽/药水无效 → 停战斗brain (防干等死)
        self.auto_healer.stop_hunting_cb = self._stop_hunting
        self.auto_healer.return_home_key = load_config().get("key", {}).get("return_home", "")

        self._running = False

    def _stop_hunting(self):
        """无药看门狗回调: 停止狩猎 (角色不再拉怪挨打), 等用户补药重开。"""
        self.combat_brain.active_hunting = False
        self.auto_healer.active_hunting = False
        log.warning("!!! 无药看门狗: 已停止狩猎 (血药用尽/药水无效), 补药后按 F1 重新开始 !!!")

    def _poll_hotkeys(self):
        """后台轮询 F1/F 热键 (GetAsyncKeyState, 不装全局钩子)。

        F1=开打, F=停。只检测"按下沿"避免长按连发。
        说明: keyboard 库的 add_hotkey 会安装 WH_KEYBOARD_LL 全局钩子,
        容易被冒险岛反作弊 (nProtect GameGuard) 检测导致掉线, 故改为轮询。
        """
        VK_F1, VK_F = 0x70, 0x46
        prev_f1 = prev_f = False
        while self._running:
            f1 = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F1) & 0x8000)
            f = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F) & 0x8000)
            if f1 and not prev_f1:
                log.info(">>> AUTO HUNTING ENABLED <<<")
                self.combat_brain.active_hunting = True
                self.auto_healer.active_hunting = True
            if f and not prev_f:
                log.info(">>> AUTO HUNTING DISABLED (STANDBY) <<<")
                self.combat_brain.active_hunting = False
                self.auto_healer.active_hunting = False
            prev_f1, prev_f = f1, f
            time.sleep(0.05)

    def start(self):
        log.info("=" * 50)
        log.info("  Ready Player One V7.0 - AUTO HUNTING AGENT (V19 怪 + 名牌玩家)")
        log.info("=" * 50)
        log.info("Modules:")
        log.info("  [See]   WindowCapture ......... OK")
        log.info("  [Sense] YOLOv8n (V19) + RTX .... OK")
        log.info("  [Think] CombatBrain 贪心决策 ... OK")
        log.info("  [Heal]  AutoHealer Thread ..... OK")
        log.info("  [Act]   GameController ........ OK")
        log.info("")
        log.info("Press F1 to start auto-hunting.")
        log.info("Press F to stop auto-hunting.")
        log.info("Press Ctrl+C in console to exit completely.")
        log.info("=" * 50)
        
        # 热键监听: GetAsyncKeyState 轮询线程 (不装全局钩子, 降低反作弊掉线风险)
        self._running = True
        threading.Thread(target=self._poll_hotkeys, daemon=True).start()

        # Start healer thread
        self.auto_healer.start()
        
        # Run combat brain on main thread
        try:
            self.combat_brain.run(
                self.capture, 
                self.controller, 
                hp_monitor=self.hp_monitor, 
                show_vision=True
            )
        except KeyboardInterrupt:
            log.info("User interrupt received. Shutting down...")
        finally:
            self._running = False
            self.combat_brain.stop()
            self.auto_healer.stop()
            log.info("Agent V7.0 shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Ready Player One V7.0 Auto-Hunting Agent")
    parser.add_argument("--process", default=None,
                        help="Game process name (默认读 config.yaml game.process_name)")
    args = parser.parse_args()

    agent = AgentV7(process_name=args.process)
    agent.start()


if __name__ == "__main__":
    main()
