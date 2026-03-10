"""
Ready Player One V5.0 - Full Auto-Hunting Agent
================================================

See  → WindowCapture (background screen grab)
Sense → YOLOv8 (real-time monster detection, RTX 2080 accelerated)  
Think → CombatBrain (state machine: scan → approach → attack → loot → patrol)
Heal → AutoHealer (independent HP/MP monitoring thread)
Act  → GameController (background DirectInput injection)
"""
import os
import sys
import time
import threading
import argparse
import keyboard

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

log = get_logger("agent_v5")


class AgentV5:
    """
    V5.0 Full Auto-Hunting Agent
    
    Architecture:
    - See:   WindowCapture grabs 1600x900 frames in background
    - Sense: YOLOv8n detects monsters at 900+ FPS on RTX 2080
    - Think: CombatBrain state machine drives hunting behavior
    - Heal:  AutoHealer monitors HP/MP independently
    - Act:   GameController injects low-level keyboard scancodes
    """
    
    def __init__(self, process_name: str = "msw.exe"):
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
            mp_threshold=0.3
        )
        
        # 4. Think + Sense (YOLO)
        self.combat_brain = CombatBrain()
        
        self._running = False
    
    def start(self):
        log.info("=" * 50)
        log.info("  Ready Player One V5.0 - AUTO HUNTING AGENT")
        log.info("=" * 50)
        log.info("Modules:")
        log.info("  [See]   WindowCapture ......... OK")
        log.info("  [Sense] YOLOv8n + RTX 2080 .... OK")
        log.info("  [Think] CombatBrain FSM ....... OK")
        log.info("  [Heal]  AutoHealer Thread ..... OK")
        log.info("  [Act]   GameController ........ OK")
        log.info("")
        log.info("Press F1 to start auto-hunting.")
        log.info("Press F to stop auto-hunting.")
        log.info("Press Ctrl+C in console to exit completely.")
        log.info("=" * 50)
        
        # 注册全局热键
        def on_f1():
            log.info(">>> AUTO HUNTING ENABLED <<<")
            self.combat_brain.active_hunting = True
            self.auto_healer.active_hunting = True

        def on_f():
            log.info(">>> AUTO HUNTING DISABLED (STANDBY) <<<")
            self.combat_brain.active_hunting = False
            self.auto_healer.active_hunting = False

        keyboard.add_hotkey('f1', on_f1)
        keyboard.add_hotkey('f', on_f)
        
        self._running = True
        
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
            log.info("Agent V5.0 shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Ready Player One V5.0 Auto-Hunting Agent")
    parser.add_argument("--process", default="msw.exe", help="Game process name")
    args = parser.parse_args()
    
    agent = AgentV5(process_name=args.process)
    agent.start()


if __name__ == "__main__":
    main()
