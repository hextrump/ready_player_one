"""
游戏键盘控制器 V2.0 — True Background (后台输入) 模式。

经过技术升级，现在支持“后台挂机”：
1.  **PostMessage 方案**：不再使用抢夺焦点的 keybd_event。
2.  **ScanCode + lParam 构造**：针对 Unity (MapleStory Worlds) 精确构造 lParam 消息位（含扫描码、扩展键位标志等），模拟底层按键。
3.  **不抢焦点**：移除 SetForegroundWindow，你可以一边刷怪一边刷网页/写代码。

注意事项：
- 游戏窗口不能“最小化”（否则显卡不渲染），但可以被其他窗口“遮挡”。
- 后台截图采用 PrintWindow (PW_RENDERFULLCONTENT)，支持遮挡抓取。
"""

from __future__ import annotations

import ctypes
import random
import time
from enum import Enum

import win32gui
import win32con
import win32process

from src.utils.logger import get_logger

log = get_logger("game_controller")

# keybd_event flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# DirectInput 扫描码
SCAN = {
    "left":  0x4B,
    "right": 0x4D,
    "up":    0x48,
    "down":  0x50,
    "alt":   0x38,
    "a":     0x1E,
    "b":     0x30,
    "s":     0x1F,
    "z":     0x2C,
    "x":     0x2D,
}

# VK codes (keybd_event 第一个参数)
VK = {
    "left":  0x25,
    "right": 0x27,
    "up":    0x26,
    "down":  0x28,
    "alt":   0x12,
    "a":     0x41,
    "b":     0x42,
    "s":     0x53,
    "z":     0x5A,
    "x":     0x58,
}

# 需要 extended key flag 的方向键
EXTENDED = {"left", "right", "up", "down"}


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class GameController:
    """
    MapleStory 键盘控制器。

    使用 AttachThreadInput + keybd_event 发送按键，
    DirectInput 游戏可以正确接收扫描码输入。

    用法:
        ctrl = GameController(hwnd=wc.hwnd)
        ctrl.move_right(0.5)       # 向右走
        ctrl.diagonal_jump(Direction.RIGHT)  # 右斜跳
        ctrl.attack_aoe()          # 群攻
    """

    def __init__(
        self,
        hwnd: int = 0,
        action_delay: float = 0.03,
        anti_detect: bool = True,
    ):
        self._hwnd = hwnd
        self.action_delay = action_delay
        self.anti_detect = anti_detect

    def set_hwnd(self, hwnd: int) -> None:
        """绑定目标窗口句柄。"""
        self._hwnd = hwnd
        log.info(f"控制器绑定后台窗口: hwnd={hwnd}")

    def _make_lparam(self, key_name: str, down: bool) -> int:
        """
        构造 Win32 消息的 lParam 参数。
        对于 Unity (UnityWndClass)，必须包含正确的扫描码以便游戏引擎识别。
        """
        vk = VK[key_name]
        scan_code = SCAN[key_name]
        
        # Win32 lParam 构造规则:
        # bit 0-15: 重复次数 (1)
        # bit 16-23: 扫描码
        # bit 24: 扩展键标志 (方向键为1)
        # bit 29: 上下文代码 (0)
        # bit 30: 之前的状态 (按下为0, 松开为1)
        # bit 31: 转换状态 (按下为0, 松开为1)
        
        lparam = 1 # Repeat count
        lparam |= (scan_code << 16)
        if key_name in EXTENDED:
            lparam |= (1 << 24)
            
        if not down:
            lparam |= (1 << 30)
            lparam |= (1 << 31)
            
        return lparam

    # ── 底层按键 (后台模式) ──

    def key_down(self, key_name: str) -> None:
        """通过 PostMessage 发送按下消息 (不抢焦点，支持后台)"""
        if not self._hwnd: return
        
        vk = VK[key_name]
        lparam = self._make_lparam(key_name, down=True)
        win32gui.PostMessage(self._hwnd, win32con.WM_KEYDOWN, vk, lparam)

    def key_up(self, key_name: str) -> None:
        """通过 PostMessage 发送弹起消息"""
        if not self._hwnd: return
        
        vk = VK[key_name]
        lparam = self._make_lparam(key_name, down=False)
        win32gui.PostMessage(self._hwnd, win32con.WM_KEYUP, vk, lparam)

    def press_key(self, key_name: str, duration: float = 0.05) -> None:
        """按住一段时间后松开。"""
        self.key_down(key_name)
        time.sleep(duration + self._jitter())
        self.key_up(key_name)
        self._post_action()

    def tap_key(self, key_name: str) -> None:
        """快速点按。"""
        self.key_down(key_name)
        time.sleep(0.02 + self._jitter())
        self.key_up(key_name)
        self._post_action()

    # ── 移动 ──

    def move_left(self, duration: float = 0.3) -> None:
        self.press_key("left", duration)

    def move_right(self, duration: float = 0.3) -> None:
        self.press_key("right", duration)

    def move_up(self, duration: float = 0.2) -> None:
        self.press_key("up", duration)

    def move_down(self, duration: float = 0.2) -> None:
        self.press_key("down", duration)

    def move_direction(self, direction: Direction, duration: float = 0.3) -> None:
        self.press_key(direction.value, duration)

    # ── 跳跃 ──

    def jump(self) -> None:
        """原地跳 (Alt)。"""
        self.tap_key("alt")

    def diagonal_jump(self, direction: Direction) -> None:
        """斜跳: 方向 + Alt。"""
        self.key_down(direction.value)
        time.sleep(0.03 + self._jitter())
        self.key_down("alt")
        time.sleep(0.03)
        self.key_up("alt")
        time.sleep(0.12 + self._jitter())
        self.key_up(direction.value)
        self._post_action()

    def jump_down(self) -> None:
        """下跳: ↓ + Alt。"""
        self.key_down("down")
        time.sleep(0.03)
        self.tap_key("alt")
        time.sleep(0.1)
        self.key_up("down")
        self._post_action()

    # ── 爬绳 ──

    def climb_up(self, duration: float = 1.0) -> None:
        self.press_key("up", duration)

    def climb_down(self, duration: float = 1.0) -> None:
        self.press_key("down", duration)

    # ── 药水 ──

    def use_hp_potion(self) -> None:
        self.tap_key("a")

    def use_mp_potion(self) -> None:
        self.tap_key("s")

    # ── 攻击 ──

    def attack_single(self) -> None:
        self.tap_key("z")

    def attack_aoe(self) -> None:
        self.tap_key("x")

    def attack_and_move(self, direction: Direction) -> None:
        self.key_down(direction.value)
        time.sleep(0.05)
        self.tap_key("x")
        time.sleep(0.1 + self._jitter())
        self.key_up(direction.value)
        self._post_action()

    def jump_attack(self, direction: Direction) -> None:
        """跳发攻击：Alt + Z 同时或快速按。用于打击高处怪物。"""
        self.key_down(direction.value)
        time.sleep(0.02)
        self.key_down("alt")  # 跳
        time.sleep(0.05)
        self.tap_key("z")      # 攻
        time.sleep(0.1)
        self.key_up("alt")
        time.sleep(0.1)
        self.key_up(direction.value)
        self._post_action()

    # ── 组合 ──

    def hunt_combo(self, direction: Direction) -> None:
        self.key_down(direction.value)
        time.sleep(0.03)
        self.tap_key("alt")
        time.sleep(0.08)
        self.tap_key("x")
        time.sleep(0.15)
        self.key_up(direction.value)
        time.sleep(0.1)
        self.tap_key("z")
        self._post_action()

    def loot_sweep(self, width: float = 0.5) -> None:
        self.move_left(width / 2)
        time.sleep(0.05)
        self.move_right(width)
        time.sleep(0.05)
        self.move_left(width / 2)

    def idle_move(self) -> None:
        d = random.choice(["left", "right"])
        self.press_key(d, random.uniform(0.05, 0.15))

    def enter_portal(self) -> None:
        self.press_key("up", 0.5)

    # ── 内部 ──

    def _jitter(self) -> float:
        if self.anti_detect:
            return random.uniform(0.005, 0.025)
        return 0.0

    def _post_action(self) -> None:
        time.sleep(self.action_delay + self._jitter())

    def __del__(self):
        self._detach()
