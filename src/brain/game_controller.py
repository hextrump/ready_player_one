"""
游戏键盘控制器 V3.0 — AttachThreadInput + keybd_event (真后台输入)

原理：
1. AttachThreadInput 把我们脚本的输入线程 attach 到游戏窗口所属线程的输入队列,
   这样 keybd_event 发出的按键会被游戏当作"前台焦点"接收。
2. keybd_event + 正确扫描码,直接走驱动层,Unity / DirectInput 游戏都能读到。
3. PostMessage 走的是 Win32 消息队列,对 DirectInput 无效,所以这版彻底放弃它。

要求:
- 必须以管理员权限运行 (AttachThreadInput 跨进程需要)
- 游戏窗口不能最小化 (最小化时 DirectInput 接收不到)
- 可以被其他窗口遮挡 (AttachThreadInput 不需要抢焦点)
"""
from __future__ import annotations

import ctypes
import random
import time
from enum import Enum

import win32api
import win32con
import win32gui
import win32process

from src.utils.logger import get_logger

log = get_logger("game_controller")

# keybd_event flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# DirectInput 扫描码 (MapleStory 用标准 PS/2 Set 1 扫描码)
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
    "c":     0x2E,
    "j":     0x24,
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
    "c":     0x43,
    "j":     0x4A,
}

# 需要 extended key flag 的方向键 (keybd_event 在方向键时要带 0x0001 标志)
EXTENDED = {"left", "right", "up", "down"}


def is_admin() -> bool:
    """检测当前进程是否以管理员权限运行 (AttachThreadInput 需要)。"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class GameController:
    """
    MapleStory Worlds 键盘控制器 (AttachThreadInput + keybd_event 方案)。

    用法:
        ctrl = GameController(hwnd=wc.hwnd)
        ctrl.move_direction(Direction.RIGHT, 0.5)  # 向右走
        ctrl.attack_single()                        # 普攻
        ctrl.jump()                                 # 跳
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

        # 检查管理员权限 (AttachThreadInput 跨进程必须)
        if not is_admin():
            log.warning(
                "⚠️ 当前不是管理员! AttachThreadInput 会被拒绝。"
                "请用 run_as_admin.bat / run_as_admin.py 重启。"
            )

        # 缓存游戏窗口的线程 ID,attach 时用
        self._game_thread = 0
        self._attached = False
        if hwnd:
            self._refresh_thread()

    # ── 线程管理 ──

    def set_hwnd(self, hwnd: int) -> None:
        """绑定目标窗口句柄。"""
        self._hwnd = hwnd
        self._refresh_thread()
        log.info(f"控制器绑定后台窗口: hwnd=0x{hwnd:08X}, thread={self._game_thread}")

    def _refresh_thread(self) -> None:
        """刷新游戏窗口的线程 ID。"""
        if not self._hwnd:
            return
        try:
            self._game_thread, _ = win32process.GetWindowThreadProcessId(self._hwnd)
        except Exception as e:
            log.warning(f"获取窗口线程失败: {e}")
            self._game_thread = 0

    def _attach(self) -> None:
        """把当前线程 attach 到游戏线程的输入队列。"""
        if not self._hwnd or self._game_thread == 0:
            return
        my_thread = win32api.GetCurrentThreadId()
        if my_thread == self._game_thread:
            return  # 已经在同一线程,不需要 attach
        try:
            ok = ctypes.windll.user32.AttachThreadInput(self._game_thread, my_thread, True)
            self._attached = bool(ok)
        except Exception as e:
            log.warning(f"AttachThreadInput 失败: {e}")
            self._attached = False

    def _detach(self) -> None:
        """detach 输入线程。"""
        if not self._attached:
            return
        try:
            my_thread = win32api.GetCurrentThreadId()
            ctypes.windll.user32.AttachThreadInput(self._game_thread, my_thread, False)
        except Exception as e:
            log.warning(f"DetachThreadInput 失败: {e}")
        finally:
            self._attached = False

    # ── 底层按键 (keybd_event) ──

    def _keybd_event(self, key_name: str, down: bool) -> None:
        """
        通过 keybd_event 发送一个按键事件。
        必须先 attach 游戏线程,否则驱动层收不到。
        """
        vk = VK[key_name]
        scan_code = SCAN[key_name]

        flags = 0
        if key_name in EXTENDED:
            flags |= KEYEVENTF_EXTENDEDKEY
        if not down:
            flags |= KEYEVENTF_KEYUP

        # 先 attach (拿焦点),再 keybd_event
        self._attach()
        try:
            ctypes.windll.user32.keybd_event(vk, scan_code, flags, 0)
        finally:
            # 短按不立刻 detach,留给后续 key_up 一起释放,减少 attach 抖动
            if down:
                pass
            else:
                self._detach()

    def key_down(self, key_name: str) -> None:
        """按下某键 (持续按住,直到 key_up)。"""
        if not self._hwnd:
            return
        self._keybd_event(key_name, down=True)

    def key_up(self, key_name: str) -> None:
        """松开某键。"""
        if not self._hwnd:
            return
        self._keybd_event(key_name, down=False)
        # key_up 时统一释放 attach
        self._detach()

    def press_key(self, key_name: str, duration: float = 0.05) -> None:
        """按住一段时间后松开。"""
        self.key_down(key_name)
        time.sleep(duration + self._jitter())
        self.key_up(key_name)
        self._post_action()

    def tap_key(self, key_name: str, post_action: bool = True,
                hold: float | None = None) -> None:
        """
        快速点按。
        post_action=True  (默认): 加 jitter + _post_action,用于普通单次点击
        post_action=False: 纯净快速连点,用于 burst 连按场景
        hold: 强制按键保持时长 (秒)。burst 模式下默认 20~45ms 随机,模拟人手按键深浅。
        """
        self.key_down(key_name)
        if post_action:
            time.sleep(0.02 + self._jitter())
        else:
            # burst 拟人: 按下时长随机,既不像"一碰就松"也不像"死死按住"
            if hold is None:
                time.sleep(random.uniform(0.020, 0.045))
            else:
                time.sleep(hold)
        self.key_up(key_name)
        if post_action:
            self._post_action()

    # ── 移动 ──

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

    def edge_jump_up(self, direction: Direction) -> None:
        """边缘上跳: 方向 + Alt, 空中保持方向稍久 (增大水平位移), 用于跳上相邻高平台。"""
        self.key_down(direction.value)
        time.sleep(0.03 + self._jitter())
        self.key_down("alt")
        time.sleep(0.03)
        self.key_up("alt")
        time.sleep(0.32 + self._jitter())  # 比斜跳多飘 ~0.2s, 更容易贴到高平台
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
        self.tap_key("x")

    def jump_attack(self, direction: Direction) -> None:
        """跳发攻击:Alt + X 同时按,用于打击高处怪物。"""
        self.key_down(direction.value)
        time.sleep(0.02)
        self.key_down("alt")
        time.sleep(0.05)
        self.tap_key("x")
        time.sleep(0.1)
        self.key_up("alt")
        time.sleep(0.1)
        self.key_up(direction.value)
        self._post_action()

    def release_all_key(self) -> None:
        """松开常用按键 (停止 bot 时兜底, 防止方向键被按住)。"""
        for key in ("left", "right", "up", "down", "x", "alt", "a", "s", "j"):
            try:
                self.key_up(key)
            except Exception:
                pass
        self._detach()

    # ── 内部 ──

    def _jitter(self) -> float:
        if self.anti_detect:
            return random.uniform(0.005, 0.025)
        return 0.0

    def _post_action(self) -> None:
        time.sleep(self.action_delay + self._jitter())

    def __del__(self):
        self._detach()