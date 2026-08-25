"""
鼠标输入探针 — 定位 SetCursorPos 完全失效在哪一层
=================================================
背景: 测谎 bot 桌面交互运行, win32api.SetCursorPos 完全不动系统光标。

这个脚本分层测, 一次看清是哪个环节断了:
  只读部分 (直接跑, 不动光标):
    - 本进程 / 前台窗口的 会话ID + 完整性级别 + 是否管理员
    - 前台窗口是哪个 (标题 / 进程) → 是不是游戏
    - 系统光标是否可见 (被游戏隐藏?)、是否被 Clip 住 (剪辑区)
    - 屏幕分辨率 vs 前台窗口矩形 → 是否全屏独占

移动测试 (--moves 才跑, 会把光标移 30px 再移回来):
    A. SetCursorPos 绝对坐标   (现在用的方案)
    B. SendInput 相对位移      (Raw Input 游戏必须用这个)
    C. SendInput 绝对坐标
  每种测试显示 SendInput 返回值 (0 = 被 UIPI 拦截, 非常关键)。

用法:
    python tools/_probe_mouse.py                     # 只读诊断
    python tools/_probe_mouse.py --moves --tag 游戏前台
    python tools/_probe_mouse.py --moves --tag 桌面前台
  建议: 先游戏前台跑一次, 再切到桌面跑一次, 对比 A/B/C 的移动结果。
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import win32api
import win32con
import win32gui
import win32process
import win32security

# ── Windows 结构 ──


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong),
                ("flags", ctypes.c_ulong),
                ("hCursor", ctypes.c_void_p),
                ("ptScreenPos", POINT)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.c_size_t)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("union", INPUT_UNION)]


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
CURSOR_SHOWING = 0x0001

# ── 系统信息 ──


def session_id(pid: int) -> int:
    pid32 = wintypes.DWORD(pid)
    sid = wintypes.DWORD()
    kernel32.ProcessIdToSessionId(pid32, ctypes.byref(sid))
    return sid.value


def integrity_sid(pid: int) -> str:
    """进程完整性级别 SID (S-1-16-xxxx): 4096=low 8192=medium 12288=high 16384=system。"""
    try:
        h = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    except Exception:
        return "(无权限)"
    try:
        tok = win32security.OpenProcessToken(h, win32security.TOKEN_QUERY)
        try:
            data = win32security.GetTokenInformation(tok, win32security.TokenIntegrityLevel)
            sid = data[0] if isinstance(data, tuple) else data
            return win32security.ConvertSidToStringSid(sid)
        finally:
            tok.Close()
    except Exception:
        return "(读取失败)"
    finally:
        h.Close()


def _integrity_num(sid: str) -> int:
    """S-1-16-xxxx → xxxx; 解析失败返回 -1。"""
    try:
        return int(sid.rsplit("-", 1)[1])
    except Exception:
        return -1


def process_name(pid: int) -> str:
    try:
        h = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            import win32process as _wp
            return _wp.GetModuleFileNameEx(h, 0).split("\\")[-1]
        finally:
            h.Close()
    except Exception:
        return "?"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def dpi_awareness() -> str:
    try:
        aw = ctypes.c_int()
        hr = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(aw))
        if hr == 0x80070005:  # E_ACCESSDENIED → 旧式 DPI aware 已设置
            return "已设(旧式/PerMonitor)"
        return {0: "无感知(系统缩放, 坐标会错)", 1: "系统DPI感知", 2: "PerMonitor"}.get(aw.value, f"未知({aw.value})")
    except Exception:
        return "读取失败"


def get_foreground() -> tuple[int, int, str, str]:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return (0, 0, "", "?")
    title = win32gui.GetWindowText(hwnd) or "(无标题)"
    tid, pid = win32process.GetWindowThreadProcessId(hwnd)
    return (hwnd, pid, title, process_name(pid))


# ── 移动测试 ──


def send_input(dx: int, dy: int, absolute: bool) -> int:
    """SendInput 一个鼠标移动事件, 返回注入成功的事件数 (0 = 被拦)。"""
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.union.mi.mouseData = 0
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = 0
    if absolute:
        sw = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        sh = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        inp.union.mi.dx = int(dx / sw * 65535)
        inp.union.mi.dy = int(dy / sh * 65535)
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    else:
        inp.union.mi.dx = dx
        inp.union.mi.dy = dy
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE
    user32.SendInput.restype = ctypes.c_uint
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def test_setcursorpos(x: int, y: int) -> tuple[bool, tuple[int, int]]:
    """SetCursorPos 到目标, 读回实际位置判断是否生效。"""
    start = win32api.GetCursorPos()
    ok = bool(user32.SetCursorPos(x, y))
    after = win32api.GetCursorPos()
    moved = (after != start)
    return (ok and moved, after)


def test_sendinput_rel(dx: int, dy: int) -> tuple[int, tuple[int, int]]:
    start = win32api.GetCursorPos()
    n = send_input(dx, dy, absolute=False)
    after = win32api.GetCursorPos()
    return (n, after)


def test_sendinput_abs(x: int, y: int) -> tuple[int, tuple[int, int]]:
    start = win32api.GetCursorPos()
    n = send_input(x, y, absolute=True)
    after = win32api.GetCursorPos()
    return (n, after)


# ── 输出 ──


def print_system() -> None:
    print("=" * 64)
    print("只读诊断 (不移动光标)")
    print("=" * 64)
    my_pid = win32api.GetCurrentProcessId()
    print(f"[本进程]  pid={my_pid}  管理员={is_admin()}  会话={session_id(my_pid)}  DPI={dpi_awareness()}")
    print(f"[本进程]  完整性={integrity_sid(my_pid)}")

    hwnd, pid, title, pname = get_foreground()
    print("-" * 64)
    if hwnd:
        print(f"[前台窗口] hwnd=0x{hwnd:08X}  pid={pid}  会话={session_id(pid)}  完整性={integrity_sid(pid)}")
        print(f"[前台窗口] 标题 = {title!r}")
        print(f"[前台窗口] 进程 = {pname}")
        if pid != my_pid:
            fg_n = _integrity_num(integrity_sid(pid))
            my_n = _integrity_num(integrity_sid(my_pid))
            if fg_n > my_n:
                print(f"[前台窗口] ⚠️ 游戏({fg_n}) 完整性 > bot({my_n}) → UIPI 会静默拒绝 SetCursorPos/SendInput!")
            elif fg_n < my_n:
                print(f"[前台窗口] bot({my_n}) 高于游戏({fg_n}) → UIPI 不是问题")
            else:
                print(f"[前台窗口] 双方权限相当 ({fg_n}) → UIPI 大概率不是问题")
    else:
        print("[前台窗口] 无 (可能没有活动窗口)")

    # 光标可见性 + 剪辑区
    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if user32.GetCursorInfo(ctypes.byref(ci)):
        vis = "可见" if (ci.flags & CURSOR_SHOWING) else "隐藏⚠️(游戏捕获了鼠标)"
        print("-" * 64)
        print(f"[光标状态] 可见性 = {vis}  位置 = ({ci.ptScreenPos.x},{ci.ptScreenPos.y})")
        rc = RECT()
        if user32.GetClipCursor(ctypes.byref(rc)):
            if (rc.left, rc.top) == (0, 0) and (rc.right, rc.bottom) == (
                win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
                win32api.GetSystemMetrics(win32con.SM_CYSCREEN),
            ):
                print("[剪辑区]   未剪辑 (全屏自由)")
            else:
                print(f"[剪辑区]   ⚠️ 被约束到 ({rc.left},{rc.top})-({rc.right},{rc.bottom}) "
                      f"(宽{rc.right - rc.left}x高{rc.bottom - rc.top})")
    else:
        print("[光标状态] GetCursorInfo 失败")

    # 前台窗口尺寸 vs 屏幕 → 是否全屏
    if hwnd:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        sw = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        sh = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        print("-" * 64)
        print(f"[窗口]     屏幕={sw}x{sh}  前台窗口=({l},{t})-({r},{b})  {r - l}x{b - t}")
        if (r - l, b - t) == (sw, sh):
            print("[窗口]     ⚠️ 前台窗口 == 全屏尺寸 (可能独占全屏, 独占模式下光标渲染归游戏管)")
        else:
            print("[窗口]     非全屏 → 独占全屏排除")


def print_move_tests(tag: str) -> None:
    print()
    print("=" * 64)
    print(f"移动测试 [{tag}]  (每次移动 30px, 会动你的光标)")
    print("=" * 64)
    cx, cy = win32api.GetCursorPos()
    print(f"起点: ({cx},{cy})\n")

    # 目标钳在屏幕内 (起点靠边时反向偏)
    sw = win32api.GetSystemMetrics(win32con.SM_CXSCREEN) - 1
    sh = win32api.GetSystemMetrics(win32con.SM_CYSCREEN) - 1
    def _target(x: int, y: int) -> tuple[int, int]:
        nx = min(sw, max(0, x + 30)) if x + 30 <= sw else x - 30
        ny = min(sh, max(0, y + 30)) if y + 30 <= sh else y - 30
        return (max(0, min(sw, nx)), max(0, min(sh, ny)))

    # A. SetCursorPos
    tx, ty = _target(cx, cy)
    ok, after = test_setcursorpos(tx, ty)
    moved = abs(after[0] - cx) > 5 or abs(after[1] - cy) > 5
    print(f"A. SetCursorPos({tx},{ty})")
    print(f"   返回={ok}  实际位置=({after[0]},{after[1]})  {'✅ 动了' if moved else '❌ 完全没动 (当前方案失效层)'}")
    test_setcursorpos(cx, cy)  # 归位

    # B. SendInput 相对
    time.sleep(0.3)
    bx, by = _target(cx, cy)
    n, after = test_sendinput_rel(bx - cx, by - cy)
    moved = abs(after[0] - cx) > 5 or abs(after[1] - cy) > 5
    print(f"\nB. SendInput 相对 ({bx - cx:+d},{by - cy:+d})")
    print(f"   SendInput返回={n} (0=被拦)  实际位置=({after[0]},{after[1]})  "
          f"{'✅ 动了' if moved else '❌ 没动'}")
    send_input(cx - bx, cy - by, absolute=False)
    time.sleep(0.3)

    # C. SendInput 绝对
    n, after = test_sendinput_abs(bx, by)
    moved = abs(after[0] - cx) > 5 or abs(after[1] - cy) > 5
    print(f"\nC. SendInput 绝对 ({bx},{by})")
    print(f"   SendInput返回={n} (0=被拦)  实际位置=({after[0]},{after[1]})  "
          f"{'✅ 动了' if moved else '❌ 没动'}")
    test_sendinput_abs(cx, cy)
    time.sleep(0.2)

    print()
    print("-" * 64)
    print("判定:")
    print("  A/B/C 都动      → 注入本身没问题, 问题在游戏不读系统光标 (Raw Input), 改用相对位移 +")
    print("                    human_mouse 换成 SendInput MOUSEEVENTF_MOVE")
    print("  A/C 不动, B 动  → 游戏用了 Raw Input, SetCursorPos 没用, 必须走相对位移")
    print("  B/C 返回 0      → SendInput 被 UIPI 拦, 需要同权限/管理员运行 bot")
    print("  A/B/C 全不动    → 注入层本身断了 (会话/驱动问题), 才考虑内核/HID 方案")
    print("-" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description="鼠标输入分层探针")
    ap.add_argument("--moves", action="store_true", help="跑移动测试 (会动光标)")
    ap.add_argument("--tag", default="", help="本次运行标签, 如 '游戏前台'")
    args = ap.parse_args()

    print_system()
    if args.moves:
        print_move_tests(args.tag or f"pid={win32api.GetCurrentProcessId()}")
    else:
        print()
        print("提示: 加 --moves 跑移动测试. 建议游戏前台/桌面前台各跑一次对比:")
        print("    python tools/_probe_mouse.py --moves --tag 游戏前台")
        print("    python tools/_probe_mouse.py --moves --tag 桌面前台")
    return 0


if __name__ == "__main__":
    sys.exit(main())
