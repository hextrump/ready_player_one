"""
列举当前所有可见窗口的进程名 + 标题, 帮助定位游戏窗口。
用法: python tools/list_windows.py [关键词]
"""
import sys
import psutil
import win32gui
import win32process


def list_windows(filter_keyword: str = ""):
    rows = []

    def enum_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            proc_name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True

        if filter_keyword and filter_keyword.lower() not in title.lower() and filter_keyword.lower() not in proc_name.lower():
            return True

        rows.append((hwnd, proc_name, title))
        return True

    win32gui.EnumWindows(enum_callback, None)
    return rows


if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else ""
    if kw:
        print(f"=== 过滤关键词: {kw!r} ===")
    else:
        print("=== 全部可见窗口 ===")
    print(f"{'HWND':<10} {'进程名':<28} 标题")
    print("-" * 90)
    for hwnd, proc, title in list_windows(kw):
        print(f"0x{hwnd:08X}  {proc:<28} {title}")
    print(f"\n共 {len(list_windows(kw))} 个窗口")
