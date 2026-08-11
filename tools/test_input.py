"""
输入自检脚本 — 验证 AttachThreadInput + keybd_event 是否能到达游戏窗口。
用法:
    1. 启动游戏窗口 (不要最小化)
    2. python main.py --process Maplestory_Classic.exe 不需要 main.py
    3. 单独跑: python tools/test_input.py
    4. 看游戏里角色是否向右走了 1 秒
"""
import sys
import os
import time

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.window_capture import WindowCapture
from src.brain.game_controller import GameController, is_admin, Direction
from src.utils.logger import get_logger

log = get_logger("test_input")


def main():
    print("=" * 60)
    print("  Ready Player One - 输入自检")
    print("=" * 60)

    if not is_admin():
        print("[FAIL] 当前不是管理员! 关闭后用 '以管理员身份运行' 重启")
        return

    # 1. 找游戏窗口
    wc = WindowCapture(process_name="Maplestory_Classic.exe")
    if not wc.find_window():
        print("[FAIL] 未找到 Maplestory_Classic.exe 窗口")
        print("       请确认游戏在前台 (未最小化) 且进程名正确")
        return

    print(f"[OK] 找到窗口: hwnd=0x{wc.hwnd:08X}, size={wc.window_size}")

    # 2. 注入测试
    ctrl = GameController(hwnd=wc.hwnd)

    print()
    print("即将向游戏发送 '右键' 1 秒...")
    print("请观察游戏里角色是否向右走了 1 秒")
    print()

    for i in range(3, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)

    print(">>> 发送 RIGHT key_down")
    ctrl.key_down("right")
    time.sleep(1.0)
    print(">>> 发送 RIGHT key_up")
    ctrl.key_up("right")

    print()
    print("=" * 60)
    print("  自检结果判定:")
    print("  - 角色向右走了 1 格 = 输入正常, 可正常挂机")
    print("  - 角色没动 = keybd_event 被反外挂拦截, 需要:")
    print("    1. 确认游戏窗口在前台 (未最小化)")
    print("    2. 确认游戏未在全屏独占模式 (无边框窗口化)")
    print("    3. 关闭任何键盘宏 / 钩子工具")
    print("    4. 如果以上都不行, 需要用虚拟 HID 驱动方案")
    print("=" * 60)


if __name__ == "__main__":
    main()
