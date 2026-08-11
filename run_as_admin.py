"""
一键以管理员身份启动 Ready Player One。
用法: 双击或命令行 `python run_as_admin.py`
机制: 如果当前进程不是管理员,自动调 ShellExecuteW 触发 UAC 重启自身。
"""
import ctypes
import sys
import os
import subprocess


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main():
    if is_admin():
        # 已经是管理员,直接拉起 main.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_py = os.path.join(script_dir, "main.py")
        print(f"[OK] Running as administrator. Launching {main_py}")
        os.chdir(script_dir)
        try:
            subprocess.run([sys.executable, main_py, "--process", "Maplestory_Classic.exe"], check=True)
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted.")
        return

    # 非管理员 → 用 ShellExecuteW ('runas') 触发 UAC
    print("[INFO] Not running as administrator. Requesting elevation...")
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    # SW_SHOWNORMAL = 1
    ret = ctypes.windll.shell32.ShellExecuteW(
        None,           # hwnd
        "runas",        # verb - 触发 UAC
        python_exe,     # file
        f'"{script_path}"',  # parameters
        None,           # working directory
        1               # show command
    )

    if ret <= 32:
        print(f"[ERROR] Elevation failed (return code {ret}). UAC was likely denied.")
        sys.exit(1)
    else:
        print("[OK] Elevation request sent. Please click 'Yes' in the UAC dialog.")
        print("     A new elevated terminal will spawn automatically.")


if __name__ == "__main__":
    main()
