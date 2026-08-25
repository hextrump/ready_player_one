"""
提权版探针 — 以管理员运行 _probe_mouse.py --moves, 结果写文件
============================================================
验证『提权 bot + 游戏前台』时 UIPI 是否放行鼠标输入。

流程 (由 PowerShell Start-Process -Verb RunAs 触发 UAC):
    1. 第一个弹窗: 让你把游戏窗口切到前台
    2. 点确定后跑移动测试 (光标动 30px 再归位)
    3. 结果写入 _probe_elevated_out.txt, 弹窗提示完成

用法: powershell
    Start-Process -FilePath python -ArgumentList 'D:/player/ready_player_one/tools/_probe_elevated.py' -Verb RunAs -Wait
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "_probe_mouse.py")
OUT = os.path.join(HERE, "_probe_elevated_out.txt")
MB_OK = 0x0
MB_ICONINFO = 0x40


def msg(text: str, title: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, text, title, MB_OK | MB_ICONINFO)


def main() -> int:
    msg("已以管理员身份运行。\n\n"
        "请把游戏(冒险岛)窗口切到前台, 点确定后开始移动测试。\n"
        "光标会移动 30px 再归位。", "提权探针")
    with open(OUT, "w", encoding="utf-8") as f:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(
            [sys.executable, PROBE, "--moves", "--tag", "提权+游戏前台"],
            stdout=f, stderr=subprocess.STDOUT, env=env,
        )
    msg(f"完成。结果已写入:\n{OUT}", "提权探针")
    return 0


if __name__ == "__main__":
    sys.exit(main())
