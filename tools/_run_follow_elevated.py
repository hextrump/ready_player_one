"""
提权版跟随准确度测试 — 以管理员运行 test_lie_follow.py (remote 后端), 结果写文件
================================================================================
UIPI 证明过: bot 非提权 (Medium) 时, 游戏 (High) 前台会静默丢弃所有鼠标注入。
此封装以管理员 (High) 启动测试, 排除 UIPI 变量, 单独测『远程检测 → 本机跟随』准确度。
另注: Claude 工具会话的 SetCursorPos 无效(非交互桌面), 本机跟随机测试必须走此提权路径。

流程 (由 PowerShell Start-Process -Verb RunAs 触发 UAC):
    1. 弹窗提示: 点确定后打开 'lie-detector-test' 小窗口播放测谎视频
    2. 鼠标自动跟随 90 秒 (视频循环, 覆盖多次测谎事件), 结果写 _follow_accuracy_out_<视频>.txt
    3. 弹窗提示完成

用法: powershell
    Start-Process -FilePath python -ArgumentList 'D:/player/ready_player_one/tools/_run_follow_elevated.py' -Verb RunAs -Wait
    # 可传视频文件名 (data/detect 下): ... _run_follow_elevated.py BV1QxcKzXEHT.mp4
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "test_lie_follow.py")
DATA = os.path.join(os.path.dirname(HERE), "data", "detect")
DEFAULT_VIDEO = "BV1XuySBvEFa.mp4"
DURATION = 90  # 秒
MB_OK = 0x0
MB_ICONINFO = 0x40


def msg(text: str, title: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, text, title, MB_OK | MB_ICONINFO)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    video = os.path.join(DATA, name)
    if not os.path.isfile(video):
        msg(f"视频不存在: {video}", "跟随准确度测试")
        return 1
    stem = os.path.splitext(name)[0]
    out = os.path.join(HERE, f"_follow_accuracy_out_{stem}.txt")
    msg(
        f"以管理员身份运行跟随准确度测试。\n\n"
        f"视频: {name}\n"
        f"将打开 'lie-detector-test' 小窗口播放测谎视频,\n"
        f"鼠标会自动跟随远程 (hhh) 检测结果。\n"
        f"跑 {DURATION} 秒后自动退出。\n\n点确定开始。",
        "跟随准确度测试",
    )
    with open(out, "w", encoding="utf-8") as f:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(
            [sys.executable, TEST, video,
             "--backend", "remote", "--auto-arm", "--auto-quit", str(DURATION)],
            stdout=f, stderr=subprocess.STDOUT, env=env,
        )
    msg(f"完成。结果已写入:\n{out}", "跟随准确度测试")
    return 0


if __name__ == "__main__":
    sys.exit(main())
