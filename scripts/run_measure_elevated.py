"""
以管理员身份运行 measure_input_ratio.py 并实时落日志。

用法 (普通权限启动即可, 自动弹 UAC):
  python scripts/run_measure_elevated.py --mode inject --duration 60
  其余参数原样透传给 measure_input_ratio.py (--region/--no-preview 等)。

机制: 非管理员时 ShellExecuteW 'runas' 提权重启自身; 提权后把工具的
stdout/stderr 实时写入 logs/measure_input_ratio_run.log 并等待结束。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "measure_input_ratio_run.log"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main() -> int:
    args = sys.argv[1:]
    if not is_admin():
        # 与 run_as_admin.py 相同的提权模式: lpParameters 只含脚本路径 (+ 透传参数),
        # 不能把 python 可执行路径也塞进参数 (否则 python 会把 exe 当脚本执行而退出)。
        script = str(Path(__file__).resolve())
        params = f'"{script}"' + ((" " + subprocess.list2cmdline(args)) if args else "")
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, params, None, 1)
        if ret <= 32:
            print(f"[ERROR] 提权失败 (return {ret}), 可能 UAC 被拒绝")
            return 1
        print("[OK] 提权请求已发出, 请在 UAC 弹窗点「是」。")
        print("     提权后的测量窗口会自动运行, 日志实时写入 logs/measure_input_ratio_run.log")
        return 0

    # ---- 已提权: 跑工具, 输出实时写日志 ----
    LOG.parent.mkdir(parents=True, exist_ok=True)
    tool = ROOT / "scripts" / "measure_input_ratio.py"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(LOG, "wb") as f:
        # 先写提权启动标记, 确认进程确实起来了
        f.write(("elevated start: " + " ".join(args) + "\n").encode("utf-8", "replace"))
        f.flush()
        p = subprocess.run(
            [sys.executable, str(tool)] + args,
            stdout=f, stderr=subprocess.STDOUT, env=env,
        )
    print(f"[DONE] exit={p.returncode}, 日志: {LOG}")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
