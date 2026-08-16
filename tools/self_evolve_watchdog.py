"""
自进化看门狗 — 持续观察 agent.log, 发现异常/问题/无进展就输出一行 (供 Monitor 唤起 Claude)。

设计思想 (事件驱动, 而非定时轮询):
  bot 一直在跑、日志一直在写; 只有"有问题/异常/没进展"才输出, 平时安静不打扰。
  输出格式: [EVOLVE] <TYPE>: <日志行>  → 由 Monitor 转成对 Claude 的唤起。

问题类型 (带冷却, 同一类型 90s 内只报一次, 避免刷屏):
  ERROR        异常/堆栈
  STUCK        巡逻卡住 (角色卡住, 脱困失败)
  BURST_CANCEL burst 被反复取消 (动作键抖动)
  JUMP_LOOP    登台跳失败→冷却循环 (跳不上去卡住)
  HP_DANGER    血量极低反复喝药 (药水无效/濒死)
  NO_PROGRESS  之前活跃但连续 N 分钟无击杀 (卡死/空图)
"""
from __future__ import annotations

import re
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "agent.log"

NO_PROGRESS_MIN = 3        # 连续 N 分钟无击杀 → 无进展
PATTERN_COOLDOWN = 90      # 同一问题类型冷却 (秒)
POLL = 1                   # 轮询间隔 (秒)

PATTERNS = [
    ("ERROR", re.compile(r"\bError\b|Traceback|异常|Exception", re.I)),
    ("STUCK", re.compile(r"巡逻卡住")),
    ("BURST_CANCEL", re.compile(r"决策变更,中止 burst")),
    ("JUMP_LOOP", re.compile(r"登台跳 \d+ 次仍不可打|不可打, 冷却")),
    ("HP_DANGER", re.compile(r"检测血量极低")),
]
ATTACK_RE = re.compile(r"\[ATTACK\] Monster")
ACTIVITY_RE = re.compile(r"\b(INFO|WARNING|ERROR)\b")


def emit(msg: str) -> None:
    print(f"[EVOLVE] {msg}", flush=True)


def main() -> int:
    pos = 0
    if LOG.exists():
        pos = LOG.stat().st_size
    last_fire: dict[str, float] = {}
    last_attack_t: float | None = None
    now = time.time()

    while True:
        try:
            if not LOG.exists():
                time.sleep(POLL)
                continue
            size = LOG.stat().st_size
            if size < pos:          # 日志轮转
                pos = 0
            if size > pos:
                with open(LOG, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
                now = time.time()
                for line in new.splitlines():
                    if ATTACK_RE.search(line):
                        last_attack_t = now
                    for name, pat in PATTERNS:
                        if pat.search(line):
                            if now - last_fire.get(name, 0) > PATTERN_COOLDOWN:
                                last_fire[name] = now
                                emit(f"{name}: {line.strip()}")

            # 无进展: 之前击杀过, 但连续 NO_PROGRESS_MIN 分钟没击杀
            if last_attack_t is not None and now - last_attack_t > NO_PROGRESS_MIN * 60:
                if now - last_fire.get("NO_PROGRESS", 0) > PATTERN_COOLDOWN:
                    last_fire["NO_PROGRESS"] = now
                    emit(f"NO_PROGRESS: {int((now - last_attack_t) / 60)} 分钟无击杀")
            time.sleep(POLL)
        except Exception:
            time.sleep(POLL * 2)


if __name__ == "__main__":
    main()
