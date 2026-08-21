"""
BrainLedger — 状态总线的"账本"层 (低频、可审计、异步落盘)。

设计思想: 日志是给人看的叙述, 账本是给系统看的事实。
  bot 跑一晚上, 我们要能回答的问题是: 卡在哪个状态? 卡了多久? 身份丢过几次?
  这些不能靠 tail 日志用眼睛数 —— 必须是可查询的行。

职责:
- 只记**状态转换级别**的事实 (每秒最多几条), 不记每帧感知。
- 写入走后台线程 + 队列: SQLite 落盘绝不阻塞决策循环 (决策循环 10Hz, 卡一下就是丢帧)。
- 队列满则丢弃最旧 (账本可以缺页, 决策不能卡)。

查询 (事后分析):
    from src.state.global_bus import GlobalBus
    from src.state.events import EventType
    gbus = GlobalBus()
    gbus.query_events(EventType.STATE_CHANGED, limit=100)
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from src.state.events import EventType, GameEvent
from src.utils.logger import get_logger

log = get_logger("ledger")

QUEUE_SIZE = 256


class BrainLedger:
    """把大脑的关键事实异步写进 GlobalBus (SQLite)。失败不影响 bot 运行。"""

    def __init__(self, db_path: str | None = None, enabled: bool = True):
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._gbus = None
        self._running = False
        self.enabled = enabled
        self.dropped = 0
        if not enabled:
            return
        try:
            from src.state.global_bus import GlobalBus
            self._gbus = GlobalBus(db_path) if db_path else GlobalBus()
        except Exception as e:
            log.warning(f"[LEDGER] 账本不可用 (继续运行, 只是不落盘): {e}")
            self.enabled = False
            return
        self._running = True
        self._thread = threading.Thread(target=self._writer, daemon=True, name="ledger")
        self._thread.start()

    # ── 记账接口 (决策线程调用, 恒不阻塞) ──

    def state_changed(self, old: str, new: str, reason: str, dwell: float, seq: int) -> None:
        self._put(EventType.STATE_CHANGED, {
            "from": old, "to": new, "reason": reason,
            "dwell_sec": round(dwell, 2), "seq": seq,
        }, priority=1)

    def identity_lost(self, last_pos: tuple, miss_frames: int, rejects: int) -> None:
        self._put(EventType.ERROR, {
            "what": "player_identity_lost",
            "last_pos": list(last_pos), "miss_frames": miss_frames, "rejects": rejects,
        }, priority=2)

    def identity_recaptured(self, pos: tuple, source: str, lost_for: float) -> None:
        self._put(EventType.PLAYER_DETECTED, {
            "what": "player_identity_recaptured",
            "pos": list(pos), "source": source, "lost_for_sec": round(lost_for, 2),
        }, priority=1)

    def kill(self, name: str, hits: int, elapsed: float, total: int) -> None:
        self._put(EventType.ACTION_COMPLETED, {
            "what": "kill", "monster": name, "hits": hits,
            "elapsed_sec": round(elapsed, 2), "total_kills": total,
        }, priority=0)

    def stuck(self, state: str, elapsed: float, action: str) -> None:
        self._put(EventType.ERROR, {
            "what": "state_timeout", "state": state,
            "elapsed_sec": round(elapsed, 2), "escalation": action,
        }, priority=2)

    # ── 内部 ──

    def _put(self, etype: EventType, payload: dict[str, Any], priority: int = 0) -> None:
        if not self.enabled:
            return
        ev = GameEvent(event_type=etype, payload=payload, source="combat_brain", priority=priority)
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            self.dropped += 1   # 账本可以缺页, 决策不能卡

    def _writer(self) -> None:
        while self._running:
            try:
                ev = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._gbus.log_event(ev, force=True)
            except Exception as e:
                log.debug(f"[LEDGER] 写入失败: {e}")

    def close(self) -> None:
        self._running = False
        if self._gbus is not None:
            try:
                time.sleep(0.05)   # 让 writer 把队尾冲掉
                self._gbus.close()
            except Exception:
                pass
