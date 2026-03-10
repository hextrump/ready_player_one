"""
全局持久化状态总线 — Agent 的"大脑皮层"。

使用 SQLite 存储关键状态变更和事件日志。
负责：
- 持久化重要事件（升级、装备交易、异常等）
- 支持状态回溯分析
- 预留去中心化存储接口
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.state.events import GameEvent, EventType
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("global_bus")

# 需要持久化的高优先级事件类型
PERSISTENT_EVENTS = {
    EventType.CAPTCHA_DETECTED,
    EventType.CAPTCHA_SOLVED,
    EventType.GM_ALERT,
    EventType.PLAYER_DEAD,
    EventType.STATE_CHANGED,
    EventType.ERROR,
}


class GlobalBus:
    """
    全局持久化事件总线。

    用法:
        gbus = GlobalBus(db_path="data/state.db")
        gbus.log_event(event)
        history = gbus.query_events(EventType.CAPTCHA_DETECTED, limit=10)
        gbus.set_state("current_map", "Henesys Hunting Ground")
        gbus.get_state("current_map")
    """

    def __init__(self, db_path: str | Path = "data/state.db"):
        db = Path(db_path)
        if not db.is_absolute():
            db = PROJECT_ROOT / db
        db.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = str(db)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        log.info(f"GlobalBus 已连接数据库: {self._db_path}")

    def _init_tables(self) -> None:
        """创建表结构。"""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL NOT NULL,
                event_type  TEXT NOT NULL,
                source      TEXT NOT NULL,
                payload     TEXT NOT NULL,
                priority    INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_time
                ON events(timestamp);
        """)
        self._conn.commit()

    def log_event(self, event: GameEvent, force: bool = False) -> None:
        """
        记录事件到数据库。

        默认只记录高优先级事件（PERSISTENT_EVENTS 列表中的）。
        设置 force=True 可强制记录任何事件。
        """
        if not force and event.event_type not in PERSISTENT_EVENTS:
            return

        self._conn.execute(
            """INSERT INTO events (timestamp, event_type, source, payload, priority)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event.timestamp,
                event.event_type.value,
                event.source,
                json.dumps(event.payload, ensure_ascii=False, default=str),
                event.priority,
            ),
        )
        self._conn.commit()
        log.debug(f"事件已持久化: {event}")

    def query_events(
        self,
        event_type: EventType | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        查询历史事件。

        Args:
            event_type: 事件类型过滤。None 表示所有。
            limit: 最大返回数量。
            since: 起始时间戳。None 表示不限。

        Returns:
            事件字典列表（按时间倒序）。
        """
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "source": row["source"],
                "payload": json.loads(row["payload"]),
                "priority": row["priority"],
            }
            for row in rows
        ]

    def set_state(self, key: str, value: Any) -> None:
        """设置/更新全局状态键值。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO state (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value, ensure_ascii=False, default=str), time.time()),
        )
        self._conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取全局状态值。"""
        row = self._conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def get_all_states(self) -> dict[str, Any]:
        """获取所有全局状态。"""
        rows = self._conn.execute("SELECT key, value FROM state").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def get_event_count(self, event_type: EventType | None = None) -> int:
        """统计事件数量。"""
        if event_type:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE event_type = ?",
                (event_type.value,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        return row["cnt"]

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()
        log.info("GlobalBus 数据库连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
