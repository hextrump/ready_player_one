"""
状态总线测试。
"""

import asyncio
import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state.events import GameEvent, EventType


class TestLocalBus:
    """LocalBus 测试。"""

    def test_import(self):
        from src.state.local_bus import LocalBus
        assert LocalBus is not None

    def test_publish_subscribe(self):
        """验证发布-订阅机制。"""
        from src.state.local_bus import LocalBus

        bus = LocalBus(queue_size=10)
        received = []

        def handler(event: GameEvent):
            received.append(event)

        bus.subscribe(EventType.MONSTER_DETECTED, handler)

        async def run_test():
            event = GameEvent(
                event_type=EventType.MONSTER_DETECTED,
                payload={"test": True},
                source="test",
            )
            await bus.publish(event)

            # 手动分发一个事件
            e = await asyncio.wait_for(bus._queue.get(), timeout=1.0)
            handlers = bus._subscribers.get(e.event_type, [])
            for h in handlers:
                h(e)

        asyncio.run(run_test())
        assert len(received) == 1
        assert received[0].payload["test"] is True

    def test_event_expiry(self):
        """验证过期事件被丢弃。"""
        from src.state.local_bus import LocalBus

        bus = LocalBus(event_ttl_ms=100)

        old_event = GameEvent(
            event_type=EventType.MONSTER_DETECTED,
            timestamp=time.time() - 1.0,  # 1秒前
        )
        assert old_event.is_expired(100)

    def test_queue_overflow(self):
        """验证队列满时丢弃旧事件。"""
        from src.state.local_bus import LocalBus

        bus = LocalBus(queue_size=2)

        async def run_test():
            for i in range(5):
                await bus.publish(GameEvent(
                    event_type=EventType.MONSTER_DETECTED,
                    payload={"index": i},
                ))
            # 队列最大 2，应该只保留最新的
            assert bus.pending_count <= 2

        asyncio.run(run_test())


class TestGlobalBus:
    """GlobalBus 测试。"""

    def test_import(self):
        from src.state.global_bus import GlobalBus
        assert GlobalBus is not None

    def test_state_set_get(self, tmp_path):
        """验证状态键值存取。"""
        from src.state.global_bus import GlobalBus

        db_path = tmp_path / "test.db"
        gbus = GlobalBus(db_path=str(db_path))

        gbus.set_state("current_map", "Henesys")
        assert gbus.get_state("current_map") == "Henesys"
        assert gbus.get_state("nonexistent", "default") == "default"

        gbus.close()

    def test_event_logging(self, tmp_path):
        """验证事件持久化。"""
        from src.state.global_bus import GlobalBus

        db_path = tmp_path / "test.db"
        gbus = GlobalBus(db_path=str(db_path))

        event = GameEvent(
            event_type=EventType.CAPTCHA_DETECTED,
            payload={"test": True},
            source="test",
        )
        gbus.log_event(event)

        history = gbus.query_events(EventType.CAPTCHA_DETECTED)
        assert len(history) == 1
        assert history[0]["payload"]["test"] is True

        gbus.close()

    def test_non_persistent_event(self, tmp_path):
        """验证非关键事件不被持久化。"""
        from src.state.global_bus import GlobalBus

        db_path = tmp_path / "test.db"
        gbus = GlobalBus(db_path=str(db_path))

        event = GameEvent(
            event_type=EventType.MONSTER_DETECTED,  # 不在 PERSISTENT_EVENTS 中
            payload={},
            source="test",
        )
        gbus.log_event(event)  # 应该被跳过

        history = gbus.query_events(EventType.MONSTER_DETECTED)
        assert len(history) == 0

        gbus.close()
