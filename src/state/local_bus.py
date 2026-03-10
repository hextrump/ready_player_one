"""
高频本地总线 — Agent 的"脊髓反射"层。

基于 asyncio.Queue 的发布-订阅模式，纯内存，亚毫秒级延迟。
用于在感知模块和执行模块之间高速传递实时坐标数据。

特性：
- 事件自动过期淘汰（只保留最新数据）
- 多订阅者支持
- 按事件类型过滤
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Callable, Awaitable

from src.state.events import GameEvent, EventType
from src.utils.logger import get_logger

log = get_logger("local_bus")

# 同步回调或异步回调
EventHandler = Callable[[GameEvent], None] | Callable[[GameEvent], Awaitable[None]]


class LocalBus:
    """
    高频本地事件总线。

    用法:
        bus = LocalBus(queue_size=100, event_ttl_ms=500)

        # 订阅
        bus.subscribe(EventType.MONSTER_DETECTED, my_handler)

        # 发布
        await bus.publish(event)

        # 启动分发循环
        await bus.run()
    """

    def __init__(self, queue_size: int = 100, event_ttl_ms: float = 500.0):
        self._queue: asyncio.Queue[GameEvent] = asyncio.Queue(maxsize=queue_size)
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_subscribers: list[EventHandler] = []
        self._event_ttl_ms = event_ttl_ms
        self._running = False
        self._stats = {"published": 0, "dispatched": 0, "expired": 0}

    def subscribe(
        self,
        event_type: EventType | None,
        handler: EventHandler,
    ) -> None:
        """
        订阅事件。

        Args:
            event_type: 事件类型。None 表示订阅所有事件。
            handler: 回调函数（同步或异步）。
        """
        if event_type is None:
            self._global_subscribers.append(handler)
            log.debug(f"全局订阅: {handler.__name__}")
        else:
            self._subscribers[event_type].append(handler)
            log.debug(f"订阅 {event_type.value}: {handler.__name__}")

    def unsubscribe(
        self,
        event_type: EventType | None,
        handler: EventHandler,
    ) -> None:
        """取消订阅。"""
        if event_type is None:
            self._global_subscribers.remove(handler)
        else:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: GameEvent) -> None:
        """
        发布事件到总线。如果队列已满，丢弃最旧的事件。
        """
        if self._queue.full():
            try:
                self._queue.get_nowait()  # 丢弃最旧
                self._stats["expired"] += 1
            except asyncio.QueueEmpty:
                pass

        await self._queue.put(event)
        self._stats["published"] += 1

    def publish_sync(self, event: GameEvent) -> None:
        """同步版本的 publish，用于非 async 上下文。"""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._stats["expired"] += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(event)
            self._stats["published"] += 1
        except asyncio.QueueFull:
            self._stats["expired"] += 1

    async def run(self) -> None:
        """启动事件分发循环。阻塞直到 stop() 被调用。"""
        self._running = True
        log.info("LocalBus 事件分发循环已启动")

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            # 过期检查
            if event.is_expired(self._event_ttl_ms):
                self._stats["expired"] += 1
                continue

            # 分发给类型订阅者
            handlers = self._subscribers.get(event.event_type, [])
            for handler in handlers:
                await self._invoke(handler, event)

            # 分发给全局订阅者
            for handler in self._global_subscribers:
                await self._invoke(handler, event)

            self._stats["dispatched"] += 1

        log.info(f"LocalBus 已停止 | 统计={self._stats}")

    async def _invoke(self, handler: EventHandler, event: GameEvent) -> None:
        """安全调用 handler（兼容同步和异步）。"""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            log.error(f"Handler 异常: {handler.__name__} → {e}")

    def stop(self) -> None:
        """停止分发循环。"""
        self._running = False

    def get_latest(self, event_type: EventType) -> GameEvent | None:
        """
        获取队列中指定类型的最新事件（不取出队列）。
        用于轮询式访问，而非回调式。
        """
        latest = None
        items = list(self._queue._queue)  # type: ignore
        for item in reversed(items):
            if item.event_type == event_type and not item.is_expired(self._event_ttl_ms):
                latest = item
                break
        return latest

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
