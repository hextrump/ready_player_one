"""
事件类型定义 — 状态总线通信的标准消息格式。

所有感知模块输出的结果，都会包装成 GameEvent 写入状态总线。
决策层通过订阅事件类型来驱动状态机流转。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """游戏事件类型枚举。"""

    # ── 动态实体 (YOLO) ──
    MONSTER_DETECTED = "monster_detected"
    PLAYER_DETECTED = "player_detected"
    NPC_DETECTED = "npc_detected"
    ITEM_DROP_DETECTED = "item_drop_detected"
    PORTAL_DETECTED = "portal_detected"

    # ── 环境 (OpenCV) ──
    PLATFORM_MAPPED = "platform_mapped"
    ROPE_MAPPED = "rope_mapped"
    MINIMAP_UPDATED = "minimap_updated"
    NAV_GRID_UPDATED = "nav_grid_updated"

    # ── 文本 (OCR) ──
    CHAT_MESSAGE = "chat_message"
    HP_UPDATED = "hp_updated"
    MP_UPDATED = "mp_updated"
    EXP_UPDATED = "exp_updated"

    # ── 异常状态 ──
    CAPTCHA_DETECTED = "captcha_detected"
    CAPTCHA_SOLVED = "captcha_solved"
    GM_ALERT = "gm_alert"
    HP_LOW = "hp_low"
    MP_LOW = "mp_low"
    INVENTORY_FULL = "inventory_full"
    PLAYER_DEAD = "player_dead"

    # ── 系统 ──
    STATE_CHANGED = "state_changed"
    ACTION_COMPLETED = "action_completed"
    ERROR = "error"


@dataclass
class BBox:
    """边界框。"""
    x: int       # 左上角 x
    y: int       # 左上角 y
    w: int       # 宽度
    h: int       # 高度

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class Detection:
    """YOLO 检测结果。"""
    class_name: str
    bbox: BBox
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


@dataclass
class Platform:
    """台阶/平台。"""
    x: int
    y: int
    width: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y)


@dataclass
class Rope:
    """绳子。"""
    x: int
    y_top: int
    y_bottom: int

    @property
    def height(self) -> int:
        return self.y_bottom - self.y_top


@dataclass
class PlayerPosition:
    """玩家坐标（从小地图提取）。"""
    x: float     # 归一化坐标 [0, 1]
    y: float
    map_name: str = ""


@dataclass
class GameEvent:
    """
    状态总线事件。所有感知模块输出统一封装为此格式。

    示例:
        GameEvent(
            event_type=EventType.MONSTER_DETECTED,
            payload={"detections": [Detection(...), ...]},
            source="yolo"
        )
    """
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    priority: int = 0       # 越高越优先

    def is_expired(self, ttl_ms: float = 500.0) -> bool:
        """检查事件是否已过期。"""
        age_ms = (time.time() - self.timestamp) * 1000
        return age_ms > ttl_ms

    def __repr__(self) -> str:
        return (
            f"GameEvent({self.event_type.value}, "
            f"source={self.source}, "
            f"priority={self.priority})"
        )
