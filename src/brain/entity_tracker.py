"""
实体跟踪器 — 给感知数据"身份与生命周期" (世界树第一步)。

问题 (无灵魂): Target 只有 cx/cy/w/h/conf, 没有 id, 每帧观测即用即焚。
攻击锁定靠像素距离 hack (100px 内算同一只), 玩家自检靠像素矩形过滤。

本模块:
- Monster 实体: 有稳定 id + 生命周期字段, 替代无身份的 Target 观测。
- MonsterTracker: 每帧把新检测匹配到已有实体 (位置最近邻), 未匹配则新建,
  长时间未观察则销毁。身份来自位置连续性, 不来自分类。

决策层读取持久实体 (跨帧存活), 而不是每帧的新观测 → 世界有了记忆。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Monster:
    """有身份的怪物实体 (字段与 Target 兼容, 决策层无需改动)。"""
    id: int
    name: str
    cx: int
    cy: int
    w: int
    h: int
    conf: float
    dist: float = 0.0      # 与玩家的距离 (感知循环每帧重算)
    seen_frames: int = 1   # 连续被观察帧数
    miss_frames: int = 0   # 连续未观察帧数
    last_seen: float = 0.0 # 最近一次被观察到的时间


@dataclass
class PlayerState:
    """玩家实体: 位置 + 收敛状态 (名牌/v13/衰减 三源收敛到一个状态)。

    替代 combat_brain 里散落的 _cached_player_pos/_player_pending/_player_miss_frames/_player_reliable。
    收敛协议 (与原来完全一致):
    - confirm: 小位移直接提交 / 大位移两帧成立 / v13 兜底命中 → 重置候选与漏检。
    - propose: 大位移候选第一帧挂起, 等下一帧同位置确认 (防锁到其它玩家名牌)。
    - reject:  本帧无可信来源 → 漏检计数 +1。
    - decay:   连续漏检后向画面中心衰减, 避免冻结在陈旧位置。
    reliable 由 find_targets 每帧末尾按本帧结果显式赋值 (名牌命中 或 v13 兜底 = 可信)。
    """

    x: int
    y: int
    reliable: bool = False        # 本帧位置是否来自可信来源 (名牌/v13), 决定脱困跳是否可信
    pending: tuple | None = None  # 大位移候选 (等两帧确认)
    miss_frames: int = 0          # 连续漏检帧 (用于位置衰减)

    def confirm(self, x: int, y: int) -> None:
        """确认位置: 更新坐标, 重置候选与漏检。"""
        self.x, self.y = int(x), int(y)
        self.pending = None
        self.miss_frames = 0

    def propose(self, x: int, y: int) -> None:
        """大位移候选第一帧: 挂起等下一帧同位置确认。"""
        self.pending = (int(x), int(y))

    def reject(self) -> None:
        """本帧无可信来源: 漏检计数 +1。"""
        self.miss_frames += 1

    def decay(self, center: tuple, step: int) -> None:
        """向画面中心衰减一步 (连续漏检时避免冻结在陈旧位置)。"""
        dx, dy = center[0] - self.x, center[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > step:
            s = step / dist
            self.x = int(round(self.x + dx * s))
            self.y = int(round(self.y + dy * s))


class MonsterTracker:
    """维护怪物的身份与生命周期。

    匹配规则: 每个新检测找最近且位置差 <= MATCH_DIST 的未使用实体; 找到则更新,
    找不到则新建。没被匹配到的实体老化, 超过 DESPAWN_AFTER 未观察则销毁。
    """

    MATCH_DIST = 80        # 位置连续性匹配半径 (px)。战斗中的怪基本不动, 80 足够稳定
    DESPAWN_AFTER = 1.2    # 实体多久没被观察到即销毁 (秒); 对应感知 ~7fps 约 8 帧

    def __init__(self):
        self._entities: Dict[int, Monster] = {}
        self._next_id = 1

    @property
    def monsters(self) -> List[Monster]:
        """当前存活实体 (按 id 排序, 顺序稳定)。"""
        return [self._entities[i] for i in sorted(self._entities)]

    def reset(self) -> None:
        """清空所有实体 (换图/重开时调用)。"""
        self._entities.clear()

    def update(self, detections, now: float | None = None) -> List[Monster]:
        """把新检测 (Target 列表) 匹配/新建/老化到实体层, 返回当前存活实体列表。

        Args:
            detections: 本帧检测出的目标 (find_targets 输出, 有 cx/cy/w/h/conf/name)。
            now: 当前时间戳 (便于测试注入)。
        """
        now = time.time() if now is None else now
        used = set()

        # 1. 匹配: 每个检测找最近的未使用实体
        for det in detections:
            best_id, best_d = None, self.MATCH_DIST
            for eid, ent in self._entities.items():
                if eid in used:
                    continue
                d = math.hypot(det.cx - ent.cx, det.cy - ent.cy)
                if d < best_d:
                    best_d, best_id = d, eid

            if best_id is not None:
                # 更新已有实体 (位置/尺寸/置信度刷新, 身份不变)
                ent = self._entities[best_id]
                ent.cx, ent.cy = det.cx, det.cy
                ent.w, ent.h = det.w, det.h
                ent.conf = det.conf
                ent.name = det.name
                ent.seen_frames += 1
                ent.miss_frames = 0
                ent.last_seen = now
                used.add(best_id)
            else:
                # 新建实体
                ent = Monster(
                    id=self._next_id, name=det.name, cx=det.cx, cy=det.cy,
                    w=det.w, h=det.h, conf=det.conf,
                    dist=0.0, seen_frames=1, miss_frames=0, last_seen=now,
                )
                self._entities[ent.id] = ent
                self._next_id += 1
                used.add(ent.id)

        # 2. 老化: 未被匹配的实体 miss_frames++, 超过 DESPAWN_AFTER 未观察则销毁
        for eid, ent in list(self._entities.items()):
            if eid not in used:
                ent.miss_frames += 1
                if now - ent.last_seen > self.DESPAWN_AFTER:
                    del self._entities[eid]

        # 3. 返回存活实体 (顺序稳定)
        return self.monsters
