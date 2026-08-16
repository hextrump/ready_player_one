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
from dataclasses import dataclass, field
from typing import Dict, List

# 决策层认怪的最低连续观察帧数: 真实怪连续存在 → 第 2 帧起可打;
# 偶尔误检 (宠物/玩家/掉落闪一帧) 断档归零 → 永远达不到, 被滤掉。
MIN_SEEN_FRAMES = 2


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
    seen_frames: int = 1   # 连续被观察帧数 (断档归零; 防偶尔误检如宠物/玩家/掉落)
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


@dataclass
class WorldState:
    """统一世界状态: 玩家实体 + 怪实体 + 地形, 跨帧存活 (世界树的地基)。

    感知线程每帧更新, 决策线程只读它 (不再读裸 dict 上下文)。
    targets/player_x/player_y 是给决策层的投影: ghost 怪不参与, 只暴露本帧观察到的。
    """

    player: PlayerState
    monsters: MonsterTracker
    platforms: list = field(default_factory=list)   # (y, x_left, x_right) 行走面
    ropes: list = field(default_factory=list)       # (x, y_top, y_bottom) 攀爬
    motion: float = 0.0                             # 帧间运动量 (卡住检测)
    fps: float = 0.0                                # 感知线程帧率

    @property
    def targets(self) -> List[Monster]:
        """本帧观察到的怪且连续确认 (决策消费; ghost/未连续确认不参与决策, 滤掉偶尔误检)。"""
        return [m for m in self.monsters.monsters
                if m.miss_frames == 0 and m.seen_frames >= MIN_SEEN_FRAMES]

    @property
    def player_x(self) -> int:
        return self.player.x

    @property
    def player_y(self) -> int:
        return self.player.y


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

        # 2. 老化: 未被匹配的实体 miss_frames++, 连续观察计数归零, 超过 DESPAWN_AFTER 未观察则销毁
        for eid, ent in list(self._entities.items()):
            if eid not in used:
                ent.miss_frames += 1
                ent.seen_frames = 0  # 断档 → 连续观察归零 (偶尔误检永远达不到确认门槛)
                if now - ent.last_seen > self.DESPAWN_AFTER:
                    del self._entities[eid]

        # 3. 返回存活实体 (顺序稳定)
        return self.monsters


@dataclass
class PlatformEntity:
    """有身份的持久平台实体 (跨帧融合)。"""
    id: int
    y: int
    x_left: int
    x_right: int
    seen_frames: int = 1
    miss_frames: int = 0
    last_seen: float = 0.0


@dataclass
class RopeEntity:
    """有身份的持久梯子实体 (跨帧融合)。"""
    id: int
    x: int
    y_top: int
    y_bottom: int
    seen_frames: int = 1
    miss_frames: int = 0
    last_seen: float = 0.0


class TerrainTracker:
    """地形持久化: 平台/梯子在【世界坐标】跨帧融合 (身份 + 融合 + 去抖 + 相机位姿)。

    设计思想: 地形不应随玩家移动 — 内部存世界坐标, 检测到相机平移就修正位姿,
    暴露时转回屏幕坐标。平台世界位置固定, 不随镜头漂移。
    - 融合: 匹配到的实体加权平均平滑 (世界坐标 y/x)
    - 确认: 连续观察 >= MIN_SEEN_FRAMES 才暴露 (滤单帧误检)
    - 生命周期: 长时间没观察到销毁
    - 相机位姿: world = screen + offset; 用匹配对 delta 中位数估算平移
    """

    MATCH_Y_TOL = 30      # 平台 y 容差
    MATCH_X_GAP = 25      # x 范围相邻/重叠判同一块
    PAN_MATCH_GAP = 120   # 平移检测用宽松 x 容差 (相机平移大时仍能匹配上)
    BLEND = 0.7           # 融合权重: 70% 旧 + 30% 新
    MIN_SEEN_FRAMES = 2   # 连续观察门槛 (滤单帧误检)
    DESPAWN_AFTER = 2.0   # 多久没观察销毁 (秒)
    PAN_THRESH = 6        # 平移检测阈值 (px, 高于抖动)

    def __init__(self):
        self._platforms: Dict[int, PlatformEntity] = {}
        self._ropes: Dict[int, RopeEntity] = {}
        self._next_pid = 1
        self._next_rid = 1
        self._world_offset = (0, 0)  # 相机位姿: world = screen + offset

    @property
    def world_offset(self) -> tuple:
        """当前相机位姿 (world = screen + offset)。"""
        return self._world_offset

    @property
    def platforms(self) -> List[tuple]:
        """当前平台屏幕坐标 [(y, x_left, x_right)] (world - offset, 供决策/viz)。"""
        ox, oy = self._world_offset
        return [(p.y - oy, p.x_left - ox, p.x_right - ox)
                for p in sorted(self._platforms.values(), key=lambda p: p.id)
                if p.seen_frames >= self.MIN_SEEN_FRAMES]

    @property
    def ropes(self) -> List[tuple]:
        """当前梯子屏幕坐标 [(x, y_top, y_bottom)]。"""
        ox, oy = self._world_offset
        return [(r.x - ox, r.y_top - oy, r.y_bottom - oy)
                for r in sorted(self._ropes.values(), key=lambda r: r.id)
                if r.seen_frames >= self.MIN_SEEN_FRAMES]

    def reset(self) -> None:
        self._platforms.clear()
        self._ropes.clear()
        self._world_offset = (0, 0)

    def update(self, platforms: list, ropes: list, now: float | None = None) -> None:
        """融合屏幕观察 (内部转世界坐标, 自动检测相机平移修正位姿)。"""
        now = time.time() if now is None else now
        ox, oy = self._world_offset

        # 屏幕 → 世界 (当前位姿)
        dets_w = [(y + oy, xl + ox, xr + ox) for (y, xl, xr) in platforms]

        # 相机平移检测: 匹配对 delta 中位数 = offset 过期量 → 修正位姿后重新转换
        pan = self._estimate_pan(dets_w)
        if pan is not None:
            ox -= pan[0]
            oy -= pan[1]
            self._world_offset = (round(ox), round(oy))
            dets_w = [(y + oy, xl + ox, xr + ox) for (y, xl, xr) in platforms]
        rope_w = [(x + ox, yt + oy, yb + oy) for (x, yt, yb) in ropes]

        # ── 平台融合 (世界坐标) ──
        used_p = set()
        for (y, xl, xr) in dets_w:
            pid = self._match_platform(y, xl, xr, used_p)
            if pid is not None:
                p = self._platforms[pid]
                b = self.BLEND
                p.y = round(p.y * b + y * (1 - b))
                p.x_left = round(p.x_left * b + xl * (1 - b))
                p.x_right = round(p.x_right * b + xr * (1 - b))
                p.seen_frames += 1
                p.miss_frames = 0
                p.last_seen = now
                used_p.add(pid)
            else:
                pid = self._next_pid
                self._next_pid += 1
                self._platforms[pid] = PlatformEntity(id=pid, y=y, x_left=xl, x_right=xr, last_seen=now)
                used_p.add(pid)
        for pid, p in list(self._platforms.items()):
            if pid not in used_p:
                p.miss_frames += 1
                # 不重置 seen_frames: 地形错帧(每 TERRAIN_EVERY 帧跑一次)的间隔帧不是"平台消失"
                if now - p.last_seen > self.DESPAWN_AFTER:
                    del self._platforms[pid]

        # ── 梯子融合 (世界坐标) ──
        used_r = set()
        for (x, yt, yb) in rope_w:
            rid = self._match_rope(x, used_r)
            if rid is not None:
                r = self._ropes[rid]
                b = self.BLEND
                r.x = round(r.x * b + x * (1 - b))
                r.y_top = round(r.y_top * b + yt * (1 - b))
                r.y_bottom = round(r.y_bottom * b + yb * (1 - b))
                r.seen_frames += 1
                r.miss_frames = 0
                r.last_seen = now
                used_r.add(rid)
            else:
                rid = self._next_rid
                self._next_rid += 1
                self._ropes[rid] = RopeEntity(id=rid, x=x, y_top=yt, y_bottom=yb, last_seen=now)
                used_r.add(rid)
        for rid, r in list(self._ropes.items()):
            if rid not in used_r:
                r.miss_frames += 1
                # 同平台: 不重置 seen_frames (错帧间隔不算平台消失)
                if now - r.last_seen > self.DESPAWN_AFTER:
                    del self._ropes[rid]

    def _estimate_pan(self, dets_w):
        """用匹配对 delta (检测世界 - 实体世界) 的中位数估算相机平移 (offset 过期量)。"""
        deltas = []
        used = set()
        for (y, xl, xr) in dets_w:
            best = None
            for pid, p in self._platforms.items():
                if pid in used:
                    continue
                if abs(p.y - y) > self.MATCH_Y_TOL * 2:   # 平移容忍更大的 y 差
                    continue
                if p.x_right < xl - self.PAN_MATCH_GAP or p.x_left > xr + self.PAN_MATCH_GAP:
                    continue
                best = pid
                break
            if best is not None:
                p = self._platforms[best]
                dcx = ((xl + xr) / 2) - ((p.x_left + p.x_right) / 2)
                dcy = y - p.y
                deltas.append((dcx, dcy))
                used.add(best)
        if len(deltas) < 2:
            return None
        # 中位数 (x_delta, y_delta)
        dxs = sorted(d[0] for d in deltas)
        dys = sorted(d[1] for d in deltas)
        mx = dxs[len(dxs) // 2]
        my = dys[len(dys) // 2]
        if abs(mx) < self.PAN_THRESH and abs(my) < self.PAN_THRESH:
            return None
        return (mx, my)

    def _match_platform(self, y, xl, xr, used_p):
        """找与检测 (y, xl, xr) 同一条平台的实体 (y 相近 + x 范围相邻/重叠)。"""
        for pid, p in self._platforms.items():
            if pid in used_p:
                continue
            if abs(p.y - y) > self.MATCH_Y_TOL:
                continue
            if p.x_right < xl - self.MATCH_X_GAP or p.x_left > xr + self.MATCH_X_GAP:
                continue
            return pid
        return None

    def _match_rope(self, x, used_r):
        """找与检测 x 同一条梯子的实体。"""
        for rid, r in self._ropes.items():
            if rid in used_r:
                continue
            if abs(r.x - x) > self.MATCH_X_GAP:
                continue
            return rid
        return None
