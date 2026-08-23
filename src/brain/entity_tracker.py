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
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Tuple

# 起批门槛: 累计被观察到这么多帧就确认为真怪 (确认后不因漏检撤销)。
# 防误检不再靠"必须连续", 而是靠 MonsterTracker.update(strong_conf=...) 的高分起批
# + DESPAWN_AFTER 生命周期 —— 见 targets 属性里的说明。
MIN_SEEN_FRAMES = 2

# 已确认的怪允许连续漏检多少帧仍参与决策 (用最后已知位置)。
# 感知 ~7fps, 3 帧 ≈ 0.4s: 怪不会因为检测器眨了下眼就消失, 但也不能一直打空气。
COAST_FRAMES = 3

# 玩家中心 → 脚底的像素偏移 (全局唯一真源; patrol_mover / nametag 定位器都对齐这个值)
PLAYER_FOOT_OFFSET = 35


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
    seen_frames: int = 1   # 累计被观察帧数 (达到门槛即 confirmed; 不再一漏检就归零)
    miss_frames: int = 0   # 连续未观察帧数
    last_seen: float = 0.0 # 最近一次被观察到的时间
    last_strong_seen: float = 0.0  # 最近一次被**强检测**观察到的时间 (防杂物给死怪续命)
    confirmed: bool = False  # 已确认是真怪 (一旦确认就保持, 短暂漏检不撤销身份)


class PlayerConfidence(str, Enum):
    """玩家位置的可信等级 (决策层据此决定敢不敢做几何动作)。"""
    CONFIRMED = "confirmed"   # 本帧有可信观察 → 坐标可用于攻击范围/跳跃判定
    STALE = "stale"           # 短暂漏检, 坐标还新鲜 → 可继续走, 但别做精细几何
    LOST = "lost"             # 长期漏检 → 坐标只是猜测, 关闭脱困跳/登台跳


@dataclass
class PlayerState:
    """玩家实体: 唯一身份 + 位置收敛 (名牌观察 → 状态)。

    设计要点 (设计思想: 对象身份不能被上下文碰瓷):
      名牌定位器每帧只产出**观察** (可能是自己, 也可能是隔壁那个人)。是否采信由本实体
      按连续性判定 —— 身份是实体自己的属性, 不是每帧从画面里重新选一个"最像的"。

    收敛协议:
    - LOST 状态: 任何可信观察都直接采信 (重新捕获, 此时没有身份需要保护)。
    - 已有身份: 小位移 (<=COMMIT_PX) 直接提交; 大位移必须**两帧落在同一处**才提交;
                超出 MAX_JUMP_PX 一律拒绝 (那是别人的名牌, 不是我瞬移了)。
    - reject:   本帧无可信观察 → 漏检 +1, 到阈值降级 STALE → LOST。
    - decay:    LOST 后向画面中心缓慢衰减, 避免冻结在陈旧坐标。
    """

    x: int
    y: int
    confidence: PlayerConfidence = PlayerConfidence.LOST
    source: str = "none"          # 本帧采信的观察来源 (ref/body/badge/none)
    pending: tuple | None = None  # 大位移候选 (等两帧确认)
    miss_frames: int = 0          # 连续漏检帧 (用于降级与衰减)
    last_confirm_t: float = 0.0   # 最近一次确认时间
    rejects: int = 0              # 累计拒绝的越界观察 (诊断: 附近有别的玩家)
    far_streak: int = 0           # 连续落在同一个"远处"的观察数 (换图/传送/锁错人的翻案证据)
    unconfirmed: int = 0          # 连续"看得见却没采信"的帧数 (卡死保护)

    # ── 门控参数 (按感知 ~7fps 标定; 玩家正常走动每帧约 35px) ──
    COMMIT_PX = 60.0        # 小位移直接提交 (走动/微移)
    CONTINUITY_PX = 90.0    # 两帧候选一致的容差
    MAX_JUMP_PX = 260.0     # 单帧最大可信位移; 超出 = 锁到别人了
    STALE_FRAMES = 8        # 连续漏检多少帧降级 STALE
    LOST_FRAMES = 30        # 连续漏检多少帧降级 LOST (开始向中心衰减)
    FAR_CONFIRM_FRAMES = 4  # 越界观察连续几帧落在同一处就翻案接受 (换图/传送 ~0.6s @7fps)
    STALL_BREAK_FRAMES = 3  # 连续几帧"看得见却不采信"就强制采信 (防位置冻死, ~0.4s @7fps)
    # 已验证身份的来源: 参考名牌图匹配 = 回答了"这是不是我", 中等位移可直接采信。
    # 其余来源 (badge/body/v13) 只回答"这里有个玩家", 必须两帧一致。
    VERIFIED_SOURCES = frozenset({"ref"})

    @property
    def reliable(self) -> bool:
        """位置是否足以支撑几何决策 (脱困跳/登台跳/攻击范围)。"""
        return self.confidence is PlayerConfidence.CONFIRMED

    @property
    def foot_y(self) -> int:
        """脚底 y (踩在哪个平台上按这个算)。"""
        return self.y + PLAYER_FOOT_OFFSET

    def observe(self, ok: bool, x: int = 0, y: int = 0, source: str = "none",
                now: float | None = None) -> bool:
        """喂入本帧的名牌/身体观察, 返回是否采信 (True=位置已更新)。

        采信力度取决于**观察回答了什么问题**:
        - source="ref": 参考名牌图匹配上了 → 回答的是"这是不是**我**"。中等位移直接采信。
        - 其它来源:     只回答"这里有**一个**玩家" → 中等位移必须两帧一致才采信。
        把两者一视同仁会两头不讨好: 严了会卡死 (实测帧 5: ref 以 0.805 分找到真人在
        (1040,432), 却因为距陈旧坐标 216px 被拒, 位置冻在错的地方直到 LOST 超时);
        松了会锁到隔壁那个人。
        """
        now = time.time() if now is None else now
        if not ok:
            self.reject()
            return False

        d = math.hypot(x - self.x, y - self.y)

        # 身份已丢失 → 无条件重新捕获 (没有身份需要保护, 拒绝只会让 bot 永远瞎着)
        if self.confidence is PlayerConfidence.LOST:
            self.confirm(x, y, source, now)
            return True

        if d <= self.COMMIT_PX:
            self.confirm(x, y, source, now)
            return True

        # 与上一帧候选是否落在同一处 (连续性证据)
        consistent = (self.pending is not None
                      and math.hypot(x - self.pending[0], y - self.pending[1]) <= self.CONTINUITY_PX)

        if d > self.MAX_JUMP_PX:
            # 瞬移太远 → 默认当作别人的名牌, 不动摇已确认的身份。
            # 但"证据持续一致"必须能推翻状态: 换图/传送/上一帧本来就锁错人时, 同一个远处位置
            # 会连续出现。连续 FAR_CONFIRM_FRAMES 帧落在同一处 → 接受。
            self.far_streak = self.far_streak + 1 if consistent else 1
            self.propose(x, y)
            if self.far_streak >= self.FAR_CONFIRM_FRAMES:
                self.confirm(x, y, source, now)
                return True
            self.rejects += 1
            self.reject()
            return False
        self.far_streak = 0

        # 中等位移 (跳跃/下跳/被击退): 已验证身份的来源直接采信, 其余要两帧一致
        if source in self.VERIFIED_SOURCES or consistent:
            self.confirm(x, y, source, now)
            return True

        # 卡死保护: 连续 N 帧"看得见却不采信" = 错的多半是我们自己 (陈旧坐标), 不是观察。
        # 没有这条, 掉帧/卡顿期间的一次中等位移能把位置冻住直到 LOST 超时 (~4s 全瞎)。
        self.propose(x, y)
        self.unconfirmed += 1
        if self.unconfirmed >= self.STALL_BREAK_FRAMES:
            self.confirm(x, y, source, now)
            return True
        self.reject()          # 本帧位置未更新, 仍算一次漏检 (但 pending 保留)
        return False

    def confirm(self, x: int, y: int, source: str = "none", now: float | None = None) -> None:
        """确认位置: 更新坐标与身份状态, 重置候选与漏检。"""
        self.x, self.y = int(x), int(y)
        self.pending = None
        self.miss_frames = 0
        self.far_streak = 0
        self.unconfirmed = 0
        self.source = source
        self.confidence = PlayerConfidence.CONFIRMED
        self.last_confirm_t = time.time() if now is None else now

    def propose(self, x: int, y: int) -> None:
        """大位移候选第一帧: 挂起等下一帧同位置确认。"""
        self.pending = (int(x), int(y))

    def reject(self) -> None:
        """本帧无可采信观察: 漏检 +1, 按阈值降级。"""
        self.miss_frames += 1
        if self.miss_frames >= self.LOST_FRAMES:
            self.confidence = PlayerConfidence.LOST
            self.source = "none"
        elif self.miss_frames >= self.STALE_FRAMES:
            self.confidence = PlayerConfidence.STALE
        elif self.confidence is PlayerConfidence.CONFIRMED:
            self.confidence = PlayerConfidence.STALE   # 本帧没看到 = 已经不是"确认"了

    def decay(self, center: tuple, step: int) -> None:
        """向画面中心衰减一步 (LOST 后避免冻结在陈旧位置)。"""
        dx, dy = center[0] - self.x, center[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > step:
            s = step / dist
            self.x = int(round(self.x + dx * s))
            self.y = int(round(self.y + dy * s))


@dataclass(frozen=True)
class WorldSnapshot:
    """世界状态的**不可变帧快照** — 决策与执行的唯一输入。

    为什么需要它 (设计思想: 上下文不能碰瓷状态):
      原来决策层每读一个字段就加一次锁 (_decide 读怪, _walk_toward 又读玩家,
      _attack 再读一次目标), 每次读到的是**不同时刻**的世界。于是"决策认为怪在射程内"
      和"执行认为怪不在射程内"能同时成立 → 攻击 0 击空转、状态 100ms 反复横跳。
      快照把"一次决策看到的世界"钉成一个值: 同一 seq 内所有判断自洽。
    """

    seq: int                      # 感知帧序号 (单调递增, 可判新旧)
    t: float                      # 快照时间戳
    px: int                       # 玩家中心 x
    py: int                       # 玩家中心 y
    confidence: PlayerConfidence  # 玩家位置可信度
    player_source: str            # 玩家位置来源 (ref/body/badge/none)
    targets: Tuple[Monster, ...]  # 本帧可决策的怪 (已连续确认, 值拷贝)
    platforms: Tuple[tuple, ...]  # (y, x_left, x_right)
    ropes: Tuple[tuple, ...]      # (x, y_top, y_bottom)
    motion: float                 # 帧间运动量 (卡住检测)
    fps: float                    # 感知线程帧率
    world_offset: tuple = (0, 0)  # 相机位姿: world = screen + offset

    # ── 世界坐标 ──
    # 相机跟着玩家走, 所以**屏幕坐标里玩家几乎不动** —— 想表达"走到那边那个平台的尽头"
    # 这种跨屏幕的目标, 必须用世界坐标, 否则目标点会随着镜头一起跑, 永远追不到。
    def to_world(self, sx: float, sy: float) -> tuple:
        ox, oy = self.world_offset
        return (sx + ox, sy + oy)

    def to_screen(self, wx: float, wy: float) -> tuple:
        ox, oy = self.world_offset
        return (wx - ox, wy - oy)

    @property
    def player_world(self) -> tuple:
        return self.to_world(self.px, self.py)

    @property
    def player_reliable(self) -> bool:
        return self.confidence is PlayerConfidence.CONFIRMED

    @property
    def foot_y(self) -> int:
        return self.py + PLAYER_FOOT_OFFSET

    def target_by_id(self, tid: int | None) -> Monster | None:
        """按实体 id 找快照里的怪 (身份查找, 不用像素距离猜)。"""
        if tid is None:
            return None
        for t in self.targets:
            if t.id == tid:
                return t
        return None

    def age_ms(self, now: float | None = None) -> float:
        return ((time.time() if now is None else now) - self.t) * 1000.0


@dataclass
class WorldState:
    """统一世界状态: 玩家实体 + 怪实体 + 地形, 跨帧存活 (世界树的地基)。

    感知线程单写, 决策线程只通过 snapshot() 读 (不再读裸 dict 上下文, 也不再散读)。
    targets/player_x/player_y 是给决策层的投影: ghost 怪不参与, 只暴露本帧观察到的。
    """

    player: PlayerState
    monsters: MonsterTracker
    platforms: list = field(default_factory=list)   # (y, x_left, x_right) 行走面
    ropes: list = field(default_factory=list)       # (x, y_top, y_bottom) 攀爬
    motion: float = 0.0                             # 帧间运动量 (卡住检测)
    fps: float = 0.0                                # 感知线程帧率
    seq: int = 0                                    # 感知帧序号
    world_offset: tuple = (0, 0)                    # 相机位姿 (world = screen + offset)

    @property
    def targets(self) -> List[Monster]:
        """决策可消费的怪: **已确认** 且 漏检不超过 COAST_FRAMES 帧。

        原来要求 `miss_frames == 0`(本帧必须看见)+ 连续确认, 等于把可打率压成检出率的平方:
        实测检出率 70% 的怪, 决策层只有 50% 的帧能打 —— 表现就是"明明有怪却打打停停"。
        怪不会因为检测器眨了下眼就消失, 所以确认过的目标允许短暂 coast (用最后已知位置),
        由 DESPAWN_AFTER 兜底销毁。
        """
        return [m for m in self.monsters.monsters
                if m.confirmed and m.miss_frames <= COAST_FRAMES]

    @property
    def player_x(self) -> int:
        return self.player.x

    @property
    def player_y(self) -> int:
        return self.player.y

    def snapshot(self, now: float | None = None) -> WorldSnapshot:
        """冻结当前世界为一个快照 (怪实体值拷贝, 感知线程后续改动不会影响已发出的决策)。

        调用方必须持有感知锁; 返回的快照之后可以无锁自由使用。
        """
        return WorldSnapshot(
            seq=self.seq,
            t=time.time() if now is None else now,
            px=self.player.x,
            py=self.player.y,
            confidence=self.player.confidence,
            player_source=self.player.source,
            targets=tuple(replace(m) for m in self.targets),
            platforms=tuple(self.platforms),
            ropes=tuple(self.ropes),
            motion=self.motion,
            fps=self.fps,
            world_offset=tuple(self.world_offset),
        )


class MonsterTracker:
    """维护怪物的身份与生命周期。

    匹配规则: 每个新检测找最近且位置差 <= MATCH_DIST 的未使用实体; 找到则更新,
    找不到则新建。没被匹配到的实体老化, 超过 DESPAWN_AFTER 未观察则销毁。
    """

    MATCH_DIST = 80        # 位置连续性匹配半径 (px)。战斗中的怪基本不动, 80 足够稳定
    DESPAWN_AFTER = 1.2    # 实体多久没被观察到即销毁 (秒); 对应感知 ~7fps 约 8 帧
    WEAK_ONLY_MAX_SEC = 1.0  # 只靠弱检测最多续命多久 (超过就正常老化 → 销毁)

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

    def apply_camera_shift(self, dx: float, dy: float) -> None:
        """把所有实体平移到本帧的屏幕坐标系 (相机滚动了多少, 画面里的怪就挪多少)。

        两个作用:
        1. coast (短暂漏检时沿用最后已知位置) 期间位置不会因为镜头移动而错位 ——
           3 帧 coast × 35px/帧 ≈ 105px, 已经超过近战射程 120px 的大半。
        2. 匹配更稳: 否则 MATCH_DIST=80 的预算要先被相机位移吃掉一大半,
           走得快时同一只怪会被判成"新目标" → 身份跳变 → 攻击承诺被打断。
        """
        if not dx and not dy:
            return
        for ent in self._entities.values():
            ent.cx = int(round(ent.cx + dx))
            ent.cy = int(round(ent.cy + dy))

    def update(self, detections, now: float | None = None,
               strong_conf: float = 0.0) -> List[Monster]:
        """把新检测 (Target 列表) 匹配/新建/老化到实体层, 返回当前存活实体列表。

        Args:
            detections: 本帧检测出的目标 (find_targets 输出, 有 cx/cy/w/h/conf/name)。
            now: 当前时间戳 (便于测试注入)。
            strong_conf: **起批门槛**。conf 低于它的检测只能"维持"已有实体, 不能新建。
                雷达那套 track-before-detect: 用高门槛决定"这是不是一个新目标",
                用低门槛决定"这个已知目标还在不在"。这样贴着阈值抖动的怪 (实测 12% 的
                检测落在 0.25~0.50 带) 不会一帧被滤一帧通过 → 决策层不再打打停停;
                而杂物 (树冠/光柱/宠物) 因为拿不到两次高分, 永远起不了批。
        """
        now = time.time() if now is None else now
        used = set()

        # 1. 匹配: 每个检测找最近的未使用实体 (高分优先, 保证强检测先占坑)
        for det in sorted(detections, key=lambda d: -getattr(d, "conf", 0.0)):
            best_id, best_d = None, self.MATCH_DIST
            for eid, ent in self._entities.items():
                if eid in used:
                    continue
                d = math.hypot(det.cx - ent.cx, det.cy - ent.cy)
                if d < best_d:
                    best_d, best_id = d, eid

            if best_id is not None:
                ent = self._entities[best_id]
                # 弱检测只能给"最近还被强检测确认过"的实体续命。否则怪死了以后,
                # 旁边一个低分杂物 (树冠/光柱) 会一直把它的轨迹续下去 → bot 对着空气打。
                if det.conf < strong_conf and (now - ent.last_strong_seen) > self.WEAK_ONLY_MAX_SEC:
                    continue
                if det.conf >= strong_conf:
                    ent.last_strong_seen = now
                # 更新已有实体 (位置/尺寸/置信度刷新, 身份不变)
                ent.cx, ent.cy = det.cx, det.cy
                ent.w, ent.h = det.w, det.h
                ent.conf = det.conf
                ent.name = det.name
                ent.seen_frames += 1
                ent.miss_frames = 0
                ent.last_seen = now
                if ent.seen_frames >= MIN_SEEN_FRAMES:
                    ent.confirmed = True     # 一旦确认就是确认了 (身份不因一帧漏检撤销)
                used.add(best_id)
            elif det.conf >= strong_conf:
                # 新建实体 (只有"强检测"能起批; 弱检测只维持已有目标)
                ent = Monster(
                    id=self._next_id, name=det.name, cx=det.cx, cy=det.cy,
                    w=det.w, h=det.h, conf=det.conf,
                    dist=0.0, seen_frames=1, miss_frames=0, last_seen=now,
                    last_strong_seen=now,
                )
                self._entities[ent.id] = ent
                self._next_id += 1
                used.add(ent.id)

        # 2. 老化: 未被匹配的实体 miss_frames++, 超过 DESPAWN_AFTER 未观察则销毁。
        # 注意**不再把 seen_frames 归零**: 原来一漏检就清零, 于是"稳定闪烁"的怪
        # (检出率 50%~70%) 永远凑不齐连续两帧 → 决策层永远看不到它, 站在猪群里说没怪。
        # 防误检改由 strong_conf 起批门槛 + DESPAWN_AFTER 生命周期负责。
        for eid, ent in list(self._entities.items()):
            if eid not in used:
                ent.miss_frames += 1
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
    BLEND = 0.7           # 融合权重: 70% 旧 + 30% 新 (只用于压检测抖动)
    SNAP_PX = 12          # 残差超过它就直接对齐到本帧检测 (那是位姿漂移, 不是抖动)
    MIN_SEEN_FRAMES = 2   # 连续观察门槛 (滤单帧误检)
    DESPAWN_AFTER = 2.0   # 多久没观察销毁 (秒)
    PAN_THRESH = 6        # 平移检测阈值 (px, 高于抖动)

    def __init__(self):
        self._platforms: Dict[int, PlatformEntity] = {}
        self._ropes: Dict[int, RopeEntity] = {}
        self._next_pid = 1
        self._next_rid = 1
        self._world_offset = (0, 0)    # 相机位姿 (整数, 对外): world = screen + offset
        self._offset_f = (0.0, 0.0)    # 同一位姿的 float 累加器 (防每帧取整丢小数 → 线性漂移)

    @property
    def world_offset(self) -> tuple:
        """当前相机位姿 (world = screen + offset)。"""
        return self._world_offset

    def apply_camera_shift(self, dx: float, dy: float) -> None:
        """把本帧测到的相机位移并入位姿 (每帧调用, 不依赖地形模型)。

        为什么必须每帧做 (2026-08-19 实测):
          地形模型每 TERRAIN_EVERY=3 帧才跑一次, 而位姿原来只在跑地形的那一帧更新。
          走路时镜头每帧滚动约 35px, 于是中间两帧暴露给决策的平台位置最多偏 80px ——
          而"到平台边缘"的容差只有 25px、抓梯子 30px。结果就是走着走着平台在决策眼里
          错位: 该转身的地方不转、以为脚下有台阶其实是空的。
          镜头位移可以用相位相关在缩略图上每帧算出来 (~1ms), 不需要重新跑模型。

        Args:
            dx, dy: 画面内容相对上一帧的位移 (屏幕像素)。内容左移 dx<0 = 镜头右移。
        """
        if dx == 0 and dy == 0:
            return
        # 用 float 累积: 每帧 round() 会把小数丢掉, 而丢掉的部分是**同号**的
        # (走同一方向时每帧都少算 0.x px) → 线性累积成漂移。实测 400 帧漂了 100px。
        fx, fy = self._offset_f
        self._offset_f = (fx - dx, fy - dy)
        self._world_offset = (round(self._offset_f[0]), round(self._offset_f[1]))

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
        self._offset_f = (0.0, 0.0)

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
            self._offset_f = (float(ox), float(oy))   # 绝对纠正后, float 累加器同步归位
            dets_w = [(y + oy, xl + ox, xr + ox) for (y, xl, xr) in platforms]
        rope_w = [(x + ox, yt + oy, yb + oy) for (x, yt, yb) in ropes]

        # ── 平台融合 (世界坐标) ──
        used_p = set()
        for (y, xl, xr) in dets_w:
            pid = self._match_platform(y, xl, xr, used_p)
            if pid is not None:
                p = self._platforms[pid]
                # 平滑(抖动) vs 对齐(漂移) 要分开处理:
                #   BLEND 平滑是为了压检测框几像素的抖动, 每次只纠正 30%。
                #   但位姿漂移是**几十像素**且持续同向, 30%/次 永远追不上 ——
                #   于是老实体带着 30px 的历史漂移、新实体带着 80px, 同屏画出来就"到处乱飘"。
                # 残差超过 SNAP_PX 就直接对齐到本帧检测: 检测是精确的, 陈旧的是我们。
                resid = max(abs(((xl + xr) / 2) - ((p.x_left + p.x_right) / 2)), abs(p.y - y))
                b = 0.0 if resid > self.SNAP_PX else self.BLEND
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
                b = 0.0 if abs(r.x - x) > self.SNAP_PX else self.BLEND
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
