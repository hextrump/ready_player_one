"""
状态总线 / 身份 / 攻击范围 的回归测试。

每个用例对应一个实测过的线上故障 (见 logs/agent.log 2026-08-19 21:08 段):
- 决策说"打"、执行说"够不着" → [ATTACK] Monster × 0 击 刷屏
- 状态 100ms 反复横跳 attacking ↔ approaching
- 名牌每帧重新选"最像的" → 站着不动玩家位置乱跳 300px
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brain.entity_tracker import (Monster, MonsterTracker, PlayerConfidence,
                                      PlayerState, WorldState)


# ────────────────────────── 玩家身份收敛 ──────────────────────────

class TestPlayerIdentity:
    """PlayerState 的身份连续性: 位置是实体的属性, 不是每帧从画面重选的结果。"""

    def _player(self):
        p = PlayerState(x=800, y=520)
        p.confirm(800, 520, "ref")
        return p

    def test_small_move_commits_immediately(self):
        p = self._player()
        assert p.observe(True, 830, 520, "ref") is True
        assert (p.x, p.y) == (830, 520)
        assert p.confidence is PlayerConfidence.CONFIRMED

    def test_teleport_is_rejected(self):
        """500px 外的名牌是别人的 —— 已确认的身份不能被它推翻。"""
        p = self._player()
        assert p.observe(True, 1400, 300, "ref") is False
        assert (p.x, p.y) == (800, 520)
        assert p.rejects == 1

    def test_medium_move_needs_two_frames_for_unverified_source(self):
        """badge/body 只能回答"这里有个玩家" → 中等位移必须两帧同位置才认。"""
        p = self._player()
        assert p.observe(True, 950, 560, "badge") is False   # 第 1 帧: 挂起
        assert (p.x, p.y) == (800, 520)
        assert p.observe(True, 955, 562, "badge") is True    # 第 2 帧: 确认
        assert (p.x, p.y) == (955, 562)

    def test_verified_source_commits_medium_move_immediately(self):
        """ref = 参考名牌图匹配上了, 回答的是"这是不是我" → 中等位移直接采信。
        回归 (实测帧 5): ref 以 0.805 分找到真人在 216px 外, 却被两帧规则拒掉,
        位置冻在错的地方直到 LOST 超时 —— 那几秒 bot 是瞎的。"""
        p = self._player()
        assert p.observe(True, 1016, 520, "ref") is True
        assert (p.x, p.y) == (1016, 520)

    def test_stall_breaker_unfreezes_position(self):
        """连续看得见却不采信 = 错的多半是我们自己的陈旧坐标, 不是观察。

        取三个互相都超出连续性容差的中等位移观察 (所以两帧规则永远凑不齐),
        第 STALL_BREAK_FRAMES 帧必须强制采信 —— 否则位置会一直冻到 LOST 超时。
        """
        p = self._player()
        assert PlayerState.STALL_BREAK_FRAMES == 3
        assert p.observe(True, 950, 560, "badge") is False    # d=150, 挂起
        assert p.observe(True, 620, 560, "badge") is False    # 与上个候选差 330px, 不一致
        assert (p.x, p.y) == (800, 520)
        assert p.observe(True, 1000, 560, "badge") is True    # 第 3 帧: 强制采信
        assert (p.x, p.y) == (1000, 560)
        assert p.confidence is PlayerConfidence.CONFIRMED

    def test_ref_hit_does_not_bypass_continuity(self):
        """回归: 原来 last_ref_confident 命中就无条件 commit → 一帧瞬移到别人身上。"""
        p = self._player()
        for _ in range(3):
            p.observe(True, 1500, 200, "ref")
        assert (p.x, p.y) == (800, 520), "参考图命中也必须过连续性门控"

    def test_sustained_far_evidence_overturns_identity(self):
        """换图/传送/上一帧本来就锁错人: 同一个远处位置连续出现 → 证据翻案。
        否则一旦锁错就只能干等 LOST 超时 (~4s), 这几秒里 bot 是瞎的。"""
        p = self._player()
        for i in range(PlayerState.FAR_CONFIRM_FRAMES - 1):
            assert p.observe(True, 1400 + i, 300, "ref") is False
        assert p.observe(True, 1401, 300, "ref") is True
        assert p.x >= 1400

    def test_far_evidence_must_be_consistent(self):
        """乱跳的远处观察 (不同位置) 不该翻案 —— 那才是"别人的名牌"。"""
        p = self._player()
        for x in (1400, 200, 1500, 100, 1450, 150):
            assert p.observe(True, x, 300, "ref") is False
        assert (p.x, p.y) == (800, 520)

    def test_lost_identity_allows_recapture(self):
        """身份已丢失时不该继续护着旧坐标, 否则 bot 永远瞎着。"""
        p = self._player()
        for _ in range(PlayerState.LOST_FRAMES):
            p.observe(False)
        assert p.confidence is PlayerConfidence.LOST
        assert p.observe(True, 1400, 300, "ref") is True
        assert (p.x, p.y) == (1400, 300)
        assert p.confidence is PlayerConfidence.CONFIRMED

    def test_confidence_degrades_then_reports_unreliable(self):
        p = self._player()
        assert p.reliable is True
        p.observe(False)
        assert p.confidence is PlayerConfidence.STALE
        assert p.reliable is False
        for _ in range(PlayerState.LOST_FRAMES):
            p.observe(False)
        assert p.confidence is PlayerConfidence.LOST


# ────────────────────────── 世界快照 ──────────────────────────

class TestWorldSnapshot:
    """快照是值: 感知线程后续改动不得影响已经发出去的决策。"""

    def _world(self):
        w = WorldState(player=PlayerState(x=800, y=520), monsters=MonsterTracker())
        w.player.confirm(800, 520, "ref")
        det = Monster(id=0, name="Monster", cx=900, cy=520, w=40, h=40, conf=0.5)
        w.monsters.update([det])
        w.monsters.update([det])   # 连续两帧才达到 MIN_SEEN_FRAMES
        return w

    def test_snapshot_is_isolated_from_later_mutation(self):
        w = self._world()
        snap = w.snapshot()
        assert len(snap.targets) == 1
        before = snap.targets[0].cx
        # 感知线程继续跑, 怪走了
        moved = Monster(id=0, name="Monster", cx=940, cy=520, w=40, h=40, conf=0.5)
        w.monsters.update([moved])
        assert snap.targets[0].cx == before, "快照里的怪不能被后续帧改写"

    def test_target_by_id_finds_by_identity(self):
        w = self._world()
        snap = w.snapshot()
        tid = snap.targets[0].id
        assert snap.target_by_id(tid) is not None
        assert snap.target_by_id(99999) is None
        assert snap.target_by_id(None) is None

    def test_snapshot_carries_player_confidence(self):
        w = self._world()
        assert w.snapshot().player_reliable is True
        w.player.observe(False)
        assert w.snapshot().player_reliable is False


# ────────────────────────── 攻击范围 / 状态机 ──────────────────────────

def _brain(mode="melee"):
    """不跑 __init__ (会加载 YOLO), 只装配决策需要的字段。"""
    from src.brain.combat_brain import BrainState, BrainStateCtx, CombatBrain
    b = CombatBrain.__new__(CombatBrain)
    b.primary_key = "x" if mode == "melee" else "b"
    b.attack_range_x_melee = 120
    b.attack_range_x_bullet = 0 if mode == "melee" else 350
    b.engage_range_x = max(b.attack_range_x_melee, b.attack_range_x_bullet)
    b.attack_range_y = 30
    b.attack_range_buffer = 80
    b.jump_attack_range_y_up = 120
    b.flat_mode = False
    b.state = BrainState.STANDBY
    b.state_ctx = BrainStateCtx(state=BrainState.STANDBY, entered_at=time.time())
    b._snap = None
    b._last_action = None
    b.active_hunting = False
    b.ledger = _NullLedger()
    return b


class _NullLedger:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _mon(cx, cy=520, mid=1):
    return Monster(id=mid, name="Monster", cx=cx, cy=cy, w=40, h=40, conf=0.5, dist=abs(cx - 800))


class TestAttackRange:
    """回归: 决策与执行必须用同一个射程判定, 否则 0 击空转。"""

    def test_engage_vs_hold(self):
        b = _brain()
        near, edge, far = _mon(900), _mon(990), _mon(975)
        assert b.is_in_attack_range(near, 800, 520) is True          # dx=100 <= 120
        assert b.is_in_attack_range(edge, 800, 520) is False         # dx=190 > 120
        assert b.is_in_attack_range(edge, 800, 520, hold=True) is True   # 190 <= 200
        assert b.is_in_attack_range(_mon(1050), 800, 520, hold=True) is False

    def test_warrior_range_not_silently_zero(self):
        """战士的 bullet 射程是 0; 忘了传 key 的旧调用对战士恒为 False → 静默失效。"""
        b = _brain("melee")
        assert b.engage_range_x == 120
        assert b.is_in_attack_range(_mon(880), 800, 520) is True

    def test_select_target_prefers_reachable_not_buffered(self):
        """选目标只能用 engage: 用 buffer 选出来的怪执行层打不到, 就会 0 击空转。"""
        b = _brain()
        targets = [_mon(990, mid=1), _mon(880, mid=2)]
        best = b.select_target(targets, 800, 520)
        assert best.id == 2
        assert b.is_in_attack_range(best, 800, 520) is True

    def test_no_zero_hit_loop(self):
        """核心回归: 凡是 select_target + engage 判定通过的目标, _attack 的起手守卫必过。"""
        b = _brain()
        for dx in range(0, 400, 10):
            t = _mon(800 + dx)
            if b.is_in_attack_range(t, 800, 520):           # 决策放行
                assert b.is_in_attack_range(t, 800, 520, hold=True), \
                    f"dx={dx}: 决策说打得到, 执行说够不着 → 0 击空转"

    def test_jump_attack_range_for_overhead_monster(self):
        b = _brain()
        overhead = _mon(830, cy=420)     # dy=100, 在头顶上方
        assert b.is_in_attack_range(overhead, 800, 520) is True
        b.flat_mode = True
        assert b.is_in_attack_range(overhead, 800, 520) is False


class TestStateMachine:
    """状态机不能被每帧上下文推翻。"""

    def test_min_dwell_blocks_downgrade(self):
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.transition_to(BrainState.SCANNING)
        assert b.transition_to(BrainState.ATTACKING) is True
        # 刚进 ATTACKING 就要降级 → 被最小驻留挡下
        assert b.transition_to(BrainState.APPROACHING) is False
        assert b.state is BrainState.ATTACKING
        assert b.state_ctx.blocked_count == 1

    def test_upgrade_is_never_blocked(self):
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.transition_to(BrainState.PATROLLING)
        assert b.transition_to(BrainState.ATTACKING) is True   # 有怪能打, 立刻响应

    def test_standby_is_global_override(self):
        """按 F 停手: 从 ATTACKING 直接进 STANDBY, 不该报"非法转换"再绕 SCANNING。"""
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.transition_to(BrainState.SCANNING)
        b.transition_to(BrainState.ATTACKING)
        assert b.transition_to(BrainState.STANDBY) is True
        assert b.state is BrainState.STANDBY

    def test_same_state_does_not_reset_dwell(self):
        """回归: 同状态重复转换若刷新 entered_at, 超时 watchdog 永远不会触发。"""
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.transition_to(BrainState.PATROLLING)
        entered = b.state_ctx.entered_at
        time.sleep(0.02)
        assert b.transition_to(BrainState.PATROLLING) is False
        assert b.state_ctx.entered_at == entered


class TestDecideCommitment:
    """决策承诺: 已锁定的怪只要还在 hold 射程内就继续打, 不每帧重选。"""

    def _snap(self, targets, px=800, py=520):
        """MonsterTracker 自己发身份 id (每个 WorldState 从 1 开始), 所以这里不假设 id 值,
        只保证同一个 fixture 里"第 n 只怪"跨快照拿到同一个 id。"""
        w = WorldState(player=PlayerState(x=px, y=py), monsters=MonsterTracker())
        w.player.confirm(px, py, "ref")
        for _ in range(2):
            w.monsters.update(targets)
        return w.snapshot()

    def test_locked_target_survives_range_jitter(self):
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.active_hunting = True
        b._last_action = None
        b.mover = _NullMover()
        b.transition_to(BrainState.SCANNING)

        snap = self._snap([_mon(880)])
        act, tgt = b._decide(snap)
        assert act == "attack"
        locked_id = tgt.id

        # 怪走到射程边缘外 (dx=190, engage=120 但 hold=200) → 仍然继续打, 不切 approach
        snap2 = self._snap([_mon(990)])
        act2, tgt2 = b._decide(snap2)
        assert (act2, tgt2.id) == ("attack", locked_id)

    def test_out_of_hold_range_falls_back_to_approach(self):
        from src.brain.combat_brain import BrainState
        b = _brain()
        b.active_hunting = True
        b.mover = _NullMover()
        b.transition_to(BrainState.SCANNING)
        b._decide(self._snap([_mon(880)]))
        b.state_ctx.entered_at -= 10.0        # 越过最小驻留
        act, _ = b._decide(self._snap([_mon(1200)]))
        assert act == "approach"


class _NullMover:
    """决策层测试用: 一切可及, 路程为 0。"""
    def is_reachable(self, *a, **k):
        return True

    def plan_move(self, *a, **k):
        from src.brain.patrol_mover import MoveKind, MovePlan
        return MovePlan(MoveKind.WALK, None, 0.0, "测试: 同层", travel_px=0.0)

    def flip(self):
        pass


# ────────────────────────── 地形位姿 / 名牌 UI 误锁 ──────────────────────────

class TestCameraPose:
    """走路时镜头滚动, 地形位姿必须每帧跟上 —— 否则决策看到的平台是错位的。"""

    def _tracker_with_platform(self):
        from src.brain.entity_tracker import TerrainTracker
        tt = TerrainTracker()
        for _ in range(TerrainTracker.MIN_SEEN_FRAMES):
            tt.update([(400, 300, 800)], [])
        assert tt.platforms, "平台应已通过连续观察确认"
        return tt

    def test_camera_shift_moves_exposed_platforms(self):
        """内容左移 35px (镜头右移) → 暴露的平台屏幕坐标也应左移 35px。"""
        tt = self._tracker_with_platform()
        before = tt.platforms[0]
        tt.apply_camera_shift(-35, 0)
        after = tt.platforms[0]
        assert after[1] == before[1] - 35
        assert after[2] == before[2] - 35
        assert after[0] == before[0]

    def test_camera_shift_accumulates_between_terrain_frames(self):
        """地形模型每 3 帧才跑一次, 中间两帧靠位姿累积 —— 这正是修前偏 75px 的地方。"""
        tt = self._tracker_with_platform()
        before = tt.platforms[0]
        for _ in range(3):                     # 3 个中间帧, 每帧滚动 35px
            tt.apply_camera_shift(-35, 0)
            tt.update([], [])                  # 非地形帧: 没有新观察
        after = tt.platforms[0]
        assert after[1] == before[1] - 105, "位姿必须逐帧累积, 而不是等地形帧才更新"

    def test_zero_shift_is_noop(self):
        tt = self._tracker_with_platform()
        before = tt.platforms[0]
        tt.apply_camera_shift(0, 0)
        assert tt.platforms[0] == before


class TestNametagUIRejection:
    """参考名牌是"深色横条+亮字", 与经验条/血条结构相似 —— 必须靠位置先验挡住底部 UI。"""

    def _canvas_with_two_matches(self):
        import numpy as np
        from src.perception.nametag_hsv_locator import NametagHSVLocator
        nl = NametagHSVLocator()
        if nl._ref is None:
            pytest.skip("当前模板没有绑定名牌图")
        h, w = 768, 1366
        rng = np.random.default_rng(0)
        canvas = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)   # 噪声背景
        tpl = cv2.cvtColor(nl._ref, cv2.COLOR_GRAY2BGR)
        th, tw = tpl.shape[:2]
        real_y, ui_y = int(h * 0.60), int(h * 0.95)                 # 真名牌 / 经验条位置
        canvas[real_y:real_y + th, 600:600 + tw] = tpl
        canvas[ui_y:ui_y + th, 700:700 + tw] = tpl                  # UI 里一模一样的图案
        return nl, canvas, real_y, ui_y

    def test_bottom_ui_match_is_rejected(self):
        nl, canvas, real_y, ui_y = self._canvas_with_two_matches()
        obs = nl.observe(canvas, anchor=None)
        assert obs.ok and obs.source == "ref"
        # 玩家中心 = 名牌顶 - FEET_TO_CENTER; 命中的必须是中部那个, 不是底部 UI 那个
        from src.perception.nametag_hsv_locator import FEET_TO_CENTER
        assert abs(obs.y - (real_y - FEET_TO_CENTER)) <= 4, f"锁到了 UI (y={obs.y}), 应锁中部名牌"

    def test_scale_calibration_ignores_bottom_ui(self):
        """回归: 标定原来全帧扫, 被经验条带偏成 scale=0.65 (真值 1.00), 之后每帧都锁 UI。"""
        nl, canvas, _, _ = self._canvas_with_two_matches()
        nl.observe(canvas, anchor=None)
        assert nl._scale is not None and abs(nl._scale - 1.0) < 0.08, f"标定尺度={nl._scale}"


import cv2  # noqa: E402  (测试末尾导入, 不影响上面的纯逻辑用例)
from src.brain.entity_tracker import TerrainTracker  # noqa: E402


# ────────────────────────── 下落: 不用 ↓+Alt, 走出边缘 ──────────────────────────

class _FakeController:
    def __init__(self):
        self.calls = []
    def key_down(self, k): self.calls.append(("key_down", k))
    def key_up(self, k): self.calls.append(("key_up", k))
    def jump_down(self): self.calls.append(("jump_down", None))
    def jump(self): self.calls.append(("jump", None))
    def edge_jump_up(self, d): self.calls.append(("edge_jump_up", d))
    def climb_up(self, t): self.calls.append(("climb_up", t))
    def climb_down(self, t): self.calls.append(("climb_down", t))
    def diagonal_jump(self, d): self.calls.append(("diagonal_jump", d))
    def move_direction(self, d, duration=0.3): self.calls.append(("move", d))
    def attack_single(self): self.calls.append(("attack", None))


class _FakeBrain:
    engage_range_x = 120
    def __init__(self, pos=(600, 380)): self._pos = pos
    def player_reliable(self): return True
    def player_pos(self): return self._pos
    def any_target_in_range(self, hold=False): return True   # 立刻结束行走循环
    def nearest_target(self): return None
    def world_moving(self, threshold=3.0): return True


class TestDescendWithoutJumpDown:
    """回归: 站在最底层反复 ↓+Alt 是空动作, 表现为原地无故抽搐。"""

    def _mover(self):
        from src.brain.patrol_mover import PatrolMover
        mv = PatrolMover()
        mv.allow_jump_down = False
        return mv

    def test_never_presses_jump_down_by_default(self):
        mv = self._mover()
        ctl, brain = _FakeController(), _FakeBrain()
        sup = (400, 500, 900)                      # 玩家脚下的平台
        tgt = _mon(700, cy=560)                    # 目标在这块平台的跨度内 (正下方)
        mv._descend(ctl, tgt, 600, 380, [sup], brain)
        assert not any(c[0] == "jump_down" for c in ctl.calls), "默认不该按 ↓+Alt"
        assert any(c[0] == "key_down" for c in ctl.calls), "应该改为走出边缘"

    def test_walks_to_nearer_edge_when_target_is_directly_below(self):
        """目标在平台跨度内 → 朝它走是走不下去的, 得先走到较近的一侧边缘迈出去。"""
        mv = self._mover()
        ctl, brain = _FakeController(), _FakeBrain()
        sup = (400, 500, 900)
        tgt = _mon(700, cy=560)
        mv._descend(ctl, tgt, 600, 380, [sup], brain)   # px=600: 左边缘更近 (100 vs 300)
        assert ("key_down", "left") in ctl.calls, f"应朝较近的左边缘走, 实际 {ctl.calls}"

    def test_walks_toward_target_when_it_is_off_platform(self):
        """目标在平台之外 → 直接朝它走, 自然会走出边缘落下。"""
        mv = self._mover()
        ctl, brain = _FakeController(), _FakeBrain()
        sup = (400, 500, 900)
        tgt = _mon(1200, cy=560)
        mv._descend(ctl, tgt, 600, 380, [sup], brain)
        assert ("key_down", "right") in ctl.calls, f"应朝目标走, 实际 {ctl.calls}"

    def test_template_can_re_enable_jump_down(self):
        mv = self._mover()
        mv.allow_jump_down = True
        ctl, brain = _FakeController(), _FakeBrain()
        mv._descend(ctl, _mon(700, cy=560), 600, 380, [(400, 500, 900)], brain)
        assert ("jump_down", None) in ctl.calls


class TestTerrainGraphDropThreshold:
    """回归: 同一层被融合抖出 4px 高度差, 也被当成了"下面还有一层"。"""

    def test_tiny_height_difference_is_not_a_lower_layer(self):
        from src.brain.terrain_graph import TerrainGraph
        g = TerrainGraph([(392, 0, 1000), (396, 100, 900)])   # 只差 4px = 同一层
        assert not g.can_jump_to(0, 1), "4px 不是一层楼"

    def test_real_drop_is_still_a_down_edge(self):
        from src.brain.terrain_graph import TerrainGraph, MIN_DROP_DY
        g = TerrainGraph([(400, 0, 1000), (400 + MIN_DROP_DY + 10, 100, 900)])
        assert g.can_jump_to(0, 1)
        assert any(a == "down" for _, a in g.reachable_from(0))

    def test_implicit_ground_no_longer_drops_onto_same_floor(self):
        """玩家踩隐含地面(无限宽)时, x 上与每块平台都重叠 —— 正是假下层的温床。"""
        from src.brain.terrain_graph import TerrainGraph
        ground = (392, -10**6, 10**6)
        g = TerrainGraph([ground, (396, 100, 900), (600, 100, 900)])
        assert not g.can_jump_to(0, 1), "同层 4px 不该是下层"
        assert g.can_jump_to(0, 2), "真的低 208px 仍应可下"


class TestTerrainDriftAccumulation:
    """回归: 挂久了地形"到处乱飘" —— 位姿累积漂移, 而平滑追不上。"""

    def _tracker(self):
        from src.brain.entity_tracker import TerrainTracker
        tt = TerrainTracker()
        for _ in range(TerrainTracker.MIN_SEEN_FRAMES):
            tt.update([(400, 300, 800)], [])
        return tt

    def test_detection_snaps_stale_entity_instead_of_blending(self):
        """位姿漂了 40px 之后, 下一帧检测必须把实体**直接拉回**, 不能只纠正 30%。"""
        tt = self._tracker()
        tt.apply_camera_shift(-40, 0)          # 位姿多走了 40px (漂移)
        tt.update([(400, 300, 800)], [])       # 检测说平台还在原来的屏幕位置
        y, xl, xr = tt.platforms[0]
        assert abs(xl - 300) <= 1 and abs(xr - 800) <= 1, \
            f"应立刻对齐到检测, 实际 {(y, xl, xr)}"

    def test_small_jitter_is_still_smoothed(self):
        """几像素的检测抖动仍然要平滑掉, 不能一有残差就硬贴。"""
        tt = self._tracker()
        before = tt.platforms[0]
        tt.update([(400, 305, 805)], [])       # 抖动 5px < SNAP_PX
        after = tt.platforms[0]
        assert before[1] < after[1] < 305, f"5px 抖动应被平滑, 实际 {after}"

    def test_no_accumulation_over_long_run(self):
        """核心回归: 相位相关每帧偏一点(实测约 0.2~0.4px, 同向), 长跑后不能越飘越远。

        构造与实机一致的情形: 内容每帧真实左移 30px, 检测如实反映; 但我们测出来是 30.4px。
        偏差每帧注入一次, 300 帧共 120px —— 只要检测到来时能把实体拉回, 就不该累积。
        """
        tt = self._tracker()
        true_x = 300.0
        for i in range(300):
            true_x -= 30                                  # 真实: 内容左移 30px/帧
            tt.apply_camera_shift(-30.4, 0)               # 测量: 偏 0.4px (同向, 会累积)
            if i % 3 == 0:                                # 每 3 帧一次地形检测 (TERRAIN_EVERY)
                tt.update([(400, round(true_x), round(true_x) + 500)], [])
            else:
                tt.update([], [])
        tt.apply_camera_shift(-30.4, 0)
        true_x -= 30
        tt.update([(400, round(true_x), round(true_x) + 500)], [])   # 最后一帧有检测
        y, xl, xr = tt.platforms[0]
        drift = abs(xl - true_x)
        assert drift <= 5, f"检测到来后仍差 {drift}px —— 累积偏差没被锚回去"


class TestMonsterTrackStability:
    """回归: 怪检测出来了但不稳定 → 决策层就是不打。"""

    def _det(self, conf=0.9, cx=500, cy=400):
        return Monster(id=0, name="M", cx=cx, cy=cy, w=60, h=60, conf=conf)

    def _live(self, tr):
        from src.brain.entity_tracker import COAST_FRAMES
        return [m for m in tr.monsters if m.confirmed and m.miss_frames <= COAST_FRAMES]

    def test_flickering_monster_still_becomes_attackable(self):
        """稳定闪烁 (看到/漏检交替) 的怪, 原来永远凑不齐"连续两帧" → 永远打不了。"""
        tr = MonsterTracker()
        t = 0.0
        for i in range(6):
            t += 0.14
            tr.update([self._det()] if i % 2 == 0 else [], now=t, strong_conf=0.35)
        assert self._live(tr), "一帧有一帧没的怪也必须能打"

    def test_confirmed_monster_coasts_through_brief_miss(self):
        """确认过的怪漏检 1~3 帧仍可打 (用最后已知位置), 不能一眨眼就消失。"""
        tr = MonsterTracker()
        t = 0.0
        for _ in range(2):
            t += 0.14
            tr.update([self._det()], now=t, strong_conf=0.35)
        assert self._live(tr)
        for _ in range(3):
            t += 0.14
            tr.update([], now=t, strong_conf=0.35)
        assert self._live(tr), "3 帧内的漏检应 coast, 不该丢目标"

    def test_low_conf_junk_never_starts_a_track(self):
        """杂物只有低分 → 永远起不了批 (这是 filter_conf 收紧要保住的性质)。"""
        tr = MonsterTracker()
        t = 0.0
        for _ in range(50):
            t += 0.14
            tr.update([self._det(conf=0.25)], now=t, strong_conf=0.35)
        assert not any(m.confirmed for m in tr.monsters)

    def test_weak_detection_maintains_confirmed_track(self):
        """已确认的怪掉到维持门槛 (0.18~0.35) 仍然跟得住 —— 这是"积极打"的关键。"""
        tr = MonsterTracker()
        t = 0.0
        for _ in range(2):
            t += 0.14
            tr.update([self._det(conf=0.9)], now=t, strong_conf=0.35)
        for _ in range(4):
            t += 0.14
            tr.update([self._det(conf=0.20)], now=t, strong_conf=0.35)
        assert self._live(tr), "确认过的目标掉分不该被丢掉"

    def test_weak_only_track_expires(self):
        """但杂物不能给死怪无限续命: 只剩弱检测超过 WEAK_ONLY_MAX_SEC 就该消亡。"""
        tr = MonsterTracker()
        t = 0.0
        for _ in range(2):
            t += 0.14
            tr.update([self._det(conf=0.9)], now=t, strong_conf=0.35)
        for _ in range(20):                      # 2.8s 只有弱检测
            t += 0.14
            tr.update([self._det(conf=0.20)], now=t, strong_conf=0.35)
        assert not tr.monsters, "只靠弱检测续命不能超过 WEAK_ONLY_MAX_SEC"

    def test_camera_shift_moves_coasting_monsters(self):
        """coast 期间镜头在动, 怪的屏幕坐标必须跟着动, 否则打的是它 0.4s 前的位置。"""
        tr = MonsterTracker()
        t = 0.0
        for _ in range(2):
            t += 0.14
            tr.update([self._det(cx=500)], now=t, strong_conf=0.35)
        tr.apply_camera_shift(-35, 0)
        assert tr.monsters[0].cx == 465


class _WorldBrain:
    """模拟"相机跟随玩家"的世界: 玩家在世界坐标里走, 屏幕坐标几乎不动。

    这正是巡逻的难点 —— 屏幕里看不出自己走了多远, 只有世界坐标能表达"去那边的尽头"。
    """

    engage_range_x = 120

    def __init__(self, world_x=0.0, screen_x=800, screen_y=431, monster_wx=None):
        self.world_x = float(world_x)
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.monster_wx = monster_wx
        self.monster_cy = None      # None = 与玩家同高
        self.in_range = False

    # 相机跟随: offset = 世界位置 - 屏幕位置 (玩家永远在屏幕中间)
    def world_offset(self):
        return (self.world_x - self.screen_x, 0)

    def player_pos(self):
        return (self.screen_x, self.screen_y)

    def player_world(self):
        return (self.world_x, float(self.screen_y))

    def player_reliable(self):
        return True

    def any_target_in_range(self, hold=False):
        return self.in_range

    def any_target_near(self, d=360):
        return self.in_range

    def world_moving(self, threshold=3.0):
        return True

    def nearest_target(self):
        if self.monster_wx is None:
            return None
        ox, _ = self.world_offset()
        cy = self.screen_y if self.monster_cy is None else self.monster_cy
        return Monster(id=1, name="M", cx=int(self.monster_wx - ox), cy=cy,
                       w=60, h=60, conf=0.9)


class TestPatrolGoal:
    """回归: 巡逻在原地左右小幅晃 —— 因为它按时间翻向, 根本没有目标。

    现在是目标驱动: 看地形 → 选最远的落脚点 (世界坐标) → 一路走过去。
    """

    def _mover(self):
        from src.brain.patrol_mover import PatrolMover
        mv = PatrolMover()
        mv.patrol_pause_chance = 0.0      # 关掉随机发呆, 保证测试确定性
        return mv

    # 屏幕坐标里的地形: 左端一块近的, 右端一块远的
    PLATFORMS = [(466, 700, 900), (466, 1000, 1340), (466, 60, 300)]

    def test_picks_farthest_terrain_point_not_nearest(self):
        """"去最远的地方" —— 选点按距离排序, 不是随便挑一个近的。"""
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, self.PLATFORMS, [], brain)
        # 屏幕 800 = 世界 0, 所以世界坐标 = 屏幕 - 800
        # 候选距离: 右端 1340→540, 左端 60→-740 (740)
        assert abs(mv._goal[0]) >= 500, f"应该挑远的, 实际 {mv._goal}"

    def test_goal_is_in_world_coords_and_survives_camera_scroll(self):
        """目标必须钉在世界坐标: 镜头滚动 300px 后, 目标的世界坐标不能变。

        这是"大幅度变换位置"的前提 —— 用屏幕坐标存目标, 走一步目标就跟着跑, 永远走不到。
        """
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, self.PLATFORMS, [], brain)
        goal_before = mv._goal
        brain.world_x = 300.0                      # 玩家走了 300px, 镜头跟着滚
        assert mv._goal == goal_before, "目标的世界坐标不该随镜头改变"

    def test_goal_persists_across_calls(self):
        """选定目标后要一路走过去, 不能每次调用都重选 (那就又变成翻烧饼了)。"""
        mv = self._mover()
        ctl, brain = _FakeController(), _WorldBrain(world_x=0.0)
        brain.in_range = True                      # 让行走循环立刻返回, 只看目标是否稳定
        mv.patrol(ctl, 800, 431, self.PLATFORMS, [], brain)
        first = mv._goal
        assert first is not None
        for _ in range(5):
            mv.patrol(ctl, 800, 431, self.PLATFORMS, [], brain)
        assert mv._goal == first, "没到达之前不该换目标"

    def test_arriving_marks_visited_and_picks_new_goal(self):
        """到达后记为"去过"并换下一个 —— 否则会在同一个点上反复横跳。"""
        mv = self._mover()
        ctl, brain = _FakeController(), _WorldBrain(world_x=0.0)
        brain.in_range = True
        mv.patrol(ctl, 800, 431, self.PLATFORMS, [], brain)
        goal = mv._goal
        brain.world_x = goal[0]                    # 走到了
        mv.patrol(ctl, 800, 431, self.PLATFORMS, [], brain)
        assert mv._recently_visited(goal), "到过的点要记下来"
        assert mv._goal != goal, "到达后应该换新目标"

    def test_visited_points_are_deprioritised(self):
        """刚扫过的地方降权, 优先去没去过的 —— 人不会在清空的角落反复刷。"""
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, self.PLATFORMS, [], brain)
        first = mv._goal
        mv._mark_visited(first)
        mv._goal = None
        mv._pick_goal(0.0, 431.0, self.PLATFORMS, [], brain)
        assert mv._goal != first, "刚去过的点不该立刻又被选中"

    def test_no_terrain_falls_back_to_long_exploration(self):
        """地形没检出来时也要走远 —— 不能因为"没看到平台"就原地不动。"""
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, [], [], brain)
        assert mv._goal_kind == "探索"
        assert abs(mv._goal[0]) >= mv.goal_explore_push - 1

    def test_implicit_ground_is_not_a_candidate(self):
        """隐含地面节点宽 2000000px, 它的"端点"是假的, 不能当目标。"""
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, [(466, -10**6, 10**6)], [], brain)
        assert mv._goal_kind == "探索", f"不该选到隐含地面端点, 实际 {mv._goal_kind}"

    def test_nearby_monster_becomes_the_goal(self):
        """附近有怪 → 目标直接设成它 (而不是"方向偏置", 那会被左右两只怪拽成钟摆)。"""
        mv = self._mover()
        ctl = _FakeController()
        brain = _WorldBrain(world_x=0.0, monster_wx=200.0)
        brain.in_range = True
        mv.patrol(ctl, 800, 431, self.PLATFORMS, [], brain)
        assert mv._goal_kind == "monster"
        assert abs(mv._goal[0] - 200.0) < 1

    def test_flip_abandons_goal(self):
        """看门狗强制换向时要放弃目标, 否则下一次又朝老目标走回去。"""
        mv = self._mover()
        brain = _WorldBrain(world_x=0.0)
        mv._pick_goal(0.0, 431.0, self.PLATFORMS, [], brain)
        assert mv._goal is not None
        mv.flip()
        assert mv._goal is None


class _PlanBrain:
    """给 approach 用的最小 brain: 玩家不动, 没有怪进范围 (只看动作序列)。"""
    engage_range_x = 120

    def __init__(self, screen_x=800, screen_y=465):
        self.screen_x = screen_x
        self.screen_y = screen_y

    def player_pos(self): return (self.screen_x, self.screen_y)
    def player_world(self): return (float(self.screen_x), float(self.screen_y))
    def world_offset(self): return (0, 0)
    def player_reliable(self): return True
    def any_target_in_range(self, hold=False): return True   # 让行走循环立刻返回
    def any_target_near(self, d=360): return False
    def world_moving(self, threshold=3.0): return True
    def nearest_target(self): return None


# ────────────────────────── MovePlan: 怎么过去 ──────────────────────────

class TestMovePlan:
    """回归: 怪在上层平台时人卡在下面平台来回蹭。

    根因是"我够不够得到那只怪"在两处各用 dx/dy 现推一遍, 且要求怪在头顶 150px 内 ——
    实测上方的怪普遍在 260~520px 外, 于是 458 个"一跳可上"的目标全被判成够不着。
    现在收敛成一个 MovePlan: plan_move 唯一生产, is_reachable / approach 都消费它。
    """

    def _mover(self):
        from src.brain.patrol_mover import PatrolMover
        mv = PatrolMover()
        mv.patrol_pause_chance = 0.0
        return mv

    # 玩家脚底 y = py + 35。下层平台 y=500, 上层平台 y=400 (落差 100, 一跳可上)
    LOWER = (500, 0, 1366)
    UPPER = (400, 900, 1300)

    def _target(self, cx, cy, h=60):
        return Monster(id=1, name="M", cx=cx, cy=cy, w=60, h=h, conf=0.9)

    def test_far_monster_on_upper_platform_is_jump_up_not_unreachable(self):
        """核心回归: 怪在上层平台、水平 300px 外 → 该跳上去, 不是"够不着"。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=370)          # 脚底 400 = 上层平台
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER, self.UPPER], [])
        assert plan.kind is MoveKind.JUMP_UP, f"{plan.kind} / {plan.reason}"
        assert plan.surface_dy > 0

    def test_takeoff_point_is_under_the_target_platform(self):
        """起跳点要在目标平台正下方 (且从两端内缩, 别贴边跳)。"""
        from src.brain.patrol_mover import MoveKind, TAKEOFF_INSET
        mv = self._mover()
        tgt = self._target(cx=1100, cy=370)
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER, self.UPPER], [])
        assert plan.kind is MoveKind.JUMP_UP
        assert self.UPPER[1] + TAKEOFF_INSET <= plan.takeoff_x <= self.UPPER[2] - TAKEOFF_INSET

    def test_monster_above_without_detected_platform_still_jumps(self):
        """怪脚下的平台没被检出来 (实测 60 例全是这样) → 仍然要跳, 起跳点取怪的 x。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=370)
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], [])   # 只有下层平台
        assert plan.kind is MoveKind.JUMP_UP
        assert abs(plan.takeoff_x - 1100) <= 1

    def test_absurdly_high_is_unreachable(self):
        """高出单跳估计 2 倍以上 → 试跳也没用, 诚实地说够不着, 让巡逻换地方。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=100)          # 脚底 130, 落差 370 > 140×2
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], [])
        assert plan.kind is MoveKind.UNREACHABLE

    def test_above_single_jump_estimate_still_tries_jumping(self):
        """核心: 地形/梯子检测不可靠, **不能从"没检测到梯子"推出"够不着"**。

        怪在上面就说明那儿有路。落差略超单跳估计时先去试跳,
        跳不上由失败计数学到 —— 而不是靠一条否定证据把目标判死。
        """
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=270)          # 脚底 300, 落差 200 (>140, <280)
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], [])
        assert plan.kind is MoveKind.JUMP_UP, f"{plan.kind} / {plan.reason}"
        assert "试跳" in plan.reason

    def test_takeoff_is_under_the_monster_not_gated_by_terrain(self):
        """起跳点取怪的正下方 —— 怪站在那儿就说明那儿有落脚点, 不依赖平台被检出来。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1150, cy=370)
        no_terrain = mv.plan_move(tgt, 800, 465, [self.LOWER], [])     # 上层平台没检出
        assert no_terrain.kind is MoveKind.JUMP_UP
        assert abs(no_terrain.takeoff_x - 1150) <= 1

    def test_too_high_with_rope_is_climb(self):
        """有一根跨越这段落差的梯子 → 走过去爬上去。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=100)
        ropes = [(1000, 100, 520)]                   # 梯子从 y=100 到 520, 覆盖两层
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], ropes)
        assert plan.kind is MoveKind.CLIMB
        assert plan.takeoff_x == 1000

    def test_same_level_is_walk(self):
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=470)          # 脚底 500 = 下层平台
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], [])
        assert plan.kind is MoveKind.WALK

    def test_below_is_drop(self):
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        tgt = self._target(cx=1100, cy=600)          # 脚底 630, 比玩家低 130
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER], [])
        assert plan.kind is MoveKind.DROP

    def test_flat_mode_disables_vertical_moves(self):
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        mv.flat_mode = True
        tgt = self._target(cx=1100, cy=370)
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER, self.UPPER], [])
        assert plan.kind is MoveKind.UNREACHABLE

    def test_is_reachable_agrees_with_plan(self):
        """不变量: 决策层放行的, 执行层一定有动作可做 (与射程 engage/hold 同构)。"""
        from src.brain.patrol_mover import MoveKind
        mv = self._mover()
        P, R = [self.LOWER, self.UPPER], [(1000, 100, 520)]
        for cx in range(200, 1350, 90):
            for cy in (100, 370, 470, 600):
                tgt = self._target(cx=cx, cy=cy)
                plan = mv.plan_move(tgt, 800, 465, P, R)
                assert mv.is_reachable(tgt, 800, 465, P, R) == (plan.kind is not MoveKind.UNREACHABLE)

    def test_approach_jumps_immediately_even_when_far_from_takeoff(self):
        """执行: 不再要求先走到精确的起跳点才跳 —— 起跳点本来就是估的 (地形常漏检),
        死磕对齐只是在拿不可靠坐标浪费时间。没对齐就边跳边靠近 (edge_jump_up 空中
        会飘一段, 本身就在缩小水平差); 对齐了就原地上跳。"""
        mv = self._mover()
        tgt = self._target(cx=1100, cy=370)
        # 远离起跳点: 应该立刻边跳边靠近, 不是先纯走过去
        ctl, brain = _FakeController(), _PlanBrain(screen_x=800)
        mv.approach(ctl, tgt, 800, 465, [self.LOWER, self.UPPER], brain, ropes=[])
        assert any(c[0] == "edge_jump_up" for c in ctl.calls), f"没对齐时该边跳边靠近: {ctl.calls}"
        assert not any(c[0] == "key_down" for c in ctl.calls), f"不该先纯走位: {ctl.calls}"
        # 站在起跳点上: 原地上跳
        plan = mv.plan_move(tgt, 800, 465, [self.LOWER, self.UPPER], [])
        ctl2, brain2 = _FakeController(), _PlanBrain(screen_x=plan.takeoff_x)
        mv.approach(ctl2, tgt, plan.takeoff_x, 465, [self.LOWER, self.UPPER], brain2, ropes=[])
        assert any(c[0] == "jump" for c in ctl2.calls), f"站在起跳点却没跳: {ctl2.calls}"

    def test_patrol_does_not_chase_unreachable_monster(self):
        """回归 (卡在下面平台): 追一只够不着的怪会走到它正下方, 而"到达"只比 x ——
        到了就换目标、怪还在又设回来, 于是在下层平台原地来回蹭。"""
        mv = self._mover()
        ctl = _FakeController()
        brain = _WorldBrain(world_x=0.0, monster_wx=200.0)
        brain.in_range = True
        brain.monster_cy = 100          # 怪在很高的上方, 没梯子 → 够不着
        mv.patrol(ctl, 800, 465, [self.LOWER], [], brain, world_offset=(-800, 0))
        assert mv._goal_kind != "monster", "不该把够不着的怪设成巡逻目标"
