"""本地冒烟测试: _ServerState 空间门/事件边界逻辑 (mock opencv/samurai, 不碰 GPU)

复现 replay 两个异常场景, 断言新逻辑行为:
  A. idx 251: countdown 中倒计时数字抢走 opencv (conf 0.65 @ (708,457), 星形在 (599,383))
     → 旧 samurai 会话必须在 countdown 首帧被掐掉, center 保持星形不跳数字。
  B. idx 358: 星形消失 2 帧后 conf=0.25 检测在远处 (368,432)
     → 不能拿 conf=0.25 直接锚 samurai 会话; 会话必须在 miss 满 2 帧时停掉。

用法: python tools/_test_server_gates.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.lie_detector.state import LiePhase  # noqa: E402


class MockOpenCV:
    ready = True
    import_error = None

    def __init__(self):
        self.seq = []          # 每帧 (active, phase, center, conf, bbox)
        self.i = 0

    def detect(self, frame, scale=1.0):
        r = self.seq[self.i]
        self.i += 1
        active, phase, center, conf, bbox = r
        if not active:
            from src.perception.lie_detector.state import LieDetectResult
            return LieDetectResult(active=False)
        from src.perception.lie_detector.state import LieDetectResult
        return LieDetectResult(active=True, phase=phase, target_center=center,
                               target_bbox=bbox, confidence=conf)


class MockSamurai:
    ready = True
    import_error = None
    model_ready = True
    build_error = None

    def __init__(self):
        self.session_active = False
        self.starts = 0
        self.steps = 0
        self.stops = 0
        self._stepped = 0

    def warm(self):
        return True

    def start(self, frame, bbox):
        self.session_active = True
        self.starts += 1
        self._track_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        return True

    def step(self, frame):
        # 追踪中: 返回跟随检测中心 (模拟 samurai 正常跟目标)
        self.steps += 1
        if self._track_center is None:
            self.session_active = False
            return None
        c = self._track_center
        self._track_center = (c[0] + 1, c[1] + 1)   # 正常小幅移动
        return (c, 0.9, (c[0] - 20, c[1] - 20, c[0] + 20, c[1] + 20))

    def stop(self):
        self.session_active = False
        self.stops += 1


def build_state(opencv, samurai):
    with patch("src.perception.lie_detector.opencv_backend.OpenCVBackend", lambda repo: opencv), \
         patch("src.perception.lie_detector.samurai_stream.SamuraiStream", lambda repo, model_size, image_size: samurai):
        from tools.lie_detect_server import _ServerState
        state = _ServerState("fake_repo", "base_plus", image_size=512)
        return state


def frame(c=None):
    return np.zeros((540, 960, 3), dtype=np.uint8)


def run(scenario, opencv_seq, expect_centers=None, expect_no_session_start_on=None):
    ocv = MockOpenCV()
    ocv.seq = opencv_seq
    sm = MockSamurai()
    state = build_state(ocv, sm)
    outs = []
    for _ in range(len(opencv_seq)):
        try:
            outs.append(state.handle_frame(frame()))
        except IndexError:
            break
    print(f"\n=== {scenario} ===")
    for o in outs:
        print(f"  active={o['active']} phase={o['phase']} conf={o['confidence']:.2f} "
              f"center={o['center']} s_bbox={'Y' if o['s_bbox'] else '-'}")
    print(f"  samurai: starts={sm.starts} steps={sm.steps} stops={sm.stops}")
    if expect_centers:
        got = [tuple(o['center']) for o in outs if o['center']]
        ok = all(g == e for g, e in zip(got, expect_centers))
        print(f"  centers {got} vs expect {expect_centers} → {'PASS' if ok else 'FAIL'}")
    return outs, sm


def main() -> int:
    all_ok = True

    # ── 场景 A: idx 251 countdown 错选数字 ──
    # 帧1-2: countdown 首 2 帧 (star @599,383), 去抖激活 + 首锚
    # 帧3-5: countdown 持续, 数字抢目标 (conf 0.65-0.68 @708,457) 且持续多帧
    # 期望: 激活后 center 全程 599,383 (数字被空间门拒, hold 星形); 无会话
    cd = LiePhase.COUNTDOWN
    seqA = [
        (True, cd, (599, 383), 0.63, (590, 375, 608, 391)),
        (True, cd, (599, 383), 0.63, (590, 375, 608, 391)),
        (True, cd, (708, 457), 0.65, (680, 430, 736, 484)),
        (True, cd, (712, 466), 0.66, (680, 439, 744, 493)),
        (True, cd, (699, 473), 0.68, (670, 448, 728, 498)),
    ]
    outsA, smA = run("A: countdown 错选数字 (idx 251 场景)", seqA)
    active_centers = [o['center'] for o in outsA if o['active']]
    a_ok = (
        all(c == [599, 383] for c in active_centers)
        and smA.starts == 0 and smA.steps == 0
    )
    all_ok &= a_ok
    print(f"  场景A 结果: {'PASS' if a_ok else 'FAIL'} (active 帧 center 应全 599,383; 会话应从不启动)")

    # ── 场景 B: idx 358 事件切换 ──
    # 帧1: tracking star @(762,358) conf 0.99, samurai 会话建立
    # 帧2-3: tracking, samurai step 正常跟随 (miss 0)
    # 帧4-5: opencv miss (目标消失)
    # 帧6: conf=0.25 @(368,432) (新事件弱检测)
    # 期望: 会话在帧5 (miss 满2) 被停; 帧6 不拿 0.25 起新会话 (starts 保持 1),
    #       且 P4 冷锚门拒 0.25 → center=None (旧逻辑会直接锚 digit 把鼠标拽走 — 这是没断言的漏网 bug)
    tr = LiePhase.TRACKING
    seqB = [
        (True, tr, (762, 358), 0.99, (750, 350, 774, 366)),
        (True, tr, (762, 364), 0.99, (750, 356, 774, 372)),
        (True, tr, (761, 362), 0.99, (749, 354, 773, 370)),
        (False, LiePhase.IDLE, None, 0.0, None),
        (False, LiePhase.IDLE, None, 0.0, None),
        (True, tr, (368, 432), 0.25, (350, 420, 386, 444)),
    ]
    outsB, smB = run("B: 事件切换 2 miss 后弱检测 (idx 358 场景)", seqB)
    b_ok = (
        smB.starts == 1                # 只有帧1 起过一次会话
        and smB.stops >= 1             # miss 满 2 停掉
        and outsB[5]['s_bbox'] is None  # 帧6 没起新会话 (0.25 被门禁拦下)
        and outsB[5]['center'] is None  # P4: 冷锚 conf 0.25 < 0.35 → center=None 不锚
    )
    all_ok &= b_ok
    print(f"  场景B 结果: {'PASS' if b_ok else 'FAIL'} (starts 应=1, stops≥1, 帧6 s_bbox=None center=None)")

    # ── 场景 C: countdown 预热 samurai (idx 256 区域) ──
    # 帧1-4: countdown 星形静止 conf 0.5 (较亮, 连续稳定)
    # 帧5:   tracking 进入 → 会话已预热 → 直接 step (s_bbox Y)
    # 期望: starts=1 (countdown 第3帧就起), 帧5 s_bbox 非 None
    cd = LiePhase.COUNTDOWN
    tr = LiePhase.TRACKING
    seqC = [
        (True, cd, (1060, 44), 0.5, (1040, 20, 1080, 68)),
        (True, cd, (1060, 44), 0.5, (1040, 20, 1080, 68)),
        (True, cd, (1060, 44), 0.5, (1040, 20, 1080, 68)),
        (True, cd, (1060, 44), 0.5, (1040, 20, 1080, 68)),
        (True, tr, (1060, 44), 0.25, (1040, 20, 1080, 68)),
    ]
    outsC, smC = run("C: countdown 预热 (idx 256 场景)", seqC)
    c_ok = (
        smC.starts == 1                 # countdown 内预热起过一次
        and outsC[4]['s_bbox'] is not None  # tracking 首帧直接 step 出 s_bbox
        and outsC[4]['center'] == [1060, 44]
    )
    all_ok &= c_ok
    print(f"  场景C 结果: {'PASS' if c_ok else 'FAIL'} (starts 应=1, 帧5 s_bbox 非 None)")

    # ── 场景 D: tracking 中途相位振荡 TRACKING→COUNTDOWN→TRACKING + 远处 conf=1.0 大块数字 ──
    # 帧1-2: tracking, samurai 会话建立+跟随
    # 帧3:   countdown 振荡 (matched 抖动) → P2 协成 tracking → 不 new_event, 会话继续 step
    # 帧4-5: 远 conf=1.0 大块 (80x120) 数字抢目标 → 锚点守卫 → 远跳确认, 确认满 2 帧后尺寸门拒 → hold 星形
    # 期望: starts=1 (不重锚), 不 new_event (帧3 coerced), center 全程星形不跳数字
    cd = LiePhase.COUNTDOWN
    seqD = [
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (601, 401), 0.99, (591, 391, 611, 411)),
        (True, cd, (600, 400), 0.5, (590, 390, 610, 410)),
        (True, tr, (900, 300), 1.0, (850, 230, 950, 370)),
        (True, tr, (900, 300), 1.0, (850, 230, 950, 370)),
    ]
    outsD, smD = run("D: 相位振荡 + 远处 conf1.0 大块 (idx 81 场景)", seqD)
    d3, d4, d5 = outsD[2], outsD[3], outsD[4]
    d_ok = (
        smD.starts == 1 and smD.stops == 0                     # 振荡不杀会话不重锚
        and d3['phase'] == 'tracking' and d3['diag']['coerced'] is True  # P2 协成生效
        and d3['diag']['phase_raw'] == 'countdown'
        and d3['diag']['new_event'] is False                    # 不误触发新事件清锚
        and d4['center'] == [601, 401] and d5['center'] == [601, 401]   # 确认中 hold 星形
        and d5['diag']['rejected'] is True                      # 远跳被拒 (尺寸门)
    )
    all_ok &= d_ok
    print(f"  场景D 结果: {'PASS' if d_ok else 'FAIL'} (振荡不杀会话; 远处数字被尺寸门拒, center 保持星形)")

    # ── 场景 E: tracking 中远处大 bbox 数字 conf=1.0 → 尺寸门拒 (无振荡干扰) ──
    # 帧1: 去抖激活用 (首帧 idle, 不锚); 帧2: 冷锚星形; 帧3-4: 远 conf1.0 大块确认 → 尺寸门拒
    seqE = [
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (900, 300), 1.0, (850, 230, 950, 370)),
        (True, tr, (900, 300), 1.0, (850, 230, 950, 370)),
    ]
    outsE, smE = run("E: tracking 远 conf1.0 大 bbox 数字 → 尺寸门拒", seqE)
    e_ok = (
        smE.starts == 1
        and outsE[2]['center'] == [600, 400]      # 确认第1帧 hold 星形
        and outsE[3]['center'] == [600, 400]      # 确认第2帧尺寸拒 → 仍 hold 最后接受中心
        and all(o['center'] != [900, 300] for o in outsE)
    )
    all_ok &= e_ok
    print(f"  场景E 结果: {'PASS' if e_ok else 'FAIL'} (数字被尺寸门拒, 永不跳 (900,300))")

    # ── 场景 F: 远处同尺寸强候选 conf=0.9 → 第2帧确认后重锚 (合法远移恢复) ──
    seqF = [
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (900, 300), 0.9, (890, 290, 910, 310)),
        (True, tr, (900, 300), 0.9, (890, 290, 910, 310)),
    ]
    outsF, smF = run("F: 远同尺寸 conf0.9 → 2帧确认重锚 (合法远移)", seqF)
    f_ok = (
        smF.starts == 2                        # 帧4重锚 (确认后真重启会话)
        and outsF[2]['center'] == [600, 400]   # 确认第1帧 hold 旧锚
        and outsF[3]['center'] == [900, 300]   # 第2帧确认通过 → 重锚到新位置
    )
    all_ok &= f_ok
    print(f"  场景F 结果: {'PASS' if f_ok else 'FAIL'} (第2帧确认后重锚 (900,300))")

    # ── 场景 J: 守卫路径 (np.bool_) 响应必须可 JSON 序列化 ──
    # 回归: 线上 json.dumps 崩 "Object of type bool is not JSON serializable" —
    # 锚点守卫 `and ... np.hypot(...) > D` 末项是 np.bool_, 落进 diag.anchor_guard。
    # 测试直调 handle_frame 不经过 _send_json, 得显式过 json.dumps 才抓得住。
    import json as _json
    seqJ = [
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (600, 400), 0.99, (590, 390, 610, 410)),
        (True, tr, (900, 300), 0.9, (890, 290, 910, 310)),
        (True, tr, (900, 300), 0.9, (890, 290, 910, 310)),
    ]
    outsJ, smJ = run("J: 守卫 np.bool_ 响应 → json.dumps 可序列化", seqJ)
    j_ok = all(_json.dumps(o) for o in outsJ)          # 每帧都能出 JSON (不崩)
    j_guard_seen = any((o.get("diag") or {}).get("anchor_guard") for o in outsJ)
    all_ok &= j_ok
    print(f"  场景J 结果: {'PASS' if j_ok else 'FAIL'} "
          f"(np.bool_ 序列化{'守卫已触发' if j_guard_seen else '未触发守卫'} json.dumps 全通过)")

    # ── 场景 G: 冷起始 conf=0.25 junk → center=None 不锚; 下帧 conf=0.5 → 正常跟随 ──
    seqG = [
        (True, tr, (368, 432), 0.25, (350, 420, 386, 444)),
        (True, tr, (368, 432), 0.25, (350, 420, 386, 444)),
        (True, tr, (370, 434), 0.5, (350, 420, 390, 448)),
    ]
    outsG, smG = run("G: 冷起始 conf0.25 junk (idx 144/250 场景)", seqG)
    g_ok = (
        smG.starts == 1                                     # 只有 conf0.5 起会话
        and outsG[1]['active'] is True and outsG[1]['center'] is None   # 帧2: 0.25 冷锚拒
        and outsG[2]['center'] == [370, 434]                # 帧3: 0.5 正常锚
    )
    all_ok &= g_ok
    print(f"  场景G 结果: {'PASS' if g_ok else 'FAIL'} (0.25 不锚 center=None; 0.5 正常跟随)")

    # ── 场景 H: countdown 冷起始超大 bbox (560x440) → P4 冷锚尺寸上限拒 → 会话不启动 ──
    seqH = [
        (True, cd, (480, 270), 0.5, (200, 50, 760, 490)),
        (True, cd, (480, 270), 0.5, (200, 50, 760, 490)),
        (True, cd, (480, 270), 0.5, (200, 50, 760, 490)),
    ]
    outsH, smH = run("H: countdown 冷起始超大亮块 → 冷锚尺寸上限拒", seqH)
    h_ok = (
        smH.starts == 0                                     # 会话不启动 (无预热)
        and outsH[1]['active'] is True and outsH[1]['center'] is None
        and outsH[2]['center'] is None
    )
    all_ok &= h_ok
    print(f"  场景H 结果: {'PASS' if h_ok else 'FAIL'} (超大块冷锚被拒, 会话不启动)")

    # ── 场景 I: 锚活着+会话死, 远 conf=0.25 候选持续 3+ 帧 → 远跳确认的 conf 门槛拒 ──
    # 回归: BV17eGn69EAM idx100-102 (countdown 锚 + 预热失败无会话 + 0.25 blob 持久 → 错抢),
    #        BV1XuySBvEFa idx274/301/415/576 同型。合法远跳 conf 均 ≥0.5。
    seqI = [
        (True, cd, (600, 400), 0.5, (590, 390, 610, 410)),
        (True, cd, (600, 400), 0.5, (590, 390, 610, 410)),
        (True, tr, (900, 300), 0.25, (890, 290, 910, 310)),
        (True, tr, (900, 300), 0.25, (890, 290, 910, 310)),
        (True, tr, (900, 300), 0.25, (890, 290, 910, 310)),
    ]
    outsI, smI = run("I: 锚活+会话死, 远 conf0.25 暗候选持久 → conf 门槛拒", seqI)
    i_ok = (
        smI.starts == 0                                    # 不重锚不起会话
        and all(o['center'] != [900, 300] for o in outsI)  # 永不跳 (900,300)
        and all(o['center'] == [600, 400] for o in outsI if o['center'])  # 持续 hold 星形
    )
    all_ok &= i_ok
    print(f"  场景I 结果: {'PASS' if i_ok else 'FAIL'} (conf0.25 远跳被 conf 门槛拒, 持续 hold)")

    print(f"\n全部: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
