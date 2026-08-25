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
    # 期望: 会话在帧5 (miss 满2) 被停; 帧6 不拿 0.25 起新会话 (starts 保持 1, s_bbox=None)
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
    )
    all_ok &= b_ok
    print(f"  场景B 结果: {'PASS' if b_ok else 'FAIL'} (starts 应=1, stops≥1, 帧6 s_bbox=None)")

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

    print(f"\n全部: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
