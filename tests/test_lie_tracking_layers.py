"""
测谎仪混合跟踪层 — 自适应背景差分 + 时序差分 + Kalman + 融合 (Phase 1)。

覆盖:
- AdaptiveBackgroundModel: 预热学背景 / 静态白元素吸收 / 移动块隔离 / freeze 防吸收
- TinyKalman: 起手 / 高置信收敛
- HybridBackend: 瞬移跟随 / 渐隐比纯白块多撑 N 帧 / 01.mp4 帧 40~199 全程回归

依赖 vendored 仓库 (models/lie_detector/) 的用例在仓库缺席时自动 skip;
背景差分 / Kalman 层不依赖仓库, 任意环境可跑。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.lie_detector.adaptive_bg import AdaptiveBackgroundModel
from src.perception.lie_detector.hybrid_backend import HybridBackend
from src.perception.lie_detector.kalman import TinyKalman
from src.perception.lie_detector.opencv_backend import OpenCVBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDORED_REPO = PROJECT_ROOT / "models" / "lie_detector"
VIDEO_01 = PROJECT_ROOT / "tools" / "01.mp4"


def _vendored_repo_or_skip() -> Path:
    if not (VENDORED_REPO / "scripts" / "auto_bbox.py").is_file():
        pytest.skip("models/lie_detector 未落地")
    return VENDORED_REPO


def _dark(size=(360, 640), v=35) -> np.ndarray:
    return np.full((size[0], size[1]), v, dtype=np.uint8)


def _white_block(gray: np.ndarray, box, v=255) -> np.ndarray:
    """在灰度帧上画一个 (v 亮度) 方块, 返回新帧 (不原地改)。"""
    out = gray.copy()
    cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), v, -1)
    return out


# ── AdaptiveBackgroundModel: 背景差分 ──


def test_bg_warmup_learns_plain_background():
    """预热 N 帧纯背景 → 背景就绪, 且背景 vs 背景残差为 0 (无目标)。"""
    bgm = AdaptiveBackgroundModel(warmup_frames=5, residual_thresh=20)
    dark = _dark()
    for _ in range(5):
        r = bgm.update_and_detect(dark)
    assert bgm.ready is True
    assert r is None  # 背景=背景, 无残差


def test_bg_static_white_element_absorbed():
    """预热期无目标 → 静态白元素学进背景; 之后同帧残差为 0 (不误报)。"""
    bgm = AdaptiveBackgroundModel(warmup_frames=3, residual_thresh=20)
    fr = _white_block(_dark(), (100, 100, 200, 200))
    for _ in range(3):
        bgm.update_and_detect(fr, freeze_bbox=None)
    assert bgm.ready
    r = bgm.update_and_detect(fr, freeze_bbox=None)
    assert r is None  # 静态白元素已被吸收进背景


def test_bg_residual_isolates_moving_block():
    """静态白元素(已吸收) + 新位置移动块 → 残差只出移动块, 中心精准。"""
    bgm = AdaptiveBackgroundModel(warmup_frames=3, residual_thresh=20)
    static = _white_block(_dark(), (100, 100, 200, 200))
    for _ in range(3):
        bgm.update_and_detect(static, freeze_bbox=None)

    # 移动块出现在新位置 (freeze 该区 = 模拟 blob 已锁定, 防吸收)
    box = (400, 100, 500, 200)
    fr = _white_block(static, box)
    r = bgm.update_and_detect(fr, freeze_bbox=box)
    assert r is not None
    assert r.center == (450, 150)   # 移动块中心, 不是静态元素处
    assert r.area >= 90 * 90        # 面积≈方块


def test_bg_moving_block_not_absorbed_by_freeze():
    """freeze 区内像素永不吸收: 目标停驻多帧残差仍强 (渐隐兜底的前提)。"""
    bgm = AdaptiveBackgroundModel(warmup_frames=3, residual_thresh=20)
    dark = _dark()
    for _ in range(3):
        bgm.update_and_detect(dark, freeze_bbox=None)
    box = (300, 200, 400, 300)
    for _ in range(10):
        fr = _white_block(dark, box)
        r = bgm.update_and_detect(fr, freeze_bbox=box)
        assert r is not None
        assert r.center == (350, 250)


# ── TinyKalman ──


def test_kalman_initialized_and_high_conf_tracks():
    """起手 (x,y) → initialized; 高置信观测应快速收敛到观测附近。"""
    k = TinyKalman()
    assert not k.initialized
    k.reset(100, 100)
    assert k.initialized
    px, py = k.predict()
    assert abs(px - 100) < 1 and abs(py - 100) < 1
    cx, cy = k.correct(300, 150, confidence=1.0)
    # 高置信 (测量噪声低) → 单步就应贴近观测
    assert abs(cx - 300) < 30
    assert abs(cy - 150) < 10


def test_kalman_low_conf_lags_behind():
    """低置信 (测量噪声高) → 平滑更信预测, 输出滞后于观测。"""
    k = TinyKalman()
    k.reset(100, 100)
    k.correct(300, 100, confidence=1.0)      # 高置信先到位
    hi_out = k.predict()
    k.correct(300, 100, confidence=1.0)
    hi_pos = k.correct(500, 100, confidence=1.0)   # 高置信 → 贴观测
    k.reset(100, 100)
    k.correct(300, 100, confidence=0.1)      # 低置信
    lo_pos = k.correct(500, 100, confidence=0.1)   # 低置信 → 更滞后
    assert abs(lo_pos[0] - 500) > abs(hi_pos[0] - 500)


# ── HybridBackend (依赖 vendored 仓库) ──


def test_hybrid_teleport_follow():
    """目标瞬移 (600px 级) 每帧跟随, 中心精确 (Kalman 瞬移 reset 直出观测)。"""
    repo = _vendored_repo_or_skip()
    hb = HybridBackend(repo, config={"teleport_dist": 120})
    f = np.zeros((540, 960, 3), dtype=np.uint8)
    for i, x in enumerate([200, 500, 300, 700, 100, 600]):
        fr = f.copy()
        fr[:] = (35, 35, 35)
        cv2.rectangle(fr, (x, 220), (x + 90, 310), (255, 255, 255), -1)
        r = hb.detect(fr)
        assert r.active, f"帧{i} 应激活"
        assert abs(r.target_center[0] - (x + 45)) <= 4, \
            f"帧{i} 中心偏: want {x+45}, got {r.target_center[0]}"


def test_hybrid_fade_longer_than_blob():
    """渐隐: 亮度逐帧降 → hybrid 残差层比纯白块多撑若干帧 (渐隐兜底)。"""
    repo = _vendored_repo_or_skip()
    hb = HybridBackend(repo, config={"min_conf": 0.15})
    blob_be = OpenCVBackend(repo)
    f = np.zeros((540, 960, 3), dtype=np.uint8)

    # 亮块预热 (blob 激活 + freeze 防吸收), 再逐帧渐隐
    seq = [255] * 8 + [220, 180, 150, 120, 100, 80, 60, 40, 30]
    hybrid_active = 0
    blob_active = 0
    for b in seq:
        fr = f.copy()
        fr[:] = (35, 35, 35)
        cv2.rectangle(fr, (200, 220), (290, 310), (b, b, b), -1)
        if hb.detect(fr).active:
            hybrid_active += 1
        if blob_be.detect(fr).active:
            blob_active += 1

    assert hybrid_active > blob_active, \
        f"hybrid 应比纯白块多撑: hybrid={hybrid_active}, blob={blob_active}"


def test_hybrid_01mp4_frames_40_199_regression():
    """真实素材回归: 帧 40~199 全程 active (与 opencv 基线 0 失败一致)。"""
    if not VIDEO_01.is_file():
        pytest.skip("tools/01.mp4 未落地")
    repo = _vendored_repo_or_skip()
    cap = cv2.VideoCapture(str(VIDEO_01))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    assert len(frames) > 200, f"视频帧数不足: {len(frames)}"

    hb = HybridBackend(repo)
    for i in range(40):
        hb.detect(frames[i])  # 预热背景模型

    inactive = [i for i in range(40, 200) if not hb.detect(frames[i]).active]
    assert inactive == [], f"帧 40~199 应全程 active, 实际失活帧: {inactive}"


# ── HybridBackend × SOT (UETrack) — mock 后端驱动锚点 re-init 逻辑 ──


class MockSOT:
    """模拟 UETrack SOT: ready 恒真, track 返回模板位置 (完美跟踪 / 可卡死)。

    init_template 记录调用次数 (锚点 re-init 触发判定靠它)。
    """

    def __init__(self):
        self.ready = True
        self.inits = 0
        self._pos = None   # 模板中心 (init 后 = 目标位置)

    def init_template(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        self._pos = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.inits += 1
        return True

    def track(self, frame):
        return (self._pos, 0.9)

    def stop(self):
        pass


def _frame_with_block(x, v=255, size=90, y=220):
    f = np.zeros((540, 960, 3), dtype=np.uint8)
    f[:] = (35, 35, 35)
    cv2.rectangle(f, (x, y), (x + size, y + size), (v, v, v), -1)
    return f


def _prime_sot_first_init(hb, sot, x=200):
    """峰值状态机: 亮块静止 3 帧 → 亮度降 2 帧 → 触发首次 init (inits==1)。"""
    for _ in range(3):
        hb.detect(_frame_with_block(x))
    hb.detect(_frame_with_block(x, v=220))
    hb.detect(_frame_with_block(x, v=200))
    assert hb.sot_inited, "峰值后下降应触发首次 init"
    assert sot.inits == 1
    return ((x + 45), (220 + 45))  # 块中心


def test_hybrid_sot_anchor_reinit_on_divergence():
    """目标瞬移: SOT 卡在旧锚点 + blob 强检测 → 锚点 re-init, 输出新锚点中心。"""
    repo = _vendored_repo_or_skip()
    sot = MockSOT()
    hb = HybridBackend(repo, config={"sot_reinit_dist": 60, "teleport_dist": 120}, uetrack=sot)
    _prime_sot_first_init(hb, sot, x=200)

    # 瞬移到新锚点 (600): SOT 还卡在 245 (mock 不跟), blob 权威 → re-init
    r = hb.detect(_frame_with_block(600))
    assert sot.inits == 2, f"SOT 卡死 + blob 强 → 应 re-init, inits={sot.inits}"
    assert abs(r.target_center[0] - 645) <= 6, \
        f"re-init 后应输出新锚点中心, got {r.target_center}"

    # 再跑一帧: 已对准, 不再 re-init
    r2 = hb.detect(_frame_with_block(600))
    assert sot.inits == 2, "目标未再动, 不应重复 re-init"


def test_hybrid_sot_no_reinit_when_agreeing():
    """目标原地不动: SOT 与 blob 一致 → 不触发 re-init (防 thrash)。"""
    repo = _vendored_repo_or_skip()
    sot = MockSOT()
    hb = HybridBackend(repo, config={"sot_reinit_dist": 60}, uetrack=sot)
    _prime_sot_first_init(hb, sot, x=200)

    for _ in range(5):
        r = hb.detect(_frame_with_block(200))
        assert abs(r.target_center[0] - 245) <= 6
    assert sot.inits == 1, "目标未动, SOT 不应 re-init"


def test_hybrid_sot_holds_center_during_fade():
    """渐隐期: blob 失效后 SOT 外观记忆兜底, 中心保持目标处 (re-init 不抢)。"""
    repo = _vendored_repo_or_skip()
    sot = MockSOT()
    hb = HybridBackend(repo, config={"min_conf": 0.15, "sot_reinit_dist": 60}, uetrack=sot)
    _prime_sot_first_init(hb, sot, x=200)

    active = 0
    for v in [120, 90, 60, 40, 30]:
        r = hb.detect(_frame_with_block(200, v=v))
        if r.active:
            active += 1
        if r.target_center is not None:
            assert abs(r.target_center[0] - 245) <= 8, \
                f"渐隐期中心应保持目标处, v={v} got {r.target_center}"
    assert active >= 2, f"SOT 应在渐隐期兜底 (active {active} 帧)"
    assert sot.inits == 1, "渐隐期 blob 弱, 不应触发 re-init"
