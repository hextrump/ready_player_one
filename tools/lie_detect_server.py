"""lie_detect_server — hhh GPU 全远程测谎检测服务

在训练机 (hhh, GPU) 上跑; 本机 bot 每帧 POST /frame 发 JPEG, 收 {active, center, confidence}。
服务端做完整检测 (opencv 弹窗激活 + samurai 会话跟踪 + 去抖/超时), 复用仓库 src 逻辑。

用法:
    python tools/lie_detect_server.py --host 0.0.0.0 --port 8600 \
        --repo "C:/Users/heyas/Documents/code/lie-detector"
    python tools/lie_detect_server.py --spike   # 自测: 合成帧 init/step 测 GPU 延迟, 仅参考不拦截

协议 (HTTP/1.1 keep-alive + JSON/base64 JPEG):
    GET  /health → {"status":"ok"|"building"|"error","model_ready":bool,"device":str,
                    "build_error":str|null,"session_active":bool,"track_count":int}
    POST /frame  {"image_b64":"<jpeg>"} → {"ok":true,"active":bool,"phase":"countdown"|"tracking",
                    "center":[cx,cy]|null,"confidence":float,"bbox":[x1,y1,x2,y2]|null,
                    "s_bbox":[x1,y1,x2,y2]|null}   # s_bbox = SAMURAI mask 跟踪框 (无则画 bbox)
    POST /clear  → {"ok":true}   (强制结束会话, 释放 GPU)

会话流式: 测谎事件内一个 SAM2 会话。
- 倒计时阶段 (opencv 高置信静态) → opencv 权威, 不起会话。
- TRACKING 阶段 (目标移动) → 首 TRACKING 帧用 opencv bbox 起会话, 之后逐帧 step。
- opencv 强检测且与 samurai 中心分歧 > TELEPORT_DIST → 信 opencv (锚点守卫)。
- 去抖解除 / 30s 超时 → 会话 stop, 释放显存。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import http.server
import socket
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.lie_detector.state import LiePhase  # noqa: E402  事件阶段枚举 (服务端决策用)

# 锚点守卫: opencv 置信 ≥ 此值 且 与 samurai 中心分歧 > 此值(px) → 信 opencv
# 0.8 (不是 0.5): opencv 在 conf=0.5-0.7 (多阈值部分命中, 常见于弱/抖动检测) 时中心常漂移,
# 若此时抢 samurai 会让鼠标满屏跳。只有 opencv 很强 (≥0.8, 接近 1.0) 才允许拽回。
ANCHOR_GUARD_MIN_CONF = 0.8
ANCHOR_GUARD_DIST = 120

# opencv 检测降采样: 长边超过此值先把帧缩到该尺寸再检测 (连通域成本 ~O(像素数),
# 全帧 150-400ms → 缩放后 ~100ms)。绝对阈值由 opencv_backend 按 scale 同步缩放,
# 返回坐标已缩回全分辨率。
DETECT_MAX_SIDE = 320

# ── 事件生命周期 + 空间门 (决策逻辑, 治"跟歪") ──
# 空间门: 目标在连续可见段内帧间位移很小 (星形倒计时静止/跟踪缓移), 中心跳变过大 = 错选
# (倒计时数字等大亮块抢"最大白块"名额)。拒绝跳变 → 输出上帧已接受中心 (hold),
# 鼠标停在真目标上而非瞬移到错对象。
COUNTDOWN_GATE_DIST = 25   # 倒计时内星形静止, 中心跳 >25px = 错选 (距离门对置信无豁免)
TRACKING_GATE_DIST = 120   # 跟踪内允许星形较快移动, >120px 且低置信才拒
GATE_HIGH_CONF = 0.8       # 置信 ≥ 此值 → 跳变豁免 (信强检测)
# 事件边界: 连续 miss 满此帧数 = 事件结束/切换 (星形消失), 旧会话/旧锚作废
EVENT_END_MISSES = 2
# 重锚定门禁: 新事件首锚 (目标位置未知) 需最低置信; 同事件重锚需靠近上帧目标
REANCHOR_MIN_CONF = 0.35
REANCHOR_CLOSE_DIST = 100
# countdown 预热 samurai: 星形倒计时静止且较亮(conf≈0.5), 连续稳定帧后起会话,
# 让 tracking 一进入就有一致的 mask (直接 tracking 才初始化时弱星形 conf 0.25 易初始化失败)
COUNTDOWN_STABLE_FRAMES = 3
COUNTDOWN_ANCHOR_MIN_CONF = 0.35


class _ServerState:
    """服务端共享状态: opencv 检测 + samurai 流式 + 去抖/超时 (单会话)。"""

    def __init__(
        self,
        repo: str | Path,
        model_size: str,
        image_size: int | None = None,
        activate_after: int = 2,
        deactivate_after: int = 6,
        timeout_sec: float = 30.0,
    ):
        # 延迟 import src (先给 CUDA_VISIBLE_DEVICES 留窗口)
        from src.perception.lie_detector.opencv_backend import OpenCVBackend
        from src.perception.lie_detector.samurai_stream import SamuraiStream
        from src.perception.lie_detector.state import (
            _DebounceState,
        )

        self._repo = Path(repo)
        self.opencv = OpenCVBackend(self._repo)
        self.samurai = SamuraiStream(self._repo, model_size=model_size, image_size=image_size)
        self._debounce = _DebounceState(
            activate_after=activate_after,
            deactivate_after=deactivate_after,
        )
        self._timeout_sec = timeout_sec
        self._activated_at = 0.0
        self._lock = threading.Lock()
        self._track_count = 0
        self._last_center: Optional[Tuple[int, int]] = None   # 最近一次已接受的目标中心 (空间门/hold)
        self._last_phase: Optional[LiePhase] = None           # 上一帧阶段 (countdown 首帧 = 新事件)
        self._countdown_stable = 0                            # countdown 连续稳定帧计数 (预热 samurai 用)
        self._seen_tracking = False                           # 本事件内是否见过 TRACKING (相位单调性, P2)
        self._device: Optional[str] = None

        if not self.opencv.ready:
            print(f"[server] 警告: OpenCV 后端不可用: {self.opencv.import_error}", file=sys.stderr)
        if not self.samurai.ready:
            print(f"[server] 警告: SAMURAI 后端不可用: {self.samurai.import_error}", file=sys.stderr)

    # ── 后台预热 (build predictor ~10-20s, 不阻塞 serve) ──

    def warm_async(self) -> None:
        def _warm():
            try:
                ok = self.samurai.warm()
                if ok:
                    import torch
                    self._device = str(torch.cuda.get_device_name(0))
                    print(f"[server] SAMURAI predictor 就绪, GPU: {self._device}")
                else:
                    print(f"[server] SAMURAI 预热失败: {self.samurai.build_error}", file=sys.stderr)
            except Exception as e:
                print(f"[server] 预热异常: {e}", file=sys.stderr)

        threading.Thread(target=_warm, daemon=True, name="samurai-warm").start()

    # ── HTTP 层直接调用 (加锁) ──

    def health(self) -> dict[str, Any]:
        status = "ok" if self.samurai.model_ready else ("error" if self.samurai.build_error else "building")
        return {
            "status": status,
            "model_ready": self.samurai.model_ready,
            "device": self._device,
            "build_error": (str(self.samurai.build_error) if self.samurai.build_error else None),
            "import_error": (str(self.samurai.import_error) if self.samurai.import_error else None),
            "session_active": self.samurai.session_active,
            "track_count": self._track_count,
            "opencv_ready": self.opencv.ready,
        }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self.samurai.stop()
            self._reset_debounce()
            return {"ok": True}

    def handle_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        with self._lock:
            return self._handle_frame_locked(frame_bgr)

    # ── 核心逐帧逻辑 ──

    def _handle_frame_locked(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        # opencv 降采样检测: 长边 > DETECT_MAX_SIDE → 缩放后检测 (阈值同步缩放, 结果缩回全分辨率)
        detect_frame = frame_bgr
        detect_scale = 1.0
        H, W = frame_bgr.shape[:2]
        max_side = max(H, W)
        if max_side > DETECT_MAX_SIDE:
            detect_scale = DETECT_MAX_SIDE / max_side
            detect_frame = cv2.resize(
                frame_bgr,
                (max(1, int(round(W * detect_scale))), max(1, int(round(H * detect_scale)))),
                interpolation=cv2.INTER_AREA,
            )
        r = self.opencv.detect(detect_frame, scale=detect_scale)   # LieDetectResult (opencv 权威)
        now = time.time()

        # 进入本帧前的连续 miss 数 (事件切换判定: 目标消失 2+ 帧 = 旧事件结束)
        miss_run = self._debounce.miss_streak
        prev_phase = self._last_phase

        # 目标丢失: 连续 miss 满 EVENT_END_MISSES → 立刻停会话 (防旧会话带旧锚点活到新事件)
        if not r.active and miss_run + 1 >= EVENT_END_MISSES:
            self._end_event("目标丢失")

        is_active = self._update_debounce(r.active, now)

        # 超时强制解除 (防 SAM2 卡住整个会话)
        if is_active:
            if self._activated_at == 0.0:
                self._activated_at = now
            elif self._timeout_sec > 0 and (now - self._activated_at) > self._timeout_sec:
                print(f"[server] 激活超时 ({self._timeout_sec:.0f}s), 强制解除会话", file=sys.stderr)
                self._end_event("超时")
                self._reset_debounce()
                return self._idle_response()
        else:
            self._activated_at = 0.0

        if not is_active:
            if self._in_event():
                self._end_event("解除")
            return self._idle_response()

        # ── 事件边界: 旧目标作废, 本帧重新锚定 ──
        # ① 连续 miss ≥ EVENT_END_MISSES 后恢复 = 事件切换 (新事件星形在新位置)
        # ② 进入 countdown = 新事件开头 (countdown 只在事件起始出现; 顺带掐掉旧会话串扰)
        new_event = False
        if r.active:
            if miss_run >= EVENT_END_MISSES:
                new_event = True
            elif r.phase == LiePhase.COUNTDOWN and prev_phase != LiePhase.COUNTDOWN:
                new_event = True
        if new_event:
            self._end_event("新事件")

        center = None
        conf = r.confidence
        s_bbox = None
        bbox_out = None
        accepted = False

        # diag: 每帧诊断 (客户端 remote_backend._parse_result 只读已知键, 未知键忽略, 安全)
        diag = {
            "oc": None if r.target_center is None else [int(r.target_center[0]), int(r.target_center[1])],
            "ob": None if r.target_bbox is None else [int(v) for v in r.target_bbox],
            "matched": int(round(r.confidence * 4)),
            "phase_raw": r.phase.value,
            "new_event": new_event,
            "miss_run": miss_run,
            "anchor": None if self._last_center is None else list(self._last_center),
            "branch": None,
            "rejected": False,
            "sam_center": None,
            "sam_conf": None,
            "anchor_guard": False,
            "cd_stable": self._countdown_stable,
            "seen_tracking": self._seen_tracking,
        }

        if r.phase == LiePhase.COUNTDOWN:
            # 倒计时: 星形静止, opencv 权威, 强空间门 (距离门对置信无豁免 — 星形不会跳)
            center, accepted = self._gate_center(r.target_center, r.confidence, COUNTDOWN_GATE_DIST,
                                                 conf_bypass=False)
            bbox_out = r.target_bbox if accepted else None
            diag["branch"] = "countdown"
            # 预热 samurai: countdown 星形静止且较亮, 连续稳定后起会话 (tracking 进入即有 mask)
            if not self.samurai.session_active and accepted and r.target_bbox is not None:
                self._countdown_stable += 1
                if (self._countdown_stable >= COUNTDOWN_STABLE_FRAMES
                        and r.confidence >= COUNTDOWN_ANCHOR_MIN_CONF):
                    try:
                        self.samurai.start(frame_bgr, tuple(int(v) for v in r.target_bbox))
                        print(f"[server] countdown 预热会话 @ {center} conf={r.confidence:.2f}",
                              file=sys.stderr)
                    except Exception as e:
                        print(f"[server] countdown 预热失败: {e}", file=sys.stderr)
            if self.samurai.session_active:
                self._countdown_stable = 0   # 已起会话, 停止计数
        elif r.phase == LiePhase.TRACKING:
            started = False
            if not self.samurai.session_active:
                # 起会话: 重锚定门禁 (新事件低置信可起; 同事件需靠近上帧目标防错选)
                if r.target_bbox is not None and self._anchoring_ok(r, new_event):
                    started = self.samurai.start(frame_bgr, tuple(int(v) for v in r.target_bbox))
            if started:
                diag["branch"] = "tracking_start"
                center, accepted = self._gate_center(r.target_center, r.confidence, TRACKING_GATE_DIST)
                conf = r.confidence
                bbox_out = r.target_bbox if accepted else None
            elif self.samurai.session_active:
                res = self.samurai.step(frame_bgr)
                if res is not None:
                    s_center, s_conf, s_bbox = res
                    diag["sam_center"] = [int(s_center[0]), int(s_center[1])]
                    diag["sam_conf"] = round(float(s_conf), 4)
                    # 锚点守卫: opencv 强检测且分歧大 → 信 opencv (并清掉 samurai 框)
                    guard = (
                        r.target_center is not None
                        and r.confidence >= ANCHOR_GUARD_MIN_CONF
                        and np.hypot(s_center[0] - r.target_center[0], s_center[1] - r.target_center[1]) > ANCHOR_GUARD_DIST
                    )
                    diag["anchor_guard"] = guard
                    diag["branch"] = "samurai_step"
                    if guard:
                        center, accepted = self._gate_center(r.target_center, r.confidence, TRACKING_GATE_DIST)
                        conf = r.confidence
                    else:
                        center, accepted = self._gate_center(s_center, s_conf, TRACKING_GATE_DIST)
                        conf = s_conf
                else:
                    # step 失败 (目标消失) → 回退 opencv (过门)
                    diag["branch"] = "samurai_fail"
                    center, accepted = self._gate_center(r.target_center, r.confidence, TRACKING_GATE_DIST)
                if not accepted:
                    s_bbox = None
                bbox_out = r.target_bbox if accepted else None
                self._track_count += 1
            else:
                # 会话未起 (门禁拦下) → opencv 输出 (过门)
                diag["branch"] = "opencv_nosession"
                center, accepted = self._gate_center(r.target_center, r.confidence, TRACKING_GATE_DIST)
                bbox_out = r.target_bbox if accepted else None
        else:
            # 激活中但 opencv miss (去抖未解除) → 无目标, 客户端 hold
            diag["branch"] = "hold_none"
            center, accepted = None, False

        diag["rejected"] = not accepted
        if r.phase == LiePhase.TRACKING and r.active:
            self._seen_tracking = True
        self._last_phase = r.phase

        return {
            "ok": True,
            "active": True,
            "phase": r.phase.value,
            "center": None if center is None else [int(center[0]), int(center[1])],
            "confidence": round(float(conf), 4),
            "bbox": None if bbox_out is None else [int(v) for v in bbox_out],
            "s_bbox": None if s_bbox is None else [int(v) for v in s_bbox],
            "diag": diag,
        }

    def _update_debounce(self, candidate_active: bool, now: float) -> bool:
        if candidate_active:
            self._debounce.hit_streak += 1
            self._debounce.miss_streak = 0
            if not self._debounce.active and self._debounce.hit_streak >= self._debounce.activate_after:
                self._debounce.active = True
        else:
            self._debounce.miss_streak += 1
            self._debounce.hit_streak = 0
            if self._debounce.active and self._debounce.miss_streak >= self._debounce.deactivate_after:
                self._debounce.active = False
        return self._debounce.active

    def _reset_debounce(self) -> None:
        self._debounce.active = False
        self._debounce.hit_streak = 0
        self._debounce.miss_streak = 0
        self._activated_at = 0.0
        self._last_center = None
        self._last_phase = None
        self._countdown_stable = 0
        self._seen_tracking = False

    # ── 事件生命周期 + 空间门 helper ──

    @staticmethod
    def _idle_response() -> dict[str, Any]:
        return {"ok": True, "active": False, "phase": "idle", "center": None,
                "confidence": 0.0, "bbox": None, "s_bbox": None}

    def _in_event(self) -> bool:
        return self.samurai.session_active or self._last_center is not None

    def _end_event(self, reason: str = "") -> None:
        """事件结束/切换: 停会话, 清目标锚 (下个事件重新锚定)。"""
        stopped = self.samurai.session_active
        if stopped:
            self.samurai.stop()
        self._last_center = None
        self._last_phase = None
        self._countdown_stable = 0
        self._seen_tracking = False
        if stopped:
            print(f"[server] 会话结束 ({reason or '事件结束'})", file=sys.stderr)

    def _gate_center(self, candidate, conf: float, gate_dist: float,
                     conf_bypass: bool = True):
        """空间门: 候选中心相对已锚定目标跳变过大 → 判定错选, 返回 (hold_center, False)。

        返回 (输出中心, 是否接受)。接受 → 更新锚; 拒绝 → 输出上帧已接受中心 (hold)。
        - 无锚 → 接受 (首检即锚)。
        - conf_bypass=True: 置信 ≥ GATE_HIGH_CONF 的跳变豁免 (强检测可信)。
          False (countdown): 距离门对置信无豁免 — 倒计时内星形静止, 任何大跳都是错选。
        """
        if candidate is None:
            return (None, False)
        cand = (int(candidate[0]), int(candidate[1]))
        if self._last_center is None:
            self._last_center = cand
            return (self._last_center, True)
        d = np.hypot(cand[0] - self._last_center[0], cand[1] - self._last_center[1])
        if d > gate_dist and (not conf_bypass or conf < GATE_HIGH_CONF):
            return (self._last_center, False)   # 错选: hold 上帧中心
        self._last_center = cand
        return (self._last_center, True)

    def _anchoring_ok(self, r, new_event: bool) -> bool:
        """重锚定门禁: 能否用本帧 opencv 检测起/重起 samurai 会话 (防 conf=0.25 junk 锚错)。

        - 新事件 (目标位置未知): 只要求最低置信。
        - 同事件重锚 (会话中途失败/目标短暂丢失): 新检测须靠近上帧目标 (星形连续移动, 置信可低)。
        """
        if r.target_center is None:
            return False
        if self._last_center is None or new_event:
            return r.confidence >= REANCHOR_MIN_CONF
        d = np.hypot(r.target_center[0] - self._last_center[0], r.target_center[1] - self._last_center[1])
        return d <= REANCHOR_CLOSE_DIST


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """http.server + TCP_NODELAY (Windows 默认 delayed-ACK 40ms, Nagle 会让每请求慢 ~60ms)。

    客户端 http.client 已设 NODELAY; 服务端收的连接默认没设 → 补上, 削掉小请求 RTT。
    """
    daemon_threads = True

    def get_request(self):
        sock, addr = super().get_request()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock, addr


# 模块级共享状态 (handler 引用)
_STATE: Optional[_ServerState] = None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 静默访问日志 (避免刷屏); 错误走 stderr

    # ── GET ──

    def do_GET(self):  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(_STATE.health())
        else:
            self._send_json({"ok": False, "error": f"not found: {self.path}"}, code=404)

    # ── POST ──

    def do_POST(self):  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/frame":
            body = self._read_body()
            try:
                payload = json.loads(body)
                frame = self._decode_jpeg(payload.get("image_b64", ""))
            except Exception as e:
                self._send_json({"ok": False, "error": f"bad frame: {e}"}, code=400)
                return
            if frame is None:
                self._send_json({"ok": False, "error": "jpeg 解码失败"}, code=400)
                return
            self._send_json(_STATE.handle_frame(frame))
        elif path == "/clear":
            self._send_json(_STATE.clear())
        else:
            self._send_json({"ok": False, "error": f"not found: {self.path}"}, code=404)

    # ── helpers ──

    def _read_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    @staticmethod
    def _decode_jpeg(image_b64: str) -> Optional[np.ndarray]:
        if not image_b64:
            return None
        buf = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)


def _synthetic_frame(step: int = 0) -> np.ndarray:
    """暗底 + 白方块 (spike 自测用)。"""
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    x = 400 + (step * 6) % 120
    cv2.rectangle(frame, (x, 240), (x + 40, 280), (255, 255, 255), -1)
    return frame


def _spike(repo: str | Path, model_size: str, image_size: int | None = None) -> int:
    """部署门禁: 构建 predictor + 合成帧 init/step, 测 GPU 延迟。"""
    print(f"[spike] repo={repo} model={model_size} image_size={image_size or 'cfg'}")
    state = _ServerState(repo, model_size, image_size=image_size)
    print(f"[spike] opencv.ready={state.opencv.ready} samurai.ready={state.samurai.ready}")
    if not state.samurai.ready:
        print(f"[spike] 失败: {state.samurai.import_error}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    if not state.samurai.warm():
        print(f"[spike] warm 失败: {state.samurai.build_error}", file=sys.stderr)
        return 1
    build_s = time.perf_counter() - t0
    print(f"[spike] warm 耗时 {build_s:.1f}s")

    frame = _synthetic_frame(0)
    r = state.opencv.detect(frame)
    bbox = r.target_bbox if r.target_bbox is not None else (380, 220, 460, 300)
    t0 = time.perf_counter()
    ok = state.samurai.start(frame, bbox)
    start_s = time.perf_counter() - t0
    if not ok:
        print("[spike] start 失败", file=sys.stderr)
        return 1
    print(f"[spike] start {start_s * 1000:.0f}ms (bbox={bbox})")

    lat = []
    for i in range(1, 6):
        f = _synthetic_frame(i)
        t0 = time.perf_counter()
        res = state.samurai.step(f)
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        print(f"[spike] step{i}: {res} ({dt:.0f}ms)")
    avg_ms = float(np.mean(lat))  # lat 已是 ms
    size_label = str(image_size) if image_size else "cfg"
    print(f"[spike] step 平均 {avg_ms:.0f}ms/帧 @ {size_label} (仅参考不拦截; bot 帧预算 140-200ms)")
    state.samurai.stop()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="hhh GPU 全远程测谎检测服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--repo", default=None, help="lie-detector 项目路径 (默认 hhh 独立项目)")
    ap.add_argument("--model-size", default="base_plus")
    ap.add_argument("--image-size", type=int, default=512,
                    help="SAM2 输入边长 (默认 512, step ~68ms; 1024 精度更高但 ~151ms)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (import torch 前生效)")
    ap.add_argument("--activate-after", type=int, default=2)
    ap.add_argument("--deactivate-after", type=int, default=6)
    ap.add_argument("--timeout-sec", type=float, default=30.0)
    ap.add_argument("--spike", action="store_true", help="部署门禁: 合成帧自测后退出")
    args = ap.parse_args()

    # GPU 选择必须在任何 torch import 之前
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    repo = args.repo
    if repo is None:
        candidates = [
            Path(r"C:/Users/heyas/Documents/code/lie-detector"),
            PROJECT_ROOT / "models" / "lie_detector",
        ]
        repo = next((p for p in candidates if p.is_dir()), candidates[0])
    print(f"[server] repo={repo} (存在: {Path(repo).is_dir()})")

    if args.spike:
        sys.exit(_spike(repo, args.model_size, args.image_size))

    global _STATE
    _STATE = _ServerState(repo, args.model_size, image_size=args.image_size,
                          activate_after=args.activate_after,
                          deactivate_after=args.deactivate_after,
                          timeout_sec=args.timeout_sec)
    _STATE.warm_async()

    server = _Server((args.host, args.port), _Handler)
    print(f"[server] listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[server] 退出")
        if _STATE.samurai.session_active:
            _STATE.samurai.stop()


if __name__ == "__main__":
    main()
