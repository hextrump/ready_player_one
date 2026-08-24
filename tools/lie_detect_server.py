"""lie_detect_server — hhh GPU 全远程测谎检测服务

在训练机 (hhh, GPU) 上跑; 本机 bot 每帧 POST /frame 发 JPEG, 收 {active, center, confidence}。
服务端做完整检测 (opencv 弹窗激活 + samurai 会话跟踪 + 去抖/超时), 复用仓库 src 逻辑。

用法:
    python tools/lie_detect_server.py --host 0.0.0.0 --port 8600 \
        --repo "C:/Users/heyas/Documents/code/lie-detector"
    python tools/lie_detect_server.py --spike   # 部署门禁: 合成帧 init/step 测 GPU 延迟, 退出

协议 (HTTP/1.1 keep-alive + JSON/base64 JPEG):
    GET  /health → {"status":"ok"|"building"|"error","model_ready":bool,"device":str,
                    "build_error":str|null,"session_active":bool,"track_count":int}
    POST /frame  {"image_b64":"<jpeg>"} → {"ok":true,"active":bool,"phase":"countdown"|"tracking",
                    "center":[cx,cy]|null,"confidence":float,"bbox":[x1,y1,x2,y2]|null}
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 锚点守卫: opencv 置信 ≥ 此值 且 与 samurai 中心分歧 > 此值(px) → 信 opencv
ANCHOR_GUARD_MIN_CONF = 0.5
ANCHOR_GUARD_DIST = 120


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
        r = self.opencv.detect(frame_bgr)   # LieDetectResult (opencv 权威: 激活 + bbox + phase)
        now = time.time()
        is_active = self._update_debounce(r.active, now)

        # 超时强制解除 (防 SAM2 卡住整个会话)
        if is_active:
            if self._activated_at == 0.0:
                self._activated_at = now
            elif self._timeout_sec > 0 and (now - self._activated_at) > self._timeout_sec:
                print(f"[server] 激活超时 ({self._timeout_sec:.0f}s), 强制解除会话", file=sys.stderr)
                self.samurai.stop()
                self._reset_debounce()
                return {"ok": True, "active": False, "phase": "idle", "center": None,
                        "confidence": 0.0, "bbox": None}
        else:
            self._activated_at = 0.0

        if not is_active:
            if self.samurai.session_active:
                self.samurai.stop()
            return {"ok": True, "active": False, "phase": "idle", "center": None,
                    "confidence": 0.0, "bbox": None}

        # 激活中: 维护 samurai 会话
        started_now = False
        if not self.samurai.session_active:
            # 首 TRACKING 帧 → 起会话 (bbox = opencv 已膨胀框); 倒计时阶段 opencv 权威
            if r.phase == "tracking" and r.target_bbox is not None:
                started_now = self.samurai.start(frame_bgr, tuple(int(v) for v in r.target_bbox))

        if self.samurai.session_active and not started_now:
            res = self.samurai.step(frame_bgr)
            center = r.target_center
            conf = r.confidence
            if res is not None:
                s_center, s_conf = res
                # 锚点守卫: opencv 强检测且分歧大 → 信 opencv
                if (
                    r.target_center is not None
                    and r.confidence >= ANCHOR_GUARD_MIN_CONF
                    and np.hypot(s_center[0] - r.target_center[0], s_center[1] - r.target_center[1]) > ANCHOR_GUARD_DIST
                ):
                    center = r.target_center
                    conf = r.confidence
                else:
                    center = s_center
                    conf = s_conf
            self._track_count += 1
        else:
            center = r.target_center
            conf = r.confidence

        return {
            "ok": True,
            "active": True,
            "phase": r.phase.value,
            "center": None if center is None else [int(center[0]), int(center[1])],
            "confidence": round(float(conf), 4),
            "bbox": None if r.target_bbox is None else [int(v) for v in r.target_bbox],
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
    print(f"[spike] step 平均 {avg_ms:.0f}ms/帧 ({'PASS <50ms' if avg_ms < 50 else 'FAIL >=50ms'})")
    state.samurai.stop()
    return 0 if avg_ms < 50 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="hhh GPU 全远程测谎检测服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--repo", default=None, help="lie-detector 项目路径 (默认 hhh 独立项目)")
    ap.add_argument("--model-size", default="base_plus")
    ap.add_argument("--image-size", type=int, default=None,
                    help="SAM2 输入边长 (默认配置 1024; 512 更快但精度降)")
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

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    print(f"[server] listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[server] 退出")
        if _STATE.samurai.session_active:
            _STATE.samurai.stop()


if __name__ == "__main__":
    main()
