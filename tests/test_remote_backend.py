"""
测谎仪远程后端 (全远程) — HTTP 客户端回归测试。

用 stdlib ThreadingHTTPServer 起假服务 (随机端口, 零依赖):
覆盖 帧→JSON 往返 / active 透传 / keep-alive 复用 / 断连重连 /
超时 / 500 / 非 JSON → active=False 安全降级 / model.py backend=remote 路由。
不需要 GPU / 不需要真实 hhh。
"""
import json
import sys
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.lie_detector.model import LieDetectorModel
from src.perception.lie_detector.remote_backend import RemoteBackend
from src.perception.lie_detector.state import LieBackend, LiePhase


# ── 可配置假服务 ──


class FakeLieServer:
    """假 hhh 服务: 行为由 self.cfg 控制 (每测试可改), 记录收到的帧。"""

    def __init__(self):
        self.cfg = {"mode": "normal"}      # normal | inactive | error500 | hang
        self.drop_on_request = None        # 第 N 次请求不响应直接断开 (模拟断连)
        self.frames_seen = 0
        self.last_frame = None
        self._conn_ids = set()             # 每个连接一个 id → 统计 keep-alive 复用
        self._n_req = 0
        self._id_counter = 0

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _make_handler(self):
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n) if n > 0 else b""
                server._conn_ids.add(id(self.connection))
                server._n_req += 1
                if server.drop_on_request == server._n_req:
                    # 模拟断连: 读到请求但不回任何响应, 直接掐连接
                    self.close_connection = True
                    try:
                        self.connection.close()
                    except Exception:
                        pass
                    return
                path = self.path.rstrip("/")
                if path == "/clear":
                    self._reply(200, {"ok": True})
                    return
                if path != "/frame":
                    self._reply(404, {"ok": False, "error": "not found"})
                    return

                mode = server.cfg["mode"]
                if mode == "hang":
                    time.sleep(0.4)                       # 超过客户端 timeout → 触发超时
                payload = json.loads(body or b"{}")
                if "image_b64" in payload:
                    server.frames_seen += 1
                    buf = np.frombuffer(__import__("base64").b64decode(payload["image_b64"]), np.uint8)
                    server.last_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                if mode == "error500":
                    self._reply(500, {"ok": False, "error": "boom"})
                elif mode == "inactive":
                    self._reply(200, {"ok": True, "active": False, "phase": "idle",
                                      "center": None, "confidence": 0.0, "bbox": None})
                else:                                     # normal
                    self._reply(200, {"ok": True, "active": True, "phase": "tracking",
                                      "center": [320, 180], "confidence": 0.92,
                                      "bbox": [270, 130, 370, 230]})

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") == "/health":
                    self._reply(200, {"status": "ok", "model_ready": True, "device": "cuda:0"})
                else:
                    self._reply(404, {"ok": False, "error": "not found"})

            def _reply(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return H

    @property
    def requests(self):
        return self._n_req

    @property
    def distinct_conns(self):
        return len(self._conn_ids)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    srv = FakeLieServer()
    yield srv
    srv.close()


def _frame(box=(270, 130, 370, 230), size=(360, 640)) -> np.ndarray:
    fr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.rectangle(fr, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
    return fr


# ── 帧→JSON 往返 + active 透传 ──


def test_remote_update_roundtrip_active(fake):
    """normal: update 发帧 → 收 active=True, center/phase/conf 正确映射到 LieDetectResult。"""
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    r = rb.update(_frame())
    assert fake.frames_seen == 1
    assert r.active is True
    assert r.phase is LiePhase.TRACKING
    assert r.target_center == (320, 180)
    assert r.confidence == pytest.approx(0.92)
    assert r.target_bbox == (270, 130, 370, 230)
    assert r.backend is LieBackend.REMOTE
    rb.close()


def test_remote_update_inactive_passthrough(fake):
    """inactive: 服务端 active=False → 客户端透传 inactive, 不误移鼠标。"""
    fake.cfg["mode"] = "inactive"
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    r = rb.update(_frame())
    assert r.active is False
    assert r.target_center is None
    rb.close()


# ── keep-alive 复用 + 断连重连 ──


def test_remote_keepalive_reuses_connection(fake):
    """同一实例连续多帧 → 复用同一 TCP 连接 (连接数远小于请求数)。"""
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    for _ in range(5):
        assert rb.update(_frame()).active is True
    assert fake.requests >= 5
    assert fake.distinct_conns <= 2, \
        f"应复用连接: {fake.requests} 请求只用了 {fake.distinct_conns} 连接"
    rb.close()


def test_remote_reconnects_after_disconnect(fake):
    """服务端掐连接一次 → 客户端 attempt 0 失败后重建连接重试成功 (不崩不丢数据)。"""
    fake.drop_on_request = 2                       # 第 2 次请求 (已 keep-alive 复用) 直接断开
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    assert rb.update(_frame()).active is True      # 建立连接 (req 1)
    r2 = rb.update(_frame())                        # req 2 被掐 → 内部重建 → req 3 正常
    assert r2.active is True, "断连后应自动重连并成功"
    assert fake.requests >= 3
    # drop 先于帧计数 → frames_seen 只计"成功送达"的帧: req1 + req3(重传) = 2
    assert fake.frames_seen >= 2, "被掐的帧应自动重传且被服务端收到"
    rb.close()


# ── 故障降级: 超时 / 500 / 非 JSON ──


def test_remote_timeout_returns_inactive(fake):
    """服务端挂起超过 timeout → update 安全返回 active=False (不盲目移鼠标)。"""
    fake.cfg["mode"] = "hang"
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.1)
    t0 = time.perf_counter()
    r = rb.update(_frame())
    assert r.active is False
    assert time.perf_counter() - t0 < 1.0          # 没被卡死
    rb.close()


def test_remote_500_returns_inactive(fake):
    """服务端 500 → active=False 降级。"""
    fake.cfg["mode"] = "error500"
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    assert rb.update(_frame()).active is False
    rb.close()


def test_remote_unreachable_returns_inactive():
    """端口没服务 → 连接失败 → active=False (安全阀, 不崩)。"""
    rb = RemoteBackend("127.0.0.1", 1, timeout=0.1)  # 端口 1 无监听
    assert rb.update(_frame()).active is False
    rb.close()


def test_remote_outage_cooldown_skips_hammers():
    """服务端不在 (outage) → 首个连接失败进冷却, 之后每帧秒回 inactive (不拖垮视觉线程)。"""
    rb = RemoteBackend("127.0.0.1", 1, timeout=0.1)
    assert rb.update(_frame()).active is False     # 连接失败 → 进冷却
    assert rb._cooldown_until > time.time()
    t1 = time.perf_counter()
    for _ in range(5):                              # 冷却期间: 立即返回, 不重连
        assert rb.update(_frame()).active is False
    assert time.perf_counter() - t1 < 0.1, "冷却内应秒回, 不付多次连接超时"
    rb.close()


# ── health / clear ──


def test_remote_health_and_clear(fake):
    rb = RemoteBackend("127.0.0.1", fake.port, timeout=0.2)
    h = rb.health()
    assert h.get("model_ready") is True
    assert rb.clear() is True
    rb.close()


# ── model.py backend=remote 路由 (不崩 + 透传) ──


def test_model_remote_routes_to_remote_backend(fake):
    """LieDetectorModel(backend=remote) → update 走 RemoteBackend, 收到服务端结果。"""
    model = LieDetectorModel(
        "__nonexistent_repo__",                       # remote 不依赖本地仓库
        backend=LieBackend.REMOTE,
        config={"remote": {"host": "127.0.0.1", "port": fake.port, "timeout": 0.2}},
    )
    assert model.backend is LieBackend.REMOTE
    r = model.update(_frame())
    assert r.active is True
    assert r.target_center == (320, 180)
    model.reset()                                     # reset 调 remote.clear, 不崩
    assert fake.frames_seen >= 1
