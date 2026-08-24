"""
lie_detector.remote_backend — 远程测谎检测客户端 (全远程)

bot 侧薄客户端: 每帧 JPEG → POST hhh 服务 /frame → 解析 JSON 构造 LieDetectResult。
检测 (opencv 弹窗 + samurai 会话跟踪 + 去抖/超时) 全部在服务端; 本机不做本地去抖。

用法 (由 LieDetectorModel backend=remote 路由):
    rb = RemoteBackend(host, port, timeout=1.0)
    r = rb.update(frame)          # → LieDetectResult
    rb.close()

失败降级: 任何网络异常/超时/非 200 → LieDetectResult(active=False) (检测不到不移鼠标)。
keep-alive: 持一个 HTTPConnection 跨请求复用; 断连重建一次再试。
可选 fallback: 传一个 callable(frame)->LieDetectResult (如本地 OpenCVBackend.detect),
远程失败时兜底 (config fallback=opencv 才启用, 默认关 — 遵"全远程")。
"""
from __future__ import annotations

import base64
import http.client
import json
import socket
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

from .state import LieBackend, LieDetectResult, LiePhase

log = get_logger("lie_detector.remote")


class RemoteBackend:
    """远程检测客户端: update(frame) → LieDetectResult。"""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 1.0,
        jpeg_quality: int = 85,
        fallback: Optional[Callable[[np.ndarray], LieDetectResult]] = None,
    ):
        self._host = host
        self._port = int(port)
        self._timeout = timeout
        self._jpeg_quality = jpeg_quality
        self._fallback = fallback
        self._conn = None
        self._cooldown_until = 0.0      # 断连冷却: 期间不发请求 (outage 不拖垮视觉线程)
        self._cooldown = 0.8            # 秒; 冷却窗口内不再尝试连接

    # ── 公共属性 ──

    @property
    def ready(self) -> bool:
        """配置合法即可用 (可达性由每帧 update 决定)。"""
        return True

    @property
    def import_error(self):
        return None

    # ── 主入口 ──

    def update(self, frame: np.ndarray) -> LieDetectResult:
        """每帧调用: 发帧 → 收 {active, center, confidence} → LieDetectResult。"""
        result = self._request_frame(frame)
        if result is not None:
            return result
        # 远程失败 → 可选本地兜底, 否则安全返回 inactive
        if self._fallback is not None:
            try:
                r = self._fallback(frame)
                if r.active:
                    return r
            except Exception as e:
                log.warning(f"[remote] 本地兜底失败: {e}")
        return LieDetectResult(active=False, backend=LieBackend.REMOTE)

    def health(self) -> dict:
        """GET /health → dict (服务端状态); 失败返回 {}。"""
        data = self._request("GET", "/health", None)
        return data if isinstance(data, dict) else {}

    def clear(self) -> bool:
        """强制结束服务端会话 (测谎意外退出后想立刻重进时由 model.reset 调)。"""
        data = self._request("POST", "/clear", None)
        return isinstance(data, dict) and data.get("ok", False)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── 内部: HTTP 往返 ──

    def _request_frame(self, frame: np.ndarray) -> Optional[LieDetectResult]:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            log.warning("[remote] JPEG 编码失败")
            return None
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        payload = json.dumps({"image_b64": image_b64}).encode("utf-8")
        data = self._request("POST", "/frame", payload)
        if not isinstance(data, dict):
            return None
        return self._parse_result(data)

    def _request(self, method: str, path: str, body: Optional[bytes]) -> Optional[dict]:
        """带 keep-alive 复用 + 断连重建一次。

        成本控制: 连接失败 (服务端不在) → 只付一次超时 + 进冷却, 期间直接返回 None
        (outage 不拖垮视觉线程 ~1s/帧); 已建立连接中途断开 → 重建一次重试。
        任何失败 → None。
        """
        now = time.time()
        if self._conn is None and now < self._cooldown_until:
            return None                       # 冷却中: 不盲打服务端
        for attempt in (0, 1):
            if self._conn is None:
                if not self._connect():
                    self._cooldown_until = time.time() + self._cooldown
                    return None               # 连不上 = 服务端不在, 重试无意义
            try:
                headers = {"Content-Type": "application/json"}
                self._conn.request(method, path, body=body, headers=headers)
                resp = self._conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    log.warning(f"[remote] {path} HTTP {resp.status}: {data[:200]!r}")
                    return None
                self._cooldown_until = 0.0    # 服务端可达, 清冷却
                try:
                    return json.loads(data)
                except json.JSONDecodeError as e:
                    log.warning(f"[remote] {path} JSON 解析失败: {e}")
                    return None
            except (ConnectionError, http.client.HTTPException, socket.error, TimeoutError, OSError) as e:
                log.warning(f"[remote] {path} 请求失败 (attempt {attempt}): {type(e).__name__}")
                self._drop_conn()
        return None

    def _connect(self) -> bool:
        try:
            c = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
            c.connect()                       # 显式建连: 连接失败在此抛出
            self._conn = c
            return True
        except Exception as e:
            log.warning(f"[remote] 连接失败 {self._host}:{self._port}: {e}")
            self._conn = None
            return False

    def _drop_conn(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # ── 内部: 结果映射 ──

    @staticmethod
    def _parse_result(data: dict) -> LieDetectResult:
        if not data.get("ok", False):
            return LieDetectResult(active=False, backend=LieBackend.REMOTE)
        active = bool(data.get("active", False))
        phase = RemoteBackend._parse_phase(data.get("phase"))
        center = RemoteBackend._to_tuple(data.get("center"))
        bbox = RemoteBackend._to_tuple(data.get("bbox"))
        return LieDetectResult(
            active=active,
            phase=phase,
            target_center=center,
            target_bbox=bbox,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            brightness=float(data.get("brightness", 0.0) or 0.0),
            backend=LieBackend.REMOTE,
        )

    @staticmethod
    def _parse_phase(s) -> LiePhase:
        try:
            return LiePhase(str(s))
        except ValueError:
            return LiePhase.IDLE

    @staticmethod
    def _to_tuple(v) -> Optional[Tuple[int, ...]]:
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            return None
        return tuple(int(x) for x in v)
