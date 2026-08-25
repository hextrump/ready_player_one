"""端到端冒烟: 视频帧 → hhh 远程服务 → LieDetectResult (真网络, 真检测)。

按 ~7fps 节奏喂前 N 秒, 报告: 首次激活时间/相位/置信/中心, 以及每帧往返延迟。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.perception.lie_detector.remote_backend import RemoteBackend

HOST, PORT = "100.118.47.94", 8600
VIDEO = "data/detect/BV1XuySBvEFa.mp4"
SECONDS = 18


def main() -> int:
    rb = RemoteBackend(HOST, PORT, timeout=3.0, jpeg_quality=85)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"打不开 {VIDEO}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(fps * SECONDS)
    t_start = time.time()
    t_next = time.time()  # 下一帧目标时刻 (~7fps, 周期 0.14s)
    first_active = None
    tracking_seen = False
    lats = []
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = rb.update(frame)
        dt = (time.perf_counter() - t0) * 1000
        lats.append(dt)
        if r.active and first_active is None:
            first_active = (i / fps, r.phase.value, r.confidence, r.target_center)
        if r.active and r.phase.value == "tracking":
            tracking_seen = True
        if i % int(fps / 2) == 0 or (r.active and i % 3 == 0):
            print(f"[{i / fps:5.1f}s] f{i}: active={r.active} phase={r.phase.value} "
                  f"center={r.target_center} conf={r.confidence:.3f} ({dt:.0f}ms)")
        t_next += 0.14
        time.sleep(max(0.0, t_next - time.time()))
    cap.release()
    rb.close()
    dt = time.time() - t_start
    avg_ms = sum(lats) / len(lats) if lats else 0.0
    print("=" * 70)
    print(f"喂了 {len(lats)} 帧, 耗时 {dt:.1f}s (实际 {(len(lats) / dt):.1f}fps)")
    print(f"往返延迟: avg {avg_ms:.0f}ms  max {max(lats):.0f}ms")
    print(f"首次激活: {first_active}")
    print(f"samurai TRACKING: {'看到' if tracking_seen else '没看到'}")
    return 0 if first_active and tracking_seen else 1


if __name__ == "__main__":
    sys.exit(main())
