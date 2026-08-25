"""量化本机↔hhh 网络: 基带 RTT + 不同大小帧的上传耗时。"""
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

HOST, PORT = "100.118.47.94", 8600


def jpeg_b64(frame: np.ndarray, q: int) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def post_frame(b64: str, label: str, repeat: int = 3) -> None:
    body = b'{"image_b64":"' + b64.encode("ascii") + b'"}'
    print(f"{label}: payload={len(body)/1024:.0f}KB")
    times = []
    import http.client
    for i in range(repeat):
        t0 = time.perf_counter()
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=15.0)
            c.request("POST", "/frame", body=body, headers={"Content-Type": "application/json"})
            resp = c.getresponse()
            n = len(resp.read())
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            print(f"   try{i}: {dt:.0f}ms (HTTP {resp.status}, {n}B)")
            c.close()
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            print(f"   try{i}: {dt:.0f}ms FAIL {type(e).__name__}: {e}")
    if times:
        print(f"   -> avg {sum(times)/len(times):.0f}ms")


def main() -> None:
    # 基带 RTT (GET /health, 空 body)
    import http.client
    print("== 基带 RTT: GET /health ==")
    for i in range(3):
        t0 = time.perf_counter()
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=10.0)
            c.request("GET", "/health")
            c.getresponse().read()
            print(f"  try{i}: {(time.perf_counter()-t0)*1000:.0f}ms")
            c.close()
        except Exception as e:
            print(f"  try{i}: FAIL {type(e).__name__}: {e}")

    # 真实帧: 原尺寸 720x1092 vs 缩小
    cap = cv2.VideoCapture("data/detect/BV1XuySBvEFa.mp4")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("视频读帧失败")
        return
    print(f"\n== 原帧 {frame.shape[1]}x{frame.shape[0]} ==")
    post_frame(jpeg_b64(frame, 85), "orig q85")
    print(f"\n== 半分辨率 {frame.shape[1]//2}x{frame.shape[0]//2} q80 ==")
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    post_frame(jpeg_b64(small, 80), "half q80")
    print(f"\n== 960x540 q80 (生产 letterbox 参考) ==")
    ref = cv2.resize(frame, (960, 540))
    post_frame(jpeg_b64(ref, 80), "960x540 q80")


if __name__ == "__main__":
    main()
