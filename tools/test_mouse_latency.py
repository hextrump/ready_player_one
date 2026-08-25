"""
鼠标延迟观察 — 播放视频, 逐帧送 hhh 远程服务, 鼠标跟随目标中心
================================================================
公平延迟测试: 每帧【先】同步送服务、拿到中心后【再】显示这一帧,
视频播放节奏跟随处理速度 → 画面里目标移动 到 光标跟上 的滞后
= 纯端到端延迟 (网络往返 + 服务端 opencv+samurai + JPEG 编解码)。

窗口左上角实时叠加 (看延迟不用盯控制台):
    LAT:   端到端往返 ms (单帧)
    PHASE: countdown/tracking/idle + conf
    TARGET: 服务端返回中心 (帧坐标)
    DELTA:  光标 vs 目标 屏幕距离 px

用法:
    python tools/test_mouse_latency.py data/detect/BV1XuySBvEFa.mp4
    python tools/test_mouse_latency.py <mp4> --host 100.118.47.94 --port 8600
    python tools/test_mouse_latency.py <mp4> --send-width 720   # 发送帧缩小 = 更快的网络传输
按键: Q 退出 / 关窗口退出
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import win32api
import win32gui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.lie_detector.remote_backend import RemoteBackend

WIN_NAME = "lie-mouse-latency"


def _find_hwnd() -> int:
    for _ in range(100):
        h = win32gui.FindWindow(None, WIN_NAME)
        if h:
            return h
        time.sleep(0.05)
    return 0


def _draw(frame: np.ndarray, lat_ms: float, r, delta: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # 目标十字
    if r.target_center is not None:
        cx, cy = r.target_center
        cv2.drawMarker(out, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    # 状态条
    lines = [
        f"LAT: {lat_ms:.0f}ms  (光标滞后 = 此值附近)",
        f"PHASE: {r.phase.value}  conf={r.confidence:.2f}  active={r.active}",
        f"TARGET: {r.target_center}  CURSOR-DELTA: {delta:.0f}px",
        f"Q 退出",
    ]
    y = 22
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="远程测谎鼠标延迟观察 (公平: 同步往返)")
    ap.add_argument("video", help="视频路径")
    ap.add_argument("--host", default="100.118.47.94")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--send-width", type=int, default=0,
                    help="发送帧宽度 (0=原始; 小=网络快但检测分辨率降; 建议 >=640)")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] 打不开视频: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rb = RemoteBackend(args.host, args.port, timeout=args.timeout, jpeg_quality=82)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    hwnd = _find_hwnd()
    if not hwnd:
        print("[ERROR] 窗口未创建")
        return 1
    ox, oy = win32gui.ClientToScreen(hwnd, (0, 0))
    print(f"窗口已创建 hwnd={hwnd} 原点=({ox},{oy})  视频 {args.video}")
    print(f"远程服务 {args.host}:{args.port}  send_width={args.send_width or '原'}  Q 退出")

    lats = []
    n_frame = 0
    n_active = 0
    t_loop = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环
                continue
            H, W = frame.shape[:2]
            if args.send_width and args.send_width != W:
                sw = args.send_width
                sh = int(H * sw / W)
                send_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
                sx = sw / W   # 帧坐标 → 发送帧坐标
                sy = sh / H
            else:
                send_frame = frame
                sx = sy = 1.0

            t0 = time.perf_counter()
            r = rb.update(send_frame)
            dt = (time.perf_counter() - t0) * 1000
            lats.append(dt)
            n_frame += 1
            if r.active:
                n_active += 1

            # 光标 → 屏幕坐标 (窗口 AUTOSIZE = 客户区 == 原始帧尺寸, scale=1)
            delta = 0.0
            if r.active and r.target_center is not None:
                cx, cy = r.target_center
                # 发送帧坐标 → 原始帧坐标
                fx, fy = cx / sx, cy / sy
                screen_x = ox + int(fx)
                screen_y = oy + int(fy)
                try:
                    cur = win32api.GetCursorPos()
                    delta = ((cur[0] - screen_x) ** 2 + (cur[1] - screen_y) ** 2) ** 0.5
                    win32api.SetCursorPos((screen_x, screen_y))
                except Exception:
                    pass

            disp = _draw(frame, dt, r, delta)
            cv2.imshow(WIN_NAME, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # 节奏: 至少 1/fps 秒/帧 (视频节奏跟随处理)
            time.sleep(max(0.0, 1.0 / fps - (time.time() - t_loop)))
            t_loop = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        rb.close()
        cv2.destroyAllWindows()

    if lats:
        print("=" * 64)
        print(f"帧数={n_frame}  active={n_active}  往返 avg={np.mean(lats):.0f}ms  "
              f"p50={np.percentile(lats, 50):.0f}  max={max(lats):.0f}")
        print("(avg 含网络+服务端; 服务端 samurai step 实测 ~70ms, 网络 ~75-120ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
