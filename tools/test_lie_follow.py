"""
测谎仪鼠标跟随 — 实机测试工具 (自诊断版)
========================================
小窗口播放测谎视频 (或 --self-test 白方块), 用生产同款 (WindowCapture + LieDetectorModel + MouseTracker)
验证: 检测到白色图形 → 鼠标自动跟随到图形中心。

窗口左上角实时叠加状态 (不用看控制台也能判断卡在哪):
    ARMED:  ON/OFF       是否在监测
    DETECT: ACTIVE/WAIT  检测状态 + conf + target (绿色框 = 检测到的目标)
    GRAB:   ok WxH / none / fail  抓帧健康度 (ok=成功数 none=空帧 fail=异常)

自诊断对照:
    ARMED 是 ON 但 GRAB 一直 none/fail → 抓屏问题 (窗口被遮挡/最小化等)
    GRAB ok  但 DETECT 一直 WAIT      → 画面里没有够大的白色图形 (换视频/加大窗口)
    DETECT ACTIVE 但鼠标不动           → 看 [FOLLOW] 日志 target_screen vs cursor

用法:
    python tools/test_lie_follow.py path/to/lie.mp4 [--window-w 960] [--window-h 540]
    python tools/test_lie_follow.py --self-test [--window-w 960] [--window-h 540]
    python tools/test_lie_follow.py data/detect/BV1XuySBvEFa.mp4 --backend remote --auto-arm
        # 远程延迟测试: 播放视频 → 抓屏 → hhh 服务端检测 → 本机鼠标跟随

按键 (GetAsyncKeyState, 后台轮询):
    F9  武装测谎监测 / F 停止 / Q 退出
    --auto-arm   启动后直接武装, 不用按键 (推荐: 排除按键变量)
    --auto-quit N 跑 N 秒后自动退出
"""
import argparse
import ctypes
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import win32api
import win32gui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.brain.mouse_tracker import MouseTracker
from src.capture.window_capture import WindowCapture
from src.perception.lie_detector import LieBackend, LieDetectorModel
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("test_lie_follow")

WINDOW_NAME = "lie-detector-test"
VK_F9, VK_F, VK_Q = 0x78, 0x46, 0x51

# 播放线程 ↔ 跟随线程共享的显示状态 (GIL 下 dict 读写原子, 足够)
STATUS = {
    "armed": False,
    "detect": False,
    "conf": 0.0,
    "target": None,
    "bbox": None,            # 检测框 (letterbox 坐标), 画框用
    "s_bbox": None,          # SAMURAI mask 跟踪框 (有它优先画它, 绿框 = samurai 在追)
    "stale": False,          # remote 网络抖动返回的上一帧缓存
    "letterbox": (1.0, 0, 0),  # wc.last_letterbox
    "grab": "idle",
    "grab_ok": 0,
    "grab_none": 0,
    "grab_fail": 0,
}

# 每帧光标-目标屏幕距离 (px), 退出时输出准确度汇总
ACC_DIST: list = []


def _key_down(vk: int) -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return False


def _screen_of(center: Tuple[int, int], letterbox: Tuple[float, int, int],
               hwnd: int) -> Optional[Tuple[int, int]]:
    """letterbox 中心 → 屏幕坐标 (诊断 [FOLLOW] 日志用)。"""
    cx, cy = center
    scale, pl, pt = letterbox
    if scale <= 0:
        return None
    cxx = (cx - pl) / scale
    cyy = (cy - pt) / scale
    try:
        ox, oy = win32gui.ClientToScreen(hwnd, (0, 0))
        return (int(ox + cxx), int(oy + cyy))
    except Exception:
        return None


def _draw_overlay(frame: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """窗口左上角叠加状态条 + 检测绿框 (半透明黑底 + 黄字, 任何画面下可读)。"""
    h, w = frame.shape[:2]

    # 跟踪框: 优先 SAMURAI mask 框 (绿, = 昨天视频里的跟踪), 否则 opencv 检测框 (黄)
    # letterbox 坐标 → 客户区 → 窗口显示坐标 (imshow 缩放比例还原)
    lb = STATUS.get("letterbox")
    box = STATUS.get("s_bbox") or STATUS.get("bbox")
    if box and lb and lb[0] > 0:
        scale, pl, pt = lb
        show_scale = min(win_w / w, win_h / h) if (w and h) else 1.0
        x1 = int((box[0] - pl) / scale / show_scale)
        y1 = int((box[1] - pt) / scale / show_scale)
        x2 = int((box[2] - pl) / scale / show_scale)
        y2 = int((box[3] - pt) / scale / show_scale)
        if STATUS.get("s_bbox") is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)     # samurai 绿框
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)   # opencv 黄框
    # 目标中心十字 (只画 samurai 在追时)
    target = STATUS.get("target")
    if STATUS.get("s_bbox") is not None and target and lb and lb[0] > 0:
        scale, pl, pt = lb
        show_scale = min(win_w / w, win_h / h) if (w and h) else 1.0
        cx = int((target[0] - pl) / scale / show_scale)
        cy = int((target[1] - pt) / scale / show_scale)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

    lines = [
        f"ARMED: {'ON ' if STATUS['armed'] else 'OFF'}   (F9 武装 / F 停 / Q 退)",
        f"DETECT: {'ACTIVE' if STATUS['detect'] else 'WAIT'}" + (
            f"  conf={STATUS['conf']:.2f} target={STATUS['target']}"
            + ("  [STALE 保活]" if STATUS.get('stale') else "")
            if STATUS['detect'] else ""
        ),
        f"GRAB: {STATUS['grab']}  (ok={STATUS['grab_ok']} none={STATUS['grab_none']} fail={STATUS['grab_fail']})",
    ]
    bar_h = min(88, max(0, h))
    if bar_h > 0:
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.5, bar, 0.5, 0)
    y = 22
    for line in lines:
        if y > h:
            break
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return frame


def play_video_loop(video_path: str, win_w: int, win_h: int, quit_evt: threading.Event) -> None:
    """循环播放测谎视频 (imshow + waitKey 必须同一线程)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"打不开视频: {video_path}")
        quit_evt.set()
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000.0 / fps))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    log.info(f"开始播放 {video_path} (窗口 '{WINDOW_NAME}')")

    while not quit_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环
            continue
        cv2.imshow(WINDOW_NAME, _draw_overlay(frame, win_w, win_h))
        k = cv2.waitKey(delay) & 0xFF
        if k == ord('q'):
            quit_evt.set()
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                quit_evt.set()  # 窗口被手动关闭
                break
        except cv2.error:
            quit_evt.set()
            break
    cap.release()
    cv2.destroyAllWindows()


def play_self_test(win_w: int, win_h: int, quit_evt: threading.Event) -> None:
    """无视频自检: 暗底 + 左右来回的白色方块 → 验证 抓屏→检测→跟随 全链路。"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    sq, x, dx = 140, 60, 5
    mid = (win_h - sq) // 2
    while not quit_evt.is_set():
        frame[:] = (35, 35, 35)                      # 暗底
        cv2.rectangle(frame, (x, mid), (x + sq, mid + sq), (255, 255, 255), -1)
        x += dx
        if x <= 20 or x + sq >= win_w - 20:
            dx = -dx
        cv2.imshow(WINDOW_NAME, _draw_overlay(frame, win_w, win_h))
        k = cv2.waitKey(16) & 0xFF
        if k == ord('q'):
            quit_evt.set()
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                quit_evt.set()
                break
        except cv2.error:
            quit_evt.set()
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="测谎仪鼠标跟随实机测试 (自诊断)")
    parser.add_argument("video", nargs="?", default=None, help="测谎视频路径 (mp4); 不传则用 --self-test")
    parser.add_argument("--self-test", action="store_true", help="无视频自检: 窗口画移动白方块")
    parser.add_argument("--window-w", type=int, default=960, help="视频窗口宽 (默认 960)")
    parser.add_argument("--window-h", type=int, default=540, help="视频窗口高 (默认 540)")
    parser.add_argument("--backend", default=None,
                        help="检测后端: opencv|hybrid|remote (默认用 config.lie_detector.backend; "
                             "remote=全远程 hhh 服务端 opencv+samurai; "
                             "hybrid=本地 UETrack SOT + 背景差分 + Kalman 融合)")
    parser.add_argument("--auto-arm", action="store_true", help="启动后直接武装, 不用按 F9")
    parser.add_argument("--auto-quit", type=int, default=0, help="跑 N 秒后自动退出 (0=不自动)")
    args = parser.parse_args()
    if args.video is None and not args.self_test:
        print("[ERROR] 需要视频路径 或 --self-test")
        sys.exit(1)
    if args.video is not None and not Path(args.video).is_file():
        print(f"[ERROR] 视频不存在: {args.video}")
        sys.exit(1)

    quit_evt = threading.Event()
    if args.self_test:
        threading.Thread(
            target=play_self_test, args=(args.window_w, args.window_h, quit_evt),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=play_video_loop, args=(args.video, args.window_w, args.window_h, quit_evt),
            daemon=True,
        ).start()

    # 等视频窗口出现, 再按窗口标题精确捕获
    wc = WindowCapture(process_name="", window_title=WINDOW_NAME)
    for _ in range(50):
        if wc.find_window():
            break
        time.sleep(0.1)
    if not wc.is_valid:
        log.error(f"找不到视频窗口 '{WINDOW_NAME}' (播放线程可能没起来)")
        sys.exit(1)
    log.info(f"已捕获视频窗口 hwnd={wc.hwnd}, 客户区 {wc._width}x{wc._height}")

    # 生产同款测谎模型 (config 的 detector_repo_path + hybrid 子配置)
    ld_cfg = load_config().get("lie_detector", {})
    repo = ld_cfg.get("detector_repo_path") or str(PROJECT_ROOT / "models" / "lie_detector")
    backend = args.backend or ld_cfg.get("backend", "opencv")
    model = LieDetectorModel(
        repo,
        backend=backend,
        config={
            "activate_after_frames": ld_cfg.get("activate_after_frames", 2),
            "deactivate_after_frames": ld_cfg.get("deactivate_after_frames", 6),
            "timeout_sec": ld_cfg.get("timeout_sec", 30.0),
            "hybrid": ld_cfg.get("hybrid"),   # hybrid 子配置 (bg/kalman/fusion/uetrack)
            "remote": ld_cfg.get("remote"),   # remote 子配置 (host/port/timeout/jpeg_quality/fallback)
        },
    )
    is_remote = model.backend == LieBackend.REMOTE
    if not (is_remote or model.opencv_ready or model.hybrid_ready):
        log.error(f"测谎模型不可用: opencv={model.opencv_ready} hybrid={model.hybrid_ready}")
        sys.exit(1)
    if backend == "hybrid" and model.hybrid_ready:
        log.info("[LIE] backend=hybrid, UETrack 13s 构建已在后台预热线程执行 (首个弹窗不会阻塞)")
    if is_remote:
        rc = ld_cfg.get("remote", {})
        log.info(f"[LIE] backend=remote: {rc.get('host')}:{rc.get('port', 8600)} "
                 f"(每帧 JPEG→hhh opencv+samurai→收 active/center/conf)")

    tracker = MouseTracker(hwnd=wc.hwnd)
    tracker.start()

    mode = "self-test 白方块" if args.self_test else f"视频 {Path(args.video).name}"
    print("=" * 64)
    print(f"  测谎仪鼠标跟随测试  [{mode}]  窗口: '{WINDOW_NAME}'")
    print(f"  状态条: 自动武装={'开' if args.auto_arm else '按F9'}  窗口左上角有状态 + 检测绿框")
    print("=" * 64)

    # 初始武装 (auto-arm: 不用按键; 否则 F9)
    armed = bool(args.auto_arm)
    if armed:
        STATUS["armed"] = True
        log.info("已自动武装 (--auto-arm), 开始抓屏+检测+跟随; 按 F 解除")

    was_active = False
    prev_f9 = prev_f = False
    last_health_t = 0.0
    last_follow_t = 0.0
    t_start = time.time()
    try:
        while not quit_evt.is_set():
            if args.auto_quit and time.time() - t_start >= args.auto_quit:
                log.info(f"自动退出 (--auto-quit {args.auto_quit}s)")
                break

            # 热键 (F9/F) 后台轮询 (auto-arm 时 F9 仍可切, F 可停)
            f9, f = _key_down(VK_F9), _key_down(VK_F)
            if f9 and not prev_f9 and not armed:
                armed = True
                STATUS["armed"] = True
                STATUS["grab_ok"] = STATUS["grab_none"] = STATUS["grab_fail"] = 0
                log.info("F9: 武装测谎监测")
            if f and not prev_f:
                armed = False
                STATUS["armed"] = False
                if was_active:
                    model.reset()
                    tracker.clear_target()
                    was_active = False
                log.info("F: 停止监测, 跟随解除")
            prev_f9, prev_f = f9, f
            if _key_down(VK_Q):
                break
            if not armed:
                time.sleep(0.1)
                continue

            # ── 抓帧 (失败/空帧计数, 让问题可见) ──
            try:
                frame = wc.grab()
                if frame is None or frame.size == 0:
                    STATUS["grab_none"] += 1
                    STATUS["grab"] = "none"
                    time.sleep(0.05)
                    continue
            except Exception as e:
                STATUS["grab_fail"] += 1
                STATUS["grab"] = "fail"
                if time.time() - last_health_t > 2.0:
                    log.warning(f"grab 异常持续: {e}")
                    last_health_t = time.time()
                time.sleep(0.2)
                continue
            STATUS["grab_ok"] += 1
            STATUS["grab"] = f"ok {frame.shape[1]}x{frame.shape[0]}"

            # ── 检测 + 喂目标 ──
            res = model.update(frame)
            # None 守卫 (P4 冷锚门): 服务端在无锚 + conf<0.35/尺寸异常时返回 center:null —
            # 客户端保持上帧目标不更新 (生产 combat_brain 已 None-safe; 这里防 target_center[0] 崩)
            if res.active and res.target_center is not None:
                scale, pl, pt = wc.last_letterbox
                tracker.update_target(
                    cx=res.target_center[0], cy=res.target_center[1],
                    confidence=res.confidence, brightness=res.brightness,
                    letterbox_scale=scale, letterbox_pad_left=pl,
                    letterbox_pad_top=pt, hwnd=wc.hwnd,
                )
                STATUS["detect"] = True
                STATUS["conf"] = res.confidence
                STATUS["target"] = res.target_center
                STATUS["bbox"] = res.target_bbox
                STATUS["s_bbox"] = res.samurai_bbox
                STATUS["stale"] = res.stale
                STATUS["letterbox"] = (scale, pl, pt)
                if not was_active:
                    log.info(f"[LIE] ACTIVE conf={res.confidence:.2f} "
                             f"phase={res.phase.value} target={res.target_center}")
                was_active = True

                # 每帧累计准确度 + 每 ~1s 光标收敛诊断 (target_screen vs cursor)
                ts = _screen_of(res.target_center, (scale, pl, pt), wc.hwnd)
                if ts:
                    cur = win32api.GetCursorPos()
                    dist = ((ts[0] - cur[0]) ** 2 + (ts[1] - cur[1]) ** 2) ** 0.5
                    ACC_DIST.append(dist)
                    if time.time() - last_follow_t > 1.0:
                        last_follow_t = time.time()
                        log.info(f"[FOLLOW] target_screen={ts} cursor={cur} dist={dist:.0f}")
            else:
                STATUS["detect"] = False
                STATUS["target"] = None
                STATUS["bbox"] = None
                STATUS["s_bbox"] = None
                STATUS["stale"] = False
                if was_active:
                    log.info("[LIE] CLEARED (目标消失)")
                was_active = False
                tracker.clear_target()

            # 每 ~2s 打印健康度, 定位卡在哪
            now = time.time()
            if now - last_health_t > 2.0:
                log.info(f"[健康度] armed={armed} grab_ok={STATUS['grab_ok']} "
                         f"none={STATUS['grab_none']} fail={STATUS['grab_fail']} "
                         f"detect={STATUS['detect']}")
                last_health_t = now

            time.sleep(0.05)  # ~20Hz 喂目标, 与生产视觉线程节奏一致
    finally:
        tracker.stop()
        quit_evt.set()  # 通知播放线程退出并销毁窗口
        # 不再在主线程 destroyAllWindows: 播放线程自己会销毁,
        # 主线程同时销毁会和 imshow/waitKey 线程竞态 → 挂死
        time.sleep(0.3)  # 给播放线程收尾
        if ACC_DIST:
            import statistics
            n = len(ACC_DIST)
            s = sorted(ACC_DIST)
            mean = statistics.mean(ACC_DIST)
            p50 = s[n // 2]
            p90 = s[min(n - 1, int(n * 0.9))]
            hit15 = sum(1 for d in ACC_DIST if d < 15) / n
            hit25 = sum(1 for d in ACC_DIST if d < 25) / n
            log.info(f"[ACCURACY] 样本={n} mean={mean:.1f}px p50={p50:.1f}px p90={p90:.1f}px "
                     f"max={s[-1]:.1f}px  命中(<15px)={hit15:.0%} (<25px)={hit25:.0%}  "
                     f"(星形 bbox 约 31x23px, dist<15px ≈ 光标在目标上)")
        log.info("已退出")


if __name__ == "__main__":
    main()
