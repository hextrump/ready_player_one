"""
measure_input_ratio.py — 步骤 0: 量化"游戏到底吃没吃到按键"
============================================================

核心问题: bot 按下的 X, 游戏实际受理了几次?
本工具同时数两路信号, 对比 ratio = ATTACK / X:

  路 A (INPUT)   X 键按下次数
  路 B (ATTACK)  游戏实际攻击次数 = 画面里"伤害数字冒泡"的次数
                 伤害数字出现 = 攻击已被游戏受理并结算 (命中/Miss 都会冒字)

两种输入模式 (同一套伤害检测, ratio 直接可比):

  --mode manual   你手动打。用低层键盘钩子 (WH_KEYBOARD_LL) 数你物理按的 X,
                  且只在游戏窗口在前台时计数 (防误按)。
  --mode inject   bot 自己的按键路径。脚本用 GameController 以 bot 的节奏
                  (随机 0.45~0.85s 间隔, keybd_event) 注入 X, 数发送次数。

为什么这么测:
  - manual ratio ≈ 1.0 → 游戏本身吃得进正常键盘输入, 问题在 bot 的注入路径。
  - manual ratio < 1.0 → 游戏自己就在吞键 (连发限制/读 scancode 差异), 换注入方式也白搭。
  - inject ratio 明显低于 manual → AttachThreadInput / keybd_event 路径有问题, 再对症改。

用法:
  python scripts/measure_input_ratio.py --mode manual --duration 90
  python scripts/measure_input_ratio.py --mode inject  --duration 90
  # 窗口找不到时指定进程名:  --process Maplestory_Classic.exe
  # 不弹预览直接测 (用 --region 指定区域或默认中央横带):  --no-preview

流程:
  1. 找到游戏窗口, 弹出预览。
  2. 鼠标框选"伤害数字冒出的区域" (站在怪旁边攻击, 数字在怪上方/角色上方冒出)。
     拖动左键框选 → 按 s 锁定并开始, 按 q 退出。
  3. manual: 切回游戏手动打怪; inject: 脚本自己打。
     测量中随时 Ctrl+Shift+Q 提前结束。
  4. 结束后打印 ratio 摘要, 写 logs/input_ratio_<mode>_<时间戳>.csv (逐事件时间戳)。
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import win32gui

from src.capture.window_capture import WindowCapture
from src.brain.game_controller import GameController
from src.utils.logger import get_logger

log = get_logger("measure_input_ratio")

# ── Win32 常量 ──
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HC_ACTION = 0
VK_X = 0x58
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004

# 伤害数字检测的显示缩放 (预览窗口固定这个宽度, 鼠标坐标按比例换算回原图)
PREVIEW_W = 1280
PREVIEW_H = 720


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.c_uint32),
        ("scanCode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


LowLevelKeyboardProc = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(KBDLLHOOKSTRUCT))


class KeyboardObserver:
    """
    单个消息泵线程, 兼两件事:
      1. 低层键盘钩子 (WH_KEYBOARD_LL): 统计指定 VK 的按键 (可要求游戏窗口在前台)。
      2. 全局热键 (RegisterHotKey): 测量中从任意窗口 Ctrl+Shift+Q 提前结束。
    """

    def __init__(self, hook_vk: int = VK_X, game_hwnd: int = 0,
                 on_key=None, hotkey_vk: int = ord("Q"), on_hotkey=None):
        self._hook_vk = hook_vk
        self._game_hwnd = game_hwnd
        self._on_key = on_key
        self._hotkey_vk = hotkey_vk
        self._on_hotkey = on_hotkey
        self._running = False
        self._thread = None
        self._hhook = None
        self._proc = LowLevelKeyboardProc(self._callback)

    def _callback(self, nCode, wParam, lParam):
        if nCode == HC_ACTION and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            k = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if k.vkCode == self._hook_vk:
                if (self._game_hwnd == 0
                        or win32gui.GetForegroundWindow() == self._game_hwnd):
                    if self._on_key:
                        self._on_key()
        return ctypes.windll.user32.CallNextHookEx(self._hhook, nCode, wParam, lParam)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self):
        # 钩子 + 热键必须在同一个持有消息泵的线程注册/使用
        self._hhook = ctypes.windll.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hhook:
            print("[HOOK] 低层键盘钩子安装失败")
            self._running = False
            return
        hotkey_id = 1
        hk_ok = bool(ctypes.windll.user32.RegisterHotKey(
            None, hotkey_id, MOD_CONTROL | MOD_SHIFT, self._hotkey_vk))
        if not hk_ok:
            print("[HOOK] 全局热键 Ctrl+Shift+Q 注册失败 (已忽略, 可用 q 键退出)")

        msg = ctypes.wintypes.MSG()
        while self._running:
            r = ctypes.windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if r in (0, -1):
                break
            if msg.message == WM_HOTKEY and self._on_hotkey:
                self._on_hotkey()
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        if hk_ok:
            ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
        if self._hhook:
            ctypes.windll.user32.UnhookWindowsHookEx(self._hhook)
            self._hhook = None

    def stop(self):
        self._running = False
        if self._thread:
            ctypes.windll.user32.PostThreadMessageW(self._thread.ident, WM_QUIT, 0, 0)
            self._thread.join(timeout=1.0)


class DamageDetector:
    """
    在 ROI 里检测"新冒出的伤害数字" = 一次被游戏受理的攻击。

    原理: 伤害数字是白(普攻)/黄(暴击)的亮字, 带深色描边, 从怪物身上冒出后
    向上飘 ~0.7s。每帧:
      1. 阈值出亮字 mask (白 + 黄)。
      2. 上一帧 mask 膨胀后取"新增"像素 → 新出现的文字团。
      3. 文字团按邻近度聚簇 (同一位数字的多段连通域算一个数)。
      4. 冷却过滤: 同一位置 ~0.5s 内不重复计 (数字上飘/闪烁不重复计数)。
    """

    WHITE_MIN = 220
    YELLOW_MIN = (200, 140, 110, 50)   # r, g, b, r-g 差值

    def __init__(self, cooldown: float = 0.5, min_w: int = 8, min_h: int = 6):
        self._prev = None
        self._recent = []  # [(ts, cx, cy)] 已计数字, 用于冷却
        self._cooldown = cooldown
        self._min_w = min_w
        self._min_h = min_h

    @staticmethod
    def build_mask(roi: np.ndarray) -> np.ndarray:
        b = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        r = roi[..., 2].astype(np.int16)
        white = (r >= DamageDetector.WHITE_MIN) & (g >= DamageDetector.WHITE_MIN) & (b >= DamageDetector.WHITE_MIN)
        yellow = (r >= 200) & (g >= 140) & (b <= 110) & (r - g >= 50)
        mask = (white | yellow).astype(np.uint8) * 255
        # 去单像素噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return mask

    @staticmethod
    def _cluster(boxes) -> list:
        """把相邻的文字连通域聚成同一个数字 (多位数字的每个 digit 是一段连通域)。"""
        boxes = list(boxes)
        if not boxes:
            return []
        used = [False] * len(boxes)
        out = []
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            used[i] = True
            group = [a]
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(boxes):
                    if used[j]:
                        continue
                    for g in group:
                        if abs(g[0] - b[0]) <= 40 and abs(g[1] - b[1]) <= 16:
                            used[j] = True
                            group.append(b)
                            changed = True
                            break
            x0 = min(g[0] for g in group)
            y0 = min(g[1] for g in group)
            x1 = max(g[0] + g[2] for g in group)
            y1 = max(g[1] + g[3] for g in group)
            out.append((x0, y0, x1 - x0, y1 - y0))
        return out

    def detect(self, roi: np.ndarray, now: float) -> tuple[int, list]:
        """返回 (本次新冒出的伤害数字数, 每个数字的 bbox)。"""
        if roi.size == 0:
            return 0, []
        mask = self.build_mask(roi)
        if self._prev is None:
            self._prev = mask
            return 0, []
        grown = cv2.dilate(self._prev, np.ones((7, 7), np.uint8))
        new = cv2.bitwise_and(mask, cv2.bitwise_not(grown))
        cnts, _ = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= 15]
        self._prev = mask

        # 冷却过滤 (同一数字上飘/闪烁不重复计)
        self._recent = [e for e in self._recent if now - e[0] <= self._cooldown]
        events = []
        for (x, y, w, h) in self._cluster(boxes):
            if w < self._min_w or h < self._min_h:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            if any(abs(cx - e[1]) <= 35 and abs(cy - e[2]) <= 18 for e in self._recent):
                continue
            self._recent.append((now, cx, cy))
            events.append((x, y, w, h))
        return len(events), events


class Measure:
    def __init__(self, args):
        self.args = args
        self._running = False
        self._lock = threading.Lock()
        self._events = []  # (ts, event, detail)
        self.t_start = 0.0
        self.x_count = 0
        self.attack_count = 0
        self._last_boxes = []
        self._obs = None

        self.wc = WindowCapture(
            process_name=args.process, window_title=args.title, target_size=None)
        self.detector = DamageDetector()

    # ── 事件记录 (线程安全) ──
    def _record(self, event: str, detail: str = "", n: int = 1):
        with self._lock:
            self._events.append((time.time(), event, detail))
            if event in ("x_down", "x_sent"):
                self.x_count += n
            elif event == "attack":
                self.attack_count += n

    # ── 区域选择 ──
    def _default_region(self, frame):
        h, w = frame.shape[:2]
        return (int(w * 0.2), int(h * 0.25), int(w * 0.6), int(h * 0.35))

    def _select_region(self, frame):
        if self.args.region:
            x, y, w, h = map(int, self.args.region.split(","))
            return (x, y, w, h)
        if self.args.no_preview:
            return self._default_region(frame)

        scale_x = frame.shape[1] / PREVIEW_W
        scale_y = frame.shape[0] / PREVIEW_H
        state = {"drag": False, "p0": None, "p1": None}
        region = None

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag"] = True
                state["p0"] = (x, y)
                state["p1"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
                state["p1"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["drag"] = False
                state["p1"] = (x, y)

        cv2.namedWindow("Region Select", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Region Select", on_mouse)
        print("在预览上【拖动左键框选伤害数字冒出的区域】(站怪旁边攻击, 数字冒在怪/角色上方)")
        print("按 s 锁定并开始测量, 按 q 退出")
        while True:
            disp = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            if state["p0"] and state["p1"]:
                cv2.rectangle(disp, state["p0"], state["p1"], (0, 255, 0), 2)
            cv2.imshow("Region Select", disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                if state["p0"] and state["p1"]:
                    x0 = int(min(state["p0"][0], state["p1"][0]) * scale_x)
                    y0 = int(min(state["p0"][1], state["p1"][1]) * scale_y)
                    x1 = int(max(state["p0"][0], state["p1"][0]) * scale_x)
                    y1 = int(max(state["p0"][1], state["p1"][1]) * scale_y)
                    region = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                    break
                print("还没框选, 先拖动鼠标!")
            elif key in (ord("q"), 27):
                break
        cv2.destroyWindow("Region Select")
        if region is None:
            print("未框选, 使用默认中央区域")
            region = self._default_region(frame)
        return region

    def _draw_preview(self, frame, region, boxes):
        disp = frame.copy()
        x, y, w, h = region
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(disp, "ATTACK ZONE", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for bx, by, bw, bh in boxes:
            cv2.rectangle(disp, (x + bx, y + by), (x + bx + bw, y + by + bh),
                          (0, 0, 255), 2)
        ratio = self.attack_count / self.x_count if self.x_count else 0.0
        elapsed = time.time() - self.t_start
        txt = (f"X={self.x_count}  ATTACK={self.attack_count}  "
               f"ratio={ratio:.2f}  [{elapsed:.0f}s]  (q 退出)")
        cv2.putText(disp, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        disp = cv2.resize(disp, (PREVIEW_W, PREVIEW_H))
        return disp

    def _status(self):
        ratio = self.attack_count / self.x_count if self.x_count else 0.0
        elapsed = time.time() - self.t_start
        log.info(f"[{elapsed:6.1f}s] X={self.x_count:4d}  "
                 f"ATTACK={self.attack_count:4d}  ratio={ratio:.2f}")

    # ── 注入线程 (bot 真实按键路径) ──
    def _inject_loop(self):
        ctrl = GameController(hwnd=self.wc.hwnd)
        nxt = 0.0
        while self._running:
            now = time.time()
            if now >= nxt:
                # 复刻 bot burst 节奏: 随机 0.45~0.85s 间隔 + 20~55ms 按住时长
                ctrl.tap_key("x", post_action=False,
                             hold=random.uniform(0.020, 0.055))
                self._record("x_sent", "keybd_event")
                nxt = now + max(0.06, random.gauss(0.65, 0.20))
            time.sleep(0.005)

    def run(self) -> int:
        if not self.wc.find_window():
            print(f"[ERROR] 找不到窗口: process={self.args.process}")
            print("        如果游戏进程名不是默认值, 用 --process <进程名> 指定")
            return 1
        print(f"[OK] 窗口: hwnd=0x{self.wc.hwnd:08X}")

        frame0 = self.wc.grab()
        if frame0 is None or frame0.size == 0:
            print("[ERROR] 抓不到游戏画面 (窗口不可见?)")
            return 1
        region = self._select_region(frame0)
        print(f"[OK] 检测区域: x={region[0]} y={region[1]} w={region[2]} h={region[3]}")

        self.t_start = time.time()
        self._running = True
        self._record("mode", self.args.mode)
        self._record("region", ",".join(map(str, region)))

        inject_thread = None
        if self.args.mode == "manual":
            self._obs = KeyboardObserver(
                hook_vk=VK_X, game_hwnd=self.wc.hwnd,
                on_key=lambda: self._record("x_down", "manual X"),
                hotkey_vk=ord("Q"), on_hotkey=self._stop)
            self._obs.start()
            print("[OK] 手动模式: 切回游戏, 站在怪旁边按 X 打怪")
            print("     Ctrl+Shift+Q 提前结束; 按 q 键 (预览窗口) 也可结束")
        else:
            inject_thread = threading.Thread(target=self._inject_loop, daemon=True)
            inject_thread.start()
            print("[OK] 注入模式: 脚本按 bot 节奏注入 X (~1.5 下/秒)")
            print("     需要管理员运行 (AttachThreadInput); 非管理员时按键可能被吞")

        if not self.args.no_preview:
            cv2.namedWindow("Measure", cv2.WINDOW_NORMAL)

        last_status = 0.0
        black_streak = 0
        black_hint = False
        try:
            while self._running:
                frame = self.wc.grab()
                if frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                # 黑屏检测: PrintWindow 对 DX 游戏可能返黑 → 已回退 BitBlt, 但仍可能黑
                if float((frame.mean(axis=2) < 10).mean()) > 0.5:
                    black_streak += 1
                    if black_streak > 15 and not black_hint:
                        log.warning("连续多帧黑屏 — 游戏窗口被最小化/遮挡? 伤害数字看不到就测不准")
                        black_hint = True
                else:
                    black_streak = 0

                x, y, w, h = region
                roi = frame[y:y + h, x:x + w]
                now = time.time()
                n, boxes = self.detector.detect(roi, now)
                if n:
                    self._record("attack", f"+{n}", n=n)
                self._last_boxes = boxes

                if now - last_status >= 5:
                    last_status = now
                    self._status()

                if not self.args.no_preview:
                    disp = self._draw_preview(frame, region, boxes)
                    cv2.imshow("Measure", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self._running = False

                if now - self.t_start >= self.args.duration:
                    self._running = False

                time.sleep(0.03)  # ~15fps, 伤害数字上升速度下足够采样
        finally:
            self._running = False
            if self._obs:
                self._obs.stop()
            self._finish()
        return 0

    def _stop(self):
        self._running = False

    def _finish(self):
        with self._lock:
            events = sorted(self._events, key=lambda e: e[0])
            x, att = self.x_count, self.attack_count
        ratio = att / x if x else 0.0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = PROJECT_ROOT / "logs" / f"input_ratio_{self.args.mode}_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "elapsed_s", "event", "detail"])
            for t, ev, det in events:
                w.writerow([f"{t:.3f}", f"{t - self.t_start:.3f}", ev, det])

        print("\n" + "=" * 56)
        print(f"  模式      : {self.args.mode}")
        print(f"  时长      : {time.time() - self.t_start:.1f}s")
        print(f"  X 按下    : {x}")
        print(f"  ATTACK    : {att}")
        print(f"  ratio     : {ratio:.2f}   (= 游戏实际攻击 / 按键)")
        print("-" * 56)
        if x == 0:
            print("  没有采集到任何按键 — 检查前台窗口/模式是否选对")
        elif ratio >= 0.95:
            print("  判断: 按键几乎全部到达游戏。注入路径基本没问题,")
            print("        问题(如果有)更可能在方向键/移动, 而不是 X 攻击键。")
        elif ratio >= 0.85:
            print("  判断: 偶发丢键 (~15%)。可尝试加 KEYEVENTF_SCANCODE 或减少 attach 抖动。")
        else:
            print("  判断: 明显丢键。优先查 AttachThreadInput 抖动 / 按键节奏, 再考虑 SendInput。")
        print(f"  CSV       : {csv_path}")
        print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="输入到达率测量 (步骤0: 量化)")
    ap.add_argument("--mode", choices=["manual", "inject"], default="manual",
                    help="manual=你手动打; inject=脚本注入 bot 路径 (默认 manual)")
    ap.add_argument("--process", default="msw.exe",
                    help="游戏进程名 (默认 msw.exe; 找不到时试 Maplestory_Classic.exe)")
    ap.add_argument("--title", default=None, help="游戏窗口标题 (进程名找不到时的备选)")
    ap.add_argument("--region", default=None,
                    help="检测区域 'x,y,w,h' (默认拖框选择)")
    ap.add_argument("--duration", type=int, default=90, help="测量时长秒数 (默认 90)")
    ap.add_argument("--no-preview", action="store_true",
                    help="不弹预览窗口, 直接测 (用默认中央区域或 --region)")
    args = ap.parse_args()

    m = Measure(args)
    return m.run()


if __name__ == "__main__":
    sys.exit(main())
