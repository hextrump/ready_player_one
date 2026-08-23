"""
采集玩家名牌模板 + 角色中心偏移 (供 NametagLocator 使用)
========================================================
用法: python tools/capture_nametag.py --player player0_warrior

操作:
  1. 左键拖框: 框住名牌 (紧贴名牌板, 不含背景)
  2. 再单击:   标记角色中心 (身体/头部)
  [R] 重抓帧并重来   [C] 预览裁剪+偏移   [S] 保存   [Q] 退出

保存:
  data/player/<player>/nametag.png         (BGR 裁剪, 与 WindowCapture.grab() 通道一致)
  data/player/<player>/nametag_offset.json {"offset_x": int, "offset_y": int}
  offset = 角色中心 − 名牌左上角
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.window_capture import WindowCapture

# 名牌是**角色的身份凭证**, 必须存进玩家模板目录 —— 不该沉淀成一个全局文件
# 被下一个角色悄悄继承 (见 src/utils/player_profile._resolve_identity)。
# 这三个全局由 set_save_target 在 main() 开头设置 (--player 必填)。
SAVE_DIR = ""
TEMPLATE_PATH = ""
OFFSET_PATH = ""


def set_save_target(player: str) -> None:
    """把落盘目标切到玩家模板目录 (data/player/<player>/nametag.png)。"""
    global SAVE_DIR, TEMPLATE_PATH, OFFSET_PATH
    d = os.path.join(PROJECT_ROOT, "data", "player", player)
    if not os.path.isdir(d):
        print(f"[ERROR] 模板目录不存在: {d}")
        print(f"        可用模板: python tools/swap_player.py --list")
        sys.exit(1)
    SAVE_DIR = d
    TEMPLATE_PATH = os.path.join(d, "nametag.png")
    OFFSET_PATH = os.path.join(d, "nametag_offset.json")
    print(f"[OK] 名牌将保存到模板 {player}: {TEMPLATE_PATH}")

# ============ 交互状态 ============
state = {
    "frame": None,    # 最新一帧 BGR
    "rect": None,     # (x1, y1, x2, y2) 名牌框 (原图坐标)
    "center": None,   # (cx, cy) 角色中心 (原图坐标)
    "dragging": False,
    "start": None,
}


def on_mouse(event, x, y, flags, param):
    orig_w, orig_h = param
    # 处理窗口缩放: 换算回原图坐标
    win_w, win_h = cv2.getWindowImageRect("Capture Nametag")[2:]
    if win_w > 0 and win_h > 0:
        real_x = int(x * orig_w / win_w)
        real_y = int(y * orig_h / win_h)
    else:
        real_x, real_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        state["dragging"] = True
        state["start"] = (real_x, real_y)
        if state["rect"] is not None and state["center"] is None:
            # 已框好名牌后的第一次点击 = 标记角色中心
            state["center"] = (real_x, real_y)
            print(f"[CENTER] 角色中心: ({real_x},{real_y})  按 S 保存")
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["dragging"] and state["center"] is None:
            state["rect"] = (state["start"][0], state["start"][1], real_x, real_y)
    elif event == cv2.EVENT_LBUTTONUP:
        if state["dragging"] and state["center"] is None:
            state["dragging"] = False
            x1, y1, x2, y2 = state["start"][0], state["start"][1], real_x, real_y
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            if right - left < 5 or bottom - top < 5:
                print("[WARN] 名牌框太小, 请重新拖框")
                state["rect"] = None
            else:
                state["rect"] = (left, top, right, bottom)
                print(f"[NAMETAG] 名牌框: ({left},{top})-({right},{bottom})")
                print("  下一步: 单击角色中心 (身体/头部)")
        elif state["dragging"]:
            state["dragging"] = False


def save():
    if state["rect"] is None or state["center"] is None:
        print("[WARN] 需先框名牌 + 点角色中心")
        return
    x1, y1, x2, y2 = state["rect"]
    cx, cy = state["center"]
    os.makedirs(SAVE_DIR, exist_ok=True)

    crop = state["frame"][y1:y2, x1:x2]
    cv2.imwrite(TEMPLATE_PATH, crop)
    offset = {"offset_x": cx - x1, "offset_y": cy - y1}
    with open(OFFSET_PATH, "w", encoding="utf-8") as f:
        json.dump(offset, f, indent=2)

    print(f"[SAVE] 模板 -> {TEMPLATE_PATH} ({x2-x1}x{y2-y1})")
    print(f"[SAVE] offset -> {OFFSET_PATH} {offset}")
    # offset = 角色中心 − 名牌左上角。不同服的名牌位置可能不同(有的在头顶上, 有的在脚下),
    # 不再按符号告警, 只提示用户确认红叉确实在角色身上。
    if abs(offset["offset_y"]) > 3 * (y2 - y1):
        print("[WARN] |offset_y| 超过 3 倍名牌高度, 请确认角色中心点选正确")


def main():
    parser = argparse.ArgumentParser(description="采集玩家名牌模板 + 角色中心偏移")
    parser.add_argument("--process", default="Maplestory_Classic.exe", help="游戏进程名")
    parser.add_argument("--player", required=True,
                        help="保存到该玩家模板目录 (data/player/<player>/nametag.png)。")
    args = parser.parse_args()
    set_save_target(args.player)

    wc = WindowCapture(process_name=args.process)
    if not wc.find_window():
        print(f"[ERROR] 找不到进程 {args.process} 的窗口")
        sys.exit(1)
    print(f"[OK] 已找到窗口 HWND={wc.hwnd}")
    wc.bring_to_front()  # 保证窗口在渲染; PrintWindow 黑屏时 BitBlt 兜底需要它可见

    win_name = "Capture Nametag"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, np.zeros((720, 1280, 3), dtype=np.uint8))  # 先显示一次, 确保窗口已创建
    cv2.waitKey(1)
    mouse_callback_set = False

    print("=" * 60)
    print("  名牌模板采集 (用于玩家定位)")
    print("=" * 60)
    print("  1. 左键拖框: 框住名牌 (紧贴, 不含背景)")
    print("  2. 再单击:   标记角色中心 (身体/头部)")
    print("  [R] 重抓帧并重来   [C] 预览   [S] 保存   [Q] 退出")
    print("=" * 60)

    while True:
        frame = wc.grab()
        if frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        state["frame"] = frame
        orig_h, orig_w = frame.shape[:2]

        # 鼠标回调只需绑定一次 (必须等窗口创建后再绑, 否则 cvSetMouseCallback 会 NULL window 崩溃)
        if not mouse_callback_set:
            cv2.setMouseCallback(win_name, on_mouse, (orig_w, orig_h))
            mouse_callback_set = True

        vis = frame.copy()
        if frame.mean() < 2.0:
            cv2.putText(vis, "BLACK FRAME - 游戏窗口需在前台可见", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if state["rect"]:
            x1, y1, x2, y2 = state["rect"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "NAMETAG", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if state["center"]:
            cx, cy = state["center"]
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(vis, "CENTER", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if state["rect"] and state["center"]:
            cv2.line(vis, (state["rect"][0], state["rect"][1]), state["center"], (255, 255, 0), 1)

        # 顶部状态条
        cv2.rectangle(vis, (0, 0), (760, 30), (0, 0, 0), -1)
        if state["rect"] is None:
            status = "DRAG nametag box"
        elif state["center"] is None:
            status = "CLICK player center"
        else:
            status = "READY - press S to save"
        cv2.putText(vis, f"{status} | R re-grab  C preview  S save  Q quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        display = cv2.resize(vis, (1280, 720))
        cv2.imshow(win_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        # 用户手动关掉窗口时干净退出 (否则后续 setMouseCallback/imshow 会崩溃)
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('r'):
            state["rect"] = None
            state["center"] = None
            print("[RESET] 已清空, 重新框名牌")
        elif key == ord('c'):
            if state["rect"]:
                x1, y1, x2, y2 = state["rect"]
                crop = state["frame"][y1:y2, x1:x2].copy()
                cv2.imshow("Template Preview", crop)
                if state["center"]:
                    print(f"[PREVIEW] offset=({state['center'][0] - x1}, {state['center'][1] - y1})")
            else:
                print("[WARN] 先框名牌再预览")
        elif key == ord('s'):
            save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
