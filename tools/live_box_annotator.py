import cv2
import json
import os
import time
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.window_capture import WindowCapture

class BoxStore:
    def __init__(self):
        self.boxes = [] # [{"cls": str, "x1": int, "y1": int, "x2": int, "y2": int}]
        self.history = []
        
    def add_box(self, cls_name, x1, y1, x2, y2):
        # 确保 x1 < x2, y1 < y2
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if right - left < 5 or bottom - top < 5:
            print("[WARN] 框太小，已忽略")
            return
            
        self.boxes.append({
            "cls": cls_name,
            "x1": left, "y1": top,
            "x2": right, "y2": bottom
        })
        self.history.append("add")
        print(f"[{cls_name}] Added Box: ({left},{top}) - ({right},{bottom})")

    def undo(self):
        if self.history:
            self.history.pop()
            self.boxes.pop()

    def clear(self):
        self.boxes.clear()
        self.history.clear()

def save_snapshot(frame_raw, frame_annotated, store, snap_dir="data/entity/snapshots"):
    os.makedirs(snap_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    import random
    suffix = "".join(random.choices("abcdef", k=4))
    prefix = f"{ts}_{suffix}"
    
    raw_path = os.path.join(snap_dir, f"{prefix}_raw.png")
    ann_path = os.path.join(snap_dir, f"{prefix}_annotated.png")
    json_path = os.path.join(snap_dir, f"{prefix}_labels.json")
    
    cv2.imwrite(raw_path, frame_raw)
    cv2.imwrite(ann_path, frame_annotated)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"boxes": store.boxes}, f, indent=2, ensure_ascii=False)
    
    print(f"[SNAPSHOT] 已保存截图: {prefix}.json ({len(store.boxes)} boxes)")

# ============ Global State ============
store = BoxStore()
mode = "Player"  # Player / Monster / HP / MP
mouse_pos = (0, 0)
drawing = False
start_pt = None

COLORS = {
    "Player": (255, 0, 0),      # Blue
    "Monster": (0, 0, 255),     # Red
    "HP": (0, 255, 255),        # Yellow 
    "MP": (255, 0, 255),        # Magenta
    "Toy Bear": (0, 255, 0)     # Green (New for Ludibrium)
}

def on_mouse(event, x, y, flags, param):
    global mouse_pos, drawing, start_pt, store, mode
    
    orig_w, orig_h = param
    # 获取真正的画面点击坐标 (处理窗口大小缩放)
    win_w, win_h = cv2.getWindowImageRect("Live Box Annotator")[2:]
    if win_w > 0 and win_h > 0:
        real_x = int(x * orig_w / win_w)
        real_y = int(y * orig_h / win_h)
    else:
        real_x, real_y = x, y
        
    mouse_pos = (real_x, real_y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "Delete":
            # 找到点击的最上面一个框并删除
            for i in reversed(range(len(store.boxes))):
                b = store.boxes[i]
                if b["x1"] <= real_x <= b["x2"] and b["y1"] <= real_y <= b["y2"]:
                    removed = store.boxes.pop(i)
                    print(f"[DELETE] 移除了 {removed['cls']} 框")
                    break
        else:
            drawing = True
            start_pt = (real_x, real_y)
            print(f"[{mode}] 开始画框: {start_pt}")
            
    elif event == cv2.EVENT_MOUSEMOVE:
        pass # just update mouse_pos
        
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            store.add_box(mode, start_pt[0], start_pt[1], real_x, real_y)
            start_pt = None
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if drawing:
            drawing = False
            start_pt = None
            print("[CANCEL] 取消画框")


def render_overlay(frame, store, mode, drawing, start_pt, mouse_pos):
    vis = frame.copy()
    
    for b in store.boxes:
        c = COLORS.get(b["cls"], (0,255,0))
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), c, 2)
        cv2.putText(vis, b["cls"], (b["x1"], b["y1"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        
    if drawing and start_pt:
        c = COLORS.get(mode, (0,255,0))
        cv2.rectangle(vis, start_pt, mouse_pos, c, 1)
        
    if mode != "Delete":
        cv2.drawMarker(vis, mouse_pos, COLORS.get(mode, (255,255,255)), cv2.MARKER_CROSS, 20, 1)
        
    # UI
    cv2.rectangle(vis, (0, 0), (1000, 35), (0,0,0), -1)
    
    # Mode indicator
    if mode == "Delete":
        cv2.putText(vis, f"MODE: DELETE (Click to remove)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(vis, f"MODE: {mode} (Click & Drag)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(mode, (255,255,255)), 2)
        
    stats = f"Boxes: {len(store.boxes)}"
    cv2.putText(vis, stats, (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    help_text = "1=Plyr 2=Mon 3=HP 4=MP 5=Bear | D=Del F=Snap Z=Undo C=Clear Q=Quit"
    cv2.putText(vis, help_text, (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return vis


def main():
    global mode
    wc = WindowCapture()
    print("Wait for window...")
    while wc.hwnd == 0 or getattr(wc, "hwnd", None) is None:
        wc.process_name = "msw.exe"
        wc.window_title = "MapleStory"
        if wc.find_window():
            break
            
        wc.process_name = "msw.exe"
        wc.window_title = None
        if wc.find_window():
            break
            
        wc.process_name = None
        wc.window_title = "MapleStory"
        if wc.find_window():
            break

        # Artale specific
        wc.process_name = None
        wc.window_title = "MapleStory Worlds-Artale (繁體中文版)"
        if wc.find_window():
            break

        print("Waiting for game window...")
        time.sleep(1)
        
    win_name = "Live Box Annotator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*60)
    print("  LIVE ENTITY ANNOTATOR (Bounding Boxes)")
    print("="*60)
    print("  [1] Player   [2] Monster")
    print("  [3] HP Bar   [4] MP Bar")
    print("  [5] Toy Bear (玩具熊)")
    print("  [D] Delete Mode")
    print("  鼠标左键: 拖拽画框 / Delete模式下点击删除")
    print("  鼠标右键: 取消当前框")
    print("  [F] 截图保存到 data/entity/snapshots/")
    print("  [Z] 撤销上一个  [C] 清除所有图距  [Q] 退出")
    print("="*60 + "\n")

    paused = False

    while True:
        if not paused:
            frame = wc.grab()
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            last_raw_frame = frame.copy()

        orig_h, orig_w = last_raw_frame.shape[:2]
        cv2.setMouseCallback(win_name, on_mouse, (orig_w, orig_h))
        
        vis = render_overlay(last_raw_frame, store, mode, drawing, start_pt, mouse_pos)
        
        if paused:
            cv2.putText(vis, "!!! PAUSED - Press SPACE to Resume !!!", (250, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(win_name, vis)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            print(f"[PAUSE] {'已暂停' if paused else '已恢复'}")
        elif key == ord('1'):
            mode = "Player"
            print(f"[MODE] → {mode}")
        elif key == ord('2'):
            mode = "Monster"
            print(f"[MODE] → {mode}")
        elif key == ord('3'):
            mode = "HP"
            print(f"[MODE] → {mode}")
        elif key == ord('4'):
            mode = "MP"
            print(f"[MODE] → {mode}")
        elif key == ord('5'):
            mode = "Toy Bear"
            print(f"[MODE] → {mode}")
        elif key == ord('d'):
            mode = "Delete"
            print(f"[MODE] → {mode}")
        elif key == ord('f'):
            if last_raw_frame is not None:
                vis_save = render_overlay(last_raw_frame, store, mode, False, None, (0,0))
                save_snapshot(last_raw_frame, vis_save, store)
        elif key == ord('z'):
            store.undo()
            print("[UNDO] 已撤销")
        elif key == ord('c'):
            store.clear()
            print("[CLEAR] 已清除标注")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
