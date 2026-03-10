"""
Live Terrain Annotator — 实时跑图标注工具

使用方法:
1. 运行此脚本，它会自动抓取游戏窗口画面
2. 在弹出的 OpenCV 窗口中用鼠标标注地形
3. 标注结果实时保存为 JSON 文件

操作说明:
  [P] 键 → 切换到平台模式 (Platform)  — 鼠标点两下画一条水平线
  [R] 键 → 切换到绳子模式 (Rope)     — 鼠标点两下画一条垂直线  
  [D] 键 → 切换到删除模式 (Delete)    — 鼠标点击删除最近的标注
  [S] 键 → 保存标注到 JSON 文件
  [L] 键 → 从 JSON 文件加载标注
  [C] 键 → 清除所有标注
  [Z] 键 → 撤销上一步
  [Q] 键 → 退出

  鼠标左键 → 放置标注点
  鼠标右键 → 取消当前正在画的标注
"""

import cv2
import numpy as np
import json
import time
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.window_capture import WindowCapture

# ============ 数据结构 ============

class AnnotationStore:
    def __init__(self):
        self.platforms = []   # [(x_left, y, x_right, y)]  水平线
        self.ropes = []       # [(x, y_top, x, y_bottom)]  垂直线
        self.history = []     # 撤销历史
    
    def add_platform(self, x1, y1, x2, y2):
        """添加平台（取两点的平均 Y）"""
        avg_y = (y1 + y2) // 2
        xl, xr = min(x1, x2), max(x1, x2)
        entry = {"type": "platform", "x_left": xl, "y": avg_y, "x_right": xr}
        self.platforms.append(entry)
        self.history.append(("add_platform", len(self.platforms) - 1))
    
    def add_rope(self, x1, y1, x2, y2):
        """添加绳子（取两点的平均 X）"""
        avg_x = (x1 + x2) // 2
        yt, yb = min(y1, y2), max(y1, y2)
        entry = {"type": "rope", "x": avg_x, "y_top": yt, "y_bottom": yb}
        self.ropes.append(entry)
        self.history.append(("add_rope", len(self.ropes) - 1))
    
    def delete_nearest(self, mx, my, threshold=30):
        """删除距离鼠标最近的标注"""
        best_dist = threshold
        best_type = None
        best_idx = None
        
        for i, p in enumerate(self.platforms):
            # 到水平线段的距离
            if p["x_left"] <= mx <= p["x_right"]:
                dist = abs(my - p["y"])
            else:
                d_left = np.sqrt((mx - p["x_left"])**2 + (my - p["y"])**2)
                d_right = np.sqrt((mx - p["x_right"])**2 + (my - p["y"])**2)
                dist = min(d_left, d_right)
            if dist < best_dist:
                best_dist = dist
                best_type = "platform"
                best_idx = i
        
        for i, r in enumerate(self.ropes):
            if r["y_top"] <= my <= r["y_bottom"]:
                dist = abs(mx - r["x"])
            else:
                d_top = np.sqrt((mx - r["x"])**2 + (my - r["y_top"])**2)
                d_bot = np.sqrt((mx - r["x"])**2 + (my - r["y_bottom"])**2)
                dist = min(d_top, d_bot)
            if dist < best_dist:
                best_dist = dist
                best_type = "rope"
                best_idx = i
        
        if best_type == "platform" and best_idx is not None:
            removed = self.platforms.pop(best_idx)
            self.history.append(("del_platform", removed))
            return True
        elif best_type == "rope" and best_idx is not None:
            removed = self.ropes.pop(best_idx)
            self.history.append(("del_rope", removed))
            return True
        return False
    
    def undo(self):
        """撤销上一步"""
        if not self.history:
            return
        action, data = self.history.pop()
        if action == "add_platform":
            self.platforms.pop(data)
        elif action == "add_rope":
            self.ropes.pop(data)
        elif action == "del_platform":
            self.platforms.append(data)
        elif action == "del_rope":
            self.ropes.append(data)
    
    def clear(self):
        self.platforms.clear()
        self.ropes.clear()
        self.history.clear()
    
    def save(self, path):
        data = {
            "platforms": self.platforms,
            "ropes": self.ropes
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {len(self.platforms)} platforms, {len(self.ropes)} ropes → {path}")
    
    def load(self, path):
        if not os.path.exists(path):
            print(f"[LOAD] File not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.platforms = data.get("platforms", [])
        self.ropes = data.get("ropes", [])
        self.history.clear()
        print(f"[LOADED] {len(self.platforms)} platforms, {len(self.ropes)} ropes ← {path}")


def save_snapshot(frame_raw, frame_annotated, store, snap_dir="data/terrain/snapshots"):
    """
    保存一组截图:
    - 原始游戏画面 (raw)
    - 带标注的画面 (annotated)
    - 标注JSON (labels)
    """
    os.makedirs(snap_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    idx = len([f for f in os.listdir(snap_dir) if f.endswith('_raw.png')])
    prefix = f"{ts}_{idx:04d}"
    
    raw_path = os.path.join(snap_dir, f"{prefix}_raw.png")
    ann_path = os.path.join(snap_dir, f"{prefix}_annotated.png")
    json_path = os.path.join(snap_dir, f"{prefix}_labels.json")
    
    cv2.imwrite(raw_path, frame_raw)
    cv2.imwrite(ann_path, frame_annotated)
    
    data = {"platforms": store.platforms, "ropes": store.ropes}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[SNAPSHOT] 已保存截图: {prefix}")
    print(f"  原图:   {raw_path}")
    print(f"  标注图: {ann_path}")
    print(f"  标签:   {json_path}")
    return prefix


# ============ 渲染引擎 ============

def render_overlay(frame, store, mode, pending_point, mouse_pos):
    """在游戏画面上渲染所有标注"""
    vis = frame.copy()
    overlay = frame.copy()
    
    PLAT_COLOR = (0, 0, 255)       # 红色
    ROPE_COLOR = (0, 255, 255)     # 黄色
    PENDING_COLOR = (0, 255, 0)    # 绿色（正在画的）
    
    # 画已有平台
    for i, p in enumerate(store.platforms):
        cv2.line(overlay, (p["x_left"], p["y"]), (p["x_right"], p["y"]), PLAT_COLOR, 6)
        cv2.putText(vis, f"P{i}", (p["x_left"], p["y"] - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, PLAT_COLOR, 1)
    
    # 画已有绳子
    for i, r in enumerate(store.ropes):
        cv2.line(overlay, (r["x"], r["y_top"]), (r["x"], r["y_bottom"]), ROPE_COLOR, 6)
        cv2.putText(vis, f"R{i}", (r["x"] - 10, r["y_top"] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ROPE_COLOR, 1)
    
    # 叠加半透明
    vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)
    
    # 画正在进行中的标注（第一个点已点下，等待第二个点）
    if pending_point is not None and mouse_pos is not None:
        px, py = pending_point
        mx, my = mouse_pos
        if mode == "platform":
            # 预览水平线（以两个点的平均 Y 画）
            avg_y = (py + my) // 2
            cv2.line(vis, (px, avg_y), (mx, avg_y), PENDING_COLOR, 3)
            cv2.circle(vis, (px, py), 5, PENDING_COLOR, -1)
        elif mode == "rope":
            # 预览垂直线
            avg_x = (px + mx) // 2
            cv2.line(vis, (avg_x, py), (avg_x, my), PENDING_COLOR, 3)
            cv2.circle(vis, (px, py), 5, PENDING_COLOR, -1)
    
    # 顶部 HUD
    mode_text = {"platform": "[P] PLATFORM MODE", "rope": "[R] ROPE MODE", "delete": "[D] DELETE MODE"}
    mode_color = {"platform": (0, 0, 255), "rope": (0, 255, 255), "delete": (0, 0, 200)}
    
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(vis, mode_text.get(mode, mode), (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color.get(mode, (255,255,255)), 2)
    
    stats = f"Platforms: {len(store.platforms)} | Ropes: {len(store.ropes)}"
    cv2.putText(vis, stats, (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    help_text = "P/R/D=Mode  F=Screenshot  S=Save  Z=Undo  C=Clear  Q=Quit"
    cv2.putText(vis, help_text, (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    
    return vis


# ============ 主程序 ============

def main():
    save_path = "data/terrain/current_map.json"
    os.makedirs("data/terrain", exist_ok=True)
    
    # 初始化游戏窗口抓取 — 按标题查找
    wc = WindowCapture(process_name=None, window_title="MapleStory")
    
    # 先按默认方式找，找不到就用标题关键词遍历
    if not wc.find_window():
        import win32gui
        def find_maple(hwnd, results):
            title = win32gui.GetWindowText(hwnd)
            if "MapleStory" in title and win32gui.IsWindowVisible(hwnd):
                results.append(hwnd)
        results = []
        win32gui.EnumWindows(find_maple, results)
        if results:
            wc._hwnd = results[0]
            wc._update_size()
            title = win32gui.GetWindowText(wc._hwnd)
            print(f"通过标题匹配找到窗口: hwnd={wc._hwnd}, title='{title}'")
        else:
            print("未找到游戏窗口！请确保 MapleStory 正在运行。")
            return
    
    print(f"已锁定游戏窗口: {wc.hwnd}")
    
    store = AnnotationStore()
    
    # 自动加载已有标注
    if os.path.exists(save_path):
        store.load(save_path)
    
    mode = "platform"  # platform / rope / delete
    pending_point = None
    mouse_pos = (0, 0)
    last_raw_frame = None    # 保存最后一帧原始画面用于截图
    
    win_name = "Terrain Annotator - LIVE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    def on_mouse(event, x, y, flags, param):
        nonlocal pending_point, mouse_pos, mode
        
        # 将显示窗口坐标映射回原始图片坐标
        # (因为我们 resize 了显示窗口)
        if param is not None:
            orig_w, orig_h = param
            win_rect = cv2.getWindowImageRect(win_name)
            if win_rect[2] > 0 and win_rect[3] > 0:
                scale_x = orig_w / win_rect[2]
                scale_y = orig_h / win_rect[3]
                x = int(x * scale_x)
                y = int(y * scale_y)
        
        mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "delete":
                deleted = store.delete_nearest(x, y, threshold=40)
                if deleted:
                    print(f"[DELETE] 已删除 ({x}, {y}) 附近的标注")
            elif mode in ("platform", "rope"):
                if pending_point is None:
                    # 第一个点
                    pending_point = (x, y)
                    print(f"[{mode.upper()}] 第一个点: ({x}, {y}) — 点击第二个点完成")
                else:
                    # 第二个点 → 完成
                    px, py = pending_point
                    if mode == "platform":
                        store.add_platform(px, py, x, y)
                        avg_y = (py + y) // 2
                        print(f"[PLATFORM] 已添加: y={avg_y}, x=[{min(px,x)} ~ {max(px,x)}]")
                    elif mode == "rope":
                        store.add_rope(px, py, x, y)
                        avg_x = (px + x) // 2
                        print(f"[ROPE] 已添加: x={avg_x}, y=[{min(py,y)} ~ {max(py,y)}]")
                    pending_point = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键取消当前标注
            if pending_point is not None:
                pending_point = None
                print("[CANCEL] 已取消当前标注")
    
    print("\n" + "="*60)
    print("  LIVE TERRAIN ANNOTATOR")
    print("="*60)
    print("  [P] 平台模式  [R] 绳子模式  [D] 删除模式")
    print("  鼠标左键: 放标注点  鼠标右键: 取消")
    print("  [F] 截图保存  [S] 保存JSON  [Z] 撤销  [C] 清除  [Q] 退出")
    print("  截图自动保存到: data/terrain/snapshots/")
    print("="*60 + "\n")
    
    while True:
        frame = wc.grab()
        if frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        
        last_raw_frame = frame.copy()
        orig_h, orig_w = frame.shape[:2]
        
        # 设置鼠标回调（传入原始尺寸用于坐标映射）
        cv2.setMouseCallback(win_name, on_mouse, (orig_w, orig_h))
        
        # 渲染
        vis = render_overlay(frame, store, mode, pending_point, mouse_pos)
        
        # 显示
        cv2.imshow(win_name, vis)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            mode = "platform"
            pending_point = None
            print("[MODE] → 平台模式 (Platform)")
        elif key == ord('r'):
            mode = "rope"
            pending_point = None
            print("[MODE] → 绳子模式 (Rope)")
        elif key == ord('d'):
            mode = "delete"
            pending_point = None
            print("[MODE] → 删除模式 (Delete)")
        elif key == ord('f'):
            # 截图保存（原图 + 标注图 + JSON}
            if last_raw_frame is not None:
                vis_for_save = render_overlay(last_raw_frame, store, mode, None, None)
                save_snapshot(last_raw_frame, vis_for_save, store)
        elif key == ord('s'):
            store.save(save_path)
            # 同时也存一份截图
            if last_raw_frame is not None:
                vis_for_save = render_overlay(last_raw_frame, store, mode, None, None)
                save_snapshot(last_raw_frame, vis_for_save, store)
        elif key == ord('z'):
            store.undo()
            print("[UNDO] 已撤销")
        elif key == ord('c'):
            store.clear()
            print("[CLEAR] 已清除所有标注")
    
    # 退出前自动保存
    store.save(save_path)
    cv2.destroyAllWindows()
    print("标注工具已关闭。")


if __name__ == "__main__":
    main()
