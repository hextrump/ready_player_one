
import os
import sys
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.capture.window_capture import WindowCapture
from src.utils.logger import get_logger

log = get_logger("capture_ludicity")

def main():
    # 查找游戏窗口
    # 根据刚才列出的窗口，标题包含 MapleStory Worlds-Artale
    target_title = "MapleStory Worlds-Artale"
    
    # 也可以尝试直接用进程名，通常是 msw.exe
    wc = WindowCapture(process_name="msw.exe", window_title=None)
    
    if not wc.find_window():
        log.warning(f"按进程名未找到，尝试按标题关键词查找...")
        # 这种方式需要更精确，但我们可以遍历一下
        import win32gui
        def find_ms(hwnd, ctx):
            title = win32gui.GetWindowText(hwnd)
            if target_title in title:
                ctx['hwnd'] = hwnd
        
        ctx = {'hwnd': 0}
        win32gui.EnumWindows(find_ms, ctx)
        if ctx['hwnd']:
            wc._hwnd = ctx['hwnd']
            wc._update_size()
            log.info(f"成功找到窗口: {win32gui.GetWindowText(wc._hwnd)}")
        else:
            log.error("未找到游戏窗口！请确保游戏已打开。")
            return

    # 截图
    try:
        frame = wc.grab()
        save_path = "data/debug/ludicity_raw.png"
        os.makedirs("data/debug", exist_ok=True)
        cv2.imwrite(save_path, frame)
        log.info(f"截图已保存至: {save_path}")
        print(f"SUCCESS:{save_path}")
    except Exception as e:
        log.error(f"截图失败: {e}")

if __name__ == "__main__":
    main()
