"""
游戏窗口后台截图工具 — 使用 PrintWindow API。

即使窗口被覆盖、最小化或在后台运行，也能完整截取。
核心: FindWindow → GetWindowDC → PrintWindow → 内存位图 → numpy

PrintWindow 比 BitBlt 更强:
- BitBlt: 只复制屏幕上可见的像素，被遮挡部分返回黑色
- PrintWindow: 发送 WM_PRINT 消息让窗口自己绘制，不依赖可见性
"""

from __future__ import annotations

import ctypes
import time
import threading
from typing import Optional

import cv2
import numpy as np

try:
    import win32gui
    import win32ui
    import win32con
    import win32process
    import win32api
except ImportError:
    raise ImportError("需要安装 pywin32: pip install pywin32")

from src.utils.logger import get_logger

log = get_logger("window_capture")

# === DPI Awareness ===
# Windows DPI 缩放会导致 GetClientRect 返回缩放后的值（如 1280x720）
# 而不是真实像素值（如 1600x900）。必须在调用任何窗口 API 之前设置。
try:
    # Windows 10 1703+ 最佳方案: Per-Monitor V2
    ctypes.windll.user32.SetProcessDpiAwarenessContext(
        ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    )
except Exception:
    try:
        # Win 8.1+ 备选
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            # Win Vista+ 最基础
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# PrintWindow flags
PW_CLIENTONLY = 0x1      # 仅客户区（不含标题栏/边框）
PW_RENDERFULLCONTENT = 0x2  # 强制完整渲染（Win 8.1+，对 DX 游戏更好）


class WindowCapture:
    """
    游戏窗口后台截图器 — PrintWindow 方案。

    支持:
    - 后台截图（窗口被遮挡也能截）
    - GPU 渲染游戏 (DirectX/OpenGL)
    - 小地图区域快速截取

    用法:
        wc = WindowCapture(process_name="msw.exe")
        wc.find_window()
        frame = wc.grab()             # 后台截图
        minimap = wc.grab_minimap()    # 小地图
    """

    def __init__(
        self,
        process_name: str = "msx.exe",
        window_title: str | None = None,
    ):
        """
        Args:
            process_name: 进程名（用于查找窗口）
            window_title: 窗口标题（备选查找方式）
        """
        self.process_name = process_name
        self.window_title = window_title
        self._hwnd: int = 0
        self._width: int = 0
        self._height: int = 0

        # 小地图区域 (相对于客户区，左上角)
        self._minimap_region: tuple[int, int, int, int] = (0, 0, 200, 150)
        self._lock = threading.Lock()

    def find_window(self) -> bool:
        """
        查找游戏窗口。优先按进程名，备选按窗口标题。

        Returns:
            是否找到窗口
        """
        self._hwnd = 0

        # 方式1: 按进程名查找
        if self.process_name:
            self._hwnd = self._find_by_process(self.process_name)

        # 方式2: 按窗口标题查找
        if not self._hwnd and self.window_title:
            self._hwnd = win32gui.FindWindow(None, self.window_title)

        if not self._hwnd:
            log.warning(f"未找到窗口: process={self.process_name}, title={self.window_title}")
            return False

        # 获取客户区尺寸
        self._update_size()

        title = win32gui.GetWindowText(self._hwnd)
        log.info(f"找到窗口: hwnd={self._hwnd}, title='{title}', "
                 f"size={self._width}x{self._height}")
        return True

    def _update_size(self) -> None:
        """更新客户区尺寸。"""
        rect = win32gui.GetClientRect(self._hwnd)
        self._width = rect[2] - rect[0]
        self._height = rect[3] - rect[1]

    def _find_by_process(self, process_name: str) -> int:
        """通过进程名查找主窗口句柄。"""
        result = [0]

        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                import psutil
                proc = psutil.Process(pid)
                if proc.name().lower() == process_name.lower():
                    result[0] = hwnd
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return True

        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception:
            pass
        return result[0]

    def grab(self) -> np.ndarray:
        """
        后台截取游戏窗口客户区 (PrintWindow)。

        即使窗口被其他窗口遮挡也能正常截取。

        Returns:
            BGR numpy 数组
        """
        with self._lock:
            if not self._hwnd or not win32gui.IsWindow(self._hwnd):
                raise RuntimeError("窗口句柄无效，请先调用 find_window()")

            # 刷新尺寸
            self._update_size()

            # 如果窗口最小化，先恢复
            if win32gui.IsIconic(self._hwnd):
                win32gui.ShowWindow(self._hwnd, win32con.SW_RESTORE)
                time.sleep(0.05)
                self._update_size()

            # --- PrintWindow 后台截图 ---
            # 1. 获取窗口 DC
            hwnd_dc = win32gui.GetWindowDC(self._hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)

            # 2. 创建兼容内存 DC
            mem_dc = mfc_dc.CreateCompatibleDC()

            # 3. 创建位图缓冲区
            # 用完整窗口尺寸（PrintWindow 截取整个窗口含标题栏）
            win_rect = win32gui.GetWindowRect(self._hwnd)
            win_w = win_rect[2] - win_rect[0]
            win_h = win_rect[3] - win_rect[1]

            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, win_w, win_h)
            old_bitmap = mem_dc.SelectObject(bitmap)

            # 4. PrintWindow: 让窗口把自己画到我们的内存 DC
            # PW_RENDERFULLCONTENT = 0x2 (Win 8.1+) 对 DX 游戏效果更好
            ctypes.windll.user32.PrintWindow(
                self._hwnd, mem_dc.GetSafeHdc(),
                PW_RENDERFULLCONTENT
            )

            # 5. 位图 → numpy
            bmp_info = bitmap.GetInfo()
            bmp_bits = bitmap.GetBitmapBits(True)
            frame = np.frombuffer(bmp_bits, dtype=np.uint8)
            frame = frame.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))

            # 6. 严谨清理资源 (防止 GDI 句柄泄漏)
            # 必须要 SelectObject 回去，否则 DeleteDC 会失败
            mem_dc.SelectObject(old_bitmap)
            win32gui.DeleteObject(bitmap.GetHandle())
            mem_dc.DeleteDC()
            # 注意: mfc_dc 是从 GetWindowDC 得到的，不需要也不应该调用 DeleteDC
            # 只需要释放底层的 hwnd_dc 即可
            win32gui.ReleaseDC(self._hwnd, hwnd_dc)

            # BGRA → BGR
            full_frame = frame[:, :, :3].copy()

            # 7. 裁剪出客户区（去掉标题栏和边框）
            client_point = win32gui.ClientToScreen(self._hwnd, (0, 0))
            x_offset = client_point[0] - win_rect[0]
            y_offset = client_point[1] - win_rect[1]

            client_frame = full_frame[
                y_offset : y_offset + self._height,
                x_offset : x_offset + self._width
            ]

            return client_frame.copy()

    def grab_minimap(self) -> np.ndarray:
        """后台截取小地图区域。"""
        frame = self.grab()
        x, y, w, h = self._minimap_region
        return frame[y:y+h, x:x+w].copy()

    def grab_region(self, region: tuple[int, int, int, int]) -> np.ndarray:
        """截取指定子区域 (x, y, width, height)。"""
        frame = self.grab()
        x, y, w, h = region
        return frame[y:y+h, x:x+w].copy()

    def set_minimap_region(self, x: int, y: int, w: int, h: int) -> None:
        """设置小地图区域坐标。"""
        self._minimap_region = (x, y, w, h)
        log.info(f"小地图区域已更新: ({x}, {y}, {w}, {h})")

    def bring_to_front(self) -> None:
        """将游戏窗口置于前台。"""
        if self._hwnd:
            try:
                win32gui.ShowWindow(self._hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self._hwnd)
                time.sleep(0.1)
                log.info("游戏窗口已前置")
            except Exception as e:
                log.warning(f"窗口前置失败: {e}")

    @property
    def hwnd(self) -> int:
        return self._hwnd

    @property
    def window_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_valid(self) -> bool:
        return self._hwnd != 0 and win32gui.IsWindow(self._hwnd)
