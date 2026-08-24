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
from src.utils.image_utils import letterbox_array

log = get_logger("window_capture")

# 规范画布尺寸: 匹配训练集, 保证训练/推理输入一致
CANONICAL_SIZE = (1366, 768)

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
        target_size: tuple[int, int] | None = CANONICAL_SIZE,
    ):
        """
        Args:
            process_name: 进程名（用于查找窗口）
            window_title: 窗口标题（备选查找方式）
            target_size: 输出帧 letterbox 到的尺寸. None = 保持原始客户区尺寸.
                          默认 (1366, 768), 与训练集一致, 保证训练/推理输入对齐.
        """
        self.process_name = process_name
        self.window_title = window_title
        self.target_size = target_size
        self._hwnd: int = 0
        self._width: int = 0
        self._height: int = 0

        # 最近一次 letterbox 参数 (测谎仪坐标映射需要: letterbox 坐标 → 客户区坐标)
        # 未启用 letterbox 时 = (1.0, 0, 0)。
        self._last_letterbox: tuple[float, int, int] = (1.0, 0, 0)
        self._last_letterbox_lock = threading.Lock()

        # 小地图区域 (相对于客户区，左上角)
        self._minimap_region: tuple[int, int, int, int] = (0, 0, 200, 150)
        self._lock = threading.Lock()

        # DXGI 抓帧 (windows_capture 库, 比 PrintWindow/BitBlt 快一个数量级)
        # 初始化失败时回退原有 PrintWindow/BitBlt 路径
        self._dxgi = None
        self._dxgi_control = None
        self._dxgi_frame = None
        self._dxgi_ok = False

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

        # 尝试 DXGI 抓帧 (失败自动回退 PrintWindow/BitBlt)
        self._init_dxgi()
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

    # ── DXGI 抓帧 (windows_capture 库, 比 PrintWindow/BitBlt 快一个数量级) ──

    def _init_dxgi(self) -> bool:
        """初始化 DXGI 抓帧。失败返回 False (调用方回退 PrintWindow/BitBlt)。"""
        if not self._hwnd:
            return False
        try:
            from windows_capture import WindowsCapture
            title = win32gui.GetWindowText(self._hwnd)
            if not title:
                return False
            self._dxgi = WindowsCapture(window_name=title)
            self._dxgi.event(self.on_frame_arrived)  # 处理器方法名必须精确匹配
            self._dxgi.event(self.on_closed)
            self._dxgi_control = self._dxgi.start_free_threaded()
            # 等第一帧 (最多 ~2s)
            for _ in range(40):
                if self._dxgi_frame is not None:
                    self._dxgi_ok = True
                    break
                time.sleep(0.05)
            if self._dxgi_ok:
                log.info("DXGI 抓帧已启用 (windows_capture)")
            else:
                log.warning("DXGI 抓帧未收到帧, 回退 PrintWindow/BitBlt")
            return self._dxgi_ok
        except Exception as e:
            log.warning(f"DXGI 抓帧初始化失败, 回退 PrintWindow/BitBlt: {e}")
            self._dxgi = None
            self._dxgi_ok = False
            return False

    def on_frame_arrived(self, frame, capture_control):
        """windows_capture 帧回调 (方法名必须精确匹配, 库按名字分发)。"""
        with self._lock:
            self._dxgi_frame = frame.frame_buffer

    def on_closed(self):
        self._dxgi_ok = False

    def stop(self):
        """停止抓帧线程 (释放 DXGI 资源)。"""
        if self._dxgi_control is not None:
            try:
                self._dxgi_control.stop()
            except Exception:
                pass
            self._dxgi_control = None
        self._dxgi_ok = False

    def grab(self) -> np.ndarray:
        """
        后台截取游戏窗口客户区。优先 DXGI (windows_capture), 失败回退 PrintWindow/BitBlt。

        即使窗口被其他窗口遮挡也能正常截取。

        Returns:
            BGR numpy 数组
        """
        with self._lock:
            if not self._hwnd or not win32gui.IsWindow(self._hwnd):
                raise RuntimeError("窗口句柄无效，请先调用 find_window()")

            # DXGI 优先 (此锁已被 grab 持有, 直接读 self._dxgi_frame, 不再加锁避免死锁)
            if self._dxgi_ok:
                f = self._dxgi_frame
                if f is not None:
                    try:
                        bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                        h, w = bgr.shape[:2]
                        ch, cw = self._height, self._width
                        if ch > 0 and cw > 0 and ch <= h and cw <= w:
                            title_h = max(0, h - ch)  # 标题栏高度
                            client_frame = bgr[title_h:title_h + ch, 0:cw]
                            if self.target_size is not None:
                                client_frame, scale, pl, pt = letterbox_array(client_frame, self.target_size)
                                with self._last_letterbox_lock:
                                    self._last_letterbox = (scale, pl, pt)
                            else:
                                with self._last_letterbox_lock:
                                    self._last_letterbox = (1.0, 0, 0)
                            return client_frame
                    except Exception as e:
                        log.warning(f"DXGI 帧处理失败, 回退 PrintWindow/BitBlt: {e}")

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

            client_frame = client_frame.copy()

            # 8. 空帧防护: 窗口最小化/尺寸异常时 PrintWindow 返回 0x0 或残缺帧,
            #    后续 letterbox 除零会崩溃 → 直接返回 None (调用方会重试)
            if client_frame.size == 0 or client_frame.shape[0] < 2 or client_frame.shape[1] < 2:
                self._last_grab_was_fallback = False
                return None

            # 9. PrintWindow 对某些 DX 游戏会整屏返回黑, 或返回"内容+右侧黑边"的残缺帧
            #    → 检测大面积近黑像素 (>25%) 就回退 BitBlt 屏幕截图 (需窗口可见/在前台)
            black_ratio = float((client_frame.mean(axis=2) < 10).mean())
            if not np.isfinite(black_ratio) or black_ratio > 0.25:
                try:
                    client_frame = self._grab_bitblt()
                    if not getattr(self, '_last_grab_was_fallback', False):
                        log.info(f"PrintWindow 黑屏/黑边(占比 {black_ratio:.0%}), 已回退 BitBlt 屏幕截图")
                    self._last_grab_was_fallback = True
                except Exception as e:
                    log.warning(f"PrintWindow 黑屏且 BitBlt 兜底失败: {e}")
                    self._last_grab_was_fallback = False
                # BitBlt 兜底也可能返回空 → 再次防护
                if client_frame is None or client_frame.size == 0 or \
                        client_frame.shape[0] < 2 or client_frame.shape[1] < 2:
                    return None
            else:
                self._last_grab_was_fallback = False

            # letterbox 到规范尺寸 (与训练集一致)
            if self.target_size is not None:
                client_frame, scale, pl, pt = letterbox_array(
                    client_frame, self.target_size
                )
                with self._last_letterbox_lock:
                    self._last_letterbox = (scale, pl, pt)
            else:
                with self._last_letterbox_lock:
                    self._last_letterbox = (1.0, 0, 0)

            return client_frame

    def _grab_bitblt(self) -> np.ndarray:
        """从屏幕 DC 截取整个窗口 (PrintWindow 返回黑屏时的兜底)。

        需要窗口可见且未被完全遮挡; 返回客户区 BGR。
        注意: 与 PrintWindow 不同, 窗口被完全遮挡时截到的是遮挡窗口的内容。
        """
        win_rect = win32gui.GetWindowRect(self._hwnd)
        win_w = win_rect[2] - win_rect[0]
        win_h = win_rect[3] - win_rect[1]

        screen_dc = win32gui.GetDC(0)
        try:
            mfc_dc = win32ui.CreateDCFromHandle(screen_dc)
            mem_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, win_w, win_h)
            old_bitmap = mem_dc.SelectObject(bitmap)
            mem_dc.BitBlt((0, 0), (win_w, win_h), mfc_dc,
                          (win_rect[0], win_rect[1]), win32con.SRCCOPY)
            bmp_info = bitmap.GetInfo()
            bits = bitmap.GetBitmapBits(True)
            mem_dc.SelectObject(old_bitmap)
            win32gui.DeleteObject(bitmap.GetHandle())
            mem_dc.DeleteDC()
            # mfc_dc 由 GetDC 派生, 不 DeleteDC, 只 ReleaseDC 底层句柄
        finally:
            win32gui.ReleaseDC(0, screen_dc)

        full_frame = np.frombuffer(bits, dtype=np.uint8)
        full_frame = full_frame.reshape(
            (bmp_info["bmHeight"], bmp_info["bmWidth"], 4))[:, :, :3].copy()

        client_point = win32gui.ClientToScreen(self._hwnd, (0, 0))
        x_offset = client_point[0] - win_rect[0]
        y_offset = client_point[1] - win_rect[1]
        return full_frame[y_offset:y_offset + self._height,
                          x_offset:x_offset + self._width].copy()

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

    def resize_window(self, client_w: int, client_h: int) -> None:
        """把游戏窗口客户区强制 resize 到指定尺寸 (参考 MapleStoryAutoLevelUp 的 auto_resize)。

        目的: 固定窗口尺寸 → 帧尺寸稳定、坐标常量有效、避免窗口过大/出屏导致"视频超出"。
        若客户区已是目标尺寸则跳过。
        """
        if not self._hwnd or not win32gui.IsWindow(self._hwnd):
            return
        self._update_size()
        if self._width == client_w and self._height == client_h:
            return
        try:
            wr = win32gui.GetWindowRect(self._hwnd)
            # 窗口尺寸 = 客户区 + 边框/标题栏 (当前差值不变)
            win_w = client_w + (wr[2] - wr[0]) - self._width
            win_h = client_h + (wr[3] - wr[1]) - self._height
            # 保证窗口在屏幕内 (左缘出屏时拉回)
            x, y = wr[0], wr[1]
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            win32gui.SetWindowPos(self._hwnd, None, x, y, win_w, win_h,
                                  win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
            time.sleep(0.1)
            self._update_size()
            log.info(f"游戏窗口已 resize 到客户区 {self._width}x{self._height}")
        except Exception as e:
            log.warning(f"窗口 resize 失败: {e}")

    @property
    def hwnd(self) -> int:
        return self._hwnd

    @property
    def window_size(self) -> tuple[int, int]:
        """返回输出帧尺寸 (letterbox 后, 若启用)"""
        if self.target_size is not None:
            return self.target_size
        return (self._width, self._height)

    @property
    def is_valid(self) -> bool:
        return self._hwnd != 0 and win32gui.IsWindow(self._hwnd)

    @property
    def last_letterbox(self) -> tuple[float, int, int]:
        """最近一次 letterbox 参数 (scale, pad_left, pad_top)。

        用于把视觉坐标 (letterbox 后) 映射回客户区坐标:
            client_x = (letterbox_x - pad_left) / scale
            client_y = (letterbox_y - pad_top) / scale
        然后再 ClientToScreen(hwnd, ...) → 屏幕坐标 → SetCursorPos。

        未启用 letterbox (target_size=None) 时返回 (1.0, 0, 0)。
        """
        with self._last_letterbox_lock:
            return self._last_letterbox
