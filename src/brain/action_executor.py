"""
ActionExecutor — 眼手分离: 决策循环只发命令, 本线程执行实际按键动作。

参考 MapleStoryAutoLevelUp 的"决策循环 set_command + 独立键盘线程"结构。

设计要点:
- 主循环每帧 `set_action()` 推送最新动作 (非阻塞)。
- 动作用"粗网格键"去重: 同一目标的轻微抖动不会触发重推, 避免把正在进行的 burst
  每帧重置 (防空转)。
- 动作方法内定期 `is_cancelled(action_id)` 检查: 决策已变则中止当前动作, 快速切换。
- 动作对象约定为 `(type, payload)`:
    ("none",    None)             待机/停手
    ("attack",  Target)           对目标 burst 连打
    ("approach", Target)          接近目标 (走/跳)
    ("patrol",  None)             地形巡逻一步
"""
from __future__ import annotations

import threading

from src.utils.logger import get_logger

log = get_logger("action_executor")


class ActionExecutor:
    def __init__(self, worker):
        """
        Args:
            worker: callable(action_id: int, action: tuple) -> None
                    在 executor 线程内执行动作; 动作内部用 is_cancelled(action_id) 中断。
        """
        self._worker = worker
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._action_id = 0
        self._pending = None      # (action_id, action)
        self._pending_key = None  # 待执行动作的去重键
        self._current = None      # 正在执行的动作
        self._current_key = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── 键: 动作去重 (目标用稳定身份 id, 防抖动跨网格边界导致 burst 被反复取消) ──
    @staticmethod
    def _key(action):
        atype, payload = action
        if payload is None:
            return (atype, None)
        tid = getattr(payload, "id", None)
        if tid is not None:
            return (atype, ("id", tid))
        return (atype, (payload.cx // 60, payload.cy // 60))

    def set_action(self, action) -> bool:
        """推送新动作 (非阻塞)。与待执行/正在执行相同则忽略, 返回是否真的变更。"""
        with self._lock:
            if action is None:
                return False
            k = self._key(action)
            if k == self._pending_key or k == self._current_key:
                return False
            self._action_id += 1
            self._pending = (self._action_id, action)
            self._pending_key = k
            self._cv.notify()
            return True

    def is_cancelled(self, action_id: int) -> bool:
        """动作方法内调用: 传入启动时拿到的 action_id, 返回是否已被更新的决策取代。"""
        with self._lock:
            return action_id != self._action_id

    def stop(self):
        with self._lock:
            self._running = False
            self._action_id += 1  # 触发正在执行的动作立即中断 (is_cancelled 返回 True)
            self._cv.notify()

    def _loop(self):
        while True:
            with self._lock:
                while self._running and self._pending is None:
                    self._cv.wait(timeout=0.2)
                if not self._running:
                    return
                action_id, action = self._pending
                self._pending = None
                self._current = action
                self._current_key = self._pending_key
                self._pending_key = None
            try:
                self._worker(action_id, action)
            except Exception as e:
                log.error(f"[EXEC] 动作执行异常: {e}", exc_info=True)
            finally:
                # 动作完成/中断后清除 current, 允许同一动作被再次推送
                with self._lock:
                    self._current = None
                    self._current_key = None
