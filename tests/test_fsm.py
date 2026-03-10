"""
状态机 (FSM) 测试。
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state.events import GameEvent, EventType


class TestAgentFSM:
    """AgentFSM 单元测试。"""

    def _make_fsm(self, tmp_path):
        """创建测试用 FSM 实例。"""
        from src.state.local_bus import LocalBus
        from src.state.global_bus import GlobalBus
        from src.brain.fsm import AgentFSM

        bus = LocalBus()
        gbus = GlobalBus(db_path=str(tmp_path / "test.db"))
        fsm = AgentFSM(local_bus=bus, global_bus=gbus)
        return fsm, gbus

    def test_initial_state(self, tmp_path):
        """验证初始状态为 idle。"""
        fsm, gbus = self._make_fsm(tmp_path)
        assert fsm.current_state == "idle"
        gbus.close()

    def test_idle_to_hunting(self, tmp_path):
        """验证 IDLE → HUNTING 转换。"""
        fsm, gbus = self._make_fsm(tmp_path)
        fsm.trigger("start_hunt")
        assert fsm.current_state == "hunting"
        gbus.close()

    def test_hunting_to_evading(self, tmp_path):
        """验证 HP 过低触发逃跑。"""
        fsm, gbus = self._make_fsm(tmp_path)
        fsm.trigger("start_hunt")
        fsm.trigger("hp_low")
        assert fsm.current_state == "evading"
        gbus.close()

    def test_captcha_from_any_state(self, tmp_path):
        """验证测谎仪可从任何状态进入。"""
        fsm, gbus = self._make_fsm(tmp_path)

        # 从 IDLE
        fsm.trigger("captcha_detected")
        assert fsm.current_state == "solving_captcha"
        assert fsm.is_paused

        # 解决后回到 IDLE
        fsm.trigger("captcha_solved")
        assert fsm.current_state == "idle"
        assert not fsm.is_paused

        # 从 HUNTING
        fsm.trigger("start_hunt")
        fsm.trigger("captcha_detected")
        assert fsm.current_state == "solving_captcha"

        gbus.close()

    def test_handle_event(self, tmp_path):
        """验证事件驱动的状态转换。"""
        fsm, gbus = self._make_fsm(tmp_path)

        event = GameEvent(
            event_type=EventType.CAPTCHA_DETECTED,
            payload={},
            source="yolo",
        )
        result = fsm.handle_event(event)
        assert result is True
        assert fsm.current_state == "solving_captcha"
        gbus.close()

    def test_full_hunting_cycle(self, tmp_path):
        """验证完整的打怪循环。"""
        fsm, gbus = self._make_fsm(tmp_path)

        fsm.trigger("start_hunt")       # idle → hunting
        assert fsm.current_state == "hunting"

        fsm.trigger("monster_killed")    # hunting → looting
        assert fsm.current_state == "looting"

        fsm.trigger("loot_done")         # looting → hunting
        assert fsm.current_state == "hunting"

        fsm.trigger("inventory_full")    # hunting → returning
        assert fsm.current_state == "returning"

        fsm.trigger("returned")          # returning → idle
        assert fsm.current_state == "idle"

        gbus.close()

    def test_action_callback(self, tmp_path):
        """验证动作回调注册和执行。"""
        fsm, gbus = self._make_fsm(tmp_path)

        callback_called = {"hunting": False}

        def on_hunting():
            callback_called["hunting"] = True

        fsm.register_action("hunting", on_hunting)
        fsm.trigger("start_hunt")

        assert callback_called["hunting"] is True
        gbus.close()

    def test_state_persistence(self, tmp_path):
        """验证状态变化被持久化到 GlobalBus。"""
        fsm, gbus = self._make_fsm(tmp_path)

        fsm.trigger("start_hunt")
        fsm.trigger("captcha_detected")

        events = gbus.query_events(EventType.STATE_CHANGED)
        assert len(events) >= 2
        gbus.close()
