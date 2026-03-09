from __future__ import annotations

import time
import unittest
from typing import Any, Callable

from robot_pipeline_app.gui_run_popout import RunControlPopout, TeleopRunPopout


class _FakeRoot:
    def __init__(self) -> None:
        self._counter = 0

    def after(self, _delay_ms: int, _callback: Callable[..., Any]) -> str:
        self._counter += 1
        return f"job-{self._counter}"

    def after_cancel(self, _job_id: str) -> None:
        return

    def bell(self) -> None:
        return


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _FakeFrame:
    def __init__(self) -> None:
        self.pack_called = False
        self.pack_forget_called = False
        self.grid_calls = 0
        self.grid_remove_calls = 0

    def pack(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        self.pack_called = True

    def pack_forget(self) -> None:
        self.pack_forget_called = True

    def grid(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        self.grid_calls += 1

    def grid_remove(self) -> None:
        self.grid_remove_calls += 1


class _FakeWindow:
    def __init__(self) -> None:
        self.withdrawn = False
        self.deiconified = False
        self.focused = False
        self.lifted = False
        self.destroyed = False

    def winfo_exists(self) -> bool:
        return not self.destroyed

    def deiconify(self) -> None:
        self.deiconified = True

    def lift(self) -> None:
        self.lifted = True

    def focus_force(self) -> None:
        self.focused = True

    def withdraw(self) -> None:
        self.withdrawn = True

    def destroy(self) -> None:
        self.destroyed = True

    def state(self) -> str:
        return "withdrawn" if self.withdrawn else "normal"


class _FakeFeed:
    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1

    def close(self) -> None:
        self.close_count += 1


class _FakeProgressbar(dict[str, Any]):
    def configure(self, **kwargs: Any) -> None:
        self.update(kwargs)


class RunControlPopoutTest(unittest.TestCase):
    def _build_start_run_popout(self) -> tuple[RunControlPopout, _FakeFeed, _FakeFrame, _FakeFrame, list[str]]:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._ensure_window = lambda: None  # type: ignore[method-assign]
        popout.window = _FakeWindow()
        popout.mode_var = _FakeVar()
        popout.episode_var = _FakeVar()
        popout.episode_timer_var = _FakeVar()
        popout.key_status_var = _FakeVar()
        popout.outcome_tags_var = _FakeVar()
        popout.outcome_status_var = _FakeVar()
        popout.outcome_summary_var = _FakeVar()
        popout.outcome_target_var = _FakeVar()
        popout.episode_progressbar = _FakeProgressbar()
        outcome_frame = _FakeFrame()
        record_host = _FakeFrame()
        feed = _FakeFeed()
        popout._outcome_frame = outcome_frame
        popout._record_camera_host = record_host
        popout._record_camera_feed = feed
        popout._update_outcome_target_label = lambda: None  # type: ignore[method-assign]
        popout._stop_reset_prompt = lambda: None  # type: ignore[method-assign]
        popout._set_outcome_controls_enabled = lambda _enabled: None  # type: ignore[method-assign]
        popout._update_outcome_summary_label = lambda: None  # type: ignore[method-assign]
        popout._refresh_outcome_history_rows = lambda: None  # type: ignore[method-assign]
        popout._refresh_outcome_button_states = lambda: None  # type: ignore[method-assign]
        popout._schedule_tick = lambda: None  # type: ignore[method-assign]
        popout._pulse_dot = lambda: None  # type: ignore[method-assign]
        popout._refresh_kb_indicator = lambda: None  # type: ignore[method-assign]
        applied_modes: list[str] = []
        popout._apply_window_size_for_mode = lambda mode: applied_modes.append(mode)  # type: ignore[method-assign]
        return popout, feed, record_host, outcome_frame, applied_modes

    def test_tick_freezes_timer_when_episode_duration_elapsed(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 2.0
        popout._episode_started_at = time.monotonic() - 4.0

        popout._tick()

        self.assertTrue(popout._awaiting_next_episode)
        self.assertIsNone(popout._episode_started_at)
        self.assertEqual(popout._timer_job, "job-1")

    def test_handle_output_line_advances_episode_and_resumes_timer(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 10.0
        popout._current_episode = 1
        popout._awaiting_next_episode = True
        popout._episode_started_at = None

        popout.handle_output_line("Recording episode 2/5")

        self.assertEqual(popout._current_episode, 2)
        self.assertFalse(popout._awaiting_next_episode)
        self.assertIsNotNone(popout._episode_started_at)

    def test_handle_output_line_maps_zero_based_episode_index(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 10.0

        popout.handle_output_line("Recording episode 0")

        self.assertEqual(popout._current_episode, 1)
        self.assertFalse(popout._awaiting_next_episode)
        self.assertIsNotNone(popout._episode_started_at)

    def test_handle_output_line_restarts_same_episode_after_reset_phase(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 10.0
        popout._current_episode = 1
        popout._awaiting_next_episode = True
        popout._episode_started_at = None

        popout.handle_output_line("Recording episode 1/5")

        self.assertEqual(popout._current_episode, 1)
        self.assertFalse(popout._awaiting_next_episode)
        self.assertIsNotNone(popout._episode_started_at)

    def test_handle_output_line_clears_previous_outcome_on_episode_rerun(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 10.0
        popout._allow_outcome_marking = True
        popout._current_episode = 1
        popout._total_episodes = 5
        popout._awaiting_next_episode = True
        popout._episode_started_at = None
        popout._episode_outcomes = {1: {"episode": 1, "result": "success", "tags": ["block-left"]}}
        popout.outcome_status_var = _FakeVar()
        popout.outcome_summary_var = _FakeVar()

        popout.handle_output_line("Recording episode 1/5")

        self.assertNotIn(1, popout._episode_outcomes)
        self.assertIn("Previous mark cleared", popout.outcome_status_var.get())
        self.assertIn("Rated: 0/5", popout.outcome_summary_var.get())

    def test_get_episode_outcome_summary_aggregates_counts_and_tags(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._allow_outcome_marking = True
        popout._total_episodes = 3
        popout._episode_outcomes = {
            1: {"episode": 1, "result": "success", "tags": ["vertical"]},
            2: {"episode": 2, "result": "failed", "tags": ["horizontal", "zone-left"]},
        }

        summary = popout.get_episode_outcome_summary()

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["rated_count"], 2)
        self.assertEqual(summary["unrated_count"], 1)
        self.assertEqual(summary["tags"], ["horizontal", "vertical", "zone-left"])

    def test_send_key_dispatches_immediately(self) -> None:
        sent: list[str] = []
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda direction: (sent.append(direction) or True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._awaiting_next_episode = False

        popout._send_key("right")

        self.assertEqual(sent, ["right"])
        self.assertIsNone(popout._pending_direction)

    def test_mark_episode_outcome_updates_selected_prior_episode(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._allow_outcome_marking = True
        popout._current_episode = 3
        popout._selected_episode = 1
        popout.outcome_tags_var = _FakeVar("retry, close")
        popout.outcome_status_var = _FakeVar()
        popout.outcome_summary_var = _FakeVar()
        popout._episode_outcomes = {1: {"episode": 1, "result": "failed", "tags": ["old"]}}

        popout._mark_episode_outcome("success")

        self.assertEqual(popout._episode_outcomes[1]["result"], "success")
        self.assertEqual(popout._episode_outcomes[1]["tags"], ["retry", "close"])
        self.assertNotIn(3, popout._episode_outcomes)

    def test_apply_episode_tags_keeps_unmarked_entry_and_rated_count(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._allow_outcome_marking = True
        popout._total_episodes = 4
        popout._current_episode = 2
        popout._selected_episode = 2
        popout.outcome_tags_var = _FakeVar("lighting, tight-turn")
        popout.outcome_status_var = _FakeVar()
        popout.outcome_summary_var = _FakeVar()

        popout._apply_episode_tags()
        summary = popout.get_episode_outcome_summary()

        self.assertEqual(popout._episode_outcomes[2]["result"], "unmarked")
        self.assertEqual(popout._episode_outcomes[2]["tags"], ["lighting", "tight-turn"])
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["rated_count"], 0)
        self.assertEqual(summary["unrated_count"], 4)
        self.assertIn("lighting", summary["tags"])

    def test_handle_output_line_clears_pending_on_episode_start(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._episode_duration_s = 10.0
        popout._pending_direction = "right"

        popout.handle_output_line("Recording episode 1/5")

        self.assertIsNone(popout._pending_direction)

    def test_show_reset_prompt_sets_message_and_job(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout._active = True
        popout._current_episode = 1
        popout.reset_prompt_var = _FakeVar()

        popout._show_reset_prompt()

        self.assertIn("Reset the environment", popout.reset_prompt_var.get())
        self.assertIsNotNone(popout._reset_prompt_job)

    def test_stop_reset_prompt_clears_message(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        popout.reset_prompt_var = _FakeVar("Reset the environment...")
        popout._reset_prompt_job = "job-1"

        popout._stop_reset_prompt()

        self.assertEqual(popout.reset_prompt_var.get(), "")
        self.assertIsNone(popout._reset_prompt_job)

    def test_start_run_record_starts_runtime_feed_and_hides_outcome_tracker(self) -> None:
        popout, feed, record_host, outcome_frame, applied_modes = self._build_start_run_popout()

        popout.start_run("record", 5, 50)

        self.assertEqual(feed.start_count, 1)
        self.assertEqual(record_host.grid_calls, 1)
        self.assertTrue(outcome_frame.pack_forget_called)
        self.assertEqual(applied_modes, ["record"])

    def test_start_run_deploy_keeps_runtime_feed_hidden(self) -> None:
        popout, feed, record_host, outcome_frame, applied_modes = self._build_start_run_popout()

        popout.start_run("deploy", 5, 50)

        self.assertEqual(feed.start_count, 0)
        self.assertEqual(feed.stop_count, 1)
        self.assertEqual(record_host.grid_remove_calls, 1)
        self.assertTrue(outcome_frame.pack_called)
        self.assertEqual(applied_modes, ["deploy"])

    def test_hide_stops_runtime_feed(self) -> None:
        popout, feed, _record_host, _outcome_frame, _applied_modes = self._build_start_run_popout()

        popout.start_run("record", 5, 50)
        popout.hide()

        self.assertEqual(feed.start_count, 1)
        self.assertEqual(feed.stop_count, 1)
        assert isinstance(popout.window, _FakeWindow)
        self.assertTrue(popout.window.withdrawn)

    def test_window_close_requests_cancel_when_active(self) -> None:
        cancel_calls: list[bool] = []
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: cancel_calls.append(True),
        )
        popout._active = True
        hide_calls: list[bool] = []
        popout.hide = lambda: hide_calls.append(True)  # type: ignore[method-assign]

        popout._handle_window_close()

        self.assertEqual(cancel_calls, [True])
        self.assertEqual(hide_calls, [])

    def test_window_close_hides_when_inactive(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        hide_calls: list[bool] = []
        popout.hide = lambda: hide_calls.append(True)  # type: ignore[method-assign]

        popout._handle_window_close()

        self.assertEqual(hide_calls, [True])

    def test_apply_theme_rebuilds_active_window(self) -> None:
        popout = RunControlPopout(
            root=_FakeRoot(),
            colors={"theme_mode": "dark"},
            on_send_key=lambda _direction: (True, "ok"),
            on_cancel=lambda: None,
        )
        original_window = _FakeWindow()
        original_feed = _FakeFeed()
        popout.window = original_window
        popout._record_camera_feed = original_feed
        popout._active = True
        calls: list[object] = []

        def _ensure_window() -> None:
            calls.append("ensure")
            popout.window = _FakeWindow()

        popout._ensure_window = _ensure_window  # type: ignore[method-assign]
        popout._sync_window_state = lambda *, show_window: calls.append(("sync", show_window))  # type: ignore[method-assign]
        popout._pulse_dot = lambda: calls.append("pulse")  # type: ignore[method-assign]
        popout._schedule_tick = lambda: calls.append("tick")  # type: ignore[method-assign]
        popout._refresh_kb_indicator = lambda: calls.append("kb")  # type: ignore[method-assign]

        updated_colors = {"theme_mode": "light", "panel": "#ffffff"}
        popout.apply_theme(updated_colors)

        self.assertIs(popout.colors, updated_colors)
        self.assertTrue(original_window.destroyed)
        self.assertEqual(original_feed.close_count, 1)
        self.assertIn("ensure", calls)
        self.assertIn(("sync", True), calls)
        self.assertIn("pulse", calls)
        self.assertIn("tick", calls)
        self.assertIn("kb", calls)


class TeleopRunPopoutTest(unittest.TestCase):
    def test_window_close_requests_cancel_when_active(self) -> None:
        cancel_calls: list[bool] = []
        popout = TeleopRunPopout(
            root=_FakeRoot(),
            colors={},
            on_cancel=lambda: cancel_calls.append(True),
        )
        popout._active = True
        hide_calls: list[bool] = []
        popout.hide = lambda: hide_calls.append(True)  # type: ignore[method-assign]

        popout._handle_window_close()

        self.assertEqual(cancel_calls, [True])
        self.assertEqual(hide_calls, [])

    def test_window_close_hides_when_inactive(self) -> None:
        popout = TeleopRunPopout(
            root=_FakeRoot(),
            colors={},
            on_cancel=lambda: None,
        )
        hide_calls: list[bool] = []
        popout.hide = lambda: hide_calls.append(True)  # type: ignore[method-assign]

        popout._handle_window_close()

        self.assertEqual(hide_calls, [True])

    def test_apply_theme_rebuilds_active_window(self) -> None:
        popout = TeleopRunPopout(
            root=_FakeRoot(),
            colors={"theme_mode": "dark"},
            on_cancel=lambda: None,
        )
        original_window = _FakeWindow()
        popout.window = original_window
        popout._active = True
        calls: list[object] = []

        def _ensure_window() -> None:
            calls.append("ensure")
            popout.window = _FakeWindow()

        popout._ensure_window = _ensure_window  # type: ignore[method-assign]
        popout._sync_window_state = lambda *, show_window: calls.append(("sync", show_window))  # type: ignore[method-assign]
        popout._pulse_dot = lambda: calls.append("pulse")  # type: ignore[method-assign]
        popout._schedule_tick = lambda: calls.append("tick")  # type: ignore[method-assign]

        updated_colors = {"theme_mode": "light", "panel": "#ffffff"}
        popout.apply_theme(updated_colors)

        self.assertIs(popout.colors, updated_colors)
        self.assertTrue(original_window.destroyed)
        self.assertIn("ensure", calls)
        self.assertIn(("sync", True), calls)
        self.assertIn("pulse", calls)
        self.assertIn("tick", calls)


if __name__ == "__main__":
    unittest.main()
