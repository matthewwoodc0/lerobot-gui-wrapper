from __future__ import annotations

import time
import unittest
from typing import Any, Callable

from robot_pipeline_app.gui_run_popout import RunControlPopout


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


class RunControlPopoutTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
