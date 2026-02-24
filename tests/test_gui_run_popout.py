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


if __name__ == "__main__":
    unittest.main()
