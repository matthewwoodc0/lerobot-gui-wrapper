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

        popout.handle_output_line("Episode 2/5")

        self.assertEqual(popout._current_episode, 2)
        self.assertFalse(popout._awaiting_next_episode)
        self.assertIsNotNone(popout._episode_started_at)


if __name__ == "__main__":
    unittest.main()
