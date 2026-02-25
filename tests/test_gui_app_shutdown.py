from __future__ import annotations

import unittest
from collections import deque
from typing import Any, Callable

from robot_pipeline_app.gui_app import _schedule_shutdown_after_cancel


class _QueuedRoot:
    def __init__(self) -> None:
        self._jobs: deque[tuple[Callable[..., Any], tuple[Any, ...]]] = deque()

    def after(self, _delay: int, callback: Callable[..., Any], *args: Any) -> str:
        self._jobs.append((callback, args))
        return "after-id"

    def run_next(self) -> None:
        if not self._jobs:
            return
        callback, args = self._jobs.popleft()
        callback(*args)

    def drain(self, *, limit: int = 100) -> None:
        steps = 0
        while self._jobs and steps < limit:
            self.run_next()
            steps += 1


class GuiAppShutdownTests(unittest.TestCase):
    def test_shutdown_runs_immediately_when_no_active_process(self) -> None:
        root = _QueuedRoot()
        canceled: list[bool] = []
        finalized: list[bool] = []

        _schedule_shutdown_after_cancel(
            root=root,
            has_active_process=lambda: False,
            request_cancel=lambda: canceled.append(True),
            finalize_shutdown=lambda: finalized.append(True),
        )

        self.assertEqual(canceled, [])
        self.assertEqual(finalized, [True])
        self.assertEqual(len(root._jobs), 0)

    def test_shutdown_polls_event_loop_until_process_exits(self) -> None:
        root = _QueuedRoot()
        cancel_calls: list[bool] = []
        finalized: list[bool] = []
        active_states = deque([True, True, False])
        ticks = {"value": 0}

        def has_active_process() -> bool:
            if active_states:
                return active_states.popleft()
            return False

        def monotonic() -> float:
            value = float(ticks["value"]) * 0.1
            ticks["value"] += 1
            return value

        _schedule_shutdown_after_cancel(
            root=root,
            has_active_process=has_active_process,
            request_cancel=lambda: cancel_calls.append(True),
            finalize_shutdown=lambda: finalized.append(True),
            timeout_s=3.0,
            monotonic=monotonic,
        )

        self.assertEqual(cancel_calls, [True])
        self.assertEqual(finalized, [])
        root.drain()
        self.assertEqual(finalized, [True])

    def test_shutdown_finalizes_after_cancel_timeout_without_blocking(self) -> None:
        root = _QueuedRoot()
        cancel_calls: list[bool] = []
        finalized: list[bool] = []
        monotonic_values = deque([0.0, 0.4, 0.8, 1.2])

        def monotonic() -> float:
            if monotonic_values:
                return monotonic_values.popleft()
            return 2.0

        _schedule_shutdown_after_cancel(
            root=root,
            has_active_process=lambda: True,
            request_cancel=lambda: cancel_calls.append(True),
            finalize_shutdown=lambda: finalized.append(True),
            timeout_s=0.3,
            monotonic=monotonic,
        )

        self.assertEqual(cancel_calls, [True])
        self.assertEqual(finalized, [])
        root.run_next()
        self.assertEqual(finalized, [True])


if __name__ == "__main__":
    unittest.main()
