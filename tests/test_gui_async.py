import unittest
from collections import Counter
from queue import Empty, Queue
from threading import Event
from time import monotonic, sleep
from typing import Callable

from robot_pipeline_app.gui_async import UiBackgroundJobs


class _FakeRoot:
    def after(self, _ms, callback):
        callback()
        return "after-id"


class _QueuedRoot:
    def __init__(self) -> None:
        self._queue: Queue[Callable[[], None]] = Queue()

    def after(self, _ms, callback):
        self._queue.put(callback)
        return "after-id"

    def drain(self, *, timeout_s: float = 0.01) -> int:
        drained = 0
        while True:
            try:
                callback = self._queue.get(timeout=timeout_s if drained == 0 else 0.0)
            except Empty:
                break
            callback()
            drained += 1
        return drained

    def wait_until(self, predicate, *, timeout_s: float = 1.0) -> None:
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            self.drain(timeout_s=0.005)
            if predicate():
                return
            sleep(0.002)
        self.drain(timeout_s=0.005)
        if not predicate():
            raise AssertionError("condition not met before timeout")


class GuiAsyncTests(unittest.TestCase):
    def test_submit_applies_latest_result_only(self):
        root = _FakeRoot()
        jobs = UiBackgroundJobs(root, max_workers=1)
        self.addCleanup(jobs.shutdown)

        applied: list[int] = []
        stale: list[bool] = []

        first = jobs.bump("k")
        jobs.submit(
            "k",
            lambda: 1,
            on_success=lambda value: applied.append(value),
            on_complete=lambda is_stale: stale.append(is_stale),
        )
        self.assertGreater(first, 0)

        self.assertEqual(applied, [1])
        self.assertEqual(stale, [False])

    def test_is_current_tracks_versions(self):
        jobs = UiBackgroundJobs(_FakeRoot(), max_workers=1)
        self.addCleanup(jobs.shutdown)

        v1 = jobs.bump("source")
        self.assertTrue(jobs.is_current("source", v1))
        v2 = jobs.bump("source")
        self.assertFalse(jobs.is_current("source", v1))
        self.assertTrue(jobs.is_current("source", v2))

    def test_submit_marks_older_result_as_stale_when_newer_finishes_first(self):
        root = _QueuedRoot()
        jobs = UiBackgroundJobs(root, max_workers=2)
        self.addCleanup(jobs.shutdown)

        release_old = Event()
        release_new = Event()
        applied: list[int] = []
        stale_flags: list[bool] = []

        jobs.submit(
            "history-refresh",
            lambda: (release_old.wait(1.0), 1)[1],
            on_success=lambda value: applied.append(value),
            on_complete=lambda is_stale: stale_flags.append(is_stale),
        )
        jobs.submit(
            "history-refresh",
            lambda: (release_new.wait(1.0), 2)[1],
            on_success=lambda value: applied.append(value),
            on_complete=lambda is_stale: stale_flags.append(is_stale),
        )

        release_new.set()
        root.wait_until(lambda: applied == [2], timeout_s=1.0)
        release_old.set()
        root.wait_until(lambda: len(stale_flags) == 2, timeout_s=1.0)

        self.assertEqual(applied, [2])
        self.assertEqual(Counter(stale_flags), Counter({False: 1, True: 1}))

    def test_submit_suppresses_stale_error_callbacks(self):
        root = _QueuedRoot()
        jobs = UiBackgroundJobs(root, max_workers=2)
        self.addCleanup(jobs.shutdown)

        release_old = Event()
        release_new = Event()
        errors: list[str] = []
        applied: list[int] = []
        stale_flags: list[bool] = []

        def stale_error_worker() -> int:
            release_old.wait(1.0)
            raise RuntimeError("stale failure")

        jobs.submit(
            "record-dataset-metadata",
            stale_error_worker,
            on_success=lambda value: applied.append(value),
            on_error=lambda exc: errors.append(str(exc)),
            on_complete=lambda is_stale: stale_flags.append(is_stale),
        )
        jobs.submit(
            "record-dataset-metadata",
            lambda: (release_new.wait(1.0), 7)[1],
            on_success=lambda value: applied.append(value),
            on_error=lambda exc: errors.append(str(exc)),
            on_complete=lambda is_stale: stale_flags.append(is_stale),
        )

        release_new.set()
        root.wait_until(lambda: applied == [7], timeout_s=1.0)
        release_old.set()
        root.wait_until(lambda: len(stale_flags) == 2, timeout_s=1.0)

        self.assertEqual(applied, [7])
        self.assertEqual(errors, [])
        self.assertEqual(Counter(stale_flags), Counter({False: 1, True: 1}))


if __name__ == "__main__":
    unittest.main()
