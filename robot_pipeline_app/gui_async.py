from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class UiBackgroundJobs:
    """Run background jobs and safely apply latest results on Tk's main loop."""

    def __init__(self, root: Any, *, max_workers: int = 4) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gui-bg")
        self._lock = Lock()
        self._version_by_key: dict[str, int] = {}
        self._shutdown = False

    def bump(self, key: str) -> int:
        with self._lock:
            next_version = self._version_by_key.get(key, 0) + 1
            self._version_by_key[key] = next_version
            return next_version

    def is_current(self, key: str, version: int) -> bool:
        with self._lock:
            return self._version_by_key.get(key, 0) == version

    def submit(
        self,
        key: str,
        worker: Callable[[], T],
        *,
        on_success: Callable[[T], None],
        on_error: Callable[[Exception], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> int:
        version = self.bump(key)
        future = self._executor.submit(worker)

        def _finish() -> None:
            if self._shutdown:
                return
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - handled by tests with fake worker
                is_stale = not self.is_current(key, version)
                if not is_stale and on_error is not None:
                    on_error(exc)
                if on_complete is not None:
                    on_complete(is_stale)
                return

            is_stale = not self.is_current(key, version)
            if not is_stale:
                on_success(result)
            if on_complete is not None:
                on_complete(is_stale)

        def _poll() -> None:
            if self._shutdown:
                return
            if future.done():
                _finish()
                return
            self._root.after(15, _poll)

        if not self._shutdown:
            self._root.after(0, _poll)
        return version

    def shutdown(self) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)
