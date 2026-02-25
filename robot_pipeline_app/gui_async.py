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

        def _run() -> None:
            try:
                result = worker()
            except Exception as exc:  # pragma: no cover - handled by tests with fake worker
                def _error_cb() -> None:
                    is_stale = not self.is_current(key, version)
                    if not is_stale and on_error is not None:
                        on_error(exc)
                    if on_complete is not None:
                        on_complete(is_stale)

                self._root.after(0, _error_cb)
                return

            def _success_cb() -> None:
                is_stale = not self.is_current(key, version)
                if not is_stale:
                    on_success(result)
                if on_complete is not None:
                    on_complete(is_stale)

            self._root.after(0, _success_cb)

        self._executor.submit(_run)
        return version

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
