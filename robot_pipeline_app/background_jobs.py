from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, TypeVar

T = TypeVar("T")


class LatestJobRunner:
    """Track latest-by-key background work independently from any UI toolkit."""

    def __init__(self, *, max_workers: int = 4) -> None:
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

    def submit(self, key: str, worker: Callable[[], T]) -> tuple[int, Future[T]]:
        if self._shutdown:
            raise RuntimeError("LatestJobRunner is shut down.")
        version = self.bump(key)
        future = self._executor.submit(worker)
        return version, future

    def shutdown(self) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)
