from __future__ import annotations

from typing import Any, Callable, TypeVar

from PySide6.QtCore import QObject, QTimer, Signal

from .background_jobs import LatestJobRunner

T = TypeVar("T")


class QtAfterAdapter(QObject):
    _dispatch = Signal(int, object, object)

    def __init__(self) -> None:
        super().__init__()
        self._dispatch.connect(self._deliver)

    def after(self, delay_ms: int, callback: object, *args: object) -> None:
        if not callable(callback):
            return
        self._dispatch.emit(max(0, int(delay_ms)), callback, tuple(args))

    def _deliver(self, delay_ms: int, callback: object, args: object) -> None:
        if not callable(callback):
            return
        payload = tuple(args) if isinstance(args, tuple) else (args,)
        QTimer.singleShot(max(0, int(delay_ms)), lambda: callback(*payload))


class UiBackgroundJobs:
    """Run background jobs and safely apply latest results on the UI event loop."""

    def __init__(self, root: Any, *, max_workers: int = 4) -> None:
        self._root = root
        self._runner = LatestJobRunner(max_workers=max_workers)
        self._shutdown = False

    def bump(self, key: str) -> int:
        return self._runner.bump(key)

    def is_current(self, key: str, version: int) -> bool:
        return self._runner.is_current(key, version)

    def submit(
        self,
        key: str,
        worker: Callable[[], T],
        *,
        on_success: Callable[[T], None],
        on_error: Callable[[Exception], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> int:
        version, future = self._runner.submit(key, worker)

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
        self._runner.shutdown()
