from __future__ import annotations

from typing import Any, Callable

from .run_controller_service import ManagedRunController

try:
    from PySide6.QtCore import QObject, Signal, Slot

    _QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - imported through gui_qt_app availability checks
    _QT_IMPORT_ERROR = exc


if _QT_IMPORT_ERROR is None:

    class _QtUiDispatcher(QObject):
        invoke = Signal(object, tuple)

        def __init__(self) -> None:
            super().__init__()
            self.invoke.connect(self._dispatch)

        @Slot(object, tuple)
        def _dispatch(self, callback: object, args: tuple[object, ...]) -> None:
            if not callable(callback):
                return
            callback(*args)

        def schedule(self, callback: Callable[..., None], *args: Any) -> None:
            self.invoke.emit(callback, tuple(args))


    class QtRunControllerBridge:
        def __init__(
            self,
            *,
            config: dict[str, Any],
            append_log: Callable[[str], None],
            external_busy: Callable[[], bool] | None = None,
            on_run_failure: Callable[[], None] | None = None,
            on_running_state_change: Callable[[bool], None] | None = None,
        ) -> None:
            self._dispatcher = _QtUiDispatcher()
            self.controller = ManagedRunController(
                config=config,
                schedule_ui=self._dispatcher.schedule,
                append_log=append_log,
                external_busy=external_busy,
                on_run_failure=on_run_failure,
                on_running_state_change=on_running_state_change,
            )

else:

    class QtRunControllerBridge:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError(f"PySide6 is unavailable: {_QT_IMPORT_ERROR}")
