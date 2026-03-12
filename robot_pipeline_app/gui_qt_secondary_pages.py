from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QWidget

from .gui_qt_config_page import QtConfigPage
from .gui_qt_experiments_page import QtExperimentsPage
from .gui_qt_history_page import QtHistoryPage
from .gui_qt_page_base import (
    _CameraSchemaEditor,
    _InputGrid,
    _PageWithOutput,
    _VideoFrameLabel,
    _VideoGalleryTile,
    _build_card,
    _json_text,
    _quiet_cv2_logging,
    _set_readonly_table,
    _set_table_headers,
)
from .gui_qt_visualizer_page import QtVisualizerPage
from .run_controller_service import ManagedRunController


def build_qt_secondary_panel(
    *,
    section_id: str,
    config: dict[str, Any],
    append_log: Callable[[str], None],
    run_controller: ManagedRunController,
    run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
    update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
) -> QWidget | None:
    if section_id == "config":
        return QtConfigPage(
            config=config,
            append_log=append_log,
            run_terminal_command=run_terminal_command,
            update_and_restart_app=update_and_restart_app,
        )
    if section_id == "experiments":
        return QtExperimentsPage(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "visualizer":
        return QtVisualizerPage(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "history":
        return QtHistoryPage(config=config, append_log=append_log, run_controller=run_controller)
    return None
