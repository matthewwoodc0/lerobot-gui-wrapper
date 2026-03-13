from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QWidget

from .gui_qt_config_page import QtConfigPage
from .gui_qt_experiments_page import QtExperimentsPage
from .gui_qt_history_page import QtHistoryPage
from .gui_qt_workflows_page import QtWorkflowsPage
from .gui_qt_visualizer_page import QtVisualizerPage
from .run_controller_service import ManagedRunController
from .workflow_queue import WorkflowQueueService


def build_qt_secondary_panel(
    *,
    section_id: str,
    config: dict[str, Any],
    append_log: Callable[[str], None],
    run_controller: ManagedRunController,
    run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
    update_and_restart_app: Callable[[], tuple[bool, str]] | None = None,
    on_config_changed: Callable[[], None] | None = None,
    workflow_queue: WorkflowQueueService | None = None,
) -> QWidget | None:
    if section_id == "config":
        return QtConfigPage(
            config=config,
            append_log=append_log,
            run_terminal_command=run_terminal_command,
            update_and_restart_app=update_and_restart_app,
            on_config_changed=on_config_changed,
        )
    if section_id == "experiments":
        return QtExperimentsPage(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "visualizer":
        return QtVisualizerPage(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "history":
        return QtHistoryPage(config=config, append_log=append_log, run_controller=run_controller)
    if section_id in {"workflows", "queue"} and workflow_queue is not None:
        return QtWorkflowsPage(config=config, append_log=append_log, workflows_service=workflow_queue)
    return None
