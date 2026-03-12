from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import QWidget

from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card, _count_preflight_failures
from .gui_qt_deploy import DeployOpsPanel, _QtModelUploadDialog
from .gui_qt_record import RecordOpsPanel
from .gui_qt_teleop import TeleopOpsPanel
from .run_controller_service import ManagedRunController


def build_qt_core_ops_panel(
    *,
    section_id: str,
    config: dict[str, Any],
    append_log: Callable[[str], None],
    run_controller: ManagedRunController,
) -> QWidget | None:
    if section_id == "record":
        return RecordOpsPanel(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "deploy":
        return DeployOpsPanel(config=config, append_log=append_log, run_controller=run_controller)
    if section_id == "teleop":
        return TeleopOpsPanel(config=config, append_log=append_log, run_controller=run_controller)
    return None
