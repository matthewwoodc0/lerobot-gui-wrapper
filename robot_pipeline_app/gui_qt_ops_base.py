from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .app_theme import SPACING_CARD, SPACING_COMPACT, SPACING_SHELL
from .camera_state import camera_mapping_summary
from .checks import has_failures, run_preflight_for_deploy, run_preflight_for_record, run_preflight_for_teleop, summarize_checks
from .artifacts import _normalize_deploy_episode_outcomes, write_deploy_episode_spreadsheet, write_deploy_notes_file
from .command_text import format_command_for_dialog
from .command_overrides import get_flag_value, get_policy_path_value
from .commands import resolve_follower_robot_id, resolve_leader_robot_id
from .config_store import _atomic_write, get_lerobot_dir, save_config
from .constants import DEFAULT_TASK
from .deploy_workflow_helpers import (
    ModelBrowserNode,
    build_model_browser_tree,
    build_model_upload_request,
    build_calibration_command,
    camera_rename_map_suggestion,
    first_model_payload_candidate,
    quick_actions_from_checks,
    resolve_payload_path,
    split_model_selection,
    summarize_model_info,
)
from .failure_inspector import (
    build_failure_explanation_text,
    build_run_summary_text,
    has_failure_details,
    raw_transcript_text,
)
from .gui_forms import (
    build_deploy_request_and_command,
    build_record_request_and_command,
    build_teleop_request_and_command,
)
from .gui_qt_camera import QtCameraWorkspace
from .gui_qt_dialogs import ask_editable_command_dialog, ask_text_dialog, ask_text_dialog_with_actions, show_text_dialog
from .gui_qt_output import QtRunOutputPanel
from .gui_qt_runtime_helpers import QtRunHelperDialog
from .repo_utils import normalize_repo_id, repo_name_from_repo_id, repo_name_only, suggest_eval_prefixed_repo_id
from .run_controller_service import ManagedRunController, RunUiHooks
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .workflows import move_recorded_dataset

def _build_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("SectionCard")
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(SPACING_SHELL, SPACING_SHELL, SPACING_SHELL, SPACING_SHELL)
    layout.setSpacing(SPACING_CARD)

    header = QLabel(title)
    header.setObjectName("SectionMeta")
    layout.addWidget(header)
    return card, layout


def _count_preflight_failures(checks: list[tuple[str, str, str]]) -> int:
    return sum(1 for level, _name, _detail in checks if str(level).strip().upper() == "FAIL")


class _InputGrid:
    def __init__(self, layout: QVBoxLayout) -> None:
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACING_COMPACT)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self._index = 0
        layout.addLayout(self._grid)

    def add_field(self, label_text: str, widget: QWidget) -> None:
        row = self._index // 2
        pair = self._index % 2
        label_col = pair * 2
        widget_col = label_col + 1
        label = QLabel(label_text)
        label.setObjectName("FormLabel")
        self._grid.addWidget(label, row, label_col)
        self._grid.addWidget(widget, row, widget_col)
        self._index += 1


class _AdvancedOptionsPanel(QFrame):
    def __init__(self, *, title: str, fields: list[tuple[str, str]]) -> None:
        super().__init__()
        self.setObjectName("SectionCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.fields = fields
        self.inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_COMPACT, SPACING_COMPACT, SPACING_COMPACT, SPACING_COMPACT)
        layout.setSpacing(10)

        header = QLabel(title)
        header.setObjectName("SectionMeta")
        layout.addWidget(header)

        form = _InputGrid(layout)
        for key, label in fields:
            widget = QLineEdit("")
            widget.setPlaceholderText(f"--{key}")
            form.add_field(label, widget)
            self.inputs[key] = widget

        self.custom_args_input = QLineEdit("")
        self.custom_args_input.setPlaceholderText("--flag value --other=value")
        form.add_field("Extra flags", self.custom_args_input)

    def build_overrides(self) -> tuple[dict[str, str] | None, str]:
        overrides: dict[str, str] = {}
        for key, _label in self.fields:
            value = self.inputs[key].text().strip()
            if value:
                overrides[key] = value
        return (overrides or None), self.custom_args_input.text().strip()

    def seed_from_command(self, cmd: list[str]) -> None:
        for key, _label in self.fields:
            value = get_flag_value(cmd, key)
            self.inputs[key].setText(value if value is not None else "")


class _CoreOpsPanel(QWidget):
    def __init__(
        self,
        *,
        title: str,
        subtitle: str,
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__()
        _ = title, subtitle
        self._append_log = append_log
        self._run_controller = run_controller
        self._action_buttons: list[QPushButton] = []
        self._cancel_button: QPushButton | None = None
        self._latest_run_artifact_path: Path | None = None
        self._latest_run_metadata: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SHELL)

        self.form_card, self.form_layout = _build_card("Workflow Inputs")
        layout.addWidget(self.form_card)

        self.output_card, output_layout = _build_card("Run Output")
        self.output_panel = QtRunOutputPanel()
        self.status_label = self.output_panel.status_label
        self._update_chip_state("success")
        self.output = self.output_panel.summary_output
        self.raw_output = self.output_panel.raw_output
        self.output_panel.explain_button.clicked.connect(self._show_failure_explanation)
        self.output_panel.explain_button.setToolTip("show a plain-language explanation of why this run failed")
        output_layout.addWidget(self.output_panel)
        layout.addWidget(self.output_card, 1)
        self.output_card.hide()
        layout.addStretch(1)

    def _register_action_button(self, button: QPushButton, *, is_cancel: bool = False) -> None:
        self._action_buttons.append(button)
        if is_cancel:
            self._cancel_button = button
            button.setEnabled(False)

    def _update_chip_state(self, state: str) -> None:
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_output(self, *, title: str, text: str, log_message: str) -> None:
        self.output_card.show()
        self.status_label.setText(title)
        self._update_chip_state("success")
        self.output_panel.set_summary_text(text)
        self.output_panel.show_summary_tab()
        self._append_log(log_message)

    def _append_output_line(self, line: str) -> None:
        self.output_card.show()
        self.output_panel.append_summary_line(line)

    def _append_output_chunk(self, chunk: str) -> None:
        self.output_panel.append_raw_text(chunk)

    def _append_output_and_log(self, line: str) -> None:
        self._append_output_line(line)
        self._append_log(line)

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        if active:
            self.status_label.setText(status_text or "Running command...")
            self._update_chip_state("running")
            self.output_panel.explain_button.setEnabled(False)
            self.output_panel.show_raw_tab()
        else:
            if is_error:
                self.status_label.setText(status_text or "Command failed.")
                self._update_chip_state("error")
            else:
                self.status_label.setText(status_text or "Ready.")
                self._update_chip_state("success")
            if self._latest_run_metadata is not None:
                self.output_panel.set_summary_text(build_run_summary_text(self._latest_run_metadata))
                self.output_panel.explain_button.setEnabled(has_failure_details(self._latest_run_metadata))
                if is_error and has_failure_details(self._latest_run_metadata):
                    self.output_panel.show_summary_tab()

        for button in self._action_buttons:
            if button is self._cancel_button:
                button.setEnabled(active)
            else:
                button.setEnabled(not active)

    def _build_hooks(self, *, on_teleop_ready: Callable[[], None] | None = None) -> RunUiHooks:
        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._handle_runtime_line,
            append_output_chunk=self._append_output_chunk,
            on_teleop_ready=on_teleop_ready,
            on_artifact_written=self._remember_run_artifact,
        )

    def _handle_runtime_line(self, line: str) -> None:
        _ = line

    def _remember_run_artifact(self, artifact_path: Path) -> None:
        self._latest_run_artifact_path = Path(artifact_path)
        self._latest_run_metadata = self._read_artifact_metadata(self._latest_run_artifact_path)

    def _read_artifact_metadata(self, run_path: Path | None) -> dict[str, Any] | None:
        if run_path is None:
            return None
        metadata_path = Path(run_path) / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["_run_path"] = str(run_path)
        payload["_metadata_path"] = str(metadata_path)
        return payload

    def _show_failure_explanation(self) -> None:
        if self._latest_run_metadata is None:
            return
        self._show_text_dialog(
            title="Failure Explanation",
            text=build_failure_explanation_text(
                self._latest_run_metadata,
                run_path=self._latest_run_artifact_path,
            ),
            wrap_mode="word",
        )

    def _show_launch_summary(
        self,
        *,
        heading: str,
        command_label: str,
        cmd: list[str],
        preflight_title: str,
        preflight_checks: list[tuple[str, str, str]],
        warning_detail: str | None = None,
    ) -> None:
        self.output_card.show()
        self._latest_run_artifact_path = None
        self._latest_run_metadata = None
        self.output_panel.explain_button.setEnabled(False)
        self.output_panel.clear_raw()
        text = (
            f"{command_label}\n\n"
            f"{format_command_for_dialog(cmd)}\n\n"
            f"{summarize_checks(preflight_checks, title=preflight_title)}"
        )
        if warning_detail:
            text += f"\n\n{warning_detail}"
        text += "\n\nStreaming output is available in the Raw Transcript tab."
        self.output_panel.set_summary_text(text)
        self.status_label.setText(heading)
        self._update_chip_state("success")
        self.output_panel.show_raw_tab()

    def _handle_launch_rejection(self, *, title: str, message: str, log_message: str) -> None:
        self.status_label.setText(title)
        self._update_chip_state("error")
        self._append_output_and_log(message)
        self._append_log(log_message)
        self.output_panel.show_summary_tab()

    def _dialog_parent(self) -> QWidget | None:
        parent = self.window()
        return parent if isinstance(parent, QWidget) else None

    def _show_text_dialog(
        self,
        *,
        title: str,
        text: str,
        copy_text: str | None = None,
        wrap_mode: str = "word",
    ) -> None:
        show_text_dialog(
            parent=self._dialog_parent(),
            title=title,
            text=text,
            copy_text=copy_text,
            wrap_mode=wrap_mode,
        )

    def _ask_editable_command_dialog(
        self,
        *,
        title: str,
        command_argv: list[str],
        intro_text: str,
        confirm_label: str,
    ) -> list[str] | None:
        return ask_editable_command_dialog(
            parent=self._dialog_parent(),
            title=title,
            command_argv=command_argv,
            intro_text=intro_text,
            confirm_label=confirm_label,
            cancel_label="Cancel",
        )

    def _ask_text_dialog_with_actions(
        self,
        *,
        title: str,
        text: str,
        actions: list[tuple[str, str]],
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        wrap_mode: str = "word",
    ) -> str:
        return ask_text_dialog_with_actions(
            parent=self._dialog_parent(),
            title=title,
            text=text,
            actions=actions,
            confirm_label=confirm_label,
            cancel_label=cancel_label,
            wrap_mode=wrap_mode,
        )

    def _confirm_preflight_review(
        self,
        *,
        title: str,
        checks: list[tuple[str, str, str]],
    ) -> bool:
        summary = summarize_checks(checks, title=title)
        if has_failures(checks):
            prompt = summary + "\n\nFAIL items detected.\nClick Confirm to continue anyway, or Cancel to stop."
            dialog_title = "Preflight Failures"
        else:
            prompt = summary + "\n\nPreflight complete.\nClick Confirm to continue, or Cancel to stop."
            dialog_title = "Preflight Review"
        return ask_text_dialog(
            parent=self._dialog_parent(),
            title=dialog_title,
            text=prompt,
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        )

    def _cancel_run(self) -> None:
        ok, message = self._run_controller.cancel_active_run()
        if not ok:
            self._set_output(title="Cancel Unavailable", text=message, log_message="Cancel request was rejected.")

    def _run_port_scan_dialog(
        self,
        *,
        title: str,
        current_follower: str,
        current_leader: str,
        apply_scope_label: str,
    ) -> tuple[str | None, str | None]:
        entries = scan_robot_serial_ports()
        report = format_robot_port_scan(entries)
        if not entries:
            self._show_text_dialog(title=title, text=report, wrap_mode="word")
            self._append_log("Robot port scan: no candidate ports found.")
            return None, None

        follower_guess, leader_guess = suggest_follower_leader_ports(
            entries,
            current_follower=current_follower,
            current_leader=current_leader,
        )
        self._append_log(
            "Robot port scan detected: "
            + ", ".join(str(item.get("path", "")) for item in entries)
        )
        if follower_guess and leader_guess:
            report += (
                "\n\nDetected candidate motor-controller ports.\n\n"
                f"Set follower -> {follower_guess}\n"
                f"Set leader -> {leader_guess}\n\n"
                f"Click Apply Detected Ports to use these as {apply_scope_label} defaults now."
            )
            action = self._ask_text_dialog_with_actions(
                title=title,
                text=report,
                actions=[("apply_ports", "Apply Detected Ports")],
                confirm_label="Close",
                cancel_label="Close",
                wrap_mode="word",
            )
            if action == "apply_ports":
                return follower_guess, leader_guess
            return None, None

        self._show_text_dialog(title=title, text=report, wrap_mode="word")
        return None, None
