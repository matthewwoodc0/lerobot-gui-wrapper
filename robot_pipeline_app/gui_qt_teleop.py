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
from .gui_forms import (
    build_deploy_request_and_command,
    build_record_request_and_command,
    build_teleop_request_and_command,
)
from .gui_qt_camera import QtCameraWorkspace
from .gui_qt_dialogs import ask_editable_command_dialog, ask_text_dialog, ask_text_dialog_with_actions, show_text_dialog
from .gui_qt_runtime_helpers import QtRunHelperDialog
from .repo_utils import normalize_repo_id, repo_name_from_repo_id, repo_name_only, suggest_eval_prefixed_repo_id
from .run_controller_service import ManagedRunController, RunUiHooks
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .workflows import move_recorded_dataset

from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card, _count_preflight_failures

class TeleopOpsPanel(_CoreOpsPanel):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Teleop",
            subtitle="Build teleop commands, run preflight checks, and launch or cancel live sessions from the GUI.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config
        self._teleop_status_detail = "Ready to validate or launch a live teleop session."
        self.run_helper_dialog = QtRunHelperDialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            mode_title="Teleop",
            on_cancel=self._cancel_run,
            show_episode_controls=False,
        )
        self.camera_preview = QtCameraWorkspace(
            config=self.config,
            append_log=self._append_log,
            title="Teleop Camera Workspace",
        )
        self.teleop_snapshot_card = self._build_snapshot_card()
        self.teleop_overview = QWidget()
        overview_layout = QGridLayout(self.teleop_overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setHorizontalSpacing(18)
        overview_layout.setVerticalSpacing(18)
        overview_layout.setColumnStretch(0, 1)
        overview_layout.setColumnStretch(1, 2)
        overview_layout.addWidget(self.teleop_snapshot_card, 0, 0)
        overview_layout.addWidget(self.camera_preview, 0, 1)
        root_layout = self.layout()
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertWidget(1, self.teleop_overview)

        form = _InputGrid(self.form_layout)

        self.follower_port_input = QLineEdit(str(config.get("follower_port", "")))
        form.add_field("Follower port", self.follower_port_input)

        self.leader_port_input = QLineEdit(str(config.get("leader_port", "")))
        form.add_field("Leader port", self.leader_port_input)

        self.follower_id_input = QLineEdit(str(config.get("follower_robot_id", "red4")).strip() or "red4")
        form.add_field("Follower id", self.follower_id_input)

        self.leader_id_input = QLineEdit(str(config.get("leader_robot_id", "white")).strip() or "white")
        form.add_field("Leader id", self.leader_id_input)

        self.control_fps_input = QLineEdit(str(config.get("teleop_control_fps", "")).strip())
        self.control_fps_input.setPlaceholderText("optional")
        form.add_field("Control FPS", self.control_fps_input)

        self.teleop_advanced_toggle = QCheckBox("Advanced command options")
        self.teleop_advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.form_layout.addWidget(self.teleop_advanced_toggle)

        self.teleop_advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Teleop Options",
            fields=[
                ("robot.type", "Robot type"),
                ("robot.port", "Follower port"),
                ("robot.id", "Follower robot id"),
                ("robot.cameras", "Robot cameras JSON"),
                ("teleop.type", "Teleop type"),
                ("teleop.port", "Leader port"),
                ("teleop.id", "Leader robot id"),
                ("control.fps", "Control FPS"),
            ],
        )
        self.teleop_advanced_panel.hide()
        self.form_layout.addWidget(self.teleop_advanced_panel)

        actions = QHBoxLayout()
        run_button = QPushButton("Run Teleop")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.run_teleop)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

        preview_button = QPushButton("Preview Command")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)
        self._register_action_button(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        scan_ports_button = QPushButton("Scan Robot Ports")
        scan_ports_button.clicked.connect(self.scan_robot_ports)
        actions.addWidget(scan_ports_button)
        self._register_action_button(scan_ports_button)

        help_button = QPushButton("Teleop Help")
        help_button.clicked.connect(self.show_teleop_help)
        actions.addWidget(help_button)
        self._register_action_button(help_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

        for widget in (
            self.follower_port_input,
            self.leader_port_input,
            self.follower_id_input,
            self.leader_id_input,
            self.control_fps_input,
        ):
            widget.textChanged.connect(self._refresh_session_snapshot)
        self._refresh_session_snapshot()

    def _build_snapshot_card(self) -> QFrame:
        card, layout = _build_card("Teleop Snapshot")

        self.connection_summary_label = QLabel("")
        self.connection_summary_label.setWordWrap(True)
        layout.addWidget(self.connection_summary_label)

        self.camera_summary_label = QLabel("")
        self.camera_summary_label.setObjectName("MutedLabel")
        self.camera_summary_label.setWordWrap(True)
        layout.addWidget(self.camera_summary_label)

        self.command_summary_label = QLabel("")
        self.command_summary_label.setObjectName("MutedLabel")
        self.command_summary_label.setWordWrap(True)
        layout.addWidget(self.command_summary_label)

        self.status_summary_label = QLabel("")
        self.status_summary_label.setObjectName("MutedLabel")
        self.status_summary_label.setWordWrap(True)
        layout.addWidget(self.status_summary_label)

        layout.addStretch(1)
        return card

    def _refresh_session_snapshot(self) -> None:
        follower_port = self.follower_port_input.text().strip() or "(unset)"
        leader_port = self.leader_port_input.text().strip() or "(unset)"
        follower_id = self.follower_id_input.text().strip() or "(unset)"
        leader_id = self.leader_id_input.text().strip() or "(unset)"
        control_fps = self.control_fps_input.text().strip() or "auto"

        self.connection_summary_label.setText(
            "Follower robot\n"
            f"{follower_port}  |  id: {follower_id}\n\n"
            "Leader robot\n"
            f"{leader_port}  |  id: {leader_id}"
        )
        self.camera_summary_label.setText(f"Runtime cameras\n{camera_mapping_summary(self.config)}")
        self.command_summary_label.setText(
            "Session settings\n"
            f"Control FPS: {control_fps}\n"
            f"Workspace: {get_lerobot_dir(self.config)}"
        )
        self.status_summary_label.setText(f"Live status\n{self._teleop_status_detail}")

    def show_teleop_help(self) -> None:
        self._show_text_dialog(
            title="Teleop Help",
            text=(
                "Use this page to validate the teleop command, run preflight, and launch or cancel the live session.\n\n"
                "Current teleop session settings:\n"
                + f"Follower: {self.follower_port_input.text().strip() or '(unset)'}\n"
                + f"Leader: {self.leader_port_input.text().strip() or '(unset)'}\n"
                + f"Follower id: {self.follower_id_input.text().strip() or '(unset)'}\n"
                + f"Leader id: {self.leader_id_input.text().strip() or '(unset)'}\n"
                + f"Control FPS: {self.control_fps_input.text().strip() or 'auto'}"
            ),
            wrap_mode="word",
        )
        self.run_helper_dialog.show()
        self.run_helper_dialog.raise_()
        self.run_helper_dialog.activateWindow()

    def _build(self) -> tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]:
        arg_overrides: dict[str, str] | None = None
        custom_args_raw = ""
        if self.teleop_advanced_toggle.isChecked():
            arg_overrides, custom_args_raw = self.teleop_advanced_panel.build_overrides()
        return build_teleop_request_and_command(
            config=self.config,
            follower_port_raw=self.follower_port_input.text(),
            leader_port_raw=self.leader_port_input.text(),
            follower_id_raw=self.follower_id_input.text(),
            leader_id_raw=self.leader_id_input.text(),
            control_fps_raw=self.control_fps_input.text(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def _toggle_advanced_options(self, checked: bool) -> None:
        if checked:
            req, cmd, _updated, error = self._build()
            if error is None and req is not None and cmd is not None:
                self.teleop_advanced_panel.seed_from_command(cmd)
            self.teleop_advanced_panel.show()
        else:
            self.teleop_advanced_panel.hide()

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)
        if not active:
            self.run_helper_dialog.finish_run(
                status_text=status_text or ("Teleop failed." if is_error else "Teleop completed.")
            )
        self._teleop_status_detail = status_text or ("Running command..." if active else "Ready to validate or launch a live teleop session.")
        self._refresh_session_snapshot()

    def _append_runtime_line(self, line: str) -> None:
        self.run_helper_dialog.handle_output_line(line)

    def _build_hooks(self, *, on_teleop_ready: Callable[[], None] | None = None) -> RunUiHooks:
        def _mark_ready() -> None:
            self._mark_teleop_ready()
            if on_teleop_ready is not None:
                on_teleop_ready()

        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_runtime_line,
            append_output_chunk=self._append_output_chunk,
            on_teleop_ready=_mark_ready,
            on_artifact_written=self._remember_run_artifact,
        )

    def _mark_teleop_ready(self) -> None:
        self.status_label.setText("Teleop connected.")
        self._append_output_and_log("Teleop session reported ready.")
        self.run_helper_dialog.set_teleop_ready(True)
        self._teleop_status_detail = "Teleop connected and the runtime helper is ready."
        self._refresh_session_snapshot()

    def preview_command(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Teleop preview failed validation.")
            return
        summary = (
            f"Follower: {req.follower_port} ({req.follower_id})\n"
            f"Leader: {req.leader_port} ({req.leader_id})\n"
            f"Control FPS: {req.control_fps if req.control_fps is not None else 'auto'}\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._append_log("Teleop preview built.")
        self._show_text_dialog(title="Teleop Command", text=summary, wrap_mode="word")

    def run_preflight(self) -> None:
        req, _cmd, updated, error = self._build()
        if error or req is None or updated is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Teleop preflight failed validation.")
            return
        run_config = {**self.config, **updated}
        checks = run_preflight_for_teleop(run_config, control_fps=req.control_fps)
        self._append_log("Teleop preflight ran.")
        self._show_text_dialog(
            title="Teleop Preflight",
            text=summarize_checks(checks, title="Teleop Preflight"),
            wrap_mode="char",
        )

    def scan_robot_ports(self) -> None:
        follower_guess, leader_guess = self._run_port_scan_dialog(
            title="Robot Port Scan",
            current_follower=self.follower_port_input.text().strip(),
            current_leader=self.leader_port_input.text().strip(),
            apply_scope_label="teleop",
        )
        if not follower_guess or not leader_guess:
            return
        self.follower_port_input.setText(follower_guess)
        self.leader_port_input.setText(leader_guess)
        self.config["follower_port"] = follower_guess
        self.config["leader_port"] = leader_guess
        save_config(self.config, quiet=True)
        self._append_output_and_log(f"Applied scanned ports: follower={follower_guess}, leader={leader_guess}")
        self._refresh_session_snapshot()

    def run_teleop(self) -> None:
        req, cmd, updated, error = self._build()
        if error or req is None or cmd is None or updated is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Teleop launch failed validation.")
            return

        run_config = {**self.config, **updated}
        checks = run_preflight_for_teleop(run_config, control_fps=req.control_fps)
        if not self._confirm_preflight_review(title="Teleop Preflight", checks=checks):
            self._append_log("Teleop canceled after preflight review.")
            return

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Teleop Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the teleop command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Teleop",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited teleop command from command editor.")
        cmd = editable_cmd

        warning_detail = None
        if any(str(level).strip().upper() == "WARN" for level, _name, _detail in checks):
            warning_detail = "Warnings were detected. The workflow continues automatically when there are no FAIL checks."
        self._show_launch_summary(
            heading="Launching teleop run...",
            command_label="Teleop command",
            cmd=cmd,
            preflight_title="Teleop Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self.config.update(updated)
        self._append_log("Teleop launch starting.")
        self.run_helper_dialog.start_run(run_mode="teleop")

        effective_follower_port = get_flag_value(cmd, "robot.port") or req.follower_port
        effective_leader_port = get_flag_value(cmd, "teleop.port") or req.leader_port
        effective_follower_id = get_flag_value(cmd, "robot.id") or resolve_follower_robot_id(run_config)
        effective_leader_id = get_flag_value(cmd, "teleop.id") or resolve_leader_robot_id(run_config)

        def after_teleop(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Teleop canceled.", False)
                self._append_output_and_log("Teleop session canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Teleop failed.", True)
                self._append_output_and_log(f"Teleop session failed with exit code {return_code}.")
                return
            self._set_running(False, "Teleop completed.", False)
            self._append_output_and_log("Teleop session completed.")

        teleop_context = {
            "follower_port": effective_follower_port,
            "follower_id": effective_follower_id,
            "leader_port": effective_leader_port,
            "leader_id": effective_leader_id,
        }
        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_teleop,
            run_mode="teleop",
            preflight_checks=checks,
            artifact_context=teleop_context,
        )
        if not ok and message:
            self._handle_launch_rejection(title="Teleop Unavailable", message=message, log_message="Teleop launch was rejected.")
