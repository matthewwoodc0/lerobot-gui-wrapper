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
from .repo_utils import next_available_dataset_name, normalize_repo_id, repo_name_from_repo_id, repo_name_only, suggest_eval_prefixed_repo_id
from .run_controller_service import ManagedRunController, RunUiHooks
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .workflows import move_recorded_dataset

from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card, _count_preflight_failures

class RecordOpsPanel(_CoreOpsPanel):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Record",
            subtitle="Build record commands, run preflight checks, and launch or cancel recording workflows.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config
        self.run_helper_dialog = QtRunHelperDialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            mode_title="Record",
            on_cancel=self._cancel_run,
        )
        self.camera_preview = QtCameraWorkspace(config=self.config, append_log=self._append_log)
        root_layout = self.layout()
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertWidget(2, self.camera_preview)

        form = _InputGrid(self.form_layout)

        default_dataset = str(config.get("last_dataset_repo_id", "")).strip() or str(config.get("last_dataset_name", "dataset_1"))
        self.dataset_input = QLineEdit(default_dataset)
        self.dataset_input.setPlaceholderText("owner/dataset_name or dataset_name")
        form.add_field("Dataset", self.dataset_input)

        self.dataset_root_input = QLineEdit(str(config.get("record_data_dir", "")))
        form.add_field("Dataset root", self.dataset_root_input)

        self.task_input = QLineEdit(str(config.get("last_task", DEFAULT_TASK)) or DEFAULT_TASK)
        form.add_field("Task", self.task_input)

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 10000)
        self.episodes_input.setValue(20)
        form.add_field("Episodes", self.episodes_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(20)
        form.add_field("Episode time (s)", self.duration_input)

        self.target_hz_input = QLineEdit(str(config.get("record_target_hz", "")).strip())
        self.target_hz_input.setPlaceholderText("optional")
        form.add_field("Target Hz", self.target_hz_input)

        self.upload_checkbox = QCheckBox("Upload to Hugging Face after record")
        self.upload_checkbox.setChecked(False)
        self.form_layout.addWidget(self.upload_checkbox)

        self.record_advanced_toggle = QCheckBox("Advanced command options")
        self.record_advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.form_layout.addWidget(self.record_advanced_toggle)

        self.record_advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Record Options",
            fields=[
                ("robot.type", "Robot type"),
                ("robot.port", "Follower port"),
                ("robot.id", "Follower robot id"),
                ("teleop.type", "Teleop type"),
                ("teleop.port", "Leader port"),
                ("teleop.id", "Leader robot id"),
            ],
        )
        self.record_advanced_panel.hide()
        self.form_layout.addWidget(self.record_advanced_panel)

        actions = QHBoxLayout()
        run_button = QPushButton("Run Record")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.run_record)
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

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("DangerButton")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)

        actions.addStretch(1)
        self.form_layout.addLayout(actions)

    def _build(self) -> tuple[Any | None, list[str] | None, str | None]:
        arg_overrides = None
        custom_args_raw = ""
        if self.record_advanced_toggle.isChecked():
            arg_overrides, custom_args_raw = self.record_advanced_panel.build_overrides()
        return build_record_request_and_command(
            config=self.config,
            dataset_input=self.dataset_input.text(),
            episodes_raw=str(self.episodes_input.value()),
            duration_raw=str(self.duration_input.value()),
            task_raw=self.task_input.text(),
            dataset_dir_raw=self.dataset_root_input.text(),
            upload_enabled=self.upload_checkbox.isChecked(),
            target_hz_raw=self.target_hz_input.text(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def _toggle_advanced_options(self, checked: bool) -> None:
        if checked:
            req, cmd, error = self._build()
            if error is None and req is not None and cmd is not None:
                self.record_advanced_panel.seed_from_command(cmd)
            self.record_advanced_panel.show()
        else:
            self.record_advanced_panel.hide()

    def _advance_dataset_name(self) -> None:
        """Auto-iterate the dataset name field to the next available name and show it in the UI."""
        current = self.dataset_input.text().strip()
        if not current:
            return
        hf_username = str(self.config.get("hf_username", "")).strip()
        dataset_root_text = self.dataset_root_input.text().strip() or str(self.config.get("record_data_dir", ""))
        dataset_root = Path(dataset_root_text).expanduser() if dataset_root_text else None
        base_name = repo_name_from_repo_id(current)
        iterated = next_available_dataset_name(
            base_name=base_name, hf_username=hf_username, dataset_root=dataset_root
        )
        if iterated != base_name:
            new_value = normalize_repo_id(hf_username, iterated) if hf_username else iterated
            self.dataset_input.setText(new_value)
            self._append_log(f"Dataset name '{base_name}' already exists — advanced to '{iterated}'.")

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)

    def refresh_from_config(self) -> None:
        self.dataset_root_input.setText(str(self.config.get("record_data_dir", "")).strip())
        self.target_hz_input.setText(str(self.config.get("record_target_hz", "")).strip())
        self._advance_dataset_name()

    def preview_command(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build record command.",
                log_message="Record preview failed validation.",
            )
            return
        summary = (
            f"Record target: {req.dataset_repo_id}\n"
            f"Episodes: {req.num_episodes}\n"
            f"Episode time: {req.episode_time_s}s\n"
            f"Upload after record: {req.upload_after_record}\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._append_log(f"Record preview built for {req.dataset_repo_id}.")
        self._show_text_dialog(title="Record Command", text=summary, wrap_mode="word")

    def run_preflight(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build record command.",
                log_message="Record preflight failed validation.",
            )
            return
        checks = run_preflight_for_record(
            config=self.config,
            dataset_root=req.dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=req.episode_time_s,
            dataset_repo_id=req.dataset_repo_id,
        )
        self._append_log(f"Record preflight ran for {req.dataset_repo_id}.")
        self._show_text_dialog(
            title="Record Preflight",
            text=summarize_checks(checks, title="Record Preflight"),
            wrap_mode="char",
        )

    def scan_robot_ports(self) -> None:
        follower_guess, leader_guess = self._run_port_scan_dialog(
            title="Robot Port Scan",
            current_follower=str(self.config.get("follower_port", "")),
            current_leader=str(self.config.get("leader_port", "")),
            apply_scope_label="record",
        )
        if not follower_guess or not leader_guess:
            return
        self.config["follower_port"] = follower_guess
        self.config["leader_port"] = leader_guess
        if self.record_advanced_toggle.isChecked():
            self.record_advanced_panel.inputs["robot.port"].setText(follower_guess)
            self.record_advanced_panel.inputs["teleop.port"].setText(leader_guess)
        save_config(self.config, quiet=True)
        self._append_output_and_log(
            f"Applied scanned record defaults: follower={follower_guess}, leader={leader_guess}"
        )

    def run_record(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build record command.", log_message="Record launch failed validation.")
            return

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Record Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the record command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Record",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited record command from command editor.")
        cmd = editable_cmd

        effective_repo_id = normalize_repo_id(
            str(self.config.get("hf_username", "")),
            get_flag_value(cmd, "dataset.repo_id") or req.dataset_repo_id,
        )
        effective_dataset_name = repo_name_from_repo_id(effective_repo_id)
        effective_dataset_root = req.dataset_root
        dataset_root_text = (get_flag_value(cmd, "dataset.root") or "").strip()
        if dataset_root_text:
            effective_dataset_root = Path(dataset_root_text).expanduser()
        episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.num_episodes)
        duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.episode_time_s)
        try:
            effective_num_episodes = int(str(episodes_text).strip())
            effective_episode_time = int(str(duration_text).strip())
        except ValueError:
            self._set_output(
                title="Validation Error",
                text="Edited command must keep episodes and episode time as integers.",
                log_message="Record launch rejected due to invalid edited command values.",
            )
            return
        if effective_num_episodes <= 0 or effective_episode_time <= 0:
            self._set_output(
                title="Validation Error",
                text="Edited command must keep episodes and episode time greater than zero.",
                log_message="Record launch rejected due to non-positive edited command values.",
            )
            return

        checks = run_preflight_for_record(
            config=self.config,
            dataset_root=effective_dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=effective_episode_time,
            dataset_repo_id=effective_repo_id,
        )
        if not self._confirm_preflight_review(title="Record Preflight", checks=checks):
            self._append_log("Record canceled after preflight review.")
            return

        warning_detail = None
        if any(str(level).strip().upper() == "WARN" for level, _name, _detail in checks):
            warning_detail = "Warnings were detected. The workflow continues automatically when there are no FAIL checks."
        self._show_launch_summary(
            heading="Launching record run...",
            command_label="Record command",
            cmd=cmd,
            preflight_title="Record Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self._append_log(f"Record launch starting for {effective_repo_id}.")
        self.run_helper_dialog.start_run(
            run_mode="record",
            expected_episodes=effective_num_episodes,
        )
        self.run_helper_dialog.show()
        self.run_helper_dialog.raise_()
        self.run_helper_dialog.activateWindow()

        def after_upload(upload_code: int, upload_canceled: bool) -> None:
            if upload_canceled:
                self._set_running(False, "Upload canceled.", False)
                self._append_output_and_log("Hugging Face dataset upload canceled.")
                return
            if upload_code != 0:
                self._set_running(False, "Upload failed.", True)
                self._append_output_and_log(f"Hugging Face dataset upload failed with exit code {upload_code}.")
                return
            self._set_running(False, "Record + upload completed.", False)
            self._append_output_and_log(f"Hugging Face dataset upload completed: {effective_repo_id}")

        def after_record(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Record canceled.", False)
                self._append_output_and_log("Record run canceled. Upload was skipped.")
                return
            if return_code != 0:
                self._set_running(False, "Record failed.", True)
                self._append_output_and_log(f"Record run failed with exit code {return_code}.")
                return

            active_dataset = move_recorded_dataset(
                lerobot_dir=get_lerobot_dir(self.config),
                dataset_name=effective_dataset_name,
                dataset_root=effective_dataset_root,
                log=self._append_output_and_log,
            )
            self.config["record_data_dir"] = str(effective_dataset_root)
            self.config["last_dataset_name"] = effective_dataset_name
            self.config["last_dataset_repo_id"] = effective_repo_id
            self._advance_dataset_name()

            if not req.upload_after_record:
                self._set_running(False, "Record completed.", False)
                self._append_output_and_log(f"Recording completed for {effective_repo_id}.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                effective_repo_id,
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]
            self._set_running(False, "Record completed. Starting upload...", False)
            self._append_output_and_log(f"Starting Hugging Face dataset upload: {effective_repo_id}")
            upload_ok, upload_error = self._run_controller.run_process_async(
                cmd=upload_cmd,
                cwd=get_lerobot_dir(self.config),
                hooks=self._build_hooks(),
                complete_callback=after_upload,
                run_mode="upload",
                artifact_context={"dataset_repo_id": effective_repo_id},
            )
            if not upload_ok and upload_error:
                self._handle_launch_rejection(
                    title="Upload Unavailable",
                    message=upload_error,
                    log_message="Record upload follow-up could not start.",
                )

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_record,
            expected_episodes=effective_num_episodes,
            expected_seconds=effective_num_episodes * effective_episode_time,
            run_mode="record",
            preflight_checks=checks,
            artifact_context={"dataset_repo_id": effective_repo_id},
        )
        if not ok and message:
            self._handle_launch_rejection(title="Record Unavailable", message=message, log_message="Record launch was rejected.")

    def open_run_helper(self) -> None:
        self.run_helper_dialog.show()
        self.run_helper_dialog.raise_()
        self.run_helper_dialog.activateWindow()

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)
        if not active:
            self.run_helper_dialog.finish_run(
                status_text=status_text or ("Record failed." if is_error else "Record completed.")
            )

    def _handle_runtime_line(self, line: str) -> None:
        self.run_helper_dialog.handle_output_line(line)
