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
    QTreeWidget,
    QTreeWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

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
from .gui_qt_camera import QtDualCameraPreview
from .gui_input_help import keyboard_input_help_text, keyboard_input_help_title
from .gui_qt_dialogs import ask_editable_command_dialog, ask_text_dialog, ask_text_dialog_with_actions, show_text_dialog
from .gui_qt_runtime_helpers import QtRunHelperDialog
from .repo_utils import normalize_repo_id, repo_name_from_repo_id, repo_name_only, suggest_eval_prefixed_repo_id
from .run_controller_service import ManagedRunController, RunUiHooks
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .workflows import move_recorded_dataset


def _build_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("SectionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)

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
        self._grid.setHorizontalSpacing(14)
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
        self.fields = fields
        self.inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
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
        form.add_field("Custom args (raw)", self.custom_args_input)

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
        self._append_log = append_log
        self._run_controller = run_controller
        self._action_buttons: list[QPushButton] = []
        self._cancel_button: QPushButton | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        hero, hero_layout = _build_card(title)
        hero.setObjectName("SectionHero")
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        hero_layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setObjectName("MutedLabel")
        hero_layout.addWidget(subtitle_label)
        layout.addWidget(hero)

        self.form_card, self.form_layout = _build_card("Workflow Inputs")
        layout.addWidget(self.form_card)

        self.output_card, output_layout = _build_card("Run Output")
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(260)
        output_layout.addWidget(self.status_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        output_layout.addWidget(self.output)
        layout.addWidget(self.output_card, 1)

    def _register_action_button(self, button: QPushButton, *, is_cancel: bool = False) -> None:
        self._action_buttons.append(button)
        if is_cancel:
            self._cancel_button = button
            button.setEnabled(False)

    def _set_output(self, *, title: str, text: str, log_message: str) -> None:
        self.status_label.setText(title)
        self.output.setPlainText(text)
        self._append_log(log_message)

    def _append_output_line(self, line: str) -> None:
        self.output.appendPlainText(str(line))

    def _append_output_and_log(self, line: str) -> None:
        self._append_output_line(line)
        self._append_log(line)

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        if active:
            self.status_label.setText(status_text or "Running command...")
        else:
            if is_error:
                self.status_label.setText(status_text or "Command failed.")
            else:
                self.status_label.setText(status_text or "Ready.")

        for button in self._action_buttons:
            if button is self._cancel_button:
                button.setEnabled(active)
            else:
                button.setEnabled(not active)

    def _build_hooks(self, *, on_teleop_ready: Callable[[], None] | None = None) -> RunUiHooks:
        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_output_line,
            on_teleop_ready=on_teleop_ready,
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
        text = (
            f"{command_label}\n\n"
            f"{format_command_for_dialog(cmd)}\n\n"
            f"{summarize_checks(preflight_checks, title=preflight_title)}"
        )
        if warning_detail:
            text += f"\n\n{warning_detail}"
        text += "\n\nStreaming output will appear below."
        self.output.setPlainText(text)
        self.status_label.setText(heading)

    def _handle_launch_rejection(self, *, title: str, message: str, log_message: str) -> None:
        self.status_label.setText(title)
        self._append_output_and_log(message)
        self._append_log(log_message)

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


class _QtModelUploadDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        default_local_model: str,
        default_owner: str,
        default_repo_name: str,
        model_options: list[str],
    ) -> None:
        super().__init__(parent)
        self.result_request: dict[str, Any] | None = None
        self.result_settings: dict[str, Any] | None = None
        self.setWindowTitle("Upload Model to Hugging Face")
        self.setModal(True)
        self.resize(900, 420)
        self.setMinimumSize(760, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QLabel(
            "Upload sends your local model/checkpoint folder to a Hugging Face model repository.\n"
            "Use this for backups and sharing of trained artifacts. It does not run deploy or eval."
        )
        intro.setObjectName("MutedLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        local_label = QLabel("Local model folder")
        local_label.setObjectName("FormLabel")
        grid.addWidget(local_label, 0, 0)
        self.local_model_input = QLineEdit(default_local_model)
        grid.addWidget(self.local_model_input, 0, 1, 1, 2)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._choose_local_model)
        grid.addWidget(browse_button, 0, 3)

        options_label = QLabel("Local model candidates")
        options_label.setObjectName("FormLabel")
        grid.addWidget(options_label, 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        grid.addWidget(self.model_combo, 1, 1, 1, 2)
        refresh_button = QPushButton("Refresh Models")
        refresh_button.clicked.connect(lambda: self.set_model_options(model_options))
        grid.addWidget(refresh_button, 1, 3)

        owner_label = QLabel("HF owner")
        owner_label.setObjectName("FormLabel")
        grid.addWidget(owner_label, 2, 0)
        self.owner_input = QLineEdit(default_owner)
        grid.addWidget(self.owner_input, 2, 1)

        repo_label = QLabel("HF model name")
        repo_label.setObjectName("FormLabel")
        grid.addWidget(repo_label, 2, 2)
        self.repo_name_input = QLineEdit(default_repo_name)
        grid.addWidget(self.repo_name_input, 2, 3)

        self.skip_if_exists_checkbox = QCheckBox("Skip upload when remote model already exists")
        self.skip_if_exists_checkbox.setChecked(True)
        layout.addWidget(self.skip_if_exists_checkbox)

        self.status_label = QLabel("Choose a local model folder, then preview or run artifact upload.")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        parity_button = QPushButton("Check Local/Remote Parity")
        parity_button.clicked.connect(self.check_parity)
        button_row.addWidget(parity_button)

        preview_button = QPushButton("Preview Upload Command")
        preview_button.clicked.connect(self.preview_upload_command)
        button_row.addWidget(preview_button)

        button_row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)

        run_button = QPushButton("Upload Model")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.accept_for_run)
        button_row.addWidget(run_button)
        layout.addLayout(button_row)

        self.model_combo.currentTextChanged.connect(self._sync_combo_selection)
        self.set_model_options(model_options)

    def set_model_options(self, model_options: list[str]) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(model_options)
        current = self.local_model_input.text().strip()
        if current:
            index = self.model_combo.findText(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        elif model_options:
            self.model_combo.setCurrentIndex(0)
            self.local_model_input.setText(model_options[0])
        self.model_combo.blockSignals(False)

    def _choose_local_model(self) -> None:
        current = self.local_model_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select local model folder", current)
        if selected:
            self.local_model_input.setText(selected)
            if not self.repo_name_input.text().strip():
                self.repo_name_input.setText(Path(selected).name)

    def _sync_combo_selection(self, value: str) -> None:
        selected = str(value or "").strip()
        if not selected:
            return
        self.local_model_input.setText(selected)
        if not self.repo_name_input.text().strip():
            self.repo_name_input.setText(Path(selected).name)

    def _build_request(self) -> tuple[dict[str, Any] | None, str | None]:
        request, error_text = build_model_upload_request(
            local_model_raw=self.local_model_input.text(),
            owner_raw=self.owner_input.text(),
            repo_name_raw=self.repo_name_input.text(),
        )
        if request is not None:
            self.repo_name_input.setText(str(request.get("repo_name", self.repo_name_input.text())))
        return request, error_text

    def check_parity(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build upload request.")
            return
        self.status_label.setText(str(request.get("parity_detail", "Parity check complete.")))

    def preview_upload_command(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build upload request.")
            return
        show_text_dialog(
            parent=self,
            title="HF Model Upload Command",
            text="Upload command:\n" + format_command_for_dialog(request["upload_cmd"]),
            copy_text=" ".join(str(part) for part in request["upload_cmd"]),
            wrap_mode="word",
        )

    def accept_for_run(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build upload request.")
            return
        self.result_request = request
        self.result_settings = {
            "local_model": self.local_model_input.text().strip(),
            "owner": self.owner_input.text().strip(),
            "repo_name": self.repo_name_input.text().strip(),
            "skip_if_exists": self.skip_if_exists_checkbox.isChecked(),
        }
        self.accept()


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
        self.camera_preview = QtDualCameraPreview(config=self.config, append_log=self._append_log)
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
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)
        self._register_action_button(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        run_button = QPushButton("Run Record")
        run_button.clicked.connect(self.run_record)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

        scan_ports_button = QPushButton("Scan Robot Ports")
        scan_ports_button.clicked.connect(self.scan_robot_ports)
        actions.addWidget(scan_ports_button)
        self._register_action_button(scan_ports_button)

        cancel_button = QPushButton("Cancel")
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

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)

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


class DeployOpsPanel(_CoreOpsPanel):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Deploy",
            subtitle="Build deploy commands, run preflight checks, and launch or cancel deployment workflows.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config
        self.rename_map_flag = str(config.get("camera_rename_flag", "rename_map")).strip().lstrip("-") or "rename_map"
        self._latest_deploy_artifact_path: Path | None = None
        self.run_helper_dialog = QtRunHelperDialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            mode_title="Deploy",
            on_cancel=self._cancel_run,
        )
        self.camera_preview = QtDualCameraPreview(config=self.config, append_log=self._append_log)

        form = _InputGrid(self.form_layout)

        models_root = str(config.get("trained_models_dir", ""))
        self.models_root_input = QLineEdit(models_root)
        form.add_field("Models root", self.models_root_input)

        last_model_name = str(config.get("last_model_name", "")).strip()
        self.model_path_input = QLineEdit(last_model_name)
        self.model_path_input.setPlaceholderText("absolute model path or relative folder under models root")
        form.add_field("Model path", self.model_path_input)

        default_eval_name = str(config.get("last_eval_dataset_name", "")).strip() or "eval_run_1"
        default_eval = normalize_repo_id(str(config.get("hf_username", "")), default_eval_name)
        self.eval_dataset_input = QLineEdit(default_eval)
        eval_row = QWidget()
        eval_row_layout = QHBoxLayout(eval_row)
        eval_row_layout.setContentsMargins(0, 0, 0, 0)
        eval_row_layout.setSpacing(8)
        eval_row_layout.addWidget(self.eval_dataset_input, 1)
        self.eval_prefix_button = QPushButton("Apply eval_ Prefix")
        self.eval_prefix_button.clicked.connect(self.apply_eval_prefix_quick_fix)
        eval_row_layout.addWidget(self.eval_prefix_button)
        form.add_field("Eval dataset", eval_row)

        self.task_input = QLineEdit(str(config.get("eval_task", DEFAULT_TASK)) or DEFAULT_TASK)
        form.add_field("Eval task", self.task_input)

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 10000)
        self.episodes_input.setValue(int(config.get("eval_num_episodes", 10) or 10))
        form.add_field("Eval episodes", self.episodes_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(int(config.get("eval_duration_s", 20) or 20))
        form.add_field("Eval duration (s)", self.duration_input)

        self.target_hz_input = QLineEdit(str(config.get("deploy_target_hz", "")).strip())
        self.target_hz_input.setPlaceholderText("optional")
        form.add_field("Target Hz", self.target_hz_input)

        self.follower_calibration_input = QLineEdit(str(config.get("follower_calibration_path", "")).strip())
        self.follower_calibration_input.setPlaceholderText("optional calibration JSON")
        form.add_field("Follower calibration", self._build_calibration_row(self.follower_calibration_input))

        self.leader_calibration_input = QLineEdit(str(config.get("leader_calibration_path", "")).strip())
        self.leader_calibration_input.setPlaceholderText("optional calibration JSON")
        form.add_field("Leader calibration", self._build_calibration_row(self.leader_calibration_input))

        self.rename_map_input = QLineEdit("")
        self.rename_map_input.setPlaceholderText("optional camera rename map JSON")
        form.add_field("Camera rename map", self.rename_map_input)

        self.deploy_advanced_toggle = QCheckBox("Advanced command options")
        self.deploy_advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.form_layout.addWidget(self.deploy_advanced_toggle)

        self.deploy_advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Deploy Options",
            fields=[
                ("robot.type", "Robot type"),
                ("robot.port", "Follower port"),
                ("robot.id", "Follower robot id"),
                ("robot.cameras", "Robot cameras JSON"),
                ("teleop.type", "Teleop type"),
                ("teleop.port", "Leader port"),
                ("teleop.id", "Leader robot id"),
                ("policy.path", "Policy path"),
                (self.rename_map_flag, "Camera rename map JSON"),
            ],
        )
        self.deploy_advanced_panel.hide()
        self.form_layout.addWidget(self.deploy_advanced_panel)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)
        self._register_action_button(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        run_button = QPushButton("Run Deploy")
        run_button.clicked.connect(self.run_deploy)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

        scan_ports_button = QPushButton("Scan Robot Ports")
        scan_ports_button.clicked.connect(self.scan_robot_ports)
        actions.addWidget(scan_ports_button)
        self._register_action_button(scan_ports_button)

        helper_button = QPushButton("Open Run Helper")
        helper_button.clicked.connect(self.open_run_helper)
        actions.addWidget(helper_button)
        self._register_action_button(helper_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)

        actions.addStretch(1)
        self.form_layout.addLayout(actions)

        self._auto_eval_hint = self.eval_dataset_input.text().strip()
        self.model_browser_card, model_browser_layout = _build_card("Model Browser")
        self._populate_model_browser_ui(model_browser_layout)
        main_layout = self.layout()
        if isinstance(main_layout, QVBoxLayout):
            main_layout.insertWidget(2, self.model_browser_card)
            main_layout.insertWidget(3, self.camera_preview)
        self.models_root_input.editingFinished.connect(self.refresh_model_browser)
        self.refresh_model_browser()
        self.restore_model_browser_selection()

    def _build_calibration_row(self, input_widget: QLineEdit) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(input_widget, 1)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self._browse_calibration_file(input_widget))
        layout.addWidget(browse_button)

        auto_button = QPushButton("Auto")
        auto_button.clicked.connect(lambda: input_widget.setText(""))
        layout.addWidget(auto_button)
        return row

    def _browse_calibration_file(self, target_input: QLineEdit) -> None:
        current_value = target_input.text().strip()
        initial_dir = str(Path(current_value).expanduser().parent) if current_value else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Robot Calibration File",
            initial_dir,
            "JSON files (*.json);;All files (*)",
        )
        if selected:
            target_input.setText(selected)

    def _preview_config(self) -> dict[str, Any]:
        preview = dict(self.config)
        preview["follower_calibration_path"] = self.follower_calibration_input.text().strip()
        preview["leader_calibration_path"] = self.leader_calibration_input.text().strip()
        return preview

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)
        if not active:
            self.run_helper_dialog.finish_run(
                status_text=status_text or ("Deploy failed." if is_error else "Deploy completed.")
            )

    def _append_runtime_line(self, line: str) -> None:
        self._append_output_line(line)
        self.run_helper_dialog.handle_output_line(line)

    def _remember_deploy_artifact(self, artifact_path: Path) -> None:
        self._latest_deploy_artifact_path = Path(artifact_path)

    def _persist_runtime_outcomes(self) -> tuple[bool, str]:
        run_path = self._latest_deploy_artifact_path
        if run_path is None:
            return False, "Deploy artifact path is not available yet."
        metadata_path = run_path / "metadata.json"
        if not metadata_path.exists():
            return False, "Deploy metadata.json is missing."
        raw_payload = self.run_helper_dialog.outcome_payload()
        if not raw_payload:
            return False, "No deploy outcomes were marked in the runtime helper."
        try:
            metadata_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"Unable to read deploy metadata.json: {exc}"
        entries: list[dict[str, Any]] = []
        total_episodes = 0
        for episode in sorted(raw_payload):
            try:
                episode_idx = int(episode)
            except (TypeError, ValueError):
                continue
            if episode_idx <= 0:
                continue
            total_episodes = max(total_episodes, episode_idx)
            entry = raw_payload.get(episode_idx, {})
            result = str(entry.get("status", "")).strip().lower()
            if result not in {"success", "failed"}:
                result = "unmarked"
            tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
            entries.append({"episode": episode_idx, "result": result, "tags": [str(tag) for tag in tags if str(tag).strip()]})
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(
            {"enabled": True, "total_episodes": total_episodes, "episode_outcomes": entries}
        )
        try:
            _atomic_write(json.dumps(metadata_data, indent=2) + "\n", metadata_path)
        except OSError as exc:
            return False, f"Failed to update deploy metadata.json: {exc}"
        write_deploy_notes_file(run_path, metadata_data, filename="notes.md")
        write_deploy_episode_spreadsheet(
            run_path,
            metadata_data,
            filename="episode_outcomes.csv",
            summary_filename="episode_outcomes_summary.csv",
        )
        return True, f"Saved deploy outcome tracker data to {metadata_path.name} and episode outcome CSV files."

    def _build_hooks(self, *, on_teleop_ready: Callable[[], None] | None = None) -> RunUiHooks:
        _ = on_teleop_ready
        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_runtime_line,
            on_artifact_written=self._remember_deploy_artifact,
        )

    def open_run_helper(self) -> None:
        self.run_helper_dialog.show()
        self.run_helper_dialog.raise_()
        self.run_helper_dialog.activateWindow()

    def _build(self) -> tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]:
        arg_overrides: dict[str, str] | None = None
        custom_args_raw = ""
        if self.deploy_advanced_toggle.isChecked():
            arg_overrides, custom_args_raw = self.deploy_advanced_panel.build_overrides()
        rename_map_value = self.rename_map_input.text().strip()
        if rename_map_value:
            arg_overrides = dict(arg_overrides or {})
            arg_overrides[self.rename_map_flag] = rename_map_value
        return build_deploy_request_and_command(
            config=self._preview_config(),
            deploy_root_raw=self.models_root_input.text(),
            deploy_model_raw=self.model_path_input.text(),
            eval_dataset_raw=self.eval_dataset_input.text(),
            eval_episodes_raw=str(self.episodes_input.value()),
            eval_duration_raw=str(self.duration_input.value()),
            eval_task_raw=self.task_input.text(),
            target_hz_raw=self.target_hz_input.text(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def _toggle_advanced_options(self, checked: bool) -> None:
        if checked:
            req, cmd, _updated, error = self._build()
            if error is None and req is not None and cmd is not None:
                self.deploy_advanced_panel.seed_from_command(cmd)
            self.deploy_advanced_panel.show()
        else:
            self.deploy_advanced_panel.hide()

    def _populate_model_browser_ui(self, layout: QVBoxLayout) -> None:
        controls = QHBoxLayout()
        controls.setSpacing(8)

        browse_root_button = QPushButton("Browse Root")
        browse_root_button.clicked.connect(self.browse_model_root)
        controls.addWidget(browse_root_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_model_browser)
        controls.addWidget(refresh_button)

        browse_model_button = QPushButton("Browse Model")
        browse_model_button.clicked.connect(self.browse_for_model)
        controls.addWidget(browse_model_button)

        upload_button = QPushButton("Upload Model")
        upload_button.clicked.connect(self.open_model_upload_dialog)
        controls.addWidget(upload_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.model_tree = QTreeWidget()
        self.model_tree.setColumnCount(2)
        self.model_tree.setHeaderLabels(["Model / Checkpoint", "Type"])
        self.model_tree.setRootIsDecorated(True)
        self.model_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.model_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.model_tree.itemSelectionChanged.connect(self._handle_model_tree_selection)
        layout.addWidget(self.model_tree)

        self.selected_model_label = QLabel("No model selected.")
        self.selected_model_label.setObjectName("MutedLabel")
        self.selected_model_label.setWordWrap(True)
        layout.addWidget(self.selected_model_label)

        self.model_info = QPlainTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setMinimumHeight(120)
        self.model_info.setPlainText("No model selected.")
        layout.addWidget(self.model_info)

    def _current_models_root(self) -> Path:
        return Path(self.models_root_input.text().strip() or str(self.config.get("trained_models_dir", ""))).expanduser()

    def _add_model_browser_node(self, parent: QTreeWidgetItem | QTreeWidget, node: ModelBrowserNode) -> None:
        item = QTreeWidgetItem([node.label, node.kind])
        item.setData(0, Qt.ItemDataRole.UserRole, str(node.path))
        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in node.children:
            self._add_model_browser_node(item, child)

    def refresh_model_browser(self) -> None:
        self.model_tree.clear()
        root_path = self._current_models_root()
        self.config["trained_models_dir"] = str(root_path)
        nodes = build_model_browser_tree(root_path)
        for node in nodes:
            self._add_model_browser_node(self.model_tree, node)
        self.selected_model_label.setText("No model selected." if not nodes else self.selected_model_label.text())

    def _apply_model_selection(self, selected_path: Path) -> None:
        root_path = self._current_models_root()
        resolved = resolve_payload_path(selected_path)
        self.model_path_input.setText(str(resolved))
        if self.deploy_advanced_toggle.isChecked():
            self.deploy_advanced_panel.inputs["policy.path"].setText(str(resolved))
        self.selected_model_label.setText(
            f"Selected: {selected_path}  |  Deploy payload: {resolved}"
            if resolved != selected_path
            else f"Selected: {selected_path}"
        )
        self.model_info.setPlainText(summarize_model_info(selected_path))

        model_folder, checkpoint = split_model_selection(root_path, selected_path)
        self.config["trained_models_dir"] = str(root_path)
        self.config["last_model_name"] = model_folder
        self.config["last_checkpoint_name"] = checkpoint
        save_config(self.config, quiet=True)

        current_eval_name = self.eval_dataset_input.text().strip()
        if not current_eval_name or current_eval_name == self._auto_eval_hint:
            suggested = normalize_repo_id(
                str(self.config.get("hf_username", "")),
                repo_name_only(selected_path.name),
            )
            self.eval_dataset_input.setText(suggested)
            self._auto_eval_hint = suggested

        self._append_log(f"Model selected: {resolved}")

    def _handle_model_tree_selection(self) -> None:
        items = self.model_tree.selectedItems()
        if not items:
            return
        path_text = str(items[0].data(0, Qt.ItemDataRole.UserRole) or "").strip()
        if not path_text:
            return
        selected_path = Path(path_text)
        self._apply_model_selection(selected_path)

    def restore_model_browser_selection(self) -> None:
        root_path = self._current_models_root()
        saved_folder = str(self.config.get("last_model_name", "")).strip()
        saved_checkpoint = str(self.config.get("last_checkpoint_name", "")).strip()
        if not saved_folder:
            return
        target = root_path / saved_folder
        if saved_checkpoint:
            target = target / saved_checkpoint
        self._select_tree_item_for_path(target)

    def _select_tree_item_for_path(self, target: Path) -> bool:
        matches = self.model_tree.findItems(target.name, Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly, 0)
        for item in matches:
            path_text = str(item.data(0, Qt.ItemDataRole.UserRole) or "").strip()
            if path_text and Path(path_text) == target:
                self.model_tree.setCurrentItem(item)
                self.model_tree.scrollToItem(item)
                return True
        return False

    def browse_model_root(self) -> None:
        current = str(self._current_models_root())
        selected = QFileDialog.getExistingDirectory(self, "Select models root", current or str(Path.home()))
        if not selected:
            return
        self.models_root_input.setText(selected)
        self.refresh_model_browser()

    def browse_for_model(self) -> None:
        current = str(self._current_models_root())
        selected = QFileDialog.getExistingDirectory(self, "Select Model or Checkpoint Folder", current or str(Path.home()))
        if not selected:
            return
        selected_path = Path(selected)
        root_path = self._current_models_root()
        if root_path not in selected_path.parents and selected_path != root_path:
            self.models_root_input.setText(str(selected_path.parent))
            self.refresh_model_browser()
        self._apply_model_selection(selected_path)
        self._select_tree_item_for_path(selected_path)

    def _model_candidate_options(self) -> list[str]:
        root_path = self._current_models_root()
        nodes = build_model_browser_tree(root_path)
        return [str(node.path) for node in nodes]

    def open_model_upload_dialog(self) -> None:
        default_local_model = self.model_path_input.text().strip()
        if not default_local_model:
            saved_folder = str(self.config.get("last_model_name", "")).strip()
            if saved_folder:
                default_local_model = str(self._current_models_root() / saved_folder)
        default_owner = str(self.config.get("deploy_hf_sync_owner", "")).strip() or str(self.config.get("hf_username", "")).strip()
        default_repo_name = repo_name_only(str(self.config.get("deploy_hf_sync_repo_name", "")).strip(), owner=default_owner)
        if not default_repo_name and default_local_model:
            default_repo_name = Path(default_local_model).name

        dialog = _QtModelUploadDialog(
            parent=self._dialog_parent(),
            default_local_model=default_local_model,
            default_owner=default_owner,
            default_repo_name=default_repo_name,
            model_options=self._model_candidate_options(),
        )
        dialog.exec()
        request = dialog.result_request
        settings = dialog.result_settings
        if request is None or settings is None:
            return

        repo_id = str(request["repo_id"])
        local_model = Path(request["local_model"])
        remote_exists = request["remote_exists"]
        if remote_exists is True and bool(settings.get("skip_if_exists", True)):
            self._append_output_and_log(f"Remote model already exists. Upload skipped: {repo_id}")
            return
        if remote_exists is True and not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Remote Model Exists",
            text=f"{repo_id} already exists on Hugging Face.\nContinue upload anyway?",
            confirm_label="Continue Upload",
            cancel_label="Cancel",
            wrap_mode="word",
        ):
            return
        if remote_exists is None and not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Parity Unknown",
            text=f"Could not verify remote parity for {repo_id}.\nContinue upload anyway?",
            confirm_label="Continue Upload",
            cancel_label="Cancel",
            wrap_mode="word",
        ):
            return
        if not self._confirm_preflight_review(title="HF Model Upload Preflight", checks=request["checks"]):
            return
        if not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Confirm HF Model Upload",
            text=(
                "Review the upload command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(request["upload_cmd"])
            ),
            copy_text=" ".join(str(part) for part in request["upload_cmd"]),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        self.config["deploy_hf_sync_local_model"] = str(settings.get("local_model", "")).strip()
        self.config["deploy_hf_sync_owner"] = str(settings.get("owner", "")).strip()
        self.config["deploy_hf_sync_repo_name"] = repo_name_only(str(settings.get("repo_name", "")).strip(), owner=str(settings.get("owner", "")))
        self.config["deploy_hf_sync_skip_if_exists"] = bool(settings.get("skip_if_exists", True))
        self.config["hf_username"] = str(settings.get("owner", "")).strip().strip("/") or str(self.config.get("hf_username", ""))
        save_config(self.config, quiet=True)

        self._show_launch_summary(
            heading="Launching model upload...",
            command_label="HF model upload command",
            cmd=request["upload_cmd"],
            preflight_title="HF Model Upload Preflight",
            preflight_checks=request["checks"],
        )
        self._append_log(f"Starting model upload: {repo_id}")

        def after_upload(upload_code: int, upload_canceled: bool) -> None:
            if upload_canceled:
                self._set_running(False, "Model upload canceled.", False)
                self._append_output_and_log("Hugging Face model upload canceled.")
                return
            if upload_code != 0:
                self._set_running(False, "Model upload failed.", True)
                self._append_output_and_log(f"Hugging Face model upload failed with exit code {upload_code}.")
                return
            self._set_running(False, "Model upload completed.", False)
            self._append_output_and_log(f"Model upload completed: {repo_id}")

        ok, message = self._run_controller.run_process_async(
            cmd=request["upload_cmd"],
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_upload,
            run_mode="upload",
            preflight_checks=request["checks"],
            artifact_context={"model_path": str(local_model), "model_repo_id": repo_id},
        )
        if not ok and message:
            self._handle_launch_rejection(
                title="Upload Unavailable",
                message=message,
                log_message="Model upload launch was rejected.",
            )

    def preview_command(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Deploy preview failed validation.")
            return
        summary = (
            f"Model path: {req.model_path}\n"
            f"Eval dataset: {req.eval_repo_id}\n"
            f"Episodes: {req.eval_num_episodes}\n"
            f"Duration: {req.eval_duration_s}s\n\n"
            f"{format_command_for_dialog(cmd)}"
        )
        self._append_log(f"Deploy preview built for {req.eval_repo_id}.")
        self._show_text_dialog(title="Deploy Command", text=summary, copy_text=summary, wrap_mode="word")

    def run_preflight(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Deploy preflight failed validation.")
            return
        checks = run_preflight_for_deploy(
            config=self._preview_config(),
            model_path=req.model_path,
            eval_repo_id=req.eval_repo_id,
            command=cmd,
        )
        self._append_log(f"Deploy preflight ran for {req.eval_repo_id}.")
        self._show_text_dialog(
            title="Deploy Preflight",
            text=summarize_checks(checks, title="Deploy Preflight"),
            wrap_mode="char",
        )

    def apply_eval_prefix_quick_fix(self) -> bool:
        current_value = self.eval_dataset_input.text().strip()
        suggested_repo_id, changed = suggest_eval_prefixed_repo_id(
            username=str(self.config.get("hf_username", "")),
            dataset_name_or_repo_id=current_value,
        )
        suggested_repo_id = normalize_repo_id(str(self.config.get("hf_username", "")), suggested_repo_id)
        if not changed:
            self._append_log(f"Eval dataset already follows convention: {suggested_repo_id}")
            return False
        self.eval_dataset_input.setText(suggested_repo_id)
        self._append_output_and_log(f"Applied eval dataset quick fix: {suggested_repo_id}")
        return True

    def scan_robot_ports(self) -> None:
        follower_guess, leader_guess = self._run_port_scan_dialog(
            title="Robot Port Scan",
            current_follower=str(self.config.get("follower_port", "")),
            current_leader=str(self.config.get("leader_port", "")),
            apply_scope_label="deploy",
        )
        if not follower_guess or not leader_guess:
            return
        self.config["follower_port"] = follower_guess
        self.config["leader_port"] = leader_guess
        if self.deploy_advanced_toggle.isChecked():
            self.deploy_advanced_panel.inputs["robot.port"].setText(follower_guess)
            self.deploy_advanced_panel.inputs["teleop.port"].setText(leader_guess)
        save_config(self.config, quiet=True)
        self._append_output_and_log(
            f"Applied scanned deploy defaults: follower={follower_guess}, leader={leader_guess}"
        )

    def run_deploy(self) -> None:
        req, cmd, updated, error = self._build()
        if error or req is None or cmd is None or updated is None:
            self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Deploy launch failed validation.")
            return

        calibration_changed = False
        preview_config = self._preview_config()
        for key in ("follower_calibration_path", "leader_calibration_path"):
            if self.config.get(key, "") != preview_config.get(key, ""):
                self.config[key] = preview_config.get(key, "")
                calibration_changed = True
        if calibration_changed:
            save_config(self.config, quiet=True)

        while True:
            req, cmd, updated, error = self._build()
            if error or req is None or cmd is None or updated is None:
                self._set_output(title="Validation Error", text=error or "Unable to build deploy command.", log_message="Deploy launch failed validation.")
                return

            checks = run_preflight_for_deploy(
                config=self._preview_config(),
                model_path=req.model_path,
                eval_repo_id=req.eval_repo_id,
                command=cmd,
            )
            quick_actions, action_context = quick_actions_from_checks(checks)
            model_candidate = first_model_payload_candidate(checks)
            rename_map_suggestion = camera_rename_map_suggestion(checks)
            rename_ctx = action_context.get("apply_rename_map", {})
            rename_map_from_context = str(rename_ctx.get("rename_map_suggestion", "")).strip()
            if not rename_map_suggestion and rename_map_from_context:
                rename_map_suggestion = rename_map_from_context

            model_ctx = action_context.get("fix_model_payload", {})
            model_candidate_from_context = str(model_ctx.get("model_candidate", "")).strip()
            if model_candidate_from_context:
                model_candidate = model_candidate_from_context
            if model_candidate and Path(model_candidate).expanduser() == req.model_path:
                quick_actions = [item for item in quick_actions if item[0] != "fix_model_payload"]

            current_eval_input = self.eval_dataset_input.text().strip() or req.eval_repo_id
            suggested_repo, missing_eval_prefix = suggest_eval_prefixed_repo_id(
                username=str(self.config.get("hf_username", "")),
                dataset_name_or_repo_id=current_eval_input,
            )
            suggested_repo = normalize_repo_id(str(self.config.get("hf_username", "")), suggested_repo)
            eval_ctx = action_context.get("fix_eval_prefix", {})
            suggested_repo = str(eval_ctx.get("suggested_eval_repo_id", suggested_repo)).strip() or suggested_repo
            if not missing_eval_prefix:
                quick_actions = [item for item in quick_actions if item[0] != "fix_eval_prefix"]

            if self.rename_map_input.text().strip():
                quick_actions = [item for item in quick_actions if item[0] != "apply_rename_map"]

            if not quick_actions:
                break

            action = self._ask_text_dialog_with_actions(
                title="Deploy Preflight Fix Center",
                text=summarize_checks(checks, title="Deploy Preflight"),
                actions=quick_actions,
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            )
            if action == "cancel":
                self._append_log("Deploy canceled from quick-fix center.")
                return
            if action == "confirm":
                break
            if action == "fix_eval_prefix":
                self.eval_dataset_input.setText(suggested_repo)
                self._append_output_and_log(f"Applied preflight quick fix: eval dataset -> {suggested_repo}")
            elif action == "apply_rename_map" and rename_map_suggestion:
                self.rename_map_input.setText(rename_map_suggestion)
                if self.deploy_advanced_toggle.isChecked():
                    self.deploy_advanced_panel.inputs[self.rename_map_flag].setText(rename_map_suggestion)
                self._append_output_and_log(f"Applied preflight quick fix: {self.rename_map_flag} -> {rename_map_suggestion}")
            elif action == "fix_model_payload" and model_candidate:
                self.model_path_input.setText(str(Path(model_candidate).expanduser()))
                if self.deploy_advanced_toggle.isChecked():
                    self.deploy_advanced_panel.inputs["policy.path"].setText(str(Path(model_candidate).expanduser()))
                self._append_output_and_log(f"Applied preflight quick fix: model payload -> {model_candidate}")
            elif action.startswith("fix_camera_fps:"):
                try:
                    new_fps = int(action.split("fix_camera_fps:", 1)[1])
                except (ValueError, IndexError):
                    new_fps = None
                if new_fps and new_fps > 0:
                    self.config["camera_fps"] = new_fps
                    save_config(self.config, quiet=True)
                    self._append_output_and_log(
                        f"Applied preflight quick fix: camera_fps -> {new_fps} Hz (matches model training FPS)"
                    )
            elif action == "browse_follower_calib":
                before = self.follower_calibration_input.text().strip()
                self._browse_calibration_file(self.follower_calibration_input)
                after = self.follower_calibration_input.text().strip()
                if after != before:
                    self.config["follower_calibration_path"] = after
                    save_config(self.config, quiet=True)
                    self._append_output_and_log(
                        f"Applied preflight quick fix: follower_calibration_path -> {after or '(auto-detect)'}"
                    )
            elif action == "browse_leader_calib":
                before = self.leader_calibration_input.text().strip()
                self._browse_calibration_file(self.leader_calibration_input)
                after = self.leader_calibration_input.text().strip()
                if after != before:
                    self.config["leader_calibration_path"] = after
                    save_config(self.config, quiet=True)
                    self._append_output_and_log(
                        f"Applied preflight quick fix: leader_calibration_path -> {after or '(auto-detect)'}"
                    )
            elif action == "show_calib_cmd":
                calib_cmd = build_calibration_command(self._preview_config())
                self._show_text_dialog(
                    title="Robot Recalibration Command",
                    text=(
                        "One or more calibration checks failed.\n"
                        "Run the command below to recalibrate the follower arm,\n"
                        "then re-run the deploy preflight.\n\n"
                        "IMPORTANT: power-cycle the arm and keep hands clear before running.\n\n"
                        + calib_cmd
                    ),
                    copy_text=calib_cmd,
                    wrap_mode="none",
                )

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Deploy Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the deploy command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Deploy",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited deploy command from command editor.")
        cmd = editable_cmd

        effective_repo_id = normalize_repo_id(
            str(self.config.get("hf_username", "")),
            get_flag_value(cmd, "dataset.repo_id") or req.eval_repo_id,
        )
        episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.eval_num_episodes)
        duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.eval_duration_s)
        effective_model_text = get_policy_path_value(cmd) or str(req.model_path)
        effective_model_path = Path(effective_model_text).expanduser()
        if not effective_model_path.is_absolute():
            models_root = Path(self.models_root_input.text().strip() or str(self.config.get("trained_models_dir", ""))).expanduser()
            effective_model_path = models_root / effective_model_path
        try:
            effective_episodes = int(str(episodes_text).strip())
            effective_duration = int(str(duration_text).strip())
        except ValueError:
            self._set_output(
                title="Validation Error",
                text="Edited command must keep eval episodes and duration as integers.",
                log_message="Deploy launch rejected due to invalid edited command values.",
            )
            return
        if effective_episodes <= 0 or effective_duration <= 0:
            self._set_output(
                title="Validation Error",
                text="Edited command must keep eval episodes and duration greater than zero.",
                log_message="Deploy launch rejected due to non-positive edited command values.",
            )
            return

        checks = run_preflight_for_deploy(
            config=self._preview_config(),
            model_path=effective_model_path,
            eval_repo_id=effective_repo_id,
            command=cmd,
        )
        if not self._confirm_preflight_review(title="Deploy Preflight", checks=checks):
            self._append_log("Deploy canceled after preflight review.")
            return

        warning_detail = None
        if any(str(level).strip().upper() == "WARN" for level, _name, _detail in checks):
            warning_detail = "Warnings were detected. The workflow continues automatically when there are no FAIL checks."
        self._show_launch_summary(
            heading="Launching deploy run...",
            command_label="Deploy command",
            cmd=cmd,
            preflight_title="Deploy Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self.config.update(updated)
        self._append_log(f"Deploy launch starting for {effective_repo_id}.")
        self._latest_deploy_artifact_path = None
        self.run_helper_dialog.start_run(run_mode="deploy", expected_episodes=effective_episodes)

        def after_deploy(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Deploy canceled.", False)
                self._append_output_and_log("Deploy run canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Deploy failed.", True)
                self._append_output_and_log(f"Deploy run failed with exit code {return_code}.")
                return
            self.config["last_dataset_repo_id"] = effective_repo_id
            self._set_running(False, "Deploy completed.", False)
            self._append_output_and_log(f"Deploy completed. Eval dataset: {effective_repo_id}")
            saved_ok, saved_message = self._persist_runtime_outcomes()
            if saved_ok:
                self._append_output_and_log(saved_message)

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_deploy,
            expected_episodes=effective_episodes,
            expected_seconds=effective_episodes * effective_duration,
            run_mode="deploy",
            preflight_checks=checks,
            artifact_context={"dataset_repo_id": effective_repo_id, "model_path": str(effective_model_path)},
        )
        if not ok and message:
            self._handle_launch_rejection(title="Deploy Unavailable", message=message, log_message="Deploy launch was rejected.")


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
            subtitle="Build teleop commands, run preflight checks, launch live sessions, and control episodes from the GUI.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config
        self._teleop_ready = False
        send_arrow = getattr(self._run_controller, "send_arrow_key", None)
        self.run_helper_dialog = QtRunHelperDialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            mode_title="Teleop",
            on_send_key=send_arrow,
            on_cancel=self._cancel_run,
        )

        form = _InputGrid(self.form_layout)

        self.follower_port_input = QLineEdit(str(config.get("follower_port", "")))
        form.add_field("Follower port", self.follower_port_input)

        self.leader_port_input = QLineEdit(str(config.get("leader_port", "")))
        form.add_field("Leader port", self.leader_port_input)

        self.follower_id_input = QLineEdit(str(config.get("follower_robot_id", "red4")).strip() or "red4")
        form.add_field("Follower id", self.follower_id_input)

        self.leader_id_input = QLineEdit(str(config.get("leader_robot_id", "white")).strip() or "white")
        form.add_field("Leader id", self.leader_id_input)

        actions = QHBoxLayout()
        preview_button = QPushButton("Preview Command")
        preview_button.setObjectName("AccentButton")
        preview_button.clicked.connect(self.preview_command)
        actions.addWidget(preview_button)
        self._register_action_button(preview_button)

        preflight_button = QPushButton("Run Preflight")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        run_button = QPushButton("Run Teleop")
        run_button.clicked.connect(self.run_teleop)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

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

        teleop_controls = QHBoxLayout()
        self.reset_episode_button = QPushButton("Reset Episode")
        self.reset_episode_button.clicked.connect(lambda: self._send_arrow("left"))
        self.reset_episode_button.setEnabled(False)
        teleop_controls.addWidget(self.reset_episode_button)

        self.next_episode_button = QPushButton("Next Episode")
        self.next_episode_button.clicked.connect(lambda: self._send_arrow("right"))
        self.next_episode_button.setEnabled(False)
        teleop_controls.addWidget(self.next_episode_button)
        teleop_controls.addStretch(1)
        self.form_layout.addLayout(teleop_controls)

    def show_teleop_help(self) -> None:
        self._show_text_dialog(
            title=keyboard_input_help_title(),
            text=keyboard_input_help_text()
            + "\n\nTeleop session helper:\n"
            + f"Follower: {self.follower_port_input.text().strip() or '(unset)'}\n"
            + f"Leader: {self.leader_port_input.text().strip() or '(unset)'}\n"
            + f"Follower id: {self.follower_id_input.text().strip() or '(unset)'}\n"
            + f"Leader id: {self.leader_id_input.text().strip() or '(unset)'}",
            wrap_mode="word",
        )
        self.run_helper_dialog.show()
        self.run_helper_dialog.raise_()
        self.run_helper_dialog.activateWindow()

    def _build(self) -> tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]:
        return build_teleop_request_and_command(
            config=self.config,
            follower_port_raw=self.follower_port_input.text(),
            leader_port_raw=self.leader_port_input.text(),
            follower_id_raw=self.follower_id_input.text(),
            leader_id_raw=self.leader_id_input.text(),
        )

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        if not active:
            self._teleop_ready = False
            self.reset_episode_button.setEnabled(False)
            self.next_episode_button.setEnabled(False)
            self.run_helper_dialog.finish_run(
                status_text=status_text or ("Teleop failed." if is_error else "Teleop completed.")
            )

    def _append_runtime_line(self, line: str) -> None:
        self._append_output_line(line)
        self.run_helper_dialog.handle_output_line(line)

    def _build_hooks(self, *, on_teleop_ready: Callable[[], None] | None = None) -> RunUiHooks:
        def _mark_ready() -> None:
            self._mark_teleop_ready()
            if on_teleop_ready is not None:
                on_teleop_ready()

        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_runtime_line,
            on_teleop_ready=_mark_ready,
        )

    def _mark_teleop_ready(self) -> None:
        self._teleop_ready = True
        self.reset_episode_button.setEnabled(True)
        self.next_episode_button.setEnabled(True)
        self.status_label.setText("Teleop connected.")
        self._append_output_and_log("Teleop session reported ready. Episode controls are now live.")
        self.run_helper_dialog.set_teleop_ready(True)

    def _send_arrow(self, direction: str) -> None:
        if not self._teleop_ready:
            self._set_output(
                title="Controls Unavailable",
                text="Episode controls become available after the teleop process reports that it is ready.",
                log_message="Teleop control request ignored before session readiness.",
            )
            return
        ok, message = self._run_controller.send_arrow_key(direction)
        if not ok:
            self._set_output(title="Dispatch Failed", text=message, log_message="Teleop arrow dispatch failed.")
            return
        self._append_output_and_log(message)

    def preview_command(self) -> None:
        req, cmd, _updated, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(title="Validation Error", text=error or "Unable to build teleop command.", log_message="Teleop preview failed validation.")
            return
        summary = (
            f"Follower: {req.follower_port} ({req.follower_id})\n"
            f"Leader: {req.leader_port} ({req.leader_id})\n\n"
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
        self._teleop_ready = False
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
