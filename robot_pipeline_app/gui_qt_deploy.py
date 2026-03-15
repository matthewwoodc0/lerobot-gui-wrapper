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
from .gui_qt_dialogs import (
    _build_dialog_panel,
    _fit_dialog_to_screen,
    ask_editable_command_dialog,
    ask_text_dialog,
    ask_text_dialog_with_actions,
    show_text_dialog,
)
from .gui_qt_runtime_helpers import QtRunHelperDialog
from .repo_utils import next_available_dataset_name, normalize_repo_id, repo_name_from_repo_id, repo_name_only, suggest_eval_prefixed_repo_id
from .run_controller_service import ManagedRunController, RunUiHooks
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .workflows import move_recorded_dataset

from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card, _count_preflight_failures

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
        self._initial_model_options = list(model_options)
        self.setWindowTitle("Upload Model to Hugging Face")
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=900,
            requested_height=420,
            requested_min_width=760,
            requested_min_height=340,
        )
        self._default_local_model = default_local_model
        self._default_owner = default_owner
        self._default_repo_name = default_repo_name
        self._build_ui()
        self.set_model_options(list(self._initial_model_options))

    def _build_ui(self) -> None:
        layout = _build_dialog_panel(
            self,
            title="Upload Model to Hugging Face",
            subtitle="Upload sends your local model or checkpoint folder to a Hugging Face model repository.",
        )

        intro = QLabel(
            "Use this for backups and sharing of trained artifacts. It does not run deploy or eval."
        )
        intro.setObjectName("DialogSubtitle")
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
        self.local_model_input = QLineEdit(self._default_local_model)
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
        refresh_button.clicked.connect(lambda: self.set_model_options(list(self._initial_model_options)))
        grid.addWidget(refresh_button, 1, 3)

        owner_label = QLabel("HF owner")
        owner_label.setObjectName("FormLabel")
        grid.addWidget(owner_label, 2, 0)
        self.owner_input = QLineEdit(self._default_owner)
        grid.addWidget(self.owner_input, 2, 1)

        repo_label = QLabel("HF model name")
        repo_label.setObjectName("FormLabel")
        grid.addWidget(repo_label, 2, 2)
        self.repo_name_input = QLineEdit(self._default_repo_name)
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


class _DeployWorkflowRunner:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        rename_map_flag: str,
        models_root_input: QLineEdit,
        model_path_input: QLineEdit,
        eval_dataset_input: QLineEdit,
        follower_calibration_input: QLineEdit,
        leader_calibration_input: QLineEdit,
        rename_map_input: QLineEdit,
        deploy_advanced_toggle: QCheckBox,
        deploy_advanced_panel: _AdvancedOptionsPanel,
        run_helper_dialog: QtRunHelperDialog,
        run_controller: ManagedRunController,
        build: Callable[[], tuple[Any | None, list[str] | None, dict[str, Any] | None, str | None]],
        preview_config: Callable[[], dict[str, Any]],
        browse_calibration_file: Callable[[QLineEdit], None],
        ask_text_dialog_with_actions: Callable[..., str],
        ask_editable_command_dialog: Callable[..., list[str] | None],
        confirm_preflight_review: Callable[..., bool],
        show_text_dialog: Callable[..., None],
        show_launch_summary: Callable[..., None],
        set_output: Callable[..., None],
        set_running: Callable[[bool, str | None, bool], None],
        append_log: Callable[[str], None],
        append_output_and_log: Callable[[str], None],
        build_hooks: Callable[[], RunUiHooks],
        handle_launch_rejection: Callable[..., None],
        persist_runtime_outcomes: Callable[[], tuple[bool, str]],
    ) -> None:
        self.config = config
        self.rename_map_flag = rename_map_flag
        self.models_root_input = models_root_input
        self.model_path_input = model_path_input
        self.eval_dataset_input = eval_dataset_input
        self.follower_calibration_input = follower_calibration_input
        self.leader_calibration_input = leader_calibration_input
        self.rename_map_input = rename_map_input
        self.deploy_advanced_toggle = deploy_advanced_toggle
        self.deploy_advanced_panel = deploy_advanced_panel
        self.run_helper_dialog = run_helper_dialog
        self.run_controller = run_controller
        self._build = build
        self._preview_config = preview_config
        self._browse_calibration_file = browse_calibration_file
        self._ask_text_dialog_with_actions = ask_text_dialog_with_actions
        self._ask_editable_command_dialog = ask_editable_command_dialog
        self._confirm_preflight_review = confirm_preflight_review
        self._show_text_dialog = show_text_dialog
        self._show_launch_summary = show_launch_summary
        self._set_output = set_output
        self._set_running = set_running
        self._append_log = append_log
        self._append_output_and_log = append_output_and_log
        self._build_hooks = build_hooks
        self._handle_launch_rejection = handle_launch_rejection
        self._persist_runtime_outcomes = persist_runtime_outcomes

    def run(self) -> None:
        req, cmd, updated, error = self._build()
        if error or req is None or cmd is None or updated is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build deploy command.",
                log_message="Deploy launch failed validation.",
            )
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
                self._set_output(
                    title="Validation Error",
                    text=error or "Unable to build deploy command.",
                    log_message="Deploy launch failed validation.",
                )
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
                self._append_output_and_log(
                    f"Applied preflight quick fix: {self.rename_map_flag} -> {rename_map_suggestion}"
                )
            elif action == "fix_model_payload" and model_candidate:
                expanded_model_candidate = str(Path(model_candidate).expanduser())
                self.model_path_input.setText(expanded_model_candidate)
                if self.deploy_advanced_toggle.isChecked():
                    self.deploy_advanced_panel.inputs["policy.path"].setText(expanded_model_candidate)
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
                self._apply_calibration_browse(
                    target_input=self.follower_calibration_input,
                    config_key="follower_calibration_path",
                )
            elif action == "browse_leader_calib":
                self._apply_calibration_browse(
                    target_input=self.leader_calibration_input,
                    config_key="leader_calibration_path",
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
            models_root = Path(
                self.models_root_input.text().strip() or str(self.config.get("trained_models_dir", ""))
            ).expanduser()
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

        ok, message = self.run_controller.run_process_async(
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
            self._handle_launch_rejection(
                title="Deploy Unavailable",
                message=message,
                log_message="Deploy launch was rejected.",
            )

    def _apply_calibration_browse(self, *, target_input: QLineEdit, config_key: str) -> None:
        before = target_input.text().strip()
        self._browse_calibration_file(target_input)
        after = target_input.text().strip()
        if after != before:
            self.config[config_key] = after
            save_config(self.config, quiet=True)
            self._append_output_and_log(
                f"Applied preflight quick fix: {config_key} -> {after or '(auto-detect)'}"
            )


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
        self.camera_preview = QtCameraWorkspace(config=self.config, append_log=self._append_log)
        self._build_form_ui()
        self._build_model_browser_ui()
        self._bind_signals()
        self._advance_eval_name()

    def _build_form_ui(self) -> QWidget:
        form = _InputGrid(self.form_layout)

        models_root = str(self.config.get("trained_models_dir", ""))
        self.models_root_input = QLineEdit(models_root)
        form.add_field("Models root", self.models_root_input)

        last_model_name = str(self.config.get("last_model_name", "")).strip()
        self.model_path_input = QLineEdit(last_model_name)
        self.model_path_input.setPlaceholderText("absolute model path or relative folder under models root")
        form.add_field("Model path", self.model_path_input)

        default_eval_name = str(self.config.get("last_eval_dataset_name", "")).strip() or "eval_run_1"
        default_eval = normalize_repo_id(str(self.config.get("hf_username", "")), default_eval_name)
        self.eval_dataset_input = QLineEdit(default_eval)
        eval_row = QWidget()
        eval_row_layout = QHBoxLayout(eval_row)
        eval_row_layout.setContentsMargins(0, 0, 0, 0)
        eval_row_layout.setSpacing(8)
        eval_row_layout.addWidget(self.eval_dataset_input, 1)
        self.eval_prefix_button = QPushButton("Apply eval_ Prefix")
        eval_row_layout.addWidget(self.eval_prefix_button)
        form.add_field("Eval dataset", eval_row)

        self.task_input = QLineEdit(str(self.config.get("eval_task", DEFAULT_TASK)) or DEFAULT_TASK)
        form.add_field("Eval task", self.task_input)

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 10000)
        self.episodes_input.setValue(int(self.config.get("eval_num_episodes", 10) or 10))
        form.add_field("Eval episodes", self.episodes_input)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(int(self.config.get("eval_duration_s", 20) or 20))
        form.add_field("Eval duration (s)", self.duration_input)

        self.target_hz_input = QLineEdit(str(self.config.get("deploy_target_hz", "")).strip())
        self.target_hz_input.setPlaceholderText("optional")
        form.add_field("Target Hz", self.target_hz_input)

        self.follower_calibration_input = QLineEdit(str(self.config.get("follower_calibration_path", "")).strip())
        self.follower_calibration_input.setPlaceholderText("optional calibration JSON")
        (
            follower_calibration_row,
            self.follower_calibration_browse_button,
            self.follower_calibration_auto_button,
        ) = self._build_calibration_row(self.follower_calibration_input)
        form.add_field("Follower calibration", follower_calibration_row)

        self.leader_calibration_input = QLineEdit(str(self.config.get("leader_calibration_path", "")).strip())
        self.leader_calibration_input.setPlaceholderText("optional calibration JSON")
        (
            leader_calibration_row,
            self.leader_calibration_browse_button,
            self.leader_calibration_auto_button,
        ) = self._build_calibration_row(self.leader_calibration_input)
        form.add_field("Leader calibration", leader_calibration_row)

        self.rename_map_input = QLineEdit("")
        self.rename_map_input.setPlaceholderText("optional camera rename map JSON")
        form.add_field("Camera rename map", self.rename_map_input)

        self.deploy_advanced_toggle = QCheckBox("Advanced command options")
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
        self.run_button = QPushButton("Run Deploy")
        self.run_button.setObjectName("AccentButton")
        actions.addWidget(self.run_button)
        self._register_action_button(self.run_button)

        self.preview_button = QPushButton("Preview Command")
        actions.addWidget(self.preview_button)
        self._register_action_button(self.preview_button)

        self.preflight_button = QPushButton("Run Preflight")
        actions.addWidget(self.preflight_button)
        self._register_action_button(self.preflight_button)

        self.scan_ports_button = QPushButton("Scan Robot Ports")
        actions.addWidget(self.scan_ports_button)
        self._register_action_button(self.scan_ports_button)

        self.helper_button = QPushButton("Open Run Helper")
        actions.addWidget(self.helper_button)
        self._register_action_button(self.helper_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("DangerButton")
        actions.addWidget(self.cancel_button)
        self._register_action_button(self.cancel_button, is_cancel=True)

        actions.addStretch(1)
        self.form_layout.addLayout(actions)

        self._auto_eval_hint = self.eval_dataset_input.text().strip()
        return self.form_card

    def _build_model_browser_ui(self) -> QWidget:
        self.model_browser_card, model_browser_layout = _build_card("Model Browser")
        self.model_browser_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._populate_model_browser_ui(model_browser_layout)
        main_layout = self.layout()
        if isinstance(main_layout, QVBoxLayout):
            main_layout.insertWidget(2, self.model_browser_card)
            main_layout.insertWidget(3, self.camera_preview)
        self.refresh_model_browser()
        self.restore_model_browser_selection()
        return self.model_browser_card

    def _bind_signals(self) -> None:
        self.eval_prefix_button.clicked.connect(self.apply_eval_prefix_quick_fix)
        self.follower_calibration_browse_button.clicked.connect(
            lambda: self._browse_calibration_file(self.follower_calibration_input)
        )
        self.follower_calibration_auto_button.clicked.connect(lambda: self.follower_calibration_input.setText(""))
        self.leader_calibration_browse_button.clicked.connect(
            lambda: self._browse_calibration_file(self.leader_calibration_input)
        )
        self.leader_calibration_auto_button.clicked.connect(lambda: self.leader_calibration_input.setText(""))
        self.deploy_advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.run_button.clicked.connect(self.run_deploy)
        self.preview_button.clicked.connect(self.preview_command)
        self.preflight_button.clicked.connect(self.run_preflight)
        self.scan_ports_button.clicked.connect(self.scan_robot_ports)
        self.helper_button.clicked.connect(self.open_run_helper)
        self.cancel_button.clicked.connect(self._cancel_run)
        self.browse_root_button.clicked.connect(self.browse_model_root)
        self.refresh_models_button.clicked.connect(self.refresh_model_browser)
        self.browse_model_button.clicked.connect(self.browse_for_model)
        self.upload_model_button.clicked.connect(self.open_model_upload_dialog)
        self.model_tree.itemSelectionChanged.connect(self._handle_model_tree_selection)
        self.models_root_input.editingFinished.connect(self.refresh_model_browser)

    def _build_calibration_row(self, input_widget: QLineEdit) -> tuple[QWidget, QPushButton, QPushButton]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(input_widget, 1)

        browse_button = QPushButton("Browse")
        layout.addWidget(browse_button)

        auto_button = QPushButton("Auto")
        layout.addWidget(auto_button)
        return row, browse_button, auto_button

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

    def _advance_eval_name(self) -> None:
        """Auto-iterate the eval dataset name field to the next available name on HF."""
        current = self.eval_dataset_input.text().strip()
        if not current:
            return
        hf_username = str(self.config.get("hf_username", "")).strip()
        base_name = repo_name_from_repo_id(current)
        iterated = next_available_dataset_name(base_name=base_name, hf_username=hf_username)
        if iterated != base_name:
            new_value = normalize_repo_id(hf_username, iterated) if hf_username else iterated
            self.eval_dataset_input.setText(new_value)
            self._append_log(f"Eval dataset '{base_name}' already exists — advanced to '{iterated}'.")

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)
        if not active:
            self.run_helper_dialog.finish_run(
                status_text=status_text or ("Deploy failed." if is_error else "Deploy completed.")
            )
            if not (status_text or "").lower().startswith("deploy fail"):
                self._advance_eval_name()

    def _append_runtime_line(self, line: str) -> None:
        self.run_helper_dialog.handle_output_line(line)

    def _remember_deploy_artifact(self, artifact_path: Path) -> None:
        super()._remember_run_artifact(artifact_path)
        self._latest_deploy_artifact_path = Path(artifact_path)

    def refresh_from_config(self) -> None:
        self.models_root_input.setText(str(self.config.get("trained_models_dir", "")).strip())
        self.target_hz_input.setText(str(self.config.get("deploy_target_hz", "")).strip())
        self.follower_calibration_input.setText(str(self.config.get("follower_calibration_path", "")).strip())
        self.leader_calibration_input.setText(str(self.config.get("leader_calibration_path", "")).strip())
        self.eval_dataset_input.setText(str(self.config.get("last_eval_dataset_name", "")).strip())
        self._advance_eval_name()

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
            append_output_chunk=self._append_output_chunk,
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

        self.browse_root_button = QPushButton("Browse Root")
        controls.addWidget(self.browse_root_button)

        self.refresh_models_button = QPushButton("Refresh")
        controls.addWidget(self.refresh_models_button)

        self.browse_model_button = QPushButton("Browse Model")
        controls.addWidget(self.browse_model_button)

        self.upload_model_button = QPushButton("Upload Model")
        controls.addWidget(self.upload_model_button)
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
            f"Model metadata\n"
            f"{summarize_model_info(req.model_path)}\n\n"
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
        _DeployWorkflowRunner(
            config=self.config,
            rename_map_flag=self.rename_map_flag,
            models_root_input=self.models_root_input,
            model_path_input=self.model_path_input,
            eval_dataset_input=self.eval_dataset_input,
            follower_calibration_input=self.follower_calibration_input,
            leader_calibration_input=self.leader_calibration_input,
            rename_map_input=self.rename_map_input,
            deploy_advanced_toggle=self.deploy_advanced_toggle,
            deploy_advanced_panel=self.deploy_advanced_panel,
            run_helper_dialog=self.run_helper_dialog,
            run_controller=self._run_controller,
            build=self._build,
            preview_config=self._preview_config,
            browse_calibration_file=self._browse_calibration_file,
            ask_text_dialog_with_actions=self._ask_text_dialog_with_actions,
            ask_editable_command_dialog=self._ask_editable_command_dialog,
            confirm_preflight_review=self._confirm_preflight_review,
            show_text_dialog=self._show_text_dialog,
            show_launch_summary=self._show_launch_summary,
            set_output=self._set_output,
            set_running=self._set_running,
            append_log=self._append_log,
            append_output_and_log=self._append_output_and_log,
            build_hooks=self._build_hooks,
            handle_launch_rejection=self._handle_launch_rejection,
            persist_runtime_outcomes=self._persist_runtime_outcomes,
        ).run()
