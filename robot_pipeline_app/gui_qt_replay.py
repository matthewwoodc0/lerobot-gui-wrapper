from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from .checks import has_failures
from .config_store import get_lerobot_dir, save_config
from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid
from .hardware_workflows import (
    build_replay_preflight_checks,
    build_replay_readiness_summary,
    build_replay_request_and_command,
    discover_replay_episodes,
)
from .run_controller_service import ManagedRunController


class ReplayOpsPanel(_CoreOpsPanel):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Replay",
            subtitle="Replay recorded episodes on hardware with command review, preflight, and artifact capture.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config

        form = _InputGrid(self.form_layout)

        default_dataset = (
            str(config.get("last_dataset_repo_id", "")).strip()
            or str(config.get("last_train_dataset", "")).strip()
            or str(config.get("last_dataset_name", "")).strip()
        )
        self.dataset_input = QLineEdit(default_dataset)
        self.dataset_input.setPlaceholderText("owner/dataset_name")
        self.dataset_input.textChanged.connect(self._refresh_episode_state)
        form.add_field("Dataset", self.dataset_input)

        self.dataset_path_input = QLineEdit("")
        self.dataset_path_input.setPlaceholderText("optional local dataset path override")
        self.dataset_path_input.textChanged.connect(self._refresh_episode_state)
        form.add_field("Dataset path", self.dataset_path_input)

        self.episode_combo = QComboBox()
        self.episode_combo.currentIndexChanged.connect(self._refresh_episode_state)
        form.add_field("Episode", self.episode_combo)

        self.episode_manual_input = QLineEdit("")
        self.episode_manual_input.setPlaceholderText("manual fallback if discovery is incomplete")
        self.episode_manual_input.textChanged.connect(self._refresh_episode_state)
        form.add_field("Manual episode", self.episode_manual_input)

        self.support_label = QLabel("")
        self.support_label.setWordWrap(True)
        self.support_label.setObjectName("MutedLabel")
        self.form_layout.addWidget(self.support_label)

        self.readiness_label = QLabel("")
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setObjectName("MutedLabel")
        self.form_layout.addWidget(self.readiness_label)

        self.replay_advanced_toggle = QCheckBox("Advanced command options")
        self.replay_advanced_toggle.toggled.connect(self._toggle_advanced_options)
        self.form_layout.addWidget(self.replay_advanced_toggle)

        self.replay_advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Replay Options",
            fields=[
                ("dataset.repo_id", "Dataset repo id"),
                ("dataset.path", "Dataset path"),
                ("dataset.root", "Dataset root"),
                ("dataset.episode", "Episode"),
                ("robot.type", "Robot type"),
                ("robot.port", "Robot port"),
                ("robot.id", "Robot id"),
            ],
        )
        self.replay_advanced_panel.hide()
        self.form_layout.addWidget(self.replay_advanced_panel)

        actions = QHBoxLayout()
        run_button = QPushButton("Run Replay")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.run_replay)
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

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("DangerButton")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)
        actions.addStretch(1)
        self.form_layout.addLayout(actions)

        self.refresh_from_config()

    def _build(self) -> tuple[Any | None, list[str] | None, Any, str | None]:
        arg_overrides = None
        custom_args_raw = ""
        if self.replay_advanced_toggle.isChecked():
            arg_overrides, custom_args_raw = self.replay_advanced_panel.build_overrides()
        return build_replay_request_and_command(
            config=self.config,
            dataset_repo_id=self.dataset_input.text(),
            episode_raw=self._episode_raw_value(),
            dataset_path_raw=self.dataset_path_input.text(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def _episode_raw_value(self) -> str:
        manual = self.episode_manual_input.text().strip()
        if manual:
            return manual
        return self.episode_combo.currentText().strip() or "0"

    def _refresh_episode_state(self) -> None:
        repo_id = self.dataset_input.text().strip()
        if not repo_id:
            self.episode_combo.clear()
            self.episode_combo.addItem("0")
            self.episode_manual_input.setEnabled(True)
            self.readiness_label.setText("Enter a dataset repo id to load local episodes and replay readiness.")
            return
        discovery = discover_replay_episodes(self.config, repo_id, dataset_path_raw=self.dataset_path_input.text())
        selected_before = self.episode_combo.currentText().strip() or "0"
        if self.episode_manual_input.text().strip():
            selected_before = self.episode_manual_input.text().strip()
        choices = [str(index) for index in discovery.episode_indices[:500]] or ["0"]
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems(choices)
        if selected_before in choices:
            self.episode_combo.setCurrentText(selected_before)
        self.episode_combo.blockSignals(False)
        self.episode_manual_input.setEnabled(discovery.manual_entry_only or not bool(discovery.episode_indices))

        request, _cmd, support, error = self._build()
        if error or request is None:
            detail = discovery.scan_error or error or "Replay readiness unavailable."
            self.readiness_label.setText(detail)
            self.support_label.setText(str(getattr(support, "detail", detail)))
            return
        summary = build_replay_readiness_summary(config=self.config, request=request, support=support)
        if discovery.scan_error:
            summary += f"\n[WARN] Episode discovery: {discovery.scan_error}"
        self.readiness_label.setText(summary)

    def _toggle_advanced_options(self, checked: bool) -> None:
        if checked:
            request, cmd, _support, error = self._build()
            if error is None and request is not None and cmd is not None:
                self.replay_advanced_panel.seed_from_command(cmd)
            self.replay_advanced_panel.show()
        else:
            self.replay_advanced_panel.hide()

    def refresh_from_config(self) -> None:
        default_dataset = str(self.config.get("last_dataset_repo_id", "")).strip() or str(
            self.config.get("last_train_dataset", "")
        ).strip()
        if default_dataset and not self.dataset_input.text().strip():
            self.dataset_input.setText(default_dataset)
        self._refresh_episode_state()

    def preview_command(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build replay command.",
                log_message="Replay preview failed validation.",
            )
            return
        dataset_path_text = str(request.dataset_path) if request.dataset_path is not None else "not resolved locally"
        self._show_text_dialog(
            title="Replay Command",
            text=(
                f"Dataset: {request.dataset_repo_id}\n"
                f"Episode: {request.episode_index}\n"
                f"Dataset path: {dataset_path_text}\n"
                f"Robot: {request.robot_type} @ {request.robot_port} ({request.robot_id})\n\n"
                f"{build_replay_readiness_summary(config=self.config, request=request, support=support)}\n\n"
                f"{support.detail}\n\n"
                f"{' '.join(str(part) for part in cmd)}"
            ),
            wrap_mode="word",
        )
        self._append_log(f"Replay preview built for {request.dataset_repo_id} episode {request.episode_index}.")

    def run_preflight(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build replay command.",
                log_message="Replay preflight failed validation.",
            )
            return
        checks = build_replay_preflight_checks(config=self.config, request=request, support=support)
        self._show_text_dialog(
            title="Replay Preflight",
            text=build_replay_readiness_summary(config=self.config, request=request, support=support)
            + "\n\n"
            + "\n".join(f"[{level}] {name}: {detail}" for level, name, detail in checks),
            wrap_mode="char",
        )
        self._append_log(f"Replay preflight ran for {request.dataset_repo_id} episode {request.episode_index}.")

    def run_replay(self) -> None:
        request, cmd, support, error = self._build()
        self.support_label.setText(str(support.detail))
        if error or request is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build replay command.",
                log_message="Replay launch failed validation.",
            )
            return

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Replay Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the replay command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Replay",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited replay command from command editor.")
        cmd = editable_cmd

        checks = build_replay_preflight_checks(config=self.config, request=request, support=support)
        if not self._confirm_preflight_review(title="Replay Preflight", checks=checks):
            self._append_log("Replay canceled after preflight review.")
            return

        warning_detail = None
        if has_failures(checks):
            warning_detail = "Replay preflight contains FAIL items. Continue only if you intentionally want to override them."
        self._show_launch_summary(
            heading="Launching replay...",
            command_label="Replay command",
            cmd=cmd,
            preflight_title="Replay Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self._append_log(f"Replay launch starting for {request.dataset_repo_id} episode {request.episode_index}.")

        def after_replay(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Replay canceled.", False)
                self._append_output_and_log("Replay canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Replay failed.", True)
                self._append_output_and_log(f"Replay failed with exit code {return_code}.")
                return
            self.config["last_dataset_repo_id"] = request.dataset_repo_id
            save_config(self.config, quiet=True)
            self._set_running(False, "Replay completed.", False)
            self._append_output_and_log(f"Replay completed for {request.dataset_repo_id} episode {request.episode_index}.")

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_replay,
            run_mode="replay",
            preflight_checks=checks,
            artifact_context={
                "dataset_repo_id": request.dataset_repo_id,
                "dataset_path": str(request.dataset_path) if request.dataset_path is not None else "",
                "replay_episode": request.episode_index,
            },
        )
        if not ok and message:
            self._handle_launch_rejection(
                title="Replay Unavailable",
                message=message,
                log_message="Replay launch was rejected.",
            )
