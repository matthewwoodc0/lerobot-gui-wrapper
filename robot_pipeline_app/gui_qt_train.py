from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

import re as _re

from .checks import run_preflight_for_train, summarize_checks
from .command_overrides import get_flag_value
from .config_store import get_lerobot_dir, save_config
from .gui_forms import build_train_request_and_command
from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid
from .repo_utils import model_exists_on_hf, normalize_repo_id
from .run_controller_service import ManagedRunController


class TrainOpsPanel(_CoreOpsPanel):
    _POLICY_TYPES = ["act", "diffusion", "tdmpc", "vq_bet", "pi0fast", "smolvla"]
    _DEVICE_OPTIONS = ["", "cuda", "mps", "cpu"]

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Train",
            subtitle="Configure and launch policy training with live terminal output.",
            append_log=append_log,
            run_controller=run_controller,
        )
        self.config = config

        form = _InputGrid(self.form_layout)

        default_dataset = str(config.get("last_dataset_repo_id", "")).strip() or str(config.get("last_train_dataset", "")).strip()
        self.dataset_input = QLineEdit(default_dataset)
        self.dataset_input.setPlaceholderText("owner/dataset_name")
        form.add_field("Dataset", self.dataset_input)

        self.policy_type_combo = QComboBox()
        self.policy_type_combo.setEditable(True)
        self.policy_type_combo.addItems(self._POLICY_TYPES)
        self.policy_type_combo.setCurrentText(str(config.get("last_train_policy_type", "act")).strip() or "act")
        form.add_field("Policy type", self.policy_type_combo)

        default_output_dir = str(config.get("trained_models_dir", "outputs/train")).strip() or "outputs/train"
        self.output_dir_input = QLineEdit(default_output_dir)
        self.output_dir_input.setPlaceholderText("outputs/train")
        form.add_field("Output dir", self._build_browse_row(self.output_dir_input, browse_kind="directory"))

        self.device_combo = QComboBox()
        self.device_combo.addItems(self._DEVICE_OPTIONS)
        self.device_combo.setCurrentIndex(0)
        self.device_combo.setToolTip("Auto-detect if empty.")
        form.add_field("Device", self.device_combo)

        self.job_name_input = QLineEdit("")
        self.job_name_input.setPlaceholderText("optional")
        form.add_field("Job name", self.job_name_input)

        self.resume_from_input = QLineEdit("")
        self.resume_from_input.setPlaceholderText("train_config.json or checkpoint folder (optional)")
        self.resume_from_input.setToolTip(
            "Resume only works when the detected LeRobot train entrypoint exposes a real checkpoint/config path flag."
        )
        form.add_field("Resume checkpoint/config", self._build_browse_row(self.resume_from_input, browse_kind="file"))

        self.wandb_checkbox = QCheckBox("Enable WandB logging")
        self.wandb_checkbox.setChecked(False)
        self.wandb_checkbox.toggled.connect(self._toggle_wandb_project)
        self.form_layout.addWidget(self.wandb_checkbox)

        self.wandb_project_row = QWidget()
        wandb_layout = QHBoxLayout(self.wandb_project_row)
        wandb_layout.setContentsMargins(0, 0, 0, 0)
        wandb_layout.setSpacing(8)
        wandb_label = QLabel("WandB project")
        wandb_label.setObjectName("FormLabel")
        wandb_layout.addWidget(wandb_label)
        self.wandb_project_input = QLineEdit("")
        self.wandb_project_input.setPlaceholderText("optional")
        wandb_layout.addWidget(self.wandb_project_input, 1)
        self.wandb_project_row.hide()
        self.form_layout.addWidget(self.wandb_project_row)

        self.train_advanced_panel = _AdvancedOptionsPanel(
            title="Advanced Training Options",
            fields=[],
        )
        self.form_layout.addWidget(self.train_advanced_panel)

        actions = QHBoxLayout()

        preflight_button = QPushButton("Run Preflight")
        preflight_button.setObjectName("AccentButton")
        preflight_button.clicked.connect(self.run_preflight)
        actions.addWidget(preflight_button)
        self._register_action_button(preflight_button)

        run_button = QPushButton("Start Training")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.run_train)
        actions.addWidget(run_button)
        self._register_action_button(run_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("DangerButton")
        cancel_button.clicked.connect(self._cancel_run)
        actions.addWidget(cancel_button)
        self._register_action_button(cancel_button, is_cancel=True)

        actions.addStretch(1)
        self.form_layout.addLayout(actions)
        self._advance_job_name()

    def _advance_job_name(self) -> None:
        """Auto-iterate job_name if the output directory for that run already exists."""
        job_name = self.job_name_input.text().strip()
        if not job_name:
            return
        output_dir_text = self.output_dir_input.text().strip() or str(self.config.get("trained_models_dir", "outputs/train"))
        output_dir = Path(output_dir_text).expanduser()
        hf_username = str(self.config.get("hf_username", "")).strip()

        bare = _re.sub(r"_\d+$", "", job_name)

        def _candidate(n: int) -> str:
            return bare if n == 1 else f"{bare}_{n}"

        def _run_dir_occupied(name: str) -> bool:
            d = output_dir / name
            return d.exists() and any(d.iterdir())

        def _hf_model_occupied(name: str) -> bool:
            if not hf_username:
                return False
            return bool(model_exists_on_hf(normalize_repo_id(hf_username, name)))

        for n in range(1, 100):
            candidate = _candidate(n)
            if not _run_dir_occupied(candidate) and not _hf_model_occupied(candidate):
                if candidate != job_name:
                    self.job_name_input.setText(candidate)
                    self._append_log(f"Job name '{job_name}' already exists — advanced to '{candidate}'.")
                return

    def _build_browse_row(self, target_input: QLineEdit, *, browse_kind: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(target_input, 1)

        browse_button = QPushButton("Browse...")
        if browse_kind == "directory":
            browse_button.clicked.connect(lambda: self._browse_directory(target_input))
        else:
            browse_button.clicked.connect(lambda: self._browse_file(target_input))
        layout.addWidget(browse_button)
        return row

    def _browse_directory(self, target_input: QLineEdit) -> None:
        current = target_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select Output Directory", current)
        if selected:
            target_input.setText(selected)

    def _browse_file(self, target_input: QLineEdit) -> None:
        current = target_input.text().strip()
        initial_dir = str(Path(current).expanduser().parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint",
            initial_dir,
            "All files (*)",
        )
        if selected:
            target_input.setText(selected)

    def _toggle_wandb_project(self, checked: bool) -> None:
        self.wandb_project_row.setVisible(checked)

    def _ensure_output_visible(self) -> None:
        self.output_card.show()

    def _set_output(self, *, title: str, text: str, log_message: str) -> None:
        self._ensure_output_visible()
        super()._set_output(title=title, text=text, log_message=log_message)

    def _append_output_line(self, line: str) -> None:
        self._ensure_output_visible()
        super()._append_output_line(line)

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
        self._ensure_output_visible()
        super()._show_launch_summary(
            heading=heading,
            command_label=command_label,
            cmd=cmd,
            preflight_title=preflight_title,
            preflight_checks=preflight_checks,
            warning_detail=warning_detail,
        )

    def _form_values(self) -> dict[str, Any]:
        return {
            "dataset_repo_id": self.dataset_input.text(),
            "policy_type": self.policy_type_combo.currentText(),
            "output_dir": self.output_dir_input.text(),
            "device": self.device_combo.currentText(),
            "wandb_enabled": self.wandb_checkbox.isChecked(),
            "wandb_project": self.wandb_project_input.text(),
            "job_name": self.job_name_input.text(),
            "resume_from": self.resume_from_input.text(),
            "custom_args": self.train_advanced_panel.custom_args_input.text(),
        }

    def _build(self) -> tuple[Any | None, list[str] | None, str | None]:
        return build_train_request_and_command(form_values=self._form_values(), config=self.config)

    def _effective_form_values(self, req: Any, cmd: list[str]) -> dict[str, Any]:
        return {
            "dataset_repo_id": get_flag_value(cmd, "dataset.repo_id") or req.dataset_repo_id,
            "policy_type": get_flag_value(cmd, "policy.type") or req.policy_type,
            "output_dir": get_flag_value(cmd, "output_dir") or req.output_dir,
            "device": get_flag_value(cmd, "policy.device") or req.device,
            "wandb_enabled": (get_flag_value(cmd, "wandb.enable") or "").strip().lower() == "true" or req.wandb_enabled,
            "wandb_project": get_flag_value(cmd, "wandb.project") or req.wandb_project,
            "job_name": get_flag_value(cmd, "job_name") or req.job_name,
            "resume_from": req.resume_from,
        }

    def run_preflight(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build training command.",
                log_message="Training preflight failed validation.",
            )
            return
        checks = run_preflight_for_train(self.config, self._effective_form_values(req, cmd))
        self._set_output(
            title="Training Preflight",
            text=summarize_checks(checks, title="Training Preflight"),
            log_message=f"Training preflight ran for {req.dataset_repo_id}.",
        )

    def run_train(self) -> None:
        req, cmd, error = self._build()
        if error or req is None or cmd is None:
            self._set_output(
                title="Validation Error",
                text=error or "Unable to build training command.",
                log_message="Training launch failed validation.",
            )
            return

        editable_cmd = self._ask_editable_command_dialog(
            title="Confirm Training Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the training command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Start Training",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited training command from command editor.")
        cmd = editable_cmd

        effective_values = self._effective_form_values(req, cmd)
        checks = run_preflight_for_train(self.config, effective_values)
        if not self._confirm_preflight_review(title="Training Preflight", checks=checks):
            self._append_log("Training canceled after preflight review.")
            return

        warning_detail = None
        if any(str(level).strip().upper() == "WARN" for level, _name, _detail in checks):
            warning_detail = "Warnings were detected. The workflow continues automatically when there are no FAIL checks."
        self._show_launch_summary(
            heading="Launching training run...",
            command_label="Training command",
            cmd=cmd,
            preflight_title="Training Preflight",
            preflight_checks=checks,
            warning_detail=warning_detail,
        )
        self._append_log(
            f"Training launch starting for {effective_values['dataset_repo_id']} ({effective_values['policy_type']})."
        )

        def after_train(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Training canceled.", False)
                self._append_output_and_log("Training run canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Training failed.", True)
                self._append_output_and_log(f"Training run failed with exit code {return_code}.")
                self._advance_job_name()
                return
            self.config["last_train_policy_type"] = str(effective_values["policy_type"])
            self.config["last_train_dataset"] = str(effective_values["dataset_repo_id"])
            save_config(self.config, quiet=True)
            self._set_running(False, "Training completed.", False)
            self._advance_job_name()
            self._append_output_and_log(
                f"Training completed for {effective_values['dataset_repo_id']} ({effective_values['policy_type']})."
            )

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_train,
            run_mode="train",
            preflight_checks=checks,
            artifact_context={
                "dataset_repo_id": str(effective_values["dataset_repo_id"]),
                "policy_type": str(effective_values["policy_type"]),
                "output_dir": str(effective_values["output_dir"]),
                "device": str(effective_values["device"]),
                "job_name": str(effective_values["job_name"]),
                "resume_from": str(effective_values["resume_from"]),
                "wandb_enabled": bool(effective_values["wandb_enabled"]),
                "wandb_project": str(effective_values["wandb_project"]),
            },
        )
        if not ok and message:
            self._handle_launch_rejection(
                title="Training Unavailable",
                message=message,
                log_message="Training launch was rejected.",
            )
