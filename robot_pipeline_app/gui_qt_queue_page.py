from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem

from .config_store import get_lerobot_dir
from .gui_qt_page_base import _InputGrid, _PageWithOutput, _build_card, _json_text, _set_readonly_table, _set_table_headers
from .history_utils import open_path_in_file_manager
from .workflow_queue import (
    WorkflowQueueService,
    build_record_upload_queue_item,
    build_train_deploy_eval_queue_item,
    build_train_sim_eval_queue_item,
)


class QtWorkflowQueuePage(_PageWithOutput):
    _POLICY_TYPES = ["act", "diffusion", "tdmpc", "vq_bet", "pi0fast", "smolvla"]
    _DEVICE_OPTIONS = ["", "cuda", "mps", "cpu"]

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        workflow_queue: WorkflowQueueService,
    ) -> None:
        super().__init__(
            title="Queue",
            subtitle="Queue lightweight local recipes such as record/upload and train/eval follow-ups.",
            append_log=append_log,
        )
        self.config = config
        self._workflow_queue = workflow_queue
        self._rows: list[dict[str, Any]] = []
        self._workflow_queue.add_listener(self.refresh_queue_views)

        self.content_layout.addWidget(self._build_record_recipe_card())
        self.content_layout.addWidget(self._build_train_defaults_card())
        self.content_layout.addWidget(self._build_sim_eval_recipe_card())
        self.content_layout.addWidget(self._build_deploy_eval_recipe_card())
        self.content_layout.addWidget(self._build_queue_status_card())
        self.content_layout.addStretch(1)
        self.refresh_from_config()
        self.refresh_queue_views()

    def _build_record_recipe_card(self):
        card, layout = _build_card("Recipe: Record -> Upload")
        form = _InputGrid(layout)

        default_dataset = str(self.config.get("last_dataset_repo_id", "")).strip() or str(self.config.get("last_dataset_name", "")).strip()
        self.record_dataset_input = QLineEdit(default_dataset)
        form.add_field("Dataset", self.record_dataset_input)

        self.record_dataset_root_input = QLineEdit(str(self.config.get("record_data_dir", "")))
        form.add_field("Dataset root", self.record_dataset_root_input)

        self.record_task_input = QLineEdit(str(self.config.get("eval_task", "")))
        form.add_field("Task", self.record_task_input)

        self.record_episodes_input = QSpinBox()
        self.record_episodes_input.setRange(1, 10_000)
        self.record_episodes_input.setValue(10)
        form.add_field("Episodes", self.record_episodes_input)

        self.record_duration_input = QSpinBox()
        self.record_duration_input.setRange(1, 3600)
        self.record_duration_input.setValue(20)
        form.add_field("Episode time (s)", self.record_duration_input)

        self.record_target_hz_input = QLineEdit(str(self.config.get("record_target_hz", "")).strip())
        self.record_target_hz_input.setPlaceholderText("optional")
        form.add_field("Target Hz", self.record_target_hz_input)

        actions = QHBoxLayout()
        enqueue_button = QPushButton("Queue Record -> Upload")
        enqueue_button.clicked.connect(self.enqueue_record_upload)
        actions.addWidget(enqueue_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_train_defaults_card(self):
        card, layout = _build_card("Shared Train Defaults")
        form = _InputGrid(layout)

        default_dataset = str(self.config.get("last_train_dataset", "")).strip() or str(self.config.get("last_dataset_repo_id", "")).strip()
        self.train_dataset_input = QLineEdit(default_dataset)
        form.add_field("Dataset", self.train_dataset_input)

        self.train_policy_type_combo = QComboBox()
        self.train_policy_type_combo.setEditable(True)
        self.train_policy_type_combo.addItems(self._POLICY_TYPES)
        self.train_policy_type_combo.setCurrentText(str(self.config.get("last_train_policy_type", "act")).strip() or "act")
        form.add_field("Policy type", self.train_policy_type_combo)

        default_output = str(self.config.get("trained_models_dir", "")).strip() or str(get_lerobot_dir(self.config) / "trained_models")
        self.train_output_dir_input = QLineEdit(default_output)
        form.add_field("Output dir", self.train_output_dir_input)

        self.train_device_combo = QComboBox()
        self.train_device_combo.addItems(self._DEVICE_OPTIONS)
        form.add_field("Device", self.train_device_combo)

        self.train_job_name_input = QLineEdit("")
        self.train_job_name_input.setPlaceholderText("optional")
        form.add_field("Job name", self.train_job_name_input)

        self.train_resume_input = QLineEdit("")
        self.train_resume_input.setPlaceholderText("optional checkpoint/config path")
        form.add_field("Resume", self.train_resume_input)

        self.train_custom_args_input = QLineEdit("")
        self.train_custom_args_input.setPlaceholderText("optional extra train flags")
        form.add_field("Extra flags", self.train_custom_args_input)

        self.train_wandb_checkbox = QCheckBox("Enable WandB")
        layout.addWidget(self.train_wandb_checkbox)

        self.train_wandb_project_input = QLineEdit("")
        self.train_wandb_project_input.setPlaceholderText("optional WandB project")
        layout.addWidget(self.train_wandb_project_input)
        return card

    def _build_sim_eval_recipe_card(self):
        card, layout = _build_card("Recipe: Train -> Sim Eval")
        form = _InputGrid(layout)

        self.sim_env_type_input = QLineEdit(str(self.config.get("workflow_sim_eval_env_type", "")).strip())
        self.sim_env_type_input.setPlaceholderText("optional if benchmark set")
        form.add_field("Env type", self.sim_env_type_input)

        self.sim_benchmark_input = QLineEdit(str(self.config.get("workflow_sim_eval_benchmark", "")).strip())
        self.sim_benchmark_input.setPlaceholderText("optional if env type set")
        form.add_field("Benchmark", self.sim_benchmark_input)

        self.sim_task_input = QLineEdit(str(self.config.get("eval_task", "")).strip())
        form.add_field("Task", self.sim_task_input)

        self.sim_episodes_input = QSpinBox()
        self.sim_episodes_input.setRange(1, 10_000)
        self.sim_episodes_input.setValue(max(int(self.config.get("eval_num_episodes", 10) or 10), 1))
        form.add_field("Episodes", self.sim_episodes_input)

        default_output = str(Path(str(self.config.get("runs_dir", "")).strip() or ".") / "sim_eval")
        self.sim_output_dir_input = QLineEdit(default_output)
        form.add_field("Output dir", self.sim_output_dir_input)

        self.sim_device_input = QLineEdit("")
        self.sim_device_input.setPlaceholderText("optional")
        form.add_field("Device", self.sim_device_input)

        self.sim_trust_remote_code_checkbox = QCheckBox("Trust remote code when required")
        layout.addWidget(self.sim_trust_remote_code_checkbox)

        actions = QHBoxLayout()
        enqueue_button = QPushButton("Queue Train -> Sim Eval")
        enqueue_button.clicked.connect(self.enqueue_train_sim_eval)
        actions.addWidget(enqueue_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_deploy_eval_recipe_card(self):
        card, layout = _build_card("Recipe: Train -> Deploy Eval")
        form = _InputGrid(layout)

        default_eval_dataset = str(self.config.get("last_eval_dataset_name", "")).strip()
        self.deploy_eval_dataset_input = QLineEdit(default_eval_dataset)
        form.add_field("Eval dataset", self.deploy_eval_dataset_input)

        self.deploy_eval_task_input = QLineEdit(str(self.config.get("eval_task", "")).strip())
        form.add_field("Task", self.deploy_eval_task_input)

        self.deploy_eval_episodes_input = QSpinBox()
        self.deploy_eval_episodes_input.setRange(1, 10_000)
        self.deploy_eval_episodes_input.setValue(max(int(self.config.get("eval_num_episodes", 10) or 10), 1))
        form.add_field("Episodes", self.deploy_eval_episodes_input)

        self.deploy_eval_duration_input = QSpinBox()
        self.deploy_eval_duration_input.setRange(1, 3600)
        self.deploy_eval_duration_input.setValue(max(int(self.config.get("eval_duration_s", 20) or 20), 1))
        form.add_field("Episode time (s)", self.deploy_eval_duration_input)

        self.deploy_target_hz_input = QLineEdit(str(self.config.get("deploy_target_hz", "")).strip())
        self.deploy_target_hz_input.setPlaceholderText("optional")
        form.add_field("Target Hz", self.deploy_target_hz_input)

        actions = QHBoxLayout()
        enqueue_button = QPushButton("Queue Train -> Deploy Eval")
        enqueue_button.clicked.connect(self.enqueue_train_deploy_eval)
        actions.addWidget(enqueue_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_queue_status_card(self):
        card, layout = _build_card("Queue Status")
        self.queue_table = QTableWidget(0, 5)
        _set_table_headers(self.queue_table, ["ID", "Recipe", "Status", "Step", "Artifacts"])
        _set_readonly_table(self.queue_table)
        self.queue_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.queue_table)

        actions = QHBoxLayout()
        cancel_button = QPushButton("Cancel Active")
        cancel_button.clicked.connect(self.cancel_active_workflow)
        actions.addWidget(cancel_button)

        open_button = QPushButton("Open Latest Artifact")
        open_button.clicked.connect(self.open_latest_artifact)
        actions.addWidget(open_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _train_form_values(self) -> dict[str, Any]:
        return {
            "dataset_repo_id": self.train_dataset_input.text(),
            "policy_type": self.train_policy_type_combo.currentText(),
            "output_dir": self.train_output_dir_input.text(),
            "device": self.train_device_combo.currentText(),
            "job_name": self.train_job_name_input.text(),
            "resume_from": self.train_resume_input.text(),
            "custom_args": self.train_custom_args_input.text(),
            "wandb_enabled": self.train_wandb_checkbox.isChecked(),
            "wandb_project": self.train_wandb_project_input.text(),
        }

    def enqueue_record_upload(self) -> None:
        queue_id = self._workflow_queue.next_queue_id()
        item = build_record_upload_queue_item(
            queue_id=queue_id,
            dataset_input=self.record_dataset_input.text(),
            episodes_raw=str(self.record_episodes_input.value()),
            duration_raw=str(self.record_duration_input.value()),
            task_raw=self.record_task_input.text(),
            dataset_dir_raw=self.record_dataset_root_input.text(),
            target_hz_raw=self.record_target_hz_input.text(),
        )
        ok, message = self._workflow_queue.enqueue(item)
        self._set_output(title="Queued" if ok else "Queue Failed", text=message, log_message=message)

    def enqueue_train_sim_eval(self) -> None:
        queue_id = self._workflow_queue.next_queue_id()
        item = build_train_sim_eval_queue_item(
            queue_id=queue_id,
            train_form_values=self._train_form_values(),
            sim_eval_settings={
                "env_type": self.sim_env_type_input.text(),
                "benchmark": self.sim_benchmark_input.text(),
                "task": self.sim_task_input.text(),
                "episodes": str(self.sim_episodes_input.value()),
                "output_dir": self.sim_output_dir_input.text(),
                "device": self.sim_device_input.text(),
                "trust_remote_code": self.sim_trust_remote_code_checkbox.isChecked(),
            },
        )
        ok, message = self._workflow_queue.enqueue(item)
        self._set_output(title="Queued" if ok else "Queue Failed", text=message, log_message=message)

    def enqueue_train_deploy_eval(self) -> None:
        queue_id = self._workflow_queue.next_queue_id()
        item = build_train_deploy_eval_queue_item(
            queue_id=queue_id,
            train_form_values=self._train_form_values(),
            deploy_settings={
                "deploy_root_raw": self.train_output_dir_input.text(),
                "eval_dataset_raw": self.deploy_eval_dataset_input.text(),
                "eval_episodes_raw": str(self.deploy_eval_episodes_input.value()),
                "eval_duration_raw": str(self.deploy_eval_duration_input.value()),
                "eval_task_raw": self.deploy_eval_task_input.text(),
                "target_hz_raw": self.deploy_target_hz_input.text(),
            },
        )
        ok, message = self._workflow_queue.enqueue(item)
        self._set_output(title="Queued" if ok else "Queue Failed", text=message, log_message=message)

    def refresh_queue_views(self) -> None:
        self._rows = list(self._workflow_queue.snapshots())
        self.queue_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = [
                row.get("queue_id", ""),
                row.get("title", ""),
                row.get("status", ""),
                row.get("current_step_label", ""),
                len(row.get("artifacts", [])) if isinstance(row.get("artifacts"), list) else 0,
            ]
            for col_index, value in enumerate(values):
                self.queue_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if self._rows and self.queue_table.currentRow() < 0:
            self.queue_table.selectRow(0)
        elif not self._rows:
            self.output_card.show()
            self._set_output(title="Queue", text="No queued workflows yet.", log_message=None)
        self._on_selection_changed()

    def _selected_row(self) -> dict[str, Any] | None:
        row = self.queue_table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def _on_selection_changed(self) -> None:
        row = self._selected_row()
        self.output_card.show()
        if row is None:
            self._set_output(title="Queue", text="Select a queued workflow to inspect it.", log_message=None)
            return
        payload = {
            "queue_id": row.get("queue_id"),
            "title": row.get("title"),
            "status": row.get("status"),
            "current_step": row.get("current_step_label"),
            "artifacts": row.get("artifacts"),
            "error_text": row.get("error_text"),
        }
        text = _json_text(payload)
        log_text = str(row.get("log_text", "")).strip()
        if log_text:
            text += "\n\nLog\n\n" + log_text
        self._set_output(title="Workflow Details", text=text, log_message=None)

    def cancel_active_workflow(self) -> None:
        ok, message = self._workflow_queue.cancel_active()
        self._set_output(title="Cancel Requested" if ok else "Cancel Failed", text=message, log_message=message)

    def open_latest_artifact(self) -> None:
        row = self._selected_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a queued workflow first.", log_message="Queue artifact open skipped with no selection.")
            return
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            self._set_output(title="No Artifact", text="This workflow has not produced any artifacts yet.", log_message="Queue artifact open skipped with no artifacts.")
            return
        latest = artifacts[-1]
        target = Path(str(latest.get("path", "")).strip())
        ok, message = open_path_in_file_manager(target)
        self._set_output(title="Open Artifact" if ok else "Open Failed", text=str(target if ok else message), log_message="Queue artifact open requested." if ok else "Queue artifact open failed.")

    def refresh_from_config(self) -> None:
        if not self.record_dataset_root_input.text().strip():
            self.record_dataset_root_input.setText(str(self.config.get("record_data_dir", "")))
        if not self.train_output_dir_input.text().strip():
            self.train_output_dir_input.setText(str(self.config.get("trained_models_dir", "")))
