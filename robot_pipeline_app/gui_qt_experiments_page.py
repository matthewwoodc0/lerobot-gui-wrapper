from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from .checks import run_preflight_for_deploy, summarize_checks
from .config_store import save_config
from .experiments_service import (
    build_experiment_comparison_payload,
    build_experiment_details_text,
    collect_experiment_runs,
    fetch_wandb_remote_snapshot,
)
from .gui_forms import build_deploy_request_and_command
from .gui_qt_page_base import _InputGrid, _PageWithOutput, _build_card, _set_readonly_table, _set_table_headers
from .history_utils import open_path_in_file_manager
from .run_controller_service import ManagedRunController, RunUiHooks
from .sim_eval import build_sim_eval_request_and_command
from .visualizer_utils import _open_path


class QtExperimentsPage(_PageWithOutput):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Experiments",
            subtitle="Compare train, deploy, and simulation eval runs with connected checkpoints and metrics.",
            append_log=append_log,
            use_output_tabs=True,
        )
        self.config = config
        self._run_controller = run_controller
        self._records: list[dict[str, Any]] = []
        self._visible_records: list[dict[str, Any]] = []
        self._checkpoint_rows: list[dict[str, Any]] = []
        self._action_buttons: list[QPushButton] = []
        self._latest_run_artifact_path: Path | None = None
        self._set_explain_callback(None)

        filters_card, filters_layout = _build_card("Experiment Runs")
        filter_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("All runs", "all")
        self.mode_combo.addItem("Train", "train")
        self.mode_combo.addItem("Deploy", "deploy")
        self.mode_combo.addItem("Sim eval", "sim_eval")
        filter_row.addWidget(self.mode_combo)

        self.status_combo = QComboBox()
        self.status_combo.addItem("All statuses", "all")
        self.status_combo.addItem("Success", "success")
        self.status_combo.addItem("Failed", "failed")
        self.status_combo.addItem("Canceled", "canceled")
        filter_row.addWidget(self.status_combo)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Filter by dataset, env, policy, checkpoint, or command")
        filter_row.addWidget(self.query_input, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_experiments)
        filter_row.addWidget(refresh_button)
        self._action_buttons.append(refresh_button)
        filters_layout.addLayout(filter_row)

        actions_row = QHBoxLayout()
        self.compare_button = QPushButton("Compare Selected")
        self.compare_button.clicked.connect(self.compare_selected_runs)
        actions_row.addWidget(self.compare_button)
        self._action_buttons.append(self.compare_button)

        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.clicked.connect(self.open_selected_run_folder)
        actions_row.addWidget(self.open_run_button)
        self._action_buttons.append(self.open_run_button)

        self.open_output_button = QPushButton("Open Output")
        self.open_output_button.clicked.connect(self.open_selected_output)
        actions_row.addWidget(self.open_output_button)
        self._action_buttons.append(self.open_output_button)

        self.open_wandb_button = QPushButton("Open WandB Run")
        self.open_wandb_button.clicked.connect(self.open_selected_wandb_run)
        actions_row.addWidget(self.open_wandb_button)
        self._action_buttons.append(self.open_wandb_button)

        actions_row.addStretch(1)
        filters_layout.addLayout(actions_row)

        self.experiment_status = QLabel("Loading experiment runs...")
        self.experiment_status.setObjectName("MutedLabel")
        self.experiment_status.setWordWrap(True)
        filters_layout.addWidget(self.experiment_status)
        self.content_layout.addWidget(filters_card)

        table_card, table_layout = _build_card("Run Table")
        self.run_table = QTableWidget(0, 7)
        _set_table_headers(
            self.run_table,
            ["Started", "Type", "Status", "Dataset / Env", "Policy", "Checkpoint", "Metrics"],
        )
        _set_readonly_table(self.run_table)
        self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.run_table.itemSelectionChanged.connect(self._on_run_selection_changed)
        table_layout.addWidget(self.run_table)
        self.content_layout.addWidget(table_card)

        comparison_card, comparison_layout = _build_card("Comparison")
        self.compare_table = QTableWidget(0, 11)
        _set_table_headers(
            self.compare_table,
            [
                "Type",
                "Status",
                "Dataset / Env",
                "Policy",
                "Checkpoint",
                "Device",
                "Duration",
                "Notes / Tags",
                "Output",
                "Metrics",
                "WandB",
            ],
        )
        _set_readonly_table(self.compare_table)
        comparison_layout.addWidget(self.compare_table)
        self.compare_status = QLabel("Select multiple runs to compare train, deploy, and sim-eval outcomes.")
        self.compare_status.setObjectName("MutedLabel")
        self.compare_status.setWordWrap(True)
        comparison_layout.addWidget(self.compare_status)
        self.content_layout.addWidget(comparison_card)

        checkpoints_card, checkpoints_layout = _build_card("Checkpoints")
        self.checkpoints_card = checkpoints_card
        self.selected_train_label = QLabel("Select a training run to inspect checkpoints.")
        self.selected_train_label.setObjectName("MutedLabel")
        self.selected_train_label.setWordWrap(True)
        checkpoints_layout.addWidget(self.selected_train_label)

        self.checkpoint_table = QTableWidget(0, 5)
        _set_table_headers(self.checkpoint_table, ["Checkpoint", "Kind", "Step", "Policy", "Path"])
        _set_readonly_table(self.checkpoint_table)
        self.checkpoint_table.itemSelectionChanged.connect(self._update_checkpoint_buttons)
        checkpoints_layout.addWidget(self.checkpoint_table)

        checkpoint_actions = QHBoxLayout()
        self.open_checkpoint_button = QPushButton("Open Checkpoint Folder")
        self.open_checkpoint_button.clicked.connect(self.open_selected_checkpoint_folder)
        checkpoint_actions.addWidget(self.open_checkpoint_button)
        self._action_buttons.append(self.open_checkpoint_button)

        self.open_train_config_button = QPushButton("Open train_config.json")
        self.open_train_config_button.clicked.connect(self.open_selected_checkpoint_config)
        checkpoint_actions.addWidget(self.open_train_config_button)
        self._action_buttons.append(self.open_train_config_button)
        checkpoint_actions.addStretch(1)
        checkpoints_layout.addLayout(checkpoint_actions)
        self.content_layout.addWidget(checkpoints_card)
        self.checkpoints_card.hide()

        deploy_card, deploy_layout = _build_card("Deploy From Checkpoint")
        deploy_form = _InputGrid(deploy_layout)
        self.deploy_dataset_input = QLineEdit(str(config.get("last_eval_dataset_name", "")).strip())
        self.deploy_dataset_input.setPlaceholderText("owner/eval_dataset")
        deploy_form.add_field("Eval dataset", self.deploy_dataset_input)

        self.deploy_episodes_input = QLineEdit(str(config.get("eval_num_episodes", 10)))
        deploy_form.add_field("Episodes", self.deploy_episodes_input)

        self.deploy_duration_input = QLineEdit(str(config.get("eval_duration_s", 20)))
        deploy_form.add_field("Duration (s)", self.deploy_duration_input)

        self.deploy_task_input = QLineEdit(str(config.get("eval_task", "")))
        self.deploy_task_input.setPlaceholderText("Eval task")
        deploy_form.add_field("Task", self.deploy_task_input)

        deploy_actions = QHBoxLayout()
        self.deploy_checkpoint_button = QPushButton("Launch Deploy Eval")
        self.deploy_checkpoint_button.setObjectName("AccentButton")
        self.deploy_checkpoint_button.clicked.connect(self.launch_deploy_from_checkpoint)
        deploy_actions.addWidget(self.deploy_checkpoint_button)
        self._action_buttons.append(self.deploy_checkpoint_button)
        deploy_actions.addStretch(1)
        deploy_layout.addLayout(deploy_actions)

        self.deploy_launch_status = QLabel("Deploy uses the selected checkpoint and the existing deploy/eval machinery.")
        self.deploy_launch_status.setObjectName("MutedLabel")
        self.deploy_launch_status.setWordWrap(True)
        deploy_layout.addWidget(self.deploy_launch_status)
        self.content_layout.addWidget(deploy_card)

        sim_eval_card, sim_eval_layout = _build_card("Sim Eval From Checkpoint")
        sim_eval_form = _InputGrid(sim_eval_layout)
        self.sim_env_type_input = QLineEdit(str(config.get("ui_sim_eval_env_type", "")).strip())
        self.sim_env_type_input.setPlaceholderText("pusht, metaworld, libero, ...")
        sim_eval_form.add_field("Env type", self.sim_env_type_input)

        self.sim_task_input = QLineEdit(str(config.get("ui_sim_eval_task", "")).strip())
        self.sim_task_input.setPlaceholderText("optional task")
        sim_eval_form.add_field("Task", self.sim_task_input)

        self.sim_benchmark_input = QLineEdit(str(config.get("ui_sim_eval_benchmark", "")).strip())
        self.sim_benchmark_input.setPlaceholderText("optional benchmark")
        sim_eval_form.add_field("Benchmark", self.sim_benchmark_input)

        self.sim_episodes_input = QLineEdit(str(config.get("ui_sim_eval_episodes", 10)))
        sim_eval_form.add_field("Episodes", self.sim_episodes_input)

        self.sim_batch_size_input = QLineEdit(str(config.get("ui_sim_eval_batch_size", "")).strip())
        self.sim_batch_size_input.setPlaceholderText("optional")
        sim_eval_form.add_field("Batch size", self.sim_batch_size_input)

        self.sim_seed_input = QLineEdit(str(config.get("ui_sim_eval_seed", "")).strip())
        self.sim_seed_input.setPlaceholderText("optional")
        sim_eval_form.add_field("Seed", self.sim_seed_input)

        self.sim_device_input = QLineEdit(str(config.get("ui_sim_eval_device", "")).strip())
        self.sim_device_input.setPlaceholderText("cuda / mps / cpu (optional)")
        sim_eval_form.add_field("Device", self.sim_device_input)

        self.sim_output_dir_input = QLineEdit(str(config.get("ui_sim_eval_output_dir", "outputs/eval")).strip() or "outputs/eval")
        sim_eval_form.add_field("Output dir", self.sim_output_dir_input)

        self.sim_job_name_input = QLineEdit(str(config.get("ui_sim_eval_job_name", "")).strip())
        self.sim_job_name_input.setPlaceholderText("optional")
        sim_eval_form.add_field("Job name", self.sim_job_name_input)

        self.sim_custom_args_input = QLineEdit("")
        self.sim_custom_args_input.setPlaceholderText("optional extra flags")
        sim_eval_form.add_field("Custom args", self.sim_custom_args_input)

        self.sim_trust_remote_code_checkbox = QCheckBox("Enable --trust_remote_code when supported")
        sim_eval_layout.addWidget(self.sim_trust_remote_code_checkbox)

        sim_eval_actions = QHBoxLayout()
        self.sim_eval_checkpoint_button = QPushButton("Launch Sim Eval")
        self.sim_eval_checkpoint_button.setObjectName("AccentButton")
        self.sim_eval_checkpoint_button.clicked.connect(self.launch_sim_eval_from_checkpoint)
        sim_eval_actions.addWidget(self.sim_eval_checkpoint_button)
        self._action_buttons.append(self.sim_eval_checkpoint_button)
        sim_eval_actions.addStretch(1)
        sim_eval_layout.addLayout(sim_eval_actions)

        self.sim_eval_status = QLabel("Sim eval uses compatibility-probed LeRobot eval entrypoints and supported flags only.")
        self.sim_eval_status.setObjectName("MutedLabel")
        self.sim_eval_status.setWordWrap(True)
        sim_eval_layout.addWidget(self.sim_eval_status)
        self.content_layout.addWidget(sim_eval_card)

        self.mode_combo.currentIndexChanged.connect(self._handle_filter_change)
        self.status_combo.currentIndexChanged.connect(self._handle_filter_change)
        self.query_input.textChanged.connect(self._handle_filter_change)
        self._restore_filters()
        self._update_checkpoint_buttons()
        self.refresh_experiments()

    def _restore_filters(self) -> None:
        mode_index = self.mode_combo.findData(str(self.config.get("ui_experiments_mode_filter", "all")).strip())
        status_index = self.status_combo.findData(str(self.config.get("ui_experiments_status_filter", "all")).strip())
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        if status_index >= 0:
            self.status_combo.setCurrentIndex(status_index)
        self.query_input.setText(str(self.config.get("ui_experiments_query", "")).strip())

    def _persist_filters(self) -> None:
        self.config["ui_experiments_mode_filter"] = str(self.mode_combo.currentData() or "all")
        self.config["ui_experiments_status_filter"] = str(self.status_combo.currentData() or "all")
        self.config["ui_experiments_query"] = self.query_input.text().strip()
        save_config(self.config, quiet=True)

    def _handle_filter_change(self, *_args: object) -> None:
        self._persist_filters()
        self.refresh_experiments()

    def _selected_run_indices(self) -> list[int]:
        return sorted({index.row() for index in self.run_table.selectionModel().selectedRows()})

    def _selected_records(self) -> list[dict[str, Any]]:
        rows = self._selected_run_indices()
        return [self._visible_records[row] for row in rows if 0 <= row < len(self._visible_records)]

    def _current_record(self) -> dict[str, Any] | None:
        row = self.run_table.currentRow()
        if row < 0 or row >= len(self._visible_records):
            return None
        return self._visible_records[row]

    def _current_checkpoint(self) -> dict[str, Any] | None:
        row = self.checkpoint_table.currentRow()
        if row < 0 or row >= len(self._checkpoint_rows):
            return None
        return self._checkpoint_rows[row]

    def refresh_experiments(self) -> None:
        current_run_id = None
        current = self._current_record()
        if current is not None:
            current_run_id = str(current.get("run_id", ""))

        payload = collect_experiment_runs(self.config, limit=0, include_wandb_remote=False)
        self._records = list(payload.get("records", []))
        self._visible_records = []

        mode_filter = str(self.mode_combo.currentData() or "all")
        status_filter = str(self.status_combo.currentData() or "all")
        query = self.query_input.text().strip().lower()

        self.run_table.setRowCount(0)
        for record in self._records:
            if mode_filter != "all" and str(record.get("mode")) != mode_filter:
                continue
            if status_filter != "all" and str(record.get("status")) != status_filter:
                continue
            if query:
                haystack = " ".join(
                    [
                        str(record.get("dataset_or_env", "")),
                        str(record.get("policy", "")),
                        str(record.get("checkpoint", "")),
                        str(record.get("command", "")),
                    ]
                ).lower()
                if query not in haystack:
                    continue

            row = self.run_table.rowCount()
            self.run_table.insertRow(row)
            self._visible_records.append(record)
            values = [
                str(record.get("started_at_iso", "")).replace("T", " ")[:19],
                str(record.get("mode", "")).replace("_", " ").title(),
                str(record.get("status", "")).title(),
                str(record.get("dataset_or_env", "-")),
                str(record.get("policy", "-")),
                str(record.get("checkpoint", "-")),
                str(record.get("metrics_summary", "-")),
            ]
            for column, value in enumerate(values):
                self.run_table.setItem(row, column, QTableWidgetItem(value))

        stats = payload.get("stats", {})
        warning_count = int(payload.get("warning_count") or 0)
        self.experiment_status.setText(
            f"{len(self._visible_records)} visible experiment runs "
            f"(train {stats.get('train', 0)}, deploy {stats.get('deploy', 0)}, sim eval {stats.get('sim_eval', 0)})"
            + (f" · skipped {warning_count} unreadable metadata files" if warning_count else "")
        )

        if current_run_id:
            for row, record in enumerate(self._visible_records):
                if str(record.get("run_id", "")) == current_run_id:
                    self.run_table.selectRow(row)
                    break
        elif self._visible_records:
            self.run_table.selectRow(0)
        else:
            self._clear_selection_state()

    def _clear_selection_state(self) -> None:
        self._checkpoint_rows = []
        self.checkpoint_table.setRowCount(0)
        self.selected_train_label.setText("Select a training run to inspect checkpoints.")
        self.checkpoints_card.hide()
        self.output_card.hide()
        self._update_checkpoint_buttons()

    def _on_run_selection_changed(self) -> None:
        records = self._selected_records()
        self.compare_button.setEnabled(len(records) >= 2)
        current = self._current_record()
        if current is None:
            self._clear_selection_state()
            return

        run_path_raw = str(current.get("record", {}).get("_run_path", "")) if isinstance(current.get("record"), dict) else ""
        if current.get("wandb"):
            wandb_info = current["wandb"]
            if isinstance(wandb_info, dict) and not wandb_info.get("remote_summary"):
                current["wandb"] = {**wandb_info, **fetch_wandb_remote_snapshot(wandb_info)}

        self.output_card.show()
        self._set_output(
            title="Experiment Details",
            text=build_experiment_details_text(current),
            log_message=None,
        )
        self._show_summary_tab()
        if run_path_raw:
            log_path = Path(run_path_raw) / "command.log"
            if log_path.exists():
                try:
                    self._set_raw_output(log_path.read_text(encoding="utf-8"))
                except OSError:
                    self._set_raw_output("Unable to read command log.")
                self._show_summary_tab()

        checkpoints = current.get("checkpoints") if isinstance(current.get("checkpoints"), list) else []
        self._checkpoint_rows = [dict(item) for item in checkpoints]
        self.checkpoint_table.setRowCount(0)
        if current.get("mode") == "train":
            self.checkpoints_card.show()
            self.selected_train_label.setText(
                f"Training run {current.get('run_id')} · {len(self._checkpoint_rows)} discovered checkpoint artifacts"
            )
            for checkpoint in self._checkpoint_rows:
                row = self.checkpoint_table.rowCount()
                self.checkpoint_table.insertRow(row)
                values = [
                    str(checkpoint.get("label", "-")),
                    str(checkpoint.get("kind", "-")),
                    str(checkpoint.get("step", "-")) if checkpoint.get("step") is not None else "-",
                    str(checkpoint.get("policy_family", "") or "-"),
                    str(checkpoint.get("path", "-")),
                ]
                for column, value in enumerate(values):
                    self.checkpoint_table.setItem(row, column, QTableWidgetItem(value))
            if self._checkpoint_rows:
                self.checkpoint_table.selectRow(0)
        else:
            self.checkpoints_card.hide()
        self._update_checkpoint_buttons()

    def compare_selected_runs(self) -> None:
        records = self._selected_records()
        payload = build_experiment_comparison_payload(records)
        self.compare_table.setRowCount(0)
        for row_payload in payload.get("rows", []):
            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)
            for column, value in enumerate(row_payload.get("values", ())):
                self.compare_table.setItem(row, column, QTableWidgetItem(str(value)))
        summary = payload.get("summary", {})
        self.compare_status.setText(
            f"Compared {summary.get('total', 0)} runs · modes {summary.get('modes', {})} · statuses {summary.get('statuses', {})}"
        )

    def _update_checkpoint_buttons(self) -> None:
        checkpoint = self._current_checkpoint()
        deployable = bool(checkpoint and checkpoint.get("is_deployable"))
        has_config = bool(checkpoint and checkpoint.get("train_config_path"))
        self.open_checkpoint_button.setEnabled(checkpoint is not None)
        self.open_train_config_button.setEnabled(has_config)
        self.deploy_checkpoint_button.setEnabled(deployable)
        self.sim_eval_checkpoint_button.setEnabled(deployable)

    def open_selected_run_folder(self) -> None:
        current = self._current_record()
        if current is None:
            return
        run_path = Path(str(current.get("record", {}).get("_run_path", "")))
        ok, message = open_path_in_file_manager(run_path)
        self.experiment_status.setText(message if ok else f"Unable to open run folder: {message}")

    def open_selected_output(self) -> None:
        current = self._current_record()
        if current is None:
            return
        target = str(current.get("output_location", "")).strip()
        if not target:
            self.experiment_status.setText("This run does not have a saved output location.")
            return
        ok, message = _open_path(target)
        self.experiment_status.setText(message if ok else f"Unable to open output: {message}")

    def open_selected_wandb_run(self) -> None:
        current = self._current_record()
        if current is None:
            return
        wandb_info = current.get("wandb") if isinstance(current.get("wandb"), dict) else {}
        target = str(wandb_info.get("run_url", "")).strip()
        if not target:
            self.experiment_status.setText("No WandB run URL was discovered for the selected run.")
            return
        ok, message = _open_path(target)
        self.experiment_status.setText(message if ok else f"Unable to open WandB run: {message}")

    def open_selected_checkpoint_folder(self) -> None:
        checkpoint = self._current_checkpoint()
        if checkpoint is None:
            return
        path = Path(str(checkpoint.get("path", "")))
        target = path if path.is_dir() else path.parent
        ok, message = open_path_in_file_manager(target)
        self.deploy_launch_status.setText(message if ok else f"Unable to open checkpoint: {message}")

    def open_selected_checkpoint_config(self) -> None:
        checkpoint = self._current_checkpoint()
        if checkpoint is None:
            return
        config_path = str(checkpoint.get("train_config_path", "")).strip()
        if not config_path:
            return
        ok, message = _open_path(config_path)
        self.deploy_launch_status.setText(message if ok else f"Unable to open config: {message}")

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        for button in self._action_buttons:
            button.setEnabled(not active)
        if active:
            self.output_card.show()
            self.status_label.setText(status_text or "Running experiment workflow...")
            self._show_raw_tab()
            return
        self.status_label.setText(status_text or ("Run failed." if is_error else "Run completed."))

    def _build_hooks(self) -> RunUiHooks:
        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_runtime_output_line,
            append_output_chunk=self._append_output_chunk,
            on_artifact_written=self._remember_artifact,
        )

    def _append_runtime_output_line(self, line: str) -> None:
        self.output_card.show()
        self._append_output_line(line)

    def _remember_artifact(self, artifact_path: Path) -> None:
        self._latest_run_artifact_path = Path(artifact_path)

    def _persist_sim_eval_defaults(self) -> None:
        self.config["ui_sim_eval_env_type"] = self.sim_env_type_input.text().strip()
        self.config["ui_sim_eval_task"] = self.sim_task_input.text().strip()
        self.config["ui_sim_eval_benchmark"] = self.sim_benchmark_input.text().strip()
        self.config["ui_sim_eval_episodes"] = self.sim_episodes_input.text().strip()
        self.config["ui_sim_eval_batch_size"] = self.sim_batch_size_input.text().strip()
        self.config["ui_sim_eval_seed"] = self.sim_seed_input.text().strip()
        self.config["ui_sim_eval_device"] = self.sim_device_input.text().strip()
        self.config["ui_sim_eval_output_dir"] = self.sim_output_dir_input.text().strip()
        self.config["ui_sim_eval_job_name"] = self.sim_job_name_input.text().strip()
        save_config(self.config, quiet=True)

    def launch_deploy_from_checkpoint(self) -> None:
        checkpoint = self._current_checkpoint()
        if checkpoint is None or not checkpoint.get("is_deployable"):
            self.deploy_launch_status.setText("Select a deployable checkpoint first.")
            return

        req, cmd, updated_config, error = build_deploy_request_and_command(
            config=self.config,
            deploy_root_raw=str(self.config.get("trained_models_dir", "")),
            deploy_model_raw=str(checkpoint.get("path", "")),
            eval_dataset_raw=self.deploy_dataset_input.text(),
            eval_episodes_raw=self.deploy_episodes_input.text(),
            eval_duration_raw=self.deploy_duration_input.text(),
            eval_task_raw=self.deploy_task_input.text(),
        )
        if error or req is None or cmd is None or updated_config is None:
            self.deploy_launch_status.setText(error or "Unable to build deploy command.")
            return

        checks = run_preflight_for_deploy(self.config, model_path=req.model_path, eval_repo_id=req.eval_repo_id, command=cmd)
        if any(str(level).upper() == "FAIL" for level, _name, _detail in checks):
            self.output_card.show()
            self._set_output(
                title="Deploy Preflight Failed",
                text=summarize_checks(checks, title="Checkpoint Deploy Preflight"),
                log_message="Checkpoint deploy blocked by preflight failures.",
            )
            self.deploy_launch_status.setText("Deploy preflight failed. Fix the listed blockers before retrying.")
            return

        self.output_card.show()
        self.output.clear()
        self._set_output(
            title="Launching deploy eval...",
            text=summarize_checks(checks, title="Checkpoint Deploy Preflight") + "\n\n" + "Command:\n" + " ".join(cmd),
            log_message=f"Launching deploy eval from checkpoint {checkpoint.get('label')}.",
        )
        self._show_raw_tab()

        def after_deploy(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self.deploy_launch_status.setText("Deploy run canceled.")
            elif return_code == 0:
                self.deploy_launch_status.setText("Deploy run completed.")
            else:
                self.deploy_launch_status.setText(f"Deploy run failed with exit code {return_code}.")
            self.refresh_experiments()

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=Path(str(self.config.get("lerobot_dir", "")).strip()).expanduser(),
            hooks=self._build_hooks(),
            complete_callback=after_deploy,
            run_mode="deploy",
            preflight_checks=checks,
            artifact_context={
                "dataset_repo_id": req.eval_repo_id,
                "model_path": str(req.model_path),
                "policy_type": str((self._current_record() or {}).get("policy", "")),
            },
        )
        if not ok and message:
            self.deploy_launch_status.setText(message)

    def launch_sim_eval_from_checkpoint(self) -> None:
        checkpoint = self._current_checkpoint()
        if checkpoint is None or not checkpoint.get("is_deployable"):
            self.sim_eval_status.setText("Select a deployable checkpoint first.")
            return

        form_values = {
            "model_path": str(checkpoint.get("path", "")),
            "output_dir": self.sim_output_dir_input.text(),
            "env_type": self.sim_env_type_input.text(),
            "task": self.sim_task_input.text(),
            "benchmark": self.sim_benchmark_input.text(),
            "episodes": self.sim_episodes_input.text(),
            "batch_size": self.sim_batch_size_input.text(),
            "seed": self.sim_seed_input.text(),
            "device": self.sim_device_input.text(),
            "job_name": self.sim_job_name_input.text(),
            "trust_remote_code": self.sim_trust_remote_code_checkbox.isChecked(),
            "custom_args": self.sim_custom_args_input.text(),
        }
        request, cmd, error = build_sim_eval_request_and_command(form_values=form_values, config=self.config)
        if error or request is None or cmd is None:
            self.sim_eval_status.setText(error or "Unable to build sim eval command.")
            return

        self._persist_sim_eval_defaults()
        self.output_card.show()
        self.output.clear()
        self._set_output(
            title="Launching sim eval...",
            text="Command:\n" + " ".join(cmd),
            log_message=f"Launching sim eval from checkpoint {checkpoint.get('label')}.",
        )
        self._show_raw_tab()

        def after_sim_eval(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self.sim_eval_status.setText("Sim eval canceled.")
            elif return_code == 0:
                self.sim_eval_status.setText("Sim eval completed.")
            else:
                self.sim_eval_status.setText(f"Sim eval failed with exit code {return_code}.")
            self.refresh_experiments()

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=Path(str(self.config.get("lerobot_dir", "")).strip()).expanduser(),
            hooks=self._build_hooks(),
            complete_callback=after_sim_eval,
            run_mode="sim_eval",
            preflight_checks=[],
            artifact_context={
                "model_path": str(request.model_path),
                "output_dir": request.output_dir,
                "env_type": request.env_type,
                "task": request.task,
                "benchmark": request.benchmark,
                "episodes": request.episodes,
                "batch_size": request.batch_size,
                "seed": request.seed,
                "device": request.device,
                "job_name": request.job_name,
                "policy_type": str((self._current_record() or {}).get("policy", "")),
            },
        )
        if not ok and message:
            self.sim_eval_status.setText(message)
