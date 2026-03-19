from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .checks import has_failures
from .config_store import get_lerobot_dir, save_config
from .deploy_workflow_helpers import DatasetBrowserNode, build_dataset_browser_tree
from .gui_async import QtAfterAdapter, UiBackgroundJobs
from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card
from .hardware_workflows import (
    build_replay_preflight_checks,
    build_replay_readiness_summary,
    build_replay_request_and_command,
    discover_replay_episodes,
)
from .repo_utils import list_hf_datasets, normalize_repo_id
from .run_controller_service import ManagedRunController
from .workspace_provenance import read_workspace_provenance


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
        self._qt_after_adapter = QtAfterAdapter()
        self._hf_dataset_jobs = UiBackgroundJobs(self._qt_after_adapter, max_workers=2)
        self._hf_dataset_rows: list[dict[str, Any]] = []
        self._dataset_selection_source = ""

        root_layout = self.layout()
        self.dataset_browser_card, dataset_browser_layout = _build_card("Datasets")
        self.dataset_browser_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._build_dataset_browser_ui(dataset_browser_layout)
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertWidget(0, self.dataset_browser_card)

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

        self.refresh_local_dataset_browser()
        self.refresh_hf_datasets()
        self.refresh_from_config()

    def _build_dataset_browser_ui(self, layout: QVBoxLayout) -> None:
        browser_layout = QHBoxLayout()
        browser_layout.setSpacing(12)

        local_panel = QWidget()
        local_layout = QVBoxLayout(local_panel)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(8)

        local_header = QLabel("Local datasets")
        local_header.setObjectName("SectionMeta")
        local_layout.addWidget(local_header)

        local_controls = QHBoxLayout()
        self.refresh_local_datasets_button = QPushButton("Refresh Local")
        self.refresh_local_datasets_button.clicked.connect(self.refresh_local_dataset_browser)
        local_controls.addWidget(self.refresh_local_datasets_button)
        local_controls.addStretch(1)
        local_layout.addLayout(local_controls)

        self.local_dataset_tree = QTreeWidget()
        self.local_dataset_tree.setColumnCount(2)
        self.local_dataset_tree.setHeaderLabels(["Dataset / Folder", "Type"])
        self.local_dataset_tree.setRootIsDecorated(True)
        self.local_dataset_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.local_dataset_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.local_dataset_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.local_dataset_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.local_dataset_tree.itemClicked.connect(lambda _item, _column: self._mark_dataset_selection_source("local"))
        self.local_dataset_tree.itemDoubleClicked.connect(lambda _item, _column: self._use_local_dataset_selection())
        local_layout.addWidget(self.local_dataset_tree)

        self.local_dataset_status_label = QLabel("Local dataset browser is ready.")
        self.local_dataset_status_label.setObjectName("MutedLabel")
        self.local_dataset_status_label.setWordWrap(True)
        local_layout.addWidget(self.local_dataset_status_label)

        browser_layout.addWidget(local_panel, 1)

        hf_panel = QWidget()
        hf_layout = QVBoxLayout(hf_panel)
        hf_layout.setContentsMargins(0, 0, 0, 0)
        hf_layout.setSpacing(8)

        hf_header = QLabel("Hugging Face datasets")
        hf_header.setObjectName("SectionMeta")
        hf_layout.addWidget(hf_header)

        hf_owner_row = QWidget()
        hf_owner_layout = QHBoxLayout(hf_owner_row)
        hf_owner_layout.setContentsMargins(0, 0, 0, 0)
        hf_owner_layout.setSpacing(8)
        owner_label = QLabel("Owner")
        owner_label.setObjectName("FormLabel")
        hf_owner_layout.addWidget(owner_label)
        self.local_hf_owner_input = QLineEdit(self._default_replay_hf_owner())
        self.local_hf_owner_input.editingFinished.connect(self._persist_replay_hf_owner)
        self.local_hf_owner_input.editingFinished.connect(self.refresh_hf_datasets)
        hf_owner_layout.addWidget(self.local_hf_owner_input, 1)
        self.refresh_hf_datasets_button = QPushButton("Refresh HF")
        self.refresh_hf_datasets_button.clicked.connect(self.refresh_hf_datasets)
        hf_owner_layout.addWidget(self.refresh_hf_datasets_button)
        hf_layout.addWidget(hf_owner_row)

        self.hf_dataset_table = QTableWidget(0, 3)
        self.hf_dataset_table.setHorizontalHeaderLabels(["Repo", "Downloads", "Likes"])
        self.hf_dataset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.hf_dataset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.hf_dataset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hf_dataset_table.verticalHeader().setVisible(False)
        self.hf_dataset_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.hf_dataset_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.hf_dataset_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.hf_dataset_table.cellClicked.connect(lambda _row, _column: self._mark_dataset_selection_source("hf"))
        self.hf_dataset_table.cellDoubleClicked.connect(lambda _row, _column: self._use_hf_dataset_selection())
        hf_layout.addWidget(self.hf_dataset_table)

        self.hf_dataset_status_label = QLabel("Enter or confirm an owner, then refresh.")
        self.hf_dataset_status_label.setObjectName("MutedLabel")
        self.hf_dataset_status_label.setWordWrap(True)
        hf_layout.addWidget(self.hf_dataset_status_label)

        browser_layout.addWidget(hf_panel, 1)
        layout.addLayout(browser_layout)

        action_row = QHBoxLayout()
        self.use_selected_dataset_button = QPushButton("Use Selected in Replay")
        self.use_selected_dataset_button.clicked.connect(self.use_selected_dataset_in_replay)
        action_row.addWidget(self.use_selected_dataset_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

    def _default_replay_hf_owner(self) -> str:
        return str(self.config.get("replay_hf_dataset_owner", "")).strip() or str(self.config.get("hf_username", "")).strip()

    def _persist_replay_hf_owner(self) -> None:
        owner = self.local_hf_owner_input.text().strip()
        self.config["replay_hf_dataset_owner"] = owner
        save_config(self.config, quiet=True)

    def _current_replay_root(self) -> Path:
        return Path(str(self.config.get("record_data_dir", "")).strip() or "data").expanduser()

    def _add_dataset_browser_node(self, parent: QTreeWidgetItem | QTreeWidget, node: DatasetBrowserNode) -> None:
        item = QTreeWidgetItem([node.label, node.kind])
        item.setData(0, Qt.ItemDataRole.UserRole, str(node.path))
        item.setData(0, Qt.ItemDataRole.UserRole + 1, node.tag)
        item.setData(0, Qt.ItemDataRole.UserRole + 2, node.repo_id)
        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in node.children:
            self._add_dataset_browser_node(item, child)

    def refresh_local_dataset_browser(self) -> None:
        self.local_dataset_tree.clear()
        root_path = self._current_replay_root()
        nodes = build_dataset_browser_tree(root_path)
        for node in nodes:
            self._add_dataset_browser_node(self.local_dataset_tree, node)
        if not root_path.exists() or not root_path.is_dir():
            self.local_dataset_status_label.setText(f"Local dataset root not found: {root_path}")
            return
        if not nodes:
            self.local_dataset_status_label.setText(f"No local datasets detected in {root_path}")
            return
        self.local_dataset_status_label.setText(f"Local datasets in {root_path}")

    def _apply_hf_dataset_rows(self, result: tuple[list[dict[str, Any]], str | None]) -> None:
        rows, error_text = result
        self._hf_dataset_rows = list(rows)
        self.hf_dataset_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.hf_dataset_table.setItem(row_index, 0, QTableWidgetItem(str(row.get("repo_id", ""))))
            self.hf_dataset_table.setItem(row_index, 1, QTableWidgetItem(str(row.get("downloads", "-"))))
            self.hf_dataset_table.setItem(row_index, 2, QTableWidgetItem(str(row.get("likes", "-"))))
        owner = self.local_hf_owner_input.text().strip()
        if error_text:
            self.hf_dataset_status_label.setText(error_text)
            return
        if rows:
            self.hf_dataset_status_label.setText(f"Hugging Face datasets for {owner}")
            self.hf_dataset_table.selectRow(0)
            return
        self.hf_dataset_status_label.setText(f"No Hugging Face datasets found for {owner}")

    def refresh_hf_datasets(self) -> None:
        owner = self.local_hf_owner_input.text().strip()
        self.config["replay_hf_dataset_owner"] = owner
        self._hf_dataset_rows = []
        self.hf_dataset_table.setRowCount(0)
        if not owner:
            self.hf_dataset_status_label.setText("Set an HF owner to load remote datasets.")
            return
        self.hf_dataset_status_label.setText(f"Loading Hugging Face datasets for {owner}...")
        self.refresh_hf_datasets_button.setEnabled(False)
        self._hf_dataset_jobs.submit(
            "replay-hf-datasets",
            lambda: list_hf_datasets(owner, limit=200),
            on_success=self._apply_hf_dataset_rows,
            on_error=lambda exc: self.hf_dataset_status_label.setText(f"Unable to load HF datasets: {exc}"),
            on_complete=lambda _is_stale: self.refresh_hf_datasets_button.setEnabled(True),
        )

    def _selected_local_dataset_path(self) -> Path | None:
        items = self.local_dataset_tree.selectedItems()
        if not items:
            return None
        item = items[0]
        tag = str(item.data(0, Qt.ItemDataRole.UserRole + 1) or "").strip()
        if tag != "dataset_root":
            return None
        path_text = str(item.data(0, Qt.ItemDataRole.UserRole) or "").strip()
        return Path(path_text) if path_text else None

    def _selected_hf_repo_id(self) -> str:
        row = self.hf_dataset_table.currentRow()
        if row < 0 or row >= len(self._hf_dataset_rows):
            return ""
        return str(self._hf_dataset_rows[row].get("repo_id", "")).strip()

    def _mark_dataset_selection_source(self, source: str) -> None:
        self._dataset_selection_source = source if source in {"local", "hf"} else ""

    def _use_local_dataset_selection(self) -> None:
        self._mark_dataset_selection_source("local")
        self.use_selected_dataset_in_replay()

    def _use_hf_dataset_selection(self) -> None:
        self._mark_dataset_selection_source("hf")
        self.use_selected_dataset_in_replay()

    def _replay_value_for_local_dataset(self, dataset_path: Path) -> str:
        provenance = read_workspace_provenance(dataset_path) or {}
        provenance_repo_id = str(provenance.get("repo_id", "")).strip()
        if provenance_repo_id:
            return provenance_repo_id
        return normalize_repo_id(str(self.config.get("hf_username", "")).strip(), dataset_path.name)

    def _apply_dataset_selection(self, *, repo_id: str, dataset_path: Path | None) -> None:
        self.dataset_input.blockSignals(True)
        self.dataset_path_input.blockSignals(True)
        self.episode_manual_input.blockSignals(True)
        self.dataset_input.setText(repo_id)
        self.dataset_path_input.setText(str(dataset_path) if dataset_path is not None else "")
        self.episode_manual_input.clear()
        self.dataset_input.blockSignals(False)
        self.dataset_path_input.blockSignals(False)
        self.episode_manual_input.blockSignals(False)
        self._refresh_episode_state()

    def use_selected_dataset_in_replay(self) -> None:
        local_dataset = self._selected_local_dataset_path()
        repo_id = self._selected_hf_repo_id()
        if self._dataset_selection_source == "hf" and repo_id:
            self._apply_dataset_selection(repo_id=repo_id, dataset_path=None)
            self._append_output_and_log(f"Using Hugging Face dataset in Replay: {repo_id}")
            return
        if local_dataset is not None:
            value = self._replay_value_for_local_dataset(local_dataset)
            self._apply_dataset_selection(repo_id=value, dataset_path=local_dataset)
            self._append_output_and_log(f"Using local dataset in Replay: {value}")
            return
        if repo_id:
            self._apply_dataset_selection(repo_id=repo_id, dataset_path=None)
            self._append_output_and_log(f"Using Hugging Face dataset in Replay: {repo_id}")
            return
        self._set_output(
            title="No Dataset Selected",
            text="Select a local or Hugging Face dataset first.",
            log_message="Replay dataset selection apply skipped with no selection.",
        )

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
            self.episode_combo.blockSignals(True)
            self.episode_combo.clear()
            self.episode_combo.addItem("0")
            self.episode_combo.blockSignals(False)
            self.episode_manual_input.setEnabled(True)
            self.support_label.setText("")
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

    def closeEvent(self, event: object) -> None:
        self._hf_dataset_jobs.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]

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
