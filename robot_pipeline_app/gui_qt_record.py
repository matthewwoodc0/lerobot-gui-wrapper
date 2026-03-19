from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .auto_names import record_dataset_seed, resolve_record_dataset_name
from .checks import run_preflight_for_record, summarize_checks
from .command_text import format_command_for_dialog
from .command_overrides import get_flag_value
from .config_store import get_lerobot_dir, save_config
from .constants import DEFAULT_TASK
from .deploy_workflow_helpers import (
    build_dataset_browser_tree,
    build_dataset_upload_request,
    DatasetBrowserNode,
)
from .gui_async import QtAfterAdapter, UiBackgroundJobs
from .gui_forms import (
    build_record_request_and_command,
)
from .gui_qt_auto_name import AutoNameController
from .gui_qt_camera import QtCameraWorkspace
from .gui_qt_dialogs import (
    _build_dialog_panel,
    _fit_dialog_to_screen,
    ask_text_dialog,
    show_text_dialog,
)
from .gui_qt_runtime_helpers import QtRunHelperDialog
from .hf_auth import has_huggingface_auth_token
from .repo_utils import (
    build_dataset_tag_upload_command,
    default_dataset_tags,
    list_hf_datasets,
    normalize_repo_id,
    repo_name_from_repo_id,
    repo_name_only,
    safe_unlink,
    write_dataset_card_temp,
)
from .run_controller_service import ManagedRunController
from .workspace_provenance import build_hf_provenance_payload, read_workspace_provenance, write_workspace_provenance
from .workflows import move_recorded_dataset

from .gui_qt_ops_base import _AdvancedOptionsPanel, _CoreOpsPanel, _InputGrid, _build_card


class _QtDatasetUploadDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        default_local_dataset: str,
        default_owner: str,
        default_repo_name: str,
        dataset_options: list[str],
    ) -> None:
        super().__init__(parent)
        self.result_request: dict[str, Any] | None = None
        self.result_settings: dict[str, Any] | None = None
        self._initial_dataset_options = list(dataset_options)
        self.setWindowTitle("Upload Dataset to Hugging Face")
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=900,
            requested_height=420,
            requested_min_width=760,
            requested_min_height=340,
        )
        self._default_local_dataset = default_local_dataset
        self._default_owner = default_owner
        self._default_repo_name = default_repo_name
        self._build_ui()
        self.set_dataset_options(list(self._initial_dataset_options))

    def _build_ui(self) -> None:
        layout = _build_dialog_panel(
            self,
            title="Upload Dataset to Hugging Face",
            subtitle="Upload a local dataset folder into a Hugging Face dataset repository.",
        )

        intro = QLabel(
            "Use this when you want to push an existing local dataset without starting a new record run."
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

        local_label = QLabel("Local dataset folder")
        local_label.setObjectName("FormLabel")
        grid.addWidget(local_label, 0, 0)
        self.local_dataset_input = QLineEdit(self._default_local_dataset)
        grid.addWidget(self.local_dataset_input, 0, 1, 1, 2)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._choose_local_dataset)
        grid.addWidget(browse_button, 0, 3)

        options_label = QLabel("Local dataset candidates")
        options_label.setObjectName("FormLabel")
        grid.addWidget(options_label, 1, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEditable(False)
        grid.addWidget(self.dataset_combo, 1, 1, 1, 2)
        refresh_button = QPushButton("Refresh Datasets")
        refresh_button.clicked.connect(lambda: self.set_dataset_options(list(self._initial_dataset_options)))
        grid.addWidget(refresh_button, 1, 3)

        owner_label = QLabel("HF owner")
        owner_label.setObjectName("FormLabel")
        grid.addWidget(owner_label, 2, 0)
        self.owner_input = QLineEdit(self._default_owner)
        grid.addWidget(self.owner_input, 2, 1)

        repo_label = QLabel("HF dataset name")
        repo_label.setObjectName("FormLabel")
        grid.addWidget(repo_label, 2, 2)
        self.repo_name_input = QLineEdit(self._default_repo_name)
        grid.addWidget(self.repo_name_input, 2, 3)

        self.status_label = QLabel("Choose a local dataset folder, then preview or run the upload.")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        parity_button = QPushButton("Check Parity")
        parity_button.clicked.connect(self.check_parity)
        button_row.addWidget(parity_button)

        preview_button = QPushButton("Preview Upload Command")
        preview_button.clicked.connect(self.preview_upload_command)
        button_row.addWidget(preview_button)

        button_row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)

        run_button = QPushButton("Upload Dataset")
        run_button.setObjectName("AccentButton")
        run_button.clicked.connect(self.accept_for_run)
        button_row.addWidget(run_button)
        layout.addLayout(button_row)

        self.dataset_combo.currentTextChanged.connect(self._sync_combo_selection)

    def set_dataset_options(self, dataset_options: list[str]) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(dataset_options)
        current = self.local_dataset_input.text().strip()
        if current:
            index = self.dataset_combo.findText(current)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        elif dataset_options:
            self.dataset_combo.setCurrentIndex(0)
            self.local_dataset_input.setText(dataset_options[0])
        self.dataset_combo.blockSignals(False)

    def _choose_local_dataset(self) -> None:
        current = self.local_dataset_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select local dataset folder", current)
        if selected:
            self.local_dataset_input.setText(selected)
            if not self.repo_name_input.text().strip():
                self.repo_name_input.setText(Path(selected).name)

    def _sync_combo_selection(self, value: str) -> None:
        selected = str(value or "").strip()
        if not selected:
            return
        self.local_dataset_input.setText(selected)
        if not self.repo_name_input.text().strip():
            self.repo_name_input.setText(Path(selected).name)

    def _build_request(self) -> tuple[dict[str, Any] | None, str | None]:
        request, error_text = build_dataset_upload_request(
            local_dataset_raw=self.local_dataset_input.text(),
            owner_raw=self.owner_input.text(),
            repo_name_raw=self.repo_name_input.text(),
        )
        if request is not None:
            self.repo_name_input.setText(str(request.get("repo_name", self.repo_name_input.text())))
        return request, error_text

    def check_parity(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build dataset upload request.")
            return
        self.status_label.setText(
            str(request.get("parity_detail", "Parity check complete."))
            + "\n"
            + str(request.get("provenance_detail", "")).strip()
        )

    def preview_upload_command(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build dataset upload request.")
            return
        show_text_dialog(
            parent=self,
            title="HF Dataset Upload Command",
            text="Upload command:\n" + format_command_for_dialog(request["upload_cmd"]),
            copy_text=" ".join(str(part) for part in request["upload_cmd"]),
            wrap_mode="word",
        )

    def accept_for_run(self) -> None:
        request, error_text = self._build_request()
        if error_text or request is None:
            self.status_label.setText(error_text or "Unable to build dataset upload request.")
            return
        self.result_request = request
        self.result_settings = {
            "local_dataset": self.local_dataset_input.text().strip(),
            "owner": self.owner_input.text().strip(),
            "repo_name": self.repo_name_input.text().strip(),
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
        self._qt_after_adapter = QtAfterAdapter()
        self._hf_dataset_jobs = UiBackgroundJobs(self._qt_after_adapter, max_workers=2)
        self._hf_dataset_rows: list[dict[str, Any]] = []
        self.run_helper_dialog = QtRunHelperDialog(
            parent=self.window() if isinstance(self.window(), QWidget) else None,
            mode_title="Record",
            on_cancel=self._cancel_run,
            on_send_key=self._run_controller.send_arrow_key,
        )
        self.camera_preview = QtCameraWorkspace(config=self.config, append_log=self._append_log)
        root_layout = self.layout()
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertWidget(2, self.camera_preview)
        self.dataset_browser_card, dataset_browser_layout = _build_card("Datasets")
        self.dataset_browser_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._build_dataset_browser_ui(dataset_browser_layout)
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertWidget(3, self.dataset_browser_card)

        form = _InputGrid(self.form_layout)

        default_dataset = record_dataset_seed(config)
        self.dataset_input = QLineEdit(default_dataset)
        self.dataset_input.setPlaceholderText("owner/dataset_name or dataset_name")
        form.add_field("Dataset", self.dataset_input)
        self._dataset_name_controller = AutoNameController(self.dataset_input)

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
        self.dataset_root_input.editingFinished.connect(self._advance_dataset_name)
        self.dataset_root_input.editingFinished.connect(self.refresh_local_dataset_browser)
        self._advance_dataset_name()
        self.local_hf_owner_input.editingFinished.connect(self._persist_record_hf_owner)
        self.refresh_local_dataset_browser()
        self.refresh_hf_datasets()

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
        self.local_hf_owner_input = QLineEdit(self._default_record_hf_owner())
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
        hf_layout.addWidget(self.hf_dataset_table)

        self.hf_dataset_status_label = QLabel("Enter or confirm an owner, then refresh.")
        self.hf_dataset_status_label.setObjectName("MutedLabel")
        self.hf_dataset_status_label.setWordWrap(True)
        hf_layout.addWidget(self.hf_dataset_status_label)

        browser_layout.addWidget(hf_panel, 1)
        layout.addLayout(browser_layout)

        action_row = QHBoxLayout()
        self.use_selected_dataset_button = QPushButton("Use Selected in Record")
        self.use_selected_dataset_button.clicked.connect(self.use_selected_dataset_in_record)
        action_row.addWidget(self.use_selected_dataset_button)

        self.upload_local_dataset_button = QPushButton("Upload Local Dataset")
        self.upload_local_dataset_button.clicked.connect(self.open_dataset_upload_dialog)
        action_row.addWidget(self.upload_local_dataset_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

    def _default_record_hf_owner(self) -> str:
        return str(self.config.get("record_hf_dataset_owner", "")).strip() or str(self.config.get("hf_username", "")).strip()

    def _persist_record_hf_owner(self) -> None:
        owner = self.local_hf_owner_input.text().strip()
        self.config["record_hf_dataset_owner"] = owner
        save_config(self.config, quiet=True)

    def _current_record_root(self) -> Path:
        return Path(self.dataset_root_input.text().strip() or str(self.config.get("record_data_dir", ""))).expanduser()

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
        root_path = self._current_record_root()
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
        self.config["record_hf_dataset_owner"] = owner
        self._hf_dataset_rows = []
        self.hf_dataset_table.setRowCount(0)
        if not owner:
            self.hf_dataset_status_label.setText("Set an HF owner to load remote datasets.")
            return
        self.hf_dataset_status_label.setText(f"Loading Hugging Face datasets for {owner}...")
        self.refresh_hf_datasets_button.setEnabled(False)
        self._hf_dataset_jobs.submit(
            "record-hf-datasets",
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

    def _record_value_for_local_dataset(self, dataset_path: Path) -> str:
        provenance = read_workspace_provenance(dataset_path) or {}
        provenance_repo_id = str(provenance.get("repo_id", "")).strip()
        if provenance_repo_id:
            return provenance_repo_id
        return normalize_repo_id(str(self.config.get("hf_username", "")).strip(), dataset_path.name)

    def use_selected_dataset_in_record(self) -> None:
        local_dataset = self._selected_local_dataset_path()
        if local_dataset is not None:
            value = self._record_value_for_local_dataset(local_dataset)
            self._dataset_name_controller.set_text(value, mode="manual")
            self._append_output_and_log(f"Using local dataset in Record: {value}")
            return
        repo_id = self._selected_hf_repo_id()
        if repo_id:
            self._dataset_name_controller.set_text(repo_id, mode="manual")
            self._append_output_and_log(f"Using Hugging Face dataset in Record: {repo_id}")
            return
        self._set_output(
            title="No Dataset Selected",
            text="Select a local or Hugging Face dataset first.",
            log_message="Record dataset selection apply skipped with no selection.",
        )

    def _collect_local_dataset_candidate_paths(self) -> list[str]:
        candidates: list[str] = []

        def visit(item: QTreeWidgetItem) -> None:
            tag = str(item.data(0, Qt.ItemDataRole.UserRole + 1) or "").strip()
            path_text = str(item.data(0, Qt.ItemDataRole.UserRole) or "").strip()
            if tag == "dataset_root" and path_text:
                candidates.append(path_text)
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.local_dataset_tree.topLevelItemCount()):
            visit(self.local_dataset_tree.topLevelItem(index))
        return candidates

    def _start_dataset_tag_upload(
        self,
        *,
        repo_id: str,
        task: str | None,
        on_complete: Callable[[int, bool, list[str]], None],
    ) -> tuple[bool, str | None]:
        tags = default_dataset_tags(config=self.config, dataset_repo_id=repo_id, task=task)
        card_path = write_dataset_card_temp(
            dataset_repo_id=repo_id,
            dataset_name=repo_name_from_repo_id(repo_id),
            tags=tags,
            task=task,
        )
        tag_cmd = build_dataset_tag_upload_command(dataset_repo_id=repo_id, card_path=card_path)
        self._append_output_and_log(f"Updating dataset tags on Hugging Face: {', '.join(tags)}")

        def after_tag(tag_code: int, tag_canceled: bool) -> None:
            safe_unlink(card_path)
            on_complete(tag_code, tag_canceled, tags)

        ok, message = self._run_controller.run_process_async(
            cmd=tag_cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_tag,
            run_mode="upload",
            artifact_context={"dataset_repo_id": repo_id},
        )
        if not ok:
            safe_unlink(card_path)
        return ok, message

    def open_dataset_upload_dialog(self) -> None:
        if not has_huggingface_auth_token():
            self._set_output(
                title="Hugging Face Login Required",
                text="No Hugging Face login was detected. Run `hf auth login` in Terminal, then try again.",
                log_message="Record dataset upload blocked because Hugging Face auth is missing.",
            )
            return

        selected_local_dataset = self._selected_local_dataset_path()
        default_local_dataset = str(selected_local_dataset) if selected_local_dataset is not None else ""
        provenance = read_workspace_provenance(selected_local_dataset) if selected_local_dataset is not None else None
        provenance_repo_id = str((provenance or {}).get("repo_id", "")).strip()
        default_owner = str(self.config.get("record_hf_upload_owner", "")).strip() or self.local_hf_owner_input.text().strip() or str(self.config.get("hf_username", "")).strip()
        if provenance_repo_id and "/" in provenance_repo_id:
            default_owner = provenance_repo_id.split("/", 1)[0]
        default_repo_name = repo_name_only(str(self.config.get("record_hf_upload_repo_name", "")).strip(), owner=default_owner)
        if provenance_repo_id:
            default_repo_name = repo_name_only(provenance_repo_id, owner=default_owner)
        elif not default_repo_name and selected_local_dataset is not None:
            default_repo_name = selected_local_dataset.name

        dialog = _QtDatasetUploadDialog(
            parent=self._dialog_parent(),
            default_local_dataset=default_local_dataset,
            default_owner=default_owner,
            default_repo_name=default_repo_name,
            dataset_options=self._collect_local_dataset_candidate_paths(),
        )
        dialog.exec()
        request = dialog.result_request
        settings = dialog.result_settings
        if request is None or settings is None:
            return

        repo_id = str(request["repo_id"])
        local_dataset = Path(request["local_dataset"])
        remote_exists = request["remote_exists"]
        provenance_repo_id = str(request.get("provenance_repo_id", "")).strip()
        if bool(request.get("provenance_matches_target", False)) and not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Dataset Already Linked",
            text=(
                f"{local_dataset} already carries Hugging Face provenance for {repo_id}.\n"
                "Continue upload anyway?"
            ),
            confirm_label="Continue Upload",
            cancel_label="Cancel",
            wrap_mode="word",
        ):
            return
        if remote_exists is True and not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Remote Dataset Exists",
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
        if provenance_repo_id and provenance_repo_id != repo_id and not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Dataset Provenance Differs",
            text=(
                f"{local_dataset} currently carries Hugging Face provenance for {provenance_repo_id}.\n"
                f"You are uploading to {repo_id}.\nContinue upload anyway?"
            ),
            confirm_label="Continue Upload",
            cancel_label="Cancel",
            wrap_mode="word",
        ):
            return
        if not self._confirm_preflight_review(title="HF Dataset Upload Preflight", checks=request["checks"]):
            return
        if not ask_text_dialog(
            parent=self._dialog_parent(),
            title="Confirm HF Dataset Upload",
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

        self.config["record_hf_upload_local_dataset"] = str(settings.get("local_dataset", "")).strip()
        upload_owner = str(settings.get("owner", "")).strip().strip("/")
        self.config["record_hf_upload_owner"] = upload_owner
        self.config["record_hf_upload_repo_name"] = repo_name_only(
            str(settings.get("repo_name", "")).strip(),
            owner=str(settings.get("owner", "")),
        )
        if upload_owner:
            self.config["record_hf_dataset_owner"] = upload_owner
            self.local_hf_owner_input.setText(upload_owner)
        save_config(self.config, quiet=True)

        self._show_launch_summary(
            heading="Launching dataset upload...",
            command_label="HF dataset upload command",
            cmd=request["upload_cmd"],
            preflight_title="HF Dataset Upload Preflight",
            preflight_checks=request["checks"],
        )
        self._append_log(f"Starting dataset upload: {repo_id}")

        def after_upload(upload_code: int, upload_canceled: bool) -> None:
            if upload_canceled:
                self._set_running(False, "Dataset upload canceled.", False)
                self._append_output_and_log("Hugging Face dataset upload canceled.")
                return
            if upload_code != 0:
                self._set_running(False, "Dataset upload failed.", True)
                self._append_output_and_log(f"Hugging Face dataset upload failed with exit code {upload_code}.")
                return
            self._append_output_and_log(f"Dataset upload completed: {repo_id}")

            def after_tag(tag_code: int, tag_canceled: bool, tags: list[str]) -> None:
                provenance_path = write_workspace_provenance(
                    local_dataset,
                    payload=build_hf_provenance_payload(
                        repo_id=repo_id,
                        asset_kind="dataset",
                        local_path=local_dataset,
                        metadata={
                            "uploaded_via": "gui_manual_upload",
                            "hf_tags": tags,
                        },
                    ),
                    prefer_meta_dir=True,
                )
                if provenance_path is not None:
                    self._append_log(f"Updated local dataset provenance: {provenance_path}")
                else:
                    self._append_output_and_log(
                        "Warning: upload succeeded but local dataset provenance could not be updated."
                    )
                if tag_canceled:
                    self._set_running(False, "Dataset upload completed; tagging canceled.", False)
                    self._append_output_and_log("Dataset tagging canceled after upload.")
                elif tag_code != 0:
                    self._set_running(False, "Dataset upload completed; tagging failed.", True)
                    self._append_output_and_log(
                        f"Dataset tagging failed with exit code {tag_code}. Dataset upload still succeeded."
                    )
                else:
                    self._set_running(False, "Dataset upload and tagging completed.", False)
                    self._append_output_and_log(f"Dataset tags updated: {', '.join(tags)}")
                self.refresh_local_dataset_browser()
                self.refresh_hf_datasets()

            tag_ok, tag_message = self._start_dataset_tag_upload(
                repo_id=repo_id,
                task=None,
                on_complete=after_tag,
            )
            if not tag_ok and tag_message:
                self._set_running(False, "Dataset upload completed; tagging could not start.", True)
                self._append_output_and_log(
                    f"Dataset upload succeeded, but dataset tagging could not start: {tag_message}"
                )
                self.refresh_local_dataset_browser()
                self.refresh_hf_datasets()

        ok, message = self._run_controller.run_process_async(
            cmd=request["upload_cmd"],
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=after_upload,
            run_mode="upload",
            preflight_checks=request["checks"],
            artifact_context={"dataset_repo_id": repo_id},
        )
        if not ok and message:
            self._handle_launch_rejection(
                title="Upload Unavailable",
                message=message,
                log_message="Record dataset upload launch was rejected.",
            )

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

    def _resolve_dataset_name(self, *, force_occupied: str | None = None) -> Any:
        return resolve_record_dataset_name(
            self._dataset_name_controller.text() or record_dataset_seed(self.config),
            config=self.config,
            dataset_root_raw=self.dataset_root_input.text(),
            force_occupied=force_occupied,
        )

    def _apply_dataset_resolution(
        self,
        resolution: Any,
        *,
        log_change: bool = False,
        preserve_mode: bool = False,
    ) -> None:
        previous_value = self._dataset_name_controller.text()
        target_value = resolution.display_value or resolution.resolved_name
        mode = self._dataset_name_controller.mode() if preserve_mode else "auto"
        self._dataset_name_controller.set_text(target_value, mode=mode)
        if log_change and previous_value and previous_value != target_value and resolution.iterated:
            self._append_log(f"Dataset name '{previous_value}' already exists — advanced to '{target_value}'.")

    def _advance_dataset_name(self, force_occupied: str | None = None, *, log_change: bool = False, preserve_manual: bool = True) -> None:
        if preserve_manual and self._dataset_name_controller.is_manual():
            return
        resolution = self._resolve_dataset_name(force_occupied=force_occupied)
        self._apply_dataset_resolution(resolution, log_change=log_change)

    def _refresh_dataset_name_if_occupied(self) -> None:
        if not self._dataset_name_controller.is_auto():
            return
        resolution = self._resolve_dataset_name()
        if not resolution.occupied:
            return
        self._apply_dataset_resolution(resolution, log_change=True)

    def _ensure_dataset_name_available(self) -> None:
        """Pre-launch check: resolve and auto-fix even in manual mode."""
        resolution = self._resolve_dataset_name()
        if resolution.occupied or resolution.iterated:
            self._apply_dataset_resolution(resolution, log_change=True)
        elif self._dataset_name_controller.is_auto():
            self._apply_dataset_resolution(resolution, log_change=False)

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        super()._set_running(active, status_text, is_error)
        self.camera_preview.set_active_run(active)

    def refresh_from_config(self) -> None:
        self.dataset_root_input.setText(str(self.config.get("record_data_dir", "")).strip())
        self.target_hz_input.setText(str(self.config.get("record_target_hz", "")).strip())
        self.local_hf_owner_input.setText(self._default_record_hf_owner())
        self._advance_dataset_name()
        self.refresh_local_dataset_browser()
        self.refresh_hf_datasets()

    def preview_command(self) -> None:
        self._ensure_dataset_name_available()
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
        self._advance_dataset_name(log_change=True)
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
        self._ensure_dataset_name_available()
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
            episode_duration_s=effective_episode_time,
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
            self._append_output_and_log(f"Hugging Face dataset upload completed: {effective_repo_id}")

            def after_tag(tag_code: int, tag_canceled: bool, tags: list[str]) -> None:
                if tag_canceled:
                    self._set_running(False, "Record + upload completed; tagging canceled.", False)
                    self._append_output_and_log("Dataset tagging canceled after upload.")
                elif tag_code != 0:
                    self._set_running(False, "Record + upload completed; tagging failed.", True)
                    self._append_output_and_log(
                        f"Dataset tagging failed with exit code {tag_code}. Dataset upload still succeeded."
                    )
                else:
                    self._set_running(False, "Record + upload + tagging completed.", False)
                    self._append_output_and_log(f"Dataset tags updated: {', '.join(tags)}")
                self.refresh_local_dataset_browser()
                self.refresh_hf_datasets()

            tag_ok, tag_message = self._start_dataset_tag_upload(
                repo_id=effective_repo_id,
                task=req.task,
                on_complete=after_tag,
            )
            if not tag_ok and tag_message:
                self._set_running(False, "Record + upload completed; tagging could not start.", True)
                self._append_output_and_log(
                    f"Dataset upload succeeded, but dataset tagging could not start: {tag_message}"
                )
                self.refresh_local_dataset_browser()
                self.refresh_hf_datasets()

        def after_record(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Record canceled.", False)
                self._append_output_and_log("Record run canceled. Upload was skipped.")
                self._advance_dataset_name(
                    force_occupied=effective_dataset_name,
                    log_change=True,
                )
                return
            if return_code != 0:
                self._set_running(False, "Record failed.", True)
                self._append_output_and_log(f"Record run failed with exit code {return_code}.")
                self._refresh_dataset_name_if_occupied()
                return

            active_dataset = move_recorded_dataset(
                lerobot_dir=get_lerobot_dir(self.config),
                dataset_name=effective_dataset_name,
                dataset_root=effective_dataset_root,
                log=self._append_output_and_log,
                dataset_repo_id=effective_repo_id,
            )
            self.config["record_data_dir"] = str(effective_dataset_root)
            self.config["last_dataset_name"] = effective_dataset_name
            self.config["last_dataset_repo_id"] = effective_repo_id
            save_config(self.config, quiet=True)
            self._advance_dataset_name(force_occupied=effective_dataset_name, log_change=True, preserve_manual=False)
            self.refresh_local_dataset_browser()

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

    def closeEvent(self, event: object) -> None:
        self._hf_dataset_jobs.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]
