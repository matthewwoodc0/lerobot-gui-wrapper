from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2 as _cv2_module  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal installs
    _cv2_module = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

from .artifacts import (
    _normalize_deploy_episode_outcomes,
    list_runs,
    normalize_deploy_result,
    write_deploy_episode_spreadsheet,
    write_deploy_notes_file,
)
from .camera_schema import apply_camera_schema_entries_to_config, camera_schema_entries_for_editor
from .checks import collect_doctor_checks, summarize_checks
from .compat_snapshot import build_compat_snapshot
from .config_store import _atomic_write, get_deploy_data_dir, get_lerobot_dir, normalize_config_without_prompts, save_config
from .constants import CONFIG_FIELDS
from .desktop_launcher import install_desktop_launcher
from .failure_inspector import (
    build_failure_explanation_text,
    build_run_summary_text,
    has_failure_details,
    raw_transcript_text,
)
from .history_utils import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    open_path_in_file_manager,
)
from .gui_qt_dialogs import (
    ask_editable_command_dialog,
    ask_replay_episode_dialog,
    ask_text_dialog,
    ask_text_dialog_with_actions,
    show_text_dialog,
)
from .hardware_workflows import (
    build_replay_preflight_checks,
    build_replay_readiness_summary,
    build_replay_request_and_command,
    discover_replay_episodes,
)
from .visualizer_utils import (
    _VisualizerRefreshSnapshot,
    _build_selection_payload,
    _collect_sources_for_refresh,
    _open_path,
    _visualizer_source_row_values,
)
from .repo_utils import normalize_deploy_rerun_command
from .run_controller_service import ManagedRunController, RunUiHooks
from .setup_wizard import build_setup_status_summary, build_setup_wizard_guide, probe_setup_wizard_status
from .workspace_compatibility import build_workspace_compatibility_summary
from .workspace_lineage import lineage_rows_for_selection

from .gui_qt_page_base import (
    _CameraSchemaEditor,
    _InputGrid,
    _PageWithOutput,
    _VideoFrameLabel,
    _VideoGalleryTile,
    _build_card,
    _json_text,
    _quiet_cv2_logging,
    _set_readonly_table,
    _set_table_headers,
)
from .gui_qt_output import QtRunOutputPanel

class QtHistoryPage(_PageWithOutput):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="History",
            subtitle="Browse run artifacts, open logs, and rerun prior commands from the main shell.",
            append_log=append_log,
            use_output_tabs=False,
        )
        self.config = config
        self._run_controller = run_controller
        self._rows: list[dict[str, Any]] = []
        self._action_buttons: list[QPushButton] = []
        self._latest_rerun_artifact_path: Path | None = None
        self._latest_rerun_metadata: dict[str, Any] | None = None
        self._set_explain_callback(None)

        # Let the splitter own the available page height.
        root_layout = self.layout()
        root_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        root_layout.setStretch(0, 1)
        root_layout.setStretch(1, 0)

        # --- Filter bar (no stretch) ---
        filter_bar = QWidget()
        filter_bar_layout = QVBoxLayout(filter_bar)
        filter_bar_layout.setContentsMargins(0, 0, 0, 0)
        filter_bar_layout.setSpacing(4)

        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("All modes", "all")
        for mode in HISTORY_MODE_VALUES:
            self.mode_combo.addItem(mode, mode)
        row.addWidget(self.mode_combo)

        self.status_combo = QComboBox()
        self.status_combo.addItem("All statuses", "all")
        self.status_combo.addItem("Success", "success")
        self.status_combo.addItem("Failed", "failed")
        self.status_combo.addItem("Canceled", "canceled")
        row.addWidget(self.status_combo)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Filter by hint, command text, or timestamp")
        row.addWidget(self.query_input, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_history)
        row.addWidget(refresh_button)
        self._action_buttons.append(refresh_button)
        filter_bar_layout.addLayout(row)

        actions = QHBoxLayout()
        self.open_run_button = QPushButton("Open Run Folder")
        self.open_run_button.clicked.connect(self.open_run_folder)
        actions.addWidget(self.open_run_button)
        self._action_buttons.append(self.open_run_button)

        self.open_log_button = QPushButton("Open Command Log")
        self.open_log_button.clicked.connect(self.open_command_log)
        actions.addWidget(self.open_log_button)
        self._action_buttons.append(self.open_log_button)

        self.rerun_button = QPushButton("Rerun Selected")
        self.rerun_button.clicked.connect(self.rerun_selected)
        actions.addWidget(self.rerun_button)
        self._action_buttons.append(self.rerun_button)

        self.replay_button = QPushButton("Replay Selected")
        self.replay_button.clicked.connect(self.replay_selected)
        actions.addWidget(self.replay_button)
        self._action_buttons.append(self.replay_button)

        actions.addStretch(1)
        filter_bar_layout.addLayout(actions)

        self.stats_label = QLabel("")
        self.stats_label.setObjectName("MutedLabel")
        filter_bar_layout.addWidget(self.stats_label)

        self.content_layout.addWidget(filter_bar)

        # --- Main splitter (stretch=1 so it takes all remaining space) ---
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)

        # Top pane: run_table directly (no wrapping card)
        self.run_table = QTableWidget(0, 6)
        _set_table_headers(self.run_table, ["Started", "Duration", "Mode", "Status", "Hint", "Command"])
        _set_readonly_table(self.run_table)
        self.run_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.run_table.setMinimumHeight(150)
        self.run_table.itemSelectionChanged.connect(self._on_selection_changed)

        header = self.run_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.run_table.setColumnWidth(0, 150)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.run_table.setColumnWidth(1, 70)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.run_table.setColumnWidth(2, 75)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.run_table.setColumnWidth(3, 80)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        main_splitter.addWidget(self.run_table)

        # Bottom pane: scroll area containing details
        _details_scroll = QScrollArea()
        _details_scroll.setWidgetResizable(True)
        _details_scroll.setFrameShape(QFrame.Shape.NoFrame)

        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)

        self.history_output_panel = QtRunOutputPanel()
        self.history_output_panel.summary_output.setObjectName("DialogText")
        self.history_output_panel.summary_output.setPlaceholderText("Select a run to see details.")
        self.history_output_panel.raw_output.setPlaceholderText("Select a run to see the raw transcript.")
        details_layout.addWidget(self.history_output_panel)

        # Repoint the inherited output helpers to the inline details panel.
        self.output_panel = self.history_output_panel
        self.status_label = self.history_output_panel.status_label
        self.output = self.history_output_panel.summary_output
        self.raw_output = self.history_output_panel.raw_output
        self._set_explain_callback(None)

        details_layout.addWidget(self._build_workspace_cards())

        # Deploy outcome + notes editor card
        deploy_card, deploy_layout = _build_card("Deploy Outcome + Notes Editor")
        self.deploy_editor_card = deploy_card
        deploy_form = _InputGrid(deploy_layout)

        self.episode_combo = QComboBox()
        self.episode_combo.currentIndexChanged.connect(self._sync_episode_editor_fields)
        deploy_form.add_field("Episode", self.episode_combo)

        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(["success", "failed", "unmarked"])
        deploy_form.add_field("Status", self.outcome_combo)

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("comma,separated,tags")
        deploy_form.add_field("Tags", self.tags_input)

        self.episode_note_input = QLineEdit()
        self.episode_note_input.setPlaceholderText("Optional note for this episode")
        deploy_form.add_field("Episode note", self.episode_note_input)

        notes_label = QLabel("Deployment overall notes")
        notes_label.setObjectName("FormLabel")
        deploy_layout.addWidget(notes_label)

        self.overall_notes = QPlainTextEdit()
        self.overall_notes.setMinimumHeight(110)
        deploy_layout.addWidget(self.overall_notes)

        editor_actions = QHBoxLayout()
        self.save_episode_button = QPushButton("Save Episode Edit")
        self.save_episode_button.clicked.connect(self.save_episode_edit)
        editor_actions.addWidget(self.save_episode_button)
        self._action_buttons.append(self.save_episode_button)

        self.save_notes_button = QPushButton("Save Deployment Notes")
        self.save_notes_button.clicked.connect(self.save_deployment_notes)
        editor_actions.addWidget(self.save_notes_button)
        self._action_buttons.append(self.save_notes_button)

        self.open_notes_button = QPushButton("Open notes.md")
        self.open_notes_button.clicked.connect(self.open_deploy_notes_file)
        editor_actions.addWidget(self.open_notes_button)
        self._action_buttons.append(self.open_notes_button)
        editor_actions.addStretch(1)
        deploy_layout.addLayout(editor_actions)

        self.deploy_editor_status = QLabel("")
        self.deploy_editor_status.setObjectName("MutedLabel")
        self.deploy_editor_status.setWordWrap(True)
        deploy_layout.addWidget(self.deploy_editor_status)

        details_layout.addWidget(deploy_card)
        self.deploy_editor_card.hide()

        details_layout.addStretch(1)
        _details_scroll.setWidget(details_container)
        main_splitter.addWidget(_details_scroll)

        main_splitter.setSizes([420, 260])

        self.content_layout.addWidget(main_splitter, 1)

        self._restore_history_filters()
        self.mode_combo.currentIndexChanged.connect(self._handle_history_filter_changed)
        self.status_combo.currentIndexChanged.connect(self._handle_history_filter_changed)
        self.query_input.textChanged.connect(self._handle_history_query_changed)
        self.refresh_history()

    def _build_workspace_cards(self) -> QFrame:
        card, layout = _build_card("Workspace Links")
        self.workspace_card = card

        self.history_compat_table = QTableWidget(0, 3)
        _set_table_headers(self.history_compat_table, ["Level", "Check", "Detail"])
        _set_readonly_table(self.history_compat_table)
        layout.addWidget(self.history_compat_table)

        self.history_lineage_table = QTableWidget(0, 2)
        _set_table_headers(self.history_lineage_table, ["Relation", "Target"])
        _set_readonly_table(self.history_lineage_table)
        layout.addWidget(self.history_lineage_table)

        actions = QHBoxLayout()
        open_link_button = QPushButton("Open Linked Target")
        open_link_button.clicked.connect(self.open_selected_lineage_target)
        actions.addWidget(open_link_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.workspace_card.hide()
        return self.workspace_card

    def _restore_history_filters(self) -> None:
        mode_value = str(self.config.get("ui_history_mode_filter", "all")).strip()
        status_value = str(self.config.get("ui_history_status_filter", "all")).strip()
        query_value = str(self.config.get("ui_history_query", "")).strip()
        mode_index = self.mode_combo.findData(mode_value)
        status_index = self.status_combo.findData(status_value)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        if status_index >= 0:
            self.status_combo.setCurrentIndex(status_index)
        self.query_input.setText(query_value)

    def _persist_history_filters(self) -> None:
        self.config["ui_history_mode_filter"] = str(self.mode_combo.currentData() or "all")
        self.config["ui_history_status_filter"] = str(self.status_combo.currentData() or "all")
        self.config["ui_history_query"] = self.query_input.text().strip()
        save_config(self.config, quiet=True)

    def _handle_history_filter_changed(self, *_args: object) -> None:
        self._persist_history_filters()
        self.refresh_history()

    def _handle_history_query_changed(self, _text: str) -> None:
        self._persist_history_filters()
        self.refresh_history()

    def _current_row(self) -> dict[str, Any] | None:
        row = self.run_table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        for button in self._action_buttons:
            button.setEnabled(not active)
        if active:
            self.status_label.setText(status_text or "Running rerun...")
            self._set_explain_callback(None)
            self._show_raw_tab()
        elif is_error:
            self.status_label.setText(status_text or "Rerun failed.")
            if self._latest_rerun_metadata is not None:
                self._set_output(
                    title=status_text or "Rerun failed.",
                    text=build_run_summary_text(self._latest_rerun_metadata),
                    log_message=None,
                )
                self._set_explain_callback(self._show_rerun_failure_explanation if has_failure_details(self._latest_rerun_metadata) else None)
                self._show_summary_tab()
        else:
            self.status_label.setText(status_text or "Ready.")
            if self._latest_rerun_metadata is not None:
                self._set_output(
                    title=status_text or "Ready.",
                    text=build_run_summary_text(self._latest_rerun_metadata),
                    log_message=None,
                )
                self._set_explain_callback(self._show_rerun_failure_explanation if has_failure_details(self._latest_rerun_metadata) else None)

    def refresh_history(self) -> None:
        runs, warning_count = list_runs(config=self.config, limit=5000)
        payload = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=warning_count,
            mode_filter=str(self.mode_combo.currentData() or "all"),
            status_filter=str(self.status_combo.currentData() or "all"),
            query=self.query_input.text().strip().lower(),
        )
        self._rows = list(payload.get("rows", []))
        self.run_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = list(row.get("values", ()))
            while len(values) < 6:
                values.append("")
            for col_index, value in enumerate(values[:6]):
                self.run_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if self._rows:
            self.run_table.selectRow(0)
        stats = payload.get("stats", {})
        total = stats.get("total", 0)
        success = stats.get("success", 0)
        failed = stats.get("failed", 0)
        canceled = stats.get("canceled", 0)
        self.stats_label.setText(
            f"Showing {total} runs — {success} success · {failed} failed · {canceled} canceled"
        )
        self._append_log(f"History refreshed {total} rows.")
        if not self._rows:
            self.deploy_editor_card.hide()
            self.workspace_card.hide()
            self._set_explain_callback(None)
            self._set_output(
                title="History Refreshed",
                text=_json_text({"stats": stats, "warning_count": warning_count}),
                log_message=None,
            )
            self._set_raw_output("")
            self._show_summary_tab()

    def _on_selection_changed(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            self.workspace_card.hide()
            self._set_explain_callback(None)
            self._set_output(title="History", text="Select a run to see details.", log_message=None)
            self._set_raw_output("")
            self._show_summary_tab()
            return
        item = row.get("item", {})
        run_path = Path(str(item.get("_run_path", "")).strip()) if str(item.get("_run_path", "")).strip() else None
        self._set_output(title="Run Details", text=build_run_summary_text(item), log_message=None)
        self._set_raw_output(raw_transcript_text(run_path))
        self._show_summary_tab()
        self._set_explain_callback(self._show_selected_failure_explanation if has_failure_details(item) else None)
        self._populate_deploy_editor()
        self._populate_workspace_links(item)

    def _populate_workspace_links(self, item: dict[str, Any]) -> None:
        runs, _warning_count = list_runs(config=self.config, limit=5000)
        compatibility = build_workspace_compatibility_summary(
            config=self.config,
            model_path=str(item.get("model_path", "")).strip() or None,
        )
        compatibility_rows = compatibility.get("issues", []) if isinstance(compatibility, dict) else []
        self.history_compat_table.setRowCount(len(compatibility_rows))
        for row_index, row in enumerate(compatibility_rows):
            if not isinstance(row, dict):
                continue
            self.history_compat_table.setItem(row_index, 0, QTableWidgetItem(str(row.get("level", ""))))
            self.history_compat_table.setItem(row_index, 1, QTableWidgetItem(str(row.get("name", ""))))
            self.history_compat_table.setItem(row_index, 2, QTableWidgetItem(str(row.get("detail", ""))))

        lineage_rows = lineage_rows_for_selection(
            selection={
                "kind": "run",
                "scope": "local",
                "run_id": str(item.get("run_id", "")).strip(),
                "run_path": str(item.get("_run_path", "")).strip(),
            },
            runs=runs,
        )
        self.history_lineage_table.setRowCount(len(lineage_rows))
        for row_index, row in enumerate(lineage_rows):
            if not isinstance(row, dict):
                continue
            self.history_lineage_table.setItem(row_index, 0, QTableWidgetItem(str(row.get("relation", ""))))
            self.history_lineage_table.setItem(row_index, 1, QTableWidgetItem(str(row.get("label", ""))))
        self.workspace_card.setVisible(bool(compatibility_rows or lineage_rows))

    def _populate_deploy_editor(self) -> None:
        row = self._current_row()
        if row is None:
            self.deploy_editor_card.hide()
            return
        item = row.get("item", {})
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self.deploy_editor_card.hide()
            return
        self.deploy_editor_card.show()
        summary, episode_map = self._deploy_episode_map(item)
        choices = self._episode_choices(summary, episode_map)
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems(choices)
        self.episode_combo.setCurrentText(choices[0])
        self.episode_combo.blockSignals(False)
        self.overall_notes.setPlainText(str(item.get("deploy_notes_summary", "")).strip())
        self.deploy_editor_status.setText("Edit episode status/tags/note and deployment notes, then save.")
        self._sync_episode_editor_fields()

    def open_selected_lineage_target(self) -> None:
        row = self.history_lineage_table.currentRow()
        current = self._current_row()
        if row < 0 or current is None:
            self._set_output(title="No Selection", text="Select a run and lineage row first.", log_message="History lineage open skipped with no selection.")
            return
        runs, _warning_count = list_runs(config=self.config, limit=5000)
        lineage_rows = lineage_rows_for_selection(
            selection={
                "kind": "run",
                "scope": "local",
                "run_id": str(current.get("item", {}).get("run_id", "")).strip(),
                "run_path": str(current.get("item", {}).get("_run_path", "")).strip(),
            },
            runs=runs,
        )
        if row >= len(lineage_rows):
            return
        target = str(lineage_rows[row].get("target", "")).strip()
        ok, message = _open_path(target)
        self._set_output(title="Open Linked Target" if ok else "Open Failed", text=target or message, log_message="History opened lineage target." if ok else "History lineage target open failed.")

    def _remember_rerun_artifact(self, artifact_path: Path) -> None:
        self._latest_rerun_artifact_path = Path(artifact_path)
        self._latest_rerun_metadata = self._read_rerun_metadata(self._latest_rerun_artifact_path)
        if self._latest_rerun_artifact_path is not None:
            self._set_raw_output(raw_transcript_text(self._latest_rerun_artifact_path))

    def _read_rerun_metadata(self, run_path: Path | None) -> dict[str, Any] | None:
        if run_path is None:
            return None
        metadata_path = Path(run_path) / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["_run_path"] = str(run_path)
        payload["_metadata_path"] = str(metadata_path)
        return payload

    def _show_selected_failure_explanation(self) -> None:
        row = self._current_row()
        if row is None:
            return
        item = row.get("item", {})
        run_path = Path(str(item.get("_run_path", "")).strip()) if str(item.get("_run_path", "")).strip() else None
        show_text_dialog(
            parent=self,
            title="Failure Explanation",
            text=build_failure_explanation_text(item, run_path=run_path),
            wrap_mode="word",
        )

    def _show_rerun_failure_explanation(self) -> None:
        if self._latest_rerun_metadata is None:
            return
        show_text_dialog(
            parent=self,
            title="Failure Explanation",
            text=build_failure_explanation_text(self._latest_rerun_metadata, run_path=self._latest_rerun_artifact_path),
            wrap_mode="word",
        )

    def open_run_folder(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-run skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        ok, message = open_path_in_file_manager(run_path)
        self._set_output(title="Open Run" if ok else "Open Failed", text=str(run_path if ok else message), log_message="History opened run folder." if ok else "History failed to open run folder.")

    def open_command_log(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History open-log skipped with no selection.")
            return
        run_path = Path(str(row.get("item", {}).get("_run_path", "")))
        log_path = run_path / "command.log"
        ok, message = open_path_in_file_manager(log_path)
        self._set_output(title="Open Log" if ok else "Open Failed", text=str(log_path if ok else message), log_message="History opened command log." if ok else "History failed to open command log.")

    def rerun_selected(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History rerun skipped with no selection.")
            return
        item = row.get("item", {})
        cmd, error = _command_from_item(item)
        if cmd is None:
            self._set_output(title="Rerun Failed", text=error or "No command stored in this history entry.", log_message="History rerun failed while parsing command.")
            return

        run_mode = str(item.get("mode", "run")).strip().lower() or "run"
        cwd_raw = str(item.get("cwd", "")).strip()
        cwd = Path(cwd_raw) if cwd_raw else None
        artifact_context: dict[str, Any] = {}
        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        model_path = str(item.get("model_path", "")).strip()
        if dataset_repo_id:
            artifact_context["dataset_repo_id"] = dataset_repo_id
        if model_path:
            artifact_context["model_path"] = model_path

        rerun_cmd = list(cmd)
        if run_mode == "deploy":
            rerun_cmd, rerun_message = normalize_deploy_rerun_command(
                command_argv=rerun_cmd,
                username=str(self.config.get("hf_username", "")),
                local_roots=[get_deploy_data_dir(self.config), get_lerobot_dir(self.config) / "data"],
            )
            if rerun_message:
                self._append_output_and_log(rerun_message)
                for arg in rerun_cmd:
                    if str(arg).startswith("--dataset.repo_id="):
                        artifact_context["dataset_repo_id"] = str(arg).split("=", 1)[1].strip()
                        break

        self._latest_rerun_artifact_path = None
        self._latest_rerun_metadata = None
        self._set_explain_callback(None)
        self._set_output(title="Rerun Starting", text="Rerunning stored command...", log_message=None)
        self._set_raw_output("")
        self._append_output_chunk(" ".join(rerun_cmd) + "\n")
        self._show_raw_tab()
        hooks = RunUiHooks(
            set_running=self._set_running,
            append_output_line=lambda _line: None,
            append_output_chunk=self._append_output_chunk,
            on_artifact_written=self._remember_rerun_artifact,
        )
        ok, message = self._run_controller.run_process_async(
            cmd=rerun_cmd,
            cwd=cwd,
            hooks=hooks,
            complete_callback=None,
            run_mode=run_mode,
            preflight_checks=None,
            artifact_context=artifact_context,
        )
        if not ok:
            self._set_output(title="Rerun Rejected", text=message or "Unable to rerun selected command.", log_message="History rerun was rejected.")
            return
        self._append_log(f"History rerun started for {run_mode}.")

    def _prompt_replay_episode(self, *, repo_id: str, dataset_path: str, default_episode: int) -> int | None:
        discovery = discover_replay_episodes(self.config, repo_id, dataset_path_raw=dataset_path)
        choices = [str(index) for index in discovery.episode_indices[:500]] or ["0"]
        selected = str(default_episode) if str(default_episode) in choices else (choices[0] if choices else "0")
        choice = ask_replay_episode_dialog(
            parent=self,
            title="Select Replay Episode",
            repo_id=repo_id,
            choices=choices,
            selected_value=selected,
            helper_text=discovery.scan_error or "Use a discovered local episode when possible. Manual override is available if the list is incomplete.",
        )
        if choice is None:
            return None
        try:
            return int(str(choice).strip())
        except (TypeError, ValueError):
            return None

    def replay_selected(self) -> None:
        row = self._current_row()
        if row is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History replay skipped with no selection.")
            return
        item = row.get("item", {})
        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        if not dataset_repo_id:
            self._set_output(
                title="Replay Unavailable",
                text="The selected run does not reference a dataset, so there is nothing to replay on hardware.",
                log_message="History replay skipped because the run had no dataset context.",
            )
            return
        dataset_path = str(item.get("dataset_path", "")).strip()
        default_episode = 0
        replay_episode = item.get("replay_episode")
        if replay_episode is not None:
            try:
                default_episode = int(replay_episode)
            except (TypeError, ValueError):
                default_episode = 0
        elif str(item.get("mode", "")).strip().lower() == "deploy":
            try:
                default_episode = int(self.episode_combo.currentText().strip() or "0")
            except (TypeError, ValueError):
                default_episode = 0

        chosen_episode = self._prompt_replay_episode(
            repo_id=dataset_repo_id,
            dataset_path=dataset_path,
            default_episode=default_episode,
        )
        if chosen_episode is None:
            return

        request, cmd, support, error = build_replay_request_and_command(
            config=self.config,
            dataset_repo_id=dataset_repo_id,
            episode_raw=str(chosen_episode),
            dataset_path_raw=dataset_path,
        )
        if error or request is None or cmd is None:
            self._set_output(title="Replay Failed", text=error or support.detail, log_message="History replay failed while building the command.")
            return

        editable_cmd = ask_editable_command_dialog(
            parent=self,
            title="Confirm Replay Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the replay command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Replay",
            cancel_label="Cancel",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("History replay is using an edited command from the command editor.")
        checks = build_replay_preflight_checks(config=self.config, request=request, support=support)
        if not ask_text_dialog(
            parent=self,
            title="Replay Preflight Review",
            text=build_replay_readiness_summary(config=self.config, request=request, support=support)
            + "\n\n"
            + "\n".join(f"[{level}] {name}: {detail}" for level, name, detail in checks)
            + "\n\nClick Confirm to continue, or Cancel to stop.",
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            self._append_log("History replay canceled after preflight review.")
            return

        self._latest_rerun_artifact_path = None
        self._latest_rerun_metadata = None
        self._set_explain_callback(None)
        self._set_output(title="Replay Starting", text="Replaying selected dataset episode on hardware...", log_message=None)
        self._set_raw_output("")
        self._append_output_chunk(" ".join(editable_cmd) + "\n")
        self._show_raw_tab()
        hooks = RunUiHooks(
            set_running=self._set_running,
            append_output_line=lambda _line: None,
            append_output_chunk=self._append_output_chunk,
            on_artifact_written=self._remember_rerun_artifact,
        )
        ok, message = self._run_controller.run_process_async(
            cmd=editable_cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=hooks,
            complete_callback=None,
            run_mode="replay",
            preflight_checks=checks,
            artifact_context={
                "dataset_repo_id": request.dataset_repo_id,
                "dataset_path": str(request.dataset_path) if request.dataset_path is not None else "",
                "replay_episode": request.episode_index,
            },
        )
        if not ok:
            self._set_output(title="Replay Rejected", text=message or "Unable to replay the selected dataset episode.", log_message="History replay was rejected.")
            return
        self._append_log(f"History replay started for {request.dataset_repo_id} episode {request.episode_index}.")

    def _read_selected_metadata(self) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
        row = self._current_row()
        if row is None:
            return None, None, None
        item = row.get("item", {})
        metadata_path_raw = str(item.get("_metadata_path", "")).strip()
        if not metadata_path_raw:
            return item, None, None
        metadata_path = Path(metadata_path_raw)
        if not metadata_path.exists():
            return item, metadata_path, None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return item, metadata_path, None
        if not isinstance(data, dict):
            return item, metadata_path, None
        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        return item, metadata_path, data

    def _deploy_episode_map(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        summary = _normalize_deploy_episode_outcomes(item.get("deploy_episode_outcomes"))
        episode_map: dict[int, dict[str, Any]] = {}
        entries = summary.get("episode_outcomes")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    episode_idx = int(entry.get("episode"))
                except (TypeError, ValueError):
                    continue
                if episode_idx <= 0:
                    continue
                episode_map[episode_idx] = dict(entry)
        return summary, episode_map

    def _episode_choices(self, summary: dict[str, Any], episode_map: dict[int, dict[str, Any]]) -> list[str]:
        total = summary.get("total_episodes")
        if isinstance(total, int) and total > 0:
            return [str(value) for value in range(1, total + 1)]
        if episode_map:
            return [str(value) for value in sorted(episode_map)]
        return ["1"]

    def _sync_episode_editor_fields(self) -> None:
        row = self._current_row()
        if row is None:
            return
        item = row.get("item", {})
        summary, episode_map = self._deploy_episode_map(item)
        try:
            current_episode = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            current_episode = 1
        if not episode_map and not summary:
            current_entry: dict[str, Any] = {}
        else:
            current_entry = episode_map.get(current_episode, {})
        self.outcome_combo.setCurrentText(normalize_deploy_result(current_entry.get("result", "unmarked")))
        tags = current_entry.get("tags") if isinstance(current_entry.get("tags"), list) else []
        self.tags_input.setText(", ".join(str(tag) for tag in tags))
        self.episode_note_input.setText(str(current_entry.get("note", "")).strip())

    def _persist_deploy_metadata(self, updated_data: dict[str, Any]) -> tuple[bool, str]:
        metadata_path_raw = str(updated_data.get("_metadata_path", "")).strip()
        run_path_raw = str(updated_data.get("_run_path", "")).strip()
        if not metadata_path_raw or not run_path_raw:
            return False, "Selected run is missing metadata/run path references."
        metadata_path = Path(metadata_path_raw)
        run_path = Path(run_path_raw)
        payload = dict(updated_data)
        payload.pop("_metadata_path", None)
        payload.pop("_run_path", None)
        try:
            _atomic_write(json.dumps(payload, indent=2) + "\n", metadata_path)
        except OSError as exc:
            return False, f"Failed to write metadata.json: {exc}"
        notes_path = write_deploy_notes_file(run_path, payload, filename="notes.md")
        if notes_path is None:
            return False, "Saved metadata, but failed to write notes.md"
        csv_path, summary_csv_path = write_deploy_episode_spreadsheet(
            run_path,
            payload,
            filename="episode_outcomes.csv",
            summary_filename="episode_outcomes_summary.csv",
        )
        if csv_path is None or summary_csv_path is None:
            return False, "Saved metadata and notes, but failed to write episode CSV files."
        row = self._current_row()
        if row is not None:
            item = row.get("item", {})
            item.clear()
            item.update(payload)
            item["_metadata_path"] = str(metadata_path)
            item["_run_path"] = str(run_path)
        return True, f"Saved deploy edits: {notes_path.name}, {csv_path.name}, {summary_csv_path.name}"

    def save_episode_edit(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History episode save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deploy episode editing is only available for deploy runs.", log_message="History episode save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History episode save failed due to unreadable metadata.")
            return
        try:
            episode_idx = int(self.episode_combo.currentText().strip() or "1")
        except ValueError:
            episode_idx = 1
        summary = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        entries = summary.get("episode_outcomes")
        episode_map: dict[int, dict[str, Any]] = {}
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    try:
                        entry_idx = int(entry.get("episode"))
                    except (TypeError, ValueError):
                        continue
                    episode_map[entry_idx] = dict(entry)
        entry = episode_map.get(episode_idx, {"episode": episode_idx})
        entry["episode"] = episode_idx
        entry["result"] = normalize_deploy_result(self.outcome_combo.currentText())
        tags = [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()]
        entry["tags"] = tags
        note = self.episode_note_input.text().strip()
        if note:
            entry["note"] = note
        else:
            entry.pop("note", None)
        episode_map[episode_idx] = entry
        total = summary.get("total_episodes")
        if not isinstance(total, int) or total < episode_idx:
            total = episode_idx
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(
            {"enabled": True, "total_episodes": total, "episode_outcomes": [episode_map[idx] for idx in sorted(episode_map)]}
        )
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Episode Edit Saved" if ok else "Save Failed", text=message, log_message=message)
        if ok:
            self._populate_deploy_editor()

    def save_deployment_notes(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes save skipped with no selection.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            self._set_output(title="History", text="Deployment notes are only available for deploy runs.", log_message="History notes save skipped for non-deploy run.")
            return
        if metadata_data is None:
            self._set_output(title="History", text="Unable to read metadata.json for this run.", log_message="History notes save failed due to unreadable metadata.")
            return
        notes = self.overall_notes.toPlainText().strip()
        if notes:
            metadata_data["deploy_notes_summary"] = notes
        else:
            metadata_data.pop("deploy_notes_summary", None)
        metadata_data["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        ok, message = self._persist_deploy_metadata(metadata_data)
        self.deploy_editor_status.setText(message)
        self._set_output(title="Deployment Notes Saved" if ok else "Save Failed", text=message, log_message=message)

    def open_deploy_notes_file(self) -> None:
        item, _metadata_path, metadata_data = self._read_selected_metadata()
        if item is None:
            self._set_output(title="No Selection", text="Select a run first.", log_message="History notes open skipped with no selection.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        if not run_path.exists():
            self._set_output(title="Open Failed", text="Run folder is missing for this history entry.", log_message="History notes open failed due to missing run path.")
            return
        notes_path = run_path / "notes.md"
        if not notes_path.exists() and metadata_data is not None:
            generated = write_deploy_notes_file(run_path, metadata_data, filename="notes.md")
            if generated is not None:
                notes_path = generated
                write_deploy_episode_spreadsheet(
                    run_path,
                    metadata_data,
                    filename="episode_outcomes.csv",
                    summary_filename="episode_outcomes_summary.csv",
                )
        ok, message = open_path_in_file_manager(notes_path)
        self._set_output(
            title="Open Notes" if ok else "Open Failed",
            text=str(notes_path if ok else message),
            log_message="History opened deploy notes." if ok else "History failed to open deploy notes.",
        )
