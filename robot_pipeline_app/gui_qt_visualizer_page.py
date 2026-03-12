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
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
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
from .history_utils import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    open_path_in_file_manager,
)
from .gui_qt_dialogs import ask_text_dialog_with_actions
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

class QtVisualizerPage(_PageWithOutput):
    _ROOT_STATE_KEYS = {
        "deployments": "ui_visualizer_deploy_root",
        "datasets": "ui_visualizer_dataset_root",
        "models": "ui_visualizer_model_root",
    }

    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None]) -> None:
        super().__init__(
            title="Visualizer",
            subtitle="Browse local deployment runs, datasets, models, and discovered video assets.",
            append_log=append_log,
        )
        self.config = config
        self._sources: list[dict[str, Any]] = []
        self._video_tiles: list[_VideoGalleryTile] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._current_source_kind = self._persisted_source_kind()

        self.content_layout.addWidget(self._build_controls_card())
        self.content_layout.addWidget(self._build_video_gallery_card())
        self.content_layout.addWidget(self._build_sources_card())
        self.content_layout.addWidget(self._build_details_card())
        self.content_layout.addWidget(self._build_insights_card())
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.addStretch(1)

        self._restore_persisted_visualizer_state()
        self.source_combo.currentIndexChanged.connect(self._handle_source_changed)
        self.hf_owner_input.editingFinished.connect(self._persist_visualizer_state)
        self.root_input.editingFinished.connect(self._persist_visualizer_state)
        self._handle_source_changed()

    def _build_controls_card(self) -> QFrame:
        card, layout = _build_card("Source Browser")

        top_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem("Deployments", "deployments")
        self.source_combo.addItem("Datasets", "datasets")
        self.source_combo.addItem("Models", "models")
        top_row.addWidget(QLabel("Source"))
        top_row.addWidget(self.source_combo)

        self.root_input = QLineEdit(self._root_text_for_source(self._current_source_kind))
        top_row.addWidget(QLabel("Root"))
        top_row.addWidget(self.root_input, 1)

        self.hf_owner_input = QLineEdit(str(self.config.get("ui_visualizer_hf_owner", self.config.get("hf_username", ""))).strip())
        top_row.addWidget(QLabel("HF owner"))
        top_row.addWidget(self.hf_owner_input)

        browse_root_button = QPushButton("Browse Root")
        browse_root_button.clicked.connect(self.browse_root)
        top_row.addWidget(browse_root_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.setObjectName("AccentButton")
        refresh_button.clicked.connect(self.refresh_sources)
        top_row.addWidget(refresh_button)
        layout.addLayout(top_row)

        actions = QHBoxLayout()
        open_source_button = QPushButton("Open Source")
        open_source_button.clicked.connect(self.open_selected_source)
        actions.addWidget(open_source_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_sources_card(self) -> QFrame:
        card, layout = _build_card("Sources")
        self.source_table = QTableWidget(0, 2)
        _set_table_headers(self.source_table, ["Source", "Name"])
        _set_readonly_table(self.source_table)
        self.source_table.setMinimumHeight(180)
        self.source_table.setMaximumHeight(280)
        self.source_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_table.itemSelectionChanged.connect(self._on_source_selection_changed)
        layout.addWidget(self.source_table)
        return card

    def _build_details_card(self) -> QFrame:
        card, layout = _build_card("Selection Details")
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(140)
        self.meta_view.setMaximumHeight(220)
        self.meta_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.meta_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.meta_view)
        return card

    def _build_insights_card(self) -> QFrame:
        self.insights_card, layout = _build_card("Deployment Insights")
        self.insights_table = QTableWidget(0, 4)
        _set_table_headers(self.insights_table, ["Episode", "Result", "Notes", "Tags"])
        _set_readonly_table(self.insights_table)
        layout.addWidget(self.insights_table)
        self.insights_card.hide()
        return self.insights_card

    def _build_video_gallery_card(self) -> QFrame:
        self.video_gallery_card, layout = _build_card("Video Gallery")
        self.video_status = QLabel("Select a source to display discovered videos.")
        self.video_status.setObjectName("MutedLabel")
        self.video_status.setWordWrap(True)
        layout.addWidget(self.video_status)

        self.video_grid_host = QWidget()
        self.video_grid = QGridLayout(self.video_grid_host)
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        self.video_grid.setHorizontalSpacing(14)
        self.video_grid.setVerticalSpacing(14)
        self.video_grid_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.video_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_scroll.setMinimumHeight(360)
        self.video_scroll.setWidget(self.video_grid_host)
        layout.addWidget(self.video_scroll, 1)
        return self.video_gallery_card

    def _handle_source_changed(self, *_args: object) -> None:
        new_source = self._active_source()
        if self._current_source_kind != new_source:
            self._persist_visualizer_root(self._current_source_kind, self.root_input.text().strip())
        self._current_source_kind = new_source
        self._sync_root_placeholder()
        self._set_insights_visible(False)
        self._persist_visualizer_state()
        self.refresh_sources()

    def browse_root(self) -> None:
        current = self.root_input.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Visualizer Root",
            current,
        )
        if not selected:
            return
        self.root_input.setText(selected)
        self._persist_visualizer_state()
        self.refresh_sources()

    def _active_source(self) -> str:
        return str(self.source_combo.currentData() or "deployments")

    def _persisted_source_kind(self) -> str:
        value = str(self.config.get("ui_visualizer_source_kind", "deployments")).strip().lower()
        if value in {"deployments", "datasets", "models"}:
            return value
        return "deployments"

    def _default_root_for_source(self, source: str) -> str:
        if source == "deployments":
            return str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
        if source == "datasets":
            return str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
        return str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))

    def _root_text_for_source(self, source: str) -> str:
        key = self._ROOT_STATE_KEYS.get(source, "")
        persisted = str(self.config.get(key, "")).strip()
        return persisted or self._default_root_for_source(source)

    def _restore_persisted_visualizer_state(self) -> None:
        source = self._persisted_source_kind()
        index = self.source_combo.findData(source)
        blocked = self.source_combo.blockSignals(True)
        if index >= 0:
            self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(blocked)
        self.root_input.setText(self._root_text_for_source(source))
        self._current_source_kind = source

    def _persist_visualizer_root(self, source: str, root_text: str) -> None:
        key = self._ROOT_STATE_KEYS.get(str(source).strip(), "")
        if not key:
            return
        value = str(root_text).strip()
        if value:
            self.config[key] = value
        else:
            self.config.pop(key, None)

    def _source_identity(self, source: dict[str, Any] | None) -> tuple[str, str, str]:
        if not isinstance(source, dict):
            return "", "", ""
        return (
            str(source.get("scope", "")).strip(),
            str(source.get("kind", "")).strip(),
            str(source.get("name", source.get("repo_id", source.get("path", "")))).strip(),
        )

    def _preferred_source_identity(self) -> tuple[str, str, str]:
        current_source = self._current_source()
        current_identity = self._source_identity(current_source)
        if any(current_identity):
            return current_identity
        return (
            str(self.config.get("ui_visualizer_selected_scope", "")).strip(),
            str(self.config.get("ui_visualizer_selected_kind", "")).strip(),
            str(self.config.get("ui_visualizer_selected_name", "")).strip(),
        )

    def _restore_source_selection(self, preferred_identity: tuple[str, str, str]) -> bool:
        if not any(preferred_identity):
            return False
        for row, source in enumerate(self._sources):
            if self._source_identity(source) == preferred_identity:
                self.source_table.selectRow(row)
                return True
        return False

    def _persist_visualizer_state(self) -> None:
        self.config["ui_visualizer_source_kind"] = self._active_source()
        self.config["ui_visualizer_hf_owner"] = self.hf_owner_input.text().strip()
        self._persist_visualizer_root(self._active_source(), self.root_input.text().strip())
        scope, kind, name = self._source_identity(self._current_source())
        if scope and kind and name:
            self.config["ui_visualizer_selected_scope"] = scope
            self.config["ui_visualizer_selected_kind"] = kind
            self.config["ui_visualizer_selected_name"] = name
        else:
            self.config.pop("ui_visualizer_selected_scope", None)
            self.config.pop("ui_visualizer_selected_kind", None)
            self.config.pop("ui_visualizer_selected_name", None)
        save_config(self.config, quiet=True)

    def _sync_root_placeholder(self) -> None:
        self.root_input.setText(self._root_text_for_source(self._active_source()))

    def _set_insights_visible(self, visible: bool) -> None:
        self.insights_card.setVisible(bool(visible))

    def _clear_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.stop()
        self._video_tiles.clear()
        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._refresh_scroll_geometry()

    def _video_gallery_columns(self, video_count: int) -> int:
        if video_count <= 1:
            return 1
        if video_count <= 4:
            return 2
        return 3

    def _start_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.start()

    def _stop_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.stop()

    def _render_video_gallery(self, videos: list[dict[str, Any]], *, empty_message: str) -> None:
        self._clear_video_tiles()
        video_items = list(videos)
        if not video_items:
            self.video_status.setText(empty_message)
            empty = QLabel(empty_message)
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("MutedLabel")
            self.video_grid.addWidget(empty, 0, 0)
            self._refresh_scroll_geometry()
            return

        count = len(video_items)
        self.video_status.setText(f"Displaying {count} video{'s' if count != 1 else ''} from the selected source.")
        columns = self._video_gallery_columns(count)
        for column in range(columns):
            self.video_grid.setColumnStretch(column, 1)
        for index, item in enumerate(video_items):
            tile = _VideoGalleryTile(item=item)
            self._video_tiles.append(tile)
            row = index // columns
            column = index % columns
            self.video_grid.addWidget(tile, row, column)
        self._refresh_scroll_geometry()
        if self.isVisible():
            self._start_video_tiles()

    def _refresh_scroll_geometry(self) -> None:
        self.video_grid_host.updateGeometry()
        self.video_grid_host.adjustSize()
        self.video_gallery_card.updateGeometry()
        self.video_gallery_card.adjustSize()
        self.updateGeometry()
        self.adjustSize()
        QTimer.singleShot(0, self._refresh_parent_scroll_area)

    def _refresh_parent_scroll_area(self) -> None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                widget = parent.widget()
                if widget is not None:
                    widget.updateGeometry()
                    widget.adjustSize()
                parent.viewport().update()
                break
            parent = parent.parentWidget()

    def _snapshot(self) -> _VisualizerRefreshSnapshot:
        source = self._active_source()
        root_text = self.root_input.text().strip()
        if source == "deployments":
            deploy_root = root_text or str(get_deploy_data_dir(self.config))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        elif source == "datasets":
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = root_text or str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        else:
            deploy_root = str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
            dataset_root = str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
            model_root = root_text or str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))
        return _VisualizerRefreshSnapshot(
            source=source,
            deploy_root=deploy_root,
            dataset_root=dataset_root,
            model_root=model_root,
            hf_owner=self.hf_owner_input.text().strip(),
        )

    def refresh_sources(self) -> None:
        preferred_identity = self._preferred_source_identity()
        snapshot = self._snapshot()
        sources, error_text, source_kind = _collect_sources_for_refresh(self.config, snapshot)
        self._sources = list(sources)
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            scope_text, name_text = _visualizer_source_row_values(source)
            self.source_table.setItem(row, 0, QTableWidgetItem(scope_text))
            self.source_table.setItem(row, 1, QTableWidgetItem(name_text))
        self.meta_view.setPlainText("")
        self.insights_table.setRowCount(0)
        self._set_insights_visible(False)
        self._render_video_gallery([], empty_message="Select a source to display discovered videos.")
        if self._sources:
            self._set_output(
                title="Sources Loaded",
                text=f"Loaded {len(self._sources)} {source_kind}.",
                log_message=f"Visualizer refreshed {len(self._sources)} {source_kind}.",
            )
            if not self._restore_source_selection(preferred_identity):
                self.source_table.selectRow(0)
        else:
            detail = error_text or f"No {source_kind} were found."
            self._set_output(title="No Sources", text=detail, log_message="Visualizer refresh returned no sources.")
            self._persist_visualizer_state()

    def _current_source(self) -> dict[str, Any] | None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self._sources):
            return None
        return self._sources[row]

    def _on_source_selection_changed(self) -> None:
        source = self._current_source()
        if source is None:
            self.meta_view.setPlainText("")
            self.insights_table.setRowCount(0)
            self._set_insights_visible(False)
            self._render_video_gallery([], empty_message="Select a source to display discovered videos.")
            self._persist_visualizer_state()
            return
        payload = _build_selection_payload(source)
        text = _json_text(payload.get("meta_payload", {}))
        self.meta_view.setPlainText(text)
        self.insights_table.setRowCount(0)
        show_insights = self._active_source() == "deployments" and bool(payload.get("insights_visible"))
        self._set_insights_visible(show_insights)
        if show_insights:
            self.insights_table.setRowCount(len(payload.get("insights_rows", [])))
            for row_index, row in enumerate(payload.get("insights_rows", [])):
                if not isinstance(row, tuple) or len(row) != 4:
                    continue
                for col_index, value in enumerate(row):
                    self.insights_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        if show_insights:
            text += "\n\n" + str(payload.get("insights_header", "Deployment Insights"))
            for row in payload.get("insights_rows", []):
                if isinstance(row, tuple) and len(row) == 4:
                    text += f"\n- Episode {row[0]} | {row[1]} | {row[2]} | {row[3]}"
        self._set_output(title="Selection Details", text=text, log_message=None)
        self._render_video_gallery(
            list(payload.get("videos", [])),
            empty_message="No videos found for the selected source.",
        )
        self._persist_visualizer_state()

    def open_selected_source(self) -> None:
        source = self._current_source()
        if source is None:
            self._set_output(title="No Selection", text="Select a source first.", log_message="Visualizer source open skipped with no selection.")
            return
        target = source.get("path")
        if not target and source.get("repo_id"):
            repo_id = str(source.get("repo_id"))
            if str(source.get("kind")) == "dataset":
                target = f"https://huggingface.co/datasets/{repo_id}"
            else:
                target = f"https://huggingface.co/{repo_id}"
        ok, message = _open_path(target or "")
        self._set_output(title="Open Source" if ok else "Open Failed", text=str(target or message), log_message="Visualizer opened source." if ok else "Visualizer source open failed.")

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[misc]
        self._start_video_tiles()

    def hideEvent(self, event: object) -> None:
        self._stop_video_tiles()
        super().hideEvent(event)  # type: ignore[misc]

    def refresh_from_config(self) -> None:
        self._restore_persisted_visualizer_state()
        self.refresh_sources()
