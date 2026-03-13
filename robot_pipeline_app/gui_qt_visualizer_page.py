from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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

from .command_text import format_command_for_dialog
from .config_store import get_deploy_data_dir, get_lerobot_dir, save_config
from .dataset_operations import DatasetOperationError, DatasetOperationService
from .repo_utils import sync_hf_asset
from .dataset_tools import (
    build_delete_episodes_command,
    build_keep_episodes_command,
    collect_local_dataset_episode_indices,
)
from .gui_qt_visualizer_cards import _DatasetToolsCard, _DatasetVisualizationCard
from .gui_qt_dialogs import ask_editable_command_dialog, ask_replay_episode_dialog, ask_text_dialog
from .visualize_tools import build_visualize_dataset_command
from .visualizer_utils import (
    _VisualizerRefreshSnapshot,
    _build_selection_payload,
    _collect_sources_for_refresh,
    _open_path,
    _visualizer_source_row_values,
)
from .run_controller_service import ManagedRunController, RunUiHooks

from .gui_qt_page_base import (
    _InputGrid,
    _PageWithOutput,
    _VideoGalleryTile,
    _build_card,
    _json_text,
    _set_readonly_table,
    _set_table_headers,
)


class _SourceBrowserPanel(QFrame):
    source_changed = Signal(Path)
    refresh_requested = Signal()
    open_requested = Signal()
    sync_requested = Signal()

    _ROOT_STATE_KEYS = {
        "deployments": "ui_visualizer_deploy_root",
        "datasets": "ui_visualizer_dataset_root",
        "models": "ui_visualizer_model_root",
    }

    def __init__(self, *, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._sources: list[dict[str, Any]] = []
        self._current_source_kind = self.persisted_source_kind()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_controls_card())
        layout.addWidget(self._build_sources_card())

        self.restore_persisted_state()
        self.source_combo.currentIndexChanged.connect(self._handle_source_changed)
        self.hf_owner_input.editingFinished.connect(self.persist_state)
        self.hf_query_input.editingFinished.connect(self.persist_state)
        self.hf_task_input.editingFinished.connect(self.persist_state)
        self.hf_tag_input.editingFinished.connect(self.persist_state)
        self.root_input.editingFinished.connect(self.persist_state)

    def _build_controls_card(self) -> QFrame:
        card, layout = _build_card("Source Browser")

        form = _InputGrid(layout)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Deployments", "deployments")
        self.source_combo.addItem("Datasets", "datasets")
        self.source_combo.addItem("Models", "models")
        form.add_field("Source", self.source_combo)

        self.root_input = QLineEdit(self.root_text_for_source(self._current_source_kind))
        root_row = QWidget()
        root_row_layout = QHBoxLayout(root_row)
        root_row_layout.setContentsMargins(0, 0, 0, 0)
        root_row_layout.setSpacing(8)
        root_row_layout.addWidget(self.root_input, 1)
        self.browse_root_button = QPushButton("Browse Root")
        self.browse_root_button.clicked.connect(self.browse_root)
        root_row_layout.addWidget(self.browse_root_button)
        form.add_field("Root", root_row)

        self.hf_owner_input = QLineEdit(str(self.config.get("ui_visualizer_hf_owner", self.config.get("hf_username", ""))).strip())
        form.add_field("HF owner", self.hf_owner_input)

        self.hf_query_input = QLineEdit(str(self.config.get("ui_visualizer_hf_query", "")).strip())
        self.hf_query_input.setPlaceholderText("HF search")
        form.add_field("HF query", self.hf_query_input)

        self.hf_task_input = QLineEdit(str(self.config.get("ui_visualizer_hf_task", "")).strip())
        self.hf_task_input.setPlaceholderText("task")
        form.add_field("HF task", self.hf_task_input)

        self.hf_tag_input = QLineEdit(str(self.config.get("ui_visualizer_hf_tag", "")).strip())
        self.hf_tag_input.setPlaceholderText("tag")
        form.add_field("HF tag", self.hf_tag_input)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("AccentButton")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        actions.addWidget(self.refresh_button)
        open_source_button = QPushButton("Open Source")
        open_source_button.clicked.connect(self.open_requested.emit)
        actions.addWidget(open_source_button)
        sync_source_button = QPushButton("Sync Selected")
        sync_source_button.clicked.connect(self.sync_requested.emit)
        actions.addWidget(sync_source_button)
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
        self.persist_state()
        self.refresh_requested.emit()

    def active_source(self) -> str:
        return str(self.source_combo.currentData() or "deployments")

    def persisted_source_kind(self) -> str:
        value = str(self.config.get("ui_visualizer_source_kind", "deployments")).strip().lower()
        if value in {"deployments", "datasets", "models"}:
            return value
        return "deployments"

    def default_root_for_source(self, source: str) -> str:
        if source == "deployments":
            return str(self.config.get("deploy_data_dir", get_deploy_data_dir(self.config)))
        if source == "datasets":
            return str(self.config.get("record_data_dir", get_lerobot_dir(self.config) / "data"))
        return str(self.config.get("trained_models_dir", get_lerobot_dir(self.config) / "trained_models"))

    def root_text_for_source(self, source: str) -> str:
        key = self._ROOT_STATE_KEYS.get(source, "")
        persisted = str(self.config.get(key, "")).strip()
        return persisted or self.default_root_for_source(source)

    def restore_persisted_state(self) -> None:
        source = self.persisted_source_kind()
        index = self.source_combo.findData(source)
        blocked = self.source_combo.blockSignals(True)
        if index >= 0:
            self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(blocked)
        self.root_input.setText(self.root_text_for_source(source))
        self.hf_owner_input.setText(str(self.config.get("ui_visualizer_hf_owner", self.config.get("hf_username", ""))).strip())
        self.hf_query_input.setText(str(self.config.get("ui_visualizer_hf_query", "")).strip())
        self.hf_task_input.setText(str(self.config.get("ui_visualizer_hf_task", "")).strip())
        self.hf_tag_input.setText(str(self.config.get("ui_visualizer_hf_tag", "")).strip())
        self._current_source_kind = source

    def persist_visualizer_root(self, source: str, root_text: str) -> None:
        key = self._ROOT_STATE_KEYS.get(str(source).strip(), "")
        if not key:
            return
        value = str(root_text).strip()
        if value:
            self.config[key] = value
        else:
            self.config.pop(key, None)

    def source_identity(self, source: dict[str, Any] | None) -> tuple[str, str, str]:
        if not isinstance(source, dict):
            return "", "", ""
        return (
            str(source.get("scope", "")).strip(),
            str(source.get("kind", "")).strip(),
            str(source.get("name", source.get("repo_id", source.get("path", "")))).strip(),
        )

    def preferred_source_identity(self) -> tuple[str, str, str]:
        current_source = self.current_source()
        current_identity = self.source_identity(current_source)
        if any(current_identity):
            return current_identity
        return (
            str(self.config.get("ui_visualizer_selected_scope", "")).strip(),
            str(self.config.get("ui_visualizer_selected_kind", "")).strip(),
            str(self.config.get("ui_visualizer_selected_name", "")).strip(),
        )

    def restore_source_selection(self, preferred_identity: tuple[str, str, str]) -> bool:
        if not any(preferred_identity):
            return False
        for row, source in enumerate(self._sources):
            if self.source_identity(source) == preferred_identity:
                self.source_table.selectRow(row)
                return True
        return False

    def persist_state(self) -> None:
        self.config["ui_visualizer_source_kind"] = self.active_source()
        self.config["ui_visualizer_hf_owner"] = self.hf_owner_input.text().strip()
        self.config["ui_visualizer_hf_query"] = self.hf_query_input.text().strip()
        self.config["ui_visualizer_hf_task"] = self.hf_task_input.text().strip()
        self.config["ui_visualizer_hf_tag"] = self.hf_tag_input.text().strip()
        self.persist_visualizer_root(self.active_source(), self.root_input.text().strip())
        scope, kind, name = self.source_identity(self.current_source())
        if scope and kind and name:
            self.config["ui_visualizer_selected_scope"] = scope
            self.config["ui_visualizer_selected_kind"] = kind
            self.config["ui_visualizer_selected_name"] = name
        else:
            self.config.pop("ui_visualizer_selected_scope", None)
            self.config.pop("ui_visualizer_selected_kind", None)
            self.config.pop("ui_visualizer_selected_name", None)
        save_config(self.config, quiet=True)

    def sync_root_placeholder(self) -> None:
        self.root_input.setText(self.root_text_for_source(self.active_source()))

    def set_sources(self, sources: list[dict[str, Any]]) -> None:
        self._sources = list(sources)
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            scope_text, name_text = _visualizer_source_row_values(source)
            self.source_table.setItem(row, 0, QTableWidgetItem(scope_text))
            self.source_table.setItem(row, 1, QTableWidgetItem(name_text))

    def current_source(self) -> dict[str, Any] | None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self._sources):
            return None
        return self._sources[row]

    def snapshot(self) -> _VisualizerRefreshSnapshot:
        source = self.active_source()
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
            hf_query=self.hf_query_input.text().strip(),
            hf_task=self.hf_task_input.text().strip(),
            hf_tag=self.hf_tag_input.text().strip(),
        )

    def _handle_source_changed(self, *_args: object) -> None:
        new_source = self.active_source()
        if self._current_source_kind != new_source:
            self.persist_visualizer_root(self._current_source_kind, self.root_input.text().strip())
        self._current_source_kind = new_source
        self.sync_root_placeholder()
        self.persist_state()
        self.refresh_requested.emit()

    def _selection_signal_path(self, source: dict[str, Any] | None) -> Path:
        if not isinstance(source, dict):
            return Path(".")
        value = str(source.get("path") or source.get("repo_id") or source.get("name") or ".").strip() or "."
        return Path(value)

    def _on_source_selection_changed(self) -> None:
        self.persist_state()
        self.source_changed.emit(self._selection_signal_path(self.current_source()))


class _VideoGalleryPanel(QFrame):
    video_selected = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self._video_tiles: list[_VideoGalleryTile] = []
        self._tile_paths: dict[_VideoGalleryTile, Path] = {}
        self._current_source = Path(".")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_gallery_card, card_layout = _build_card("Video Gallery")
        layout.addWidget(self.video_gallery_card)

        self.video_status = QLabel("Select a source to display discovered videos.")
        self.video_status.setObjectName("MutedLabel")
        self.video_status.setWordWrap(True)
        card_layout.addWidget(self.video_status)

        self.video_detail = QLabel("No video selected.")
        self.video_detail.setObjectName("MutedLabel")
        self.video_detail.setWordWrap(True)
        card_layout.addWidget(self.video_detail)

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
        card_layout.addWidget(self.video_scroll, 1)

    def load_source(self, path: Path) -> None:
        self._current_source = Path(path)
        self.video_detail.setText(f"Source: {self._current_source}")
        self.set_videos([], empty_message="Select a source to display discovered videos.")

    def clear_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.removeEventFilter(self)
            tile.stop()
        self._video_tiles.clear()
        self._tile_paths.clear()
        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.refresh_scroll_geometry()

    def video_gallery_columns(self, video_count: int) -> int:
        if video_count <= 1:
            return 1
        if video_count <= 4:
            return 2
        return 3

    def start_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.start()

    def stop_video_tiles(self) -> None:
        for tile in self._video_tiles:
            tile.stop()

    def set_videos(self, videos: list[dict[str, Any]], *, empty_message: str) -> None:
        self.clear_video_tiles()
        video_items = list(videos)
        if not video_items:
            self.video_status.setText(empty_message)
            empty = QLabel(empty_message)
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("MutedLabel")
            self.video_grid.addWidget(empty, 0, 0)
            self.refresh_scroll_geometry()
            return

        count = len(video_items)
        self.video_status.setText(f"Displaying {count} video{'s' if count != 1 else ''} from the selected source.")
        columns = self.video_gallery_columns(count)
        for column in range(columns):
            self.video_grid.setColumnStretch(column, 1)
        for index, item in enumerate(video_items):
            tile = _VideoGalleryTile(item=item)
            tile.installEventFilter(self)
            self._video_tiles.append(tile)
            self._tile_paths[tile] = self._tile_path(item)
            row = index // columns
            column = index % columns
            self.video_grid.addWidget(tile, row, column)
        self._set_selected_video(self._tile_paths.get(self._video_tiles[0], self._current_source))
        self.refresh_scroll_geometry()
        if self.isVisible():
            self.start_video_tiles()

    def refresh_scroll_geometry(self) -> None:
        self.video_grid_host.updateGeometry()
        self.video_grid_host.adjustSize()
        self.video_gallery_card.updateGeometry()
        self.video_gallery_card.adjustSize()
        self.updateGeometry()
        self.adjustSize()
        QTimer.singleShot(0, self.refresh_parent_scroll_area)

    def refresh_parent_scroll_area(self) -> None:
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

    def eventFilter(self, watched: object, event: object) -> bool:
        if isinstance(watched, _VideoGalleryTile) and isinstance(event, QEvent):
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._set_selected_video(self._tile_paths.get(watched, self._current_source))
        return super().eventFilter(watched, event)

    def _set_selected_video(self, path: Path) -> None:
        self.video_detail.setText(f"Selected video: {path}")
        self.video_selected.emit(path)

    def _tile_path(self, item: dict[str, Any]) -> Path:
        value = str(item.get("path") or item.get("url") or item.get("relative_path") or self._current_source).strip()
        return Path(value or ".")


class _SelectionDetailsPanel(QFrame):
    lineage_open_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._lineage_rows: list[dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.details_card, details_layout = _build_card("Selection Details")
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(140)
        self.meta_view.setMaximumHeight(220)
        self.meta_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.meta_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        details_layout.addWidget(self.meta_view)
        layout.addWidget(self.details_card)

        self.compatibility_card, compatibility_layout = _build_card("Compatibility")
        self.compatibility_table = QTableWidget(0, 3)
        _set_table_headers(self.compatibility_table, ["Level", "Check", "Detail"])
        _set_readonly_table(self.compatibility_table)
        compatibility_layout.addWidget(self.compatibility_table)
        self.compatibility_card.hide()
        layout.addWidget(self.compatibility_card)

        self.lineage_card, lineage_layout = _build_card("Lineage")
        self.lineage_table = QTableWidget(0, 2)
        _set_table_headers(self.lineage_table, ["Relation", "Target"])
        _set_readonly_table(self.lineage_table)
        lineage_layout.addWidget(self.lineage_table)
        lineage_actions = QHBoxLayout()
        open_lineage_button = QPushButton("Open Linked Target")
        open_lineage_button.clicked.connect(self.lineage_open_requested.emit)
        lineage_actions.addWidget(open_lineage_button)
        lineage_actions.addStretch(1)
        lineage_layout.addLayout(lineage_actions)
        self.lineage_card.hide()
        layout.addWidget(self.lineage_card)

        self.insights_card, insights_layout = _build_card("Deployment Insights")
        self.insights_table = QTableWidget(0, 4)
        _set_table_headers(self.insights_table, ["Episode", "Result", "Notes", "Tags"])
        _set_readonly_table(self.insights_table)
        insights_layout.addWidget(self.insights_table)
        self.insights_card.hide()
        layout.addWidget(self.insights_card)

    def clear(self) -> None:
        self.meta_view.setPlainText("")
        self.compatibility_table.setRowCount(0)
        self.compatibility_card.hide()
        self.lineage_table.setRowCount(0)
        self.lineage_card.hide()
        self.insights_table.setRowCount(0)
        self.insights_card.hide()
        self._lineage_rows = []

    def set_insights_visible(self, visible: bool) -> None:
        self.insights_card.setVisible(bool(visible))

    def selected_lineage_target(self) -> str:
        row = self.lineage_table.currentRow()
        if row < 0 or row >= len(self._lineage_rows):
            return ""
        return str(self._lineage_rows[row].get("target", "")).strip()

    def apply_payload(self, payload: dict[str, Any], *, show_insights: bool) -> str:
        text = _json_text(payload.get("meta_payload", {}))
        self.meta_view.setPlainText(text)

        compatibility_rows = [row for row in payload.get("compatibility_rows", []) if isinstance(row, dict)]
        self.compatibility_table.setRowCount(len(compatibility_rows))
        self.compatibility_card.setVisible(bool(compatibility_rows))
        for row_index, row in enumerate(compatibility_rows):
            self.compatibility_table.setItem(row_index, 0, QTableWidgetItem(str(row.get("level", ""))))
            self.compatibility_table.setItem(row_index, 1, QTableWidgetItem(str(row.get("name", ""))))
            self.compatibility_table.setItem(row_index, 2, QTableWidgetItem(str(row.get("detail", ""))))

        self._lineage_rows = [row for row in payload.get("lineage_rows", []) if isinstance(row, dict)]
        self.lineage_table.setRowCount(len(self._lineage_rows))
        self.lineage_card.setVisible(bool(self._lineage_rows))
        for row_index, row in enumerate(self._lineage_rows):
            self.lineage_table.setItem(row_index, 0, QTableWidgetItem(str(row.get("relation", ""))))
            self.lineage_table.setItem(row_index, 1, QTableWidgetItem(str(row.get("label", ""))))

        self.insights_table.setRowCount(0)
        self.set_insights_visible(show_insights)
        if show_insights:
            insights_rows = [row for row in payload.get("insights_rows", []) if isinstance(row, tuple) and len(row) == 4]
            self.insights_table.setRowCount(len(insights_rows))
            for row_index, row in enumerate(insights_rows):
                for col_index, value in enumerate(row):
                    self.insights_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
            text += "\n\n" + str(payload.get("insights_header", "Deployment Insights"))
            for row in insights_rows:
                text += f"\n- Episode {row[0]} | {row[1]} | {row[2]} | {row[3]}"
        return text


def _visualizer_refresh_dataset_episodes(page: "QtVisualizerPage") -> None:
    repo_id = page._dataset_repo_id_for_tools()
    if not repo_id:
        page.dataset_tools_card.clear_episode_rows()
        page.dataset_tools_card.set_status("Dataset not found locally. Download it first or check the dataset path.")
        return
    page._sync_dataset_inputs(repo_id)
    selected_dataset_path = ""
    source = page._current_source()
    if isinstance(source, dict) and str(source.get("kind", "")).strip().lower() == "dataset":
        source_repo_id = str(source.get("repo_id") or source.get("name") or "").strip()
        if source_repo_id == repo_id or source_repo_id.split("/", 1)[-1] == repo_id.split("/", 1)[-1]:
            selected_dataset_path = str(source.get("path", "")).strip()
    episode_indices, error = collect_local_dataset_episode_indices(page.config, repo_id, selected_dataset_path=selected_dataset_path or None)
    if error:
        page.dataset_tools_card.clear_episode_rows()
        page.dataset_tools_card.set_status(error)
        return
    page.dataset_tools_card.set_episode_rows(episode_indices)
    page.dataset_tools_card.set_status(f"Loaded {len(episode_indices)} episode(s) from {repo_id}.")


def _visualizer_refresh_sources(page: "QtVisualizerPage") -> None:
    preferred_identity = page._preferred_source_identity()
    sources, error_text, source_kind = _collect_sources_for_refresh(page.config, page.source_browser_panel.snapshot())
    page.source_browser_panel.set_sources(list(sources))
    page.selection_details_panel.clear()
    page.video_gallery_panel.load_source(Path("."))
    if sources:
        page._set_output(title="Sources Loaded", text=f"Loaded {len(sources)} {source_kind}.", log_message=f"Visualizer refreshed {len(sources)} {source_kind}.")
        if not page._restore_source_selection(preferred_identity):
            page.source_table.selectRow(0)
        return
    page._set_output(title="No Sources", text=error_text or f"No {source_kind} were found.", log_message="Visualizer refresh returned no sources.")
    page._persist_visualizer_state()


def _visualizer_handle_source_selected(page: "QtVisualizerPage") -> None:
    source = page._current_source()
    if source is None:
        page.selection_details_panel.clear()
        page._render_video_gallery([], empty_message="Select a source to display discovered videos.")
        page._persist_visualizer_state()
        return
    repo_id = page._selected_dataset_repo_id()
    if repo_id:
        page._sync_dataset_inputs(repo_id)
    page.video_gallery_panel.load_source(Path(str(source.get("path") or source.get("repo_id") or source.get("name") or ".")))
    payload = _build_selection_payload(source, config=page.config)
    text = page.selection_details_panel.apply_payload(payload, show_insights=page._active_source() == "deployments" and bool(payload.get("insights_visible")))
    page._set_output(title="Selection Details", text=text, log_message=None)
    page._render_video_gallery(list(payload.get("videos", [])), empty_message="No videos found for the selected source.")
    page._persist_visualizer_state()


def _visualizer_run_command(page: "QtVisualizerPage", *, cmd: list[str], heading: str, run_mode: str, artifact_context: dict[str, Any] | None, start_log: str, unavailable_title: str, unavailable_log: str, complete_callback: Callable[[int, bool], None] | None) -> None:
    page._show_output_card()
    page.output.setPlainText(f"{heading}\n\n{format_command_for_dialog(cmd)}\n\nStreaming output will appear below.")
    page.status_label.setText(heading)
    ok, message = page._run_controller.run_process_async(cmd=cmd, cwd=get_lerobot_dir(page.config), hooks=page._build_hooks(), complete_callback=complete_callback, run_mode=run_mode, preflight_checks=None, artifact_context=artifact_context)
    if not ok:
        page._set_output(title=unavailable_title, text=message or "Unable to start command.", log_message=unavailable_log)
        return
    page._append_log(start_log)


def _visualizer_open_dataset_visualization(page: "QtVisualizerPage") -> None:
    repo_id = page._dataset_repo_id_for_visualization()
    if not repo_id:
        page._show_output_card()
        page._set_output(title="Dataset Required", text="Enter a dataset repo id before opening the Rerun viewer.", log_message="Visualizer dataset visualization launch failed validation.")
        return
    page._sync_dataset_inputs(repo_id)
    episode_index = page.dataset_visualization_card.episode_index()
    cmd = build_visualize_dataset_command(page.config, repo_id, episode_index)

    def after_visualize(return_code: int, was_canceled: bool) -> None:
        if was_canceled:
            page._set_running(False, "Visualization canceled.", False)
            page._append_output_and_log("Dataset visualization canceled.")
        elif return_code != 0:
            page._set_running(False, "Visualization failed.", True)
            page._append_output_and_log(f"Dataset visualization exited with code {return_code}.")
        else:
            page._set_running(False, "Visualization closed.", False)
            page._append_output_and_log("Dataset visualization closed.")

    _visualizer_run_command(page, cmd=cmd, heading="Opening dataset visualization...", run_mode="visualize", artifact_context={"dataset_repo_id": repo_id}, start_log=f"Visualizer dataset visualization launch starting for {repo_id} episode {episode_index}.", unavailable_title="Visualization Unavailable", unavailable_log="Visualizer dataset visualization launch was rejected.", complete_callback=after_visualize)


def _visualizer_replay_dataset_episode(page: "QtVisualizerPage") -> None:
    source = page._current_source()
    dataset_path = str(source.get("path", "")).strip() if isinstance(source, dict) and str(source.get("kind", "")).strip().lower() == "dataset" else ""
    selection = page._dataset_operations.build_replay_selection(config=page.config, dataset_repo_id=page._dataset_repo_id_for_visualization(), dataset_path=dataset_path, selected_episode=page.dataset_visualization_card.episode_index())
    if isinstance(selection, DatasetOperationError):
        page._show_dataset_operation_error(selection)
        return
    selected_episode = ask_replay_episode_dialog(parent=page._dialog_parent(), title="Select Replay Episode", repo_id=selection.repo_id, choices=selection.episode_choices, selected_value=selection.selected_value, helper_text=selection.helper_text)
    if selected_episode is None:
        return
    plan = page._dataset_operations.replay_dataset_episode(config=page.config, dataset_repo_id=selection.repo_id, episode_raw=selected_episode, dataset_path=selection.dataset_path)
    if isinstance(plan, DatasetOperationError):
        page._show_dataset_operation_error(plan)
        return
    editable_cmd = ask_editable_command_dialog(parent=page._dialog_parent(), title="Confirm Replay Command", command_argv=plan.command_argv, intro_text="Review or edit the replay command below.\nThe exact command text here will be executed and saved to run history.", confirm_label="Run Replay", cancel_label="Cancel")
    if editable_cmd is None:
        return
    if editable_cmd != plan.command_argv:
        page._append_log("Visualizer replay is using an edited command from the command editor.")
    if not ask_text_dialog(parent=page._dialog_parent(), title="Replay Preflight Review", text=plan.preflight_text, confirm_label="Confirm", cancel_label="Cancel", wrap_mode="char"):
        page._append_log("Visualizer replay canceled after preflight review.")
        return

    def after_replay(return_code: int, was_canceled: bool) -> None:
        if was_canceled:
            page._set_running(False, "Replay canceled.", False)
            page._append_output_and_log("Replay canceled.")
        elif return_code != 0:
            page._set_running(False, "Replay failed.", True)
            page._append_output_and_log(f"Replay failed with exit code {return_code}.")
        else:
            page._set_running(False, "Replay completed.", False)
            page._append_output_and_log(f"Replay completed for {plan.request.dataset_repo_id} episode {plan.request.episode_index}.")

    _visualizer_run_command(page, cmd=editable_cmd, heading="Replaying dataset episode on hardware...", run_mode="replay", artifact_context=plan.artifact_context, start_log=plan.start_log, unavailable_title="Replay Unavailable", unavailable_log="Visualizer replay launch was rejected.", complete_callback=after_replay)


def _visualizer_run_dataset_edit_operation(page: "QtVisualizerPage", *, operation_name: str, confirm_title: str, confirm_text: str, command_builder: Callable[[dict[str, Any], str, list[int]], list[str]]) -> None:
    plan = page._dataset_operations._run_dataset_edit_operation(config=page.config, repo_id=page._dataset_repo_id_for_tools(), selected_indices=page.dataset_tools_card.selected_episode_indices(), operation_name=operation_name, command_builder=command_builder)
    if isinstance(plan, DatasetOperationError):
        page._show_dataset_operation_error(plan)
        return
    if not ask_text_dialog(parent=page._dialog_parent(), title=confirm_title, text=confirm_text, confirm_label="Confirm", cancel_label="Cancel", wrap_mode="word"):
        return
    editable_cmd = ask_editable_command_dialog(parent=page._dialog_parent(), title=f"Confirm {operation_name} Command", command_argv=plan.command_argv, intro_text="Review or edit the dataset edit command below.\nThe exact command text here will be executed and saved to run history.", confirm_label=operation_name, cancel_label="Cancel")
    if editable_cmd is None:
        return
    if editable_cmd != plan.command_argv:
        page._append_log(f"Running edited {operation_name.lower()} command from command editor.")
    effective_repo_id = page._dataset_operations._command_dataset_repo_id(editable_cmd, plan.repo_id)
    page._sync_dataset_inputs(effective_repo_id)

    def after_edit(return_code: int, was_canceled: bool) -> None:
        if was_canceled:
            page._set_running(False, f"{operation_name} canceled.", False)
            page._append_output_and_log(f"{operation_name} canceled.")
        elif return_code != 0:
            page._set_running(False, f"{operation_name} failed.", True)
            page._append_output_and_log(f"{operation_name} failed with exit code {return_code}.")
        else:
            page._set_running(False, f"{operation_name} completed.", False)
            page._append_output_and_log(f"{operation_name} completed for {effective_repo_id}.")
        page.refresh_dataset_episodes()

    _visualizer_run_command(page, cmd=editable_cmd, heading=f"Running {operation_name.lower()}...", run_mode="dataset_edit", artifact_context={"dataset_repo_id": effective_repo_id}, start_log=f"Visualizer {operation_name.lower()} starting for {effective_repo_id}.", unavailable_title=f"{operation_name} Unavailable", unavailable_log=f"Visualizer {operation_name.lower()} launch was rejected.", complete_callback=after_edit)


def _visualizer_merge_datasets(page: "QtVisualizerPage") -> None:
    plan = page._dataset_operations.merge_datasets(config=page.config, output_repo_id=page.dataset_tools_card.merge_output_repo_id(), source_repo_ids=page.dataset_tools_card.merge_source_repo_ids())
    if isinstance(plan, DatasetOperationError):
        page._show_dataset_operation_error(plan)
        return
    if not ask_text_dialog(parent=page._dialog_parent(), title="Merge Datasets", text=f"Merge {len(plan.source_repo_ids)} datasets into {plan.output_repo_id}?\nSources: [{', '.join(plan.source_repo_ids)}]", confirm_label="Confirm", cancel_label="Cancel", wrap_mode="word"):
        return
    editable_cmd = ask_editable_command_dialog(parent=page._dialog_parent(), title="Confirm Merge Datasets Command", command_argv=plan.command_argv, intro_text="Review or edit the dataset merge command below.\nThe exact command text here will be executed and saved to run history.", confirm_label="Merge Datasets", cancel_label="Cancel")
    if editable_cmd is None:
        return
    if editable_cmd != plan.command_argv:
        page._append_log("Running edited merge datasets command from command editor.")
    effective_repo_id = page._dataset_operations._command_dataset_repo_id(editable_cmd, plan.output_repo_id)
    page.dataset_tools_card.set_merge_output_repo_id(effective_repo_id)

    def after_merge(return_code: int, was_canceled: bool) -> None:
        if was_canceled:
            page._set_running(False, "Merge canceled.", False)
            page._append_output_and_log("Merge datasets canceled.")
        elif return_code != 0:
            page._set_running(False, "Merge failed.", True)
            page._append_output_and_log(f"Merge datasets failed with exit code {return_code}.")
        else:
            page._sync_dataset_inputs(effective_repo_id)
            page._set_running(False, "Merge completed.", False)
            page._append_output_and_log(f"Merge datasets completed for {effective_repo_id}.")
            page.refresh_sources()
        page.refresh_dataset_episodes()

    _visualizer_run_command(page, cmd=editable_cmd, heading="Running merge datasets...", run_mode="dataset_edit", artifact_context={"dataset_repo_id": effective_repo_id}, start_log=f"Visualizer merge datasets starting for {effective_repo_id}.", unavailable_title="Merge Datasets Unavailable", unavailable_log="Visualizer merge-datasets launch was rejected.", complete_callback=after_merge)


def _visualizer_open_selected_source(page: "QtVisualizerPage") -> None:
    source = page._current_source()
    if source is None:
        page._set_output(title="No Selection", text="Select a source first.", log_message="Visualizer source open skipped with no selection.")
        return
    target = source.get("path") or (f"https://huggingface.co/datasets/{source['repo_id']}" if source.get("repo_id") and str(source.get("kind")) == "dataset" else f"https://huggingface.co/{source['repo_id']}" if source.get("repo_id") else "")
    ok, message = _open_path(str(target or ""))
    page._set_output(title="Open Source" if ok else "Open Failed", text=str(target or message), log_message="Visualizer opened source." if ok else "Visualizer source open failed.")


def _visualizer_sync_selected_source(page: "QtVisualizerPage") -> None:
    source = page._current_source()
    if source is None:
        page._set_output(title="No Selection", text="Select a source first.", log_message="Visualizer sync skipped with no selection.")
        return
    if str(source.get("scope", "local")) != "huggingface":
        page._set_output(title="Sync Unavailable", text="Sync is only available for Hugging Face sources.", log_message="Visualizer sync skipped for local source.")
        return
    repo_id = str(source.get("repo_id", "")).strip()
    result, error_text = sync_hf_asset(page.config, repo_id=repo_id, kind=str(source.get("kind", "")).strip() or "dataset")
    if error_text:
        page._set_output(title="Sync Failed", text=error_text, log_message="Visualizer HF sync failed.")
        return
    page._set_output(title="Sync Complete", text=_json_text(result or {}), log_message=f"Visualizer synced {repo_id}.")
    page.refresh_sources()


def _visualizer_open_selected_lineage_target(page: "QtVisualizerPage") -> None:
    target = page.selection_details_panel.selected_lineage_target()
    if not target:
        page._set_output(title="No Lineage Selection", text="Select a lineage row first.", log_message="Visualizer lineage open skipped with no selection.")
        return
    ok, message = _open_path(target)
    page._set_output(title="Open Linked Target" if ok else "Open Failed", text=target or message, log_message="Visualizer opened lineage target." if ok else "Visualizer lineage target open failed.")


class QtVisualizerPage(_PageWithOutput):
    def __init__(self, *, config: dict[str, Any], append_log: Callable[[str], None], run_controller: ManagedRunController) -> None:
        super().__init__(title="Visualizer", subtitle="Browse local deployment runs, datasets, models, and discovered video assets.", append_log=append_log)
        self.config = config
        self._run_controller = run_controller
        self._dataset_operations = DatasetOperationService()
        self._action_buttons: list[QPushButton] = []
        self._cancel_button: QPushButton | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.source_browser_panel = _SourceBrowserPanel(config=self.config)
        self.video_gallery_panel = _VideoGalleryPanel()
        self.selection_details_panel = _SelectionDetailsPanel()
        for widget in (self.source_browser_panel, self.video_gallery_panel):
            self.content_layout.addWidget(widget)
        self._build_visualizer_tool_cards()
        self.content_layout.addWidget(self.selection_details_panel)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.addStretch(1)

        self._alias_panel_widgets()
        self.source_browser_panel.refresh_requested.connect(self.refresh_sources)
        self.source_browser_panel.open_requested.connect(self.open_selected_source)
        self.source_browser_panel.sync_requested.connect(self.sync_selected_source)
        self.source_browser_panel.source_changed.connect(self._on_source_selected)
        self.selection_details_panel.lineage_open_requested.connect(self.open_selected_lineage_target)
        self.refresh_sources()

    def _alias_panel_widgets(self) -> None:
        for name in ("source_combo", "root_input", "browse_root_button", "hf_owner_input", "hf_query_input", "hf_task_input", "hf_tag_input", "refresh_button", "source_table"):
            setattr(self, name, getattr(self.source_browser_panel, name))
        for name in ("video_gallery_card", "video_status", "video_detail", "video_grid_host", "video_grid", "video_scroll"):
            setattr(self, name, getattr(self.video_gallery_panel, name))
        for name in ("meta_view", "compatibility_card", "compatibility_table", "lineage_card", "lineage_table", "insights_card", "insights_table"):
            setattr(self, name, getattr(self.selection_details_panel, name))
        self._video_tiles = self.video_gallery_panel._video_tiles

    def _register_action_button(self, button: QPushButton, *, is_cancel: bool = False) -> None:
        self._action_buttons.append(button)
        if is_cancel:
            self._cancel_button = button
            button.setEnabled(False)

    def _build_visualizer_tool_cards(self) -> None:
        default_repo_id = self._default_dataset_repo_id()
        self.dataset_visualization_card = _DatasetVisualizationCard(default_repo_id=default_repo_id, on_open=self.open_dataset_visualization, on_replay=self.replay_dataset_episode, on_cancel=self._cancel_run, register_action_button=self._register_action_button, register_cancel_button=lambda button: self._register_action_button(button, is_cancel=True))
        self.dataset_tools_card = _DatasetToolsCard(default_repo_id=default_repo_id, on_refresh=self.refresh_dataset_episodes, on_select_all=self.select_all_dataset_episodes, on_select_none=self.select_no_dataset_episodes, on_delete_selected=self.delete_selected_episodes, on_keep_selected=self.keep_selected_episodes, on_merge=self.merge_datasets, register_action_button=self._register_action_button)
        self.content_layout.addWidget(self.dataset_visualization_card)
        self.content_layout.addWidget(self.dataset_tools_card)
        self.visualize_dataset_input = self.dataset_visualization_card.dataset_input
        self.visualize_episode_input = self.dataset_visualization_card.episode_input
        self.visualize_open_button = self.dataset_visualization_card.open_button
        self.replay_episode_button = self.dataset_visualization_card.replay_button
        self.visualize_cancel_button = self.dataset_visualization_card.cancel_button
        self.dataset_tools_input = self.dataset_tools_card.dataset_input
        self.dataset_tools_refresh_button = self.dataset_tools_card.refresh_button
        self.select_all_episodes_button = self.dataset_tools_card.select_all_button
        self.select_no_episodes_button = self.dataset_tools_card.select_none_button
        self.dataset_episodes_table = self.dataset_tools_card.episodes_table
        self.dataset_tools_status = self.dataset_tools_card.status_label
        self.delete_selected_episodes_button = self.dataset_tools_card.delete_selected_button
        self.keep_selected_episodes_button = self.dataset_tools_card.keep_selected_button
        self.merge_output_dataset_input = self.dataset_tools_card.merge_output_dataset_input
        self.merge_source_datasets_input = self.dataset_tools_card.merge_source_datasets_input
        self.merge_datasets_button = self.dataset_tools_card.merge_button

    def _dialog_parent(self) -> QWidget | None:
        parent = self.window()
        return parent if isinstance(parent, QWidget) else None

    def _show_output_card(self) -> None:
        self.output_card.show()

    def _append_output_line(self, line: str) -> None:
        self._show_output_card()
        super()._append_output_line(line)

    def _set_running(self, active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        for button in self._action_buttons:
            button.setEnabled(active if button is self._cancel_button else not active)
        self.status_label.setText(status_text or ("Running command..." if active else "Command failed." if is_error else "Ready."))

    def _build_hooks(self) -> RunUiHooks:
        return RunUiHooks(set_running=self._set_running, append_output_line=self._append_output_line)

    def _cancel_run(self) -> None:
        ok, message = self._run_controller.cancel_active_run()
        if not ok:
            self._show_output_card()
            super()._set_output(title="Cancel Unavailable", text=message, log_message="Visualizer cancel request was rejected.")

    def browse_root(self) -> None:
        self.source_browser_panel.browse_root()

    def _active_source(self) -> str:
        return self.source_browser_panel.active_source()

    def _persisted_source_kind(self) -> str:
        return self.source_browser_panel.persisted_source_kind()

    def _default_root_for_source(self, source: str) -> str:
        return self.source_browser_panel.default_root_for_source(source)

    def _root_text_for_source(self, source: str) -> str:
        return self.source_browser_panel.root_text_for_source(source)

    def _restore_persisted_visualizer_state(self) -> None:
        self.source_browser_panel.restore_persisted_state()

    def _persist_visualizer_root(self, source: str, root_text: str) -> None:
        self.source_browser_panel.persist_visualizer_root(source, root_text)

    def _source_identity(self, source: dict[str, Any] | None) -> tuple[str, str, str]:
        return self.source_browser_panel.source_identity(source)

    def _preferred_source_identity(self) -> tuple[str, str, str]:
        return self.source_browser_panel.preferred_source_identity()

    def _restore_source_selection(self, preferred_identity: tuple[str, str, str]) -> bool:
        return self.source_browser_panel.restore_source_selection(preferred_identity)

    def _persist_visualizer_state(self) -> None:
        self.source_browser_panel.persist_state()

    def _sync_root_placeholder(self) -> None:
        self.source_browser_panel.sync_root_placeholder()

    def _current_source(self) -> dict[str, Any] | None:
        return self.source_browser_panel.current_source()

    def _clear_video_tiles(self) -> None:
        self.video_gallery_panel.clear_video_tiles()
        self._video_tiles = self.video_gallery_panel._video_tiles

    def _render_video_gallery(self, videos: list[dict[str, Any]], *, empty_message: str) -> None:
        self.video_gallery_panel.set_videos(videos, empty_message=empty_message)
        self._video_tiles = self.video_gallery_panel._video_tiles

    def _refresh_scroll_geometry(self) -> None:
        self.video_gallery_panel.refresh_scroll_geometry()

    def _refresh_parent_scroll_area(self) -> None:
        self.video_gallery_panel.refresh_parent_scroll_area()

    def _default_dataset_repo_id(self) -> str:
        return self._selected_dataset_repo_id() or str(self.config.get("last_dataset_repo_id", "")).strip() or str(self.config.get("last_train_dataset", "")).strip()

    def _selected_dataset_repo_id(self) -> str:
        source = self._current_source()
        if not isinstance(source, dict):
            return ""
        if str(source.get("kind", "")).strip().lower() == "dataset":
            return str(source.get("repo_id") or source.get("name") or "").strip()
        metadata = source.get("metadata")
        return str(metadata.get("dataset_repo_id", "")).strip() if isinstance(metadata, dict) else ""

    def _sync_dataset_inputs(self, repo_id: str) -> None:
        self.dataset_visualization_card.set_repo_id(repo_id)
        self.dataset_tools_card.set_repo_id(repo_id)

    def _dataset_repo_id_for_tools(self) -> str:
        return self.dataset_tools_card.repo_id(self._default_dataset_repo_id())

    def _dataset_repo_id_for_visualization(self) -> str:
        return self.dataset_visualization_card.repo_id(self._default_dataset_repo_id())

    def _show_dataset_operation_error(self, result: DatasetOperationError) -> None:
        self._show_output_card()
        super()._set_output(title=result.title, text=result.text, log_message=result.log_message)

    def select_all_dataset_episodes(self) -> None:
        self.dataset_tools_card.select_all_episodes()

    def select_no_dataset_episodes(self) -> None:
        self.dataset_tools_card.select_no_episodes()

    def refresh_dataset_episodes(self) -> None:
        _visualizer_refresh_dataset_episodes(self)

    def refresh_sources(self) -> None:
        _visualizer_refresh_sources(self)

    def _on_source_selected(self, _selected_path: Path) -> None:
        _visualizer_handle_source_selected(self)

    def _run_command(self, *, cmd: list[str], heading: str, run_mode: str, artifact_context: dict[str, Any] | None, start_log: str, unavailable_title: str, unavailable_log: str, complete_callback: Callable[[int, bool], None] | None) -> None:
        _visualizer_run_command(self, cmd=cmd, heading=heading, run_mode=run_mode, artifact_context=artifact_context, start_log=start_log, unavailable_title=unavailable_title, unavailable_log=unavailable_log, complete_callback=complete_callback)

    def open_dataset_visualization(self) -> None:
        _visualizer_open_dataset_visualization(self)

    def replay_dataset_episode(self) -> None:
        _visualizer_replay_dataset_episode(self)

    def _confirm_dataset_operation(self, *, title: str, text: str) -> bool:
        return ask_text_dialog(parent=self._dialog_parent(), title=title, text=text, confirm_label="Confirm", cancel_label="Cancel", wrap_mode="word")

    def _run_dataset_edit_operation(self, *, operation_name: str, confirm_title: str, confirm_text: str, command_builder: Callable[[dict[str, Any], str, list[int]], list[str]]) -> None:
        _visualizer_run_dataset_edit_operation(self, operation_name=operation_name, confirm_title=confirm_title, confirm_text=confirm_text, command_builder=command_builder)

    def delete_selected_episodes(self) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        selected_indices = self.dataset_tools_card.selected_episode_indices()
        episodes_text = ", ".join(str(index) for index in selected_indices)
        self._run_dataset_edit_operation(operation_name="Delete Episodes", confirm_title="Delete Episodes", confirm_text=f"Delete {len(selected_indices)} episodes from {repo_id}?\nEpisodes: [{episodes_text}]\n\nThis cannot be undone.", command_builder=build_delete_episodes_command)

    def keep_selected_episodes(self) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        selected_indices = self.dataset_tools_card.selected_episode_indices()
        remaining = max(self.dataset_tools_card.episode_row_count() - len(selected_indices), 0)
        self._run_dataset_edit_operation(operation_name="Keep Episodes", confirm_title="Keep Selected Episodes", confirm_text=f"Keep only {len(selected_indices)} episodes and delete {remaining} others from {repo_id}?", command_builder=build_keep_episodes_command)

    def merge_datasets(self) -> None:
        _visualizer_merge_datasets(self)

    def open_selected_source(self) -> None:
        _visualizer_open_selected_source(self)

    def sync_selected_source(self) -> None:
        _visualizer_sync_selected_source(self)

    def open_selected_lineage_target(self) -> None:
        _visualizer_open_selected_lineage_target(self)

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[misc]
        self.video_gallery_panel.start_video_tiles()

    def hideEvent(self, event: object) -> None:
        self.video_gallery_panel.stop_video_tiles()
        super().hideEvent(event)  # type: ignore[misc]

    def refresh_from_config(self) -> None:
        self._restore_persisted_visualizer_state()
        self.refresh_sources()
