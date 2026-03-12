from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
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
    QWidget,
)

try:
    import cv2 as _cv2_module  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal installs
    _cv2_module = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

from .command_overrides import get_flag_value
from .command_text import format_command_for_dialog
from .config_store import get_deploy_data_dir, get_lerobot_dir, save_config
from .dataset_tools import (
    build_delete_episodes_command,
    build_keep_episodes_command,
    build_merge_datasets_command,
    collect_local_dataset_episode_indices,
)
from .gui_qt_visualizer_cards import _DatasetToolsCard, _DatasetVisualizationCard
from .gui_qt_dialogs import ask_editable_command_dialog, ask_text_dialog
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
    _PageWithOutput,
    _VideoGalleryTile,
    _build_card,
    _json_text,
    _set_readonly_table,
    _set_table_headers,
)

class QtVisualizerPage(_PageWithOutput):
    _ROOT_STATE_KEYS = {
        "deployments": "ui_visualizer_deploy_root",
        "datasets": "ui_visualizer_dataset_root",
        "models": "ui_visualizer_model_root",
    }

    def __init__(
        self,
        *,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        run_controller: ManagedRunController,
    ) -> None:
        super().__init__(
            title="Visualizer",
            subtitle="Browse local deployment runs, datasets, models, and discovered video assets.",
            append_log=append_log,
        )
        self.config = config
        self._run_controller = run_controller
        self._sources: list[dict[str, Any]] = []
        self._video_tiles: list[_VideoGalleryTile] = []
        self._action_buttons: list[QPushButton] = []
        self._cancel_button: QPushButton | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._current_source_kind = self._persisted_source_kind()

        self.content_layout.addWidget(self._build_controls_card())
        self.content_layout.addWidget(self._build_video_gallery_card())
        self._build_visualizer_tool_cards()
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

    def _register_action_button(self, button: QPushButton, *, is_cancel: bool = False) -> None:
        self._action_buttons.append(button)
        if is_cancel:
            self._cancel_button = button
            button.setEnabled(False)

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
            if button is self._cancel_button:
                button.setEnabled(active)
            else:
                button.setEnabled(not active)
        if active:
            self.status_label.setText(status_text or "Running command...")
        elif is_error:
            self.status_label.setText(status_text or "Command failed.")
        else:
            self.status_label.setText(status_text or "Ready.")

    def _build_hooks(self) -> RunUiHooks:
        return RunUiHooks(
            set_running=self._set_running,
            append_output_line=self._append_output_line,
        )

    def _cancel_run(self) -> None:
        ok, message = self._run_controller.cancel_active_run()
        if not ok:
            self._show_output_card()
            super()._set_output(title="Cancel Unavailable", text=message, log_message="Visualizer cancel request was rejected.")

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

    def _build_visualizer_tool_cards(self) -> None:
        default_repo_id = self._default_dataset_repo_id()
        self.dataset_visualization_card = _DatasetVisualizationCard(
            default_repo_id=default_repo_id,
            on_open=self.open_dataset_visualization,
            on_cancel=self._cancel_run,
            register_action_button=self._register_action_button,
            register_cancel_button=lambda button: self._register_action_button(button, is_cancel=True),
        )
        self.content_layout.addWidget(self.dataset_visualization_card)

        self.dataset_tools_card = _DatasetToolsCard(
            default_repo_id=default_repo_id,
            on_refresh=self.refresh_dataset_episodes,
            on_select_all=self.select_all_dataset_episodes,
            on_select_none=self.select_no_dataset_episodes,
            on_delete_selected=self.delete_selected_episodes,
            on_keep_selected=self.keep_selected_episodes,
            on_merge=self.merge_datasets,
            register_action_button=self._register_action_button,
        )
        self.content_layout.addWidget(self.dataset_tools_card)

        # Preserve existing attribute access used by tests and page logic.
        self.visualize_dataset_input = self.dataset_visualization_card.dataset_input
        self.visualize_episode_input = self.dataset_visualization_card.episode_input
        self.visualize_open_button = self.dataset_visualization_card.open_button
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

    def _default_dataset_repo_id(self) -> str:
        selected = self._selected_dataset_repo_id()
        if selected:
            return selected
        return str(self.config.get("last_dataset_repo_id", "")).strip() or str(self.config.get("last_train_dataset", "")).strip()

    def _selected_dataset_repo_id(self) -> str:
        source = self._current_source()
        if not isinstance(source, dict):
            return ""
        kind = str(source.get("kind", "")).strip().lower()
        if kind == "dataset":
            return str(source.get("repo_id") or source.get("name") or "").strip()
        metadata = source.get("metadata")
        if isinstance(metadata, dict):
            return str(metadata.get("dataset_repo_id", "")).strip()
        return ""

    def _sync_dataset_inputs(self, repo_id: str) -> None:
        self.dataset_visualization_card.set_repo_id(repo_id)
        self.dataset_tools_card.set_repo_id(repo_id)

    def _dataset_repo_id_for_tools(self) -> str:
        return self.dataset_tools_card.repo_id(self._default_dataset_repo_id())

    def _dataset_repo_id_for_visualization(self) -> str:
        return self.dataset_visualization_card.repo_id(self._default_dataset_repo_id())

    def _set_dataset_tools_status(self, text: str) -> None:
        self.dataset_tools_card.set_status(text)

    def _command_dataset_repo_id(self, argv: list[str], fallback: str = "") -> str:
        return (
            get_flag_value(argv, "repo_id")
            or get_flag_value(argv, "dataset.repo_id")
            or str(fallback).strip()
        )

    def _set_dataset_episode_rows(self, episode_indices: list[int]) -> None:
        self.dataset_tools_card.set_episode_rows(episode_indices)

    def _selected_episode_indices(self) -> list[int]:
        return self.dataset_tools_card.selected_episode_indices()

    def select_all_dataset_episodes(self) -> None:
        self.dataset_tools_card.select_all_episodes()

    def select_no_dataset_episodes(self) -> None:
        self.dataset_tools_card.select_no_episodes()

    def refresh_dataset_episodes(self) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        if not repo_id:
            self.dataset_tools_card.clear_episode_rows()
            self._set_dataset_tools_status("Dataset not found locally. Download it first or check the dataset path.")
            return
        self._sync_dataset_inputs(repo_id)
        selected_dataset_path = ""
        source = self._current_source()
        if isinstance(source, dict) and str(source.get("kind", "")).strip().lower() == "dataset":
            source_repo_id = str(source.get("repo_id") or source.get("name") or "").strip()
            source_repo_name = source_repo_id.split("/", 1)[-1]
            requested_repo_name = repo_id.split("/", 1)[-1]
            if source_repo_id == repo_id or (source_repo_name and source_repo_name == requested_repo_name):
                selected_dataset_path = str(source.get("path", "")).strip()
        episode_indices, error = collect_local_dataset_episode_indices(
            self.config,
            repo_id,
            selected_dataset_path=selected_dataset_path or None,
        )
        if error:
            self.dataset_tools_card.clear_episode_rows()
            self._set_dataset_tools_status(error)
            return
        self._set_dataset_episode_rows(episode_indices)
        self._set_dataset_tools_status(f"Loaded {len(episode_indices)} episode(s) from {repo_id}.")

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
        if not hasattr(self, "source_table"):
            return None
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
        selected_repo_id = self._selected_dataset_repo_id()
        if selected_repo_id:
            self._sync_dataset_inputs(selected_repo_id)
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

    def _run_command(
        self,
        *,
        cmd: list[str],
        heading: str,
        run_mode: str,
        artifact_context: dict[str, Any] | None,
        start_log: str,
        unavailable_title: str,
        unavailable_log: str,
        complete_callback: Callable[[int, bool], None] | None,
    ) -> None:
        self._show_output_card()
        self.output.setPlainText(f"{heading}\n\n{format_command_for_dialog(cmd)}\n\nStreaming output will appear below.")
        self.status_label.setText(heading)
        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=get_lerobot_dir(self.config),
            hooks=self._build_hooks(),
            complete_callback=complete_callback,
            run_mode=run_mode,
            preflight_checks=None,
            artifact_context=artifact_context,
        )
        if not ok:
            super()._set_output(
                title=unavailable_title,
                text=message or "Unable to start command.",
                log_message=unavailable_log,
            )
            return
        self._append_log(start_log)

    def open_dataset_visualization(self) -> None:
        repo_id = self._dataset_repo_id_for_visualization()
        if not repo_id:
            self._show_output_card()
            super()._set_output(
                title="Dataset Required",
                text="Enter a dataset repo id before opening the Rerun viewer.",
                log_message="Visualizer dataset visualization launch failed validation.",
            )
            return
        self._sync_dataset_inputs(repo_id)
        episode_index = self.dataset_visualization_card.episode_index()
        cmd = build_visualize_dataset_command(self.config, repo_id, episode_index)

        def after_visualize(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Visualization canceled.", False)
                self._append_output_and_log("Dataset visualization canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Visualization failed.", True)
                self._append_output_and_log(f"Dataset visualization exited with code {return_code}.")
                return
            self._set_running(False, "Visualization closed.", False)
            self._append_output_and_log("Dataset visualization closed.")

        self._run_command(
            cmd=cmd,
            heading="Opening dataset visualization...",
            run_mode="visualize",
            artifact_context={"dataset_repo_id": repo_id},
            start_log=f"Visualizer dataset visualization launch starting for {repo_id} episode {episode_index}.",
            unavailable_title="Visualization Unavailable",
            unavailable_log="Visualizer dataset visualization launch was rejected.",
            complete_callback=after_visualize,
        )

    def _editable_dataset_operation_command(
        self,
        *,
        title: str,
        intro_text: str,
        confirm_label: str,
        command_argv: list[str],
    ) -> list[str] | None:
        return ask_editable_command_dialog(
            parent=self._dialog_parent(),
            title=title,
            command_argv=command_argv,
            intro_text=intro_text,
            confirm_label=confirm_label,
            cancel_label="Cancel",
        )

    def _confirm_dataset_operation(self, *, title: str, text: str) -> bool:
        return ask_text_dialog(
            parent=self._dialog_parent(),
            title=title,
            text=text,
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="word",
        )

    def _run_dataset_edit_operation(
        self,
        *,
        operation_name: str,
        confirm_title: str,
        confirm_text: str,
        command_builder: Callable[[dict[str, Any], str, list[int]], list[str]],
    ) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        if not repo_id:
            self._show_output_card()
            super()._set_output(
                title="Dataset Required",
                text="Enter a dataset repo id before running dataset tools.",
                log_message="Visualizer dataset tools launch failed validation.",
            )
            return

        selected_indices = self._selected_episode_indices()
        if not selected_indices:
            self._show_output_card()
            super()._set_output(
                title="No Episodes Selected",
                text="Select at least one episode first.",
                log_message="Visualizer dataset tools launch skipped with no selected episodes.",
            )
            return

        if not self._confirm_dataset_operation(title=confirm_title, text=confirm_text):
            return

        cmd = command_builder(self.config, repo_id, selected_indices)
        editable_cmd = self._editable_dataset_operation_command(
            title=f"Confirm {operation_name} Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the dataset edit command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label=operation_name,
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log(f"Running edited {operation_name.lower()} command from command editor.")
        effective_repo_id = self._command_dataset_repo_id(editable_cmd, repo_id)
        self._sync_dataset_inputs(effective_repo_id)

        def after_edit(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, f"{operation_name} canceled.", False)
                self._append_output_and_log(f"{operation_name} canceled.")
                self.refresh_dataset_episodes()
                return
            if return_code != 0:
                self._set_running(False, f"{operation_name} failed.", True)
                self._append_output_and_log(f"{operation_name} failed with exit code {return_code}.")
                self.refresh_dataset_episodes()
                return
            self._set_running(False, f"{operation_name} completed.", False)
            self._append_output_and_log(f"{operation_name} completed for {effective_repo_id}.")
            self.refresh_dataset_episodes()

        self._run_command(
            cmd=editable_cmd,
            heading=f"Running {operation_name.lower()}...",
            run_mode="dataset_edit",
            artifact_context={"dataset_repo_id": effective_repo_id},
            start_log=f"Visualizer {operation_name.lower()} starting for {effective_repo_id}.",
            unavailable_title=f"{operation_name} Unavailable",
            unavailable_log=f"Visualizer {operation_name.lower()} launch was rejected.",
            complete_callback=after_edit,
        )

    def delete_selected_episodes(self) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        selected_indices = self._selected_episode_indices()
        if not repo_id or not selected_indices:
            self._show_output_card()
            super()._set_output(
                title="No Episodes Selected",
                text="Select a dataset and at least one episode first.",
                log_message="Visualizer delete-episodes launch skipped with no selected episodes.",
            )
            return
        episodes_text = ", ".join(str(index) for index in selected_indices)
        self._run_dataset_edit_operation(
            operation_name="Delete Episodes",
            confirm_title="Delete Episodes",
            confirm_text=(
                f"Delete {len(selected_indices)} episodes from {repo_id}?\n"
                f"Episodes: [{episodes_text}]\n\n"
                "This cannot be undone."
            ),
            command_builder=build_delete_episodes_command,
        )

    def keep_selected_episodes(self) -> None:
        repo_id = self._dataset_repo_id_for_tools()
        selected_indices = self._selected_episode_indices()
        if not repo_id or not selected_indices:
            self._show_output_card()
            super()._set_output(
                title="No Episodes Selected",
                text="Select a dataset and at least one episode first.",
                log_message="Visualizer keep-episodes launch skipped with no selected episodes.",
            )
            return
        remaining = max(self.dataset_tools_card.episode_row_count() - len(selected_indices), 0)
        self._run_dataset_edit_operation(
            operation_name="Keep Episodes",
            confirm_title="Keep Selected Episodes",
            confirm_text=f"Keep only {len(selected_indices)} episodes and delete {remaining} others from {repo_id}?",
            command_builder=build_keep_episodes_command,
        )

    def merge_datasets(self) -> None:
        output_repo_id = self.dataset_tools_card.merge_output_repo_id()
        source_repo_ids = self.dataset_tools_card.merge_source_repo_ids()
        if not output_repo_id:
            self._show_output_card()
            super()._set_output(
                title="Output Dataset Required",
                text="Enter the output dataset repo id before merging.",
                log_message="Visualizer merge-datasets launch failed validation.",
            )
            return
        if len(source_repo_ids) < 2:
            self._show_output_card()
            super()._set_output(
                title="Source Datasets Required",
                text="Enter at least two source dataset repo ids to merge.",
                log_message="Visualizer merge-datasets launch failed validation.",
            )
            return

        source_text = ", ".join(source_repo_ids)
        if not self._confirm_dataset_operation(
            title="Merge Datasets",
            text=(
                f"Merge {len(source_repo_ids)} datasets into {output_repo_id}?\n"
                f"Sources: [{source_text}]"
            ),
        ):
            return

        cmd = build_merge_datasets_command(self.config, output_repo_id, source_repo_ids)
        editable_cmd = self._editable_dataset_operation_command(
            title="Confirm Merge Datasets Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the dataset merge command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Merge Datasets",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            self._append_log("Running edited merge datasets command from command editor.")
        effective_output_repo_id = self._command_dataset_repo_id(editable_cmd, output_repo_id)
        self.dataset_tools_card.set_merge_output_repo_id(effective_output_repo_id)

        def after_merge(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                self._set_running(False, "Merge canceled.", False)
                self._append_output_and_log("Merge datasets canceled.")
                return
            if return_code != 0:
                self._set_running(False, "Merge failed.", True)
                self._append_output_and_log(f"Merge datasets failed with exit code {return_code}.")
                return
            self._sync_dataset_inputs(effective_output_repo_id)
            self._set_running(False, "Merge completed.", False)
            self._append_output_and_log(f"Merge datasets completed for {effective_output_repo_id}.")
            self.refresh_sources()
            self.refresh_dataset_episodes()

        self._run_command(
            cmd=editable_cmd,
            heading="Running merge datasets...",
            run_mode="dataset_edit",
            artifact_context={"dataset_repo_id": effective_output_repo_id},
            start_log=f"Visualizer merge datasets starting for {effective_output_repo_id}.",
            unavailable_title="Merge Datasets Unavailable",
            unavailable_log="Visualizer merge-datasets launch was rejected.",
            complete_callback=after_merge,
        )

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
