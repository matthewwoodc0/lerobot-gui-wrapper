from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dataset_tools import parse_dataset_repo_ids
from .gui_qt_page_base import _InputGrid, _build_card, _set_table_headers


class _DatasetVisualizationCard(QWidget):
    def __init__(
        self,
        *,
        default_repo_id: str,
        on_open: Callable[[], None],
        on_cancel: Callable[[], None],
        register_action_button: Callable[[QPushButton], None],
        register_cancel_button: Callable[[QPushButton], None],
    ) -> None:
        super().__init__()
        host_layout = QVBoxLayout(self)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        self.card, layout = _build_card("Dataset Visualization")
        host_layout.addWidget(self.card)

        form = _InputGrid(layout)
        self.dataset_input = QLineEdit(str(default_repo_id).strip())
        self.dataset_input.setPlaceholderText("owner/dataset_name")
        form.add_field("Dataset", self.dataset_input)

        self.episode_input = QSpinBox()
        self.episode_input.setRange(0, 9999)
        self.episode_input.setValue(0)
        form.add_field("Episode", self.episode_input)

        actions = QHBoxLayout()
        self.open_button = QPushButton("Open in Rerun")
        self.open_button.setObjectName("AccentButton")
        self.open_button.clicked.connect(on_open)
        actions.addWidget(self.open_button)
        register_action_button(self.open_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("DangerButton")
        self.cancel_button.clicked.connect(on_cancel)
        actions.addWidget(self.cancel_button)
        register_cancel_button(self.cancel_button)

        actions.addStretch(1)
        layout.addLayout(actions)

        info = QLabel("Opens the Rerun viewer showing camera feeds, actions, and state for the selected episode.")
        info.setWordWrap(True)
        info.setObjectName("MutedLabel")
        layout.addWidget(info)

    def repo_id(self, fallback: str = "") -> str:
        return self.dataset_input.text().strip() or str(fallback).strip()

    def set_repo_id(self, repo_id: str) -> None:
        value = str(repo_id).strip()
        if value:
            self.dataset_input.setText(value)

    def episode_index(self) -> int:
        return int(self.episode_input.value())


class _DatasetToolsCard(QWidget):
    def __init__(
        self,
        *,
        default_repo_id: str,
        on_refresh: Callable[[], None],
        on_select_all: Callable[[], None],
        on_select_none: Callable[[], None],
        on_delete_selected: Callable[[], None],
        on_keep_selected: Callable[[], None],
        on_merge: Callable[[], None],
        register_action_button: Callable[[QPushButton], None],
    ) -> None:
        super().__init__()
        host_layout = QVBoxLayout(self)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        self.card, layout = _build_card("Dataset Tools")
        host_layout.addWidget(self.card)

        top_row = QHBoxLayout()
        dataset_label = QLabel("Dataset")
        dataset_label.setObjectName("FormLabel")
        top_row.addWidget(dataset_label)

        self.dataset_input = QLineEdit(str(default_repo_id).strip())
        self.dataset_input.setPlaceholderText("owner/dataset_name")
        top_row.addWidget(self.dataset_input, 1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(on_refresh)
        top_row.addWidget(self.refresh_button)
        register_action_button(self.refresh_button)
        layout.addLayout(top_row)

        selection_row = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(on_select_all)
        selection_row.addWidget(self.select_all_button)
        register_action_button(self.select_all_button)

        self.select_none_button = QPushButton("Select None")
        self.select_none_button.clicked.connect(on_select_none)
        selection_row.addWidget(self.select_none_button)
        register_action_button(self.select_none_button)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        self.episodes_table = QTableWidget(0, 2)
        self.episodes_table.setObjectName("DatasetEpisodesTable")
        _set_table_headers(self.episodes_table, ["Episode", "Select"])
        self.episodes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.episodes_table.verticalHeader().setVisible(False)
        self.episodes_table.setAlternatingRowColors(True)
        self.episodes_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.episodes_table.setMinimumHeight(180)
        layout.addWidget(self.episodes_table)

        self.status_label = QLabel("Select a dataset and refresh to load episodes.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        self.delete_selected_button = QPushButton("Delete Selected Episodes")
        self.delete_selected_button.setObjectName("DangerButton")
        self.delete_selected_button.clicked.connect(on_delete_selected)
        actions.addWidget(self.delete_selected_button)
        register_action_button(self.delete_selected_button)

        self.keep_selected_button = QPushButton("Keep Selected Only")
        self.keep_selected_button.setObjectName("AccentButton")
        self.keep_selected_button.clicked.connect(on_keep_selected)
        actions.addWidget(self.keep_selected_button)
        register_action_button(self.keep_selected_button)

        actions.addStretch(1)
        layout.addLayout(actions)

        merge_header = QLabel("Merge Datasets")
        merge_header.setObjectName("FormLabel")
        layout.addWidget(merge_header)

        merge_info = QLabel("Create a new dataset by merging multiple local datasets from the configured data root.")
        merge_info.setObjectName("MutedLabel")
        merge_info.setWordWrap(True)
        layout.addWidget(merge_info)

        merge_form = _InputGrid(layout)
        self.merge_output_dataset_input = QLineEdit("")
        self.merge_output_dataset_input.setPlaceholderText("owner/merged_dataset")
        merge_form.add_field("Output dataset", self.merge_output_dataset_input)

        self.merge_source_datasets_input = QPlainTextEdit()
        self.merge_source_datasets_input.setPlaceholderText("owner/dataset_a\nowner/dataset_b")
        self.merge_source_datasets_input.setMinimumHeight(96)
        self.merge_source_datasets_input.setMaximumHeight(160)
        merge_form.add_field("Source datasets", self.merge_source_datasets_input)

        merge_actions = QHBoxLayout()
        self.merge_button = QPushButton("Merge Datasets")
        self.merge_button.setObjectName("AccentButton")
        self.merge_button.clicked.connect(on_merge)
        merge_actions.addWidget(self.merge_button)
        register_action_button(self.merge_button)
        merge_actions.addStretch(1)
        layout.addLayout(merge_actions)

    def repo_id(self, fallback: str = "") -> str:
        return self.dataset_input.text().strip() or str(fallback).strip()

    def set_repo_id(self, repo_id: str) -> None:
        value = str(repo_id).strip()
        if value:
            self.dataset_input.setText(value)

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text))

    def set_episode_rows(self, episode_indices: list[int]) -> None:
        self.episodes_table.setRowCount(len(episode_indices))
        for row, episode_index in enumerate(episode_indices):
            episode_item = QTableWidgetItem(str(episode_index))
            self.episodes_table.setItem(row, 0, episode_item)
            checkbox = QCheckBox()
            checkbox.setChecked(False)
            self.episodes_table.setCellWidget(row, 1, checkbox)

    def clear_episode_rows(self) -> None:
        self.episodes_table.setRowCount(0)

    def selected_episode_indices(self) -> list[int]:
        indices: list[int] = []
        for row in range(self.episodes_table.rowCount()):
            item = self.episodes_table.item(row, 0)
            widget = self.episodes_table.cellWidget(row, 1)
            if item is None or not isinstance(widget, QCheckBox) or not widget.isChecked():
                continue
            try:
                indices.append(int(item.text().strip()))
            except ValueError:
                continue
        return indices

    def select_all_episodes(self) -> None:
        for row in range(self.episodes_table.rowCount()):
            widget = self.episodes_table.cellWidget(row, 1)
            if isinstance(widget, QCheckBox):
                widget.setChecked(True)

    def select_no_episodes(self) -> None:
        for row in range(self.episodes_table.rowCount()):
            widget = self.episodes_table.cellWidget(row, 1)
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)

    def episode_row_count(self) -> int:
        return int(self.episodes_table.rowCount())

    def merge_output_repo_id(self) -> str:
        return self.merge_output_dataset_input.text().strip()

    def set_merge_output_repo_id(self, repo_id: str) -> None:
        self.merge_output_dataset_input.setText(str(repo_id).strip())

    def merge_source_repo_ids(self) -> list[str]:
        return parse_dataset_repo_ids(self.merge_source_datasets_input.toPlainText())
