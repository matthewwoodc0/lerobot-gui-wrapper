from __future__ import annotations

import time
from typing import Any, Callable

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gui_input_help import keyboard_input_help_text, keyboard_input_help_title
from .gui_qt_dialogs import _build_dialog_panel, _fit_dialog_to_screen, show_text_dialog
from .runtime_log_parsing import is_episode_reset_phase_line, is_episode_start_line, parse_episode_progress_line, parse_outcome_tags


class QtRunHelperDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        mode_title: str,
        on_send_key: Callable[[str], tuple[bool, str]] | None = None,
        on_cancel: Callable[[], None] | None = None,
        show_episode_controls: bool = True,
        show_outcome_tracker: bool = True,
        cancel_button_text: str = "Cancel Run",
        cancel_button_marks_success: bool = False,
    ) -> None:
        super().__init__(parent)
        self._mode_title = mode_title
        self._on_send_key = on_send_key
        self._on_cancel = on_cancel
        self._show_episode_controls = bool(show_episode_controls)
        self._show_outcome_tracker = bool(show_outcome_tracker)
        self._cancel_button_marks_success = bool(cancel_button_marks_success)
        self._total_episodes = 0
        self._current_episode = 0
        self._episode_outcomes: dict[int, dict[str, Any]] = {}
        self._selected_episode: int | None = None
        self._controls_ready = False
        self._run_started_at: float | None = None
        self._normal_stop_requested = False
        self._episode_duration_s: int = 0
        self._episode_started_at: float | None = None

        self.setModal(False)
        self.setWindowTitle(f"{mode_title} Helper")
        _fit_dialog_to_screen(
            self,
            requested_width=860,
            requested_height=720,
            requested_min_width=720,
            requested_min_height=560,
        )

        layout = _build_dialog_panel(
            self,
            title=f"{mode_title} Helper",
            subtitle=None,
        )

        header = QHBoxLayout()
        header.setSpacing(10)

        self.status_chip = QLabel("Idle")
        self.status_chip.setObjectName("StatusChip")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setMaximumWidth(220)
        header.addWidget(self.status_chip)

        self.elapsed_label = QLabel("Elapsed: --:--")
        self.elapsed_label.setObjectName("MutedLabel")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.elapsed_label)
        header.addStretch(1)

        self.cancel_button: QPushButton | None = None
        if on_cancel is not None:
            self.cancel_button = QPushButton(cancel_button_text)
            self.cancel_button.clicked.connect(self._handle_cancel)
            header.addWidget(self.cancel_button)
        layout.addLayout(header)

        self.summary_label = QLabel(self._idle_summary_text())
        self.summary_label.setObjectName("MutedLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.episode_progress_bar = QProgressBar()
        self.episode_progress_bar.setObjectName("EpisodeProgressBar")
        self.episode_progress_bar.setRange(0, 1000)
        self.episode_progress_bar.setValue(0)
        self.episode_progress_bar.setTextVisible(False)
        self.episode_progress_bar.setFixedHeight(6)
        self.episode_progress_bar.setVisible(False)
        layout.addWidget(self.episode_progress_bar)

        self.episode_time_label = QLabel("")
        self.episode_time_label.setObjectName("MutedLabel")
        self.episode_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.episode_time_label.setVisible(False)
        layout.addWidget(self.episode_time_label)

        self.reset_button: QPushButton | None = None
        self.next_button: QPushButton | None = None
        if self._show_episode_controls:
            control_row = QHBoxLayout()
            control_row.setSpacing(8)
            self.reset_button = QPushButton("Reset Episode")
            self.reset_button.clicked.connect(lambda: self._dispatch_key("left"))
            self.reset_button.setEnabled(False)
            control_row.addWidget(self.reset_button)

            self.next_button = QPushButton("Next Episode")
            self.next_button.clicked.connect(lambda: self._dispatch_key("right"))
            self.next_button.setEnabled(False)
            control_row.addWidget(self.next_button)

            help_button = QPushButton("Keyboard Help")
            help_button.clicked.connect(self.show_keyboard_help)
            control_row.addWidget(help_button)
            control_row.addStretch(1)
            layout.addLayout(control_row)

        self.key_status_label = QLabel(self._idle_status_text())
        self.key_status_label.setObjectName("MutedLabel")
        self.key_status_label.setWordWrap(True)
        layout.addWidget(self.key_status_label)

        # Splitter: top = runtime log, bottom = outcome tracker
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # --- Top pane: runtime log ---
        log_pane = QWidget()
        log_pane_layout = QVBoxLayout(log_pane)
        log_pane_layout.setContentsMargins(0, 0, 0, 0)
        log_pane_layout.setSpacing(4)

        self.runtime_log_title = QLabel("Runtime Log")
        self.runtime_log_title.setObjectName("SectionMeta")
        log_pane_layout.addWidget(self.runtime_log_title)

        self.runtime_log_output = QPlainTextEdit()
        self.runtime_log_output.setObjectName("DialogText")
        self.runtime_log_output.setReadOnly(True)
        self.runtime_log_output.setMinimumHeight(80)
        self.runtime_log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_pane_layout.addWidget(self.runtime_log_output, 1)

        splitter.addWidget(log_pane)

        # --- Bottom pane: outcome tracker (scrollable) ---
        outcomes_scroll = QScrollArea()
        outcomes_scroll.setWidgetResizable(True)
        outcomes_scroll.setFrameShape(outcomes_scroll.Shape.NoFrame)

        outcomes_container = QWidget()
        outcomes_container_layout = QVBoxLayout(outcomes_container)
        outcomes_container_layout.setContentsMargins(0, 4, 0, 0)
        outcomes_container_layout.setSpacing(8)

        self.outcomes_wrap = QWidget()
        outcomes_layout = QGridLayout(self.outcomes_wrap)
        outcomes_layout.setContentsMargins(0, 0, 0, 0)
        outcomes_layout.setHorizontalSpacing(12)
        outcomes_layout.setVerticalSpacing(8)

        outcomes_title = QLabel("Episode Outcome Tracker")
        outcomes_title.setObjectName("SectionMeta")
        outcomes_layout.addWidget(outcomes_title, 0, 0, 1, 4)

        self.target_episode_label = QLabel("Selected episode: --")
        self.target_episode_label.setObjectName("MutedLabel")
        outcomes_layout.addWidget(self.target_episode_label, 1, 0, 1, 4)

        tags_label = QLabel("Tags")
        tags_label.setObjectName("FormLabel")
        outcomes_layout.addWidget(tags_label, 2, 0)
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("optional comma-separated tags")
        outcomes_layout.addWidget(self.tags_input, 2, 1, 1, 3)

        self.success_button = QPushButton("Mark Success")
        self.success_button.clicked.connect(lambda: self._mark_selected("success"))
        self.success_button.setEnabled(False)
        outcomes_layout.addWidget(self.success_button, 3, 1)

        self.failed_button = QPushButton("Mark Failed")
        self.failed_button.clicked.connect(lambda: self._mark_selected("failed"))
        self.failed_button.setEnabled(False)
        outcomes_layout.addWidget(self.failed_button, 3, 2)

        self.apply_tags_button = QPushButton("Apply Tags")
        self.apply_tags_button.clicked.connect(self._apply_tags_to_selected)
        self.apply_tags_button.setEnabled(False)
        outcomes_layout.addWidget(self.apply_tags_button, 3, 3)

        outcomes_container_layout.addWidget(self.outcomes_wrap)

        self.outcome_table = QTableWidget(0, 3)
        self.outcome_table.setHorizontalHeaderLabels(["Episode", "Status", "Tags"])
        self.outcome_table.verticalHeader().setVisible(False)
        self.outcome_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.outcome_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.outcome_table.itemSelectionChanged.connect(self._sync_selected_episode_from_table)
        self.outcome_table.setMinimumHeight(80)
        outcomes_container_layout.addWidget(self.outcome_table, 1)

        outcomes_scroll.setWidget(outcomes_container)
        self._outcomes_scroll = outcomes_scroll
        splitter.addWidget(outcomes_scroll)

        # Give log ~45% and outcomes ~55% of the splitter height
        splitter.setSizes([260, 320])
        layout.addWidget(splitter, 1)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed_label)

        self._episode_tick_timer = QTimer(self)
        self._episode_tick_timer.setInterval(100)
        self._episode_tick_timer.timeout.connect(self._tick_episode_progress)

        self._set_outcome_tracker_visible(self._show_outcome_tracker)

    def _set_outcome_tracker_visible(self, visible: bool) -> None:
        self._outcomes_scroll.setVisible(visible)

    def _idle_summary_text(self) -> str:
        if self._show_episode_controls:
            return "Open during a live session for progress, episode controls, and outcome notes."
        if self._show_outcome_tracker:
            return "Open during a live session for runtime progress, connection status, and outcome notes."
        return "Open during a live session for elapsed time, connection status, and console logs."

    def _waiting_summary_text(self) -> str:
        if self._show_episode_controls:
            return "Waiting for live output to report episode progress and runtime status."
        if self._show_outcome_tracker:
            return "Waiting for live output to report runtime status."
        return "Waiting for live output to report runtime status."

    def _idle_status_text(self) -> str:
        if self._show_episode_controls:
            return "Arrow-key controls become active when the session reports readiness."
        return "Waiting for teleop readiness."

    def start_run(self, *, run_mode: str, expected_episodes: int | None = None, episode_duration_s: int = 0) -> None:
        self._total_episodes = max(0, int(expected_episodes or 0))
        self._current_episode = 0
        self._episode_outcomes.clear()
        self._selected_episode = None
        self._normal_stop_requested = False
        self._run_started_at = time.monotonic()
        self._episode_duration_s = max(0, int(episode_duration_s))
        self._episode_started_at = None
        self.status_chip.setText(f"{run_mode.title()} running")
        self.summary_label.setText(self._waiting_summary_text())
        self.key_status_label.setText(self._idle_status_text())
        self.tags_input.clear()
        self.runtime_log_output.clear()
        self._reload_outcome_table()
        self._set_controls_enabled(run_mode == "deploy" and self._show_outcome_tracker, ready=False)
        self._refresh_elapsed_label()
        self._reset_episode_progress()
        self._elapsed_timer.start()
        self.show()
        self.raise_()
        self.activateWindow()

    def finish_run(self, *, status_text: str) -> None:
        self._elapsed_timer.stop()
        self._episode_tick_timer.stop()
        self._refresh_elapsed_label()
        self._reset_episode_progress()
        self.status_chip.setText(status_text)
        self.key_status_label.setText("Session finished.")
        if self.reset_button is not None:
            self.reset_button.setEnabled(False)
        if self.next_button is not None:
            self.next_button.setEnabled(False)
        self._controls_ready = False
        if not self._show_outcome_tracker and self._run_started_at is not None:
            self.summary_label.setText(
                f"Session finished after {self._format_elapsed_seconds(time.monotonic() - self._run_started_at)}."
            )

    def set_teleop_ready(self, ready: bool) -> None:
        self._set_controls_enabled(allow_outcomes=False, ready=ready)
        if ready:
            if self._show_episode_controls:
                self.key_status_label.setText("Session ready. Reset and Next episode controls are now live.")
            else:
                self.key_status_label.setText("Session ready.")
        else:
            self.key_status_label.setText(self._idle_status_text())

    def handle_output_line(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        self.runtime_log_output.appendPlainText(text)
        scrollbar = self.runtime_log_output.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
        if is_episode_start_line(text):
            if not self._controls_ready:
                self.set_teleop_ready(True)
            self._start_episode_progress()
            if self._show_outcome_tracker:
                self.summary_label.setText("Episode started.")
            return
        if is_episode_reset_phase_line(text):
            self._reset_episode_progress()
            if self._show_outcome_tracker:
                self.summary_label.setText("Episode reset phase detected. Use outcome buttons when the take completes.")
            return
        if not self._show_outcome_tracker:
            return
        progress = parse_episode_progress_line(text)
        if progress is not None:
            episode, total = progress
            self._current_episode = max(1, int(episode))
            if total is not None:
                self._total_episodes = max(self._total_episodes, int(total))
            self.summary_label.setText(
                f"Episode {self._current_episode}"
                + (f" of {self._total_episodes}" if self._total_episodes else "")
            )
            self._ensure_episode_row(self._current_episode)

    def outcome_payload(self) -> dict[int, dict[str, Any]]:
        return dict(self._episode_outcomes)

    def show_keyboard_help(self) -> None:
        show_text_dialog(
            parent=self,
            title=keyboard_input_help_title(),
            text=keyboard_input_help_text(),
            wrap_mode="word",
        )

    def _handle_cancel(self) -> None:
        if self._cancel_button_marks_success:
            self._normal_stop_requested = True
            self.status_chip.setText(f"Ending {self._mode_title.lower()}")
            self.summary_label.setText("User requested a normal stop. Waiting for the runtime to exit cleanly.")
            self.key_status_label.setText("Stop requested.")
        if self._on_cancel is not None:
            self._on_cancel()

    def consume_normal_stop_request(self) -> bool:
        requested = self._normal_stop_requested
        self._normal_stop_requested = False
        return requested

    def _dispatch_key(self, direction: str) -> None:
        if self._on_send_key is None:
            return
        ok, message = self._on_send_key(direction)
        self.key_status_label.setText(message if ok else f"Dispatch failed: {message}")

    def _set_controls_enabled(self, allow_outcomes: bool, ready: bool) -> None:
        self._controls_ready = bool(ready)
        if self.reset_button is not None:
            self.reset_button.setEnabled(ready and self._on_send_key is not None)
        if self.next_button is not None:
            self.next_button.setEnabled(ready and self._on_send_key is not None)
        has_selection = self._selected_episode is not None
        self.success_button.setEnabled(allow_outcomes and has_selection)
        self.failed_button.setEnabled(allow_outcomes and has_selection)
        self.apply_tags_button.setEnabled(allow_outcomes and has_selection)

    def _ensure_episode_row(self, episode: int) -> None:
        if episode <= 0:
            return
        if episode not in self._episode_outcomes:
            self._episode_outcomes[episode] = {"status": "", "tags": []}
        self._reload_outcome_table(select_episode=episode)

    def _reload_outcome_table(self, *, select_episode: int | None = None) -> None:
        episodes = sorted(self._episode_outcomes)
        self.outcome_table.setRowCount(len(episodes))
        for row, episode in enumerate(episodes):
            outcome = self._episode_outcomes.get(episode, {})
            self.outcome_table.setItem(row, 0, QTableWidgetItem(str(episode)))
            self.outcome_table.setItem(row, 1, QTableWidgetItem(str(outcome.get("status", ""))))
            self.outcome_table.setItem(row, 2, QTableWidgetItem(", ".join(outcome.get("tags", []))))
        if select_episode is not None:
            for row, episode in enumerate(episodes):
                if episode == select_episode:
                    self.outcome_table.selectRow(row)
                    break

    def _sync_selected_episode_from_table(self) -> None:
        if not self._show_outcome_tracker:
            return
        selected = self.outcome_table.selectionModel().selectedRows()
        if not selected:
            self._selected_episode = None
            self.target_episode_label.setText("Selected episode: --")
            self._set_controls_enabled(allow_outcomes=True, ready=self._controls_ready)
            return
        row = selected[0].row()
        item = self.outcome_table.item(row, 0)
        if item is None:
            return
        try:
            self._selected_episode = int(item.text())
        except ValueError:
            self._selected_episode = None
        if self._selected_episode is not None:
            outcome = self._episode_outcomes.get(self._selected_episode, {})
            self.target_episode_label.setText(f"Selected episode: {self._selected_episode}")
            self.tags_input.setText(", ".join(outcome.get("tags", [])))
        ready = self.reset_button.isEnabled() if self.reset_button is not None else self._controls_ready
        self._set_controls_enabled(allow_outcomes=True, ready=ready)

    def _mark_selected(self, status: str) -> None:
        if self._selected_episode is None:
            return
        self._ensure_episode_row(self._selected_episode)
        outcome = self._episode_outcomes.setdefault(self._selected_episode, {"status": "", "tags": []})
        outcome["status"] = status
        outcome["tags"] = parse_outcome_tags(self.tags_input.text())
        self._reload_outcome_table(select_episode=self._selected_episode)

    def _apply_tags_to_selected(self) -> None:
        if self._selected_episode is None:
            return
        self._ensure_episode_row(self._selected_episode)
        outcome = self._episode_outcomes.setdefault(self._selected_episode, {"status": "", "tags": []})
        outcome["tags"] = parse_outcome_tags(self.tags_input.text())
        self._reload_outcome_table(select_episode=self._selected_episode)

    def _reset_episode_progress(self) -> None:
        self._episode_tick_timer.stop()
        self._episode_started_at = None
        self.episode_progress_bar.setValue(0)
        self.episode_progress_bar.setVisible(False)
        self.episode_time_label.setVisible(False)

    def _start_episode_progress(self) -> None:
        self._episode_started_at = time.monotonic()
        if self._episode_duration_s > 0:
            self.episode_progress_bar.setValue(0)
            self.episode_progress_bar.setVisible(True)
            self.episode_time_label.setText(f"0s / {self._episode_duration_s}s")
            self.episode_time_label.setVisible(True)
            self._episode_tick_timer.start()

    def _tick_episode_progress(self) -> None:
        if self._episode_started_at is None or self._episode_duration_s <= 0:
            return
        elapsed = time.monotonic() - self._episode_started_at
        fraction = min(1.0, elapsed / self._episode_duration_s)
        self.episode_progress_bar.setValue(int(fraction * 1000))
        self.episode_time_label.setText(f"{elapsed:.0f}s / {self._episode_duration_s}s")

    def _refresh_elapsed_label(self) -> None:
        if self._run_started_at is None:
            self.elapsed_label.setText("Elapsed: --:--")
            return
        self.elapsed_label.setText(
            "Elapsed: " + self._format_elapsed_seconds(time.monotonic() - self._run_started_at)
        )

    def _format_elapsed_seconds(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
        return f"{minutes:02d}:{remaining_seconds:02d}"
