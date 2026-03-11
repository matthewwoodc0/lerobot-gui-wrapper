from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .app_icon import find_app_icon_png
from .app_theme import build_theme_colors, normalize_theme_mode
from .artifacts import list_runs
from .camera_state import camera_mapping_summary
from .config_store import normalize_config_without_prompts, save_config
from .history_utils import is_visible_history_mode, open_path_in_file_manager
from .gui_qt_theme import build_qt_stylesheet
from .gui_terminal_shell import GuiTerminalShell

try:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    from .gui_qt_core_ops import build_qt_core_ops_panel
    from .gui_qt_secondary_pages import build_qt_secondary_panel
    from .gui_qt_runner import QtRunControllerBridge
    from .gui_qt_terminal import QtTerminalEmulator

    _QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised through availability helpers
    QtTerminalEmulator = None  # type: ignore[assignment]
    _QT_IMPORT_ERROR = exc


@dataclass(frozen=True)
class QtSectionDefinition:
    id: str
    title: str
    subtitle: str
    stage: str
    summary: str
    focus: str
    status: str
    highlights: tuple[str, ...]


_QT_SECTIONS: tuple[QtSectionDefinition, ...] = (
    QtSectionDefinition(
        id="record",
        title="Record",
        subtitle="Dataset capture, dataset browser, and Hugging Face sync.",
        stage="Core ops",
        summary="Supports shared command assembly, record preflight, and live record execution with cancel support.",
        focus="Next step is closing parity gaps around command editing, dataset browsing, and post-run polish.",
        status="Live run flow online",
        highlights=(
            "The shared run controller streams live output into the main shell without toolkit-specific glue.",
            "Record upload follow-up runs are launched from the same shared execution layer.",
            "Camera preview and dataset browser parity remain the main risk areas.",
        ),
    ),
    QtSectionDefinition(
        id="deploy",
        title="Deploy",
        subtitle="Local model selection, eval runs, and deployment diagnostics.",
        stage="Core ops",
        summary="Supports shared deploy command assembly, deploy preflight, and live deploy execution with cancel support.",
        focus="Next step is replacing manual model entry with a browser/model view and restoring quick-fix UX.",
        status="Live run flow online",
        highlights=(
            "Model tree + parity popouts map cleanly to model/view widgets.",
            "Runtime diagnostics and artifact writing flow through the shared runner into the main shell.",
            "Quick-fix dialogs, command editing, and eval outcome UX still need equivalents.",
        ),
    ),
    QtSectionDefinition(
        id="teleop",
        title="Teleop",
        subtitle="Robot connection setup and live teleoperation launch.",
        stage="Core ops",
        summary="Supports shared teleop command assembly, teleop preflight, and live launch/cancel.",
        focus="Next step is adding camera preview, command editing, and the final teleop control UX polish.",
        status="Live run flow online",
        highlights=(
            "Teleop uses the same shared streaming controller as record/deploy, including calibration auto-accept logic.",
            "Camera pause/resume behavior and richer run controls are still pending.",
            "Teleop now stays focused on connection status and live-session control instead of episode stepping.",
        ),
    ),
    QtSectionDefinition(
        id="config",
        title="Config",
        subtitle="Environment setup, diagnostics, and launcher management.",
        stage="Secondary",
        summary="Owns config editing, setup diagnostics, and launcher installation.",
        focus="Next step is adding native file pickers and finishing config parity for the less-used fields.",
        status="Workflow live",
        highlights=(
            "Doctor and setup-wizard summaries already live in shared non-visual helpers.",
            "Launcher validation now targets PySide6.",
            "A few low-frequency config prompts still need richer affordances.",
        ),
    ),
    QtSectionDefinition(
        id="visualizer",
        title="Visualizer",
        subtitle="Read-only browsing for datasets, deploy sources, and videos.",
        stage="Secondary",
        summary="Browses local deployment runs, datasets, models, and discovered videos using the shared discovery helpers.",
        focus="Next step is polishing the metadata/details layout and adding richer source actions.",
        status="Workflow live",
        highlights=(
            "Existing helper functions already covered most source-discovery logic.",
            "Video and insight payloads now feed a native browser.",
            "The remaining gap is mostly UX polish, not business logic.",
        ),
    ),
    QtSectionDefinition(
        id="history",
        title="History",
        subtitle="Run artifacts, reruns, and deployment notes.",
        stage="Secondary",
        summary="Lists run artifacts, opens logs/folders, and can rerun stored commands through the shared controller.",
        focus="Next step is bringing over deploy notes editing and deeper artifact inspection.",
        status="Workflow live",
        highlights=(
            "History payload shaping already lived outside widget rendering.",
            "Reruns now use the same shared streaming controller as the core workflow pages.",
            "Deploy notes editing remains the main history-specific gap.",
        ),
    ),
)


def qt_available() -> tuple[bool, str | None]:
    if _QT_IMPORT_ERROR is None:
        return True, None
    return False, str(_QT_IMPORT_ERROR)


def ensure_qt_application(argv: list[str] | None = None) -> tuple[Any, bool]:
    if _QT_IMPORT_ERROR is not None:
        raise RuntimeError(f"PySide6 is unavailable: {_QT_IMPORT_ERROR}")
    app = QApplication.instance()
    if app is not None:
        return app, False
    return QApplication(list(argv or ["robot_pipeline.py", "gui-qt"])), True


def qt_preview_sections() -> tuple[QtSectionDefinition, ...]:
    return _QT_SECTIONS


if _QT_IMPORT_ERROR is None:

    def _config_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        return bool(value)

    class _QtAfterAdapter(QObject):
        _dispatch = Signal(int, object, object)

        def __init__(self) -> None:
            super().__init__()
            self._dispatch.connect(self._deliver)

        def after(self, delay_ms: int, callback: object, *args: object) -> None:
            if not callable(callback):
                return
            self._dispatch.emit(max(0, int(delay_ms)), callback, tuple(args))

        def _deliver(self, delay_ms: int, callback: object, args: object) -> None:
            if not callable(callback):
                return
            payload = tuple(args) if isinstance(args, tuple) else (args,)
            if delay_ms <= 0:
                callback(*payload)
                return
            QTimer.singleShot(delay_ms, lambda: callback(*payload))


    class _QtLogPanel(QFrame):
        def __init__(
            self,
            *,
            on_submit_bytes: Callable[[bytes], tuple[bool, str]] | None = None,
            on_interrupt: Callable[[], None] | None = None,
            on_activate: Callable[[], tuple[bool, str]] | None = None,
            on_resize_terminal: Callable[[int, int], None] | None = None,
        ) -> None:
            super().__init__()
            self.setObjectName("TerminalPanel")
            self._on_submit_bytes = on_submit_bytes
            self._on_interrupt = on_interrupt
            self._on_activate = on_activate
            self._on_resize_terminal = on_resize_terminal
            self._activity_messages: list[str] = []
            self._status_text = ""

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._terminal = QtTerminalEmulator(
                send_input=self._on_submit_bytes,
                send_interrupt=self._on_interrupt,
                on_status=self.set_status,
                resize_terminal=self._on_resize_terminal,
            )
            self._terminal.setMinimumHeight(160)
            self._terminal.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self._terminal, 1)

        def append_log(self, message: str) -> None:
            text = str(message)
            self._activity_messages.append(text)
            self.set_status(text)

        def contents(self) -> str:
            return "\n".join(self._activity_messages)

        def append_terminal_output(self, chunk: str) -> None:
            self._terminal.feed_output(str(chunk))

        def terminal_contents(self) -> str:
            return self._terminal.terminal_text()

        def send_interrupt(self) -> None:
            if self._on_interrupt is not None:
                self._on_interrupt()

        def activate_shell(self) -> None:
            if self._on_activate is not None:
                ok, message = self._on_activate()
                if message:
                    self.set_status(message)
                elif ok:
                    self.set_status("Environment activation command sent to the terminal.")

        def clear_terminal(self) -> None:
            self._terminal.clear_terminal_buffer()

        def set_status(self, message: str) -> None:
            self._status_text = str(message)

        def focus_terminal(self) -> None:
            self._terminal.setFocus()


    class _NavItemWidget(QFrame):
        def __init__(self, *, title: str, status: str) -> None:
            super().__init__()
            self.setObjectName("NavItem")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(2)

            self._title_label = QLabel(title)
            self._title_label.setObjectName("NavItemTitle")
            layout.addWidget(self._title_label)

            self._status_label = QLabel(status)
            self._status_label.setObjectName("NavItemMeta")
            self._status_label.setWordWrap(True)
            layout.addWidget(self._status_label)

        def set_selected(self, selected: bool) -> None:
            self.setProperty("selected", selected)
            self._title_label.setProperty("selected", selected)
            self._status_label.setProperty("selected", selected)
            self.style().unpolish(self)
            self.style().polish(self)
            self._title_label.style().unpolish(self._title_label)
            self._title_label.style().polish(self._title_label)
            self._status_label.style().unpolish(self._status_label)
            self._status_label.style().polish(self._status_label)


    class QtPreviewWindow(QMainWindow):
        def __init__(self, *, config: dict[str, Any]) -> None:
            super().__init__()
            self.config = dict(config)
            self.theme_mode = normalize_theme_mode(config.get("ui_theme_mode", "dark"))
            self.colors = build_theme_colors(ui_font="Inter", mono_font="JetBrains Mono", theme_mode=self.theme_mode)
            self._sections = qt_preview_sections()
            self._section_index = {section.id: idx for idx, section in enumerate(self._sections)}
            self._nav_widgets: list[_NavItemWidget] = []
            self._pending_logs: list[str] = []
            self._terminal_visible = _config_bool(self.config.get("ui_terminal_visible", True), True)
            self._root_adapter = _QtAfterAdapter()
            self._latest_artifact_path: Path | None = None
            self._run_bridge = QtRunControllerBridge(
                config=self.config,
                append_log=self.append_log,
                on_running_state_change=self._on_running_state_change,
            )
            self._run_controller = self._run_bridge.controller
            self._terminal_shell = GuiTerminalShell(
                root=self._root_adapter,
                config=self.config,
                append_log=self.append_log,
                is_pipeline_active=self._run_controller.has_active_process,
                send_pipeline_stdin=self._run_controller.send_stdin,
                append_terminal_output=self._append_terminal_output,
            )

            self.setWindowTitle("LeRobot GUI")
            self.setMinimumSize(1080, 760)
            icon_path = find_app_icon_png()
            if icon_path is not None:
                self.setWindowIcon(QIcon(str(icon_path)))

            self._build_ui()
            self._apply_terminal_visibility(announce=False, persist=False, focus_terminal=False)
            self.apply_theme()
            self._apply_initial_geometry()
            self.select_section("record")
            self._start_terminal_shell()
            self.append_log("LeRobot GUI initialized.")
            self.append_log("Core workflows and secondary pages are running through the main shell.")

        def _build_ui(self) -> None:
            root = QWidget()
            outer = QHBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 18)
            outer.setSpacing(18)
            self.setCentralWidget(root)

            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(300)
            self.sidebar = sidebar
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(20, 20, 20, 20)
            sidebar_layout.setSpacing(14)

            brand = QLabel("LeRobot GUI")
            brand.setObjectName("BrandLabel")
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(8)
            title_row.addWidget(brand)
            title_row.addStretch(1)

            self.theme_button = QPushButton()
            self.theme_button.setObjectName("ThemeToggleButton")
            self.theme_button.setFixedSize(30, 30)
            self.theme_button.clicked.connect(self.toggle_theme_mode)
            title_row.addWidget(self.theme_button)
            sidebar_layout.addLayout(title_row)

            self.nav_list = QListWidget()
            self.nav_list.setSpacing(2)
            self.nav_list.currentRowChanged.connect(self._on_nav_changed)
            for section in self._sections:
                item = QListWidgetItem()
                nav_widget = _NavItemWidget(title=section.title, status=section.status)
                item.setSizeHint(nav_widget.sizeHint())
                self.nav_list.addItem(item)
                self.nav_list.setItemWidget(item, nav_widget)
                self._nav_widgets.append(nav_widget)
            sidebar_layout.addWidget(self.nav_list, 1)

            shell_status = QLabel("Shell status")
            shell_status.setObjectName("SectionMeta")
            sidebar_layout.addWidget(shell_status)

            self.sidebar_status = QLabel("Ready for record, deploy, teleop, config, visualizer, and history.")
            self.sidebar_status.setWordWrap(True)
            self.sidebar_status.setObjectName("MutedLabel")
            sidebar_layout.addWidget(self.sidebar_status)

            self.terminal_button = QPushButton()
            self.terminal_button.setObjectName("TerminalToggleButton")
            self.terminal_button.clicked.connect(self.toggle_terminal_panel)
            sidebar_layout.addWidget(self.terminal_button)

            outer.addWidget(sidebar)

            surface = QFrame()
            surface.setObjectName("ContentSurface")
            surface_layout = QVBoxLayout(surface)
            surface_layout.setContentsMargins(20, 20, 20, 20)
            surface_layout.setSpacing(18)

            self.log_panel = _QtLogPanel(
                on_submit_bytes=self._handle_terminal_input_bytes,
                on_interrupt=self._send_terminal_interrupt,
                on_activate=self._activate_terminal_environment,
                on_resize_terminal=self._resize_terminal,
            )
            self.page_stack = QStackedWidget()
            for section in self._sections:
                self.page_stack.addWidget(self._build_page(section))
            if self._pending_logs:
                for message in self._pending_logs:
                    self.log_panel.append_log(message)
                self._pending_logs.clear()

            self.splitter = QSplitter(Qt.Orientation.Vertical)
            self.splitter.setObjectName("MainSplitter")
            self.splitter.setHandleWidth(10)
            self.splitter.addWidget(self.page_stack)
            self.splitter.addWidget(self.log_panel)
            self.splitter.setStretchFactor(0, 4)
            self.splitter.setStretchFactor(1, 1)
            self.splitter.setSizes([620, 220])

            surface_layout.addWidget(self.splitter, 1)
            outer.addWidget(surface, 1)

            status = QStatusBar()
            status.showMessage("LeRobot GUI ready.")
            self.setStatusBar(status)

        def _build_page(self, section: QtSectionDefinition) -> QWidget:
            core_ops_panel = build_qt_core_ops_panel(
                section_id=section.id,
                config=self.config,
                append_log=self.append_log,
                run_controller=self._run_controller,
            )
            if core_ops_panel is not None:
                return self._wrap_panel(core_ops_panel)
            secondary_panel = build_qt_secondary_panel(
                section_id=section.id,
                config=self.config,
                append_log=self.append_log,
                run_controller=self._run_controller,
                run_terminal_command=self._send_terminal_command,
                update_and_restart_app=self._update_and_restart_app,
            )
            if secondary_panel is not None:
                return self._wrap_panel(secondary_panel)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)

            body = QWidget()
            layout = QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(18)

            hero = QFrame()
            hero.setObjectName("SectionHero")
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(22, 22, 22, 22)
            hero_layout.setSpacing(10)

            stage = QLabel(section.stage)
            stage.setObjectName("SectionMeta")
            hero_layout.addWidget(stage)

            title = QLabel(section.title)
            title.setObjectName("PageTitle")
            hero_layout.addWidget(title)

            subtitle = QLabel(section.subtitle)
            subtitle.setWordWrap(True)
            hero_layout.addWidget(subtitle)

            chip = QLabel(section.status)
            chip.setObjectName("StatusChip")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setMaximumWidth(220)
            hero_layout.addWidget(chip)
            layout.addWidget(hero)

            cards = QGridLayout()
            cards.setHorizontalSpacing(16)
            cards.setVerticalSpacing(16)
            cards.addWidget(self._build_card("Why this page matters", section.summary), 0, 0)
            cards.addWidget(self._build_card("Current port focus", section.focus), 0, 1)
            cards.addWidget(
                self._build_card(
                    "Shared layer reused now",
                    "Theme tokens, command formatting/parsing, and latest-job coordination are already toolkit-neutral.",
                ),
                1,
                0,
            )
            cards.addWidget(
                self._build_card(
                    "Config snapshot",
                    (
                        f"Theme: {self.theme_mode}\n"
                        f"HF owner: {str(self.config.get('hf_username', '')).strip() or '(unset)'}\n"
                        f"Camera map: {camera_mapping_summary(self.config)}"
                    ),
                ),
                1,
                1,
            )
            layout.addLayout(cards)

            highlights_card = QFrame()
            highlights_card.setObjectName("SectionCard")
            highlights_layout = QVBoxLayout(highlights_card)
            highlights_layout.setContentsMargins(18, 18, 18, 18)
            highlights_layout.setSpacing(10)

            highlights_title = QLabel("Section Notes")
            highlights_title.setObjectName("SectionMeta")
            highlights_layout.addWidget(highlights_title)
            for detail in section.highlights:
                item = QLabel(f"- {detail}")
                item.setWordWrap(True)
                highlights_layout.addWidget(item)
            layout.addWidget(highlights_card)
            layout.addStretch(1)

            scroll.setWidget(body)
            return scroll

        def _build_card(self, title: str, text: str) -> QFrame:
            card = QFrame()
            card.setObjectName("SectionCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(8)

            header = QLabel(title)
            header.setObjectName("SectionMeta")
            layout.addWidget(header)

            body = QLabel(text)
            body.setWordWrap(True)
            body.setObjectName("MutedLabel")
            layout.addWidget(body)
            layout.addStretch(1)
            return card

        def _wrap_panel(self, panel: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(panel)
            return scroll

        def _apply_initial_geometry(self) -> None:
            app = QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
            if screen is None:
                self.resize(1320, 880)
                return
            rect = screen.availableGeometry()
            final_w = min(1380, max(1080, rect.width() - 48))
            final_h = min(940, max(760, rect.height() - 64))
            self.resize(final_w, final_h)
            self.move(
                max(rect.x() + (rect.width() - final_w) // 2, 8),
                max(rect.y() + (rect.height() - final_h) // 2, 8),
            )

        def _refresh_theme_button(self) -> None:
            target = "light" if self.theme_mode == "dark" else "dark"
            self.theme_button.setText("☀" if target == "light" else "☾")
            self.theme_button.setToolTip(f"Switch to {target.title()} Theme")
            self._refresh_terminal_button()

        def _refresh_terminal_button(self) -> None:
            if not hasattr(self, "terminal_button"):
                return
            if self._terminal_visible:
                self.terminal_button.setText("Hide Terminal")
                self.terminal_button.setToolTip("Collapse the terminal panel")
            else:
                self.terminal_button.setText("Show Terminal")
                self.terminal_button.setToolTip("Expand the terminal panel")

        def _persist_terminal_visibility(self) -> None:
            self.config["ui_terminal_visible"] = self._terminal_visible
            save_config(self.config, quiet=True)

        def _apply_terminal_visibility(
            self,
            *,
            announce: bool,
            persist: bool,
            focus_terminal: bool,
        ) -> None:
            if self._terminal_visible:
                self.log_panel.show()
                self.splitter.setSizes([620, 220])
                if announce:
                    self.statusBar().showMessage("Terminal shown.")
                    self.append_log("Terminal shown.")
                if focus_terminal:
                    self.log_panel.focus_terminal()
            else:
                self.log_panel.hide()
                self.splitter.setSizes([1, 0])
                if announce:
                    self.statusBar().showMessage("Terminal hidden.")
                    self.append_log("Terminal hidden.")
            if persist:
                self._persist_terminal_visibility()
            self._refresh_terminal_button()

        def _on_nav_changed(self, row: int) -> None:
            if row < 0 or row >= len(self._sections):
                return
            for index, widget in enumerate(self._nav_widgets):
                widget.set_selected(index == row)
            self.page_stack.setCurrentIndex(row)
            self._refresh_visible_page_runtime_state(row)
            section = self._sections[row]
            self.statusBar().showMessage(f"{section.title}: {section.status}")
            self.append_log(f"Switched to {section.title}.")

        def _refresh_visible_page_runtime_state(self, row: int) -> None:
            page = self.page_stack.widget(row)
            if page is None:
                return
            panel = page.widget() if hasattr(page, "widget") and callable(page.widget) else page
            refresh = getattr(panel, "refresh_from_config", None)
            if callable(refresh):
                refresh()
            for child in panel.findChildren(QWidget):
                child_refresh = getattr(child, "refresh_from_config", None)
                if callable(child_refresh):
                    child_refresh()

        def _on_running_state_change(self, active: bool) -> None:
            if active:
                self.sidebar_status.setText("A workflow is currently running.")
            else:
                self.sidebar_status.setText("Ready for a new workflow.")

        def apply_theme(self) -> None:
            self.colors = build_theme_colors(ui_font="Inter", mono_font="JetBrains Mono", theme_mode=self.theme_mode)
            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet(build_qt_stylesheet(self.colors))
            self._refresh_theme_button()

        def toggle_theme_mode(self) -> None:
            self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
            self.apply_theme()
            self.append_log(f"Theme switched to {self.theme_mode}.")

        def toggle_terminal_panel(self) -> None:
            self._terminal_visible = not self._terminal_visible
            self._apply_terminal_visibility(announce=True, persist=True, focus_terminal=self._terminal_visible)

        def terminal_visible(self) -> bool:
            return self._terminal_visible

        def append_log(self, message: str) -> None:
            log_panel = getattr(self, "log_panel", None)
            if log_panel is None:
                self._pending_logs.append(str(message))
                return
            log_panel.append_log(message)
            if str(message).startswith("Run artifacts saved: "):
                artifact_text = str(message).split("Run artifacts saved: ", 1)[1].strip()
                if artifact_text:
                    self._latest_artifact_path = Path(artifact_text)

        def log_contents(self) -> str:
            return self.log_panel.contents()

        def terminal_contents(self) -> str:
            return self.log_panel.terminal_contents()

        def current_section_id(self) -> str:
            row = self.nav_list.currentRow()
            if row < 0:
                return ""
            return self._sections[row].id

        def section_titles(self) -> list[str]:
            return [section.title for section in self._sections]

        def select_section(self, section_id: str) -> None:
            row = self._section_index.get(section_id)
            if row is None:
                raise KeyError(section_id)
            self.nav_list.setCurrentRow(row)

        def _append_terminal_output(self, chunk: str) -> None:
            log_panel = getattr(self, "log_panel", None)
            if log_panel is not None:
                log_panel.append_terminal_output(chunk)

        def _start_terminal_shell(self) -> None:
            ok, message = self._terminal_shell.start()
            if message:
                self.append_log(message)
            if not ok and not message:
                self.append_log("Interactive shell is unavailable on this platform.")
            if self._terminal_visible:
                self.log_panel.focus_terminal()

        def _activate_terminal_environment(self) -> tuple[bool, str]:
            ok, message = self._terminal_shell.activate_environment()
            if message:
                self.append_log(message)
            elif ok:
                self.append_log("Terminal environment activation command sent.")
            return ok, message

        def _handle_terminal_input_bytes(self, payload: bytes) -> tuple[bool, str]:
            ok, message = self._terminal_shell.handle_terminal_input(payload)
            if not ok and message:
                self.append_log(f"Terminal send failed: {message}")
            return ok, message

        def _resize_terminal(self, columns: int, rows: int) -> None:
            self._terminal_shell.resize_terminal(columns, rows)

        def _handle_terminal_submit(self, text: str) -> None:
            ok, message = self._terminal_shell.handle_terminal_submit(text)
            if not ok and message:
                self.append_log(f"Terminal send failed: {message}")

        def _send_terminal_interrupt(self) -> None:
            ok, message = self._terminal_shell.send_interrupt()
            if not ok and message:
                self.append_log(f"Terminal interrupt failed: {message}")

        def _send_terminal_command(self, command: str) -> tuple[bool, str]:
            ok, message = self._terminal_shell.handle_terminal_submit(command)
            if ok:
                self.append_log(f"Terminal command sent: {command}")
            return ok, message

        def _update_and_restart_app(self) -> tuple[bool, str]:
            if self._run_controller.has_active_process():
                return False, "Cannot update while a workflow is active."

            repo_dir = Path(__file__).resolve().parents[1]
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=str(repo_dir),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except FileNotFoundError:
                return False, "git is not available in PATH."
            except subprocess.TimeoutExpired:
                return False, "git pull timed out."
            except Exception as exc:
                return False, f"Unable to run git pull: {exc}"

            details = "\n".join(
                part.strip()
                for part in (result.stdout or "", result.stderr or "")
                if str(part).strip()
            ).strip()
            if result.returncode != 0:
                return False, details or f"git pull failed (exit code {result.returncode})."

            restart_args = list(sys.argv)
            if not restart_args:
                restart_args = [str(repo_dir / "robot_pipeline.py"), "gui"]

            def _restart() -> None:
                try:
                    self._terminal_shell.shutdown()
                except Exception:
                    pass
                os.execv(sys.executable, [sys.executable, *restart_args])

            QTimer.singleShot(150, _restart)
            if details:
                return True, f"Update complete.\n{details}\nRestarting app..."
            return True, "Update complete. Restarting app..."

        def open_latest_artifact(self) -> None:
            latest = self._latest_artifact_path
            if latest is None or not latest.exists():
                runs, _warning_count = list_runs(self.config, limit=25)
                visible_runs = [item for item in runs if is_visible_history_mode(item.get("mode", "run"))]
                if visible_runs:
                    run_path = Path(str(visible_runs[0].get("_run_path", ""))).expanduser()
                    if run_path.exists():
                        latest = run_path
                        self._latest_artifact_path = latest
            if latest is None or not latest.exists():
                self.append_log("No run artifacts are available yet.")
                self.statusBar().showMessage("No run artifacts found.")
                return
            ok, message = open_path_in_file_manager(latest)
            if ok:
                self.append_log(f"Opened latest artifact: {latest}")
                self.statusBar().showMessage("Opened latest artifact.")
            else:
                self.append_log(f"Failed to open latest artifact: {message}")
                self.statusBar().showMessage("Failed to open latest artifact.")

        def closeEvent(self, event: Any) -> None:
            try:
                self._terminal_shell.shutdown()
            finally:
                super().closeEvent(event)

else:

    class QtPreviewWindow:
        def __init__(self, *, config: dict[str, Any]) -> None:
            _ = config
            raise RuntimeError(f"PySide6 is unavailable: {_QT_IMPORT_ERROR}")


def create_qt_preview_window(raw_config: dict[str, Any]) -> QtPreviewWindow:
    config = normalize_config_without_prompts(raw_config)
    return QtPreviewWindow(config=config)


def run_gui_qt_mode(raw_config: dict[str, Any]) -> None:
    ok, detail = qt_available()
    if not ok:
        print("GUI is unavailable on this device.")
        print(f"Details: {detail}")
        return

    app, _created = ensure_qt_application(sys.argv)
    app.setApplicationName("LeRobot GUI")
    window = create_qt_preview_window(raw_config)
    window.show()
    app.exec()
