from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

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
        QTabBar,
        QTabWidget,
        QToolButton,
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
        status="Dataset capture",
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
        status="Model evaluation",
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
        status="Robot control",
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
        status="Setup tools",
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
        status="Source browser",
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
        status="Run artifacts",
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


_HF_TOKEN_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def _huggingface_token_paths(*, env: Mapping[str, str] | None = None, home: Path | None = None) -> tuple[Path, ...]:
    env_map = env if env is not None else os.environ
    home_dir = home if home is not None else Path.home()
    candidates: list[Path] = []

    raw_token_path = str(env_map.get("HF_TOKEN_PATH", "")).strip()
    if raw_token_path:
        candidates.append(Path(os.path.expandvars(raw_token_path)).expanduser())

    raw_hf_home = str(env_map.get("HF_HOME", "")).strip()
    if raw_hf_home:
        hf_home = Path(os.path.expandvars(raw_hf_home)).expanduser()
    else:
        raw_xdg_cache = str(env_map.get("XDG_CACHE_HOME", "")).strip()
        cache_root = Path(os.path.expandvars(raw_xdg_cache)).expanduser() if raw_xdg_cache else home_dir / ".cache"
        hf_home = cache_root / "huggingface"
    candidates.append(hf_home / "token")
    candidates.append(home_dir / ".huggingface" / "token")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return tuple(unique_candidates)


def _has_huggingface_auth_token(*, env: Mapping[str, str] | None = None, home: Path | None = None) -> bool:
    env_map = env if env is not None else os.environ
    for key in _HF_TOKEN_ENV_KEYS:
        if str(env_map.get(key, "")).strip():
            return True
    for token_path in _huggingface_token_paths(env=env_map, home=home):
        try:
            if token_path.is_file() and token_path.read_text(encoding="utf-8").strip():
                return True
        except OSError:
            continue
    return False


def _huggingface_status_text(config: Mapping[str, Any]) -> str:
    username = str(config.get("hf_username", "")).strip()
    auth_present = _has_huggingface_auth_token()
    if auth_present and username:
        return f"Logged in to Hugging Face as {username}."
    if auth_present:
        return "Hugging Face token detected. Open Config and set your username, or run hf auth whoami in Terminal to confirm the account."
    if username:
        return (
            f"No Hugging Face login detected for {username}. "
            "In Terminal run hf auth login, paste your access token when prompted, then reopen the app."
        )
    return (
        "Not logged in. In Terminal run hf auth login, paste your access token when prompted, "
        "then open Config and set your Hugging Face username."
    )


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


    @dataclass
    class _TerminalSession:
        session_id: int
        title: str
        panel: _QtLogPanel | None
        shell: GuiTerminalShell | None


    class _QtLogPanel(QFrame):
        def __init__(
            self,
            *,
            on_submit_bytes: Callable[[bytes], tuple[bool, str]] | None = None,
            on_interrupt: Callable[[], None] | None = None,
            on_activate: Callable[[], tuple[bool, str]] | None = None,
            on_resize_terminal: Callable[[int, int], None] | None = None,
            on_status_change: Callable[[str], None] | None = None,
        ) -> None:
            super().__init__()
            self.setObjectName("TerminalPanel")
            self._on_submit_bytes = on_submit_bytes
            self._on_interrupt = on_interrupt
            self._on_activate = on_activate
            self._on_resize_terminal = on_resize_terminal
            self._on_status_change = on_status_change
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

        def status_text(self) -> str:
            return self._status_text

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
            if self._on_status_change is not None:
                self._on_status_change(self._status_text)

        def focus_terminal(self) -> None:
            self._terminal.setFocus()

        def refresh_terminal_geometry(self) -> None:
            self._terminal.refresh_terminal_geometry()


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
            self._activity_messages: list[str] = []
            self._nav_widgets: list[_NavItemWidget] = []
            self._sidebar_collapsed = _config_bool(self.config.get("ui_sidebar_collapsed", False), False)
            self._terminal_visible = _config_bool(self.config.get("ui_terminal_visible", True), True)
            self._terminal_split_ratio = 0.28
            self._root_adapter = _QtAfterAdapter()
            self._latest_artifact_path: Path | None = None
            self._terminal_sessions: list[_TerminalSession] = []
            self._next_terminal_session_id = 1
            self._run_bridge = QtRunControllerBridge(
                config=self.config,
                append_log=self.append_log,
                on_running_state_change=self._on_running_state_change,
            )
            self._run_controller = self._run_bridge.controller

            self.setWindowTitle("LeRobot GUI")
            self.setMinimumSize(1080, 760)
            icon_path = find_app_icon_png()
            if icon_path is not None:
                self.setWindowIcon(QIcon(str(icon_path)))

            self._build_ui()
            self._refresh_huggingface_status()
            self._apply_sidebar_visibility(announce=False, persist=False)
            self._apply_terminal_visibility(announce=False, persist=False, focus_terminal=False)
            self.apply_theme()
            self._apply_initial_geometry()
            self.select_section("record")
            self.append_log("LeRobot GUI initialized.")
            self.append_log("Core workflows and secondary pages are wired into the terminal workspace.")

        def _build_ui(self) -> None:
            root = QWidget()
            outer = QHBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 18)
            outer.setSpacing(18)
            self.setCentralWidget(root)

            self.sidebar = self._build_sidebar()
            outer.addWidget(self.sidebar)

            self.sidebar_rail = self._build_sidebar_rail()
            outer.addWidget(self.sidebar_rail)

            surface = QFrame()
            surface.setObjectName("ContentSurface")
            surface_layout = QVBoxLayout(surface)
            surface_layout.setContentsMargins(18, 18, 18, 18)
            surface_layout.setSpacing(18)

            self.page_stack = QStackedWidget()
            for section in self._sections:
                self.page_stack.addWidget(self._build_page(section))

            self.workspace_window = self._build_workspace_window()
            self.terminal_window = self._build_terminal_window()
            self._create_terminal_session(focus=False)

            self.workspace_splitter = QSplitter(Qt.Orientation.Vertical)
            self.workspace_splitter.setObjectName("WorkspaceSplitter")
            self.workspace_splitter.setHandleWidth(14)
            self.workspace_splitter.addWidget(self.workspace_window)
            self.workspace_splitter.addWidget(self.terminal_window)
            self.workspace_splitter.setStretchFactor(0, 5)
            self.workspace_splitter.setStretchFactor(1, 2)
            self.workspace_splitter.splitterMoved.connect(self._remember_terminal_split_ratio)

            surface_layout.addWidget(self.workspace_splitter, 1)
            outer.addWidget(surface, 1)

            status = QStatusBar()
            status.showMessage("LeRobot GUI ready.")
            self.setStatusBar(status)

        def _build_sidebar(self) -> QFrame:
            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(300)

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

            self.sidebar_collapse_button = QPushButton()
            self.sidebar_collapse_button.setObjectName("SidebarChromeButton")
            self.sidebar_collapse_button.setFixedSize(30, 30)
            self.sidebar_collapse_button.clicked.connect(self.toggle_sidebar)
            title_row.addWidget(self.sidebar_collapse_button)
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
            return sidebar

        def _build_sidebar_rail(self) -> QFrame:
            rail = QFrame()
            rail.setObjectName("SidebarRail")
            rail.setFixedWidth(56)

            rail_layout = QVBoxLayout(rail)
            rail_layout.setContentsMargins(8, 12, 8, 12)
            rail_layout.setSpacing(12)

            self.sidebar_expand_button = QPushButton()
            self.sidebar_expand_button.setObjectName("SidebarChromeButton")
            self.sidebar_expand_button.setFixedSize(40, 40)
            self.sidebar_expand_button.clicked.connect(self.toggle_sidebar)
            rail_layout.addWidget(
                self.sidebar_expand_button,
                0,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            )
            rail_layout.addStretch(1)
            return rail

        def _build_workspace_window(self) -> QFrame:
            window = QFrame()
            window.setObjectName("WorkspaceWindow")

            layout = QVBoxLayout(window)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)

            header = QFrame()
            header.setObjectName("PaneHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(16)

            heading_layout = QVBoxLayout()
            heading_layout.setContentsMargins(0, 0, 0, 0)
            heading_layout.setSpacing(2)

            self.workspace_meta_label = QLabel("Main workspace")
            self.workspace_meta_label.setObjectName("PaneEyebrow")
            heading_layout.addWidget(self.workspace_meta_label)

            self.workspace_title_label = QLabel("Workspace")
            self.workspace_title_label.setObjectName("PaneTitle")
            heading_layout.addWidget(self.workspace_title_label)

            self.workspace_subtitle_label = QLabel("Select a workflow from the sidebar.")
            self.workspace_subtitle_label.setObjectName("PaneSubtitle")
            self.workspace_subtitle_label.setWordWrap(True)
            heading_layout.addWidget(self.workspace_subtitle_label)

            header_layout.addLayout(heading_layout, 1)

            account_layout = QVBoxLayout()
            account_layout.setContentsMargins(0, 0, 0, 0)
            account_layout.setSpacing(2)

            self.hf_status_title_label = QLabel("Hugging Face")
            self.hf_status_title_label.setObjectName("SectionMeta")
            self.hf_status_title_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            account_layout.addWidget(self.hf_status_title_label)

            self.hf_status_label = QLabel("Checking Hugging Face login...")
            self.hf_status_label.setObjectName("PaneSubtitle")
            self.hf_status_label.setWordWrap(True)
            self.hf_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            self.hf_status_label.setMinimumWidth(280)
            self.hf_status_label.setMaximumWidth(320)
            account_layout.addWidget(self.hf_status_label)

            header_layout.addLayout(account_layout, 0)

            layout.addWidget(header)
            layout.addWidget(self.page_stack, 1)
            return window

        def _build_terminal_window(self) -> QFrame:
            window = QFrame()
            window.setObjectName("TerminalWindow")

            layout = QVBoxLayout(window)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)

            header = QFrame()
            header.setObjectName("PaneHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(16)

            heading_layout = QVBoxLayout()
            heading_layout.setContentsMargins(0, 0, 0, 0)
            heading_layout.setSpacing(2)

            eyebrow = QLabel("Runtime shell")
            eyebrow.setObjectName("PaneEyebrow")
            heading_layout.addWidget(eyebrow)

            title = QLabel("Terminal")
            title.setObjectName("PaneTitle")
            heading_layout.addWidget(title)

            subtitle = QLabel("Interactive shell and live workflow output.")
            subtitle.setObjectName("PaneSubtitle")
            heading_layout.addWidget(subtitle)

            header_layout.addLayout(heading_layout, 1)

            controls_layout = QVBoxLayout()
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(6)

            self.terminal_status_label = QLabel("Interactive shell ready.")
            self.terminal_status_label.setObjectName("PaneSubtitle")
            self.terminal_status_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.terminal_status_label.setWordWrap(True)
            self.terminal_status_label.setMinimumWidth(260)
            controls_layout.addWidget(self.terminal_status_label)

            header_layout.addLayout(controls_layout, 0)

            self.terminal_tabs = QTabWidget()
            self.terminal_tabs.setObjectName("TerminalTabs")
            self.terminal_tabs.setDocumentMode(True)
            self.terminal_tabs.setTabsClosable(False)
            self.terminal_tabs.currentChanged.connect(self._on_terminal_tab_changed)
            self.terminal_tabs.tabBar().setDrawBase(False)

            self.new_terminal_tab_button = QToolButton()
            self.new_terminal_tab_button.setObjectName("TerminalTabAddButton")
            self.new_terminal_tab_button.setText("+")
            self.new_terminal_tab_button.clicked.connect(self.new_terminal_session)
            self.terminal_tabs.setCornerWidget(self.new_terminal_tab_button, Qt.Corner.TopRightCorner)

            layout.addWidget(header)
            layout.addWidget(self.terminal_tabs, 1)
            return window

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

            scroll = self._build_page_scroll_area()

            body = QWidget()
            layout = QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(18)

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
            scroll = self._build_page_scroll_area()
            scroll.setWidget(panel)
            return scroll

        def _build_page_scroll_area(self) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            return scroll

        def _session_for_id(self, session_id: int) -> _TerminalSession | None:
            for session in self._terminal_sessions:
                if session.session_id == session_id:
                    return session
            return None

        def _active_terminal_session(self) -> _TerminalSession | None:
            if not hasattr(self, "terminal_tabs"):
                return None
            index = self.terminal_tabs.currentIndex()
            if index < 0 or index >= len(self._terminal_sessions):
                return None
            return self._terminal_sessions[index]

        def _sync_active_terminal_status(self) -> None:
            session = self._active_terminal_session()
            if session is None or session.panel is None:
                self.terminal_status_label.setText("Interactive shell ready.")
                return
            text = session.panel.status_text().strip()
            self.terminal_status_label.setText(text or "Interactive shell ready.")

        def _refresh_all_terminal_geometries(self) -> None:
            for session in self._terminal_sessions:
                if session.panel is not None:
                    session.panel.refresh_terminal_geometry()

        def _rebuild_terminal_tab_controls(self) -> None:
            tab_bar = self.terminal_tabs.tabBar()
            for index, session in enumerate(self._terminal_sessions):
                close_button = QToolButton(tab_bar)
                close_button.setObjectName("TerminalTabCloseButton")
                close_button.setText("x")
                close_button.setAutoRaise(True)
                close_button.clicked.connect(
                    lambda _checked=False, sid=session.session_id: self.close_terminal_session(sid)
                )
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, close_button)

        def _create_terminal_session(self, *, focus: bool) -> _TerminalSession:
            session_id = self._next_terminal_session_id
            self._next_terminal_session_id += 1
            title = f"Terminal {session_id}"
            session = _TerminalSession(session_id=session_id, title=title, panel=None, shell=None)

            panel = _QtLogPanel(
                on_submit_bytes=lambda payload, sid=session_id: self._handle_terminal_input_bytes(sid, payload),
                on_interrupt=lambda sid=session_id: self._send_terminal_interrupt(sid),
                on_activate=lambda sid=session_id: self._activate_terminal_environment(sid),
                on_resize_terminal=lambda columns, rows, sid=session_id: self._resize_terminal(sid, columns, rows),
                on_status_change=lambda _message, sid=session_id: self._update_terminal_session_status(sid),
            )
            shell = GuiTerminalShell(
                root=self._root_adapter,
                config=self.config,
                append_log=self.append_log,
                is_pipeline_active=self._run_controller.has_active_process,
                send_pipeline_stdin=self._run_controller.send_stdin,
                append_terminal_output=lambda chunk, sid=session_id: self._append_terminal_output(sid, chunk),
                set_status_message=panel.set_status,
            )

            session.panel = panel
            session.shell = shell
            self._terminal_sessions.append(session)
            self.terminal_tabs.addTab(panel, title)
            self.terminal_tabs.setCurrentIndex(self.terminal_tabs.count() - 1)
            self.terminal_tabs.setTabToolTip(self.terminal_tabs.count() - 1, "Interactive LeRobot shell")
            self._rebuild_terminal_tab_controls()

            ok, message = shell.start()
            if message:
                self.append_log(message)
            if not ok and not message:
                self.append_log(f"{title}: interactive shell is unavailable on this platform.")
                panel.set_status("Interactive shell is unavailable on this platform.")
            elif not message:
                panel.set_status("Interactive shell ready.")

            self._sync_active_terminal_status()
            QTimer.singleShot(0, panel.refresh_terminal_geometry)
            if focus and self._terminal_visible:
                panel.focus_terminal()
            return session

        def new_terminal_session(self) -> None:
            session = self._create_terminal_session(focus=True)
            self.append_log(f"Opened {session.title}.")
            self.statusBar().showMessage(f"{session.title} opened.")

        def close_terminal_session(self, session_id: int) -> None:
            for index, session in enumerate(self._terminal_sessions):
                if session.session_id == session_id:
                    self.close_terminal_session_at(index)
                    return

        def close_terminal_session_at(self, index: int) -> None:
            if index < 0 or index >= len(self._terminal_sessions):
                return
            session = self._terminal_sessions.pop(index)
            widget = self.terminal_tabs.widget(index)
            if widget is not None:
                self.terminal_tabs.removeTab(index)
            if session.shell is not None:
                session.shell.append_terminal_output = None
                session.shell.set_status_message = None
                session.shell.shutdown()
            if widget is not None:
                widget.deleteLater()
            self.append_log(f"Closed {session.title}.")

            if not self._terminal_sessions:
                replacement = self._create_terminal_session(focus=self._terminal_visible)
                self.append_log(f"Opened {replacement.title} to keep the shell available.")
            else:
                self._rebuild_terminal_tab_controls()
                self._sync_active_terminal_status()
                active_session = self._active_terminal_session()
                if self._terminal_visible and active_session is not None and active_session.panel is not None:
                    active_session.panel.refresh_terminal_geometry()
                    active_session.panel.focus_terminal()

        def _on_terminal_tab_changed(self, _index: int) -> None:
            self._sync_active_terminal_status()
            session = self._active_terminal_session()
            if session is not None and session.panel is not None:
                session.panel.refresh_terminal_geometry()

        def _update_terminal_session_status(self, session_id: int) -> None:
            session = self._session_for_id(session_id)
            if session is None:
                return
            active_session = self._active_terminal_session()
            if active_session is not None and active_session.session_id == session_id:
                self._sync_active_terminal_status()

        def terminal_session_count(self) -> int:
            return len(self._terminal_sessions)

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
            self._refresh_sidebar_toggle_buttons()
            self._refresh_terminal_button()

        def _refresh_sidebar_toggle_buttons(self) -> None:
            if hasattr(self, "sidebar_collapse_button"):
                self.sidebar_collapse_button.setText("<")
                self.sidebar_collapse_button.setToolTip("Minimize the sidebar")
            if hasattr(self, "sidebar_expand_button"):
                self.sidebar_expand_button.setText(">")
                self.sidebar_expand_button.setToolTip("Expand the sidebar")

        def _refresh_terminal_button(self) -> None:
            if not hasattr(self, "terminal_button"):
                return
            if self._terminal_visible:
                self.terminal_button.setText("Hide Terminal")
                self.terminal_button.setToolTip("Collapse the terminal panel")
            else:
                self.terminal_button.setText("Show Terminal")
                self.terminal_button.setToolTip("Expand the terminal panel")

        def _persist_sidebar_visibility(self) -> None:
            self.config["ui_sidebar_collapsed"] = self._sidebar_collapsed
            save_config(self.config, quiet=True)

        def _persist_terminal_visibility(self) -> None:
            self.config["ui_terminal_visible"] = self._terminal_visible
            save_config(self.config, quiet=True)

        def _apply_sidebar_visibility(self, *, announce: bool, persist: bool) -> None:
            if self._sidebar_collapsed:
                self.sidebar.hide()
                self.sidebar_rail.show()
                if announce:
                    self.statusBar().showMessage("Sidebar minimized.")
                    self.append_log("Sidebar minimized.")
            else:
                self.sidebar.show()
                self.sidebar_rail.hide()
                if announce:
                    self.statusBar().showMessage("Sidebar expanded.")
                    self.append_log("Sidebar expanded.")
            if persist:
                self._persist_sidebar_visibility()
            self._refresh_sidebar_toggle_buttons()

        def _remember_terminal_split_ratio(self, *_args: object) -> None:
            if not getattr(self, "_terminal_visible", False):
                return
            sizes = self.workspace_splitter.sizes()
            if len(sizes) < 2:
                return
            total = sum(sizes)
            if total <= 0 or sizes[1] <= 0:
                return
            ratio = sizes[1] / total
            self._terminal_split_ratio = min(0.55, max(0.18, ratio))

        def _apply_terminal_visibility(
            self,
            *,
            announce: bool,
            persist: bool,
            focus_terminal: bool,
        ) -> None:
            if self._terminal_visible:
                self.terminal_window.show()
                self.terminal_tabs.show()
                splitter_height = max(self.workspace_splitter.height(), sum(self.workspace_splitter.sizes()), 720)
                terminal_size = max(220, int(splitter_height * self._terminal_split_ratio))
                main_size = max(360, splitter_height - terminal_size)
                self.workspace_splitter.setSizes([main_size, terminal_size])
                session = self._active_terminal_session()
                if session is not None and session.panel is not None:
                    session.panel.refresh_terminal_geometry()
                if announce:
                    self.statusBar().showMessage("Terminal shown.")
                    self.append_log("Terminal shown.")
                if focus_terminal:
                    if session is not None and session.panel is not None:
                        session.panel.focus_terminal()
            else:
                self._remember_terminal_split_ratio()
                self.terminal_window.hide()
                self.terminal_tabs.hide()
                self.workspace_splitter.setSizes([1, 0])
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
            self._update_workspace_header(section)
            self._refresh_huggingface_status()
            self.statusBar().showMessage(f"{section.title} selected.")
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

        def _update_workspace_header(self, section: QtSectionDefinition) -> None:
            self.workspace_meta_label.setText(section.stage)
            self.workspace_title_label.setText(section.title)
            self.workspace_subtitle_label.setText(section.subtitle)

        def _refresh_huggingface_status(self) -> None:
            if not hasattr(self, "hf_status_label"):
                return
            self.hf_status_label.setText(_huggingface_status_text(self.config))

        def apply_theme(self) -> None:
            self.colors = build_theme_colors(ui_font="Inter", mono_font="JetBrains Mono", theme_mode=self.theme_mode)
            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet(build_qt_stylesheet(self.colors))
            self._refresh_theme_button()
            QTimer.singleShot(0, self._refresh_all_terminal_geometries)

        def toggle_theme_mode(self) -> None:
            self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
            self.apply_theme()
            self.append_log(f"Theme switched to {self.theme_mode}.")

        def toggle_sidebar(self) -> None:
            self._sidebar_collapsed = not self._sidebar_collapsed
            self._apply_sidebar_visibility(announce=True, persist=True)

        def toggle_terminal_panel(self) -> None:
            self._terminal_visible = not self._terminal_visible
            self._apply_terminal_visibility(announce=True, persist=True, focus_terminal=self._terminal_visible)

        def sidebar_collapsed(self) -> bool:
            return self._sidebar_collapsed

        def terminal_visible(self) -> bool:
            return self._terminal_visible

        def append_log(self, message: str) -> None:
            text = str(message)
            self._activity_messages.append(text)
            if str(message).startswith("Run artifacts saved: "):
                artifact_text = str(message).split("Run artifacts saved: ", 1)[1].strip()
                if artifact_text:
                    self._latest_artifact_path = Path(artifact_text)

        def log_contents(self) -> str:
            return "\n".join(self._activity_messages)

        def terminal_contents(self) -> str:
            session = self._active_terminal_session()
            if session is None or session.panel is None:
                return ""
            return session.panel.terminal_contents()

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

        def _append_terminal_output(self, session_id: int, chunk: str) -> None:
            session = self._session_for_id(session_id)
            if session is not None and session.panel is not None:
                session.panel.append_terminal_output(chunk)

        def _activate_terminal_environment(self, session_id: int) -> tuple[bool, str]:
            session = self._session_for_id(session_id)
            if session is None or session.shell is None:
                return False, "Terminal session is unavailable."
            ok, message = session.shell.activate_environment()
            if message:
                self.append_log(message)
            elif ok:
                self.append_log("Terminal environment activation command sent.")
            return ok, message

        def _handle_terminal_input_bytes(self, session_id: int, payload: bytes) -> tuple[bool, str]:
            session = self._session_for_id(session_id)
            if session is None or session.shell is None:
                return False, "Terminal session is unavailable."
            ok, message = session.shell.handle_terminal_input(payload)
            if not ok and message:
                self.append_log(f"Terminal send failed: {message}")
            return ok, message

        def _resize_terminal(self, session_id: int, columns: int, rows: int) -> None:
            session = self._session_for_id(session_id)
            if session is not None and session.shell is not None:
                session.shell.resize_terminal(columns, rows)

        def _handle_terminal_submit(self, text: str) -> None:
            session = self._active_terminal_session()
            if session is None or session.shell is None:
                return
            ok, message = session.shell.handle_terminal_submit(text)
            if not ok and message:
                self.append_log(f"Terminal send failed: {message}")

        def _send_terminal_interrupt(self, session_id: int) -> None:
            session = self._session_for_id(session_id)
            if session is None or session.shell is None:
                return
            ok, message = session.shell.send_interrupt()
            if not ok and message:
                self.append_log(f"Terminal interrupt failed: {message}")

        def _send_terminal_command(self, command: str) -> tuple[bool, str]:
            session = self._active_terminal_session()
            if session is None or session.shell is None:
                return False, "No terminal session is available."
            ok, message = session.shell.handle_terminal_submit(command)
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
                    for session in list(self._terminal_sessions):
                        if session.shell is not None:
                            session.shell.append_terminal_output = None
                            session.shell.set_status_message = None
                            session.shell.shutdown()
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
                for session in list(self._terminal_sessions):
                    if session.shell is not None:
                        session.shell.append_terminal_output = None
                        session.shell.set_status_message = None
                        session.shell.shutdown()
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
