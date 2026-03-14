from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .app_icon import find_app_icon_png
from .app_theme import SPACING_CARD, SPACING_COMPACT, SPACING_META, SPACING_PANE, SPACING_SHELL, build_theme_colors, normalize_theme_mode
from .artifacts import list_runs
from .camera_state import camera_mapping_summary
from .config_store import normalize_config_without_prompts, save_config
from .history_utils import is_visible_history_mode, open_path_in_file_manager
from .gui_qt_theme import build_qt_stylesheet
from .gui_terminal_shell import GuiTerminalShell
from .qt_bootstrap import ensure_safe_qt_bootstrap, ensure_supported_qt_platform, prepare_qt_environment
from .workflow_queue import WorkflowQueueService

prepare_qt_environment()

try:
    from PySide6.QtCore import QObject, QEvent, Qt, QTimer, Signal
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
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

    from .gui_qt_common import _build_text_card
    from .gui_qt_core_ops import build_qt_core_ops_panel
    from .gui_qt_secondary_pages import build_qt_secondary_panel
    from .gui_qt_sidebar import _NavItemWidget, _SidebarController, build_sidebar, build_sidebar_rail
    from .gui_qt_runner import QtRunControllerBridge
    from .gui_qt_terminal import QtTerminalEmulator

    _QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised through availability helpers
    QtTerminalEmulator = None  # type: ignore[assignment]
    _QT_IMPORT_ERROR = exc


if _QT_IMPORT_ERROR is None:
    class _WheelInputGuard(QObject):
        def eventFilter(self, watched: QObject, event: object) -> bool:
            if isinstance(event, QEvent) and event.type() == QEvent.Type.Wheel:
                if isinstance(watched, (QComboBox, QAbstractSpinBox)):
                    event.ignore()
                    return True
            return super().eventFilter(watched, event)


    def _install_wheel_input_guard(app: Any) -> None:
        if getattr(app, "_robot_pipeline_wheel_input_guard", None) is not None:
            return
        guard = _WheelInputGuard(app)
        app.installEventFilter(guard)
        setattr(app, "_robot_pipeline_wheel_input_guard", guard)
else:
    class _WheelInputGuard:  # pragma: no cover - Qt import failed
        pass


    def _install_wheel_input_guard(app: Any) -> None:  # pragma: no cover - Qt import failed
        _ = app


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
        id="replay",
        title="Replay",
        subtitle="Replay recorded dataset episodes on hardware with preflight, review, and artifact capture.",
        stage="Core ops",
        summary="Runs hardware replay using the detected LeRobot replay entrypoint when available, with graceful fallback messaging when it is not.",
        focus="Next step is expanding upstream flag coverage as more replay entrypoints stabilize, not creating a separate hardware runner.",
        status="Hardware replay",
        highlights=(
            "Replay reuses the shared run controller, editable command review, and artifact capture.",
            "Visualizer and History can launch replay directly into the same history stream.",
            "If the configured LeRobot runtime lacks replay support, the UI explains that before launch.",
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
        id="motor_setup",
        title="Motor Setup",
        subtitle="Port selection, first-time servo bring-up, and setup logging.",
        stage="Core ops",
        summary="Runs dedicated motor setup entrypoints when present and falls back to calibration-oriented bring-up when they are not.",
        focus="Next step is broadening runtime flag detection for more upstream servo setup commands, not inventing a custom serial protocol layer.",
        status="Servo bring-up",
        highlights=(
            "Motor setup uses the same review, preflight, streaming output, cancel, and artifact hooks as the other core workflows.",
            "Successful bring-up can feed updated ports, ids, and robot types back into config state.",
            "When the runtime only exposes calibration, the UI makes the missing ID/baudrate support explicit.",
        ),
    ),
    QtSectionDefinition(
        id="train",
        title="Train",
        subtitle="Policy training, checkpoint management, and training monitoring.",
        stage="Core ops",
        summary="Configure and launch policy training with live output streaming.",
        focus="Supports ACT, Diffusion, and other policy types with dataset selection and device configuration.",
        status="Policy training",
        highlights=(
            "Training uses the shared run controller for live output streaming.",
            "Entrypoint resolution supports multiple LeRobot versions.",
            "WandB integration is configurable, and checkpoint resume is version-checked before launch.",
        ),
    ),
    QtSectionDefinition(
        id="workflows",
        title="Workflows",
        subtitle="Local step-by-step recipes for common record, train, and eval flows.",
        stage="Secondary",
        summary="Chains lightweight record/upload and train/eval workflows on the shared local run controller.",
        focus="Next step is refining recipe ergonomics, not building cluster scheduling or concurrent multi-robot orchestration.",
        status="Workflow recipes",
        highlights=(
            "Workflow steps keep using the shared run controller and normal artifact history instead of a second execution backend.",
            "Train follow-up steps resolve checkpoints from freshly written local artifacts.",
            "Scope stays intentionally local and sequential.",
        ),
    ),
    QtSectionDefinition(
        id="experiments",
        title="Experiments",
        subtitle="Connected train, checkpoint, deploy, and sim-eval analysis.",
        stage="Secondary",
        summary="Compares experiment runs, surfaces local metrics and artifacts, and launches checkpoint-centric deploy or sim-eval workflows.",
        focus="Next step is broadening parser coverage as more upstream train/eval artifacts appear, not inventing a second run database.",
        status="Experiment console",
        highlights=(
            "Training metadata, checkpoint discovery, deploy outcomes, and sim eval artifacts all resolve from the shared run artifacts folder.",
            "Checkpoint actions reuse the existing deploy command builder and the compatibility-driven sim eval entrypoint probe.",
            "WandB stays optional; local artifacts still anchor the experiment surface when remote metadata is unavailable.",
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
        _install_wheel_input_guard(app)
        return app, False
    ensure_supported_qt_platform()
    ensure_safe_qt_bootstrap()
    app = QApplication(list(argv or ["robot_pipeline.py", "gui-qt"]))
    _install_wheel_input_guard(app)
    return app, True


def qt_preview_sections() -> tuple[QtSectionDefinition, ...]:
    return _QT_SECTIONS


_HF_TOKEN_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


@dataclass(frozen=True)
class _HuggingFaceStatusPresentation:
    chip_text: str
    chip_state: str
    summary: str
    tooltip: str


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
            QTimer.singleShot(max(0, int(delay_ms)), lambda: callback(*payload))


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


    class _WorkspacePulseController(QObject):
        def __init__(self, eyebrow_label: QLabel) -> None:
            super().__init__(eyebrow_label)
            self._eyebrow_label = eyebrow_label
            self._pulse_timer = QTimer(self)
            self._pulse_timer.setInterval(600)
            self._pulse_timer.timeout.connect(self._tick_pulse)
            self._pulse_bright = True

        def start(self) -> None:
            self._pulse_timer.start()

        def stop(self) -> None:
            self._reset_workspace_pulse()

        def _tick_pulse(self) -> None:
            self._pulse_bright = not self._pulse_bright
            value = "" if self._pulse_bright else "dim"
            self._eyebrow_label.setProperty("pulsing", value)
            self._eyebrow_label.style().unpolish(self._eyebrow_label)
            self._eyebrow_label.style().polish(self._eyebrow_label)

        def _reset_workspace_pulse(self) -> None:
            self._pulse_timer.stop()
            self._pulse_bright = True
            self._eyebrow_label.setProperty("pulsing", "")
            self._eyebrow_label.style().unpolish(self._eyebrow_label)
            self._eyebrow_label.style().polish(self._eyebrow_label)


    class _HuggingFaceStatusController:
        def __init__(self, chip_label: QLabel, summary_label: QLabel) -> None:
            self._chip_label = chip_label
            self._summary_label = summary_label

        def refresh(self, config: Mapping[str, Any]) -> _HuggingFaceStatusPresentation:
            return self._refresh_hf_status(config)

        def _huggingface_status_presentation(self, config: Mapping[str, Any]) -> _HuggingFaceStatusPresentation:
            username = str(config.get("hf_username", "")).strip()
            auth_present = _has_huggingface_auth_token()
            if auth_present and username:
                return _HuggingFaceStatusPresentation(
                    chip_text="logged in",
                    chip_state="success",
                    summary=username,
                    tooltip=f"Logged in to Hugging Face as {username}.",
                )
            if auth_present:
                return _HuggingFaceStatusPresentation(
                    chip_text="token found",
                    chip_state="running",
                    summary="set username in Config",
                    tooltip=(
                        "A Hugging Face token is present, but no username is saved. "
                        "Open Config or run hf auth whoami in Terminal to confirm the account."
                    ),
                )
            return _HuggingFaceStatusPresentation(
                chip_text="not logged in",
                chip_state="error",
                summary="run hf auth login",
                tooltip=(
                    "No Hugging Face login was detected. "
                    "In Terminal run hf auth login, paste your access token when prompted, then set your username in Config."
                ),
            )

        def _refresh_hf_status(self, config: Mapping[str, Any]) -> _HuggingFaceStatusPresentation:
            status = self._huggingface_status_presentation(config)
            self._chip_label.setText(status.chip_text)
            self._summary_label.setText(status.summary)
            return status


    class _TerminalTabManager(QObject):
        close_requested = Signal(int)
        tab_changed = Signal(int)

        def __init__(self, tab_widget: QTabWidget) -> None:
            super().__init__(tab_widget)
            self._tab_widget = tab_widget
            self._session_ids: list[int] = []
            self._titles: dict[int, str] = {}
            self._tab_widget.currentChanged.connect(self._on_terminal_tab_changed)

        def add_tab(self, *, session_id: int, widget: QWidget, title: str, tool_tip: str) -> int:
            self._session_ids.append(session_id)
            self._titles[session_id] = title
            index = self._tab_widget.addTab(widget, title)
            self._tab_widget.setCurrentIndex(index)
            self._tab_widget.setTabToolTip(index, tool_tip)
            self._rebuild_close_buttons()
            return index

        def close_session(self, session_id: int) -> int | None:
            try:
                index = self._session_ids.index(session_id)
            except ValueError:
                return None
            self.close_tab(index)
            return index

        def close_tab(self, index: int) -> int | None:
            if index < 0 or index >= len(self._session_ids):
                return None
            session_id = self._session_ids.pop(index)
            self._tab_widget.removeTab(index)
            self._titles.pop(session_id, None)
            self._rebuild_close_buttons()
            return session_id

        def rename_tab(self, session_id: int, title: str, *, tool_tip: str | None = None) -> None:
            try:
                index = self._session_ids.index(session_id)
            except ValueError:
                return
            self._titles[session_id] = title
            self._tab_widget.setTabText(index, title)
            if tool_tip is not None:
                self._tab_widget.setTabToolTip(index, tool_tip)
            self._rebuild_close_buttons()

        def session_id_for_index(self, index: int) -> int | None:
            if index < 0 or index >= len(self._session_ids):
                return None
            return self._session_ids[index]

        def current_session_id(self) -> int | None:
            return self.session_id_for_index(self._tab_widget.currentIndex())

        def session_count(self) -> int:
            return len(self._session_ids)

        def _rebuild_close_buttons(self) -> None:
            tab_bar = self._tab_widget.tabBar()
            hide_instead_of_close = len(self._session_ids) == 1
            for index, session_id in enumerate(self._session_ids):
                title = self._titles.get(session_id, f"Terminal {session_id}")
                close_button = QToolButton(tab_bar)
                close_button.setObjectName("TerminalTabCloseButton")
                close_button.setText("\u00d7")
                close_button.setAutoRaise(True)
                close_button.setToolTip(
                    "Hide the terminal panel" if hide_instead_of_close else f"Close {title}"
                )
                close_button.clicked.connect(
                    lambda _checked=False, sid=session_id: self.close_requested.emit(sid)
                )
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, close_button)

        def _on_terminal_tab_changed(self, index: int) -> None:
            self.tab_changed.emit(index)


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
            self._sidebar_controller = _SidebarController(
                window=self,
                config=self.config,
                collapsed=_config_bool(self.config.get("ui_sidebar_collapsed", False), False),
                persist_config=lambda config: save_config(config, quiet=True),
            )
            self._terminal_visible = _config_bool(self.config.get("ui_terminal_visible", False), False)
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
            self._workflow_queue = WorkflowQueueService(
                config=self.config,
                run_controller=self._run_controller,
                append_log=self.append_log,
            )

            self.setWindowTitle("LeRobot GUI")
            self.setMinimumSize(1080, 760)
            icon_path = find_app_icon_png()
            if icon_path is not None:
                self.setWindowIcon(QIcon(str(icon_path)))

            self._build_ui()
            self._pulse_controller = _WorkspacePulseController(self.workspace_meta_label)
            self._hf_status_controller = _HuggingFaceStatusController(
                self.hf_status_chip,
                self.hf_status_label,
            )
            self._refresh_huggingface_status()
            self._sidebar_controller.apply(announce=False, persist=False)
            self._apply_terminal_visibility(announce=False, persist=False, focus_terminal=False)
            self.apply_theme()
            self._apply_initial_geometry()
            self.select_section("record")
            self.append_log("LeRobot GUI initialized.")
            self.append_log("Core workflows and secondary pages are wired into the terminal workspace.")

        def _build_ui(self) -> None:
            root = QWidget()
            outer = QHBoxLayout(root)
            outer.setContentsMargins(SPACING_SHELL, SPACING_SHELL, SPACING_SHELL, SPACING_SHELL)
            outer.setSpacing(SPACING_SHELL)
            self.setCentralWidget(root)

            sidebar_widgets = build_sidebar(
                sections=self._sections,
                on_nav_changed=self._on_nav_changed,
                on_toggle_theme=self.toggle_theme_mode,
                on_toggle_sidebar=self.toggle_sidebar,
                on_toggle_terminal_panel=self.toggle_terminal_panel,
            )
            self.sidebar = sidebar_widgets.frame
            self.theme_button = sidebar_widgets.theme_button
            self.sidebar_collapse_button = sidebar_widgets.collapse_button
            self.nav_list = sidebar_widgets.nav_list
            self.sidebar_status = sidebar_widgets.status_label
            self.terminal_button = sidebar_widgets.terminal_button
            self._nav_widgets = sidebar_widgets.nav_widgets
            outer.addWidget(self.sidebar)

            rail_widgets = build_sidebar_rail(on_toggle_sidebar=self.toggle_sidebar)
            self.sidebar_rail = rail_widgets.frame
            self.sidebar_expand_button = rail_widgets.expand_button
            outer.addWidget(self.sidebar_rail)

            surface = QFrame()
            surface.setObjectName("ContentSurface")
            surface_layout = QVBoxLayout(surface)
            surface_layout.setContentsMargins(SPACING_SHELL, SPACING_SHELL, SPACING_SHELL, SPACING_SHELL)
            surface_layout.setSpacing(SPACING_SHELL)

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

        def _build_workspace_window(self) -> QFrame:
            window = QFrame()
            window.setObjectName("WorkspaceWindow")

            layout = QVBoxLayout(window)
            layout.setContentsMargins(SPACING_PANE, SPACING_PANE, SPACING_PANE, SPACING_PANE)
            layout.setSpacing(SPACING_COMPACT)

            header = QFrame()
            header.setObjectName("PaneHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(SPACING_COMPACT)

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

            self.hf_status_card = QFrame()
            self.hf_status_card.setObjectName("HeaderStatusBlock")
            self.hf_status_card.setMinimumWidth(210)
            self.hf_status_card.setMaximumWidth(240)
            account_layout = QVBoxLayout(self.hf_status_card)
            account_layout.setContentsMargins(SPACING_COMPACT, SPACING_CARD, SPACING_COMPACT, SPACING_CARD)
            account_layout.setSpacing(SPACING_META + 2)

            status_row = QHBoxLayout()
            status_row.setContentsMargins(0, 0, 0, 0)
            status_row.setSpacing(8)

            self.hf_status_title_label = QLabel("Hugging Face")
            self.hf_status_title_label.setObjectName("SectionMeta")
            status_row.addWidget(self.hf_status_title_label)
            status_row.addStretch(1)

            self.hf_status_chip = QLabel("checking")
            self.hf_status_chip.setObjectName("StatusChip")
            self.hf_status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hf_status_chip.setMaximumWidth(120)
            status_row.addWidget(self.hf_status_chip)
            account_layout.addLayout(status_row)

            self.hf_status_label = QLabel("Checking login...")
            self.hf_status_label.setObjectName("HeaderStatusSummary")
            self.hf_status_label.setWordWrap(True)
            self.hf_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            account_layout.addWidget(self.hf_status_label)

            header_layout.addWidget(
                self.hf_status_card,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            )

            layout.addWidget(header)
            layout.addWidget(self.page_stack, 1)
            return window

        def _build_terminal_window(self) -> QFrame:
            window = QFrame()
            window.setObjectName("TerminalWindow")

            layout = QVBoxLayout(window)
            layout.setContentsMargins(SPACING_PANE, SPACING_PANE, SPACING_PANE, SPACING_PANE)
            layout.setSpacing(SPACING_COMPACT)

            header = QFrame()
            header.setObjectName("PaneHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(SPACING_COMPACT)

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
            self.terminal_tabs.tabBar().setDrawBase(False)
            self._terminal_tab_manager = _TerminalTabManager(self.terminal_tabs)
            self._terminal_tab_manager.close_requested.connect(self.close_terminal_session)
            self._terminal_tab_manager.tab_changed.connect(self._on_terminal_tab_changed)

            self.new_terminal_tab_button = QToolButton()
            self.new_terminal_tab_button.setObjectName("TerminalTabAddButton")
            self.new_terminal_tab_button.setText("+")
            self.new_terminal_tab_button.setToolTip("open a new terminal session")
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
                on_config_changed=self._handle_config_changed,
                workflow_queue=self._workflow_queue,
            )
            if secondary_panel is not None:
                return self._wrap_panel(secondary_panel)

            scroll = self._build_page_scroll_area()

            body = QWidget()
            layout = QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(SPACING_SHELL)

            cards = QGridLayout()
            cards.setHorizontalSpacing(16)
            cards.setVerticalSpacing(16)
            cards.addWidget(_build_text_card("Why this page matters", section.summary), 0, 0)
            cards.addWidget(_build_text_card("Current port focus", section.focus), 0, 1)
            cards.addWidget(
                _build_text_card(
                    "Shared layer reused now",
                    "Theme tokens, command formatting/parsing, and latest-job coordination are already toolkit-neutral.",
                ),
                1,
                0,
            )
            cards.addWidget(
                _build_text_card(
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
            highlights_layout.setContentsMargins(SPACING_SHELL, SPACING_SHELL, SPACING_SHELL, SPACING_SHELL)
            highlights_layout.setSpacing(SPACING_CARD)

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
            if not hasattr(self, "_terminal_tab_manager"):
                return None
            session_id = self._terminal_tab_manager.current_session_id()
            if session_id is None:
                return None
            return self._session_for_id(session_id)

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
            self._terminal_tab_manager.add_tab(
                session_id=session_id,
                widget=panel,
                title=title,
                tool_tip="Interactive LeRobot shell",
            )

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
            was_last_session = len(self._terminal_sessions) == 1
            session = self._terminal_sessions.pop(index)
            widget = session.panel
            self._terminal_tab_manager.close_tab(index)
            if session.shell is not None:
                session.shell.append_terminal_output = None
                session.shell.set_status_message = None
                session.shell.shutdown()
            if widget is not None:
                widget.deleteLater()
            self.append_log(f"Closed {session.title}.")

            self._sync_active_terminal_status()
            if was_last_session:
                self._terminal_visible = False
                self._apply_terminal_visibility(announce=True, persist=True, focus_terminal=False)
                return

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
            return self._terminal_tab_manager.session_count()

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
            self.theme_button.setToolTip("toggle light / dark theme")
            self._sidebar_controller.refresh_buttons()
            self._refresh_terminal_button()

        def _refresh_terminal_button(self) -> None:
            if not hasattr(self, "terminal_button"):
                return
            if self._terminal_visible:
                self.terminal_button.setText("Hide Terminal")
                self.terminal_button.setToolTip("Hide the terminal panel")
            else:
                self.terminal_button.setText("Show Terminal")
                self.terminal_button.setToolTip("Show the terminal panel")

        def _persist_terminal_visibility(self) -> None:
            self.config["ui_terminal_visible"] = self._terminal_visible
            save_config(self.config, quiet=True)

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
                if not self._terminal_sessions:
                    self._create_terminal_session(focus=False)
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
                self._pulse_controller.start()
                self.sidebar_status.setText("A workflow is currently running.")
            else:
                self._pulse_controller.stop()
                self.sidebar_status.setText("Ready for a new workflow.")
                self._workflow_queue.start_if_idle()

        def _set_status_chip_state(self, label: QLabel, state: str) -> None:
            label.setProperty("state", state)
            label.style().unpolish(label)
            label.style().polish(label)

        def _update_workspace_header(self, section: QtSectionDefinition) -> None:
            self.workspace_meta_label.setText(section.stage)
            self.workspace_title_label.setText(section.title)
            self.workspace_subtitle_label.setText(section.subtitle)

        def _refresh_huggingface_status(self) -> None:
            if not hasattr(self, "hf_status_label"):
                return
            status = self._hf_status_controller.refresh(self.config)
            self._set_status_chip_state(self.hf_status_chip, status.chip_state)
            for widget in (
                self.hf_status_card,
                self.hf_status_title_label,
                self.hf_status_chip,
                self.hf_status_label,
            ):
                widget.setToolTip(status.tooltip)

        def _handle_config_changed(self) -> None:
            self._refresh_huggingface_status()
            row = self.nav_list.currentRow()
            if row >= 0:
                self._refresh_visible_page_runtime_state(row)

        def apply_theme(self) -> None:
            self.colors = build_theme_colors(ui_font="Inter", mono_font="JetBrains Mono", theme_mode=self.theme_mode)
            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet(build_qt_stylesheet(self.colors))
            self._refresh_theme_button()
            QTimer.singleShot(0, self._refresh_all_terminal_geometries)

        def toggle_theme_mode(self) -> None:
            self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
            self.config["ui_theme_mode"] = self.theme_mode
            save_config(self.config)
            self.apply_theme()
            self.append_log(f"Theme switched to {self.theme_mode}.")

        def toggle_sidebar(self) -> None:
            self._sidebar_controller.toggle()

        def toggle_terminal_panel(self) -> None:
            self._terminal_visible = not self._terminal_visible
            self._apply_terminal_visibility(announce=True, persist=True, focus_terminal=self._terminal_visible)

        def sidebar_collapsed(self) -> bool:
            return self._sidebar_controller.collapsed

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
            if section_id == "queue":
                section_id = "workflows"
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
            if not self._terminal_visible:
                self._terminal_visible = True
                self._apply_terminal_visibility(announce=True, persist=True, focus_terminal=False)
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
                self._pulse_controller.stop()
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
    def _print_gui_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        traceback.print_exception(exc_type, exc_value, exc_tb)

    def _print_thread_exception(args: threading.ExceptHookArgs) -> None:
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _print_gui_exception
    threading.excepthook = _print_thread_exception
    ok, detail = qt_available()
    if not ok:
        print("GUI is unavailable on this device.")
        print(f"Details: {detail}")
        return

    try:
        app, _created = ensure_qt_application(sys.argv)
    except RuntimeError as exc:
        print("GUI failed to start.")
        print(f"Details: {exc}")
        return
    app.setApplicationName("LeRobot GUI")
    window = create_qt_preview_window(raw_config)
    window.show()
    app.exec()
