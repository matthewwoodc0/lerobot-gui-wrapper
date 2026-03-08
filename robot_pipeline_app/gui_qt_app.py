from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .app_icon import find_app_icon_png
from .app_theme import build_theme_colors, normalize_theme_mode
from .camera_state import camera_mapping_summary
from .config_store import normalize_config_without_prompts
from .gui_qt_theme import build_qt_stylesheet

try:
    from PySide6.QtCore import Qt
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
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    _QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised through availability helpers
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
        summary="First migration target because it couples live process control, validation, and data browser flows.",
        focus="Port command assembly, preflight, preview/edit dialogs, and dataset browser models next.",
        status="Queued for parity pass",
        highlights=(
            "Interactive command preview/edit flow already extracted into shared command-text helpers.",
            "Dataset metadata and parity helpers remain reusable from the existing app.",
            "Camera preview and process stdin handling are the main risk areas.",
        ),
    ),
    QtSectionDefinition(
        id="deploy",
        title="Deploy",
        subtitle="Local model selection, eval runs, and deployment diagnostics.",
        stage="Core ops",
        summary="Second migration target after the Qt shell is stable because it shares most of the hard runtime behavior.",
        focus="Reuse run-controller orchestration and keep model browser logic thin in the view layer.",
        status="Queued for parity pass",
        highlights=(
            "Model tree + parity popouts map cleanly to Qt model/view widgets.",
            "Runtime diagnostics and artifact writing already live outside the widget tree.",
            "Keyboard help, cancellation, and eval outcome handling still need Qt equivalents.",
        ),
    ),
    QtSectionDefinition(
        id="teleop",
        title="Teleop",
        subtitle="Robot connection setup and live teleoperation launch.",
        stage="Core ops",
        summary="Port alongside Record/Deploy so the process-control layer is validated against all live workflows early.",
        focus="Keep the launch surface compact and share the same command preview + cancel path.",
        status="Queued for parity pass",
        highlights=(
            "Teleop uses the same validation and process infrastructure as record/deploy.",
            "Camera pause/resume behavior will be driven by the shared controller seam.",
            "Run popout UX is still a placeholder in this preview shell.",
        ),
    ),
    QtSectionDefinition(
        id="config",
        title="Config",
        subtitle="Environment setup, diagnostics, and launcher management.",
        stage="Secondary",
        summary="Follows core ops once the shared shell is locked in.",
        focus="Replace Tk dialogs and launcher checks after the preview shell proves out.",
        status="Waiting on core shell",
        highlights=(
            "Config normalization already sits outside the GUI and is immediately reusable.",
            "Tk-specific folder pickers and launcher validation are deferred until the cutover phase.",
            "Doctor output can move over with minimal business-logic churn.",
        ),
    ),
    QtSectionDefinition(
        id="training",
        title="Training",
        subtitle="Train command generation and HIL workflow support.",
        stage="Secondary",
        summary="Mostly form-driven and a good fit for later Qt form components once the shell and log pane settle.",
        focus="Preserve the current workflow, but present generated commands in a cleaner split-panel layout.",
        status="Waiting on core shell",
        highlights=(
            "Training command builders are already strongly separated from the UI layer.",
            "Clipboard and dialog replacements are straightforward once shared dialogs land.",
            "No live camera dependency keeps this port relatively low risk.",
        ),
    ),
    QtSectionDefinition(
        id="visualizer",
        title="Visualizer",
        subtitle="Read-only browsing for datasets, deploy sources, and videos.",
        stage="Secondary",
        summary="A strong follow-up target after core ops because it benefits most from Qt's model/view widgets.",
        focus="Move source/video tables to Qt models and keep data collection fully shared.",
        status="Waiting on core shell",
        highlights=(
            "Existing helper functions already cover most source-discovery logic.",
            "Video and insight tables map naturally to Qt item views.",
            "Background refresh behavior can reuse the extracted latest-job runner.",
        ),
    ),
    QtSectionDefinition(
        id="history",
        title="History",
        subtitle="Run artifacts, reruns, and deployment notes.",
        stage="Secondary",
        summary="Can shift once command parsing, log output, and rerun flow are solid in the Qt shell.",
        focus="Share the history payload builder and port only the view/editor mechanics.",
        status="Waiting on core shell",
        highlights=(
            "History payload shaping already lives outside widget rendering.",
            "Qt table widgets should simplify filter + details coordination.",
            "Notes and rerun dialogs still need dedicated Qt implementations.",
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

    class _QtLogPanel(QFrame):
        def __init__(self) -> None:
            super().__init__()
            self.setObjectName("SectionCard")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(10)

            title = QLabel("Migration Log")
            title.setObjectName("SectionMeta")
            layout.addWidget(title)

            self._text = QPlainTextEdit()
            self._text.setReadOnly(True)
            self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self._text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self._text, 1)

        def append_log(self, message: str) -> None:
            self._text.appendPlainText(str(message))

        def contents(self) -> str:
            return self._text.toPlainText()


    class QtPreviewWindow(QMainWindow):
        def __init__(self, *, config: dict[str, Any]) -> None:
            super().__init__()
            self.config = dict(config)
            self.theme_mode = normalize_theme_mode(config.get("ui_theme_mode", "dark"))
            self.colors = build_theme_colors(ui_font="Inter", mono_font="JetBrains Mono", theme_mode=self.theme_mode)
            self._sections = qt_preview_sections()
            self._section_index = {section.id: idx for idx, section in enumerate(self._sections)}

            self.setWindowTitle("LeRobot Pipeline Manager [Qt Preview]")
            self.setMinimumSize(1080, 760)
            icon_path = find_app_icon_png()
            if icon_path is not None:
                self.setWindowIcon(QIcon(str(icon_path)))

            self._build_ui()
            self.apply_theme()
            self._apply_initial_geometry()
            self.select_section("record")
            self.append_log("Qt preview shell initialized.")
            self.append_log("Tk remains the default GUI while shared services and parity work continue.")

        def _build_ui(self) -> None:
            root = QWidget()
            outer = QHBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 18)
            outer.setSpacing(18)
            self.setCentralWidget(root)

            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(300)
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(20, 20, 20, 20)
            sidebar_layout.setSpacing(14)

            brand_row = QHBoxLayout()
            brand_row.setSpacing(0)
            brand = QLabel("LeRobot")
            brand.setObjectName("BrandLabel")
            brand_suffix = QLabel(" Qt Preview")
            brand_suffix.setObjectName("BrandSuffix")
            brand_row.addWidget(brand)
            brand_row.addWidget(brand_suffix)
            brand_row.addStretch(1)
            sidebar_layout.addLayout(brand_row)

            preview_note = QLabel("Separate migration shell with shared theme + command-text foundation.")
            preview_note.setWordWrap(True)
            preview_note.setObjectName("MutedLabel")
            sidebar_layout.addWidget(preview_note)

            self.theme_button = QPushButton()
            self.theme_button.setObjectName("AccentButton")
            self.theme_button.clicked.connect(self.toggle_theme_mode)
            sidebar_layout.addWidget(self.theme_button)

            self.nav_list = QListWidget()
            self.nav_list.setSpacing(2)
            self.nav_list.currentRowChanged.connect(self._on_nav_changed)
            for section in self._sections:
                item = QListWidgetItem(f"{section.title}\n{section.status}")
                self.nav_list.addItem(item)
            sidebar_layout.addWidget(self.nav_list, 1)

            shell_status = QLabel("Shell status")
            shell_status.setObjectName("SectionMeta")
            sidebar_layout.addWidget(shell_status)

            self.sidebar_status = QLabel(
                "Sweep 3 in progress: shared foundations extracted, Qt shell online, workflow pages staged."
            )
            self.sidebar_status.setWordWrap(True)
            self.sidebar_status.setObjectName("MutedLabel")
            sidebar_layout.addWidget(self.sidebar_status)

            outer.addWidget(sidebar)

            surface = QFrame()
            surface.setObjectName("ContentSurface")
            surface_layout = QVBoxLayout(surface)
            surface_layout.setContentsMargins(20, 20, 20, 20)
            surface_layout.setSpacing(18)

            self.page_stack = QStackedWidget()
            for section in self._sections:
                self.page_stack.addWidget(self._build_page(section))

            self.log_panel = _QtLogPanel()

            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.addWidget(self.page_stack)
            splitter.addWidget(self.log_panel)
            splitter.setStretchFactor(0, 4)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([620, 220])

            surface_layout.addWidget(splitter, 1)
            outer.addWidget(surface, 1)

            status = QStatusBar()
            status.showMessage("Qt preview ready.")
            self.setStatusBar(status)

        def _build_page(self, section: QtSectionDefinition) -> QWidget:
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

            highlights_title = QLabel("Migration Notes")
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
            self.theme_button.setText(f"Switch to {target.title()} Theme")

        def _on_nav_changed(self, row: int) -> None:
            if row < 0 or row >= len(self._sections):
                return
            self.page_stack.setCurrentIndex(row)
            section = self._sections[row]
            self.statusBar().showMessage(f"{section.title}: {section.status}")
            self.append_log(f"Switched to {section.title} preview.")

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

        def append_log(self, message: str) -> None:
            self.log_panel.append_log(message)

        def log_contents(self) -> str:
            return self.log_panel.contents()

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
        print("Qt preview GUI is unavailable on this device.")
        print(f"Details: {detail}")
        return

    app, _created = ensure_qt_application(sys.argv)
    app.setApplicationName("LeRobot Pipeline Manager")
    window = create_qt_preview_window(raw_config)
    window.show()
    app.exec()
