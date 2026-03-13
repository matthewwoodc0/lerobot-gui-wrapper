from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from .app_theme import SPACING_COMPACT, SPACING_PANE


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


@dataclass(frozen=True)
class _SidebarWidgets:
    frame: QFrame
    theme_button: QPushButton
    collapse_button: QPushButton
    nav_list: QListWidget
    status_label: QLabel
    terminal_button: QPushButton
    nav_widgets: list[_NavItemWidget]


@dataclass(frozen=True)
class _SidebarRailWidgets:
    frame: QFrame
    expand_button: QPushButton


class _SidebarController:
    def __init__(
        self,
        *,
        window: Any,
        config: dict[str, Any],
        collapsed: bool,
        persist_config: Callable[[dict[str, Any]], None],
    ) -> None:
        self._window = window
        self._config = config
        self._persist_config = persist_config
        self.collapsed = collapsed

    def refresh_buttons(self) -> None:
        if hasattr(self._window, "sidebar_collapse_button"):
            self._window.sidebar_collapse_button.setText("<")
            self._window.sidebar_collapse_button.setToolTip("collapse sidebar")
        if hasattr(self._window, "sidebar_expand_button"):
            self._window.sidebar_expand_button.setText(">")
            self._window.sidebar_expand_button.setToolTip("expand sidebar")

    def persist(self) -> None:
        self._config["ui_sidebar_collapsed"] = self.collapsed
        self._persist_config(self._config)

    def apply(self, *, announce: bool, persist: bool) -> None:
        if not hasattr(self._window, "sidebar") or not hasattr(self._window, "sidebar_rail"):
            return
        if self.collapsed:
            self._window.sidebar.hide()
            self._window.sidebar_rail.show()
            if announce:
                self._window.statusBar().showMessage("Sidebar minimized.")
                self._window.append_log("Sidebar minimized.")
        else:
            self._window.sidebar.show()
            self._window.sidebar_rail.hide()
            if announce:
                self._window.statusBar().showMessage("Sidebar expanded.")
                self._window.append_log("Sidebar expanded.")
        if persist:
            self.persist()
        self.refresh_buttons()

    def toggle(self) -> None:
        self.collapsed = not self.collapsed
        self.apply(announce=True, persist=True)


def build_sidebar(
    *,
    sections: Sequence[Any],
    on_nav_changed: Callable[[int], None],
    on_toggle_theme: Callable[[], None],
    on_toggle_sidebar: Callable[[], None],
    on_toggle_terminal_panel: Callable[[], None],
) -> _SidebarWidgets:
    sidebar = QFrame()
    sidebar.setObjectName("Sidebar")
    sidebar.setFixedWidth(280)

    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(SPACING_PANE, SPACING_PANE, SPACING_PANE, SPACING_PANE)
    sidebar_layout.setSpacing(SPACING_COMPACT)

    brand = QLabel("LeRobot GUI")
    brand.setObjectName("BrandLabel")
    title_row = QHBoxLayout()
    title_row.setContentsMargins(0, 0, 0, 0)
    title_row.setSpacing(8)
    title_row.addWidget(brand)
    title_row.addStretch(1)

    theme_button = QPushButton()
    theme_button.setObjectName("ThemeToggleButton")
    theme_button.setFixedSize(30, 30)
    theme_button.setToolTip("toggle light / dark theme")
    theme_button.clicked.connect(on_toggle_theme)
    title_row.addWidget(theme_button)

    collapse_button = QPushButton()
    collapse_button.setObjectName("SidebarChromeButton")
    collapse_button.setFixedSize(30, 30)
    collapse_button.setToolTip("collapse sidebar")
    collapse_button.clicked.connect(on_toggle_sidebar)
    title_row.addWidget(collapse_button)
    sidebar_layout.addLayout(title_row)

    nav_list = QListWidget()
    nav_list.setSpacing(2)
    nav_list.currentRowChanged.connect(on_nav_changed)

    nav_widgets: list[_NavItemWidget] = []
    for section in sections:
        item = QListWidgetItem()
        nav_widget = _NavItemWidget(title=section.title, status=section.status)
        nav_widget.setToolTip(section.subtitle)
        item.setSizeHint(nav_widget.sizeHint())
        nav_list.addItem(item)
        nav_list.setItemWidget(item, nav_widget)
        nav_widgets.append(nav_widget)
    sidebar_layout.addWidget(nav_list, 1)

    shell_status = QLabel("Shell status")
    shell_status.setObjectName("SectionMeta")
    sidebar_layout.addWidget(shell_status)

    status_label = QLabel("Ready for local record, replay, deploy, motor setup, workflows, and analysis.")
    status_label.setWordWrap(True)
    status_label.setObjectName("MutedLabel")
    sidebar_layout.addWidget(status_label)

    terminal_button = QPushButton()
    terminal_button.setObjectName("TerminalToggleButton")
    terminal_button.setToolTip("show terminal panel")
    terminal_button.clicked.connect(on_toggle_terminal_panel)
    sidebar_layout.addWidget(terminal_button)

    return _SidebarWidgets(
        frame=sidebar,
        theme_button=theme_button,
        collapse_button=collapse_button,
        nav_list=nav_list,
        status_label=status_label,
        terminal_button=terminal_button,
        nav_widgets=nav_widgets,
    )


def build_sidebar_rail(*, on_toggle_sidebar: Callable[[], None]) -> _SidebarRailWidgets:
    rail = QFrame()
    rail.setObjectName("SidebarRail")
    rail.setFixedWidth(56)

    rail_layout = QVBoxLayout(rail)
    rail_layout.setContentsMargins(8, 12, 8, 12)
    rail_layout.setSpacing(12)

    expand_button = QPushButton()
    expand_button.setObjectName("SidebarChromeButton")
    expand_button.setFixedSize(40, 40)
    expand_button.setToolTip("expand sidebar")
    expand_button.clicked.connect(on_toggle_sidebar)
    rail_layout.addWidget(
        expand_button,
        0,
        Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
    )
    rail_layout.addStretch(1)
    return _SidebarRailWidgets(frame=rail, expand_button=expand_button)
