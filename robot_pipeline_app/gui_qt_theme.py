from __future__ import annotations

from .app_theme import (
    RADIUS_BUTTON,
    RADIUS_CARD,
    RADIUS_CHIP,
    RADIUS_DIALOG,
    RADIUS_NAV,
    RADIUS_PANE,
    RADIUS_SHELL,
)


def build_qt_stylesheet(colors: dict[str, str]) -> str:
    error_chip_bg = "#fde8e8" if colors["theme_mode"] == "light" else "#3a1212"
    success_chip_bg = "#e6f7ed" if colors["theme_mode"] == "light" else "#0d2e1a"
    return f"""
QMainWindow {{
    background: {colors["bg"]};
}}
QDialog#AppDialog {{
    background: {colors["bg"]};
}}
QWidget {{
    color: {colors["text"]};
    font-family: "{colors["font_ui"]}";
    font-size: 10pt;
}}
QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}
QFrame#Sidebar {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_PANE}px;
}}
QFrame#SidebarRail {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_PANE}px;
}}
QFrame#ContentSurface {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_SHELL}px;
}}
QFrame#WorkspaceWindow,
QFrame#TerminalWindow {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_PANE}px;
}}
QFrame#PaneHeader,
QFrame#TerminalPanel {{
    background: transparent;
    border: none;
}}
QFrame#DialogPanel {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_DIALOG}px;
}}
QFrame#DialogHeader,
QFrame#DialogFooter,
QFrame#DialogActionBar {{
    background: transparent;
    border: none;
}}
QFrame#SectionCard {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CARD}px;
}}
QFrame#SectionHero {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_DIALOG}px;
}}
QFrame#HeaderStatusBlock {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CARD}px;
}}
QLabel#BrandLabel {{
    color: {colors["accent"]};
    font-size: 22pt;
    font-weight: 700;
}}
QLabel#PaneEyebrow {{
    color: {colors["accent"]};
    font-size: 8.5pt;
    font-weight: 700;
    text-transform: uppercase;
}}
QLabel#PaneEyebrow[pulsing="dim"] {{
    color: {colors["running_dim"]};
}}
QLabel#PaneTitle {{
    color: {colors["text"]};
    font-size: 17pt;
    font-weight: 700;
}}
QLabel#PaneSubtitle {{
    color: {colors["muted"]};
    font-size: 9.5pt;
}}
QLabel#HeaderStatusSummary {{
    color: {colors["text"]};
    font-size: 9pt;
    font-weight: 600;
}}
QLabel#DialogTitle {{
    color: {colors["text"]};
    font-size: 16pt;
    font-weight: 700;
}}
QLabel#DialogSubtitle {{
    color: {colors["muted"]};
    font-size: 9.5pt;
}}
QLabel#PageTitle {{
    color: {colors["text"]};
    font-size: 19pt;
    font-weight: 700;
}}
QLabel#FormLabel {{
    color: {colors["muted"]};
    font-size: 9pt;
    font-weight: 700;
}}
QLabel#SectionMeta {{
    color: {colors["accent"]};
    font-size: 9pt;
    font-weight: 700;
    text-transform: uppercase;
}}
QLabel#MutedLabel {{
    color: {colors["muted"]};
}}
QLabel#DialogErrorLabel {{
    color: {colors["error"]};
    font-weight: 700;
}}
QLabel#StatusChip {{
    background: {colors["accent_soft"]};
    color: {colors["accent"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CHIP}px;
    padding: 3px 8px;
    font-weight: 700;
}}
QLabel#StatusChip[state="running"] {{
    background: {colors["accent_soft"]};
    color: {colors["accent"]};
    border-color: {colors["accent"]};
}}
QLabel#StatusChip[state="error"] {{
    background: {error_chip_bg};
    color: {colors["error"]};
    border-color: {colors["error"]};
}}
QLabel#StatusChip[state="success"] {{
    background: {success_chip_bg};
    color: {colors["success"]};
    border-color: {colors["success"]};
}}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    margin: 0 0 8px 0;
    padding: 0;
    background: transparent;
    border: none;
}}
QListWidget::item:selected,
QListWidget::item:selected:active,
QListWidget::item:selected:!active,
QListWidget::item:hover {{
    background: transparent;
    border: none;
    outline: none;
}}
QFrame#NavItem {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_NAV}px;
}}
QFrame#NavItem[selected="true"] {{
    background: {colors["accent"]};
    border-color: {colors["accent"]};
}}
QFrame#NavItem:hover {{
    background: {colors["accent_soft"]};
    border-color: {colors["accent"]};
}}
QLabel#NavItemTitle {{
    color: {colors["text"]};
    font-size: 11.5pt;
    font-weight: 800;
}}
QLabel#NavItemTitle[selected="true"] {{
    color: #000000;
}}
QLabel#NavItemMeta {{
    color: {colors["muted"]};
    font-size: 8.5pt;
    font-weight: 600;
}}
QLabel#NavItemMeta[selected="true"] {{
    color: #2f2100;
}}
QPushButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    padding: 8px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {colors["accent"]};
}}
QPushButton#AccentButton {{
    background: {colors["accent"]};
    color: #000000;
    border-color: {colors["accent"]};
}}
QPushButton#AccentButton:hover {{
    background: {colors["accent_dark"]};
}}
QPushButton#AccentButton:pressed {{
    background: #a87200;
}}
QPushButton[assigned="true"] {{
    background: {colors["accent"]};
    color: #000000;
    border-color: {colors["accent"]};
}}
QPushButton[assigned="true"]:hover {{
    background: {colors["accent_dark"]};
}}
QPushButton[assigned="true"]:pressed {{
    background: #a87200;
}}
QPushButton#DangerButton {{
    background: {colors["error"]};
    color: #ffffff;
    border-color: {colors["error"]};
}}
QPushButton#DangerButton:hover {{
    background: #c73333;
}}
QPushButton#DangerButton:pressed {{
    background: #a82a2a;
}}
QPushButton#TerminalToggleButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    padding: 7px 12px;
    font-size: 9pt;
    font-weight: 700;
}}
QPushButton#TerminalToggleButton:hover {{
    border-color: {colors["accent"]};
}}
QPushButton#TerminalChromeButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    padding: 7px 10px;
    font-size: 8.75pt;
    font-weight: 700;
}}
QPushButton#TerminalChromeButton:hover {{
    border-color: {colors["accent"]};
}}
QToolButton#TerminalTabAddButton,
QToolButton#TerminalTabCloseButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    font-size: 9pt;
    font-weight: 700;
    padding: 0;
}}
QToolButton#TerminalTabAddButton {{
    min-width: 24px;
    min-height: 24px;
}}
QToolButton#TerminalTabCloseButton {{
    min-width: 18px;
    min-height: 18px;
}}
QToolButton#TerminalTabAddButton:hover,
QToolButton#TerminalTabCloseButton:hover {{
    border-color: {colors["accent"]};
    color: {colors["accent"]};
}}
QPushButton#ThemeToggleButton {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    font-size: 11pt;
    padding: 0;
}}
QPushButton#ThemeToggleButton:hover {{
    border-color: {colors["accent"]};
}}
QPushButton#SidebarChromeButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_BUTTON}px;
    font-size: 10pt;
    font-weight: 700;
    padding: 0;
}}
QPushButton#SidebarChromeButton:hover {{
    border-color: {colors["accent"]};
}}
QPushButton:disabled {{
    color: {colors["muted"]};
    border-color: {colors["border"]};
}}
QLineEdit,
QComboBox,
QSpinBox,
QTableWidget,
QTreeWidget {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_NAV}px;
    min-height: 22px;
    padding: 7px 9px;
    selection-background-color: {colors["accent"]};
    selection-color: #000000;
}}
QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QTableWidget:focus,
QTreeWidget:focus {{
    border-color: {colors["accent"]};
}}
QComboBox::drop-down,
QSpinBox::up-button,
QSpinBox::down-button {{
    border: none;
    background: transparent;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    selection-background-color: {colors["accent"]};
    selection-color: #000000;
    outline: none;
}}
QTableWidget,
QTreeWidget {{
    alternate-background-color: {colors["surface"]};
    gridline-color: {colors["border"]};
}}
QTableWidget::item:selected,
QTreeWidget::item:selected {{
    background: {colors["accent"]};
    color: #000000;
}}
QHeaderView::section {{
    background: {colors["surface"]};
    color: {colors["muted"]};
    border: none;
    border-bottom: 1px solid {colors["border"]};
    border-right: 1px solid {colors["border"]};
    padding: 8px;
    font-weight: 700;
}}
QCheckBox {{
    spacing: 10px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {colors["border"]};
    border-radius: 6px;
    background: {colors["surface"]};
}}
QCheckBox::indicator:checked {{
    background: {colors["accent"]};
    border-color: {colors["accent"]};
}}
QPlainTextEdit {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CARD}px;
    selection-background-color: {colors["accent"]};
    font-family: "{colors["font_mono"]}";
    font-size: 10pt;
    padding: 10px;
}}
QPlainTextEdit#DialogText {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CARD}px;
    selection-background-color: {colors["accent_soft"]};
    selection-color: {colors["text"]};
    font-family: "{colors["font_mono"]}";
    font-size: 10pt;
    padding: 10px;
}}
QFrame#TerminalPanel QPlainTextEdit,
QPlainTextEdit#EmbeddedTerminal {{
    background: {colors["header"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {RADIUS_CARD}px;
    selection-background-color: {colors["accent_soft"]};
    selection-color: {colors["text"]};
    font-family: "{colors["font_mono"]}";
    font-size: 10pt;
    padding: 10px;
}}
QTabWidget#TerminalTabs {{
    background: transparent;
}}
QTabWidget#TerminalTabs::pane {{
    border: none;
    background: transparent;
    margin: 0;
    padding: 0;
}}
QTabWidget#TerminalTabs QTabBar {{
    background: transparent;
    border: none;
}}
QTabWidget#TerminalTabs QTabBar::tab {{
    background: {colors["surface"]};
    color: {colors["muted"]};
    border: 1px solid {colors["border"]};
    border-bottom: none;
    border-top-left-radius: {RADIUS_BUTTON}px;
    border-top-right-radius: {RADIUS_BUTTON}px;
    padding: 7px 10px;
    margin-right: 6px;
    margin-bottom: 2px;
    font-weight: 700;
}}
QTabWidget#TerminalTabs QTabBar::tab:selected {{
    background: {colors["header"]};
    color: {colors["text"]};
    border-color: {colors["accent"]};
}}
QTabWidget#TerminalTabs QTabBar::tab:hover {{
    color: {colors["text"]};
    border-color: {colors["accent"]};
}}
QStatusBar {{
    background: {colors["panel"]};
    color: {colors["muted"]};
}}
QSplitter::handle {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 4px 0 4px 0;
}}
QScrollBar::handle:vertical {{
    background: {colors["scrollbar_handle"]};
    border-radius: 6px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {colors["accent_soft"]};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""
