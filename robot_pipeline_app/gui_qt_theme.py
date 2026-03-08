from __future__ import annotations


def build_qt_stylesheet(colors: dict[str, str]) -> str:
    return f"""
QMainWindow {{
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
    border-radius: 20px;
}}
QFrame#ContentSurface {{
    background: {colors["panel"]};
    border: 1px solid {colors["border"]};
    border-radius: 24px;
}}
QFrame#SectionCard {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 18px;
}}
QFrame#SectionHero {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 22px;
}}
QLabel#BrandLabel {{
    color: {colors["accent"]};
    font-size: 22pt;
    font-weight: 700;
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
QLabel#StatusChip {{
    background: {colors["accent_soft"]};
    color: {colors["accent"]};
    border: 1px solid {colors["border"]};
    border-radius: 11px;
    padding: 4px 10px;
    font-weight: 700;
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
    border-radius: 14px;
}}
QFrame#NavItem[selected="true"] {{
    background: {colors["accent"]};
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
    border-radius: 12px;
    padding: 10px 14px;
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
QPushButton#TerminalToggleButton {{
    background: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 9pt;
    font-weight: 700;
}}
QPushButton#TerminalToggleButton:hover {{
    border-color: {colors["accent"]};
}}
QPushButton#ThemeToggleButton {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 15px;
    font-size: 11pt;
    padding: 0;
}}
QPushButton#ThemeToggleButton:hover {{
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
    border-radius: 14px;
    min-height: 22px;
    padding: 8px 10px;
    selection-background-color: {colors["accent"]};
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
QTableWidget,
QTreeWidget {{
    alternate-background-color: {colors["surface"]};
    gridline-color: {colors["border"]};
}}
QHeaderView::section {{
    background: {colors["surface"]};
    color: {colors["muted"]};
    border: 1px solid {colors["border"]};
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
    border-radius: 16px;
    selection-background-color: {colors["accent"]};
    font-family: "{colors["font_mono"]}";
    font-size: 10pt;
    padding: 12px;
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
    background: {colors["surface_alt"]};
    border-radius: 6px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""
