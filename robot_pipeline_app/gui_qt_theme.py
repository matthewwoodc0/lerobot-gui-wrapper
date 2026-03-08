from __future__ import annotations


def build_qt_stylesheet(colors: dict[str, str]) -> str:
    return f"""
QMainWindow {{
    background: {colors["bg"]};
}}
QWidget {{
    background: {colors["bg"]};
    color: {colors["text"]};
    font-family: "{colors["font_ui"]}";
    font-size: 10pt;
}}
QFrame#Sidebar {{
    background: {colors["header"]};
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
    background: {colors["surface_elevated"]};
    border: 1px solid {colors["border"]};
    border-radius: 22px;
}}
QLabel#BrandLabel {{
    color: {colors["accent"]};
    font-size: 22pt;
    font-weight: 700;
}}
QLabel#BrandSuffix {{
    color: {colors["text"]};
    font-size: 22pt;
    font-weight: 700;
}}
QLabel#PageTitle {{
    color: {colors["text"]};
    font-size: 19pt;
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
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 14px;
    margin: 0 0 8px 0;
    padding: 12px;
}}
QListWidget::item:selected {{
    background: {colors["accent"]};
    color: #000000;
    border: 1px solid {colors["accent"]};
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
    background: {colors["header"]};
    color: {colors["muted"]};
}}
QScrollBar:vertical {{
    background: {colors["panel"]};
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
