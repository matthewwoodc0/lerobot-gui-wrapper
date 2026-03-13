from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from .app_theme import SPACING_CARD, SPACING_COMPACT, SPACING_SHELL


def _build_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("SectionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(SPACING_SHELL, SPACING_SHELL, SPACING_SHELL, SPACING_SHELL)
    layout.setSpacing(SPACING_CARD)

    header = QLabel(title)
    header.setObjectName("SectionMeta")
    layout.addWidget(header)
    return card, layout


class _InputGrid:
    def __init__(self, layout: QVBoxLayout) -> None:
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACING_COMPACT)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self._index = 0
        layout.addLayout(self._grid)

    def add_field(self, label_text: str, widget: QWidget) -> None:
        row = self._index // 2
        pair = self._index % 2
        label_col = pair * 2
        widget_col = label_col + 1
        label = QLabel(label_text)
        label.setObjectName("FormLabel")
        self._grid.addWidget(label, row, label_col)
        self._grid.addWidget(widget, row, widget_col)
        self._index += 1
