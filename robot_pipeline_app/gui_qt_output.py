from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class QtRunOutputPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusChip")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(280)
        header.addWidget(self.status_label)

        header.addStretch(1)

        self.explain_button = QPushButton("Explain Failure")
        self.explain_button.setEnabled(False)
        header.addWidget(self.explain_button)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.summary_output = QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setMinimumHeight(120)
        self.summary_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.tabs.addTab(self.summary_output, "Summary")

        self.raw_output = QPlainTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setMinimumHeight(120)
        self.raw_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self.raw_output, "Raw Transcript")
        layout.addWidget(self.tabs, 1)

    def set_summary_text(self, text: str) -> None:
        self.summary_output.setPlainText(str(text))
        self.summary_output.moveCursor(QTextCursor.MoveOperation.Start)

    def append_summary_line(self, line: str) -> None:
        self.summary_output.appendPlainText(str(line))

    def set_raw_text(self, text: str) -> None:
        self.raw_output.setPlainText(str(text))
        self.raw_output.moveCursor(QTextCursor.MoveOperation.Start)

    def append_raw_text(self, text: str) -> None:
        if not text:
            return
        cursor = self.raw_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(str(text))
        self.raw_output.setTextCursor(cursor)
        self.raw_output.ensureCursorVisible()

    def clear_raw(self) -> None:
        self.raw_output.clear()

    def show_summary_tab(self) -> None:
        self.tabs.setCurrentWidget(self.summary_output)

    def show_raw_tab(self) -> None:
        self.tabs.setCurrentWidget(self.raw_output)
