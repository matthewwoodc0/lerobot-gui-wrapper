from __future__ import annotations

from .command_text import format_command_for_editing, parse_command_text

from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _text_wrap_mode(wrap_mode: str) -> QPlainTextEdit.LineWrapMode:
    normalized = str(wrap_mode or "none").strip().lower()
    if normalized == "none":
        return QPlainTextEdit.LineWrapMode.NoWrap
    return QPlainTextEdit.LineWrapMode.WidgetWidth


def _fit_dialog_to_screen(
    dialog: QDialog,
    *,
    requested_width: int,
    requested_height: int,
    requested_min_width: int,
    requested_min_height: int,
) -> None:
    dialog.setMinimumSize(requested_min_width, requested_min_height)
    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        dialog.resize(requested_width, requested_height)
        return
    rect = screen.availableGeometry()
    final_width = min(requested_width, max(requested_min_width, rect.width() - 80))
    final_height = min(requested_height, max(requested_min_height, rect.height() - 80))
    dialog.resize(final_width, final_height)


def _copy_text_to_clipboard(text: str) -> None:
    clipboard = QApplication.clipboard()
    if clipboard is None:
        return
    clipboard.setText(str(text))


class QtTextDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        title: str,
        text: str,
        copy_text: str | None = None,
        width: int = 980,
        height: int = 520,
        wrap_mode: str = "none",
    ) -> None:
        super().__init__(parent)
        self._copy_text = text if copy_text is None else str(copy_text)
        self.setWindowTitle(title)
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=width,
            requested_height=height,
            requested_min_width=700,
            requested_min_height=360,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("DialogText")
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(_text_wrap_mode(wrap_mode))
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.text_edit, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_current_text)
        footer.addWidget(copy_button)
        footer.addStretch(1)
        close_button = QPushButton("Close")
        close_button.setObjectName("AccentButton")
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        layout.addLayout(footer)

    def copy_current_text(self) -> None:
        _copy_text_to_clipboard(self._copy_text)


class QtEditableCommandDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        title: str,
        command_argv: list[str],
        intro_text: str,
        confirm_label: str = "Run",
        cancel_label: str = "Cancel",
        width: int = 980,
        height: int = 540,
    ) -> None:
        super().__init__(parent)
        self.result_argv: list[str] | None = None
        self._initial_text = format_command_for_editing([str(part) for part in command_argv if str(part)]).strip()
        self.setWindowTitle(title)
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=width,
            requested_height=height,
            requested_min_width=700,
            requested_min_height=420,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setObjectName("MutedLabel")
        layout.addWidget(intro_label)

        shortcut_label = QLabel("Edit one argument per line. Press Ctrl+Enter or Command+Enter to run.")
        shortcut_label.setWordWrap(True)
        shortcut_label.setObjectName("MutedLabel")
        layout.addWidget(shortcut_label)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("DialogText")
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setPlainText(self._initial_text)
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.editor, 1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("DialogErrorLabel")
        self.error_label.setStyleSheet("color: #ef4444;")
        layout.addWidget(self.error_label)

        footer = QHBoxLayout()
        footer.setSpacing(8)

        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_current_text)
        footer.addWidget(copy_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_text)
        footer.addWidget(reset_button)

        footer.addStretch(1)

        cancel_button = QPushButton(cancel_label)
        cancel_button.clicked.connect(self.cancel_dialog)
        footer.addWidget(cancel_button)

        confirm_button = QPushButton(confirm_label)
        confirm_button.setObjectName("AccentButton")
        confirm_button.clicked.connect(self.confirm_dialog)
        footer.addWidget(confirm_button)

        layout.addLayout(footer)

    def current_text(self) -> str:
        return self.editor.toPlainText().strip()

    def copy_current_text(self) -> None:
        _copy_text_to_clipboard(self.current_text())

    def reset_text(self) -> None:
        self.editor.setPlainText(self._initial_text)
        self.error_label.setText("")

    def cancel_dialog(self) -> None:
        self.result_argv = None
        self.reject()

    def confirm_dialog(self) -> None:
        parsed, parse_error = parse_command_text(self.current_text())
        if parse_error or parsed is None:
            self.error_label.setText(parse_error or "Unable to parse command.")
            return
        self.result_argv = parsed
        self.accept()


class QtActionChoiceDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        title: str,
        text: str,
        actions: list[tuple[str, str]] | None = None,
        copy_text: str | None = None,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        width: int = 980,
        height: int = 560,
        wrap_mode: str = "word",
    ) -> None:
        super().__init__(parent)
        self.result_choice = "cancel"
        self._copy_text = text if copy_text is None else str(copy_text)
        self.setWindowTitle(title)
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=width,
            requested_height=height,
            requested_min_width=760,
            requested_min_height=420,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("DialogText")
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(_text_wrap_mode(wrap_mode))
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        layout.addWidget(self.text_edit, 1)

        if actions:
            action_frame = QFrame()
            action_layout = QHBoxLayout(action_frame)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(8)
            for action_id, label in actions:
                button = QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=action_id: self.choose_action(value))
                action_layout.addWidget(button)
            action_layout.addStretch(1)
            layout.addWidget(action_frame)

        footer = QHBoxLayout()
        footer.setSpacing(8)

        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_current_text)
        footer.addWidget(copy_button)
        footer.addStretch(1)

        cancel_button = QPushButton(cancel_label)
        cancel_button.clicked.connect(self.cancel_dialog)
        footer.addWidget(cancel_button)

        confirm_button = QPushButton(confirm_label)
        confirm_button.setObjectName("AccentButton")
        confirm_button.clicked.connect(self.confirm_dialog)
        footer.addWidget(confirm_button)
        layout.addLayout(footer)

    def copy_current_text(self) -> None:
        _copy_text_to_clipboard(self._copy_text)

    def choose_action(self, action_id: str) -> None:
        self.result_choice = str(action_id)
        self.accept()

    def confirm_dialog(self) -> None:
        self.result_choice = "confirm"
        self.accept()

    def cancel_dialog(self) -> None:
        self.result_choice = "cancel"
        self.reject()


def show_text_dialog(
    *,
    parent: QWidget | None,
    title: str,
    text: str,
    copy_text: str | None = None,
    width: int = 980,
    height: int = 520,
    wrap_mode: str = "none",
) -> None:
    dialog = QtTextDialog(
        parent=parent,
        title=title,
        text=text,
        copy_text=copy_text,
        width=width,
        height=height,
        wrap_mode=wrap_mode,
    )
    dialog.exec()


def ask_editable_command_dialog(
    *,
    parent: QWidget | None,
    title: str,
    command_argv: list[str],
    intro_text: str,
    confirm_label: str = "Run",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 540,
) -> list[str] | None:
    dialog = QtEditableCommandDialog(
        parent=parent,
        title=title,
        command_argv=command_argv,
        intro_text=intro_text,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
        width=width,
        height=height,
    )
    dialog.exec()
    return dialog.result_argv


def ask_text_dialog(
    *,
    parent: QWidget | None,
    title: str,
    text: str,
    copy_text: str | None = None,
    confirm_label: str = "Continue",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 540,
    wrap_mode: str = "word",
) -> bool:
    dialog = QtActionChoiceDialog(
        parent=parent,
        title=title,
        text=text,
        copy_text=copy_text,
        actions=None,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
        width=width,
        height=height,
        wrap_mode=wrap_mode,
    )
    dialog.exec()
    return dialog.result_choice == "confirm"


def ask_text_dialog_with_actions(
    *,
    parent: QWidget | None,
    title: str,
    text: str,
    actions: list[tuple[str, str]],
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 560,
    wrap_mode: str = "word",
) -> str:
    dialog = QtActionChoiceDialog(
        parent=parent,
        title=title,
        text=text,
        actions=actions,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
        width=width,
        height=height,
        wrap_mode=wrap_mode,
    )
    dialog.exec()
    return dialog.result_choice
