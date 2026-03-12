from __future__ import annotations

from .command_text import format_command_for_editing, parse_command_text

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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


def _build_dialog_panel(
    dialog: QDialog,
    *,
    title: str,
    subtitle: str | None = None,
) -> QVBoxLayout:
    dialog.setObjectName("AppDialog")
    dialog.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    outer_layout = QVBoxLayout(dialog)
    outer_layout.setContentsMargins(18, 18, 18, 18)
    outer_layout.setSpacing(0)

    panel = QFrame()
    panel.setObjectName("DialogPanel")
    panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    outer_layout.addWidget(panel, 1)

    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(18, 18, 18, 18)
    panel_layout.setSpacing(12)

    header = QFrame()
    header.setObjectName("DialogHeader")
    header_layout = QVBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)

    title_label = QLabel(title)
    title_label.setObjectName("DialogTitle")
    title_label.setWordWrap(True)
    header_layout.addWidget(title_label)

    if subtitle:
        subtitle_label = QLabel(str(subtitle))
        subtitle_label.setObjectName("DialogSubtitle")
        subtitle_label.setWordWrap(True)
        header_layout.addWidget(subtitle_label)

    panel_layout.addWidget(header)
    return panel_layout


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

        layout = _build_dialog_panel(self, title=title, subtitle="Review the details below.")

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("DialogText")
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(_text_wrap_mode(wrap_mode))
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.text_edit, 1)

        footer_frame = QFrame()
        footer_frame.setObjectName("DialogFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_current_text)
        footer.addWidget(copy_button)
        footer.addStretch(1)
        close_button = QPushButton("Close")
        close_button.setObjectName("AccentButton")
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        layout.addWidget(footer_frame)

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

        layout = _build_dialog_panel(self, title=title, subtitle=intro_text)

        shortcut_label = QLabel("Edit one argument per line. Press Ctrl+Enter or Command+Enter to run.")
        shortcut_label.setWordWrap(True)
        shortcut_label.setObjectName("DialogSubtitle")
        layout.addWidget(shortcut_label)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("DialogText")
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setPlainText(self._initial_text)
        self.editor.moveCursor(QTextCursor.MoveOperation.Start)
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.editor, 1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("DialogErrorLabel")
        layout.addWidget(self.error_label)

        footer_frame = QFrame()
        footer_frame.setObjectName("DialogFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(0, 0, 0, 0)
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

        layout.addWidget(footer_frame)

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

        subtitle = "Review the details below and choose how to proceed." if actions else "Review the details below."
        layout = _build_dialog_panel(self, title=title, subtitle=subtitle)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("DialogText")
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(_text_wrap_mode(wrap_mode))
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        layout.addWidget(self.text_edit, 1)

        if actions:
            action_frame = QFrame()
            action_frame.setObjectName("DialogActionBar")
            action_layout = QHBoxLayout(action_frame)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(8)
            for action_id, label in actions:
                button = QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=action_id: self.choose_action(value))
                action_layout.addWidget(button)
            action_layout.addStretch(1)
            layout.addWidget(action_frame)

        footer_frame = QFrame()
        footer_frame.setObjectName("DialogFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(0, 0, 0, 0)
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
        layout.addWidget(footer_frame)

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


class QtReplayEpisodeDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None,
        title: str,
        repo_id: str,
        choices: list[str],
        selected_value: str,
        helper_text: str,
    ) -> None:
        super().__init__(parent)
        self.result_value: str | None = None
        self.setWindowTitle(title)
        self.setModal(True)
        _fit_dialog_to_screen(
            self,
            requested_width=760,
            requested_height=420,
            requested_min_width=640,
            requested_min_height=320,
        )

        layout = _build_dialog_panel(
            self,
            title=title,
            subtitle=f"Choose a replay episode for {repo_id}. Use manual entry if local discovery is incomplete.",
        )

        form_frame = QFrame()
        form_layout = QFormLayout(form_frame)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)

        self.episode_combo = QComboBox()
        self.episode_combo.setEditable(False)
        if choices:
            self.episode_combo.addItems(choices)
            if selected_value in choices:
                self.episode_combo.setCurrentText(selected_value)
        form_layout.addRow("Discovered episodes", self.episode_combo)

        self.manual_input = QLineEdit("")
        self.manual_input.setPlaceholderText("Optional manual episode index")
        form_layout.addRow("Manual override", self.manual_input)
        layout.addWidget(form_frame)

        helper_label = QLabel(helper_text)
        helper_label.setWordWrap(True)
        helper_label.setObjectName("DialogSubtitle")
        layout.addWidget(helper_label, 1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("DialogErrorLabel")
        layout.addWidget(self.error_label)

        footer_frame = QFrame()
        footer_frame.setObjectName("DialogFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        footer.addWidget(cancel_button)
        confirm_button = QPushButton("Use Episode")
        confirm_button.setObjectName("AccentButton")
        confirm_button.clicked.connect(self.confirm_dialog)
        footer.addWidget(confirm_button)
        layout.addWidget(footer_frame)

    def confirm_dialog(self) -> None:
        raw = self.manual_input.text().strip() or self.episode_combo.currentText().strip()
        try:
            value = int(raw)
        except (TypeError, ValueError):
            self.error_label.setText("Episode index must be an integer.")
            return
        if value < 0:
            self.error_label.setText("Episode index must be zero or greater.")
            return
        self.result_value = str(value)
        self.accept()


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


def ask_replay_episode_dialog(
    *,
    parent: QWidget | None,
    title: str,
    repo_id: str,
    choices: list[str],
    selected_value: str,
    helper_text: str,
) -> str | None:
    dialog = QtReplayEpisodeDialog(
        parent=parent,
        title=title,
        repo_id=repo_id,
        choices=choices,
        selected_value=selected_value,
        helper_text=helper_text,
    )
    dialog.exec()
    return dialog.result_value
