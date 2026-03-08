from __future__ import annotations

from typing import Any, Callable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    _QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - imported indirectly in non-Qt test envs
    _QT_IMPORT_ERROR = exc


TerminalInputCallback = Callable[[bytes], tuple[bool, str]]
StatusCallback = Callable[[str], None]
InterruptCallback = Callable[[], None]


if _QT_IMPORT_ERROR is None:

    class QtTerminalEmulator(QPlainTextEdit):
        """A small embedded VT-like terminal view for the PTY shell."""

        def __init__(
            self,
            *,
            send_input: TerminalInputCallback | None = None,
            send_interrupt: InterruptCallback | None = None,
            on_status: StatusCallback | None = None,
        ) -> None:
            super().__init__()
            self._send_input = send_input
            self._send_interrupt = send_interrupt
            self._on_status = on_status
            self._pending_escape = ""
            self._saved_cursor = (0, 0)
            self._buffer_lines = [""]
            self._cursor_row = 0
            self._cursor_col = 0
            self.setReadOnly(False)
            self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.setUndoRedoEnabled(False)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setCursorWidth(2)
            self.setTabChangesFocus(False)

        def clear_terminal_buffer(self) -> None:
            self._pending_escape = ""
            self._saved_cursor = (0, 0)
            self._buffer_lines = [""]
            self._cursor_row = 0
            self._cursor_col = 0
            self.setPlainText("")

        def terminal_text(self) -> str:
            return "\n".join(self._buffer_lines)

        def focusInEvent(self, event: Any) -> None:
            super().focusInEvent(event)
            self.ensureCursorVisible()

        def insertFromMimeData(self, source: Any) -> None:
            text = ""
            if source is not None and hasattr(source, "text"):
                try:
                    text = str(source.text() or "")
                except Exception:
                    text = ""
            if text:
                self._send_bytes(text.encode("utf-8", errors="ignore"))

        def feed_output(self, data: str) -> None:
            if not data:
                return

            payload = self._pending_escape + str(data)
            self._pending_escape = ""
            index = 0
            while index < len(payload):
                char = payload[index]
                if char == "\x1b":
                    consumed = self._consume_escape(payload, index)
                    if consumed == 0:
                        self._pending_escape = payload[index:]
                        break
                    index += consumed
                    continue
                if char == "\r":
                    self._terminal_carriage_return()
                elif char == "\n":
                    self._terminal_line_feed()
                elif char == "\b":
                    self._terminal_backspace()
                elif char == "\t":
                    for _ in range(4):
                        self._terminal_put_char(" ")
                elif char == "\x07":
                    pass
                elif ord(char) >= 32:
                    self._terminal_put_char(char)
                index += 1
            self._render_buffer()

        def _ensure_line(self, row: int) -> None:
            while row >= len(self._buffer_lines):
                self._buffer_lines.append("")

        def _set_cursor(self, row: int, col: int) -> None:
            row = max(0, row)
            self._ensure_line(row)
            line_text = self._buffer_lines[row]
            self._cursor_row = row
            self._cursor_col = max(0, min(col, len(line_text)))

        def _cursor_line_col(self) -> tuple[int, int]:
            self._ensure_line(self._cursor_row)
            return self._cursor_row, self._cursor_col

        def _terminal_carriage_return(self) -> None:
            self._cursor_col = 0

        def _terminal_line_feed(self) -> None:
            row, col = self._cursor_line_col()
            self._ensure_line(row + 1)
            self._set_cursor(row + 1, col)

        def _terminal_backspace(self) -> None:
            if self._cursor_col > 0:
                self._cursor_col -= 1
                return
            if self._cursor_row > 0:
                self._cursor_row -= 1
                self._cursor_col = len(self._buffer_lines[self._cursor_row])

        def _terminal_put_char(self, char: str) -> None:
            row, col = self._cursor_line_col()
            line = self._buffer_lines[row]
            if col < len(line):
                line = line[:col] + char + line[col + 1 :]
            else:
                line = line + (" " * (col - len(line))) + char
            self._buffer_lines[row] = line
            self._cursor_col += 1

        def _erase_line(self, mode: int) -> None:
            row, col = self._cursor_line_col()
            line = self._buffer_lines[row]
            if mode == 1:
                self._buffer_lines[row] = " " * col + line[col:]
                return
            if mode == 2:
                self._buffer_lines[row] = ""
                self._cursor_col = 0
                return
            self._buffer_lines[row] = line[:col]

        def _erase_display(self, mode: int) -> None:
            row, col = self._cursor_line_col()
            if mode == 1:
                for line_index in range(row):
                    self._buffer_lines[line_index] = ""
                current = self._buffer_lines[row]
                self._buffer_lines[row] = current[col:]
                self._cursor_col = 0
                return
            if mode == 2:
                self.clear_terminal_buffer()
                return
            self._buffer_lines[row] = self._buffer_lines[row][:col]
            del self._buffer_lines[row + 1 :]

        def _handle_csi(self, params_text: str, command: str) -> None:
            clean = params_text.strip()
            parts = clean.split(";") if clean else []

            def _as_int(value: str, default: int) -> int:
                stripped = value.strip().lstrip("?")
                if not stripped:
                    return default
                try:
                    return int(stripped)
                except ValueError:
                    return default

            count = _as_int(parts[0], 1) if parts else 1

            if command == "A":
                row, col = self._cursor_line_col()
                self._set_cursor(row - count, col)
                return
            if command == "B":
                row, col = self._cursor_line_col()
                self._set_cursor(row + count, col)
                return
            if command == "C":
                row, col = self._cursor_line_col()
                self._set_cursor(row, col + count)
                return
            if command == "D":
                row, col = self._cursor_line_col()
                self._set_cursor(row, col - count)
                return
            if command in {"H", "f"}:
                target_row = _as_int(parts[0], 1) if parts else 1
                target_col = _as_int(parts[1], 1) if len(parts) > 1 else 1
                self._set_cursor(target_row - 1, target_col - 1)
                return
            if command == "G":
                target_col = _as_int(parts[0], 1)
                row, _ = self._cursor_line_col()
                self._set_cursor(row, target_col - 1)
                return
            if command == "K":
                self._erase_line(_as_int(parts[0], 0) if parts else 0)
                return
            if command == "J":
                self._erase_display(_as_int(parts[0], 0) if parts else 0)
                return
            if command == "s":
                self._saved_cursor = self._cursor_line_col()
                return
            if command == "u":
                row, col = self._saved_cursor
                self._set_cursor(row, col)

        def _consume_escape(self, payload: str, start: int) -> int:
            if start + 1 >= len(payload):
                return 0
            lead = payload[start + 1]
            if lead == "[":
                index = start + 2
                while index < len(payload):
                    code = payload[index]
                    if "@" <= code <= "~":
                        self._handle_csi(payload[start + 2 : index], code)
                        return index - start + 1
                    index += 1
                return 0
            if lead == "]":
                index = start + 2
                while index < len(payload):
                    code = payload[index]
                    if code == "\x07":
                        return index - start + 1
                    if code == "\x1b" and index + 1 < len(payload) and payload[index + 1] == "\\":
                        return index - start + 2
                    index += 1
                return 0
            return 2

        def _render_buffer(self) -> None:
            text = self.terminal_text()
            self.setPlainText(text)
            absolute = 0
            for row_index, line in enumerate(self._buffer_lines):
                if row_index == self._cursor_row:
                    absolute += min(self._cursor_col, len(line))
                    break
                absolute += len(line) + 1
            cursor = self.textCursor()
            cursor.setPosition(min(absolute, len(text)))
            self.setTextCursor(cursor)
            self.ensureCursorVisible()

        def _send_bytes(self, payload: bytes) -> None:
            if self._send_input is None:
                return
            ok, message = self._send_input(payload)
            if not ok and message and self._on_status is not None:
                self._on_status(message)

        def keyPressEvent(self, event: Any) -> None:
            modifiers = event.modifiers()
            key = event.key()
            text = event.text()
            ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            meta_pressed = bool(modifiers & Qt.KeyboardModifier.MetaModifier)

            if ctrl_pressed and key == Qt.Key.Key_C:
                if self.textCursor().hasSelection():
                    super().keyPressEvent(event)
                elif self._send_interrupt is not None:
                    self._send_interrupt()
                return

            special_payloads = {
                Qt.Key.Key_Return: b"\r",
                Qt.Key.Key_Enter: b"\r",
                Qt.Key.Key_Backspace: b"\x7f",
                Qt.Key.Key_Tab: b"\t",
                Qt.Key.Key_Escape: b"\x1b",
                Qt.Key.Key_Up: b"\x1b[A",
                Qt.Key.Key_Down: b"\x1b[B",
                Qt.Key.Key_Right: b"\x1b[C",
                Qt.Key.Key_Left: b"\x1b[D",
                Qt.Key.Key_Home: b"\x1b[H",
                Qt.Key.Key_End: b"\x1b[F",
                Qt.Key.Key_Delete: b"\x1b[3~",
                Qt.Key.Key_PageUp: b"\x1b[5~",
                Qt.Key.Key_PageDown: b"\x1b[6~",
                Qt.Key.Key_Insert: b"\x1b[2~",
            }
            payload = special_payloads.get(key)
            if payload is not None:
                self._send_bytes(payload)
                return

            if ctrl_pressed and key == Qt.Key.Key_Space:
                self._send_bytes(b"\x00")
                return

            if ctrl_pressed and Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
                control_code = key - Qt.Key.Key_A + 1
                self._send_bytes(bytes([control_code]))
                return

            if event.matches(QKeySequence.StandardKey.Copy):
                super().keyPressEvent(event)
                return
            if event.matches(QKeySequence.StandardKey.SelectAll):
                super().keyPressEvent(event)
                return
            if event.matches(QKeySequence.StandardKey.Paste):
                clipboard = QApplication.clipboard()
                text = clipboard.text() if clipboard is not None else ""
                if text:
                    self._send_bytes(text.encode("utf-8", errors="ignore"))
                return

            if text and not meta_pressed:
                self._send_bytes(text.encode("utf-8", errors="ignore"))
                return

            super().keyPressEvent(event)

        def mousePressEvent(self, event: Any) -> None:
            super().mousePressEvent(event)
            self.setFocus()
