from __future__ import annotations

import os
import unittest

from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
from robot_pipeline_app.gui_qt_terminal import TerminalScreen

_QT_AVAILABLE, _QT_REASON = qt_available()

if _QT_AVAILABLE:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from robot_pipeline_app.gui_qt_terminal import QtTerminalEmulator


class TerminalScreenTests(unittest.TestCase):
    def test_screen_handles_carriage_return_and_clear_line(self) -> None:
        screen = TerminalScreen()
        screen.feed_output("hello\rbye\x1b[K")
        self.assertEqual(screen.terminal_text(), "bye")

    def test_screen_handles_cursor_motion(self) -> None:
        screen = TerminalScreen()
        screen.feed_output("hello\x1b[2D!!")
        self.assertEqual(screen.terminal_text(), "hel!!")

    def test_screen_hides_trailing_spaces_left_by_backspace_erase(self) -> None:
        screen = TerminalScreen()
        screen.feed_output("pwd\b \b\b \b\b \b")
        self.assertEqual(screen.terminal_text(), "")

    def test_screen_treats_shell_erase_sequence_as_real_deletion(self) -> None:
        screen = TerminalScreen()
        screen.feed_output("abcdef\b \b\b \b")
        self.assertEqual(screen.terminal_text(), "abcd")

    def test_screen_treats_raw_del_as_destructive_backspace(self) -> None:
        screen = TerminalScreen()
        screen.feed_output("abcdef\x7f\x7f")
        self.assertEqual(screen.terminal_text(), "abcd")

    def test_screen_handles_prompt_redraw_without_leaking_previous_command(self) -> None:
        screen = TerminalScreen()
        redraw = (
            "((.venv) ) user@host repo % hello world\r\n"
            "zsh: command not found: hello\r\n"
            "%                                                                              \r \r\r"
            "((.venv) ) user@host repo % ls"
        )
        screen.feed_output(redraw)
        final_line = screen.terminal_text().splitlines()[-1]
        self.assertTrue(final_line.endswith("% ls"))
        self.assertNotIn("hello world", final_line)


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtTerminalWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def test_terminal_emulator_maps_control_shortcuts_to_terminal_bytes(self) -> None:
        sent: list[bytes] = []
        interrupts: list[bool] = []
        widget = QtTerminalEmulator(
            send_input=lambda payload: (sent.append(payload), (True, ""))[1],
            send_interrupt=lambda: interrupts.append(True),
        )
        self.addCleanup(widget.close)

        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier, "a"))
        widget.keyPressEvent(
            QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier, "\r")
        )
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier, "c"))

        self.assertEqual(sent[:2], [b"\x01", b"\r"])
        self.assertEqual(interrupts, [True])


if __name__ == "__main__":
    unittest.main()
