from __future__ import annotations

import os
import unittest

from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available

_QT_AVAILABLE, _QT_REASON = qt_available()

if _QT_AVAILABLE:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from robot_pipeline_app.gui_qt_terminal import QtTerminalEmulator


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtTerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def test_terminal_emulator_handles_carriage_return_and_clear_line(self) -> None:
        widget = QtTerminalEmulator()
        self.addCleanup(widget.close)

        widget.feed_output("hello\rbye\x1b[K")

        self.assertEqual(widget.terminal_text(), "bye")

    def test_terminal_emulator_handles_cursor_motion(self) -> None:
        widget = QtTerminalEmulator()
        self.addCleanup(widget.close)

        widget.feed_output("hello\x1b[2D!!")

        self.assertEqual(widget.terminal_text(), "hel!!")

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
