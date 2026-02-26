from __future__ import annotations

import unittest

from robot_pipeline_app.gui_input_help import keyboard_input_help_text, keyboard_input_help_title


class GuiInputHelpTests(unittest.TestCase):
    def test_keyboard_input_help_title(self) -> None:
        self.assertEqual(keyboard_input_help_title(), "Keyboard Input Setup")

    def test_keyboard_input_help_text_for_macos_mentions_permissions(self) -> None:
        text = keyboard_input_help_text("darwin")
        self.assertIn("Input Monitoring", text)
        self.assertIn("Accessibility", text)

    def test_keyboard_input_help_text_for_linux_mentions_wayland(self) -> None:
        text = keyboard_input_help_text("linux")
        self.assertIn("Wayland", text)
        self.assertIn("X11", text)

    def test_keyboard_input_help_text_for_windows_mentions_windows(self) -> None:
        text = keyboard_input_help_text("win32")
        self.assertIn("Windows", text)


if __name__ == "__main__":
    unittest.main()
