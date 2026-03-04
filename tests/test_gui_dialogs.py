from __future__ import annotations

import unittest

from robot_pipeline_app.gui_dialogs import format_command_for_dialog


class GuiDialogsFormatCommandTests(unittest.TestCase):
    def test_format_command_for_dialog_empty(self) -> None:
        self.assertEqual(format_command_for_dialog([]), "(empty command)")

    def test_format_command_for_dialog_includes_shell_and_argv_views(self) -> None:
        cmd = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_record",
            "--dataset.single_task=Pick up block",
            '--rename_map={"observation.images.laptop":"observation.images.camera1"}',
        ]
        text = format_command_for_dialog(cmd)

        self.assertIn("Shell-safe command (copy/paste):", text)
        self.assertIn("Exact argv passed to subprocess (no shell quoting here):", text)
        self.assertIn("[0] python3", text)
        self.assertIn("[3] --dataset.single_task=Pick up block", text)
        self.assertIn('[4] --rename_map={"observation.images.laptop":"observation.images.camera1"}', text)


if __name__ == "__main__":
    unittest.main()
