from __future__ import annotations

import unittest

from robot_pipeline_app.gui_dialogs import format_command_for_dialog, parse_command_text


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


class GuiDialogsParseCommandTests(unittest.TestCase):
    def test_parse_command_text_empty(self) -> None:
        cmd, error = parse_command_text("   ")
        self.assertIsNone(cmd)
        self.assertEqual(error, "Command is empty.")

    def test_parse_command_text_parse_error(self) -> None:
        cmd, error = parse_command_text('python3 -c "print(1)')
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)
        self.assertIn("Unable to parse command", str(error))

    def test_parse_command_text_success(self) -> None:
        cmd, error = parse_command_text('python3 -m lerobot.record --dataset.repo_id="alice/demo one"')
        self.assertIsNone(error)
        self.assertEqual(
            cmd,
            ["python3", "-m", "lerobot.record", "--dataset.repo_id=alice/demo one"],
        )


if __name__ == "__main__":
    unittest.main()
