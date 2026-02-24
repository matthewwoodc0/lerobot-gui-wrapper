from __future__ import annotations

import unittest

from robot_pipeline_app.gui_history_tab import _command_from_item, _derive_status


class GuiHistoryTabHelpersTest(unittest.TestCase):
    def test_derive_status_prefers_explicit_status(self) -> None:
        self.assertEqual(_derive_status({"status": "success", "exit_code": 1}), "success")

    def test_derive_status_from_exit_code_and_cancel(self) -> None:
        self.assertEqual(_derive_status({"canceled": True, "exit_code": None}), "canceled")
        self.assertEqual(_derive_status({"exit_code": 0}), "success")
        self.assertEqual(_derive_status({"exit_code": 2}), "failed")

    def test_command_from_item_prefers_command_argv(self) -> None:
        cmd, error = _command_from_item({"command_argv": ["python3", "-V"], "command": "ignored"})
        self.assertIsNone(error)
        self.assertEqual(cmd, ["python3", "-V"])

    def test_command_from_item_parses_legacy_command(self) -> None:
        cmd, error = _command_from_item({"command": "python3 -m unittest"})
        self.assertIsNone(error)
        self.assertEqual(cmd, ["python3", "-m", "unittest"])

    def test_command_from_item_handles_missing_command(self) -> None:
        cmd, error = _command_from_item({})
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
