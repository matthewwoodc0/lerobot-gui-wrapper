from __future__ import annotations

import unittest

from robot_pipeline_app.gui_history_tab import (
    HISTORY_MODE_VALUES,
    _command_from_item,
    _derive_status,
    _normalize_deploy_episode_outcomes,
    _parse_tags_csv,
)


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

    def test_history_mode_values_include_training_modes(self) -> None:
        self.assertIn("train_sync", HISTORY_MODE_VALUES)
        self.assertIn("train_launch", HISTORY_MODE_VALUES)
        self.assertIn("train_attach", HISTORY_MODE_VALUES)

    def test_parse_tags_csv_dedupes_and_trims(self) -> None:
        tags = _parse_tags_csv(" vertical, horizontal ,vertical ,, ")
        self.assertEqual(tags, ["vertical", "horizontal"])

    def test_normalize_deploy_episode_outcomes_counts_success_failed_and_pending(self) -> None:
        summary = _normalize_deploy_episode_outcomes(
            {
                "total_episodes": 4,
                "episode_outcomes": [
                    {"episode": 1, "result": "success", "tags": ["a"]},
                    {"episode": 2, "result": "failed", "tags": ["b"], "note": "dropped"},
                    {"episode": 3, "result": "pending", "tags": ["c"], "note": "interesting behavior"},
                ],
            }
        )
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["rated_count"], 2)
        self.assertEqual(summary["unmarked_count"], 2)
        self.assertEqual(summary["unrated_count"], 2)
        self.assertEqual(summary["tags"], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
