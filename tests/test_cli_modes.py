from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.cli_modes import parse_args, run_record_mode
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.types import RunResult


class CliModesTest(unittest.TestCase):
    def test_parse_args_history_limit(self) -> None:
        with patch("sys.argv", ["robot_pipeline.py", "history", "--limit", "30"]):
            args = parse_args()
        self.assertEqual(args.mode, "history")
        self.assertEqual(args.limit, 30)

    def test_parse_args_accepts_all_modes(self) -> None:
        for mode in ("record", "deploy", "config", "doctor", "gui"):
            with patch("sys.argv", ["robot_pipeline.py", mode]):
                args = parse_args()
            self.assertEqual(args.mode, mode)

    def test_run_record_mode_skips_upload_when_disabled(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["lerobot_dir"] = "/tmp"
        config["record_data_dir"] = "/tmp/data"

        with patch("robot_pipeline_app.cli_modes.suggest_dataset_name", return_value=("demo_2", False)), patch(
            "robot_pipeline_app.cli_modes.prompt_text",
            side_effect=["demo_2", "Stack the blocks"],
        ), patch("robot_pipeline_app.cli_modes.prompt_path", return_value="/tmp/data"), patch(
            "robot_pipeline_app.cli_modes.prompt_int",
            side_effect=[2, 20],
        ), patch("robot_pipeline_app.cli_modes.dataset_exists_on_hf", return_value=False), patch(
            "robot_pipeline_app.cli_modes.prompt_yes_no",
            side_effect=[False, True],
        ), patch(
            "robot_pipeline_app.cli_modes.run_preflight_for_record",
            return_value=[("PASS", "ok", "ok")],
        ), patch("robot_pipeline_app.cli_modes.has_failures", return_value=False), patch(
            "robot_pipeline_app.cli_modes.execute_command_with_artifacts",
            return_value=RunResult(exit_code=0, canceled=False, output_lines=[], artifact_path=None),
        ), patch("robot_pipeline_app.cli_modes.move_recorded_dataset", return_value=Path("/tmp/data/demo_2")), patch(
            "robot_pipeline_app.cli_modes.upload_dataset_with_artifacts",
        ) as mocked_upload, patch("robot_pipeline_app.cli_modes.save_config"):
            run_record_mode(config)

        mocked_upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
