from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.cli_modes import _require_venv_on_macos, parse_args, run_deploy_mode, run_record_mode
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.types import RunResult


class CliModesTest(unittest.TestCase):
    def test_parse_args_history_limit(self) -> None:
        with patch("sys.argv", ["robot_pipeline.py", "history", "--limit", "30"]):
            args = parse_args()
        self.assertEqual(args.mode, "history")
        self.assertEqual(args.limit, 30)

    def test_parse_args_accepts_all_modes(self) -> None:
        for mode in ("record", "deploy", "config", "doctor", "gui", "install-launcher"):
            with patch("sys.argv", ["robot_pipeline.py", mode]):
                args = parse_args()
            self.assertEqual(args.mode, mode)

    def test_require_venv_on_macos_exits_when_no_virtual_env(self) -> None:
        with patch("robot_pipeline_app.cli_modes.sys.platform", "darwin"), patch(
            "robot_pipeline_app.cli_modes.sys.prefix",
            "/usr",
        ), patch(
            "robot_pipeline_app.cli_modes.sys.base_prefix",
            "/usr",
        ), patch.dict(
            "robot_pipeline_app.cli_modes.os.environ",
            {},
            clear=False,
        ):
            with self.assertRaises(SystemExit):
                _require_venv_on_macos()

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

    def test_run_deploy_mode_applies_eval_prefix_quick_fix(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        config["lerobot_dir"] = "/tmp"

        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir) / "models"
            model_dir = models_root / "model_a"
            model_dir.mkdir(parents=True, exist_ok=True)

            with patch(
                "robot_pipeline_app.cli_modes.prompt_path",
                side_effect=[str(models_root), str(model_dir)],
            ), patch(
                "robot_pipeline_app.cli_modes.validate_model_path",
                return_value=(True, "ok", []),
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_text",
                side_effect=["alice/run_1", "Eval task"],
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_int",
                side_effect=[2, 20],
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_yes_no",
                side_effect=[True, True],
            ), patch(
                "robot_pipeline_app.cli_modes.resolve_unique_repo_id",
                return_value=("alice/eval_run_1", False, True),
            ) as mocked_resolve, patch(
                "robot_pipeline_app.cli_modes.dataset_exists_on_hf",
                return_value=False,
            ), patch(
                "robot_pipeline_app.cli_modes.run_preflight_for_deploy",
                return_value=[("PASS", "ok", "ok")],
            ) as mocked_preflight, patch(
                "robot_pipeline_app.cli_modes.has_failures",
                return_value=False,
            ), patch(
                "robot_pipeline_app.cli_modes.execute_command_with_artifacts",
                return_value=RunResult(exit_code=0, canceled=False, output_lines=[], artifact_path=None),
            ), patch("robot_pipeline_app.cli_modes.save_config"):
                run_deploy_mode(config)

        self.assertEqual(mocked_resolve.call_args.kwargs["dataset_name_or_repo_id"], "alice/eval_run_1")
        self.assertEqual(mocked_preflight.call_args.kwargs["eval_repo_id"], "alice/eval_run_1")

    def test_run_deploy_mode_cancels_when_eval_prefix_quick_fix_declined(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        config["lerobot_dir"] = "/tmp"

        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir) / "models"
            model_dir = models_root / "model_a"
            model_dir.mkdir(parents=True, exist_ok=True)

            with patch(
                "robot_pipeline_app.cli_modes.prompt_path",
                side_effect=[str(models_root), str(model_dir)],
            ), patch(
                "robot_pipeline_app.cli_modes.validate_model_path",
                return_value=(True, "ok", []),
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_text",
                side_effect=["alice/run_1"],
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_yes_no",
                side_effect=[False],
            ), patch("robot_pipeline_app.cli_modes.resolve_unique_repo_id") as mocked_resolve, patch(
                "robot_pipeline_app.cli_modes.execute_command_with_artifacts",
            ) as mocked_execute:
                run_deploy_mode(config)

        mocked_resolve.assert_not_called()
        mocked_execute.assert_not_called()

    def test_run_deploy_mode_quick_fix_for_bare_name_omits_owner_in_prompt_value(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["hf_username"] = "alice"
        config["lerobot_dir"] = "/tmp"

        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir) / "models"
            model_dir = models_root / "model_a"
            model_dir.mkdir(parents=True, exist_ok=True)

            with patch(
                "robot_pipeline_app.cli_modes.prompt_path",
                side_effect=[str(models_root), str(model_dir)],
            ), patch(
                "robot_pipeline_app.cli_modes.validate_model_path",
                return_value=(True, "ok", []),
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_text",
                side_effect=["run_1", "Eval task"],
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_int",
                side_effect=[2, 20],
            ), patch(
                "robot_pipeline_app.cli_modes.prompt_yes_no",
                side_effect=[True, True],
            ), patch(
                "robot_pipeline_app.cli_modes.resolve_unique_repo_id",
                return_value=("alice/eval_run_1", False, True),
            ) as mocked_resolve, patch(
                "robot_pipeline_app.cli_modes.dataset_exists_on_hf",
                return_value=False,
            ), patch(
                "robot_pipeline_app.cli_modes.run_preflight_for_deploy",
                return_value=[("PASS", "ok", "ok")],
            ), patch(
                "robot_pipeline_app.cli_modes.has_failures",
                return_value=False,
            ), patch(
                "robot_pipeline_app.cli_modes.execute_command_with_artifacts",
                return_value=RunResult(exit_code=0, canceled=False, output_lines=[], artifact_path=None),
            ), patch("robot_pipeline_app.cli_modes.save_config"):
                run_deploy_mode(config)

        self.assertEqual(mocked_resolve.call_args.kwargs["dataset_name_or_repo_id"], "eval_run_1")


if __name__ == "__main__":
    unittest.main()
