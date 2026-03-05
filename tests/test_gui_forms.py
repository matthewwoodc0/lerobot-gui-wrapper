from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from robot_pipeline_app.constants import CONFIG_FIELDS, DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_forms import (
    build_deploy_request_and_command,
    build_record_request_and_command,
    coerce_config_from_vars,
)


class FakeVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class GuiFormsTest(unittest.TestCase):
    def test_build_record_request_and_command(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_5",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=True,
        )
        self.assertIsNone(error)
        assert req is not None and cmd is not None
        self.assertEqual(req.dataset_repo_id, "alice/demo_5")
        self.assertEqual(req.num_episodes, 3)
        self.assertTrue(req.upload_after_record)
        self.assertIn("--dataset.repo_id=alice/demo_5", cmd)

    def test_build_record_request_and_command_with_target_hz(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_hz",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=False,
            target_hz_raw="35",
        )
        self.assertIsNone(error)
        assert req is not None and cmd is not None
        self.assertIn("--dataset.fps=35", cmd)

    def test_build_record_request_fails_on_invalid_target_hz(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_hz",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=False,
            target_hz_raw="abc",
        )
        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertEqual(error, "Target Hz must be an integer.")

    def test_build_record_request_handles_none_dataset_input(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input=None,  # type: ignore[arg-type]
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=False,
        )
        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertEqual(error, "Dataset name is required.")

    def test_build_deploy_request_missing_model(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, updated, error = build_deploy_request_and_command(
            config=config,
            deploy_root_raw="/tmp/models",
            deploy_model_raw="/tmp/does_not_exist",
            eval_dataset_raw="alice/eval_1",
            eval_episodes_raw="2",
            eval_duration_raw="20",
            eval_task_raw="Test",
        )
        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNone(updated)
        self.assertIsNotNone(error)

    def test_coerce_config_from_vars(self) -> None:
        base = dict(DEFAULT_CONFIG_VALUES)
        vars_map = {field["key"]: FakeVar(str(base[field["key"]])) for field in CONFIG_FIELDS}
        vars_map["camera_laptop_index"] = FakeVar("9")
        parsed, error = coerce_config_from_vars(base, vars_map, CONFIG_FIELDS)
        self.assertIsNone(error)
        assert parsed is not None
        self.assertEqual(parsed["camera_laptop_index"], 9)

    def test_coerce_config_from_vars_uses_defaults_for_blank_entries(self) -> None:
        base = dict(DEFAULT_CONFIG_VALUES)
        vars_map = {field["key"]: FakeVar(str(base[field["key"]])) for field in CONFIG_FIELDS}
        vars_map["lerobot_dir"] = FakeVar("")
        vars_map["camera_fps"] = FakeVar("")
        vars_map["hf_username"] = FakeVar("")

        parsed, error = coerce_config_from_vars(base, vars_map, CONFIG_FIELDS)

        self.assertIsNone(error)
        assert parsed is not None
        self.assertEqual(parsed["lerobot_dir"], base["lerobot_dir"])
        self.assertEqual(parsed["camera_fps"], int(base["camera_fps"]))
        self.assertEqual(parsed["hf_username"], base["hf_username"])

    def test_coerce_config_from_vars_keeps_blank_optional_calibration_paths(self) -> None:
        base = dict(DEFAULT_CONFIG_VALUES)
        vars_map = {field["key"]: FakeVar(str(base[field["key"]])) for field in CONFIG_FIELDS}
        vars_map["follower_calibration_path"] = FakeVar("")
        vars_map["leader_calibration_path"] = FakeVar("")

        parsed, error = coerce_config_from_vars(base, vars_map, CONFIG_FIELDS)

        self.assertIsNone(error)
        assert parsed is not None
        self.assertEqual(parsed["follower_calibration_path"], "")
        self.assertEqual(parsed["leader_calibration_path"], "")

    def test_build_deploy_request_success(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            import os

            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw="alice/eval_7",
                eval_episodes_raw="4",
                eval_duration_raw="25",
                eval_task_raw="Pick and place",
            )

        self.assertIsNone(error)
        assert req is not None and cmd is not None and updated is not None
        self.assertEqual(req.eval_repo_id, "alice/eval_7")
        self.assertEqual(updated["last_model_name"], "model_a")
        self.assertTrue(any(arg.startswith("--policy.path=") or arg.startswith("--policy=") for arg in cmd))
        self.assertTrue(all(not arg.startswith("--warmup_time_s=") for arg in cmd))

    def test_build_deploy_request_with_target_hz(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            import os

            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw="alice/eval_7",
                eval_episodes_raw="4",
                eval_duration_raw="25",
                eval_task_raw="Pick and place",
                target_hz_raw="22",
            )

        self.assertIsNone(error)
        assert req is not None and cmd is not None and updated is not None
        self.assertIn("--dataset.fps=22", cmd)
        self.assertEqual(updated["deploy_target_hz"], "22")

    def test_build_deploy_request_does_not_run_blocking_compat_probe(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            import os

            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")

            with patch(
                "robot_pipeline_app.commands.probe_lerobot_capabilities",
                side_effect=AssertionError("GUI form build should not trigger blocking compat probe"),
            ):
                req, cmd, updated, error = build_deploy_request_and_command(
                    config=config,
                    deploy_root_raw=tmpdir,
                    deploy_model_raw=model_dir,
                    eval_dataset_raw="alice/eval_7",
                    eval_episodes_raw="4",
                    eval_duration_raw="25",
                    eval_task_raw="Pick and place",
                )

        self.assertIsNone(error)
        assert req is not None and cmd is not None and updated is not None
        self.assertTrue(any(arg.startswith("--policy.path=") or arg.startswith("--policy=") for arg in cmd))

    def test_build_deploy_request_invalid_model_payload(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/empty_model"
            import os

            os.makedirs(model_dir, exist_ok=True)
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw="alice/eval_8",
                eval_episodes_raw="1",
                eval_duration_raw="20",
                eval_task_raw="Test",
            )

        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNone(updated)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("does not look deployable", error)

    def test_build_deploy_request_handles_none_eval_dataset(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            import os

            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw=None,  # type: ignore[arg-type]
                eval_episodes_raw="2",
                eval_duration_raw="20",
                eval_task_raw="Test",
            )

        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNone(updated)
        self.assertEqual(error, "Eval dataset name is required.")

    def test_build_record_request_with_advanced_overrides(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_5",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=False,
            arg_overrides={
                "dataset.repo_id": "org/custom_run",
                "dataset.num_episodes": "9",
                "dataset.episode_time_s": "11",
            },
            custom_args_raw="--device cuda",
        )
        self.assertIsNone(error)
        assert req is not None and cmd is not None
        self.assertEqual(req.dataset_repo_id, "org/custom_run")
        self.assertEqual(req.num_episodes, 9)
        self.assertEqual(req.episode_time_s, 11)
        self.assertIn("--dataset.repo_id=org/custom_run", cmd)
        self.assertEqual(cmd[-2:], ["--device", "cuda"])

    def test_build_record_request_fails_on_invalid_custom_args(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        req, cmd, error = build_record_request_and_command(
            config=config,
            dataset_input="alice/demo_5",
            episodes_raw="3",
            duration_raw="15",
            task_raw="Move the cube",
            dataset_dir_raw="/tmp/datasets",
            upload_enabled=False,
            custom_args_raw='--foo "bar',
        )
        self.assertIsNone(req)
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)

    def test_build_deploy_request_with_advanced_policy_override(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model_a"
            model_dir_2 = f"{tmpdir}/model_b"
            import os

            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(model_dir_2, exist_ok=True)
            with open(f"{model_dir}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")
            with open(f"{model_dir_2}/config.json", "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(f"{model_dir_2}/model.safetensors", "w", encoding="utf-8") as handle:
                handle.write("weights\n")
            req, cmd, updated, error = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=tmpdir,
                deploy_model_raw=model_dir,
                eval_dataset_raw="alice/eval_7",
                eval_episodes_raw="4",
                eval_duration_raw="25",
                eval_task_raw="Pick and place",
                arg_overrides={"policy.path": model_dir_2, "dataset.num_episodes": "6"},
                custom_args_raw="--batch-size 2",
            )

        self.assertIsNone(error)
        assert req is not None and cmd is not None and updated is not None
        self.assertEqual(str(req.model_path), model_dir_2)
        self.assertEqual(req.eval_num_episodes, 6)
        self.assertIn(f"--policy.path={model_dir_2}", cmd)
        self.assertEqual(cmd[-2:], ["--batch-size", "2"])
        self.assertEqual(updated["last_model_name"], "model_b")


if __name__ == "__main__":
    unittest.main()
