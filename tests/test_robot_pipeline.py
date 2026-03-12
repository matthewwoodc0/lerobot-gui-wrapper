#!/usr/bin/env python3
"""Unit tests for robot_pipeline helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import robot_pipeline as rp


class RobotPipelineHelpersTest(unittest.TestCase):
    def test_normalize_repo_id(self) -> None:
        self.assertEqual(rp.normalize_repo_id("alice", "dataset_a"), "alice/dataset_a")
        self.assertEqual(rp.normalize_repo_id("alice", "bob/dataset_a"), "bob/dataset_a")
        self.assertEqual(rp.normalize_repo_id("alice", None), "alice/dataset_1")

    def test_repo_name_from_repo_id(self) -> None:
        self.assertEqual(rp.repo_name_from_repo_id("alice/train_set_1"), "train_set_1")
        self.assertEqual(rp.repo_name_from_repo_id("train_set_2"), "train_set_2")

    def test_camera_arg_uses_default_shape_when_probe_fails(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["camera_laptop_index"] = 1
        config["camera_phone_index"] = 2
        config["camera_fps"] = 24
        with patch("robot_pipeline_app.commands.probe_camera_capture", return_value=(False, "camera not opened")):
            cameras = json.loads(rp.camera_arg(config))
        self.assertEqual(cameras["laptop"]["width"], 640)
        self.assertEqual(cameras["laptop"]["height"], 480)
        self.assertEqual(cameras["laptop"]["fps"], 24)
        self.assertEqual(cameras["phone"]["index_or_path"], 2)

    def test_camera_arg_uses_detected_resolution_when_probe_succeeds(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["camera_laptop_index"] = 0
        config["camera_phone_index"] = 6
        with patch(
            "robot_pipeline_app.commands.probe_camera_capture",
            side_effect=[(True, "frame=640x360"), (True, "frame=640x480")],
        ) as mocked:
            cameras = json.loads(rp.camera_arg(config))
        self.assertEqual(cameras["laptop"]["width"], 640)
        self.assertEqual(cameras["laptop"]["height"], 360)
        self.assertEqual(cameras["phone"]["width"], 640)
        self.assertEqual(cameras["phone"]["height"], 480)
        self.assertEqual(mocked.call_count, 2)

    def test_camera_arg_resolution_backoff_prefers_lower_supported_mode(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["camera_laptop_index"] = 0
        config["camera_phone_index"] = 1
        config["camera_fps"] = 30

        def fake_probe(index: int, width: int, height: int) -> tuple[bool, str]:
            if index == 0:
                if (width, height) == (640, 360):
                    return True, "frame=1920x1080"
                if (width, height) == (640, 480):
                    return True, "frame=640x480"
                return True, "frame=1280x720"
            if index == 1:
                return True, "frame=640x360"
            return False, "camera not opened"

        with patch("robot_pipeline_app.commands.probe_camera_capture", side_effect=fake_probe):
            cameras = json.loads(rp.camera_arg(config))

        self.assertEqual(cameras["laptop"]["width"], 640)
        self.assertEqual(cameras["laptop"]["height"], 480)
        self.assertEqual(cameras["phone"]["width"], 640)
        self.assertEqual(cameras["phone"]["height"], 360)

    def test_build_lerobot_record_command_with_policy_omits_warmup_time(self) -> None:
        class _Caps:
            record_entrypoint = "lerobot.scripts.lerobot_record"
            calibrate_entrypoint = "lerobot.calibrate"
            teleop_entrypoint = "lerobot.teleoperate"
            teleop_uses_legacy_control = False
            policy_path_flag = "policy.path"

        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=_Caps()):
            cmd = rp.build_lerobot_record_command(
                config=config,
                dataset_repo_id="alice/eval_run_1",
                num_episodes=5,
                task="Grab the cube",
                episode_time=20,
                policy_path=Path("/tmp/model_x"),
            )
        self.assertIn("lerobot.scripts.lerobot_record", cmd)
        self.assertIn("--dataset.repo_id=alice/eval_run_1", cmd)
        self.assertIn("--dataset.num_episodes=5", cmd)
        self.assertNotIn("--warmup_time_s=5", cmd)
        self.assertIn("--policy.path=/tmp/model_x", cmd)

    def test_build_lerobot_record_command_skips_capability_probe_when_disabled(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["compat_probe_enabled"] = False
        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities") as mocked_probe:
            cmd = rp.build_lerobot_record_command(
                config=config,
                dataset_repo_id="alice/demo_1",
                num_episodes=1,
                task="Pick and place",
                episode_time=10,
                policy_path=Path("/tmp/model_x"),
            )
        mocked_probe.assert_not_called()
        self.assertTrue(any(arg.startswith("--policy.path=") for arg in cmd))

    def test_build_lerobot_record_command_uses_capability_policy_flag_fallback(self) -> None:
        class _Caps:
            record_entrypoint = "lerobot.scripts.lerobot_record"
            calibrate_entrypoint = "lerobot.calibrate"
            teleop_entrypoint = "lerobot.teleoperate"
            teleop_uses_legacy_control = False
            policy_path_flag = "policy"

        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=_Caps()):
            cmd = rp.build_lerobot_record_command(
                config=config,
                dataset_repo_id="alice/eval_run_1",
                num_episodes=5,
                task="Grab the cube",
                episode_time=20,
                policy_path=Path("/tmp/model_x"),
            )
        self.assertIn("--policy=/tmp/model_x", cmd)
        self.assertTrue(all(not arg.startswith("--policy.path=") for arg in cmd))

    def test_build_lerobot_record_command_without_policy_omits_warmup_time(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_1",
            num_episodes=2,
            task="Grab the cube",
            episode_time=20,
        )
        self.assertTrue(all(not arg.startswith("--warmup_time_s=") for arg in cmd))

    def test_build_lerobot_record_command_includes_target_hz_when_provided(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_hz",
            num_episodes=1,
            task="Pick and place",
            episode_time=10,
            target_hz=42,
        )
        self.assertIn("--dataset.fps=42", cmd)

    def test_build_lerobot_record_command_uses_role_specific_target_hz_defaults(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["record_target_hz"] = "18"
        config["deploy_target_hz"] = "12"

        record_cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_1",
            num_episodes=1,
            task="Pick and place",
            episode_time=10,
        )
        deploy_cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/eval_1",
            num_episodes=1,
            task="Pick and place",
            episode_time=10,
            policy_path=Path("/tmp/model_payload"),
        )

        self.assertIn("--dataset.fps=18", record_cmd)
        self.assertIn("--dataset.fps=12", deploy_cmd)

    def test_build_lerobot_record_command_includes_calibration_dirs_from_selected_files(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_calibration_path"] = "/tmp/calibration/red4.json"
        config["leader_calibration_path"] = "/tmp/calibration/white.json"
        cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_1",
            num_episodes=2,
            task="Grab the cube",
            episode_time=20,
        )
        self.assertIn("--robot.calibration_dir=/tmp/calibration", cmd)
        self.assertIn("--teleop.calibration_dir=/tmp/calibration", cmd)

    def test_build_lerobot_record_command_uses_configured_robot_ids(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_robot_id"] = "arm_alpha"
        config["leader_robot_id"] = "operator_beta"
        cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_1",
            num_episodes=2,
            task="Grab the cube",
            episode_time=20,
        )
        self.assertIn("--robot.id=arm_alpha", cmd)
        self.assertIn("--teleop.id=operator_beta", cmd)

    def test_build_lerobot_record_command_infers_robot_ids_from_selected_calibration_files(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_calibration_path"] = "/tmp/calibration/arm_alpha.json"
        config["leader_calibration_path"] = "/tmp/calibration/operator_beta.json"
        # Keep defaults: command should infer IDs from selected calibration files.
        config["follower_robot_id"] = "red4"
        config["leader_robot_id"] = "white"
        cmd = rp.build_lerobot_record_command(
            config=config,
            dataset_repo_id="alice/demo_1",
            num_episodes=2,
            task="Grab the cube",
            episode_time=20,
        )
        self.assertIn("--robot.id=arm_alpha", cmd)
        self.assertIn("--teleop.id=operator_beta", cmd)

    def test_build_lerobot_record_command_prefers_configured_checkout_modern_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir)
            (lerobot_dir / "lerobot").mkdir(parents=True, exist_ok=True)
            (lerobot_dir / "lerobot" / "record.py").write_text("# stub\n", encoding="utf-8")

            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)
            config["compat_probe_enabled"] = False

            cmd = rp.build_lerobot_record_command(
                config=config,
                dataset_repo_id="alice/demo_1",
                num_episodes=1,
                task="Pick and place",
                episode_time=10,
            )

        self.assertIn("lerobot.record", cmd)

    def test_build_lerobot_teleop_command_defaults_to_lerobot_teleoperate_module(self) -> None:
        class _Caps:
            record_entrypoint = "lerobot.scripts.lerobot_record"
            calibrate_entrypoint = "lerobot.calibrate"
            teleop_entrypoint = "lerobot.teleoperate"
            teleop_uses_legacy_control = False
            policy_path_flag = "policy.path"

        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=_Caps()):
            cmd = rp.build_lerobot_teleop_command(
                config=config,
                follower_robot_id="f_red",
                leader_robot_id="l_white",
                control_fps=24,
            )
        self.assertIn("lerobot.teleoperate", cmd)
        self.assertIn("--robot.port=/dev/ttyA", cmd)
        self.assertIn("--robot.cameras={}", cmd)
        self.assertIn("--teleop.port=/dev/ttyB", cmd)
        self.assertIn("--robot.id=f_red", cmd)
        self.assertIn("--teleop.id=l_white", cmd)
        self.assertNotIn("--control.type=teleoperate", cmd)
        self.assertNotIn("--control.fps=24", cmd)

    def test_build_lerobot_teleop_command_skips_capability_probe_when_disabled(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["compat_probe_enabled"] = False
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities") as mocked_probe:
            cmd = rp.build_lerobot_teleop_command(config=config)
        mocked_probe.assert_not_called()
        self.assertTrue(any(arg.startswith("--robot.port=/dev/ttyA") for arg in cmd))

    def test_build_lerobot_teleop_command_includes_calibration_dirs_from_selected_files(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        config["follower_calibration_path"] = "/tmp/calib/follower.json"
        config["leader_calibration_path"] = "/tmp/calib/leader.json"
        cmd = rp.build_lerobot_teleop_command(
            config=config,
            follower_robot_id="f_red",
            leader_robot_id="l_white",
        )
        self.assertIn("--robot.calibration_dir=/tmp/calib", cmd)
        self.assertIn("--teleop.calibration_dir=/tmp/calib", cmd)

    def test_build_lerobot_teleop_command_uses_configured_ids_when_no_override(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        config["follower_robot_id"] = "arm_alpha"
        config["leader_robot_id"] = "operator_beta"
        cmd = rp.build_lerobot_teleop_command(config=config)
        self.assertIn("--robot.id=arm_alpha", cmd)
        self.assertIn("--teleop.id=operator_beta", cmd)

    def test_build_lerobot_teleop_command_inferrs_ids_from_calibration_files_when_default_ids(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        config["follower_robot_id"] = "red4"
        config["leader_robot_id"] = "white"
        config["follower_calibration_path"] = "/tmp/calib/arm_alpha.json"
        config["leader_calibration_path"] = "/tmp/calib/operator_beta.json"
        cmd = rp.build_lerobot_teleop_command(config=config)
        self.assertIn("--robot.id=arm_alpha", cmd)
        self.assertIn("--teleop.id=operator_beta", cmd)

    def test_build_lerobot_teleop_command_prefers_source_checkout_script_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir)
            scripts_dir = lerobot_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "lerobot_teleoperate.py").write_text("# stub\n", encoding="utf-8")

            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)
            config["follower_port"] = "/dev/ttyA"
            config["leader_port"] = "/dev/ttyB"
            cmd = rp.build_lerobot_teleop_command(config=config)

        self.assertIn("scripts.lerobot_teleoperate", cmd)
        self.assertIn("--robot.cameras={}", cmd)
        self.assertNotIn("--control.type=teleoperate", cmd)

    def test_build_lerobot_teleop_command_prefers_configured_checkout_modern_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir)
            (lerobot_dir / "lerobot").mkdir(parents=True, exist_ok=True)
            (lerobot_dir / "lerobot" / "teleoperate.py").write_text("# stub\n", encoding="utf-8")

            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)
            config["follower_port"] = "/dev/ttyA"
            config["leader_port"] = "/dev/ttyB"
            config["compat_probe_enabled"] = False
            config["teleop_av1_fallback"] = False
            cmd = rp.build_lerobot_teleop_command(config=config)

        self.assertIn("lerobot.teleoperate", cmd)
        self.assertNotIn("lerobot.scripts.control_robot", cmd)

    def test_build_lerobot_teleop_command_falls_back_to_legacy_control_robot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir)
            legacy_dir = lerobot_dir / "lerobot" / "scripts"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "control_robot.py").write_text("# stub\n", encoding="utf-8")

            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)
            config["follower_port"] = "/dev/ttyA"
            config["leader_port"] = "/dev/ttyB"
            cmd = rp.build_lerobot_teleop_command(config=config, control_fps=24)

        self.assertIn("lerobot.scripts.control_robot", cmd)
        self.assertIn("--control.type=teleoperate", cmd)
        self.assertIn("--control.fps=24", cmd)

    def test_build_lerobot_teleop_command_macos_prefers_legacy_for_av1_fallback(self) -> None:
        class _Caps:
            record_entrypoint = "lerobot.scripts.lerobot_record"
            calibrate_entrypoint = "lerobot.calibrate"
            teleop_entrypoint = "lerobot.scripts.control_robot"
            teleop_uses_legacy_control = True
            policy_path_flag = "policy.path"

        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"

        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=_Caps()):
            cmd = rp.build_lerobot_teleop_command(config=config)

        self.assertIn("lerobot.scripts.control_robot", cmd)
        self.assertIn("--control.type=teleoperate", cmd)
        self.assertNotIn("lerobot.teleoperate", cmd)

    def test_build_lerobot_teleop_command_macos_fallback_can_be_disabled(self) -> None:
        class _Caps:
            record_entrypoint = "lerobot.scripts.lerobot_record"
            calibrate_entrypoint = "lerobot.calibrate"
            teleop_entrypoint = "lerobot.teleoperate"
            teleop_uses_legacy_control = False
            policy_path_flag = "policy.path"

        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/ttyA"
        config["leader_port"] = "/dev/ttyB"
        config["teleop_av1_fallback"] = False

        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=_Caps()):
            cmd = rp.build_lerobot_teleop_command(config=config)

        self.assertIn("lerobot.teleoperate", cmd)
        self.assertNotIn("lerobot.scripts.control_robot", cmd)
        self.assertNotIn("--control.type=teleoperate", cmd)

    def test_suggest_eval_dataset_name_increments_previous(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        config["last_eval_dataset_name"] = "eval_model_9"
        self.assertEqual(rp.suggest_eval_dataset_name(config, "model"), "eval_model_10")

    def test_write_run_artifacts_creates_log_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = tmpdir

            started = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
            ended = datetime(2026, 2, 23, 12, 0, 5, tzinfo=timezone.utc)
            run_path = rp.write_run_artifacts(
                config=config,
                mode="record",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path("/tmp"),
                started_at=started,
                ended_at=ended,
                exit_code=0,
                canceled=False,
                preflight_checks=[("PASS", "LeRobot folder", "/tmp/lerobot"), ("WARN", "Follower port", "/dev/ttyACM1")],
                output_lines=["line1", "line2"],
                dataset_repo_id="alice/demo_1",
            )

            self.assertIsNotNone(run_path)
            assert run_path is not None
            self.assertTrue((run_path / "command.log").exists())
            self.assertTrue((run_path / "metadata.json").exists())

            metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "record")
            self.assertEqual(metadata["exit_code"], 0)
            self.assertEqual(metadata["dataset_repo_id"], "alice/demo_1")
            self.assertEqual(metadata["preflight_warn_count"], 1)
            self.assertEqual(metadata["preflight_fail_count"], 0)
            self.assertEqual(metadata["status"], "success")
            self.assertEqual(metadata["source"], "pipeline")
            self.assertEqual(metadata["command_argv"][0], "python3")
            self.assertEqual(metadata["diagnostic_version"], "v2")
            self.assertIn("preflight_diagnostics", metadata)
            self.assertIsNone(metadata["first_failure_code"])

    def test_write_run_artifacts_persists_extra_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = tmpdir

            started = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
            ended = datetime(2026, 2, 23, 12, 0, 5, tzinfo=timezone.utc)
            run_path = rp.write_run_artifacts(
                config=config,
                mode="deploy",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path("/tmp"),
                started_at=started,
                ended_at=ended,
                exit_code=0,
                canceled=False,
                preflight_checks=[],
                output_lines=["line1"],
                metadata_extra={
                    "deploy_episode_outcomes": {
                        "success_count": 1,
                        "failed_count": 0,
                        "rated_count": 1,
                    }
                },
            )

            self.assertIsNotNone(run_path)
            assert run_path is not None
            metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("deploy_episode_outcomes", metadata)
            self.assertEqual(metadata["deploy_episode_outcomes"]["success_count"], 1)

    def test_preflight_deploy_marks_missing_model_as_fail(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with patch("robot_pipeline._run_common_preflight_checks", return_value=[]):
            checks = rp.run_preflight_for_deploy(config=config, model_path=Path("/tmp/definitely_missing_model"))
        self.assertTrue(any(level == "FAIL" and name == "Model folder" for level, name, _ in checks))

    def test_preflight_deploy_marks_invalid_model_payload_as_fail(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_empty"
            model_dir.mkdir(parents=True, exist_ok=True)
            with patch("robot_pipeline._run_common_preflight_checks", return_value=[]):
                checks = rp.run_preflight_for_deploy(config=config, model_path=model_dir)
        self.assertTrue(any(level == "FAIL" and name == "Model payload" for level, name, _ in checks))

    def test_preflight_deploy_eval_dataset_prefix_fail(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
            with patch("robot_pipeline._run_common_preflight_checks", return_value=[]):
                checks = rp.run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/run_1",
                )
        self.assertTrue(any(level == "FAIL" and name == "Eval dataset naming" for level, name, _ in checks))

    def test_preflight_deploy_eval_dataset_prefix_pass(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_ok"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
            with patch("robot_pipeline._run_common_preflight_checks", return_value=[]):
                checks = rp.run_preflight_for_deploy(
                    config=config,
                    model_path=model_dir,
                    eval_repo_id="alice/eval_run_1",
                )
        self.assertTrue(any(level == "PASS" and name == "Eval dataset naming" for level, name, _ in checks))

    def test_preflight_record_marks_missing_hf_cli_as_fail_when_upload_enabled(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "datasets"
            with patch("robot_pipeline._run_common_preflight_checks", return_value=[]), patch(
                "robot_pipeline.shutil.which",
                return_value=None,
            ):
                checks = rp.run_preflight_for_record(config=config, dataset_root=dataset_root, upload_enabled=True)
        self.assertTrue(any(level == "FAIL" and name == "huggingface-cli" for level, name, _ in checks))

    def test_preflight_teleop_fails_for_invalid_fps(self) -> None:
        config = dict(rp.DEFAULT_CONFIG_VALUES)
        checks = rp.run_preflight_for_teleop(
            config=config,
            control_fps=0,
            common_checks_fn=lambda _: [],
        )
        self.assertTrue(any(level == "FAIL" and name == "Teleop control FPS" for level, name, _ in checks))

    def test_list_runs_sorted_desc_with_limit_and_warning_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(rp.DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = tmpdir

            old_started = datetime(2026, 2, 23, 10, 0, 0, tzinfo=timezone.utc)
            old_ended = datetime(2026, 2, 23, 10, 0, 2, tzinfo=timezone.utc)
            new_started = datetime(2026, 2, 23, 11, 0, 0, tzinfo=timezone.utc)
            new_ended = datetime(2026, 2, 23, 11, 0, 2, tzinfo=timezone.utc)

            old_path = rp.write_run_artifacts(
                config=config,
                mode="record",
                command=["echo", "old"],
                cwd=Path("/tmp"),
                started_at=old_started,
                ended_at=old_ended,
                exit_code=0,
                canceled=False,
                preflight_checks=[],
                output_lines=["old"],
            )
            new_path = rp.write_run_artifacts(
                config=config,
                mode="deploy",
                command=["echo", "new"],
                cwd=Path("/tmp"),
                started_at=new_started,
                ended_at=new_ended,
                exit_code=1,
                canceled=False,
                preflight_checks=[],
                output_lines=["new"],
            )

            invalid_dir = Path(tmpdir) / "bad_run"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            (invalid_dir / "metadata.json").write_text("{ bad json", encoding="utf-8")

            runs, warnings = rp.list_runs(config=config, limit=1)

            self.assertEqual(warnings, 1)
            self.assertEqual(len(runs), 1)
            self.assertIsNotNone(new_path)
            assert new_path is not None
            self.assertEqual(runs[0]["run_id"], new_path.name)
            self.assertNotEqual(old_path, new_path)

    def test_summarize_probe_error_returns_last_meaningful_line(self) -> None:
        raw = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'cv2'"
        self.assertEqual(rp.summarize_probe_error(raw), "ModuleNotFoundError: No module named 'cv2'")


if __name__ == "__main__":
    unittest.main()
