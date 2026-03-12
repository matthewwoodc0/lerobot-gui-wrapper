from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.hardware_workflows import (
    MotorSetupSupport,
    ReplayRequest,
    ReplaySupport,
    apply_motor_setup_success,
    build_motor_setup_preflight_checks,
    build_motor_setup_result_summary,
    build_replay_readiness_summary,
    build_motor_setup_request_and_command,
    build_replay_preflight_checks,
    build_replay_request_and_command,
    discover_replay_episodes,
)


class HardwareWorkflowsTests(unittest.TestCase):
    def test_build_replay_request_and_command_uses_local_dataset_path_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "alice" / "demo"
            dataset_path.mkdir(parents=True)

            config = dict(DEFAULT_CONFIG_VALUES)
            config["follower_port"] = "/dev/ttyUSB0"
            config["follower_robot_id"] = "arm_follower"
            config["follower_robot_type"] = "so100_follower"
            config["follower_calibration_path"] = str(Path(tmpdir) / "calibration" / "follower.json")

            support = ReplaySupport(
                available=True,
                entrypoint="lerobot.replay",
                detail="Replay entrypoint detected.",
                supported_flags=("dataset.path", "dataset.episode", "robot.type", "robot.port", "robot.id"),
                dataset_flag=None,
                dataset_root_flag=None,
                dataset_path_flag="dataset.path",
                episode_flag="dataset.episode",
                robot_type_flag="robot.type",
                robot_port_flag="robot.port",
                robot_id_flag="robot.id",
                calibration_dir_flag="robot.calibration_dir",
            )

            with patch("robot_pipeline_app.hardware_replay.probe_replay_support", return_value=support):
                request, cmd, returned_support, error = build_replay_request_and_command(
                    config=config,
                    dataset_repo_id="alice/demo",
                    episode_raw="3",
                    dataset_path_raw=str(dataset_path),
                )

        self.assertIsNone(error)
        self.assertIs(returned_support, support)
        assert request is not None
        assert cmd is not None
        self.assertEqual(request.dataset_repo_id, "alice/demo")
        self.assertEqual(request.dataset_path, dataset_path)
        self.assertEqual(request.episode_index, 3)
        self.assertEqual(cmd[1:3], ["-m", "lerobot.replay"])
        self.assertIn(f"--dataset.path={dataset_path}", cmd)
        self.assertIn("--dataset.episode=3", cmd)
        self.assertIn("--robot.type=so100_follower", cmd)
        self.assertIn("--robot.port=/dev/ttyUSB0", cmd)
        self.assertIn("--robot.id=arm_follower", cmd)
        self.assertIn(f"--robot.calibration_dir={dataset_path.parent.parent / 'calibration'}", cmd)

    def test_replay_preflight_flags_missing_episode_in_local_dataset(self) -> None:
        request_support = ReplaySupport(
            available=True,
            entrypoint="lerobot.replay",
            detail="Replay entrypoint detected.",
            supported_flags=("dataset.path", "dataset.episode"),
            dataset_flag=None,
            dataset_root_flag=None,
            dataset_path_flag="dataset.path",
            episode_flag="dataset.episode",
            robot_type_flag=None,
            robot_port_flag="robot.port",
            robot_id_flag=None,
            calibration_dir_flag=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            dataset_path.mkdir()
            config = dict(DEFAULT_CONFIG_VALUES)
            config["follower_port"] = "/dev/ttyUSB0"

            with patch("robot_pipeline_app.hardware_replay.probe_replay_support", return_value=request_support):
                request, _cmd, support, error = build_replay_request_and_command(
                    config=config,
                    dataset_repo_id="alice/demo",
                    episode_raw="5",
                    dataset_path_raw=str(dataset_path),
                )

            self.assertIsNone(error)
            assert request is not None

            with patch("robot_pipeline_app.hardware_replay.collect_local_dataset_episode_indices", return_value=([0, 1, 2], None)):
                checks = build_replay_preflight_checks(config=config, request=request, support=support)

        detail_by_name = {name: (level, detail) for level, name, detail in checks}
        self.assertEqual(detail_by_name["Replay episode"][0], "FAIL")
        self.assertIn("Episode 5 is not present", detail_by_name["Replay episode"][1])

    def test_build_motor_setup_request_and_command_uses_detected_setup_flags(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["leader_robot_type"] = "so101_leader"
        config["leader_robot_id"] = "leader_old"

        support = MotorSetupSupport(
            available=True,
            entrypoint="lerobot.setup_motors",
            detail="Motor setup entrypoint detected.",
            supported_flags=("robot.role", "robot.type", "robot.port", "robot.id", "robot.new_id", "robot.baudrate"),
            role_flag="robot.role",
            type_flag="robot.type",
            port_flag="robot.port",
            id_flag="robot.id",
            new_id_flag="robot.new_id",
            baudrate_flag="robot.baudrate",
            uses_calibrate_fallback=False,
        )

        with patch("robot_pipeline_app.hardware_motor_setup.probe_motor_setup_support", return_value=support):
            request, cmd, returned_support, error = build_motor_setup_request_and_command(
                config=config,
                role="leader",
                port_raw="/dev/ttyUSB1",
                robot_id_raw="leader_old",
                new_id_raw="leader_new",
                baudrate_raw="1000000",
                robot_type_raw="so101_leader",
            )

        self.assertIsNone(error)
        self.assertIs(returned_support, support)
        assert request is not None
        assert cmd is not None
        self.assertEqual(request.role, "leader")
        self.assertEqual(request.port, "/dev/ttyUSB1")
        self.assertEqual(request.baudrate, 1_000_000)
        self.assertEqual(cmd[1:3], ["-m", "lerobot.setup_motors"])
        self.assertIn("--robot.role=leader", cmd)
        self.assertIn("--robot.type=so101_leader", cmd)
        self.assertIn("--robot.port=/dev/ttyUSB1", cmd)
        self.assertIn("--robot.id=leader_old", cmd)
        self.assertIn("--robot.new_id=leader_new", cmd)
        self.assertIn("--robot.baudrate=1000000", cmd)

        checks = build_motor_setup_preflight_checks(request=request, support=support)
        detail_by_name = {name: (level, detail) for level, name, detail in checks}
        self.assertEqual(detail_by_name["Motor id reassignment"][0], "PASS")
        self.assertEqual(detail_by_name["Motor baudrate"][0], "PASS")

    def test_motor_setup_calibrate_fallback_warns_and_preserves_current_id(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_robot_type"] = "so100_follower"
        config["follower_robot_id"] = "follower_old"

        support = MotorSetupSupport(
            available=True,
            entrypoint="lerobot.calibrate",
            detail="Calibration fallback only.",
            supported_flags=(),
            role_flag=None,
            type_flag=None,
            port_flag=None,
            id_flag=None,
            new_id_flag=None,
            baudrate_flag=None,
            uses_calibrate_fallback=True,
        )

        with patch("robot_pipeline_app.hardware_motor_setup.probe_motor_setup_support", return_value=support):
            request, cmd, _returned_support, error = build_motor_setup_request_and_command(
                config=config,
                role="follower",
                port_raw="/dev/ttyUSB2",
                robot_id_raw="follower_old",
                new_id_raw="follower_new",
                baudrate_raw="2000000",
                robot_type_raw="so100_follower",
            )

        self.assertIsNone(error)
        assert request is not None
        assert cmd is not None
        self.assertEqual(cmd[1:3], ["-m", "lerobot.calibrate"])

        checks = build_motor_setup_preflight_checks(request=request, support=support)
        detail_by_name = {name: (level, detail) for level, name, detail in checks}
        self.assertEqual(detail_by_name["Motor id reassignment"][0], "WARN")
        self.assertEqual(detail_by_name["Motor baudrate"][0], "WARN")

        updated = apply_motor_setup_success(config, request=request, support=support)
        self.assertEqual(updated["follower_port"], "/dev/ttyUSB2")
        self.assertEqual(updated["follower_robot_type"], "so100_follower")
        self.assertEqual(updated["follower_robot_id"], "follower_old")

        summary = build_motor_setup_result_summary(
            previous_config=config,
            updated_config=updated,
            request=request,
            support=support,
        )
        self.assertIn("Motor id update: Informational only.", summary)
        self.assertIn("Baudrate update: Informational only.", summary)

    def test_discover_replay_episodes_reports_manual_fallback_on_scan_error(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            dataset_path.mkdir()
            with patch(
                "robot_pipeline_app.hardware_replay.collect_local_dataset_episode_indices",
                return_value=([], "episodes.jsonl missing"),
            ):
                discovery = discover_replay_episodes(config, "alice/demo", dataset_path_raw=str(dataset_path))

        self.assertTrue(discovery.manual_entry_only)
        self.assertEqual(discovery.scan_error, "episodes.jsonl missing")

    def test_replay_readiness_summary_includes_missing_dataset_episode_and_robot_config(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        request = ReplayRequest(
            dataset_repo_id="alice/demo",
            dataset_path=None,
            episode_index=5,
            robot_type="",
            robot_port="",
            robot_id="",
            calibration_dir=None,
        )
        support = ReplaySupport(
            available=True,
            entrypoint="lerobot.replay",
            detail="Replay entrypoint detected.",
            supported_flags=("dataset.path", "dataset.episode"),
            dataset_flag=None,
            dataset_root_flag=None,
            dataset_path_flag="dataset.path",
            episode_flag="dataset.episode",
            robot_type_flag="robot.type",
            robot_port_flag="robot.port",
            robot_id_flag="robot.id",
            calibration_dir_flag="robot.calibration_dir",
        )

        summary = build_replay_readiness_summary(config=config, request=request, support=support)

        self.assertIn("[FAIL] Local dataset path", summary)
        self.assertIn("[FAIL] Replay robot port", summary)
        self.assertIn("[WARN] Replay robot id", summary)
        self.assertIn("[WARN] Replay robot type", summary)


if __name__ == "__main__":
    unittest.main()
