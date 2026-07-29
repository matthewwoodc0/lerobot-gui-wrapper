from __future__ import annotations

import unittest
from pathlib import Path

from robot_pipeline_app.compat import _parse_help_flags
from robot_pipeline_app.commands import build_lerobot_deploy_command
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from unittest.mock import patch


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "compat"


class CompatHelpFixtureTests(unittest.TestCase):
    def test_0_6_help_fixtures_expose_required_workflow_flags(self) -> None:
        record = _parse_help_flags((FIXTURES / "lerobot_0_6_record_help.txt").read_text(encoding="utf-8"))
        train = _parse_help_flags((FIXTURES / "lerobot_0_6_train_help.txt").read_text(encoding="utf-8"))
        teleop = _parse_help_flags((FIXTURES / "lerobot_0_6_teleop_help.txt").read_text(encoding="utf-8"))
        replay = _parse_help_flags((FIXTURES / "lerobot_0_6_replay_help.txt").read_text(encoding="utf-8"))
        calibrate = _parse_help_flags((FIXTURES / "lerobot_0_6_calibrate_help.txt").read_text(encoding="utf-8"))
        eval_flags = _parse_help_flags((FIXTURES / "lerobot_0_6_eval_help.txt").read_text(encoding="utf-8"))
        rollout = _parse_help_flags((FIXTURES / "lerobot_0_6_rollout_help.txt").read_text(encoding="utf-8"))
        viz = _parse_help_flags((FIXTURES / "lerobot_0_6_dataset_viz_help.txt").read_text(encoding="utf-8"))

        self.assertIn("dataset.repo_id", record)
        self.assertIn("policy.path", record)
        self.assertIn("policy.type", train)
        self.assertIn("dataset.repo_id", train)
        self.assertIn("robot.port", teleop)
        self.assertIn("dataset.episode", replay)
        self.assertIn("robot.type", calibrate)
        self.assertIn("policy.path", eval_flags)
        self.assertIn("env.type", eval_flags)
        self.assertIn("policy.path", rollout)
        self.assertIn("strategy.type", rollout)
        self.assertTrue(viz)  # dataset viz help is non-empty

    def test_deploy_prefers_rollout_when_supported(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/tty.fake"
        config["leader_port"] = "/dev/tty.fake-leader"
        config["compat_probe_enabled"] = True

        class Caps:
            supports_rollout = True
            rollout_entrypoint = "lerobot.scripts.lerobot_rollout"
            rollout_policy_path_flag = "policy.path"
            rollout_strategy_type_flag = "strategy.type"
            rollout_task_flag = "task"
            rollout_duration_flag = "duration"
            preferred_deploy_path = "rollout"
            record_entrypoint = "lerobot.scripts.lerobot_record"
            policy_path_flag = "policy.path"

        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=Caps()), patch(
            "robot_pipeline_app.commands.get_cached_lerobot_capabilities",
            return_value=Caps(),
        ), patch(
            "robot_pipeline_app.commands.resolve_rollout_entrypoint",
            return_value="lerobot.scripts.lerobot_rollout",
        ):
            cmd, path = build_lerobot_deploy_command(
                config=config,
                policy_path="/tmp/model",
                task="pick",
                duration_s=30,
                num_episodes=2,
                dataset_repo_id="alice/eval_demo",
            )

        self.assertEqual(path, "rollout")
        self.assertTrue(any("lerobot_rollout" in part or "rollout" in part for part in cmd))
        self.assertTrue(any(part.startswith("--policy.path=") for part in cmd))
        self.assertTrue(any(part.startswith("--strategy.type=") for part in cmd))

    def test_deploy_falls_back_to_record_policy_when_rollout_missing(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["follower_port"] = "/dev/tty.fake"
        config["leader_port"] = "/dev/tty.fake-leader"
        config["compat_probe_enabled"] = True

        class Caps:
            supports_rollout = False
            rollout_entrypoint = ""
            rollout_policy_path_flag = None
            rollout_strategy_type_flag = None
            rollout_task_flag = None
            rollout_duration_flag = None
            preferred_deploy_path = "record_policy_path"
            record_entrypoint = "lerobot.scripts.lerobot_record"
            policy_path_flag = "policy.path"

        with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=Caps()), patch(
            "robot_pipeline_app.commands.get_cached_lerobot_capabilities",
            return_value=Caps(),
        ), patch(
            "robot_pipeline_app.commands.resolve_rollout_entrypoint",
            return_value="",
        ):
            cmd, path = build_lerobot_deploy_command(
                config=config,
                policy_path="/tmp/model",
                task="pick",
                duration_s=30,
                num_episodes=2,
                dataset_repo_id="alice/eval_demo",
            )

        self.assertEqual(path, "record_policy_path")
        self.assertTrue(any("record" in part for part in cmd))
        self.assertTrue(any(part.startswith("--policy.path=") for part in cmd))


if __name__ == "__main__":
    unittest.main()
