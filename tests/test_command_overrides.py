from __future__ import annotations

import json
import unittest

from robot_pipeline_app.command_overrides import apply_command_overrides, get_flag_value, parse_custom_args


class CommandOverridesTest(unittest.TestCase):
    def test_apply_command_overrides_replaces_existing_value(self) -> None:
        base = ["python", "-m", "tool", "--dataset.repo_id=alice/demo_1", "--dataset.num_episodes=10"]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides={"dataset.repo_id": "alice/demo_2", "--dataset.num_episodes": "5"},
            custom_args_raw="",
        )
        self.assertIsNone(error)
        assert updated is not None
        self.assertIn("--dataset.repo_id=alice/demo_2", updated)
        self.assertIn("--dataset.num_episodes=5", updated)

    def test_apply_command_overrides_appends_new_value(self) -> None:
        base = ["python", "-m", "tool"]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides={"policy.path": "/tmp/model"},
            custom_args_raw="--foo bar",
        )
        self.assertIsNone(error)
        assert updated is not None
        self.assertIn("--policy.path=/tmp/model", updated)
        self.assertEqual(updated[-2:], ["--foo", "bar"])

    def test_apply_command_overrides_rejects_invalid_key(self) -> None:
        updated, error = apply_command_overrides(
            base_cmd=["python"],
            overrides={"bad key": "x"},
            custom_args_raw="",
        )
        self.assertIsNone(updated)
        self.assertEqual(error, "Invalid advanced option key: 'bad key'")

    def test_parse_custom_args_reports_unbalanced_quote(self) -> None:
        args, error = parse_custom_args('--foo "bar')
        self.assertIsNone(args)
        self.assertIsNotNone(error)

    def test_get_flag_value_prefers_last_occurrence(self) -> None:
        cmd = ["python", "--dataset.repo_id=alice/demo_1", "--dataset.repo_id=alice/demo_2"]
        self.assertEqual(get_flag_value(cmd, "dataset.repo_id"), "alice/demo_2")

    def test_apply_command_overrides_rewrites_rename_map_for_lerobot_record_module(self) -> None:
        base = ["python3", "-m", "lerobot.scripts.lerobot_record", "--foo=bar"]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides=None,
            custom_args_raw='--rename_map={"observation.images.laptop":"observation.images.camera1"}',
        )
        self.assertIsNone(error)
        assert updated is not None
        self.assertIn(
            '--dataset.rename_map={"observation.images.laptop":"observation.images.camera1"}',
            updated,
        )
        self.assertFalse(any(part.startswith("--rename_map") for part in updated))

    def test_apply_command_overrides_keeps_rename_map_for_non_lerobot_record_module(self) -> None:
        base = ["python3", "-m", "some.other.module"]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides=None,
            custom_args_raw='--rename_map={"a":"b"}',
        )
        self.assertIsNone(error)
        assert updated is not None
        self.assertIn('--rename_map={"a":"b"}', updated)

    def test_apply_command_overrides_rewrites_robot_cameras_keys_using_rename_map(self) -> None:
        base = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_record",
            '--robot.cameras={"laptop":{"index_or_path":1},"phone":{"index_or_path":0}}',
        ]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides=None,
            custom_args_raw=(
                '--rename_map={"observation.images.laptop":"observation.images.camera1",'
                '"observation.images.phone":"observation.images.camera2"}'
            ),
        )
        self.assertIsNone(error)
        assert updated is not None
        cameras_raw = get_flag_value(updated, "robot.cameras")
        self.assertIsNotNone(cameras_raw)
        payload = json.loads(str(cameras_raw))
        self.assertEqual(sorted(payload.keys()), ["camera1", "camera2"])

    def test_apply_command_overrides_rewrites_rename_map_for_lerobot_record_legacy_module(self) -> None:
        base = ["python3", "-m", "lerobot.record", "--foo=bar"]
        updated, error = apply_command_overrides(
            base_cmd=base,
            overrides=None,
            custom_args_raw='--rename_map={"observation.images.a":"observation.images.camera1"}',
        )
        self.assertIsNone(error)
        assert updated is not None
        self.assertIn(
            '--dataset.rename_map={"observation.images.a":"observation.images.camera1"}',
            updated,
        )
        self.assertFalse(any(part.startswith("--rename_map") for part in updated))


if __name__ == "__main__":
    unittest.main()
