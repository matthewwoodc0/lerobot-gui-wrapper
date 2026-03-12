from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.profile_io import export_profile, import_profile, profile_preset_payloads


class ProfileIoTest(unittest.TestCase):
    def test_export_profile_writes_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            output = Path(tmpdir) / "profile.yaml"
            result = export_profile(config, output_path=output, include_paths=True)
            self.assertTrue(result.ok)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "community_profile.v1")
            self.assertIn("paths", payload)

    def test_export_profile_skips_machine_specific_robot_fields_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["follower_port"] = "/dev/ttyUSB7"
            config["leader_port"] = "/dev/ttyUSB8"
            config["follower_calibration_path"] = "/tmp/follower.json"
            config["leader_calibration_path"] = "/tmp/leader.json"
            output = Path(tmpdir) / "profile.yaml"

            result = export_profile(config, output_path=output, include_paths=False)

            self.assertTrue(result.ok)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("paths", payload)
            self.assertNotIn("port", payload["robot"]["follower"])
            self.assertNotIn("port", payload["robot"]["leader"])
            self.assertNotIn("calibration_path", payload["robot"]["follower"])
            self.assertNotIn("calibration_path", payload["robot"]["leader"])

    def test_import_profile_rejects_unsupported_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = Path(tmpdir) / "bad_profile.yaml"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "community_profile.v1",
                        "robot": {"follower": {"id": "arm_a"}, "leader": {"id": "arm_b"}},
                        "unknown_key": "nope",
                    }
                ),
                encoding="utf-8",
            )
            result = import_profile(dict(DEFAULT_CONFIG_VALUES), input_path=profile)
            self.assertFalse(result.ok)
            self.assertIn("unsupported keys", result.message)

    def test_import_profile_applies_fields_and_skips_paths_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = Path(tmpdir) / "profile.yaml"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "community_profile.v1",
                        "robot": {
                            "follower": {"id": "arm_alpha", "port": "/dev/ttyUSB7"},
                            "leader": {"id": "arm_beta", "port": "/dev/ttyUSB8"},
                        },
                        "camera": {
                            "schema_json": {"cam1": {"index_or_path": 0}},
                            "rename_flag": "dataset.rename_map",
                        },
                        "defaults": {"camera_fps": 15, "compat_policy": "latest_plus_n_minus_1"},
                        "paths": {"lerobot_dir": "/opt/lerobot_shared"},
                    }
                ),
                encoding="utf-8",
            )

            result = import_profile(dict(DEFAULT_CONFIG_VALUES), input_path=profile, apply_paths=False)
            self.assertTrue(result.ok)
            assert result.updated_config is not None
            self.assertEqual(result.updated_config["follower_robot_id"], "arm_alpha")
            self.assertEqual(result.updated_config["camera_fps"], 15)
            self.assertEqual(result.updated_config["camera_rename_flag"], "dataset.rename_map")
            self.assertNotEqual(result.updated_config["lerobot_dir"], "/opt/lerobot_shared")
            self.assertEqual(result.updated_config["follower_port"], DEFAULT_CONFIG_VALUES["follower_port"])
            self.assertEqual(result.updated_config["leader_port"], DEFAULT_CONFIG_VALUES["leader_port"])
            self.assertIn("lerobot_dir", result.skipped_keys)
            self.assertIn("follower_port", result.skipped_keys)
            self.assertIn("leader_port", result.skipped_keys)

    def test_import_profile_applies_paths_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = Path(tmpdir) / "profile.yaml"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "community_profile.v1",
                        "paths": {
                            "lerobot_dir": "/opt/lerobot_shared",
                            "follower_port": "/dev/ttyUSB7",
                            "leader_calibration_path": "/tmp/leader.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = import_profile(dict(DEFAULT_CONFIG_VALUES), input_path=profile, apply_paths=True)
            self.assertTrue(result.ok)
            assert result.updated_config is not None
            self.assertEqual(result.updated_config["lerobot_dir"], "/opt/lerobot_shared")
            self.assertEqual(result.updated_config["follower_port"], "/dev/ttyUSB7")
            self.assertEqual(result.updated_config["leader_calibration_path"], "/tmp/leader.json")

    def test_profile_presets_expose_camera_schema_and_robot_defaults(self) -> None:
        payload = profile_preset_payloads()["SO-101 Lab Dual Cam"]
        self.assertEqual(payload["robot"]["follower"]["type"], "so101_follower")
        self.assertIn("wrist", payload["camera"]["schema_json"])


if __name__ == "__main__":
    unittest.main()
