from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.profile_io import export_profile, import_profile


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
            self.assertIn("lerobot_dir", result.skipped_keys)

    def test_import_profile_applies_paths_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = Path(tmpdir) / "profile.yaml"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "community_profile.v1",
                        "paths": {"lerobot_dir": "/opt/lerobot_shared"},
                    }
                ),
                encoding="utf-8",
            )
            result = import_profile(dict(DEFAULT_CONFIG_VALUES), input_path=profile, apply_paths=True)
            self.assertTrue(result.ok)
            assert result.updated_config is not None
            self.assertEqual(result.updated_config["lerobot_dir"], "/opt/lerobot_shared")


if __name__ == "__main__":
    unittest.main()
