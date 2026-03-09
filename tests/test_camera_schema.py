from __future__ import annotations

import unittest

from robot_pipeline_app.camera_schema import (
    apply_camera_schema_entries_to_config,
    build_observation_rename_map,
    camera_schema_entries_for_editor,
    format_observation_rename_map,
    resolve_camera_feature_mapping,
    resolve_camera_schema,
)
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES


class CameraSchemaTest(unittest.TestCase):
    def test_resolve_camera_schema_falls_back_to_legacy_fields(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        resolution = resolve_camera_schema(config)
        self.assertEqual(len(resolution.specs), 2)
        self.assertEqual(resolution.specs[0].name, config["camera_laptop_name"])
        self.assertEqual(resolution.specs[1].name, config["camera_phone_name"])

    def test_resolve_camera_schema_supports_n_cameras(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = (
            '{"camera1":{"index_or_path":0},"camera2":{"index_or_path":1},"camera3":{"index_or_path":2}}'
        )
        resolution = resolve_camera_schema(config)
        self.assertEqual([spec.name for spec in resolution.specs], ["camera1", "camera2", "camera3"])
        self.assertFalse(resolution.errors)

    def test_resolve_camera_feature_mapping_auto_maps_laptop_phone_to_camera12(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        mapping, error = resolve_camera_feature_mapping(
            config=config,
            runtime_keys={"laptop", "phone"},
            model_keys={"camera1", "camera2"},
        )
        self.assertIsNone(error)
        assert mapping is not None
        self.assertEqual(mapping["laptop"], "camera1")
        self.assertEqual(mapping["phone"], "camera2")

    def test_resolve_camera_feature_mapping_respects_explicit_map(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_policy_feature_map_json"] = (
            '{"observation.images.laptop":"observation.images.wrist","phone":"overhead"}'
        )
        mapping, error = resolve_camera_feature_mapping(
            config=config,
            runtime_keys={"laptop", "phone"},
            model_keys={"wrist", "overhead"},
        )
        self.assertIsNone(error)
        assert mapping is not None
        self.assertEqual(mapping, {"laptop": "wrist", "phone": "overhead"})

    def test_resolve_camera_feature_mapping_fails_on_count_mismatch(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        mapping, error = resolve_camera_feature_mapping(
            config=config,
            runtime_keys={"camera1", "camera2"},
            model_keys={"camera1", "camera2", "camera3"},
        )
        self.assertIsNone(mapping)
        assert error is not None
        self.assertIn("camera count mismatch", error)

    def test_camera_schema_entries_for_editor_reflects_runtime_schema(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = (
            '{"wrist":{"index_or_path":2},"overhead":{"index_or_path":5,"fps":15}}'
        )
        entries = camera_schema_entries_for_editor(config)
        self.assertEqual([entry.name for entry in entries], ["wrist", "overhead"])
        self.assertEqual(entries[0].source, 2)
        self.assertEqual(entries[1].fps, 15)

    def test_apply_camera_schema_entries_to_config_updates_runtime_and_legacy_fields(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        updated = apply_camera_schema_entries_to_config(
            config,
            [
                {"name": "wrist", "source": 3, "camera_type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5},
                {"name": "overhead", "source": 7, "camera_type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5},
                {"name": "side", "source": 8, "camera_type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5},
            ],
        )
        self.assertEqual(updated["camera_laptop_name"], "wrist")
        self.assertEqual(updated["camera_laptop_index"], 3)
        self.assertEqual(updated["camera_phone_name"], "overhead")
        self.assertEqual(updated["camera_phone_index"], 7)
        self.assertIn('"wrist":{"index_or_path":3}', updated["camera_schema_json"])

    def test_format_observation_rename_map(self) -> None:
        mapping = {"laptop": "camera1", "phone": "camera2"}
        rename_map = build_observation_rename_map(mapping)
        self.assertEqual(
            rename_map,
            {
                "observation.images.laptop": "observation.images.camera1",
                "observation.images.phone": "observation.images.camera2",
            },
        )
        self.assertIn('"observation.images.laptop":"observation.images.camera1"', format_observation_rename_map(mapping))


if __name__ == "__main__":
    unittest.main()
