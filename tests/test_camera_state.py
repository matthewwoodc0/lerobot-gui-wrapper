from __future__ import annotations

import unittest

from robot_pipeline_app.camera_state import (
    assign_named_camera_source,
    assign_camera_role,
    camera_input_fps_summary,
    camera_mapping_summary,
    export_camera_preview_state,
    restore_camera_preview_state,
)


class CameraStateTests(unittest.TestCase):
    def test_assign_camera_role_rehomes_other_role_when_needed(self) -> None:
        assignment = assign_camera_role(
            config={"camera_laptop_index": 0, "camera_phone_index": 1},
            detected_indices=[0, 1, 2],
            detected_frame_sizes={1: (1280, 720)},
            role="laptop",
            index=1,
            fingerprint="cam-1",
        )

        self.assertTrue(assignment.ok)
        self.assertEqual(assignment.updated_config["camera_laptop_index"], 1)
        self.assertEqual(assignment.updated_config["camera_phone_index"], 0)
        self.assertEqual(assignment.updated_config["camera_laptop_fingerprint"], "cam-1")
        self.assertIn("Mapped laptop camera 1 (detected frame 1280x720).", assignment.messages)

    def test_assign_camera_role_rejects_single_port_conflict(self) -> None:
        assignment = assign_camera_role(
            config={"camera_laptop_index": 0, "camera_phone_index": 0},
            detected_indices=[0],
            detected_frame_sizes={},
            role="phone",
            index=0,
            fingerprint=None,
        )

        self.assertFalse(assignment.ok)
        self.assertEqual(
            assignment.messages,
            ("Could not assign role: laptop/phone must use two different ports.",),
        )

    def test_camera_summary_helpers_report_mapping_and_fps(self) -> None:
        config = {"camera_laptop_index": 3, "camera_phone_index": 7}
        self.assertEqual(camera_mapping_summary(config), "laptop=3 phone=7")
        self.assertEqual(
            camera_input_fps_summary(config, {3: 29.97}),
            "input fps laptop=30.0 phone=n/a",
        )

    def test_camera_summary_helpers_report_schema_based_mapping(self) -> None:
        config = {
            "camera_schema_json": (
                '{"wrist":{"index_or_path":2},"overhead":{"index_or_path":5},"side":{"index_or_path":7}}'
            )
        }
        self.assertEqual(camera_mapping_summary(config), "wrist=2 overhead=5 side=7")
        self.assertEqual(
            camera_input_fps_summary(config, {2: 29.97, 7: 14.5}),
            "input fps wrist=30.0 overhead=n/a side=14.5",
        )

    def test_assign_named_camera_source_swaps_conflicting_camera(self) -> None:
        assignment = assign_named_camera_source(
            config={
                "camera_schema_json": (
                    '{"wrist":{"index_or_path":0},"overhead":{"index_or_path":1},"side":{"index_or_path":2}}'
                )
            },
            detected_indices=[0, 1, 2, 3],
            detected_frame_sizes={1: (1280, 720)},
            camera_name="wrist",
            index=1,
            fingerprint="cam-1",
        )

        self.assertTrue(assignment.ok)
        self.assertIn('"wrist":{"index_or_path":1}', assignment.updated_config["camera_schema_json"])
        self.assertIn('"overhead":{"index_or_path":0}', assignment.updated_config["camera_schema_json"])
        self.assertEqual(assignment.updated_config["camera_wrist_fingerprint"], "cam-1")
        self.assertIn("Mapped wrist camera 1 (detected frame 1280x720).", assignment.messages)

    def test_export_and_restore_camera_preview_state_roundtrip(self) -> None:
        state = export_camera_preview_state(
            detected_indices=[0, 2],
            detected_frame_sizes={0: (640, 480)},
            detected_input_fps={0: 29.97},
            status_text="Preview refreshed",
            detected_ports_text="Detected open camera ports: 0, 2",
            scan_limit="16",
            live_fps_cap="12",
            live_enabled=True,
            pause_on_run=False,
            run_active=True,
            live_paused_for_run=True,
        )

        restored = restore_camera_preview_state(state)

        self.assertEqual(restored.detected_indices, [0, 2])
        self.assertEqual(restored.detected_frame_sizes, {0: (640, 480)})
        self.assertEqual(restored.detected_input_fps, {0: 29.97})
        self.assertEqual(restored.status_text, "Preview refreshed")
        self.assertEqual(restored.live_fps_cap, "12")
        self.assertTrue(restored.live_enabled)
        self.assertFalse(restored.pause_on_run)
        self.assertTrue(restored.run_active)
        self.assertTrue(restored.live_paused_for_run)


if __name__ == "__main__":
    unittest.main()
