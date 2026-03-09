from __future__ import annotations

import unittest

from robot_pipeline_app.camera_schema import CameraSpec
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.gui_runtime_camera_feed import RuntimeCameraFeed, resolve_runtime_feed_specs


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class RuntimeCameraFeedTests(unittest.TestCase):
    def test_resolve_runtime_feed_specs_uses_all_runtime_cameras(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["camera_schema_json"] = (
            '{"wrist":{"index_or_path":2},"overhead":{"index_or_path":5},"side":{"index_or_path":7}}'
        )

        specs, warnings, errors = resolve_runtime_feed_specs(config)

        self.assertEqual([spec.name for spec in specs], ["wrist", "overhead", "side"])
        self.assertEqual([spec.source for spec in specs], [2, 5, 7])
        self.assertEqual(warnings, [])
        self.assertEqual(errors, [])

    def test_start_marks_feed_unavailable_when_cv2_probe_failed(self) -> None:
        feed = object.__new__(RuntimeCameraFeed)
        feed._camera_specs = [
            CameraSpec(name="wrist", source=2, camera_type="opencv", width=640, height=480, fps=30, warmup_s=5)
        ]
        feed.cv2_probe_ok = False
        feed.cv2_probe_error = "missing cv2"
        feed._active = False
        details: list[str] = []
        scheduled: list[int] = []
        feed.refresh_schema = lambda: None  # type: ignore[method-assign]
        feed._set_unavailable_state = lambda detail: details.append(detail)  # type: ignore[method-assign]
        feed._schedule_refresh = lambda delay_ms=0: scheduled.append(delay_ms)  # type: ignore[method-assign]

        RuntimeCameraFeed.start(feed)

        self.assertTrue(feed._active)
        self.assertEqual(scheduled, [])
        self.assertEqual(len(details), 1)
        self.assertIn("Live feed unavailable", details[0])

    def test_apply_frames_draws_placeholder_for_unavailable_source(self) -> None:
        feed = object.__new__(RuntimeCameraFeed)
        feed._camera_specs = [
            CameraSpec(name="wrist", source=0, camera_type="opencv", width=640, height=480, fps=30, warmup_s=5)
        ]
        state_var = _FakeVar()
        placeholders: list[tuple[str, str]] = []
        rendered: list[str] = []
        feed._cards = {
            "wrist": {
                "canvas": "canvas-1",
                "state_var": state_var,
                "photo": None,
            }
        }
        feed.status_var = _FakeVar()
        feed._draw_placeholder = lambda canvas, text: placeholders.append((canvas, text))  # type: ignore[method-assign]
        feed._render_frame = lambda spec, frame: rendered.append(spec.name)  # type: ignore[method-assign]

        RuntimeCameraFeed._apply_frames(feed, {"wrist": None})

        self.assertEqual(placeholders, [("canvas-1", "Unavailable")])
        self.assertEqual(rendered, [])
        self.assertIn("Unable to open source 0", state_var.get())


if __name__ == "__main__":
    unittest.main()
