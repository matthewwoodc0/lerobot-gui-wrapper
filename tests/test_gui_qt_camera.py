from __future__ import annotations

import os
import time
import unittest
from typing import Callable
from unittest.mock import patch

try:
    from robot_pipeline_app.gui_qt_app import ensure_qt_application, qt_available
    from robot_pipeline_app.gui_qt_camera import QtDualCameraPreview
except Exception as exc:  # pragma: no cover - exercised only when Qt imports fail
    ensure_qt_application = None  # type: ignore[assignment]
    QtDualCameraPreview = None  # type: ignore[assignment]
    _QT_AVAILABLE, _QT_REASON = False, str(exc)
else:
    _QT_AVAILABLE, _QT_REASON = qt_available()


class _FakeCapture:
    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def set(self, _prop: object, _value: object) -> bool:
        return True

    def get(self, _prop: object) -> float:
        return 30.0


class _FakeClosedCapture(_FakeCapture):
    def isOpened(self) -> bool:
        return False

    def get(self, _prop: object) -> float:
        return 0.0


class _FakeFrame:
    shape = (480, 640, 3)


class _DelayedFrameCapture(_FakeCapture):
    def __init__(self, failures_before_frame: int = 2) -> None:
        self.failures_before_frame = failures_before_frame
        self.released = False

    def read(self) -> tuple[bool, object | None]:
        if self.failures_before_frame > 0:
            self.failures_before_frame -= 1
            return False, None
        return True, _FakeFrame()

    def release(self) -> None:
        self.released = True


class _FakeCv2Module:
    CAP_AVFOUNDATION = 1200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 6

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def VideoCapture(self, index: int, backend: int | None = None) -> _FakeCapture:
        self.calls.append((index, backend))
        if backend == self.CAP_AVFOUNDATION:
            return _FakeCapture()
        return _FakeClosedCapture()


@unittest.skipUnless(_QT_AVAILABLE, _QT_REASON or "PySide6 unavailable")
class GuiQtCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app, _ = ensure_qt_application(["robot_pipeline.py", "gui"])

    def _make_preview(
        self,
        *,
        config: dict[str, object] | None = None,
        append_log: Callable[[str], None] | None = None,
    ) -> "QtDualCameraPreview":
        with patch("robot_pipeline_app.gui_qt_camera.probe_module_import", return_value=(True, "")):
            preview = QtDualCameraPreview(config=config or {}, append_log=append_log or (lambda _msg: None))
        return preview

    def _wait_for_scan(self, preview: "QtDualCameraPreview", timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while preview._scan_in_progress and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

    def test_candidate_scan_indices_on_macos_use_configured_runtime_sources_only(self) -> None:
        preview = self._make_preview(
            config={
                "camera_schema_json": (
                    '{"front":{"index_or_path":6},"wrist":{"index_or_path":2},"ignored":{"index_or_path":20}}'
                )
            }
        )
        self.addCleanup(preview.close)

        with patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"):
            candidates = preview._candidate_scan_indices(14)

        self.assertEqual(candidates, [2, 6])

    def test_macos_fallback_scan_indices_cover_extended_common_range(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        with patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"):
            fallback = preview._macos_fallback_scan_indices(6, [4, 6])

        self.assertEqual(fallback, [0, 1, 2, 3, 5])

    def test_scan_camera_ports_on_macos_falls_back_when_no_configured_sources_exist(self) -> None:
        logs: list[str] = []
        preview = self._make_preview(
            config={"camera_schema_json": '{"front":{"index_or_path":"/dev/camera0"}}'},
            append_log=logs.append,
        )
        self.addCleanup(preview.close)

        def _probe(index: int) -> tuple[bool, str]:
            if index == 0:
                return True, "frame=640x480"
            return False, "camera not opened"

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"),
            patch.object(preview, "_scan_index_with_preview_capture", side_effect=_probe),
        ):
            preview.scan_camera_ports()
            self._wait_for_scan(preview)

        self.assertFalse(preview._scan_in_progress)
        self.assertEqual(preview._detected_indices, [0])
        self.assertEqual(preview.detected_ports_label.text(), "Detected open camera ports: 0")
        self.assertEqual(
            preview.status_label.text(),
            "Scan complete using macOS fallback ports. Click Refresh Camera Preview to open detected cameras safely.",
        )
        self.assertTrue(any("no configured runtime camera indices" in message for message in logs))

    def test_scan_camera_ports_clears_in_progress_and_updates_ui_after_worker_finishes(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch.object(preview, "_candidate_scan_indices", return_value=[0]),
            patch.object(preview, "_scan_index_with_preview_capture", return_value=(True, "frame=640x480")),
        ):
            preview.scan_camera_ports()
            self.assertTrue(preview._scan_in_progress)
            self._wait_for_scan(preview)

        self.assertFalse(preview._scan_in_progress)
        self.assertTrue(preview.scan_button.isEnabled())
        self.assertEqual(preview._detected_indices, [0])
        self.assertEqual(preview.detected_ports_label.text(), "Detected open camera ports: 0")
        self.assertEqual(
            preview.status_label.text(),
            "Scan complete. Click Refresh Camera Preview to open detected cameras safely.",
        )

    def test_scan_camera_ports_does_not_auto_refresh_after_scan(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"),
            patch.object(preview, "_candidate_scan_indices", return_value=[0, 1]),
            patch.object(
                preview,
                "_scan_index_with_preview_capture",
                side_effect=[(True, "frame=640x480"), (False, "no preview frame")],
            ) as mocked_probe,
            patch.object(preview, "refresh_camera_previews") as mocked_refresh,
        ):
            preview.scan_camera_ports()
            self._wait_for_scan(preview)

        self.assertEqual(preview._detected_indices, [0])
        mocked_refresh.assert_not_called()
        self.assertEqual(mocked_probe.call_count, 2)
        self.assertEqual([call.args[0] for call in mocked_probe.call_args_list], [0, 1])

    def test_scan_camera_ports_surfaces_probe_failure_in_status_and_log(self) -> None:
        logs: list[str] = []
        preview = self._make_preview(append_log=logs.append)
        self.addCleanup(preview.close)

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"),
            patch.object(preview, "_candidate_scan_indices", return_value=[4, 6]),
            patch.object(preview, "_macos_fallback_scan_indices", return_value=[]),
            patch.object(
                preview,
                "_scan_index_with_preview_capture",
                side_effect=[
                    (False, "OpenCV: camera access has been denied."),
                    (False, "OpenCV: camera access has been denied."),
                ],
            ),
        ):
            preview.scan_camera_ports()
            self._wait_for_scan(preview)

        self.assertEqual(preview.detected_ports_label.text(), "Detected open camera ports: none found")
        self.assertEqual(
            preview.status_label.text(),
            "Scan failed on configured ports: OpenCV: camera access has been denied.",
        )
        self.assertIn("Camera scan failed on port 4: OpenCV: camera access has been denied.", logs)
        self.assertIn("Camera scan failed on port 6: OpenCV: camera access has been denied.", logs)

    def test_scan_camera_ports_uses_fallback_ports_after_configured_failures(self) -> None:
        logs: list[str] = []
        preview = self._make_preview(append_log=logs.append)
        self.addCleanup(preview.close)

        def _probe(index: int) -> tuple[bool, str]:
            if index == 3:
                return True, "frame=640x480"
            return False, "camera not opened"

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"),
            patch.object(preview, "_candidate_scan_indices", return_value=[4, 6]),
            patch.object(preview, "_scan_index_with_preview_capture", side_effect=_probe) as mocked_probe,
        ):
            preview.scan_camera_ports()
            self._wait_for_scan(preview)

        self.assertEqual(preview._detected_indices, [3])
        self.assertEqual(preview.detected_ports_label.text(), "Detected open camera ports: 3")
        self.assertEqual(
            preview.status_label.text(),
            "Scan complete using macOS fallback ports. Click Refresh Camera Preview to open detected cameras safely.",
        )
        self.assertEqual([call.args[0] for call in mocked_probe.call_args_list], [4, 6, 0, 1, 2, 3, 5])
        self.assertTrue(any("fallback camera scan succeeded after configured camera ports failed to open" in message for message in logs))

    def test_refresh_camera_previews_skips_when_refresh_already_running(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)
        preview._detected_indices = [0]
        preview._rebuild_cards()
        preview._refresh_in_progress = True

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch.object(preview, "_capture_frame_with_fps") as mocked_capture,
            patch.object(preview, "_capture_live_frame_with_fps") as mocked_live_capture,
        ):
            preview.refresh_camera_previews()

        mocked_capture.assert_not_called()
        mocked_live_capture.assert_not_called()

    def test_scan_index_with_preview_capture_uses_live_capture_path(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        with (
            patch.object(preview, "_capture_live_frame_with_fps", return_value=(_FakeFrame(), 22.0)) as mocked_capture,
            patch.object(preview, "_release_pooled_capture_locked") as mocked_release,
        ):
            opened, detail = preview._scan_index_with_preview_capture(3)

        self.assertTrue(opened)
        self.assertEqual(detail, "frame=640x480 @ 22.0 FPS")
        mocked_capture.assert_called_once_with(3)
        mocked_release.assert_called_once_with(3)

    def test_manual_refresh_uses_live_capture_path_and_releases_pool_after_refresh(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)
        preview._detected_indices = [0]
        preview._rebuild_cards()

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch.object(preview._live_timer, "isActive", return_value=False),
            patch.object(preview, "_capture_live_frame_with_fps", return_value=(_FakeFrame(), 30.0)) as mocked_live_capture,
            patch.object(preview, "_capture_frame_with_fps") as mocked_capture,
            patch.object(preview, "_render_frame") as mocked_render,
            patch.object(preview, "_release_all_pooled_captures") as mocked_release,
        ):
            preview.refresh_camera_previews()

        mocked_live_capture.assert_called_once_with(0)
        mocked_capture.assert_not_called()
        mocked_render.assert_called_once()
        mocked_release.assert_called_once()
        self.assertEqual(preview._detected_cards[0]["fps"].text(), "Input: 640x480 @ 30.0 FPS")
        self.assertEqual(preview.status_label.text(), "Preview refreshed for 1/1 detected ports.")

    def test_manual_refresh_reports_missing_frames_and_resets_fps(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)
        preview._detected_indices = [0, 1]
        preview._rebuild_cards()

        with (
            patch.object(preview, "_ensure_cv2_module", return_value=True),
            patch.object(preview._live_timer, "isActive", return_value=False),
            patch.object(preview, "_capture_live_frame_with_fps", side_effect=[(None, None), (None, None)]),
            patch.object(preview, "_release_all_pooled_captures") as mocked_release,
        ):
            preview.refresh_camera_previews()

        mocked_release.assert_called_once()
        self.assertEqual(preview._detected_cards[0]["fps"].text(), "Input: n/a @ n/a FPS")
        self.assertEqual(preview._detected_cards[1]["fps"].text(), "Input: n/a @ n/a FPS")
        self.assertEqual(preview.status_label.text(), "No preview frame received from detected ports: 0, 1.")

    def test_camera_preview_cards_use_main_page_scroll_instead_of_nested_scroll_area(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        self.assertFalse(hasattr(preview, "_scroll"))
        self.assertGreaterEqual(preview.layout().indexOf(preview._cards_wrap), 0)

    def test_open_capture_prefers_avfoundation_backend_on_macos(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)
        fake_cv2 = _FakeCv2Module()
        preview._cv2_module = fake_cv2

        with patch("robot_pipeline_app.gui_qt_camera.sys.platform", "darwin"):
            capture = preview._open_capture(7)

        self.assertIsNotNone(capture)
        self.assertEqual(fake_cv2.calls[0], (7, fake_cv2.CAP_AVFOUNDATION))

    def test_capture_frame_with_fps_waits_for_delayed_first_frame(self) -> None:
        preview = self._make_preview(config={"camera_warmup_s": 1})
        self.addCleanup(preview.close)
        preview._cv2_module = _FakeCv2Module()
        delayed_capture = _DelayedFrameCapture(failures_before_frame=2)

        with (
            patch.object(preview, "_open_capture", return_value=delayed_capture),
            patch("robot_pipeline_app.gui_qt_camera.time.sleep", return_value=None),
        ):
            frame, fps = preview._capture_frame_with_fps(0)

        self.assertIsNotNone(frame)
        self.assertEqual(fps, 30.0)
        self.assertTrue(delayed_capture.released)

    def test_live_preview_has_no_pause_on_run_option(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        self.assertFalse(hasattr(preview, "pause_on_run_checkbox"))

    def test_set_active_run_stops_live_preview_without_auto_resume(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)

        with (
            patch.object(preview._live_timer, "isActive", return_value=True),
            patch.object(preview._live_timer, "stop") as mocked_stop,
            patch.object(preview, "_release_all_pooled_captures") as mocked_release,
            patch.object(preview, "_restart_live_timer") as mocked_restart,
        ):
            preview.set_active_run(True)
            preview.set_active_run(False)

        mocked_stop.assert_called_once()
        mocked_release.assert_called_once()
        mocked_restart.assert_not_called()
        self.assertEqual(preview.live_button.text(), "Start Live")
        self.assertEqual(preview.status_label.text(), "Live preview stopped while a workflow is active.")

    def test_toggle_live_preview_is_blocked_while_run_is_active(self) -> None:
        preview = self._make_preview()
        self.addCleanup(preview.close)
        preview._run_active = True

        with (
            patch.object(preview, "_restart_live_timer") as mocked_restart,
            patch.object(preview, "refresh_camera_previews") as mocked_refresh,
        ):
            preview.toggle_live_preview()

        mocked_restart.assert_not_called()
        mocked_refresh.assert_not_called()
        self.assertEqual(preview.live_button.text(), "Start Live")
        self.assertEqual(preview.status_label.text(), "Live preview unavailable while a workflow is active.")


if __name__ == "__main__":
    unittest.main()
