from __future__ import annotations

import unittest

from robot_pipeline_app.gui_camera import DualCameraPreview, _normalize_scan_limit


class _FakeCapture:
    def release(self) -> None:
        return


class GuiCameraHelpersTest(unittest.TestCase):
    def test_normalize_scan_limit_clamps_and_defaults(self) -> None:
        self.assertEqual(_normalize_scan_limit("12"), 12)
        self.assertEqual(_normalize_scan_limit("0"), 1)
        self.assertEqual(_normalize_scan_limit("999"), 64)
        self.assertEqual(_normalize_scan_limit("nope"), 14)

    def test_scan_ports_worker_uses_explicit_limit_input(self) -> None:
        preview = object.__new__(DualCameraPreview)
        captured_limits: list[int] = []

        def candidate_indices(limit: int) -> list[int]:
            captured_limits.append(limit)
            return [0, 1, 2]

        def open_capture(index: int):
            if index in {0, 2}:
                return _FakeCapture()
            return None

        preview._candidate_scan_indices = candidate_indices  # type: ignore[attr-defined]
        preview._open_capture = open_capture  # type: ignore[attr-defined]

        detected, total = DualCameraPreview._scan_ports_worker(preview, 9)
        self.assertEqual(captured_limits, [9])
        self.assertEqual(detected, [0, 2])
        self.assertEqual(total, 3)


if __name__ == "__main__":
    unittest.main()
