from __future__ import annotations

import unittest

from robot_pipeline_app.gui_deploy_tab import _camera_resolution_fixes_from_checks, _first_model_payload_candidate


class GuiDeployTabHelpersTest(unittest.TestCase):
    def test_camera_resolution_fixes_from_checks_parses_detected_sizes(self) -> None:
        checks = [
            ("WARN", "Laptop camera resolution", "configured=640x360; detected=640x480; quick fix: set ..."),
            ("PASS", "Phone camera resolution", "configured=640x360; detected=640x360"),
            ("WARN", "Phone camera resolution", "configured=640x360; detected=640x400; quick fix: set ..."),
        ]
        fixes = _camera_resolution_fixes_from_checks(checks)
        self.assertEqual(fixes["laptop"], (640, 480))
        self.assertEqual(fixes["phone"], (640, 400))

    def test_first_model_payload_candidate_returns_first(self) -> None:
        checks = [
            ("PASS", "Model payload", "ok"),
            ("WARN", "Model payload candidates", "/tmp/model_a, /tmp/model_b"),
        ]
        self.assertEqual(_first_model_payload_candidate(checks), "/tmp/model_a")


if __name__ == "__main__":
    unittest.main()
