from __future__ import annotations

import unittest

from robot_pipeline_app.probes import parse_frame_dimensions


class ProbesTest(unittest.TestCase):
    def test_parse_frame_dimensions(self) -> None:
        self.assertEqual(parse_frame_dimensions("frame=640x480"), (640, 480))
        self.assertIsNone(parse_frame_dimensions("camera not opened"))


if __name__ == "__main__":
    unittest.main()
