from __future__ import annotations

import unittest

from robot_pipeline_app.probes import camera_fingerprint, parse_frame_dimensions, serial_port_fingerprint


class ProbesTest(unittest.TestCase):
    def test_parse_frame_dimensions(self) -> None:
        self.assertEqual(parse_frame_dimensions("frame=640x480"), (640, 480))
        self.assertIsNone(parse_frame_dimensions("camera not opened"))

    def test_camera_and_serial_fingerprint_return_none_for_missing_devices(self) -> None:
        self.assertIsNone(camera_fingerprint(9999))
        self.assertIsNone(serial_port_fingerprint("/tmp/definitely_missing_tty_device"))


if __name__ == "__main__":
    unittest.main()
