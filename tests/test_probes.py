from __future__ import annotations

import unittest
from unittest.mock import patch

from robot_pipeline_app.probes import camera_fingerprint, in_virtual_env, parse_frame_dimensions, serial_port_fingerprint


class ProbesTest(unittest.TestCase):
    def test_parse_frame_dimensions(self) -> None:
        self.assertEqual(parse_frame_dimensions("frame=640x480"), (640, 480))
        self.assertIsNone(parse_frame_dimensions("camera not opened"))

    def test_camera_and_serial_fingerprint_return_none_for_missing_devices(self) -> None:
        self.assertIsNone(camera_fingerprint(9999))
        self.assertIsNone(serial_port_fingerprint("/tmp/definitely_missing_tty_device"))

    def test_in_virtual_env_true_when_prefix_differs(self) -> None:
        with patch("robot_pipeline_app.probes.os.environ", {}), patch(
            "robot_pipeline_app.probes.sys.prefix", "/tmp/env_prefix"
        ), patch("robot_pipeline_app.probes.sys.base_prefix", "/usr/bin/python"):
            self.assertTrue(in_virtual_env())

    def test_in_virtual_env_true_for_conda_meta_prefix(self) -> None:
        with patch("robot_pipeline_app.probes.os.environ", {}), patch(
            "robot_pipeline_app.probes.sys.prefix", "/tmp/conda_env"
        ), patch("robot_pipeline_app.probes.sys.base_prefix", "/tmp/conda_env"), patch(
            "robot_pipeline_app.probes.Path.is_dir",
            return_value=True,
        ):
            self.assertTrue(in_virtual_env())


if __name__ == "__main__":
    unittest.main()
