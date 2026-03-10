from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from robot_pipeline_app.probes import (
    camera_fingerprint,
    in_virtual_env,
    parse_frame_dimensions,
    probe_camera_capture,
    probe_module_import,
    serial_port_fingerprint,
    summarize_probe_error,
)


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

    def test_probe_module_import_returns_timeout_message(self) -> None:
        with patch(
            "robot_pipeline_app.probes.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=10.0),
        ):
            ok, detail = probe_module_import("cv2", timeout_s=10.0)

        self.assertFalse(ok)
        self.assertEqual(detail, "module import timed out after 10.0s")

    def test_probe_camera_capture_passes_backend_name_and_timeout(self) -> None:
        completed = subprocess.CompletedProcess(args=["python"], returncode=0, stdout="frame=640x480\n", stderr="")
        with patch("robot_pipeline_app.probes.subprocess.run", return_value=completed) as mocked_run:
            ok, detail = probe_camera_capture(0, 640, 480, timeout_s=2.5, backend_name="CAP_AVFOUNDATION")

        self.assertTrue(ok)
        self.assertEqual(detail, "frame=640x480")
        self.assertEqual(mocked_run.call_args.args[0][-1], "CAP_AVFOUNDATION")
        self.assertEqual(mocked_run.call_args.kwargs["timeout"], 2.5)
        self.assertIs(mocked_run.call_args.kwargs["stderr"], subprocess.PIPE)

    def test_probe_camera_capture_returns_timeout_message(self) -> None:
        with patch(
            "robot_pipeline_app.probes.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=2.5),
        ):
            ok, detail = probe_camera_capture(0, 640, 480, timeout_s=2.5)

        self.assertFalse(ok)
        self.assertEqual(detail, "camera probe timed out after 2.5s")

    def test_summarize_probe_error_prefers_permission_denied_line(self) -> None:
        detail = summarize_probe_error(
            "OpenCV: camera access has been denied.\nOpenCV: camera failed to properly initialize!"
        )

        self.assertEqual(detail, "OpenCV: camera access has been denied.")


if __name__ == "__main__":
    unittest.main()
