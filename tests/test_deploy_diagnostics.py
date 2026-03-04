from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.deploy_diagnostics import (
    _file_markers,
    diagnose_deploy_failure_events,
    diagnose_runtime_failure_events,
    explain_deploy_failure,
    explain_runtime_failure,
    explain_runtime_slowdown,
    find_nested_model_candidates,
    is_runnable_model_path,
    summarize_camera_command_load,
    validate_model_path,
)


class DeployDiagnosticsTest(unittest.TestCase):
    def test_file_markers_skips_unreadable_entries(self) -> None:
        class _BadEntry:
            name = "debug-core.log"

            def is_file(self) -> bool:
                raise PermissionError("permission denied")

        class _GoodEntry:
            name = "model.safetensors"

            def is_file(self) -> bool:
                return True

        path = Path("/tmp/does-not-matter")
        with patch.object(Path, "iterdir", return_value=[_BadEntry(), _GoodEntry()]):  # type: ignore[list-item]
            has_weights, has_config = _file_markers(path)
        self.assertTrue(has_weights)
        self.assertFalse(has_config)

    def test_is_runnable_model_path_true_when_config_and_weights_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_a"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("x\n", encoding="utf-8")
            self.assertTrue(is_runnable_model_path(model_dir))

    def test_validate_model_path_suggests_nested_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run_1"
            payload = root / "checkpoints" / "last" / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "policy_config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("x\n", encoding="utf-8")

            ok, detail, candidates = validate_model_path(root)
            self.assertFalse(ok)
            self.assertIn("Choose a nested model payload folder", detail)
            self.assertIn(payload, candidates)

    def test_find_nested_model_candidates_prefers_latest_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run_2"
            older = root / "checkpoints" / "checkpoint-010000" / "pretrained_model"
            newer = root / "checkpoints" / "checkpoint-020000" / "pretrained_model"
            older.mkdir(parents=True, exist_ok=True)
            newer.mkdir(parents=True, exist_ok=True)
            (older / "config.json").write_text("{}\n", encoding="utf-8")
            (older / "model.safetensors").write_text("x\n", encoding="utf-8")
            (newer / "config.json").write_text("{}\n", encoding="utf-8")
            (newer / "model.safetensors").write_text("x\n", encoding="utf-8")

            candidates = find_nested_model_candidates(root)

            self.assertGreaterEqual(len(candidates), 2)
            self.assertEqual(candidates[0], newer)
            self.assertIn(older, candidates)

    def test_explain_deploy_failure_maps_common_errors(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            "ModuleNotFoundError: No module named 'lerobot'",
            "unrecognized arguments: --policy.path=/tmp/model",
        ]
        hints = explain_deploy_failure(lines, Path("/tmp/model"))
        self.assertTrue(any("environment error" in hint.lower() for hint in hints))
        self.assertTrue(any("argument error" in hint.lower() for hint in hints))

    def test_explain_deploy_failure_camera_resolution_hint(self) -> None:
        lines = [
            "RuntimeError: OpenCVCamera(0) failed to set capture_height=360 (actual_height=480, height_success=True).",
        ]
        hints = explain_deploy_failure(lines, Path("/tmp/model"))
        self.assertTrue(any("resolution negotiation failed" in hint.lower() for hint in hints))

    def test_explain_deploy_failure_motor_recovery_hints(self) -> None:
        lines = [
            "ERROR: servo joint timed out and motor not responding",
        ]
        hints = explain_deploy_failure(lines, Path("/tmp/model"))
        self.assertTrue(any("motor/servo communication issue" in hint.lower() for hint in hints))
        self.assertTrue(any("power-cycle" in hint.lower() for hint in hints))
        self.assertTrue(any("fuser -k" in hint.lower() for hint in hints))

    def test_explain_runtime_slowdown_reports_loop_gap(self) -> None:
        lines = [
            "WARNING ... Record loop is running slower (2.7 Hz) than the target FPS (30 Hz).",
            "WARNING ... Record loop is running slower (5.6 Hz) than the target FPS (30 Hz).",
        ]
        hints = explain_runtime_slowdown(lines)
        self.assertTrue(any("2.7-5.6 hz" in hint.lower() for hint in hints))
        self.assertTrue(any("reduce camera_fps" in hint.lower() for hint in hints))

    def test_summarize_camera_command_load_reports_aggregate_pixels(self) -> None:
        cmd = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_record",
            '--robot.cameras={"laptop":{"width":640,"height":480,"fps":30},"phone":{"width":640,"height":360,"fps":30}}',
        ]
        summary = summarize_camera_command_load(cmd)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("laptop=640x480@30", summary)
        self.assertIn("phone=640x360@30", summary)
        self.assertIn("16.1 MPix/s", summary)

    def test_explain_runtime_slowdown_includes_command_load_and_ui_overhead_metrics(self) -> None:
        lines = [
            "WARNING ... Record loop is running slower (8.0 Hz) than the target FPS (30 Hz).",
            "Runtime I/O optimization: suppressed 240 carriage-return progress updates.",
        ]
        cmd = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_record",
            '--robot.cameras={"laptop":{"width":1280,"height":720,"fps":30},"phone":{"width":1280,"height":720,"fps":30}}',
        ]
        hints = explain_runtime_slowdown(lines, cmd)
        self.assertTrue(any("command camera load" in hint.lower() for hint in hints))
        self.assertTrue(any("suppressed 240 carriage-return progress updates" in hint.lower() for hint in hints))
        self.assertTrue(any("capture + video encode/disk i/o" in hint.lower() for hint in hints))

    def test_explain_runtime_failure_no_status_packet_includes_motor_and_calibration_hints(self) -> None:
        lines = [
            "ConnectionError: Failed to write 'Torque_Enable' on id_=6 with '1' after 1 tries. [TxRxResult] There is no status packet!",
        ]
        hints = explain_runtime_failure(lines, run_mode="teleop")
        self.assertTrue(any("motor bus timeout" in hint.lower() for hint in hints))
        self.assertTrue(any("id 6" in hint.lower() for hint in hints))
        self.assertTrue(any("calibration" in hint.lower() for hint in hints))

    def test_explain_runtime_failure_serial_permission_includes_ports(self) -> None:
        lines = [
            "serial.serialutil.SerialException: [Errno 13] Permission denied: '/dev/ttyACM0'",
        ]
        cmd = [
            "python3",
            "-m",
            "lerobot.teleoperate",
            "--robot.port=/dev/ttyACM0",
            "--teleop.port=/dev/ttyACM1",
        ]
        hints = explain_runtime_failure(lines, command=cmd, run_mode="record")
        self.assertTrue(any("/dev/ttyacm0" in hint.lower() for hint in hints))
        self.assertTrue(any("/dev/ttyacm1" in hint.lower() for hint in hints))
        self.assertTrue(any("rerun preflight" in hint.lower() for hint in hints))

    def test_diagnose_deploy_failure_events_include_stable_codes(self) -> None:
        lines = [
            "ModuleNotFoundError: No module named 'lerobot'",
            "unrecognized arguments: --policy.path=/tmp/model",
        ]
        events = diagnose_deploy_failure_events(lines, Path("/tmp/model"))
        self.assertTrue(events)
        self.assertTrue(any(event.code for event in events))
        self.assertTrue(any(event.code.startswith("ENV-") or event.code.startswith("CLI-") for event in events))

    def test_diagnose_runtime_failure_events_include_stable_codes(self) -> None:
        lines = [
            "serial.serialutil.SerialException: [Errno 13] Permission denied: '/dev/ttyACM0'",
        ]
        events = diagnose_runtime_failure_events(lines, run_mode="record")
        self.assertTrue(events)
        self.assertTrue(any(event.code.startswith("SER-") for event in events))


if __name__ == "__main__":
    unittest.main()
