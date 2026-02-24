from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.deploy_diagnostics import (
    explain_deploy_failure,
    explain_runtime_slowdown,
    find_nested_model_candidates,
    is_runnable_model_path,
    validate_model_path,
)


class DeployDiagnosticsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
