from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.deploy_diagnostics import (
    explain_deploy_failure,
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

    def test_explain_deploy_failure_maps_common_errors(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            "ModuleNotFoundError: No module named 'lerobot'",
            "unrecognized arguments: --policy.path=/tmp/model",
        ]
        hints = explain_deploy_failure(lines, Path("/tmp/model"))
        self.assertTrue(any("environment error" in hint.lower() for hint in hints))
        self.assertTrue(any("argument error" in hint.lower() for hint in hints))


if __name__ == "__main__":
    unittest.main()
