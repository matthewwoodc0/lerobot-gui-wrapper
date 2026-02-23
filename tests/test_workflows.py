from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.workflows import execute_command_with_artifacts, upload_dataset_with_artifacts


class WorkflowExecutionTest(unittest.TestCase):
    def _config(self, runs_dir: str) -> dict[str, object]:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["runs_dir"] = runs_dir
        config["lerobot_dir"] = "/tmp"
        return config

    def test_execute_command_success_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            fake = subprocess.CompletedProcess(
                args=["echo", "ok"],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )
            with patch("robot_pipeline_app.workflows.run_command", return_value=fake):
                result = execute_command_with_artifacts(
                    config=config,
                    mode="record",
                    cmd=["echo", "ok"],
                    cwd=Path("/tmp"),
                    preflight_checks=[("PASS", "example", "ok")],
                    dataset_repo_id="alice/demo_1",
                )

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.canceled)
            self.assertIsNotNone(result.artifact_path)
            assert result.artifact_path is not None
            metadata = json.loads((result.artifact_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "record")
            self.assertEqual(metadata["dataset_repo_id"], "alice/demo_1")

    def test_execute_command_failure_exit_code_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            fake = subprocess.CompletedProcess(
                args=["echo", "bad"],
                returncode=7,
                stdout="",
                stderr="bad\n",
            )
            with patch("robot_pipeline_app.workflows.run_command", return_value=fake):
                result = execute_command_with_artifacts(
                    config=config,
                    mode="deploy",
                    cmd=["echo", "bad"],
                    cwd=Path("/tmp"),
                    preflight_checks=[],
                    dataset_repo_id="alice/eval_1",
                )

            self.assertEqual(result.exit_code, 7)
            self.assertFalse(result.canceled)

    def test_execute_command_cancel_sets_canceled_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            with patch("robot_pipeline_app.workflows.run_command", side_effect=KeyboardInterrupt):
                result = execute_command_with_artifacts(
                    config=config,
                    mode="record",
                    cmd=["python3", "-c", "print('x')"],
                    cwd=Path("/tmp"),
                    preflight_checks=[],
                    dataset_repo_id="alice/demo_2",
                )

            self.assertIsNone(result.exit_code)
            self.assertTrue(result.canceled)
            self.assertIsNotNone(result.artifact_path)
            assert result.artifact_path is not None
            metadata = json.loads((result.artifact_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["canceled"])
            self.assertIsNone(metadata["exit_code"])

    def test_upload_dataset_uses_upload_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            with patch("robot_pipeline_app.workflows.execute_command_with_artifacts") as mocked:
                upload_dataset_with_artifacts(
                    config=config,
                    dataset_repo_id="alice/demo_3",
                    upload_path=Path("/tmp/demo_3"),
                )
            self.assertTrue(mocked.called)
            kwargs = mocked.call_args.kwargs
            self.assertEqual(kwargs["mode"], "upload")
            self.assertEqual(kwargs["dataset_repo_id"], "alice/demo_3")


if __name__ == "__main__":
    unittest.main()
