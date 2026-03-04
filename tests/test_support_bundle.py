from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.artifacts import write_run_artifacts
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.support_bundle import create_support_bundle


class SupportBundleTest(unittest.TestCase):
    def _base_config(self, tmpdir: str) -> dict[str, object]:
        config = dict(DEFAULT_CONFIG_VALUES)
        root = Path(tmpdir)
        config["runs_dir"] = str(root / "runs")
        config["lerobot_dir"] = str(root / "lerobot")
        Path(str(config["runs_dir"])).mkdir(parents=True, exist_ok=True)
        Path(str(config["lerobot_dir"])).mkdir(parents=True, exist_ok=True)
        return config

    def test_create_support_bundle_latest_includes_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            started = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
            ended = datetime(2026, 3, 1, 10, 2, 0, tzinfo=timezone.utc)
            write_run_artifacts(
                config=config,  # type: ignore[arg-type]
                mode="deploy",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path(tmpdir),
                started_at=started,
                ended_at=ended,
                exit_code=1,
                canceled=False,
                preflight_checks=[
                    ("FAIL", "Eval dataset naming", "Suggested quick fix: alice/eval_demo_1"),
                ],
                output_lines=["runtime failed"],
                dataset_repo_id="alice/eval_demo_1",
            )

            bundle_path = Path(tmpdir) / "bundle.zip"
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="latest",
                output_path=bundle_path,
            )
            self.assertTrue(result.ok, msg=result.message)
            self.assertTrue(bundle_path.exists())

            with zipfile.ZipFile(bundle_path, "r") as archive:
                names = set(archive.namelist())
                self.assertIn("metadata.json", names)
                self.assertIn("command.log", names)
                self.assertIn("preflight_report.json", names)
                self.assertIn("preflight_report.txt", names)
                self.assertIn("config_snapshot.json", names)
                self.assertIn("compatibility_snapshot.json", names)
                self.assertIn("environment_probe.json", names)

                preflight = json.loads(archive.read("preflight_report.json").decode("utf-8"))
                self.assertEqual(preflight["diagnostic_version"], "v2")
                self.assertTrue(preflight["events"])
                self.assertTrue(preflight["events"][0]["code"])

    def test_bundle_redacts_home_path_and_hf_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            home = str(Path.home())
            write_run_artifacts(
                config=config,  # type: ignore[arg-type]
                mode="record",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path(tmpdir),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                exit_code=1,
                canceled=False,
                preflight_checks=[],
                output_lines=[f"path={home}/secret", "token=hf_abcdefghijklmnopqrstuvwxyz123456"],
            )

            bundle_path = Path(tmpdir) / "bundle.zip"
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="latest",
                output_path=bundle_path,
            )
            self.assertTrue(result.ok, msg=result.message)

            with zipfile.ZipFile(bundle_path, "r") as archive:
                command_log = archive.read("command.log").decode("utf-8")
                self.assertIn("path=~/secret", command_log)
                self.assertNotIn(home, command_log)
                self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz123456", command_log)
                self.assertIn("hf_***REDACTED***", command_log)

    def test_bundle_redacts_sensitive_env_values_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            write_run_artifacts(
                config=config,  # type: ignore[arg-type]
                mode="record",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path(tmpdir),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                exit_code=0,
                canceled=False,
                preflight_checks=[],
                output_lines=["ok"],
            )

            with patch.dict("os.environ", {"HF_TOKEN": "hf_SECRET_TOKEN_1234567890"}, clear=False):
                bundle_path = Path(tmpdir) / "bundle.zip"
                result = create_support_bundle(
                    config=config,  # type: ignore[arg-type]
                    run_id="latest",
                    output_path=bundle_path,
                )
            self.assertTrue(result.ok, msg=result.message)

            with zipfile.ZipFile(bundle_path, "r") as archive:
                env_probe = json.loads(archive.read("environment_probe.json").decode("utf-8"))
                self.assertEqual(env_probe["env"]["HF_TOKEN"], "***REDACTED***")

    def test_bundle_redacts_generic_secret_assignment_and_bearer_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            write_run_artifacts(
                config=config,  # type: ignore[arg-type]
                mode="deploy",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path(tmpdir),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                exit_code=1,
                canceled=False,
                preflight_checks=[],
                output_lines=[
                    "MY_API_KEY=abc123",
                    "Authorization: Bearer tokenVALUE0123456789",
                ],
            )

            bundle_path = Path(tmpdir) / "bundle.zip"
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="latest",
                output_path=bundle_path,
            )
            self.assertTrue(result.ok, msg=result.message)

            with zipfile.ZipFile(bundle_path, "r") as archive:
                command_log = archive.read("command.log").decode("utf-8")
                self.assertIn("MY_API_KEY=***REDACTED***", command_log)
                self.assertNotIn("MY_API_KEY=abc123", command_log)
                self.assertIn("Authorization: Bearer ***REDACTED***", command_log)
                self.assertNotIn("Bearer tokenVALUE0123456789", command_log)

    def test_bundle_fails_when_no_runs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="latest",
                output_path=Path(tmpdir) / "bundle.zip",
            )
            self.assertFalse(result.ok)

    def test_bundle_fails_when_output_parent_cannot_be_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            write_run_artifacts(
                config=config,  # type: ignore[arg-type]
                mode="record",
                command=["python3", "-m", "lerobot.scripts.lerobot_record"],
                cwd=Path(tmpdir),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                exit_code=0,
                canceled=False,
                preflight_checks=[],
                output_lines=["ok"],
            )
            blocker = Path(tmpdir) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="latest",
                output_path=blocker / "bundle.zip",
            )
            self.assertFalse(result.ok)

    def test_bundle_rejects_path_traversal_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            result = create_support_bundle(
                config=config,  # type: ignore[arg-type]
                run_id="../../etc",
                output_path=Path(tmpdir) / "bundle.zip",
            )
            self.assertFalse(result.ok)
            self.assertIn("path traversal", result.message.lower())


if __name__ == "__main__":
    unittest.main()
