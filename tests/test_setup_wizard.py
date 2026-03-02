from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.setup_wizard import (
    build_setup_status_summary,
    build_setup_wizard_commands,
    build_setup_wizard_guide,
    probe_setup_wizard_status,
)


class SetupWizardTest(unittest.TestCase):
    def test_probe_setup_status_ready_when_env_and_module_are_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot"
            (lerobot_dir / "lerobot_env").mkdir(parents=True, exist_ok=True)
            config = {"lerobot_dir": str(lerobot_dir)}

            with patch("robot_pipeline_app.setup_wizard.in_virtual_env", return_value=True):
                status = probe_setup_wizard_status(
                    config,
                    module_probe_fn=lambda _: (True, "ok"),
                    update_probe_fn=lambda _app_dir: ("up_to_date", "up to date with origin/main"),
                )

        self.assertTrue(status.ready)
        self.assertFalse(status.needs_bootstrap)
        self.assertTrue(status.lerobot_dir_exists)
        self.assertTrue(status.venv_dir_exists)

    def test_probe_setup_status_detects_bootstrap_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot_missing"
            config = {"lerobot_dir": str(lerobot_dir)}

            with patch("robot_pipeline_app.setup_wizard.in_virtual_env", return_value=False):
                status = probe_setup_wizard_status(
                    config,
                    module_probe_fn=lambda _: (False, "No module named lerobot"),
                    update_probe_fn=lambda _app_dir: ("unknown", "remote unreachable"),
                )

        self.assertFalse(status.ready)
        self.assertTrue(status.needs_bootstrap)
        self.assertFalse(status.lerobot_dir_exists)
        self.assertFalse(status.venv_dir_exists)

    def test_build_setup_wizard_commands_uses_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot"
            config = {"lerobot_dir": str(lerobot_dir)}
            with patch("robot_pipeline_app.setup_wizard.in_virtual_env", return_value=False):
                status = probe_setup_wizard_status(
                    config,
                    module_probe_fn=lambda _: (False, "missing"),
                    update_probe_fn=lambda _app_dir: ("up_to_date", "up to date with origin/main"),
                )

        commands = build_setup_wizard_commands(status)
        self.assertIn("git clone https://github.com/huggingface/lerobot", commands)
        self.assertIn(f"python3 -m venv {status.venv_dir}", commands)
        self.assertIn(f"source {status.venv_dir / 'bin' / 'activate'}", commands)
        self.assertIn("python3 -m pip install -e .", commands)

    def test_setup_wizard_guide_calls_out_bootstrap_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot_missing"
            config = {"lerobot_dir": str(lerobot_dir)}
            with patch("robot_pipeline_app.setup_wizard.in_virtual_env", return_value=False):
                status = probe_setup_wizard_status(
                    config,
                    module_probe_fn=lambda _: (False, "missing"),
                    update_probe_fn=lambda _app_dir: ("update_available", "3 commit(s) behind origin/main (ahead 0)"),
                )

        guide = build_setup_wizard_guide(status)
        summary = build_setup_status_summary(status)

        self.assertIn("Detected first-time bootstrap state", guide)
        self.assertIn("Run the setup commands below in your terminal", guide)
        self.assertIn("[ACTION] Neither virtual env nor lerobot import is working.", summary)
        self.assertIn("[ACTION] Update available. Would you like to update and restart now?", summary)

    def test_not_ready_when_env_inactive_even_if_lerobot_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot"
            (lerobot_dir / "lerobot_env").mkdir(parents=True, exist_ok=True)
            config = {"lerobot_dir": str(lerobot_dir)}
            with patch("robot_pipeline_app.setup_wizard.in_virtual_env", return_value=False):
                status = probe_setup_wizard_status(
                    config,
                    module_probe_fn=lambda _: (True, "ok"),
                    update_probe_fn=lambda _app_dir: ("up_to_date", "up to date with origin/main"),
                )

        self.assertFalse(status.ready)
        summary = build_setup_status_summary(status)
        guide = build_setup_wizard_guide(status)
        self.assertIn("no active virtual/conda environment", summary.lower())
        self.assertIn("not inside an active environment", guide.lower())


if __name__ == "__main__":
    unittest.main()
