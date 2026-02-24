from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from robot_pipeline_app.training_auth import (
    delete_ssh_password,
    load_ssh_password,
    save_ssh_password,
)
from robot_pipeline_app.types import TrainingProfile


def _profile() -> TrainingProfile:
    return TrainingProfile(
        id="demo",
        name="Demo",
        host="example.org",
        port=22,
        username="alice",
        auth_mode="password",
        identity_file="",
        remote_models_root="~/models",
        remote_project_root="~/project",
        env_activate_cmd="source env/bin/activate",
        default_tmux_session="train",
        default_srun_prefix="srun --pty bash -lc",
    )


class TrainingAuthTest(unittest.TestCase):
    def test_save_password_requires_secret_tool(self) -> None:
        with patch("robot_pipeline_app.training_auth.secret_tool_path", return_value=None):
            ok, detail = save_ssh_password(_profile(), "pw")
        self.assertFalse(ok)
        self.assertIn("secret-tool", detail)

    def test_save_password_success(self) -> None:
        result = subprocess.CompletedProcess(args=["secret-tool"], returncode=0, stdout="", stderr="")
        with patch("robot_pipeline_app.training_auth.secret_tool_path", return_value="/usr/bin/secret-tool"), patch(
            "robot_pipeline_app.training_auth.subprocess.run",
            return_value=result,
        ) as mocked:
            ok, detail = save_ssh_password(_profile(), "pw123")
        self.assertTrue(ok)
        self.assertIn("stored", detail.lower())
        self.assertTrue(mocked.called)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs.get("input"), "pw123\n")

    def test_load_password_success(self) -> None:
        result = subprocess.CompletedProcess(args=["secret-tool"], returncode=0, stdout="secret\n", stderr="")
        with patch("robot_pipeline_app.training_auth.secret_tool_path", return_value="/usr/bin/secret-tool"), patch(
            "robot_pipeline_app.training_auth.subprocess.run",
            return_value=result,
        ):
            password, error = load_ssh_password(_profile())
        self.assertEqual(password, "secret")
        self.assertIsNone(error)

    def test_load_password_missing_item_returns_none(self) -> None:
        result = subprocess.CompletedProcess(args=["secret-tool"], returncode=1, stdout="", stderr="No such secret item")
        with patch("robot_pipeline_app.training_auth.secret_tool_path", return_value="/usr/bin/secret-tool"), patch(
            "robot_pipeline_app.training_auth.subprocess.run",
            return_value=result,
        ):
            password, error = load_ssh_password(_profile())
        self.assertIsNone(password)
        self.assertIsNone(error)

    def test_delete_password_failure(self) -> None:
        result = subprocess.CompletedProcess(args=["secret-tool"], returncode=2, stdout="", stderr="failed")
        with patch("robot_pipeline_app.training_auth.secret_tool_path", return_value="/usr/bin/secret-tool"), patch(
            "robot_pipeline_app.training_auth.subprocess.run",
            return_value=result,
        ):
            ok, detail = delete_ssh_password(_profile())
        self.assertFalse(ok)
        self.assertIn("failed", detail.lower())


if __name__ == "__main__":
    unittest.main()
