from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.training_remote import (
    build_pull_command,
    build_remote_launch_command,
    build_sftp_pull_command,
    command_uses_binary,
    ensure_host_trusted,
    run_pull_with_fallback,
)
from robot_pipeline_app.types import TrainingProfile


def _profile(auth_mode: str = "password") -> TrainingProfile:
    return TrainingProfile(
        id="demo",
        name="Demo",
        host="example.org",
        port=22,
        username="alice",
        auth_mode=auth_mode,
        identity_file="/tmp/id_rsa",
        remote_models_root="~/models",
        remote_project_root="~/project",
        env_activate_cmd="source env/bin/activate",
        default_tmux_session="train",
        default_srun_prefix="srun --pty bash -lc",
    )


class _MessageBox:
    def __init__(self, approved: bool) -> None:
        self.approved = approved

    def askyesno(self, _title: str, _message: str) -> bool:
        return self.approved


class TrainingRemoteTest(unittest.TestCase):
    def test_build_remote_launch_command_password_uses_expect_wrapper(self) -> None:
        cmd = build_remote_launch_command(_profile("password"), "echo ok")
        self.assertGreater(len(cmd), 3)
        self.assertEqual(cmd[0], "expect")
        self.assertTrue(command_uses_binary(cmd, "ssh"))
        self.assertNotIn("pw123", " ".join(cmd))

    def test_build_remote_launch_command_ssh_key_uses_ssh_directly(self) -> None:
        cmd = build_remote_launch_command(_profile("ssh_key"), "echo ok")
        self.assertEqual(cmd[0], "ssh")
        self.assertFalse(command_uses_binary(cmd, "expect"))

    def test_build_pull_command_prefers_rsync_when_available(self) -> None:
        with patch("robot_pipeline_app.training_remote.shutil.which", return_value="/usr/bin/rsync"):
            cmd = build_pull_command(_profile("password"), "~/models/a", Path("/tmp/a"), prefer_rsync=True)
        self.assertTrue(command_uses_binary(cmd, "rsync"))
        self.assertIn("$HOME", " ".join(cmd))

    def test_build_sftp_pull_command_returns_batch_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd, batch_file = build_sftp_pull_command(_profile("ssh_key"), "/remote/path", Path(tmpdir) / "dest")
            self.assertTrue(batch_file.exists())
            self.assertTrue(command_uses_binary(cmd, "sftp"))
            batch_file.unlink(missing_ok=True)

    def test_run_pull_with_fallback_uses_sftp_after_rsync_failure(self) -> None:
        with patch("robot_pipeline_app.training_remote.shutil.which", return_value="/usr/bin/rsync"):
            calls: list[str] = []

            def run_fn(cmd: list[str]) -> int:
                if command_uses_binary(cmd, "rsync"):
                    calls.append("rsync")
                    return 23
                if command_uses_binary(cmd, "sftp"):
                    calls.append("sftp")
                    return 0
                return 1

            mode, code = run_pull_with_fallback(
                _profile("ssh_key"),
                "/remote/model",
                Path("/tmp/model"),
                run_fn,
            )
        self.assertEqual(mode, "sftp")
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["rsync", "sftp"])

    def test_ensure_host_trusted_already_known(self) -> None:
        known = subprocess.CompletedProcess(args=["ssh-keygen"], returncode=0, stdout="found\n", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "robot_pipeline_app.training_remote.Path.home",
            return_value=Path(tmpdir),
        ), patch(
            "robot_pipeline_app.training_remote.subprocess.run",
            return_value=known,
        ):
            ok, detail = ensure_host_trusted(_profile("ssh_key"), _MessageBox(approved=True))
        self.assertTrue(ok)
        self.assertIn("already trusted", detail.lower())

    def test_ensure_host_trusted_unknown_host_denied(self) -> None:
        missing = subprocess.CompletedProcess(args=["ssh-keygen"], returncode=1, stdout="", stderr="")
        scan = subprocess.CompletedProcess(args=["ssh-keyscan"], returncode=0, stdout="example.org ssh-ed25519 key", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "robot_pipeline_app.training_remote.Path.home",
            return_value=Path(tmpdir),
        ), patch(
            "robot_pipeline_app.training_remote.subprocess.run",
            side_effect=[missing, scan],
        ):
            ok, detail = ensure_host_trusted(_profile("ssh_key"), _MessageBox(approved=False))
        self.assertFalse(ok)
        self.assertIn("not approved", detail.lower())

    def test_ensure_host_trusted_unknown_host_approved_writes_known_hosts(self) -> None:
        missing = subprocess.CompletedProcess(args=["ssh-keygen"], returncode=1, stdout="", stderr="")
        scan = subprocess.CompletedProcess(args=["ssh-keyscan"], returncode=0, stdout="example.org ssh-ed25519 key", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "robot_pipeline_app.training_remote.Path.home",
            return_value=Path(tmpdir),
        ), patch(
            "robot_pipeline_app.training_remote.subprocess.run",
            side_effect=[missing, scan],
        ):
            ok, detail = ensure_host_trusted(_profile("ssh_key"), _MessageBox(approved=True))
            known_hosts = Path(tmpdir) / ".ssh" / "known_hosts"
            contents = known_hosts.read_text(encoding="utf-8")
        self.assertTrue(ok)
        self.assertIn("saved", detail.lower())
        self.assertIn("example.org", contents)


if __name__ == "__main__":
    unittest.main()
