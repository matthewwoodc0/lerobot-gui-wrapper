from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.checks import run_preflight_for_training
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


class ChecksTrainingTest(unittest.TestCase):
    def test_training_preflight_password_mode_checks_password_and_binaries(self) -> None:
        with patch("robot_pipeline_app.training_auth.is_secret_tool_available", return_value=True):
            checks = run_preflight_for_training(
                profile=_profile("password"),
                local_destination=Path("/tmp/model"),
                remote_path="~/models/demo",
                use_rsync=True,
                which_fn=lambda name: f"/usr/bin/{name}" if name in {"ssh", "sftp", "rsync", "expect"} else None,
                has_password_fn=lambda _profile: (True, None),
                remote_path_exists_fn=lambda _profile, _path: (True, None),
            )
        self.assertTrue(any(level == "PASS" and name == "Stored SSH password" for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and name == "ssh binary" for level, name, _ in checks))
        self.assertTrue(any(level == "PASS" and name == "Remote path" for level, name, _ in checks))

    def test_training_preflight_fails_when_required_values_missing(self) -> None:
        bad = _profile("password")
        bad = TrainingProfile(
            id=bad.id,
            name=bad.name,
            host="",
            port=bad.port,
            username="",
            auth_mode=bad.auth_mode,
            identity_file=bad.identity_file,
            remote_models_root="",
            remote_project_root="",
            env_activate_cmd=bad.env_activate_cmd,
            default_tmux_session=bad.default_tmux_session,
            default_srun_prefix=bad.default_srun_prefix,
        )
        with patch("robot_pipeline_app.training_auth.is_secret_tool_available", return_value=False):
            checks = run_preflight_for_training(
                profile=bad,
                which_fn=lambda _name: None,
                has_password_fn=lambda _profile: (False, "missing"),
                remote_path_exists_fn=lambda _profile, _path: (False, "not found"),
            )
        self.assertTrue(any(level == "FAIL" and name == "Training profile" for level, name, _ in checks))
        self.assertTrue(any(level == "FAIL" and name == "ssh binary" for level, name, _ in checks))

    def test_training_preflight_validates_template_rendering(self) -> None:
        checks = run_preflight_for_training(
            profile=_profile("ssh_key"),
            template_text="{a} {b}",
            template_variables={"a": "x"},
            rendered_remote_command="",
            which_fn=lambda name: f"/usr/bin/{name}" if name in {"ssh", "sftp", "expect"} else None,
        )
        self.assertTrue(any(level == "FAIL" and name == "Launch template" for level, name, _ in checks))
        self.assertTrue(any(level == "FAIL" and name == "Rendered remote command" for level, name, _ in checks))

    def test_training_preflight_checks_local_destination_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "dest" / "model"
            checks = run_preflight_for_training(
                profile=_profile("ssh_key"),
                local_destination=destination,
                which_fn=lambda name: f"/usr/bin/{name}" if name in {"ssh", "sftp"} else None,
            )
        self.assertTrue(any(name == "Local destination" for _, name, _ in checks))


if __name__ == "__main__":
    unittest.main()
