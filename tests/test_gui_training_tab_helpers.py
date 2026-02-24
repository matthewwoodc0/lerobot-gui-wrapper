from __future__ import annotations

import unittest

from robot_pipeline_app.gui_training_tab import _interactive_ssh_command, _remote_exec_command
from robot_pipeline_app.types import TrainingProfile


def _profile() -> TrainingProfile:
    return TrainingProfile(
        id="olympus",
        name="Olympus",
        host="olympus.ece.tamu.edu",
        port=22,
        username="alice",
        auth_mode="password",
        identity_file="",
        remote_models_root="~/lerobot/trained_models",
        remote_project_root="~/lerobot",
        env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
        default_tmux_session="",
        default_srun_prefix="",
    )


class GuiTrainingTabHelpersTest(unittest.TestCase):
    def test_remote_exec_command_includes_activate_and_cd(self) -> None:
        cmd = _remote_exec_command(
            project_root="~/lerobot",
            env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
            train_command="python -m lerobot.scripts.train --help",
        )
        self.assertIn("bash -lc ", cmd)
        self.assertIn("source ~/lerobot/lerobot_env/bin/activate", cmd)
        self.assertIn("cd ~/lerobot", cmd)
        self.assertIn("python -m lerobot.scripts.train --help", cmd)

    def test_remote_exec_command_allows_command_only(self) -> None:
        cmd = _remote_exec_command(
            project_root="",
            env_activate_cmd="",
            train_command="python -m lerobot.scripts.train --help",
        )
        self.assertIn("bash -lc ", cmd)
        self.assertIn("python -m lerobot.scripts.train --help", cmd)

    def test_interactive_ssh_command_is_password_prompt_friendly(self) -> None:
        cmd = _interactive_ssh_command(_profile(), "bash -lc 'echo hi'")
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=no", cmd)
        self.assertIn("alice@olympus.ece.tamu.edu", cmd)
        self.assertEqual(cmd[-1], "bash -lc 'echo hi'")


if __name__ == "__main__":
    unittest.main()
