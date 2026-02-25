from __future__ import annotations

import unittest

from robot_pipeline_app.gui_training_tab import (
    _build_train_base_command,
    _expected_pretrained_model_path,
    _interactive_ssh_command,
    _normalize_train_command_default,
    _remote_exec_command,
    _wrap_train_with_srun,
    _wrap_train_with_tmux,
)
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
            train_command="python3 -m lerobot.scripts.lerobot_train --help",
        )
        self.assertIn("bash -lc ", cmd)
        self.assertIn("source ~/lerobot/lerobot_env/bin/activate", cmd)
        self.assertIn("cd ~/lerobot", cmd)
        self.assertIn("python3 -m lerobot.scripts.lerobot_train --help", cmd)

    def test_remote_exec_command_allows_command_only(self) -> None:
        cmd = _remote_exec_command(
            project_root="",
            env_activate_cmd="",
            train_command="python3 -m lerobot.scripts.lerobot_train --help",
        )
        self.assertIn("bash -lc ", cmd)
        self.assertIn("python3 -m lerobot.scripts.lerobot_train --help", cmd)

    def test_interactive_ssh_command_is_password_prompt_friendly(self) -> None:
        cmd = _interactive_ssh_command(_profile(), "bash -lc 'echo hi'")
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=no", cmd)
        self.assertIn("alice@olympus.ece.tamu.edu", cmd)
        self.assertEqual(cmd[-1], "bash -lc 'echo hi'")

    def test_build_train_base_command_generates_lerobot_train_flags(self) -> None:
        cmd, error = _build_train_base_command(
            policy_type="act",
            dataset_repo_id="lerobot/smol_block_je",
            output_dir="outputs/train/smol_block_je",
            job_name="smol_block_je",
            device="cuda",
            batch_size=8,
            steps=100000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="--seed 7",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertIn("python3 -m lerobot.scripts.lerobot_train", cmd)
        self.assertIn("--dataset.repo_id=lerobot/smol_block_je", cmd)
        self.assertIn("--wandb.enable=true", cmd)
        self.assertIn("--policy.push_to_hub=false", cmd)
        self.assertIn("--seed 7", cmd)

    def test_wrap_train_with_srun_and_tmux(self) -> None:
        base = "python3 -m lerobot.scripts.lerobot_train --help"
        srun_cmd = _wrap_train_with_srun(base, "-p gpu-research --pty")
        self.assertTrue(srun_cmd.startswith("srun -p gpu-research --pty "))
        self.assertIn(base, srun_cmd)

        tmux_cmd = _wrap_train_with_tmux(srun_cmd, "train")
        self.assertIn("tmux has-session -t train", tmux_cmd)
        self.assertIn("tmux new-session -d -s train", tmux_cmd)
        self.assertIn("tmux send-keys -t train", tmux_cmd)

    def test_expected_pretrained_model_path_joins_project_and_output(self) -> None:
        path = _expected_pretrained_model_path(
            "/mnt/shared-scratch/Shakkottai_S/lukeduncan/lerobot/src",
            "outputs/train/smol_block_je",
        )
        self.assertEqual(
            path,
            "/mnt/shared-scratch/Shakkottai_S/lukeduncan/lerobot/src/outputs/train/smol_block_je/checkpoints/last/pretrained_model",
        )

    def test_expected_pretrained_model_path_keeps_absolute_output(self) -> None:
        path = _expected_pretrained_model_path(
            "/mnt/shared-scratch/Shakkottai_S/lukeduncan/lerobot/src",
            "/tmp/custom_output",
        )
        self.assertEqual(path, "/tmp/custom_output/checkpoints/last/pretrained_model")

    def test_normalize_train_command_default_migrates_legacy_help(self) -> None:
        migrated = _normalize_train_command_default("python -m lerobot.scripts.train --help")
        self.assertEqual(migrated, "python3 -m lerobot.scripts.lerobot_train --help")


if __name__ == "__main__":
    unittest.main()
