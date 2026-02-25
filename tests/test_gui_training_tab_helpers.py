from __future__ import annotations

import unittest

from robot_pipeline_app.gui_training_tab import (
    _build_generated_train_command,
    _build_train_base_command,
    _default_dataset_repo_id,
    _default_output_name,
    _expected_pretrained_model_path,
    _wrap_train_with_srun,
    _wrap_train_with_tmux,
)


class GuiTrainingTabHelpersTest(unittest.TestCase):
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

    def test_build_train_base_command_rejects_invalid_extra_args(self) -> None:
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
            extra_args="--seed '7",
        )
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("Invalid extra args", error)

    def test_build_generated_train_command_without_wrappers(self) -> None:
        cmd, error = _build_generated_train_command(
            policy_type="act",
            dataset_repo_id="lerobot/smol_block_je",
            output_dir="outputs/train/smol_block_je",
            job_name="smol_block_je",
            device="cuda",
            batch_size=8,
            steps=100000,
            wandb_enable=False,
            push_to_hub=False,
            extra_args="",
            use_srun=False,
            srun_prefix="",
            use_tmux=False,
            tmux_session="",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertTrue(cmd.startswith("python3 -m lerobot.scripts.lerobot_train "))
        self.assertNotIn("srun ", cmd)
        self.assertNotIn("tmux ", cmd)

    def test_build_generated_train_command_wraps_with_srun_and_tmux(self) -> None:
        cmd, error = _build_generated_train_command(
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
            use_srun=True,
            srun_prefix="-p gpu-research --pty",
            use_tmux=True,
            tmux_session="train",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertIn("tmux has-session -t train", cmd)
        self.assertIn("tmux send-keys -t train", cmd)
        self.assertIn("srun -p gpu-research --pty", cmd)
        self.assertIn("lerobot.scripts.lerobot_train", cmd)

    def test_wrap_train_with_srun_and_tmux(self) -> None:
        base = "python3 -m lerobot.scripts.lerobot_train --help"
        srun_cmd = _wrap_train_with_srun(base, "-p gpu-research --pty")
        self.assertTrue(srun_cmd.startswith("srun -p gpu-research --pty "))
        self.assertIn(base, srun_cmd)

        prefixed = _wrap_train_with_srun(base, "srun -p gpu-research --pty")
        self.assertEqual(prefixed, srun_cmd)

        tmux_cmd = _wrap_train_with_tmux(srun_cmd, "train")
        self.assertIn("tmux has-session -t train", tmux_cmd)
        self.assertIn("tmux new-session -d -s train", tmux_cmd)
        self.assertIn("tmux send-keys -t train", tmux_cmd)

    def test_wrap_train_with_tmux_without_session_returns_base(self) -> None:
        base = "python3 -m lerobot.scripts.lerobot_train --help"
        self.assertEqual(_wrap_train_with_tmux(base, ""), base)

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

    def test_default_dataset_repo_id_prefers_lerobot_namespace(self) -> None:
        repo_id = _default_dataset_repo_id({"hf_username": "alice", "last_dataset_name": "my_data"})
        self.assertEqual(repo_id, "lerobot/my_data")

    def test_default_dataset_repo_id_uses_configured_training_repo_id(self) -> None:
        repo_id = _default_dataset_repo_id(
            {"training_gen_dataset_repo_id": "org/custom_data", "last_dataset_name": "my_data"}
        )
        self.assertEqual(repo_id, "org/custom_data")

    def test_default_output_name_matches_repo_tail(self) -> None:
        name = _default_output_name({"training_gen_dataset_repo_id": "lerobot/smol_block_je"})
        self.assertEqual(name, "smol_block_je")


if __name__ == "__main__":
    unittest.main()
