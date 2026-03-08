from __future__ import annotations

import unittest

from robot_pipeline_app.gui_training_tab import (
    _build_generated_train_command,
    _build_hil_workflow_text,
    _build_srun_prefix,
    _build_train_base_command,
    _default_dataset_repo_id,
    _default_output_name,
    _expected_pretrained_model_path,
    _with_hil_suffix,
    _wrap_train_with_srun,
)


class GuiTrainingTabHelpersTest(unittest.TestCase):
    def test_build_train_base_command_generates_expected_flags(self) -> None:
        cmd, error = _build_train_base_command(
            python_bin="python",
            train_entrypoint="lerobot.scripts.lerobot_train",
            policy_path="lerobot/smolvla_base",
            policy_input_features="null",
            policy_output_features="null",
            dataset_repo_id="matthewwoodc0/jeffrey_20",
            output_dir="outputs/train/smolvla_b16_jeffrey_20",
            job_name="smolvla_b16_jeffrey_20",
            device="cuda",
            batch_size=16,
            steps=50000,
            save_freq=5000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="--seed 7",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertTrue(cmd.startswith("python -m lerobot.scripts.lerobot_train "))
        self.assertIn("--policy.path=lerobot/smolvla_base", cmd)
        self.assertIn("--policy.input_features=null", cmd)
        self.assertIn("--policy.output_features=null", cmd)
        self.assertIn("--dataset.repo_id=matthewwoodc0/jeffrey_20", cmd)
        self.assertIn("--batch_size=16", cmd)
        self.assertIn("--steps=50000", cmd)
        self.assertIn("--save_freq=5000", cmd)
        self.assertIn("--wandb.enable=true", cmd)
        self.assertIn("--policy.push_to_hub=false", cmd)
        self.assertIn("--seed 7", cmd)

    def test_build_train_base_command_rejects_invalid_extra_args(self) -> None:
        cmd, error = _build_train_base_command(
            python_bin="python",
            train_entrypoint="lerobot.scripts.lerobot_train",
            policy_path="lerobot/smolvla_base",
            policy_input_features="null",
            policy_output_features="null",
            dataset_repo_id="matthewwoodc0/jeffrey_20",
            output_dir="outputs/train/smolvla_b16_jeffrey_20",
            job_name="smolvla_b16_jeffrey_20",
            device="cuda",
            batch_size=16,
            steps=50000,
            save_freq=5000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="--seed '7",
        )
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("Invalid extra args", error)

    def test_build_srun_prefix_generates_expected_shape(self) -> None:
        prefix, error = _build_srun_prefix(
            partition="gpu-research",
            cpus_per_task=8,
            gres="gpu:a100:1",
            srun_job_name="smolvla_b16_jeffrey_20",
            queue="olympus-research-gpu",
            extra_args="",
        )
        self.assertIsNone(error)
        assert prefix is not None
        self.assertEqual(
            prefix,
            "srun -p gpu-research --cpus-per-task=8 --gres=gpu:a100:1 -J smolvla_b16_jeffrey_20 -q olympus-research-gpu --pty",
        )

    def test_build_generated_train_command_without_srun(self) -> None:
        cmd, error = _build_generated_train_command(
            python_bin="python",
            train_entrypoint="lerobot.scripts.lerobot_train",
            policy_path="lerobot/smolvla_base",
            policy_input_features="null",
            policy_output_features="null",
            dataset_repo_id="matthewwoodc0/jeffrey_20",
            output_dir="outputs/train/smolvla_b16_jeffrey_20",
            job_name="smolvla_b16_jeffrey_20",
            device="cuda",
            batch_size=16,
            steps=50000,
            save_freq=5000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="",
            use_srun=False,
            srun_partition="gpu-research",
            srun_cpus_per_task=8,
            srun_gres="gpu:a100:1",
            srun_job_name="smolvla_b16_jeffrey_20",
            srun_queue="olympus-research-gpu",
            srun_extra_args="",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertTrue(cmd.startswith("python -m lerobot.scripts.lerobot_train "))
        self.assertNotIn("srun ", cmd)

    def test_build_generated_train_command_matches_target_pattern(self) -> None:
        cmd, error = _build_generated_train_command(
            python_bin="python",
            train_entrypoint="lerobot.scripts.lerobot_train",
            policy_path="lerobot/smolvla_base",
            policy_input_features="null",
            policy_output_features="null",
            dataset_repo_id="matthewwoodc0/jeffrey_20",
            output_dir="outputs/train/smolvla_b16_jeffrey_20",
            job_name="smolvla_b16_jeffrey_20",
            device="cuda",
            batch_size=16,
            steps=50000,
            save_freq=5000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="",
            use_srun=True,
            srun_partition="gpu-research",
            srun_cpus_per_task=8,
            srun_gres="gpu:a100:1",
            srun_job_name="smolvla_b16_jeffrey_20",
            srun_queue="olympus-research-gpu",
            srun_extra_args="",
        )
        self.assertIsNone(error)
        self.assertEqual(
            cmd,
            "srun -p gpu-research --cpus-per-task=8 --gres=gpu:a100:1 -J smolvla_b16_jeffrey_20 -q olympus-research-gpu --pty python -m lerobot.scripts.lerobot_train --policy.path=lerobot/smolvla_base --policy.input_features=null --policy.output_features=null --dataset.repo_id=matthewwoodc0/jeffrey_20 --batch_size=16 --steps=50000 --output_dir=outputs/train/smolvla_b16_jeffrey_20 --job_name=smolvla_b16_jeffrey_20 --policy.device=cuda --wandb.enable=true --policy.push_to_hub=false --save_freq=5000",
        )

    def test_build_generated_train_command_uses_resolved_train_entrypoint(self) -> None:
        cmd, error = _build_generated_train_command(
            python_bin="python",
            train_entrypoint="lerobot.train",
            policy_path="lerobot/smolvla_base",
            policy_input_features="null",
            policy_output_features="null",
            dataset_repo_id="matthewwoodc0/jeffrey_20",
            output_dir="outputs/train/smolvla_b16_jeffrey_20",
            job_name="smolvla_b16_jeffrey_20",
            device="cuda",
            batch_size=16,
            steps=50000,
            save_freq=5000,
            wandb_enable=True,
            push_to_hub=False,
            extra_args="",
            use_srun=False,
            srun_partition="gpu-research",
            srun_cpus_per_task=8,
            srun_gres="gpu:a100:1",
            srun_job_name="smolvla_b16_jeffrey_20",
            srun_queue="olympus-research-gpu",
            srun_extra_args="",
        )
        self.assertIsNone(error)
        assert cmd is not None
        self.assertTrue(cmd.startswith("python -m lerobot.train "))

    def test_wrap_train_with_srun(self) -> None:
        base = "python -m lerobot.scripts.lerobot_train --help"
        srun_cmd = _wrap_train_with_srun(base, "-p gpu-research --pty")
        self.assertTrue(srun_cmd.startswith("srun -p gpu-research --pty "))
        self.assertIn(base, srun_cmd)

        prefixed = _wrap_train_with_srun(base, "srun -p gpu-research --pty")
        self.assertEqual(prefixed, srun_cmd)

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

    def test_default_dataset_repo_id_uses_configured_training_repo_id(self) -> None:
        repo_id = _default_dataset_repo_id(
            {"training_gen_dataset_repo_id": "org/custom_data", "last_dataset_name": "my_data"}
        )
        self.assertEqual(repo_id, "org/custom_data")

    def test_default_dataset_repo_id_prefers_last_recorded_repo_id(self) -> None:
        repo_id = _default_dataset_repo_id(
            {
                "hf_username": "alice",
                "last_dataset_name": "my_data",
                "last_dataset_repo_id": "alice/actual_recorded_data",
            }
        )
        self.assertEqual(repo_id, "alice/actual_recorded_data")

    def test_default_dataset_repo_id_uses_owner_and_last_dataset_name(self) -> None:
        repo_id = _default_dataset_repo_id({"hf_username": "alice", "last_dataset_name": "my_data"})
        self.assertEqual(repo_id, "alice/my_data")

    def test_default_dataset_repo_id_returns_blank_without_owner_or_last_repo(self) -> None:
        repo_id = _default_dataset_repo_id({"last_dataset_name": "my_data"})
        self.assertEqual(repo_id, "")

    def test_default_output_name_matches_repo_tail(self) -> None:
        name = _default_output_name({"training_gen_dataset_repo_id": "lerobot/smol_block_je"})
        self.assertEqual(name, "smol_block_je")

    def test_default_output_name_falls_back_to_last_dataset_name(self) -> None:
        name = _default_output_name({"last_dataset_name": "my_data"})
        self.assertEqual(name, "my_data")

    def test_with_hil_suffix_adds_suffix_once(self) -> None:
        self.assertEqual(_with_hil_suffix("outputs/train/my_run"), "outputs/train/my_run_hil")
        self.assertEqual(_with_hil_suffix("my_run_hil"), "my_run_hil")

    def test_build_hil_workflow_text_includes_incremental_loop(self) -> None:
        text = _build_hil_workflow_text(
            project_root="~/lerobot/src",
            env_activate_cmd="source env/bin/activate",
            intervention_repo_id="lerobot/interventions",
            base_model_path="outputs/train/base/checkpoints/last/pretrained_model",
            command="python -m lerobot.scripts.lerobot_train --steps=3000",
            expected_model_path="outputs/train/base_hil/checkpoints/last/pretrained_model",
        )
        self.assertIn("Human Intervention learning loop", text)
        self.assertIn("lerobot/interventions", text)
        self.assertIn("--steps=3000", text)
        self.assertIn("outputs/train/base_hil/checkpoints/last/pretrained_model", text)


if __name__ == "__main__":
    unittest.main()
