from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.sim_eval import build_sim_eval_request_and_command


class SimEvalBuilderTests(unittest.TestCase):
    def test_build_sim_eval_request_and_command_uses_compatibility_probed_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            fake_caps = SimpleNamespace(
                supports_sim_eval=True,
                sim_eval_support_detail="Simulation eval is supported.",
                sim_eval_entrypoint="lerobot.scripts.lerobot_eval",
                sim_eval_policy_path_flag="policy.pretrained_path",
                sim_eval_output_dir_flag="output_dir",
                sim_eval_env_type_flag="env.type",
                sim_eval_task_flag="env.task",
                sim_eval_benchmark_flag=None,
                sim_eval_episodes_flag="eval.n_episodes",
                sim_eval_batch_size_flag="eval.batch_size",
                sim_eval_seed_flag="seed",
                sim_eval_device_flag="policy.device",
                sim_eval_job_name_flag="job_name",
                supported_sim_eval_flags=("policy.pretrained_path", "output_dir", "env.type", "env.task", "eval.n_episodes", "eval.batch_size", "seed", "policy.device", "job_name", "trust_remote_code"),
            )
            config = dict(DEFAULT_CONFIG_VALUES)

            with patch("robot_pipeline_app.commands.probe_lerobot_capabilities", return_value=fake_caps), patch(
                "robot_pipeline_app.sim_eval.probe_lerobot_capabilities",
                return_value=fake_caps,
            ):
                request, cmd, error = build_sim_eval_request_and_command(
                    form_values={
                        "model_path": str(model_dir),
                        "output_dir": "outputs/eval",
                        "env_type": "pusht",
                        "task": "push",
                        "episodes": "5",
                        "batch_size": "2",
                        "seed": "0",
                        "device": "cuda",
                        "job_name": "eval-nightly",
                        "trust_remote_code": True,
                    },
                    config=config,
                )

        self.assertIsNone(error)
        assert request is not None and cmd is not None
        self.assertEqual(request.seed, 0)
        self.assertIn("--policy.pretrained_path=" + str(model_dir), cmd)
        self.assertIn("--env.type=pusht", cmd)
        self.assertIn("--env.task=push", cmd)
        self.assertIn("--eval.n_episodes=5", cmd)
        self.assertIn("--eval.batch_size=2", cmd)
        self.assertIn("--seed=0", cmd)
        self.assertIn("--policy.device=cuda", cmd)
        self.assertIn("--job_name=eval-nightly", cmd)
        self.assertIn("--trust_remote_code=true", cmd)

    def test_build_sim_eval_request_and_command_rejects_missing_env_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            request, cmd, error = build_sim_eval_request_and_command(
                form_values={
                    "model_path": str(model_dir),
                    "episodes": "5",
                },
                config=dict(DEFAULT_CONFIG_VALUES),
            )

        self.assertIsNone(request)
        self.assertIsNone(cmd)
        self.assertEqual(error, "Simulation eval needs an env type or benchmark selection.")


if __name__ == "__main__":
    unittest.main()
