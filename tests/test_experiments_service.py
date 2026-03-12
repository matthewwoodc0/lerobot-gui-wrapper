from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.experiments_service import (
    build_experiment_comparison_payload,
    build_sim_eval_metadata_extra,
    build_train_metadata_extra,
    collect_experiment_runs,
    discover_checkpoint_artifacts,
    extract_wandb_run_metadata,
)


class ExperimentsServiceTests(unittest.TestCase):
    def test_discover_checkpoint_artifacts_finds_deployable_payloads_and_train_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "train"
            payload = output_dir / "checkpoints" / "checkpoint-020000" / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("weights\n", encoding="utf-8")
            (payload.parent / "train_config.json").write_text("{}\n", encoding="utf-8")

            artifacts = discover_checkpoint_artifacts(output_dir)

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["kind"], "checkpoint")
        self.assertEqual(artifacts[0]["step"], 20000)
        self.assertTrue(artifacts[0]["is_deployable"])
        self.assertTrue(str(artifacts[0]["train_config_path"]).endswith("train_config.json"))

    def test_build_train_metadata_extra_captures_metrics_wandb_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "train"
            output_dir.mkdir(parents=True, exist_ok=True)
            trainer_state = {
                "global_step": 2500,
                "log_history": [
                    {"loss": 0.42, "learning_rate": 1e-4},
                ],
            }
            (output_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
            payload = output_dir / "checkpoints" / "checkpoint-002500" / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("weights\n", encoding="utf-8")
            (payload.parent / "train_config.json").write_text("{}\n", encoding="utf-8")
            wandb_dir = output_dir / "wandb" / "latest-run" / "files"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            (wandb_dir / "wandb-summary.json").write_text(json.dumps({"loss": 0.41, "_step": 2500}), encoding="utf-8")
            (wandb_dir / "wandb-metadata.json").write_text(
                json.dumps({"run_id": "abc123", "project": "research", "entity": "alice", "display_name": "nightly"}),
                encoding="utf-8",
            )

            metadata = build_train_metadata_extra(
                context={
                    "policy_type": "diffusion",
                    "output_dir": str(output_dir),
                    "device": "cuda",
                    "job_name": "nightly",
                    "resume_from": str(payload.parent / "train_config.json"),
                    "wandb_enabled": True,
                    "wandb_project": "research",
                },
                output_lines=[
                    "step: 2400 loss: 0.45",
                    "wandb: https://wandb.ai/alice/research/runs/abc123",
                ],
                cwd=Path(tmpdir),
            )

        self.assertEqual(metadata["policy_type"], "diffusion")
        self.assertEqual(metadata["train_metrics"]["step"], 2400)
        self.assertEqual(metadata["train_metrics"]["loss"], 0.45)
        self.assertEqual(metadata["wandb"]["run_id"], "abc123")
        self.assertEqual(metadata["wandb"]["run_url"], "https://wandb.ai/alice/research/runs/abc123")
        self.assertEqual(len(metadata["checkpoint_artifacts"]), 1)

    def test_extract_wandb_run_metadata_prefers_discovered_local_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "train"
            wandb_dir = output_dir / "wandb" / "latest-run" / "files"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            (wandb_dir / "wandb-summary.json").write_text(
                json.dumps({"train/loss": 0.11, "eval/pc_success": 0.9}),
                encoding="utf-8",
            )
            (wandb_dir / "wandb-metadata.json").write_text(
                json.dumps({"run_id": "run555", "project": "proj", "entity": "alice", "display_name": "best-run"}),
                encoding="utf-8",
            )

            payload = extract_wandb_run_metadata(
                output_lines=["wandb: syncing run"],
                output_dir=output_dir,
                enabled=True,
                project="proj",
                job_name="nightly",
            )

        self.assertEqual(payload["run_id"], "run555")
        self.assertEqual(payload["run_name"], "best-run")
        self.assertEqual(payload["run_url"], "https://wandb.ai/alice/proj/runs/run555")
        self.assertEqual(payload["summary"]["train/loss"], 0.11)

    def test_build_sim_eval_metadata_extra_reads_eval_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            eval_info = {
                "overall": {
                    "avg_sum_reward": 0.75,
                    "pc_success": 0.6,
                    "eval_s": 12.5,
                    "eval_ep_s": 2.5,
                    "video_paths": ["videos/episode_1.mp4"],
                },
                "pusht": {"avg_sum_reward": 0.75},
            }
            (output_dir / "eval_info.json").write_text(json.dumps(eval_info), encoding="utf-8")

            metadata = build_sim_eval_metadata_extra(
                context={
                    "model_path": "/tmp/model",
                    "output_dir": str(output_dir),
                    "device": "cuda",
                    "env_type": "pusht",
                    "task": "push",
                    "episodes": 5,
                    "seed": 0,
                },
                output_lines=["Overall Aggregated Metrics:", "{'avg_sum_reward': 0.7}"],
                cwd=Path(tmpdir),
            )

        self.assertEqual(metadata["sim_eval"]["env_type"], "pusht")
        self.assertEqual(metadata["sim_eval_metrics"]["pc_success"], 0.6)
        self.assertEqual(metadata["sim_eval_metrics"]["avg_sum_reward"], 0.75)
        self.assertEqual(metadata["sim_eval_metrics"]["task_groups"], ["pusht"])

    def test_collect_experiment_runs_and_comparison_payload_cover_train_deploy_and_sim_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            train_run = runs_dir / "train_20260312_100000"
            deploy_run = runs_dir / "deploy_20260312_110000"
            sim_eval_run = runs_dir / "sim_eval_20260312_120000"
            train_run.mkdir(parents=True, exist_ok=True)
            deploy_run.mkdir(parents=True, exist_ok=True)
            sim_eval_run.mkdir(parents=True, exist_ok=True)

            train_output_dir = Path(tmpdir) / "outputs" / "train_1"
            payload = train_output_dir / "checkpoints" / "checkpoint-010000" / "pretrained_model"
            payload.mkdir(parents=True, exist_ok=True)
            (payload / "config.json").write_text("{}\n", encoding="utf-8")
            (payload / "model.safetensors").write_text("weights\n", encoding="utf-8")
            (payload.parent / "train_config.json").write_text("{}\n", encoding="utf-8")

            (train_run / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": train_run.name,
                        "mode": "train",
                        "status": "success",
                        "started_at_iso": "2026-03-12T10:00:00+00:00",
                        "duration_s": 120.0,
                        "command": "python -m lerobot train",
                        "command_argv": ["python", "-m", "lerobot.train", "--policy.type=act"],
                        "dataset_repo_id": "alice/demo_train",
                        "policy_type": "act",
                        "output_dir": str(train_output_dir),
                        "output_dir_resolved": str(train_output_dir),
                        "train_metrics": {"step": 10000, "loss": 0.2},
                        "checkpoint_artifacts": discover_checkpoint_artifacts(train_output_dir),
                        "wandb": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            (train_run / "command.log").write_text("step: 10000 loss: 0.2\n", encoding="utf-8")

            (deploy_run / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": deploy_run.name,
                        "mode": "deploy",
                        "status": "success",
                        "started_at_iso": "2026-03-12T11:00:00+00:00",
                        "duration_s": 90.0,
                        "command": "python -m lerobot deploy",
                        "command_argv": ["python", "-m", "lerobot.record", "--policy.device=cuda"],
                        "dataset_repo_id": "alice/eval_demo",
                        "model_path": str(payload),
                        "deploy_episode_outcomes": {
                            "total_episodes": 4,
                            "episode_outcomes": [
                                {"episode": 1, "result": "success", "tags": ["best"]},
                                {"episode": 2, "result": "success", "tags": []},
                                {"episode": 3, "result": "failed", "tags": ["grasp"]},
                                {"episode": 4, "result": "unmarked", "tags": []},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (deploy_run / "command.log").write_text("deploy output\n", encoding="utf-8")

            sim_eval_output_dir = Path(tmpdir) / "outputs" / "eval_1"
            sim_eval_output_dir.mkdir(parents=True, exist_ok=True)
            (sim_eval_output_dir / "eval_info.json").write_text(
                json.dumps({"overall": {"avg_sum_reward": 0.8, "pc_success": 0.7, "eval_s": 10.0}}),
                encoding="utf-8",
            )
            (sim_eval_run / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": sim_eval_run.name,
                        "mode": "sim_eval",
                        "status": "success",
                        "started_at_iso": "2026-03-12T12:00:00+00:00",
                        "duration_s": 30.0,
                        "command": "python -m lerobot eval",
                        "command_argv": ["python", "-m", "lerobot.scripts.lerobot_eval", "--policy.device=cuda"],
                        "model_path": str(payload),
                        "policy_type": "act",
                        "output_dir": str(sim_eval_output_dir),
                        "output_dir_resolved": str(sim_eval_output_dir),
                        "sim_eval": {"env_type": "pusht", "task": "push"},
                        "sim_eval_metrics": {"avg_sum_reward": 0.8, "pc_success": 0.7, "eval_s": 10.0},
                    }
                ),
                encoding="utf-8",
            )
            (sim_eval_run / "command.log").write_text("Overall Aggregated Metrics:\n{'pc_success': 0.7}\n", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["runs_dir"] = str(runs_dir)
            payload = collect_experiment_runs(config)
            comparison = build_experiment_comparison_payload(payload["records"])

        self.assertEqual(payload["stats"]["train"], 1)
        self.assertEqual(payload["stats"]["deploy"], 1)
        self.assertEqual(payload["stats"]["sim_eval"], 1)
        self.assertEqual(len(comparison["rows"]), 3)
        metrics_column = [row["values"][9] for row in comparison["rows"]]
        self.assertTrue(any("loss 0.2000" in value for value in metrics_column))
        self.assertTrue(any("success 2/4" in value for value in metrics_column))
        self.assertTrue(any("success 70.0%" in value for value in metrics_column))


if __name__ == "__main__":
    unittest.main()
