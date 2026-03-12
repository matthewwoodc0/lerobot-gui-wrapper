from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.workflow_queue import (
    WorkflowQueueService,
    build_record_upload_queue_item,
    build_train_sim_eval_queue_item,
)


class _FakeQueueRunController:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._active = False
        self._hooks: Any = None
        self._complete_callback: Any = None

    def has_active_process(self) -> bool:
        return self._active

    def run_process_async(self, *, cmd, cwd, hooks, complete_callback, **kwargs):  # type: ignore[no-untyped-def]
        self._active = True
        self.calls.append(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                **kwargs,
            }
        )
        self._hooks = hooks
        self._complete_callback = complete_callback
        return True, None

    def cancel_active_run(self) -> tuple[bool, str]:
        if not self._active:
            return False, "No active run."
        return True, "Cancel requested."

    def finish(self, *, artifact_path: Path | None = None, return_code: int = 0, canceled: bool = False) -> None:
        if artifact_path is not None and self._hooks is not None and self._hooks.on_artifact_written is not None:
            self._hooks.on_artifact_written(artifact_path)
        callback = self._complete_callback
        self._active = False
        self._hooks = None
        self._complete_callback = None
        if callback is not None:
            callback(return_code, canceled)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class WorkflowQueueTests(unittest.TestCase):
    def test_record_upload_recipe_runs_record_then_upload_with_history_context(self) -> None:
        controller = _FakeQueueRunController()
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            dataset_root = Path(tmpdir) / "datasets"
            active_dataset = dataset_root / "demo"

            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=logs.append)
            item = build_record_upload_queue_item(
                queue_id=queue.next_queue_id(),
                dataset_input="alice/demo",
                episodes_raw="2",
                duration_raw="20",
                task_raw="pick place",
                dataset_dir_raw=str(dataset_root),
                target_hz_raw="30",
            )

            request = SimpleNamespace(
                dataset_repo_id="alice/demo",
                dataset_name="demo",
                dataset_root=dataset_root,
                num_episodes=2,
                episode_time_s=20,
            )

            with patch("robot_pipeline_app.workflow_queue.build_record_request_and_command", return_value=(request, ["python3", "-m", "lerobot.record"], None)), patch(
                "robot_pipeline_app.workflow_queue.move_recorded_dataset",
                return_value=active_dataset,
            ):
                ok, _message = queue.enqueue(item)

                self.assertTrue(ok)
                self.assertEqual(len(controller.calls), 1)
                first = controller.calls[0]
                self.assertEqual(first["run_mode"], "record")
                self.assertEqual(first["artifact_context"]["workflow_recipe"], "record_upload")
                self.assertEqual(first["artifact_context"]["workflow_step_label"], "Record")
                self.assertEqual(first["artifact_context"]["dataset_repo_id"], "alice/demo")

                record_artifact = Path(tmpdir) / "runs" / "record_1"
                _write_metadata(record_artifact, {"run_id": "record_1", "mode": "record"})
                controller.finish(artifact_path=record_artifact)

                self.assertEqual(len(controller.calls), 2)
                second = controller.calls[1]
                self.assertEqual(second["run_mode"], "upload")
                self.assertEqual(
                    second["cmd"],
                    ["huggingface-cli", "upload", "alice/demo", str(active_dataset), "--repo-type", "dataset"],
                )
                self.assertEqual(second["artifact_context"]["workflow_prev_run_id"], "record_1")
                self.assertEqual(second["artifact_context"]["dataset_path"], str(active_dataset))

                upload_artifact = Path(tmpdir) / "runs" / "upload_1"
                _write_metadata(upload_artifact, {"run_id": "upload_1", "mode": "upload"})
                controller.finish(artifact_path=upload_artifact)

        snapshot = queue.snapshots()[0]
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(len(snapshot["artifacts"]), 2)
        self.assertTrue(any("Queued workflow #1" in line for line in logs))

    def test_train_sim_eval_recipe_uses_discovered_checkpoint_for_followup(self) -> None:
        controller = _FakeQueueRunController()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            item = build_train_sim_eval_queue_item(
                queue_id=queue.next_queue_id(),
                train_form_values={
                    "dataset_repo_id": "alice/trainset",
                    "policy_type": "act",
                    "output_dir": str(Path(tmpdir) / "trained_models"),
                    "device": "cuda",
                },
                sim_eval_settings={
                    "env_type": "aloha",
                    "task": "pick",
                    "episodes": "5",
                    "output_dir": str(Path(tmpdir) / "sim_eval"),
                    "device": "cuda",
                },
            )

            train_request = SimpleNamespace(
                dataset_repo_id="alice/trainset",
                policy_type="act",
                output_dir=str(Path(tmpdir) / "trained_models"),
                device="cuda",
                job_name="",
                resume_from="",
                wandb_enabled=False,
                wandb_project="",
            )

            with patch("robot_pipeline_app.workflow_queue.build_train_request_and_command", return_value=(train_request, ["python3", "-m", "lerobot.train"], None)), patch(
                "robot_pipeline_app.workflow_queue.build_lerobot_sim_eval_command",
                return_value=["python3", "-m", "lerobot.eval"],
            ):
                ok, _message = queue.enqueue(item)

                self.assertTrue(ok)
                self.assertEqual(len(controller.calls), 1)
                first = controller.calls[0]
                self.assertEqual(first["run_mode"], "train")
                self.assertEqual(first["artifact_context"]["workflow_recipe"], "train_sim_eval")
                self.assertEqual(first["artifact_context"]["dataset_repo_id"], "alice/trainset")

                train_artifact = Path(tmpdir) / "runs" / "train_1"
                _write_metadata(
                    train_artifact,
                    {
                        "run_id": "train_1",
                        "mode": "train",
                        "checkpoint_artifacts": [{"path": str(Path(tmpdir) / "checkpoints" / "last")}],
                    },
                )
                controller.finish(artifact_path=train_artifact)

                self.assertEqual(len(controller.calls), 2)
                second = controller.calls[1]
                self.assertEqual(second["run_mode"], "sim_eval")
                self.assertEqual(second["cmd"], ["python3", "-m", "lerobot.eval"])
                self.assertEqual(second["artifact_context"]["workflow_prev_run_id"], "train_1")
                self.assertEqual(second["artifact_context"]["model_path"], str(Path(tmpdir) / "checkpoints" / "last"))

                sim_eval_artifact = Path(tmpdir) / "runs" / "sim_eval_1"
                _write_metadata(sim_eval_artifact, {"run_id": "sim_eval_1", "mode": "sim_eval"})
                controller.finish(artifact_path=sim_eval_artifact)

        snapshot = queue.snapshots()[0]
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(len(snapshot["artifacts"]), 2)


if __name__ == "__main__":
    unittest.main()
