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
    build_train_deploy_eval_queue_item,
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
            config["runs_dir"] = str(Path(tmpdir) / "runs")
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
            ), patch("robot_pipeline_app.workflow_queue.save_config") as mocked_save:
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
                self.assertEqual(config["last_dataset_name"], "demo")
                self.assertEqual(config["last_dataset_repo_id"], "alice/demo")
                mocked_save.assert_called()

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
        self.assertTrue(any("Added workflow #1" in line for line in logs))

    def test_train_sim_eval_recipe_uses_discovered_checkpoint_for_followup(self) -> None:
        controller = _FakeQueueRunController()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
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
            ), patch("robot_pipeline_app.workflow_queue.save_config") as mocked_save:
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
                self.assertEqual(config["last_train_dataset"], "alice/trainset")
                self.assertEqual(config["last_train_policy_type"], "act")
                mocked_save.assert_called()

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

    def test_train_workflow_auto_job_name_resolves_before_launch(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
            config["hf_username"] = "alice"
            config["last_train_job_name"] = "demo_train_act_1"
            trained_models_dir = Path(tmpdir) / "trained_models"
            occupied = trained_models_dir / "demo_train_act_1"
            occupied.mkdir(parents=True, exist_ok=True)
            (occupied / "checkpoint.txt").write_text("occupied\n", encoding="utf-8")
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            item = build_train_sim_eval_queue_item(
                queue_id=queue.next_queue_id(),
                train_form_values={
                    "dataset_repo_id": "alice/demo-train",
                    "policy_type": "act",
                    "output_dir": str(trained_models_dir),
                    "device": "cuda",
                    "job_name": "",
                },
                train_job_name_state={"value": "", "mode": "auto"},
                sim_eval_settings={"episodes": "5", "output_dir": str(Path(tmpdir) / "sim_eval")},
            )

            captured: dict[str, Any] = {}

            def fake_build_train_request_and_command(*, form_values, config):  # type: ignore[no-untyped-def]
                captured.update(form_values)
                request = SimpleNamespace(
                    dataset_repo_id=form_values["dataset_repo_id"],
                    policy_type=form_values["policy_type"],
                    output_dir=form_values["output_dir"],
                    device=form_values["device"],
                    job_name=form_values["job_name"],
                    resume_from="",
                    wandb_enabled=False,
                    wandb_project="",
                )
                return request, ["python3", "-m", "lerobot.train"], None

            with patch("robot_pipeline_app.workflow_queue.build_train_request_and_command", side_effect=fake_build_train_request_and_command), patch(
                "robot_pipeline_app.repo_utils.model_exists_on_hf",
                return_value=False,
            ):
                ok, _message = queue.enqueue(item)

            self.assertTrue(ok)
            self.assertEqual(captured["job_name"], "demo_train_act_2")

    def test_train_deploy_eval_auto_name_resolves_before_deploy_step(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
            config["hf_username"] = "alice"
            deploy_data_dir = Path(tmpdir) / "deploy_data"
            config["deploy_data_dir"] = str(deploy_data_dir)
            (deploy_data_dir / "eval_run_1").mkdir(parents=True, exist_ok=True)
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            item = build_train_deploy_eval_queue_item(
                queue_id=queue.next_queue_id(),
                train_form_values={
                    "dataset_repo_id": "alice/trainset",
                    "policy_type": "act",
                    "output_dir": str(Path(tmpdir) / "trained_models"),
                    "device": "cuda",
                },
                train_job_name_state={"value": "", "mode": "auto"},
                deploy_settings={
                    "deploy_root_raw": str(Path(tmpdir) / "trained_models"),
                    "eval_dataset_raw": "alice/eval_run_1",
                    "eval_episodes_raw": "5",
                    "eval_duration_raw": "20",
                    "eval_task_raw": "pick",
                },
                deploy_eval_name_state={"value": "alice/eval_run_1", "mode": "auto"},
            )

            train_request = SimpleNamespace(
                dataset_repo_id="alice/trainset",
                policy_type="act",
                output_dir=str(Path(tmpdir) / "trained_models"),
                device="cuda",
                job_name="demo_train_act_1",
                resume_from="",
                wandb_enabled=False,
                wandb_project="",
            )

            captured: dict[str, Any] = {}

            def fake_build_deploy_request_and_command(*, config, deploy_root_raw, deploy_model_raw, eval_dataset_raw, eval_episodes_raw, eval_duration_raw, eval_task_raw, target_hz_raw):  # type: ignore[no-untyped-def]
                captured["eval_dataset_raw"] = eval_dataset_raw
                request = SimpleNamespace(
                    model_path=Path(deploy_model_raw),
                    eval_repo_id=eval_dataset_raw,
                    eval_num_episodes=int(eval_episodes_raw),
                    eval_duration_s=int(eval_duration_raw),
                    eval_task=eval_task_raw,
                )
                return request, ["python3", "-m", "lerobot.deploy"], {"last_eval_dataset_name": "eval_run_2"}, None

            with patch("robot_pipeline_app.workflow_queue.build_train_request_and_command", return_value=(train_request, ["python3", "-m", "lerobot.train"], None)), patch(
                "robot_pipeline_app.workflow_queue.build_deploy_request_and_command",
                side_effect=fake_build_deploy_request_and_command,
            ), patch("robot_pipeline_app.workflow_queue.save_config"):
                ok, _message = queue.enqueue(item)
                self.assertTrue(ok)
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

            self.assertEqual(captured["eval_dataset_raw"], "alice/eval_run_2")

    def test_enqueue_persists_workflow_state(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            item = build_record_upload_queue_item(
                queue_id=queue.next_queue_id(),
                dataset_input="alice/demo",
                episodes_raw="2",
                duration_raw="20",
                task_raw="pick place",
                dataset_dir_raw=str(Path(tmpdir) / "datasets"),
            )

            request = SimpleNamespace(
                dataset_repo_id="alice/demo",
                dataset_name="demo",
                dataset_root=Path(tmpdir) / "datasets",
                num_episodes=2,
                episode_time_s=20,
            )

            with patch("robot_pipeline_app.workflow_queue.build_record_request_and_command", return_value=(request, ["python3", "-m", "lerobot.record"], None)):
                queue.enqueue(item)

            state_path = Path(config["runs_dir"]) / "workflow_state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["items"][0]["recipe_type"], "record_upload")
            self.assertEqual(payload["items"][0]["status"], "running")

    def test_restart_converts_running_to_interrupted(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            state_path = runs_dir / "workflow_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "next_queue_id": 2,
                        "items": [
                            {
                                "queue_id": 1,
                                "recipe_type": "record_upload",
                                "title": "Record -> Upload (alice/demo)",
                                "step_labels": ["Record", "Upload"],
                                "payload": {"dataset_input": "alice/demo"},
                                "status": "running",
                                "current_step_index": 0,
                                "current_command": ["python3", "-m", "lerobot.record"],
                                "artifacts": [],
                                "error_text": "",
                                "log_text": "Starting step: Record",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(runs_dir)

            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)

            snapshot = queue.snapshots()[0]
            self.assertEqual(snapshot["status"], "interrupted")
            self.assertIn("App exited before completion", snapshot["error_text"])

    def test_restart_loads_legacy_queue_state_path_and_rewrites_new_file(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            legacy_state_path = runs_dir / "queue_state.json"
            legacy_state_path.write_text(
                json.dumps(
                    {
                        "next_queue_id": 2,
                        "items": [
                            {
                                "queue_id": 1,
                                "recipe_type": "record_upload",
                                "title": "Record -> Upload (alice/demo)",
                                "step_labels": ["Record", "Upload"],
                                "payload": {"dataset_input": "alice/demo"},
                                "status": "queued",
                                "current_step_index": 0,
                                "current_command": [],
                                "artifacts": [],
                                "error_text": "",
                                "log_text": "",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(runs_dir)

            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)

            snapshot = queue.snapshots()[0]
            self.assertEqual(snapshot["status"], "queued")
            self.assertTrue((runs_dir / "workflow_state.json").exists())

    def test_resume_pending_ignores_interrupted_items(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            (runs_dir / "workflow_state.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "queue_id": 1,
                                "recipe_type": "record_upload",
                                "title": "Queued",
                                "step_labels": ["Record", "Upload"],
                                "payload": {"dataset_input": "alice/demo"},
                                "status": "queued",
                                "current_step_index": 0,
                                "current_command": [],
                                "artifacts": [],
                                "error_text": "",
                                "log_text": "",
                            },
                            {
                                "queue_id": 2,
                                "recipe_type": "record_upload",
                                "title": "Interrupted",
                                "step_labels": ["Record", "Upload"],
                                "payload": {"dataset_input": "alice/demo"},
                                "status": "interrupted",
                                "current_step_index": 1,
                                "current_command": ["python3", "-m", "lerobot.record"],
                                "artifacts": [],
                                "error_text": "App exited before completion.",
                                "log_text": "",
                            },
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(runs_dir)
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)

            request = SimpleNamespace(
                dataset_repo_id="alice/demo",
                dataset_name="demo",
                dataset_root=Path(tmpdir) / "datasets",
                num_episodes=2,
                episode_time_s=20,
            )
            with patch("robot_pipeline_app.workflow_queue.build_record_request_and_command", return_value=(request, ["python3", "-m", "lerobot.record"], None)):
                ok, _message = queue.resume_pending()

            self.assertTrue(ok)
            self.assertEqual(len(controller.calls), 1)
            self.assertEqual(queue.snapshots()[1]["status"], "interrupted")

    def test_retry_interrupted_step_restarts_same_step(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            item = build_train_sim_eval_queue_item(
                queue_id=queue.next_queue_id(),
                train_form_values={"dataset_repo_id": "alice/trainset"},
                sim_eval_settings={"episodes": "5"},
            )
            item.status = "interrupted"
            item.current_step_index = 1
            item.current_command = ["python3", "-m", "lerobot.eval"]
            item.current_artifact_metadata = {"checkpoint_artifacts": [{"path": str(Path(tmpdir) / "checkpoints" / "last")}]}
            queue._items.append(item)
            queue._persist_state()

            with patch("robot_pipeline_app.workflow_queue.build_lerobot_sim_eval_command", return_value=["python3", "-m", "lerobot.eval"]):
                ok, _message = queue.retry_interrupted_step(item.queue_id)

            self.assertTrue(ok)
            self.assertEqual(len(controller.calls), 1)
            self.assertEqual(controller.calls[0]["run_mode"], "sim_eval")

    def test_clear_finished_interrupted_rewrites_persisted_state(self) -> None:
        controller = _FakeQueueRunController()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = tmpdir
            config["runs_dir"] = str(Path(tmpdir) / "runs")
            queue = WorkflowQueueService(config=config, run_controller=controller, append_log=lambda _line: None)
            queue._items = [
                build_record_upload_queue_item(
                    queue_id=1,
                    dataset_input="alice/demo",
                    episodes_raw="1",
                    duration_raw="20",
                    task_raw="pick",
                    dataset_dir_raw=str(Path(tmpdir) / "datasets"),
                ),
                build_record_upload_queue_item(
                    queue_id=2,
                    dataset_input="alice/demo",
                    episodes_raw="1",
                    duration_raw="20",
                    task_raw="pick",
                    dataset_dir_raw=str(Path(tmpdir) / "datasets"),
                ),
            ]
            queue._items[0].status = "success"
            queue._items[1].status = "interrupted"
            queue._persist_state()

            ok, _message = queue.clear_finished_interrupted()

            self.assertTrue(ok)
            payload = json.loads((Path(config["runs_dir"]) / "workflow_state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["items"], [])


if __name__ == "__main__":
    unittest.main()
