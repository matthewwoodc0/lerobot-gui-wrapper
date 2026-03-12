from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .commands import build_lerobot_sim_eval_command
from .config_store import get_lerobot_dir
from .experiments_service import discover_checkpoint_artifacts
from .gui_forms import build_deploy_request_and_command, build_record_request_and_command, build_train_request_and_command
from .run_controller_service import ManagedRunController, RunUiHooks
from .workflow_queue_models import WorkflowQueueItem
from .workflow_queue_recipes import (
    build_record_upload_queue_item,
    build_train_deploy_eval_queue_item,
    build_train_sim_eval_queue_item,
)
from .workflows import move_recorded_dataset


QueueListener = Callable[[], None]


class WorkflowQueueService:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        run_controller: ManagedRunController,
        append_log: Callable[[str], None],
    ) -> None:
        self._config = config
        self._run_controller = run_controller
        self._append_log = append_log
        self._listeners: list[QueueListener] = []
        self._items: list[WorkflowQueueItem] = []
        self._next_queue_id = 1
        self._active_queue_id: int | None = None
        self._state_path = self._queue_state_path()
        self._load_persisted_state()

    def next_queue_id(self) -> int:
        queue_id = self._next_queue_id
        self._next_queue_id += 1
        return queue_id

    def add_listener(self, listener: QueueListener) -> None:
        self._listeners.append(listener)

    def snapshots(self) -> list[dict[str, Any]]:
        return [item.snapshot() for item in self._items]

    def active_snapshot(self) -> dict[str, Any] | None:
        item = self._active_item()
        return item.snapshot() if item is not None else None

    def has_pending_work(self) -> bool:
        return any(item.status in {"queued", "running"} for item in self._items)

    def enqueue(self, item: WorkflowQueueItem) -> tuple[bool, str]:
        item.resume_required = False
        self._items.append(item)
        self._append_log(f"Queued workflow #{item.queue_id}: {item.title}")
        self._persist_state()
        self._notify()
        self.start_if_idle()
        return True, f"Queued workflow #{item.queue_id}: {item.title}"

    def cancel_active(self) -> tuple[bool, str]:
        item = self._active_item()
        if item is None:
            return False, "No queued workflow is active."
        if item.status == "queued":
            item.status = "canceled"
            item.error_text = "Canceled before launch."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            return True, "Queued workflow canceled."
        ok, message = self._run_controller.cancel_active_run()
        if ok:
            item.log_lines.append("Cancel requested.")
            self._persist_state()
            self._notify()
        return ok, message

    def resume_pending(self) -> tuple[bool, str]:
        resumed = 0
        for item in self._items:
            if item.status == "queued" and item.resume_required:
                item.resume_required = False
                resumed += 1
        self._persist_state()
        self._notify()
        self.start_if_idle()
        if resumed == 0:
            return False, "No paused queued workflows are waiting for resume."
        return True, f"Resumed {resumed} queued workflow(s)."

    def retry_interrupted_step(self, queue_id: int) -> tuple[bool, str]:
        item = next((entry for entry in self._items if entry.queue_id == queue_id), None)
        if item is None:
            return False, f"Queue item #{queue_id} was not found."
        if item.status != "interrupted":
            return False, "Only interrupted workflows can retry the current step."
        item.status = "queued"
        item.resume_required = False
        item.current_command = []
        item.error_text = ""
        item.log_lines.append(f"Retrying interrupted step: {item.current_step_label or 'current step'}")
        self._persist_state()
        self._notify()
        self.start_if_idle()
        return True, f"Retrying workflow #{queue_id} from step '{item.current_step_label or item.current_step_index}'."

    def clear_finished_interrupted(self) -> tuple[bool, str]:
        removable = {"success", "failed", "canceled", "interrupted"}
        before = len(self._items)
        self._items = [item for item in self._items if item.status not in removable]
        after = len(self._items)
        if self._active_queue_id is not None and not any(item.queue_id == self._active_queue_id for item in self._items):
            self._active_queue_id = None
        self._persist_state()
        self._notify()
        removed = before - after
        if removed == 0:
            return False, "No finished or interrupted workflows were cleared."
        return True, f"Cleared {removed} finished/interrupted workflow(s)."

    def start_if_idle(self) -> None:
        if self._active_queue_id is not None:
            return
        if self._run_controller.has_active_process():
            return
        next_item = next((item for item in self._items if item.status == "queued" and not item.resume_required), None)
        if next_item is None:
            return
        self._active_queue_id = next_item.queue_id
        self._start_current_step(next_item)

    def _queue_state_path(self) -> Path:
        runs_dir = Path(str(self._config.get("runs_dir", "")).strip() or ".")
        return runs_dir / "queue_state.json"

    def _atomic_write(self, destination: Path, payload: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _persist_state(self) -> None:
        payload = {
            "next_queue_id": self._next_queue_id,
            "items": [item.to_persisted_payload() for item in self._items],
        }
        self._atomic_write(self._state_path, json.dumps(payload, indent=2) + "\n")

    def _load_persisted_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items_raw = payload.get("items")
        if not isinstance(items_raw, list):
            return
        loaded_items: list[WorkflowQueueItem] = []
        converted_running = False
        for item_raw in items_raw:
            if not isinstance(item_raw, dict):
                continue
            item = WorkflowQueueItem.from_persisted_payload(item_raw)
            if item is None:
                continue
            if item.status == "running":
                item.status = "interrupted"
                item.error_text = "App exited before completion."
                item.log_lines.append("App exited before completion.")
                converted_running = True
            elif item.status == "queued":
                item.resume_required = True
            loaded_items.append(item)
        self._items = loaded_items
        next_queue_id = payload.get("next_queue_id")
        if isinstance(next_queue_id, int) and next_queue_id > 0:
            self._next_queue_id = next_queue_id
        elif self._items:
            self._next_queue_id = max(item.queue_id for item in self._items) + 1
        if converted_running or any(item.resume_required for item in self._items):
            self._persist_state()

    def _notify(self) -> None:
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:
                continue

    def _active_item(self) -> WorkflowQueueItem | None:
        if self._active_queue_id is None:
            return None
        return next((item for item in self._items if item.queue_id == self._active_queue_id), None)

    def _read_metadata(self, artifact_path: Path | None) -> dict[str, Any] | None:
        if artifact_path is None:
            return None
        metadata_path = artifact_path / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _workflow_context(self, item: WorkflowQueueItem) -> dict[str, Any]:
        latest_run_id = ""
        if item.artifacts:
            latest_run_id = str(item.artifacts[-1].get("run_id", "")).strip()
        context = {
            "workflow_queue_id": item.queue_id,
            "workflow_recipe": item.recipe_type,
            "workflow_step_index": item.current_step_index,
            "workflow_step_label": item.current_step_label,
        }
        if latest_run_id:
            context["workflow_prev_run_id"] = latest_run_id
        return context

    def _append_item_log(self, item: WorkflowQueueItem, line: str) -> None:
        text = str(line)
        item.log_lines.append(text)
        self._notify()

    def _remember_artifact(self, item: WorkflowQueueItem, artifact_path: Path) -> None:
        item.current_artifact_path = Path(artifact_path)
        item.current_artifact_metadata = self._read_metadata(item.current_artifact_path)
        self._persist_state()
        self._notify()

    def _launch_step(
        self,
        item: WorkflowQueueItem,
        *,
        label: str,
        run_mode: str,
        cmd: list[str],
        cwd: Path | None,
        artifact_context: dict[str, Any],
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
    ) -> None:
        item.status = "running"
        item.current_command = list(cmd)
        item.error_text = ""
        item.current_artifact_path = None
        item.current_artifact_metadata = None
        item.resume_required = False
        item.log_lines.append(f"Starting step: {label}")
        item.log_lines.append(" ".join(str(part) for part in cmd))
        self._persist_state()
        self._notify()

        hooks = RunUiHooks(
            set_running=lambda _active, _status, _is_error: None,
            append_output_line=lambda line: self._append_item_log(item, line),
            append_output_chunk=None,
            on_artifact_written=lambda artifact_path: self._remember_artifact(item, artifact_path),
        )

        ok, message = self._run_controller.run_process_async(
            cmd=cmd,
            cwd=cwd,
            hooks=hooks,
            complete_callback=lambda return_code, canceled: self._finish_step(item, return_code, canceled),
            expected_episodes=expected_episodes,
            expected_seconds=expected_seconds,
            run_mode=run_mode,
            preflight_checks=None,
            artifact_context=artifact_context,
        )
        if not ok:
            item.status = "failed"
            item.error_text = message or f"Unable to start {label.lower()}."
            item.log_lines.append(item.error_text)
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            self.start_if_idle()

    def _finish_step(self, item: WorkflowQueueItem, return_code: int, canceled: bool) -> None:
        metadata = self._read_metadata(item.current_artifact_path)
        if item.current_artifact_path is not None:
            item.artifacts.append(
                {
                    "step_label": item.current_step_label,
                    "path": str(item.current_artifact_path),
                    "run_id": str(metadata.get("run_id", "")) if isinstance(metadata, dict) else "",
                    "mode": str(metadata.get("mode", "")) if isinstance(metadata, dict) else "",
                }
            )
        item.current_artifact_metadata = metadata
        if canceled:
            item.status = "canceled"
            item.error_text = "Canceled by user."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            self.start_if_idle()
            return
        if return_code != 0:
            item.status = "interrupted"
            item.error_text = f"Step interrupted with exit code {return_code}."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            self.start_if_idle()
            return

        item.current_step_index += 1
        if item.current_step_index >= len(item.step_labels):
            item.status = "success"
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            self.start_if_idle()
            return
        item.status = "queued"
        item.current_command = []
        self._persist_state()
        self._notify()
        self._start_current_step(item)

    def _selected_model_path(self, item: WorkflowQueueItem) -> str:
        metadata = item.current_artifact_metadata or {}
        checkpoints = metadata.get("checkpoint_artifacts")
        if isinstance(checkpoints, list):
            for checkpoint in checkpoints:
                if not isinstance(checkpoint, dict):
                    continue
                path_value = str(checkpoint.get("path", "")).strip()
                if path_value:
                    return path_value
        output_dir = str(metadata.get("output_dir_resolved", "")).strip() or str(metadata.get("output_dir", "")).strip()
        if output_dir:
            artifacts = discover_checkpoint_artifacts(output_dir)
            for artifact in artifacts:
                path_value = str(artifact.get("path", "")).strip()
                if path_value:
                    return path_value
            return output_dir
        return ""

    def _start_current_step(self, item: WorkflowQueueItem) -> None:
        if item.recipe_type == "record_upload":
            self._start_record_upload_step(item)
            return
        if item.recipe_type == "train_sim_eval":
            self._start_train_sim_eval_step(item)
            return
        if item.recipe_type == "train_deploy_eval":
            self._start_train_deploy_eval_step(item)
            return
        item.status = "failed"
        item.error_text = f"Unknown workflow recipe: {item.recipe_type}"
        self._active_queue_id = None
        self._persist_state()
        self._notify()

    def _start_record_upload_step(self, item: WorkflowQueueItem) -> None:
        payload = item.payload
        if item.current_step_index == 0:
            request, cmd, error = build_record_request_and_command(
                config=self._config,
                dataset_input=str(payload.get("dataset_input", "")),
                episodes_raw=str(payload.get("episodes_raw", "")),
                duration_raw=str(payload.get("duration_raw", "")),
                task_raw=str(payload.get("task_raw", "")),
                dataset_dir_raw=str(payload.get("dataset_dir_raw", "")),
                upload_enabled=False,
                target_hz_raw=str(payload.get("target_hz_raw", "")),
            )
            if error or request is None or cmd is None:
                item.status = "failed"
                item.error_text = error or "Unable to build record command."
                self._active_queue_id = None
                self._persist_state()
                self._notify()
                return
            payload["dataset_repo_id"] = request.dataset_repo_id
            payload["dataset_name"] = request.dataset_name
            payload["dataset_root"] = str(request.dataset_root)
            self._launch_step(
                item,
                label="Record",
                run_mode="record",
                cmd=cmd,
                cwd=get_lerobot_dir(self._config),
                expected_episodes=request.num_episodes,
                expected_seconds=request.num_episodes * request.episode_time_s,
                artifact_context={
                    **self._workflow_context(item),
                    "dataset_repo_id": request.dataset_repo_id,
                },
            )
            return

        dataset_repo_id = str(payload.get("dataset_repo_id", "")).strip()
        dataset_name = str(payload.get("dataset_name", "")).strip()
        dataset_root = Path(str(payload.get("dataset_root", "")).strip() or ".")
        active_dataset = move_recorded_dataset(
            lerobot_dir=get_lerobot_dir(self._config),
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            log=lambda line: self._append_item_log(item, line),
        )
        payload["resolved_dataset_path"] = str(active_dataset)
        upload_cmd = [
            "huggingface-cli",
            "upload",
            dataset_repo_id,
            str(active_dataset),
            "--repo-type",
            "dataset",
        ]
        self._launch_step(
            item,
            label="Upload",
            run_mode="upload",
            cmd=upload_cmd,
            cwd=get_lerobot_dir(self._config),
            artifact_context={
                **self._workflow_context(item),
                "dataset_repo_id": dataset_repo_id,
                "dataset_path": str(active_dataset),
            },
        )

    def _start_train_sim_eval_step(self, item: WorkflowQueueItem) -> None:
        payload = item.payload
        train_form_values = dict(payload.get("train_form_values", {}))
        sim_eval_settings = dict(payload.get("sim_eval_settings", {}))
        if item.current_step_index == 0:
            request, cmd, error = build_train_request_and_command(form_values=train_form_values, config=self._config)
            if error or request is None or cmd is None:
                item.status = "failed"
                item.error_text = error or "Unable to build training command."
                self._active_queue_id = None
                self._persist_state()
                self._notify()
                return
            payload["dataset_repo_id"] = request.dataset_repo_id
            self._launch_step(
                item,
                label="Train",
                run_mode="train",
                cmd=cmd,
                cwd=get_lerobot_dir(self._config),
                artifact_context={
                    **self._workflow_context(item),
                    "dataset_repo_id": request.dataset_repo_id,
                    "policy_type": request.policy_type,
                    "output_dir": request.output_dir,
                    "device": request.device,
                    "job_name": request.job_name,
                    "resume_from": request.resume_from,
                    "wandb_enabled": request.wandb_enabled,
                    "wandb_project": request.wandb_project,
                },
            )
            return

        model_path = self._selected_model_path(item)
        if not model_path:
            item.status = "failed"
            item.error_text = "Training finished, but no checkpoint/model artifact was found for sim eval."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            return
        payload["model_path"] = model_path
        try:
            sim_eval_cmd = build_lerobot_sim_eval_command(
                self._config,
                request={
                    "model_path": model_path,
                    "output_dir": str(sim_eval_settings.get("output_dir", "")).strip(),
                    "env_type": str(sim_eval_settings.get("env_type", "")).strip(),
                    "task": str(sim_eval_settings.get("task", "")).strip(),
                    "benchmark": str(sim_eval_settings.get("benchmark", "")).strip(),
                    "episodes": str(sim_eval_settings.get("episodes", "")).strip(),
                    "batch_size": str(sim_eval_settings.get("batch_size", "")).strip(),
                    "seed": str(sim_eval_settings.get("seed", "")).strip(),
                    "device": str(sim_eval_settings.get("device", "")).strip(),
                    "job_name": str(sim_eval_settings.get("job_name", "")).strip(),
                    "trust_remote_code": bool(sim_eval_settings.get("trust_remote_code", False)),
                },
            )
        except ValueError as exc:
            item.status = "failed"
            item.error_text = str(exc)
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            return
        self._launch_step(
            item,
            label="Sim Eval",
            run_mode="sim_eval",
            cmd=sim_eval_cmd,
            cwd=get_lerobot_dir(self._config),
            artifact_context={
                **self._workflow_context(item),
                "dataset_repo_id": str(payload.get("dataset_repo_id", "")).strip(),
                "model_path": model_path,
                "output_dir": str(sim_eval_settings.get("output_dir", "")).strip(),
            },
        )

    def _start_train_deploy_eval_step(self, item: WorkflowQueueItem) -> None:
        payload = item.payload
        train_form_values = dict(payload.get("train_form_values", {}))
        deploy_settings = dict(payload.get("deploy_settings", {}))
        if item.current_step_index == 0:
            request, cmd, error = build_train_request_and_command(form_values=train_form_values, config=self._config)
            if error or request is None or cmd is None:
                item.status = "failed"
                item.error_text = error or "Unable to build training command."
                self._active_queue_id = None
                self._persist_state()
                self._notify()
                return
            payload["dataset_repo_id"] = request.dataset_repo_id
            self._launch_step(
                item,
                label="Train",
                run_mode="train",
                cmd=cmd,
                cwd=get_lerobot_dir(self._config),
                artifact_context={
                    **self._workflow_context(item),
                    "dataset_repo_id": request.dataset_repo_id,
                    "policy_type": request.policy_type,
                    "output_dir": request.output_dir,
                    "device": request.device,
                    "job_name": request.job_name,
                    "resume_from": request.resume_from,
                },
            )
            return

        model_path = self._selected_model_path(item)
        if not model_path:
            item.status = "failed"
            item.error_text = "Training finished, but no checkpoint/model artifact was found for deploy eval."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            return
        payload["model_path"] = model_path
        request, cmd, updated_config, error = build_deploy_request_and_command(
            config=self._config,
            deploy_root_raw=str(deploy_settings.get("deploy_root_raw", self._config.get("trained_models_dir", ""))),
            deploy_model_raw=model_path,
            eval_dataset_raw=str(deploy_settings.get("eval_dataset_raw", "")),
            eval_episodes_raw=str(deploy_settings.get("eval_episodes_raw", "")),
            eval_duration_raw=str(deploy_settings.get("eval_duration_raw", "")),
            eval_task_raw=str(deploy_settings.get("eval_task_raw", "")),
            target_hz_raw=str(deploy_settings.get("target_hz_raw", "")),
        )
        if error or request is None or cmd is None:
            item.status = "failed"
            item.error_text = error or "Unable to build deploy eval command."
            self._active_queue_id = None
            self._persist_state()
            self._notify()
            return
        _ = updated_config
        self._launch_step(
            item,
            label="Deploy Eval",
            run_mode="deploy",
            cmd=cmd,
            cwd=get_lerobot_dir(self._config),
            artifact_context={
                **self._workflow_context(item),
                "dataset_repo_id": request.eval_repo_id,
                "model_path": str(request.model_path),
            },
        )
