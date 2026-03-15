from __future__ import annotations

from typing import Any

from .workflow_queue_models import WorkflowQueueItem


def build_record_upload_queue_item(
    *,
    queue_id: int,
    dataset_input: str,
    dataset_name_state: dict[str, Any] | None = None,
    episodes_raw: str,
    duration_raw: str,
    task_raw: str,
    dataset_dir_raw: str,
    target_hz_raw: str = "",
) -> WorkflowQueueItem:
    return WorkflowQueueItem(
        queue_id=queue_id,
        recipe_type="record_upload",
        title=f"Record -> Upload ({str(dataset_input or '').strip() or 'dataset'})",
        step_labels=["Record", "Upload"],
        payload={
            "dataset_input": str(dataset_input or "").strip(),
            "dataset_name_state": dict(dataset_name_state or {}),
            "episodes_raw": str(episodes_raw or "").strip(),
            "duration_raw": str(duration_raw or "").strip(),
            "task_raw": str(task_raw or "").strip(),
            "dataset_dir_raw": str(dataset_dir_raw or "").strip(),
            "target_hz_raw": str(target_hz_raw or "").strip(),
        },
    )


def build_train_sim_eval_queue_item(
    *,
    queue_id: int,
    train_form_values: dict[str, Any],
    train_job_name_state: dict[str, Any] | None = None,
    sim_eval_settings: dict[str, Any],
) -> WorkflowQueueItem:
    dataset_value = str(train_form_values.get("dataset_repo_id", "")).strip() or "dataset"
    return WorkflowQueueItem(
        queue_id=queue_id,
        recipe_type="train_sim_eval",
        title=f"Train -> Sim Eval ({dataset_value})",
        step_labels=["Train", "Sim Eval"],
        payload={
            "train_form_values": dict(train_form_values),
            "train_job_name_state": dict(train_job_name_state or {}),
            "sim_eval_settings": dict(sim_eval_settings),
        },
    )


def build_train_deploy_eval_queue_item(
    *,
    queue_id: int,
    train_form_values: dict[str, Any],
    train_job_name_state: dict[str, Any] | None = None,
    deploy_settings: dict[str, Any],
    deploy_eval_name_state: dict[str, Any] | None = None,
) -> WorkflowQueueItem:
    dataset_value = str(train_form_values.get("dataset_repo_id", "")).strip() or "dataset"
    return WorkflowQueueItem(
        queue_id=queue_id,
        recipe_type="train_deploy_eval",
        title=f"Train -> Deploy Eval ({dataset_value})",
        step_labels=["Train", "Deploy Eval"],
        payload={
            "train_form_values": dict(train_form_values),
            "train_job_name_state": dict(train_job_name_state or {}),
            "deploy_settings": dict(deploy_settings),
            "deploy_eval_name_state": dict(deploy_eval_name_state or {}),
        },
    )
