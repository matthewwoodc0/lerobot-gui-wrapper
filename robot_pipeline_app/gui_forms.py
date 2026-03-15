from __future__ import annotations

from pathlib import Path
from typing import Any

from .command_overrides import apply_command_overrides, get_flag_value, get_policy_path_value
from .compat import normalize_train_resume_path, probe_lerobot_capabilities
from .commands import (
    build_lerobot_train_command,
    build_lerobot_record_command,
    build_lerobot_teleop_command,
    resolve_follower_robot_id,
    resolve_leader_robot_id,
)
from .config_store import default_for_key, normalize_path
from .constants import DEFAULT_TASK
from .deploy_diagnostics import validate_model_path
from .repo_utils import normalize_repo_id, repo_name_from_repo_id
from .types import DeployRequest, RecordRequest, TeleopRequest, TrainRequest


def _parse_optional_target_hz(raw_value: str, *, label: str) -> tuple[int | None, str | None]:
    text = str(raw_value or "").strip()
    if not text:
        return None, None
    try:
        value = int(text)
    except ValueError:
        return None, f"{label} must be an integer."
    if value <= 0:
        return None, f"{label} must be greater than zero."
    return value, None


def _validate_effective_target_hz_from_cmd(cmd: list[str], *, label: str) -> str | None:
    hz_text = get_flag_value(cmd, "dataset.fps") or get_flag_value(cmd, "fps")
    if hz_text is None:
        return None
    try:
        value = int(str(hz_text).strip())
    except ValueError:
        return f"{label} must be an integer."
    if value <= 0:
        return f"{label} must be greater than zero."
    return None


def _parse_form_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_train_dataset_target(config: dict[str, Any], dataset_input: str) -> str:
    raw_value = str(dataset_input or "").strip()
    if not raw_value:
        return ""
    candidate = Path(normalize_path(raw_value))
    try:
        if candidate.exists():
            return str(candidate)
    except OSError:
        pass
    return normalize_repo_id(str(config.get("hf_username", "")), raw_value)


def coerce_config_from_vars(
    base_config: dict[str, Any],
    config_vars: dict[str, Any],
    config_fields: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    preview = dict(base_config)
    for field in config_fields:
        key = field["key"]
        raw_value = str(config_vars[key].get()).strip()
        if raw_value == "":
            fallback = default_for_key(key, preview)
            if field["type"] == "int":
                preview[key] = int(fallback)
            elif field["type"] == "path":
                fallback_raw = str(fallback).strip()
                if key in {"follower_calibration_path", "leader_calibration_path"} and not fallback_raw:
                    preview[key] = ""
                else:
                    preview[key] = normalize_path(fallback_raw)
            else:
                preview[key] = str(fallback)
            continue
        if field["type"] == "int":
            try:
                preview[key] = int(raw_value)
            except ValueError:
                return None, f"{field['prompt']} must be an integer."
        elif field["type"] == "path":
            if key in {"follower_calibration_path", "leader_calibration_path"} and not raw_value:
                preview[key] = ""
            else:
                preview[key] = normalize_path(raw_value)
        else:
            preview[key] = raw_value
    return preview, None


def build_record_request_and_command(
    config: dict[str, Any],
    dataset_input: str,
    episodes_raw: str,
    duration_raw: str,
    task_raw: str,
    dataset_dir_raw: str,
    upload_enabled: bool,
    target_hz_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[RecordRequest | None, list[str] | None, str | None]:
    dataset_input = str(dataset_input or "").strip()
    if not dataset_input:
        return None, None, "Dataset name is required."

    try:
        episodes = int(str(episodes_raw or "").strip())
        episode_time = int(str(duration_raw or "").strip())
    except ValueError:
        return None, None, "Episodes and episode time must be integers."

    if episodes <= 0 or episode_time <= 0:
        return None, None, "Episodes and episode time must be greater than zero."
    target_hz, target_hz_error = _parse_optional_target_hz(target_hz_raw, label="Target Hz")
    if target_hz_error is not None:
        return None, None, target_hz_error

    task = str(task_raw or "").strip() or DEFAULT_TASK
    dataset_root = Path(normalize_path(str(dataset_dir_raw or "").strip() or str(config["record_data_dir"])))
    dataset_repo_id = normalize_repo_id(str(config["hf_username"]), dataset_input)
    dataset_name = repo_name_from_repo_id(dataset_repo_id)

    req = RecordRequest(
        dataset_repo_id=dataset_repo_id,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        num_episodes=episodes,
        episode_time_s=episode_time,
        task=task,
        upload_after_record=upload_enabled,
    )
    cmd = build_lerobot_record_command(
        config=config,
        dataset_repo_id=req.dataset_repo_id,
        num_episodes=req.num_episodes,
        task=req.task,
        episode_time=req.episode_time_s,
        push_to_hub=False,
        target_hz=target_hz,
        allow_blocking_compat_probe=False,
    )
    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, override_error or "Unable to apply advanced options."
    target_hz_validation_error = _validate_effective_target_hz_from_cmd(cmd, label="Target Hz")
    if target_hz_validation_error is not None:
        return None, None, target_hz_validation_error

    raw_repo_id = get_flag_value(cmd, "dataset.repo_id") or req.dataset_repo_id
    effective_repo_id = normalize_repo_id(str(config["hf_username"]), raw_repo_id)
    effective_task = get_flag_value(cmd, "dataset.single_task") or req.task
    episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.num_episodes)
    duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.episode_time_s)
    try:
        effective_episodes = int(episodes_text)
        effective_duration = int(duration_text)
    except ValueError:
        return None, None, "Advanced options must keep episodes and episode time as integers."
    if effective_episodes <= 0 or effective_duration <= 0:
        return None, None, "Advanced options must keep episodes and episode time greater than zero."

    effective_req = RecordRequest(
        dataset_repo_id=effective_repo_id,
        dataset_name=repo_name_from_repo_id(effective_repo_id),
        dataset_root=req.dataset_root,
        num_episodes=effective_episodes,
        episode_time_s=effective_duration,
        task=effective_task,
        upload_after_record=upload_enabled,
    )
    return effective_req, cmd, None


def build_deploy_request_and_command(
    config: dict[str, Any],
    deploy_root_raw: str,
    deploy_model_raw: str,
    eval_dataset_raw: str,
    eval_episodes_raw: str,
    eval_duration_raw: str,
    eval_task_raw: str,
    target_hz_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[DeployRequest | None, list[str] | None, dict[str, Any] | None, str | None]:
    models_root = Path(normalize_path(str(deploy_root_raw or "").strip() or str(config["trained_models_dir"])))
    model_path = Path(normalize_path(str(deploy_model_raw or "").strip()))
    if not model_path.is_absolute():
        model_path = models_root / model_path

    if not model_path.exists() or not model_path.is_dir():
        return None, None, None, f"Model folder not found:\n{model_path}"
    is_valid_model, detail, _ = validate_model_path(model_path)
    if not is_valid_model:
        return None, None, None, detail

    eval_dataset_input = str(eval_dataset_raw or "").strip()
    if not eval_dataset_input:
        return None, None, None, "Eval dataset name is required."

    try:
        eval_episodes = int(str(eval_episodes_raw or "").strip())
        eval_duration = int(str(eval_duration_raw or "").strip())
    except ValueError:
        return None, None, None, "Eval episodes and duration must be integers."

    if eval_episodes <= 0 or eval_duration <= 0:
        return None, None, None, "Eval episodes and duration must be greater than zero."
    target_hz, target_hz_error = _parse_optional_target_hz(target_hz_raw, label="Deploy target Hz")
    if target_hz_error is not None:
        return None, None, None, target_hz_error

    eval_task = str(eval_task_raw or "").strip() or DEFAULT_TASK
    eval_repo_id = normalize_repo_id(str(config["hf_username"]), eval_dataset_input)

    req = DeployRequest(
        model_path=model_path,
        eval_repo_id=eval_repo_id,
        eval_num_episodes=eval_episodes,
        eval_duration_s=eval_duration,
        eval_task=eval_task,
    )

    # Compute relative path from models_root for better persistence
    try:
        rel = model_path.relative_to(models_root)
        last_model_name_str = str(rel.parts[0]) if rel.parts else model_path.name
        last_checkpoint_str = str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else ""
    except ValueError:
        last_model_name_str = model_path.name
        last_checkpoint_str = ""

    updated_config = {
        "trained_models_dir": str(models_root),
        "last_model_name": last_model_name_str,
        "last_checkpoint_name": last_checkpoint_str,
        "eval_num_episodes": eval_episodes,
        "eval_duration_s": eval_duration,
        "eval_task": eval_task,
        "last_eval_dataset_name": eval_repo_id.split("/", 1)[1],
        "deploy_target_hz": str(target_hz) if target_hz is not None else "",
    }

    cmd = build_lerobot_record_command(
        config={**config, **updated_config},
        dataset_repo_id=req.eval_repo_id,
        num_episodes=req.eval_num_episodes,
        task=req.eval_task,
        episode_time=req.eval_duration_s,
        policy_path=req.model_path,
        push_to_hub=False,
        target_hz=target_hz,
        allow_blocking_compat_probe=False,
    )
    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, None, override_error or "Unable to apply advanced options."
    target_hz_validation_error = _validate_effective_target_hz_from_cmd(cmd, label="Deploy target Hz")
    if target_hz_validation_error is not None:
        return None, None, None, target_hz_validation_error
    effective_target_hz = (get_flag_value(cmd, "dataset.fps") or get_flag_value(cmd, "fps") or "").strip()
    updated_config["deploy_target_hz"] = effective_target_hz

    raw_repo_id = get_flag_value(cmd, "dataset.repo_id") or req.eval_repo_id
    effective_repo_id = normalize_repo_id(str(config["hf_username"]), raw_repo_id)
    effective_task = get_flag_value(cmd, "dataset.single_task") or req.eval_task
    episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.eval_num_episodes)
    duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.eval_duration_s)
    policy_text = get_policy_path_value(cmd) or str(req.model_path)

    try:
        effective_episodes = int(episodes_text)
        effective_duration = int(duration_text)
    except ValueError:
        return None, None, None, "Advanced options must keep eval episodes and duration as integers."
    if effective_episodes <= 0 or effective_duration <= 0:
        return None, None, None, "Advanced options must keep eval episodes and duration greater than zero."

    effective_model_path = Path(normalize_path(policy_text))
    if not effective_model_path.is_absolute():
        effective_model_path = models_root / effective_model_path
    if not effective_model_path.exists() or not effective_model_path.is_dir():
        return None, None, None, f"Model folder not found:\n{effective_model_path}"
    is_valid_model, detail, _ = validate_model_path(effective_model_path)
    if not is_valid_model:
        return None, None, None, detail

    effective_req = DeployRequest(
        model_path=effective_model_path,
        eval_repo_id=effective_repo_id,
        eval_num_episodes=effective_episodes,
        eval_duration_s=effective_duration,
        eval_task=effective_task,
    )

    try:
        rel = effective_model_path.relative_to(models_root)
        last_model_name_str = str(rel.parts[0]) if rel.parts else effective_model_path.name
        last_checkpoint_str = str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else ""
    except ValueError:
        last_model_name_str = effective_model_path.name
        last_checkpoint_str = ""

    updated_config["last_model_name"] = last_model_name_str
    updated_config["last_checkpoint_name"] = last_checkpoint_str
    updated_config["eval_num_episodes"] = effective_episodes
    updated_config["eval_duration_s"] = effective_duration
    updated_config["eval_task"] = effective_task
    updated_config["last_eval_dataset_name"] = repo_name_from_repo_id(effective_repo_id)

    return effective_req, cmd, updated_config, None


def build_teleop_request_and_command(
    config: dict[str, Any],
    follower_port_raw: str,
    leader_port_raw: str,
    follower_id_raw: str,
    leader_id_raw: str,
    control_fps_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[TeleopRequest | None, list[str] | None, dict[str, Any] | None, str | None]:
    follower_port = str(follower_port_raw or "").strip()
    leader_port = str(leader_port_raw or "").strip()
    follower_id = str(follower_id_raw or "").strip() or "red4"
    leader_id = str(leader_id_raw or "").strip() or "white"

    if not follower_port:
        return None, None, None, "Follower port is required."
    if not leader_port:
        return None, None, None, "Leader port is required."

    control_fps: int | None = None
    control_fps_text = str(control_fps_raw or "").strip()
    if control_fps_text:
        try:
            control_fps = int(control_fps_text)
        except ValueError:
            return None, None, None, "Teleop control FPS must be an integer."
        if control_fps <= 0:
            return None, None, None, "Teleop control FPS must be greater than zero."

    updated_config = {
        "follower_port": follower_port,
        "leader_port": leader_port,
        "follower_robot_id": follower_id,
        "leader_robot_id": leader_id,
        "teleop_control_fps": control_fps_text,
    }
    run_config = {**config, **updated_config}
    cmd = build_lerobot_teleop_command(
        run_config,
        follower_robot_id=follower_id,
        leader_robot_id=leader_id,
        control_fps=control_fps,
    )
    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, None, override_error or "Unable to apply advanced options."

    effective_control_fps_text = (get_flag_value(cmd, "control.fps") or "").strip()
    effective_control_fps: int | None = None
    if effective_control_fps_text:
        try:
            effective_control_fps = int(effective_control_fps_text)
        except ValueError:
            return None, None, None, "Teleop control FPS must be an integer."
        if effective_control_fps <= 0:
            return None, None, None, "Teleop control FPS must be greater than zero."

    effective_follower_port = str(get_flag_value(cmd, "robot.port") or follower_port).strip()
    effective_leader_port = str(get_flag_value(cmd, "teleop.port") or leader_port).strip()
    effective_follower_id = str(get_flag_value(cmd, "robot.id") or resolve_follower_robot_id(run_config)).strip()
    effective_leader_id = str(get_flag_value(cmd, "teleop.id") or resolve_leader_robot_id(run_config)).strip()
    updated_config = {
        "follower_port": effective_follower_port,
        "leader_port": effective_leader_port,
        "follower_robot_id": effective_follower_id,
        "leader_robot_id": effective_leader_id,
        "teleop_control_fps": effective_control_fps_text,
    }
    req = TeleopRequest(
        follower_port=effective_follower_port,
        leader_port=effective_leader_port,
        follower_id=effective_follower_id,
        leader_id=effective_leader_id,
        control_fps=effective_control_fps,
    )
    return req, cmd, updated_config, None


def build_train_request_and_command(
    form_values: dict[str, Any],
    config: dict[str, Any],
) -> tuple[TrainRequest | None, list[str] | None, str | None]:
    dataset_input = str(form_values.get("dataset_repo_id", "")).strip()
    if not dataset_input:
        return None, None, "Dataset name is required."

    policy_type = str(form_values.get("policy_type", "")).strip()
    if not policy_type:
        return None, None, "Policy type is required."

    default_output_dir = str(config.get("trained_models_dir", "outputs/train")).strip() or "outputs/train"
    output_dir = normalize_path(str(form_values.get("output_dir", "")).strip() or default_output_dir)
    device = str(form_values.get("device", "")).strip()
    wandb_enabled = _parse_form_bool(form_values.get("wandb_enabled"), default=False)
    wandb_project = str(form_values.get("wandb_project", "")).strip()
    job_name = str(form_values.get("job_name", "")).strip()
    resume_from_raw = str(form_values.get("resume_from", "")).strip()
    resume_from = normalize_train_resume_path(resume_from_raw) if resume_from_raw else ""
    custom_args = str(form_values.get("custom_args", "")).strip()
    dataset_episodes = str(form_values.get("dataset_episodes", "")).strip()

    request = TrainRequest(
        dataset_repo_id=_normalize_train_dataset_target(config, dataset_input),
        policy_type=policy_type,
        output_dir=output_dir,
        device=device,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        job_name=job_name,
        resume_from=resume_from,
        custom_args=custom_args,
    )

    if request.resume_from:
        resume_path = Path(request.resume_from)
        if not resume_path.exists():
            return None, None, f"Resume checkpoint/config not found:\n{resume_path}"
        capabilities = probe_lerobot_capabilities(config, include_flag_probe=True)
        if not capabilities.supports_train_resume:
            return None, None, capabilities.train_resume_detail
        resume_flag = str(capabilities.train_resume_path_flag or "").strip().lower()
        if "config" in resume_flag and not resume_path.is_file():
            return (
                None,
                None,
                (
                    f"{capabilities.train_resume_detail} "
                    "Select a train_config.json file or a checkpoint folder containing one."
                ),
            )

    try:
        cmd = build_lerobot_train_command(
            config=config,
            request={
                "dataset_repo_id": request.dataset_repo_id,
                "policy_type": request.policy_type,
                "output_dir": request.output_dir,
                "device": request.device,
                "dataset_episodes": dataset_episodes,
                "wandb_enabled": request.wandb_enabled,
                "wandb_project": request.wandb_project,
                "job_name": request.job_name,
                "resume_from": request.resume_from,
            },
        )
    except ValueError as exc:
        return None, None, str(exc)
    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        custom_args_raw=request.custom_args,
    )
    if override_error is not None or cmd is None:
        return None, None, override_error or "Unable to apply advanced options."
    return request, cmd, None
