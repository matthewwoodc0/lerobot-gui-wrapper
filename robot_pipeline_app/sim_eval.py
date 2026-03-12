from __future__ import annotations

from pathlib import Path
from typing import Any

from .command_overrides import apply_command_overrides, get_flag_value
from .commands import build_lerobot_sim_eval_command
from .compat import probe_lerobot_capabilities
from .config_store import normalize_path
from .deploy_diagnostics import validate_model_path
from .types import SimEvalRequest


def _parse_optional_positive_int(raw_value: Any, *, label: str) -> tuple[int | None, str | None]:
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


def _parse_optional_non_negative_int(raw_value: Any, *, label: str) -> tuple[int | None, str | None]:
    text = str(raw_value or "").strip()
    if not text:
        return None, None
    try:
        value = int(text)
    except ValueError:
        return None, f"{label} must be an integer."
    if value < 0:
        return None, f"{label} must be zero or greater."
    return value, None


def _effective_flag_value(cmd: list[str], *keys: str | None) -> str | None:
    for key in keys:
        if not key:
            continue
        value = get_flag_value(cmd, key)
        if value is not None:
            return value
    return None


def build_sim_eval_request_and_command(
    *,
    form_values: dict[str, Any],
    config: dict[str, Any],
) -> tuple[SimEvalRequest | None, list[str] | None, str | None]:
    model_raw = str(form_values.get("model_path", "")).strip()
    if not model_raw:
        return None, None, "Model/checkpoint path is required."
    model_path = Path(normalize_path(model_raw))
    if not model_path.is_dir():
        return None, None, f"Model/checkpoint folder not found:\n{model_path}"
    is_valid_model, detail, _candidates = validate_model_path(model_path)
    if not is_valid_model:
        return None, None, detail

    env_type = str(form_values.get("env_type", "")).strip()
    benchmark = str(form_values.get("benchmark", "")).strip()
    if not env_type and not benchmark:
        return None, None, "Simulation eval needs an env type or benchmark selection."
    task = str(form_values.get("task", "")).strip()
    output_dir = normalize_path(str(form_values.get("output_dir", "")).strip() or "outputs/eval")
    device = str(form_values.get("device", "")).strip()
    job_name = str(form_values.get("job_name", "")).strip()
    trust_remote_code = bool(form_values.get("trust_remote_code"))
    custom_args = str(form_values.get("custom_args", "")).strip()

    episodes_text = str(form_values.get("episodes", "")).strip()
    try:
        episodes = int(episodes_text)
    except ValueError:
        return None, None, "Simulation eval episodes must be an integer."
    if episodes <= 0:
        return None, None, "Simulation eval episodes must be greater than zero."

    batch_size, batch_error = _parse_optional_positive_int(form_values.get("batch_size"), label="Batch size")
    if batch_error is not None:
        return None, None, batch_error
    seed, seed_error = _parse_optional_non_negative_int(form_values.get("seed"), label="Seed")
    if seed_error is not None:
        return None, None, seed_error

    request = SimEvalRequest(
        model_path=model_path,
        output_dir=output_dir,
        env_type=env_type,
        task=task,
        benchmark=benchmark,
        episodes=episodes,
        batch_size=batch_size,
        seed=seed,
        device=device,
        job_name=job_name,
        trust_remote_code=trust_remote_code,
        custom_args=custom_args,
    )

    try:
        cmd = build_lerobot_sim_eval_command(
            config,
            {
                "model_path": str(request.model_path),
                "output_dir": request.output_dir,
                "env_type": request.env_type,
                "task": request.task,
                "benchmark": request.benchmark,
                "episodes": request.episodes,
                "batch_size": request.batch_size,
                "seed": request.seed,
                "device": request.device,
                "job_name": request.job_name,
                "trust_remote_code": request.trust_remote_code,
            },
        )
    except ValueError as exc:
        return None, None, str(exc)

    cmd, override_error = apply_command_overrides(base_cmd=cmd, custom_args_raw=request.custom_args)
    if override_error is not None or cmd is None:
        return None, None, override_error or "Unable to apply advanced options."

    caps = probe_lerobot_capabilities(config, include_flag_probe=True)
    effective_model_path = _effective_flag_value(cmd, caps.sim_eval_policy_path_flag) or str(request.model_path)
    effective_output_dir = _effective_flag_value(cmd, caps.sim_eval_output_dir_flag) or request.output_dir
    effective_env_type = _effective_flag_value(cmd, caps.sim_eval_env_type_flag) or request.env_type
    effective_task = _effective_flag_value(cmd, caps.sim_eval_task_flag) or request.task
    effective_benchmark = _effective_flag_value(cmd, caps.sim_eval_benchmark_flag) or request.benchmark
    effective_episodes = _effective_flag_value(cmd, caps.sim_eval_episodes_flag) or str(request.episodes)
    effective_batch_size = _effective_flag_value(cmd, caps.sim_eval_batch_size_flag)
    effective_seed = _effective_flag_value(cmd, caps.sim_eval_seed_flag)
    effective_device = _effective_flag_value(cmd, caps.sim_eval_device_flag) or request.device
    effective_job_name = _effective_flag_value(cmd, caps.sim_eval_job_name_flag) or request.job_name

    effective_model = Path(normalize_path(effective_model_path))
    if not effective_model.is_dir():
        return None, None, f"Model/checkpoint folder not found:\n{effective_model}"
    is_valid_effective_model, detail, _candidates = validate_model_path(effective_model)
    if not is_valid_effective_model:
        return None, None, detail

    try:
        effective_episodes_value = int(effective_episodes)
    except ValueError:
        return None, None, "Simulation eval episodes must remain an integer after advanced options."
    if effective_episodes_value <= 0:
        return None, None, "Simulation eval episodes must remain greater than zero after advanced options."

    if not effective_env_type and not effective_benchmark:
        return None, None, "Simulation eval needs an env type or benchmark after advanced options are applied."

    effective_batch_size_value: int | None = None
    if effective_batch_size:
        try:
            effective_batch_size_value = int(str(effective_batch_size).strip())
        except ValueError:
            return None, None, "Batch size must remain an integer after advanced options."
        if effective_batch_size_value <= 0:
            return None, None, "Batch size must remain greater than zero after advanced options."

    effective_seed_value: int | None = None
    if effective_seed:
        try:
            effective_seed_value = int(str(effective_seed).strip())
        except ValueError:
            return None, None, "Seed must remain an integer after advanced options."
        if effective_seed_value < 0:
            return None, None, "Seed must remain zero or greater after advanced options."

    effective_request = SimEvalRequest(
        model_path=effective_model,
        output_dir=effective_output_dir,
        env_type=effective_env_type,
        task=effective_task,
        benchmark=effective_benchmark,
        episodes=effective_episodes_value,
        batch_size=effective_batch_size_value,
        seed=effective_seed_value,
        device=effective_device,
        job_name=effective_job_name,
        trust_remote_code=request.trust_remote_code,
        custom_args=request.custom_args,
    )
    return effective_request, cmd, None
