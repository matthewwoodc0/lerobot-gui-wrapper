from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

from .compat import probe_lerobot_capabilities
from .compat_policy import (
    WORKFLOW_PASS_GATE_NOTE,
    compatibility_policy_display,
    evaluate_python_compatibility,
    match_validated_track,
    validated_tracks_payload,
)
from .config_store import normalize_config_without_prompts
from .feature_flags import compat_probe_enabled
from .lerobot_runtime import detect_runtime_module_version, detect_runtime_python_version


def _python_version_tuple(version_text: str) -> tuple[int, int, int]:
    parsed: list[int] = []
    for chunk in str(version_text or "").strip().split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parsed.append(int(digits) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return parsed[0], parsed[1], parsed[2]


def build_compat_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_config_without_prompts(config)
    lerobot_version = detect_runtime_module_version(normalized, "lerobot")
    python_version = detect_runtime_python_version(normalized)
    python_tuple = _python_version_tuple(python_version)
    python_compatibility = evaluate_python_compatibility(lerobot_version, python_tuple)
    if not compat_probe_enabled(normalized):
        return {
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "compat_policy": compatibility_policy_display(str(normalized.get("compat_policy", "latest_plus_n_minus_1"))),
            "compat_probe_enabled": False,
            "validated_tracks": validated_tracks_payload(),
            "validated_track": match_validated_track(lerobot_version).to_dict() if match_validated_track(lerobot_version) else None,
            "python_requirement": python_compatibility.requirement,
            "python_compatibility_status": python_compatibility.status,
            "python_compatibility_detail": python_compatibility.detail,
            "python_hard_compat_fail": python_compatibility.hard_fail,
            "workflow_pass_gate_note": WORKFLOW_PASS_GATE_NOTE,
            "lerobot_version": lerobot_version,
            "python_version": python_version,
            "platform": platform.platform(),
        }

    capabilities = probe_lerobot_capabilities(normalized, include_flag_probe=True)
    return {
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "compat_policy": compatibility_policy_display(str(normalized.get("compat_policy", "latest_plus_n_minus_1"))),
        "compat_probe_enabled": True,
        "validated_tracks": validated_tracks_payload(),
        "validated_track": match_validated_track(capabilities.lerobot_version).to_dict()
        if match_validated_track(capabilities.lerobot_version)
        else None,
        "workflow_pass_gate_note": WORKFLOW_PASS_GATE_NOTE,
        "record_entrypoint": capabilities.record_entrypoint,
        "train_entrypoint": capabilities.train_entrypoint,
        "sim_eval_entrypoint": capabilities.sim_eval_entrypoint,
        "teleop_entrypoint": capabilities.teleop_entrypoint,
        "calibrate_entrypoint": capabilities.calibrate_entrypoint,
        "record_help_available": capabilities.record_help_available,
        "train_help_available": capabilities.train_help_available,
        "sim_eval_help_available": capabilities.sim_eval_help_available,
        "camera_rename_flag": capabilities.active_rename_flag,
        "supported_record_flags": list(capabilities.supported_record_flags),
        "supported_train_flags": list(capabilities.supported_train_flags),
        "supported_sim_eval_flags": list(capabilities.supported_sim_eval_flags),
        "supports_sim_eval": capabilities.supports_sim_eval,
        "sim_eval_policy_path_flag": capabilities.sim_eval_policy_path_flag,
        "sim_eval_output_dir_flag": capabilities.sim_eval_output_dir_flag,
        "sim_eval_env_type_flag": capabilities.sim_eval_env_type_flag,
        "sim_eval_task_flag": capabilities.sim_eval_task_flag,
        "sim_eval_benchmark_flag": capabilities.sim_eval_benchmark_flag,
        "sim_eval_episodes_flag": capabilities.sim_eval_episodes_flag,
        "sim_eval_batch_size_flag": capabilities.sim_eval_batch_size_flag,
        "sim_eval_seed_flag": capabilities.sim_eval_seed_flag,
        "sim_eval_device_flag": capabilities.sim_eval_device_flag,
        "sim_eval_job_name_flag": capabilities.sim_eval_job_name_flag,
        "sim_eval_support_detail": capabilities.sim_eval_support_detail,
        "missing_train_flags": list(capabilities.missing_train_flags),
        "supports_train_resume": capabilities.supports_train_resume,
        "train_resume_path_flag": capabilities.train_resume_path_flag,
        "train_resume_toggle_flag": capabilities.train_resume_toggle_flag,
        "train_resume_detail": capabilities.train_resume_detail,
        "policy_path_flag": capabilities.policy_path_flag,
        "python_requirement": capabilities.python_requirement,
        "python_compatibility_status": capabilities.python_compatibility_status,
        "python_compatibility_detail": capabilities.python_compatibility_detail,
        "python_hard_compat_fail": capabilities.python_hard_compat_fail,
        "fallback_notes": list(capabilities.fallback_notes),
        "lerobot_version": capabilities.lerobot_version if capabilities.lerobot_version else lerobot_version,
        "python_version": capabilities.python_version or python_version,
        "platform": platform.platform(),
    }
