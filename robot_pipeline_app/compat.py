from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compat_policy import (
    TRAIN_REQUIRED_FLAGS,
    WORKFLOW_PASS_GATE_NOTE,
    compatibility_policy_display,
    evaluate_python_compatibility,
    match_validated_track,
    validated_tracks_payload,
    validated_tracks_summary,
)
from .config_store import normalize_path
from .lerobot_runtime import (
    build_lerobot_module_command,
    configured_lerobot_dir as runtime_configured_lerobot_dir,
    detect_runtime_module_version,
    detect_runtime_python_version,
    lerobot_runtime_cwd,
    runtime_module_available,
    runtime_signature,
)


_FLAG_PATTERN = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_.-]*)")
_CAP_CACHE: dict[tuple[str, ...], "LeRobotCapabilities"] = {}
_TRAIN_RESUME_PATH_FLAG_CANDIDATES: tuple[str, ...] = (
    "config_path",
    "config.path",
    "resume_path",
    "resume.path",
    "checkpoint_path",
    "checkpoint.path",
    "resume_from",
    "resume.from",
)
_TRAIN_RESUME_CONFIG_HINTS: tuple[str, ...] = ("config", "json", "file")
_SIM_EVAL_POLICY_PATH_FLAG_CANDIDATES: tuple[str, ...] = (
    "policy.path",
    "policy.pretrained_path",
    "policy.pretrained_model_path",
    "policy.model_path",
    "model_path",
    "checkpoint_path",
)
_SIM_EVAL_OUTPUT_DIR_FLAG_CANDIDATES: tuple[str, ...] = ("output_dir", "policy.output_dir")
_SIM_EVAL_ENV_TYPE_FLAG_CANDIDATES: tuple[str, ...] = ("env.type", "env", "env.name", "env.environment")
_SIM_EVAL_TASK_FLAG_CANDIDATES: tuple[str, ...] = ("env.task", "task", "env.task_id")
_SIM_EVAL_BENCHMARK_FLAG_CANDIDATES: tuple[str, ...] = ("benchmark", "env.benchmark", "benchmark.name")
_SIM_EVAL_EPISODES_FLAG_CANDIDATES: tuple[str, ...] = ("eval.n_episodes", "num_episodes", "episodes", "eval.episodes")
_SIM_EVAL_BATCH_SIZE_FLAG_CANDIDATES: tuple[str, ...] = ("eval.batch_size", "batch_size", "env.num_envs")
_SIM_EVAL_SEED_FLAG_CANDIDATES: tuple[str, ...] = ("seed", "env.seed")
_SIM_EVAL_DEVICE_FLAG_CANDIDATES: tuple[str, ...] = ("policy.device", "env.device", "device")
_SIM_EVAL_JOB_NAME_FLAG_CANDIDATES: tuple[str, ...] = ("job_name",)


@dataclass(frozen=True)
class LeRobotCapabilities:
    detected_at_iso: str
    lerobot_version: str
    python_version: str
    record_entrypoint: str
    train_entrypoint: str
    sim_eval_entrypoint: str
    teleop_entrypoint: str
    teleop_uses_legacy_control: bool
    calibrate_entrypoint: str
    record_help_available: bool
    record_help_error: str
    supported_record_flags: tuple[str, ...]
    train_help_available: bool
    train_help_error: str
    supported_train_flags: tuple[str, ...]
    sim_eval_help_available: bool
    sim_eval_help_error: str
    supported_sim_eval_flags: tuple[str, ...]
    supports_sim_eval: bool
    sim_eval_policy_path_flag: str | None
    sim_eval_output_dir_flag: str | None
    sim_eval_env_type_flag: str | None
    sim_eval_task_flag: str | None
    sim_eval_benchmark_flag: str | None
    sim_eval_episodes_flag: str | None
    sim_eval_batch_size_flag: str | None
    sim_eval_seed_flag: str | None
    sim_eval_device_flag: str | None
    sim_eval_job_name_flag: str | None
    sim_eval_support_detail: str
    missing_train_flags: tuple[str, ...]
    supports_train_resume: bool
    train_resume_path_flag: str | None
    train_resume_toggle_flag: str | None
    train_resume_detail: str
    supports_policy_path: bool
    policy_path_flag: str | None
    supported_rename_flags: tuple[str, ...]
    active_rename_flag: str
    python_requirement: str
    python_compatibility_status: str
    python_compatibility_detail: str
    python_hard_compat_fail: bool
    fallback_notes: tuple[str, ...]
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        validated_track = match_validated_track(self.lerobot_version)
        return {
            "detected_at_iso": self.detected_at_iso,
            "lerobot_version": self.lerobot_version,
            "python_version": self.python_version,
            "record_entrypoint": self.record_entrypoint,
            "train_entrypoint": self.train_entrypoint,
            "sim_eval_entrypoint": self.sim_eval_entrypoint,
            "teleop_entrypoint": self.teleop_entrypoint,
            "teleop_uses_legacy_control": self.teleop_uses_legacy_control,
            "calibrate_entrypoint": self.calibrate_entrypoint,
            "record_help_available": self.record_help_available,
            "record_help_error": self.record_help_error,
            "supported_record_flags": list(self.supported_record_flags),
            "train_help_available": self.train_help_available,
            "train_help_error": self.train_help_error,
            "supported_train_flags": list(self.supported_train_flags),
            "sim_eval_help_available": self.sim_eval_help_available,
            "sim_eval_help_error": self.sim_eval_help_error,
            "supported_sim_eval_flags": list(self.supported_sim_eval_flags),
            "supports_sim_eval": self.supports_sim_eval,
            "sim_eval_policy_path_flag": self.sim_eval_policy_path_flag,
            "sim_eval_output_dir_flag": self.sim_eval_output_dir_flag,
            "sim_eval_env_type_flag": self.sim_eval_env_type_flag,
            "sim_eval_task_flag": self.sim_eval_task_flag,
            "sim_eval_benchmark_flag": self.sim_eval_benchmark_flag,
            "sim_eval_episodes_flag": self.sim_eval_episodes_flag,
            "sim_eval_batch_size_flag": self.sim_eval_batch_size_flag,
            "sim_eval_seed_flag": self.sim_eval_seed_flag,
            "sim_eval_device_flag": self.sim_eval_device_flag,
            "sim_eval_job_name_flag": self.sim_eval_job_name_flag,
            "sim_eval_support_detail": self.sim_eval_support_detail,
            "missing_train_flags": list(self.missing_train_flags),
            "supports_train_resume": self.supports_train_resume,
            "train_resume_path_flag": self.train_resume_path_flag,
            "train_resume_toggle_flag": self.train_resume_toggle_flag,
            "train_resume_detail": self.train_resume_detail,
            "supports_policy_path": self.supports_policy_path,
            "policy_path_flag": self.policy_path_flag,
            "supported_rename_flags": list(self.supported_rename_flags),
            "active_rename_flag": self.active_rename_flag,
            "validated_track": validated_track.to_dict() if validated_track is not None else None,
            "validated_tracks": validated_tracks_payload(),
            "python_requirement": self.python_requirement,
            "python_compatibility_status": self.python_compatibility_status,
            "python_compatibility_detail": self.python_compatibility_detail,
            "python_hard_compat_fail": self.python_hard_compat_fail,
            "workflow_pass_gate_note": WORKFLOW_PASS_GATE_NOTE,
            "fallback_notes": list(self.fallback_notes),
            "cache_hit": self.cache_hit,
        }


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _configured_lerobot_dir(config: dict[str, Any]) -> Path | None:
    return runtime_configured_lerobot_dir(config)


def _lerobot_module_available(config: dict[str, Any], module_name: str) -> bool:
    return runtime_module_available(config, module_name)


def _parse_python_version_tuple(version_text: str) -> tuple[int, int, int]:
    parts = str(version_text or "").strip().split(".")
    parsed: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            parsed.append(0)
            continue
        parsed.append(int(digits))
    while len(parsed) < 3:
        parsed.append(0)
    return parsed[0], parsed[1], parsed[2]


def normalize_train_resume_path(raw_value: str) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""

    candidate = Path(normalize_path(raw))
    if candidate.is_dir():
        for nested in (
            candidate / "train_config.json",
            candidate / "pretrained_model" / "train_config.json",
        ):
            if nested.is_file():
                return str(nested)
    return str(candidate)


def _train_resume_flag_requires_file(path_flag: str | None) -> bool:
    normalized = str(path_flag or "").strip().lower()
    return any(hint in normalized for hint in _TRAIN_RESUME_CONFIG_HINTS)


def _resolve_checkout_module(
    lerobot_dir: Path | None,
    candidates: tuple[tuple[str, str], ...],
) -> str | None:
    if lerobot_dir is None:
        return None
    for relative_path, module_name in candidates:
        if (lerobot_dir / relative_path).exists():
            return module_name
    return None


def resolve_record_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_record_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "record.py").exists():
            return "lerobot.record"
        if (lerobot_dir / "scripts" / "record.py").exists():
            return "scripts.record"
        if (lerobot_dir / "lerobot" / "scripts" / "record.py").exists():
            return "lerobot.scripts.record"
        if (lerobot_dir / "scripts" / "lerobot_record.py").exists():
            return "scripts.lerobot_record"
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_record.py").exists():
            return "lerobot.scripts.lerobot_record"

    for module_name in (
        "lerobot.record",
        "lerobot.scripts.record",
        "lerobot.scripts.lerobot_record",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.scripts.lerobot_record"


def resolve_train_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_train_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "train.py").exists():
            return "lerobot.train"
        if (lerobot_dir / "scripts" / "train.py").exists():
            return "scripts.train"
        if (lerobot_dir / "lerobot" / "scripts" / "train.py").exists():
            return "lerobot.scripts.train"
        if (lerobot_dir / "scripts" / "lerobot_train.py").exists():
            return "scripts.lerobot_train"
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_train.py").exists():
            return "lerobot.scripts.lerobot_train"

    for module_name in (
        "lerobot.train",
        "lerobot.scripts.train",
        "lerobot.scripts.lerobot_train",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.scripts.lerobot_train"


def resolve_sim_eval_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_sim_eval_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    resolved = _resolve_checkout_module(
        lerobot_dir,
        (
            ("src/lerobot/scripts/lerobot_eval.py", "lerobot.scripts.lerobot_eval"),
            ("src/lerobot/scripts/eval.py", "lerobot.scripts.eval"),
            ("src/lerobot/eval.py", "lerobot.eval"),
            ("lerobot/scripts/lerobot_eval.py", "lerobot.scripts.lerobot_eval"),
            ("lerobot/scripts/eval.py", "lerobot.scripts.eval"),
            ("lerobot/eval.py", "lerobot.eval"),
            ("scripts/lerobot_eval.py", "scripts.lerobot_eval"),
            ("scripts/eval.py", "scripts.eval"),
        ),
    )
    if resolved:
        return resolved

    for module_name in (
        "lerobot.scripts.lerobot_eval",
        "lerobot.scripts.eval",
        "lerobot.eval",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.scripts.lerobot_eval"


def resolve_edit_dataset_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_edit_dataset_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    resolved = _resolve_checkout_module(
        lerobot_dir,
        (
            ("src/lerobot/edit_dataset.py", "lerobot.edit_dataset"),
            ("src/lerobot/scripts/edit_dataset.py", "lerobot.scripts.edit_dataset"),
            ("src/lerobot/scripts/lerobot_edit_dataset.py", "lerobot.scripts.lerobot_edit_dataset"),
            ("lerobot/edit_dataset.py", "lerobot.edit_dataset"),
            ("scripts/edit_dataset.py", "scripts.edit_dataset"),
            ("lerobot/scripts/edit_dataset.py", "lerobot.scripts.edit_dataset"),
            ("scripts/lerobot_edit_dataset.py", "scripts.lerobot_edit_dataset"),
            ("lerobot/scripts/lerobot_edit_dataset.py", "lerobot.scripts.lerobot_edit_dataset"),
        ),
    )
    if resolved:
        return resolved

    for module_name in (
        "lerobot.edit_dataset",
        "lerobot.scripts.edit_dataset",
        "lerobot.scripts.lerobot_edit_dataset",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.edit_dataset"


def resolve_visualize_dataset_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_visualize_dataset_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    resolved = _resolve_checkout_module(
        lerobot_dir,
        (
            ("src/lerobot/visualize_dataset.py", "lerobot.visualize_dataset"),
            ("src/lerobot/scripts/visualize_dataset.py", "lerobot.scripts.visualize_dataset"),
            ("src/lerobot/scripts/lerobot_dataset_viz.py", "lerobot.scripts.lerobot_dataset_viz"),
            ("lerobot/visualize_dataset.py", "lerobot.visualize_dataset"),
            ("scripts/visualize_dataset.py", "scripts.visualize_dataset"),
            ("lerobot/scripts/visualize_dataset.py", "lerobot.scripts.visualize_dataset"),
            ("scripts/lerobot_dataset_viz.py", "scripts.lerobot_dataset_viz"),
            ("lerobot/scripts/lerobot_dataset_viz.py", "lerobot.scripts.lerobot_dataset_viz"),
        ),
    )
    if resolved:
        return resolved

    for module_name in (
        "lerobot.visualize_dataset",
        "lerobot.scripts.visualize_dataset",
        "lerobot.scripts.lerobot_dataset_viz",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.scripts.visualize_dataset"


def resolve_replay_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_replay_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    resolved = _resolve_checkout_module(
        lerobot_dir,
        (
            ("src/lerobot/replay.py", "lerobot.replay"),
            ("src/lerobot/scripts/replay.py", "lerobot.scripts.replay"),
            ("src/lerobot/scripts/lerobot_replay.py", "lerobot.scripts.lerobot_replay"),
            ("lerobot/replay.py", "lerobot.replay"),
            ("scripts/replay.py", "scripts.replay"),
            ("lerobot/scripts/replay.py", "lerobot.scripts.replay"),
            ("scripts/lerobot_replay.py", "scripts.lerobot_replay"),
            ("lerobot/scripts/lerobot_replay.py", "lerobot.scripts.lerobot_replay"),
        ),
    )
    if resolved:
        return resolved

    for module_name in (
        "lerobot.replay",
        "lerobot.scripts.replay",
        "lerobot.scripts.lerobot_replay",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name
    return ""


def resolve_calibrate_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_calibrate_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "calibrate.py").exists():
            return "lerobot.calibrate"
        if (lerobot_dir / "scripts" / "calibrate.py").exists():
            return "scripts.calibrate"
        if (lerobot_dir / "lerobot" / "scripts" / "calibrate.py").exists():
            return "lerobot.scripts.calibrate"

    for module_name in ("lerobot.calibrate", "lerobot.scripts.calibrate"):
        if _lerobot_module_available(config, module_name):
            return module_name

    return "lerobot.calibrate"


def resolve_motor_setup_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_motor_setup_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    resolved = _resolve_checkout_module(
        lerobot_dir,
        (
            ("src/lerobot/setup_motors.py", "lerobot.setup_motors"),
            ("src/lerobot/motor_setup.py", "lerobot.motor_setup"),
            ("src/lerobot/configure_motors.py", "lerobot.configure_motors"),
            ("src/lerobot/scripts/setup_motors.py", "lerobot.scripts.setup_motors"),
            ("src/lerobot/scripts/lerobot_setup_motors.py", "lerobot.scripts.lerobot_setup_motors"),
            ("lerobot/setup_motors.py", "lerobot.setup_motors"),
            ("lerobot/motor_setup.py", "lerobot.motor_setup"),
            ("lerobot/configure_motors.py", "lerobot.configure_motors"),
            ("scripts/setup_motors.py", "scripts.setup_motors"),
            ("lerobot/scripts/setup_motors.py", "lerobot.scripts.setup_motors"),
            ("scripts/lerobot_setup_motors.py", "scripts.lerobot_setup_motors"),
            ("lerobot/scripts/lerobot_setup_motors.py", "lerobot.scripts.lerobot_setup_motors"),
        ),
    )
    if resolved:
        return resolved

    for module_name in (
        "lerobot.setup_motors",
        "lerobot.motor_setup",
        "lerobot.configure_motors",
        "lerobot.scripts.setup_motors",
        "lerobot.scripts.lerobot_setup_motors",
    ):
        if _lerobot_module_available(config, module_name):
            return module_name
    return ""


def _resolve_legacy_teleop_entrypoint(config: dict[str, Any], lerobot_dir: Path | None) -> tuple[str, bool] | None:
    if _lerobot_module_available(config, "lerobot.scripts.control_robot"):
        return "lerobot.scripts.control_robot", True

    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "scripts" / "control_robot.py").exists():
            return "lerobot.scripts.control_robot", True
        if (lerobot_dir / "scripts" / "control_robot.py").exists():
            return "scripts.control_robot", True
    return None


def resolve_teleop_entrypoint(config: dict[str, Any]) -> tuple[str, bool]:
    prefer_non_av1_path = sys.platform == "darwin" and _parse_bool(
        config.get("teleop_av1_fallback", sys.platform == "darwin"),
        sys.platform == "darwin",
    )
    lerobot_dir = _configured_lerobot_dir(config)

    if prefer_non_av1_path:
        legacy = _resolve_legacy_teleop_entrypoint(config, lerobot_dir)
        if legacy is not None:
            return legacy

    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "teleoperate.py").exists():
            return "lerobot.teleoperate", False
        if (lerobot_dir / "scripts" / "teleoperate.py").exists():
            return "scripts.teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "teleoperate.py").exists():
            return "lerobot.scripts.teleoperate", False
        if (lerobot_dir / "scripts" / "lerobot_teleoperate.py").exists():
            return "scripts.lerobot_teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_teleoperate.py").exists():
            return "lerobot.scripts.lerobot_teleoperate", False

    if _lerobot_module_available(config, "lerobot.teleoperate"):
        return "lerobot.teleoperate", False
    if _lerobot_module_available(config, "lerobot.scripts.teleoperate"):
        return "lerobot.scripts.teleoperate", False
    if _lerobot_module_available(config, "lerobot.scripts.lerobot_teleoperate"):
        return "lerobot.scripts.lerobot_teleoperate", False

    legacy = _resolve_legacy_teleop_entrypoint(config, lerobot_dir)
    if legacy is not None:
        return legacy

    return "lerobot.teleoperate", False


def _parse_help_flags(text: str) -> set[str]:
    return {match.group(1).strip() for match in _FLAG_PATTERN.finditer(text) if match.group(1).strip()}


def _probe_help_flags(config: dict[str, Any], module_entrypoint: str) -> tuple[set[str], str]:
    cwd = lerobot_runtime_cwd(config)
    variants = (
        [*build_lerobot_module_command(config, module_entrypoint), "--help"],
        [*build_lerobot_module_command(config, module_entrypoint), "-h"],
    )

    errors: list[str] = []
    for cmd in variants:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=12,
                cwd=str(cwd) if cwd is not None else None,
            )
        except subprocess.TimeoutExpired:
            errors.append("help probe timed out")
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue

        text = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if not text:
            errors.append("empty help output")
            continue

        flags = _parse_help_flags(text)
        if flags:
            return flags, ""
        errors.append("no flags parsed from help output")

    return set(), "; ".join(errors)


def _probe_record_help_flags(config: dict[str, Any], record_entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, record_entrypoint)


def probe_entrypoint_help_flags(config: dict[str, Any], entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, entrypoint)


def _probe_train_help_flags(config: dict[str, Any], train_entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, train_entrypoint)


def _probe_sim_eval_help_flags(config: dict[str, Any], sim_eval_entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, sim_eval_entrypoint)


def _missing_required_train_flags(flags: set[str]) -> list[str]:
    return [flag for flag in TRAIN_REQUIRED_FLAGS if flag not in flags]


def _choose_policy_path_flag(flags: set[str]) -> str | None:
    if "policy.path" in flags:
        return "policy.path"
    if "policy" in flags:
        return "policy"
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if "policy" in normalized and "path" in normalized:
            return candidate
    return None


def _choose_train_resume_path_flag(flags: set[str]) -> str | None:
    for candidate in _TRAIN_RESUME_PATH_FLAG_CANDIDATES:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if "resume" in normalized and "path" in normalized:
            return candidate
        if "checkpoint" in normalized and "path" in normalized:
            return candidate
        if "config" in normalized and "path" in normalized:
            return candidate
    return None


def _choose_train_resume_toggle_flag(flags: set[str]) -> str | None:
    if "resume" in flags:
        return "resume"
    for candidate in sorted(flags):
        if candidate.lower() == "resume":
            return candidate
    return None


def _choose_flag(flags: set[str], candidates: tuple[str, ...], *, keywords: tuple[str, ...] = ()) -> str | None:
    for candidate in candidates:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if keywords and all(keyword in normalized for keyword in keywords):
            return candidate
    return None


def _choose_sim_eval_policy_path_flag(flags: set[str]) -> str | None:
    for candidate in _SIM_EVAL_POLICY_PATH_FLAG_CANDIDATES:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if "policy" in normalized and "path" in normalized:
            return candidate
        if "checkpoint" in normalized and "path" in normalized:
            return candidate
        if "model" in normalized and "path" in normalized:
            return candidate
    return None


def _choose_sim_eval_output_dir_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_OUTPUT_DIR_FLAG_CANDIDATES, keywords=("output", "dir"))


def _choose_sim_eval_env_type_flag(flags: set[str]) -> str | None:
    for candidate in _SIM_EVAL_ENV_TYPE_FLAG_CANDIDATES:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if "env" in normalized and any(token in normalized for token in ("type", "name", "environment")):
            return candidate
    return None


def _choose_sim_eval_task_flag(flags: set[str]) -> str | None:
    for candidate in _SIM_EVAL_TASK_FLAG_CANDIDATES:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if "task" in normalized and "tasks" not in normalized:
            return candidate
    return None


def _choose_sim_eval_benchmark_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_BENCHMARK_FLAG_CANDIDATES, keywords=("benchmark",))


def _choose_sim_eval_episodes_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_EPISODES_FLAG_CANDIDATES, keywords=("episode",))


def _choose_sim_eval_batch_size_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_BATCH_SIZE_FLAG_CANDIDATES, keywords=("batch",))


def _choose_sim_eval_seed_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_SEED_FLAG_CANDIDATES, keywords=("seed",))


def _choose_sim_eval_device_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_DEVICE_FLAG_CANDIDATES, keywords=("device",))


def _choose_sim_eval_job_name_flag(flags: set[str]) -> str | None:
    return _choose_flag(flags, _SIM_EVAL_JOB_NAME_FLAG_CANDIDATES, keywords=("job", "name"))


def _normalize_flag_name(raw: str) -> str:
    return str(raw or "").strip().lstrip("-")


def _rename_flag_candidates(config: dict[str, Any]) -> list[str]:
    configured = _normalize_flag_name(str(config.get("camera_rename_flag", "rename_map")))
    candidates = [
        configured or "rename_map",
        "dataset.rename_map",
        "rename_map",
        "dataset.image_features_to_rename",
        "image_features_to_rename",
        "observation.rename_map",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _detect_lerobot_version(config: dict[str, Any]) -> str:
    return detect_runtime_module_version(config, "lerobot")


def _cache_key(
    config: dict[str, Any],
    include_flag_probe: bool,
    *,
    lerobot_version: str | None = None,
) -> tuple[str, ...]:
    python_executable, cwd = runtime_signature(config)
    version = str(lerobot_version or _detect_lerobot_version(config))
    return (
        python_executable,
        cwd,
        str(sys.platform),
        version,
        str(config.get("lerobot_venv_dir", "")),
        str(config.get("lerobot_dir", "")),
        str(config.get("lerobot_record_entrypoint", "")),
        str(config.get("lerobot_train_entrypoint", "")),
        str(config.get("lerobot_calibrate_entrypoint", "")),
        str(config.get("camera_rename_flag", "")),
        str(config.get("teleop_av1_fallback", "")),
        "probe_flags" if include_flag_probe else "entrypoints_only",
    )


def get_cached_lerobot_capabilities(
    config: dict[str, Any],
    *,
    include_flag_probe: bool = False,
) -> LeRobotCapabilities | None:
    exact = _CAP_CACHE.get(_cache_key(config, include_flag_probe))
    if exact is not None:
        return LeRobotCapabilities(**{**exact.__dict__, "cache_hit": True})
    if include_flag_probe:
        return None
    probed = _CAP_CACHE.get(_cache_key(config, True))
    if probed is not None:
        return LeRobotCapabilities(**{**probed.__dict__, "cache_hit": True})
    return None


def probe_lerobot_capabilities(
    config: dict[str, Any],
    *,
    include_flag_probe: bool = True,
    force_refresh: bool = False,
) -> LeRobotCapabilities:
    lerobot_version = _detect_lerobot_version(config)
    python_version = detect_runtime_python_version(config)
    python_compatibility = evaluate_python_compatibility(
        lerobot_version,
        _parse_python_version_tuple(python_version),
    )
    key = _cache_key(config, include_flag_probe, lerobot_version=lerobot_version)
    if not force_refresh and key in _CAP_CACHE:
        cached = _CAP_CACHE[key]
        return LeRobotCapabilities(**{**cached.__dict__, "cache_hit": True})

    record_entrypoint = resolve_record_entrypoint(config)
    train_entrypoint = resolve_train_entrypoint(config)
    sim_eval_entrypoint = resolve_sim_eval_entrypoint(config)
    teleop_entrypoint, teleop_uses_legacy = resolve_teleop_entrypoint(config)
    calibrate_entrypoint = resolve_calibrate_entrypoint(config)
    configured_rename = _normalize_flag_name(str(config.get("camera_rename_flag", "rename_map"))) or "rename_map"

    fallback_notes: list[str] = []
    record_help_available = False
    record_help_error = ""
    supported_flags: set[str] = set()
    train_help_available = False
    train_help_error = ""
    supported_train_flags: set[str] = set()
    sim_eval_help_available = False
    sim_eval_help_error = ""
    supported_sim_eval_flags: set[str] = set()
    sim_eval_policy_path_flag: str | None = None
    sim_eval_output_dir_flag: str | None = None
    sim_eval_env_type_flag: str | None = None
    sim_eval_task_flag: str | None = None
    sim_eval_benchmark_flag: str | None = None
    sim_eval_episodes_flag: str | None = None
    sim_eval_batch_size_flag: str | None = None
    sim_eval_seed_flag: str | None = None
    sim_eval_device_flag: str | None = None
    sim_eval_job_name_flag: str | None = None
    sim_eval_support_detail = (
        f"Could not confirm simulation eval support for {sim_eval_entrypoint}; help output was not probed."
    )
    missing_train_flags: list[str] = []
    train_resume_path_flag: str | None = None
    train_resume_toggle_flag: str | None = None
    train_resume_detail = (
        f"Could not confirm checkpoint-path resume support for {train_entrypoint}; help output was not probed."
    )
    policy_path_flag: str | None = "policy.path"

    if include_flag_probe:
        supported_flags, record_help_error = _probe_record_help_flags(config, record_entrypoint)
        record_help_available = bool(supported_flags)
        policy_path_flag = _choose_policy_path_flag(supported_flags)
        if not record_help_available and record_help_error:
            fallback_notes.append(
                f"Unable to probe --help flags for {record_entrypoint}; using compatibility defaults."
            )
        supported_train_flags, train_help_error = _probe_train_help_flags(config, train_entrypoint)
        train_help_available = bool(supported_train_flags)
        if not train_help_available and train_help_error:
            fallback_notes.append(
                f"Unable to probe --help flags for {train_entrypoint}; train compatibility remains unverified."
            )
        supported_sim_eval_flags, sim_eval_help_error = _probe_sim_eval_help_flags(config, sim_eval_entrypoint)
        sim_eval_help_available = bool(supported_sim_eval_flags)
        if not sim_eval_help_available and sim_eval_help_error:
            fallback_notes.append(
                f"Unable to probe --help flags for {sim_eval_entrypoint}; sim eval compatibility remains unverified."
            )
    else:
        cached_probe_key = _cache_key(config, True, lerobot_version=lerobot_version)
        cached_probe = _CAP_CACHE.get(cached_probe_key)
        if cached_probe is not None and cached_probe.supported_record_flags:
            supported_flags = set(cached_probe.supported_record_flags)
            record_help_available = cached_probe.record_help_available
            record_help_error = cached_probe.record_help_error
            policy_path_flag = cached_probe.policy_path_flag
            supported_train_flags = set(cached_probe.supported_train_flags)
            train_help_available = cached_probe.train_help_available
            train_help_error = cached_probe.train_help_error
            supported_sim_eval_flags = set(cached_probe.supported_sim_eval_flags)
            sim_eval_help_available = cached_probe.sim_eval_help_available
            sim_eval_help_error = cached_probe.sim_eval_help_error
            sim_eval_policy_path_flag = cached_probe.sim_eval_policy_path_flag
            sim_eval_output_dir_flag = cached_probe.sim_eval_output_dir_flag
            sim_eval_env_type_flag = cached_probe.sim_eval_env_type_flag
            sim_eval_task_flag = cached_probe.sim_eval_task_flag
            sim_eval_benchmark_flag = cached_probe.sim_eval_benchmark_flag
            sim_eval_episodes_flag = cached_probe.sim_eval_episodes_flag
            sim_eval_batch_size_flag = cached_probe.sim_eval_batch_size_flag
            sim_eval_seed_flag = cached_probe.sim_eval_seed_flag
            sim_eval_device_flag = cached_probe.sim_eval_device_flag
            sim_eval_job_name_flag = cached_probe.sim_eval_job_name_flag
            sim_eval_support_detail = cached_probe.sim_eval_support_detail
            train_resume_path_flag = cached_probe.train_resume_path_flag
            train_resume_toggle_flag = cached_probe.train_resume_toggle_flag
            train_resume_detail = cached_probe.train_resume_detail

    rename_candidates = _rename_flag_candidates(config)
    supported_rename_flags: list[str] = []
    if supported_flags:
        supported_rename_flags = [flag for flag in rename_candidates if flag in supported_flags]
        if not supported_rename_flags:
            discovered = sorted(
                flag
                for flag in supported_flags
                if "rename" in flag.lower() and ("map" in flag.lower() or "feature" in flag.lower())
            )
            supported_rename_flags.extend(discovered)

    active_rename_flag = configured_rename
    if supported_rename_flags:
        active_rename_flag = supported_rename_flags[0]
    if active_rename_flag != configured_rename:
        fallback_notes.append(
            f"Configured rename flag '--{configured_rename}' unsupported; using '--{active_rename_flag}'."
        )

    supports_policy_path = bool(policy_path_flag)
    if policy_path_flag and policy_path_flag != "policy.path":
        fallback_notes.append(
            f"'--policy.path' unsupported; using '--{policy_path_flag}' for policy selection."
        )
    if not policy_path_flag and include_flag_probe:
        fallback_notes.append("No policy-path style flag detected in record entrypoint help output.")

    if train_help_available:
        missing_train_flags = _missing_required_train_flags(supported_train_flags)
        if missing_train_flags:
            fallback_notes.append(
                "Missing required train flags in help output: "
                + ", ".join(f"--{flag}" for flag in missing_train_flags)
            )

    if supported_train_flags:
        train_resume_path_flag = train_resume_path_flag or _choose_train_resume_path_flag(supported_train_flags)
        train_resume_toggle_flag = train_resume_toggle_flag or _choose_train_resume_toggle_flag(supported_train_flags)

    if supported_sim_eval_flags:
        sim_eval_policy_path_flag = sim_eval_policy_path_flag or _choose_sim_eval_policy_path_flag(supported_sim_eval_flags)
        sim_eval_output_dir_flag = sim_eval_output_dir_flag or _choose_sim_eval_output_dir_flag(supported_sim_eval_flags)
        sim_eval_env_type_flag = sim_eval_env_type_flag or _choose_sim_eval_env_type_flag(supported_sim_eval_flags)
        sim_eval_task_flag = sim_eval_task_flag or _choose_sim_eval_task_flag(supported_sim_eval_flags)
        sim_eval_benchmark_flag = sim_eval_benchmark_flag or _choose_sim_eval_benchmark_flag(supported_sim_eval_flags)
        sim_eval_episodes_flag = sim_eval_episodes_flag or _choose_sim_eval_episodes_flag(supported_sim_eval_flags)
        sim_eval_batch_size_flag = sim_eval_batch_size_flag or _choose_sim_eval_batch_size_flag(supported_sim_eval_flags)
        sim_eval_seed_flag = sim_eval_seed_flag or _choose_sim_eval_seed_flag(supported_sim_eval_flags)
        sim_eval_device_flag = sim_eval_device_flag or _choose_sim_eval_device_flag(supported_sim_eval_flags)
        sim_eval_job_name_flag = sim_eval_job_name_flag or _choose_sim_eval_job_name_flag(supported_sim_eval_flags)

    supports_train_resume = bool(train_resume_path_flag)
    if train_help_available:
        if supports_train_resume:
            if train_resume_toggle_flag:
                train_resume_detail = (
                    f"Checkpoint resume is supported via --{train_resume_toggle_flag} and --{train_resume_path_flag}."
                )
            else:
                train_resume_detail = f"Checkpoint resume is supported via --{train_resume_path_flag}."
        else:
            train_resume_detail = (
                f"{train_entrypoint} help output did not expose a checkpoint/config-path resume flag."
            )
            fallback_notes.append(
                f"Checkpoint-path training resume is unavailable for {train_entrypoint} in the detected runtime."
            )
    elif train_help_error:
        train_resume_detail = (
            f"Could not confirm checkpoint-path resume support for {train_entrypoint}: {train_help_error}"
        )

    supports_sim_eval = bool(sim_eval_policy_path_flag and (sim_eval_env_type_flag or sim_eval_benchmark_flag))
    if sim_eval_help_available:
        if supports_sim_eval:
            supported_parts = [f"--{sim_eval_policy_path_flag}"]
            if sim_eval_env_type_flag:
                supported_parts.append(f"--{sim_eval_env_type_flag}")
            if sim_eval_benchmark_flag:
                supported_parts.append(f"--{sim_eval_benchmark_flag}")
            if sim_eval_task_flag:
                supported_parts.append(f"--{sim_eval_task_flag}")
            sim_eval_support_detail = "Simulation eval is supported via " + ", ".join(supported_parts) + "."
        else:
            sim_eval_support_detail = (
                f"{sim_eval_entrypoint} help output did not expose the policy/environment flags required for GUI sim eval."
            )
            fallback_notes.append(
                f"Simulation eval is unavailable for {sim_eval_entrypoint} in the detected runtime."
            )
    elif sim_eval_help_error:
        sim_eval_support_detail = f"Could not confirm simulation eval support for {sim_eval_entrypoint}: {sim_eval_help_error}"

    capabilities = LeRobotCapabilities(
        detected_at_iso=datetime.now(timezone.utc).isoformat(),
        lerobot_version=lerobot_version,
        python_version=python_version,
        record_entrypoint=record_entrypoint,
        train_entrypoint=train_entrypoint,
        sim_eval_entrypoint=sim_eval_entrypoint,
        teleop_entrypoint=teleop_entrypoint,
        teleop_uses_legacy_control=teleop_uses_legacy,
        calibrate_entrypoint=calibrate_entrypoint,
        record_help_available=record_help_available,
        record_help_error=record_help_error,
        supported_record_flags=tuple(sorted(supported_flags)),
        train_help_available=train_help_available,
        train_help_error=train_help_error,
        supported_train_flags=tuple(sorted(supported_train_flags)),
        sim_eval_help_available=sim_eval_help_available,
        sim_eval_help_error=sim_eval_help_error,
        supported_sim_eval_flags=tuple(sorted(supported_sim_eval_flags)),
        supports_sim_eval=supports_sim_eval,
        sim_eval_policy_path_flag=sim_eval_policy_path_flag,
        sim_eval_output_dir_flag=sim_eval_output_dir_flag,
        sim_eval_env_type_flag=sim_eval_env_type_flag,
        sim_eval_task_flag=sim_eval_task_flag,
        sim_eval_benchmark_flag=sim_eval_benchmark_flag,
        sim_eval_episodes_flag=sim_eval_episodes_flag,
        sim_eval_batch_size_flag=sim_eval_batch_size_flag,
        sim_eval_seed_flag=sim_eval_seed_flag,
        sim_eval_device_flag=sim_eval_device_flag,
        sim_eval_job_name_flag=sim_eval_job_name_flag,
        sim_eval_support_detail=sim_eval_support_detail,
        missing_train_flags=tuple(missing_train_flags),
        supports_train_resume=supports_train_resume,
        train_resume_path_flag=train_resume_path_flag,
        train_resume_toggle_flag=train_resume_toggle_flag,
        train_resume_detail=train_resume_detail,
        supports_policy_path=supports_policy_path,
        policy_path_flag=policy_path_flag,
        supported_rename_flags=tuple(supported_rename_flags),
        active_rename_flag=active_rename_flag,
        python_requirement=python_compatibility.requirement,
        python_compatibility_status=python_compatibility.status,
        python_compatibility_detail=python_compatibility.detail,
        python_hard_compat_fail=python_compatibility.hard_fail,
        fallback_notes=tuple(fallback_notes),
        cache_hit=False,
    )
    _CAP_CACHE[key] = capabilities
    return capabilities


def compatibility_checks(
    config: dict[str, Any],
    *,
    include_flag_probe: bool = False,
) -> list[tuple[str, str, str]]:
    caps = probe_lerobot_capabilities(config, include_flag_probe=include_flag_probe)
    checks: list[tuple[str, str, str]] = []
    policy = compatibility_policy_display(str(config.get("compat_policy", "latest_plus_n_minus_1")).strip())
    checks.append(("PASS", "Compatibility policy", policy))
    checks.append(("PASS", "Validated tracks", validated_tracks_summary()))
    checks.append(("PASS", "Workflow validation gate", WORKFLOW_PASS_GATE_NOTE))
    checks.append(
        (
            "PASS" if caps.lerobot_version != "unknown" else "WARN",
            "LeRobot version",
            caps.lerobot_version if caps.lerobot_version != "unknown" else "unable to detect installed lerobot version",
        )
    )
    checks.append(
        (
            caps.python_compatibility_status,
            "Python compatibility",
            caps.python_compatibility_detail,
        )
    )
    matched_track = match_validated_track(caps.lerobot_version)
    checks.append(
        (
            "PASS" if matched_track is not None else "WARN",
            "Validated LeRobot track",
            (
                f"{matched_track.label} ({matched_track.version_spec}, validated {matched_track.status_date})"
                if matched_track is not None
                else "installed version is outside the validated current/N-1 tracks"
            ),
        )
    )
    checks.append(("PASS", "Record entrypoint", caps.record_entrypoint))
    checks.append(("PASS", "Train entrypoint", caps.train_entrypoint))
    checks.append(("PASS", "Sim eval entrypoint", caps.sim_eval_entrypoint))
    teleop_detail = caps.teleop_entrypoint + (" (legacy control path)" if caps.teleop_uses_legacy_control else "")
    checks.append(("PASS", "Teleop entrypoint", teleop_detail))
    checks.append(("PASS", "Calibrate entrypoint", caps.calibrate_entrypoint))
    if caps.supports_policy_path:
        policy_flag = caps.policy_path_flag or "policy.path"
        checks.append(("PASS", "Policy path flag", f"--{policy_flag}"))
    else:
        checks.append(
            (
                "WARN",
                "Policy path flag",
                "No supported policy path flag detected in current LeRobot record entrypoint.",
            )
        )

    if caps.supported_rename_flags:
        checks.append(
            (
                "PASS",
                "Rename-map flag",
                f"active --{caps.active_rename_flag}; supported={list(caps.supported_rename_flags)}",
            )
        )
    else:
        checks.append(
            (
                "WARN",
                "Rename-map flag",
                f"could not confirm supported rename-map flags; using --{caps.active_rename_flag} by default",
            )
        )

    if caps.train_help_available:
        if caps.missing_train_flags:
            checks.append(
                (
                    "WARN",
                    "Train flags",
                    "missing required flags: "
                    + ", ".join(f"--{flag}" for flag in caps.missing_train_flags),
                )
            )
        else:
            checks.append(("PASS", "Train flags", f"required flags confirmed for {caps.train_entrypoint}"))
    else:
        checks.append(
            (
                "WARN",
                "Train flags",
                f"could not confirm required train flags; help output unavailable for {caps.train_entrypoint}",
            )
        )

    checks.append(
        (
            "PASS" if caps.supports_train_resume else "WARN",
            "Train resume",
            caps.train_resume_detail,
        )
    )
    checks.append(
        (
            "PASS" if caps.supports_sim_eval else "WARN",
            "Sim eval support",
            caps.sim_eval_support_detail,
        )
    )

    if caps.fallback_notes:
        for note in caps.fallback_notes:
            checks.append(("WARN", "Compatibility fallback", note))
    return checks
