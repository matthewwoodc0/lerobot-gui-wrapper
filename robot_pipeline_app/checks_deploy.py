from __future__ import annotations

import difflib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .camera_schema import (
    build_observation_rename_map,
    format_observation_rename_map,
    resolve_camera_feature_mapping,
    resolve_camera_schema,
)
from .commands import (
    follower_robot_action_dim,
    follower_robot_type,
    leader_robot_type,
    resolve_record_entrypoint,
)
from .compat import compatibility_checks, probe_lerobot_capabilities
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .diagnostics import checks_to_events
from .deploy_diagnostics import validate_model_path
from .feature_flags import compat_probe_enabled
from .model_metadata import extract_model_metadata
from .probes import (
    camera_fingerprint,
    in_virtual_env,
    parse_frame_dimensions,
    probe_camera_capture,
    probe_module_import,
    serial_port_fingerprint,
    summarize_probe_error,
)
from .repo_utils import (
    dataset_exists_on_hf,
    has_eval_prefix,
    increment_dataset_name,
    next_available_dataset_name,
    normalize_repo_id,
    repo_name_from_repo_id,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .types import CheckResult, DiagnosticEvent, PreflightReport

CommonChecksFn = Callable[[dict[str, Any]], list[CheckResult]]
WhichFn = Callable[[str], Optional[str]]

_DEFAULT_FOLLOWER_ROBOT_ID = "red4"
_DEFAULT_LEADER_ROBOT_ID = "white"

# Calibration sanity bounds (STS3215 Feetech servo, 12-bit ADC → 0–4095 ticks)
_CALIB_DRIVE_MODE_VALID = frozenset({0, 1})
_CALIB_HOMING_OFFSET_BOUND = 8192   # generous: ±4096 is 1 full revolution; >8192 implies corruption
_CALIB_RAW_POSITION_MAX = 4095      # 12-bit max
_CALIB_MIN_RANGE_TICKS = 200        # narrower than this → likely bad calibration zero-point
_HEAVY_MODEL_PATTERNS = (
    ("smolvlm", "SmolVLM"),
    ("vision_language", "vision-language"),
    ("vision-language", "vision-language"),
    ("video-instruct", "video-instruct"),
    ("vlm", "VLM"),
)

from .checks_common import (
    CommonChecksFn,
    _check_robot_calibration,
    _follower_robot_id,
    _leader_robot_id,
    _run_common_preflight_checks,
)

def _extract_model_config_fields(model_path: Path) -> tuple[dict[str, Any] | None, str]:
    metadata = extract_model_metadata(model_path)
    if metadata.errors:
        return None, metadata.errors[0]

    found: dict[str, Any] = {}
    if metadata.fps is not None:
        found["fps"] = metadata.fps
    if metadata.robot_type is not None:
        found["robot_type"] = metadata.robot_type
    if metadata.motor_names:
        found["motor_names"] = list(metadata.motor_names)
    if metadata.action_dim is not None:
        found["action_dim"] = metadata.action_dim
    if metadata.normalization_stats is not None:
        found["normalization_stats"] = metadata.normalization_stats

    if not found and metadata.normalization_present is not True:
        return None, metadata.metadata_source or "could not extract training config fields from model JSON files"

    return found, metadata.metadata_source or "detected via model metadata"


def _extract_model_camera_keys(model_path: Path) -> tuple[set[str] | None, str]:
    metadata = extract_model_metadata(model_path)
    if metadata.errors:
        return None, metadata.errors[0]
    if not metadata.camera_keys:
        return None, "could not infer camera keys from model metadata JSON files"
    return set(metadata.camera_keys), metadata.metadata_source or "detected via model metadata"


def _probe_torch_accelerator() -> tuple[str, str]:
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'imported': False, 'error': str(exc)}))\n"
        "    raise SystemExit(0)\n"
        "cuda = bool(torch.cuda.is_available())\n"
        "mps_backend = getattr(torch.backends, 'mps', None)\n"
        "mps = bool(mps_backend and mps_backend.is_available())\n"
        "print(json.dumps({'imported': True, 'cuda': cuda, 'mps': mps, 'torch': getattr(torch, '__version__', '')}))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return "unknown", f"Unable to probe torch runtime: {exc}"

    payload = (result.stdout or "").strip()
    if not payload:
        return "unknown", "torch probe returned no output."

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "unknown", summarize_probe_error(payload)

    if not isinstance(data, dict):
        return "unknown", "torch probe returned invalid payload."

    if not bool(data.get("imported")):
        return "unknown", f"torch import unavailable: {data.get('error', 'unknown error')}"

    torch_version = str(data.get("torch", "")).strip()
    suffix = f" (torch {torch_version})" if torch_version else ""
    if bool(data.get("cuda")):
        return "cuda", f"CUDA available{suffix}"
    if bool(data.get("mps")):
        return "mps", f"MPS available{suffix}"
    return "cpu", f"CPU-only runtime{suffix}"


def _infer_model_runtime_risk(model_path: Path) -> str | None:
    lowered_name = model_path.name.lower()
    for token, label in _HEAVY_MODEL_PATTERNS:
        if token in lowered_name:
            return f"{label} hint from model path name"

    try:
        json_files = [path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() == ".json"]
    except OSError:
        return None

    for json_path in json_files[:14]:
        try:
            text = json_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for token, label in _HEAVY_MODEL_PATTERNS:
            if token in text:
                return f"{label} hint from {json_path.name}"
    return None


def _extract_flag_value(argv: list[str] | None, flag_name: str) -> str | None:
    if not argv:
        return None
    normalized = str(flag_name or "").strip().lstrip("-")
    if not normalized:
        return None
    prefixed = f"--{normalized}"
    for idx in range(len(argv) - 1, -1, -1):
        current = str(argv[idx])
        if current.startswith(f"{prefixed}="):
            return current.split("=", 1)[1]
        if current == prefixed and idx + 1 < len(argv):
            return str(argv[idx + 1])
    return None


def _camera_rename_flag(config: dict[str, Any]) -> str:
    raw = str(config.get("camera_rename_flag", "rename_map")).strip().lstrip("-")
    return raw or "rename_map"


def _candidate_rename_flags(
    config: dict[str, Any],
    *,
    capabilities: Any | None = None,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        normalized = str(value or "").strip().lstrip("-")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    _add(_camera_rename_flag(config))
    if capabilities is not None:
        _add(getattr(capabilities, "active_rename_flag", ""))
        for supported in getattr(capabilities, "supported_rename_flags", ()) or ():
            _add(supported)

    for fallback in (
        "rename_map",
        "dataset.rename_map",
        "dataset.image_features_to_rename",
        "image_features_to_rename",
        "observation.rename_map",
    ):
        _add(fallback)
    return candidates


def _extract_first_flag_value(
    argv: list[str] | None,
    flag_names: list[str],
) -> tuple[str | None, str | None]:
    for flag_name in flag_names:
        value = _extract_flag_value(argv, flag_name)
        if value is not None:
            return flag_name, value
    return None, None


def _probe_record_flag_support(config: dict[str, Any], flag_name: str) -> CheckResult:
    normalized_flag = str(flag_name or "").strip().lstrip("-")
    flag = f"--{normalized_flag}"
    if not compat_probe_enabled(config):
        return (
            "WARN",
            f"lerobot_record flag: {flag}",
            "Compatibility probe disabled by config (compat_probe_enabled=false).",
        )
    capabilities = probe_lerobot_capabilities(config, include_flag_probe=True)
    module_name = capabilities.record_entrypoint or resolve_record_entrypoint(config)
    supported = set(capabilities.supported_record_flags)

    if not capabilities.record_help_available:
        detail = capabilities.record_help_error or "help output unavailable"
        return (
            "WARN",
            f"lerobot_record flag: {flag}",
            f"Could not confirm '{flag}' support for {module_name} ({detail}).",
        )

    if normalized_flag in supported:
        return ("PASS", f"lerobot_record flag: {flag}", f"{flag} supported by {module_name}")

    if normalized_flag == "policy.path" and capabilities.policy_path_flag:
        alt = capabilities.policy_path_flag
        return (
            "WARN",
            f"lerobot_record flag: {flag}",
            f"{flag} not supported by {module_name}; fallback available via --{alt}.",
        )

    if normalized_flag in {"rename_map", "dataset.rename_map", _camera_rename_flag(config)} and capabilities.active_rename_flag:
        alt = capabilities.active_rename_flag
        if alt in supported:
            return (
                "WARN",
                f"lerobot_record flag: {flag}",
                f"{flag} not supported by {module_name}; fallback available via --{alt}.",
            )

    return (
        "WARN",
        f"lerobot_record flag: {flag}",
        f"Could not confirm '{flag}' in {module_name} help output (non-blocking).",
    )


def _probe_policy_path_support(config: dict[str, Any]) -> CheckResult:
    return _probe_record_flag_support(config, "policy.path")


def _probe_rename_map_support(config: dict[str, Any]) -> CheckResult:
    if not compat_probe_enabled(config):
        return (
            "WARN",
            "lerobot_record flag: rename_map",
            "Compatibility probe disabled by config (compat_probe_enabled=false).",
        )
    capabilities = probe_lerobot_capabilities(config, include_flag_probe=True)
    rename_flag = str(capabilities.active_rename_flag or "").strip() or _camera_rename_flag(config)
    return _probe_record_flag_support(config, rename_flag)


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    command: list[str] | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)

    if compat_probe_enabled(config):
        for level, name, detail in compatibility_checks(config, include_flag_probe=True):
            checks.append((level, f"Deploy compatibility: {name}", detail))
    else:
        checks.append(("WARN", "Deploy compatibility", "compatibility probe disabled by config (compat_probe_enabled=false)"))

    username = str(config.get("hf_username", "")).strip()
    eval_repo = str(eval_repo_id or "").strip()
    if not eval_repo:
        fallback_dataset = str(config.get("last_eval_dataset_name", "")).strip()
        eval_repo = normalize_repo_id(username, fallback_dataset)
    suggested_eval_repo, _ = suggest_eval_prefixed_repo_id(username, eval_repo)
    has_prefix = has_eval_prefix(eval_repo)

    checks.append(
        (
            "PASS" if has_prefix else "FAIL",
            "Eval dataset naming",
            eval_repo
            if has_prefix
            else f"Eval dataset repo must begin with 'eval_' (dataset part). Suggested quick fix: {suggested_eval_repo}",
        )
    )

    if has_prefix and eval_repo and username:
        eval_name = repo_name_from_repo_id(eval_repo)
        exists_on_hf = bool(dataset_exists_on_hf(eval_repo))
        if exists_on_hf:
            suggested = next_available_dataset_name(base_name=eval_name, hf_username=username)
            suggested_repo = f"{username}/{suggested}"
            checks.append((
                "FAIL",
                "Eval dataset already exists",
                f"'{eval_repo}' already exists on Hugging Face. "
                f"Rename it to keep runs separate. Next available name: '{suggested_repo}'.",
            ))
        else:
            checks.append(("PASS", "Eval dataset name", f"'{eval_name}' is available on Hugging Face."))

    # Check ML dependencies required for policy inference.
    # These are optional extras that may not be installed in the base lerobot env.
    for ml_dep, fix_hint in (
        (
            "transformers",
            "pip install transformers  (or: cd ~/lerobot && pip install -e '.[smolvla]')",
        ),
        (
            "torch",
            "pip install torch  (follow PyTorch install guide for your CUDA version)",
        ),
    ):
        dep_ok, dep_msg = probe_module_import(ml_dep)
        checks.append((
            "PASS" if dep_ok else "FAIL",
            f"Python module: {ml_dep}",
            "import ok" if dep_ok else (
                summarize_probe_error(dep_msg)
                + f" — Fix: {fix_hint}"
            ),
        ))

    is_valid_model, detail, candidates = validate_model_path(model_path)
    checks.append(("PASS" if model_path.exists() and model_path.is_dir() else "FAIL", "Model folder", str(model_path)))
    if model_path.exists() and model_path.is_dir():
        model_access = os.access(str(model_path), os.R_OK | os.X_OK)
        checks.append(
            (
                "PASS" if model_access else "FAIL",
                "Model folder access",
                "read/execute ok"
                if model_access
                else f"permission denied for model folder: {model_path}",
            )
        )
    checks.append(("PASS" if is_valid_model else "FAIL", "Model payload", detail))
    if candidates:
        checks.append(
            (
                "WARN",
                "Model payload candidates",
                ", ".join(str(path) for path in candidates[:3]),
            )
        )

    model_metadata = extract_model_metadata(model_path)
    if model_metadata.errors:
        checks.append(("WARN", "Model metadata", model_metadata.errors[0]))
    else:
        checks.append(("PASS", "Model metadata", model_metadata.metadata_source))
        checks.append(
            (
                "PASS" if (model_metadata.policy_family or model_metadata.policy_class) else "WARN",
                "Policy family/class",
                f"{model_metadata.policy_family or 'unknown'} / {model_metadata.policy_class or 'unknown'}",
            )
        )
        if model_metadata.plugin_package:
            plugin_ok, plugin_msg = probe_module_import(model_metadata.plugin_package)
            plugin_detail = (
                f"import ok ({model_metadata.plugin_package})"
                if plugin_ok
                else (
                    summarize_probe_error(plugin_msg)
                    + f" — Fix: pip install {model_metadata.plugin_package}"
                )
            )
            checks.append(
                (
                    "PASS" if plugin_ok else "FAIL",
                    "Policy plugin package",
                    plugin_detail,
                )
            )
        if model_metadata.runtime_labels:
            checks.append(("PASS", "Model runtime labels", ", ".join(model_metadata.runtime_labels)))
        if model_metadata.supports_rtc is not None:
            checks.append(
                (
                    "PASS",
                    "Model RTC capability",
                    "supported" if model_metadata.supports_rtc else "not declared by metadata",
                )
            )
        if model_metadata.normalization_present is not None:
            checks.append(
                (
                    "PASS" if model_metadata.normalization_present else "WARN",
                    "Model normalization",
                    "normalization stats detected" if model_metadata.normalization_present else "normalization stats not detected",
                )
            )

    schema = resolve_camera_schema(config)
    runtime_keys = {spec.name for spec in schema.specs}
    if schema.errors:
        for message in schema.errors:
            checks.append(("FAIL", "Runtime camera schema", message))
    if schema.warnings:
        for message in schema.warnings:
            checks.append(("WARN", "Runtime camera schema", message))
    if runtime_keys:
        checks.append(("PASS", "Runtime camera schema", f"{len(runtime_keys)} key(s): {sorted(runtime_keys)}"))
    else:
        checks.append(("FAIL", "Runtime camera schema", "no runtime camera keys configured"))

    model_camera_keys, camera_key_detail = _extract_model_camera_keys(model_path)
    if model_camera_keys is None:
        checks.append(("WARN", "Model camera keys", camera_key_detail))
    else:
        if model_camera_keys == runtime_keys:
            checks.append(("PASS", "Model camera keys", f"matches runtime keys {sorted(runtime_keys)}"))
        else:
            mapping, mapping_error = resolve_camera_feature_mapping(
                config=config,
                runtime_keys=runtime_keys,
                model_keys=model_camera_keys,
            )
            if mapping is None:
                checks.append(
                    (
                        "FAIL",
                        "Model camera keys",
                        (
                            f"model={sorted(model_camera_keys)}; runtime={sorted(runtime_keys)}; "
                            f"{mapping_error or 'camera key mismatch between training and deployment'}"
                        ),
                    )
                )
            else:
                rename_map = build_observation_rename_map(mapping)
                deploy_capabilities = probe_lerobot_capabilities(config, include_flag_probe=True) if compat_probe_enabled(config) else None
                rename_flag = str(getattr(deploy_capabilities, "active_rename_flag", "") or "").strip() or _camera_rename_flag(config)
                rename_candidates = _candidate_rename_flags(config, capabilities=deploy_capabilities)
                used_rename_flag, rename_flag_value = _extract_first_flag_value(command, rename_candidates)
                effective_rename_flag = used_rename_flag or rename_flag
                if rename_map:
                    suggested_json = format_observation_rename_map(mapping)
                    rename_map_valid = False
                    rename_map_error: str | None = None
                    if rename_flag_value:
                        try:
                            parsed = json.loads(rename_flag_value)
                        except json.JSONDecodeError as exc:
                            rename_map_error = f"configured --{effective_rename_flag} is not valid JSON: {exc}"
                        else:
                            if isinstance(parsed, dict):
                                if parsed == rename_map:
                                    rename_map_valid = True
                                else:
                                    rename_map_error = (
                                        f"--{effective_rename_flag} does not match suggested mapping; "
                                        f"suggested={suggested_json}"
                                    )
                            else:
                                rename_map_error = f"--{effective_rename_flag} must decode to a JSON object"
                    else:
                        support_level, support_name, support_detail = _probe_rename_map_support(config)
                        checks.append((support_level, support_name, support_detail))
                        rename_map_error = f"camera keys require --{rename_flag} before deploy"

                    if rename_map_valid:
                        checks.append(
                            (
                                "PASS",
                                "Model camera keys",
                                (
                                    f"model={sorted(model_camera_keys)}; runtime={sorted(runtime_keys)}; "
                                    f"mapped via --{effective_rename_flag}"
                                ),
                            )
                        )
                        checks.append(
                            (
                                "PASS",
                                "Camera rename map",
                                f"--{effective_rename_flag} matches suggested mapping",
                            )
                        )
                    else:
                        checks.append(
                            (
                                "FAIL",
                                "Model camera keys",
                                (
                                    f"model={sorted(model_camera_keys)}; runtime={sorted(runtime_keys)}; "
                                    f"{rename_map_error or f'camera keys require --{rename_flag} before deploy'}"
                                ),
                            )
                        )
                        checks.append(
                            (
                                "WARN",
                                "Camera rename map suggestion",
                                suggested_json,
                            )
                        )
                        if rename_map_error:
                            checks.append(("FAIL", "Camera rename map", rename_map_error))
                else:
                    checks.append(("PASS", "Model camera keys", "runtime and model keys match without rename map"))

    # ------------------------------------------------------------------ #
    # Training config vs. deploy config comparison                        #
    # ------------------------------------------------------------------ #
    model_config_fields, model_config_source = _extract_model_config_fields(model_path)
    if model_config_fields is None:
        checks.append(("WARN", "Model training config", model_config_source))
    else:
        # FPS check
        model_fps = model_config_fields.get("fps")
        if model_fps is None:
            checks.append(("WARN", "Training vs deploy FPS", f"fps not found in model metadata ({model_config_source})"))
        else:
            runtime_fps = int(config.get("camera_fps", 30))
            if int(model_fps) == runtime_fps:
                checks.append(("PASS", "Training vs deploy FPS", f"match: {runtime_fps} Hz"))
            else:
                checks.append((
                    "FAIL",
                    "Training vs deploy FPS",
                    (
                        f"model trained at {int(model_fps)} Hz but camera_fps={runtime_fps}; "
                        "FPS mismatch causes timing drift and degraded policy performance"
                    ),
                ))

        # Robot type check
        model_robot_type = model_config_fields.get("robot_type")
        deploy_robot_type = follower_robot_type(config)
        if model_robot_type is None:
            checks.append(("WARN", "Training vs deploy robot type", f"robot_type not found in model metadata ({model_config_source})"))
        elif model_robot_type == deploy_robot_type:
            checks.append(("PASS", "Training vs deploy robot type", f"match: {deploy_robot_type}"))
        else:
            checks.append((
                "FAIL",
                "Training vs deploy robot type",
                (
                    f"model trained on '{model_robot_type}', deploying to '{deploy_robot_type}'; "
                    "robot type mismatch will cause action space errors at runtime"
                ),
            ))

        # Action dimension check
        model_action_dim = model_config_fields.get("action_dim")
        deploy_action_dim = follower_robot_action_dim(config)
        if model_action_dim is not None:
            if model_action_dim == deploy_action_dim:
                checks.append(("PASS", "Training vs deploy action dim", f"match: {deploy_action_dim} DOF"))
            else:
                checks.append((
                    "FAIL",
                    "Training vs deploy action dim",
                    (
                        f"model outputs {model_action_dim} actions but {deploy_robot_type} "
                        f"expects {deploy_action_dim} DOF; shape mismatch will crash at inference"
                    ),
                ))

    # ------------------------------------------------------------------ #
    # Robot calibration — follower and leader                             #
    # ------------------------------------------------------------------ #
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_follower_robot_id(config), robot_type=follower_robot_type(config),
        config_key="follower_calibration_path", label="Follower",
        model_config_fields=model_config_fields,
    ))
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_leader_robot_id(config), robot_type=leader_robot_type(config),
        config_key="leader_calibration_path", label="Leader",
    ))

    checks.append(_probe_policy_path_support(config))

    fps = int(config.get("camera_fps", 30))
    accelerator, accel_detail = _probe_torch_accelerator()
    accel_level = "PASS" if accelerator in {"cuda", "mps"} else "WARN"
    checks.append((accel_level, "Compute accelerator", accel_detail))

    if accelerator == "cpu" and fps >= 25:
        checks.append(
            (
                "WARN",
                "Deploy loop performance risk",
                (
                    f"camera_fps={fps} with CPU-only runtime. "
                    "Target 30Hz often drops to single-digit Hz during policy inference. "
                    "Consider camera_fps=8-15, smaller model, or GPU/MPS acceleration."
                ),
            )
        )

    model_risk = _infer_model_runtime_risk(model_path)
    if model_risk:
        checks.append(
            (
                "WARN",
                "Model inference load",
                f"{model_risk}. VLM-style policies are commonly slower than 30Hz without acceleration.",
            )
        )
    return checks


def run_preflight_for_deploy_events(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    command: list[str] | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[DiagnosticEvent]:
    return checks_to_events(
        run_preflight_for_deploy(
            config=config,
            model_path=model_path,
            eval_repo_id=eval_repo_id,
            command=command,
            common_checks_fn=common_checks_fn,
        )
    )
