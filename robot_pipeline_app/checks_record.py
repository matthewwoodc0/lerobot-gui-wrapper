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
    WhichFn,
    _check_robot_calibration,
    _follower_robot_id,
    _leader_robot_id,
    _nearest_existing_parent,
    _run_common_preflight_checks,
)

def _collect_local_dataset_names(roots: list[Path]) -> set[str]:
    names: set[str] = set()
    for root in roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
            for child in root.iterdir():
                if child.is_dir():
                    names.add(child.name)
        except OSError:
            continue
    return names


def _is_numbered_neighbor(a: str, b: str) -> bool:
    try:
        return increment_dataset_name(a) == b or increment_dataset_name(b) == a
    except Exception:
        return False


def _possible_dataset_typo(dataset_repo_id: str, known_names: set[str]) -> str | None:
    dataset_name = repo_name_from_repo_id(dataset_repo_id)
    if not dataset_name or dataset_name in known_names:
        return None

    candidates = sorted(name for name in known_names if name and name != dataset_name)
    if not candidates:
        return None

    close_matches = difflib.get_close_matches(dataset_name, candidates, n=1, cutoff=0.8)
    if not close_matches:
        return None

    closest = close_matches[0]
    if _is_numbered_neighbor(dataset_name, closest):
        return None

    return (
        f"'{dataset_name}' is very close to existing dataset '{closest}'. "
        "Confirm this isn't a typo before recording/upload."
    )


def run_preflight_for_record(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
    which_fn: WhichFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)
    which = which_fn or shutil.which

    if dataset_root.exists() and not dataset_root.is_dir():
        checks.append(("FAIL", "Dataset root folder", f"Not a directory: {dataset_root}"))
    else:
        probe_path = dataset_root if dataset_root.exists() else _nearest_existing_parent(dataset_root)
        if probe_path is None:
            checks.append(("FAIL", "Dataset root folder", f"No existing parent for path: {dataset_root}"))
        elif not os.access(str(probe_path), os.W_OK):
            checks.append(("FAIL", "Dataset root folder", f"No write permission for: {probe_path}"))
        else:
            checks.append(("PASS", "Dataset root folder", f"Writable path available via: {probe_path}"))

    if dataset_repo_id:
        dataset_name = repo_name_from_repo_id(dataset_repo_id)
        hf_username = str(config.get("hf_username", "")).strip()
        exists_locally = (dataset_root / dataset_name).exists()
        exists_on_hf = bool(dataset_exists_on_hf(dataset_repo_id)) if hf_username else False
        if exists_locally or exists_on_hf:
            where = []
            if exists_locally:
                where.append(f"locally at {dataset_root / dataset_name}")
            if exists_on_hf:
                where.append(f"on Hugging Face ({dataset_repo_id})")
            suggested = next_available_dataset_name(
                base_name=dataset_name, hf_username=hf_username, dataset_root=dataset_root
            )
            checks.append((
                "WARN",
                "Dataset already exists",
                f"'{dataset_name}' already exists {' and '.join(where)}. "
                f"Recording will append episodes to it. To start fresh, rename to '{suggested}'.",
            ))
        else:
            checks.append(("PASS", "Dataset name", f"'{dataset_name}' is available."))

    if episode_time_s is not None:
        if episode_time_s <= 0:
            checks.append(("FAIL", "Episode duration", f"{episode_time_s}s is invalid; must be greater than zero."))
        elif episode_time_s < 8:
            checks.append(
                (
                    "WARN",
                    "Episode duration",
                    f"{episode_time_s}s is likely too short and may truncate tasks (recommended: 8-120s).",
                )
            )
        elif episode_time_s > 180:
            checks.append(
                (
                    "WARN",
                    "Episode duration",
                    f"{episode_time_s}s is long and may add dead time (recommended: 8-120s).",
                )
            )
        else:
            checks.append(("PASS", "Episode duration", f"{episode_time_s}s"))

    if dataset_repo_id:
        known_names = {
            str(config.get("last_dataset_name", "")).strip(),
            str(config.get("last_eval_dataset_name", "")).strip(),
        }
        known_names.update(_collect_local_dataset_names([dataset_root, get_lerobot_dir(config) / "data"]))
        typo_warning = _possible_dataset_typo(dataset_repo_id, {name for name in known_names if name})
        if typo_warning:
            checks.append(("WARN", "Dataset repo typo risk", typo_warning))

    if upload_enabled:
        hf_cli = which("huggingface-cli")
        checks.append(
            (
                "PASS" if hf_cli else "FAIL",
                "huggingface-cli",
                hf_cli or "not found in PATH",
            )
        )

    # Calibration checks for both arms (recording uses both)
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_follower_robot_id(config), robot_type=follower_robot_type(config),
        config_key="follower_calibration_path", label="Follower",
    ))
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_leader_robot_id(config), robot_type=leader_robot_type(config),
        config_key="leader_calibration_path", label="Leader",
    ))

    return checks


def run_preflight_for_record_events(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
    which_fn: WhichFn | None = None,
) -> list[DiagnosticEvent]:
    return checks_to_events(
        run_preflight_for_record(
            config=config,
            dataset_root=dataset_root,
            upload_enabled=upload_enabled,
            episode_time_s=episode_time_s,
            dataset_repo_id=dataset_repo_id,
            common_checks_fn=common_checks_fn,
            which_fn=which_fn,
        )
    )
