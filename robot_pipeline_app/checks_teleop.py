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
    has_eval_prefix,
    increment_dataset_name,
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

def run_preflight_for_teleop(
    config: dict[str, Any],
    control_fps: int | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)

    if control_fps is not None:
        if control_fps <= 0:
            checks.append(("FAIL", "Teleop control FPS", f"{control_fps} is invalid; must be greater than zero."))
        elif control_fps > 120:
            checks.append(("WARN", "Teleop control FPS", f"{control_fps} is high and may be unstable on CPU-bound hosts."))
        else:
            checks.append(("PASS", "Teleop control FPS", str(control_fps)))

    # Calibration checks for both arms
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


def run_preflight_for_teleop_events(
    config: dict[str, Any],
    control_fps: int | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[DiagnosticEvent]:
    return checks_to_events(
        run_preflight_for_teleop(
            config=config,
            control_fps=control_fps,
            common_checks_fn=common_checks_fn,
        )
    )
