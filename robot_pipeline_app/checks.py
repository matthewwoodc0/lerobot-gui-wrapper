from __future__ import annotations

import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .deploy_diagnostics import validate_model_path
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
from .types import CheckResult, PreflightReport

CommonChecksFn = Callable[[dict[str, Any]], list[CheckResult]]
WhichFn = Callable[[str], Optional[str]]

_CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_DEPLOY_ROBOT_TYPE = "so101_follower"  # must match commands.py build_lerobot_record_command
_DEPLOY_ROBOT_ID = "red4"             # must match commands.py build_lerobot_record_command
_DEPLOY_ROBOT_MOTOR_COUNT = 6         # so101_follower has 6 DOF
_LEADER_ROBOT_TYPE = "so101_leader"   # must match commands.py build_lerobot_teleop_command
_LEADER_ROBOT_ID = "white"            # must match commands.py build_lerobot_teleop_command

# Calibration sanity bounds (STS3215 Feetech servo, 12-bit ADC → 0–4095 ticks)
_CALIB_DRIVE_MODE_VALID = frozenset({0, 1})
_CALIB_HOMING_OFFSET_BOUND = 8192   # generous: ±4096 is 1 full revolution; >8192 implies corruption
_CALIB_RAW_POSITION_MAX = 4095      # 12-bit max
_CALIB_MIN_RANGE_TICKS = 200        # narrower than this → likely bad calibration zero-point
_CAMERA_NAME_BLOCKLIST = {
    "width",
    "height",
    "fps",
    "shape",
    "dtype",
    "type",
    "index",
    "path",
    "device",
    "name",
    "mean",
    "std",
    "low",
    "high",
    "channels",
    "color_space",
    "normalization",
}
_HEAVY_MODEL_PATTERNS = (
    ("smolvlm", "SmolVLM"),
    ("vision_language", "vision-language"),
    ("vision-language", "vision-language"),
    ("video-instruct", "video-instruct"),
    ("vlm", "VLM"),
)


def _check_counts(checks: list[CheckResult]) -> tuple[int, int, int]:
    pass_count = sum(1 for level, _, _ in checks if level == "PASS")
    warn_count = sum(1 for level, _, _ in checks if level == "WARN")
    fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
    return pass_count, warn_count, fail_count


def build_preflight_report(checks: list[CheckResult]) -> PreflightReport:
    pass_count, warn_count, fail_count = _check_counts(checks)
    return PreflightReport(
        checks=checks,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
    )


def summarize_checks(checks: list[CheckResult], title: str = "Checks") -> str:
    pass_count, warn_count, fail_count = _check_counts(checks)
    lines = [title]
    lines.extend(f"[{level:4}] {name}: {detail}" for level, name, detail in checks)
    lines.append("")
    lines.append(f"Summary: PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return "\n".join(lines)


def has_failures(checks: list[CheckResult]) -> bool:
    return any(level == "FAIL" for level, _, _ in checks)


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current == current.parent:
            return None
        current = current.parent
    return current



def _dialout_membership() -> tuple[bool | None, str]:
    if not sys.platform.startswith("linux"):
        return None, "dialout group check is only applicable on Linux."

    try:
        import grp
        import pwd
    except Exception as exc:
        return None, f"Unable to import group modules: {exc}"

    try:
        dialout_group = grp.getgrnam("dialout")
    except KeyError:
        return None, "Group 'dialout' not found on this system."

    try:
        user_name = pwd.getpwuid(os.getuid()).pw_name
        user_groups = set(os.getgroups())
    except Exception as exc:
        return None, f"Unable to inspect current user groups: {exc}"

    in_group = dialout_group.gr_gid in user_groups or user_name in dialout_group.gr_mem
    if in_group:
        return True, f"User '{user_name}' is in dialout."

    return (
        False,
        (
            f"User '{user_name}' is not in dialout. "
            "Fix: sudo usermod -a -G dialout $USER (then log out/in or run newgrp dialout)."
        ),
    )


def _serial_lock_check(ports: list[str]) -> CheckResult:
    unique_ports = sorted({str(port).strip() for port in ports if str(port).strip()})
    if not unique_ports:
        return ("WARN", "Serial port lock", "No accessible serial ports available for lock inspection.")

    lsof_path = shutil.which("lsof")
    if not lsof_path:
        return ("WARN", "Serial port lock", "lsof not found in PATH; cannot detect active serial lock holders.")

    try:
        result = subprocess.run(
            [lsof_path, *unique_ports],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return ("WARN", "Serial port lock", f"Unable to run lsof on configured serial ports: {exc}")

    stdout = result.stdout or ""
    if result.returncode == 1 and not stdout.strip():
        return ("PASS", "Serial port lock", "No active process holds configured serial ports.")

    if result.returncode == 0:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        holders = lines[1:] if len(lines) > 1 else []
        holder_preview = "; ".join(holders[:3]) if holders else "one or more lock holders detected"
        ports_text = " ".join(unique_ports)
        return (
            "FAIL",
            "Serial port lock",
            (
                f"Configured serial ports are currently in use ({holder_preview}). "
                f"Stop the holder first. Quick reset: lsof {ports_text}; sudo fuser -k {ports_text}"
            ),
        )

    detail = summarize_probe_error(result.stderr or stdout or f"exit code {result.returncode}")
    return ("WARN", "Serial port lock", f"Could not determine serial lock state: {detail}")


def _append_fingerprint_check(
    *,
    add: Callable[[str, str, str], None],
    check_name: str,
    baseline_value: str,
    current_value: str | None,
    missing_detail: str,
) -> None:
    baseline = str(baseline_value or "").strip()
    if current_value is None:
        add("WARN", check_name, missing_detail)
        return

    if baseline and baseline == current_value:
        add("PASS", check_name, "matches saved baseline")
        return

    if baseline and baseline != current_value:
        add("FAIL", check_name, f"saved={baseline}; current={current_value} (mapping drift detected)")
        return

    add("WARN", check_name, "no baseline saved yet; scan/assign once to lock mapping.")


def _inferred_role_from_fingerprint(fingerprint: str | None) -> str | None:
    if not fingerprint:
        return None
    lowered = fingerprint.lower()
    has_leader = "leader" in lowered
    has_follower = "follower" in lowered
    if has_leader and not has_follower:
        return "leader"
    if has_follower and not has_leader:
        return "follower"
    return None


def _append_robot_port_checks(config: dict[str, Any], add: Callable[[str, str, str], None]) -> None:
    follower_port = str(config.get("follower_port", "")).strip()
    leader_port = str(config.get("leader_port", "")).strip()
    named_ports = (("Follower port", "follower", follower_port), ("Leader port", "leader", leader_port))

    resolved_paths: dict[str, str] = {}
    accessible_ports: list[str] = []
    fingerprints: dict[str, str | None] = {}

    for label, role_key, port in named_ports:
        if not port:
            add("FAIL", label, "(empty)")
            continue

        port_path = Path(port)
        if not port_path.exists():
            add("FAIL", label, f"{port} (not found)")
            continue

        add("PASS", label, port)
        try:
            resolved_paths[role_key] = str(port_path.resolve())
        except OSError:
            resolved_paths[role_key] = port

        has_rw = os.access(port, os.R_OK | os.W_OK)
        add(
            "PASS" if has_rw else "FAIL",
            f"{label} access",
            "read/write ok" if has_rw else f"permission denied for {port}",
        )
        if has_rw:
            accessible_ports.append(port)

        fingerprint = serial_port_fingerprint(port)
        fingerprints[role_key] = fingerprint
        _append_fingerprint_check(
            add=add,
            check_name=f"{label} fingerprint",
            baseline_value=str(config.get(f"{role_key}_port_fingerprint", "")),
            current_value=fingerprint,
            missing_detail=f"could not fingerprint {port}; use stable /dev/serial/by-id path when available.",
        )

        inferred_role = _inferred_role_from_fingerprint(fingerprint)
        if inferred_role is None:
            add("WARN", f"{label} role inference", "could not infer role from serial fingerprint text.")
        elif inferred_role == role_key:
            add("PASS", f"{label} role inference", f"fingerprint label suggests '{role_key}'.")
        else:
            add(
                "FAIL",
                f"{label} role inference",
                f"fingerprint label suggests '{inferred_role}', not configured role '{role_key}'.",
            )

    if follower_port and leader_port:
        same_path = follower_port == leader_port
        same_resolved = (
            resolved_paths.get("follower") is not None
            and resolved_paths.get("leader") is not None
            and resolved_paths["follower"] == resolved_paths["leader"]
        )
        add(
            "FAIL" if same_path or same_resolved else "PASS",
            "Leader/Follower port uniqueness",
            (
                "leader and follower point to the same serial device"
                if same_path or same_resolved
                else "leader and follower ports are distinct"
            ),
        )

    follower_fp = fingerprints.get("follower")
    leader_fp = fingerprints.get("leader")
    if follower_fp and leader_fp:
        add(
            "FAIL" if follower_fp == leader_fp else "PASS",
            "Leader/Follower port identity",
            (
                "both role ports fingerprint to the same hardware device"
                if follower_fp == leader_fp
                else "role ports fingerprint to different hardware devices"
            ),
        )

    linux_acm_ports = [
        port
        for port in (follower_port, leader_port)
        if port and "ttyacm" in port.lower()
    ]
    if linux_acm_ports:
        in_dialout, dialout_detail = _dialout_membership()
        if in_dialout is True:
            add("PASS", "dialout group", dialout_detail)
        elif in_dialout is False:
            # If user is NOT in dialout but all ports are R/W accessible
            # (e.g. via udev rules, ACLs, or chmod), this is fine — downgrade to WARN.
            all_ports_accessible = all(
                os.access(p, os.R_OK | os.W_OK)
                for p in linux_acm_ports
                if Path(p).exists()
            )
            if all_ports_accessible and accessible_ports:
                add(
                    "WARN",
                    "dialout group",
                    dialout_detail + " (Ports are accessible — a udev rule or ACL may be granting access.)",
                )
            else:
                add("FAIL", "dialout group", dialout_detail)
        else:
            add("WARN", "dialout group", dialout_detail)

    lock_level, lock_name, lock_detail = _serial_lock_check(accessible_ports)
    add(lock_level, lock_name, lock_detail)


def _append_camera_checks(config: dict[str, Any], add: Callable[[str, str, str], None]) -> None:
    try:
        laptop_idx = int(config.get("camera_laptop_index", 0))
    except (TypeError, ValueError):
        laptop_idx = 0
    try:
        phone_idx = int(config.get("camera_phone_index", 1))
    except (TypeError, ValueError):
        phone_idx = 1

    add(
        "PASS" if laptop_idx != phone_idx else "FAIL",
        "Camera indices",
        f"laptop={laptop_idx}, phone={phone_idx}",
    )

    cv2_ok, cv2_msg = probe_module_import("cv2")
    add(
        "PASS" if cv2_ok else "FAIL",
        "Python module: cv2",
        "import ok" if cv2_ok else summarize_probe_error(cv2_msg),
    )
    if not cv2_ok:
        return

    width = 640
    height = 360
    role_specs = (
        (
            "laptop",
            "Laptop",
            laptop_idx,
            width,
            height,
        ),
        (
            "phone",
            "Phone",
            phone_idx,
            width,
            height,
        ),
    )

    role_fingerprints: dict[str, str | None] = {}
    for role, role_label, index, target_width, target_height in role_specs:
        opened, probe_detail = probe_camera_capture(index, target_width, target_height)
        if opened:
            add("PASS", f"Camera {index} probe", probe_detail)
        else:
            detail = summarize_probe_error(probe_detail)
            add(
                "FAIL",
                f"Camera {index} probe",
                f"{detail}; verify index {index} exists and camera is connected.",
            )

        actual = parse_frame_dimensions(probe_detail)
        if actual is not None:
            actual_width, actual_height = actual
            if (actual_width, actual_height) == (target_width, target_height):
                add(
                    "PASS",
                    f"{role_label} camera resolution",
                    f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}",
                )
            else:
                add(
                    "WARN",
                    f"{role_label} camera resolution",
                    (
                        f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}; "
                        "runtime will auto-detect camera frame size when building the command."
                    ),
                )
        elif opened:
            add("WARN", f"{role_label} camera resolution", "opened camera but could not parse detected frame size.")

        fingerprint = camera_fingerprint(index)
        role_fingerprints[role] = fingerprint
        _append_fingerprint_check(
            add=add,
            check_name=f"{role_label} camera fingerprint",
            baseline_value=str(config.get(f"camera_{role}_fingerprint", "")),
            current_value=fingerprint,
            missing_detail=f"could not fingerprint camera index {index} on this platform.",
        )

    laptop_fp = role_fingerprints.get("laptop")
    phone_fp = role_fingerprints.get("phone")
    if laptop_fp and phone_fp:
        add(
            "FAIL" if laptop_fp == phone_fp else "PASS",
            "Laptop/Phone camera identity",
            (
                "both roles map to the same camera fingerprint"
                if laptop_fp == phone_fp
                else "camera role fingerprints are distinct"
            ),
        )


def _extract_top_level_fps(payload: Any) -> float | None:
    """Return the top-level 'fps' or 'control_fps' value from a JSON dict, if positive."""
    if not isinstance(payload, dict):
        return None
    for key in ("fps", "control_fps"):
        val = payload.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return None


def _extract_robot_type_from_payload(payload: Any) -> str | None:
    """Return a robot type string from a JSON dict (top-level or under a nested section)."""
    if not isinstance(payload, dict):
        return None
    # Direct top-level key
    rt = payload.get("robot_type")
    if isinstance(rt, str) and rt.strip():
        return rt.strip()
    # Nested under common section keys
    for section_key in ("robot", "env", "config"):
        section = payload.get(section_key)
        if isinstance(section, dict):
            rt = section.get("robot_type") or section.get("type")
            if isinstance(rt, str) and rt.strip():
                return rt.strip()
    return None


def _extract_motor_info_from_payload(payload: Any) -> tuple[list[str] | None, int | None]:
    """Extract motor names and action dimension from a JSON dict.

    Returns ``(motor_names, action_dim)``.  Either member may be ``None`` when
    the relevant key is absent from the payload.
    """
    if not isinstance(payload, dict):
        return None, None

    motor_names: list[str] | None = None
    action_dim: int | None = None

    # LeRobot stores motor names under several possible keys
    for key in ("motor_names", "motors", "joint_names", "actuator_names"):
        val = payload.get(key)
        if isinstance(val, list) and all(isinstance(n, str) for n in val) and val:
            motor_names = [str(n) for n in val]
            break

    # Action dimension from output_shapes / action_space / features
    for shapes_key in ("output_shapes", "action_space", "features"):
        shapes = payload.get(shapes_key)
        if not isinstance(shapes, dict):
            continue
        for action_key in ("action",):
            entry = shapes.get(action_key)
            if isinstance(entry, dict):
                shape = entry.get("shape") or entry.get("size")
                if isinstance(shape, list) and shape:
                    try:
                        action_dim = int(shape[0])
                    except (TypeError, ValueError):
                        pass
                elif isinstance(shape, int) and shape > 0:
                    action_dim = shape
            elif isinstance(entry, list) and entry:
                try:
                    action_dim = int(entry[0])
                except (TypeError, ValueError):
                    pass
        if action_dim is not None:
            break

    return motor_names, action_dim


def _extract_model_config_fields(model_path: Path) -> tuple[dict[str, Any] | None, str]:
    """Scan JSON files in the model folder and return key training-time config fields.

    Returns ``(fields_dict, source_description)``.  ``fields_dict`` may contain:

    - ``"fps"``          (float)      – control/recording Hz used during training
    - ``"robot_type"``   (str)        – robot hardware class used during training
    - ``"motor_names"``  (list[str])  – ordered motor/joint names the policy acts on
    - ``"action_dim"``   (int)        – number of action dimensions expected by the policy

    Returns ``(None, error_message)`` when no relevant fields can be found.
    """
    if not model_path.exists() or not model_path.is_dir():
        return None, f"model path not readable: {model_path}"

    try:
        json_files = sorted(
            p for p in model_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"
        )
    except OSError as exc:
        return None, f"cannot list model folder: {exc}"

    if not json_files:
        return None, "no JSON metadata files found in model folder"

    found: dict[str, Any] = {}
    sources: list[str] = []

    _all_done = {"fps", "robot_type", "motor_names", "action_dim", "normalization_stats"}

    for json_path in json_files[:12]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        changed = False

        if "fps" not in found:
            fps_val = _extract_top_level_fps(payload)
            if fps_val is not None:
                found["fps"] = fps_val
                changed = True

        if "robot_type" not in found:
            rt_val = _extract_robot_type_from_payload(payload)
            if rt_val is not None:
                found["robot_type"] = rt_val
                changed = True

        if "motor_names" not in found or "action_dim" not in found:
            motor_names, action_dim = _extract_motor_info_from_payload(payload)
            if motor_names is not None and "motor_names" not in found:
                found["motor_names"] = motor_names
                changed = True
            if action_dim is not None and "action_dim" not in found:
                found["action_dim"] = action_dim
                changed = True

        # Normalization stats — look for min/max arrays under common stat keys
        if "normalization_stats" not in found and isinstance(payload, dict):
            for stats_key in ("stats", "normalization_stats", "norm_stats"):
                stats_section = payload.get(stats_key)
                if not isinstance(stats_section, dict):
                    continue
                # LeRobot stores stats under "observation.state" or "action"
                for obs_key in ("observation.state", "action", "state"):
                    obs_stats = stats_section.get(obs_key)
                    if isinstance(obs_stats, dict):
                        if isinstance(obs_stats.get("min"), list) and isinstance(obs_stats.get("max"), list):
                            found["normalization_stats"] = obs_stats
                            changed = True
                            break
                if "normalization_stats" in found:
                    break

        if changed and json_path.name not in sources:
            sources.append(json_path.name)

        if set(found.keys()) >= _all_done:
            break

    if not found:
        return None, "could not extract training config fields from model JSON files"

    return found, f"extracted from: {', '.join(sources[:4])}"


def _find_robot_calibration_path(
    config: dict[str, Any],
    *,
    robot_id: str = _DEPLOY_ROBOT_ID,
    robot_type: str = _DEPLOY_ROBOT_TYPE,
    config_key: str = "follower_calibration_path",
) -> Path | None:
    """Return the calibration JSON path for a robot, or None.

    Precedence:
    1. Explicit ``config[config_key]`` if set and points to an existing file.
    2. Auto-discovery by probing conventional LeRobot locations for *robot_id*.
    3. ``None`` if no file found.
    """
    # ── User-specified calibration file override ──
    user_calib = str(config.get(config_key, "")).strip()
    # Also check legacy single key for backward compat
    if not user_calib and config_key == "follower_calibration_path":
        user_calib = str(config.get("calibration_path", "")).strip()
    if user_calib:
        try:
            user_path = Path(normalize_path(user_calib))
            if user_path.is_file():
                return user_path
        except (OSError, ValueError):
            pass  # fall through to auto-discovery

    # ── Auto-discovery ──
    lerobot_dir = str(config.get("lerobot_dir", "~/lerobot"))
    candidates = [
        # Modern LeRobot: .cache/huggingface/lerobot/calibration/<type>/<id>.json
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / robot_type
        / f"{robot_id}.json",
        # Also checked without the type subdirectory
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / f"{robot_id}.json",
        # lerobot_dir/calibration/<id>.json
        Path(normalize_path(lerobot_dir)) / "calibration" / f"{robot_id}.json",
        # lerobot_dir/.cache/calibration/<id>.json
        Path(normalize_path(lerobot_dir)) / ".cache" / "calibration" / f"{robot_id}.json",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _extract_calibration_motor_names(calib_path: Path) -> list[str] | None:
    """Extract ordered motor/joint names from a LeRobot calibration JSON file.

    Calibration files are dicts keyed by joint name; meta keys beginning with
    ``"_"`` are skipped.  Returns ``None`` if the file cannot be read or parsed.
    """
    try:
        payload = json.loads(calib_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    names = [k for k in payload.keys() if k and not k.startswith("_")]
    return names if names else None


def _is_suspicious_float(value: Any) -> bool:
    """Return True if *value* is inf, -inf, or NaN — signs of corrupt calibration data."""
    try:
        f = float(value)
        return math.isinf(f) or math.isnan(f)
    except (TypeError, ValueError):
        return False


def _validate_calibration_array_format(payload: dict[str, Any]) -> list[CheckResult]:
    """Validate old-style LeRobot calibration that stores per-motor data in parallel arrays."""
    checks: list[CheckResult] = []

    motor_names: list[Any] = payload.get("motor_names", []) or []
    homing_offsets: list[Any] = payload.get("homing_offset", []) or []
    drive_modes: list[Any] = payload.get("drive_mode", []) or []

    n = len(motor_names) or len(homing_offsets)
    if n == 0:
        checks.append(("WARN", "Calibration motor count", "old-format file has no motor_names or homing_offset array"))
        return checks

    checks.append(("PASS", "Calibration format", f"old array format — {n} motors"))

    # Array length consistency
    for arr_name, arr in (("homing_offset", homing_offsets), ("drive_mode", drive_modes)):
        if isinstance(arr, list) and arr and len(arr) != n:
            checks.append((
                "WARN",
                "Calibration array lengths",
                f"motor_names has {n} entries but {arr_name} has {len(arr)} — file may be truncated",
            ))

    bad_inf: list[str] = []
    bad_offset: list[str] = []
    bad_drive: list[str] = []

    for i, offset in enumerate(homing_offsets if isinstance(homing_offsets, list) else []):
        name = str(motor_names[i]) if i < len(motor_names) else f"motor_{i}"
        if _is_suspicious_float(offset):
            bad_inf.append(name)
        elif isinstance(offset, (int, float)) and abs(offset) > _CALIB_HOMING_OFFSET_BOUND:
            bad_offset.append(f"{name}={int(offset)}")

    for i, dm in enumerate(drive_modes if isinstance(drive_modes, list) else []):
        name = str(motor_names[i]) if i < len(motor_names) else f"motor_{i}"
        if dm not in _CALIB_DRIVE_MODE_VALID:
            bad_drive.append(f"{name}={dm}")

    if bad_inf:
        checks.append((
            "FAIL",
            "Calibration inf/NaN offsets",
            f"inf or NaN in homing_offset for: {bad_inf} — recalibrate immediately; robot will move dangerously",
        ))
    if bad_offset:
        checks.append((
            "WARN",
            "Calibration homing offsets",
            f"unusually large offsets suggest motor slip or uint16 wrap-around: {bad_offset[:4]}",
        ))
    if not bad_inf and not bad_offset:
        checks.append(("PASS", "Calibration homing offsets", "all within expected range"))

    if bad_drive:
        checks.append((
            "FAIL",
            "Calibration drive modes",
            f"drive_mode must be 0 or 1 — invalid values will invert motor direction: {bad_drive[:4]}",
        ))
    elif isinstance(drive_modes, list) and drive_modes:
        checks.append(("PASS", "Calibration drive modes", "all 0 or 1"))

    return checks


def _validate_calibration_per_motor_format(
    payload: dict[str, Any],
    model_config_fields: dict[str, Any] | None,
) -> list[CheckResult]:
    """Validate new-style LeRobot calibration where each joint is a named object."""
    checks: list[CheckResult] = []

    motor_entries = {
        k: v for k, v in payload.items() if isinstance(v, dict) and not k.startswith("_")
    }

    if not motor_entries:
        checks.append(("WARN", "Calibration format", "new per-motor format detected but no motor objects found"))
        return checks

    n = len(motor_entries)
    checks.append(("PASS", "Calibration format", f"new per-motor format — {n} motors"))

    bad_inf: list[str] = []
    bad_drive: list[str] = []
    bad_offset: list[str] = []
    bad_range: list[str] = []
    narrow_range: list[str] = []
    seen_ids: dict[int, str] = {}
    duplicate_ids: list[str] = []

    for motor_name, motor_data in motor_entries.items():
        if not isinstance(motor_data, dict):
            continue

        # inf / NaN in any field
        for field, val in motor_data.items():
            if _is_suspicious_float(val):
                bad_inf.append(f"{motor_name}.{field}={val}")

        # drive_mode
        dm = motor_data.get("drive_mode")
        if dm is not None and not _is_suspicious_float(dm) and dm not in _CALIB_DRIVE_MODE_VALID:
            bad_drive.append(f"{motor_name}={dm}")

        # homing_offset
        ho = motor_data.get("homing_offset")
        if ho is not None and not _is_suspicious_float(ho):
            if isinstance(ho, (int, float)) and abs(ho) > _CALIB_HOMING_OFFSET_BOUND:
                bad_offset.append(f"{motor_name}={int(ho)}")

        # range_min / range_max
        rmin = motor_data.get("range_min")
        rmax = motor_data.get("range_max")
        if rmin is not None and rmax is not None:
            if _is_suspicious_float(rmin) or _is_suspicious_float(rmax):
                pass  # already caught by inf check above
            elif not isinstance(rmin, (int, float)) or not isinstance(rmax, (int, float)):
                pass
            elif rmin >= rmax:
                bad_range.append(f"{motor_name}: range_min={int(rmin)} >= range_max={int(rmax)}")
            elif (rmax - rmin) < _CALIB_MIN_RANGE_TICKS:
                narrow_range.append(f"{motor_name}: only {int(rmax - rmin)} ticks wide")

        # Motor ID uniqueness
        mid = motor_data.get("id")
        if isinstance(mid, int):
            if mid in seen_ids:
                duplicate_ids.append(f"{motor_name} and {seen_ids[mid]} both id={mid}")
            else:
                seen_ids[mid] = motor_name

    if bad_inf:
        checks.append((
            "FAIL",
            "Calibration inf/NaN values",
            (
                f"inf or NaN in calibration fields: {', '.join(bad_inf[:4])} — "
                "motor will move to extreme position and may damage hardware; recalibrate immediately"
            ),
        ))

    if bad_drive:
        checks.append((
            "FAIL",
            "Calibration drive modes",
            (
                f"drive_mode must be 0 or 1 — wrong value inverts motor direction and will cause collisions: "
                f"{', '.join(bad_drive[:4])}"
            ),
        ))
    else:
        checks.append(("PASS", "Calibration drive modes", "all 0 or 1"))

    if bad_offset:
        checks.append((
            "WARN",
            "Calibration homing offsets",
            (
                f"homing_offset > {_CALIB_HOMING_OFFSET_BOUND} suggests motor slip or uint16 wrap-around "
                f"(see LeRobot issue #1342): {', '.join(bad_offset[:4])}"
            ),
        ))
    elif not bad_inf:
        checks.append(("PASS", "Calibration homing offsets", "all within expected range"))

    if bad_range:
        checks.append((
            "FAIL",
            "Calibration motor ranges",
            (
                f"range_min >= range_max makes normalisation undefined — "
                f"robot will output NaN actions: {'; '.join(bad_range[:4])}"
            ),
        ))
    elif narrow_range:
        checks.append((
            "WARN",
            "Calibration motor ranges",
            (
                f"suspiciously narrow joint range (< {_CALIB_MIN_RANGE_TICKS} ticks) — "
                f"arm may not have been moved through full range during calibration: "
                f"{', '.join(narrow_range[:4])}"
            ),
        ))
    else:
        checks.append(("PASS", "Calibration motor ranges", "all range_min < range_max with reasonable width"))

    if duplicate_ids:
        checks.append((
            "FAIL",
            "Calibration motor IDs",
            f"duplicate motor IDs will cause bus collisions: {'; '.join(duplicate_ids[:4])}",
        ))
    elif seen_ids:
        checks.append((
            "PASS",
            "Calibration motor IDs",
            f"IDs {sorted(seen_ids.keys())} are unique",
        ))

    # Normalization drift: compare calibration ranges against model's observed stats
    if model_config_fields is not None:
        model_motor_names: list[str] | None = model_config_fields.get("motor_names")
        model_stats: dict[str, Any] | None = model_config_fields.get("normalization_stats")
        if model_motor_names and model_stats and motor_entries:
            drift_issues: list[str] = []
            for i, joint_name in enumerate(model_motor_names):
                calib = motor_entries.get(joint_name)
                if calib is None:
                    continue
                rmin = calib.get("range_min")
                rmax = calib.get("range_max")
                if not isinstance(rmin, (int, float)) or not isinstance(rmax, (int, float)):
                    continue
                calib_width = rmax - rmin
                # Model stats store per-joint min/max as flat lists (index = joint index)
                stat_min_list = model_stats.get("min")
                stat_max_list = model_stats.get("max")
                if isinstance(stat_min_list, list) and isinstance(stat_max_list, list):
                    if i < len(stat_min_list) and i < len(stat_max_list):
                        try:
                            stat_width = float(stat_max_list[i]) - float(stat_min_list[i])
                        except (TypeError, ValueError):
                            continue
                        if stat_width > 0 and calib_width > 0:
                            ratio = calib_width / stat_width
                            # Flag if calibration range is more than 40% wider or narrower than training
                            if ratio < 0.6 or ratio > 1.6:
                                drift_issues.append(
                                    f"{joint_name}: calib={calib_width:.0f}ticks vs train={stat_width:.0f}ticks "
                                    f"({ratio:.1f}x)"
                                )
            if drift_issues:
                checks.append((
                    "WARN",
                    "Calibration vs training normalization",
                    (
                        "joint ranges differ significantly from training-time observations — "
                        "policy outputs may be poorly scaled: "
                        + "; ".join(drift_issues[:4])
                    ),
                ))
            elif model_motor_names and model_stats:
                checks.append((
                    "PASS",
                    "Calibration vs training normalization",
                    "calibration joint ranges consistent with training-time observations",
                ))
        elif model_motor_names and not model_stats and motor_entries:
            # We have a calibration file and know the motor names, but the model JSON did
            # not include normalization_stats.  We cannot numerically verify calibration
            # drift, but we can warn the user: if the robot was recalibrated after the
            # training dataset was collected the model will act in the wrong position space.
            checks.append((
                "WARN",
                "Calibration vs training normalization",
                (
                    "model checkpoint does not embed normalization stats — cannot verify "
                    "that current calibration matches training-time calibration. "
                    "If the robot has been recalibrated since the dataset was recorded, "
                    "motor positions will normalize to different values than the model expects, "
                    "causing degraded or unpredictable policy behaviour. "
                    "Re-record a dataset with the current calibration, or restore the "
                    "original calibration file used during training."
                ),
            ))

    return checks


def _validate_calibration_values(
    calib_path: Path,
    model_config_fields: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Deep-validate a robot calibration JSON file.

    Handles both old (parallel-array) and new (per-motor object) LeRobot formats.

    Checks:
    - inf/NaN in any field (causes undefined motor moves / hardware damage)
    - drive_mode values are 0 or 1 (wrong value inverts joint direction)
    - homing_offset within ±8192 (large values suggest slip or uint16 wrap)
    - range_min < range_max (inverted range makes normalisation undefined)
    - Joint range ≥ 200 ticks (suspiciously narrow → incomplete calibration sweep)
    - Motor IDs unique (duplicates cause bus collisions)
    - Calibration joint ranges consistent with model normalization stats (if available)
    """
    try:
        payload = json.loads(calib_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [("FAIL", "Calibration file parse", f"cannot read/parse {calib_path.name}: {exc}")]

    if not isinstance(payload, dict):
        return [("FAIL", "Calibration file format", "calibration JSON root is not an object")]

    # Distinguish old (array) from new (per-motor) format by the type of "homing_offset"
    if isinstance(payload.get("homing_offset"), list):
        return _validate_calibration_array_format(payload)
    return _validate_calibration_per_motor_format(payload, model_config_fields)


def _check_robot_calibration(
    config: dict[str, Any],
    *,
    robot_id: str,
    robot_type: str,
    config_key: str,
    label: str,
    model_config_fields: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Run full calibration checks for a single robot (find → validate → compare).

    *label* is a human-readable prefix such as ``"Follower"`` or ``"Leader"``
    prepended to each check name so users can tell which robot a check belongs to.
    """
    checks: list[CheckResult] = []
    calib_path = _find_robot_calibration_path(
        config, robot_id=robot_id, robot_type=robot_type, config_key=config_key,
    )
    if calib_path is None:
        checks.append((
            "WARN",
            f"{label} calibration file",
            (
                f"no calibration file found for robot '{robot_id}' ({robot_type}); "
                "run LeRobot calibration before running to ensure motor offsets are current"
            ),
        ))
        return checks

    checks.append(("PASS", f"{label} calibration file", str(calib_path)))
    calib_motor_names = _extract_calibration_motor_names(calib_path)
    if calib_motor_names is None:
        checks.append(("WARN", f"{label} calibration motors", f"could not parse motor names from {calib_path.name}"))
    else:
        checks.append(("PASS", f"{label} calibration motors", f"{len(calib_motor_names)} joints: {', '.join(calib_motor_names)}"))

        # Compare calibration motors with model's expected motor names (if available)
        model_motor_names = (model_config_fields or {}).get("motor_names")
        if model_motor_names is not None:
            calib_set = set(calib_motor_names)
            model_set = set(model_motor_names)
            missing_from_calib = sorted(model_set - calib_set)
            extra_in_calib = sorted(calib_set - model_set)
            if not missing_from_calib and not extra_in_calib:
                checks.append(("PASS", f"{label} model vs calibration motors", f"all {len(model_motor_names)} joint names match"))
            else:
                detail_parts = []
                if missing_from_calib:
                    detail_parts.append(f"missing from calibration: {missing_from_calib}")
                if extra_in_calib:
                    detail_parts.append(f"extra in calibration: {extra_in_calib}")
                checks.append((
                    "FAIL",
                    f"{label} model vs calibration motors",
                    "; ".join(detail_parts) + " — motor name mismatch will produce wrong joint actions",
                ))

    # Deep calibration validation — values that make the robot run badly
    calib_value_checks = _validate_calibration_values(calib_path, model_config_fields)
    checks.extend(calib_value_checks)

    return checks


def _collect_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned} if cleaned else set()
    if isinstance(value, list):
        items: set[str] = set()
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.add(cleaned)
        return items
    return set()


def _looks_like_camera_name(name: str) -> bool:
    normalized = name.strip()
    if not normalized:
        return False
    if normalized.lower() in _CAMERA_NAME_BLOCKLIST:
        return False
    return _CAMERA_NAME_PATTERN.match(normalized) is not None


def _collect_camera_keys_from_json(payload: Any, out: set[str]) -> None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).lower()
            if key in {"camera_keys", "image_keys", "observation_image_keys", "observation_camera_keys"}:
                out.update({item for item in _collect_strings(value) if _looks_like_camera_name(item)})
            elif key in {"cameras", "images"} and isinstance(value, dict):
                out.update({name for name in value.keys() if _looks_like_camera_name(str(name))})
            _collect_camera_keys_from_json(value, out)
        return

    if isinstance(payload, list):
        for item in payload:
            _collect_camera_keys_from_json(item, out)


def _extract_model_camera_keys(model_path: Path) -> tuple[set[str] | None, str]:
    if not model_path.exists() or not model_path.is_dir():
        return None, f"model path is not readable: {model_path}"

    json_files = sorted(path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    if not json_files:
        return None, "no JSON metadata files found in model payload"

    discovered: set[str] = set()
    source_files: list[str] = []

    for json_path in json_files[:12]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        before = len(discovered)
        _collect_camera_keys_from_json(payload, discovered)
        if len(discovered) > before:
            source_files.append(json_path.name)

    if not discovered:
        return None, "could not infer camera keys from model metadata JSON files"

    return discovered, f"detected via: {', '.join(source_files[:4])}"


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


def _run_common_preflight_checks(config: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def add(level: str, name: str, detail: str) -> None:
        checks.append((level, name, detail))

    lerobot_dir = get_lerobot_dir(config)
    add(
        "PASS" if lerobot_dir.exists() else "FAIL",
        "LeRobot folder",
        str(lerobot_dir),
    )

    lerobot_ok, lerobot_msg = probe_module_import("lerobot")
    add(
        "PASS" if lerobot_ok else "FAIL",
        "Python module: lerobot",
        "import ok" if lerobot_ok else summarize_probe_error(lerobot_msg),
    )

    # Motor driver SDK — required for SO-100 / Feetech servo arms.
    scs_ok, scs_msg = probe_module_import("scservo_sdk")
    add(
        "PASS" if scs_ok else "FAIL",
        "Python module: scservo_sdk",
        "import ok" if scs_ok else (
            summarize_probe_error(scs_msg)
            + " — Fix: pip install feetech-servo-sdk  (required for Feetech SO-100 arms)"
        ),
    )

    add(
        "PASS" if in_virtual_env() else "WARN",
        "Python environment",
        f"executable={sys.executable}",
    )

    _append_robot_port_checks(config, add)
    _append_camera_checks(config, add)

    return checks


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


def _probe_policy_path_support() -> CheckResult:
    timeout_s = 20
    cmd_variants = (
        [sys.executable, "-m", "lerobot.scripts.lerobot_record", "--help"],
        [sys.executable, "-m", "lerobot.scripts.lerobot_record", "-h"],
    )
    saw_timeout = False
    for cmd in cmd_variants:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            saw_timeout = True
            continue
        except Exception as exc:
            return ("WARN", "lerobot_record policy flag", f"Unable to probe help output: {exc}")

        text = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
        if "--policy.path" in text:
            return ("PASS", "lerobot_record policy flag", "--policy.path supported")
        if "modulenotfounderror" in text and "lerobot" in text:
            return ("FAIL", "lerobot_record policy flag", "lerobot module not importable in active env")

    if saw_timeout:
        return (
            "WARN",
            "lerobot_record policy flag",
            f"Help probe timed out after {timeout_s}s; skipping policy flag verification (non-blocking).",
        )

    return (
        "WARN",
        "lerobot_record policy flag",
        "Could not confirm '--policy.path' in lerobot_record help output (non-blocking).",
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
        robot_id=_DEPLOY_ROBOT_ID, robot_type=_DEPLOY_ROBOT_TYPE,
        config_key="follower_calibration_path", label="Follower",
    ))
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_LEADER_ROBOT_ID, robot_type=_LEADER_ROBOT_TYPE,
        config_key="leader_calibration_path", label="Leader",
    ))

    return checks


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
        robot_id=_DEPLOY_ROBOT_ID, robot_type=_DEPLOY_ROBOT_TYPE,
        config_key="follower_calibration_path", label="Follower",
    ))
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_LEADER_ROBOT_ID, robot_type=_LEADER_ROBOT_TYPE,
        config_key="leader_calibration_path", label="Leader",
    ))

    return checks


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)

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

    is_valid_model, detail, candidates = validate_model_path(model_path)
    checks.append(("PASS" if model_path.exists() and model_path.is_dir() else "FAIL", "Model folder", str(model_path)))
    checks.append(("PASS" if is_valid_model else "FAIL", "Model payload", detail))
    if candidates:
        checks.append(
            (
                "WARN",
                "Model payload candidates",
                ", ".join(str(path) for path in candidates[:3]),
            )
        )

    expected_camera_keys = {"laptop", "phone"}
    model_camera_keys, camera_key_detail = _extract_model_camera_keys(model_path)
    if model_camera_keys is None:
        checks.append(("WARN", "Model camera keys", camera_key_detail))
    elif model_camera_keys == expected_camera_keys:
        checks.append(("PASS", "Model camera keys", f"matches runtime keys {sorted(expected_camera_keys)}"))
    else:
        checks.append(
            (
                "FAIL",
                "Model camera keys",
                (
                    f"model={sorted(model_camera_keys)}; runtime={sorted(expected_camera_keys)}; "
                    "camera key mismatch between training and deployment"
                ),
            )
        )

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
        if model_robot_type is None:
            checks.append(("WARN", "Training vs deploy robot type", f"robot_type not found in model metadata ({model_config_source})"))
        elif model_robot_type == _DEPLOY_ROBOT_TYPE:
            checks.append(("PASS", "Training vs deploy robot type", f"match: {_DEPLOY_ROBOT_TYPE}"))
        else:
            checks.append((
                "FAIL",
                "Training vs deploy robot type",
                (
                    f"model trained on '{model_robot_type}', deploying to '{_DEPLOY_ROBOT_TYPE}'; "
                    "robot type mismatch will cause action space errors at runtime"
                ),
            ))

        # Action dimension check
        model_action_dim = model_config_fields.get("action_dim")
        if model_action_dim is not None:
            if model_action_dim == _DEPLOY_ROBOT_MOTOR_COUNT:
                checks.append(("PASS", "Training vs deploy action dim", f"match: {_DEPLOY_ROBOT_MOTOR_COUNT} DOF"))
            else:
                checks.append((
                    "FAIL",
                    "Training vs deploy action dim",
                    (
                        f"model outputs {model_action_dim} actions but {_DEPLOY_ROBOT_TYPE} "
                        f"has {_DEPLOY_ROBOT_MOTOR_COUNT} DOF; shape mismatch will crash at inference"
                    ),
                ))

    # ------------------------------------------------------------------ #
    # Robot calibration — follower and leader                             #
    # ------------------------------------------------------------------ #
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_DEPLOY_ROBOT_ID, robot_type=_DEPLOY_ROBOT_TYPE,
        config_key="follower_calibration_path", label="Follower",
        model_config_fields=model_config_fields,
    ))
    checks.extend(_check_robot_calibration(
        config,
        robot_id=_LEADER_ROBOT_ID, robot_type=_LEADER_ROBOT_TYPE,
        config_key="leader_calibration_path", label="Leader",
    ))

    checks.append(_probe_policy_path_support())

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


def collect_doctor_checks(config: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def add(level: str, name: str, detail: str) -> None:
        checks.append((level, name, detail))

    add("PASS", "Python", f"{sys.executable} ({sys.version.split()[0]})")

    lerobot_dir = get_lerobot_dir(config)
    add(
        "PASS" if lerobot_dir.exists() else "FAIL",
        "LeRobot folder",
        str(lerobot_dir),
    )

    record_data_dir = Path(normalize_path(config.get("record_data_dir", "")))
    add(
        "PASS" if record_data_dir.exists() else "WARN",
        "Record data dir",
        str(record_data_dir),
    )

    deploy_data_dir = get_deploy_data_dir(config)
    add(
        "PASS" if deploy_data_dir.exists() else "WARN",
        "Deploy data dir",
        str(deploy_data_dir),
    )

    models_dir = Path(normalize_path(config.get("trained_models_dir", "")))
    add(
        "PASS" if models_dir.exists() else "WARN",
        "Trained models dir",
        str(models_dir),
    )

    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    add(
        "PASS" if runs_dir.exists() else "WARN",
        "Runs artifacts dir",
        str(runs_dir),
    )

    next_record_name = increment_dataset_name(repo_name_from_repo_id(str(config.get("last_dataset_name", "dataset_1"))))
    record_targets = [record_data_dir / next_record_name, lerobot_dir / "data" / next_record_name]
    existing_record_targets = [str(path) for path in record_targets if path.exists()]
    add(
        "WARN" if existing_record_targets else "PASS",
        "Next record dataset collision",
        ", ".join(existing_record_targets)
        if existing_record_targets
        else f"{next_record_name} is free in record/lerobot data targets",
    )

    next_eval_name = repo_name_from_repo_id(
        suggest_eval_dataset_name(config, str(config.get("last_model_name", "")))
    )
    eval_targets = [deploy_data_dir / next_eval_name, lerobot_dir / "data" / next_eval_name]
    existing_eval_targets = [str(path) for path in eval_targets if path.exists()]
    add(
        "WARN" if existing_eval_targets else "PASS",
        "Next eval dataset collision",
        ", ".join(existing_eval_targets)
        if existing_eval_targets
        else f"{next_eval_name} is free in deploy/lerobot data targets",
    )

    huggingface_cli = shutil.which("huggingface-cli")
    add(
        "PASS" if huggingface_cli else "WARN",
        "huggingface-cli",
        huggingface_cli or "not found in PATH",
    )

    common_checks = _run_common_preflight_checks(config)
    for level, name, detail in common_checks:
        if name == "LeRobot folder":
            continue
        add(level, name, detail)

    return checks
