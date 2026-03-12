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
        diagnostics=checks_to_events(checks),
    )


def summarize_checks(checks: list[CheckResult], title: str = "Checks") -> str:
    pass_count, warn_count, fail_count = _check_counts(checks)
    lines = [title]
    events = checks_to_events(checks)
    for (level, name, detail), event in zip(checks, events):
        lines.append(f"[{level:4}] {event.code} {name}: {detail}")
    lines.append("")
    lines.append(f"Summary: PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return "\n".join(lines)


def has_failures(checks: list[CheckResult]) -> bool:
    return any(level == "FAIL" for level, _, _ in checks)


def diagnostics_from_checks(checks: list[CheckResult]) -> list[DiagnosticEvent]:
    return checks_to_events(checks)


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current == current.parent:
            return None
        current = current.parent
    return current


def _follower_robot_id(config: dict[str, Any]) -> str:
    value = str(config.get("follower_robot_id", "")).strip()
    inferred = _robot_id_from_calibration_selection(
        config.get("follower_calibration_path")
    ) or _robot_id_from_calibration_selection(config.get("calibration_path"))
    if inferred and (not value or value == _DEFAULT_FOLLOWER_ROBOT_ID):
        return inferred
    return value or _DEFAULT_FOLLOWER_ROBOT_ID


def _leader_robot_id(config: dict[str, Any]) -> str:
    value = str(config.get("leader_robot_id", "")).strip()
    inferred = _robot_id_from_calibration_selection(config.get("leader_calibration_path"))
    if inferred and (not value or value == _DEFAULT_LEADER_ROBOT_ID):
        return inferred
    return value or _DEFAULT_LEADER_ROBOT_ID


def _robot_id_from_calibration_selection(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw in {".", "./"}:
        return None
    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() != ".json":
        return None
    stem = candidate.stem.strip()
    return stem or None


def _configured_env_dir(config: dict[str, Any]) -> Path:
    return Path(
        normalize_path(
            str(config.get("lerobot_venv_dir", get_lerobot_dir(config) / "lerobot_env"))
        )
    )


def _activation_config_check(config: dict[str, Any]) -> tuple[str, str]:
    custom_activate_cmd = str(config.get("setup_venv_activate_cmd", "")).strip()
    if custom_activate_cmd:
        return (
            "PASS",
            f"custom activation command configured ({custom_activate_cmd})",
        )

    configured_env_dir = _configured_env_dir(config)
    activate_script = configured_env_dir / "bin" / "activate"
    if activate_script.is_file():
        return ("PASS", f"activate script found at {activate_script}")

    conda_meta = configured_env_dir / "conda-meta"
    if conda_meta.is_dir():
        return (
            "PASS",
            (
                f"conda environment folder detected at {configured_env_dir} "
                "(conda-meta present)"
            ),
        )

    return (
        "FAIL",
        (
            f"missing activate script at {activate_script}. "
            "Set 'LeRobot venv folder path' or 'setup_venv_activate_cmd' in Config."
        ),
    )


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
        if has_rw:
            access_detail = "read/write ok"
        elif sys.platform.startswith("linux"):
            access_detail = f"permission denied for {port} (Linux fix: add user to dialout/uucp or apply udev ACL rule)"
        elif sys.platform == "darwin":
            access_detail = (
                f"permission denied for {port} "
                "(macOS: prefer /dev/cu.* ports and verify terminal access under Privacy & Security)"
            )
        else:
            access_detail = f"permission denied for {port}"
        add(
            "PASS" if has_rw else "FAIL",
            f"{label} access",
            access_detail,
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

    linux_serial_ports = [
        port
        for port in (follower_port, leader_port)
        if port and any(token in port.lower() for token in ("ttyacm", "ttyusb", "/dev/serial/by-id/"))
    ]
    if linux_serial_ports:
        in_dialout, dialout_detail = _dialout_membership()
        if in_dialout is True:
            add("PASS", "dialout group", dialout_detail)
        elif in_dialout is False:
            # If user is NOT in dialout but all ports are R/W accessible
            # (e.g. via udev rules, ACLs, or chmod), this is fine — downgrade to WARN.
            all_ports_accessible = all(
                os.access(p, os.R_OK | os.W_OK)
                for p in linux_serial_ports
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
    schema = resolve_camera_schema(config)
    if schema.errors:
        for message in schema.errors:
            add("FAIL", "Camera schema", message)
    if schema.warnings:
        for message in schema.warnings:
            add("WARN", "Camera schema", message)
    if not schema.specs:
        add("FAIL", "Camera schema", "no cameras configured; set camera_schema_json or legacy camera indices in Config.")
        return

    names = [spec.name for spec in schema.specs]
    add("PASS", "Camera schema", f"{len(schema.specs)} camera(s): {', '.join(names)}")

    source_counts: dict[str, int] = {}
    for spec in schema.specs:
        source_key = str(spec.source)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
    duplicate_sources = sorted(source for source, count in source_counts.items() if count > 1)
    if duplicate_sources:
        add(
            "FAIL",
            "Camera source uniqueness",
            f"multiple camera names map to the same source: {duplicate_sources}",
        )
    else:
        add("PASS", "Camera source uniqueness", "all configured camera sources are distinct")

    cv2_ok, cv2_msg = probe_module_import("cv2")
    add(
        "PASS" if cv2_ok else "FAIL",
        "Python module: cv2",
        "import ok" if cv2_ok else summarize_probe_error(cv2_msg),
    )
    if not cv2_ok:
        return

    fingerprints: dict[str, str | None] = {}
    opened_camera_names: list[str] = []
    failed_camera_names: list[str] = []
    for spec in schema.specs:
        target_width = int(spec.width)
        target_height = int(spec.height)
        source_text = str(spec.source)
        opened, probe_detail = probe_camera_capture(spec.source, target_width, target_height)
        if opened:
            opened_camera_names.append(spec.name)
            add("PASS", f"Camera '{spec.name}' probe", probe_detail)
        else:
            failed_camera_names.append(spec.name)
            detail = summarize_probe_error(probe_detail)
            add(
                "FAIL",
                f"Camera '{spec.name}' probe",
                f"{detail}; verify source '{source_text}' exists and camera is connected.",
            )

        actual = parse_frame_dimensions(probe_detail)
        if actual is not None:
            actual_width, actual_height = actual
            if (actual_width, actual_height) == (target_width, target_height):
                add(
                    "PASS",
                    f"Camera '{spec.name}' resolution",
                    f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}",
                )
            else:
                add(
                    "WARN",
                    f"Camera '{spec.name}' resolution",
                    (
                        f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}; "
                        "runtime will auto-detect camera frame size when building the command."
                    ),
                )
        elif opened:
            add(
                "WARN",
                f"Camera '{spec.name}' resolution",
                "opened camera but could not parse detected frame size.",
            )

        fingerprint = camera_fingerprint(spec.source)
        fingerprints[spec.name] = fingerprint
        _append_fingerprint_check(
            add=add,
            check_name=f"Camera '{spec.name}' fingerprint",
            baseline_value=str(config.get(f"camera_{spec.name}_fingerprint", config.get(f"camera_{spec.name.lower()}_fingerprint", ""))),
            current_value=fingerprint,
            missing_detail=f"could not fingerprint camera source '{source_text}' on this platform.",
        )

    fp_to_names: dict[str, list[str]] = {}
    for camera_name, fingerprint in fingerprints.items():
        if not fingerprint:
            continue
        fp_to_names.setdefault(fingerprint, []).append(camera_name)

    collisions = sorted(
        sorted(names)
        for names in fp_to_names.values()
        if len(names) > 1
    )
    if collisions:
        add(
            "FAIL",
            "Camera identity uniqueness",
            f"multiple runtime camera names resolve to the same device fingerprint: {collisions}",
        )
    else:
        add("PASS", "Camera identity uniqueness", "camera fingerprints are distinct for configured sources")

    if len(schema.specs) > 1:
        if failed_camera_names:
            add(
                "FAIL",
                "Configured camera linkage",
                (
                    f"only {len(opened_camera_names)}/{len(schema.specs)} configured cameras opened successfully; "
                    f"missing: {', '.join(failed_camera_names)}. "
                    "Re-scan cameras and assign each configured camera name to a real connected device before running."
                ),
            )
        else:
            add(
                "PASS",
                "Configured camera linkage",
                f"all {len(schema.specs)} configured camera names resolved to real connected cameras",
            )


def _find_robot_calibration_path(
    config: dict[str, Any],
    *,
    robot_id: str = _DEFAULT_FOLLOWER_ROBOT_ID,
    robot_type: str = "so101_follower",
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
    # Compatibility: older config normalization could turn empty optional path
    # fields into "."; treat these sentinel-ish values as empty.
    if user_calib in {".", "./"}:
        user_calib = ""
    if user_calib:
        try:
            user_path = Path(normalize_path(user_calib))
            if user_path.is_file():
                return user_path
            if user_path.is_dir():
                nested = user_path / f"{robot_id}.json"
                if nested.is_file():
                    return nested
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

    if compat_probe_enabled(config):
        for level, name, detail in compatibility_checks(config, include_flag_probe=False):
            add(level, name, detail)
    else:
        add("WARN", "Compatibility probe", "disabled by config (compat_probe_enabled=false)")

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

    env_active = in_virtual_env()
    activation_level, activation_detail = _activation_config_check(config)
    if activation_level == "FAIL" and env_active:
        activation_level = "WARN"
        activation_detail = (
            activation_detail
            + " Current process already has an active environment; update Config to avoid launcher mismatch."
        )
    add(activation_level, "Environment activation", activation_detail)

    configured_env_dir = _configured_env_dir(config)
    activate_script = configured_env_dir / "bin" / "activate"
    if env_active:
        add(
            "PASS",
            "Python environment",
            f"active environment detected (executable={sys.executable})",
        )
    else:
        add(
            "FAIL",
            "Python environment",
            (
                f"no active virtual/conda environment detected (executable={sys.executable}). "
                f"Fix: source {activate_script}  or conda activate <env> before launching GUI."
            ),
        )

    _append_robot_port_checks(config, add)
    _append_camera_checks(config, add)

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


def collect_doctor_events(config: dict[str, Any]) -> list[DiagnosticEvent]:
    return checks_to_events(collect_doctor_checks(config))
