from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .types import CheckResult

_CALIB_DRIVE_MODE_VALID = frozenset({0, 1})
_CALIB_HOMING_OFFSET_BOUND = 8192
_CALIB_MIN_RANGE_TICKS = 200


def _extract_calibration_motor_names(calib_path: Path) -> list[str] | None:
    """Extract ordered motor/joint names from a LeRobot calibration JSON file."""
    try:
        payload = json.loads(calib_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    names = [key for key in payload.keys() if key and not key.startswith("_")]
    return names if names else None


def _is_suspicious_float(value: Any) -> bool:
    """Return True if *value* is inf, -inf, or NaN."""
    try:
        numeric = float(value)
        return math.isinf(numeric) or math.isnan(numeric)
    except (TypeError, ValueError):
        return False


def _validate_calibration_array_format(payload: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    motor_names: list[Any] = payload.get("motor_names", []) or []
    homing_offsets: list[Any] = payload.get("homing_offset", []) or []
    drive_modes: list[Any] = payload.get("drive_mode", []) or []

    n = len(motor_names) or len(homing_offsets)
    if n == 0:
        checks.append(("WARN", "Calibration motor count", "old-format file has no motor_names or homing_offset array"))
        return checks

    checks.append(("PASS", "Calibration format", f"old array format — {n} motors"))

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

    for index, offset in enumerate(homing_offsets if isinstance(homing_offsets, list) else []):
        name = str(motor_names[index]) if index < len(motor_names) else f"motor_{index}"
        if _is_suspicious_float(offset):
            bad_inf.append(name)
        elif isinstance(offset, (int, float)) and abs(offset) > _CALIB_HOMING_OFFSET_BOUND:
            bad_offset.append(f"{name}={int(offset)}")

    for index, drive_mode in enumerate(drive_modes if isinstance(drive_modes, list) else []):
        name = str(motor_names[index]) if index < len(motor_names) else f"motor_{index}"
        if drive_mode not in _CALIB_DRIVE_MODE_VALID:
            bad_drive.append(f"{name}={drive_mode}")

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
    checks: list[CheckResult] = []

    motor_entries = {
        key: value for key, value in payload.items() if isinstance(value, dict) and not key.startswith("_")
    }

    if not motor_entries:
        checks.append(("WARN", "Calibration format", "new per-motor format detected but no motor objects found"))
        return checks

    checks.append(("PASS", "Calibration format", f"new per-motor format — {len(motor_entries)} motors"))

    bad_inf: list[str] = []
    bad_drive: list[str] = []
    bad_offset: list[str] = []
    bad_range: list[str] = []
    narrow_range: list[str] = []
    seen_ids: dict[int, str] = {}
    duplicate_ids: list[str] = []

    for motor_name, motor_data in motor_entries.items():
        for field, value in motor_data.items():
            if _is_suspicious_float(value):
                bad_inf.append(f"{motor_name}.{field}={value}")

        drive_mode = motor_data.get("drive_mode")
        if drive_mode is not None and not _is_suspicious_float(drive_mode) and drive_mode not in _CALIB_DRIVE_MODE_VALID:
            bad_drive.append(f"{motor_name}={drive_mode}")

        homing_offset = motor_data.get("homing_offset")
        if homing_offset is not None and not _is_suspicious_float(homing_offset):
            if isinstance(homing_offset, (int, float)) and abs(homing_offset) > _CALIB_HOMING_OFFSET_BOUND:
                bad_offset.append(f"{motor_name}={int(homing_offset)}")

        range_min = motor_data.get("range_min")
        range_max = motor_data.get("range_max")
        if range_min is not None and range_max is not None:
            if _is_suspicious_float(range_min) or _is_suspicious_float(range_max):
                pass
            elif not isinstance(range_min, (int, float)) or not isinstance(range_max, (int, float)):
                pass
            elif range_min >= range_max:
                bad_range.append(f"{motor_name}: range_min={int(range_min)} >= range_max={int(range_max)}")
            elif (range_max - range_min) < _CALIB_MIN_RANGE_TICKS:
                narrow_range.append(f"{motor_name}: only {int(range_max - range_min)} ticks wide")

        motor_id = motor_data.get("id")
        if isinstance(motor_id, int):
            if motor_id in seen_ids:
                duplicate_ids.append(f"{motor_name} and {seen_ids[motor_id]} both id={motor_id}")
            else:
                seen_ids[motor_id] = motor_name

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
        checks.append(("PASS", "Calibration motor IDs", f"IDs {sorted(seen_ids.keys())} are unique"))

    if model_config_fields is not None:
        model_motor_names: list[str] | None = model_config_fields.get("motor_names")
        model_stats: dict[str, Any] | None = model_config_fields.get("normalization_stats")
        if model_motor_names and model_stats and motor_entries:
            drift_issues: list[str] = []
            for index, joint_name in enumerate(model_motor_names):
                calibration_entry = motor_entries.get(joint_name)
                if calibration_entry is None:
                    continue
                range_min = calibration_entry.get("range_min")
                range_max = calibration_entry.get("range_max")
                if not isinstance(range_min, (int, float)) or not isinstance(range_max, (int, float)):
                    continue
                calibration_width = range_max - range_min
                stat_min_list = model_stats.get("min")
                stat_max_list = model_stats.get("max")
                if isinstance(stat_min_list, list) and isinstance(stat_max_list, list):
                    if index < len(stat_min_list) and index < len(stat_max_list):
                        try:
                            stat_width = float(stat_max_list[index]) - float(stat_min_list[index])
                        except (TypeError, ValueError):
                            continue
                        if stat_width > 0 and calibration_width > 0:
                            ratio = calibration_width / stat_width
                            if ratio < 0.6 or ratio > 1.6:
                                drift_issues.append(
                                    f"{joint_name}: calib={calibration_width:.0f}ticks vs train={stat_width:.0f}ticks "
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
            else:
                checks.append((
                    "PASS",
                    "Calibration vs training normalization",
                    "calibration joint ranges consistent with training-time observations",
                ))
        elif model_motor_names and not model_stats and motor_entries:
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
    try:
        payload = json.loads(calib_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [("FAIL", "Calibration file parse", f"cannot read/parse {calib_path.name}: {exc}")]

    if not isinstance(payload, dict):
        return [("FAIL", "Calibration file format", "calibration JSON root is not an object")]

    if isinstance(payload.get("homing_offset"), list):
        return _validate_calibration_array_format(payload)
    return _validate_calibration_per_motor_format(payload, model_config_fields)
