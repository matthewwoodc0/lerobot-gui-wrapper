from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .camera_schema import resolve_camera_schema
from .compat import compatibility_checks
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .checks_calibration import (
    _extract_calibration_motor_names,
    _validate_calibration_values,
)
from .diagnostics_formatters import diagnostics_from_checks
from .feature_flags import compat_probe_enabled
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
    increment_dataset_name,
    repo_name_from_repo_id,
    suggest_eval_dataset_name,
)
from .types import CheckResult, DiagnosticEvent

CommonChecksFn = Callable[[dict[str, Any]], list[CheckResult]]
WhichFn = Callable[[str], Optional[str]]

_DEFAULT_FOLLOWER_ROBOT_ID = "red4"
_DEFAULT_LEADER_ROBOT_ID = "white"
_HEAVY_MODEL_PATTERNS = (
    ("smolvlm", "SmolVLM"),
    ("vision_language", "vision-language"),
    ("vision-language", "vision-language"),
    ("video-instruct", "video-instruct"),
    ("vlm", "VLM"),
)


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


class PreflightChecker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._checks: list[CheckResult] = []

    def _reset(self) -> None:
        self._checks = []

    def _add(self, level: str, name: str, detail: str) -> None:
        self._checks.append((level, name, detail))

    def _append_common_preflight_checks(self, *, include_lerobot_folder: bool) -> None:
        lerobot_dir = get_lerobot_dir(self.config)
        if include_lerobot_folder:
            self._add(
                "PASS" if lerobot_dir.exists() else "FAIL",
                "LeRobot folder",
                str(lerobot_dir),
            )

        if compat_probe_enabled(self.config):
            for level, name, detail in compatibility_checks(self.config, include_flag_probe=False):
                self._add(level, name, detail)
        else:
            self._add("WARN", "Compatibility probe", "disabled by config (compat_probe_enabled=false)")

        lerobot_ok, lerobot_msg = probe_module_import("lerobot")
        self._add(
            "PASS" if lerobot_ok else "FAIL",
            "Python module: lerobot",
            "import ok" if lerobot_ok else summarize_probe_error(lerobot_msg),
        )

        scs_ok, scs_msg = probe_module_import("scservo_sdk")
        self._add(
            "PASS" if scs_ok else "FAIL",
            "Python module: scservo_sdk",
            "import ok" if scs_ok else (
                summarize_probe_error(scs_msg)
                + " — Fix: pip install feetech-servo-sdk  (required for Feetech SO-100 arms)"
            ),
        )

        env_active = in_virtual_env()
        activation_level, activation_detail = _activation_config_check(self.config)
        if activation_level == "FAIL" and env_active:
            activation_level = "WARN"
            activation_detail = (
                activation_detail
                + " Current process already has an active environment; update Config to avoid launcher mismatch."
            )
        self._add(activation_level, "Environment activation", activation_detail)

        configured_env_dir = _configured_env_dir(self.config)
        activate_script = configured_env_dir / "bin" / "activate"
        if env_active:
            self._add(
                "PASS",
                "Python environment",
                f"active environment detected (executable={sys.executable})",
            )
        else:
            self._add(
                "FAIL",
                "Python environment",
                (
                    f"no active virtual/conda environment detected (executable={sys.executable}). "
                    f"Fix: source {activate_script}  or conda activate <env> before launching GUI."
                ),
            )

        _append_robot_port_checks(self.config, self._add)
        _append_camera_checks(self.config, self._add)

    def run_common_preflight_checks(self) -> list[CheckResult]:
        self._reset()
        self._append_common_preflight_checks(include_lerobot_folder=True)
        return list(self._checks)

    def collect_doctor_checks(self) -> list[CheckResult]:
        self._reset()
        self._add("PASS", "Python", f"{sys.executable} ({sys.version.split()[0]})")

        lerobot_dir = get_lerobot_dir(self.config)
        self._add(
            "PASS" if lerobot_dir.exists() else "FAIL",
            "LeRobot folder",
            str(lerobot_dir),
        )

        record_data_dir = Path(normalize_path(self.config.get("record_data_dir", "")))
        self._add(
            "PASS" if record_data_dir.exists() else "WARN",
            "Record data dir",
            str(record_data_dir),
        )

        deploy_data_dir = get_deploy_data_dir(self.config)
        self._add(
            "PASS" if deploy_data_dir.exists() else "WARN",
            "Deploy data dir",
            str(deploy_data_dir),
        )

        models_dir = Path(normalize_path(self.config.get("trained_models_dir", "")))
        self._add(
            "PASS" if models_dir.exists() else "WARN",
            "Trained models dir",
            str(models_dir),
        )

        runs_dir = Path(normalize_path(self.config.get("runs_dir", DEFAULT_RUNS_DIR)))
        self._add(
            "PASS" if runs_dir.exists() else "WARN",
            "Runs artifacts dir",
            str(runs_dir),
        )

        next_record_name = increment_dataset_name(
            repo_name_from_repo_id(str(self.config.get("last_dataset_name", "dataset_1")))
        )
        record_targets = [record_data_dir / next_record_name, lerobot_dir / "data" / next_record_name]
        existing_record_targets = [str(path) for path in record_targets if path.exists()]
        self._add(
            "WARN" if existing_record_targets else "PASS",
            "Next record dataset collision",
            ", ".join(existing_record_targets)
            if existing_record_targets
            else f"{next_record_name} is free in record/lerobot data targets",
        )

        next_eval_name = repo_name_from_repo_id(
            suggest_eval_dataset_name(self.config, str(self.config.get("last_model_name", "")))
        )
        eval_targets = [deploy_data_dir / next_eval_name, lerobot_dir / "data" / next_eval_name]
        existing_eval_targets = [str(path) for path in eval_targets if path.exists()]
        self._add(
            "WARN" if existing_eval_targets else "PASS",
            "Next eval dataset collision",
            ", ".join(existing_eval_targets)
            if existing_eval_targets
            else f"{next_eval_name} is free in deploy/lerobot data targets",
        )

        huggingface_cli = shutil.which("huggingface-cli")
        self._add(
            "PASS" if huggingface_cli else "WARN",
            "huggingface-cli",
            huggingface_cli or "not found in PATH",
        )

        self._append_common_preflight_checks(include_lerobot_folder=False)
        return list(self._checks)

    def collect_doctor_events(self) -> list[DiagnosticEvent]:
        return diagnostics_from_checks(self.collect_doctor_checks())


def _run_common_preflight_checks(config: dict[str, Any]) -> list[CheckResult]:
    return PreflightChecker(config).run_common_preflight_checks()


def collect_doctor_checks(config: dict[str, Any]) -> list[CheckResult]:
    return PreflightChecker(config).collect_doctor_checks()


def collect_doctor_events(config: dict[str, Any]) -> list[DiagnosticEvent]:
    return PreflightChecker(config).collect_doctor_events()
