from __future__ import annotations

import importlib.metadata
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
    match_validated_track,
    validated_tracks_payload,
    validated_tracks_summary,
)


_FLAG_PATTERN = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_.-]*)")
_CAP_CACHE: dict[tuple[str, ...], "LeRobotCapabilities"] = {}


@dataclass(frozen=True)
class LeRobotCapabilities:
    detected_at_iso: str
    lerobot_version: str
    record_entrypoint: str
    train_entrypoint: str
    teleop_entrypoint: str
    teleop_uses_legacy_control: bool
    calibrate_entrypoint: str
    record_help_available: bool
    record_help_error: str
    supported_record_flags: tuple[str, ...]
    train_help_available: bool
    train_help_error: str
    supported_train_flags: tuple[str, ...]
    missing_train_flags: tuple[str, ...]
    supports_policy_path: bool
    policy_path_flag: str | None
    supported_rename_flags: tuple[str, ...]
    active_rename_flag: str
    fallback_notes: tuple[str, ...]
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_at_iso": self.detected_at_iso,
            "lerobot_version": self.lerobot_version,
            "record_entrypoint": self.record_entrypoint,
            "train_entrypoint": self.train_entrypoint,
            "teleop_entrypoint": self.teleop_entrypoint,
            "teleop_uses_legacy_control": self.teleop_uses_legacy_control,
            "calibrate_entrypoint": self.calibrate_entrypoint,
            "record_help_available": self.record_help_available,
            "record_help_error": self.record_help_error,
            "supported_record_flags": list(self.supported_record_flags),
            "train_help_available": self.train_help_available,
            "train_help_error": self.train_help_error,
            "supported_train_flags": list(self.supported_train_flags),
            "missing_train_flags": list(self.missing_train_flags),
            "supports_policy_path": self.supports_policy_path,
            "policy_path_flag": self.policy_path_flag,
            "supported_rename_flags": list(self.supported_rename_flags),
            "active_rename_flag": self.active_rename_flag,
            "validated_track": (
                match_validated_track(self.lerobot_version).to_dict()
                if match_validated_track(self.lerobot_version) is not None
                else None
            ),
            "validated_tracks": validated_tracks_payload(),
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
    raw = str(config.get("lerobot_dir", "")).strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except Exception:
        return None


def resolve_record_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_record_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
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
        if _module_available(module_name):
            return module_name

    return "lerobot.scripts.lerobot_record"


def resolve_train_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_train_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
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
        if _module_available(module_name):
            return module_name

    return "lerobot.scripts.lerobot_train"


def resolve_calibrate_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_calibrate_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir = _configured_lerobot_dir(config)
    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "calibrate.py").exists():
            return "scripts.calibrate"
        if (lerobot_dir / "lerobot" / "scripts" / "calibrate.py").exists():
            return "lerobot.scripts.calibrate"

    for module_name in ("lerobot.calibrate", "lerobot.scripts.calibrate"):
        if _module_available(module_name):
            return module_name

    return "lerobot.calibrate"


def _resolve_legacy_teleop_entrypoint(lerobot_dir: Path | None) -> tuple[str, bool] | None:
    if _module_available("lerobot.scripts.control_robot"):
        return "lerobot.scripts.control_robot", True

    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "scripts" / "control_robot.py").exists():
            return "lerobot.scripts.control_robot", True
        if (lerobot_dir / "scripts" / "control_robot.py").exists():
            return "scripts.control_robot", True
    return None


def resolve_teleop_entrypoint(config: dict[str, Any]) -> tuple[str, bool]:
    lerobot_dir = _configured_lerobot_dir(config)
    prefer_non_av1_path = sys.platform == "darwin" and _parse_bool(
        config.get("teleop_av1_fallback", sys.platform == "darwin"),
        sys.platform == "darwin",
    )

    if prefer_non_av1_path:
        legacy = _resolve_legacy_teleop_entrypoint(lerobot_dir)
        if legacy is not None:
            return legacy

    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "teleoperate.py").exists():
            return "scripts.teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "teleoperate.py").exists():
            return "lerobot.scripts.teleoperate", False
        if (lerobot_dir / "scripts" / "lerobot_teleoperate.py").exists():
            return "scripts.lerobot_teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_teleoperate.py").exists():
            return "lerobot.scripts.lerobot_teleoperate", False

    if _module_available("lerobot.teleoperate"):
        return "lerobot.teleoperate", False
    if _module_available("lerobot.scripts.teleoperate"):
        return "lerobot.scripts.teleoperate", False
    if _module_available("lerobot.scripts.lerobot_teleoperate"):
        return "lerobot.scripts.lerobot_teleoperate", False

    legacy = _resolve_legacy_teleop_entrypoint(lerobot_dir)
    if legacy is not None:
        return legacy

    return "lerobot.teleoperate", False


def _probe_help_flags(config: dict[str, Any], module_entrypoint: str) -> tuple[set[str], str]:
    lerobot_dir = _configured_lerobot_dir(config)
    cwd = str(lerobot_dir) if lerobot_dir is not None and lerobot_dir.exists() else None
    variants = (
        [sys.executable, "-m", module_entrypoint, "--help"],
        [sys.executable, "-m", module_entrypoint, "-h"],
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
                cwd=cwd,
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

        flags = {match.group(1).strip() for match in _FLAG_PATTERN.finditer(text) if match.group(1).strip()}
        if flags:
            return flags, ""
        errors.append("no flags parsed from help output")

    return set(), "; ".join(errors)


def _probe_record_help_flags(config: dict[str, Any], record_entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, record_entrypoint)


def _probe_train_help_flags(config: dict[str, Any], train_entrypoint: str) -> tuple[set[str], str]:
    return _probe_help_flags(config, train_entrypoint)


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


def _detect_lerobot_version() -> str:
    try:
        return str(importlib.metadata.version("lerobot"))
    except Exception:
        return "unknown"


def _cache_key(
    config: dict[str, Any],
    include_flag_probe: bool,
    *,
    lerobot_version: str | None = None,
) -> tuple[str, ...]:
    version = str(lerobot_version or _detect_lerobot_version())
    return (
        sys.executable,
        str(sys.platform),
        version,
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
    lerobot_version = _detect_lerobot_version()
    key = _cache_key(config, include_flag_probe, lerobot_version=lerobot_version)
    if not force_refresh and key in _CAP_CACHE:
        cached = _CAP_CACHE[key]
        return LeRobotCapabilities(**{**cached.__dict__, "cache_hit": True})

    record_entrypoint = resolve_record_entrypoint(config)
    train_entrypoint = resolve_train_entrypoint(config)
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
    missing_train_flags: list[str] = []
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
        missing_train_flags = [flag for flag in TRAIN_REQUIRED_FLAGS if flag not in supported_train_flags]
        if missing_train_flags:
            fallback_notes.append(
                "Missing required train flags in help output: "
                + ", ".join(f"--{flag}" for flag in missing_train_flags)
            )

    capabilities = LeRobotCapabilities(
        detected_at_iso=datetime.now(timezone.utc).isoformat(),
        lerobot_version=lerobot_version,
        record_entrypoint=record_entrypoint,
        train_entrypoint=train_entrypoint,
        teleop_entrypoint=teleop_entrypoint,
        teleop_uses_legacy_control=teleop_uses_legacy,
        calibrate_entrypoint=calibrate_entrypoint,
        record_help_available=record_help_available,
        record_help_error=record_help_error,
        supported_record_flags=tuple(sorted(supported_flags)),
        train_help_available=train_help_available,
        train_help_error=train_help_error,
        supported_train_flags=tuple(sorted(supported_train_flags)),
        missing_train_flags=tuple(missing_train_flags),
        supports_policy_path=supports_policy_path,
        policy_path_flag=policy_path_flag,
        supported_rename_flags=tuple(supported_rename_flags),
        active_rename_flag=active_rename_flag,
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

    if caps.fallback_notes:
        for note in caps.fallback_notes:
            checks.append(("WARN", "Compatibility fallback", note))
    return checks
