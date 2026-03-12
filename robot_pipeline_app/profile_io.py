from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_store import normalize_config_without_prompts

_SCHEMA_VERSION = "community_profile.v1"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "description",
    "robot",
    "camera",
    "defaults",
    "paths",
    "mapping_hints",
    "comments",
}
_ROBOT_KEYS = {"follower", "leader"}
_ROBOT_ROLE_KEYS = {"type", "id", "port", "calibration_path", "action_dim"}
_CAMERA_KEYS = {
    "schema_json",
    "policy_feature_map_json",
    "rename_flag",
    "laptop_name",
    "phone_name",
    "laptop_index",
    "phone_index",
}
_DEFAULT_KEYS = {
    "camera_fps",
    "camera_warmup_s",
    "record_target_hz",
    "deploy_target_hz",
    "eval_num_episodes",
    "eval_duration_s",
    "eval_task",
    "compat_policy",
    "auto_fix_mode",
}
_PATH_KEYS = {
    "lerobot_dir",
    "lerobot_venv_dir",
    "record_data_dir",
    "deploy_data_dir",
    "trained_models_dir",
    "runs_dir",
    "follower_port",
    "leader_port",
    "follower_calibration_path",
    "leader_calibration_path",
}
_MAPPING_HINT_KEYS = {"camera_roles", "feature_notes", "notes"}


@dataclass(frozen=True)
class ProfileExportResult:
    ok: bool
    message: str
    output_path: Path | None = None


@dataclass(frozen=True)
class ProfileImportResult:
    ok: bool
    message: str
    updated_config: dict[str, Any] | None = None
    applied_keys: tuple[str, ...] = ()
    skipped_keys: tuple[str, ...] = ()


def profile_preset_payloads() -> dict[str, dict[str, Any]]:
    return {
        "SO-101 Lab Dual Cam": {
            "schema_version": _SCHEMA_VERSION,
            "name": "SO-101 Lab Dual Cam",
            "description": "SO-101 portable lab preset with wrist and overhead cameras.",
            "robot": {
                "follower": {"type": "so101_follower", "action_dim": 6},
                "leader": {"type": "so101_leader"},
            },
            "camera": {
                "schema_json": {
                    "wrist": {"index_or_path": 0},
                    "overhead": {"index_or_path": 1},
                },
                "policy_feature_map_json": {
                    "observation.images.wrist": "observation.images.camera1",
                    "observation.images.overhead": "observation.images.camera2",
                },
            },
            "defaults": {"camera_fps": 30, "compat_policy": "latest_plus_n_minus_1"},
        },
        "SO-100 Bench Cam": {
            "schema_version": _SCHEMA_VERSION,
            "name": "SO-100 Bench Cam",
            "description": "SO-100 portable bench preset with front camera naming.",
            "robot": {
                "follower": {"type": "so100_follower", "action_dim": 6},
                "leader": {"type": "so100_leader"},
            },
            "camera": {
                "schema_json": {
                    "front": {"index_or_path": 0},
                },
            },
            "defaults": {"camera_fps": 30, "compat_policy": "latest_plus_n_minus_1"},
        },
        "Unitree G1 Front Cam": {
            "schema_version": _SCHEMA_VERSION,
            "name": "Unitree G1 Front Cam",
            "description": "Unitree G1 preset with single front camera and 29 DoF.",
            "robot": {
                "follower": {"type": "unitree_g1_29dof", "action_dim": 29},
                "leader": {"type": "unitree_g1_29dof"},
            },
            "camera": {
                "schema_json": {
                    "front": {"index_or_path": 0},
                },
            },
            "defaults": {"camera_fps": 20, "compat_policy": "latest_plus_n_minus_1"},
        },
    }


def _json_or_raw(value: Any) -> Any:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _load_profile_payload(input_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"Unable to read profile file: {exc}"

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload, None
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return None, (
            "Profile file is not JSON-formatted YAML and PyYAML is unavailable. "
            "Install pyyaml or use JSON-formatted YAML profile files."
        )

    try:
        payload = yaml.safe_load(raw)
    except Exception as exc:
        return None, f"Unable to parse profile file: {exc}"
    if not isinstance(payload, dict):
        return None, "Profile file must contain a top-level mapping object."
    return payload, None


def _validate_unknown_keys(path: str, payload: dict[str, Any], allowed_keys: set[str]) -> list[str]:
    unknown = sorted(str(key) for key in payload.keys() if str(key) not in allowed_keys)
    if not unknown:
        return []
    return [f"{path}: unsupported keys {unknown}"]


def validate_profile_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_unknown_keys("root", payload, _TOP_LEVEL_KEYS))

    schema = str(payload.get("schema_version", "")).strip()
    if schema and schema != _SCHEMA_VERSION:
        errors.append(
            f"root.schema_version: unsupported '{schema}'. Expected '{_SCHEMA_VERSION}'."
        )

    robot = payload.get("robot")
    if robot is not None:
        if not isinstance(robot, dict):
            errors.append("robot: must be an object.")
        else:
            errors.extend(_validate_unknown_keys("robot", robot, _ROBOT_KEYS))
            for role in _ROBOT_KEYS:
                role_payload = robot.get(role)
                if role_payload is None:
                    continue
                if not isinstance(role_payload, dict):
                    errors.append(f"robot.{role}: must be an object.")
                    continue
                errors.extend(_validate_unknown_keys(f"robot.{role}", role_payload, _ROBOT_ROLE_KEYS))

    camera = payload.get("camera")
    if camera is not None:
        if not isinstance(camera, dict):
            errors.append("camera: must be an object.")
        else:
            errors.extend(_validate_unknown_keys("camera", camera, _CAMERA_KEYS))

    defaults = payload.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            errors.append("defaults: must be an object.")
        else:
            errors.extend(_validate_unknown_keys("defaults", defaults, _DEFAULT_KEYS))

    paths = payload.get("paths")
    if paths is not None:
        if not isinstance(paths, dict):
            errors.append("paths: must be an object.")
        else:
            errors.extend(_validate_unknown_keys("paths", paths, _PATH_KEYS))

    mapping_hints = payload.get("mapping_hints")
    if mapping_hints is not None:
        if not isinstance(mapping_hints, dict):
            errors.append("mapping_hints: must be an object.")
        else:
            errors.extend(_validate_unknown_keys("mapping_hints", mapping_hints, _MAPPING_HINT_KEYS))

    comments = payload.get("comments")
    if comments is not None and not isinstance(comments, list):
        errors.append("comments: must be a list when provided.")

    return errors


def export_profile(
    config: dict[str, Any],
    *,
    output_path: Path,
    name: str = "",
    description: str = "",
    include_paths: bool = False,
) -> ProfileExportResult:
    normalized = normalize_config_without_prompts(config)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "name": str(name or "").strip() or "LeRobot Community Profile",
        "description": str(description or "").strip() or "Portable lab profile for LeRobot GUI Wrapper.",
        "robot": {
            "follower": {
                "type": str(normalized.get("follower_robot_type", "")),
                "id": str(normalized.get("follower_robot_id", "")),
                "action_dim": int(normalized.get("follower_robot_action_dim", 6)),
            },
            "leader": {
                "type": str(normalized.get("leader_robot_type", "")),
                "id": str(normalized.get("leader_robot_id", "")),
            },
        },
        "camera": {
            "schema_json": _json_or_raw(normalized.get("camera_schema_json", "")),
            "policy_feature_map_json": _json_or_raw(normalized.get("camera_policy_feature_map_json", "")),
            "rename_flag": str(normalized.get("camera_rename_flag", "rename_map")),
            "laptop_name": str(normalized.get("camera_laptop_name", "")),
            "phone_name": str(normalized.get("camera_phone_name", "")),
            "laptop_index": int(normalized.get("camera_laptop_index", 0)),
            "phone_index": int(normalized.get("camera_phone_index", 1)),
        },
        "defaults": {
            "camera_fps": int(normalized.get("camera_fps", 30)),
            "camera_warmup_s": int(normalized.get("camera_warmup_s", 5)),
            "record_target_hz": str(normalized.get("record_target_hz", "")),
            "deploy_target_hz": str(normalized.get("deploy_target_hz", "")),
            "eval_num_episodes": int(normalized.get("eval_num_episodes", 10)),
            "eval_duration_s": int(normalized.get("eval_duration_s", 20)),
            "eval_task": str(normalized.get("eval_task", "")),
            "compat_policy": str(normalized.get("compat_policy", "latest_plus_n_minus_1")),
            "auto_fix_mode": str(normalized.get("auto_fix_mode", "safe")),
        },
        "mapping_hints": {
            "camera_roles": [
                "Define camera_schema_json for 1/2/3+ cameras as needed per lab.",
                "Use policy_feature_map_json when model image keys differ from runtime keys.",
            ],
            "feature_notes": str(normalized.get("camera_policy_feature_map_json", "")).strip(),
        },
        "comments": [],
    }

    if include_paths:
        payload["paths"] = {key: str(normalized.get(key, "")) for key in sorted(_PATH_KEYS)}

    output = output_path.expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        return ProfileExportResult(
            ok=False,
            message=f"Unable to write profile file: {exc}",
            output_path=None,
        )

    return ProfileExportResult(
        ok=True,
        message="Profile exported successfully.",
        output_path=output,
    )


def apply_profile_payload(
    config: dict[str, Any],
    *,
    payload: dict[str, Any],
    apply_paths: bool = False,
) -> ProfileImportResult:
    validation_errors = validate_profile_payload(payload)
    if validation_errors:
        return ProfileImportResult(
            ok=False,
            message="Profile validation failed:\n- " + "\n- ".join(validation_errors),
        )

    normalized = normalize_config_without_prompts(config)
    updated = dict(normalized)
    applied_keys: list[str] = []
    skipped_keys: list[str] = []

    robot = payload.get("robot", {}) if isinstance(payload.get("robot"), dict) else {}
    follower = robot.get("follower", {}) if isinstance(robot.get("follower"), dict) else {}
    leader = robot.get("leader", {}) if isinstance(robot.get("leader"), dict) else {}
    robot_key_map = {
        "follower_robot_type": follower.get("type"),
        "follower_robot_id": follower.get("id"),
        "follower_robot_action_dim": follower.get("action_dim"),
        "leader_robot_type": leader.get("type"),
        "leader_robot_id": leader.get("id"),
    }
    for key, value in robot_key_map.items():
        if value is None:
            continue
        updated[key] = value
        applied_keys.append(key)

    camera = payload.get("camera", {}) if isinstance(payload.get("camera"), dict) else {}
    camera_schema = camera.get("schema_json")
    if camera_schema is not None:
        updated["camera_schema_json"] = (
            json.dumps(camera_schema, separators=(",", ":"))
            if isinstance(camera_schema, (dict, list))
            else str(camera_schema)
        )
        applied_keys.append("camera_schema_json")
    camera_policy_map = camera.get("policy_feature_map_json")
    if camera_policy_map is not None:
        updated["camera_policy_feature_map_json"] = (
            json.dumps(camera_policy_map, separators=(",", ":"))
            if isinstance(camera_policy_map, (dict, list))
            else str(camera_policy_map)
        )
        applied_keys.append("camera_policy_feature_map_json")

    camera_field_map = {
        "camera_rename_flag": camera.get("rename_flag"),
        "camera_laptop_name": camera.get("laptop_name"),
        "camera_phone_name": camera.get("phone_name"),
        "camera_laptop_index": camera.get("laptop_index"),
        "camera_phone_index": camera.get("phone_index"),
    }
    for key, value in camera_field_map.items():
        if value is None:
            continue
        updated[key] = value
        applied_keys.append(key)

    defaults = payload.get("defaults", {}) if isinstance(payload.get("defaults"), dict) else {}
    for key in sorted(_DEFAULT_KEYS):
        if key not in defaults:
            continue
        updated[key] = defaults[key]
        applied_keys.append(key)

    paths = payload.get("paths", {}) if isinstance(payload.get("paths"), dict) else {}
    legacy_path_fields = {
        "follower_port": follower.get("port"),
        "leader_port": leader.get("port"),
        "follower_calibration_path": follower.get("calibration_path"),
        "leader_calibration_path": leader.get("calibration_path"),
    }
    for key in sorted(_PATH_KEYS):
        value = paths.get(key)
        if value is None and key in legacy_path_fields:
            value = legacy_path_fields[key]
        if value is None:
            continue
        if apply_paths:
            updated[key] = value
            applied_keys.append(key)
        else:
            skipped_keys.append(key)

    return ProfileImportResult(
        ok=True,
        message="Profile imported successfully.",
        updated_config=updated,
        applied_keys=tuple(sorted(set(applied_keys))),
        skipped_keys=tuple(sorted(set(skipped_keys))),
    )


def import_profile(
    config: dict[str, Any],
    *,
    input_path: Path,
    apply_paths: bool = False,
) -> ProfileImportResult:
    payload, error = _load_profile_payload(input_path)
    if error is not None or payload is None:
        return ProfileImportResult(ok=False, message=error or "Unable to parse profile file.")
    return apply_profile_payload(config, payload=payload, apply_paths=apply_paths)
