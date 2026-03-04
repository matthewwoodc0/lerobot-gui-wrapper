from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .camera_schema import resolve_camera_schema
from .probes import parse_frame_dimensions, probe_camera_capture

_CAMERA_DEFAULT_WIDTH = 640
_CAMERA_DEFAULT_HEIGHT = 360
_CAMERA_SOFT_CAP_PIXELS = 640 * 480
_CAMERA_BACKOFF_TARGETS: tuple[tuple[int, int], ...] = (
    (640, 480),
    (848, 480),
    (960, 540),
    (1280, 720),
)
_DEFAULT_FOLLOWER_ROBOT_TYPE = "so101_follower"
_DEFAULT_LEADER_ROBOT_TYPE = "so101_leader"
_DEFAULT_FOLLOWER_ACTION_DIM = 6


def _calibration_dir_from_config_value(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw in {".", "./"}:
        return None

    candidate = Path(raw).expanduser()
    # Config fields store explicit calibration FILE paths; runtime expects
    # calibration_dir. Accept either form and normalize to a directory.
    if candidate.suffix.lower() == ".json":
        candidate = candidate.parent
    return str(candidate)


def _follower_calibration_dir(config: dict[str, Any]) -> str | None:
    direct = _calibration_dir_from_config_value(config.get("follower_calibration_path"))
    if direct:
        return direct
    # Backward compatibility with legacy single calibration_path key.
    legacy = _calibration_dir_from_config_value(config.get("calibration_path"))
    if legacy:
        return legacy
    return None


def _leader_calibration_dir(config: dict[str, Any]) -> str | None:
    return _calibration_dir_from_config_value(config.get("leader_calibration_path"))


def _robot_id_from_calibration_path(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw in {".", "./"}:
        return None
    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() != ".json":
        return None
    stem = candidate.stem.strip()
    return stem or None


def _follower_robot_id(config: dict[str, Any]) -> str:
    value = str(config.get("follower_robot_id", "")).strip()
    inferred = _robot_id_from_calibration_path(config.get("follower_calibration_path")) or _robot_id_from_calibration_path(
        config.get("calibration_path")
    )
    if inferred and (not value or value == "red4"):
        return inferred
    return value or "red4"


def _leader_robot_id(config: dict[str, Any]) -> str:
    value = str(config.get("leader_robot_id", "")).strip()
    inferred = _robot_id_from_calibration_path(config.get("leader_calibration_path"))
    if inferred and (not value or value == "white"):
        return inferred
    return value or "white"


def resolve_follower_robot_id(config: dict[str, Any]) -> str:
    """Return follower robot id after applying calibration-path inference."""
    return _follower_robot_id(config)


def resolve_leader_robot_id(config: dict[str, Any]) -> str:
    """Return leader robot id after applying calibration-path inference."""
    return _leader_robot_id(config)


def follower_robot_type(config: dict[str, Any]) -> str:
    value = str(config.get("follower_robot_type", "")).strip()
    return value or _DEFAULT_FOLLOWER_ROBOT_TYPE


def leader_robot_type(config: dict[str, Any]) -> str:
    value = str(config.get("leader_robot_type", "")).strip()
    return value or _DEFAULT_LEADER_ROBOT_TYPE


def follower_robot_action_dim(config: dict[str, Any]) -> int:
    try:
        parsed = int(str(config.get("follower_robot_action_dim", _DEFAULT_FOLLOWER_ACTION_DIM)).strip())
    except Exception:
        parsed = _DEFAULT_FOLLOWER_ACTION_DIM
    return parsed if parsed > 0 else _DEFAULT_FOLLOWER_ACTION_DIM


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    if parsed <= 0:
        return default
    return parsed


def _camera_resolution_soft_cap_pixels(config: dict[str, Any]) -> int:
    return _parse_positive_int(config.get("camera_resolution_soft_cap_pixels"), _CAMERA_SOFT_CAP_PIXELS)


def _camera_resolution_backoff_enabled(config: dict[str, Any]) -> bool:
    return _parse_bool(config.get("camera_resolution_backoff", True), True)


def _probe_resolution(index_or_path: int | str, width: int, height: int) -> tuple[int, int] | None:
    opened, detail = probe_camera_capture(index_or_path, width, height)
    parsed = parse_frame_dimensions(detail)
    if opened and parsed is not None:
        return parsed
    return None


def _resolve_camera_dimensions(
    config: dict[str, Any],
    role: str,
    index_or_path: int | str,
    default_width: int,
    default_height: int,
) -> tuple[int, int]:
    _ = (config, role)
    parsed = _probe_resolution(index_or_path, default_width, default_height)
    if parsed is None:
        return default_width, default_height

    best_width, best_height = parsed
    best_area = int(best_width) * int(best_height)
    soft_cap = _camera_resolution_soft_cap_pixels(config)
    if best_area <= soft_cap or not _camera_resolution_backoff_enabled(config):
        return best_width, best_height

    # If the camera ignored the low default request and returned a large frame,
    # probe a few common low-latency modes and keep the smallest successful size.
    for probe_width, probe_height in _CAMERA_BACKOFF_TARGETS:
        candidate = _probe_resolution(index_or_path, probe_width, probe_height)
        if candidate is None:
            continue
        cand_width, cand_height = candidate
        cand_area = int(cand_width) * int(cand_height)
        if cand_area < best_area:
            best_width, best_height = cand_width, cand_height
            best_area = cand_area
        if best_area <= soft_cap:
            break

    return best_width, best_height


def camera_arg(config: dict[str, Any]) -> str:
    resolution = resolve_camera_schema(config)
    cameras: dict[str, dict[str, Any]] = {}
    for spec in resolution.specs:
        target_width = int(spec.width or _CAMERA_DEFAULT_WIDTH)
        target_height = int(spec.height or _CAMERA_DEFAULT_HEIGHT)
        resolved_width, resolved_height = _resolve_camera_dimensions(
            config,
            spec.name,
            spec.source,
            target_width,
            target_height,
        )
        cameras[spec.name] = {
            "type": spec.camera_type,
            "index_or_path": spec.source,
            "width": resolved_width,
            "height": resolved_height,
            "fps": int(spec.fps),
            "warmup_s": int(spec.warmup_s),
    }
    return json.dumps(cameras, separators=(",", ":"))


def _resolve_record_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_record_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir_value = str(config.get("lerobot_dir", "")).strip()
    lerobot_dir = Path(lerobot_dir_value).expanduser() if lerobot_dir_value else None
    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "lerobot_record.py").exists():
            return "scripts.lerobot_record"
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_record.py").exists():
            return "lerobot.scripts.lerobot_record"
        if (lerobot_dir / "scripts" / "record.py").exists():
            return "scripts.record"

    for module_name in (
        "lerobot.scripts.lerobot_record",
        "lerobot.record",
        "lerobot.scripts.record",
    ):
        if _module_available(module_name):
            return module_name

    return "lerobot.scripts.lerobot_record"


def resolve_record_entrypoint(config: dict[str, Any]) -> str:
    return _resolve_record_entrypoint(config)


def resolve_calibrate_entrypoint(config: dict[str, Any]) -> str:
    configured = str(config.get("lerobot_calibrate_entrypoint", "")).strip()
    if configured:
        return configured

    lerobot_dir_value = str(config.get("lerobot_dir", "")).strip()
    lerobot_dir = Path(lerobot_dir_value).expanduser() if lerobot_dir_value else None
    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "calibrate.py").exists():
            return "scripts.calibrate"
        if (lerobot_dir / "lerobot" / "scripts" / "calibrate.py").exists():
            return "lerobot.scripts.calibrate"

    for module_name in ("lerobot.calibrate", "lerobot.scripts.calibrate"):
        if _module_available(module_name):
            return module_name

    return "lerobot.calibrate"


def build_lerobot_calibrate_command(config: dict[str, Any], *, role: str = "follower") -> list[str]:
    selected_role = role.strip().lower()
    if selected_role == "leader":
        robot_type = leader_robot_type(config)
        robot_port = str(config.get("leader_port", "")).strip()
        robot_id = _leader_robot_id(config)
    else:
        robot_type = follower_robot_type(config)
        robot_port = str(config.get("follower_port", "")).strip()
        robot_id = _follower_robot_id(config)

    cmd = [
        sys.executable,
        "-m",
        resolve_calibrate_entrypoint(config),
        f"--robot.type={robot_type}",
    ]
    if robot_port:
        cmd.append(f"--robot.port={robot_port}")
    if robot_id:
        cmd.append(f"--robot.id={robot_id}")
    return cmd


def build_lerobot_record_command(
    config: dict[str, Any],
    dataset_repo_id: str,
    num_episodes: int,
    task: str,
    episode_time: int,
    policy_path: Path | None = None,
    push_to_hub: bool | None = None,
) -> list[str]:
    follower_calibration_dir = _follower_calibration_dir(config)
    leader_calibration_dir = _leader_calibration_dir(config)
    follower_robot_id = _follower_robot_id(config)
    leader_robot_id = _leader_robot_id(config)
    record_module = _resolve_record_entrypoint(config)
    cmd = [
        sys.executable,
        "-m",
        record_module,
        f"--robot.type={follower_robot_type(config)}",
        f"--robot.port={config['follower_port']}",
        f"--robot.id={follower_robot_id}",
        f"--robot.cameras={camera_arg(config)}",
        f"--teleop.type={leader_robot_type(config)}",
        f"--teleop.port={config['leader_port']}",
        f"--teleop.id={leader_robot_id}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time}",
    ]
    if follower_calibration_dir:
        cmd.append(f"--robot.calibration_dir={follower_calibration_dir}")
    if leader_calibration_dir:
        cmd.append(f"--teleop.calibration_dir={leader_calibration_dir}")
    if push_to_hub is not None:
        cmd.append(f"--dataset.push_to_hub={'true' if push_to_hub else 'false'}")
    if policy_path is not None:
        cmd.append(f"--policy.path={policy_path}")
    return cmd


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _use_macos_av1_fallback(config: dict[str, Any]) -> bool:
    default = sys.platform == "darwin"
    return _parse_bool(config.get("teleop_av1_fallback", default), default)


def _resolve_legacy_teleop_entrypoint(lerobot_dir: Path | None) -> tuple[str, bool] | None:
    if _module_available("lerobot.scripts.control_robot"):
        return "lerobot.scripts.control_robot", True

    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "scripts" / "control_robot.py").exists():
            return "lerobot.scripts.control_robot", True
        if (lerobot_dir / "scripts" / "control_robot.py").exists():
            return "scripts.control_robot", True

    return None


def _resolve_teleop_entrypoint(config: dict[str, Any]) -> tuple[str, bool]:
    lerobot_dir_value = str(config.get("lerobot_dir", "")).strip()
    lerobot_dir = Path(lerobot_dir_value).expanduser() if lerobot_dir_value else None
    prefer_non_av1_path = sys.platform == "darwin" and _use_macos_av1_fallback(config)

    # macOS fallback: choose legacy control path first when available to avoid
    # teleoperate AV1 hardware decode requirements on unsupported systems.
    if prefer_non_av1_path:
        legacy = _resolve_legacy_teleop_entrypoint(lerobot_dir)
        if legacy is not None:
            return legacy

    # Source checkout layout (works when cwd is the LeRobot root).
    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "lerobot_teleoperate.py").exists():
            return "scripts.lerobot_teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_teleoperate.py").exists():
            return "lerobot.scripts.lerobot_teleoperate", False

    # Installed package layouts.
    if _module_available("lerobot.teleoperate"):
        return "lerobot.teleoperate", False
    if _module_available("lerobot.scripts.lerobot_teleoperate"):
        return "lerobot.scripts.lerobot_teleoperate", False

    # Legacy LeRobot fallback.
    legacy = _resolve_legacy_teleop_entrypoint(lerobot_dir)
    if legacy is not None:
        return legacy

    # Default to modern package entrypoint; preflight/setup will surface missing lerobot installs.
    return "lerobot.teleoperate", False


def build_lerobot_teleop_command(
    config: dict[str, Any],
    *,
    follower_robot_id: str | None = None,
    leader_robot_id: str | None = None,
    control_fps: int | None = None,
) -> list[str]:
    module_name, use_legacy_control = _resolve_teleop_entrypoint(config)
    follower_calibration_dir = _follower_calibration_dir(config)
    leader_calibration_dir = _leader_calibration_dir(config)
    resolved_follower_id = str(follower_robot_id or "").strip() or _follower_robot_id(config)
    resolved_leader_id = str(leader_robot_id or "").strip() or _leader_robot_id(config)
    cmd = [
        sys.executable,
        "-m",
        module_name,
    ]
    if use_legacy_control:
        cmd.append("--control.type=teleoperate")
    cmd.extend(
        [
            f"--robot.type={follower_robot_type(config)}",
            f"--robot.port={config['follower_port']}",
            f"--robot.cameras={camera_arg(config)}" if use_legacy_control else "--robot.cameras={}",
            f"--robot.id={resolved_follower_id}",
            f"--teleop.type={leader_robot_type(config)}",
            f"--teleop.port={config['leader_port']}",
            f"--teleop.id={resolved_leader_id}",
        ]
    )
    if follower_calibration_dir:
        cmd.append(f"--robot.calibration_dir={follower_calibration_dir}")
    if leader_calibration_dir:
        cmd.append(f"--teleop.calibration_dir={leader_calibration_dir}")
    if use_legacy_control and control_fps is not None:
        cmd.append(f"--control.fps={control_fps}")
    return cmd
