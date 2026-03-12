from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command_overrides import apply_command_overrides
from .commands import (
    build_lerobot_calibrate_command,
    follower_robot_type,
    leader_robot_type,
    resolve_follower_robot_id,
    resolve_leader_robot_id,
)
from .compat import (
    probe_entrypoint_help_flags,
    resolve_calibrate_entrypoint,
    resolve_motor_setup_entrypoint,
    resolve_replay_entrypoint,
)
from .config_store import normalize_path
from .dataset_tools import collect_local_dataset_episode_indices, dataset_local_path_candidates
from .lerobot_runtime import build_lerobot_module_command
from .repo_utils import repo_name_from_repo_id

_REPLAY_DATASET_FLAG_CANDIDATES: tuple[str, ...] = (
    "dataset.repo_id",
    "repo_id",
    "dataset",
)
_REPLAY_DATASET_ROOT_FLAG_CANDIDATES: tuple[str, ...] = (
    "dataset.root",
    "root",
    "dataset_root",
)
_REPLAY_DATASET_PATH_FLAG_CANDIDATES: tuple[str, ...] = (
    "dataset.path",
    "dataset_path",
    "input.path",
)
_REPLAY_EPISODE_FLAG_CANDIDATES: tuple[str, ...] = (
    "dataset.episode",
    "dataset.episode_index",
    "dataset.episode_idx",
    "episode",
    "episode_index",
)
_REPLAY_ROBOT_TYPE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.type",)
_REPLAY_ROBOT_PORT_FLAG_CANDIDATES: tuple[str, ...] = ("robot.port",)
_REPLAY_ROBOT_ID_FLAG_CANDIDATES: tuple[str, ...] = ("robot.id",)
_REPLAY_CALIBRATION_FLAG_CANDIDATES: tuple[str, ...] = ("robot.calibration_dir", "calibration_dir")

_MOTOR_ROLE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.role", "role")
_MOTOR_TYPE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.type", "type")
_MOTOR_PORT_FLAG_CANDIDATES: tuple[str, ...] = ("robot.port", "port")
_MOTOR_ID_FLAG_CANDIDATES: tuple[str, ...] = ("robot.id", "id", "current_id")
_MOTOR_NEW_ID_FLAG_CANDIDATES: tuple[str, ...] = ("robot.new_id", "new_id", "target_id")
_MOTOR_BAUDRATE_FLAG_CANDIDATES: tuple[str, ...] = ("robot.baudrate", "baudrate", "serial.baudrate")


@dataclass(frozen=True)
class ReplaySupport:
    available: bool
    entrypoint: str
    detail: str
    supported_flags: tuple[str, ...]
    dataset_flag: str | None
    dataset_root_flag: str | None
    dataset_path_flag: str | None
    episode_flag: str | None
    robot_type_flag: str | None
    robot_port_flag: str | None
    robot_id_flag: str | None
    calibration_dir_flag: str | None
    used_fallback_flags: bool = False


@dataclass(frozen=True)
class ReplayRequest:
    dataset_repo_id: str
    dataset_path: Path | None
    episode_index: int
    robot_type: str
    robot_port: str
    robot_id: str
    calibration_dir: str | None


@dataclass(frozen=True)
class MotorSetupSupport:
    available: bool
    entrypoint: str
    detail: str
    supported_flags: tuple[str, ...]
    role_flag: str | None
    type_flag: str | None
    port_flag: str | None
    id_flag: str | None
    new_id_flag: str | None
    baudrate_flag: str | None
    uses_calibrate_fallback: bool = False
    used_fallback_flags: bool = False


@dataclass(frozen=True)
class MotorSetupRequest:
    role: str
    robot_type: str
    port: str
    robot_id: str
    new_id: str
    baudrate: int | None


def _choose_flag(flags: set[str], candidates: tuple[str, ...], *, keywords: tuple[str, ...] = ()) -> str | None:
    for candidate in candidates:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if keywords and all(keyword in normalized for keyword in keywords):
            return candidate
    return None


def _parse_positive_int(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _calibration_dir_from_path(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = Path(normalize_path(raw))
    if candidate.suffix.lower() == ".json":
        candidate = candidate.parent
    return str(candidate)


def _resolve_local_dataset_path(
    config: dict[str, Any],
    repo_id: str,
    dataset_path_raw: str = "",
) -> Path | None:
    explicit = str(dataset_path_raw or "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(normalize_path(explicit)))
    candidates.extend(dataset_local_path_candidates(config, repo_id))
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _repo_id_for_root_style(
    config: dict[str, Any],
    *,
    repo_id: str,
    dataset_path: Path | None,
) -> tuple[str, str | None]:
    default_root = Path(str(config.get("record_data_dir", "")).strip() or ".").expanduser()
    root = default_root
    local_repo_id = str(repo_id or "").strip()
    if dataset_path is None:
        return local_repo_id, str(root)
    try:
        relative = dataset_path.relative_to(default_root)
    except ValueError:
        root = dataset_path.parent
        local_repo_id = dataset_path.name
    else:
        local_repo_id = str(relative).replace("\\", "/")
    return local_repo_id, str(root)


def probe_replay_support(config: dict[str, Any]) -> ReplaySupport:
    entrypoint = resolve_replay_entrypoint(config)
    if not entrypoint:
        return ReplaySupport(
            available=False,
            entrypoint="",
            detail="Replay is unavailable: no LeRobot replay entrypoint was detected in the configured runtime.",
            supported_flags=(),
            dataset_flag=None,
            dataset_root_flag=None,
            dataset_path_flag=None,
            episode_flag=None,
            robot_type_flag=None,
            robot_port_flag=None,
            robot_id_flag=None,
            calibration_dir_flag=None,
        )

    flags, error_text = probe_entrypoint_help_flags(config, entrypoint)
    dataset_flag = _choose_flag(flags, _REPLAY_DATASET_FLAG_CANDIDATES, keywords=("dataset",))
    dataset_root_flag = _choose_flag(flags, _REPLAY_DATASET_ROOT_FLAG_CANDIDATES, keywords=("root",))
    dataset_path_flag = _choose_flag(flags, _REPLAY_DATASET_PATH_FLAG_CANDIDATES, keywords=("dataset", "path"))
    episode_flag = _choose_flag(flags, _REPLAY_EPISODE_FLAG_CANDIDATES, keywords=("episode",))
    robot_type_flag = _choose_flag(flags, _REPLAY_ROBOT_TYPE_FLAG_CANDIDATES, keywords=("robot", "type"))
    robot_port_flag = _choose_flag(flags, _REPLAY_ROBOT_PORT_FLAG_CANDIDATES, keywords=("robot", "port"))
    robot_id_flag = _choose_flag(flags, _REPLAY_ROBOT_ID_FLAG_CANDIDATES, keywords=("robot", "id"))
    calibration_dir_flag = _choose_flag(
        flags,
        _REPLAY_CALIBRATION_FLAG_CANDIDATES,
        keywords=("calibration", "dir"),
    )

    used_fallback_flags = False
    detail = f"Replay entrypoint detected: {entrypoint}."
    if flags:
        available = bool(episode_flag and (dataset_flag or dataset_path_flag))
        if not available:
            detail = (
                f"Replay entrypoint '{entrypoint}' was detected, but its help output did not expose "
                "dataset/episode flags compatible with GUI replay."
            )
    else:
        # Keep replay available with conservative defaults so users can still
        # review and edit the command before launch.
        used_fallback_flags = True
        dataset_flag = dataset_flag or "dataset.repo_id"
        episode_flag = episode_flag or "dataset.episode"
        robot_type_flag = robot_type_flag or "robot.type"
        robot_port_flag = robot_port_flag or "robot.port"
        robot_id_flag = robot_id_flag or "robot.id"
        calibration_dir_flag = calibration_dir_flag or "robot.calibration_dir"
        available = True
        detail = (
            f"Replay entrypoint detected: {entrypoint}. Help probing was unavailable"
            + (f" ({error_text})." if error_text else ".")
            + " The GUI is using fallback replay flags; review the command before launch."
        )

    return ReplaySupport(
        available=available,
        entrypoint=entrypoint,
        detail=detail,
        supported_flags=tuple(sorted(flags)),
        dataset_flag=dataset_flag,
        dataset_root_flag=dataset_root_flag,
        dataset_path_flag=dataset_path_flag,
        episode_flag=episode_flag,
        robot_type_flag=robot_type_flag,
        robot_port_flag=robot_port_flag,
        robot_id_flag=robot_id_flag,
        calibration_dir_flag=calibration_dir_flag,
        used_fallback_flags=used_fallback_flags,
    )


def build_replay_request_and_command(
    *,
    config: dict[str, Any],
    dataset_repo_id: str,
    episode_raw: str,
    dataset_path_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[ReplayRequest | None, list[str] | None, ReplaySupport, str | None]:
    support = probe_replay_support(config)
    if not support.available:
        return None, None, support, support.detail

    repo_id = str(dataset_repo_id or "").strip()
    if not repo_id:
        return None, None, support, "Dataset repo id is required for replay."

    try:
        episode_index = int(str(episode_raw or "").strip())
    except (TypeError, ValueError):
        return None, None, support, "Episode index must be an integer."
    if episode_index < 0:
        return None, None, support, "Episode index must be zero or greater."

    dataset_path = _resolve_local_dataset_path(config, repo_id, dataset_path_raw)
    if support.dataset_path_flag and dataset_path is None:
        return None, None, support, "Replay requires a local dataset path, but the dataset was not found locally."

    robot_port = str(config.get("follower_port", "")).strip()
    if not robot_port:
        return None, None, support, "Follower port is required for hardware replay."
    robot_type = follower_robot_type(config)
    robot_id = resolve_follower_robot_id(config)
    calibration_dir = _calibration_dir_from_path(config.get("follower_calibration_path"))

    request = ReplayRequest(
        dataset_repo_id=repo_id,
        dataset_path=dataset_path,
        episode_index=episode_index,
        robot_type=robot_type,
        robot_port=robot_port,
        robot_id=robot_id,
        calibration_dir=calibration_dir,
    )

    cmd = [*build_lerobot_module_command(config, support.entrypoint)]
    if support.dataset_path_flag:
        if dataset_path is None:
            return None, None, support, "Replay requires a local dataset path, but the dataset was not found locally."
        cmd.append(f"--{support.dataset_path_flag}={dataset_path}")
    elif support.dataset_root_flag and support.dataset_flag:
        local_repo_id, root = _repo_id_for_root_style(config, repo_id=repo_id, dataset_path=dataset_path)
        cmd.append(f"--{support.dataset_flag}={local_repo_id}")
        if root:
            cmd.append(f"--{support.dataset_root_flag}={root}")
    elif support.dataset_flag:
        cmd.append(f"--{support.dataset_flag}={repo_id}")
    else:
        return None, None, support, support.detail

    if support.episode_flag:
        cmd.append(f"--{support.episode_flag}={episode_index}")
    if support.robot_type_flag:
        cmd.append(f"--{support.robot_type_flag}={robot_type}")
    if support.robot_port_flag:
        cmd.append(f"--{support.robot_port_flag}={robot_port}")
    if support.robot_id_flag and robot_id:
        cmd.append(f"--{support.robot_id_flag}={robot_id}")
    if support.calibration_dir_flag and calibration_dir:
        cmd.append(f"--{support.calibration_dir_flag}={calibration_dir}")

    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, support, override_error or "Unable to apply replay command overrides."
    return request, cmd, support, None


def build_replay_preflight_checks(
    *,
    config: dict[str, Any],
    request: ReplayRequest,
    support: ReplaySupport,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    checks.append(("PASS" if support.available else "FAIL", "Replay entrypoint", support.detail))
    checks.append(
        (
            "PASS" if request.robot_port else "FAIL",
            "Replay robot port",
            request.robot_port or "Follower port is not configured.",
        )
    )
    if request.dataset_path is None:
        checks.append(
            (
                "FAIL" if support.dataset_path_flag else "WARN",
                "Local dataset path",
                "Dataset was not found locally. Sync/download it before replaying on hardware.",
            )
        )
    else:
        checks.append(("PASS", "Local dataset path", str(request.dataset_path)))
        indices, error_text = collect_local_dataset_episode_indices(
            config,
            request.dataset_repo_id,
            selected_dataset_path=request.dataset_path,
        )
        if error_text:
            checks.append(("WARN", "Dataset episode scan", error_text))
        elif request.episode_index not in set(indices):
            checks.append(
                (
                    "FAIL",
                    "Replay episode",
                    f"Episode {request.episode_index} is not present in the local dataset.",
                )
            )
        else:
            checks.append(("PASS", "Replay episode", f"Episode {request.episode_index} exists locally."))
    if support.used_fallback_flags:
        checks.append(
            (
                "WARN",
                "Replay flag probe",
                "Help probing was unavailable, so the GUI is using fallback replay flags.",
            )
        )
    return checks


def probe_motor_setup_support(config: dict[str, Any]) -> MotorSetupSupport:
    entrypoint = resolve_motor_setup_entrypoint(config)
    if not entrypoint:
        calibrate_entrypoint = resolve_calibrate_entrypoint(config)
        if not calibrate_entrypoint:
            return MotorSetupSupport(
                available=False,
                entrypoint="",
                detail="Motor setup is unavailable: no dedicated setup or calibrate entrypoint was detected.",
                supported_flags=(),
                role_flag=None,
                type_flag=None,
                port_flag=None,
                id_flag=None,
                new_id_flag=None,
                baudrate_flag=None,
            )
        return MotorSetupSupport(
            available=True,
            entrypoint=calibrate_entrypoint,
            detail=(
                "Dedicated motor setup entrypoint was not detected. "
                "The GUI will fall back to calibration-only bring-up, so ID and baudrate edits are informational."
            ),
            supported_flags=(),
            role_flag=None,
            type_flag=None,
            port_flag=None,
            id_flag=None,
            new_id_flag=None,
            baudrate_flag=None,
            uses_calibrate_fallback=True,
        )

    flags, error_text = probe_entrypoint_help_flags(config, entrypoint)
    role_flag = _choose_flag(flags, _MOTOR_ROLE_FLAG_CANDIDATES, keywords=("role",))
    type_flag = _choose_flag(flags, _MOTOR_TYPE_FLAG_CANDIDATES, keywords=("robot", "type"))
    port_flag = _choose_flag(flags, _MOTOR_PORT_FLAG_CANDIDATES, keywords=("port",))
    id_flag = _choose_flag(flags, _MOTOR_ID_FLAG_CANDIDATES, keywords=("id",))
    new_id_flag = _choose_flag(flags, _MOTOR_NEW_ID_FLAG_CANDIDATES, keywords=("new", "id"))
    baudrate_flag = _choose_flag(flags, _MOTOR_BAUDRATE_FLAG_CANDIDATES, keywords=("baud",))

    used_fallback_flags = False
    detail = f"Motor setup entrypoint detected: {entrypoint}."
    if not flags:
        used_fallback_flags = True
        type_flag = type_flag or "robot.type"
        port_flag = port_flag or "robot.port"
        id_flag = id_flag or "robot.id"
        new_id_flag = new_id_flag or "robot.new_id"
        baudrate_flag = baudrate_flag or "robot.baudrate"
        detail = (
            f"Motor setup entrypoint detected: {entrypoint}. Help probing was unavailable"
            + (f" ({error_text})." if error_text else ".")
            + " The GUI is using fallback motor-setup flags; review the command before launch."
        )

    return MotorSetupSupport(
        available=True,
        entrypoint=entrypoint,
        detail=detail,
        supported_flags=tuple(sorted(flags)),
        role_flag=role_flag,
        type_flag=type_flag,
        port_flag=port_flag,
        id_flag=id_flag,
        new_id_flag=new_id_flag,
        baudrate_flag=baudrate_flag,
        uses_calibrate_fallback=False,
        used_fallback_flags=used_fallback_flags,
    )


def build_motor_setup_request_and_command(
    *,
    config: dict[str, Any],
    role: str,
    port_raw: str,
    robot_id_raw: str,
    new_id_raw: str = "",
    baudrate_raw: str = "",
    robot_type_raw: str = "",
    arg_overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[MotorSetupRequest | None, list[str] | None, MotorSetupSupport, str | None]:
    selected_role = str(role or "follower").strip().lower()
    if selected_role not in {"follower", "leader"}:
        selected_role = "follower"

    support = probe_motor_setup_support(config)
    if not support.available:
        return None, None, support, support.detail

    port = str(port_raw or "").strip()
    if not port:
        return None, None, support, "Robot port is required for motor bring-up."

    default_robot_type = follower_robot_type(config) if selected_role == "follower" else leader_robot_type(config)
    default_robot_id = resolve_follower_robot_id(config) if selected_role == "follower" else resolve_leader_robot_id(config)
    robot_type = str(robot_type_raw or "").strip() or default_robot_type
    robot_id = str(robot_id_raw or "").strip() or default_robot_id
    new_id = str(new_id_raw or "").strip()
    baudrate = _parse_positive_int(baudrate_raw)
    if str(baudrate_raw or "").strip() and baudrate is None:
        return None, None, support, "Baudrate must be a positive integer."

    request = MotorSetupRequest(
        role=selected_role,
        robot_type=robot_type,
        port=port,
        robot_id=robot_id,
        new_id=new_id,
        baudrate=baudrate,
    )

    if support.uses_calibrate_fallback:
        runtime_config = dict(config)
        if selected_role == "leader":
            runtime_config["leader_port"] = port
            runtime_config["leader_robot_id"] = robot_id
            runtime_config["leader_robot_type"] = robot_type
        else:
            runtime_config["follower_port"] = port
            runtime_config["follower_robot_id"] = robot_id
            runtime_config["follower_robot_type"] = robot_type
        cmd = build_lerobot_calibrate_command(runtime_config, role=selected_role)
    else:
        cmd = [*build_lerobot_module_command(config, support.entrypoint)]
        if support.role_flag:
            cmd.append(f"--{support.role_flag}={selected_role}")
        if support.type_flag:
            cmd.append(f"--{support.type_flag}={robot_type}")
        if support.port_flag:
            cmd.append(f"--{support.port_flag}={port}")
        if support.id_flag and robot_id:
            cmd.append(f"--{support.id_flag}={robot_id}")
        if support.new_id_flag and new_id:
            cmd.append(f"--{support.new_id_flag}={new_id}")
        if support.baudrate_flag and baudrate is not None:
            cmd.append(f"--{support.baudrate_flag}={baudrate}")

    cmd, override_error = apply_command_overrides(
        base_cmd=cmd,
        overrides=arg_overrides,
        custom_args_raw=custom_args_raw,
    )
    if override_error or cmd is None:
        return None, None, support, override_error or "Unable to apply motor setup overrides."
    return request, cmd, support, None


def build_motor_setup_preflight_checks(
    *,
    request: MotorSetupRequest,
    support: MotorSetupSupport,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    level = "WARN" if support.uses_calibrate_fallback else "PASS"
    checks.append((level, "Motor setup entrypoint", support.detail))
    checks.append(("PASS" if request.port else "FAIL", "Robot port", request.port or "Robot port is missing."))
    checks.append(("PASS" if request.robot_id else "WARN", "Robot id", request.robot_id or "Robot id is not set."))
    if request.new_id:
        checks.append(
            (
                "PASS" if bool(support.new_id_flag and not support.uses_calibrate_fallback) else "WARN",
                "Motor id reassignment",
                (
                    f"Will request new motor id '{request.new_id}'."
                    if support.new_id_flag and not support.uses_calibrate_fallback
                    else "Current runtime does not expose a dedicated new-id flag; the new id will not be applied automatically."
                ),
            )
        )
    if request.baudrate is not None:
        checks.append(
            (
                "PASS" if bool(support.baudrate_flag and not support.uses_calibrate_fallback) else "WARN",
                "Motor baudrate",
                (
                    f"Will request baudrate {request.baudrate}."
                    if support.baudrate_flag and not support.uses_calibrate_fallback
                    else "Current runtime does not expose a dedicated baudrate flag; the baudrate will not be applied automatically."
                ),
            )
        )
    if support.used_fallback_flags:
        checks.append(
            (
                "WARN",
                "Motor setup flag probe",
                "Help probing was unavailable, so the GUI is using fallback motor-setup flags.",
            )
        )
    return checks


def apply_motor_setup_success(
    config: dict[str, Any],
    *,
    request: MotorSetupRequest,
    support: MotorSetupSupport,
) -> dict[str, Any]:
    updated = dict(config)
    role = request.role
    id_value = request.robot_id
    if request.new_id and support.new_id_flag and not support.uses_calibrate_fallback:
        id_value = request.new_id
    if role == "leader":
        updated["leader_port"] = request.port
        updated["leader_robot_id"] = id_value
        updated["leader_robot_type"] = request.robot_type
    else:
        updated["follower_port"] = request.port
        updated["follower_robot_id"] = id_value
        updated["follower_robot_type"] = request.robot_type
    return updated


def suggested_episode_values(config: dict[str, Any], repo_id: str, *, dataset_path: str = "") -> list[str]:
    resolved_dataset_path = _resolve_local_dataset_path(config, repo_id, dataset_path)
    if resolved_dataset_path is None:
        return ["0"]
    episode_indices, error_text = collect_local_dataset_episode_indices(
        config,
        repo_id,
        selected_dataset_path=resolved_dataset_path,
    )
    if error_text or not episode_indices:
        return ["0"]
    return [str(index) for index in episode_indices[:500]]


def default_dataset_name(repo_id: str) -> str:
    return repo_name_from_repo_id(str(repo_id or "").strip())
