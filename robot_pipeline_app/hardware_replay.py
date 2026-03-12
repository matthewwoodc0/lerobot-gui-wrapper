from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command_overrides import apply_command_overrides
from .commands import follower_robot_type, resolve_follower_robot_id
from .compat import probe_entrypoint_help_flags, resolve_replay_entrypoint
from .config_store import normalize_path
from .dataset_tools import collect_local_dataset_episode_indices, dataset_local_path_candidates
from .lerobot_runtime import build_lerobot_module_command

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
class ReplayDiscovery:
    dataset_path: Path | None
    episode_indices: tuple[int, ...]
    scan_error: str | None
    manual_entry_only: bool


def _choose_flag(flags: set[str], candidates: tuple[str, ...], *, keywords: tuple[str, ...] = ()) -> str | None:
    for candidate in candidates:
        if candidate in flags:
            return candidate
    for candidate in sorted(flags):
        normalized = candidate.lower()
        if keywords and all(keyword in normalized for keyword in keywords):
            return candidate
    return None


def _calibration_dir_from_path(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = Path(normalize_path(raw))
    if candidate.suffix.lower() == ".json":
        candidate = candidate.parent
    return str(candidate)


def resolve_local_dataset_path(
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
    calibration_dir_flag = _choose_flag(flags, _REPLAY_CALIBRATION_FLAG_CANDIDATES, keywords=("calibration", "dir"))

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


def discover_replay_episodes(
    config: dict[str, Any],
    repo_id: str,
    *,
    dataset_path_raw: str = "",
) -> ReplayDiscovery:
    dataset_path = resolve_local_dataset_path(config, repo_id, dataset_path_raw)
    if dataset_path is None:
        return ReplayDiscovery(
            dataset_path=None,
            episode_indices=(),
            scan_error="Dataset was not found locally. Enter an episode manually after syncing the dataset.",
            manual_entry_only=True,
        )
    episode_indices, error_text = collect_local_dataset_episode_indices(
        config,
        repo_id,
        selected_dataset_path=dataset_path,
    )
    if error_text:
        return ReplayDiscovery(
            dataset_path=dataset_path,
            episode_indices=(),
            scan_error=error_text,
            manual_entry_only=True,
        )
    return ReplayDiscovery(
        dataset_path=dataset_path,
        episode_indices=tuple(int(index) for index in episode_indices),
        scan_error=None,
        manual_entry_only=not bool(episode_indices),
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

    dataset_path = resolve_local_dataset_path(config, repo_id, dataset_path_raw)
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
    checks.append(("PASS" if request.robot_port else "FAIL", "Replay robot port", request.robot_port or "Follower port is not configured."))
    checks.append(("PASS" if request.robot_id else "WARN", "Replay robot id", request.robot_id or "Follower id is not configured."))
    checks.append(("PASS" if request.robot_type else "WARN", "Replay robot type", request.robot_type or "Follower robot type is not configured."))
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
            checks.append(("FAIL", "Replay episode", f"Episode {request.episode_index} is not present in the local dataset."))
        else:
            checks.append(("PASS", "Replay episode", f"Episode {request.episode_index} exists locally."))
    if not request.calibration_dir:
        checks.append(("WARN", "Follower calibration", "No follower calibration directory is selected; replay will rely on runtime defaults."))
    if support.used_fallback_flags:
        checks.append(("WARN", "Replay flag probe", "Help probing was unavailable, so the GUI is using fallback replay flags."))
    return checks


def build_replay_readiness_summary(
    *,
    config: dict[str, Any],
    request: ReplayRequest,
    support: ReplaySupport,
) -> str:
    checks = build_replay_preflight_checks(config=config, request=request, support=support)
    lines = [
        f"Dataset: {request.dataset_repo_id}",
        f"Episode: {request.episode_index}",
    ]
    for level, name, detail in checks:
        if name in {"Local dataset path", "Replay episode", "Replay robot port", "Replay robot id", "Replay robot type", "Replay flag probe", "Dataset episode scan", "Follower calibration"}:
            lines.append(f"[{level}] {name}: {detail}")
    lines.append(f"[{'PASS' if support.available else 'FAIL'}] Compatibility: {support.detail}")
    return "\n".join(lines)


def suggested_episode_values(config: dict[str, Any], repo_id: str, *, dataset_path: str = "") -> list[str]:
    discovery = discover_replay_episodes(config, repo_id, dataset_path_raw=dataset_path)
    if not discovery.episode_indices:
        return ["0"]
    return [str(index) for index in discovery.episode_indices[:500]]
