from __future__ import annotations

from .hardware_motor_setup import (
    MotorSetupRequest,  # noqa: F401
    MotorSetupSupport,  # noqa: F401
    apply_motor_setup_success,  # noqa: F401
    build_motor_setup_preflight_checks,  # noqa: F401
    build_motor_setup_request_and_command,  # noqa: F401
    build_motor_setup_result_summary,  # noqa: F401
    probe_motor_setup_support,  # noqa: F401
)
from .hardware_replay import (
    ReplayDiscovery,  # noqa: F401
    ReplayRequest,  # noqa: F401
    ReplaySupport,  # noqa: F401
    build_replay_preflight_checks,  # noqa: F401
    build_replay_readiness_summary,  # noqa: F401
    build_replay_request_and_command,  # noqa: F401
    discover_replay_episodes,  # noqa: F401
    probe_replay_support,  # noqa: F401
    resolve_local_dataset_path,  # noqa: F401
    suggested_episode_values,  # noqa: F401
)
from .repo_utils import repo_name_from_repo_id


def default_dataset_name(repo_id: str) -> str:
    return repo_name_from_repo_id(str(repo_id or "").strip())
