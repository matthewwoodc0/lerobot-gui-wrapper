from __future__ import annotations

from .hardware_motor_setup import (
    MotorSetupRequest,
    MotorSetupSupport,
    apply_motor_setup_success,
    build_motor_setup_preflight_checks,
    build_motor_setup_request_and_command,
    build_motor_setup_result_summary,
    probe_motor_setup_support,
)
from .hardware_replay import (
    ReplayDiscovery,
    ReplayRequest,
    ReplaySupport,
    build_replay_preflight_checks,
    build_replay_readiness_summary,
    build_replay_request_and_command,
    discover_replay_episodes,
    probe_replay_support,
    resolve_local_dataset_path,
    suggested_episode_values,
)
from .repo_utils import repo_name_from_repo_id


def default_dataset_name(repo_id: str) -> str:
    return repo_name_from_repo_id(str(repo_id or "").strip())
