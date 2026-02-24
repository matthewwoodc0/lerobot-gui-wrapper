#!/usr/bin/env python3
"""LeRobot local pipeline manager for SO-101 recording and local deployment.

Compatibility shim that re-exports legacy helper names from robot_pipeline_app.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from robot_pipeline_app import checks as _checks
from robot_pipeline_app.artifacts import build_run_id, list_runs, run_history_mode, write_run_artifacts
from robot_pipeline_app.checks import (
    _check_counts,
    collect_doctor_checks,
    has_failures,
    run_preflight_for_teleop,
    summarize_checks,
)
from robot_pipeline_app.cli_modes import (
    main,
    parse_args,
    run_config_mode,
    run_deploy_mode,
    run_doctor_mode,
    run_record_mode,
)
from robot_pipeline_app.commands import build_lerobot_record_command, build_lerobot_teleop_command, camera_arg
from robot_pipeline_app.config_store import (
    CONFIG_FIELDS,
    DEFAULT_CONFIG_VALUES,
    DEFAULT_LEROBOT_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SECONDARY_CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    PRIMARY_CONFIG_PATH,
    default_for_key,
    ensure_config,
    ensure_runs_dir,
    get_lerobot_dir,
    get_secondary_config_path,
    load_raw_config,
    normalize_config_without_prompts,
    normalize_path,
    pick_directory,
    print_section,
    prompt_int,
    prompt_path,
    prompt_text,
    prompt_yes_no,
    save_config,
)
from robot_pipeline_app.constants import DEFAULT_TASK
from robot_pipeline_app.gui_app import run_gui_mode
from robot_pipeline_app.probes import probe_camera_capture, probe_module_import, summarize_probe_error
from robot_pipeline_app.repo_utils import (
    dataset_exists_on_hf,
    has_eval_prefix,
    increment_dataset_name,
    normalize_repo_id,
    repo_name_from_repo_id,
    suggest_dataset_name,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from robot_pipeline_app.runner import run_command

_run_common_preflight_checks = _checks._run_common_preflight_checks
_nearest_existing_parent = _checks._nearest_existing_parent


def run_preflight_for_record(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
) -> list[tuple[str, str, str]]:
    return _checks.run_preflight_for_record(
        config=config,
        dataset_root=dataset_root,
        upload_enabled=upload_enabled,
        episode_time_s=episode_time_s,
        dataset_repo_id=dataset_repo_id,
        common_checks_fn=_run_common_preflight_checks,
        which_fn=shutil.which,
    )


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
) -> list[tuple[str, str, str]]:
    return _checks.run_preflight_for_deploy(
        config=config,
        model_path=model_path,
        eval_repo_id=eval_repo_id,
        common_checks_fn=_run_common_preflight_checks,
    )


__all__ = [
    "CONFIG_FIELDS",
    "DEFAULT_CONFIG_VALUES",
    "DEFAULT_LEROBOT_DIR",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_SECONDARY_CONFIG_PATH",
    "LEGACY_CONFIG_PATH",
    "PRIMARY_CONFIG_PATH",
    "DEFAULT_TASK",
    "print_section",
    "normalize_path",
    "prompt_text",
    "prompt_int",
    "pick_directory",
    "prompt_path",
    "prompt_yes_no",
    "get_secondary_config_path",
    "load_raw_config",
    "save_config",
    "default_for_key",
    "ensure_config",
    "increment_dataset_name",
    "dataset_exists_on_hf",
    "has_eval_prefix",
    "suggest_dataset_name",
    "camera_arg",
    "normalize_repo_id",
    "repo_name_from_repo_id",
    "probe_module_import",
    "summarize_probe_error",
    "probe_camera_capture",
    "suggest_eval_dataset_name",
    "suggest_eval_prefixed_repo_id",
    "build_lerobot_record_command",
    "build_lerobot_teleop_command",
    "run_command",
    "get_lerobot_dir",
    "ensure_runs_dir",
    "build_run_id",
    "_check_counts",
    "summarize_checks",
    "has_failures",
    "_nearest_existing_parent",
    "_run_common_preflight_checks",
    "run_preflight_for_record",
    "run_preflight_for_deploy",
    "run_preflight_for_teleop",
    "write_run_artifacts",
    "list_runs",
    "run_history_mode",
    "collect_doctor_checks",
    "run_doctor_mode",
    "run_record_mode",
    "run_deploy_mode",
    "run_config_mode",
    "normalize_config_without_prompts",
    "run_gui_mode",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting safely.")
        raise SystemExit(1)
