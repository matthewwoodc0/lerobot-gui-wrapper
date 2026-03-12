from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import checks_common as _checks_common
from . import checks_deploy as _checks_deploy
from . import checks_record as _checks_record
from . import checks_teleop as _checks_teleop
from .checks_common import (
    CommonChecksFn,
    WhichFn,
    _check_counts,
    _check_robot_calibration,
    _configured_env_dir,
    _dialout_membership,
    _find_robot_calibration_path,
    _is_suspicious_float,
    _nearest_existing_parent,
    _run_common_preflight_checks as _run_common_preflight_checks_impl,
    _serial_lock_check,
    _validate_calibration_values,
    build_preflight_report,
    collect_doctor_events as _collect_doctor_events_impl,
    diagnostics_from_checks,
    has_failures,
    summarize_checks,
)
from .checks_deploy import (
    _extract_model_camera_keys,
    _probe_policy_path_support,
    _probe_torch_accelerator,
    run_preflight_for_deploy as _run_preflight_for_deploy_impl,
    run_preflight_for_deploy_events as _run_preflight_for_deploy_events_impl,
)
from .checks_record import (
    run_preflight_for_record as _run_preflight_for_record_impl,
    run_preflight_for_record_events as _run_preflight_for_record_events_impl,
)
from .checks_teleop import (
    run_preflight_for_teleop as _run_preflight_for_teleop_impl,
    run_preflight_for_teleop_events as _run_preflight_for_teleop_events_impl,
)

# Re-export imported probe/config helpers so existing patches against
# `robot_pipeline_app.checks` keep working through the compatibility shim.
camera_fingerprint = _checks_common.camera_fingerprint
get_lerobot_dir = _checks_common.get_lerobot_dir
probe_camera_capture = _checks_common.probe_camera_capture
probe_module_import = _checks_common.probe_module_import
serial_port_fingerprint = _checks_common.serial_port_fingerprint


def _sync_checks_bindings() -> None:
    common_names = (
        "camera_fingerprint",
        "get_lerobot_dir",
        "probe_camera_capture",
        "probe_module_import",
        "serial_port_fingerprint",
        "_find_robot_calibration_path",
        "_dialout_membership",
        "_serial_lock_check",
    )
    for name in common_names:
        setattr(_checks_common, name, globals()[name])

    record_names = (
        "_check_robot_calibration",
        "_find_robot_calibration_path",
        "_run_common_preflight_checks",
    )
    for name in record_names:
        setattr(_checks_record, name, globals()[name])

    teleop_names = record_names
    for name in teleop_names:
        setattr(_checks_teleop, name, globals()[name])

    deploy_names = record_names + ("_probe_policy_path_support", "_probe_torch_accelerator")
    for name in deploy_names:
        setattr(_checks_deploy, name, globals()[name])


def _run_common_preflight_checks(config: dict[str, Any]):
    _sync_checks_bindings()
    return _run_common_preflight_checks_impl(config)


def collect_doctor_checks(config: dict[str, Any]):
    _sync_checks_bindings()
    return _checks_common.collect_doctor_checks(config)


def collect_doctor_events(config: dict[str, Any]):
    _sync_checks_bindings()
    return _collect_doctor_events_impl(config)


def run_preflight_for_record(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
    which_fn: WhichFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_record_impl(
        config=config,
        dataset_root=dataset_root,
        upload_enabled=upload_enabled,
        episode_time_s=episode_time_s,
        dataset_repo_id=dataset_repo_id,
        common_checks_fn=common_checks_fn,
        which_fn=which_fn,
    )


def run_preflight_for_record_events(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
    which_fn: WhichFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_record_events_impl(
        config=config,
        dataset_root=dataset_root,
        upload_enabled=upload_enabled,
        episode_time_s=episode_time_s,
        dataset_repo_id=dataset_repo_id,
        common_checks_fn=common_checks_fn,
        which_fn=which_fn,
    )


def run_preflight_for_teleop(
    config: dict[str, Any],
    control_fps: int | None = None,
    common_checks_fn: CommonChecksFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_teleop_impl(
        config=config,
        control_fps=control_fps,
        common_checks_fn=common_checks_fn,
    )


def run_preflight_for_teleop_events(
    config: dict[str, Any],
    control_fps: int | None = None,
    common_checks_fn: CommonChecksFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_teleop_events_impl(
        config=config,
        control_fps=control_fps,
        common_checks_fn=common_checks_fn,
    )


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    command: list[str] | None = None,
    common_checks_fn: CommonChecksFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_deploy_impl(
        config=config,
        model_path=model_path,
        eval_repo_id=eval_repo_id,
        command=command,
        common_checks_fn=common_checks_fn,
    )


def run_preflight_for_deploy_events(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    command: list[str] | None = None,
    common_checks_fn: CommonChecksFn | None = None,
):
    _sync_checks_bindings()
    return _run_preflight_for_deploy_events_impl(
        config=config,
        model_path=model_path,
        eval_repo_id=eval_repo_id,
        command=command,
        common_checks_fn=common_checks_fn,
    )
