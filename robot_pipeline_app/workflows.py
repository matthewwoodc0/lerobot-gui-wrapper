from __future__ import annotations

import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_run_artifacts
from .config_store import get_lerobot_dir
from .deploy_diagnostics import explain_deploy_failure
from .runner import run_command
from .types import CheckResult, RunResult


LogFn = Callable[[str], None]


def execute_command_with_artifacts(
    config: dict[str, Any],
    mode: str,
    cmd: list[str],
    cwd: Path | None,
    preflight_checks: list[CheckResult] | None = None,
    dataset_repo_id: str | None = None,
    model_path: Path | str | None = None,
    log: LogFn = print,
) -> RunResult:
    checks = preflight_checks or []
    started = datetime.now(timezone.utc)
    output_lines = ["$ " + shlex.join(cmd)]

    try:
        result = run_command(cmd, cwd=cwd, capture_output=True)
    except KeyboardInterrupt:
        ended = datetime.now(timezone.utc)
        output_lines.append("Interrupted by user.")
        artifact_path = write_run_artifacts(
            config=config,
            mode=mode,
            command=cmd,
            cwd=cwd,
            started_at=started,
            ended_at=ended,
            exit_code=None,
            canceled=True,
            preflight_checks=checks,
            output_lines=output_lines,
            dataset_repo_id=dataset_repo_id,
            model_path=model_path,
        )
        if artifact_path is not None:
            log(f"Run artifacts saved: {artifact_path}")
        log("Interrupted by user.")
        return RunResult(exit_code=None, canceled=True, output_lines=output_lines, artifact_path=artifact_path)

    ended = datetime.now(timezone.utc)
    if result is None:
        output_lines.append(f"Command not found: {cmd[0]}")
        artifact_path = write_run_artifacts(
            config=config,
            mode=mode,
            command=cmd,
            cwd=cwd,
            started_at=started,
            ended_at=ended,
            exit_code=-1,
            canceled=False,
            preflight_checks=checks,
            output_lines=output_lines,
            dataset_repo_id=dataset_repo_id,
            model_path=model_path,
        )
        if artifact_path is not None:
            log(f"Run artifacts saved: {artifact_path}")
        return RunResult(exit_code=-1, canceled=False, output_lines=output_lines, artifact_path=artifact_path)

    if result.stdout:
        print(result.stdout, end="")
        output_lines.extend(result.stdout.splitlines())
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
        output_lines.extend(result.stderr.splitlines())

    if result.returncode != 0 and mode == "deploy":
        deploy_hints = explain_deploy_failure(output_lines, Path(str(model_path)) if model_path else None)
        if deploy_hints:
            output_lines.append("Deploy diagnostics:")
            log("Deploy diagnostics:")
            for hint in deploy_hints:
                output_lines.append(f"- {hint}")
                log(f"- {hint}")

    output_lines.append(f"[exit code {result.returncode}]")
    artifact_path = write_run_artifacts(
        config=config,
        mode=mode,
        command=cmd,
        cwd=cwd,
        started_at=started,
        ended_at=ended,
        exit_code=result.returncode,
        canceled=False,
        preflight_checks=checks,
        output_lines=output_lines,
        dataset_repo_id=dataset_repo_id,
        model_path=model_path,
    )
    if artifact_path is not None:
        log(f"Run artifacts saved: {artifact_path}")

    return RunResult(
        exit_code=result.returncode,
        canceled=False,
        output_lines=output_lines,
        artifact_path=artifact_path,
    )


def move_recorded_dataset(
    lerobot_dir: Path,
    dataset_name: str,
    dataset_root: Path,
    log: LogFn = print,
) -> Path:
    source_dataset = lerobot_dir / "data" / dataset_name
    target_dataset = dataset_root / dataset_name
    active_dataset_path = source_dataset

    if source_dataset.exists() and source_dataset.resolve() != target_dataset.resolve():
        target_dataset.parent.mkdir(parents=True, exist_ok=True)
        if target_dataset.exists():
            log(f"Warning: target dataset folder already exists: {target_dataset}")
            log("Keeping recorded data at original location.")
        else:
            shutil.move(str(source_dataset), str(target_dataset))
            active_dataset_path = target_dataset
            log(f"Moved dataset to: {target_dataset}")
            log("Done! ✓")
    elif target_dataset.exists():
        active_dataset_path = target_dataset

    return active_dataset_path


def upload_dataset_with_artifacts(
    config: dict[str, Any],
    dataset_repo_id: str,
    upload_path: Path,
    log: LogFn = print,
) -> RunResult:
    upload_cmd = [
        "huggingface-cli",
        "upload",
        dataset_repo_id,
        str(upload_path),
        "--repo-type",
        "dataset",
    ]
    return execute_command_with_artifacts(
        config=config,
        mode="upload",
        cmd=upload_cmd,
        cwd=get_lerobot_dir(config),
        preflight_checks=[],
        dataset_repo_id=dataset_repo_id,
        model_path=None,
        log=log,
    )
