from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checks import _check_counts
from .config_store import ensure_runs_dir, normalize_path, print_section
from .constants import DEFAULT_RUNS_DIR
from .types import CheckResult


def build_run_id(mode: str) -> str:
    safe_mode = re.sub(r"[^a-zA-Z0-9_-]+", "_", mode.strip() or "run")
    return f"{safe_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_run_artifacts(
    config: dict[str, Any],
    mode: str,
    command: list[str] | str,
    cwd: Path | str | None,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int | None,
    canceled: bool,
    preflight_checks: list[CheckResult] | None,
    output_lines: list[str] | str,
    dataset_repo_id: str | None = None,
    model_path: Path | str | None = None,
    run_id: str | None = None,
    command_argv: list[str] | None = None,
    source: str = "pipeline",
) -> Path | None:
    try:
        runs_dir = ensure_runs_dir(config)
    except OSError:
        return None

    created_run_id = run_id or build_run_id(mode)
    run_path = runs_dir / created_run_id
    suffix = 1
    while run_path.exists():
        run_path = runs_dir / f"{created_run_id}_{suffix}"
        suffix += 1

    try:
        run_path.mkdir(parents=True, exist_ok=False)
    except OSError:
        return None

    command_text = shlex.join(command) if isinstance(command, list) else str(command)
    command_argv_value: list[str] | None = command_argv
    if command_argv_value is None and isinstance(command, list):
        command_argv_value = [str(part) for part in command]
    cwd_text = str(cwd) if cwd is not None else ""

    if isinstance(output_lines, list):
        log_text = "\n".join(output_lines)
    else:
        log_text = str(output_lines)
    if log_text and not log_text.endswith("\n"):
        log_text += "\n"

    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=timezone.utc)

    if canceled:
        status = "canceled"
    elif exit_code == 0:
        status = "success"
    else:
        status = "failed"

    pass_count, warn_count, fail_count = _check_counts(preflight_checks or [])
    metadata = {
        "run_id": run_path.name,
        "mode": mode,
        "command": command_text,
        "command_argv": command_argv_value,
        "cwd": cwd_text,
        "started_at_iso": started_at.isoformat(),
        "ended_at_iso": ended_at.isoformat(),
        "duration_s": round(max((ended_at - started_at).total_seconds(), 0.0), 3),
        "exit_code": exit_code,
        "canceled": bool(canceled),
        "status": status,
        "preflight_fail_count": fail_count,
        "preflight_warn_count": warn_count,
        "preflight_pass_count": pass_count,
        "dataset_repo_id": dataset_repo_id,
        "model_path": str(model_path) if model_path is not None else None,
        "source": source,
    }

    try:
        (run_path / "command.log").write_text(log_text, encoding="utf-8")
        (run_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return None

    return run_path


def list_runs(config: dict[str, Any], limit: int = 15) -> tuple[list[dict[str, Any]], int]:
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    if not runs_dir.exists() or not runs_dir.is_dir():
        return [], 0

    warning_count = 0
    runs: list[dict[str, Any]] = []

    def parse_iso(raw: Any) -> datetime:
        text = str(raw or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warning_count += 1
            continue

        if not isinstance(data, dict):
            warning_count += 1
            continue

        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        runs.append(data)

    runs.sort(key=lambda item: parse_iso(item.get("started_at_iso")), reverse=True)

    if limit > 0:
        runs = runs[:limit]

    return runs, warning_count


def run_history_mode(config: dict[str, Any], limit: int = 15) -> None:
    print_section("=== 📜 HISTORY MODE ===")
    runs, warning_count = list_runs(config, limit=limit)

    if not runs:
        print("No run artifacts found yet.")
        runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
        print(f"Runs folder: {runs_dir}")
        if warning_count:
            print(f"Skipped unreadable metadata files: {warning_count}")
        return

    print("Time                | Mode    | Exit      | Duration | Hint")
    print("-" * 88)
    for item in runs:
        started = str(item.get("started_at_iso", "")).replace("T", " ")[:19]
        mode = str(item.get("mode", "run"))[:7].ljust(7)
        canceled = bool(item.get("canceled", False))
        exit_code = item.get("exit_code")
        exit_text = "CANCELED" if canceled else f"{exit_code}"
        duration = f"{float(item.get('duration_s', 0.0)):.1f}s"
        hint = str(item.get("dataset_repo_id") or item.get("model_path") or "-")
        if hint not in {"-", ""} and "/" in hint and not hint.count("/") == 1:
            hint = Path(hint).name
        print(f"{started:19} | {mode:7} | {exit_text:9} | {duration:8} | {hint}")

    if warning_count:
        print(f"\nSkipped unreadable metadata files: {warning_count}")
