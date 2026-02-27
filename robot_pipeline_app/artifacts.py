from __future__ import annotations

import csv
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


def normalize_deploy_result(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"success", "failed"}:
        return text
    if text in {"pending", "unmarked"}:
        return "unmarked"
    return "unmarked"


def _normalize_tag_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for raw in value:
        tag = str(raw).strip()
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags


def non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _normalize_deploy_episode_outcomes(raw_summary: Any) -> dict[str, Any]:
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    total_raw = summary.get("total_episodes")
    try:
        total_episodes = int(total_raw)
    except (TypeError, ValueError):
        total_episodes = 0
    if total_episodes < 0:
        total_episodes = 0

    entries_raw = summary.get("episode_outcomes")
    entries_map: dict[int, dict[str, Any]] = {}
    if isinstance(entries_raw, list):
        for raw_entry in entries_raw:
            if not isinstance(raw_entry, dict):
                continue
            try:
                episode_idx = int(raw_entry.get("episode"))
            except (TypeError, ValueError):
                continue
            if episode_idx <= 0:
                continue
            result = normalize_deploy_result(raw_entry.get("result"))
            tags = _normalize_tag_list(raw_entry.get("tags"))
            note = str(raw_entry.get("note", "")).strip()
            entry: dict[str, Any] = {
                "episode": episode_idx,
                "result": result,
                "tags": tags,
            }
            if note:
                entry["note"] = note
            updated_raw = raw_entry.get("updated_at_epoch_s")
            if updated_raw is not None:
                try:
                    entry["updated_at_epoch_s"] = float(updated_raw)
                except (TypeError, ValueError):
                    pass
            entries_map[episode_idx] = entry

    if entries_map:
        max_episode = max(entries_map)
        if total_episodes <= 0 or total_episodes < max_episode:
            total_episodes = max_episode

    episode_indices = range(1, total_episodes + 1) if total_episodes > 0 else sorted(entries_map)
    entries: list[dict[str, Any]] = []
    for episode_idx in episode_indices:
        entry = entries_map.get(episode_idx)
        if entry is None:
            entry = {
                "episode": episode_idx,
                "result": "unmarked",
                "tags": [],
            }
        entries.append(entry)

    success_count = sum(1 for entry in entries if entry.get("result") == "success")
    failed_count = sum(1 for entry in entries if entry.get("result") == "failed")
    rated_count = sum(1 for entry in entries if entry.get("result") in {"success", "failed"})
    if not entries:
        provided_success = non_negative_int(summary.get("success_count"))
        provided_failed = non_negative_int(summary.get("failed_count"))
        provided_rated = non_negative_int(summary.get("rated_count"))
        if provided_success is not None:
            success_count = provided_success
        if provided_failed is not None:
            failed_count = provided_failed
        if provided_rated is not None:
            rated_count = provided_rated
        else:
            rated_count = success_count + failed_count
        if rated_count < success_count + failed_count:
            rated_count = success_count + failed_count
        if total_episodes <= 0 and rated_count > 0:
            total_episodes = rated_count
    tag_set: set[str] = set()
    for entry in entries:
        for tag in entry.get("tags", []):
            tag_set.add(str(tag))

    return {
        "enabled": bool(summary.get("enabled", True)),
        "total_episodes": total_episodes if total_episodes > 0 else None,
        "rated_count": rated_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "unmarked_count": max(total_episodes - rated_count, 0) if total_episodes > 0 else None,
        "unrated_count": max(total_episodes - rated_count, 0) if total_episodes > 0 else None,
        "tags": sorted(tag_set),
        "episode_outcomes": entries,
    }


def build_deploy_notes_markdown(metadata: dict[str, Any]) -> str:
    run_id = str(metadata.get("run_id", "-"))
    model_path = str(metadata.get("model_path", "-"))
    dataset_repo_id = str(metadata.get("dataset_repo_id", "-"))
    started_at = str(metadata.get("started_at_iso", "-"))
    ended_at = str(metadata.get("ended_at_iso", "-"))
    duration = metadata.get("duration_s", "-")
    exit_code = metadata.get("exit_code")
    status = str(metadata.get("status", "-"))
    command = str(metadata.get("command", "")).strip()
    overall_notes = str(metadata.get("deploy_notes_summary", "")).strip()

    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    episodes = summary.get("episode_outcomes", [])

    lines = [
        "# Deployment Notes",
        "",
        "## Deployment Summary",
        f"- Run ID: {run_id}",
        f"- Status: {status}",
        f"- Exit code: {exit_code}",
        f"- Model: {model_path}",
        f"- Eval dataset: {dataset_repo_id}",
        f"- Started: {started_at}",
        f"- Ended: {ended_at}",
        f"- Duration (s): {duration}",
    ]
    if command:
        lines.extend(
            [
                "- Command:",
                "",
                "```bash",
                command,
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Episode Outcomes",
            "",
            "| Episode | Status | Tags | Note |",
            "| --- | --- | --- | --- |",
        ]
    )
    if episodes:
        for entry in episodes:
            episode_idx = entry.get("episode", "-")
            status_text = str(entry.get("result", "note")).strip().title()
            tags = entry.get("tags")
            if isinstance(tags, list) and tags:
                tag_text = ", ".join(str(tag) for tag in tags)
            else:
                tag_text = "-"
            note_text = str(entry.get("note", "")).strip().replace("\n", " ")
            lines.append(f"| {episode_idx} | {status_text} | {tag_text} | {note_text or '-'} |")
    else:
        lines.append("| - | - | - | - |")

    lines.extend(
        [
            "",
            "## Overall Notes",
            "",
            overall_notes if overall_notes else "_No overall notes yet._",
            "",
        ]
    )
    return "\n".join(lines)


def write_deploy_notes_file(run_path: Path, metadata: dict[str, Any], filename: str = "notes.md") -> Path | None:
    target_dir = Path(run_path)
    if not target_dir.exists() or not target_dir.is_dir():
        return None
    text = build_deploy_notes_markdown(metadata)
    notes_path = target_dir / str(filename)
    try:
        notes_path.write_text(text, encoding="utf-8")
    except OSError:
        return None
    return notes_path


def _safe_tag_column_name(tag: str) -> str:
    text = str(tag).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "tag"


def _episode_rows_for_export(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    episode_map: dict[int, dict[str, Any]] = {}
    entries = summary.get("episode_outcomes")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                episode_idx = int(entry.get("episode"))
            except (TypeError, ValueError):
                continue
            if episode_idx <= 0:
                continue
            episode_map[episode_idx] = entry

    total = summary.get("total_episodes")
    episode_indices: list[int] = []
    if isinstance(total, int) and total > 0:
        episode_indices = list(range(1, total + 1))
    else:
        episode_indices = sorted(episode_map)

    tags = summary.get("tags")
    tag_list = [str(tag) for tag in tags] if isinstance(tags, list) else []

    rows: list[dict[str, Any]] = []
    for episode_idx in episode_indices:
        entry = episode_map.get(episode_idx, {})
        result = normalize_deploy_result(entry.get("result", "unmarked"))
        is_unmarked = 1 if result == "unmarked" else 0
        row = {
            "episode": episode_idx,
            "status": result,
            "is_success": 1 if result == "success" else 0,
            "is_failed": 1 if result == "failed" else 0,
            "is_unmarked": is_unmarked,
            "is_pending": is_unmarked,
            "tags": ", ".join(str(tag) for tag in entry.get("tags", []) if str(tag).strip()),
            "tags_count": len(entry.get("tags", [])) if isinstance(entry.get("tags"), list) else 0,
            "note": str(entry.get("note", "")).strip(),
            "updated_at_epoch_s": entry.get("updated_at_epoch_s", ""),
        }
        entry_tags = {str(tag).strip().lower() for tag in entry.get("tags", [])} if isinstance(entry.get("tags"), list) else set()
        for tag in tag_list:
            row[f"tag__{tag}"] = 1 if tag.strip().lower() in entry_tags else 0
        rows.append(row)
    return rows, tag_list


def write_deploy_episode_spreadsheet(
    run_path: Path,
    metadata: dict[str, Any],
    filename: str = "episode_outcomes.csv",
    summary_filename: str = "episode_outcomes_summary.csv",
) -> tuple[Path | None, Path | None]:
    target_dir = Path(run_path)
    if not target_dir.exists() or not target_dir.is_dir():
        return None, None

    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    rows, tag_list = _episode_rows_for_export(summary)

    # Build stable, collision-safe tag columns.
    tag_columns: list[str] = []
    used_columns: set[str] = set()
    tag_column_by_name: dict[str, str] = {}
    for tag in tag_list:
        base = f"tag__{_safe_tag_column_name(tag)}"
        column = base
        suffix = 2
        while column in used_columns:
            column = f"{base}_{suffix}"
            suffix += 1
        used_columns.add(column)
        tag_columns.append(column)
        tag_column_by_name[tag] = column

    headers = [
        "episode",
        "status",
        "is_success",
        "is_failed",
        "is_unmarked",
        "is_pending",
        "tags",
        "tags_count",
        "note",
        "updated_at_epoch_s",
        *tag_columns,
    ]

    main_csv_path = target_dir / str(filename)
    summary_csv_path = target_dir / str(summary_filename)
    try:
        with main_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                export_row = dict(row)
                for tag_name, col in tag_column_by_name.items():
                    export_row[col] = int(export_row.pop(f"tag__{tag_name}", 0))
                for col in tag_columns:
                    export_row.setdefault(col, 0)
                writer.writerow(export_row)

        total_for_summary = summary.get("total_episodes") or len(rows)
        unmarked_metric = summary.get("unmarked_count")
        if not isinstance(unmarked_metric, int):
            unmarked_metric = max(total_for_summary - int(summary.get("rated_count", 0)), 0)

        summary_rows = [
            ("total_episodes", total_for_summary),
            ("rated_count", summary.get("rated_count", 0)),
            ("success_count", summary.get("success_count", 0)),
            ("failed_count", summary.get("failed_count", 0)),
            ("unmarked_count", unmarked_metric),
            ("pending_count", unmarked_metric),
        ]
        with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for metric, value in summary_rows:
                writer.writerow([metric, value])
            if tag_columns:
                writer.writerow([])
                writer.writerow(["tag", "count"])
                for tag_name, col in tag_column_by_name.items():
                    count = sum(int(row.get(col, 0) or 0) for row in rows)
                    writer.writerow([tag_name, count])
    except OSError:
        return None, None

    return main_csv_path, summary_csv_path


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
    metadata_extra: dict[str, Any] | None = None,
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
    if metadata_extra:
        metadata.update(metadata_extra)
    if mode == "deploy":
        metadata["deploy_episode_outcomes"] = _normalize_deploy_episode_outcomes(
            metadata.get("deploy_episode_outcomes")
        )

    try:
        (run_path / "command.log").write_text(log_text, encoding="utf-8")
        (run_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if mode == "deploy":
            write_deploy_notes_file(run_path, metadata, filename="notes.md")
            write_deploy_episode_spreadsheet(
                run_path,
                metadata,
                filename="episode_outcomes.csv",
                summary_filename="episode_outcomes_summary.csv",
            )
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

        try:
            data["_metadata_path"] = str(metadata_path)
            data["_run_path"] = str(metadata_path.parent)
            runs.append(data)
        except Exception:
            warning_count += 1
            continue

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
