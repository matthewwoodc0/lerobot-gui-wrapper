from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HIDDEN_HISTORY_MODES = {"train_sync", "train_launch", "train_attach"}
HISTORY_MODE_VALUES = ["all", "record", "deploy", "teleop", "train", "upload", "shell", "doctor"]


def is_visible_history_mode(mode: Any) -> bool:
    normalized = str(mode or "").strip().lower() or "run"
    return normalized not in HIDDEN_HISTORY_MODES


def open_path_in_file_manager(path: Path) -> tuple[bool, str]:
    target = Path(path)
    if not target.exists():
        return False, f"Path does not exist: {target}"

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        elif os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:
        return False, f"Unable to open path: {exc}"
    return True, "Opened path."


def _derive_status(item: dict[str, Any]) -> str:
    raw = str(item.get("status", "")).strip().lower()
    if raw in {"success", "failed", "canceled"}:
        return raw
    if bool(item.get("canceled", False)):
        return "canceled"
    try:
        code = int(item.get("exit_code"))
    except Exception:
        return "failed"
    return "success" if code == 0 else "failed"


def _status_display_text(status: str) -> str:
    normalized = str(status).strip().lower()
    if normalized == "success":
        return "Success"
    if normalized == "failed":
        return "Failed"
    if normalized == "canceled":
        return "Canceled"
    return normalized.title() if normalized else "-"


def _command_from_item(item: dict[str, Any]) -> tuple[list[str] | None, str | None]:
    command_argv = item.get("command_argv")
    if isinstance(command_argv, list):
        cmd = [str(part) for part in command_argv if str(part)]
        if cmd:
            return cmd, None

    command_text = str(item.get("command", "")).strip()
    if not command_text:
        return None, "No command stored in this history entry."
    try:
        return shlex.split(command_text), None
    except ValueError as exc:
        return None, f"Unable to parse legacy command string: {exc}"


def _build_history_refresh_payload_from_runs(
    *,
    runs: list[dict[str, Any]],
    warning_count: int,
    mode_filter: str,
    status_filter: str,
    query: str,
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    row_index = 0
    success_count = 0
    failed_count = 0
    canceled_count = 0

    for item in runs:
        mode = str(item.get("mode", "run")).strip().lower() or "run"
        if not is_visible_history_mode(mode):
            continue
        status = _derive_status(item)
        hint = str(item.get("dataset_repo_id") or item.get("model_path") or "-")
        command_text = str(item.get("command", "")).strip()
        started = str(item.get("started_at_iso", "")).replace("T", " ")[:19]
        try:
            duration = f"{float(item.get('duration_s') or 0.0):.1f}s"
        except (TypeError, ValueError):
            duration = "-"
        status_display = _status_display_text(status)

        if mode_filter != "all" and mode != mode_filter:
            continue
        if status_filter != "all" and status != status_filter:
            continue
        if query:
            haystack = " ".join([started, mode, status, hint, command_text]).lower()
            if query not in haystack:
                continue

        run_id = str(item.get("run_id") or item.get("_run_path") or len(rows))
        iid = run_id
        suffix = 1
        while iid in seen_ids:
            iid = f"{run_id}_{suffix}"
            suffix += 1
        seen_ids.add(iid)

        row_tag = "even" if row_index % 2 == 0 else "odd"
        status_tag = f"{status}_row" if status in {"success", "failed", "canceled"} else row_tag
        rows.append(
            {
                "iid": iid,
                "item": item,
                "values": (started, duration, mode, status_display, hint, command_text[:220]),
                "tags": (row_tag, status_tag),
            }
        )
        if status == "success":
            success_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "canceled":
            canceled_count += 1
        row_index += 1

    return {
        "rows": rows,
        "warning_count": warning_count,
        "stats": {
            "total": len(rows),
            "success": success_count,
            "failed": failed_count,
            "canceled": canceled_count,
        },
    }
