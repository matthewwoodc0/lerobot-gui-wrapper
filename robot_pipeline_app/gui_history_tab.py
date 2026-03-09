from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import list_runs, non_negative_int, write_deploy_episode_spreadsheet, write_deploy_notes_file
from .config_store import _atomic_write
from .gui_async import UiBackgroundJobs
from .gui_dialogs import ask_text_dialog, format_command_for_dialog
from .gui_scroll import bind_yview_wheel_scroll
from .gui_theme import configure_treeview_style

_HISTORY_BOTTOM_SPACER_ROWS = 2
HIDDEN_HISTORY_MODES = {"train_sync", "train_launch", "train_attach"}
HISTORY_MODE_VALUES = ["all", "record", "deploy", "teleop", "upload", "shell", "doctor"]


def is_visible_history_mode(mode: Any) -> bool:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        normalized = "run"
    return normalized not in HIDDEN_HISTORY_MODES


def _cancel_debounce_job(root: Any, job_state: dict[str, Any], key: str = "id") -> None:
    pending = job_state.get(key)
    if pending is None:
        return
    try:
        root.after_cancel(pending)
    except Exception:
        pass
    job_state[key] = None


def _schedule_debounce_job(
    *,
    root: Any,
    job_state: dict[str, Any],
    callback: Callable[[], None],
    delay_ms: int = 220,
    key: str = "id",
) -> Any:
    _cancel_debounce_job(root, job_state, key)

    def _run() -> None:
        job_state[key] = None
        callback()

    job_state[key] = root.after(delay_ms, _run)
    return job_state[key]


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


@dataclass
class HistoryTabHandles:
    refresh: Callable[[], None]
    select_tab: Callable[[], None]
    apply_theme: Callable[[dict[str, str]], None]


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


def _status_display_text(status: str) -> str:
    normalized = str(status).strip().lower()
    if normalized == "success":
        return "Success"
    if normalized == "failed":
        return "Failed"
    if normalized == "canceled":
        return "Canceled"
    return normalized.title() if normalized else "-"


def _normalize_outcome_result(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"success", "failed"}:
        return text
    if text in {"pending", "unmarked"}:
        return "unmarked"
    return None


def _parse_tags_csv(raw: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").split(","):
        tag = chunk.strip()
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags



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
            result = _normalize_outcome_result(raw_entry.get("result")) or "unmarked"
            tags = raw_entry.get("tags") if isinstance(raw_entry.get("tags"), list) else []
            normalized_tags = _parse_tags_csv(",".join(str(tag) for tag in tags))
            note = str(raw_entry.get("note", "")).strip()
            entry: dict[str, Any] = {
                "episode": episode_idx,
                "result": result,
                "tags": normalized_tags,
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


def setup_history_tab(
    *,
    root: Any,
    notebook: Any,
    history_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    log_panel: Any,
    messagebox: Any,
    on_rerun_pipeline: Callable[[list[str], Path | None, str, dict[str, Any]], tuple[bool, str]],
    on_rerun_shell: Callable[[str], tuple[bool, str]],
    background_jobs: UiBackgroundJobs | None = None,
) -> HistoryTabHandles:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import ttk

    rows_by_id: dict[str, dict[str, Any]] = {}

    frame = ttk.Frame(history_tab, style="Panel.TFrame", padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=3)
    frame.rowconfigure(3, weight=2)

    style = ttk.Style(root)
    ui_font = str(colors.get("font_ui", "TkDefaultFont"))
    header_font = (ui_font, 10, "bold")
    body_font = (ui_font, 10)
    try:
        metrics_font = tkfont.Font(root=root, family=ui_font, size=10)
        row_height = max(22, int(metrics_font.metrics("linespace")) + 10)
    except Exception:
        row_height = 24

    configure_treeview_style(
        style=style,
        style_name="History.Treeview",
        colors=colors,
        body_font=body_font,
        heading_font=header_font,
        rowheight=row_height,
    )

    filters = ttk.Frame(frame, style="Panel.TFrame")
    filters.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    mode_var = tk.StringVar(value="all")
    status_var = tk.StringVar(value="all")
    search_var = tk.StringVar(value="")
    refresh_status_var = tk.StringVar(value="")
    _search_refresh_job: dict[str, Any] = {"id": None}
    _history_busy_job: dict[str, Any] = {"id": None, "ticks": 0}

    ttk.Label(filters, text="Mode", style="Field.TLabel").pack(side="left")
    mode_combo = ttk.Combobox(
        filters,
        textvariable=mode_var,
        values=HISTORY_MODE_VALUES,
        width=12,
        state="readonly",
        style="Dark.TCombobox",
    )
    mode_combo.pack(side="left", padx=(6, 10))

    ttk.Label(filters, text="Status", style="Field.TLabel").pack(side="left")
    status_combo = ttk.Combobox(
        filters,
        textvariable=status_var,
        values=["all", "success", "failed", "canceled"],
        width=12,
        state="readonly",
        style="Dark.TCombobox",
    )
    status_combo.pack(side="left", padx=(6, 10))

    ttk.Label(filters, text="Search", style="Field.TLabel").pack(side="left")
    search_entry = ttk.Entry(filters, textvariable=search_var, width=36)
    search_entry.pack(side="left", padx=(6, 10), fill="x", expand=True)

    refresh_button = ttk.Button(filters, text="Refresh")
    refresh_button.pack(side="right")
    ttk.Label(filters, textvariable=refresh_status_var, style="Muted.TLabel").pack(side="right", padx=(0, 8))

    # ── Run stats strip ──────────────────────────────────────────────────────
    surface = colors.get("surface", "#1a1a1a")
    stats_strip = tk.Frame(frame, bg=surface, padx=10, pady=5)
    stats_strip.grid(row=1, column=0, sticky="ew", pady=(0, 6))

    _stat_label_font = (colors.get("font_ui", "TkDefaultFont"), 9)
    _stat_val_font = (colors.get("font_mono", "TkFixedFont"), 9, "bold")

    showing_label = tk.Label(stats_strip, text="Showing:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font)
    showing_label.pack(side="left")
    _stats_total_var = tk.StringVar(value="0")
    showing_value_label = tk.Label(
        stats_strip,
        textvariable=_stats_total_var,
        bg=surface,
        fg=colors.get("text", "#eeeeee"),
        font=_stat_val_font,
    )
    showing_value_label.pack(side="left", padx=(4, 12))

    success_label = tk.Label(stats_strip, text="Success:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font)
    success_label.pack(side="left")
    _stats_success_var = tk.StringVar(value="0")
    success_value_label = tk.Label(
        stats_strip,
        textvariable=_stats_success_var,
        bg=surface,
        fg=colors.get("success", "#22c55e"),
        font=_stat_val_font,
    )
    success_value_label.pack(side="left", padx=(4, 12))

    failed_label = tk.Label(stats_strip, text="Failed:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font)
    failed_label.pack(side="left")
    _stats_failed_var = tk.StringVar(value="0")
    failed_value_label = tk.Label(
        stats_strip,
        textvariable=_stats_failed_var,
        bg=surface,
        fg=colors.get("error", "#ef4444"),
        font=_stat_val_font,
    )
    failed_value_label.pack(side="left", padx=(4, 12))

    canceled_label = tk.Label(stats_strip, text="Canceled:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font)
    canceled_label.pack(side="left")
    _stats_canceled_var = tk.StringVar(value="0")
    canceled_value_label = tk.Label(
        stats_strip,
        textvariable=_stats_canceled_var,
        bg=surface,
        fg=colors.get("muted", "#777777"),
        font=_stat_val_font,
    )
    canceled_value_label.pack(side="left", padx=(4, 0))

    tree_frame = ttk.Frame(frame, style="Panel.TFrame")
    tree_frame.grid(row=2, column=0, sticky="nsew")

    columns = ("time", "duration", "mode", "status", "hint", "command")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12, style="History.Treeview")
    tree.heading("time", text="Time")
    tree.heading("duration", text="Duration")
    tree.heading("mode", text="Mode")
    tree.heading("status", text="Status")
    tree.heading("hint", text="Hint")
    tree.heading("command", text="Command")

    tree.column("time", width=155, anchor="w")
    tree.column("duration", width=90, anchor="w")
    tree.column("mode", width=80, anchor="w")
    tree.column("status", width=95, anchor="w")
    tree.column("hint", width=220, anchor="w")
    tree.column("command", width=520, anchor="w")

    tree.tag_configure("even", background=colors.get("surface", "#1a1a1a"))
    tree.tag_configure("odd", background=colors.get("panel", "#111111"))
    tree.tag_configure("success_row", foreground=colors.get("success", "#22c55e"), font=(ui_font, 10, "bold"))
    tree.tag_configure("failed_row", foreground=colors.get("error", "#ef4444"), font=(ui_font, 10, "bold"))
    tree.tag_configure("canceled_row", foreground=colors.get("muted", "#777777"), font=(ui_font, 10, "bold"))
    tree.tag_configure(
        "spacer_row",
        background=colors.get("surface", "#1a1a1a"),
        foreground=colors.get("surface", "#1a1a1a"),
    )

    tree_scroll = ttk.Scrollbar(
        tree_frame,
        orient="vertical",
        command=tree.yview,
        style="Dark.Vertical.TScrollbar",
    )
    tree.configure(yscrollcommand=tree_scroll.set)

    tree.grid(row=0, column=0, sticky="nsew")
    tree_scroll.grid(row=0, column=1, sticky="ns")
    bind_yview_wheel_scroll(tree)
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    details_frame = ttk.LabelFrame(frame, text="Selected History Entry", style="Section.TLabelframe", padding=10)
    details_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
    details_frame.columnconfigure(0, weight=1)
    details_frame.rowconfigure(0, weight=1)

    details = tk.Text(
        details_frame,
        wrap="word",
        height=12,
        bg=colors.get("surface", "#111827"),
        fg=colors.get("text", "#e6edf7"),
        insertbackground="#f8fafc",
        relief="flat",
        font=(colors.get("font_mono", "TkFixedFont"), 10),
        padx=8,
        pady=8,
    )
    details.grid(row=0, column=0, sticky="nsew")
    details.configure(state="disabled")

    buttons = ttk.Frame(details_frame, style="Panel.TFrame")
    buttons.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    open_run_button = ttk.Button(buttons, text="Open Artifact Folder")
    open_run_button.pack(side="left")
    open_log_button = ttk.Button(buttons, text="Open command.log")
    open_log_button.pack(side="left", padx=(8, 0))
    copy_button = ttk.Button(buttons, text="Copy Command")
    copy_button.pack(side="left", padx=(8, 0))
    rerun_button = ttk.Button(buttons, text="Rerun")
    rerun_button.pack(side="right")

    deploy_editor = ttk.LabelFrame(
        details_frame,
        text="Deploy Outcome + Notes Editor",
        style="Section.TLabelframe",
        padding=10,
    )
    deploy_editor.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    deploy_editor.columnconfigure(1, weight=1)
    deploy_editor.columnconfigure(3, weight=1)
    deploy_editor.columnconfigure(5, weight=1)

    episode_edit_var = tk.StringVar(value="")
    outcome_edit_var = tk.StringVar(value="success")
    tags_edit_var = tk.StringVar(value="")
    episode_note_var = tk.StringVar(value="")

    ttk.Label(deploy_editor, text="Episode", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    episode_combo = ttk.Combobox(
        deploy_editor,
        textvariable=episode_edit_var,
        values=[],
        width=10,
        state="readonly",
        style="Dark.TCombobox",
    )
    episode_combo.grid(row=0, column=1, sticky="w", pady=3)

    ttk.Label(deploy_editor, text="Status", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 6), pady=3)
    outcome_combo = ttk.Combobox(
        deploy_editor,
        textvariable=outcome_edit_var,
        values=["success", "failed", "unmarked"],
        width=12,
        state="readonly",
        style="Dark.TCombobox",
    )
    outcome_combo.grid(row=0, column=3, sticky="w", pady=3)

    ttk.Label(deploy_editor, text="Tags", style="Field.TLabel").grid(row=0, column=4, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(deploy_editor, textvariable=tags_edit_var, width=38).grid(row=0, column=5, sticky="ew", pady=3)

    ttk.Label(deploy_editor, text="Episode note", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(deploy_editor, textvariable=episode_note_var, width=80).grid(row=1, column=1, columnspan=5, sticky="ew", pady=3)

    deploy_editor_button_row = ttk.Frame(deploy_editor, style="Panel.TFrame")
    deploy_editor_button_row.grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 6))
    save_episode_button = ttk.Button(deploy_editor_button_row, text="Save Episode Edit")
    save_episode_button.pack(side="left")

    ttk.Label(
        deploy_editor,
        text="Deployment overall notes",
        style="Field.TLabel",
    ).grid(row=3, column=0, sticky="w", pady=(2, 4))
    overall_notes_text = tk.Text(
        deploy_editor,
        wrap="word",
        height=4,
        bg=colors.get("surface", "#111827"),
        fg=colors.get("text", "#e6edf7"),
        insertbackground="#f8fafc",
        relief="flat",
        font=(colors.get("font_mono", "TkFixedFont"), 10),
        padx=8,
        pady=6,
    )
    overall_notes_text.grid(row=4, column=0, columnspan=6, sticky="ew")

    deploy_notes_button_row = ttk.Frame(deploy_editor, style="Panel.TFrame")
    deploy_notes_button_row.grid(row=5, column=0, columnspan=6, sticky="w", pady=(6, 0))
    save_overall_notes_button = ttk.Button(deploy_notes_button_row, text="Save Deployment Notes")
    save_overall_notes_button.pack(side="left")
    open_notes_button = ttk.Button(deploy_notes_button_row, text="Open notes.md")
    open_notes_button.pack(side="left", padx=(8, 0))

    deploy_editor_status_var = tk.StringVar(value="")
    ttk.Label(deploy_editor, textvariable=deploy_editor_status_var, style="Muted.TLabel").grid(
        row=6,
        column=0,
        columnspan=6,
        sticky="w",
        pady=(6, 0),
    )

    def get_selected() -> dict[str, Any] | None:
        selected = tree.selection()
        if not selected:
            return None
        return rows_by_id.get(selected[0])

    def set_details_text(text: str) -> None:
        details.configure(state="normal")
        details.delete("1.0", "end")
        details.insert("1.0", text)
        details.configure(state="disabled")

    def _read_selected_metadata() -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
        item = get_selected()
        if item is None:
            return None, None, None
        metadata_path_raw = str(item.get("_metadata_path", "")).strip()
        if not metadata_path_raw:
            return item, None, None
        metadata_path = Path(metadata_path_raw)
        if not metadata_path.exists():
            return item, metadata_path, None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return item, metadata_path, None
        if not isinstance(data, dict):
            return item, metadata_path, None
        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        return item, metadata_path, data

    def _persist_deploy_metadata(updated_data: dict[str, Any]) -> tuple[bool, str]:
        metadata_path_raw = str(updated_data.get("_metadata_path", "")).strip()
        run_path_raw = str(updated_data.get("_run_path", "")).strip()
        if not metadata_path_raw or not run_path_raw:
            return False, "Selected run is missing metadata/run path references."
        metadata_path = Path(metadata_path_raw)
        run_path = Path(run_path_raw)

        payload = dict(updated_data)
        payload.pop("_metadata_path", None)
        payload.pop("_run_path", None)

        try:
            _atomic_write(json.dumps(payload, indent=2) + "\n", metadata_path)
        except OSError as exc:
            return False, f"Failed to write metadata.json: {exc}"

        notes_path = write_deploy_notes_file(run_path, payload, filename="notes.md")
        if notes_path is None:
            return False, "Saved metadata, but failed to write notes.md"
        csv_path, summary_csv_path = write_deploy_episode_spreadsheet(
            run_path,
            payload,
            filename="episode_outcomes.csv",
            summary_filename="episode_outcomes_summary.csv",
        )
        if csv_path is None or summary_csv_path is None:
            return False, "Saved metadata and notes, but failed to write episode CSV files."

        selected = get_selected()
        if selected is not None:
            selected.clear()
            selected.update(payload)
            selected["_metadata_path"] = str(metadata_path)
            selected["_run_path"] = str(run_path)
        return (
            True,
            (
                f"Saved deploy edits: {notes_path.name}, "
                f"{csv_path.name}, {summary_csv_path.name}"
            ),
        )

    def _deploy_episode_map_from_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        summary = _normalize_deploy_episode_outcomes(item.get("deploy_episode_outcomes"))
        episode_map: dict[int, dict[str, Any]] = {}
        episode_outcomes = summary.get("episode_outcomes")
        if isinstance(episode_outcomes, list):
            for entry in episode_outcomes:
                if not isinstance(entry, dict):
                    continue
                try:
                    episode_idx = int(entry.get("episode"))
                except (TypeError, ValueError):
                    continue
                if episode_idx <= 0:
                    continue
                episode_map[episode_idx] = entry
        return summary, episode_map

    def _episode_choices_from_summary(summary: dict[str, Any], episode_map: dict[int, dict[str, Any]]) -> list[str]:
        choices: list[int] = []
        total = summary.get("total_episodes")
        if isinstance(total, int) and total > 0:
            choices.extend(range(1, total + 1))
        if not choices:
            choices.extend(sorted(episode_map))
        if not choices:
            choices = [1]
        return [str(value) for value in choices]

    def _populate_deploy_editor_from_selected() -> None:
        item = get_selected()
        if item is None or str(item.get("mode", "")).strip().lower() != "deploy":
            deploy_editor.grid_remove()
            return
        deploy_editor.grid()
        summary, episode_map = _deploy_episode_map_from_item(item)
        choices = _episode_choices_from_summary(summary, episode_map)
        episode_combo.configure(values=choices)

        current_choice = episode_edit_var.get().strip()
        if current_choice not in choices:
            current_choice = choices[0]
            episode_edit_var.set(current_choice)

        try:
            current_episode = int(current_choice)
        except ValueError:
            current_episode = 1
        current_entry = episode_map.get(current_episode, {})
        outcome_value = _normalize_outcome_result(current_entry.get("result"))
        outcome_edit_var.set(outcome_value if outcome_value is not None else "unmarked")
        tags = current_entry.get("tags") if isinstance(current_entry.get("tags"), list) else []
        tags_edit_var.set(", ".join(str(tag) for tag in tags))
        episode_note_var.set(str(current_entry.get("note", "")).strip())

        overall = str(item.get("deploy_notes_summary", "")).strip()
        overall_notes_text.delete("1.0", "end")
        overall_notes_text.insert("1.0", overall)
        deploy_editor_status_var.set("Edit episode status/tags/note and deployment notes, then save.")

    def render_selected(_: Any = None) -> None:
        item = get_selected()
        if item is None:
            set_details_text("Select a run to inspect command and metadata.")
            deploy_editor.grid_remove()
            return

        status = _derive_status(item)
        lines = [
            f"Run ID: {item.get('run_id', '-')}",
            f"Mode: {item.get('mode', '-')}",
            f"Status: {status}",
            f"Exit: {item.get('exit_code')}",
            f"Canceled: {item.get('canceled')}",
            f"Started: {item.get('started_at_iso', '-')}",
            f"Ended: {item.get('ended_at_iso', '-')}",
            f"Duration: {item.get('duration_s', '-')}",
            f"Source: {item.get('source', '-')}",
            f"Dataset: {item.get('dataset_repo_id', '-')}",
            f"Model: {item.get('model_path', '-')}",
            f"CWD: {item.get('cwd', '-')}",
            f"Artifact Path: {item.get('_run_path', '-')}",
            "",
            "Command:",
            str(item.get("command", "")),
        ]
        mode_name = str(item.get("mode", "")).strip().lower()
        run_path_text = str(item.get("_run_path", "")).strip()
        if mode_name == "deploy" and run_path_text:
            run_path = Path(run_path_text)
            lines.extend(
                [
                    "",
                    "Deploy Artifacts:",
                    f"Notes: {run_path / 'notes.md'}",
                    f"Episode CSV: {run_path / 'episode_outcomes.csv'}",
                    f"Summary CSV: {run_path / 'episode_outcomes_summary.csv'}",
                ]
            )

        outcome_summary = item.get("deploy_episode_outcomes")
        if isinstance(outcome_summary, dict):
            normalized_summary = _normalize_deploy_episode_outcomes(outcome_summary)
            success_count = normalized_summary.get("success_count")
            failed_count = normalized_summary.get("failed_count")
            unmarked_count = normalized_summary.get("unmarked_count")
            rated_count = normalized_summary.get("rated_count")
            total_episodes = normalized_summary.get("total_episodes")
            unmarked_display = unmarked_count if unmarked_count is not None else "--"
            tags = normalized_summary.get("tags") if isinstance(normalized_summary.get("tags"), list) else []
            lines.extend(
                [
                    "",
                    "Deploy Episode Outcomes:",
                    (
                        f"Success: {success_count} | Failed: {failed_count} | "
                        f"Unmarked: {unmarked_display} | Rated: {rated_count}/{total_episodes if total_episodes else '--'}"
                    ),
                    f"Tags: {', '.join(str(tag) for tag in tags) if tags else '(none)'}",
                ]
            )
            episode_outcomes = normalized_summary.get("episode_outcomes")
            if isinstance(episode_outcomes, list) and episode_outcomes:
                for entry in episode_outcomes:
                    if not isinstance(entry, dict):
                        continue
                    episode_idx = entry.get("episode", "?")
                    result = str(entry.get("result", "-")).strip().lower() or "-"
                    result_display = _status_display_text(result)
                    entry_tags = entry.get("tags")
                    if isinstance(entry_tags, list) and entry_tags:
                        tag_text = ", ".join(str(tag) for tag in entry_tags)
                    else:
                        tag_text = "(none)"
                    note_text = str(entry.get("note", "")).strip()
                    if note_text:
                        lines.append(f"Episode {episode_idx}: {result_display} | tags: {tag_text} | note: {note_text}")
                    else:
                        lines.append(f"Episode {episode_idx}: {result_display} | tags: {tag_text}")
        overall_notes = str(item.get("deploy_notes_summary", "")).strip()
        if overall_notes:
            lines.extend(
                [
                    "",
                    "Deployment Notes:",
                    overall_notes,
                ]
            )
        set_details_text("\n".join(lines))
        _populate_deploy_editor_from_selected()

    def _stop_refresh_busy_status(final_text: str = "") -> None:
        pending = _history_busy_job.get("id")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
            _history_busy_job["id"] = None
        refresh_status_var.set(final_text)
        refresh_button.configure(state="normal")

    def _start_refresh_busy_status(base_text: str) -> None:
        _stop_refresh_busy_status()
        refresh_button.configure(state="disabled")
        _history_busy_job["ticks"] = 0

        def _tick() -> None:
            ticks = int(_history_busy_job.get("ticks", 0))
            dots = "." * ((ticks % 3) + 1)
            refresh_status_var.set(f"{base_text}{dots}")
            _history_busy_job["ticks"] = ticks + 1
            _history_busy_job["id"] = root.after(280, _tick)

        _tick()

    def _build_refresh_payload(mode_filter: str, status_filter: str, query: str) -> dict[str, Any]:
        runs, warning_count = list_runs(config=config, limit=5000)
        return _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=warning_count,
            mode_filter=mode_filter,
            status_filter=status_filter,
            query=query,
        )

    def _apply_refresh_payload(payload: dict[str, Any], *, preserve_selection_id: str | None = None) -> None:
        rows_by_id.clear()
        for row_id in tree.get_children(""):
            tree.delete(row_id)

        rows = payload.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                iid = str(row.get("iid", "")).strip()
                if not iid:
                    continue
                item = row.get("item")
                if not isinstance(item, dict):
                    continue
                values = row.get("values")
                tags = row.get("tags")
                values_tuple = tuple(values) if isinstance(values, (list, tuple)) else ("", "", "", "", "", "")
                tags_tuple = tuple(tags) if isinstance(tags, (list, tuple)) else tuple()
                rows_by_id[iid] = item
                tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=values_tuple,
                    tags=tags_tuple,
                )

        for _ in range(_HISTORY_BOTTOM_SPACER_ROWS):
            tree.insert(
                "",
                "end",
                values=("", "", "", "", "", ""),
                tags=("spacer_row",),
            )

        warning_count = payload.get("warning_count")
        if isinstance(warning_count, int) and warning_count > 0:
            log_panel.append_log(f"History: skipped unreadable metadata files: {warning_count}")

        stats = payload.get("stats")
        if isinstance(stats, dict):
            _stats_total_var.set(str(stats.get("total", 0)))
            _stats_success_var.set(str(stats.get("success", 0)))
            _stats_failed_var.set(str(stats.get("failed", 0)))
            _stats_canceled_var.set(str(stats.get("canceled", 0)))
        else:
            _stats_total_var.set(str(len(rows_by_id)))
            _stats_success_var.set("0")
            _stats_failed_var.set("0")
            _stats_canceled_var.set("0")

        if preserve_selection_id and tree.exists(preserve_selection_id):
            tree.selection_set(preserve_selection_id)
            tree.focus(preserve_selection_id)
            tree.see(preserve_selection_id)

        render_selected()

    def refresh(*, preserve_selection: bool = False) -> None:
        selected = tree.selection()
        preserve_selection_id = selected[0] if preserve_selection and selected else None
        mode_filter = mode_var.get().strip().lower()
        status_filter = status_var.get().strip().lower()
        query = search_var.get().strip().lower()

        if background_jobs is None:
            payload = _build_refresh_payload(mode_filter, status_filter, query)
            _apply_refresh_payload(payload, preserve_selection_id=preserve_selection_id)
            return

        _start_refresh_busy_status("Refreshing history")
        background_jobs.submit(
            "history-refresh",
            lambda: _build_refresh_payload(mode_filter, status_filter, query),
            on_success=lambda payload: _apply_refresh_payload(payload, preserve_selection_id=preserve_selection_id),
            on_error=lambda exc: log_panel.append_log(f"History refresh failed: {exc}"),
            on_complete=lambda is_stale: None if is_stale else _stop_refresh_busy_status(),
        )

    def _refresh_preserving_selection() -> None:
        refresh(preserve_selection=True)

    def on_episode_choice_changed(_: Any = None) -> None:
        item = get_selected()
        if item is None or str(item.get("mode", "")).strip().lower() != "deploy":
            return
        _, episode_map = _deploy_episode_map_from_item(item)
        try:
            episode_idx = int(episode_edit_var.get().strip())
        except ValueError:
            return
        entry = episode_map.get(episode_idx, {})
        result = _normalize_outcome_result(entry.get("result"))
        outcome_edit_var.set(result if result is not None else "unmarked")
        tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
        tags_edit_var.set(", ".join(str(tag) for tag in tags))
        episode_note_var.set(str(entry.get("note", "")).strip())

    def save_episode_edit() -> None:
        item, metadata_path, metadata_data = _read_selected_metadata()
        if item is None:
            messagebox.showinfo("History", "Select a deploy row first.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            messagebox.showinfo("History", "Episode editing is only available for deploy runs.")
            return
        if metadata_path is None or metadata_data is None:
            messagebox.showerror("History", "Could not load metadata for this run.")
            return

        try:
            episode_idx = int(episode_edit_var.get().strip())
        except ValueError:
            messagebox.showerror("Deploy Edit", "Episode must be an integer.")
            return
        if episode_idx <= 0:
            messagebox.showerror("Deploy Edit", "Episode must be greater than zero.")
            return

        result_choice = str(outcome_edit_var.get()).strip().lower()
        if result_choice not in {"success", "failed", "unmarked"}:
            messagebox.showerror("Deploy Edit", "Status must be success, failed, or unmarked.")
            return
        tags = _parse_tags_csv(tags_edit_var.get())
        note_text = str(episode_note_var.get()).strip()

        summary = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        _, episode_map = _deploy_episode_map_from_item({"deploy_episode_outcomes": summary})
        if result_choice == "unmarked" and not tags and not note_text:
            episode_map.pop(episode_idx, None)
        else:
            entry: dict[str, Any] = {
                "episode": episode_idx,
                "result": result_choice,
                "tags": tags,
                "updated_at_epoch_s": round(time.time(), 3),
            }
            if note_text:
                entry["note"] = note_text
            episode_map[episode_idx] = entry

        total = summary.get("total_episodes")
        total_episodes = int(total) if isinstance(total, int) and total > 0 else 0
        if total_episodes > 0 and episode_idx > total_episodes:
            total_episodes = episode_idx
        if total_episodes == 0 and episode_map:
            total_episodes = max(episode_map)

        updated_summary = _normalize_deploy_episode_outcomes(
            {
                "enabled": True,
                "total_episodes": total_episodes if total_episodes > 0 else None,
                "episode_outcomes": [episode_map[idx] for idx in sorted(episode_map)],
            }
        )
        metadata_data["deploy_episode_outcomes"] = updated_summary

        ok, detail = _persist_deploy_metadata(metadata_data)
        if not ok:
            messagebox.showerror("Deploy Edit", detail)
            return

        deploy_editor_status_var.set(f"Saved episode {episode_idx}: {result_choice}.")
        log_panel.append_log(detail)
        _refresh_preserving_selection()

    def save_deployment_notes() -> None:
        item, metadata_path, metadata_data = _read_selected_metadata()
        if item is None:
            messagebox.showinfo("History", "Select a deploy row first.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            messagebox.showinfo("History", "Deployment notes are only available for deploy runs.")
            return
        if metadata_path is None or metadata_data is None:
            messagebox.showerror("History", "Could not load metadata for this run.")
            return

        notes = overall_notes_text.get("1.0", "end").strip()
        if notes:
            metadata_data["deploy_notes_summary"] = notes
        else:
            metadata_data.pop("deploy_notes_summary", None)

        summary = _normalize_deploy_episode_outcomes(metadata_data.get("deploy_episode_outcomes"))
        metadata_data["deploy_episode_outcomes"] = summary

        ok, detail = _persist_deploy_metadata(metadata_data)
        if not ok:
            messagebox.showerror("Deploy Notes", detail)
            return

        deploy_editor_status_var.set("Saved deployment notes.")
        log_panel.append_log(detail)
        _refresh_preserving_selection()

    def open_deploy_notes_file() -> None:
        item, _, metadata_data = _read_selected_metadata()
        if item is None:
            messagebox.showinfo("History", "Select a deploy row first.")
            return
        if str(item.get("mode", "")).strip().lower() != "deploy":
            messagebox.showinfo("History", "Notes file is only available for deploy runs.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        if not run_path.exists():
            messagebox.showerror("Deploy Notes", f"Run path does not exist: {run_path}")
            return

        notes_path = run_path / "notes.md"
        if not notes_path.exists() and metadata_data is not None:
            payload = dict(metadata_data)
            payload.pop("_metadata_path", None)
            payload.pop("_run_path", None)
            generated = write_deploy_notes_file(run_path, payload, filename="notes.md")
            if generated is None:
                messagebox.showerror("Deploy Notes", "Unable to generate notes.md for this run.")
                return
            notes_path = generated
            write_deploy_episode_spreadsheet(
                run_path,
                payload,
                filename="episode_outcomes.csv",
                summary_filename="episode_outcomes_summary.csv",
            )

        ok, message = open_path_in_file_manager(notes_path)
        if not ok:
            messagebox.showerror("Deploy Notes", message)

    def open_selected_run_folder() -> None:
        item = get_selected()
        if item is None:
            messagebox.showinfo("History", "Select a history row first.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        ok, message = open_path_in_file_manager(run_path)
        if not ok:
            messagebox.showerror("Open Artifact Folder", message)

    def open_selected_log() -> None:
        item = get_selected()
        if item is None:
            messagebox.showinfo("History", "Select a history row first.")
            return
        run_path = Path(str(item.get("_run_path", "")).strip())
        log_path = run_path / "command.log"
        ok, message = open_path_in_file_manager(log_path)
        if not ok:
            messagebox.showerror("Open command.log", message)

    def copy_selected_command() -> None:
        item = get_selected()
        if item is None:
            messagebox.showinfo("History", "Select a history row first.")
            return
        command = str(item.get("command", "")).strip()
        if not command:
            messagebox.showinfo("History", "Selected row does not contain a command.")
            return
        root.clipboard_clear()
        root.clipboard_append(command)
        log_panel.append_log("Copied history command to clipboard.")

    def rerun_selected() -> None:
        item = get_selected()
        if item is None:
            messagebox.showinfo("History", "Select a history row first.")
            return

        command_text = str(item.get("command", "")).strip()
        if not command_text:
            messagebox.showerror("Rerun", "Selected history entry has no command text.")
            return

        cmd, parse_error = _command_from_item(item)
        if parse_error is not None or cmd is None:
            messagebox.showerror("Rerun", parse_error or "Unable to parse command.")
            return

        if not ask_text_dialog(
            root=root,
            title="Confirm Rerun",
            text=(
                "Rerun the selected command?\n"
                "Click Confirm to execute, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        mode = str(item.get("mode", "run") or "run")
        if mode == "shell":
            ok, message = on_rerun_shell(command_text)
        else:
            cwd_raw = str(item.get("cwd", "")).strip()
            cwd = Path(cwd_raw) if cwd_raw else None
            context = {
                "dataset_repo_id": item.get("dataset_repo_id"),
                "model_path": item.get("model_path"),
            }
            ok, message = on_rerun_pipeline(cmd, cwd, mode, context)

        if not ok:
            messagebox.showerror("Rerun Failed", message)
            return
        log_panel.append_log(message)

    def _cancel_scheduled_search_refresh() -> None:
        _cancel_debounce_job(root, _search_refresh_job, "id")

    def _schedule_search_refresh(_: Any = None) -> None:
        _schedule_debounce_job(
            root=root,
            job_state=_search_refresh_job,
            callback=refresh,
            delay_ms=220,
            key="id",
        )

    def select_tab() -> None:
        notebook.select(history_tab)
        _cancel_scheduled_search_refresh()
        refresh()

    tree.bind("<<TreeviewSelect>>", render_selected)
    mode_combo.bind("<<ComboboxSelected>>", lambda _: (_cancel_scheduled_search_refresh(), refresh()))
    status_combo.bind("<<ComboboxSelected>>", lambda _: (_cancel_scheduled_search_refresh(), refresh()))
    search_entry.bind("<KeyRelease>", _schedule_search_refresh)
    episode_combo.bind("<<ComboboxSelected>>", on_episode_choice_changed)

    refresh_button.configure(command=lambda: (_cancel_scheduled_search_refresh(), refresh()))
    open_run_button.configure(command=open_selected_run_folder)
    open_log_button.configure(command=open_selected_log)
    copy_button.configure(command=copy_selected_command)
    rerun_button.configure(command=rerun_selected)
    save_episode_button.configure(command=save_episode_edit)
    save_overall_notes_button.configure(command=save_deployment_notes)
    open_notes_button.configure(command=open_deploy_notes_file)

    deploy_editor.grid_remove()

    refresh()

    def apply_theme(updated_colors: dict[str, str]) -> None:
        surface_color = updated_colors.get("surface", "#1a1a1a")
        panel_color = updated_colors.get("panel", "#111111")
        surface_alt_color = updated_colors.get("surface_alt", surface_color)
        muted_color = updated_colors.get("muted", "#777777")
        text_color = updated_colors.get("text", "#eeeeee")
        success_color = updated_colors.get("success", "#22c55e")
        error_color = updated_colors.get("error", "#ef4444")
        ui_font_updated = str(updated_colors.get("font_ui", "TkDefaultFont"))
        mono_font_updated = str(updated_colors.get("font_mono", "TkFixedFont"))

        configure_treeview_style(
            style=style,
            style_name="History.Treeview",
            colors=updated_colors,
            body_font=(ui_font_updated, 10),
            heading_font=(ui_font_updated, 10, "bold"),
            rowheight=row_height,
        )

        stats_strip.configure(bg=surface_color)
        showing_label.configure(bg=surface_color, fg=muted_color, font=(ui_font_updated, 9))
        showing_value_label.configure(bg=surface_color, fg=text_color, font=(mono_font_updated, 9, "bold"))
        success_label.configure(bg=surface_color, fg=muted_color, font=(ui_font_updated, 9))
        success_value_label.configure(bg=surface_color, fg=success_color, font=(mono_font_updated, 9, "bold"))
        failed_label.configure(bg=surface_color, fg=muted_color, font=(ui_font_updated, 9))
        failed_value_label.configure(bg=surface_color, fg=error_color, font=(mono_font_updated, 9, "bold"))
        canceled_label.configure(bg=surface_color, fg=muted_color, font=(ui_font_updated, 9))
        canceled_value_label.configure(bg=surface_color, fg=muted_color, font=(mono_font_updated, 9, "bold"))

        details.configure(
            bg=surface_color,
            fg=text_color,
            insertbackground=text_color,
            font=(mono_font_updated, 10),
        )
        overall_notes_text.configure(
            bg=surface_color,
            fg=text_color,
            insertbackground=text_color,
            font=(mono_font_updated, 10),
        )
        tree.tag_configure("even", background=surface_color)
        tree.tag_configure("odd", background=panel_color)
        tree.tag_configure("success_row", foreground=success_color, font=(ui_font_updated, 10, "bold"))
        tree.tag_configure("failed_row", foreground=error_color, font=(ui_font_updated, 10, "bold"))
        tree.tag_configure("canceled_row", foreground=muted_color, font=(ui_font_updated, 10, "bold"))
        tree.tag_configure("spacer_row", background=surface_alt_color, foreground=surface_alt_color)

    return HistoryTabHandles(refresh=refresh, select_tab=select_tab, apply_theme=apply_theme)
