from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import list_runs
from .gui_dialogs import ask_text_dialog, format_command_for_dialog

_HISTORY_BOTTOM_SPACER_ROWS = 2


def _wheel_units(event: Any) -> int:
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    delta = int(getattr(event, "delta", 0))
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        return int(-delta / 120)
    return -1 if delta > 0 else 1


def _bind_tree_wheel_scroll(tree_widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = _wheel_units(event)
        if units == 0:
            return None
        before = tree_widget.yview()
        tree_widget.yview_scroll(units, "units")
        after = tree_widget.yview()
        if before != after:
            return "break"
        return None

    tree_widget.bind("<MouseWheel>", on_wheel, add="+")
    tree_widget.bind("<Button-4>", on_wheel, add="+")
    tree_widget.bind("<Button-5>", on_wheel, add="+")


@dataclass
class HistoryTabHandles:
    refresh: Callable[[], None]
    select_tab: Callable[[], None]


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

    style.configure(
        "History.Treeview",
        font=body_font,
        rowheight=row_height,
        background=colors.get("surface", "#1a1a1a"),
        foreground=colors.get("text", "#eeeeee"),
        fieldbackground=colors.get("surface", "#1a1a1a"),
        borderwidth=0,
    )
    style.configure("History.Treeview.Heading", font=header_font, background=colors.get("panel", "#111111"), foreground=colors.get("accent", "#f0a500"))
    style.map(
        "History.Treeview",
        background=[("selected", colors.get("accent", "#f0a500"))],
        foreground=[("selected", "#000000")],
    )

    filters = ttk.Frame(frame, style="Panel.TFrame")
    filters.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    mode_var = tk.StringVar(value="all")
    status_var = tk.StringVar(value="all")
    search_var = tk.StringVar(value="")

    ttk.Label(filters, text="Mode", style="Field.TLabel").pack(side="left")
    mode_combo = ttk.Combobox(
        filters,
        textvariable=mode_var,
        values=["all", "record", "deploy", "upload", "shell", "doctor"],
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

    # ── Run stats strip ──────────────────────────────────────────────────────
    surface = colors.get("surface", "#1a1a1a")
    stats_strip = tk.Frame(frame, bg=surface, padx=10, pady=5)
    stats_strip.grid(row=1, column=0, sticky="ew", pady=(0, 6))

    _stat_label_font = (colors.get("font_ui", "TkDefaultFont"), 9)
    _stat_val_font = (colors.get("font_mono", "TkFixedFont"), 9, "bold")

    tk.Label(stats_strip, text="Showing:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font).pack(side="left")
    _stats_total_var = tk.StringVar(value="0")
    tk.Label(stats_strip, textvariable=_stats_total_var, bg=surface, fg=colors.get("text", "#eeeeee"), font=_stat_val_font).pack(side="left", padx=(4, 12))

    tk.Label(stats_strip, text="Success:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font).pack(side="left")
    _stats_success_var = tk.StringVar(value="0")
    tk.Label(stats_strip, textvariable=_stats_success_var, bg=surface, fg=colors.get("success", "#22c55e"), font=_stat_val_font).pack(side="left", padx=(4, 12))

    tk.Label(stats_strip, text="Failed:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font).pack(side="left")
    _stats_failed_var = tk.StringVar(value="0")
    tk.Label(stats_strip, textvariable=_stats_failed_var, bg=surface, fg=colors.get("error", "#ef4444"), font=_stat_val_font).pack(side="left", padx=(4, 12))

    tk.Label(stats_strip, text="Canceled:", bg=surface, fg=colors.get("muted", "#777777"), font=_stat_label_font).pack(side="left")
    _stats_canceled_var = tk.StringVar(value="0")
    tk.Label(stats_strip, textvariable=_stats_canceled_var, bg=surface, fg=colors.get("muted", "#777777"), font=_stat_val_font).pack(side="left", padx=(4, 0))

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
    _bind_tree_wheel_scroll(tree)
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

    def render_selected(_: Any = None) -> None:
        item = get_selected()
        if item is None:
            set_details_text("Select a run to inspect command and metadata.")
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

        outcome_summary = item.get("deploy_episode_outcomes")
        if isinstance(outcome_summary, dict):
            success_count = outcome_summary.get("success_count")
            failed_count = outcome_summary.get("failed_count")
            rated_count = outcome_summary.get("rated_count")
            total_episodes = outcome_summary.get("total_episodes")
            tags = outcome_summary.get("tags") if isinstance(outcome_summary.get("tags"), list) else []
            lines.extend(
                [
                    "",
                    "Deploy Episode Outcomes:",
                    (
                        f"Success: {success_count} | Failed: {failed_count} | "
                        f"Rated: {rated_count}/{total_episodes if total_episodes else '--'}"
                    ),
                    f"Tags: {', '.join(str(tag) for tag in tags) if tags else '(none)'}",
                ]
            )
            episode_outcomes = outcome_summary.get("episode_outcomes")
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
                    lines.append(f"Episode {episode_idx}: {result_display} | tags: {tag_text}")
        set_details_text("\n".join(lines))

    def refresh() -> None:
        runs, warning_count = list_runs(config=config, limit=5000)
        rows_by_id.clear()
        for row_id in tree.get_children(""):
            tree.delete(row_id)

        mode_filter = mode_var.get().strip().lower()
        status_filter = status_var.get().strip().lower()
        query = search_var.get().strip().lower()
        row_index = 0

        for item in runs:
            mode = str(item.get("mode", "run")).strip().lower() or "run"
            status = _derive_status(item)
            hint = str(item.get("dataset_repo_id") or item.get("model_path") or "-")
            command_text = str(item.get("command", "")).strip()
            started = str(item.get("started_at_iso", "")).replace("T", " ")[:19]
            duration = f"{float(item.get('duration_s', 0.0)):.1f}s"
            status_display = _status_display_text(status)

            if mode_filter != "all" and mode != mode_filter:
                continue
            if status_filter != "all" and status != status_filter:
                continue
            if query:
                haystack = " ".join([started, mode, status, hint, command_text]).lower()
                if query not in haystack:
                    continue

            run_id = str(item.get("run_id") or item.get("_run_path") or len(rows_by_id))
            iid = run_id
            suffix = 1
            while iid in rows_by_id:
                iid = f"{run_id}_{suffix}"
                suffix += 1

            rows_by_id[iid] = item
            row_tag = "even" if row_index % 2 == 0 else "odd"
            status_tag = f"{status}_row" if status in {"success", "failed", "canceled"} else row_tag
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(started, duration, mode, status_display, hint, command_text[:220]),
                tags=(row_tag, status_tag),
            )
            row_index += 1

        for spacer_idx in range(_HISTORY_BOTTOM_SPACER_ROWS):
            tree.insert(
                "",
                "end",
                values=("", "", "", "", "", ""),
                tags=("spacer_row",),
            )

        if warning_count:
            log_panel.append_log(f"History: skipped unreadable metadata files: {warning_count}")

        # Update stats strip
        n_success = sum(1 for item in rows_by_id.values() if _derive_status(item) == "success")
        n_failed = sum(1 for item in rows_by_id.values() if _derive_status(item) == "failed")
        n_canceled = sum(1 for item in rows_by_id.values() if _derive_status(item) == "canceled")
        _stats_total_var.set(str(len(rows_by_id)))
        _stats_success_var.set(str(n_success))
        _stats_failed_var.set(str(n_failed))
        _stats_canceled_var.set(str(n_canceled))

        render_selected()

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

    def select_tab() -> None:
        notebook.select(history_tab)
        refresh()

    tree.bind("<<TreeviewSelect>>", render_selected)
    mode_combo.bind("<<ComboboxSelected>>", lambda _: refresh())
    status_combo.bind("<<ComboboxSelected>>", lambda _: refresh())
    search_entry.bind("<KeyRelease>", lambda _: refresh())

    refresh_button.configure(command=refresh)
    open_run_button.configure(command=open_selected_run_folder)
    open_log_button.configure(command=open_selected_log)
    copy_button.configure(command=copy_selected_command)
    rerun_button.configure(command=rerun_selected)

    refresh()

    return HistoryTabHandles(refresh=refresh, select_tab=select_tab)
