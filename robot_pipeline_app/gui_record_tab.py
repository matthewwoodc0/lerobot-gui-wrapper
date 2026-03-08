from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .camera_schema import resolve_camera_schema
from .command_overrides import get_flag_value
from .checks import run_preflight_for_record
from .config_store import get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_dialogs import ask_editable_command_dialog, ask_text_dialog, format_command_for_dialog, show_text_dialog
from .gui_file_dialogs import ask_directory_dialog
from .gui_forms import build_record_request_and_command
from .gui_log import GuiLogPanel
from .gui_window import fit_window_to_screen
from .repo_utils import (
    compose_repo_id,
    dataset_exists_on_hf,
    get_hf_dataset_info,
    increment_dataset_name,
    list_hf_datasets,
    repo_name_only,
    repo_name_from_repo_id,
    resolve_unique_repo_id,
    suggest_dataset_name,
)
from .runner import format_command
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .types import GuiRunProcessAsync
from .workflows import move_recorded_dataset

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class RecordTabHandles:
    record_dir_var: Any
    camera_slot: Any
    refresh_summary: Callable[[], None]
    apply_theme: Callable[[dict[str, str]], None]
    action_buttons: list[Any]


def _compose_repo_id(owner: str, dataset_name_or_repo_id: str) -> str | None:
    """Backward-compatible alias used by helper tests and older call sites."""
    return compose_repo_id(owner, dataset_name_or_repo_id)


def _list_local_dataset_dirs(record_data_dir: Path, lerobot_dir: Path) -> list[Path]:
    roots = [
        Path(normalize_path(str(record_data_dir))),
        Path(normalize_path(str(lerobot_dir / "data"))),
    ]
    seen: set[str] = set()
    datasets: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            children = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        except OSError:
            continue
        for child in children:
            key = str(child.resolve())
            if key in seen:
                continue
            seen.add(key)
            datasets.append(child)
    return sorted(datasets, key=lambda path: path.name.lower())


def _build_local_dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    path = Path(dataset_path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "files_scanned": 0,
        "video_files": 0,
        "total_size_bytes": 0,
        "marker_files": [],
        "sample_files": [],
        "truncated_scan": False,
    }
    if not path.exists() or not path.is_dir():
        return summary

    marker_names = {"episodes.parquet", "meta.json", "stats.json"}
    sample_files: list[str] = []
    marker_files: set[str] = set()
    total_size = 0
    file_count = 0
    video_count = 0
    limit = 4000

    for current_root, dirnames, filenames in os.walk(path, topdown=True):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in sorted(filenames):
            if file_count >= limit:
                summary["truncated_scan"] = True
                break
            file_count += 1
            full_path = Path(current_root) / filename
            try:
                stat = full_path.stat()
                total_size += int(stat.st_size)
            except OSError:
                pass
            suffix = full_path.suffix.lower()
            if suffix in _VIDEO_EXTENSIONS:
                video_count += 1
            if filename in marker_names:
                marker_files.add(filename)
            if len(sample_files) < 25:
                try:
                    sample_files.append(str(full_path.relative_to(path)))
                except Exception:
                    sample_files.append(str(full_path))
        if summary["truncated_scan"]:
            break

    summary["files_scanned"] = file_count
    summary["video_files"] = video_count
    summary["total_size_bytes"] = total_size
    summary["marker_files"] = sorted(marker_files)
    summary["sample_files"] = sample_files
    return summary



def _hf_parity_detail(exists: bool | None, repo_id: str) -> tuple[str, str]:
    if exists is True:
        return "WARN", f"Remote dataset already exists: {repo_id}"
    if exists is False:
        return "PASS", f"Remote dataset not found yet: {repo_id}"
    return "WARN", f"Unable to confirm if remote dataset exists: {repo_id}"


def _build_v30_convert_command(repo_id: str, python_bin: str | None = None) -> list[str]:
    py = python_bin or sys.executable or shutil.which("python3") or shutil.which("python") or "python3"
    return [
        py,
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
    ]

def setup_record_tab(
    *,
    root: Any,
    record_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    choose_folder: Callable[[Any], None],
    log_panel: GuiLogPanel,
    messagebox: Any,
    set_running: Callable[[bool, str | None, bool], None],
    run_process_async: GuiRunProcessAsync,
    refresh_header_subtitle: Callable[[], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
    background_jobs: Any | None = None,
) -> RecordTabHandles:
    import tkinter as tk
    from tkinter import ttk

    record_container = ttk.Frame(record_tab, style="Panel.TFrame")
    record_container.pack(fill="both", expand=True)

    _local_dataset_initial = increment_dataset_name(str(config.get("last_dataset_name", "dataset_1")))
    record_dataset_var = tk.StringVar(value=_local_dataset_initial)
    record_episodes_var = tk.StringVar(value="20")
    record_duration_var = tk.StringVar(value="20")
    record_target_hz_var = tk.StringVar(value=str(config.get("record_target_hz", "")).strip())
    record_task_var = tk.StringVar(value=DEFAULT_TASK)
    record_dir_var = tk.StringVar(value=str(config["record_data_dir"]))
    record_upload_var = tk.BooleanVar(value=False)
    record_hf_username_var = tk.StringVar(value=str(config.get("hf_username", "")).strip())
    record_hf_repo_name_var = tk.StringVar(value=repo_name_from_repo_id(record_dataset_var.get().strip()))
    record_tag_after_upload_var = tk.BooleanVar(value=bool(config.get("record_tag_after_upload", True)))
    record_advanced_enabled_var = tk.BooleanVar(value=False)
    record_custom_args_var = tk.StringVar(value="")
    lerobot_dir = get_lerobot_dir(config)

    if background_jobs is not None:
        def _apply_suggested_name(result: tuple[str, bool]) -> None:
            if record_dataset_var.get() == _local_dataset_initial:
                record_dataset_var.set(result[0])
                record_hf_repo_name_var.set(repo_name_from_repo_id(result[0]))

        background_jobs.submit(
            "hf-dataset-suggest",
            lambda: suggest_dataset_name(config),
            on_success=_apply_suggested_name,
            on_error=lambda exc: None,
        )

    record_form = ttk.LabelFrame(record_container, text="Recording Setup", style="Section.TLabelframe", padding=12)
    record_form.pack(fill="x")
    record_form.columnconfigure(1, weight=1)

    _record_name_colors: dict[str, str] = dict(colors)
    _record_name_check_job: list[str | None] = [None]
    _record_name_hf_check_key = "record-name-hf-check"

    ttk.Label(record_form, text="Dataset Name", style="SectionTitle.TLabel").grid(
        row=0, column=0, sticky="nw", padx=(0, 6), pady=(8, 2),
    )
    _record_name_cell = ttk.Frame(record_form, style="Panel.TFrame")
    _record_name_cell.grid(row=0, column=1, sticky="ew", pady=(8, 2))
    _record_name_cell.columnconfigure(0, weight=1)
    ttk.Entry(_record_name_cell, textvariable=record_dataset_var, width=52).grid(row=0, column=0, sticky="ew")
    record_name_status_label = tk.Label(
        _record_name_cell,
        text="",
        anchor="w",
        bg=colors.get("panel", "#111111"),
        fg=colors.get("muted", "#777777"),
        font=(colors.get("font_ui", "TkDefaultFont"), 9),
    )
    record_name_status_label.grid(row=1, column=0, sticky="ew")
    ttk.Button(
        record_form,
        text="Suggest Next",
        command=lambda: record_dataset_var.set(suggest_dataset_name(config)[0]),
    ).grid(row=0, column=2, sticky="nw", padx=(6, 0), pady=(8, 2))

    def _run_record_name_check() -> None:
        _record_name_check_job[0] = None
        name = record_dataset_var.get().strip()
        if not name:
            if background_jobs is not None:
                background_jobs.bump(_record_name_hf_check_key)
            record_name_status_label.configure(text="", fg=_record_name_colors.get("muted", "#777777"))
            return
        local_name = repo_name_from_repo_id(name)
        record_data_root = Path(normalize_path(record_dir_var.get()))
        lerobot_data_root = Path(normalize_path(str(lerobot_dir / "data")))
        local_exists = (record_data_root / local_name).exists() or (lerobot_data_root / local_name).exists()
        if local_exists:
            if background_jobs is not None:
                background_jobs.bump(_record_name_hf_check_key)
            record_name_status_label.configure(
                text="Dataset exists locally — recording will overwrite it",
                fg=_record_name_colors.get("error", "#f87171"),
            )
            return
        username = record_hf_username_var.get().strip()
        repo_id = compose_repo_id(username, name) if username else None

        def _apply_hf_result(expected_name: str, expected_repo_id: str, exists: bool | None) -> None:
            current_name = record_dataset_var.get().strip()
            current_owner = record_hf_username_var.get().strip()
            current_repo_id = compose_repo_id(current_owner, current_name) if current_owner else None
            if current_name != expected_name or current_repo_id != expected_repo_id:
                return
            if exists is True:
                record_name_status_label.configure(
                    text=f"Already exists on HF ({expected_repo_id}) — this will fail",
                    fg=_record_name_colors.get("error", "#f87171"),
                )
            elif exists is False:
                record_name_status_label.configure(
                    text="Available — no conflicts found",
                    fg=_record_name_colors.get("success", "#4ade80"),
                )
            else:
                record_name_status_label.configure(
                    text="HF check unavailable — name format is valid, but conflict status is unknown",
                    fg=_record_name_colors.get("muted", "#777777"),
                )

        if repo_id:
            record_name_status_label.configure(
                text="Checking Hugging Face...",
                fg=_record_name_colors.get("muted", "#777777"),
            )
            if background_jobs is not None:
                background_jobs.submit(
                    _record_name_hf_check_key,
                    lambda rid=repo_id: dataset_exists_on_hf(rid),
                    on_success=lambda exists, expected_name=name, expected_repo_id=repo_id: _apply_hf_result(
                        expected_name,
                        expected_repo_id,
                        exists,
                    ),
                    on_error=lambda _exc, expected_name=name, expected_repo_id=repo_id: _apply_hf_result(
                        expected_name,
                        expected_repo_id,
                        None,
                    ),
                )
            else:
                _apply_hf_result(name, repo_id, dataset_exists_on_hf(repo_id))
        else:
            if background_jobs is not None:
                background_jobs.bump(_record_name_hf_check_key)
            record_name_status_label.configure(
                text="Available locally",
                fg=_record_name_colors.get("success", "#4ade80"),
            )

    def _schedule_record_name_check(*_args: object) -> None:
        if _record_name_check_job[0] is not None:
            root.after_cancel(_record_name_check_job[0])
        _record_name_check_job[0] = root.after(600, _run_record_name_check)

    record_dataset_var.trace_add("write", _schedule_record_name_check)
    record_hf_username_var.trace_add("write", _schedule_record_name_check)
    record_dir_var.trace_add("write", _schedule_record_name_check)
    root.after(200, _run_record_name_check)

    ttk.Label(record_form, text="Local dataset save folder", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_dir_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(record_form, text="Browse", command=lambda: choose_folder(record_dir_var)).grid(
        row=1,
        column=2,
        sticky="w",
        padx=(6, 0),
        pady=4,
    )

    ttk.Label(record_form, text="Episodes", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_episodes_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Episode time (seconds)", style="Field.TLabel").grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_duration_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Target Hz (optional)", style="Field.TLabel").grid(
        row=4,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_target_hz_var, width=20).grid(row=4, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Task description", style="Field.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_task_var, width=52).grid(row=5, column=1, sticky="ew", pady=4)

    ttk.Checkbutton(record_form, text="Upload to Hugging Face after recording", variable=record_upload_var).grid(
        row=6,
        column=1,
        sticky="w",
        pady=(8, 8),
    )

    upload_options = ttk.LabelFrame(record_form, text="Upload Options", style="Section.TLabelframe", padding=10)
    upload_options.columnconfigure(1, weight=1)
    ttk.Label(upload_options, text="Hugging Face username", style="Field.TLabel").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_username_var, width=24).grid(
        row=0,
        column=1,
        sticky="w",
        pady=4,
    )

    ttk.Label(upload_options, text="Dataset name on Hugging Face", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_repo_name_var, width=40).grid(
        row=1,
        column=1,
        sticky="ew",
        pady=4,
    )
    ttk.Button(
        upload_options,
        text="Use Dataset Field",
        command=lambda: record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip())),
    ).grid(row=1, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Checkbutton(
        upload_options,
        text="Run LeRobot v3.0 conversion/tagging after upload",
        variable=record_tag_after_upload_var,
    ).grid(row=2, column=1, sticky="w", pady=(6, 2))

    ttk.Label(
        upload_options,
        text="Runs: python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=<repo>",
        style="Muted.TLabel",
    ).grid(row=3, column=1, sticky="w", pady=(0, 2))

    record_advanced_fields = [
        ("robot.type", "Robot type"),
        ("robot.port", "Follower port"),
        ("robot.id", "Follower robot id"),
        ("robot.cameras", "Robot cameras JSON"),
        ("teleop.type", "Teleop type"),
        ("teleop.port", "Leader port"),
        ("teleop.id", "Leader robot id"),
    ]
    record_advanced_vars: dict[str, Any] = {
        key: tk.StringVar(value="")
        for key, _ in record_advanced_fields
    }

    ttk.Checkbutton(
        record_form,
        text="Advanced command options",
        variable=record_advanced_enabled_var,
    ).grid(row=8, column=1, sticky="w", pady=(6, 0))

    record_advanced_frame = ttk.LabelFrame(
        record_form,
        text="Advanced Record Options",
        style="Section.TLabelframe",
        padding=10,
    )
    record_advanced_frame.columnconfigure(1, weight=1)
    ttk.Label(
        record_advanced_frame,
        text=(
            "Advanced is for low-level command flags. "
            "Set dataset name/episodes/task/time/target Hz in the main setup above."
        ),
        style="Muted.TLabel",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    for idx, (key, label) in enumerate(record_advanced_fields, start=1):
        ttk.Label(record_advanced_frame, text=f"{label} (--{key})", style="Field.TLabel").grid(
            row=idx,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        ttk.Entry(record_advanced_frame, textvariable=record_advanced_vars[key], width=58).grid(
            row=idx,
            column=1,
            sticky="ew",
            pady=3,
        )

    custom_row = len(record_advanced_fields) + 1
    ttk.Label(record_advanced_frame, text="Custom args (raw)", style="Field.TLabel").grid(
        row=custom_row,
        column=0,
        sticky="w",
        padx=(0, 8),
        pady=(8, 3),
    )
    ttk.Entry(record_advanced_frame, textvariable=record_custom_args_var, width=58).grid(
        row=custom_row,
        column=1,
        sticky="ew",
        pady=(8, 3),
    )

    record_buttons = ttk.Frame(record_form, style="Panel.TFrame")
    record_buttons.grid(row=10, column=1, sticky="w", pady=(8, 0))
    preview_record_button = ttk.Button(record_buttons, text="Preview Command")
    preview_record_button.pack(side="left")
    run_record_button = ttk.Button(record_buttons, text="Run Record", style="Accent.TButton")
    run_record_button.pack(side="left", padx=(10, 0))
    scan_ports_button = ttk.Button(record_buttons, text="Scan Robot Ports")
    scan_ports_button.pack(side="left", padx=(10, 0))

    record_summary_var = tk.StringVar(value="")
    record_summary_panel = ttk.LabelFrame(record_container, text="Current Robot Snapshot", style="Section.TLabelframe", padding=10)
    record_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(record_summary_panel, textvariable=record_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    camera_slot = ttk.Frame(record_container, style="Panel.TFrame")
    camera_slot.pack(fill="x")

    def refresh_record_summary() -> None:
        schema = resolve_camera_schema(config)
        camera_names = ", ".join(spec.name for spec in schema.specs) if schema.specs else "(none)"
        record_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Runtime camera keys: {camera_keys}\n"
            "Camera stream size: auto-detected at runtime | FPS: {fps} (warmup {warmup}s) | Target Hz: {target_hz}".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                camera_keys=camera_names,
                fps=config.get("camera_fps", 30),
                warmup=config["camera_warmup_s"],
                target_hz=(str(config.get("record_target_hz", "")).strip() or "auto"),
            )
        )

    def _local_dataset_candidates() -> list[Path]:
        root_value = record_dir_var.get().strip() or str(config["record_data_dir"])
        return _list_local_dataset_dirs(Path(normalize_path(root_value)), lerobot_dir)

    dataset_browser_state: dict[str, Any] = {"selected": None}
    dataset_browser_frame = ttk.LabelFrame(record_container, text="Dataset Browser", style="Section.TLabelframe", padding=10)
    dataset_browser_frame.pack(fill="both", expand=True, pady=(10, 0))
    dataset_browser_frame.columnconfigure(0, weight=2)
    dataset_browser_frame.columnconfigure(1, weight=3)
    dataset_browser_frame.rowconfigure(1, weight=1)

    dataset_source_var = tk.StringVar(value="local")
    dataset_owner_var = tk.StringVar(value=str(config.get("hf_username", "")).strip())
    dataset_status_var = tk.StringVar(value="Browse local and Hugging Face datasets.")
    dataset_sources: dict[str, dict[str, Any]] = {}
    _dataset_refresh_job: dict[str, Any] = {"id": None}
    _dataset_busy_job: dict[str, Any] = {"id": None, "ticks": 0}
    _dataset_status_baseline: dict[str, str] = {"text": dataset_status_var.get()}

    browser_toolbar = ttk.Frame(dataset_browser_frame, style="Panel.TFrame")
    browser_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    browser_toolbar.columnconfigure(1, weight=1)

    source_row = ttk.Frame(browser_toolbar, style="Panel.TFrame")
    source_row.grid(row=0, column=0, sticky="w")
    ttk.Label(source_row, text="Source", style="Field.TLabel").pack(side="left")
    ttk.Radiobutton(source_row, text="Local", value="local", variable=dataset_source_var, style="TRadiobutton").pack(side="left", padx=(8, 6))
    ttk.Radiobutton(source_row, text="Hugging Face", value="huggingface", variable=dataset_source_var, style="TRadiobutton").pack(side="left")

    local_controls = ttk.Frame(browser_toolbar, style="Panel.TFrame")
    ttk.Label(local_controls, text="Local root", style="Field.TLabel").pack(side="left", padx=(12, 6))
    local_root_entry = ttk.Entry(local_controls, textvariable=record_dir_var, width=44)
    local_root_entry.pack(side="left", fill="x", expand=True)
    browse_dataset_root_button = ttk.Button(local_controls, text="Browse Root")
    browse_dataset_root_button.pack(side="left", padx=(6, 0))

    hf_controls = ttk.Frame(browser_toolbar, style="Panel.TFrame")
    ttk.Label(hf_controls, text="HF owner", style="Field.TLabel").pack(side="left", padx=(12, 6))
    hf_owner_entry = ttk.Entry(hf_controls, textvariable=dataset_owner_var, width=28)
    hf_owner_entry.pack(side="left")

    browser_actions = ttk.Frame(browser_toolbar, style="Panel.TFrame")
    browser_actions.grid(row=0, column=2, sticky="e")
    refresh_dataset_browser_button = ttk.Button(browser_actions, text="Refresh")
    refresh_dataset_browser_button.pack(side="left")
    apply_dataset_selection_button = ttk.Button(browser_actions, text="Prefill Record Form from Selection")
    apply_dataset_selection_button.pack(side="left", padx=(8, 0))
    sync_local_dataset_button = ttk.Button(
        browser_actions,
        text="Deploy Selected Local Dataset to Hugging Face...",
    )

    dataset_tree = ttk.Treeview(
        dataset_browser_frame,
        columns=("scope", "name"),
        show="headings",
        style="History.Treeview",
        selectmode="browse",
        height=10,
    )
    dataset_tree.heading("scope", text="Source")
    dataset_tree.heading("name", text="Dataset")
    dataset_tree.column("scope", width=120, anchor="w")
    dataset_tree.column("name", width=360, anchor="w")
    dataset_tree.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
    dataset_tree_scroll = ttk.Scrollbar(
        dataset_browser_frame,
        orient="vertical",
        command=dataset_tree.yview,
        style="Dark.Vertical.TScrollbar",
    )
    dataset_tree.configure(yscrollcommand=dataset_tree_scroll.set)
    dataset_tree_scroll.grid(row=1, column=0, sticky="nse")

    dataset_meta_wrap = ttk.Frame(dataset_browser_frame, style="Panel.TFrame")
    dataset_meta_wrap.grid(row=1, column=1, sticky="nsew")
    dataset_meta_wrap.columnconfigure(0, weight=1)
    dataset_meta_wrap.rowconfigure(0, weight=1)

    dataset_meta_text = tk.Text(
        dataset_meta_wrap,
        height=10,
        wrap="word",
        bg=colors.get("surface", "#1a1a1a"),
        fg=colors.get("text", "#eeeeee"),
        insertbackground=colors.get("text", "#eeeeee"),
        relief="flat",
        font=(colors.get("font_mono", "TkFixedFont"), 10),
        padx=8,
        pady=8,
    )
    dataset_meta_scroll = ttk.Scrollbar(
        dataset_meta_wrap,
        orient="vertical",
        command=dataset_meta_text.yview,
        style="Dark.Vertical.TScrollbar",
    )
    dataset_meta_text.configure(yscrollcommand=dataset_meta_scroll.set)
    dataset_meta_text.grid(row=0, column=0, sticky="nsew")
    dataset_meta_scroll.grid(row=0, column=1, sticky="ns")
    dataset_meta_text.configure(state="disabled")

    ttk.Label(dataset_browser_frame, textvariable=dataset_status_var, style="Muted.TLabel", justify="left").grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="w",
        pady=(8, 0),
    )

    def _render_dataset_metadata(payload: dict[str, Any]) -> None:
        dataset_meta_text.configure(state="normal")
        dataset_meta_text.delete("1.0", "end")
        dataset_meta_text.insert("1.0", json.dumps(payload, indent=2, default=str))
        dataset_meta_text.see("1.0")
        dataset_meta_text.configure(state="disabled")

    def _set_browser_empty_state(message: str) -> None:
        _render_dataset_metadata({"message": message})

    def _collect_hf_dataset_sources(owner: str) -> tuple[list[dict[str, Any]], str | None]:
        rows, error_text = list_hf_datasets(owner, limit=200)
        collected: list[dict[str, Any]] = []
        for row in rows:
            repo_id = str(row.get("repo_id", "")).strip()
            if not repo_id:
                continue
            collected.append(
                {
                    "id": f"hf::{repo_id}",
                    "scope": "huggingface",
                    "repo_id": repo_id,
                    "name": repo_id,
                    "metadata": row,
                }
            )
        return collected, error_text

    def _stop_dataset_busy_status(final_text: str | None = None) -> None:
        pending = _dataset_busy_job.get("id")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
            _dataset_busy_job["id"] = None
        if final_text is not None:
            dataset_status_var.set(final_text)

    def _set_dataset_status(message: str) -> None:
        _dataset_status_baseline["text"] = message
        dataset_status_var.set(message)

    def _start_dataset_busy_status(base_text: str) -> None:
        _stop_dataset_busy_status()
        _dataset_busy_job["ticks"] = 0

        def _tick() -> None:
            ticks = int(_dataset_busy_job.get("ticks", 0))
            dots = "." * ((ticks % 3) + 1)
            dataset_status_var.set(f"{base_text}{dots}")
            _dataset_busy_job["ticks"] = ticks + 1
            _dataset_busy_job["id"] = root.after(280, _tick)

        _tick()

    def _refresh_dataset_browser_async(*, preserve_selection: bool = True) -> None:
        if background_jobs is None:
            _refresh_dataset_browser(preserve_selection=preserve_selection)
            return

        selected_before = dataset_tree.selection()
        selected_before_key = selected_before[0] if selected_before else ""
        for item in dataset_tree.get_children():
            dataset_tree.delete(item)
        dataset_sources.clear()
        dataset_browser_state["selected"] = None
        _start_dataset_busy_status("Scanning datasets")
        refresh_dataset_browser_button.configure(state="disabled")

        source_value = dataset_source_var.get()
        owner_value = dataset_owner_var.get().strip()
        record_root_value = record_dir_var.get().strip() or str(config["record_data_dir"])

        def _worker() -> tuple[list[dict[str, Any]], str | None, str]:
            if source_value == "local":
                rows: list[dict[str, Any]] = []
                for dataset_path in _list_local_dataset_dirs(Path(normalize_path(record_root_value)), lerobot_dir):
                    rows.append(
                        {
                            "id": f"local::{dataset_path}",
                            "scope": "local",
                            "path": dataset_path,
                            "name": dataset_path.name,
                        }
                    )
                return rows, None, "local"
            rows, error_text = _collect_hf_dataset_sources(owner_value)
            return rows, error_text, "huggingface"

        def _apply(result: tuple[list[dict[str, Any]], str | None, str]) -> None:
            rows, error_text, mode = result
            if mode == "local":
                if rows:
                    _stop_dataset_busy_status(f"Local datasets in {record_root_value}")
                else:
                    _stop_dataset_busy_status("No local datasets detected in the configured record roots.")
            else:
                if error_text:
                    _stop_dataset_busy_status(error_text)
                else:
                    _stop_dataset_busy_status(f"Hugging Face datasets for {owner_value or '(owner missing)'}")
            _dataset_status_baseline["text"] = dataset_status_var.get()

            for idx, row in enumerate(rows):
                iid = f"dataset-{idx}"
                dataset_sources[iid] = row
                scope_text = "Local" if row.get("scope") == "local" else "Hugging Face"
                dataset_tree.insert("", "end", iid=iid, values=(scope_text, row.get("name", "-")))

            if not rows:
                _set_browser_empty_state("No datasets found. Switch source or refresh.")
                _sync_dataset_action_buttons()
                return

            if preserve_selection and selected_before_key in dataset_sources:
                dataset_tree.selection_set(selected_before_key)
                dataset_tree.see(selected_before_key)
            else:
                dataset_tree.selection_set("dataset-0")
                dataset_tree.see("dataset-0")

            selected_now = dataset_tree.selection()
            if selected_now:
                _render_selected_dataset_metadata(selected_now[0])
            _sync_dataset_action_buttons()

        def _done(_: bool) -> None:
            refresh_dataset_browser_button.configure(state="normal")

        def _on_refresh_error(exc: Exception) -> None:
            message = f"Dataset refresh failed: {exc}"
            _stop_dataset_busy_status(message)
            _dataset_status_baseline["text"] = message

        background_jobs.submit(
            "record-dataset-refresh",
            _worker,
            on_success=_apply,
            on_error=_on_refresh_error,
            on_complete=_done,
        )

    def _refresh_dataset_browser(*, preserve_selection: bool = True) -> None:
        selected_before = dataset_tree.selection()
        selected_before_key = selected_before[0] if selected_before else ""
        for item in dataset_tree.get_children():
            dataset_tree.delete(item)
        dataset_sources.clear()
        dataset_browser_state["selected"] = None

        if dataset_source_var.get() == "local":
            rows = []
            for dataset_path in _local_dataset_candidates():
                rows.append(
                    {
                        "id": f"local::{dataset_path}",
                        "scope": "local",
                        "path": dataset_path,
                        "name": dataset_path.name,
                    }
                )
            if rows:
                _set_dataset_status(f"Local datasets in {record_dir_var.get().strip() or config['record_data_dir']}")
            else:
                _set_dataset_status("No local datasets detected in the configured record roots.")
        else:
            owner = dataset_owner_var.get().strip()
            rows, error_text = _collect_hf_dataset_sources(owner)
            if error_text:
                _set_dataset_status(error_text)
            else:
                _set_dataset_status(f"Hugging Face datasets for {owner or '(owner missing)'}")

        for idx, row in enumerate(rows):
            iid = f"dataset-{idx}"
            dataset_sources[iid] = row
            scope_text = "Local" if row.get("scope") == "local" else "Hugging Face"
            dataset_tree.insert("", "end", iid=iid, values=(scope_text, row.get("name", "-")))

        if not rows:
            _set_browser_empty_state("No datasets found. Switch source or refresh.")
            _sync_dataset_action_buttons()
            return

        if preserve_selection and selected_before_key in dataset_sources:
            dataset_tree.selection_set(selected_before_key)
            dataset_tree.see(selected_before_key)
        else:
            dataset_tree.selection_set("dataset-0")
            dataset_tree.see("dataset-0")

        selected_now = dataset_tree.selection()
        if selected_now:
            _render_selected_dataset_metadata(selected_now[0])
        _sync_dataset_action_buttons()

    def _build_selected_dataset_metadata_payload(source: dict[str, Any]) -> dict[str, Any]:
        scope = source.get("scope")
        if scope == "local":
            dataset_path = Path(source["path"])
            return {
                "scope": "local",
                "name": source.get("name"),
                "path": str(dataset_path),
                "summary": _build_local_dataset_metadata(dataset_path),
            }

        repo_id = str(source.get("repo_id", "")).strip()
        info, error_text = get_hf_dataset_info(repo_id)
        payload = {
            "scope": "huggingface",
            "repo_id": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "summary": source.get("metadata", {}),
        }
        if info is not None:
            payload["metadata"] = info
        else:
            payload["metadata_error"] = error_text or "Unable to fetch Hugging Face metadata."
        return payload

    def _render_selected_dataset_metadata(iid: str) -> None:
        source = dataset_sources.get(iid)
        if source is None:
            return
        dataset_browser_state["selected"] = source

        def _apply(payload: dict[str, Any]) -> None:
            _render_dataset_metadata(payload)

        if background_jobs is None:
            _apply(_build_selected_dataset_metadata_payload(source))
            return

        _start_dataset_busy_status("Loading dataset metadata")
        background_jobs.submit(
            "record-dataset-metadata",
            lambda: _build_selected_dataset_metadata_payload(source),
            on_success=_apply,
            on_error=lambda exc: _render_dataset_metadata({"error": f"Dataset metadata load failed: {exc}"}),
            on_complete=lambda is_stale: None
            if is_stale
            else _stop_dataset_busy_status(_dataset_status_baseline["text"]),
        )

    def _selected_dataset_source() -> dict[str, Any] | None:
        selected = dataset_tree.selection()
        if not selected:
            return None
        return dataset_sources.get(selected[0])

    def _has_selected_local_dataset() -> bool:
        source = _selected_dataset_source()
        if source is None or source.get("scope") != "local":
            return False
        path_raw = source.get("path")
        path = path_raw if isinstance(path_raw, Path) else Path(str(path_raw)) if path_raw else None
        return bool(path is not None and path.exists() and path.is_dir())

    def _sync_dataset_action_buttons() -> None:
        selected_source = _selected_dataset_source()
        apply_dataset_selection_button.configure(state="normal" if selected_source is not None else "disabled")

        local_mode = dataset_source_var.get() == "local"
        if local_mode and _has_selected_local_dataset():
            if not sync_local_dataset_button.winfo_manager():
                sync_local_dataset_button.pack(side="left", padx=(8, 0))
            sync_local_dataset_button.configure(state="normal")
        else:
            if sync_local_dataset_button.winfo_manager():
                sync_local_dataset_button.pack_forget()

    def _apply_selected_dataset_to_record() -> None:
        selected = dataset_tree.selection()
        if not selected:
            messagebox.showinfo("Dataset Browser", "Select a dataset first.")
            return
        source = dataset_sources.get(selected[0])
        if source is None:
            return

        if source.get("scope") == "local":
            dataset_path = Path(source["path"])
            record_dir_var.set(str(dataset_path.parent))
            record_dataset_var.set(dataset_path.name)
            record_hf_repo_name_var.set(dataset_path.name)
            _set_dataset_status(
                f"Prefilled recording form from local dataset '{dataset_path.name}' (save folder + dataset name)."
            )
            return

        repo_id = str(source.get("repo_id", "")).strip()
        if not repo_id:
            return
        record_dataset_var.set(repo_id)
        if "/" in repo_id:
            owner, dataset_name = repo_id.split("/", 1)
            record_hf_username_var.set(owner)
            record_hf_repo_name_var.set(dataset_name)
            dataset_owner_var.set(owner)
        _set_dataset_status(
            f"Prefilled recording form from Hugging Face dataset '{repo_id}' (dataset/repo fields updated)."
        )

    def _sync_dataset_browser_controls() -> None:
        is_local = dataset_source_var.get() == "local"
        if is_local:
            if hf_controls.winfo_manager():
                hf_controls.grid_remove()
            if not local_controls.winfo_manager():
                local_controls.grid(row=0, column=1, sticky="ew")
            local_root_entry.configure(state="normal")
        else:
            if local_controls.winfo_manager():
                local_controls.grid_remove()
            if not hf_controls.winfo_manager():
                hf_controls.grid(row=0, column=1, sticky="w")
        _sync_dataset_action_buttons()
        _refresh_dataset_browser_async(preserve_selection=False)

    def _on_dataset_tree_selected(_: Any) -> None:
        selected = dataset_tree.selection()
        if not selected:
            _sync_dataset_action_buttons()
            return
        _render_selected_dataset_metadata(selected[0])
        _sync_dataset_action_buttons()

    def _schedule_dataset_browser_refresh(*_: Any) -> None:
        pending = _dataset_refresh_job.get("id")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        _dataset_refresh_job["id"] = root.after(260, lambda: _refresh_dataset_browser_async(preserve_selection=False))

    def _browse_dataset_root() -> None:
        choose_folder(record_dir_var)
        _refresh_dataset_browser_async(preserve_selection=False)

    browse_dataset_root_button.configure(command=_browse_dataset_root)
    refresh_dataset_browser_button.configure(command=lambda: _refresh_dataset_browser_async(preserve_selection=False))
    apply_dataset_selection_button.configure(command=_apply_selected_dataset_to_record)
    sync_local_dataset_button.configure(command=lambda: open_sync_to_hf_popup(require_selected_local_dataset=True))
    dataset_tree.bind("<<TreeviewSelect>>", _on_dataset_tree_selected)
    hf_owner_entry.bind("<Return>", lambda *_: _refresh_dataset_browser_async(preserve_selection=False))
    dataset_source_var.trace_add("write", lambda *_: _sync_dataset_browser_controls())
    record_dir_var.trace_add("write", _schedule_dataset_browser_refresh)
    _sync_dataset_browser_controls()

    hf_sync_popup_state: dict[str, Any] = {"window": None}

    def open_sync_to_hf_popup(*, require_selected_local_dataset: bool = False) -> None:
        if require_selected_local_dataset and not _has_selected_local_dataset():
            messagebox.showinfo("Dataset Browser", "Select an existing local dataset first.")
            return

        popup = hf_sync_popup_state.get("window")
        if popup is not None and bool(popup.winfo_exists()):
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            return

        from tkinter import filedialog as _fd

        dataset_candidates = _local_dataset_candidates()
        selected_browser_entry = dataset_browser_state.get("selected")
        default_local_dataset = ""
        if isinstance(selected_browser_entry, dict) and selected_browser_entry.get("scope") == "local":
            selected_path = selected_browser_entry.get("path")
            if isinstance(selected_path, Path):
                default_local_dataset = str(selected_path)
            elif selected_path:
                default_local_dataset = str(selected_path)
        if not default_local_dataset:
            default_local_dataset = str(config.get("record_hf_sync_local_dataset", "")).strip()
        if not default_local_dataset and dataset_candidates:
            default_local_dataset = str(dataset_candidates[0])

        default_owner = (
            str(config.get("record_hf_sync_owner", "")).strip()
            or str(dataset_owner_var.get()).strip()
            or str(config.get("hf_username", "")).strip()
        )
        default_dataset_name = repo_name_only(str(config.get("record_hf_sync_repo_name", "")).strip(), owner=default_owner)
        if not default_dataset_name and default_local_dataset:
            default_dataset_name = Path(default_local_dataset).name

        local_dataset_var = tk.StringVar(value=default_local_dataset)
        owner_var = tk.StringVar(value=default_owner)
        repo_name_var = tk.StringVar(value=default_dataset_name)
        convert_var = tk.BooleanVar(value=bool(config.get("record_hf_convert_after_upload", record_tag_after_upload_var.get())))
        skip_if_exists_var = tk.BooleanVar(value=bool(config.get("record_hf_sync_skip_if_exists", True)))
        status_var = tk.StringVar(value="Upload local dataset to Hugging Face with local/remote parity checks.")

        popup = tk.Toplevel(root)
        popup.title("Deploy Dataset to Hugging Face")
        fit_window_to_screen(
            window=popup,
            requested_width=940,
            requested_height=430,
            requested_min_width=840,
            requested_min_height=370,
        )
        popup.configure(bg=colors.get("panel", "#111111"))
        popup.transient(root)
        hf_sync_popup_state["window"] = popup

        def _on_close() -> None:
            hf_sync_popup_state["window"] = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)

        container = ttk.Frame(popup, style="Panel.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        ttk.Label(container, text="Local dataset folder", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=local_dataset_var, width=68).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4,
        )
        browse_local_button = ttk.Button(container, text="Browse...")
        browse_local_button.grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Detected local datasets", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        dataset_combo = ttk.Combobox(
            container,
            values=[str(path) for path in dataset_candidates],
            state="readonly",
            width=68,
            style="Dark.TCombobox",
        )
        dataset_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        refresh_datasets_button = ttk.Button(container, text="Refresh Datasets")
        refresh_datasets_button.grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(container, text="HF owner", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=owner_var, width=28).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(container, text="HF dataset name", style="Field.TLabel").grid(
            row=2, column=2, sticky="w", padx=(12, 6), pady=4,
        )
        ttk.Entry(container, textvariable=repo_name_var, width=30).grid(row=2, column=3, sticky="ew", pady=4)

        parity_row = ttk.Frame(container, style="Panel.TFrame")
        parity_row.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))
        check_parity_button = ttk.Button(parity_row, text="Check Local/Remote Parity")
        check_parity_button.pack(side="left")

        controls_row = ttk.Frame(container, style="Panel.TFrame")
        controls_row.grid(row=4, column=1, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            controls_row,
            text="Skip upload when remote dataset already exists",
            variable=skip_if_exists_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Run LeRobot v3.0 conversion/tagging after upload",
            variable=convert_var,
        ).pack(anchor="w")

        buttons_row = ttk.Frame(container, style="Panel.TFrame")
        buttons_row.grid(row=5, column=1, columnspan=3, sticky="w", pady=(10, 0))
        preview_sync_button = ttk.Button(buttons_row, text="Preview HF Upload Command")
        preview_sync_button.pack(side="left")
        run_sync_button = ttk.Button(buttons_row, text="Deploy to Hugging Face", style="Accent.TButton")
        run_sync_button.pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=status_var, style="Muted.TLabel", justify="left").grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        def _refresh_dataset_options() -> None:
            options = [str(path) for path in _local_dataset_candidates()]
            dataset_combo.configure(values=options)
            current = local_dataset_var.get().strip()
            if not current and options:
                local_dataset_var.set(options[0])
            if current and current in options:
                dataset_combo.set(current)

        def _choose_local_dataset() -> None:
            current = local_dataset_var.get().strip()
            initial_dir = current or str(record_dir_var.get().strip() or config["record_data_dir"])
            selected = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=initial_dir,
                title="Select local dataset folder",
            )
            if selected:
                local_dataset_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        def _save_hf_sync_settings() -> None:
            cleaned_repo_name = repo_name_only(repo_name_var.get(), owner=owner_var.get())
            if cleaned_repo_name != str(repo_name_var.get()).strip():
                repo_name_var.set(cleaned_repo_name)
            config["record_hf_sync_local_dataset"] = local_dataset_var.get().strip()
            config["record_hf_sync_owner"] = owner_var.get().strip()
            config["record_hf_sync_repo_name"] = cleaned_repo_name
            config["record_hf_convert_after_upload"] = bool(convert_var.get())
            config["record_hf_sync_skip_if_exists"] = bool(skip_if_exists_var.get())
            save_config(config, quiet=True)

        def _build_hf_sync_request() -> tuple[dict[str, Any] | None, str | None]:
            local_dataset = Path(normalize_path(local_dataset_var.get().strip()))
            if not local_dataset.exists() or not local_dataset.is_dir():
                return None, f"Local dataset folder not found: {local_dataset}"

            cleaned_repo_name = repo_name_only(repo_name_var.get(), owner=owner_var.get())
            if cleaned_repo_name != str(repo_name_var.get()).strip():
                repo_name_var.set(cleaned_repo_name)
            repo_id = compose_repo_id(owner_var.get(), cleaned_repo_name)
            if repo_id is None:
                return None, "Hugging Face owner and dataset name are required."

            hf_cli_path = shutil.which("huggingface-cli")
            if hf_cli_path is None:
                return None, "huggingface-cli not found in PATH."

            upload_cmd = [
                "huggingface-cli",
                "upload",
                repo_id,
                str(local_dataset),
                "--repo-type",
                "dataset",
            ]

            exists = dataset_exists_on_hf(repo_id)
            parity_level, parity_detail = _hf_parity_detail(exists, repo_id)
            checks: list[tuple[str, str, str]] = [
                ("PASS", "Local dataset", str(local_dataset)),
                ("PASS", "Target repo", repo_id),
                ("PASS", "huggingface-cli", hf_cli_path),
                (parity_level, "Parity", parity_detail),
            ]

            convert_cmd = _build_v30_convert_command(repo_id) if convert_var.get() else None
            if convert_cmd is not None:
                checks.append(("PASS", "v3.0 convert command", format_command(convert_cmd)))

            return {
                "local_dataset": local_dataset,
                "repo_id": repo_id,
                "upload_cmd": upload_cmd,
                "convert_cmd": convert_cmd,
                "remote_exists": exists,
                "parity_detail": parity_detail,
                "checks": checks,
            }, None

        def _run_hf_convert_step(*, repo_id: str, convert_cmd: list[str], local_dataset: Path) -> None:
            set_running(False, "Upload completed. Running v3.0 conversion/tagging...")

            def after_convert(convert_code: int, convert_canceled: bool) -> None:
                if convert_canceled:
                    set_running(False, "HF upload completed. v3.0 conversion canceled.")
                    messagebox.showwarning(
                        "Upload Completed",
                        (
                            "Dataset upload completed, but v3.0 conversion was canceled.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}"
                        ),
                    )
                    return
                if convert_code != 0:
                    set_running(False, "HF upload completed. v3.0 conversion failed.", True)
                    messagebox.showwarning(
                        "Upload Completed (Conversion Warning)",
                        (
                            "Dataset upload completed, but v3.0 conversion/tagging failed.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}\n"
                            f"Conversion exit code: {convert_code}"
                        ),
                    )
                    return
                set_running(False, "HF upload + v3.0 conversion completed.")
                messagebox.showinfo(
                    "Done",
                    (
                        "Dataset upload and v3.0 conversion/tagging completed.\n\n"
                        f"Local dataset: {local_dataset}\n"
                        f"Hugging Face repo: {repo_id}"
                    ),
                )

            run_process_async(
                convert_cmd,
                lerobot_dir,
                after_convert,
                None,
                None,
                "upload",
                None,
                {"dataset_repo_id": repo_id},
            )

        def check_parity_now() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("HF Parity Check", error_text or "Unable to build parity check.")
                return
            status_var.set(str(request.get("parity_detail", "Parity check complete.")))

        def preview_sync_command() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("Deploy to Hugging Face", error_text or "Unable to build upload command.")
                return
            last_command_state["value"] = format_command(request["upload_cmd"])
            details = [
                "Upload command:",
                format_command_for_dialog(request["upload_cmd"]),
            ]
            convert_cmd = request["convert_cmd"]
            if convert_cmd is not None:
                details.append("\nThen run:")
                details.append(format_command_for_dialog(convert_cmd))
            show_text_dialog(
                root=root,
                title="HF Upload Command",
                text="\n".join(details),
                copy_text=last_command_state["value"],
                wrap_mode="word",
            )

        def run_sync_command() -> None:
            request, error_text = _build_hf_sync_request()
            if error_text or request is None:
                messagebox.showerror("Deploy to Hugging Face", error_text or "Unable to build upload command.")
                return

            repo_id = request["repo_id"]
            local_dataset = request["local_dataset"]
            remote_exists = request["remote_exists"]
            if remote_exists is True and skip_if_exists_var.get():
                messagebox.showinfo(
                    "Skipped",
                    f"Remote dataset already exists. Upload skipped:\n{repo_id}",
                )
                return
            if remote_exists is True and not messagebox.askyesno(
                "Remote Dataset Exists",
                f"{repo_id} already exists on Hugging Face.\nContinue upload anyway?",
            ):
                return
            if remote_exists is None and not messagebox.askyesno(
                "Parity Unknown",
                (
                    f"Could not verify remote parity for {repo_id}.\n"
                    "Continue upload anyway?"
                ),
            ):
                return

            if not confirm_preflight_in_gui("HF Dataset Deploy Preflight", request["checks"]):
                return

            if not ask_text_dialog(
                root=root,
                title="Confirm HF Dataset Deploy",
                text=(
                    "Review the upload command below.\n"
                    "Click Confirm to run it, or Cancel to stop.\n\n"
                    + format_command_for_dialog(request["upload_cmd"])
                ),
                copy_text=format_command(request["upload_cmd"]),
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            ):
                return

            _save_hf_sync_settings()
            config["hf_username"] = str(owner_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            config["record_tag_after_upload"] = bool(convert_var.get())
            save_config(config, quiet=True)
            refresh_header_subtitle()

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "HF upload canceled.")
                    messagebox.showinfo("Canceled", "Hugging Face upload was canceled.")
                    return
                if upload_code != 0:
                    set_running(False, "HF upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Hugging Face upload failed with exit code {upload_code}.")
                    return

                convert_cmd = request["convert_cmd"]
                if convert_cmd is None:
                    set_running(False, "HF upload completed.")
                    messagebox.showinfo(
                        "Done",
                        (
                            "Dataset upload to Hugging Face completed.\n\n"
                            f"Local dataset: {local_dataset}\n"
                            f"Hugging Face repo: {repo_id}\n"
                            "v3.0 conversion/tagging: skipped"
                        ),
                    )
                    return

                _run_hf_convert_step(repo_id=repo_id, convert_cmd=convert_cmd, local_dataset=local_dataset)

            run_process_async(
                request["upload_cmd"],
                lerobot_dir,
                after_upload,
                None,
                None,
                "upload",
                request["checks"],
                {"dataset_repo_id": repo_id},
            )

        def _on_dataset_combo_selected(_: Any) -> None:
            selected = dataset_combo.get().strip()
            if selected:
                local_dataset_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        browse_local_button.configure(command=_choose_local_dataset)
        refresh_datasets_button.configure(command=_refresh_dataset_options)
        dataset_combo.bind("<<ComboboxSelected>>", _on_dataset_combo_selected)
        check_parity_button.configure(command=check_parity_now)
        preview_sync_button.configure(command=preview_sync_command)
        run_sync_button.configure(command=run_sync_command)
        _refresh_dataset_options()

    def _seed_record_advanced_from_current() -> None:
        dataset_input, dataset_error = _record_dataset_input_from_ui()
        if dataset_error is not None or dataset_input is None:
            return
        req, cmd, error_text = build_record_request_and_command(
            config=config,
            dataset_input=dataset_input,
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
            target_hz_raw=record_target_hz_var.get(),
        )
        if error_text or req is None or cmd is None:
            return
        for key, _ in record_advanced_fields:
            value = get_flag_value(cmd, key)
            if value is not None:
                record_advanced_vars[key].set(value)

    def _refresh_record_advanced_visibility(*_: Any) -> None:
        if record_advanced_enabled_var.get():
            _seed_record_advanced_from_current()
            record_advanced_frame.grid(row=9, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            record_advanced_frame.grid_remove()

    def preview_record() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return
        last_command_state["value"] = format_command(cmd)
        command_for_dialog = format_command_for_dialog(cmd)
        log_panel.append_log("Preview record command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Record Command",
            text=command_for_dialog,
            wrap_mode="word",
        )

    def scan_robot_ports() -> None:
        entries = scan_robot_serial_ports()
        report = format_robot_port_scan(entries)
        show_text_dialog(
            root=root,
            title="Robot Port Scan",
            text=report,
            wrap_mode="word",
        )
        if not entries:
            log_panel.append_log("Robot port scan: no candidate ports found.")
            return
        follower_guess, leader_guess = suggest_follower_leader_ports(
            entries,
            current_follower=str(config.get("follower_port", "")),
            current_leader=str(config.get("leader_port", "")),
        )
        log_panel.append_log(
            "Robot port scan detected: "
            + ", ".join(str(item.get("path", "")) for item in entries)
        )
        if follower_guess and leader_guess:
            apply_guess = messagebox.askyesno(
                "Apply Detected Ports",
                (
                    "Detected candidate motor-controller ports.\n\n"
                    f"Set follower -> {follower_guess}\n"
                    f"Set leader -> {leader_guess}\n\n"
                    "Apply these as record defaults now?"
                ),
            )
            if apply_guess:
                config["follower_port"] = follower_guess
                config["leader_port"] = leader_guess
                if record_advanced_enabled_var.get():
                    record_advanced_vars["robot.port"].set(follower_guess)
                    record_advanced_vars["teleop.port"].set(leader_guess)
                save_config(config, quiet=True)
                refresh_header_subtitle()
                refresh_record_summary()
                log_panel.append_log(
                    f"Applied scanned record defaults: follower={follower_guess}, leader={leader_guess}"
                )

    def run_record_from_gui() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return

        config["record_data_dir"] = str(req.dataset_root)
        config["record_target_hz"] = str(record_target_hz_var.get()).strip()
        resolved_repo_id, adjusted, _ = resolve_unique_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=req.dataset_repo_id,
            local_roots=[req.dataset_root, lerobot_dir / "data"],
        )
        if adjusted:
            if record_upload_var.get():
                owner, name = resolved_repo_id.split("/", 1)
                record_hf_username_var.set(owner)
                record_hf_repo_name_var.set(name)
                log_panel.append_log(f"Auto-iterated Hugging Face dataset to avoid existing target: {resolved_repo_id}")
            else:
                record_dataset_var.set(resolved_repo_id)
                log_panel.append_log(f"Auto-iterated dataset to avoid existing target: {resolved_repo_id}")

            req, cmd, error_text = build_current_record_from_ui()
            if error_text or req is None or cmd is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build command.")
                return

        editable_cmd = ask_editable_command_dialog(
            root=root,
            title="Confirm Record Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the record command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Record",
            cancel_label="Cancel",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            log_panel.append_log("Running edited record command from command editor.")
        cmd = editable_cmd

        effective_repo_id = normalize_repo_id(
            str(config["hf_username"]),
            get_flag_value(cmd, "dataset.repo_id") or req.dataset_repo_id,
        )
        effective_dataset_name = repo_name_from_repo_id(effective_repo_id)
        effective_dataset_root = req.dataset_root
        effective_dataset_root_text = (get_flag_value(cmd, "dataset.root") or "").strip()
        if effective_dataset_root_text:
            effective_dataset_root = Path(normalize_path(effective_dataset_root_text))
        episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.num_episodes)
        duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.episode_time_s)
        try:
            effective_num_episodes = int(str(episodes_text).strip())
            effective_episode_time = int(str(duration_text).strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Edited command must keep episodes and episode time as integers.")
            return
        if effective_num_episodes <= 0 or effective_episode_time <= 0:
            messagebox.showerror("Validation Error", "Edited command must keep episodes and episode time greater than zero.")
            return

        exists = dataset_exists_on_hf(effective_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{effective_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        preflight_checks = run_preflight_for_record(
            config=config,
            dataset_root=effective_dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=effective_episode_time,
            dataset_repo_id=effective_repo_id,
        )
        if not confirm_preflight_in_gui("Record Preflight", preflight_checks):
            return

        last_command_state["value"] = format_command(cmd)

        def after_record(return_code: int, was_canceled: bool) -> None:
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Record canceled.")
                    messagebox.showinfo("Canceled", "Record command was canceled.")
                else:
                    set_running(False, "Recording failed.", True)
                    messagebox.showerror("Record Failed", f"Recording failed with exit code {return_code}.")
                return

            active_dataset = move_recorded_dataset(
                lerobot_dir=lerobot_dir,
                dataset_name=effective_dataset_name,
                dataset_root=effective_dataset_root,
                log=log_panel.append_log,
            )

            config["record_data_dir"] = str(effective_dataset_root)
            config["last_dataset_name"] = effective_dataset_name
            config["last_dataset_repo_id"] = effective_repo_id
            if record_upload_var.get():
                config["hf_username"] = str(record_hf_username_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            config["record_tag_after_upload"] = bool(record_tag_after_upload_var.get())
            save_config(config)
            next_dataset_name, _ = suggest_dataset_name(config)
            record_dataset_var.set(next_dataset_name)
            if record_upload_var.get():
                record_hf_repo_name_var.set(next_dataset_name)
            refresh_record_summary()
            refresh_header_subtitle()

            if was_canceled:
                set_running(False, "Record canceled.")
                messagebox.showinfo("Canceled", "Record command was canceled. Upload was skipped.")
                return

            if not req.upload_after_record:
                set_running(False, "Record completed.")
                messagebox.showinfo("Done", "Recording completed.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                effective_repo_id,
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]

            set_running(False, "Record completed. Starting upload...")

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "Upload canceled.")
                    messagebox.showinfo("Canceled", "Upload command was canceled.")
                elif upload_code != 0:
                    set_running(False, "Upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Upload failed with exit code {upload_code}.")
                else:
                    config["last_dataset_repo_id"] = effective_repo_id
                    save_config(config, quiet=True)
                    if not record_tag_after_upload_var.get():
                        set_running(False, "Record + upload completed.")
                        messagebox.showinfo(
                            "Done",
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {effective_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded dataset: {effective_dataset_name}\n"
                            f"Hugging Face repo: {effective_repo_id}\n"
                            "v3.0 conversion/tagging: skipped",
                        )
                        return

                    convert_cmd = _build_v30_convert_command(effective_repo_id)

                    set_running(False, "Upload completed. Running v3.0 conversion/tagging...")

                    def after_convert(convert_code: int, convert_canceled: bool) -> None:
                        base_details = (
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {effective_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded name: {effective_dataset_name}\n"
                            f"Hugging Face repo: {effective_repo_id}\n"
                        )
                        if convert_canceled:
                            set_running(False, "Upload completed. v3.0 conversion canceled.")
                            messagebox.showwarning(
                                "Upload Completed",
                                base_details + "v3.0 conversion/tagging: canceled",
                            )
                            return
                        if convert_code != 0:
                            set_running(False, "Upload completed. v3.0 conversion failed.", True)
                            messagebox.showwarning(
                                "Upload Completed (Conversion Warning)",
                                base_details + f"v3.0 conversion/tagging: failed (exit code {convert_code})",
                            )
                            return

                        config["last_dataset_repo_id"] = effective_repo_id
                        save_config(config, quiet=True)
                        set_running(False, "Record + upload + v3.0 conversion completed.")
                        messagebox.showinfo(
                            "Done",
                            base_details + "v3.0 conversion/tagging: success",
                        )

                    run_process_async(
                        cmd=convert_cmd,
                        cwd=lerobot_dir,
                        complete_callback=after_convert,
                        expected_episodes=None,
                        expected_seconds=None,
                        run_mode="upload",
                        preflight_checks=None,
                        artifact_context={"dataset_repo_id": effective_repo_id},
                    )

            run_process_async(
                upload_cmd,
                get_lerobot_dir(config),
                after_upload,
                None,
                None,
                "upload",
                None,
                {"dataset_repo_id": effective_repo_id},
            )

        run_process_async(
            cmd,
            get_lerobot_dir(config),
            after_record,
            effective_num_episodes,
            effective_num_episodes * effective_episode_time,
            "record",
            preflight_checks,
            {"dataset_repo_id": effective_repo_id},
        )

    def _record_dataset_input_from_ui() -> tuple[str | None, str | None]:
        dataset_input = record_dataset_var.get()
        if record_upload_var.get():
            hf_username = str(record_hf_username_var.get()).strip().strip("/")
            hf_dataset_name = repo_name_only(record_hf_repo_name_var.get(), owner=hf_username)
            if hf_dataset_name != str(record_hf_repo_name_var.get()).strip().strip("/"):
                record_hf_repo_name_var.set(hf_dataset_name)
            if not hf_username:
                return None, "Hugging Face username is required when upload is enabled."
            if not hf_dataset_name:
                return None, "Hugging Face dataset name is required when upload is enabled."
            dataset_input = f"{hf_username}/{hf_dataset_name}"
        return dataset_input, None

    def _record_advanced_overrides_from_ui() -> tuple[dict[str, str] | None, str]:
        if not record_advanced_enabled_var.get():
            return None, ""
        overrides: dict[str, str] = {}
        for key, _ in record_advanced_fields:
            value = str(record_advanced_vars[key].get()).strip()
            if value:
                overrides[key] = value
        return overrides or None, str(record_custom_args_var.get())

    def build_current_record_from_ui() -> tuple[Any, Any, str | None]:
        dataset_input, dataset_error = _record_dataset_input_from_ui()
        if dataset_error is not None or dataset_input is None:
            return None, None, dataset_error or "Dataset name is required."
        arg_overrides, custom_args_raw = _record_advanced_overrides_from_ui()
        return build_record_request_and_command(
            config=config,
            dataset_input=dataset_input,
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
            target_hz_raw=record_target_hz_var.get(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def refresh_upload_options_visibility(*_: Any) -> None:
        if record_upload_var.get():
            if not str(record_hf_username_var.get()).strip():
                record_hf_username_var.set(str(config.get("hf_username", "")).strip())
            if not str(record_hf_repo_name_var.get()).strip():
                record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip()))
            upload_options.grid(row=7, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            upload_options.grid_remove()

    refresh_upload_options_visibility()
    record_upload_var.trace_add("write", refresh_upload_options_visibility)
    _refresh_record_advanced_visibility()
    record_advanced_enabled_var.trace_add("write", _refresh_record_advanced_visibility)

    preview_record_button.configure(command=preview_record)
    run_record_button.configure(command=run_record_from_gui)
    scan_ports_button.configure(command=scan_robot_ports)

    def apply_theme(updated_colors: dict[str, str]) -> None:
        _record_name_colors.update(updated_colors)
        record_name_status_label.configure(
            bg=updated_colors.get("panel", "#111111"),
            font=(updated_colors.get("font_ui", "TkDefaultFont"), 9),
        )
        dataset_meta_text.configure(
            bg=updated_colors.get("surface", "#1a1a1a"),
            fg=updated_colors.get("text", "#eeeeee"),
            insertbackground=updated_colors.get("text", "#eeeeee"),
            font=(updated_colors.get("font_mono", "TkFixedFont"), 10),
        )

    return RecordTabHandles(
        record_dir_var=record_dir_var,
        camera_slot=camera_slot,
        refresh_summary=refresh_record_summary,
        apply_theme=apply_theme,
        action_buttons=[preview_record_button, run_record_button, scan_ports_button, sync_local_dataset_button],
    )
