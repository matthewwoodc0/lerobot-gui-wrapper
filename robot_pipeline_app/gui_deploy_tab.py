from __future__ import annotations

import posixpath
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .command_overrides import get_flag_value
from .checks import run_preflight_for_deploy, summarize_checks
from .deploy_diagnostics import find_nested_model_candidates, is_runnable_model_path
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import (
    ask_text_dialog,
    ask_text_dialog_with_actions,
    format_command_for_dialog,
    show_text_dialog,
)
from .gui_file_dialogs import ask_directory_dialog
from .gui_forms import build_deploy_request_and_command
from .gui_log import GuiLogPanel
from .repo_utils import (
    dataset_exists_on_hf,
    resolve_unique_repo_id,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .runner import format_command
from .training_profiles import load_training_profiles
from .training_remote import (
    build_pull_command,
    build_sftp_pull_command,
    command_uses_binary,
    ensure_host_trusted,
)
from .types import GuiRunProcessAsync, TrainingProfile

_MODEL_TREE_MAX_DEPTH = 4
_MODEL_TREE_BOTTOM_SPACER_ROWS = 2


def _remote_path_name(remote_path: str) -> str:
    raw = str(remote_path or "").strip().rstrip("/")
    if not raw:
        return "remote_model"
    name = posixpath.basename(raw)
    return name or "remote_model"


def _default_pull_destination(trained_models_dir: str, remote_path: str) -> Path:
    root = Path(normalize_path(trained_models_dir))
    return root / _remote_path_name(remote_path)


def _increment_pull_destination(path: Path) -> Path:
    parent = path.parent
    base = path.name
    idx = 2
    candidate = parent / f"{base}_{idx}"
    while candidate.exists():
        idx += 1
        candidate = parent / f"{base}_{idx}"
    return candidate


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


def _first_model_payload_candidate(checks: list[tuple[str, str, str]]) -> str | None:
    for _, name, detail in checks:
        if name.strip().lower() != "model payload candidates":
            continue
        candidate = detail.split(",", 1)[0].strip()
        return candidate or None
    return None


def _resolve_payload_path(path: Path) -> Path:
    if is_runnable_model_path(path):
        return path
    candidates = find_nested_model_candidates(path)
    return candidates[0] if candidates else path


def _model_tree_node_kind(path: Path) -> tuple[str, str]:
    if is_runnable_model_path(path):
        return "Model", "model_root"

    candidates = find_nested_model_candidates(path, max_depth=3, limit=1)
    if candidates:
        if "checkpoint" in path.name.lower():
            return "Checkpoint -> model", "resolved"
        return "Contains model", "resolved"

    try:
        has_subdirs = any(p.is_dir() for p in path.iterdir())
    except OSError:
        has_subdirs = False

    if has_subdirs and "checkpoint" in path.name.lower():
        return "Checkpoint", "checkpoint"
    return ("Folder", "folder") if has_subdirs else ("", "folder")


def _needs_eval_prefix_quick_fix(username: str, dataset_name_or_repo_id: Any) -> bool:
    _, changed = suggest_eval_prefixed_repo_id(
        username=username,
        dataset_name_or_repo_id=dataset_name_or_repo_id,
    )
    return changed


@dataclass
class DeployTabHandles:
    deploy_root_var: Any
    deploy_eval_episodes_var: Any
    deploy_eval_duration_var: Any
    deploy_eval_task_var: Any
    deploy_camera_preview: DualCameraPreview
    refresh_local_models: Callable[[], None]
    select_model_path: Callable[[Path], bool]
    action_buttons: list[Any]


def setup_deploy_tab(
    *,
    root: Any,
    deploy_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    cv2_probe_ok: bool,
    cv2_probe_error: str,
    choose_folder: Callable[[Any], None],
    log_panel: GuiLogPanel,
    messagebox: Any,
    set_running: Callable[[bool, str | None, bool], None],
    run_process_async: GuiRunProcessAsync,
    on_camera_indices_changed: Callable[[int, int], None],
    refresh_header_subtitle: Callable[[], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
) -> DeployTabHandles:
    import tkinter as tk
    from tkinter import ttk

    accent = colors.get("accent", "#f0a500")
    surface = colors.get("surface", "#1a1a1a")
    panel = colors.get("panel", "#111111")
    border = colors.get("border", "#2d2d2d")
    text_col = colors.get("text", "#eeeeee")
    muted = colors.get("muted", "#777777")
    mono_font = colors.get("font_mono", "TkFixedFont")

    deploy_container = ttk.Frame(deploy_tab, style="Panel.TFrame")
    deploy_container.pack(fill="both", expand=True)

    deploy_root_var = tk.StringVar(value=str(config["trained_models_dir"]))

    # Compute initial model path from stored folder + optional checkpoint
    _last_model_folder = str(config.get("last_model_name", "")).strip()
    _last_checkpoint = str(config.get("last_checkpoint_name", "")).strip()
    if _last_model_folder:
        _init_model = str(Path(config["trained_models_dir"]) / _last_model_folder)
        if _last_checkpoint:
            _init_model = str(Path(_init_model) / _last_checkpoint)
    else:
        _init_model = str(config["trained_models_dir"])

    deploy_model_var = tk.StringVar(value=_init_model)
    deploy_eval_dataset_var = tk.StringVar(
        value=str(config.get("last_eval_dataset_name", "")).strip()
        or suggest_eval_dataset_name(config, _last_model_folder)
    )
    deploy_eval_episodes_var = tk.StringVar(value=str(config.get("eval_num_episodes", 10)))
    deploy_eval_duration_var = tk.StringVar(value=str(config.get("eval_duration_s", 20)))
    deploy_eval_task_var = tk.StringVar(value=str(config.get("eval_task", DEFAULT_TASK)))
    deploy_advanced_enabled_var = tk.BooleanVar(value=False)
    deploy_custom_args_var = tk.StringVar(value="")

    # ── Deploy form ───────────────────────────────────────────────────────────
    deploy_form = ttk.LabelFrame(deploy_container, text="Deploy / Eval Setup", style="Section.TLabelframe", padding=12)
    deploy_form.pack(fill="x")
    deploy_form.columnconfigure(1, weight=1)

    ttk.Label(deploy_form, text="Eval dataset name (or repo id)", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    quick_fix_eval_button = ttk.Button(deploy_form, text="Quick Fix eval_")
    quick_fix_eval_grid_kwargs = {"row": 0, "column": 2, "sticky": "w", "padx": (6, 0), "pady": 4}
    quick_fix_eval_button.grid(**quick_fix_eval_grid_kwargs)

    ttk.Label(deploy_form, text="Eval episodes", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_episodes_var, width=20).grid(row=1, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval episode time (seconds)", style="Field.TLabel").grid(
        row=2, column=0, sticky="w", padx=(0, 6), pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_duration_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval task description", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_task_var, width=52).grid(row=3, column=1, sticky="ew", pady=4)

    deploy_buttons = ttk.Frame(deploy_form, style="Panel.TFrame")
    deploy_buttons.grid(row=4, column=1, sticky="w", pady=(8, 0))
    preview_deploy_button = ttk.Button(deploy_buttons, text="Preview Command")
    preview_deploy_button.pack(side="left")
    run_deploy_button = ttk.Button(deploy_buttons, text="Run Deploy", style="Accent.TButton")
    run_deploy_button.pack(side="left", padx=(10, 0))

    deploy_advanced_fields = [
        ("robot.type", "Robot type"),
        ("robot.port", "Follower port"),
        ("robot.id", "Follower robot id"),
        ("robot.cameras", "Robot cameras JSON"),
        ("teleop.type", "Teleop type"),
        ("teleop.port", "Leader port"),
        ("teleop.id", "Leader robot id"),
        ("dataset.repo_id", "Eval dataset repo id"),
        ("dataset.num_episodes", "Eval episodes"),
        ("dataset.single_task", "Eval task"),
        ("dataset.episode_time_s", "Eval episode time (s)"),
        ("policy.path", "Policy path"),
    ]
    deploy_advanced_vars: dict[str, Any] = {
        key: tk.StringVar(value="")
        for key, _ in deploy_advanced_fields
    }

    ttk.Checkbutton(
        deploy_form,
        text="Advanced command options",
        variable=deploy_advanced_enabled_var,
    ).grid(row=5, column=1, sticky="w", pady=(6, 0))

    deploy_advanced_frame = ttk.LabelFrame(
        deploy_form,
        text="Advanced Deploy Options",
        style="Section.TLabelframe",
        padding=10,
    )
    deploy_advanced_frame.columnconfigure(1, weight=1)
    ttk.Label(
        deploy_advanced_frame,
        text="Fields are prefilled from the setup panel. Edit to override, or clear a field to use setup defaults.",
        style="Muted.TLabel",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    for idx, (key, label) in enumerate(deploy_advanced_fields, start=1):
        ttk.Label(deploy_advanced_frame, text=f"{label} (--{key})", style="Field.TLabel").grid(
            row=idx,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3,
        )
        ttk.Entry(deploy_advanced_frame, textvariable=deploy_advanced_vars[key], width=58).grid(
            row=idx,
            column=1,
            sticky="ew",
            pady=3,
        )

    deploy_custom_row = len(deploy_advanced_fields) + 1
    ttk.Label(deploy_advanced_frame, text="Custom args (raw)", style="Field.TLabel").grid(
        row=deploy_custom_row,
        column=0,
        sticky="w",
        padx=(0, 8),
        pady=(8, 3),
    )
    ttk.Entry(deploy_advanced_frame, textvariable=deploy_custom_args_var, width=58).grid(
        row=deploy_custom_row,
        column=1,
        sticky="ew",
        pady=(8, 3),
    )

    # ── Expandable model tree browser ─────────────────────────────────────────
    model_section = ttk.LabelFrame(deploy_container, text="Model Selection", style="Section.TLabelframe", padding=10)
    model_section.pack(fill="x", pady=(10, 0))
    model_section.columnconfigure(0, weight=1)

    ui_font = colors.get("font_ui", "TkDefaultFont")

    # Root dir row
    root_row = tk.Frame(model_section, bg=panel)
    root_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    root_row.columnconfigure(1, weight=1)
    tk.Label(root_row, text="Root:", bg=panel, fg=muted, font=(ui_font, 10)).grid(
        row=0, column=0, sticky="w", padx=(0, 6),
    )
    root_entry = tk.Entry(
        root_row,
        textvariable=deploy_root_var,
        bg=surface,
        fg=text_col,
        insertbackground=text_col,
        relief="flat",
        highlightthickness=1,
        highlightbackground=border,
        font=(mono_font, 10),
    )
    root_entry.grid(row=0, column=1, sticky="ew")
    ttk.Button(
        root_row, text="Browse Root",
        command=lambda: (choose_folder(deploy_root_var), refresh_local_models()),
    ).grid(row=0, column=2, sticky="w", padx=(6, 0))

    # Tree view style
    _ts = ttk.Style(root)
    _ts.configure(
        "Model.Treeview",
        font=(mono_font, 10),
        rowheight=26,
        background=surface,
        foreground=text_col,
        fieldbackground=surface,
        borderwidth=0,
        indent=18,
    )
    _ts.configure(
        "Model.Treeview.Heading",
        font=(ui_font, 10, "bold"),
        background=panel,
        foreground=accent,
        relief="flat",
    )
    _ts.map(
        "Model.Treeview",
        background=[("selected", accent)],
        foreground=[("selected", "#000000")],
    )

    tree_frame = tk.Frame(model_section, bg=panel)
    tree_frame.grid(row=1, column=0, sticky="nsew")
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    model_tree = ttk.Treeview(
        tree_frame,
        columns=("kind",),
        show="tree headings",
        height=10,
        style="Model.Treeview",
    )
    model_tree.heading("#0", text="Model / Checkpoint", anchor="w")
    model_tree.heading("kind", text="Type", anchor="w")
    model_tree.column("#0", stretch=True, minwidth=200)
    model_tree.column("kind", width=260, minwidth=220, stretch=True, anchor="w")

    # Green = model itself is directly runnable (has weights + config)
    # Yellow = folder contains a runnable payload deeper inside
    # Default = subfolder / unknown
    # Muted = empty / no model found
    model_tree.tag_configure("model_root", foreground=colors.get("success", "#22c55e"))
    model_tree.tag_configure("resolved", foreground=accent)
    model_tree.tag_configure("checkpoint", foreground=text_col)
    model_tree.tag_configure("folder", foreground=muted)
    model_tree.tag_configure("spacer", foreground=surface, background=surface)

    model_tree_scroll = ttk.Scrollbar(
        tree_frame,
        orient="vertical",
        command=model_tree.yview,
        style="Dark.Vertical.TScrollbar",
    )
    model_tree.configure(yscrollcommand=model_tree_scroll.set)

    model_tree.grid(row=0, column=0, sticky="nsew")
    model_tree_scroll.grid(row=0, column=1, sticky="ns")
    _bind_tree_wheel_scroll(model_tree)

    # Bottom row: Refresh + Browse Model + Pull + selected path display
    bottom_row = tk.Frame(model_section, bg=panel)
    bottom_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    bottom_row.columnconfigure(4, weight=1)

    refresh_models_button = ttk.Button(bottom_row, text="Refresh")
    refresh_models_button.grid(row=0, column=0, sticky="w")

    browse_model_button = ttk.Button(bottom_row, text="Browse Model...")
    browse_model_button.grid(row=0, column=1, sticky="w", padx=(6, 0))

    pull_model_button = ttk.Button(bottom_row, text="Pull New Model...")
    pull_model_button.grid(row=0, column=2, sticky="w", padx=(6, 0))

    selected_path_var = tk.StringVar(value="No model selected.")
    path_border = tk.Frame(bottom_row, bg=accent, width=3)
    path_border.grid(row=0, column=3, sticky="ns", padx=(12, 4))
    path_border.grid_propagate(False)
    tk.Label(
        bottom_row,
        textvariable=selected_path_var,
        bg=panel,
        fg=muted,
        font=(mono_font, 9),
        anchor="w",
        justify="left",
        wraplength=520,
    ).grid(row=0, column=4, sticky="ew")

    # ── Model info panel ─────────────────────────────────────────────────────
    model_info_var = tk.StringVar(value="No model selected.")
    model_info_panel = ttk.LabelFrame(deploy_container, text="Selected Model Info", style="Section.TLabelframe", padding=10)
    model_info_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(model_info_panel, textvariable=model_info_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    # ── Camera preview ───────────────────────────────────────────────────────
    deploy_camera_preview = DualCameraPreview(
        root=root,
        parent=deploy_container,
        title="Deploy Camera Preview",
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        append_log=log_panel.append_log,
        on_camera_indices_changed=on_camera_indices_changed,
    )

    auto_eval_hint = {"value": deploy_eval_dataset_var.get().strip()}

    def refresh_eval_quick_fix_button_visibility(*_: Any) -> None:
        needs_quick_fix = _needs_eval_prefix_quick_fix(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=deploy_eval_dataset_var.get().strip(),
        )
        is_visible = bool(quick_fix_eval_button.winfo_manager())
        if needs_quick_fix and not is_visible:
            quick_fix_eval_button.grid(**quick_fix_eval_grid_kwargs)
        elif not needs_quick_fix and is_visible:
            quick_fix_eval_button.grid_remove()

    deploy_eval_dataset_var.trace_add("write", refresh_eval_quick_fix_button_visibility)

    # ── Internal state ───────────────────────────────────────────────────────
    _state: dict[str, str] = {
        "model_folder": _last_model_folder,
        "checkpoint": _last_checkpoint,
    }
    # Maps treeview item IDs → absolute paths
    _tree_paths: dict[str, Path] = {}

    def _resolve_model_path() -> Path | None:
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        folder = _state["model_folder"]
        if not folder:
            return None
        p = root_path / folder
        ckpt = _state["checkpoint"]
        if ckpt:
            p = p / ckpt
        return p

    def _update_selected_path_display() -> None:
        p = _resolve_model_path()
        if p is None:
            selected_path_var.set("No model selected.")
            deploy_model_var.set(str(config["trained_models_dir"]))
            if deploy_advanced_enabled_var.get():
                deploy_advanced_vars["policy.path"].set(str(config["trained_models_dir"]))
            return

        resolved = _resolve_payload_path(p)

        if resolved != p:
            selected_path_var.set(f"Selected: {p}  |  Deploy payload: {resolved}")
        else:
            selected_path_var.set(f"Selected: {p}")
        deploy_model_var.set(str(resolved))
        if deploy_advanced_enabled_var.get():
            deploy_advanced_vars["policy.path"].set(str(resolved))

    def update_model_info(model_path: Path | None) -> None:
        if model_path is None or not model_path.exists() or not model_path.is_dir():
            model_info_var.set("No model selected.")
            return
        entries = sorted(model_path.iterdir())
        child_names = [p.name for p in entries[:8]]
        checkpoints = [p.name for p in entries if p.is_dir() and "checkpoint" in p.name.lower()]
        has_config = any((model_path / name).exists() for name in ("config.json", "model_config.json"))
        is_direct = is_runnable_model_path(model_path)
        candidates = find_nested_model_candidates(model_path) if not is_direct else []
        deploy_payload = _resolve_payload_path(model_path)
        info_lines = [
            f"Selected path: {model_path}",
            f"Deploy payload: {deploy_payload}",
            f"Directly runnable: {'yes' if is_direct else 'no'}  |  Config file: {'yes' if has_config else 'no'}  |  Items: {len(entries)}",
        ]
        if not is_direct and candidates:
            info_lines.append(f"Resolved payload: {candidates[0]}")
        if checkpoints:
            info_lines.append(f"Checkpoint-like folders: {', '.join(checkpoints[:6])}")
        info_lines.append(f"Contents: {', '.join(child_names) if child_names else '(empty)'}")
        model_info_var.set("\n".join(info_lines))

    def _save_selection_to_config() -> None:
        config["last_model_name"] = _state["model_folder"]
        config["last_checkpoint_name"] = _state["checkpoint"]
        save_config(config, quiet=True)

    def _populate_tree() -> None:
        for iid in list(model_tree.get_children()):
            model_tree.delete(iid)
        _tree_paths.clear()

        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        if not root_path.exists():
            return

        try:
            top_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())
        except OSError:
            return

        def add_subtree(parent_iid: str, node_path: Path, depth: int) -> None:
            if depth >= _MODEL_TREE_MAX_DEPTH:
                return
            try:
                subdirs = sorted(p for p in node_path.iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                return
            for subdir in subdirs:
                sub_kind, sub_tag = _model_tree_node_kind(subdir)
                sub_iid = model_tree.insert(
                    parent_iid,
                    "end",
                    text=subdir.name,
                    values=(sub_kind,),
                    tags=(sub_tag,),
                    open=False,
                )
                _tree_paths[sub_iid] = subdir
                add_subtree(sub_iid, subdir, depth + 1)

        for model_dir in top_dirs:
            kind_label, tag = _model_tree_node_kind(model_dir)
            iid = model_tree.insert(
                "", "end",
                text=model_dir.name,
                values=(kind_label,),
                tags=(tag,),
                open=False,
            )
            _tree_paths[iid] = model_dir
            add_subtree(iid, model_dir, 1)

        for _ in range(_MODEL_TREE_BOTTOM_SPACER_ROWS):
            model_tree.insert("", "end", text=" ", values=(" ",), tags=("spacer",))

    def on_tree_select(_: Any) -> None:
        selected = model_tree.selection()
        if not selected:
            return
        iid = selected[0]
        path = _tree_paths.get(iid)
        if path is None:
            return

        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        try:
            rel = path.relative_to(root_path)
            parts = rel.parts
            _state["model_folder"] = parts[0] if parts else path.name
            _state["checkpoint"] = str(Path(*parts[1:])) if len(parts) > 1 else ""
        except ValueError:
            _state["model_folder"] = path.name
            _state["checkpoint"] = ""

        folder_name = _state["model_folder"]
        current_eval_name = deploy_eval_dataset_var.get().strip()
        if not current_eval_name or current_eval_name == auto_eval_hint["value"]:
            suggested = suggest_eval_dataset_name(config, folder_name)
            deploy_eval_dataset_var.set(suggested)
            auto_eval_hint["value"] = suggested

        _update_selected_path_display()
        update_model_info(path)
        _save_selection_to_config()

    def _apply_browsed_path(selected_path: Path) -> None:
        """Update state and display after the user browses to a model folder."""
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        try:
            rel = selected_path.relative_to(root_path)
            parts = rel.parts
            _state["model_folder"] = parts[0] if parts else selected_path.name
            _state["checkpoint"] = str(Path(*parts[1:])) if len(parts) > 1 else ""
        except ValueError:
            _state["model_folder"] = selected_path.name
            _state["checkpoint"] = ""

        resolved = _resolve_payload_path(selected_path)

        if not is_runnable_model_path(selected_path) and not find_nested_model_candidates(selected_path):
            log_panel.append_log(f"Warning: {selected_path.name} does not appear to contain a runnable model.")

        if resolved != selected_path:
            selected_path_var.set(f"Selected: {selected_path}  |  Deploy payload: {resolved}")
        else:
            selected_path_var.set(f"Selected: {selected_path}")
        deploy_model_var.set(str(resolved))
        if deploy_advanced_enabled_var.get():
            deploy_advanced_vars["policy.path"].set(str(resolved))

        update_model_info(selected_path)
        _save_selection_to_config()

        # Highlight the matching node in the tree if visible
        for iid, path in _tree_paths.items():
            if path == selected_path:
                parent = model_tree.parent(iid)
                if parent:
                    model_tree.item(parent, open=True)
                model_tree.selection_set(iid)
                model_tree.see(iid)
                break

        log_panel.append_log(f"Model selected: {resolved}")

    def select_model_path(selected_path: Path) -> bool:
        candidate = Path(selected_path)
        if not candidate.exists() or not candidate.is_dir():
            return False
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        if root_path not in candidate.parents and candidate != root_path:
            deploy_root_var.set(str(candidate.parent))
            config["trained_models_dir"] = str(candidate.parent)
            save_config(config, quiet=True)
            _populate_tree()
        _apply_browsed_path(candidate)
        return True

    def browse_for_model() -> None:
        from tkinter import filedialog as _fd

        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        start = str(root_path) if root_path.exists() else str(Path.home())
        raw = ask_directory_dialog(
            root=root,
            filedialog=_fd,
            initial_dir=start,
            title="Select Model or Checkpoint Folder",
        )
        if not raw:
            return
        _apply_browsed_path(Path(raw))

    model_tree.bind("<<TreeviewSelect>>", on_tree_select)

    def refresh_local_models() -> None:
        _populate_tree()
        _restore_selection_from_config()

    def _restore_selection_from_config() -> None:
        saved_folder = str(config.get("last_model_name", "")).strip()
        saved_ckpt = str(config.get("last_checkpoint_name", "")).strip()
        if not saved_folder:
            return

        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        target_path = root_path / saved_folder
        if saved_ckpt:
            target_path = target_path / saved_ckpt

        # Try to find and select the exact saved path in the tree
        for iid, path in _tree_paths.items():
            if path == target_path:
                parent = model_tree.parent(iid)
                if parent:
                    model_tree.item(parent, open=True)
                model_tree.selection_set(iid)
                model_tree.see(iid)
                _state["model_folder"] = saved_folder
                _state["checkpoint"] = saved_ckpt
                _update_selected_path_display()
                update_model_info(target_path)
                return

        # Fallback: select the top-level model folder
        for iid, path in _tree_paths.items():
            if path == root_path / saved_folder and model_tree.parent(iid) == "":
                model_tree.selection_set(iid)
                model_tree.see(iid)
                _state["model_folder"] = saved_folder
                _state["checkpoint"] = ""
                _update_selected_path_display()
                update_model_info(path)
                return

    refresh_models_button.configure(command=refresh_local_models)
    browse_model_button.configure(command=browse_for_model)

    pull_popup_state: dict[str, Any] = {"window": None}

    def _cleanup_batch_file(batch_file: Path | None) -> None:
        if batch_file is None:
            return
        try:
            batch_file.unlink()
        except OSError:
            pass

    def open_pull_model_popup() -> None:
        popup = pull_popup_state.get("window")
        if popup is not None and bool(popup.winfo_exists()):
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            return

        from tkinter import filedialog as _fd

        profiles, active_profile_id = load_training_profiles(config)
        active_profile = next((item for item in profiles if item.id == active_profile_id), None)
        if active_profile is None and profiles:
            active_profile = profiles[0]

        default_host = str(config.get("deploy_pull_host", "")).strip() or (active_profile.host if active_profile else "")
        default_username = str(config.get("deploy_pull_username", "")).strip() or (
            active_profile.username if active_profile else ""
        )
        default_port_raw = str(config.get("deploy_pull_port", "")).strip()
        if default_port_raw:
            default_port = default_port_raw
        elif active_profile is not None:
            default_port = str(active_profile.port)
        else:
            default_port = "22"

        default_remote_path = str(config.get("deploy_pull_remote_path", "")).strip() or (
            active_profile.remote_models_root if active_profile else "~/lerobot/trained_models"
        )
        computed_destination = _default_pull_destination(str(config["trained_models_dir"]), default_remote_path)
        default_local_destination = str(config.get("deploy_pull_local_destination", "")).strip() or str(computed_destination)

        popup = tk.Toplevel(root)
        popup.title("Pull New Model")
        popup.geometry("920x420")
        popup.minsize(820, 360)
        popup.configure(bg=panel)
        popup.transient(root)
        pull_popup_state["window"] = popup

        def _on_close() -> None:
            pull_popup_state["window"] = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)

        container = ttk.Frame(popup, style="Panel.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        host_var = tk.StringVar(value=default_host)
        port_var = tk.StringVar(value=default_port)
        username_var = tk.StringVar(value=default_username)
        remote_path_var = tk.StringVar(value=default_remote_path)
        local_destination_var = tk.StringVar(value=default_local_destination)
        prefer_rsync_var = tk.BooleanVar(value=bool(config.get("deploy_pull_prefer_rsync", True)))
        attach_terminal_var = tk.BooleanVar(value=bool(config.get("deploy_pull_attach_terminal", True)))
        auto_local_var = tk.BooleanVar(value=bool(config.get("deploy_pull_auto_local", True)))
        auto_rename_var = tk.BooleanVar(value=bool(config.get("deploy_pull_auto_rename", True)))
        status_var = tk.StringVar(value="Pull a remote model folder into local trained_models.")

        ttk.Label(container, text="Host", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(container, textvariable=host_var, width=28).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(container, text="Port", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        ttk.Entry(container, textvariable=port_var, width=10).grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Username", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(container, textvariable=username_var, width=28).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(container, text="Remote model path", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=remote_path_var, width=66).grid(
            row=2, column=1, columnspan=3, sticky="ew", pady=4,
        )

        ttk.Label(container, text="Local destination", style="Field.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=local_destination_var, width=66).grid(
            row=3, column=1, columnspan=2, sticky="ew", pady=4,
        )

        browse_destination_button = ttk.Button(container, text="Browse Parent...")
        browse_destination_button.grid(row=3, column=3, sticky="w", pady=4)

        controls_row = ttk.Frame(container, style="Panel.TFrame")
        controls_row.grid(row=4, column=1, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            controls_row,
            text="Prefer rsync (auto fallback to sftp on failure)",
            variable=prefer_rsync_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Attach terminal input (recommended for SSH password prompt)",
            variable=attach_terminal_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Auto-fill local destination from remote path",
            variable=auto_local_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_row,
            text="Auto-rename local destination when folder already exists",
            variable=auto_rename_var,
        ).pack(anchor="w")

        buttons_row = ttk.Frame(container, style="Panel.TFrame")
        buttons_row.grid(row=5, column=1, columnspan=3, sticky="w", pady=(10, 0))
        preview_pull_button = ttk.Button(buttons_row, text="Preview Pull Command")
        preview_pull_button.pack(side="left")
        run_pull_button = ttk.Button(buttons_row, text="Pull Model", style="Accent.TButton")
        run_pull_button.pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=status_var, style="Muted.TLabel", justify="left").grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        def _save_pull_settings() -> None:
            config["deploy_pull_host"] = host_var.get().strip()
            config["deploy_pull_port"] = port_var.get().strip()
            config["deploy_pull_username"] = username_var.get().strip()
            config["deploy_pull_remote_path"] = remote_path_var.get().strip()
            config["deploy_pull_local_destination"] = local_destination_var.get().strip()
            config["deploy_pull_prefer_rsync"] = bool(prefer_rsync_var.get())
            config["deploy_pull_attach_terminal"] = bool(attach_terminal_var.get())
            config["deploy_pull_auto_local"] = bool(auto_local_var.get())
            config["deploy_pull_auto_rename"] = bool(auto_rename_var.get())
            save_config(config, quiet=True)

        def _apply_auto_destination() -> None:
            if not auto_local_var.get():
                return
            local_destination_var.set(
                str(_default_pull_destination(str(config["trained_models_dir"]), remote_path_var.get().strip()))
            )

        def _choose_local_destination_parent() -> None:
            current_local_raw = local_destination_var.get().strip()
            current_local = Path(normalize_path(current_local_raw)) if current_local_raw else Path(config["trained_models_dir"])
            selected = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=str(current_local.parent if current_local.parent.exists() else Path.home()),
                title="Select local destination parent folder",
            )
            if not selected:
                return
            name = current_local.name if current_local.name else _remote_path_name(remote_path_var.get())
            local_destination_var.set(str(Path(selected) / name))
            auto_local_var.set(False)

        browse_destination_button.configure(command=_choose_local_destination_parent)
        remote_path_var.trace_add("write", lambda *_: _apply_auto_destination())
        auto_local_var.trace_add("write", lambda *_: _apply_auto_destination())

        def _build_pull_request() -> tuple[dict[str, Any] | None, str | None]:
            host = host_var.get().strip()
            username = username_var.get().strip()
            if not host:
                return None, "Host is required."
            if not username:
                return None, "Username is required."

            try:
                port = int(port_var.get().strip() or "22")
            except ValueError:
                return None, "Port must be an integer."
            if port <= 0 or port > 65535:
                return None, "Port must be between 1 and 65535."

            remote_path = remote_path_var.get().strip()
            if not remote_path:
                return None, "Remote model path is required."

            local_raw = local_destination_var.get().strip()
            if not local_raw:
                return None, "Local destination is required."

            local_destination = Path(normalize_path(local_raw))
            renamed = False
            if local_destination.exists():
                if auto_rename_var.get():
                    local_destination = _increment_pull_destination(local_destination)
                    local_destination_var.set(str(local_destination))
                    renamed = True
                else:
                    return None, f"Local destination already exists: {local_destination}"

            try:
                local_destination.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return None, f"Unable to create destination parent folder: {exc}"

            profile = TrainingProfile(
                id="deploy_pull",
                name="Deploy Pull",
                host=host,
                port=port,
                username=username,
                auth_mode="ssh_key",
                identity_file="",
                remote_models_root=remote_path,
                remote_project_root="~",
                env_activate_cmd="",
                default_tmux_session="",
                default_srun_prefix="",
            )

            ssh_path = shutil.which("ssh")
            sftp_path = shutil.which("sftp")
            rsync_path = shutil.which("rsync")
            if not ssh_path:
                return None, "ssh binary not found in PATH."
            if not sftp_path and not rsync_path:
                return None, "Neither sftp nor rsync is available in PATH."

            prefer_rsync = bool(prefer_rsync_var.get())
            if prefer_rsync and rsync_path:
                primary_cmd = build_pull_command(profile, remote_path, local_destination, prefer_rsync=True)
                primary_mode = "rsync" if command_uses_binary(primary_cmd, "rsync") else "sftp"
                primary_batch = None
                if primary_mode == "sftp":
                    primary_cmd, primary_batch = build_sftp_pull_command(profile, remote_path, local_destination)
            else:
                primary_cmd, primary_batch = build_sftp_pull_command(profile, remote_path, local_destination)
                primary_mode = "sftp"

            checks: list[tuple[str, str, str]] = [
                ("PASS", "SSH target", f"{username}@{host}:{port}"),
                ("PASS", "Remote model path", remote_path),
                ("PASS", "Local destination", str(local_destination)),
                ("PASS" if ssh_path else "FAIL", "ssh binary", ssh_path or "not found"),
                ("PASS" if sftp_path else "FAIL", "sftp binary", sftp_path or "not found"),
                (
                    "PASS" if rsync_path else "WARN",
                    "rsync binary",
                    rsync_path or "not found (sftp only)",
                ),
                (
                    "PASS" if attach_terminal_var.get() else "WARN",
                    "Interactive terminal attach",
                    "enabled" if attach_terminal_var.get() else "disabled (password prompts may fail)",
                ),
            ]
            if primary_mode == "rsync" and sftp_path:
                checks.append(("PASS", "Fallback", "sftp fallback available if rsync fails"))
            elif primary_mode == "rsync":
                checks.append(("WARN", "Fallback", "sftp fallback unavailable"))
            if renamed:
                checks.append(("WARN", "Local destination rename", f"destination exists; using {local_destination.name}"))

            return {
                "profile": profile,
                "remote_path": remote_path,
                "local_destination": local_destination,
                "primary_cmd": primary_cmd,
                "primary_mode": primary_mode,
                "primary_batch": primary_batch,
                "allow_fallback": bool(primary_mode == "rsync" and sftp_path),
                "checks": checks,
            }, None

        def _start_pull_command(
            *,
            request: dict[str, Any],
            command: list[str],
            mode: str,
            batch_file: Path | None,
            allow_fallback: bool,
        ) -> None:
            profile = request["profile"]
            remote_path = request["remote_path"]
            local_destination = request["local_destination"]
            run_mode = "train_attach" if attach_terminal_var.get() else "run"

            if run_mode == "train_attach":
                log_panel.append_log(
                    "Model pull terminal is interactive. Enter SSH password in the terminal input if prompted."
                )

            def on_start_error(_exc: Exception) -> None:
                _cleanup_batch_file(batch_file)

            def on_complete(return_code: int, was_canceled: bool) -> None:
                _cleanup_batch_file(batch_file)
                if was_canceled:
                    set_running(False, "Model pull canceled.")
                    messagebox.showinfo("Canceled", "Model pull was canceled.")
                    return
                if return_code == 0:
                    _save_pull_settings()
                    refresh_local_models()
                    selected = select_model_path(local_destination)
                    set_running(False, f"Model pull completed ({mode}).")
                    if selected:
                        messagebox.showinfo("Model Pull Complete", f"Model pulled and selected:\n{local_destination}")
                    else:
                        messagebox.showinfo("Model Pull Complete", f"Model pulled to:\n{local_destination}")
                    return

                if mode == "rsync" and allow_fallback:
                    retry = messagebox.askyesno(
                        "rsync failed",
                        "rsync model pull failed. Retry once with sftp fallback?",
                    )
                    if retry:
                        fallback_cmd, fallback_batch = build_sftp_pull_command(profile, remote_path, local_destination)
                        set_running(False, "Retrying model pull with sftp...")
                        _start_pull_command(
                            request=request,
                            command=fallback_cmd,
                            mode="sftp",
                            batch_file=fallback_batch,
                            allow_fallback=False,
                        )
                        return

                set_running(False, "Model pull failed.", True)
                messagebox.showerror("Model Pull Failed", f"Model pull failed with exit code {return_code}.")

            run_process_async(
                command,
                None,
                on_complete,
                None,
                None,
                run_mode,
                request["checks"],
                {
                    "training_profile": "Deploy Pull",
                    "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
                    "remote_path": remote_path,
                    "local_path": str(local_destination),
                    "template_name": "deploy_pull_model",
                    "training_transport": mode,
                },
                on_start_error,
            )

        def preview_pull_command() -> None:
            request, error_text = _build_pull_request()
            if error_text or request is None:
                messagebox.showerror("Pull Model", error_text or "Unable to build pull command.")
                return
            try:
                last_command_state["value"] = format_command(request["primary_cmd"])
                details = [
                    "Primary pull command:",
                    format_command_for_dialog(request["primary_cmd"]),
                ]
                if request["allow_fallback"]:
                    details.append("\nFallback: if rsync fails, the app retries once with sftp.")
                show_text_dialog(
                    root=root,
                    title="Model Pull Command",
                    text="\n".join(details),
                    copy_text=last_command_state["value"],
                    wrap_mode="word",
                )
            finally:
                _cleanup_batch_file(request["primary_batch"])

        def run_pull_command() -> None:
            request, error_text = _build_pull_request()
            if error_text or request is None:
                messagebox.showerror("Pull Model", error_text or "Unable to build pull command.")
                return

            if not confirm_preflight_in_gui("Model Pull Preflight", request["checks"]):
                _cleanup_batch_file(request["primary_batch"])
                return

            if not ask_text_dialog(
                root=root,
                title="Confirm Model Pull",
                text=(
                    "Review the model pull command below.\n"
                    "Click Confirm to run it, or Cancel to stop.\n\n"
                    + format_command_for_dialog(request["primary_cmd"])
                ),
                copy_text=format_command(request["primary_cmd"]),
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            ):
                _cleanup_batch_file(request["primary_batch"])
                return

            trusted, trust_detail = ensure_host_trusted(request["profile"], messagebox)
            if not trusted:
                _cleanup_batch_file(request["primary_batch"])
                messagebox.showerror("SSH Trust", trust_detail)
                return
            log_panel.append_log(f"SSH trust check: {trust_detail}")

            _save_pull_settings()
            _start_pull_command(
                request=request,
                command=request["primary_cmd"],
                mode=request["primary_mode"],
                batch_file=request["primary_batch"],
                allow_fallback=bool(request["allow_fallback"]),
            )

        preview_pull_button.configure(command=preview_pull_command)
        run_pull_button.configure(command=run_pull_command)
        _apply_auto_destination()

    pull_model_button.configure(command=open_pull_model_popup)

    # ── Deploy logic ─────────────────────────────────────────────────────────
    def _seed_deploy_advanced_from_current() -> None:
        req, cmd, _, error_text = build_deploy_request_and_command(
            config=config,
            deploy_root_raw=deploy_root_var.get(),
            deploy_model_raw=deploy_model_var.get(),
            eval_dataset_raw=deploy_eval_dataset_var.get(),
            eval_episodes_raw=deploy_eval_episodes_var.get(),
            eval_duration_raw=deploy_eval_duration_var.get(),
            eval_task_raw=deploy_eval_task_var.get(),
        )
        if error_text or req is None or cmd is None:
            return
        for key, _ in deploy_advanced_fields:
            value = get_flag_value(cmd, key)
            if value is not None:
                deploy_advanced_vars[key].set(value)

    def _refresh_deploy_advanced_visibility(*_: Any) -> None:
        if deploy_advanced_enabled_var.get():
            _seed_deploy_advanced_from_current()
            deploy_advanced_frame.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            deploy_advanced_frame.grid_remove()

    def _deploy_advanced_overrides_from_ui() -> tuple[dict[str, str] | None, str]:
        if not deploy_advanced_enabled_var.get():
            return None, ""
        overrides: dict[str, str] = {}
        for key, _ in deploy_advanced_fields:
            value = str(deploy_advanced_vars[key].get()).strip()
            if value:
                overrides[key] = value
        return overrides or None, str(deploy_custom_args_var.get())

    def build_current_deploy() -> tuple[Any, Any, Any, Any]:
        arg_overrides, custom_args_raw = _deploy_advanced_overrides_from_ui()
        return build_deploy_request_and_command(
            config=config,
            deploy_root_raw=deploy_root_var.get(),
            deploy_model_raw=deploy_model_var.get(),
            eval_dataset_raw=deploy_eval_dataset_var.get(),
            eval_episodes_raw=deploy_eval_episodes_var.get(),
            eval_duration_raw=deploy_eval_duration_var.get(),
            eval_task_raw=deploy_eval_task_var.get(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def apply_eval_prefix_quick_fix() -> bool:
        current_value = deploy_eval_dataset_var.get().strip()
        suggested_repo_id, changed = suggest_eval_prefixed_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=current_value,
        )
        if not changed:
            log_panel.append_log(f"Eval dataset already follows convention: {suggested_repo_id}")
            return False
        deploy_eval_dataset_var.set(suggested_repo_id)
        if deploy_advanced_enabled_var.get():
            deploy_advanced_vars["dataset.repo_id"].set(suggested_repo_id)
        log_panel.append_log(f"Applied eval dataset quick fix: {suggested_repo_id}")
        return True

    def preview_deploy() -> None:
        req, cmd, _, error_text = build_current_deploy()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
            return
        last_command_state["value"] = format_command(cmd)
        command_for_dialog = format_command_for_dialog(cmd)
        log_panel.append_log("Preview deploy command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Deploy Command",
            text=command_for_dialog,
            copy_text=last_command_state["value"],
            wrap_mode="word",
        )

    def run_deploy_from_gui() -> None:
        req, cmd, updated_config, error_text = build_current_deploy()
        if error_text or req is None or cmd is None or updated_config is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
            return

        current_eval_input = deploy_eval_dataset_var.get().strip() or req.eval_repo_id
        suggested_eval_repo_id, requires_quick_fix = suggest_eval_prefixed_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=current_eval_input,
        )
        if requires_quick_fix:
            proceed = ask_text_dialog(
                root=root,
                title="Eval Dataset Prefix Required",
                text=(
                    "Deploy eval dataset names must start with 'eval_'.\n\n"
                    f"Current: {current_eval_input}\n"
                    f"Suggested: {suggested_eval_repo_id}\n\n"
                    "Click Apply Quick Fix to continue, or Cancel Deploy to stop."
                ),
                confirm_label="Apply Quick Fix",
                cancel_label="Cancel Deploy",
                wrap_mode="char",
            )
            if not proceed:
                return
            deploy_eval_dataset_var.set(suggested_eval_repo_id)
            if deploy_advanced_enabled_var.get():
                deploy_advanced_vars["dataset.repo_id"].set(suggested_eval_repo_id)
            log_panel.append_log(f"Applied eval dataset quick fix: {suggested_eval_repo_id}")
            req, cmd, updated_config, error_text = build_current_deploy()
            if error_text or req is None or cmd is None or updated_config is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
                return

        lerobot_dir = get_lerobot_dir(config)
        deploy_data_dir = get_deploy_data_dir(config)
        resolved_repo_id, adjusted, _ = resolve_unique_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=req.eval_repo_id,
            local_roots=[deploy_data_dir, lerobot_dir / "data"],
        )
        if adjusted:
            deploy_eval_dataset_var.set(resolved_repo_id)
            if deploy_advanced_enabled_var.get():
                deploy_advanced_vars["dataset.repo_id"].set(resolved_repo_id)
            log_panel.append_log(f"Auto-iterated eval dataset to avoid existing target: {resolved_repo_id}")
            req, cmd, updated_config, error_text = build_current_deploy()
            if error_text or req is None or cmd is None or updated_config is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
                return

        exists = dataset_exists_on_hf(req.eval_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{req.eval_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not ask_text_dialog(
            root=root,
            title="Confirm Deploy",
            text=(
                "Review the deploy command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            copy_text=format_command(cmd),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        command_changed_after_confirm = False
        while True:
            preflight_checks = run_preflight_for_deploy(
                config=config,
                model_path=req.model_path,
                eval_repo_id=req.eval_repo_id,
            )

            model_candidate = _first_model_payload_candidate(preflight_checks)
            current_eval_input = deploy_eval_dataset_var.get().strip() or req.eval_repo_id
            suggested_repo, missing_eval_prefix = suggest_eval_prefixed_repo_id(
                username=str(config["hf_username"]),
                dataset_name_or_repo_id=current_eval_input,
            )

            quick_actions: list[tuple[str, str]] = []
            if missing_eval_prefix:
                quick_actions.append(("fix_eval_prefix", "Apply eval_ Prefix"))
            if model_candidate and Path(model_candidate) != req.model_path:
                quick_actions.append(("fix_model_payload", "Use Suggested Model Payload"))

            if not quick_actions:
                if not confirm_preflight_in_gui("Deploy Preflight", preflight_checks):
                    return
                break

            action = ask_text_dialog_with_actions(
                root=root,
                title="Deploy Preflight Fix Center",
                text=summarize_checks(preflight_checks, title="Deploy Preflight"),
                actions=quick_actions,
                confirm_label="Confirm",
                cancel_label="Cancel",
                wrap_mode="char",
            )
            if action == "cancel":
                return
            if action == "confirm":
                break

            if action == "fix_eval_prefix":
                deploy_eval_dataset_var.set(suggested_repo)
                if deploy_advanced_enabled_var.get():
                    deploy_advanced_vars["dataset.repo_id"].set(suggested_repo)
                log_panel.append_log(f"Applied preflight quick fix: eval dataset -> {suggested_repo}")
                command_changed_after_confirm = True
            elif action == "fix_model_payload" and model_candidate:
                deploy_model_var.set(str(model_candidate))
                if deploy_advanced_enabled_var.get():
                    deploy_advanced_vars["policy.path"].set(str(model_candidate))
                log_panel.append_log(f"Applied preflight quick fix: model payload -> {model_candidate}")
                command_changed_after_confirm = True

            req, cmd, updated_config, error_text = build_current_deploy()
            if error_text or req is None or cmd is None or updated_config is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
                return

        if command_changed_after_confirm and not ask_text_dialog(
            root=root,
            title="Confirm Updated Deploy Command",
            text=(
                "Preflight quick fixes updated the command.\n"
                "Click Confirm to run the updated command, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            copy_text=format_command(cmd),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        config.update(updated_config)
        save_config(config)
        deploy_eval_dataset_var.set(suggest_eval_dataset_name(config, req.model_path.name))
        if deploy_advanced_enabled_var.get():
            deploy_advanced_vars["dataset.repo_id"].set(deploy_eval_dataset_var.get().strip())
        refresh_header_subtitle()

        def after_deploy(return_code: int, was_canceled: bool) -> None:
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Deploy canceled.")
                    messagebox.showinfo("Canceled", "Deploy command was canceled.")
                else:
                    set_running(False, "Deploy failed.", True)
                    messagebox.showerror("Deploy Failed", f"Deploy failed with exit code {return_code}.")
            else:
                set_running(False, "Deploy completed.")
                messagebox.showinfo(
                    "Done",
                    (
                        f"Deployment completed.\nModel: {req.model_path}\nEval dataset: {req.eval_repo_id}\n\n"
                        "Open History and use the Deploy Outcome + Notes Editor to finalize episode edits and overall notes."
                    ),
                )

        run_process_async(
            cmd,
            lerobot_dir,
            after_deploy,
            req.eval_num_episodes,
            req.eval_num_episodes * req.eval_duration_s,
            "deploy",
            preflight_checks,
            {"dataset_repo_id": req.eval_repo_id, "model_path": str(req.model_path)},
        )

    preview_deploy_button.configure(command=preview_deploy)
    run_deploy_button.configure(command=run_deploy_from_gui)
    quick_fix_eval_button.configure(command=apply_eval_prefix_quick_fix)
    refresh_eval_quick_fix_button_visibility()
    _refresh_deploy_advanced_visibility()
    deploy_advanced_enabled_var.trace_add("write", _refresh_deploy_advanced_visibility)

    refresh_local_models()
    update_model_info(_resolve_model_path())

    return DeployTabHandles(
        deploy_root_var=deploy_root_var,
        deploy_eval_episodes_var=deploy_eval_episodes_var,
        deploy_eval_duration_var=deploy_eval_duration_var,
        deploy_eval_task_var=deploy_eval_task_var,
        deploy_camera_preview=deploy_camera_preview,
        refresh_local_models=refresh_local_models,
        select_model_path=select_model_path,
        action_buttons=[preview_deploy_button, run_deploy_button, quick_fix_eval_button, refresh_models_button, pull_model_button],
    )
