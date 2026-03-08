from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Callable

from .command_overrides import get_flag_value, get_policy_path_value
from .checks import run_preflight_for_deploy, summarize_checks
from .commands import build_lerobot_calibrate_command
from .diagnostics import checks_to_events
from .deploy_diagnostics import find_nested_model_candidates, is_runnable_model_path
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_dialogs import (
    ask_editable_command_dialog,
    ask_text_dialog,
    ask_text_dialog_with_actions,
    format_command_for_dialog,
    show_text_dialog,
)
from .gui_file_dialogs import ask_directory_dialog, ask_openfilename_dialog
from .gui_forms import build_deploy_request_and_command
from .gui_input_help import keyboard_input_help_text, keyboard_input_help_title
from .gui_log import GuiLogPanel
from .gui_scroll import bind_yview_wheel_scroll
from .gui_theme import configure_treeview_style
from .gui_window import fit_window_to_screen
from .repo_utils import (
    compose_repo_id,
    dataset_exists_on_hf,
    model_exists_on_hf,
    normalize_repo_id,
    repo_name_only,
    resolve_unique_repo_id,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .runner import format_command
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .types import DiagnosticEvent, GuiRunProcessAsync

_MODEL_TREE_MAX_DEPTH = 4
_MODEL_TREE_BOTTOM_SPACER_ROWS = 2


def _compose_repo_id(owner: str, dataset_name_or_repo_id: str) -> str | None:
    """Backward-compatible alias used by helper tests and older call sites."""
    return compose_repo_id(owner, dataset_name_or_repo_id)


def _first_model_payload_candidate(checks: list[tuple[str, str, str]]) -> str | None:
    for _, name, detail in checks:
        if name.strip().lower() != "model payload candidates":
            continue
        candidate = detail.split(",", 1)[0].strip()
        return candidate or None
    return None


def _build_calibration_command(config: dict[str, Any]) -> str:
    """Return the LeRobot calibration CLI command for the current follower robot."""
    return format_command(build_lerobot_calibrate_command(config, role="follower"))


def _camera_rename_map_suggestion(checks: list[tuple[str, str, str]]) -> str | None:
    for _, name, detail in checks:
        if name.strip().lower() == "camera rename map suggestion":
            value = detail.strip()
            return value or None
    return None


def _quick_actions_from_diagnostics(events: list[DiagnosticEvent]) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    action_labels = {
        "fix_eval_prefix": "Apply eval_ Prefix",
        "fix_model_payload": "Use Suggested Model Payload",
        "apply_rename_map": "Apply Camera Rename Map",
        "browse_follower_calib": "Browse Follower Calibration",
        "browse_leader_calib": "Browse Leader Calibration",
        "show_calib_cmd": "Show Recalibration Command",
    }
    actions: list[tuple[str, str]] = []
    context_by_action: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    def add(action_id: str, label: str, context: dict[str, Any] | None = None) -> None:
        if action_id in seen:
            return
        seen.add(action_id)
        actions.append((action_id, label))
        if context:
            context_by_action[action_id] = dict(context)

    for event in events:
        action_id = str(event.quick_action_id or "").strip()
        if not action_id:
            continue
        context = dict(event.context or {})
        if action_id == "fix_camera_fps":
            suggested_fps = context.get("suggested_fps")
            if isinstance(suggested_fps, int) and suggested_fps > 0:
                scoped_action = f"fix_camera_fps:{suggested_fps}"
                add(scoped_action, f"Set camera_fps -> {suggested_fps} Hz (match training)", context)
            continue
        label = action_labels.get(action_id)
        if label:
            add(action_id, label, context)

    return actions, context_by_action


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



def _model_hf_parity_detail(exists: bool | None, repo_id: str) -> tuple[str, str]:
    if exists is True:
        return "WARN", f"Remote model already exists: {repo_id}"
    if exists is False:
        return "PASS", f"Remote model not found yet: {repo_id}"
    return "WARN", f"Unable to confirm if remote model exists: {repo_id}"


@dataclass
class DeployTabHandles:
    deploy_root_var: Any
    deploy_eval_episodes_var: Any
    deploy_eval_duration_var: Any
    deploy_eval_task_var: Any
    camera_slot: Any
    refresh_local_models: Callable[[], None]
    select_model_path: Callable[[Path], bool]
    apply_theme: Callable[[dict[str, str]], None]
    action_buttons: list[Any]


def setup_deploy_tab(
    *,
    root: Any,
    deploy_tab: Any,
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
    _initial_eval_dataset = (
        str(config.get("last_eval_dataset_name", "")).strip()
        or suggest_eval_dataset_name(config, _last_model_folder)
    )
    deploy_eval_dataset_var = tk.StringVar(
        value=normalize_repo_id(str(config.get("hf_username", "")), _initial_eval_dataset)
    )
    deploy_eval_episodes_var = tk.StringVar(value=str(config.get("eval_num_episodes", 10)))
    deploy_eval_duration_var = tk.StringVar(value=str(config.get("eval_duration_s", 20)))
    deploy_target_hz_var = tk.StringVar(value=str(config.get("deploy_target_hz", "")).strip())
    deploy_eval_task_var = tk.StringVar(value=str(config.get("eval_task", DEFAULT_TASK)))
    deploy_advanced_enabled_var = tk.BooleanVar(value=False)
    deploy_custom_args_var = tk.StringVar(value="")
    deploy_follower_calib_var = tk.StringVar(
        value=str(config.get("follower_calibration_path", "")).strip()
    )
    deploy_leader_calib_var = tk.StringVar(
        value=str(config.get("leader_calibration_path", "")).strip()
    )

    def _browse_calibration_file(target_var: Any) -> None:
        """Open a JSON file picker for calibration files."""
        from tkinter import filedialog as _fd

        current_dir = None
        current_val = str(target_var.get()).strip()
        if current_val:
            try:
                current_dir = str(Path(current_val).parent)
            except Exception:
                pass
        if not current_dir:
            current_dir = str(config.get("lerobot_dir", str(Path.home())))
        selected = ask_openfilename_dialog(
            root=root,
            filedialog=_fd,
            initial_dir=current_dir,
            title="Select Robot Calibration File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            target_var.set(selected)

    # ── Deploy form ───────────────────────────────────────────────────────────
    deploy_form = ttk.LabelFrame(deploy_container, text="Deploy / Eval Setup", style="Section.TLabelframe", padding=12)
    deploy_form.pack(fill="x")
    deploy_form.columnconfigure(1, weight=1)

    ttk.Label(deploy_form, text="Eval dataset repo id (owner/eval_name)", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    quick_fix_eval_button = ttk.Button(deploy_form, text="Apply eval_ Prefix")
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

    ttk.Label(deploy_form, text="Target Hz (optional)", style="Field.TLabel").grid(
        row=4, column=0, sticky="w", padx=(0, 6), pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_target_hz_var, width=20).grid(row=4, column=1, sticky="w", pady=4)

    # ── Calibration files (optional, browse for .json) ──
    for _cal_row, _cal_label, _cal_var in [
        (5, "Follower calibration (optional)", deploy_follower_calib_var),
        (6, "Leader calibration (optional)", deploy_leader_calib_var),
    ]:
        ttk.Label(deploy_form, text=_cal_label, style="Field.TLabel").grid(
            row=_cal_row, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(deploy_form, textvariable=_cal_var, width=52).grid(
            row=_cal_row, column=1, sticky="ew", pady=4,
        )
        _cal_btn = ttk.Frame(deploy_form, style="Panel.TFrame")
        _cal_btn.grid(row=_cal_row, column=2, sticky="w", padx=(6, 0), pady=4)
        ttk.Button(
            _cal_btn, text="Browse",
            command=lambda v=_cal_var: _browse_calibration_file(v),
        ).pack(side="left")
        ttk.Button(
            _cal_btn, text="Auto",
            command=lambda v=_cal_var: v.set(""),
        ).pack(side="left", padx=(4, 0))

    deploy_buttons = ttk.Frame(deploy_form, style="Panel.TFrame")
    deploy_buttons.grid(row=7, column=1, sticky="w", pady=(8, 0))
    preview_deploy_button = ttk.Button(deploy_buttons, text="Preview Command")
    preview_deploy_button.pack(side="left")
    run_deploy_button = ttk.Button(deploy_buttons, text="Run Deploy", style="Accent.TButton")
    run_deploy_button.pack(side="left", padx=(10, 0))
    scan_ports_button = ttk.Button(deploy_buttons, text="Scan Robot Ports")
    scan_ports_button.pack(side="left", padx=(10, 0))
    keyboard_help_button = ttk.Button(
        deploy_buttons,
        text="Keyboard Help",
        command=lambda: messagebox.showinfo(
            keyboard_input_help_title(),
            keyboard_input_help_text(),
        ),
    )
    keyboard_help_button.pack(side="left", padx=(10, 0))

    rename_map_flag = str(config.get("camera_rename_flag", "rename_map")).strip().lstrip("-") or "rename_map"
    deploy_advanced_fields = [
        ("robot.type", "Robot type"),
        ("robot.port", "Follower port"),
        ("robot.id", "Follower robot id"),
        ("robot.cameras", "Robot cameras JSON"),
        ("teleop.type", "Teleop type"),
        ("teleop.port", "Leader port"),
        ("teleop.id", "Leader robot id"),
        ("policy.path", "Policy path"),
        (rename_map_flag, "Camera rename map JSON"),
    ]
    deploy_advanced_vars: dict[str, Any] = {
        key: tk.StringVar(value="")
        for key, _ in deploy_advanced_fields
    }

    ttk.Checkbutton(
        deploy_form,
        text="Advanced command options",
        variable=deploy_advanced_enabled_var,
    ).grid(row=8, column=1, sticky="w", pady=(6, 0))

    deploy_advanced_frame = ttk.LabelFrame(
        deploy_form,
        text="Advanced Deploy Options",
        style="Section.TLabelframe",
        padding=10,
    )
    deploy_advanced_frame.columnconfigure(1, weight=1)
    ttk.Label(
        deploy_advanced_frame,
        text=(
            "Advanced is for low-level command flags. "
            "Set eval dataset/episodes/task/time/target Hz in the main setup above."
        ),
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
    root_label = tk.Label(root_row, text="Root:", bg=panel, fg=muted, font=(ui_font, 10))
    root_label.grid(
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

    # Tree view style (centralized with shared theme helper)
    _ts = ttk.Style(root)
    configure_treeview_style(
        style=_ts,
        style_name="Model.Treeview",
        colors=colors,
        body_font=(mono_font, 10),
        heading_font=(ui_font, 10, "bold"),
        rowheight=26,
    )
    _ts.configure("Model.Treeview", indent=18)

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
    bind_yview_wheel_scroll(model_tree)

    # Bottom row: Refresh + Browse Model + Pull + selected path display
    bottom_row = tk.Frame(model_section, bg=panel)
    bottom_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    bottom_row.columnconfigure(4, weight=1)

    refresh_models_button = ttk.Button(bottom_row, text="Refresh")
    refresh_models_button.grid(row=0, column=0, sticky="w")

    browse_model_button = ttk.Button(bottom_row, text="Browse Model...")
    browse_model_button.grid(row=0, column=1, sticky="w", padx=(6, 0))

    sync_model_hf_button = ttk.Button(bottom_row, text="Upload Model to Hugging Face...")
    sync_model_hf_button.grid(row=0, column=2, sticky="w", padx=(6, 0))

    selected_path_var = tk.StringVar(value="No model selected.")
    path_border = tk.Frame(bottom_row, bg=accent, width=3)
    path_border.grid(row=0, column=3, sticky="ns", padx=(12, 4))
    path_border.grid_propagate(False)
    selected_path_label = tk.Label(
        bottom_row,
        textvariable=selected_path_var,
        bg=panel,
        fg=muted,
        font=(mono_font, 9),
        anchor="w",
        justify="left",
        wraplength=520,
    )
    selected_path_label.grid(row=0, column=4, sticky="ew")

    # ── Model info panel ─────────────────────────────────────────────────────
    model_info_var = tk.StringVar(value="No model selected.")
    model_info_panel = ttk.LabelFrame(deploy_container, text="Selected Model Info", style="Section.TLabelframe", padding=10)
    model_info_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(model_info_panel, textvariable=model_info_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    # ── Camera preview slot (shared preview placed here by gui_app) ─────────
    camera_slot = ttk.Frame(deploy_container, style="Panel.TFrame")
    camera_slot.pack(fill="x")

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
        try:
            entries = sorted(model_path.iterdir())
        except PermissionError as exc:
            model_info_var.set(
                "\n".join(
                    [
                        f"Selected path: {model_path}",
                        f"Access error: permission denied ({exc})",
                        "Fix: grant read+execute permissions to this model folder, then refresh.",
                    ]
                )
            )
            return
        except OSError as exc:
            model_info_var.set(
                "\n".join(
                    [
                        f"Selected path: {model_path}",
                        f"Access error: {exc}",
                        "Fix: verify the path exists and is readable, then refresh.",
                    ]
                )
            )
            return
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
            suggested = normalize_repo_id(
                str(config.get("hf_username", "")),
                suggest_eval_dataset_name(config, folder_name),
            )
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

    hf_model_sync_popup_state: dict[str, Any] = {"window": None}

    def open_model_sync_popup() -> None:
        popup = hf_model_sync_popup_state.get("window")
        if popup is not None and bool(popup.winfo_exists()):
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            return

        from tkinter import filedialog as _fd

        resolved_model_path = _resolve_model_path()
        default_local_model = str(config.get("deploy_hf_sync_local_model", "")).strip()
        if not default_local_model and resolved_model_path is not None:
            default_local_model = str(_resolve_payload_path(resolved_model_path))
        if not default_local_model:
            default_local_model = deploy_model_var.get().strip()

        default_owner = str(config.get("deploy_hf_sync_owner", "")).strip() or str(config.get("hf_username", "")).strip()
        default_repo_name = repo_name_only(str(config.get("deploy_hf_sync_repo_name", "")).strip(), owner=default_owner)
        if not default_repo_name and default_local_model:
            default_repo_name = Path(default_local_model).name

        local_model_var = tk.StringVar(value=default_local_model)
        owner_var = tk.StringVar(value=default_owner)
        repo_name_var = tk.StringVar(value=default_repo_name)
        skip_if_exists_var = tk.BooleanVar(value=bool(config.get("deploy_hf_sync_skip_if_exists", True)))
        status_var = tk.StringVar(
            value="Choose a local model folder, then preview or run artifact upload to Hugging Face."
        )

        popup = tk.Toplevel(root)
        popup.title("Upload Model to Hugging Face")
        fit_window_to_screen(
            window=popup,
            requested_width=940,
            requested_height=420,
            requested_min_width=840,
            requested_min_height=350,
        )
        popup.configure(bg=colors.get("panel", "#111111"))
        popup.transient(root)
        hf_model_sync_popup_state["window"] = popup

        def _on_close() -> None:
            hf_model_sync_popup_state["window"] = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)

        container = ttk.Frame(popup, style="Panel.TFrame", padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        ttk.Label(
            container,
            text=(
                "Upload sends your local model/checkpoint folder to a Hugging Face model repository.\n"
                "Use this for backups/sharing of trained artifacts. It does not run robot deploy/eval.\n"
                "Prerequisite: install and authenticate the HF CLI first (`huggingface-cli login`).\n"
                "Preview shows the exact command; parity check warns if the target repo already exists."
            ),
            style="Muted.TLabel",
            justify="left",
            wraplength=860,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(container, text="Local model folder", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=local_model_var, width=68).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=4,
        )
        browse_model_local_button = ttk.Button(container, text="Browse...")
        browse_model_local_button.grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(container, text="Local model candidates", style="Field.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        model_combo = ttk.Combobox(container, state="readonly", width=68, style="Dark.TCombobox")
        model_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        refresh_local_model_options_button = ttk.Button(container, text="Refresh Models")
        refresh_local_model_options_button.grid(row=2, column=3, sticky="w", pady=4)

        ttk.Label(container, text="HF owner", style="Field.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=4,
        )
        ttk.Entry(container, textvariable=owner_var, width=28).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(container, text="HF model name", style="Field.TLabel").grid(
            row=3, column=2, sticky="w", padx=(12, 6), pady=4,
        )
        ttk.Entry(container, textvariable=repo_name_var, width=30).grid(row=3, column=3, sticky="ew", pady=4)

        parity_row = ttk.Frame(container, style="Panel.TFrame")
        parity_row.grid(row=4, column=1, columnspan=3, sticky="w", pady=(6, 0))
        check_parity_button = ttk.Button(parity_row, text="Check Local/Remote Parity")
        check_parity_button.pack(side="left")

        controls_row = ttk.Frame(container, style="Panel.TFrame")
        controls_row.grid(row=5, column=1, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            controls_row,
            text="Skip upload when remote model already exists",
            variable=skip_if_exists_var,
        ).pack(anchor="w")

        buttons_row = ttk.Frame(container, style="Panel.TFrame")
        buttons_row.grid(row=6, column=1, columnspan=3, sticky="w", pady=(10, 0))
        preview_sync_button = ttk.Button(buttons_row, text="Preview Upload Command")
        preview_sync_button.pack(side="left")
        run_sync_button = ttk.Button(buttons_row, text="Upload Model to Hugging Face", style="Accent.TButton")
        run_sync_button.pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=status_var, style="Muted.TLabel", justify="left").grid(
            row=7,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        def _model_candidates() -> list[Path]:
            root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
            if not root_path.exists() or not root_path.is_dir():
                return []
            try:
                children = sorted(path for path in root_path.iterdir() if path.is_dir() and not path.name.startswith("."))
            except OSError:
                return []
            return children[:200]

        def _refresh_model_options() -> None:
            options = [str(path) for path in _model_candidates()]
            model_combo.configure(values=options)
            current = local_model_var.get().strip()
            if not current and options:
                local_model_var.set(options[0])
                current = options[0]
            if current and current in options:
                model_combo.set(current)

        def _choose_local_model() -> None:
            current = local_model_var.get().strip()
            initial_dir = current or str(deploy_root_var.get().strip() or config["trained_models_dir"])
            selected = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=initial_dir,
                title="Select local model folder",
            )
            if selected:
                local_model_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        def _save_model_sync_settings() -> None:
            cleaned_repo_name = repo_name_only(repo_name_var.get(), owner=owner_var.get())
            if cleaned_repo_name != str(repo_name_var.get()).strip():
                repo_name_var.set(cleaned_repo_name)
            config["deploy_hf_sync_local_model"] = local_model_var.get().strip()
            config["deploy_hf_sync_owner"] = owner_var.get().strip()
            config["deploy_hf_sync_repo_name"] = cleaned_repo_name
            config["deploy_hf_sync_skip_if_exists"] = bool(skip_if_exists_var.get())
            save_config(config, quiet=True)

        def _build_model_sync_request() -> tuple[dict[str, Any] | None, str | None]:
            local_model = Path(normalize_path(local_model_var.get().strip()))
            if not local_model.exists() or not local_model.is_dir():
                return None, f"Local model folder not found: {local_model}"

            cleaned_repo_name = repo_name_only(repo_name_var.get(), owner=owner_var.get())
            if cleaned_repo_name != str(repo_name_var.get()).strip():
                repo_name_var.set(cleaned_repo_name)
            repo_id = compose_repo_id(owner_var.get(), cleaned_repo_name)
            if repo_id is None:
                return None, "Hugging Face owner and model name are required."

            hf_cli = shutil.which("huggingface-cli")
            if hf_cli is None:
                return None, "huggingface-cli not found in PATH."

            upload_cmd = [
                "huggingface-cli",
                "upload",
                repo_id,
                str(local_model),
                "--repo-type",
                "model",
            ]

            exists = model_exists_on_hf(repo_id)
            parity_level, parity_detail = _model_hf_parity_detail(exists, repo_id)
            checks: list[tuple[str, str, str]] = [
                ("PASS", "Local model folder", str(local_model)),
                ("PASS", "Target model repo", repo_id),
                ("PASS", "huggingface-cli", hf_cli),
                (parity_level, "Parity", parity_detail),
            ]
            return {
                "local_model": local_model,
                "repo_id": repo_id,
                "upload_cmd": upload_cmd,
                "remote_exists": exists,
                "parity_detail": parity_detail,
                "checks": checks,
            }, None

        def _check_parity_now() -> None:
            request, error_text = _build_model_sync_request()
            if error_text or request is None:
                messagebox.showerror("HF Model Parity Check", error_text or "Unable to build parity check.")
                return
            status_var.set(str(request.get("parity_detail", "Parity check complete.")))

        def _preview_sync_command() -> None:
            request, error_text = _build_model_sync_request()
            if error_text or request is None:
                messagebox.showerror("Upload Model to Hugging Face", error_text or "Unable to build upload command.")
                return
            last_command_state["value"] = format_command(request["upload_cmd"])
            show_text_dialog(
                root=root,
                title="HF Model Upload Command",
                text=(
                    "Upload command:\n"
                    + format_command_for_dialog(request["upload_cmd"])
                ),
                copy_text=last_command_state["value"],
                wrap_mode="word",
            )

        def _run_sync_command() -> None:
            request, error_text = _build_model_sync_request()
            if error_text or request is None:
                messagebox.showerror("Upload Model to Hugging Face", error_text or "Unable to build upload command.")
                return

            repo_id = request["repo_id"]
            local_model = request["local_model"]
            remote_exists = request["remote_exists"]
            if remote_exists is True and skip_if_exists_var.get():
                messagebox.showinfo("Skipped", f"Remote model already exists. Upload skipped:\n{repo_id}")
                return
            if remote_exists is True and not messagebox.askyesno(
                "Remote Model Exists",
                f"{repo_id} already exists on Hugging Face.\nContinue upload anyway?",
            ):
                return
            if remote_exists is None and not messagebox.askyesno(
                "Parity Unknown",
                f"Could not verify remote parity for {repo_id}.\nContinue upload anyway?",
            ):
                return

            if not confirm_preflight_in_gui("HF Model Upload Preflight", request["checks"]):
                return

            if not ask_text_dialog(
                root=root,
                title="Confirm HF Model Upload",
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

            _save_model_sync_settings()
            config["hf_username"] = str(owner_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            save_config(config, quiet=True)
            refresh_header_subtitle()

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "HF model upload canceled.")
                    messagebox.showinfo("Canceled", "Hugging Face model upload was canceled.")
                    return
                if upload_code != 0:
                    set_running(False, "HF model upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Hugging Face model upload failed with exit code {upload_code}.")
                    return
                set_running(False, "HF model upload completed.")
                messagebox.showinfo(
                    "Done",
                    (
                        "Model upload to Hugging Face completed.\n\n"
                        f"Local model: {local_model}\n"
                        f"Hugging Face model repo: {repo_id}"
                    ),
                )

            run_process_async(
                request["upload_cmd"],
                get_lerobot_dir(config),
                after_upload,
                None,
                None,
                "upload",
                request["checks"],
                {"model_path": str(local_model), "model_repo_id": repo_id},
            )

        def _on_model_combo_selected(_: Any) -> None:
            selected = model_combo.get().strip()
            if selected:
                local_model_var.set(selected)
                if not repo_name_var.get().strip():
                    repo_name_var.set(Path(selected).name)

        browse_model_local_button.configure(command=_choose_local_model)
        refresh_local_model_options_button.configure(command=_refresh_model_options)
        model_combo.bind("<<ComboboxSelected>>", _on_model_combo_selected)
        check_parity_button.configure(command=_check_parity_now)
        preview_sync_button.configure(command=_preview_sync_command)
        run_sync_button.configure(command=_run_sync_command)
        _refresh_model_options()

    refresh_models_button.configure(command=refresh_local_models)
    browse_model_button.configure(command=browse_for_model)
    sync_model_hf_button.configure(command=open_model_sync_popup)

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
            target_hz_raw=deploy_target_hz_var.get(),
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
            deploy_advanced_frame.grid(row=9, column=1, columnspan=2, sticky="ew", pady=(2, 8))
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
            target_hz_raw=deploy_target_hz_var.get(),
            arg_overrides=arg_overrides,
            custom_args_raw=custom_args_raw,
        )

    def apply_eval_prefix_quick_fix() -> bool:
        current_value = deploy_eval_dataset_var.get().strip()
        suggested_repo_id, changed = suggest_eval_prefixed_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=current_value,
        )
        suggested_repo_id = normalize_repo_id(str(config.get("hf_username", "")), suggested_repo_id)
        if not changed:
            log_panel.append_log(f"Eval dataset already follows convention: {suggested_repo_id}")
            return False
        deploy_eval_dataset_var.set(suggested_repo_id)
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
                    "Apply these as deploy defaults now?"
                ),
            )
            if apply_guess:
                config["follower_port"] = follower_guess
                config["leader_port"] = leader_guess
                if deploy_advanced_enabled_var.get():
                    deploy_advanced_vars["robot.port"].set(follower_guess)
                    deploy_advanced_vars["teleop.port"].set(leader_guess)
                save_config(config, quiet=True)
                refresh_header_subtitle()
                log_panel.append_log(
                    f"Applied scanned deploy defaults: follower={follower_guess}, leader={leader_guess}"
                )

    def run_deploy_from_gui() -> None:
        # Persist user's calibration path choices so preflight can see them
        _f_calib = deploy_follower_calib_var.get().strip()
        _l_calib = deploy_leader_calib_var.get().strip()
        _calib_changed = False
        if config.get("follower_calibration_path", "") != _f_calib:
            config["follower_calibration_path"] = _f_calib
            _calib_changed = True
        if config.get("leader_calibration_path", "") != _l_calib:
            config["leader_calibration_path"] = _l_calib
            _calib_changed = True
        if _calib_changed:
            save_config(config, quiet=True)

        req, cmd, updated_config, error_text = build_current_deploy()
        if error_text or req is None or cmd is None or updated_config is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
            return

        current_eval_input = deploy_eval_dataset_var.get().strip() or req.eval_repo_id
        suggested_eval_repo_id, requires_quick_fix = suggest_eval_prefixed_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=current_eval_input,
        )
        suggested_eval_repo_id = normalize_repo_id(str(config.get("hf_username", "")), suggested_eval_repo_id)
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

        while True:
            preflight_checks = run_preflight_for_deploy(
                config=config,
                model_path=req.model_path,
                eval_repo_id=req.eval_repo_id,
                command=cmd,
            )

            model_candidate = _first_model_payload_candidate(preflight_checks)
            current_eval_input = deploy_eval_dataset_var.get().strip() or req.eval_repo_id
            suggested_repo, missing_eval_prefix = suggest_eval_prefixed_repo_id(
                username=str(config["hf_username"]),
                dataset_name_or_repo_id=current_eval_input,
            )
            suggested_repo = normalize_repo_id(str(config.get("hf_username", "")), suggested_repo)
            preflight_events = checks_to_events(preflight_checks)
            quick_actions, action_context = _quick_actions_from_diagnostics(preflight_events)

            rename_map_suggestion = _camera_rename_map_suggestion(preflight_checks)
            rename_map_applied = bool(get_flag_value(cmd, rename_map_flag))
            rename_ctx = action_context.get("apply_rename_map", {})
            rename_map_from_context = str(rename_ctx.get("rename_map_suggestion", "")).strip()
            if not rename_map_suggestion and rename_map_from_context:
                rename_map_suggestion = rename_map_from_context
            if rename_map_applied:
                quick_actions = [item for item in quick_actions if item[0] != "apply_rename_map"]

            model_ctx = action_context.get("fix_model_payload", {})
            model_candidate_from_context = str(model_ctx.get("model_candidate", "")).strip()
            if model_candidate_from_context:
                model_candidate = model_candidate_from_context
            if model_candidate and Path(model_candidate) == req.model_path:
                quick_actions = [item for item in quick_actions if item[0] != "fix_model_payload"]

            eval_ctx = action_context.get("fix_eval_prefix", {})
            suggested_repo = str(eval_ctx.get("suggested_eval_repo_id", suggested_repo)).strip() or suggested_repo
            if not missing_eval_prefix:
                quick_actions = [item for item in quick_actions if item[0] != "fix_eval_prefix"]

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
                log_panel.append_log(f"Applied preflight quick fix: eval dataset -> {suggested_repo}")
            elif action == "apply_rename_map" and rename_map_suggestion:
                if not deploy_advanced_enabled_var.get():
                    deploy_advanced_enabled_var.set(True)
                deploy_advanced_vars[rename_map_flag].set(rename_map_suggestion)
                log_panel.append_log(f"Applied preflight quick fix: {rename_map_flag} -> {rename_map_suggestion}")
            elif action == "fix_model_payload" and model_candidate:
                deploy_model_var.set(str(model_candidate))
                if deploy_advanced_enabled_var.get():
                    deploy_advanced_vars["policy.path"].set(str(model_candidate))
                log_panel.append_log(f"Applied preflight quick fix: model payload -> {model_candidate}")
            elif action.startswith("fix_camera_fps:"):
                try:
                    new_fps = int(action.split("fix_camera_fps:", 1)[1])
                except (ValueError, IndexError):
                    new_fps = None
                if new_fps and new_fps > 0:
                    config["camera_fps"] = new_fps
                    save_config(config, quiet=True)
                    log_panel.append_log(f"Applied preflight quick fix: camera_fps -> {new_fps} Hz (matches model training FPS)")
            elif action in ("browse_follower_calib", "browse_leader_calib"):
                if action == "browse_follower_calib":
                    _target_var, _cfg_key, _lbl = deploy_follower_calib_var, "follower_calibration_path", "follower"
                else:
                    _target_var, _cfg_key, _lbl = deploy_leader_calib_var, "leader_calibration_path", "leader"
                _old_calib = _target_var.get().strip()
                _browse_calibration_file(_target_var)
                _new_calib = _target_var.get().strip()
                if _new_calib != _old_calib:
                    config[_cfg_key] = _new_calib
                    save_config(config, quiet=True)
                    log_panel.append_log(f"Applied preflight quick fix: {_cfg_key} -> {_new_calib or '(auto-detect)'}")
            elif action == "show_calib_cmd":
                calib_cmd = _build_calibration_command(config)
                show_text_dialog(
                    root=root,
                    title="Robot Recalibration Command",
                    text=(
                        "One or more calibration checks failed.\n"
                        "Run the command below to recalibrate the follower arm,\n"
                        "then re-run the deploy preflight.\n\n"
                        "IMPORTANT: power-cycle the arm and keep hands clear before running.\n\n"
                        + calib_cmd
                    ),
                    copy_text=calib_cmd,
                    width=900,
                    height=340,
                    wrap_mode="none",
                )

            req, cmd, updated_config, error_text = build_current_deploy()
            if error_text or req is None or cmd is None or updated_config is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
                return

        editable_cmd = ask_editable_command_dialog(
            root=root,
            title="Confirm Deploy Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the deploy command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Deploy",
            cancel_label="Cancel",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            log_panel.append_log("Running edited deploy command from command editor.")
        cmd = editable_cmd

        models_root = Path(normalize_path(str(updated_config.get("trained_models_dir", config["trained_models_dir"]))))
        effective_repo_id = normalize_repo_id(
            str(config.get("hf_username", "")),
            get_flag_value(cmd, "dataset.repo_id") or req.eval_repo_id,
        )
        episodes_text = get_flag_value(cmd, "dataset.num_episodes") or str(req.eval_num_episodes)
        duration_text = get_flag_value(cmd, "dataset.episode_time_s") or str(req.eval_duration_s)
        effective_task = get_flag_value(cmd, "dataset.single_task") or req.eval_task
        effective_model_text = get_policy_path_value(cmd) or str(req.model_path)
        effective_model_path = Path(normalize_path(str(effective_model_text)))
        if not effective_model_path.is_absolute():
            effective_model_path = models_root / effective_model_path
        try:
            effective_episodes = int(str(episodes_text).strip())
            effective_duration = int(str(duration_text).strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Edited command must keep eval episodes and duration as integers.")
            return
        if effective_episodes <= 0 or effective_duration <= 0:
            messagebox.showerror("Validation Error", "Edited command must keep eval episodes and duration greater than zero.")
            return

        preflight_checks = run_preflight_for_deploy(
            config=config,
            model_path=effective_model_path,
            eval_repo_id=effective_repo_id,
            command=cmd,
        )
        if not confirm_preflight_in_gui("Deploy Preflight", preflight_checks):
            return

        updated_config["eval_num_episodes"] = effective_episodes
        updated_config["eval_duration_s"] = effective_duration
        updated_config["eval_task"] = effective_task
        updated_config["last_eval_dataset_name"] = repo_name_only(effective_repo_id, owner=str(config.get("hf_username", "")))
        updated_config["deploy_target_hz"] = (get_flag_value(cmd, "dataset.fps") or get_flag_value(cmd, "fps") or "").strip()
        try:
            rel = effective_model_path.relative_to(models_root)
            updated_config["last_model_name"] = str(rel.parts[0]) if rel.parts else effective_model_path.name
            updated_config["last_checkpoint_name"] = str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else ""
        except ValueError:
            updated_config["last_model_name"] = effective_model_path.name
            updated_config["last_checkpoint_name"] = ""

        last_command_state["value"] = format_command(cmd)
        config.update(updated_config)
        save_config(config)
        deploy_eval_dataset_var.set(
            normalize_repo_id(
                str(config.get("hf_username", "")),
                suggest_eval_dataset_name(config, effective_model_path.name),
            )
        )
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
                config["last_dataset_repo_id"] = effective_repo_id
                save_config(config, quiet=True)
                set_running(False, "Deploy completed.")
                messagebox.showinfo(
                    "Done",
                    (
                        f"Deployment completed.\nModel: {effective_model_path}\nEval dataset: {effective_repo_id}\n\n"
                        "Open History and use the Deploy Outcome + Notes Editor to finalize episode edits and overall notes."
                    ),
                )

        run_process_async(
            cmd,
            lerobot_dir,
            after_deploy,
            effective_episodes,
            effective_episodes * effective_duration,
            "deploy",
            preflight_checks,
            {"dataset_repo_id": effective_repo_id, "model_path": str(effective_model_path)},
        )

    preview_deploy_button.configure(command=preview_deploy)
    run_deploy_button.configure(command=run_deploy_from_gui)
    scan_ports_button.configure(command=scan_robot_ports)
    quick_fix_eval_button.configure(command=apply_eval_prefix_quick_fix)
    refresh_eval_quick_fix_button_visibility()
    _refresh_deploy_advanced_visibility()
    deploy_advanced_enabled_var.trace_add("write", _refresh_deploy_advanced_visibility)

    refresh_local_models()
    update_model_info(_resolve_model_path())

    def apply_theme(updated_colors: dict[str, str]) -> None:
        nonlocal panel, surface, text_col, accent, border, muted, mono_font, ui_font
        panel = updated_colors.get("panel", "#111111")
        surface = updated_colors.get("surface", "#1a1a1a")
        text_col = updated_colors.get("text", "#eeeeee")
        accent = updated_colors.get("accent", "#f0a500")
        border = updated_colors.get("border", "#2d2d2d")
        muted = updated_colors.get("muted", "#777777")
        mono_font = updated_colors.get("font_mono", "TkFixedFont")
        ui_font = updated_colors.get("font_ui", "TkDefaultFont")
        for widget in (root_row, tree_frame, bottom_row):
            widget.configure(bg=panel)
        root_label.configure(bg=panel, fg=muted, font=(ui_font, 10))
        root_entry.configure(
            bg=surface,
            fg=text_col,
            insertbackground=text_col,
            highlightbackground=border,
            highlightcolor=border,
            font=(mono_font, 10),
        )
        selected_path_label.configure(bg=panel, fg=muted, font=(mono_font, 9))

        configure_treeview_style(
            style=_ts,
            style_name="Model.Treeview",
            colors=updated_colors,
            body_font=(mono_font, 10),
            heading_font=(ui_font, 10, "bold"),
            rowheight=26,
        )
        _ts.configure("Model.Treeview", indent=18)
        model_tree.tag_configure("model_root", foreground=updated_colors.get("success", "#22c55e"))
        model_tree.tag_configure("resolved", foreground=accent)
        model_tree.tag_configure("checkpoint", foreground=text_col)
        model_tree.tag_configure("folder", foreground=muted)
        model_tree.tag_configure("spacer", foreground=surface, background=surface)
        path_border.configure(bg=accent)

    return DeployTabHandles(
        deploy_root_var=deploy_root_var,
        deploy_eval_episodes_var=deploy_eval_episodes_var,
        deploy_eval_duration_var=deploy_eval_duration_var,
        deploy_eval_task_var=deploy_eval_task_var,
        camera_slot=camera_slot,
        refresh_local_models=refresh_local_models,
        select_model_path=select_model_path,
        apply_theme=apply_theme,
        action_buttons=[
            preview_deploy_button,
            run_deploy_button,
            scan_ports_button,
            keyboard_help_button,
            quick_fix_eval_button,
            refresh_models_button,
            sync_model_hf_button,
        ],
    )
