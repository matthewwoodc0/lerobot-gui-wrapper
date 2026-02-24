from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checks import run_preflight_for_deploy, summarize_checks
from .config_store import get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import (
    ask_text_dialog,
    ask_text_dialog_with_actions,
    format_command_for_dialog,
    show_text_dialog,
)
from .gui_forms import build_deploy_request_and_command
from .gui_log import GuiLogPanel
from .repo_utils import (
    dataset_exists_on_hf,
    resolve_unique_repo_id,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .runner import format_command
from .types import GuiRunProcessAsync

_CAMERA_FIX_DETAIL_PATTERN = re.compile(r"configured=(\d+)x(\d+);\s*detected=(\d+)x(\d+)")


def _camera_resolution_fixes_from_checks(checks: list[tuple[str, str, str]]) -> dict[str, tuple[int, int]]:
    fixes: dict[str, tuple[int, int]] = {}
    for level, name, detail in checks:
        if level not in {"WARN", "FAIL"}:
            continue
        lowered = name.strip().lower()
        if lowered == "laptop camera resolution":
            role = "laptop"
        elif lowered == "phone camera resolution":
            role = "phone"
        else:
            continue

        match = _CAMERA_FIX_DETAIL_PATTERN.search(detail)
        if not match:
            continue
        detected_w = int(match.group(3))
        detected_h = int(match.group(4))
        fixes[role] = (detected_w, detected_h)
    return fixes


def _first_model_payload_candidate(checks: list[tuple[str, str, str]]) -> str | None:
    for _, name, detail in checks:
        if name.strip().lower() != "model payload candidates":
            continue
        candidate = detail.split(",", 1)[0].strip()
        return candidate or None
    return None


@dataclass
class DeployTabHandles:
    deploy_root_var: Any
    deploy_eval_episodes_var: Any
    deploy_eval_duration_var: Any
    deploy_eval_task_var: Any
    deploy_camera_preview: DualCameraPreview
    refresh_local_models: Callable[[], None]
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

    # ── Deploy form ───────────────────────────────────────────────────────────
    deploy_form = ttk.LabelFrame(deploy_container, text="Deploy / Eval Setup", style="Section.TLabelframe", padding=12)
    deploy_form.pack(fill="x")
    deploy_form.columnconfigure(1, weight=1)

    ttk.Label(deploy_form, text="Eval dataset name (or repo id)", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    quick_fix_eval_button = ttk.Button(deploy_form, text="Quick Fix eval_")
    quick_fix_eval_button.grid(row=0, column=2, sticky="w", padx=(6, 0), pady=4)

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

    # ── Two-panel model/checkpoint browser ───────────────────────────────────
    model_section = ttk.LabelFrame(deploy_container, text="Model Selection", style="Section.TLabelframe", padding=10)
    model_section.pack(fill="x", pady=(10, 0))
    model_section.columnconfigure(0, weight=1)

    # Root row
    root_row = tk.Frame(model_section, bg=panel)
    root_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    root_row.columnconfigure(1, weight=1)
    tk.Label(root_row, text="Root:", bg=panel, fg=muted, font=(colors.get("font_ui", "TkDefaultFont"), 10)).grid(
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
    ttk.Button(root_row, text="Browse", command=lambda: choose_folder(deploy_root_var)).grid(
        row=0, column=2, sticky="w", padx=(6, 0),
    )

    # Two listboxes side by side
    panels_frame = tk.Frame(model_section, bg=panel)
    panels_frame.grid(row=1, column=0, sticky="ew")
    panels_frame.columnconfigure(0, weight=1)
    panels_frame.columnconfigure(1, weight=1)

    # Model list (left)
    model_list_frame = tk.Frame(panels_frame, bg=panel)
    model_list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    model_list_frame.columnconfigure(0, weight=1)
    tk.Label(model_list_frame, text="Models", bg=panel, fg=accent, font=(colors.get("font_ui", "TkDefaultFont"), 10, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 4),
    )
    model_listbox = tk.Listbox(
        model_list_frame,
        height=8,
        bg=surface,
        fg=text_col,
        selectbackground=accent,
        selectforeground="#000000",
        font=(mono_font, 10),
        relief="flat",
        highlightthickness=1,
        highlightbackground=border,
        activestyle="none",
        exportselection=False,
    )
    model_listbox.grid(row=1, column=0, sticky="ew")
    model_sb = ttk.Scrollbar(model_list_frame, orient="vertical", command=model_listbox.yview)
    model_sb.grid(row=1, column=1, sticky="ns")
    model_listbox.configure(yscrollcommand=model_sb.set)

    # Checkpoint list (right)
    ckpt_list_frame = tk.Frame(panels_frame, bg=panel)
    ckpt_list_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    ckpt_list_frame.columnconfigure(0, weight=1)
    tk.Label(ckpt_list_frame, text="Checkpoints", bg=panel, fg=accent, font=(colors.get("font_ui", "TkDefaultFont"), 10, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 4),
    )
    checkpoint_listbox = tk.Listbox(
        ckpt_list_frame,
        height=8,
        bg=surface,
        fg=text_col,
        selectbackground=accent,
        selectforeground="#000000",
        font=(mono_font, 10),
        relief="flat",
        highlightthickness=1,
        highlightbackground=border,
        activestyle="none",
        exportselection=False,
    )
    checkpoint_listbox.grid(row=1, column=0, sticky="ew")
    ckpt_sb = ttk.Scrollbar(ckpt_list_frame, orient="vertical", command=checkpoint_listbox.yview)
    ckpt_sb.grid(row=1, column=1, sticky="ns")
    checkpoint_listbox.configure(yscrollcommand=ckpt_sb.set)

    # Refresh + selected path display
    bottom_row = tk.Frame(model_section, bg=panel)
    bottom_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    bottom_row.columnconfigure(1, weight=1)

    refresh_models_button = ttk.Button(bottom_row, text="Refresh")
    refresh_models_button.grid(row=0, column=0, sticky="w")

    selected_path_var = tk.StringVar(value="No model selected.")
    # Yellow left-border accent for selected path
    path_border = tk.Frame(bottom_row, bg=accent, width=3)
    path_border.grid(row=0, column=1, sticky="ns", padx=(12, 4))
    path_border.grid_propagate(False)
    tk.Label(
        bottom_row,
        textvariable=selected_path_var,
        bg=panel,
        fg=muted,
        font=(mono_font, 9),
        anchor="w",
        justify="left",
    ).grid(row=0, column=2, sticky="ew")

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

    # ── Internal state ───────────────────────────────────────────────────────
    _state: dict[str, str] = {
        "model_folder": _last_model_folder,
        "checkpoint": _last_checkpoint,
    }

    def _resolve_model_path() -> Path | None:
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        folder = _state["model_folder"]
        if not folder:
            return None
        p = root_path / folder
        ckpt = _state["checkpoint"]
        if ckpt and ckpt != "(model root)":
            p = p / ckpt
        return p

    def _update_selected_path_display() -> None:
        p = _resolve_model_path()
        if p is None:
            selected_path_var.set("No model selected.")
            deploy_model_var.set(str(config["trained_models_dir"]))
        else:
            selected_path_var.set(str(p))
            deploy_model_var.set(str(p))

    def update_model_info(model_path: Path | None) -> None:
        if model_path is None or not model_path.exists() or not model_path.is_dir():
            model_info_var.set("No model selected.")
            return
        entries = sorted(model_path.iterdir())
        child_names = [p.name for p in entries[:8]]
        checkpoints = [p.name for p in entries if p.is_dir() and "checkpoint" in p.name.lower()]
        has_config = any((model_path / name).exists() for name in ("config.json", "model_config.json"))
        info_lines = [
            f"Path: {model_path}",
            f"Items: {len(entries)} | Config file: {'yes' if has_config else 'no'}",
            f"Checkpoint-like folders: {', '.join(checkpoints[:4]) if checkpoints else 'none'}",
            f"Sample contents: {', '.join(child_names) if child_names else '(empty)'}",
        ]
        model_info_var.set("\n".join(info_lines))

    def _populate_checkpoints(model_folder_name: str) -> None:
        checkpoint_listbox.delete(0, "end")
        if not model_folder_name:
            checkpoint_listbox.insert("end", "(select a model first)")
            return
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        model_path = root_path / model_folder_name
        checkpoint_listbox.insert("end", "(model root)")
        if model_path.exists() and model_path.is_dir():
            for p in sorted(model_path.iterdir()):
                if p.is_dir() and "checkpoint" in p.name.lower():
                    checkpoint_listbox.insert("end", p.name)

    def _save_selection_to_config() -> None:
        config["last_model_name"] = _state["model_folder"]
        config["last_checkpoint_name"] = _state["checkpoint"]
        save_config(config, quiet=True)

    def on_model_select(_: Any) -> None:
        selected = model_listbox.curselection()
        if not selected:
            return
        folder_name = model_listbox.get(selected[0])
        _state["model_folder"] = folder_name
        _state["checkpoint"] = ""

        _populate_checkpoints(folder_name)
        # Auto-select "(model root)" by default
        checkpoint_listbox.selection_clear(0, "end")
        checkpoint_listbox.selection_set(0)

        _update_selected_path_display()
        update_model_info(_resolve_model_path())

        current_eval_name = deploy_eval_dataset_var.get().strip()
        if not current_eval_name or current_eval_name == auto_eval_hint["value"]:
            suggested = suggest_eval_dataset_name(config, folder_name)
            deploy_eval_dataset_var.set(suggested)
            auto_eval_hint["value"] = suggested

        _save_selection_to_config()

    def on_checkpoint_select(_: Any) -> None:
        selected = checkpoint_listbox.curselection()
        if not selected:
            return
        ckpt_name = checkpoint_listbox.get(selected[0])
        _state["checkpoint"] = "" if ckpt_name == "(model root)" else ckpt_name
        _update_selected_path_display()
        update_model_info(_resolve_model_path())
        _save_selection_to_config()

    model_listbox.bind("<<ListboxSelect>>", on_model_select)
    checkpoint_listbox.bind("<<ListboxSelect>>", on_checkpoint_select)

    def refresh_local_models() -> None:
        model_listbox.delete(0, "end")
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        if not root_path.exists():
            return
        for folder in sorted(p.name for p in root_path.iterdir() if p.is_dir()):
            model_listbox.insert("end", folder)
        _restore_selection_from_config()

    def _restore_selection_from_config() -> None:
        saved_folder = str(config.get("last_model_name", "")).strip()
        saved_ckpt = str(config.get("last_checkpoint_name", "")).strip()
        if not saved_folder:
            return

        # Find and select the model folder in the left list
        all_models = list(model_listbox.get(0, "end"))
        if saved_folder not in all_models:
            return
        idx = all_models.index(saved_folder)
        model_listbox.selection_clear(0, "end")
        model_listbox.selection_set(idx)
        model_listbox.see(idx)
        _state["model_folder"] = saved_folder

        # Populate checkpoints, then restore selection
        _populate_checkpoints(saved_folder)
        all_ckpts = list(checkpoint_listbox.get(0, "end"))
        if saved_ckpt and saved_ckpt in all_ckpts:
            cidx = all_ckpts.index(saved_ckpt)
            checkpoint_listbox.selection_set(cidx)
            checkpoint_listbox.see(cidx)
            _state["checkpoint"] = saved_ckpt
        else:
            # Default to "(model root)"
            checkpoint_listbox.selection_set(0)
            _state["checkpoint"] = ""

        _update_selected_path_display()
        update_model_info(_resolve_model_path())

    refresh_models_button.configure(command=refresh_local_models)

    # ── Deploy logic ─────────────────────────────────────────────────────────
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
        log_panel.append_log(f"Applied eval dataset quick fix: {suggested_repo_id}")
        return True

    def preview_deploy() -> None:
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
            wrap_mode="word",
        )

    def run_deploy_from_gui() -> None:
        def build_current_deploy() -> tuple[Any, Any, Any, Any]:
            return build_deploy_request_and_command(
                config=config,
                deploy_root_raw=deploy_root_var.get(),
                deploy_model_raw=deploy_model_var.get(),
                eval_dataset_raw=deploy_eval_dataset_var.get(),
                eval_episodes_raw=deploy_eval_episodes_var.get(),
                eval_duration_raw=deploy_eval_duration_var.get(),
                eval_task_raw=deploy_eval_task_var.get(),
            )

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
            log_panel.append_log(f"Applied eval dataset quick fix: {suggested_eval_repo_id}")
            req, cmd, updated_config, error_text = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=deploy_root_var.get(),
                deploy_model_raw=deploy_model_var.get(),
                eval_dataset_raw=suggested_eval_repo_id,
                eval_episodes_raw=deploy_eval_episodes_var.get(),
                eval_duration_raw=deploy_eval_duration_var.get(),
                eval_task_raw=deploy_eval_task_var.get(),
            )
            if error_text or req is None or cmd is None or updated_config is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build deploy command.")
                return

        lerobot_dir = get_lerobot_dir(config)
        resolved_repo_id, adjusted, _ = resolve_unique_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=req.eval_repo_id,
            local_roots=[lerobot_dir / "data"],
        )
        if adjusted:
            deploy_eval_dataset_var.set(resolved_repo_id)
            log_panel.append_log(f"Auto-iterated eval dataset to avoid existing target: {resolved_repo_id}")
            req, cmd, updated_config, error_text = build_deploy_request_and_command(
                config=config,
                deploy_root_raw=deploy_root_var.get(),
                deploy_model_raw=deploy_model_var.get(),
                eval_dataset_raw=resolved_repo_id,
                eval_episodes_raw=deploy_eval_episodes_var.get(),
                eval_duration_raw=deploy_eval_duration_var.get(),
                eval_task_raw=deploy_eval_task_var.get(),
            )
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

            camera_fixes = _camera_resolution_fixes_from_checks(preflight_checks)
            model_candidate = _first_model_payload_candidate(preflight_checks)
            current_eval_input = deploy_eval_dataset_var.get().strip() or req.eval_repo_id
            suggested_repo, missing_eval_prefix = suggest_eval_prefixed_repo_id(
                username=str(config["hf_username"]),
                dataset_name_or_repo_id=current_eval_input,
            )

            quick_actions: list[tuple[str, str]] = []
            if missing_eval_prefix:
                quick_actions.append(("fix_eval_prefix", "Apply eval_ Prefix"))
            if camera_fixes:
                quick_actions.append(("fix_camera", "Fix Camera Resolution"))
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
                log_panel.append_log(f"Applied preflight quick fix: eval dataset -> {suggested_repo}")
                command_changed_after_confirm = True
            elif action == "fix_camera":
                for role, (width, height) in camera_fixes.items():
                    config[f"camera_{role}_width"] = int(width)
                    config[f"camera_{role}_height"] = int(height)
                    log_panel.append_log(f"Applied preflight quick fix: {role} camera resolution -> {width}x{height}")
                save_config(config, quiet=True)
                refresh_header_subtitle()
                command_changed_after_confirm = True
            elif action == "fix_model_payload" and model_candidate:
                deploy_model_var.set(str(model_candidate))
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
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        config.update(updated_config)
        save_config(config)
        deploy_eval_dataset_var.set(suggest_eval_dataset_name(config, req.model_path.name))
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
                    f"Deployment completed.\nModel: {req.model_path}\nEval dataset: {req.eval_repo_id}",
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

    refresh_local_models()
    update_model_info(_resolve_model_path())

    return DeployTabHandles(
        deploy_root_var=deploy_root_var,
        deploy_eval_episodes_var=deploy_eval_episodes_var,
        deploy_eval_duration_var=deploy_eval_duration_var,
        deploy_eval_task_var=deploy_eval_task_var,
        deploy_camera_preview=deploy_camera_preview,
        refresh_local_models=refresh_local_models,
        action_buttons=[preview_deploy_button, run_deploy_button, quick_fix_eval_button, refresh_models_button],
    )
