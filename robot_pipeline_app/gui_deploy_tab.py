from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checks import run_preflight_for_deploy
from .config_store import get_lerobot_dir, normalize_path, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import ask_text_dialog, format_command_for_dialog, show_text_dialog
from .gui_forms import build_deploy_request_and_command
from .gui_log import GuiLogPanel
from .repo_utils import dataset_exists_on_hf, resolve_unique_repo_id, suggest_eval_dataset_name
from .runner import format_command
from .types import GuiRunProcessAsync


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

    deploy_container = ttk.Frame(deploy_tab, style="Panel.TFrame")
    deploy_container.pack(fill="both", expand=True)

    deploy_root_var = tk.StringVar(value=str(config["trained_models_dir"]))
    default_model_path = (
        str(Path(config["trained_models_dir"]) / config["last_model_name"])
        if str(config.get("last_model_name", "")).strip()
        else str(config["trained_models_dir"])
    )
    deploy_model_var = tk.StringVar(value=default_model_path)
    deploy_eval_dataset_var = tk.StringVar(
        value=str(config.get("last_eval_dataset_name", "")).strip()
        or suggest_eval_dataset_name(config, str(config.get("last_model_name", "")))
    )
    deploy_eval_episodes_var = tk.StringVar(value=str(config.get("eval_num_episodes", 10)))
    deploy_eval_duration_var = tk.StringVar(value=str(config.get("eval_duration_s", 20)))
    deploy_eval_task_var = tk.StringVar(value=str(config.get("eval_task", DEFAULT_TASK)))

    deploy_form = ttk.LabelFrame(deploy_container, text="Deploy / Eval Setup", style="Section.TLabelframe", padding=12)
    deploy_form.pack(fill="x")
    deploy_form.columnconfigure(1, weight=1)

    ttk.Label(deploy_form, text="Local model root folder", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_root_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_form, text="Browse", command=lambda: choose_folder(deploy_root_var)).grid(
        row=0,
        column=2,
        sticky="w",
        padx=(6, 0),
        pady=4,
    )

    ttk.Label(deploy_form, text="Model folder to deploy", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_model_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_form, text="Browse", command=lambda: choose_folder(deploy_model_var)).grid(
        row=1,
        column=2,
        sticky="w",
        padx=(6, 0),
        pady=4,
    )

    ttk.Label(deploy_form, text="Eval dataset name (or repo id)", style="Field.TLabel").grid(
        row=2,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_dataset_var, width=52).grid(row=2, column=1, sticky="ew", pady=4)

    ttk.Label(deploy_form, text="Eval episodes", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_episodes_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval episode time (seconds)", style="Field.TLabel").grid(
        row=4,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_duration_var, width=20).grid(row=4, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval task description", style="Field.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_task_var, width=52).grid(row=5, column=1, sticky="ew", pady=4)

    deploy_buttons = ttk.Frame(deploy_form, style="Panel.TFrame")
    deploy_buttons.grid(row=6, column=1, sticky="w", pady=(8, 0))
    preview_deploy_button = ttk.Button(deploy_buttons, text="Preview Command")
    preview_deploy_button.pack(side="left")
    run_deploy_button = ttk.Button(deploy_buttons, text="Run Deploy", style="Accent.TButton")
    run_deploy_button.pack(side="left", padx=(10, 0))

    model_section = ttk.LabelFrame(deploy_container, text="Local Models", style="Section.TLabelframe", padding=10)
    model_section.pack(fill="x", pady=(10, 0))
    model_section.columnconfigure(0, weight=1)
    model_listbox = tk.Listbox(
        model_section,
        height=8,
        bg="#111827",
        fg="#e5e7eb",
        selectbackground=colors["accent"],
        selectforeground="#ffffff",
        relief="flat",
        highlightthickness=1,
        highlightbackground=colors["border"],
    )
    model_listbox.grid(row=0, column=0, sticky="ew")
    refresh_models_button = ttk.Button(model_section, text="Refresh Model List")
    refresh_models_button.grid(row=0, column=1, sticky="n", padx=(8, 0))

    model_info_var = tk.StringVar(value="No model selected.")
    model_info_panel = ttk.LabelFrame(deploy_container, text="Selected Model Info", style="Section.TLabelframe", padding=10)
    model_info_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(model_info_panel, textvariable=model_info_var, style="Muted.TLabel", justify="left").pack(anchor="w")

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

    def refresh_local_models() -> None:
        model_listbox.delete(0, "end")
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        if not root_path.exists():
            return
        for folder in sorted(p.name for p in root_path.iterdir() if p.is_dir()):
            model_listbox.insert("end", folder)

    def on_model_select(_: Any) -> None:
        selected = model_listbox.curselection()
        if not selected:
            return
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        folder_name = model_listbox.get(selected[0])
        model_path = root_path / folder_name
        deploy_model_var.set(str(model_path))
        update_model_info(model_path)
        current_eval_name = deploy_eval_dataset_var.get().strip()
        if not current_eval_name or current_eval_name == auto_eval_hint["value"]:
            suggested = suggest_eval_dataset_name(config, folder_name)
            deploy_eval_dataset_var.set(suggested)
            auto_eval_hint["value"] = suggested

    model_listbox.bind("<<ListboxSelect>>", on_model_select)
    refresh_models_button.configure(command=refresh_local_models)

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
        req, cmd, updated_config, error_text = build_deploy_request_and_command(
            config=config,
            deploy_root_raw=deploy_root_var.get(),
            deploy_model_raw=deploy_model_var.get(),
            eval_dataset_raw=deploy_eval_dataset_var.get(),
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
            text=format_command_for_dialog(cmd),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="word",
        ):
            return

        preflight_checks = run_preflight_for_deploy(config=config, model_path=req.model_path)
        if not confirm_preflight_in_gui("Deploy Preflight", preflight_checks):
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
    refresh_local_models()
    update_model_info(Path(deploy_model_var.get()) if deploy_model_var.get().strip() else None)

    return DeployTabHandles(
        deploy_root_var=deploy_root_var,
        deploy_eval_episodes_var=deploy_eval_episodes_var,
        deploy_eval_duration_var=deploy_eval_duration_var,
        deploy_eval_task_var=deploy_eval_task_var,
        deploy_camera_preview=deploy_camera_preview,
        refresh_local_models=refresh_local_models,
        action_buttons=[preview_deploy_button, run_deploy_button, refresh_models_button],
    )
