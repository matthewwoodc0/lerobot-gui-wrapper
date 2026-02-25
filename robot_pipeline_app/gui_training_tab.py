from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable

from .config_store import save_config
from .gui_dialogs import show_text_dialog
from .gui_log import GuiLogPanel
from .repo_utils import normalize_repo_id, repo_name_from_repo_id
from .types import GuiRunProcessAsync


@dataclass
class TrainingTabHandles:
    action_buttons: list[Any]
    refresh: Callable[[], None]


DEFAULT_SRUN_PREFIX = (
    "srun --gres=gpu:1 --cpus-per-task=8 --pty"
)
DEFAULT_TMUX_SESSION = "train"
DEFAULT_PROJECT_ROOT = "~/lerobot/src"
DEFAULT_ENV_ACTIVATE = "source ~/lerobot/lerobot_env/bin/activate"


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _default_dataset_repo_id(config: dict[str, Any]) -> str:
    configured = str(config.get("training_gen_dataset_repo_id", "")).strip().strip("/")
    if configured:
        return configured
    dataset_name = str(config.get("last_dataset_name", "dataset_1")).strip() or "dataset_1"
    return normalize_repo_id("lerobot", dataset_name)


def _default_output_name(config: dict[str, Any]) -> str:
    dataset_repo = _default_dataset_repo_id(config)
    return repo_name_from_repo_id(dataset_repo) or "train_run"


def _build_train_base_command(
    *,
    policy_type: str,
    dataset_repo_id: str,
    output_dir: str,
    job_name: str,
    device: str,
    batch_size: int,
    steps: int,
    wandb_enable: bool,
    push_to_hub: bool,
    extra_args: str = "",
) -> tuple[str | None, str | None]:
    policy = str(policy_type).strip()
    dataset = str(dataset_repo_id).strip()
    out_dir = str(output_dir).strip()
    job = str(job_name).strip()
    device_name = str(device).strip()
    if not policy:
        return None, "Policy type is required."
    if not dataset:
        return None, "Dataset repo id is required."
    if not out_dir:
        return None, "Output dir is required."
    if not job:
        return None, "Job name is required."
    if not device_name:
        return None, "Policy device is required."
    if batch_size <= 0:
        return None, "Batch size must be greater than zero."
    if steps <= 0:
        return None, "Steps must be greater than zero."

    args = [
        "python3",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.type={policy}",
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--dataset.repo_id={dataset}",
        f"--output_dir={out_dir}",
        f"--job_name={job}",
        f"--policy.device={device_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--wandb.enable={'true' if wandb_enable else 'false'}",
    ]
    extra_text = str(extra_args or "").strip()
    if extra_text:
        try:
            args.extend(shlex.split(extra_text))
        except ValueError as exc:
            return None, f"Invalid extra args: {exc}"

    return shlex.join(args), None


def _wrap_train_with_srun(train_command: str, srun_prefix: str) -> str:
    base = str(train_command or "").strip()
    prefix = str(srun_prefix or "").strip()
    if not prefix:
        return base
    if prefix.startswith("srun "):
        return f"{prefix} {base}".strip()
    return f"srun {prefix} {base}".strip()


def _wrap_train_with_tmux(train_command: str, tmux_session: str) -> str:
    base = str(train_command or "").strip()
    session = str(tmux_session or "").strip()
    if not session:
        return base

    session_q = shlex.quote(session)
    base_q = shlex.quote(base)
    return (
        f"tmux has-session -t {session_q} 2>/dev/null && "
        f"tmux send-keys -t {session_q} {base_q} C-m || "
        f"tmux new-session -d -s {session_q} {base_q}; "
        "tmux ls"
    )


def _expected_pretrained_model_path(project_root: str, output_dir: str) -> str:
    root = str(project_root or "").strip().rstrip("/")
    raw_out_dir = str(output_dir or "").strip()
    suffix = "checkpoints/last/pretrained_model"
    if not raw_out_dir:
        return suffix
    if raw_out_dir.startswith("/"):
        return f"{raw_out_dir.rstrip('/')}/{suffix}"
    out_dir = raw_out_dir.strip("/")
    if not root:
        return f"{out_dir}/{suffix}"
    return f"{root}/{out_dir}/{suffix}"


def _build_generated_train_command(
    *,
    policy_type: str,
    dataset_repo_id: str,
    output_dir: str,
    job_name: str,
    device: str,
    batch_size: int,
    steps: int,
    wandb_enable: bool,
    push_to_hub: bool,
    extra_args: str,
    use_srun: bool,
    srun_prefix: str,
    use_tmux: bool,
    tmux_session: str,
) -> tuple[str | None, str | None]:
    base_command, error = _build_train_base_command(
        policy_type=policy_type,
        dataset_repo_id=dataset_repo_id,
        output_dir=output_dir,
        job_name=job_name,
        device=device,
        batch_size=batch_size,
        steps=steps,
        wandb_enable=wandb_enable,
        push_to_hub=push_to_hub,
        extra_args=extra_args,
    )
    if base_command is None:
        return None, error or "Unable to build command."

    cmd = base_command
    if use_srun:
        cmd = _wrap_train_with_srun(cmd, srun_prefix)
    if use_tmux:
        cmd = _wrap_train_with_tmux(cmd, tmux_session)
    return cmd, None


def setup_training_tab(
    *,
    root: Any,
    training_tab: Any,
    config: dict[str, Any],
    filedialog: Any,
    log_panel: GuiLogPanel,
    messagebox: Any,
    run_process_async: GuiRunProcessAsync,
    set_running: Callable[[bool, str | None, bool], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
) -> TrainingTabHandles:
    _ = (filedialog, run_process_async, set_running, confirm_preflight_in_gui)

    import tkinter as tk
    from tkinter import ttk

    frame = ttk.Frame(training_tab, style="Panel.TFrame")
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)

    default_name = _default_output_name(config)
    default_output_dir = f"outputs/train/{default_name}"

    policy_var = tk.StringVar(value=str(config.get("training_gen_policy_type", "act")).strip() or "act")
    repo_var = tk.StringVar(value=str(config.get("training_gen_dataset_repo_id", _default_dataset_repo_id(config))).strip())
    output_dir_var = tk.StringVar(
        value=str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir
    )
    job_name_var = tk.StringVar(value=str(config.get("training_gen_job_name", default_name)).strip() or default_name)
    device_var = tk.StringVar(value=str(config.get("training_gen_device", "cuda")).strip() or "cuda")
    batch_var = tk.StringVar(value=str(config.get("training_gen_batch_size", 8)))
    steps_var = tk.StringVar(value=str(config.get("training_gen_steps", 100000)))
    wandb_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_wandb_enable"), True))
    push_hub_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_push_to_hub"), False))
    extra_args_var = tk.StringVar(value=str(config.get("training_gen_extra_args", "")))
    use_srun_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_use_srun"), True))
    srun_prefix_var = tk.StringVar(
        value=str(config.get("training_gen_srun_prefix", "")).strip() or DEFAULT_SRUN_PREFIX
    )
    use_tmux_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_use_tmux"), True))
    tmux_session_var = tk.StringVar(
        value=str(config.get("training_gen_tmux_session", "")).strip() or DEFAULT_TMUX_SESSION
    )
    project_root_var = tk.StringVar(
        value=str(config.get("training_gen_project_root", "")).strip() or DEFAULT_PROJECT_ROOT
    )
    env_activate_var = tk.StringVar(
        value=str(config.get("training_gen_env_activate_cmd", "")).strip() or DEFAULT_ENV_ACTIVATE
    )

    status_var = tk.StringVar(
        value=(
            "Main flow: generate command, copy it, paste into your own terminal, "
            "and edit it in the mini editor if needed."
        )
    )

    builder_section = ttk.LabelFrame(frame, text="Training Command Generator", style="Section.TLabelframe", padding=10)
    builder_section.pack(fill="x")
    builder_section.columnconfigure(1, weight=1)
    builder_section.columnconfigure(3, weight=1)

    ttk.Label(builder_section, text="Policy type", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=policy_var, width=20).grid(row=0, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Dataset repo id", style="Field.TLabel").grid(
        row=0, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=repo_var, width=44).grid(row=0, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Output dir", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=output_dir_var, width=34).grid(row=1, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Job name", style="Field.TLabel").grid(row=1, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=job_name_var, width=28).grid(row=1, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Device", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=device_var, width=16).grid(row=2, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Batch size", style="Field.TLabel").grid(row=2, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=batch_var, width=12).grid(row=2, column=3, sticky="w", pady=3)

    ttk.Label(builder_section, text="Steps", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=steps_var, width=16).grid(row=3, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="srun prefix", style="Field.TLabel").grid(row=3, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_prefix_var, width=44).grid(row=3, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="tmux session", style="Field.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=tmux_session_var, width=20).grid(row=4, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Project root", style="Field.TLabel").grid(row=4, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=project_root_var, width=34).grid(row=4, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Env activate cmd", style="Field.TLabel").grid(
        row=5, column=0, sticky="w", padx=(0, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=env_activate_var, width=70).grid(
        row=5, column=1, columnspan=3, sticky="ew", pady=3
    )

    ttk.Label(builder_section, text="Extra args", style="Field.TLabel").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=extra_args_var, width=70).grid(
        row=6, column=1, columnspan=3, sticky="ew", pady=3
    )

    toggles = ttk.Frame(builder_section, style="Panel.TFrame")
    toggles.grid(row=7, column=1, columnspan=3, sticky="w", pady=(2, 0))
    ttk.Checkbutton(toggles, text="W&B enabled", variable=wandb_var).pack(side="left")
    ttk.Checkbutton(toggles, text="Push to hub", variable=push_hub_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(toggles, text="Wrap with srun", variable=use_srun_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(toggles, text="Wrap with tmux", variable=use_tmux_var).pack(side="left", padx=(12, 0))

    button_row = ttk.Frame(builder_section, style="Panel.TFrame")
    button_row.grid(row=8, column=1, columnspan=3, sticky="w", pady=(8, 0))
    generate_button = ttk.Button(button_row, text="Generate Command")
    generate_button.pack(side="left")
    copy_button = ttk.Button(button_row, text="Copy Command", style="Accent.TButton")
    copy_button.pack(side="left", padx=(8, 0))
    preview_button = ttk.Button(button_row, text="Preview Guidance")
    preview_button.pack(side="left", padx=(8, 0))
    save_button = ttk.Button(button_row, text="Save Defaults")
    save_button.pack(side="left", padx=(8, 0))

    editor_section = ttk.LabelFrame(frame, text="Generated Command (Editable)", style="Section.TLabelframe", padding=10)
    editor_section.pack(fill="both", expand=True, pady=(10, 0))
    editor_section.columnconfigure(0, weight=1)
    editor_section.rowconfigure(0, weight=1)

    command_text = tk.Text(editor_section, height=10, wrap="word")
    command_text.grid(row=0, column=0, sticky="nsew")
    command_scroll = ttk.Scrollbar(editor_section, orient="vertical", command=command_text.yview)
    command_scroll.grid(row=0, column=1, sticky="ns")
    command_text.configure(yscrollcommand=command_scroll.set)

    ttk.Label(frame, textvariable=status_var, style="Muted.TLabel", justify="left").pack(anchor="w", pady=(8, 0))

    def _editor_get() -> str:
        return str(command_text.get("1.0", "end")).strip()

    def _editor_set(value: str) -> None:
        command_text.delete("1.0", "end")
        text = str(value or "").strip()
        if text:
            command_text.insert("1.0", text)

    def _save_generator_settings() -> None:
        config["training_gen_policy_type"] = policy_var.get().strip() or "act"
        config["training_gen_dataset_repo_id"] = repo_var.get().strip()
        config["training_gen_output_dir"] = output_dir_var.get().strip()
        config["training_gen_job_name"] = job_name_var.get().strip()
        config["training_gen_device"] = device_var.get().strip() or "cuda"
        config["training_gen_batch_size"] = batch_var.get().strip()
        config["training_gen_steps"] = steps_var.get().strip()
        config["training_gen_wandb_enable"] = bool(wandb_var.get())
        config["training_gen_push_to_hub"] = bool(push_hub_var.get())
        config["training_gen_extra_args"] = extra_args_var.get().strip()
        config["training_gen_use_srun"] = bool(use_srun_var.get())
        config["training_gen_srun_prefix"] = srun_prefix_var.get().strip()
        config["training_gen_use_tmux"] = bool(use_tmux_var.get())
        config["training_gen_tmux_session"] = tmux_session_var.get().strip()
        config["training_gen_project_root"] = project_root_var.get().strip()
        config["training_gen_env_activate_cmd"] = env_activate_var.get().strip()
        config["training_generated_command"] = _editor_get()

    def _generate_command() -> tuple[str | None, str | None]:
        try:
            batch_size = int(batch_var.get().strip())
        except ValueError:
            return None, "Batch size must be an integer."
        try:
            steps = int(steps_var.get().strip())
        except ValueError:
            return None, "Steps must be an integer."

        return _build_generated_train_command(
            policy_type=policy_var.get().strip(),
            dataset_repo_id=repo_var.get().strip(),
            output_dir=output_dir_var.get().strip(),
            job_name=job_name_var.get().strip(),
            device=device_var.get().strip(),
            batch_size=batch_size,
            steps=steps,
            wandb_enable=bool(wandb_var.get()),
            push_to_hub=bool(push_hub_var.get()),
            extra_args=extra_args_var.get().strip(),
            use_srun=bool(use_srun_var.get()),
            srun_prefix=srun_prefix_var.get().strip(),
            use_tmux=bool(use_tmux_var.get()),
            tmux_session=tmux_session_var.get().strip(),
        )

    def generate_command() -> None:
        command, error = _generate_command()
        if command is None:
            messagebox.showerror("Training Command Generator", error or "Unable to generate command.")
            return

        _editor_set(command)
        last_command_state["value"] = command
        expected_path = _expected_pretrained_model_path(project_root_var.get().strip(), output_dir_var.get().strip())
        status_var.set(
            "Generated command. Copy and paste into your terminal. "
            f"Expected model path: {expected_path}"
        )
        _save_generator_settings()
        save_config(config, quiet=True)
        log_panel.append_log("Generated training command.")

    def copy_command() -> None:
        command = _editor_get()
        if not command:
            generated, error = _generate_command()
            if generated is None:
                messagebox.showerror("Training Command Generator", error or "Unable to generate command.")
                return
            command = generated
            _editor_set(command)

        try:
            root.clipboard_clear()
            root.clipboard_append(command)
        except Exception as exc:
            messagebox.showerror("Training Command Generator", f"Failed to copy command: {exc}")
            return

        last_command_state["value"] = command
        _save_generator_settings()
        save_config(config, quiet=True)
        status_var.set("Copied command to clipboard. Paste it into your terminal.")
        log_panel.append_log("Copied generated training command to clipboard.")

    def preview_guidance() -> None:
        command = _editor_get()
        if not command:
            generated, error = _generate_command()
            if generated is None:
                messagebox.showerror("Training Command Generator", error or "Unable to generate command.")
                return
            command = generated
            _editor_set(command)

        expected_path = _expected_pretrained_model_path(project_root_var.get().strip(), output_dir_var.get().strip())
        shell_steps = (
            "Manual terminal flow:\n"
            "1. Open your training terminal/session.\n"
            f"2. cd {project_root_var.get().strip() or DEFAULT_PROJECT_ROOT}\n"
            f"3. {env_activate_var.get().strip() or DEFAULT_ENV_ACTIVATE}\n"
            f"4. Paste and run command:\n{command}\n\n"
            f"Expected model path:\n{expected_path}"
        )
        last_command_state["value"] = command
        show_text_dialog(
            root=root,
            title="Training Command + Terminal Steps",
            text=shell_steps,
            copy_text=command,
            wrap_mode="word",
        )

    def save_defaults() -> None:
        _save_generator_settings()
        save_config(config, quiet=True)
        status_var.set("Saved training generator defaults.")
        log_panel.append_log("Saved training generator defaults.")

    def refresh() -> None:
        default_name = _default_output_name(config)
        default_output_dir = f"outputs/train/{default_name}"

        policy_var.set(str(config.get("training_gen_policy_type", "act")).strip() or "act")
        repo_var.set(str(config.get("training_gen_dataset_repo_id", _default_dataset_repo_id(config))).strip())
        output_dir_var.set(str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir)
        job_name_var.set(str(config.get("training_gen_job_name", default_name)).strip() or default_name)
        device_var.set(str(config.get("training_gen_device", "cuda")).strip() or "cuda")
        batch_var.set(str(config.get("training_gen_batch_size", 8)))
        steps_var.set(str(config.get("training_gen_steps", 100000)))
        wandb_var.set(_coerce_bool(config.get("training_gen_wandb_enable"), True))
        push_hub_var.set(_coerce_bool(config.get("training_gen_push_to_hub"), False))
        extra_args_var.set(str(config.get("training_gen_extra_args", "")))
        use_srun_var.set(_coerce_bool(config.get("training_gen_use_srun"), True))
        srun_prefix_var.set(str(config.get("training_gen_srun_prefix", "")).strip() or DEFAULT_SRUN_PREFIX)
        use_tmux_var.set(_coerce_bool(config.get("training_gen_use_tmux"), True))
        tmux_session_var.set(str(config.get("training_gen_tmux_session", "")).strip() or DEFAULT_TMUX_SESSION)
        project_root_var.set(str(config.get("training_gen_project_root", "")).strip() or DEFAULT_PROJECT_ROOT)
        env_activate_var.set(str(config.get("training_gen_env_activate_cmd", "")).strip() or DEFAULT_ENV_ACTIVATE)

        stored_command = str(config.get("training_generated_command", "")).strip()
        if stored_command:
            _editor_set(stored_command)
        else:
            command, _ = _generate_command()
            if command:
                _editor_set(command)

    generate_button.configure(command=generate_command)
    copy_button.configure(command=copy_command)
    preview_button.configure(command=preview_guidance)
    save_button.configure(command=save_defaults)

    refresh()

    action_buttons = [generate_button, copy_button, preview_button, save_button]
    return TrainingTabHandles(action_buttons=action_buttons, refresh=refresh)
