from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config_store import normalize_path, save_config
from .gui_dialogs import ask_text_dialog, format_command_for_dialog, show_text_dialog
from .gui_file_dialogs import ask_directory_dialog
from .gui_log import GuiLogPanel
from .repo_utils import normalize_repo_id, repo_name_from_repo_id
from .runner import format_command
from .training_profiles import load_training_profiles, save_training_profiles
from .types import GuiRunProcessAsync, TrainingProfile


@dataclass
class TrainingTabHandles:
    action_buttons: list[Any]
    refresh: Callable[[], None]


DEFAULT_TRAIN_HELP_COMMAND = "python3 -m lerobot.scripts.lerobot_train --help"
DEFAULT_OLYMPUS_SRUN_PREFIX = (
    "srun -p gpu-research --cpus-per-task=8 --gres=gpu:tesla:1 "
    "-J gpu-job1 -q olympus-research-gpu --pty"
)


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
    username = str(config.get("hf_username", "")).strip() or "lerobot"
    dataset_name = str(config.get("last_dataset_name", "dataset_1")).strip() or "dataset_1"
    return normalize_repo_id(username, dataset_name)


def _default_output_name(config: dict[str, Any]) -> str:
    dataset_repo = _default_dataset_repo_id(config)
    return repo_name_from_repo_id(dataset_repo) or "train_run"


def _legacy_train_help_command() -> str:
    return "python -m lerobot.scripts.train --help"


def _normalize_train_command_default(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_TRAIN_HELP_COMMAND
    if text == _legacy_train_help_command():
        return DEFAULT_TRAIN_HELP_COMMAND
    return text


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


def _ssh_terminal_command(profile: TrainingProfile) -> list[str]:
    return [
        "ssh",
        "-p",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=no",
        f"{profile.username}@{profile.host}",
    ]


def _default_olympus_profile() -> TrainingProfile:
    return TrainingProfile(
        id="olympus",
        name="Olympus",
        host="olympus.ece.tamu.edu",
        port=22,
        username="",
        auth_mode="password",
        identity_file="",
        remote_models_root="~/lerobot/trained_models",
        remote_project_root="~/lerobot/src",
        env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
        default_tmux_session="train",
        default_srun_prefix=DEFAULT_OLYMPUS_SRUN_PREFIX,
    )


def _pick_olympus_profile(config: dict[str, Any]) -> TrainingProfile:
    profiles, _ = load_training_profiles(config)
    for profile in profiles:
        if profile.id == "olympus":
            return profile
    for profile in profiles:
        if str(profile.host).strip().lower() == "olympus.ece.tamu.edu":
            return profile
    return profiles[0] if profiles else _default_olympus_profile()


def _remote_exec_command(
    *,
    project_root: str,
    env_activate_cmd: str,
    train_command: str,
) -> str:
    segments: list[str] = []
    if env_activate_cmd:
        segments.append(env_activate_cmd)
    if project_root:
        segments.append(f"cd {project_root}")
    segments.append(train_command)
    remote_shell = " && ".join(segments)
    return f"bash -lc {shlex.quote(remote_shell)}"


def _interactive_ssh_command(profile: TrainingProfile, remote_exec: str) -> list[str]:
    return [
        "ssh",
        "-p",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=no",
        f"{profile.username}@{profile.host}",
        remote_exec,
    ]


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
    import tkinter as tk
    from tkinter import ttk

    olympus_profile = _pick_olympus_profile(config)

    frame = ttk.Frame(training_tab, style="Panel.TFrame")
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)

    # --- Local training -----------------------------------------------------
    local_section = ttk.LabelFrame(frame, text="Local Training (On Device)", style="Section.TLabelframe", padding=10)
    local_section.grid(row=0, column=0, sticky="ew")
    local_section.columnconfigure(1, weight=1)

    default_local_workdir = str(config.get("lerobot_dir", str(Path.home() / "lerobot")))
    local_workdir_var = tk.StringVar(
        value=str(config.get("training_local_workdir", default_local_workdir))
    )
    local_command_var = tk.StringVar(
        value=_normalize_train_command_default(str(config.get("training_local_command", DEFAULT_TRAIN_HELP_COMMAND)))
    )
    local_status_var = tk.StringVar(value="Run local training command on this device.")
    default_repo = _default_dataset_repo_id(config)
    default_name = _default_output_name(config)
    default_output_dir = f"outputs/train/{default_name}"
    builder_policy_var = tk.StringVar(value=str(config.get("training_gen_policy_type", "act")).strip() or "act")
    builder_repo_var = tk.StringVar(value=str(config.get("training_gen_dataset_repo_id", default_repo)).strip() or default_repo)
    builder_output_dir_var = tk.StringVar(
        value=str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir
    )
    builder_job_name_var = tk.StringVar(value=str(config.get("training_gen_job_name", default_name)).strip() or default_name)
    builder_device_var = tk.StringVar(value=str(config.get("training_gen_device", "cuda")).strip() or "cuda")
    builder_batch_var = tk.StringVar(value=str(config.get("training_gen_batch_size", 8)))
    builder_steps_var = tk.StringVar(value=str(config.get("training_gen_steps", 100000)))
    builder_wandb_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_wandb_enable"), True))
    builder_push_hub_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_push_to_hub"), False))
    builder_extra_args_var = tk.StringVar(value=str(config.get("training_gen_extra_args", "")))
    builder_srun_prefix_var = tk.StringVar(
        value=(
            str(config.get("training_gen_srun_prefix", "")).strip()
            or str(olympus_profile.default_srun_prefix).strip()
            or DEFAULT_OLYMPUS_SRUN_PREFIX
        )
    )
    builder_tmux_session_var = tk.StringVar(
        value=(
            str(config.get("training_gen_tmux_session", "")).strip()
            or str(olympus_profile.default_tmux_session).strip()
            or "train"
        )
    )
    builder_local_tmux_session_var = tk.StringVar(value=str(config.get("training_gen_local_tmux_session", "")).strip())

    ttk.Label(local_section, text="Working directory", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(local_section, textvariable=local_workdir_var, width=62).grid(row=0, column=1, sticky="ew", pady=3)

    def choose_local_workdir() -> None:
        selected = ask_directory_dialog(
            root=root,
            filedialog=filedialog,
            initial_dir=local_workdir_var.get().strip() or str(Path.home()),
            title="Select local training working directory",
        )
        if selected:
            local_workdir_var.set(selected)

    ttk.Button(local_section, text="Browse", command=choose_local_workdir).grid(
        row=0, column=2, sticky="w", padx=(6, 0), pady=3,
    )

    ttk.Label(local_section, text="Train command", style="Field.TLabel").grid(
        row=1, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(local_section, textvariable=local_command_var, width=62).grid(
        row=1, column=1, columnspan=2, sticky="ew", pady=3,
    )

    local_buttons = ttk.Frame(local_section, style="Panel.TFrame")
    local_buttons.grid(row=2, column=1, sticky="w", pady=(8, 0))
    preview_local_button = ttk.Button(local_buttons, text="Preview Local Command")
    preview_local_button.pack(side="left")
    run_local_button = ttk.Button(local_buttons, text="Run Local Training", style="Accent.TButton")
    run_local_button.pack(side="left", padx=(8, 0))

    ttk.Label(local_section, textvariable=local_status_var, style="Muted.TLabel").grid(
        row=3,
        column=1,
        columnspan=2,
        sticky="w",
        pady=(6, 0),
    )

    # --- Shared command builder --------------------------------------------
    builder_section = ttk.LabelFrame(frame, text="Train Command Builder", style="Section.TLabelframe", padding=10)
    builder_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    builder_section.columnconfigure(1, weight=1)
    builder_section.columnconfigure(3, weight=1)

    ttk.Label(builder_section, text="Policy type", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_policy_var, width=20).grid(row=0, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Dataset repo id", style="Field.TLabel").grid(
        row=0, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_repo_var, width=44).grid(row=0, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Output dir", style="Field.TLabel").grid(
        row=1, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_output_dir_var, width=34).grid(row=1, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Job name", style="Field.TLabel").grid(
        row=1, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_job_name_var, width=28).grid(row=1, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Device", style="Field.TLabel").grid(
        row=2, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_device_var, width=16).grid(row=2, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Batch size", style="Field.TLabel").grid(
        row=2, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_batch_var, width=12).grid(row=2, column=3, sticky="w", pady=3)

    ttk.Label(builder_section, text="Steps", style="Field.TLabel").grid(
        row=3, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_steps_var, width=16).grid(row=3, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Olympus srun prefix", style="Field.TLabel").grid(
        row=3, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_srun_prefix_var, width=44).grid(row=3, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Olympus tmux session", style="Field.TLabel").grid(
        row=4, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_tmux_session_var, width=20).grid(row=4, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Local tmux session (optional)", style="Field.TLabel").grid(
        row=4, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_local_tmux_session_var, width=24).grid(row=4, column=3, sticky="w", pady=3)

    ttk.Label(builder_section, text="Extra args", style="Field.TLabel").grid(
        row=5, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(builder_section, textvariable=builder_extra_args_var, width=70).grid(
        row=5, column=1, columnspan=3, sticky="ew", pady=3,
    )

    builder_checks = ttk.Frame(builder_section, style="Panel.TFrame")
    builder_checks.grid(row=6, column=1, columnspan=3, sticky="w", pady=(2, 0))
    ttk.Checkbutton(builder_checks, text="W&B enabled", variable=builder_wandb_var).pack(side="left")
    ttk.Checkbutton(builder_checks, text="Push to hub", variable=builder_push_hub_var).pack(side="left", padx=(12, 0))

    builder_status_var = tk.StringVar(
        value="Build once, then fill Local/Olympus train command fields. Olympus fill includes srun + tmux wrappers."
    )
    ttk.Label(builder_section, textvariable=builder_status_var, style="Muted.TLabel").grid(
        row=7,
        column=1,
        columnspan=3,
        sticky="w",
        pady=(6, 0),
    )

    builder_buttons = ttk.Frame(builder_section, style="Panel.TFrame")
    builder_buttons.grid(row=8, column=1, columnspan=3, sticky="w", pady=(8, 0))
    fill_local_from_builder_button = ttk.Button(builder_buttons, text="Fill Local Command")
    fill_local_from_builder_button.pack(side="left")
    fill_olympus_from_builder_button = ttk.Button(builder_buttons, text="Fill Olympus Command")
    fill_olympus_from_builder_button.pack(side="left", padx=(8, 0))
    preview_generated_button = ttk.Button(builder_buttons, text="Preview Generated")
    preview_generated_button.pack(side="left", padx=(8, 0))

    # --- Olympus training ---------------------------------------------------
    olympus_section = ttk.LabelFrame(frame, text="Olympus SSH Training", style="Section.TLabelframe", padding=10)
    olympus_section.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    olympus_section.columnconfigure(1, weight=1)
    olympus_section.columnconfigure(3, weight=1)

    olympus_host_var = tk.StringVar(value=olympus_profile.host)
    olympus_port_var = tk.StringVar(value=str(olympus_profile.port))
    olympus_user_var = tk.StringVar(value=olympus_profile.username)
    olympus_project_root_var = tk.StringVar(value=olympus_profile.remote_project_root)
    olympus_env_activate_var = tk.StringVar(value=olympus_profile.env_activate_cmd)
    olympus_command_var = tk.StringVar(
        value=_normalize_train_command_default(str(config.get("training_remote_command", DEFAULT_TRAIN_HELP_COMMAND)))
    )
    olympus_status_var = tk.StringVar(
        value="Manual-first: preview/copy generated commands, or run directly over interactive SSH."
    )

    ttk.Label(olympus_section, text="Host", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_host_var, width=30).grid(row=0, column=1, sticky="ew", pady=3)

    ttk.Label(olympus_section, text="Port", style="Field.TLabel").grid(
        row=0, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_port_var, width=12).grid(row=0, column=3, sticky="w", pady=3)

    ttk.Label(olympus_section, text="Username", style="Field.TLabel").grid(
        row=1, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_user_var, width=30).grid(row=1, column=1, sticky="ew", pady=3)

    ttk.Label(olympus_section, text="Remote project root", style="Field.TLabel").grid(
        row=1, column=2, sticky="w", padx=(10, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_project_root_var, width=34).grid(
        row=1, column=3, sticky="ew", pady=3,
    )

    ttk.Label(olympus_section, text="Env activate cmd", style="Field.TLabel").grid(
        row=2, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_env_activate_var, width=62).grid(
        row=2, column=1, columnspan=3, sticky="ew", pady=3,
    )

    ttk.Label(olympus_section, text="Train command", style="Field.TLabel").grid(
        row=3, column=0, sticky="w", padx=(0, 6), pady=3,
    )
    ttk.Entry(olympus_section, textvariable=olympus_command_var, width=62).grid(
        row=3, column=1, columnspan=3, sticky="ew", pady=3,
    )

    olympus_buttons = ttk.Frame(olympus_section, style="Panel.TFrame")
    olympus_buttons.grid(row=4, column=1, sticky="w", pady=(8, 0))
    save_olympus_button = ttk.Button(olympus_buttons, text="Save Settings")
    save_olympus_button.pack(side="left")
    preview_olympus_button = ttk.Button(olympus_buttons, text="Preview SSH Command")
    preview_olympus_button.pack(side="left", padx=(8, 0))
    open_ssh_shell_button = ttk.Button(olympus_buttons, text="Open SSH Shell")
    open_ssh_shell_button.pack(side="left", padx=(8, 0))
    run_olympus_button = ttk.Button(olympus_buttons, text="Run Olympus Training", style="Accent.TButton")
    run_olympus_button.pack(side="left", padx=(8, 0))

    ttk.Label(olympus_section, textvariable=olympus_status_var, style="Muted.TLabel").grid(
        row=5,
        column=1,
        columnspan=3,
        sticky="w",
        pady=(6, 0),
    )

    def _build_olympus_profile() -> tuple[TrainingProfile | None, str | None]:
        host = olympus_host_var.get().strip()
        username = olympus_user_var.get().strip()
        if not host:
            return None, "Host is required."
        if not username:
            return None, "Username is required."

        try:
            port = int(olympus_port_var.get().strip() or "22")
        except ValueError:
            return None, "Port must be an integer."
        if port <= 0 or port > 65535:
            return None, "Port must be between 1 and 65535."

        profile = TrainingProfile(
            id="olympus",
            name="Olympus",
            host=host,
            port=port,
            username=username,
            auth_mode="password",
            identity_file="",
            remote_models_root=olympus_profile.remote_models_root,
            remote_project_root=olympus_project_root_var.get().strip() or "~/lerobot/src",
            env_activate_cmd=olympus_env_activate_var.get().strip() or "source ~/lerobot/lerobot_env/bin/activate",
            default_tmux_session=builder_tmux_session_var.get().strip() or "train",
            default_srun_prefix=builder_srun_prefix_var.get().strip() or DEFAULT_OLYMPUS_SRUN_PREFIX,
        )
        return profile, None

    def _persist_olympus_profile(profile: TrainingProfile) -> None:
        profiles, _ = load_training_profiles(config)
        others = [item for item in profiles if item.id != profile.id]
        save_training_profiles(config, others + [profile], profile.id)

    def _save_generator_settings() -> None:
        config["training_gen_policy_type"] = builder_policy_var.get().strip() or "act"
        config["training_gen_dataset_repo_id"] = builder_repo_var.get().strip()
        config["training_gen_output_dir"] = builder_output_dir_var.get().strip()
        config["training_gen_job_name"] = builder_job_name_var.get().strip()
        config["training_gen_device"] = builder_device_var.get().strip() or "cuda"
        config["training_gen_batch_size"] = builder_batch_var.get().strip()
        config["training_gen_steps"] = builder_steps_var.get().strip()
        config["training_gen_wandb_enable"] = bool(builder_wandb_var.get())
        config["training_gen_push_to_hub"] = bool(builder_push_hub_var.get())
        config["training_gen_extra_args"] = builder_extra_args_var.get().strip()
        config["training_gen_srun_prefix"] = builder_srun_prefix_var.get().strip()
        config["training_gen_tmux_session"] = builder_tmux_session_var.get().strip()
        config["training_gen_local_tmux_session"] = builder_local_tmux_session_var.get().strip()

    def _save_local_settings() -> None:
        config["training_remote_command"] = _normalize_train_command_default(olympus_command_var.get().strip())
        config["training_local_workdir"] = normalize_path(local_workdir_var.get().strip() or config.get("lerobot_dir", ""))
        config["training_local_command"] = _normalize_train_command_default(local_command_var.get().strip())
        _save_generator_settings()
        save_config(config, quiet=True)

    def _save_training_settings(profile: TrainingProfile) -> None:
        _persist_olympus_profile(profile)
        _save_local_settings()

    def _generated_commands_payload() -> tuple[dict[str, str] | None, str | None]:
        try:
            batch_size = int(builder_batch_var.get().strip())
        except ValueError:
            return None, "Batch size must be an integer."
        try:
            steps = int(builder_steps_var.get().strip())
        except ValueError:
            return None, "Steps must be an integer."

        base_command, error = _build_train_base_command(
            policy_type=builder_policy_var.get().strip(),
            dataset_repo_id=builder_repo_var.get().strip(),
            output_dir=builder_output_dir_var.get().strip(),
            job_name=builder_job_name_var.get().strip(),
            device=builder_device_var.get().strip(),
            batch_size=batch_size,
            steps=steps,
            wandb_enable=bool(builder_wandb_var.get()),
            push_to_hub=bool(builder_push_hub_var.get()),
            extra_args=builder_extra_args_var.get().strip(),
        )
        if base_command is None:
            return None, error or "Unable to build train command."

        local_command = _wrap_train_with_tmux(base_command, builder_local_tmux_session_var.get().strip())
        remote_command = _wrap_train_with_srun(base_command, builder_srun_prefix_var.get().strip())
        olympus_command = _wrap_train_with_tmux(remote_command, builder_tmux_session_var.get().strip())
        expected_remote_model = _expected_pretrained_model_path(
            olympus_project_root_var.get().strip(),
            builder_output_dir_var.get().strip(),
        )
        expected_local_model = _expected_pretrained_model_path(
            local_workdir_var.get().strip(),
            builder_output_dir_var.get().strip(),
        )
        return {
            "base_command": base_command,
            "local_command": local_command,
            "olympus_command": olympus_command,
            "expected_remote_model": expected_remote_model,
            "expected_local_model": expected_local_model,
        }, None

    def fill_local_from_builder() -> None:
        payload, error = _generated_commands_payload()
        if payload is None:
            messagebox.showerror("Train Command Builder", error or "Unable to generate local command.")
            return
        local_command_var.set(payload["local_command"])
        builder_status_var.set("Local command updated from builder fields.")
        local_status_var.set(f"Generated local command. Expected model path: {payload['expected_local_model']}")
        _save_generator_settings()
        save_config(config, quiet=True)
        log_panel.append_log("Generated local training command from builder.")

    def fill_olympus_from_builder() -> None:
        payload, error = _generated_commands_payload()
        if payload is None:
            messagebox.showerror("Train Command Builder", error or "Unable to generate Olympus command.")
            return
        olympus_command_var.set(payload["olympus_command"])
        builder_status_var.set("Olympus command updated from builder fields.")
        olympus_status_var.set(
            "Generated Olympus command with srun+tmux wrappers. "
            f"Expected remote model path: {payload['expected_remote_model']}"
        )
        _save_generator_settings()
        save_config(config, quiet=True)
        log_panel.append_log("Generated Olympus training command from builder.")

    def preview_generated_commands() -> None:
        payload, error = _generated_commands_payload()
        if payload is None:
            messagebox.showerror("Train Command Builder", error or "Unable to generate commands.")
            return
        text = (
            "Base lerobot_train command:\n"
            f"{payload['base_command']}\n\n"
            "Local command:\n"
            f"{payload['local_command']}\n\n"
            "Olympus command (srun + tmux):\n"
            f"{payload['olympus_command']}\n\n"
            "Expected trained model paths:\n"
            f"- Local: {payload['expected_local_model']}\n"
            f"- Olympus: {payload['expected_remote_model']}\n\n"
            "Manual Olympus flow:\n"
            f"ssh -p {olympus_port_var.get().strip() or '22'} "
            f"{olympus_user_var.get().strip()}@{olympus_host_var.get().strip()}\n"
            f"tmux new-session -s {builder_tmux_session_var.get().strip() or 'train'}\n"
            f"cd {olympus_project_root_var.get().strip() or '~/lerobot/src'}\n"
            f"{olympus_env_activate_var.get().strip() or 'source ~/lerobot/lerobot_env/bin/activate'}\n"
            f"{_wrap_train_with_srun(payload['base_command'], builder_srun_prefix_var.get().strip())}"
        )
        last_command_state["value"] = payload["olympus_command"]
        show_text_dialog(
            root=root,
            title="Generated Training Commands",
            text=text,
            copy_text=text,
            wrap_mode="word",
        )

    def _local_command_payload() -> tuple[list[str] | None, Path | None, list[tuple[str, str, str]], str | None]:
        command_text = local_command_var.get().strip()
        if not command_text:
            return None, None, [], "Local train command is required."

        cwd_raw = local_workdir_var.get().strip()
        if not cwd_raw:
            return None, None, [], "Local working directory is required."

        cwd = Path(normalize_path(cwd_raw))
        if not cwd.exists() or not cwd.is_dir():
            return None, None, [], f"Working directory not found: {cwd}"

        bash_path = shutil.which("bash") or shutil.which("sh")
        if bash_path is None:
            return None, None, [], "Neither bash nor sh was found in PATH."

        cmd = [bash_path, "-lc", command_text]
        checks: list[tuple[str, str, str]] = [
            ("PASS", "Working directory", str(cwd)),
            ("PASS", "Train command", command_text),
            ("PASS", "Shell binary", bash_path),
        ]
        return cmd, cwd, checks, None

    def _remote_command_payload() -> tuple[list[str] | None, TrainingProfile | None, list[tuple[str, str, str]], str | None]:
        profile, profile_error = _build_olympus_profile()
        if profile is None:
            return None, None, [], profile_error or "Invalid Olympus settings."

        train_cmd = olympus_command_var.get().strip()
        if not train_cmd:
            return None, None, [], "Remote train command is required."

        remote_exec = _remote_exec_command(
            project_root=olympus_project_root_var.get().strip(),
            env_activate_cmd=olympus_env_activate_var.get().strip(),
            train_command=train_cmd,
        )
        cmd = _interactive_ssh_command(profile, remote_exec)

        ssh_path = shutil.which("ssh")
        checks: list[tuple[str, str, str]] = [
            ("PASS" if ssh_path else "FAIL", "ssh binary", ssh_path or "not found in PATH"),
            ("PASS", "SSH target", f"{profile.username}@{profile.host}:{profile.port}"),
            ("PASS", "Remote command", remote_exec),
            ("WARN", "Password entry", "Password will be requested in terminal during SSH login."),
        ]
        expected_model = _expected_pretrained_model_path(
            olympus_project_root_var.get().strip(),
            builder_output_dir_var.get().strip(),
        )
        if expected_model:
            checks.append(("PASS", "Expected model path", expected_model))
        if "srun " not in train_cmd:
            checks.append(("WARN", "srun wrapper", "Train command does not include srun; GPU queue may not be used."))
        if "tmux " not in train_cmd:
            checks.append(("WARN", "tmux wrapper", "Train command does not include tmux; session may close on disconnect."))
        return cmd, profile, checks, None

    def save_olympus_settings() -> None:
        profile, error = _build_olympus_profile()
        if profile is None:
            messagebox.showerror("Training Settings", error or "Invalid settings.")
            return
        _save_training_settings(profile)
        olympus_status_var.set("Saved local + Olympus settings.")
        log_panel.append_log("Saved training settings for local and Olympus modes.")

    def preview_local() -> None:
        cmd, _, _, error = _local_command_payload()
        if error or cmd is None:
            messagebox.showerror("Local Training", error or "Unable to build local command.")
            return
        command_text = format_command(cmd)
        last_command_state["value"] = command_text
        show_text_dialog(
            root=root,
            title="Local Training Command",
            text=format_command_for_dialog(cmd),
            copy_text=command_text,
            wrap_mode="word",
        )

    def run_local() -> None:
        cmd, cwd, checks, error = _local_command_payload()
        if error or cmd is None or cwd is None:
            messagebox.showerror("Local Training", error or "Unable to build local command.")
            return

        if not confirm_preflight_in_gui("Local Training Preflight", checks):
            return

        if not ask_text_dialog(
            root=root,
            title="Confirm Local Training",
            text=(
                "Review the local training command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            copy_text=format_command(cmd),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        _save_local_settings()
        last_command_state["value"] = format_command(cmd)

        def on_complete(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                set_running(False, "Local training canceled.")
                messagebox.showinfo("Canceled", "Local training was canceled.")
                return
            if return_code != 0:
                set_running(False, "Local training failed.", True)
                messagebox.showerror("Local Training Failed", f"Local training failed with exit code {return_code}.")
                return
            set_running(False, "Local training completed.")
            messagebox.showinfo("Local Training", "Local training command completed.")

        run_process_async(
            cmd,
            cwd,
            on_complete,
            None,
            None,
            "train_launch",
            checks,
            {
                "training_profile": "Local",
                "remote_host": "",
                "remote_path": str(cwd),
                "local_path": str(cwd),
                "template_name": "local_command",
                "training_transport": "local",
            },
        )

    def preview_olympus() -> None:
        cmd, _, _, error = _remote_command_payload()
        if error or cmd is None:
            messagebox.showerror("Olympus Training", error or "Unable to build SSH command.")
            return

        command_text = format_command(cmd)
        last_command_state["value"] = command_text
        show_text_dialog(
            root=root,
            title="Olympus SSH Training Command",
            text=format_command_for_dialog(cmd),
            copy_text=command_text,
            wrap_mode="word",
        )

    def open_ssh_shell() -> None:
        profile, profile_error = _build_olympus_profile()
        if profile is None:
            messagebox.showerror("Olympus Training", profile_error or "Invalid Olympus settings.")
            return

        cmd = _ssh_terminal_command(profile)
        ssh_path = shutil.which("ssh")
        checks: list[tuple[str, str, str]] = [
            ("PASS" if ssh_path else "FAIL", "ssh binary", ssh_path or "not found in PATH"),
            ("PASS", "SSH target", f"{profile.username}@{profile.host}:{profile.port}"),
            (
                "PASS",
                "Manual mode",
                "Interactive shell will open; run tmux/srun commands manually in terminal pane.",
            ),
        ]
        if not confirm_preflight_in_gui("Olympus SSH Preflight", checks):
            return

        last_command_state["value"] = format_command(cmd)
        olympus_status_var.set("Opened SSH shell. Run tmux/srun commands manually from the terminal pane.")

        def on_complete(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                set_running(False, "Olympus shell closed.")
                return
            if return_code != 0:
                set_running(False, "Olympus shell failed.", True)
                messagebox.showerror("Olympus SSH", f"SSH session exited with code {return_code}.")
                return
            set_running(False, "Olympus shell closed.")

        run_process_async(
            cmd,
            None,
            on_complete,
            None,
            None,
            "train_attach",
            checks,
            {
                "training_profile": profile.name,
                "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
                "remote_path": profile.remote_project_root,
                "local_path": "",
                "template_name": "olympus_shell",
                "training_transport": "ssh_manual",
            },
        )

    def run_olympus() -> None:
        cmd, profile, checks, error = _remote_command_payload()
        if error or cmd is None or profile is None:
            messagebox.showerror("Olympus Training", error or "Unable to build SSH command.")
            return

        if not confirm_preflight_in_gui("Olympus Training Preflight", checks):
            return

        if not ask_text_dialog(
            root=root,
            title="Confirm Olympus Training",
            text=(
                "Review the SSH training command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            copy_text=format_command(cmd),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        _save_training_settings(profile)
        last_command_state["value"] = format_command(cmd)

        def on_complete(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                set_running(False, "Olympus training canceled.")
                messagebox.showinfo("Canceled", "Olympus training was canceled.")
                return
            if return_code != 0:
                set_running(False, "Olympus training failed.", True)
                messagebox.showerror("Olympus Training Failed", f"Olympus training failed with exit code {return_code}.")
                return
            set_running(False, "Olympus training completed.")
            messagebox.showinfo("Olympus Training", "Olympus training command completed.")

        run_process_async(
            cmd,
            None,
            on_complete,
            None,
            None,
            "train_attach",
            checks,
            {
                "training_profile": profile.name,
                "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
                "remote_path": profile.remote_project_root,
                "local_path": "",
                "template_name": "olympus_ssh",
                "training_transport": "ssh_prompt",
            },
        )

    def refresh() -> None:
        nonlocal olympus_profile
        olympus_profile = _pick_olympus_profile(config)

        olympus_host_var.set(olympus_profile.host)
        olympus_port_var.set(str(olympus_profile.port))
        olympus_user_var.set(olympus_profile.username)
        olympus_project_root_var.set(olympus_profile.remote_project_root)
        olympus_env_activate_var.set(olympus_profile.env_activate_cmd)
        olympus_command_var.set(
            _normalize_train_command_default(str(config.get("training_remote_command", DEFAULT_TRAIN_HELP_COMMAND)))
        )

        local_workdir_var.set(
            str(config.get("training_local_workdir", config.get("lerobot_dir", str(Path.home() / "lerobot"))))
        )
        local_command_var.set(
            _normalize_train_command_default(str(config.get("training_local_command", DEFAULT_TRAIN_HELP_COMMAND)))
        )

        default_repo = _default_dataset_repo_id(config)
        default_name = _default_output_name(config)
        default_output_dir = f"outputs/train/{default_name}"
        builder_policy_var.set(str(config.get("training_gen_policy_type", "act")).strip() or "act")
        builder_repo_var.set(str(config.get("training_gen_dataset_repo_id", default_repo)).strip() or default_repo)
        builder_output_dir_var.set(
            str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir
        )
        builder_job_name_var.set(str(config.get("training_gen_job_name", default_name)).strip() or default_name)
        builder_device_var.set(str(config.get("training_gen_device", "cuda")).strip() or "cuda")
        builder_batch_var.set(str(config.get("training_gen_batch_size", 8)))
        builder_steps_var.set(str(config.get("training_gen_steps", 100000)))
        builder_wandb_var.set(_coerce_bool(config.get("training_gen_wandb_enable"), True))
        builder_push_hub_var.set(_coerce_bool(config.get("training_gen_push_to_hub"), False))
        builder_extra_args_var.set(str(config.get("training_gen_extra_args", "")))
        builder_srun_prefix_var.set(
            str(config.get("training_gen_srun_prefix", "")).strip()
            or str(olympus_profile.default_srun_prefix).strip()
            or DEFAULT_OLYMPUS_SRUN_PREFIX
        )
        builder_tmux_session_var.set(
            str(config.get("training_gen_tmux_session", "")).strip()
            or str(olympus_profile.default_tmux_session).strip()
            or "train"
        )
        builder_local_tmux_session_var.set(str(config.get("training_gen_local_tmux_session", "")).strip())

    preview_local_button.configure(command=preview_local)
    run_local_button.configure(command=run_local)
    fill_local_from_builder_button.configure(command=fill_local_from_builder)
    fill_olympus_from_builder_button.configure(command=fill_olympus_from_builder)
    preview_generated_button.configure(command=preview_generated_commands)
    save_olympus_button.configure(command=save_olympus_settings)
    preview_olympus_button.configure(command=preview_olympus)
    open_ssh_shell_button.configure(command=open_ssh_shell)
    run_olympus_button.configure(command=run_olympus)

    refresh()

    action_buttons = [
        preview_local_button,
        run_local_button,
        fill_local_from_builder_button,
        fill_olympus_from_builder_button,
        preview_generated_button,
        save_olympus_button,
        preview_olympus_button,
        open_ssh_shell_button,
        run_olympus_button,
    ]
    return TrainingTabHandles(action_buttons=action_buttons, refresh=refresh)
