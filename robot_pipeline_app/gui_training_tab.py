from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable

from .compat import resolve_train_entrypoint
from .compat_policy import TRAINING_COMMAND_LABEL
from .config_store import save_config
from .gui_dialogs import show_text_dialog
from .gui_log import GuiLogPanel
from .repo_utils import normalize_repo_id, repo_name_from_repo_id
from .types import GuiRunProcessAsync


@dataclass
class TrainingTabHandles:
    action_buttons: list[Any]
    refresh: Callable[[], None]
    apply_theme: Callable[[dict[str, str]], None]


DEFAULT_PYTHON_BIN = "python"
DEFAULT_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_POLICY_INPUT_FEATURES = "null"
DEFAULT_POLICY_OUTPUT_FEATURES = "null"
DEFAULT_SRUN_PARTITION = "gpu-research"
DEFAULT_SRUN_CPUS_PER_TASK = 8
DEFAULT_SRUN_GRES = "gpu:a100:1"
DEFAULT_SRUN_QUEUE = "olympus-research-gpu"
DEFAULT_PROJECT_ROOT = "~/lerobot/src"
DEFAULT_ENV_ACTIVATE = "source ~/lerobot/lerobot_env/bin/activate"
DEFAULT_BATCH_SIZE = 16
DEFAULT_STEPS = 50000
DEFAULT_SAVE_FREQ = 5000
DEFAULT_HIL_BATCH_SIZE = 8
DEFAULT_HIL_STEPS = 3000
DEFAULT_HIL_SAVE_FREQ = 300


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
    last_repo_id = str(config.get("last_dataset_repo_id", "")).strip().strip("/")
    if last_repo_id:
        return last_repo_id
    owner = str(config.get("hf_username", "")).strip().strip("/")
    dataset_name = str(config.get("last_dataset_name", "")).strip().strip("/")
    if owner and dataset_name:
        return normalize_repo_id(owner, dataset_name)
    return ""


def _default_output_name(config: dict[str, Any]) -> str:
    dataset_repo = _default_dataset_repo_id(config)
    if dataset_repo:
        return repo_name_from_repo_id(dataset_repo) or "train_run"
    dataset_name = str(config.get("last_dataset_name", "")).strip()
    return dataset_name or "train_run"


def _build_train_base_command(
    *,
    python_bin: str,
    train_entrypoint: str,
    policy_path: str,
    policy_input_features: str,
    policy_output_features: str,
    dataset_repo_id: str,
    output_dir: str,
    job_name: str,
    device: str,
    batch_size: int,
    steps: int,
    save_freq: int,
    wandb_enable: bool,
    push_to_hub: bool,
    extra_args: str = "",
) -> tuple[str | None, str | None]:
    python_cmd = str(python_bin).strip()
    module_entrypoint = str(train_entrypoint).strip()
    policy = str(policy_path).strip()
    policy_inputs = str(policy_input_features).strip()
    policy_outputs = str(policy_output_features).strip()
    dataset = str(dataset_repo_id).strip()
    out_dir = str(output_dir).strip()
    job = str(job_name).strip()
    device_name = str(device).strip()

    if not python_cmd:
        return None, "Python binary is required."
    if not module_entrypoint:
        return None, "LeRobot train entrypoint is required."
    if not policy:
        return None, "Policy path is required."
    if not policy_inputs:
        return None, "Policy input features value is required."
    if not policy_outputs:
        return None, "Policy output features value is required."
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
    if save_freq <= 0:
        return None, "Save frequency must be greater than zero."

    args = [
        python_cmd,
        "-m",
        module_entrypoint,
        f"--policy.path={policy}",
        f"--policy.input_features={policy_inputs}",
        f"--policy.output_features={policy_outputs}",
        f"--dataset.repo_id={dataset}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--output_dir={out_dir}",
        f"--job_name={job}",
        f"--policy.device={device_name}",
        f"--wandb.enable={'true' if wandb_enable else 'false'}",
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--save_freq={save_freq}",
    ]
    extra_text = str(extra_args or "").strip()
    if extra_text:
        try:
            args.extend(shlex.split(extra_text))
        except ValueError as exc:
            return None, f"Invalid extra args: {exc}"

    return shlex.join(args), None


def _build_srun_prefix(
    *,
    partition: str,
    cpus_per_task: int,
    gres: str,
    srun_job_name: str,
    queue: str,
    extra_args: str = "",
) -> tuple[str | None, str | None]:
    part = str(partition).strip()
    gres_value = str(gres).strip()
    job_name = str(srun_job_name).strip()
    queue_name = str(queue).strip()

    if not part:
        return None, "srun partition is required."
    if cpus_per_task <= 0:
        return None, "srun cpus-per-task must be greater than zero."
    if not gres_value:
        return None, "srun gres is required."
    if not job_name:
        return None, "srun job name is required."
    if not queue_name:
        return None, "srun queue is required."

    args = [
        "srun",
        "-p",
        part,
        f"--cpus-per-task={cpus_per_task}",
        f"--gres={gres_value}",
        "-J",
        job_name,
        "-q",
        queue_name,
        "--pty",
    ]

    extra_text = str(extra_args or "").strip()
    if extra_text:
        try:
            args.extend(shlex.split(extra_text))
        except ValueError as exc:
            return None, f"Invalid srun extra args: {exc}"

    return shlex.join(args), None


def _wrap_train_with_srun(train_command: str, srun_prefix: str) -> str:
    base = str(train_command or "").strip()
    prefix = str(srun_prefix or "").strip()
    if not prefix:
        return base
    if prefix.startswith("srun "):
        return f"{prefix} {base}".strip()
    return f"srun {prefix} {base}".strip()


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


def _with_hil_suffix(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "hil_run"
    if text.endswith("_hil") or text.endswith("-hil"):
        return text
    return f"{text}_hil"


def _build_hil_workflow_text(
    *,
    project_root: str,
    env_activate_cmd: str,
    intervention_repo_id: str,
    base_model_path: str,
    command: str,
    expected_model_path: str,
) -> str:
    intervention_repo = str(intervention_repo_id or "").strip() or "<org/intervention_dataset_repo>"
    base_model = str(base_model_path or "").strip() or "<path_or_hub_id_of_previous_model>"
    cmd = str(command or "").strip() or "<generated_command>"

    return (
        "Human Intervention learning loop (incremental update, not full retraining):\n"
        "1. Capture human correction episodes for failure modes in teleop.\n"
        f"2. Push merged intervention dataset to: {intervention_repo}\n"
        f"3. Set Policy path to previous model checkpoint/hub id: {base_model}\n"
        "4. Use smaller values for Batch size / Steps / Save freq to run a short adaptation pass.\n"
        "5. Generate + run command from terminal:\n"
        f"   cd {project_root}\n"
        f"   {env_activate_cmd}\n"
        f"   {cmd}\n"
        "6. Validate behavior and repeat only on new intervention slices.\n\n"
        f"Expected updated model path:\n{expected_model_path}"
    )


def _build_generated_train_command(
    *,
    python_bin: str,
    train_entrypoint: str,
    policy_path: str,
    policy_input_features: str,
    policy_output_features: str,
    dataset_repo_id: str,
    output_dir: str,
    job_name: str,
    device: str,
    batch_size: int,
    steps: int,
    save_freq: int,
    wandb_enable: bool,
    push_to_hub: bool,
    extra_args: str,
    use_srun: bool,
    srun_partition: str,
    srun_cpus_per_task: int,
    srun_gres: str,
    srun_job_name: str,
    srun_queue: str,
    srun_extra_args: str,
) -> tuple[str | None, str | None]:
    base_command, error = _build_train_base_command(
        python_bin=python_bin,
        train_entrypoint=train_entrypoint,
        policy_path=policy_path,
        policy_input_features=policy_input_features,
        policy_output_features=policy_output_features,
        dataset_repo_id=dataset_repo_id,
        output_dir=output_dir,
        job_name=job_name,
        device=device,
        batch_size=batch_size,
        steps=steps,
        save_freq=save_freq,
        wandb_enable=wandb_enable,
        push_to_hub=push_to_hub,
        extra_args=extra_args,
    )
    if base_command is None:
        return None, error or "Unable to build command."

    cmd = base_command
    if use_srun:
        srun_prefix, srun_error = _build_srun_prefix(
            partition=srun_partition,
            cpus_per_task=srun_cpus_per_task,
            gres=srun_gres,
            srun_job_name=srun_job_name,
            queue=srun_queue,
            extra_args=srun_extra_args,
        )
        if srun_prefix is None:
            return None, srun_error or "Unable to build srun prefix."
        cmd = _wrap_train_with_srun(cmd, srun_prefix)
    return cmd, None


def setup_training_tab(
    *,
    root: Any,
    training_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str] | None = None,
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
    theme_colors = colors or {}

    default_name = _default_output_name(config)
    default_output_dir = f"outputs/train/{default_name}"
    detected_train_entrypoint = resolve_train_entrypoint(config)

    python_var = tk.StringVar(value=str(config.get("training_gen_python_bin", DEFAULT_PYTHON_BIN)).strip() or DEFAULT_PYTHON_BIN)
    policy_path_var = tk.StringVar(
        value=str(config.get("training_gen_policy_path", DEFAULT_POLICY_PATH)).strip() or DEFAULT_POLICY_PATH
    )
    policy_input_features_var = tk.StringVar(
        value=str(config.get("training_gen_policy_input_features", DEFAULT_POLICY_INPUT_FEATURES)).strip()
        or DEFAULT_POLICY_INPUT_FEATURES
    )
    policy_output_features_var = tk.StringVar(
        value=str(config.get("training_gen_policy_output_features", DEFAULT_POLICY_OUTPUT_FEATURES)).strip()
        or DEFAULT_POLICY_OUTPUT_FEATURES
    )
    repo_var = tk.StringVar(value=str(config.get("training_gen_dataset_repo_id", _default_dataset_repo_id(config))).strip())
    output_dir_var = tk.StringVar(
        value=str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir
    )
    job_name_var = tk.StringVar(value=str(config.get("training_gen_job_name", default_name)).strip() or default_name)
    device_var = tk.StringVar(value=str(config.get("training_gen_device", "cuda")).strip() or "cuda")
    batch_var = tk.StringVar(value=str(config.get("training_gen_batch_size", DEFAULT_BATCH_SIZE)))
    steps_var = tk.StringVar(value=str(config.get("training_gen_steps", DEFAULT_STEPS)))
    save_freq_var = tk.StringVar(value=str(config.get("training_gen_save_freq", DEFAULT_SAVE_FREQ)))
    wandb_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_wandb_enable"), True))
    push_hub_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_push_to_hub"), False))
    extra_args_var = tk.StringVar(value=str(config.get("training_gen_extra_args", "")))
    use_srun_var = tk.BooleanVar(value=_coerce_bool(config.get("training_gen_use_srun"), True))
    srun_partition_var = tk.StringVar(
        value=str(config.get("training_gen_srun_partition", DEFAULT_SRUN_PARTITION)).strip() or DEFAULT_SRUN_PARTITION
    )
    srun_cpus_var = tk.StringVar(
        value=str(config.get("training_gen_srun_cpus_per_task", DEFAULT_SRUN_CPUS_PER_TASK))
    )
    srun_gres_var = tk.StringVar(
        value=str(config.get("training_gen_srun_gres", DEFAULT_SRUN_GRES)).strip() or DEFAULT_SRUN_GRES
    )
    srun_job_var = tk.StringVar(
        value=str(config.get("training_gen_srun_job_name", default_name)).strip() or default_name
    )
    srun_queue_var = tk.StringVar(
        value=str(config.get("training_gen_srun_queue", DEFAULT_SRUN_QUEUE)).strip() or DEFAULT_SRUN_QUEUE
    )
    srun_extra_args_var = tk.StringVar(value=str(config.get("training_gen_srun_extra_args", "")))
    project_root_var = tk.StringVar(
        value=str(config.get("training_gen_project_root", "")).strip() or DEFAULT_PROJECT_ROOT
    )
    env_activate_var = tk.StringVar(
        value=str(config.get("training_gen_env_activate_cmd", "")).strip() or DEFAULT_ENV_ACTIVATE
    )
    hil_intervention_repo_var = tk.StringVar(
        value=str(config.get("training_gen_hil_intervention_repo_id", "")).strip()
    )
    hil_base_model_var = tk.StringVar(
        value=str(config.get("training_gen_hil_base_model_path", "")).strip()
    )

    status_var = tk.StringVar(
        value=(
            "Human Intervention Learning mode: apply the HIL preset, then copy/paste the command into your terminal. "
            "Use this tab for short intervention adaptation runs only."
        )
    )

    builder_section = ttk.LabelFrame(frame, text="Human Intervention Learning (HIL)", style="Section.TLabelframe", padding=10)
    builder_section.pack(fill="x")
    builder_section.columnconfigure(1, weight=1)
    builder_section.columnconfigure(3, weight=1)

    ttk.Label(builder_section, text="Policy path", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=policy_path_var, width=30).grid(row=0, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Dataset repo id", style="Field.TLabel").grid(
        row=0, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=repo_var, width=42).grid(row=0, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Output dir", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=output_dir_var, width=34).grid(row=1, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Job name", style="Field.TLabel").grid(row=1, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=job_name_var, width=26).grid(row=1, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Device", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=device_var, width=14).grid(row=2, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Batch size", style="Field.TLabel").grid(row=2, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=batch_var, width=12).grid(row=2, column=3, sticky="w", pady=3)

    ttk.Label(builder_section, text="Steps", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=steps_var, width=16).grid(row=3, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Save freq", style="Field.TLabel").grid(row=3, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=save_freq_var, width=12).grid(row=3, column=3, sticky="w", pady=3)

    ttk.Label(builder_section, text="Policy input features", style="Field.TLabel").grid(
        row=4, column=0, sticky="w", padx=(0, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=policy_input_features_var, width=30).grid(row=4, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Policy output features", style="Field.TLabel").grid(
        row=4, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=policy_output_features_var, width=30).grid(row=4, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Python binary", style="Field.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=python_var, width=20).grid(row=5, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="Extra train args", style="Field.TLabel").grid(
        row=5, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=extra_args_var, width=42).grid(row=5, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="srun partition", style="Field.TLabel").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_partition_var, width=22).grid(row=6, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="srun queue", style="Field.TLabel").grid(row=6, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_queue_var, width=30).grid(row=6, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="srun cpus/task", style="Field.TLabel").grid(row=7, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_cpus_var, width=14).grid(row=7, column=1, sticky="w", pady=3)

    ttk.Label(builder_section, text="srun gres", style="Field.TLabel").grid(row=7, column=2, sticky="w", padx=(10, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_gres_var, width=30).grid(row=7, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="srun job name", style="Field.TLabel").grid(row=8, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=srun_job_var, width=24).grid(row=8, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="srun extra args", style="Field.TLabel").grid(
        row=8, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=srun_extra_args_var, width=42).grid(row=8, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Project root", style="Field.TLabel").grid(row=9, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(builder_section, textvariable=project_root_var, width=34).grid(row=9, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="Env activate cmd", style="Field.TLabel").grid(
        row=9, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=env_activate_var, width=42).grid(row=9, column=3, sticky="ew", pady=3)

    ttk.Label(builder_section, text="HIL intervention repo", style="Field.TLabel").grid(
        row=10, column=0, sticky="w", padx=(0, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=hil_intervention_repo_var, width=34).grid(row=10, column=1, sticky="ew", pady=3)

    ttk.Label(builder_section, text="HIL base model path", style="Field.TLabel").grid(
        row=10, column=2, sticky="w", padx=(10, 6), pady=3
    )
    ttk.Entry(builder_section, textvariable=hil_base_model_var, width=42).grid(row=10, column=3, sticky="ew", pady=3)

    toggles = ttk.Frame(builder_section, style="Panel.TFrame")
    toggles.grid(row=11, column=1, columnspan=3, sticky="w", pady=(2, 0))
    ttk.Checkbutton(toggles, text="W&B enabled", variable=wandb_var).pack(side="left")
    ttk.Checkbutton(toggles, text="Push to hub", variable=push_hub_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(toggles, text="Wrap with srun", variable=use_srun_var).pack(side="left", padx=(12, 0))

    button_row = ttk.Frame(builder_section, style="Panel.TFrame")
    button_row.grid(row=12, column=1, columnspan=3, sticky="w", pady=(8, 0))
    hil_button = ttk.Button(button_row, text="Apply HIL Preset", style="Accent.TButton")
    hil_button.pack(side="left")
    copy_button = ttk.Button(button_row, text="Copy HIL Command")
    copy_button.pack(side="left", padx=(8, 0))
    save_button = ttk.Button(button_row, text="Save HIL Defaults")
    save_button.pack(side="left", padx=(8, 0))

    editor_section = ttk.LabelFrame(
        frame,
        text=f"{TRAINING_COMMAND_LABEL} (Editable)",
        style="Section.TLabelframe",
        padding=10,
    )
    editor_section.pack(fill="both", expand=True, pady=(10, 0))
    editor_section.columnconfigure(0, weight=1)
    editor_section.rowconfigure(0, weight=1)

    command_text = tk.Text(
        editor_section,
        height=10,
        wrap="word",
        bg=theme_colors.get("surface", "#2f2f2f"),
        fg=theme_colors.get("text", "#ffffff"),
        insertbackground=theme_colors.get("text", "#ffffff"),
        selectbackground=theme_colors.get("surface_alt", "#4b5563"),
        selectforeground=theme_colors.get("text", "#ffffff"),
        relief="flat",
        highlightthickness=1,
        highlightbackground=theme_colors.get("border", "#3f3f3f"),
        highlightcolor=theme_colors.get("border", "#3f3f3f"),
        font=(theme_colors.get("font_mono", "TkFixedFont"), 10),
        padx=8,
        pady=8,
    )
    command_text.grid(row=0, column=0, sticky="nsew")
    command_scroll = ttk.Scrollbar(
        editor_section,
        orient="vertical",
        command=command_text.yview,
        style="Dark.Vertical.TScrollbar",
    )
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
        config["training_gen_python_bin"] = python_var.get().strip() or DEFAULT_PYTHON_BIN
        config["training_gen_policy_path"] = policy_path_var.get().strip()
        config["training_gen_policy_input_features"] = policy_input_features_var.get().strip()
        config["training_gen_policy_output_features"] = policy_output_features_var.get().strip()
        config["training_gen_dataset_repo_id"] = repo_var.get().strip()
        config["training_gen_output_dir"] = output_dir_var.get().strip()
        config["training_gen_job_name"] = job_name_var.get().strip()
        config["training_gen_device"] = device_var.get().strip() or "cuda"
        config["training_gen_batch_size"] = batch_var.get().strip()
        config["training_gen_steps"] = steps_var.get().strip()
        config["training_gen_save_freq"] = save_freq_var.get().strip()
        config["training_gen_wandb_enable"] = bool(wandb_var.get())
        config["training_gen_push_to_hub"] = bool(push_hub_var.get())
        config["training_gen_extra_args"] = extra_args_var.get().strip()
        config["training_gen_use_srun"] = bool(use_srun_var.get())
        config["training_gen_srun_partition"] = srun_partition_var.get().strip()
        config["training_gen_srun_cpus_per_task"] = srun_cpus_var.get().strip()
        config["training_gen_srun_gres"] = srun_gres_var.get().strip()
        config["training_gen_srun_job_name"] = srun_job_var.get().strip()
        config["training_gen_srun_queue"] = srun_queue_var.get().strip()
        config["training_gen_srun_extra_args"] = srun_extra_args_var.get().strip()
        config["training_gen_project_root"] = project_root_var.get().strip()
        config["training_gen_env_activate_cmd"] = env_activate_var.get().strip()
        config["training_gen_hil_intervention_repo_id"] = hil_intervention_repo_var.get().strip()
        config["training_gen_hil_base_model_path"] = hil_base_model_var.get().strip()
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
        try:
            save_freq = int(save_freq_var.get().strip())
        except ValueError:
            return None, "Save freq must be an integer."

        srun_cpus_per_task = DEFAULT_SRUN_CPUS_PER_TASK
        if bool(use_srun_var.get()):
            try:
                srun_cpus_per_task = int(srun_cpus_var.get().strip())
            except ValueError:
                return None, "srun cpus-per-task must be an integer."

        return _build_generated_train_command(
            python_bin=python_var.get().strip() or DEFAULT_PYTHON_BIN,
            train_entrypoint=resolve_train_entrypoint(config) or detected_train_entrypoint,
            policy_path=policy_path_var.get().strip(),
            policy_input_features=policy_input_features_var.get().strip(),
            policy_output_features=policy_output_features_var.get().strip(),
            dataset_repo_id=repo_var.get().strip(),
            output_dir=output_dir_var.get().strip(),
            job_name=job_name_var.get().strip(),
            device=device_var.get().strip(),
            batch_size=batch_size,
            steps=steps,
            save_freq=save_freq,
            wandb_enable=bool(wandb_var.get()),
            push_to_hub=bool(push_hub_var.get()),
            extra_args=extra_args_var.get().strip(),
            use_srun=bool(use_srun_var.get()),
            srun_partition=srun_partition_var.get().strip(),
            srun_cpus_per_task=srun_cpus_per_task,
            srun_gres=srun_gres_var.get().strip(),
            srun_job_name=srun_job_var.get().strip() or job_name_var.get().strip(),
            srun_queue=srun_queue_var.get().strip(),
            srun_extra_args=srun_extra_args_var.get().strip(),
        )

    def copy_command() -> None:
        command = _editor_get()
        if not command:
            generated, error = _generate_command()
            if generated is None:
                messagebox.showerror("Human Intervention Learning", error or "Unable to generate HIL command.")
                return
            command = generated
            _editor_set(command)

        try:
            root.clipboard_clear()
            root.clipboard_append(command)
        except Exception as exc:
            messagebox.showerror("Human Intervention Learning", f"Failed to copy command: {exc}")
            return

        last_command_state["value"] = command
        _save_generator_settings()
        save_config(config, quiet=True)
        status_var.set("Copied HIL command to clipboard. Paste it into your terminal.")
        log_panel.append_log("Copied LeRobot training command to clipboard.")

    def apply_hil_preset() -> None:
        previous_output_dir = output_dir_var.get().strip()
        previous_job_name = job_name_var.get().strip()
        previous_expected_path = _expected_pretrained_model_path(project_root_var.get().strip(), previous_output_dir)

        batch_var.set(str(DEFAULT_HIL_BATCH_SIZE))
        steps_var.set(str(DEFAULT_HIL_STEPS))
        save_freq_var.set(str(DEFAULT_HIL_SAVE_FREQ))

        hil_output_dir = _with_hil_suffix(previous_output_dir)
        if not hil_output_dir.startswith("outputs/"):
            hil_output_dir = f"outputs/train/{hil_output_dir.strip('/')}"
        output_dir_var.set(hil_output_dir)
        job_name_var.set(_with_hil_suffix(previous_job_name))

        if not policy_path_var.get().strip():
            policy_path_var.set(previous_expected_path)

        command, error = _generate_command()
        if command is None:
            messagebox.showerror("Human Intervention Learning", error or "Unable to generate HIL command.")
            return

        expected_path = _expected_pretrained_model_path(project_root_var.get().strip(), output_dir_var.get().strip())
        guidance_text = _build_hil_workflow_text(
            project_root=project_root_var.get().strip() or DEFAULT_PROJECT_ROOT,
            env_activate_cmd=env_activate_var.get().strip() or DEFAULT_ENV_ACTIVATE,
            intervention_repo_id=hil_intervention_repo_var.get().strip() or repo_var.get().strip(),
            base_model_path=hil_base_model_var.get().strip() or previous_expected_path,
            command=command,
            expected_model_path=expected_path,
        )

        _editor_set(command)
        last_command_state["value"] = command
        status_var.set(
            "Applied HIL preset (smaller step budget) for intervention fine-tuning. "
            f"Expected model path: {expected_path}"
        )
        _save_generator_settings()
        save_config(config, quiet=True)
        log_panel.append_log("Applied HIL preset and generated a LeRobot training command.")

        show_text_dialog(
            root=root,
            title="Human Intervention Learning Plan",
            text=guidance_text,
            copy_text=command,
            wrap_mode="word",
        )

    def save_defaults() -> None:
        _save_generator_settings()
        save_config(config, quiet=True)
        status_var.set("Saved HIL defaults.")
        log_panel.append_log("Saved HIL defaults.")

    def refresh() -> None:
        default_name = _default_output_name(config)
        default_output_dir = f"outputs/train/{default_name}"

        python_var.set(str(config.get("training_gen_python_bin", DEFAULT_PYTHON_BIN)).strip() or DEFAULT_PYTHON_BIN)
        policy_path_var.set(str(config.get("training_gen_policy_path", DEFAULT_POLICY_PATH)).strip() or DEFAULT_POLICY_PATH)
        policy_input_features_var.set(
            str(config.get("training_gen_policy_input_features", DEFAULT_POLICY_INPUT_FEATURES)).strip()
            or DEFAULT_POLICY_INPUT_FEATURES
        )
        policy_output_features_var.set(
            str(config.get("training_gen_policy_output_features", DEFAULT_POLICY_OUTPUT_FEATURES)).strip()
            or DEFAULT_POLICY_OUTPUT_FEATURES
        )
        repo_var.set(str(config.get("training_gen_dataset_repo_id", _default_dataset_repo_id(config))).strip())
        output_dir_var.set(str(config.get("training_gen_output_dir", default_output_dir)).strip() or default_output_dir)
        job_name_var.set(str(config.get("training_gen_job_name", default_name)).strip() or default_name)
        device_var.set(str(config.get("training_gen_device", "cuda")).strip() or "cuda")
        batch_var.set(str(config.get("training_gen_batch_size", DEFAULT_BATCH_SIZE)))
        steps_var.set(str(config.get("training_gen_steps", DEFAULT_STEPS)))
        save_freq_var.set(str(config.get("training_gen_save_freq", DEFAULT_SAVE_FREQ)))
        wandb_var.set(_coerce_bool(config.get("training_gen_wandb_enable"), True))
        push_hub_var.set(_coerce_bool(config.get("training_gen_push_to_hub"), False))
        extra_args_var.set(str(config.get("training_gen_extra_args", "")))
        use_srun_var.set(_coerce_bool(config.get("training_gen_use_srun"), True))
        srun_partition_var.set(
            str(config.get("training_gen_srun_partition", DEFAULT_SRUN_PARTITION)).strip() or DEFAULT_SRUN_PARTITION
        )
        srun_cpus_var.set(str(config.get("training_gen_srun_cpus_per_task", DEFAULT_SRUN_CPUS_PER_TASK)))
        srun_gres_var.set(str(config.get("training_gen_srun_gres", DEFAULT_SRUN_GRES)).strip() or DEFAULT_SRUN_GRES)
        srun_job_var.set(str(config.get("training_gen_srun_job_name", default_name)).strip() or default_name)
        srun_queue_var.set(str(config.get("training_gen_srun_queue", DEFAULT_SRUN_QUEUE)).strip() or DEFAULT_SRUN_QUEUE)
        srun_extra_args_var.set(str(config.get("training_gen_srun_extra_args", "")))
        project_root_var.set(str(config.get("training_gen_project_root", "")).strip() or DEFAULT_PROJECT_ROOT)
        env_activate_var.set(str(config.get("training_gen_env_activate_cmd", "")).strip() or DEFAULT_ENV_ACTIVATE)
        hil_intervention_repo_var.set(str(config.get("training_gen_hil_intervention_repo_id", "")).strip())
        hil_base_model_var.set(str(config.get("training_gen_hil_base_model_path", "")).strip())

        stored_command = str(config.get("training_generated_command", "")).strip()
        if stored_command:
            _editor_set(stored_command)
        else:
            command, _ = _generate_command()
            if command:
                _editor_set(command)

    hil_button.configure(command=apply_hil_preset)
    copy_button.configure(command=copy_command)
    save_button.configure(command=save_defaults)

    refresh()

    def apply_theme(updated_colors: dict[str, str]) -> None:
        command_text.configure(
            bg=updated_colors.get("surface", "#2f2f2f"),
            fg=updated_colors.get("text", "#ffffff"),
            insertbackground=updated_colors.get("text", "#ffffff"),
            selectbackground=updated_colors.get("surface_alt", "#4b5563"),
            selectforeground=updated_colors.get("text", "#ffffff"),
            highlightbackground=updated_colors.get("border", "#3f3f3f"),
            highlightcolor=updated_colors.get("border", "#3f3f3f"),
            font=(updated_colors.get("font_mono", "TkFixedFont"), 10),
        )

    action_buttons = [hil_button, copy_button, save_button]
    return TrainingTabHandles(action_buttons=action_buttons, refresh=refresh, apply_theme=apply_theme)
