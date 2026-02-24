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
from .runner import format_command
from .training_profiles import load_training_profiles, save_training_profiles
from .types import GuiRunProcessAsync, TrainingProfile


@dataclass
class TrainingTabHandles:
    action_buttons: list[Any]
    refresh: Callable[[], None]


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
        remote_project_root="~/lerobot",
        env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
        default_tmux_session="",
        default_srun_prefix="",
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

    local_workdir_var = tk.StringVar(
        value=str(config.get("training_local_workdir", config.get("lerobot_dir", str(Path.home() / "lerobot"))))
    )
    local_command_var = tk.StringVar(
        value=str(config.get("training_local_command", "python -m lerobot.scripts.train --help"))
    )
    local_status_var = tk.StringVar(value="Run local training command on this device.")

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

    # --- Olympus training ---------------------------------------------------
    olympus_section = ttk.LabelFrame(frame, text="Olympus SSH Training", style="Section.TLabelframe", padding=10)
    olympus_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    olympus_section.columnconfigure(1, weight=1)
    olympus_section.columnconfigure(3, weight=1)

    olympus_host_var = tk.StringVar(value=olympus_profile.host)
    olympus_port_var = tk.StringVar(value=str(olympus_profile.port))
    olympus_user_var = tk.StringVar(value=olympus_profile.username)
    olympus_project_root_var = tk.StringVar(value=olympus_profile.remote_project_root)
    olympus_env_activate_var = tk.StringVar(value=olympus_profile.env_activate_cmd)
    olympus_command_var = tk.StringVar(
        value=str(config.get("training_remote_command", "python -m lerobot.scripts.train --help"))
    )
    olympus_status_var = tk.StringVar(
        value="Uses interactive SSH. Enter your password in terminal input if prompted."
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
            remote_project_root=olympus_project_root_var.get().strip() or "~/lerobot",
            env_activate_cmd=olympus_env_activate_var.get().strip() or "source ~/lerobot/lerobot_env/bin/activate",
            default_tmux_session="",
            default_srun_prefix="",
        )
        return profile, None

    def _persist_olympus_profile(profile: TrainingProfile) -> None:
        profiles, _ = load_training_profiles(config)
        others = [item for item in profiles if item.id != profile.id]
        save_training_profiles(config, others + [profile], profile.id)

    def _save_local_settings() -> None:
        config["training_remote_command"] = olympus_command_var.get().strip()
        config["training_local_workdir"] = normalize_path(local_workdir_var.get().strip() or config.get("lerobot_dir", ""))
        config["training_local_command"] = local_command_var.get().strip()
        save_config(config, quiet=True)

    def _save_training_settings(profile: TrainingProfile) -> None:
        _persist_olympus_profile(profile)
        _save_local_settings()

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
        olympus_command_var.set(str(config.get("training_remote_command", "python -m lerobot.scripts.train --help")))

        local_workdir_var.set(
            str(config.get("training_local_workdir", config.get("lerobot_dir", str(Path.home() / "lerobot"))))
        )
        local_command_var.set(str(config.get("training_local_command", "python -m lerobot.scripts.train --help")))

    preview_local_button.configure(command=preview_local)
    run_local_button.configure(command=run_local)
    save_olympus_button.configure(command=save_olympus_settings)
    preview_olympus_button.configure(command=preview_olympus)
    run_olympus_button.configure(command=run_olympus)

    refresh()

    action_buttons = [
        preview_local_button,
        run_local_button,
        save_olympus_button,
        preview_olympus_button,
        run_olympus_button,
    ]
    return TrainingTabHandles(action_buttons=action_buttons, refresh=refresh)
