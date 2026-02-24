from __future__ import annotations

import json
import posixpath
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checks import run_preflight_for_training
from .config_store import normalize_path, save_config
from .gui_dialogs import ask_text_dialog, ask_text_dialog_with_actions, format_command_for_dialog, show_text_dialog
from .gui_file_dialogs import ask_directory_dialog
from .gui_log import GuiLogPanel
from .runner import format_command
from .training_auth import (
    delete_ssh_password,
    has_ssh_password,
    is_secret_tool_available,
    save_ssh_password,
)
from .training_profiles import load_training_profiles, save_training_profiles
from .training_remote import (
    build_pull_command,
    build_remote_launch_command,
    build_sftp_pull_command,
    command_uses_binary,
    ensure_host_trusted,
    list_remote_dirs,
)
from .training_templates import default_training_templates, render_template
from .types import GuiRunProcessAsync, TrainingProfile


@dataclass
class TrainingTabHandles:
    action_buttons: list[Any]
    refresh: Callable[[], None]


def _profile_label(profile: TrainingProfile) -> str:
    return f"{profile.name} ({profile.id})"


def _next_profile_id(existing_ids: set[str]) -> str:
    base = "profile"
    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if candidate not in existing_ids:
            return candidate
        idx += 1


def _default_local_destination(config: dict[str, Any], remote_path: str) -> Path:
    root = Path(normalize_path(str(config.get("trained_models_dir", Path.home() / "lerobot" / "trained_models"))))
    name = posixpath.basename(str(remote_path or "").rstrip("/")) or "remote_model"
    return root / name


def _increment_destination(path: Path) -> Path:
    base_name = path.name
    parent = path.parent
    idx = 2
    candidate = parent / f"{base_name}_{idx}"
    while candidate.exists():
        idx += 1
        candidate = parent / f"{base_name}_{idx}"
    return candidate


def _resolve_destination_collision(
    *,
    root: Any,
    target: Path,
) -> Path | None:
    if not target.exists():
        return target

    action = ask_text_dialog_with_actions(
        root=root,
        title="Destination Exists",
        text=(
            f"Local destination already exists:\n{target}\n\n"
            "Choose Increment Name to keep existing files, Overwrite to replace, or Cancel."
        ),
        actions=[
            ("increment", "Increment Name"),
            ("overwrite", "Overwrite"),
        ],
        confirm_label="Continue",
        cancel_label="Cancel",
        wrap_mode="char",
    )
    if action == "cancel":
        return None
    if action == "increment":
        return _increment_destination(target)
    if action == "overwrite":
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return target
    return None


def _remote_parent(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return text
    if text in {"/", "~"}:
        return text
    stripped = text.rstrip("/")
    parent = posixpath.dirname(stripped)
    if not parent:
        return "/"
    return parent


def _remote_join(base: str, child: str) -> str:
    current = str(base or "").strip() or "."
    name = str(child or "").strip()
    if not name:
        return current
    if name.startswith("/"):
        return name
    if current.endswith("/"):
        return current + name
    if current in {"~", "."}:
        return f"{current}/{name}"
    return posixpath.join(current, name)


def setup_training_tab(
    *,
    root: Any,
    training_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    filedialog: Any,
    log_panel: GuiLogPanel,
    messagebox: Any,
    run_process_async: GuiRunProcessAsync,
    set_running: Callable[[bool, str | None, bool], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
    on_model_pulled: Callable[[Path], None],
) -> TrainingTabHandles:
    import tkinter as tk
    from tkinter import simpledialog, ttk

    templates = default_training_templates()
    templates_by_id = {template["id"]: template for template in templates}

    profiles, active_profile_id = load_training_profiles(config)
    state: dict[str, Any] = {
        "profiles": profiles,
        "active_profile_id": active_profile_id,
        "remote_base_path": "",
    }

    ui_font = colors.get("font_ui", "TkDefaultFont")
    panel = colors.get("panel", "#111111")
    surface = colors.get("surface", "#1a1a1a")
    text_col = colors.get("text", "#eeeeee")
    border = colors.get("border", "#2d2d2d")
    style = ttk.Style(root)
    style.configure(
        "Training.Treeview",
        font=(colors.get("font_mono", "TkFixedFont"), 10),
        rowheight=24,
        background=surface,
        foreground=text_col,
        fieldbackground=surface,
        borderwidth=0,
    )
    style.configure(
        "Training.Treeview.Heading",
        font=(ui_font, 10, "bold"),
        background=panel,
        foreground=colors.get("accent", "#f0a500"),
    )
    style.map(
        "Training.Treeview",
        background=[("selected", colors.get("accent", "#f0a500"))],
        foreground=[("selected", "#000000")],
    )

    def all_profiles() -> list[TrainingProfile]:
        return list(state["profiles"])

    def active_profile() -> TrainingProfile | None:
        active = str(state.get("active_profile_id") or "")
        for profile in all_profiles():
            if profile.id == active:
                return profile
        return all_profiles()[0] if all_profiles() else None

    def persist_profiles() -> None:
        save_training_profiles(config, all_profiles(), str(state.get("active_profile_id") or ""))
        save_config(config, quiet=True)

    container = ttk.Frame(training_tab, style="Panel.TFrame")
    container.pack(fill="both", expand=True)
    container.columnconfigure(0, weight=1)

    # ── Profile management ──────────────────────────────────────────────────
    profile_section = ttk.LabelFrame(container, text="Training Profiles", style="Section.TLabelframe", padding=10)
    profile_section.grid(row=0, column=0, sticky="ew")
    profile_section.columnconfigure(1, weight=1)

    profile_label_to_id: dict[str, str] = {}
    profile_choice_var = tk.StringVar(value="")
    profile_name_var = tk.StringVar(value="")
    profile_id_var = tk.StringVar(value="")
    profile_host_var = tk.StringVar(value="")
    profile_port_var = tk.StringVar(value="22")
    profile_user_var = tk.StringVar(value="")
    profile_auth_mode_var = tk.StringVar(value="password")
    profile_identity_var = tk.StringVar(value="")
    profile_remote_models_root_var = tk.StringVar(value="")
    profile_remote_project_root_var = tk.StringVar(value="")
    profile_env_activate_var = tk.StringVar(value="")
    profile_tmux_var = tk.StringVar(value="")
    profile_srun_var = tk.StringVar(value="")
    profile_password_status_var = tk.StringVar(value="")

    ttk.Label(profile_section, text="Active profile", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
    profile_combo = ttk.Combobox(
        profile_section,
        textvariable=profile_choice_var,
        values=[],
        state="readonly",
        style="Dark.TCombobox",
        width=42,
    )
    profile_combo.grid(row=0, column=1, sticky="ew")

    profile_button_row = ttk.Frame(profile_section, style="Panel.TFrame")
    profile_button_row.grid(row=0, column=2, sticky="w", padx=(8, 0))
    add_profile_button = ttk.Button(profile_button_row, text="New")
    add_profile_button.pack(side="left")
    delete_profile_button = ttk.Button(profile_button_row, text="Delete")
    delete_profile_button.pack(side="left", padx=(6, 0))
    save_profile_button = ttk.Button(profile_button_row, text="Save Profile")
    save_profile_button.pack(side="left", padx=(6, 0))

    profile_fields = ttk.Frame(profile_section, style="Panel.TFrame")
    profile_fields.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    profile_fields.columnconfigure(1, weight=1)
    profile_fields.columnconfigure(3, weight=1)

    def add_profile_field(row: int, column: int, label: str, variable: Any, width: int = 28) -> None:
        ttk.Label(profile_fields, text=label, style="Field.TLabel").grid(row=row, column=column * 2, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(profile_fields, textvariable=variable, width=width).grid(row=row, column=column * 2 + 1, sticky="ew", pady=3, padx=(0, 12 if column == 0 else 0))

    add_profile_field(0, 0, "Name", profile_name_var)
    add_profile_field(0, 1, "ID", profile_id_var)
    add_profile_field(1, 0, "Host", profile_host_var)
    add_profile_field(1, 1, "Port", profile_port_var)
    add_profile_field(2, 0, "Username", profile_user_var)
    ttk.Label(profile_fields, text="Auth mode", style="Field.TLabel").grid(row=2, column=2, sticky="w", padx=(0, 6), pady=3)
    profile_auth_combo = ttk.Combobox(
        profile_fields,
        textvariable=profile_auth_mode_var,
        values=["password", "ssh_key"],
        state="readonly",
        style="Dark.TCombobox",
        width=24,
    )
    profile_auth_combo.grid(row=2, column=3, sticky="ew", pady=3)
    add_profile_field(3, 0, "SSH key file", profile_identity_var)
    add_profile_field(3, 1, "Remote models root", profile_remote_models_root_var)
    add_profile_field(4, 0, "Remote project root", profile_remote_project_root_var)
    add_profile_field(4, 1, "Env activate cmd", profile_env_activate_var)
    add_profile_field(5, 0, "Default tmux session", profile_tmux_var)
    add_profile_field(5, 1, "Default srun prefix", profile_srun_var)

    # ── Connection tools ────────────────────────────────────────────────────
    connection_section = ttk.LabelFrame(container, text="Connection Tools", style="Section.TLabelframe", padding=10)
    connection_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    connection_section.columnconfigure(1, weight=1)

    connection_buttons = ttk.Frame(connection_section, style="Panel.TFrame")
    connection_buttons.grid(row=0, column=0, sticky="w")
    test_connection_button = ttk.Button(connection_buttons, text="Test Connection")
    test_connection_button.pack(side="left")
    set_password_button = ttk.Button(connection_buttons, text="Set / Update Password")
    set_password_button.pack(side="left", padx=(8, 0))
    clear_password_button = ttk.Button(connection_buttons, text="Clear Password")
    clear_password_button.pack(side="left", padx=(8, 0))

    ttk.Label(connection_section, textvariable=profile_password_status_var, style="Muted.TLabel").grid(
        row=0, column=1, sticky="w", padx=(12, 0),
    )

    # ── Remote browser ──────────────────────────────────────────────────────
    remote_section = ttk.LabelFrame(container, text="Remote Model Browser", style="Section.TLabelframe", padding=10)
    remote_section.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
    remote_section.columnconfigure(0, weight=1)
    remote_section.rowconfigure(2, weight=1)
    container.rowconfigure(2, weight=1)

    remote_path_var = tk.StringVar(value=str(config.get("training_last_remote_path", "")).strip())
    remote_status_var = tk.StringVar(value="Select a profile and refresh.")

    remote_path_row = ttk.Frame(remote_section, style="Panel.TFrame")
    remote_path_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    remote_path_row.columnconfigure(1, weight=1)
    ttk.Label(remote_path_row, text="Current remote path", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
    remote_path_entry = ttk.Entry(remote_path_row, textvariable=remote_path_var, width=56)
    remote_path_entry.grid(row=0, column=1, sticky="ew")
    remote_home_button = ttk.Button(remote_path_row, text="Home")
    remote_home_button.grid(row=0, column=2, padx=(6, 0))
    remote_up_button = ttk.Button(remote_path_row, text="Up")
    remote_up_button.grid(row=0, column=3, padx=(6, 0))
    remote_refresh_button = ttk.Button(remote_path_row, text="Refresh")
    remote_refresh_button.grid(row=0, column=4, padx=(6, 0))

    remote_tree_frame = ttk.Frame(remote_section, style="Panel.TFrame")
    remote_tree_frame.grid(row=1, column=0, sticky="nsew")
    remote_tree_frame.columnconfigure(0, weight=1)
    remote_tree_frame.rowconfigure(0, weight=1)
    remote_section.rowconfigure(1, weight=1)

    remote_tree = ttk.Treeview(
        remote_tree_frame,
        columns=("name",),
        show="headings",
        height=8,
        style="Training.Treeview",
    )
    remote_tree.heading("name", text="Remote Directories")
    remote_tree.column("name", width=620, anchor="w")
    remote_tree.grid(row=0, column=0, sticky="nsew")
    remote_tree_scroll = ttk.Scrollbar(
        remote_tree_frame,
        orient="vertical",
        command=remote_tree.yview,
        style="Dark.Vertical.TScrollbar",
    )
    remote_tree_scroll.grid(row=0, column=1, sticky="ns")
    remote_tree.configure(yscrollcommand=remote_tree_scroll.set)

    ttk.Label(remote_section, textvariable=remote_status_var, style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(6, 0))

    # ── Pull panel ──────────────────────────────────────────────────────────
    pull_section = ttk.LabelFrame(container, text="Pull Model / Checkpoint", style="Section.TLabelframe", padding=10)
    pull_section.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    pull_section.columnconfigure(1, weight=1)

    pull_remote_var = tk.StringVar(value=remote_path_var.get().strip())
    pull_local_var = tk.StringVar(value=str(config.get("training_last_local_destination", "")).strip())
    pull_prefer_rsync_var = tk.BooleanVar(value=True)
    pull_status_var = tk.StringVar(value="")

    ttk.Label(pull_section, text="Remote folder", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(pull_section, textvariable=pull_remote_var, width=56).grid(row=0, column=1, sticky="ew", pady=3)
    ttk.Button(pull_section, text="Use Current Path", command=lambda: pull_remote_var.set(remote_path_var.get().strip())).grid(
        row=0, column=2, sticky="w", padx=(6, 0), pady=3,
    )

    ttk.Label(pull_section, text="Local destination", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(pull_section, textvariable=pull_local_var, width=56).grid(row=1, column=1, sticky="ew", pady=3)

    def choose_pull_destination() -> None:
        selected = ask_directory_dialog(
            root=root,
            filedialog=filedialog,
            initial_dir=str(Path(pull_local_var.get() or config.get("trained_models_dir", str(Path.home()))).parent),
            title="Select destination parent folder",
        )
        if not selected:
            return
        remote_basename = posixpath.basename(str(pull_remote_var.get() or "").rstrip("/")) or "remote_model"
        pull_local_var.set(str(Path(selected) / remote_basename))

    ttk.Button(pull_section, text="Browse", command=choose_pull_destination).grid(row=1, column=2, sticky="w", padx=(6, 0), pady=3)
    ttk.Checkbutton(
        pull_section,
        text="Prefer rsync (fallback to sftp)",
        variable=pull_prefer_rsync_var,
    ).grid(row=2, column=1, sticky="w", pady=(4, 0))

    pull_button_row = ttk.Frame(pull_section, style="Panel.TFrame")
    pull_button_row.grid(row=3, column=1, sticky="w", pady=(8, 0))
    preview_pull_button = ttk.Button(pull_button_row, text="Preview Pull Command")
    preview_pull_button.pack(side="left")
    run_pull_button = ttk.Button(pull_button_row, text="Run Pull", style="Accent.TButton")
    run_pull_button.pack(side="left", padx=(8, 0))

    ttk.Label(pull_section, textvariable=pull_status_var, style="Muted.TLabel").grid(row=4, column=1, sticky="w", pady=(6, 0))

    # ── Launch panel ────────────────────────────────────────────────────────
    launch_section = ttk.LabelFrame(container, text="Remote Launch", style="Section.TLabelframe", padding=10)
    launch_section.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    launch_section.columnconfigure(1, weight=1)

    launch_template_var = tk.StringVar(value=templates[0]["id"])
    launch_template_desc_var = tk.StringVar(value=templates[0]["description"])
    launch_train_command_var = tk.StringVar(value="python -m lerobot.scripts.train --help")
    launch_tmux_session_var = tk.StringVar(value="")
    launch_srun_prefix_var = tk.StringVar(value="")
    launch_project_root_var = tk.StringVar(value="")
    launch_env_activate_var = tk.StringVar(value="")
    launch_extra_vars_var = tk.StringVar(value="")
    launch_preview_var = tk.StringVar(value="")

    ttk.Label(launch_section, text="Template", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    launch_template_combo = ttk.Combobox(
        launch_section,
        textvariable=launch_template_var,
        values=[template["id"] for template in templates],
        state="readonly",
        style="Dark.TCombobox",
        width=22,
    )
    launch_template_combo.grid(row=0, column=1, sticky="w", pady=3)
    ttk.Label(launch_section, textvariable=launch_template_desc_var, style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=3)

    ttk.Label(launch_section, text="Train command", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_train_command_var, width=56).grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)

    ttk.Label(launch_section, text="tmux session", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_tmux_session_var, width=20).grid(row=2, column=1, sticky="w", pady=3)

    ttk.Label(launch_section, text="srun prefix", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_srun_prefix_var, width=56).grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)

    ttk.Label(launch_section, text="Remote project root", style="Field.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_project_root_var, width=56).grid(row=4, column=1, columnspan=2, sticky="ew", pady=3)

    ttk.Label(launch_section, text="Env activate cmd", style="Field.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_env_activate_var, width=56).grid(row=5, column=1, columnspan=2, sticky="ew", pady=3)

    ttk.Label(launch_section, text="Extra vars JSON", style="Field.TLabel").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(launch_section, textvariable=launch_extra_vars_var, width=56).grid(row=6, column=1, columnspan=2, sticky="ew", pady=3)

    custom_template_label = ttk.Label(launch_section, text="Custom template text", style="Field.TLabel")
    custom_template_text = tk.Text(
        launch_section,
        wrap="word",
        height=4,
        bg=surface,
        fg=text_col,
        insertbackground=text_col,
        relief="flat",
        highlightthickness=1,
        highlightbackground=border,
        font=(ui_font, 10),
        padx=6,
        pady=6,
    )

    launch_button_row = ttk.Frame(launch_section, style="Panel.TFrame")
    launch_button_row.grid(row=9, column=1, sticky="w", pady=(8, 0))
    preview_launch_button = ttk.Button(launch_button_row, text="Preview Launch Command")
    preview_launch_button.pack(side="left")
    run_launch_button = ttk.Button(launch_button_row, text="Run Launch", style="Accent.TButton")
    run_launch_button.pack(side="left", padx=(8, 0))

    ttk.Label(launch_section, textvariable=launch_preview_var, style="Muted.TLabel").grid(
        row=10,
        column=1,
        columnspan=2,
        sticky="w",
        pady=(6, 0),
    )

    def refresh_profile_combo() -> None:
        labels: list[str] = []
        profile_label_to_id.clear()
        for profile in all_profiles():
            label = _profile_label(profile)
            labels.append(label)
            profile_label_to_id[label] = profile.id
        profile_combo.configure(values=labels)
        current_profile = active_profile()
        if current_profile is not None:
            profile_choice_var.set(_profile_label(current_profile))
        elif labels:
            profile_choice_var.set(labels[0])
            state["active_profile_id"] = profile_label_to_id[labels[0]]
        else:
            profile_choice_var.set("")

    def update_password_status() -> None:
        profile = active_profile()
        if profile is None:
            profile_password_status_var.set("No active profile.")
            set_password_button.configure(state="disabled")
            clear_password_button.configure(state="disabled")
            return
        if profile.auth_mode == "ssh_key":
            profile_password_status_var.set("SSH key auth mode. Password is not required.")
            set_password_button.configure(state="disabled")
            clear_password_button.configure(state="disabled")
            return
        set_password_button.configure(state="normal")
        clear_password_button.configure(state="normal")
        if not is_secret_tool_available():
            profile_password_status_var.set("secret-tool unavailable (install libsecret-tools).")
            return
        has_password, error = has_ssh_password(profile)
        if error:
            profile_password_status_var.set(f"Password status: {error}")
            return
        profile_password_status_var.set("Password stored securely." if has_password else "No stored password.")

    def populate_profile_form(profile: TrainingProfile | None) -> None:
        if profile is None:
            profile_name_var.set("")
            profile_id_var.set("")
            profile_host_var.set("")
            profile_port_var.set("22")
            profile_user_var.set("")
            profile_auth_mode_var.set("password")
            profile_identity_var.set("")
            profile_remote_models_root_var.set("")
            profile_remote_project_root_var.set("")
            profile_env_activate_var.set("")
            profile_tmux_var.set("")
            profile_srun_var.set("")
            return

        profile_name_var.set(profile.name)
        profile_id_var.set(profile.id)
        profile_host_var.set(profile.host)
        profile_port_var.set(str(profile.port))
        profile_user_var.set(profile.username)
        profile_auth_mode_var.set(profile.auth_mode)
        profile_identity_var.set(profile.identity_file)
        profile_remote_models_root_var.set(profile.remote_models_root)
        profile_remote_project_root_var.set(profile.remote_project_root)
        profile_env_activate_var.set(profile.env_activate_cmd)
        profile_tmux_var.set(profile.default_tmux_session)
        profile_srun_var.set(profile.default_srun_prefix)

        current_remote = remote_path_var.get().strip()
        if not current_remote:
            remote_path_var.set(profile.remote_models_root)
        if not pull_remote_var.get().strip():
            pull_remote_var.set(remote_path_var.get().strip())
        if not pull_local_var.get().strip():
            pull_local_var.set(str(_default_local_destination(config, pull_remote_var.get().strip())))

        launch_tmux_session_var.set(profile.default_tmux_session)
        launch_srun_prefix_var.set(profile.default_srun_prefix)
        launch_project_root_var.set(profile.remote_project_root)
        launch_env_activate_var.set(profile.env_activate_cmd)

    def save_profile_from_form() -> bool:
        original = active_profile()
        if original is None:
            messagebox.showerror("Training Profiles", "No active profile selected.")
            return False

        profile_id = str(profile_id_var.get()).strip()
        if not profile_id:
            messagebox.showerror("Training Profiles", "Profile ID is required.")
            return False
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in profile_id):
            messagebox.showerror("Training Profiles", "Profile ID can only contain letters, numbers, '_' and '-'.")
            return False

        try:
            port = int(str(profile_port_var.get()).strip() or "22")
        except ValueError:
            messagebox.showerror("Training Profiles", "Port must be an integer.")
            return False
        if port <= 0 or port > 65535:
            messagebox.showerror("Training Profiles", "Port must be between 1 and 65535.")
            return False

        auth_mode = str(profile_auth_mode_var.get()).strip() or "password"
        if auth_mode not in {"password", "ssh_key"}:
            messagebox.showerror("Training Profiles", "Auth mode must be 'password' or 'ssh_key'.")
            return False

        updated = TrainingProfile(
            id=profile_id,
            name=str(profile_name_var.get()).strip() or profile_id,
            host=str(profile_host_var.get()).strip(),
            port=port,
            username=str(profile_user_var.get()).strip(),
            auth_mode=auth_mode,
            identity_file=str(profile_identity_var.get()).strip(),
            remote_models_root=str(profile_remote_models_root_var.get()).strip() or "~/lerobot/trained_models",
            remote_project_root=str(profile_remote_project_root_var.get()).strip() or "~/lerobot",
            env_activate_cmd=str(profile_env_activate_var.get()).strip() or "source ~/lerobot/lerobot_env/bin/activate",
            default_tmux_session=str(profile_tmux_var.get()).strip() or "lerobot_train",
            default_srun_prefix=str(profile_srun_var.get()).strip() or "srun --gres=gpu:1 --pty bash -lc",
        )

        others = [profile for profile in all_profiles() if profile.id != original.id]
        if any(profile.id == updated.id for profile in others):
            messagebox.showerror("Training Profiles", f"Profile ID '{updated.id}' already exists.")
            return False
        state["profiles"] = others + [updated]
        state["profiles"].sort(key=lambda item: item.name.lower())
        state["active_profile_id"] = updated.id
        persist_profiles()
        refresh_profile_combo()
        populate_profile_form(updated)
        update_password_status()
        log_panel.append_log(f"Saved training profile: {updated.name} ({updated.id})")
        return True

    def select_profile_by_label(label: str) -> None:
        profile_id = profile_label_to_id.get(label)
        if not profile_id:
            return
        state["active_profile_id"] = profile_id
        persist_profiles()
        populate_profile_form(active_profile())
        update_password_status()

    def add_profile() -> None:
        existing_ids = {profile.id for profile in all_profiles()}
        new_id = _next_profile_id(existing_ids)
        template = active_profile()
        if template is None:
            messagebox.showerror("Training Profiles", "Cannot create profile without an existing template.")
            return
        created = TrainingProfile(
            id=new_id,
            name=f"Profile {new_id.split('_')[-1]}",
            host=template.host,
            port=template.port,
            username=template.username,
            auth_mode=template.auth_mode,
            identity_file=template.identity_file,
            remote_models_root=template.remote_models_root,
            remote_project_root=template.remote_project_root,
            env_activate_cmd=template.env_activate_cmd,
            default_tmux_session=template.default_tmux_session,
            default_srun_prefix=template.default_srun_prefix,
        )
        state["profiles"] = all_profiles() + [created]
        state["active_profile_id"] = created.id
        persist_profiles()
        refresh_profile_combo()
        populate_profile_form(created)
        update_password_status()

    def delete_profile() -> None:
        profile = active_profile()
        if profile is None:
            return
        if len(all_profiles()) <= 1:
            messagebox.showerror("Training Profiles", "At least one training profile is required.")
            return
        if not messagebox.askyesno("Delete Profile", f"Delete training profile '{profile.name}'?"):
            return
        state["profiles"] = [item for item in all_profiles() if item.id != profile.id]
        state["active_profile_id"] = state["profiles"][0].id
        persist_profiles()
        refresh_profile_combo()
        populate_profile_form(active_profile())
        update_password_status()

    def _ensure_profile_ready() -> TrainingProfile | None:
        profile = active_profile()
        if profile is None:
            messagebox.showerror("Training", "No active training profile.")
            return None
        if not profile.host or not profile.username:
            messagebox.showerror("Training", "Profile host and username are required.")
            return None
        return profile

    def test_connection() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        trusted, trust_detail = ensure_host_trusted(profile, messagebox)
        if not trusted:
            messagebox.showerror("Training Connection", trust_detail)
            return

        cmd = build_remote_launch_command(profile, "echo __TRAINING_CONNECTION_OK__")
        last_command_state["value"] = format_command(cmd)
        log_panel.append_log("Testing SSH training connection...")
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            messagebox.showerror("Training Connection", f"Connection test failed to run: {exc}")
            return

        output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode == 0 and "__TRAINING_CONNECTION_OK__" in output:
            messagebox.showinfo("Training Connection", f"Connection successful.\n{profile.username}@{profile.host}:{profile.port}")
            log_panel.append_log(f"Training connection successful: {profile.username}@{profile.host}:{profile.port}")
            return
        messagebox.showerror(
            "Training Connection",
            "Connection test failed.\n\n"
            f"Exit code: {result.returncode}\n"
            f"Detail: {output[:1000] or '(no output)'}",
        )

    def set_or_update_password() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        if profile.auth_mode != "password":
            messagebox.showinfo("SSH Password", "This profile uses SSH key auth mode.")
            return
        password = simpledialog.askstring(
            "SSH Password",
            f"Enter SSH password for {profile.username}@{profile.host}:{profile.port}",
            parent=root,
            show="*",
        )
        if password is None:
            return
        ok, detail = save_ssh_password(profile, password)
        if ok:
            messagebox.showinfo("SSH Password", detail)
        else:
            messagebox.showerror("SSH Password", detail)
        update_password_status()

    def clear_password() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        if profile.auth_mode != "password":
            messagebox.showinfo("SSH Password", "This profile uses SSH key auth mode.")
            return
        ok, detail = delete_ssh_password(profile)
        if ok:
            messagebox.showinfo("SSH Password", detail)
        else:
            messagebox.showerror("SSH Password", detail)
        update_password_status()

    def _refresh_remote_list(target_path: str | None = None) -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        desired = str(target_path if target_path is not None else remote_path_var.get()).strip() or profile.remote_models_root
        trusted, trust_detail = ensure_host_trusted(profile, messagebox)
        if not trusted:
            remote_status_var.set(trust_detail)
            return

        children, error = list_remote_dirs(profile, desired)
        for iid in list(remote_tree.get_children()):
            remote_tree.delete(iid)
        if error is not None:
            remote_status_var.set(error)
            return

        state["remote_base_path"] = desired
        remote_path_var.set(desired)
        pull_remote_var.set(desired)
        if not pull_local_var.get().strip():
            pull_local_var.set(str(_default_local_destination(config, desired)))

        for child in children or []:
            remote_tree.insert("", "end", values=(child,))
        remote_status_var.set(f"Listed {len(children or [])} directories under {desired}")
        config["training_last_remote_path"] = desired
        save_config(config, quiet=True)

    def open_selected_remote_dir(_: Any = None) -> None:
        selected = remote_tree.selection()
        if not selected:
            return
        child_name = str(remote_tree.item(selected[0], "values")[0]).strip()
        if not child_name:
            return
        _refresh_remote_list(_remote_join(str(state.get("remote_base_path") or remote_path_var.get().strip()), child_name))

    def go_remote_home() -> None:
        profile = active_profile()
        if profile is None:
            return
        _refresh_remote_list(profile.remote_models_root)

    def go_remote_up() -> None:
        current = remote_path_var.get().strip()
        _refresh_remote_list(_remote_parent(current))

    def _pull_context(profile: TrainingProfile, remote_path: str, local_destination: Path, mode: str, template_name: str | None = None) -> dict[str, Any]:
        context = {
            "training_profile": profile.name,
            "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
            "remote_path": remote_path,
            "local_path": str(local_destination),
            "training_transport": mode,
        }
        if template_name:
            context["template_name"] = template_name
        return context

    def preview_pull_command() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        remote_path = str(pull_remote_var.get()).strip()
        local_raw = str(pull_local_var.get()).strip()
        if not remote_path:
            messagebox.showerror("Training Pull", "Remote path is required.")
            return
        if not local_raw:
            local_raw = str(_default_local_destination(config, remote_path))
            pull_local_var.set(local_raw)
        local_destination = Path(normalize_path(local_raw))

        cmd = build_pull_command(profile, remote_path, local_destination, prefer_rsync=bool(pull_prefer_rsync_var.get()))
        last_command_state["value"] = format_command(cmd)
        show_text_dialog(
            root=root,
            title="Training Pull Command",
            text=format_command_for_dialog(cmd),
            copy_text=last_command_state["value"],
            wrap_mode="word",
        )

    def run_pull() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return

        remote_path = str(pull_remote_var.get()).strip()
        if not remote_path:
            messagebox.showerror("Training Pull", "Remote path is required.")
            return

        local_destination_raw = str(pull_local_var.get()).strip()
        if not local_destination_raw:
            local_destination_raw = str(_default_local_destination(config, remote_path))
            pull_local_var.set(local_destination_raw)
        local_destination = Path(normalize_path(local_destination_raw))

        try:
            local_destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Training Pull", f"Unable to create destination parent folder: {exc}")
            return

        resolved_destination = _resolve_destination_collision(root=root, target=local_destination)
        if resolved_destination is None:
            return
        local_destination = resolved_destination
        pull_local_var.set(str(local_destination))

        trusted, trust_detail = ensure_host_trusted(profile, messagebox)
        if not trusted:
            messagebox.showerror("Training Pull", trust_detail)
            return

        preflight_checks = run_preflight_for_training(
            profile=profile,
            local_destination=local_destination,
            remote_path=remote_path,
            use_rsync=bool(pull_prefer_rsync_var.get()),
        )
        if not confirm_preflight_in_gui("Training Pull Preflight", preflight_checks):
            return

        primary_cmd = build_pull_command(profile, remote_path, local_destination, prefer_rsync=bool(pull_prefer_rsync_var.get()))
        primary_is_rsync = command_uses_binary(primary_cmd, "rsync")

        def on_fallback_complete(return_code: int, was_canceled: bool, batch_file: Path) -> None:
            try:
                batch_file.unlink()
            except OSError:
                pass
            if was_canceled:
                set_running(False, "Training pull canceled.")
                messagebox.showinfo("Canceled", "Training pull was canceled.")
                return
            if return_code != 0:
                set_running(False, "Training pull failed.", True)
                messagebox.showerror("Training Pull Failed", f"Training pull failed with exit code {return_code}.")
                return
            set_running(False, "Training pull completed.")
            config["training_last_remote_path"] = remote_path
            config["training_last_local_destination"] = str(local_destination)
            save_config(config, quiet=True)
            on_model_pulled(local_destination)
            pull_status_var.set(f"Pulled via sftp: {local_destination}")
            messagebox.showinfo(
                "Training Pull Complete",
                f"Pulled model successfully.\n\nRemote: {remote_path}\nLocal: {local_destination}\nMode: sftp fallback",
            )

        def on_primary_complete(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                set_running(False, "Training pull canceled.")
                messagebox.showinfo("Canceled", "Training pull was canceled.")
                return
            if return_code == 0:
                set_running(False, "Training pull completed.")
                config["training_last_remote_path"] = remote_path
                config["training_last_local_destination"] = str(local_destination)
                save_config(config, quiet=True)
                on_model_pulled(local_destination)
                mode_name = "rsync" if primary_is_rsync else "sftp"
                pull_status_var.set(f"Pulled via {mode_name}: {local_destination}")
                messagebox.showinfo(
                    "Training Pull Complete",
                    f"Pulled model successfully.\n\nRemote: {remote_path}\nLocal: {local_destination}\nMode: {mode_name}",
                )
                return

            if primary_is_rsync:
                set_running(False, "Rsync failed, starting sftp fallback...", True)
                pull_status_var.set("Rsync failed. Starting sftp fallback...")
                fallback_cmd, batch_file = build_sftp_pull_command(profile, remote_path, local_destination)
                root.after(
                    0,
                    lambda: run_process_async(
                        fallback_cmd,
                        None,
                        lambda rc, canceled: on_fallback_complete(rc, canceled, batch_file),
                        None,
                        None,
                        "train_sync",
                        preflight_checks,
                        _pull_context(profile, remote_path, local_destination, "sftp"),
                    ),
                )
                return

            set_running(False, "Training pull failed.", True)
            messagebox.showerror("Training Pull Failed", f"Training pull failed with exit code {return_code}.")

        last_command_state["value"] = format_command(primary_cmd)
        run_process_async(
            primary_cmd,
            None,
            on_primary_complete,
            None,
            None,
            "train_sync",
            preflight_checks,
            _pull_context(profile, remote_path, local_destination, "rsync" if primary_is_rsync else "sftp"),
        )

    def current_template_text() -> str:
        template_id = str(launch_template_var.get()).strip()
        if template_id == "custom":
            return custom_template_text.get("1.0", "end").strip()
        template = templates_by_id.get(template_id)
        return str(template.get("template", "")).strip() if template else ""

    def build_template_variables(profile: TrainingProfile) -> tuple[dict[str, str] | None, str | None]:
        variables = {
            "env_activate_cmd": str(launch_env_activate_var.get()).strip(),
            "remote_project_root": str(launch_project_root_var.get()).strip(),
            "tmux_session": str(launch_tmux_session_var.get()).strip(),
            "srun_prefix": str(launch_srun_prefix_var.get()).strip(),
            "train_command": str(launch_train_command_var.get()).strip(),
        }
        extra_text = str(launch_extra_vars_var.get()).strip()
        if extra_text:
            try:
                parsed = json.loads(extra_text)
            except json.JSONDecodeError as exc:
                return None, f"Extra vars JSON is invalid: {exc}"
            if not isinstance(parsed, dict):
                return None, "Extra vars JSON must be an object."
            for key, value in parsed.items():
                variables[str(key)] = str(value)

        if not variables["tmux_session"]:
            variables["tmux_session"] = profile.default_tmux_session
        if not variables["srun_prefix"]:
            variables["srun_prefix"] = profile.default_srun_prefix
        if not variables["remote_project_root"]:
            variables["remote_project_root"] = profile.remote_project_root
        if not variables["env_activate_cmd"]:
            variables["env_activate_cmd"] = profile.env_activate_cmd
        return variables, None

    def render_launch_command() -> tuple[list[str] | None, str | None, str | None, dict[str, str] | None]:
        profile = _ensure_profile_ready()
        if profile is None:
            return None, None, "No active profile.", None
        template_text = current_template_text()
        variables, variable_error = build_template_variables(profile)
        if variable_error or variables is None:
            return None, None, variable_error or "Invalid template variables.", None

        rendered, render_error = render_template(template_text, variables)
        if render_error or rendered is None:
            return None, None, render_error or "Unable to render template.", None

        command = build_remote_launch_command(profile, rendered)
        return command, rendered, None, variables

    def preview_launch() -> None:
        command, rendered_remote, error, _ = render_launch_command()
        if error or command is None or rendered_remote is None:
            messagebox.showerror("Training Launch", error or "Unable to build launch command.")
            return

        last_command_state["value"] = format_command(command)
        launch_preview_var.set(rendered_remote)
        show_text_dialog(
            root=root,
            title="Training Launch Command",
            text=format_command_for_dialog(command),
            copy_text=last_command_state["value"],
            wrap_mode="word",
        )

    def run_launch() -> None:
        profile = _ensure_profile_ready()
        if profile is None:
            return
        command, rendered_remote, error, variables = render_launch_command()
        if error or command is None or rendered_remote is None or variables is None:
            messagebox.showerror("Training Launch", error or "Unable to build launch command.")
            return

        trusted, trust_detail = ensure_host_trusted(profile, messagebox)
        if not trusted:
            messagebox.showerror("Training Launch", trust_detail)
            return

        template_id = str(launch_template_var.get()).strip()
        template_name = template_id
        template = templates_by_id.get(template_id)
        if template is not None:
            template_name = str(template.get("name", template_id))

        preflight_checks = run_preflight_for_training(
            profile=profile,
            rendered_remote_command=rendered_remote,
            template_text=current_template_text(),
            template_variables=variables,
            use_rsync=False,
        )
        if not confirm_preflight_in_gui("Training Launch Preflight", preflight_checks):
            return

        if not ask_text_dialog(
            root=root,
            title="Confirm Training Launch",
            text=(
                "Review the remote launch command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(command)
            ),
            copy_text=format_command(command),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        last_command_state["value"] = format_command(command)
        launch_preview_var.set(rendered_remote)

        def on_launch_complete(return_code: int, was_canceled: bool) -> None:
            if was_canceled:
                set_running(False, "Training launch canceled.")
                messagebox.showinfo("Canceled", "Training launch was canceled.")
                return
            if return_code != 0:
                set_running(False, "Training launch failed.", True)
                messagebox.showerror("Training Launch Failed", f"Training launch failed with exit code {return_code}.")
                return
            set_running(False, "Training launch completed.")
            messagebox.showinfo(
                "Training Launch Complete",
                f"Training launch completed.\nProfile: {profile.name}\nTemplate: {template_name}",
            )

        run_process_async(
            command,
            None,
            on_launch_complete,
            None,
            None,
            "train_launch",
            preflight_checks,
            {
                "training_profile": profile.name,
                "remote_host": f"{profile.username}@{profile.host}:{profile.port}",
                "remote_path": profile.remote_project_root,
                "local_path": "",
                "template_name": template_name,
            },
        )

    def refresh_template_controls(*_: Any) -> None:
        template_id = str(launch_template_var.get()).strip()
        template = templates_by_id.get(template_id)
        if template is not None:
            launch_template_desc_var.set(str(template.get("description", "")))
        else:
            launch_template_desc_var.set("")
        if template_id == "custom":
            custom_template_label.grid(row=7, column=0, sticky="nw", padx=(0, 6), pady=3)
            custom_template_text.grid(row=7, column=1, columnspan=2, sticky="ew", pady=3)
            if not custom_template_text.get("1.0", "end").strip():
                custom_template_text.insert("1.0", "{env_activate_cmd} && cd {remote_project_root} && {train_command}")
        else:
            custom_template_label.grid_remove()
            custom_template_text.grid_remove()

    def refresh() -> None:
        refresh_profile_combo()
        populate_profile_form(active_profile())
        update_password_status()
        if not remote_path_var.get().strip():
            profile = active_profile()
            if profile is not None:
                remote_path_var.set(profile.remote_models_root)
        if not pull_remote_var.get().strip():
            pull_remote_var.set(remote_path_var.get().strip())
        if not pull_local_var.get().strip():
            pull_local_var.set(str(_default_local_destination(config, pull_remote_var.get().strip())))
        refresh_template_controls()

    def on_remote_double_click(_: Any) -> None:
        open_selected_remote_dir()

    def on_profile_selected(_: Any = None) -> None:
        label = profile_choice_var.get().strip()
        if not label:
            return
        select_profile_by_label(label)

    profile_combo.bind("<<ComboboxSelected>>", on_profile_selected)
    add_profile_button.configure(command=add_profile)
    delete_profile_button.configure(command=delete_profile)
    save_profile_button.configure(command=save_profile_from_form)

    test_connection_button.configure(command=test_connection)
    set_password_button.configure(command=set_or_update_password)
    clear_password_button.configure(command=clear_password)

    remote_home_button.configure(command=go_remote_home)
    remote_up_button.configure(command=go_remote_up)
    remote_refresh_button.configure(command=lambda: _refresh_remote_list(remote_path_var.get().strip()))
    remote_tree.bind("<Double-1>", on_remote_double_click)

    preview_pull_button.configure(command=preview_pull_command)
    run_pull_button.configure(command=run_pull)
    preview_launch_button.configure(command=preview_launch)
    run_launch_button.configure(command=run_launch)
    launch_template_combo.bind("<<ComboboxSelected>>", refresh_template_controls)

    def sync_pull_local_from_remote(*_: Any) -> None:
        remote = pull_remote_var.get().strip()
        if not remote:
            return
        current = pull_local_var.get().strip()
        expected = str(_default_local_destination(config, remote))
        if not current or current == str(_default_local_destination(config, remote_path_var.get().strip())):
            pull_local_var.set(expected)

    pull_remote_var.trace_add("write", sync_pull_local_from_remote)

    refresh()

    action_buttons = [
        add_profile_button,
        delete_profile_button,
        save_profile_button,
        test_connection_button,
        set_password_button,
        clear_password_button,
        remote_refresh_button,
        preview_pull_button,
        run_pull_button,
        preview_launch_button,
        run_launch_button,
    ]
    return TrainingTabHandles(
        action_buttons=action_buttons,
        refresh=refresh,
    )
