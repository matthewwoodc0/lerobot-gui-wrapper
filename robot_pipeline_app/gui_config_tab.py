from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Callable

from .checks import collect_doctor_checks, summarize_checks
from .config_store import default_for_key, save_config
from .constants import CONFIG_FIELDS, DEFAULT_TASK
from .desktop_launcher import install_desktop_launcher
from .gui_dialogs import ask_text_dialog_with_actions
from .gui_forms import coerce_config_from_vars
from .gui_log import GuiLogPanel
from .setup_wizard import (
    build_setup_status_summary,
    build_setup_wizard_commands,
    build_setup_wizard_guide,
    probe_setup_wizard_status,
)

DEFAULT_SETUP_VENV_ACTIVATE_CMD = "source ~/lerobot/lerobot_env/bin/activate"


@dataclass
class ConfigTabHandles:
    action_buttons: list[Any]
    sync_from_config: Callable[[], None]
    apply_theme: Callable[[dict[str, str]], None]


def setup_config_tab(
    *,
    root: Any,
    config_tab: Any,
    config: dict[str, Any],
    choose_folder: Callable[[Any], None],
    log_panel: GuiLogPanel,
    messagebox: Any,
    refresh_header_subtitle: Callable[[], None],
    refresh_record_summary: Callable[[], None],
    refresh_local_models: Callable[[], None],
    record_camera_preview: Any,
    deploy_camera_preview: Any,
    record_dir_var: Any,
    deploy_root_var: Any,
    deploy_eval_episodes_var: Any,
    deploy_eval_duration_var: Any,
    deploy_eval_task_var: Any,
    run_terminal_command: Callable[[str], tuple[bool, str]] | None = None,
    show_terminal: Callable[[], None] | None = None,
) -> ConfigTabHandles:
    import tkinter as tk
    from tkinter import ttk

    config_vars: dict[str, Any] = {}
    path_keys = {"lerobot_dir", "lerobot_venv_dir", "runs_dir", "record_data_dir", "deploy_data_dir", "trained_models_dir"}
    field_lookup = {field["key"]: field for field in CONFIG_FIELDS}
    group_layout = [
        ("Paths", ["lerobot_dir", "lerobot_venv_dir", "runs_dir", "record_data_dir", "deploy_data_dir", "trained_models_dir"]),
        ("Robot Ports", ["follower_port", "leader_port"]),
        (
            "Cameras",
            [
                "camera_laptop_index",
                "camera_phone_index",
                "camera_warmup_s",
                "camera_fps",
            ],
        ),
        (
            "Hugging Face + Defaults",
            [
                "hf_username",
                "eval_num_episodes",
                "eval_duration_s",
                "eval_task",
                "ui_theme_mode",
            ],
        ),
    ]

    def add_config_group(parent: Any, title: str, keys: list[str]) -> None:
        frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        for row, key in enumerate(keys):
            field = field_lookup[key]
            ttk.Label(frame, text=field["prompt"], style="Field.TLabel").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 6),
                pady=4,
            )
            current = config.get(key)
            if current in (None, ""):
                current = default_for_key(key, config)
            value_var = tk.StringVar(value=str(current))
            config_vars[key] = value_var
            ttk.Entry(frame, textvariable=value_var, width=52).grid(row=row, column=1, sticky="ew", pady=4)
            if key in path_keys:
                ttk.Button(frame, text="Browse", command=lambda var=value_var: choose_folder(var)).grid(
                    row=row,
                    column=2,
                    sticky="w",
                    padx=(6, 0),
                    pady=4,
                )

    for group_title, keys in group_layout:
        add_config_group(config_tab, group_title, keys)

    setup_wizard_frame = ttk.LabelFrame(config_tab, text="First-Time Setup Wizard", style="Section.TLabelframe", padding=10)
    setup_wizard_frame.pack(fill="x", pady=(0, 10))
    ttk.Label(
        setup_wizard_frame,
        text=(
            "Checks whether this Python environment can import LeRobot and whether a virtual environment is active.\n"
            "If no venv is active, the popout can run a quick activate command and lets you enter a custom source command."
        ),
        style="Field.TLabel",
        justify="left",
    ).pack(anchor="w")
    setup_status_var = tk.StringVar(value="Run Setup Check to inspect environment readiness.")
    setup_status_label = ttk.Label(setup_wizard_frame, textvariable=setup_status_var, style="Field.TLabel", justify="left")
    setup_status_label.pack(anchor="w", fill="x", pady=(8, 8))
    setup_controls = ttk.Frame(setup_wizard_frame, style="Panel.TFrame")
    setup_controls.pack(fill="x")
    setup_status_state: dict[str, Any] = {"value": None}
    setup_wizard_prompted: dict[str, bool] = {"value": False}

    def _setup_preview_config() -> tuple[dict[str, Any] | None, str | None]:
        parsed_config, error_text = coerce_config_from_vars(config, config_vars, CONFIG_FIELDS)
        if parsed_config is None:
            return None, error_text or "Invalid config values."
        return parsed_config, None

    def _refresh_setup_status() -> Any | None:
        preview, error_text = _setup_preview_config()
        if preview is None:
            setup_status_state["value"] = None
            setup_status_var.set(f"[FAIL] Setup check could not run: {error_text}")
            return None
        status = probe_setup_wizard_status(preview)
        setup_status_state["value"] = status
        setup_status_var.set(build_setup_status_summary(status))
        return status

    def _copy_setup_commands(status: Any) -> None:
        commands = build_setup_wizard_commands(status)
        root.clipboard_clear()
        root.clipboard_append(commands)
        log_panel.append_log("Copied LeRobot setup commands to clipboard.")

    def _normalize_activate_command(raw_command: str) -> str:
        value = str(raw_command).strip()
        if not value:
            return ""
        if value.startswith("source ") or value.startswith(". "):
            return value
        if " " not in value and ("/" in value or value.startswith("~")):
            return f"source {value}"
        return value

    def _venv_activate_cmd_from_dir(venv_dir_raw: str) -> str:
        clean = str(venv_dir_raw).strip()
        if not clean:
            return DEFAULT_SETUP_VENV_ACTIVATE_CMD
        activate_path = Path(clean).expanduser() / "bin" / "activate"
        return f"source {shlex.quote(str(activate_path))}"

    def _venv_dir_from_activate_command(command_text: str) -> Path | None:
        normalized = _normalize_activate_command(command_text)
        if not normalized:
            return None
        try:
            parts = shlex.split(normalized)
        except ValueError:
            return None
        if len(parts) < 2 or parts[0] not in {"source", "."}:
            return None
        activate_path = Path(parts[1]).expanduser()
        if activate_path.name != "activate" or activate_path.parent.name != "bin":
            return None
        return activate_path.parent.parent

    def _default_activate_command_from_ui() -> str:
        venv_dir = str(config_vars.get("lerobot_venv_dir").get()).strip()
        if not venv_dir:
            venv_dir = str(default_for_key("lerobot_venv_dir", config))
        return _venv_activate_cmd_from_dir(venv_dir)

    def _activate_venv_in_terminal(command_text: str, *, remember_as_default: bool) -> tuple[bool, str]:
        command = _normalize_activate_command(command_text)
        if not command:
            return False, "Activation command is empty."
        if run_terminal_command is None:
            return False, "Terminal command runner is unavailable."
        if show_terminal is not None:
            show_terminal()
        ok, message = run_terminal_command(command)
        if not ok:
            return False, message
        if remember_as_default:
            config["setup_venv_activate_cmd"] = command
            parsed_venv_dir = _venv_dir_from_activate_command(command)
            if parsed_venv_dir is not None:
                config["lerobot_venv_dir"] = str(parsed_venv_dir)
                venv_var = config_vars.get("lerobot_venv_dir")
                if venv_var is not None:
                    venv_var.set(str(parsed_venv_dir))
            save_config(config, quiet=True)
        setup_status_var.set(
            setup_status_var.get()
            + "\n[INFO] Sent venv activation command to terminal. "
            "If it fails, use 'Enter Custom Venv Source'."
        )
        log_panel.append_log(f"Setup wizard sent terminal command: {command}")
        return True, ""

    def _prompt_custom_venv_source(default_command: str) -> str | None:
        from tkinter import simpledialog

        custom = simpledialog.askstring(
            "Custom Venv Source",
            (
                "Enter the venv activation command to run in terminal.\n"
                "Example: source ~/lerobot/lerobot_env/bin/activate"
            ),
            initialvalue=_normalize_activate_command(default_command) or DEFAULT_SETUP_VENV_ACTIVATE_CMD,
            parent=root,
        )
        if custom is None:
            return None
        value = _normalize_activate_command(custom)
        return value or None

    def _should_auto_open_wizard(status: Any | None) -> bool:
        if status is None:
            return False
        return not bool(getattr(status, "virtual_env_active", False))

    def apply_setup_path_defaults() -> None:
        lerobot_dir_raw = str(config_vars["lerobot_dir"].get()).strip()
        if not lerobot_dir_raw:
            messagebox.showerror("Setup Wizard", "LeRobot folder path is empty.")
            return
        lerobot_dir = Path(lerobot_dir_raw).expanduser()
        hf_username = str(config_vars["hf_username"].get()).strip() or str(config.get("hf_username", ""))
        config_vars["lerobot_venv_dir"].set(str(lerobot_dir / "lerobot_env"))
        config_vars["record_data_dir"].set(str(lerobot_dir / "data"))
        config_vars["deploy_data_dir"].set(str(default_for_key("deploy_data_dir", {"hf_username": hf_username})))
        config_vars["trained_models_dir"].set(str(lerobot_dir / "trained_models"))
        setup_status_var.set(
            setup_status_var.get()
            + "\n[INFO] Applied venv/record/deploy/models paths. Click Save Config to persist."
        )
        log_panel.append_log("Applied setup path defaults from LeRobot folder path.")

    def open_setup_wizard_popout() -> None:
        while True:
            status = _refresh_setup_status()
            if status is None:
                messagebox.showerror("Setup Wizard", "Cannot open wizard until current config fields are valid.")
                return
            default_activate_cmd = _default_activate_command_from_ui()
            action = ask_text_dialog_with_actions(
                root=root,
                title="LeRobot Setup Wizard",
                text=build_setup_wizard_guide(status),
                actions=[
                    ("activate_venv", "Activate Venv"),
                    ("custom_activate", "Enter Custom Venv Source"),
                    ("copy_commands", "Copy Setup Commands"),
                    ("apply_paths", "Apply Path Defaults"),
                    ("recheck", "Re-check Environment"),
                ],
                confirm_label="Done",
                cancel_label="Close",
                width=1020,
                height=620,
                wrap_mode="word",
            )
            if action == "copy_commands":
                _copy_setup_commands(status)
                continue
            if action == "activate_venv":
                ok, error_text = _activate_venv_in_terminal(default_activate_cmd, remember_as_default=False)
                if not ok:
                    retry_custom = messagebox.askyesno(
                        "Setup Wizard",
                        (
                            "Unable to run default activation command.\n\n"
                            f"{error_text}\n\n"
                            "Do you want to enter a custom venv source command now?"
                        ),
                    )
                    if retry_custom:
                        custom_command = _prompt_custom_venv_source(default_activate_cmd)
                        if custom_command:
                            custom_ok, custom_error_text = _activate_venv_in_terminal(
                                custom_command,
                                remember_as_default=True,
                            )
                            if not custom_ok:
                                messagebox.showerror(
                                    "Setup Wizard",
                                    f"Unable to run activation command:\n{custom_error_text}",
                                )
                continue
            if action == "custom_activate":
                custom_command = _prompt_custom_venv_source(default_activate_cmd)
                if custom_command is None:
                    continue
                ok, error_text = _activate_venv_in_terminal(custom_command, remember_as_default=True)
                if not ok:
                    messagebox.showerror("Setup Wizard", f"Unable to run activation command:\n{error_text}")
                continue
            if action == "apply_paths":
                apply_setup_path_defaults()
                continue
            if action == "recheck":
                continue
            break

    def run_setup_check_from_gui(*, allow_auto_wizard: bool = True) -> None:
        status = _refresh_setup_status()
        if status is None:
            messagebox.showerror("Setup Wizard", "Setup check could not run with current config values.")
            return
        log_panel.append_log("Ran setup wizard environment check from Config tab.")
        if allow_auto_wizard and _should_auto_open_wizard(status) and not setup_wizard_prompted["value"]:
            setup_wizard_prompted["value"] = True
            open_setup_wizard_popout()

    def copy_setup_commands_from_gui() -> None:
        status = _refresh_setup_status()
        if status is None:
            messagebox.showerror("Setup Wizard", "Setup commands unavailable while config fields are invalid.")
            return
        _copy_setup_commands(status)

    run_setup_check_button = ttk.Button(setup_controls, text="Run Setup Check", command=run_setup_check_from_gui)
    run_setup_check_button.pack(side="left")
    open_setup_wizard_button = ttk.Button(setup_controls, text="Open Setup Wizard", command=open_setup_wizard_popout)
    open_setup_wizard_button.pack(side="left", padx=(8, 0))
    copy_setup_commands_button = ttk.Button(setup_controls, text="Copy Setup Commands", command=copy_setup_commands_from_gui)
    copy_setup_commands_button.pack(side="left", padx=(8, 0))
    apply_setup_paths_button = ttk.Button(setup_controls, text="Apply Path Defaults", command=apply_setup_path_defaults)
    apply_setup_paths_button.pack(side="left", padx=(8, 0))

    diagnostics_frame = ttk.LabelFrame(config_tab, text="Diagnostics", style="Section.TLabelframe", padding=10)
    diagnostics_frame.pack(fill="both", expand=True, pady=(0, 10))
    diagnostics_controls = ttk.Frame(diagnostics_frame, style="Panel.TFrame")
    diagnostics_controls.pack(fill="x", pady=(0, 6))

    doctor_report_var = tk.StringVar(value="")
    doctor_text = tk.Text(
        diagnostics_frame,
        height=9,
        state="disabled",
        bg=log_panel.colors.get("surface", "#1a1a1a"),
        fg=log_panel.colors.get("text", "#eeeeee"),
        insertbackground="#f8fafc",
        font=(log_panel.colors.get("font_mono", "TkFixedFont"), 10),
        relief="flat",
        padx=8,
        pady=8,
    )
    doctor_text.tag_configure("pass", foreground="#4ade80")
    doctor_text.tag_configure("warn", foreground=log_panel.colors.get("accent", "#f0a500"))
    doctor_text.tag_configure("fail", foreground="#f87171")
    doctor_text.tag_configure("default", foreground=log_panel.colors.get("text", "#eeeeee"))
    doctor_text.pack(fill="both", expand=True)

    def _apply_doctor_text_theme(theme_colors: dict[str, str]) -> None:
        doctor_text.configure(
            bg=theme_colors.get("surface", "#1a1a1a"),
            fg=theme_colors.get("text", "#eeeeee"),
            insertbackground=theme_colors.get("text", "#f8fafc"),
            font=(theme_colors.get("font_mono", "TkFixedFont"), 10),
        )
        doctor_text.tag_configure("pass", foreground=theme_colors.get("success", "#4ade80"))
        doctor_text.tag_configure("warn", foreground=theme_colors.get("accent", "#f0a500"))
        doctor_text.tag_configure("fail", foreground=theme_colors.get("error", "#f87171"))
        doctor_text.tag_configure("default", foreground=theme_colors.get("text", "#eeeeee"))

    def render_doctor_report(checks: list[tuple[str, str, str]]) -> None:
        summary = summarize_checks(checks, title="Doctor Report")
        doctor_report_var.set(summary)
        doctor_text.configure(state="normal")
        doctor_text.delete("1.0", "end")
        for line in summary.splitlines():
            tag = "default"
            if line.startswith("[PASS"):
                tag = "pass"
            elif line.startswith("[WARN"):
                tag = "warn"
            elif line.startswith("[FAIL"):
                tag = "fail"
            doctor_text.insert("end", line + "\n", (tag,))
        doctor_text.see("end")
        doctor_text.configure(state="disabled")

    def run_doctor_from_gui() -> None:
        preview, error_text = coerce_config_from_vars(config, config_vars, CONFIG_FIELDS)
        if preview is None:
            messagebox.showerror("Validation Error", error_text or "Invalid config values.")
            return
        checks = collect_doctor_checks(preview)
        render_doctor_report(checks)
        log_panel.append_log("Ran Doctor from Config tab.")

    def copy_doctor_report() -> None:
        report_text = doctor_report_var.get()
        if not report_text.strip():
            messagebox.showinfo("Copy Doctor Report", "No doctor report available yet.")
            return
        root.clipboard_clear()
        root.clipboard_append(report_text)
        log_panel.append_log("Copied doctor report to clipboard.")

    run_doctor_button = ttk.Button(diagnostics_controls, text="Run Doctor", command=run_doctor_from_gui)
    run_doctor_button.pack(side="left")
    copy_doctor_button = ttk.Button(diagnostics_controls, text="Copy Doctor Report", command=copy_doctor_report)
    copy_doctor_button.pack(side="left", padx=(8, 0))

    launcher_frame = ttk.LabelFrame(config_tab, text="Desktop App Launcher", style="Section.TLabelframe", padding=10)
    launcher_frame.pack(fill="x", pady=(0, 10))
    ttk.Label(
        launcher_frame,
        text=(
            "Create/update an app-menu launcher that opens this GUI with the active Python environment.\n"
            "After install, you can launch without keeping a terminal open."
        ),
        style="Field.TLabel",
        justify="left",
    ).pack(anchor="w")
    launcher_controls = ttk.Frame(launcher_frame, style="Panel.TFrame")
    launcher_controls.pack(fill="x", pady=(8, 0))

    def install_launcher_from_gui() -> None:
        preview, error_text = _setup_preview_config()
        if preview is None:
            messagebox.showerror("Desktop Launcher", error_text or "Invalid config values.")
            return
        venv_dir = Path(str(preview.get("lerobot_venv_dir", default_for_key("lerobot_venv_dir", preview)))).expanduser()
        report = install_desktop_launcher(
            app_dir=Path(__file__).resolve().parents[1],
            venv_dir=venv_dir,
        )
        if not report.ok:
            messagebox.showerror("Desktop Launcher", report.message)
            log_panel.append_log(f"Desktop launcher install failed: {report.message}")
            return
        script_path = str(report.script_path) if report.script_path is not None else "(unknown)"
        desktop_path = str(report.desktop_entry_path) if report.desktop_entry_path is not None else "(unknown)"
        icon_path = str(report.icon_path) if report.icon_path is not None else "(not installed)"
        messagebox.showinfo(
            "Desktop Launcher Installed",
            (
                "Launcher installed.\n\n"
                f"Script: {script_path}\n"
                f"Launcher path: {desktop_path}\n\n"
                f"Icon: {icon_path}\n\n"
                f"Venv path: {venv_dir}\n\n"
                "Open 'LeRobot Pipeline Manager' from your app menu or Applications folder."
            ),
        )
        log_panel.append_log(f"Desktop launcher installed: {script_path} (icon: {icon_path})")

    install_launcher_button = ttk.Button(
        launcher_controls,
        text="Install Desktop Launcher",
        command=install_launcher_from_gui,
    )
    install_launcher_button.pack(side="left")

    def save_config_from_gui() -> None:
        parsed_config, error_text = coerce_config_from_vars(config, config_vars, CONFIG_FIELDS)
        if parsed_config is None:
            messagebox.showerror("Validation Error", error_text or "Invalid config values.")
            return

        parsed_config["setup_venv_activate_cmd"] = _venv_activate_cmd_from_dir(
            str(parsed_config.get("lerobot_venv_dir", default_for_key("lerobot_venv_dir", parsed_config)))
        )
        config.update(parsed_config)
        save_config(config)
        record_dir_var.set(str(config["record_data_dir"]))
        deploy_root_var.set(str(config["trained_models_dir"]))
        deploy_eval_episodes_var.set(str(config.get("eval_num_episodes", 10)))
        deploy_eval_duration_var.set(str(config.get("eval_duration_s", 20)))
        deploy_eval_task_var.set(str(config.get("eval_task", DEFAULT_TASK)))
        refresh_local_models()
        refresh_record_summary()
        refresh_header_subtitle()
        record_camera_preview.refresh_labels()
        deploy_camera_preview.refresh_labels()
        messagebox.showinfo("Saved", "Configuration saved.")

    save_config_button = ttk.Button(config_tab, text="Save Config", style="Accent.TButton", command=save_config_from_gui)
    save_config_button.pack(anchor="w", pady=(2, 0))

    def sync_from_config() -> None:
        for field in CONFIG_FIELDS:
            key = field["key"]
            var = config_vars.get(key)
            if var is None:
                continue
            value = config.get(key)
            if value in (None, ""):
                value = default_for_key(key, config)
            var.set(str(value))
        _refresh_setup_status()

    _refresh_setup_status()
    if _should_auto_open_wizard(setup_status_state["value"]) and not setup_wizard_prompted["value"]:
        setup_wizard_prompted["value"] = True
        open_setup_wizard_popout()

    def apply_theme(updated_colors: dict[str, str]) -> None:
        _apply_doctor_text_theme(updated_colors)

    return ConfigTabHandles(
        action_buttons=[
            run_setup_check_button,
            open_setup_wizard_button,
            copy_setup_commands_button,
            apply_setup_paths_button,
            run_doctor_button,
            copy_doctor_button,
            install_launcher_button,
            save_config_button,
        ],
        sync_from_config=sync_from_config,
        apply_theme=apply_theme,
    )
