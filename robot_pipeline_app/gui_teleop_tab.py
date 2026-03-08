from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .camera_schema import resolve_camera_schema
from .command_overrides import get_flag_value
from .checks import run_preflight_for_teleop
from .commands import resolve_follower_robot_id, resolve_leader_robot_id
from .config_store import get_lerobot_dir, save_config
from .gui_dialogs import ask_editable_command_dialog, format_command_for_dialog, show_text_dialog
from .gui_forms import build_teleop_request_and_command
from .gui_log import GuiLogPanel
from .runner import format_command
from .serial_scan import format_robot_port_scan, scan_robot_serial_ports, suggest_follower_leader_ports
from .types import GuiRunProcessAsync


@dataclass
class TeleopTabHandles:
    refresh_summary: Callable[[], None]
    apply_theme: Callable[[dict[str, str]], None]
    action_buttons: list[Any]


def setup_teleop_tab(
    *,
    root: Any,
    teleop_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    log_panel: GuiLogPanel,
    messagebox: Any,
    set_running: Callable[[bool, str | None, bool], None],
    run_process_async: GuiRunProcessAsync,
    refresh_header_subtitle: Callable[[], None],
    last_command_state: dict[str, str],
    confirm_preflight_in_gui: Callable[[str, list[tuple[str, str, str]]], bool],
) -> TeleopTabHandles:
    import tkinter as tk
    from tkinter import ttk

    teleop_container = ttk.Frame(teleop_tab, style="Panel.TFrame")
    teleop_container.pack(fill="both", expand=True)

    follower_port_var = tk.StringVar(value=str(config.get("follower_port", "")))
    leader_port_var = tk.StringVar(value=str(config.get("leader_port", "")))
    follower_id_var = tk.StringVar(value=str(config.get("follower_robot_id", "red4")).strip() or "red4")
    leader_id_var = tk.StringVar(value=str(config.get("leader_robot_id", "white")).strip() or "white")

    teleop_form = ttk.LabelFrame(teleop_container, text="Teleop Setup", style="Section.TLabelframe", padding=12)
    teleop_form.pack(fill="x")
    teleop_form.columnconfigure(1, weight=1)

    ttk.Label(teleop_form, text="Follower port", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(teleop_form, textvariable=follower_port_var, width=30).grid(row=0, column=1, sticky="w", pady=4)

    ttk.Label(teleop_form, text="Leader port", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(teleop_form, textvariable=leader_port_var, width=30).grid(row=1, column=1, sticky="w", pady=4)

    ttk.Label(teleop_form, text="Follower robot id", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(teleop_form, textvariable=follower_id_var, width=30).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(teleop_form, text="Leader robot id", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(teleop_form, textvariable=leader_id_var, width=30).grid(row=3, column=1, sticky="w", pady=4)

    teleop_buttons = ttk.Frame(teleop_form, style="Panel.TFrame")
    teleop_buttons.grid(row=4, column=1, sticky="w", pady=(8, 0))
    preview_teleop_button = ttk.Button(teleop_buttons, text="Preview Command")
    preview_teleop_button.pack(side="left")
    run_teleop_button = ttk.Button(teleop_buttons, text="Run Teleop", style="Accent.TButton")
    run_teleop_button.pack(side="left", padx=(10, 0))
    scan_ports_button = ttk.Button(teleop_buttons, text="Scan Robot Ports")
    scan_ports_button.pack(side="left", padx=(10, 0))

    teleop_summary_var = tk.StringVar(value="")
    teleop_summary_panel = ttk.LabelFrame(teleop_container, text="Teleop Snapshot", style="Section.TLabelframe", padding=10)
    teleop_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(teleop_summary_panel, textvariable=teleop_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    teleop_decode_notice: dict[str, bool] = {"shown": False}

    def refresh_teleop_summary() -> None:
        schema = resolve_camera_schema(config)
        camera_names = ", ".join(spec.name for spec in schema.specs) if schema.specs else "(none)"
        teleop_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Follower id: {follower_id} | Leader id: {leader_id}\n"
            "Runtime camera keys: {camera_keys}".format(
                follower=str(follower_port_var.get()).strip() or "-",
                leader=str(leader_port_var.get()).strip() or "-",
                follower_id=str(follower_id_var.get()).strip() or "-",
                leader_id=str(leader_id_var.get()).strip() or "-",
                camera_keys=camera_names,
            )
        )

    def _build_current_teleop_command() -> tuple[dict[str, Any] | None, list[str] | None, str | None]:
        _req, cmd, updated_config, error_text = build_teleop_request_and_command(
            config=config,
            follower_port_raw=follower_port_var.get(),
            leader_port_raw=leader_port_var.get(),
            follower_id_raw=follower_id_var.get(),
            leader_id_raw=leader_id_var.get(),
        )
        if error_text or updated_config is None or cmd is None:
            return None, None, error_text or "Unable to build teleop command."
        run_config = dict(config)
        run_config.update(updated_config)
        return run_config, cmd, None

    def _persist_config_updates(run_config: dict[str, Any]) -> None:
        updated = False
        for key in ("follower_port", "leader_port", "follower_robot_id", "leader_robot_id"):
            new_value = run_config.get(key)
            if config.get(key) != new_value:
                config[key] = new_value
                updated = True
        if updated:
            save_config(config, quiet=True)
            refresh_header_subtitle()
            log_panel.append_log("Saved teleop connection defaults to config.")

    def preview_teleop() -> None:
        _, cmd, error_text = _build_current_teleop_command()
        if error_text or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build teleop command.")
            return
        last_command_state["value"] = format_command(cmd)
        command_for_dialog = format_command_for_dialog(cmd)
        log_panel.append_log("Preview teleop command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Teleop Command",
            text=command_for_dialog,
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
            current_follower=str(follower_port_var.get()),
            current_leader=str(leader_port_var.get()),
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
                    "Apply these now?"
                ),
            )
            if apply_guess:
                follower_port_var.set(follower_guess)
                leader_port_var.set(leader_guess)
                log_panel.append_log(
                    f"Applied scanned ports: follower={follower_guess}, leader={leader_guess}"
                )

    def run_teleop() -> None:
        run_config, cmd, error_text = _build_current_teleop_command()
        if error_text or run_config is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build teleop command.")
            return
        if any("control_robot" in str(part) for part in cmd) and not teleop_decode_notice["shown"]:
            teleop_decode_notice["shown"] = True
            log_panel.append_log(
                "Teleop AV1 fallback active: using legacy control module for compatibility on systems without AV1 HW decode."
            )
        _persist_config_updates(run_config)
        refresh_teleop_summary()

        preflight_checks = run_preflight_for_teleop(run_config)
        if not confirm_preflight_in_gui("Teleop Preflight", preflight_checks):
            log_panel.append_log("Teleop canceled after preflight review.")
            return

        editable_cmd = ask_editable_command_dialog(
            root=root,
            title="Confirm Teleop Command",
            command_argv=cmd,
            intro_text=(
                "Review or edit the teleop command below.\n"
                "The exact command text here will be executed and saved to run history."
            ),
            confirm_label="Run Teleop",
            cancel_label="Cancel",
        )
        if editable_cmd is None:
            return
        if editable_cmd != cmd:
            log_panel.append_log("Running edited teleop command from command editor.")
        cmd = editable_cmd

        effective_follower_port = get_flag_value(cmd, "robot.port") or run_config.get("follower_port", "")
        effective_leader_port = get_flag_value(cmd, "teleop.port") or run_config.get("leader_port", "")
        effective_follower_id = get_flag_value(cmd, "robot.id") or resolve_follower_robot_id(run_config)
        effective_leader_id = get_flag_value(cmd, "teleop.id") or resolve_leader_robot_id(run_config)

        last_command_state["value"] = format_command(cmd)
        log_panel.append_log("Running teleop session...")

        def on_complete(return_code: int, canceled: bool) -> None:
            if canceled:
                set_running(False, "Teleop canceled.")
                log_panel.append_log("Teleop session canceled.")
                return
            if return_code == 0:
                set_running(False, "Teleop completed.")
                log_panel.append_log("Teleop session completed.")
                return
            set_running(False, "Teleop failed.", True)
            log_panel.append_log(f"Teleop session failed (exit code {return_code}).")

        teleop_context: dict[str, Any] = {
            "follower_port": effective_follower_port,
            "follower_id": effective_follower_id,
            "leader_port": effective_leader_port,
            "leader_id": effective_leader_id,
        }
        run_process_async(
            cmd,
            get_lerobot_dir(run_config),
            on_complete,
            None,
            None,
            "teleop",
            preflight_checks,
            teleop_context,
        )

    preview_teleop_button.configure(command=preview_teleop)
    run_teleop_button.configure(command=run_teleop)
    scan_ports_button.configure(command=scan_robot_ports)

    for var in (follower_port_var, leader_port_var, follower_id_var, leader_id_var):
        var.trace_add("write", lambda *_: refresh_teleop_summary())
    refresh_teleop_summary()

    return TeleopTabHandles(
        refresh_summary=refresh_teleop_summary,
        apply_theme=lambda updated_colors: None,
        action_buttons=[preview_teleop_button, run_teleop_button, scan_ports_button],
    )
