from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .checks import run_preflight_for_teleop
from .commands import build_lerobot_teleop_command
from .config_store import get_lerobot_dir, save_config
from .gui_camera import DualCameraPreview
from .gui_dialogs import format_command_for_dialog, show_text_dialog
from .gui_log import GuiLogPanel
from .runner import format_command
from .types import GuiRunProcessAsync


@dataclass
class TeleopTabHandles:
    teleop_camera_preview: DualCameraPreview
    refresh_summary: Callable[[], None]
    action_buttons: list[Any]


def setup_teleop_tab(
    *,
    root: Any,
    teleop_tab: Any,
    config: dict[str, Any],
    colors: dict[str, str],
    cv2_probe_ok: bool,
    cv2_probe_error: str,
    log_panel: GuiLogPanel,
    messagebox: Any,
    set_running: Callable[[bool, str | None, bool], None],
    run_process_async: GuiRunProcessAsync,
    on_camera_indices_changed: Callable[[int, int], None],
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
    follower_id_var = tk.StringVar(value="red4")
    leader_id_var = tk.StringVar(value="white")

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

    teleop_summary_var = tk.StringVar(value="")
    teleop_summary_panel = ttk.LabelFrame(teleop_container, text="Teleop Snapshot", style="Section.TLabelframe", padding=10)
    teleop_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(teleop_summary_panel, textvariable=teleop_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    teleop_camera_preview = DualCameraPreview(
        root=root,
        parent=teleop_container,
        title="Teleop Camera Preview",
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        append_log=log_panel.append_log,
        on_camera_indices_changed=on_camera_indices_changed,
    )

    def refresh_teleop_summary() -> None:
        teleop_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Follower id: {follower_id} | Leader id: {leader_id}\n"
            "Camera mapping: laptop idx {laptop}, phone idx {phone}".format(
                follower=str(follower_port_var.get()).strip() or "-",
                leader=str(leader_port_var.get()).strip() or "-",
                follower_id=str(follower_id_var.get()).strip() or "-",
                leader_id=str(leader_id_var.get()).strip() or "-",
                laptop=config.get("camera_laptop_index", "-"),
                phone=config.get("camera_phone_index", "-"),
            )
        )

    def _build_current_teleop_command() -> tuple[dict[str, Any] | None, list[str] | None, str | None]:
        follower_port = str(follower_port_var.get()).strip()
        leader_port = str(leader_port_var.get()).strip()
        follower_id = str(follower_id_var.get()).strip() or "red4"
        leader_id = str(leader_id_var.get()).strip() or "white"
        if not follower_port:
            return None, None, "Follower port is required."
        if not leader_port:
            return None, None, "Leader port is required."

        run_config = dict(config)
        run_config["follower_port"] = follower_port
        run_config["leader_port"] = leader_port
        cmd = build_lerobot_teleop_command(
            run_config,
            follower_robot_id=follower_id,
            leader_robot_id=leader_id,
        )
        return run_config, cmd, None

    def _persist_config_updates(run_config: dict[str, Any]) -> None:
        updated = False
        for key in ("follower_port", "leader_port"):
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

    def run_teleop() -> None:
        run_config, cmd, error_text = _build_current_teleop_command()
        if error_text or run_config is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build teleop command.")
            return
        _persist_config_updates(run_config)
        refresh_teleop_summary()

        preflight_checks = run_preflight_for_teleop(run_config)
        if not confirm_preflight_in_gui("Teleop Preflight", preflight_checks):
            log_panel.append_log("Teleop canceled after preflight review.")
            return

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

        run_process_async(
            cmd,
            get_lerobot_dir(run_config),
            on_complete,
            None,
            None,
            "teleop",
            preflight_checks,
            None,
        )

    preview_teleop_button.configure(command=preview_teleop)
    run_teleop_button.configure(command=run_teleop)

    for var in (follower_port_var, leader_port_var, follower_id_var, leader_id_var):
        var.trace_add("write", lambda *_: refresh_teleop_summary())
    refresh_teleop_summary()

    return TeleopTabHandles(
        teleop_camera_preview=teleop_camera_preview,
        refresh_summary=refresh_teleop_summary,
        action_buttons=[preview_teleop_button, run_teleop_button],
    )
