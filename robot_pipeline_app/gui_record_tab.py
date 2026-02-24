from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .checks import run_preflight_for_record
from .config_store import get_lerobot_dir, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import ask_text_dialog, show_text_dialog
from .gui_forms import build_record_request_and_command
from .gui_log import GuiLogPanel
from .repo_utils import dataset_exists_on_hf, suggest_dataset_name
from .runner import format_command
from .types import GuiRunProcessAsync
from .workflows import move_recorded_dataset


@dataclass
class RecordTabHandles:
    record_dir_var: Any
    record_camera_preview: DualCameraPreview
    refresh_summary: Callable[[], None]
    action_buttons: list[Any]


def setup_record_tab(
    *,
    root: Any,
    record_tab: Any,
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
) -> RecordTabHandles:
    import tkinter as tk
    from tkinter import ttk

    record_container = ttk.Frame(record_tab, style="Panel.TFrame")
    record_container.pack(fill="both", expand=True)

    suggested_dataset, _ = suggest_dataset_name(config)
    record_dataset_var = tk.StringVar(value=suggested_dataset)
    record_episodes_var = tk.StringVar(value="20")
    record_duration_var = tk.StringVar(value="20")
    record_task_var = tk.StringVar(value=DEFAULT_TASK)
    record_dir_var = tk.StringVar(value=str(config["record_data_dir"]))
    record_upload_var = tk.BooleanVar(value=False)

    record_form = ttk.LabelFrame(record_container, text="Recording Setup", style="Section.TLabelframe", padding=12)
    record_form.pack(fill="x")
    record_form.columnconfigure(1, weight=1)

    ttk.Label(record_form, text="Dataset name (or repo id)", style="Field.TLabel").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(
        record_form,
        text="Suggest Next",
        command=lambda: record_dataset_var.set(suggest_dataset_name(config)[0]),
    ).grid(row=0, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Label(record_form, text="Local dataset save folder", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_dir_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(record_form, text="Browse", command=lambda: choose_folder(record_dir_var)).grid(
        row=1,
        column=2,
        sticky="w",
        padx=(6, 0),
        pady=4,
    )

    ttk.Label(record_form, text="Episodes", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_episodes_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Episode time (seconds)", style="Field.TLabel").grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(record_form, textvariable=record_duration_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Task description", style="Field.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_task_var, width=52).grid(row=4, column=1, sticky="ew", pady=4)

    ttk.Checkbutton(record_form, text="Upload to Hugging Face after recording", variable=record_upload_var).grid(
        row=5,
        column=1,
        sticky="w",
        pady=(8, 8),
    )

    record_buttons = ttk.Frame(record_form, style="Panel.TFrame")
    record_buttons.grid(row=6, column=1, sticky="w", pady=(8, 0))
    preview_record_button = ttk.Button(record_buttons, text="Preview Command")
    preview_record_button.pack(side="left")
    run_record_button = ttk.Button(record_buttons, text="Run Record", style="Accent.TButton")
    run_record_button.pack(side="left", padx=(10, 0))

    record_summary_var = tk.StringVar(value="")
    record_summary_panel = ttk.LabelFrame(record_container, text="Current Robot Snapshot", style="Section.TLabelframe", padding=10)
    record_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(record_summary_panel, textvariable=record_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    record_camera_preview = DualCameraPreview(
        root=root,
        parent=record_container,
        title="Record Camera Preview",
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        append_log=log_panel.append_log,
        on_camera_indices_changed=on_camera_indices_changed,
    )

    def refresh_record_summary() -> None:
        record_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Laptop camera idx: {laptop} | Phone camera idx: {phone}\n"
            "Camera stream: {w}x{h} @ {fps}fps (warmup {warmup}s)".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                w=config.get("camera_width", 640),
                h=config.get("camera_height", 360),
                fps=config.get("camera_fps", 30),
                warmup=config["camera_warmup_s"],
            )
        )

    def preview_record() -> None:
        req, cmd, error_text = build_record_request_and_command(
            config=config,
            dataset_input=record_dataset_var.get(),
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
        )
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return
        last_command_state["value"] = format_command(cmd)
        log_panel.append_log("Preview record command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Record Command",
            text=last_command_state["value"],
            wrap_mode="none",
        )

    def run_record_from_gui() -> None:
        req, cmd, error_text = build_record_request_and_command(
            config=config,
            dataset_input=record_dataset_var.get(),
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
        )
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return

        config["record_data_dir"] = str(req.dataset_root)

        exists = dataset_exists_on_hf(req.dataset_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{req.dataset_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not ask_text_dialog(
            root=root,
            title="Confirm Record",
            text=format_command(cmd),
            confirm_label="Run",
            cancel_label="Cancel",
            wrap_mode="none",
        ):
            return

        preflight_checks = run_preflight_for_record(
            config=config,
            dataset_root=req.dataset_root,
            upload_enabled=req.upload_after_record,
        )
        if not confirm_preflight_in_gui("Record Preflight", preflight_checks):
            return

        def after_record(return_code: int, was_canceled: bool) -> None:
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Record canceled.")
                    messagebox.showinfo("Canceled", "Record command was canceled.")
                else:
                    set_running(False, "Recording failed.", True)
                    messagebox.showerror("Record Failed", f"Recording failed with exit code {return_code}.")
                return

            lerobot_dir = get_lerobot_dir(config)
            active_dataset = move_recorded_dataset(
                lerobot_dir=lerobot_dir,
                dataset_name=req.dataset_name,
                dataset_root=req.dataset_root,
                log=log_panel.append_log,
            )

            config["last_dataset_name"] = req.dataset_name
            save_config(config)
            refresh_record_summary()
            refresh_header_subtitle()

            if was_canceled:
                set_running(False, "Record canceled.")
                messagebox.showinfo("Canceled", "Record command was canceled. Upload was skipped.")
                return

            if not req.upload_after_record:
                set_running(False, "Record completed.")
                messagebox.showinfo("Done", "Recording completed.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                req.dataset_repo_id,
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]

            set_running(False, "Record completed. Starting upload...")

            def after_upload(upload_code: int, upload_canceled: bool) -> None:
                if upload_canceled:
                    set_running(False, "Upload canceled.")
                    messagebox.showinfo("Canceled", "Upload command was canceled.")
                elif upload_code != 0:
                    set_running(False, "Upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Upload failed with exit code {upload_code}.")
                else:
                    set_running(False, "Record + upload completed.")
                    messagebox.showinfo("Done", "Recording and upload completed.")

            run_process_async(
                upload_cmd,
                get_lerobot_dir(config),
                after_upload,
                None,
                None,
                "upload",
                None,
                {"dataset_repo_id": req.dataset_repo_id},
            )

        run_process_async(
            cmd,
            get_lerobot_dir(config),
            after_record,
            req.num_episodes,
            req.num_episodes * req.episode_time_s,
            "record",
            preflight_checks,
            {"dataset_repo_id": req.dataset_repo_id},
        )

    preview_record_button.configure(command=preview_record)
    run_record_button.configure(command=run_record_from_gui)

    return RecordTabHandles(
        record_dir_var=record_dir_var,
        record_camera_preview=record_camera_preview,
        refresh_summary=refresh_record_summary,
        action_buttons=[preview_record_button, run_record_button],
    )
