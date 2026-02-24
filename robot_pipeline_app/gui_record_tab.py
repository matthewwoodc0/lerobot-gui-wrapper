from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .checks import run_preflight_for_record
from .config_store import get_lerobot_dir, save_config
from .constants import DEFAULT_TASK
from .gui_camera import DualCameraPreview
from .gui_dialogs import ask_text_dialog, format_command_for_dialog, show_text_dialog
from .gui_forms import build_record_request_and_command
from .gui_log import GuiLogPanel
from .hf_tagging import (
    build_dataset_tag_upload_command,
    default_dataset_tags,
    safe_unlink,
    write_dataset_card_temp,
)
from .repo_utils import dataset_exists_on_hf, repo_name_from_repo_id, resolve_unique_repo_id, suggest_dataset_name
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
    record_hf_username_var = tk.StringVar(value=str(config.get("hf_username", "")).strip())
    record_hf_repo_name_var = tk.StringVar(value=repo_name_from_repo_id(record_dataset_var.get().strip()))
    record_tag_after_upload_var = tk.BooleanVar(value=bool(config.get("record_tag_after_upload", True)))

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

    upload_options = ttk.LabelFrame(record_form, text="Upload Options", style="Section.TLabelframe", padding=10)
    upload_options.columnconfigure(1, weight=1)
    ttk.Label(upload_options, text="Hugging Face username", style="Field.TLabel").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_username_var, width=24).grid(
        row=0,
        column=1,
        sticky="w",
        pady=4,
    )

    ttk.Label(upload_options, text="Dataset name on Hugging Face", style="Field.TLabel").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 6),
        pady=4,
    )
    ttk.Entry(upload_options, textvariable=record_hf_repo_name_var, width=40).grid(
        row=1,
        column=1,
        sticky="ew",
        pady=4,
    )
    ttk.Button(
        upload_options,
        text="Use Dataset Field",
        command=lambda: record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip())),
    ).grid(row=1, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Checkbutton(
        upload_options,
        text="Apply dataset tags after upload",
        variable=record_tag_after_upload_var,
    ).grid(row=2, column=1, sticky="w", pady=(6, 2))

    ttk.Label(
        upload_options,
        text="Tagging runs automatically after upload.",
        style="Muted.TLabel",
    ).grid(row=3, column=1, sticky="w", pady=(0, 2))

    record_buttons = ttk.Frame(record_form, style="Panel.TFrame")
    record_buttons.grid(row=7, column=1, sticky="w", pady=(8, 0))
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
            "Camera stream size: auto-detected at runtime | FPS: {fps} (warmup {warmup}s)".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                fps=config.get("camera_fps", 30),
                warmup=config["camera_warmup_s"],
            )
        )

    def preview_record() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return
        last_command_state["value"] = format_command(cmd)
        command_for_dialog = format_command_for_dialog(cmd)
        log_panel.append_log("Preview record command:")
        log_panel.append_log(last_command_state["value"])
        show_text_dialog(
            root=root,
            title="Record Command",
            text=command_for_dialog,
            wrap_mode="word",
        )

    def run_record_from_gui() -> None:
        req, cmd, error_text = build_current_record_from_ui()
        if error_text or req is None or cmd is None:
            messagebox.showerror("Validation Error", error_text or "Unable to build command.")
            return

        lerobot_dir = get_lerobot_dir(config)
        config["record_data_dir"] = str(req.dataset_root)
        resolved_repo_id, adjusted, _ = resolve_unique_repo_id(
            username=str(config["hf_username"]),
            dataset_name_or_repo_id=req.dataset_repo_id,
            local_roots=[req.dataset_root, lerobot_dir / "data"],
        )
        if adjusted:
            if record_upload_var.get():
                owner, name = resolved_repo_id.split("/", 1)
                record_hf_username_var.set(owner)
                record_hf_repo_name_var.set(name)
                log_panel.append_log(f"Auto-iterated Hugging Face dataset to avoid existing target: {resolved_repo_id}")
            else:
                record_dataset_var.set(resolved_repo_id)
                log_panel.append_log(f"Auto-iterated dataset to avoid existing target: {resolved_repo_id}")

            req, cmd, error_text = build_current_record_from_ui()
            if error_text or req is None or cmd is None:
                messagebox.showerror("Validation Error", error_text or "Unable to build command.")
                return

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
            text=(
                "Review the record command below.\n"
                "Click Confirm to run it, or Cancel to stop.\n\n"
                + format_command_for_dialog(cmd)
            ),
            confirm_label="Confirm",
            cancel_label="Cancel",
            wrap_mode="char",
        ):
            return

        preflight_checks = run_preflight_for_record(
            config=config,
            dataset_root=req.dataset_root,
            upload_enabled=req.upload_after_record,
            episode_time_s=req.episode_time_s,
            dataset_repo_id=req.dataset_repo_id,
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

            active_dataset = move_recorded_dataset(
                lerobot_dir=lerobot_dir,
                dataset_name=req.dataset_name,
                dataset_root=req.dataset_root,
                log=log_panel.append_log,
            )

            config["last_dataset_name"] = req.dataset_name
            if record_upload_var.get():
                config["hf_username"] = str(record_hf_username_var.get()).strip().strip("/") or str(config.get("hf_username", ""))
            config["record_tag_after_upload"] = bool(record_tag_after_upload_var.get())
            save_config(config)
            next_dataset_name, _ = suggest_dataset_name(config)
            record_dataset_var.set(next_dataset_name)
            if record_upload_var.get():
                record_hf_repo_name_var.set(next_dataset_name)
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
                    if not record_tag_after_upload_var.get():
                        set_running(False, "Record + upload completed.")
                        messagebox.showinfo(
                            "Done",
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {req.dataset_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded dataset: {req.dataset_name}\n"
                            f"Hugging Face repo: {req.dataset_repo_id}\n"
                            "Tagging status: skipped",
                        )
                        return

                    tags = default_dataset_tags(
                        config=config,
                        dataset_repo_id=req.dataset_repo_id,
                        task=req.task,
                    )
                    card_path = write_dataset_card_temp(
                        dataset_repo_id=req.dataset_repo_id,
                        dataset_name=req.dataset_name,
                        tags=tags,
                        task=req.task,
                    )
                    tag_cmd = build_dataset_tag_upload_command(
                        dataset_repo_id=req.dataset_repo_id,
                        card_path=card_path,
                    )

                    set_running(False, "Upload completed. Applying tags...")

                    def after_tag_upload(tag_code: int, tag_canceled: bool) -> None:
                        safe_unlink(card_path)
                        base_details = (
                            "Recording and upload completed.\n\n"
                            f"Hugging Face account: {req.dataset_repo_id.split('/', 1)[0]}\n"
                            f"Uploaded name: {req.dataset_name}\n"
                            f"Hugging Face repo: {req.dataset_repo_id}\n"
                            f"Tags: {', '.join(tags)}\n"
                        )
                        if tag_canceled:
                            set_running(False, "Upload completed. Tagging canceled.")
                            messagebox.showwarning(
                                "Upload Completed",
                                base_details + "Tagging status: canceled",
                            )
                            return
                        if tag_code != 0:
                            set_running(False, "Upload completed. Tagging failed.", True)
                            messagebox.showwarning(
                                "Upload Completed (Tagging Warning)",
                                base_details + f"Tagging status: failed (exit code {tag_code})",
                            )
                            return

                        set_running(False, "Record + upload + tagging completed.")
                        messagebox.showinfo(
                            "Done",
                            base_details + "Tagging status: success",
                        )

                    try:
                        run_process_async(
                            cmd=tag_cmd,
                            cwd=get_lerobot_dir(config),
                            complete_callback=after_tag_upload,
                            expected_episodes=None,
                            expected_seconds=None,
                            run_mode="upload",
                            preflight_checks=None,
                            artifact_context={"dataset_repo_id": req.dataset_repo_id},
                            start_error_callback=lambda _exc: safe_unlink(card_path),
                        )
                    except Exception:
                        safe_unlink(card_path)
                        raise

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

    def build_current_record_from_ui() -> tuple[Any, Any, str | None]:
        dataset_input = record_dataset_var.get()
        if record_upload_var.get():
            hf_username = str(record_hf_username_var.get()).strip().strip("/")
            hf_dataset_name = str(record_hf_repo_name_var.get()).strip().strip("/")
            if not hf_username:
                return None, None, "Hugging Face username is required when upload is enabled."
            if not hf_dataset_name:
                return None, None, "Hugging Face dataset name is required when upload is enabled."
            dataset_input = f"{hf_username}/{hf_dataset_name}"
        return build_record_request_and_command(
            config=config,
            dataset_input=dataset_input,
            episodes_raw=record_episodes_var.get(),
            duration_raw=record_duration_var.get(),
            task_raw=record_task_var.get(),
            dataset_dir_raw=record_dir_var.get(),
            upload_enabled=record_upload_var.get(),
        )

    def refresh_upload_options_visibility(*_: Any) -> None:
        if record_upload_var.get():
            if not str(record_hf_username_var.get()).strip():
                record_hf_username_var.set(str(config.get("hf_username", "")).strip())
            if not str(record_hf_repo_name_var.get()).strip():
                record_hf_repo_name_var.set(repo_name_from_repo_id(record_dataset_var.get().strip()))
            upload_options.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(2, 8))
        else:
            upload_options.grid_remove()

    refresh_upload_options_visibility()
    record_upload_var.trace_add("write", refresh_upload_options_visibility)

    preview_record_button.configure(command=preview_record)
    run_record_button.configure(command=run_record_from_gui)

    return RecordTabHandles(
        record_dir_var=record_dir_var,
        record_camera_preview=record_camera_preview,
        refresh_summary=refresh_record_summary,
        action_buttons=[preview_record_button, run_record_button],
    )
