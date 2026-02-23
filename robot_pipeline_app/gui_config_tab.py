from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .checks import collect_doctor_checks, summarize_checks
from .config_store import default_for_key, save_config
from .constants import CONFIG_FIELDS, DEFAULT_TASK
from .gui_forms import coerce_config_from_vars
from .gui_log import GuiLogPanel


@dataclass
class ConfigTabHandles:
    action_buttons: list[Any]


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
) -> ConfigTabHandles:
    import tkinter as tk
    from tkinter import scrolledtext, ttk

    config_vars: dict[str, Any] = {}
    path_keys = {"lerobot_dir", "runs_dir", "record_data_dir", "trained_models_dir"}
    field_lookup = {field["key"]: field for field in CONFIG_FIELDS}
    group_layout = [
        ("Paths", ["lerobot_dir", "runs_dir", "record_data_dir", "trained_models_dir"]),
        ("Robot Ports", ["follower_port", "leader_port"]),
        (
            "Cameras",
            [
                "camera_laptop_index",
                "camera_phone_index",
                "camera_warmup_s",
                "camera_width",
                "camera_height",
                "camera_fps",
            ],
        ),
        (
            "Hugging Face + Defaults",
            [
                "hf_username",
                "last_dataset_name",
                "eval_num_episodes",
                "eval_duration_s",
                "eval_task",
                "last_eval_dataset_name",
                "last_model_name",
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

    diagnostics_frame = ttk.LabelFrame(config_tab, text="Diagnostics", style="Section.TLabelframe", padding=10)
    diagnostics_frame.pack(fill="both", expand=True, pady=(0, 10))
    diagnostics_controls = ttk.Frame(diagnostics_frame, style="Panel.TFrame")
    diagnostics_controls.pack(fill="x", pady=(0, 6))

    doctor_report_var = tk.StringVar(value="")
    doctor_text = scrolledtext.ScrolledText(
        diagnostics_frame,
        height=9,
        state="disabled",
        bg="#111827",
        fg="#d4d4d4",
        insertbackground="#f8fafc",
        font=("Menlo", 10),
        relief="flat",
        padx=8,
        pady=8,
    )
    doctor_text.tag_configure("pass", foreground="#4ade80")
    doctor_text.tag_configure("warn", foreground="#fbbf24")
    doctor_text.tag_configure("fail", foreground="#f87171")
    doctor_text.tag_configure("default", foreground="#d4d4d4")
    doctor_text.pack(fill="both", expand=True)

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

    def save_config_from_gui() -> None:
        parsed_config, error_text = coerce_config_from_vars(config, config_vars, CONFIG_FIELDS)
        if parsed_config is None:
            messagebox.showerror("Validation Error", error_text or "Invalid config values.")
            return

        config.clear()
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

    return ConfigTabHandles(action_buttons=[run_doctor_button, copy_doctor_button, save_config_button])
