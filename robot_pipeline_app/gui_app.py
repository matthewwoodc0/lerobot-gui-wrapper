from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .checks import has_failures, summarize_checks
from .config_store import normalize_config_without_prompts, normalize_path, save_config
from .gui_config_tab import setup_config_tab
from .gui_deploy_tab import setup_deploy_tab
from .gui_dialogs import ask_text_dialog, show_text_dialog
from .gui_log import GuiLogPanel
from .gui_record_tab import setup_record_tab
from .gui_runner import create_run_controller
from .probes import probe_module_import


def run_gui_mode(raw_config: dict[str, Any]) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print("Tkinter GUI is unavailable on this device.")
        print(f"Details: {exc}")
        return

    if "LEROBOT_DISABLE_CAMERA_PREVIEW" in os.environ:
        cv2_probe_ok, cv2_probe_error = False, "Disabled by LEROBOT_DISABLE_CAMERA_PREVIEW"
    else:
        cv2_probe_ok, cv2_probe_error = probe_module_import("cv2")

    config = normalize_config_without_prompts(raw_config)
    if not raw_config:
        save_config(config)

    root = tk.Tk()
    root.title("LeRobot Pipeline Manager")
    root.geometry("1240x900")
    root.minsize(1080, 760)

    colors = {
        "bg": "#0b1220",
        "panel": "#111a2e",
        "header": "#111827",
        "border": "#273449",
        "text": "#e2e8f0",
        "muted": "#9ca3af",
        "accent": "#0ea5e9",
        "accent_dark": "#0284c7",
        "running": "#f59e0b",
        "ready": "#22c55e",
        "error": "#ef4444",
    }
    root.configure(bg=colors["bg"])

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Panel.TFrame", background=colors["bg"])
    style.configure("Section.TLabelframe", background=colors["panel"], bordercolor=colors["border"])
    style.configure(
        "Section.TLabelframe.Label",
        background=colors["panel"],
        foreground=colors["text"],
        font=("Helvetica", 11, "bold"),
    )
    style.configure("Field.TLabel", background=colors["panel"], foreground=colors["text"], font=("Helvetica", 10))
    style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"], font=("Helvetica", 10))
    style.configure("SectionTitle.TLabel", background=colors["bg"], foreground=colors["text"], font=("Helvetica", 12, "bold"))
    style.configure("TNotebook", background=colors["bg"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=colors["panel"],
        foreground=colors["muted"],
        padding=(14, 8),
        font=("Helvetica", 11, "bold"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["accent"]), ("active", colors["accent_dark"])],
        foreground=[("selected", "#ffffff"), ("active", "#ffffff")],
    )
    style.configure("Accent.TButton", padding=(12, 6), font=("Helvetica", 10, "bold"))
    style.map(
        "Accent.TButton",
        background=[("active", colors["accent_dark"]), ("!disabled", colors["accent"])],
        foreground=[("!disabled", "#ffffff")],
    )
    style.configure("Accent.Horizontal.TProgressbar", troughcolor="#1f2937", bordercolor="#1f2937", background=colors["accent"])
    style.configure("Time.Horizontal.TProgressbar", troughcolor="#1f2937", bordercolor="#1f2937", background="#34d399")

    status_var = tk.StringVar(value="Ready.")
    header_subtitle_var = tk.StringVar()
    hf_var = tk.StringVar()
    action_buttons: list[Any] = []
    last_command_state: dict[str, str] = {"value": ""}

    header_bar = tk.Frame(root, bg=colors["header"], padx=14, pady=10)
    header_bar.pack(fill="x")

    title_frame = tk.Frame(header_bar, bg=colors["header"])
    title_frame.pack(side="left", fill="x", expand=True)
    tk.Label(
        title_frame,
        text="LeRobot Pipeline Manager",
        fg=colors["text"],
        bg=colors["header"],
        font=("Helvetica", 20, "bold"),
    ).pack(anchor="w")
    tk.Label(
        title_frame,
        textvariable=header_subtitle_var,
        fg=colors["muted"],
        bg=colors["header"],
        font=("Helvetica", 10),
    ).pack(anchor="w")

    status_frame = tk.Frame(header_bar, bg=colors["header"])
    status_frame.pack(side="right", anchor="e")
    status_dot_canvas = tk.Canvas(status_frame, width=16, height=16, bg=colors["header"], highlightthickness=0)
    status_dot_canvas.grid(row=0, column=0, padx=(0, 6))
    status_dot = status_dot_canvas.create_oval(2, 2, 14, 14, fill=colors["ready"], outline=colors["ready"])
    tk.Label(status_frame, textvariable=status_var, fg=colors["text"], bg=colors["header"], font=("Helvetica", 11, "bold")).grid(
        row=0,
        column=1,
        sticky="w",
    )
    tk.Label(status_frame, textvariable=hf_var, fg=colors["muted"], bg=colors["header"], font=("Helvetica", 9)).grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="e",
    )

    main_pane = tk.PanedWindow(
        root,
        orient="vertical",
        sashwidth=8,
        background=colors["bg"],
        bd=0,
    )
    main_pane.pack(fill="both", expand=True, padx=12, pady=(10, 8))

    notebook_host = ttk.Frame(main_pane, style="Panel.TFrame")
    output_host = ttk.Frame(main_pane, style="Panel.TFrame")
    main_pane.add(notebook_host, minsize=380)
    main_pane.add(output_host, minsize=240)

    notebook = ttk.Notebook(notebook_host)
    notebook.pack(fill="both", expand=True)
    managed_scroll_canvases: set[str] = set()

    def _find_managed_canvas(widget: Any) -> Any | None:
        current = widget
        while current is not None:
            if str(current) in managed_scroll_canvases:
                return current
            try:
                parent_name = current.winfo_parent()
            except Exception:
                break
            if not parent_name:
                break
            try:
                current = current.nametowidget(parent_name)
            except Exception:
                break
        return None

    def _on_mousewheel(event: Any) -> str | None:
        canvas = _find_managed_canvas(event.widget)
        if canvas is None:
            return None

        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
            return "break"
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
            return "break"

        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return None
        if abs(delta) >= 120:
            units = int(-delta / 120)
        else:
            units = -1 if delta > 0 else 1
        canvas.yview_scroll(units, "units")
        return "break"

    root.bind_all("<MouseWheel>", _on_mousewheel, add="+")
    root.bind_all("<Button-4>", _on_mousewheel, add="+")
    root.bind_all("<Button-5>", _on_mousewheel, add="+")

    def build_scroll_tab(title: str) -> tuple[Any, Any]:
        outer = ttk.Frame(notebook, style="Panel.TFrame")
        canvas = tk.Canvas(
            outer,
            bg=colors["bg"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        v_scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        content = ttk.Frame(canvas, style="Panel.TFrame", padding=12)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        managed_scroll_canvases.add(str(canvas))

        def sync_scroll_region(_: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_content_width(event: Any) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_content_width)

        bottom_spacer = ttk.Frame(content, style="Panel.TFrame")
        bottom_spacer.pack(side="bottom", fill="x")
        bottom_spacer.configure(height=320)
        bottom_spacer.pack_propagate(False)

        notebook.add(outer, text=title)
        return outer, content

    record_tab_outer, record_tab = build_scroll_tab("Record")
    deploy_tab_outer, deploy_tab = build_scroll_tab("Deploy")
    config_tab_outer, config_tab = build_scroll_tab("Config")

    output_panel = ttk.Frame(output_host, style="Panel.TFrame", padding=(0, 0, 0, 0))
    output_panel.pack(fill="both", expand=True)

    log_panel = GuiLogPanel(
        root=root,
        parent=output_panel,
        colors=colors,
        on_cancel=lambda: None,
        get_last_command=lambda: last_command_state["value"],
    )

    def refresh_header_subtitle() -> None:
        header_subtitle_var.set(
            "Follower {follower} | Leader {leader} | Cameras: laptop idx {laptop}, phone idx {phone} @ {w}x{h} {fps}fps".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                w=config.get("camera_width", 640),
                h=config.get("camera_height", 360),
                fps=config.get("camera_fps", 30),
            )
        )
        hf_var.set(f"Hugging Face: {config['hf_username']}")

    def set_status_dot(color: str) -> None:
        status_dot_canvas.itemconfig(status_dot, fill=color, outline=color)

    def confirm_preflight_in_gui(title: str, checks: list[tuple[str, str, str]]) -> bool:
        summary = summarize_checks(checks, title=title)
        if has_failures(checks):
            return ask_text_dialog(
                root=root,
                title="Preflight Failures",
                text=summary + "\n\nFAIL items detected. Continue anyway?",
                confirm_label="Continue",
                cancel_label="Cancel",
            )
        show_text_dialog(
            root=root,
            title="Preflight",
            text=summary,
            wrap_mode="word",
        )
        return True

    run_controller = create_run_controller(
        root=root,
        config=config,
        colors=colors,
        status_var=status_var,
        set_status_dot=set_status_dot,
        log_panel=log_panel,
        messagebox=messagebox,
        action_buttons=action_buttons,
        last_command_state=last_command_state,
    )
    log_panel.set_cancel_callback(run_controller.cancel_active_run)
    log_panel.set_is_running_callback(run_controller.is_running)

    def choose_folder(var: Any) -> None:
        selected = filedialog.askdirectory(
            initialdir=normalize_path(str(var.get() or Path.home())),
            title="Select folder",
        )
        if selected:
            var.set(normalize_path(selected))

    preview_handles: dict[str, Any] = {"record": None, "deploy": None}
    config_tab_handles: dict[str, Any] = {"handles": None}

    def on_camera_indices_changed(laptop_idx: int, phone_idx: int) -> None:
        laptop = int(laptop_idx)
        phone = int(phone_idx)
        changed = (
            int(config.get("camera_laptop_index", -1)) != laptop
            or int(config.get("camera_phone_index", -1)) != phone
        )
        config["camera_laptop_index"] = laptop
        config["camera_phone_index"] = phone
        if changed:
            save_config(config, quiet=True)
            log_panel.append_log(f"Saved camera mapping: laptop={laptop}, phone={phone}.")

        refresh_header_subtitle()
        record_preview = preview_handles.get("record")
        if record_preview is not None:
            record_preview.refresh_summary()
            record_preview.record_camera_preview.refresh_labels()
        deploy_preview = preview_handles.get("deploy")
        if deploy_preview is not None:
            deploy_preview.deploy_camera_preview.refresh_labels()
        cfg_handles = config_tab_handles.get("handles")
        if cfg_handles is not None:
            cfg_handles.sync_from_config()

    record_handles = setup_record_tab(
        root=root,
        record_tab=record_tab,
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        choose_folder=choose_folder,
        log_panel=log_panel,
        messagebox=messagebox,
        set_running=run_controller.set_running,
        run_process_async=run_controller.run_process_async,
        on_camera_indices_changed=on_camera_indices_changed,
        refresh_header_subtitle=refresh_header_subtitle,
        last_command_state=last_command_state,
        confirm_preflight_in_gui=confirm_preflight_in_gui,
    )
    preview_handles["record"] = record_handles
    action_buttons.extend(record_handles.action_buttons)

    deploy_handles = setup_deploy_tab(
        root=root,
        deploy_tab=deploy_tab,
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        choose_folder=choose_folder,
        log_panel=log_panel,
        messagebox=messagebox,
        set_running=run_controller.set_running,
        run_process_async=run_controller.run_process_async,
        on_camera_indices_changed=on_camera_indices_changed,
        refresh_header_subtitle=refresh_header_subtitle,
        last_command_state=last_command_state,
        confirm_preflight_in_gui=confirm_preflight_in_gui,
    )
    preview_handles["deploy"] = deploy_handles
    action_buttons.extend(deploy_handles.action_buttons)

    config_handles = setup_config_tab(
        root=root,
        config_tab=config_tab,
        config=config,
        choose_folder=choose_folder,
        log_panel=log_panel,
        messagebox=messagebox,
        refresh_header_subtitle=refresh_header_subtitle,
        refresh_record_summary=record_handles.refresh_summary,
        refresh_local_models=deploy_handles.refresh_local_models,
        record_camera_preview=record_handles.record_camera_preview,
        deploy_camera_preview=deploy_handles.deploy_camera_preview,
        record_dir_var=record_handles.record_dir_var,
        deploy_root_var=deploy_handles.deploy_root_var,
        deploy_eval_episodes_var=deploy_handles.deploy_eval_episodes_var,
        deploy_eval_duration_var=deploy_handles.deploy_eval_duration_var,
        deploy_eval_task_var=deploy_handles.deploy_eval_task_var,
    )
    config_tab_handles["handles"] = config_handles
    action_buttons.extend(config_handles.action_buttons)

    def on_tab_changed(_: Any) -> None:
        selected = notebook.select()
        if selected != str(record_tab_outer):
            record_handles.record_camera_preview.stop()
        if selected != str(deploy_tab_outer):
            deploy_handles.deploy_camera_preview.stop()

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    def on_close() -> None:
        record_handles.record_camera_preview.close()
        deploy_handles.deploy_camera_preview.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_header_subtitle()
    record_handles.refresh_summary()
    record_handles.record_camera_preview.refresh_labels()
    deploy_handles.deploy_camera_preview.refresh_labels()
    log_panel.append_log("GUI ready. Configure tabs, preview cameras, then run record/deploy.")

    def set_initial_split() -> None:
        try:
            total_h = max(root.winfo_height(), 760)
            main_pane.sash_place(0, 0, int(total_h * 0.60))
        except Exception:
            pass

    root.after(80, set_initial_split)
    root.mainloop()
