from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .app_icon import apply_tk_app_icon
from .artifacts import list_runs
from .checks import has_failures, summarize_checks
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_config_without_prompts, save_config
from .gui_async import UiBackgroundJobs
from .gui_config_tab import setup_config_tab
from .gui_deploy_tab import setup_deploy_tab
from .gui_dialogs import ask_text_dialog
from .gui_file_dialogs import ask_directory_dialog
from .gui_history_tab import open_path_in_file_manager, setup_history_tab
from .gui_log import GuiLogPanel
from .gui_record_tab import setup_record_tab
from .gui_runner import create_run_controller
from .gui_scroll import at_scroll_edge, bind_canvas_scroll_recursive, scroll_widget_yview, widget_yview, wheel_units
from .gui_teleop_tab import setup_teleop_tab
from .gui_terminal_shell import GuiTerminalShell
from .gui_training_tab import setup_training_tab
from .gui_theme import apply_gui_theme
from .gui_tokens import normalize_theme_mode
from .gui_visualizer_tab import setup_visualizer_tab
from .gui_window import fit_window_to_screen
from .probes import probe_module_import
from .repo_utils import normalize_deploy_rerun_command


def _apply_runtime_theme_to_components(
    *,
    colors: dict[str, str],
    log_panel: Any,
    preview_handles: dict[str, Any],
    training_handles_ref: dict[str, Any],
    config_tab_handles: dict[str, Any],
    visualizer_handles_ref: dict[str, Any],
    history_handles_ref: dict[str, Any],
) -> None:
    log_panel.apply_theme(colors)
    for key in ("record", "deploy", "teleop"):
        preview = preview_handles.get(key)
        if preview is not None and hasattr(preview, "apply_theme"):
            preview.apply_theme(colors)
    train_handles = training_handles_ref.get("handles")
    if train_handles is not None and hasattr(train_handles, "apply_theme"):
        train_handles.apply_theme(colors)
    cfg_handles = config_tab_handles.get("handles")
    if cfg_handles is not None and hasattr(cfg_handles, "apply_theme"):
        cfg_handles.apply_theme(colors)
    viz_handles = visualizer_handles_ref.get("handles")
    if viz_handles is not None and hasattr(viz_handles, "apply_theme"):
        viz_handles.apply_theme(colors)
    hist_handles = history_handles_ref.get("handles")
    if hist_handles is not None and hasattr(hist_handles, "apply_theme"):
        hist_handles.apply_theme(colors)


def _schedule_shutdown_after_cancel(
    *,
    root: Any,
    has_active_process: Callable[[], bool],
    request_cancel: Callable[[], None],
    finalize_shutdown: Callable[[], None],
    log: Callable[[str], None] | None = None,
    timeout_s: float = 3.0,
    poll_interval_ms: int = 50,
    monotonic: Callable[[], float] | None = None,
) -> None:
    log_fn = log or (lambda _msg: None)
    monotonic_fn = monotonic or time.monotonic
    if not has_active_process():
        finalize_shutdown()
        return

    log_fn("[on_close] Active subprocess detected — sending cancel signal.")
    request_cancel()
    deadline = monotonic_fn() + max(float(timeout_s), 0.0)

    def _poll_cancel_completion() -> None:
        if not has_active_process():
            log_fn("[on_close] Subprocess exited cleanly.")
            finalize_shutdown()
            return
        if monotonic_fn() >= deadline:
            log_fn("[on_close] Subprocess did not exit within timeout — proceeding with destroy.")
            finalize_shutdown()
            return
        root.after(max(int(poll_interval_ms), 1), _poll_cancel_completion)

    root.after(max(int(poll_interval_ms), 1), _poll_cancel_completion)


def run_gui_mode(raw_config: dict[str, Any]) -> None:
    try:
        import tkinter as tk
        import tkinter.font as tkfont
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
    apply_tk_app_icon(root=root, tk_module=tk)
    root.title("LeRobot Pipeline Manager")
    fit_window_to_screen(
        window=root,
        requested_width=1240,
        requested_height=900,
        requested_min_width=1080,
        requested_min_height=760,
    )

    theme_mode_var = tk.StringVar(value=normalize_theme_mode(config.get("ui_theme_mode", "dark")))
    config["ui_theme_mode"] = theme_mode_var.get()
    colors = apply_gui_theme(root=root, tkfont=tkfont, ttk=ttk, theme_mode=theme_mode_var.get())
    ui_font = colors["font_ui"]

    status_var = tk.StringVar(value="Ready.")
    header_subtitle_var = tk.StringVar()
    hf_var = tk.StringVar()
    action_buttons: list[Any] = []
    last_command_state: dict[str, str] = {"value": ""}

    header_bar = tk.Frame(root, bg=colors["header"], padx=14, pady=10)
    header_bar.pack(fill="x")

    title_frame = tk.Frame(header_bar, bg=colors["header"])
    title_frame.pack(side="left", fill="x", expand=True)

    # "LeRobot" in yellow, " Pipeline Manager" in white — two labels side-by-side
    title_row = tk.Frame(title_frame, bg=colors["header"])
    title_row.pack(anchor="w")
    title_brand_label = tk.Label(
        title_row,
        text="LeRobot",
        fg=colors["accent"],
        bg=colors["header"],
        font=(ui_font, 20, "bold"),
    )
    title_brand_label.pack(side="left")
    title_suffix_label = tk.Label(
        title_row,
        text=" Pipeline Manager",
        fg=colors["text"],
        bg=colors["header"],
        font=(ui_font, 20, "bold"),
    )
    title_suffix_label.pack(side="left")

    subtitle_label = tk.Label(
        title_frame,
        textvariable=header_subtitle_var,
        fg=colors["muted"],
        bg=colors["header"],
        font=(ui_font, 10),
    )
    subtitle_label.pack(anchor="w")

    # Last-run indicator: "Last: record · success · 2m ago"
    last_run_label = tk.Label(
        title_frame,
        text="",
        fg=colors["muted"],
        bg=colors["header"],
        font=(ui_font, 9),
    )
    last_run_label.pack(anchor="w")

    status_frame = tk.Frame(header_bar, bg=colors["header"])
    status_frame.pack(side="right", anchor="e")
    status_dot_canvas = tk.Canvas(status_frame, width=16, height=16, bg=colors["header"], highlightthickness=0)
    status_dot_canvas.grid(row=0, column=0, padx=(0, 6))
    status_dot = status_dot_canvas.create_oval(2, 2, 14, 14, fill=colors["ready"], outline=colors["ready"])
    status_text_label = tk.Label(status_frame, textvariable=status_var, fg=colors["text"], bg=colors["header"], font=(ui_font, 11, "bold"))
    status_text_label.grid(
        row=0,
        column=1,
        sticky="w",
    )
    hf_text_label = tk.Label(status_frame, textvariable=hf_var, fg=colors["muted"], bg=colors["header"], font=(ui_font, 9))
    hf_text_label.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="e",
    )
    terminal_toggle_header_button = ttk.Button(
        status_frame,
        text="Hide Terminal",
        style="Secondary.TButton",
    )
    terminal_toggle_header_button.grid(row=2, column=0, columnspan=2, sticky="e", pady=(6, 0))

    theme_toggle_button = ttk.Button(
        status_frame,
        text="Switch to Light",
        style="Secondary.TButton",
    )
    theme_toggle_button.grid(row=3, column=0, columnspan=2, sticky="e", pady=(6, 0))
    main_pane = tk.PanedWindow(
        root,
        orient="vertical",
        sashwidth=6,
        sashrelief="flat",
        background=colors["border"],
        bd=0,
    )
    main_pane.pack(fill="both", expand=True, padx=12, pady=(10, 8))

    notebook_host = ttk.Frame(main_pane, style="Panel.TFrame")
    output_host = ttk.Frame(main_pane, style="Panel.TFrame")
    main_pane.add(notebook_host, minsize=380)
    main_pane.add(output_host, minsize=240)

    notebook = ttk.Notebook(notebook_host)
    notebook.pack(fill="both", expand=True)
    managed_scroll_canvases: dict[str, Any] = {}
    canvas_by_outer: dict[str, Any] = {}
    scroll_content_by_canvas: dict[str, Any] = {}

    def _find_managed_canvas(widget: Any) -> Any | None:
        current = widget
        while current is not None:
            canvas = managed_scroll_canvases.get(str(current))
            if canvas is not None:
                return canvas
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

    def _widget_class_name(widget: Any) -> str:
        try:
            return str(widget.winfo_class()).lower()
        except Exception:
            return ""

    def _find_scrollable_widget(widget: Any) -> Any | None:
        current = widget
        while current is not None:
            if widget_yview(current) is not None:
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

    def _selected_tab_canvas() -> Any | None:
        try:
            selected = str(notebook.select())
        except Exception:
            return None
        return canvas_by_outer.get(selected)

    def _on_mousewheel(event: Any) -> str | None:
        units = wheel_units(event)
        if units == 0:
            return None

        event_widget = getattr(event, "widget", None)
        try:
            pointer_widget = root.winfo_containing(getattr(event, "x_root", 0), getattr(event, "y_root", 0))
        except Exception:
            pointer_widget = None
        if pointer_widget is not None:
            event_widget = pointer_widget

        target = _find_scrollable_widget(event_widget)
        if target is not None:
            target_class = _widget_class_name(target)
            # Canvas widgets generally need explicit wheel scrolling.
            if target_class == "canvas":
                if scroll_widget_yview(target, units) or (sys.platform == "darwin" and scroll_widget_yview(target, -units)):
                    return "break"
            # For native scroll widgets (Text/Treeview), only hand off when
            # they are already at an edge in the scroll direction.
            elif sys.platform == "darwin":
                # macOS trackpads can route wheel events through bind_all with
                # an unexpected event.widget, so scroll explicitly when possible.
                if scroll_widget_yview(target, units) or scroll_widget_yview(target, -units):
                    return "break"
            elif not at_scroll_edge(target, units):
                return None

        fallback_canvas = _find_managed_canvas(event_widget) or _selected_tab_canvas()
        if (
            fallback_canvas is not None
            and fallback_canvas is not target
            and (
                scroll_widget_yview(fallback_canvas, units)
                or (sys.platform == "darwin" and scroll_widget_yview(fallback_canvas, -units))
            )
        ):
            return "break"
        return None

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
        canvas.pack(side="left", fill="both", expand=True)

        content = ttk.Frame(canvas, style="Panel.TFrame", padding=12)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        managed_scroll_canvases[str(canvas)] = canvas
        scroll_content_by_canvas[str(canvas)] = content
        canvas_by_outer[str(outer)] = canvas

        def sync_scroll_region(_: Any = None) -> None:
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        if sys.platform == "darwin":
            _bind_job: dict[str, Any] = {"id": None}

            def _schedule_canvas_bindings() -> None:
                pending = _bind_job.get("id")
                if pending is not None:
                    try:
                        root.after_cancel(pending)
                    except Exception:
                        pass

                def _apply() -> None:
                    _bind_job["id"] = None
                    try:
                        bind_canvas_scroll_recursive(canvas, content)
                    except Exception:
                        pass

                _bind_job["id"] = root.after(120, _apply)

            content.bind("<Configure>", lambda *_: _schedule_canvas_bindings(), add="+")
            root.after(0, _schedule_canvas_bindings)

        def sync_content_width(event: Any) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_content_width)
        root.after(250, sync_scroll_region)

        bottom_spacer = ttk.Frame(content, style="Panel.TFrame")
        bottom_spacer.pack(side="bottom", fill="x")
        bottom_spacer.configure(height=320)
        bottom_spacer.pack_propagate(False)

        notebook.add(outer, text=title)
        return outer, content

    record_tab_outer, record_tab = build_scroll_tab("Record")
    deploy_tab_outer, deploy_tab = build_scroll_tab("Deploy")
    teleop_tab_outer, teleop_tab = build_scroll_tab("Teleop")
    training_tab_outer, training_tab = build_scroll_tab("Training")
    visualizer_tab_outer, visualizer_tab = build_scroll_tab("Visualizer")
    config_tab_outer, config_tab = build_scroll_tab("Config")
    history_tab = ttk.Frame(notebook, style="Panel.TFrame")
    notebook.add(history_tab, text="History")

    output_panel = ttk.Frame(output_host, style="Panel.TFrame", padding=(0, 0, 0, 0))
    output_panel.pack(fill="both", expand=True)

    log_panel = GuiLogPanel(
        root=root,
        parent=output_panel,
        colors=colors,
        on_cancel=lambda: None,
        get_last_command=lambda: last_command_state["value"],
    )

    # Both panes start in the split view, so initial state is visible=True.
    # The startup call to set_terminal_visible(False) will then correctly hide it.
    terminal_state: dict[str, bool] = {"visible": True}

    def _style_terminal_toggle(visible: bool) -> None:
        if visible:
            terminal_toggle_header_button.configure(
                text="Hide Terminal",
                style="Accent.TButton",
            )
        else:
            terminal_toggle_header_button.configure(
                text="Show Terminal",
                style="Secondary.TButton",
            )

    def set_terminal_visible(visible: bool) -> None:
        target = bool(visible)
        current = bool(terminal_state["visible"])
        if target != current:
            try:
                if target:
                    main_pane.add(output_host, minsize=240)
                    total_h = max(root.winfo_height(), 760)
                    main_pane.sash_place(0, 0, int(total_h * 0.60))
                else:
                    main_pane.forget(output_host)
            except Exception:
                pass
            terminal_state["visible"] = target

        _style_terminal_toggle(target)
        log_panel.set_terminal_visible(target)

    def toggle_terminal_visibility() -> None:
        set_terminal_visible(not bool(terminal_state["visible"]))

    def _refresh_theme_button_text() -> None:
        if theme_mode_var.get() == "light":
            theme_toggle_button.configure(text="Switch to Dark")
        else:
            theme_toggle_button.configure(text="Switch to Light")

    def _apply_theme_to_header_widgets() -> None:
        root.configure(bg=colors["bg"])
        header_bar.configure(bg=colors["header"])
        title_frame.configure(bg=colors["header"])
        title_row.configure(bg=colors["header"])
        title_brand_label.configure(bg=colors["header"], fg=colors["accent"], font=(colors["font_ui"], 20, "bold"))
        title_suffix_label.configure(bg=colors["header"], fg=colors["text"], font=(colors["font_ui"], 20, "bold"))
        subtitle_label.configure(bg=colors["header"], fg=colors["muted"], font=(colors["font_ui"], 10))
        status_frame.configure(bg=colors["header"])
        status_dot_canvas.configure(bg=colors["header"])
        last_run_label.configure(bg=colors["header"], fg=colors["muted"], font=(colors["font_ui"], 9))
        status_text_label.configure(bg=colors["header"], fg=colors["text"], font=(colors["font_ui"], 11, "bold"))
        hf_text_label.configure(bg=colors["header"], fg=colors["muted"], font=(colors["font_ui"], 9))
        main_pane.configure(background=colors["border"])
        _style_terminal_toggle(bool(terminal_state["visible"]))

    def set_theme_mode(mode: str, *, persist: bool = True) -> None:
        normalized_mode = normalize_theme_mode(mode)
        theme_mode_var.set(normalized_mode)
        updated = apply_gui_theme(root=root, tkfont=tkfont, ttk=ttk, theme_mode=normalized_mode)
        colors.clear()
        colors.update(updated)
        config["ui_theme_mode"] = normalized_mode
        _apply_theme_to_header_widgets()
        _apply_runtime_theme_to_components(
            colors=colors,
            log_panel=log_panel,
            preview_handles=preview_handles,
            training_handles_ref=training_handles_ref,
            config_tab_handles=config_tab_handles,
            visualizer_handles_ref=visualizer_handles_ref,
            history_handles_ref=history_handles_ref,
        )
        _refresh_theme_button_text()
        status_var.set("Theme updated.")
        root.after(1000, lambda: status_var.set("Ready.") if not run_controller.has_active_process() else None)
        if persist:
            save_config(config, quiet=True)

    def toggle_theme_mode() -> None:
        set_theme_mode("light" if theme_mode_var.get() == "dark" else "dark", persist=True)

    history_handles_ref: dict[str, Any] = {"handles": None}
    visualizer_handles_ref: dict[str, Any] = {"handles": None}

    def refresh_history_if_ready() -> None:
        handles = history_handles_ref.get("handles")
        if handles is not None:
            handles.refresh()

    run_controller_ref: dict[str, Any] = {"controller": None}

    def is_pipeline_active() -> bool:
        controller = run_controller_ref.get("controller")
        if controller is None:
            return False
        return bool(controller.has_active_process())

    def send_pipeline_stdin(text: str) -> tuple[bool, str]:
        controller = run_controller_ref.get("controller")
        if controller is None:
            return False, "No active process is available."
        ok, message = controller.send_stdin(text)
        if ok:
            return True, ""
        return False, message

    shell_manager = GuiTerminalShell(
        root=root,
        config=config,
        append_log=log_panel.append_log,
        is_pipeline_active=is_pipeline_active,
        send_pipeline_stdin=send_pipeline_stdin,
        on_artifact_written=refresh_history_if_ready,
    )

    def refresh_header_subtitle() -> None:
        header_subtitle_var.set(
            "Follower {follower} | Leader {leader} | Cameras: laptop idx {laptop}, phone idx {phone} @ {fps}fps (auto-size)".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                fps=config.get("camera_fps", 30),
            )
        )
        hf_var.set(f"Hugging Face: {config['hf_username']}")

    _pulse_job: dict[str, str | None] = {"job": None}
    _dot_bright: dict[str, bool] = {"value": True}

    def _stop_pulse() -> None:
        job = _pulse_job.get("job")
        if job is not None:
            root.after_cancel(job)
            _pulse_job["job"] = None

    def _pulse_running_dot() -> None:
        _stop_pulse()
        color = colors["running"] if _dot_bright["value"] else colors.get("running_dim", colors["running"])
        status_dot_canvas.itemconfig(status_dot, fill=color, outline=color)
        _dot_bright["value"] = not _dot_bright["value"]
        _pulse_job["job"] = root.after(600, _pulse_running_dot)

    def set_status_dot(color: str) -> None:
        _stop_pulse()
        status_dot_canvas.itemconfig(status_dot, fill=color, outline=color)
        if color == colors["running"]:
            _dot_bright["value"] = True
            _pulse_running_dot()

    def confirm_preflight_in_gui(title: str, checks: list[tuple[str, str, str]]) -> bool:
        summary = summarize_checks(checks, title=title)
        if has_failures(checks):
            prompt = summary + "\n\nFAIL items detected.\nClick Confirm to continue anyway, or Cancel to stop."
            confirm_label = "Confirm"
            dialog_title = "Preflight Failures"
        else:
            prompt = summary + "\n\nPreflight complete.\nClick Confirm to continue, or Cancel to stop."
            confirm_label = "Confirm"
            dialog_title = "Preflight Review"
        return ask_text_dialog(
            root=root,
            title=dialog_title,
            text=prompt,
            confirm_label=confirm_label,
            cancel_label="Cancel",
            wrap_mode="char",
        )

    def on_run_failure() -> None:
        if not terminal_state["visible"]:
            set_terminal_visible(True)
        log_panel.scroll_to_first_error()

    def _update_last_run_indicator() -> None:
        try:
            from datetime import datetime, timezone as _tz
            runs, _ = list_runs(config=config, limit=1)
            if not runs:
                return
            item = runs[0]
            mode = str(item.get("mode", "run"))
            canceled = bool(item.get("canceled"))
            if canceled:
                status, color = "canceled", colors["muted"]
            else:
                try:
                    code = int(item.get("exit_code", -1))
                    if code == 0:
                        status, color = "success", colors["success"]
                    else:
                        status, color = "failed", colors["error"]
                except Exception:
                    status, color = "?", colors["muted"]

            ended = str(item.get("ended_at_iso", ""))
            ago = ""
            if ended:
                try:
                    dt = datetime.fromisoformat(ended)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_tz.utc)
                    secs = int((datetime.now(_tz.utc) - dt).total_seconds())
                    if secs < 60:
                        ago = f"  {secs}s ago"
                    elif secs < 3600:
                        ago = f"  {secs // 60}m ago"
                    elif secs < 86400:
                        ago = f"  {secs // 3600}h ago"
                    else:
                        ago = f"  {secs // 86400}d ago"
                except Exception:
                    pass

            last_run_label.configure(text=f"Last: {mode}  ·  {status}{ago}", fg=color)
        except Exception:
            pass

    def _on_artifact_written() -> None:
        refresh_history_if_ready()
        _update_last_run_indicator()

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
        external_busy=shell_manager.is_busy,
        on_run_failure=on_run_failure,
        on_artifact_written=_on_artifact_written,
    )
    run_controller_ref["controller"] = run_controller
    log_panel.set_cancel_callback(run_controller.cancel_active_run)
    log_panel.set_is_running_callback(run_controller.has_active_process)
    log_panel.set_submit_callback(shell_manager.handle_terminal_submit)
    log_panel.set_interrupt_callback(shell_manager.send_interrupt)

    def choose_folder(var: Any) -> None:
        selected = ask_directory_dialog(
            root=root,
            filedialog=filedialog,
            initial_dir=str(var.get() or ""),
            title="Select folder",
        )
        if selected:
            var.set(selected)


    background_jobs = UiBackgroundJobs(root)

    preview_handles: dict[str, Any] = {"record": None, "deploy": None}
    config_tab_handles: dict[str, Any] = {"handles": None}
    training_handles_ref: dict[str, Any] = {"handles": None}

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
        teleop_preview = preview_handles.get("teleop")
        if teleop_preview is not None:
            teleop_preview.refresh_summary()
            teleop_preview.teleop_camera_preview.refresh_labels()
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
        background_jobs=background_jobs,
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
        background_jobs=background_jobs,
    )
    preview_handles["deploy"] = deploy_handles
    action_buttons.extend(deploy_handles.action_buttons)

    teleop_handles = setup_teleop_tab(
        root=root,
        teleop_tab=teleop_tab,
        config=config,
        colors=colors,
        cv2_probe_ok=cv2_probe_ok,
        cv2_probe_error=cv2_probe_error,
        log_panel=log_panel,
        messagebox=messagebox,
        set_running=run_controller.set_running,
        run_process_async=run_controller.run_process_async,
        on_camera_indices_changed=on_camera_indices_changed,
        refresh_header_subtitle=refresh_header_subtitle,
        last_command_state=last_command_state,
        confirm_preflight_in_gui=confirm_preflight_in_gui,
        background_jobs=background_jobs,
    )
    preview_handles["teleop"] = teleop_handles
    action_buttons.extend(teleop_handles.action_buttons)

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
        run_terminal_command=shell_manager.run_command_from_history,
        show_terminal=lambda: set_terminal_visible(True),
    )
    config_tab_handles["handles"] = config_handles
    action_buttons.extend(config_handles.action_buttons)

    training_handles = setup_training_tab(
        root=root,
        training_tab=training_tab,
        config=config,
        colors=colors,
        filedialog=filedialog,
        log_panel=log_panel,
        messagebox=messagebox,
        run_process_async=run_controller.run_process_async,
        set_running=run_controller.set_running,
        last_command_state=last_command_state,
        confirm_preflight_in_gui=confirm_preflight_in_gui,
    )
    training_handles_ref["handles"] = training_handles
    action_buttons.extend(training_handles.action_buttons)

    visualizer_handles = setup_visualizer_tab(
        root=root,
        visualizer_tab=visualizer_tab,
        config=config,
        colors=colors,
        log_panel=log_panel,
        messagebox=messagebox,
        background_jobs=background_jobs,
    )
    visualizer_handles_ref["handles"] = visualizer_handles

    def rerun_pipeline_command(
        cmd: list[str],
        cwd: Path | None,
        run_mode: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        if shell_manager.is_busy():
            return False, "A shell command is currently running. Wait for it to finish first."
        if run_controller.has_active_process():
            return False, "Another command is already running."

        rerun_cmd = list(cmd)
        rerun_context = dict(context)
        if run_mode == "deploy":
            rerun_cmd, rerun_message = normalize_deploy_rerun_command(
                command_argv=rerun_cmd,
                username=str(config.get("hf_username", "")),
                local_roots=[get_deploy_data_dir(config), get_lerobot_dir(config) / "data"],
            )
            if rerun_message:
                log_panel.append_log(rerun_message)
                for arg in rerun_cmd:
                    if str(arg).startswith("--dataset.repo_id="):
                        rerun_context["dataset_repo_id"] = str(arg).split("=", 1)[1].strip()
                        break
        run_controller.run_process_async(
            rerun_cmd,
            cwd,
            None,
            None,
            None,
            run_mode,
            None,
            rerun_context,
        )
        return True, f"Started rerun for {run_mode}."

    def rerun_shell_command(command: str) -> tuple[bool, str]:
        if run_controller.has_active_process():
            return False, "Cannot start shell rerun while record/deploy is active."
        return shell_manager.run_command_from_history(command)

    history_handles = setup_history_tab(
        root=root,
        notebook=notebook,
        history_tab=history_tab,
        config=config,
        colors=colors,
        log_panel=log_panel,
        messagebox=messagebox,
        on_rerun_pipeline=rerun_pipeline_command,
        on_rerun_shell=rerun_shell_command,
        background_jobs=background_jobs,
    )
    history_handles_ref["handles"] = history_handles

    # Final pass after all tabs are built, in case any widgets were added late.
    if sys.platform == "darwin":
        for _scroll_canvas in managed_scroll_canvases.values():
            try:
                _content = scroll_content_by_canvas.get(str(_scroll_canvas))
                if _content is not None:
                    bind_canvas_scroll_recursive(_scroll_canvas, _content)
            except Exception:
                pass

    def open_latest_artifact() -> None:
        runs, _ = list_runs(config=config, limit=1)
        if not runs:
            messagebox.showinfo("Run Artifacts", "No run artifacts found yet.")
            return
        latest_path = Path(str(runs[0].get("_run_path", "")).strip())
        ok, message = open_path_in_file_manager(latest_path)
        if not ok:
            messagebox.showerror("Open Latest Artifact", message)

    log_panel.set_toggle_terminal_callback(toggle_terminal_visibility)
    log_panel.set_show_history_callback(history_handles.select_tab)
    log_panel.set_open_latest_artifact_callback(open_latest_artifact)
    terminal_toggle_header_button.configure(command=toggle_terminal_visibility)
    theme_toggle_button.configure(command=toggle_theme_mode)
    set_theme_mode(theme_mode_var.get(), persist=False)

    def on_tab_changed(_: Any) -> None:
        selected = notebook.select()
        if selected != str(record_tab_outer):
            record_handles.record_camera_preview.stop()
        if selected != str(deploy_tab_outer):
            deploy_handles.deploy_camera_preview.stop()
        if selected != str(teleop_tab_outer):
            teleop_handles.teleop_camera_preview.stop()
        if sys.platform == "darwin":
            selected_canvas = canvas_by_outer.get(str(selected))
            if selected_canvas is not None:
                selected_content = scroll_content_by_canvas.get(str(selected_canvas))
                if selected_content is not None:
                    try:
                        bind_canvas_scroll_recursive(selected_canvas, selected_content)
                    except Exception:
                        pass

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    close_state: dict[str, bool] = {"in_progress": False}

    def _finalize_close() -> None:
        shell_manager.shutdown()
        background_jobs.shutdown()
        root.destroy()

    def on_close() -> None:
        if close_state["in_progress"]:
            return
        close_state["in_progress"] = True
        record_handles.record_camera_preview.close()
        deploy_handles.deploy_camera_preview.close()
        teleop_handles.teleop_camera_preview.close()
        _schedule_shutdown_after_cancel(
            root=root,
            has_active_process=run_controller.has_active_process,
            request_cancel=run_controller.cancel_active_run,
            finalize_shutdown=_finalize_close,
            log=print,
        )

    root.protocol("WM_DELETE_WINDOW", on_close)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    # On macOS the Command key maps to Meta in Tkinter; use Ctrl on other platforms.
    _is_mac = os.path.exists("/usr/bin/osascript")
    _mod = "Meta" if _is_mac else "Control"

    def _focus_terminal() -> None:
        if not terminal_state["visible"]:
            set_terminal_visible(True)
        log_panel.focus_input()

    def _guarded(fn: Any) -> Any:
        """Only fire shortcut if a text input widget does not have focus."""
        def _handler(event: Any) -> str | None:
            w = root.focus_get()
            if isinstance(w, tk.Text):
                return None
            fn()
            return "break"
        return _handler

    root.bind_all(f"<{_mod}-Key-1>", _guarded(lambda: notebook.select(record_tab_outer)))
    root.bind_all(f"<{_mod}-Key-2>", _guarded(lambda: notebook.select(deploy_tab_outer)))
    root.bind_all(f"<{_mod}-Key-3>", _guarded(lambda: notebook.select(teleop_tab_outer)))
    root.bind_all(f"<{_mod}-Key-4>", _guarded(lambda: notebook.select(training_tab_outer)))
    root.bind_all(f"<{_mod}-Key-5>", _guarded(lambda: notebook.select(visualizer_tab_outer)))
    root.bind_all(f"<{_mod}-Key-6>", _guarded(lambda: notebook.select(config_tab_outer)))
    root.bind_all(f"<{_mod}-Key-7>", _guarded(history_handles.select_tab))
    root.bind_all("<F2>", _guarded(_focus_terminal))

    refresh_header_subtitle()
    record_handles.refresh_summary()
    teleop_handles.refresh_summary()
    record_handles.record_camera_preview.refresh_labels()
    deploy_handles.deploy_camera_preview.refresh_labels()
    teleop_handles.teleop_camera_preview.refresh_labels()
    set_terminal_visible(False)
    _update_last_run_indicator()
    _shortcut_label = "Cmd" if _is_mac else "Ctrl"
    log_panel.append_log(
        f"GUI ready.  Shortcuts: {_shortcut_label}+1/2/3/4/5/6/7 = tabs  |  F2 = focus terminal  |  Copy Command = last run cmd"
    )

    def set_initial_split() -> None:
        try:
            total_h = max(root.winfo_height(), 760)
            main_pane.sash_place(0, 0, int(total_h * 0.60))
        except Exception:
            pass

    root.after(80, set_initial_split)
    root.mainloop()
