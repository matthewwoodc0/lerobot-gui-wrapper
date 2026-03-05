from __future__ import annotations

import shlex
from typing import Any

from .gui_scroll import TOUCHPAD_SCROLL_SEQ, wheel_units
from .gui_window import fit_window_to_screen


def _style_lookup(style: Any, style_name: str, option: str, fallback: str) -> str:
    try:
        value = style.lookup(style_name, option)
    except Exception:
        value = ""
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _style_map_lookup(style: Any, style_name: str, option: str, state_token: str, fallback: str) -> str:
    try:
        entries = style.map(style_name, option)
    except Exception:
        entries = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        states, mapped_value = entry
        if isinstance(states, (tuple, list)):
            state_text = " ".join(str(item) for item in states)
        else:
            state_text = str(states)
        if state_token in state_text:
            return str(mapped_value)
    for entry in entries:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            return str(entry[1])
    return fallback


def _dialog_theme(root: Any) -> dict[str, str]:
    import tkinter.font as tkfont
    from tkinter import ttk

    style = ttk.Style(root)
    panel = _style_lookup(style, "Panel.TFrame", "background", "#121212")
    surface = _style_lookup(style, "TEntry", "fieldbackground", "#1b1b1b")
    border = _style_lookup(style, "TEntry", "bordercolor", "#303030")
    text = _style_lookup(style, "TLabel", "foreground", "#f2f2f2")
    muted = _style_lookup(style, "Muted.TLabel", "foreground", "#8f8f8f")
    accent = _style_lookup(style, "Accent.TButton", "background", "#f0a500")
    accent_active = _style_map_lookup(style, "Accent.TButton", "background", "active", accent)
    danger = _style_lookup(style, "Danger.TButton", "background", "#ef4444")
    danger_active = _style_map_lookup(style, "Danger.TButton", "background", "active", danger)
    surface_alt = _style_map_lookup(style, "TButton", "background", "active", surface)

    try:
        ui_font = str(tkfont.nametofont("TkDefaultFont").cget("family"))
    except Exception:
        ui_font = "TkDefaultFont"
    try:
        mono_font = str(tkfont.nametofont("TkFixedFont").cget("family"))
    except Exception:
        mono_font = "TkFixedFont"

    return {
        "panel": panel,
        "surface": surface,
        "surface_alt": surface_alt,
        "border": border,
        "text": text,
        "muted": muted,
        "accent": accent,
        "accent_active": accent_active,
        "danger": danger,
        "danger_active": danger_active,
        "font_ui": ui_font,
        "font_mono": mono_font,
    }


def format_command_for_dialog(cmd: list[str]) -> str:
    if not cmd:
        return "(empty command)"
    shell_safe = shlex.join(cmd)
    lines = [
        "Shell-safe command (copy/paste):",
        shell_safe,
        "",
        "Exact argv passed to subprocess (no shell quoting here):",
    ]
    for idx, arg in enumerate(cmd):
        lines.append(f"[{idx}] {arg}")
    return "\n".join(lines)


def parse_command_text(command_text: str) -> tuple[list[str] | None, str | None]:
    raw = str(command_text or "").strip()
    if not raw:
        return None, "Command is empty."
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        return None, f"Unable to parse command: {exc}"
    if not parts:
        return None, "Command is empty."
    return [str(part) for part in parts], None


def _bind_text_wheel_scroll(text_widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = wheel_units(event)
        if units == 0:
            return None
        text_widget.yview_scroll(units, "units")
        return "break"

    text_widget.bind("<MouseWheel>", on_wheel, add="+")
    text_widget.bind("<Button-4>", on_wheel, add="+")
    text_widget.bind("<Button-5>", on_wheel, add="+")
    try:
        text_widget.bind(TOUCHPAD_SCROLL_SEQ, on_wheel, add="+")
    except Exception:
        pass


def show_text_dialog(
    *,
    root: Any,
    title: str,
    text: str,
    copy_text: str | None = None,
    width: int = 980,
    height: int = 520,
    wrap_mode: str = "none",
) -> None:
    import tkinter as tk
    from tkinter import ttk

    theme = _dialog_theme(root)
    dialog_bg = theme["panel"]
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    fit_window_to_screen(
        window=window,
        requested_width=width,
        requested_height=height,
        requested_min_width=700,
        requested_min_height=360,
    )
    window.transient(root)
    window.grab_set()
    window.lift()
    window.focus_force()

    # Footer first to guarantee it gets space
    buttons = tk.Frame(window, bg=dialog_bg, pady=12, padx=16)
    buttons.pack(side="bottom", fill="x")

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg=theme["surface"],
        fg=theme["text"],
        insertbackground=theme["text"],
        relief="flat",
        font=(theme["font_mono"], 10),
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=theme["border"],
    )
    text_widget.grid(row=0, column=0, sticky="nsew")
    _bind_text_wheel_scroll(text_widget)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    def copy_to_clipboard() -> None:
        payload = text if copy_text is None else str(copy_text)
        root.clipboard_clear()
        root.clipboard_append(payload)

    tk.Button(
        buttons,
        text="Copy",
        command=copy_to_clipboard,
        padx=12,
        pady=8,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    ).pack(side="left")

    tk.Button(
        buttons,
        text="Close",
        command=window.destroy,
        padx=12,
        pady=8,
        bg=theme["accent"],
        fg="#000000",
        activebackground=theme["accent_active"],
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=(theme["font_ui"], 10, "bold"),
    ).pack(side="right")

    window.bind("<Escape>", lambda _: window.destroy())
    window.wait_window()


def ask_editable_command_dialog(
    *,
    root: Any,
    title: str,
    command_argv: list[str],
    intro_text: str,
    confirm_label: str = "Run",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 540,
) -> list[str] | None:
    import tkinter as tk
    from tkinter import ttk

    theme = _dialog_theme(root)
    dialog_bg = theme["panel"]
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    fit_window_to_screen(
        window=window,
        requested_width=width,
        requested_height=height,
        requested_min_width=700,
        requested_min_height=420,
    )
    window.transient(root)
    window.grab_set()
    window.lift()
    window.focus_force()

    initial_text = shlex.join([str(part) for part in command_argv if str(part)]).strip()
    result: dict[str, list[str] | None] = {"value": None}

    footer = tk.Frame(window, bg=dialog_bg, padx=16, pady=14)
    footer.pack(side="bottom", fill="x")

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(1, weight=1)
    body.columnconfigure(0, weight=1)

    intro_label = ttk.Label(
        body,
        text=intro_text,
        style="Muted.TLabel",
        justify="left",
        anchor="w",
    )
    intro_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    text_widget = tk.Text(
        body,
        wrap="none",
        bg=theme["surface"],
        fg=theme["text"],
        insertbackground=theme["text"],
        relief="flat",
        font=(theme["font_mono"], 10),
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=theme["border"],
    )
    text_widget.grid(row=1, column=0, sticky="nsew")
    _bind_text_wheel_scroll(text_widget)
    text_widget.insert("1.0", initial_text)
    text_widget.see("1.0")

    error_var = tk.StringVar(value="")
    error_label = tk.Label(
        body,
        textvariable=error_var,
        anchor="w",
        justify="left",
        bg=dialog_bg,
        fg=theme["danger"],
        font=(theme["font_ui"], 10),
    )
    error_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _current_text() -> str:
        return text_widget.get("1.0", "end").strip()

    def on_confirm() -> None:
        parsed, parse_error = parse_command_text(_current_text())
        if parse_error or parsed is None:
            error_var.set(parse_error or "Unable to parse command.")
            return
        result["value"] = parsed
        window.destroy()

    def on_cancel() -> None:
        result["value"] = None
        window.destroy()

    def on_copy() -> None:
        root.clipboard_clear()
        root.clipboard_append(_current_text())

    def on_reset() -> None:
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", initial_text)
        error_var.set("")

    tk.Button(
        footer,
        text="Copy",
        command=on_copy,
        width=12,
        padx=10,
        pady=9,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    ).pack(side="left")

    tk.Button(
        footer,
        text="Reset",
        command=on_reset,
        width=12,
        padx=10,
        pady=9,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    ).pack(side="left", padx=(8, 0))

    cancel_button = tk.Button(
        footer,
        text=cancel_label,
        command=on_cancel,
        width=16,
        padx=10,
        pady=9,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    )
    cancel_button.pack(side="right", padx=(8, 0))

    confirm_button = tk.Button(
        footer,
        text=confirm_label,
        command=on_confirm,
        width=16,
        padx=10,
        pady=9,
        bg=theme["accent"],
        fg="#000000",
        activebackground=theme["accent_active"],
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=(theme["font_ui"], 10, "bold"),
    )
    confirm_button.pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.bind("<Escape>", lambda _: on_cancel())
    window.bind("<Return>", lambda _: on_confirm())
    window.bind("<KP_Enter>", lambda _: on_confirm())
    confirm_button.focus_set()
    window.wait_window()
    return result["value"]


def ask_text_dialog(
    *,
    root: Any,
    title: str,
    text: str,
    copy_text: str | None = None,
    confirm_label: str = "Continue",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 540,
    wrap_mode: str = "word",
) -> bool:
    import tkinter as tk
    from tkinter import ttk

    theme = _dialog_theme(root)
    dialog_bg = theme["panel"]
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    fit_window_to_screen(
        window=window,
        requested_width=width,
        requested_height=height,
        requested_min_width=700,
        requested_min_height=420,
    )
    window.transient(root)
    window.grab_set()
    window.lift()
    window.focus_force()

    result: dict[str, bool] = {"value": False}

    # Pack footer FIRST to guarantee button visibility
    footer = tk.Frame(window, bg=dialog_bg, padx=16, pady=14)
    footer.pack(side="bottom", fill="x")

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg=theme["surface"],
        fg=theme["text"],
        insertbackground=theme["text"],
        relief="flat",
        font=(theme["font_mono"], 10),
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=theme["border"],
    )
    text_widget.grid(row=0, column=0, sticky="nsew")
    _bind_text_wheel_scroll(text_widget)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    def on_confirm() -> None:
        result["value"] = True
        window.destroy()

    def on_cancel() -> None:
        result["value"] = False
        window.destroy()

    if copy_text is not None:
        def on_copy() -> None:
            root.clipboard_clear()
            root.clipboard_append(str(copy_text))

        tk.Button(
            footer,
            text="Copy",
            command=on_copy,
            width=12,
            padx=10,
            pady=9,
            bg=theme["surface"],
            fg=theme["text"],
            activebackground=theme["surface_alt"],
            activeforeground=theme["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=theme["border"],
            font=(theme["font_ui"], 10),
        ).pack(side="left")

    cancel_button = tk.Button(
        footer,
        text=cancel_label,
        command=on_cancel,
        width=16,
        padx=10,
        pady=9,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    )
    cancel_button.pack(side="right", padx=(8, 0))

    confirm_button = tk.Button(
        footer,
        text=confirm_label,
        command=on_confirm,
        width=16,
        padx=10,
        pady=9,
        bg=theme["accent"],
        fg="#000000",
        activebackground=theme["accent_active"],
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=(theme["font_ui"], 10, "bold"),
    )
    confirm_button.pack(side="right")

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.bind("<Escape>", lambda _: on_cancel())
    window.bind("<Return>", lambda _: on_confirm())
    window.bind("<KP_Enter>", lambda _: on_confirm())
    confirm_button.focus_set()
    window.wait_window()
    return result["value"]


def ask_text_dialog_with_actions(
    *,
    root: Any,
    title: str,
    text: str,
    actions: list[tuple[str, str]],
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 560,
    wrap_mode: str = "word",
) -> str:
    import tkinter as tk
    from tkinter import ttk

    theme = _dialog_theme(root)
    dialog_bg = theme["panel"]
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    fit_window_to_screen(
        window=window,
        requested_width=width,
        requested_height=height,
        requested_min_width=760,
        requested_min_height=440,
    )
    window.transient(root)
    window.grab_set()
    window.lift()
    window.focus_force()

    result: dict[str, str] = {"value": "cancel"}

    def on_choose(action_id: str) -> None:
        result["value"] = action_id
        window.destroy()

    # Pack footer FIRST to guarantee button visibility
    footer = tk.Frame(window, bg=dialog_bg, padx=16, pady=14)
    footer.pack(side="bottom", fill="x")

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg=theme["surface"],
        fg=theme["text"],
        insertbackground=theme["text"],
        relief="flat",
        font=(theme["font_mono"], 10),
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=theme["border"],
    )
    text_widget.grid(row=0, column=0, sticky="nsew")
    _bind_text_wheel_scroll(text_widget)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    if actions:
        quick_action_row = tk.Frame(footer, bg=dialog_bg)
        quick_action_row.pack(fill="x", pady=(0, 10))
        for action_id, label in actions:
            tk.Button(
                quick_action_row,
                text=label,
                command=lambda value=action_id: on_choose(value),
                padx=10,
                pady=7,
                bg=theme["surface"],
                fg=theme["accent"],
                activebackground=theme["surface_alt"],
                activeforeground=theme["accent"],
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=theme["accent"],
                font=(theme["font_ui"], 10),
            ).pack(side="left", padx=(0, 8))

    button_row = tk.Frame(footer, bg=dialog_bg)
    button_row.pack(fill="x")

    cancel_button = tk.Button(
        button_row,
        text=cancel_label,
        command=lambda: on_choose("cancel"),
        width=14,
        padx=8,
        pady=9,
        bg=theme["surface"],
        fg=theme["text"],
        activebackground=theme["surface_alt"],
        activeforeground=theme["text"],
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=theme["border"],
        font=(theme["font_ui"], 10),
    )
    cancel_button.pack(side="right", padx=(8, 0))

    confirm_button = tk.Button(
        button_row,
        text=confirm_label,
        command=lambda: on_choose("confirm"),
        width=14,
        padx=8,
        pady=9,
        bg=theme["accent"],
        fg="#000000",
        activebackground=theme["accent_active"],
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=(theme["font_ui"], 10, "bold"),
    )
    confirm_button.pack(side="right")

    window.protocol("WM_DELETE_WINDOW", lambda: on_choose("cancel"))
    window.bind("<Escape>", lambda _: on_choose("cancel"))
    window.bind("<Return>", lambda _: on_choose("confirm"))
    window.bind("<KP_Enter>", lambda _: on_choose("confirm"))
    confirm_button.focus_set()
    window.wait_window()
    return result["value"]
