from __future__ import annotations

import shlex
from typing import Any


def format_command_for_dialog(cmd: list[str]) -> str:
    if not cmd:
        return "(empty command)"
    lines = [shlex.quote(cmd[0])]
    for arg in cmd[1:]:
        lines.append(f"  {shlex.quote(arg)}")
    return "\n".join(lines)


def _fit_and_center_dialog(
    *,
    window: Any,
    requested_width: int,
    requested_height: int,
    requested_min_width: int,
    requested_min_height: int,
) -> None:
    screen_w = int(window.winfo_screenwidth() or requested_width)
    screen_h = int(window.winfo_screenheight() or requested_height)

    max_w = max(360, screen_w - 24)
    max_h = max(280, screen_h - 48)
    final_w = min(requested_width, max_w)
    final_h = min(requested_height, max_h)

    min_w = min(requested_min_width, final_w)
    min_h = min(requested_min_height, final_h)
    window.minsize(min_w, min_h)

    x = max((screen_w - final_w) // 2, 8)
    y = max((screen_h - final_h) // 2, 8)
    window.geometry(f"{final_w}x{final_h}+{x}+{y}")


def _wheel_units(event: Any) -> int:
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    try:
        delta = float(getattr(event, "delta", 0.0))
    except (TypeError, ValueError):
        return 0
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        units = int(-delta / 120)
        if units != 0:
            return units
    return -1 if delta > 0 else 1


def _bind_text_wheel_scroll(text_widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = _wheel_units(event)
        if units == 0:
            return None
        text_widget.yview_scroll(units, "units")
        return "break"

    text_widget.bind("<MouseWheel>", on_wheel, add="+")
    text_widget.bind("<Button-4>", on_wheel, add="+")
    text_widget.bind("<Button-5>", on_wheel, add="+")


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

    dialog_bg = "#0f0f0f"
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    _fit_and_center_dialog(
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
        bg="#141414",
        fg="#cccccc",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground="#2d2d2d",
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
        bg="#252525",
        fg="#eeeeee",
        activebackground="#333333",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#2d2d2d",
        font=("TkDefaultFont", 10),
    ).pack(side="left")

    tk.Button(
        buttons,
        text="Close",
        command=window.destroy,
        padx=12,
        pady=8,
        bg="#f0a500",
        fg="#000000",
        activebackground="#c88a00",
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=("TkDefaultFont", 10, "bold"),
    ).pack(side="right")

    window.bind("<Escape>", lambda _: window.destroy())
    window.wait_window()


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

    dialog_bg = "#0f0f0f"
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    _fit_and_center_dialog(
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
        bg="#141414",
        fg="#cccccc",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground="#2d2d2d",
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
            bg="#252525",
            fg="#eeeeee",
            activebackground="#333333",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#444444",
            font=("TkDefaultFont", 10),
        ).pack(side="left")

    cancel_button = tk.Button(
        footer,
        text=cancel_label,
        command=on_cancel,
        width=16,
        padx=10,
        pady=9,
        bg="#252525",
        fg="#eeeeee",
        activebackground="#333333",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#444444",
        font=("TkDefaultFont", 10),
    )
    cancel_button.pack(side="right", padx=(8, 0))

    confirm_button = tk.Button(
        footer,
        text=confirm_label,
        command=on_confirm,
        width=16,
        padx=10,
        pady=9,
        bg="#f0a500",
        fg="#000000",
        activebackground="#c88a00",
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=("TkDefaultFont", 10, "bold"),
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

    dialog_bg = "#0f0f0f"
    window = tk.Toplevel(root)
    window.title(title)
    window.configure(bg=dialog_bg)
    _fit_and_center_dialog(
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
        bg="#141414",
        fg="#cccccc",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground="#2d2d2d",
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
                bg="#1a1a1a",
                fg="#f0a500",
                activebackground="#252525",
                activeforeground="#f0a500",
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground="#f0a500",
                font=("TkDefaultFont", 10),
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
        bg="#252525",
        fg="#eeeeee",
        activebackground="#333333",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#444444",
        font=("TkDefaultFont", 10),
    )
    cancel_button.pack(side="right", padx=(8, 0))

    confirm_button = tk.Button(
        button_row,
        text=confirm_label,
        command=lambda: on_choose("confirm"),
        width=14,
        padx=8,
        pady=9,
        bg="#f0a500",
        fg="#000000",
        activebackground="#c88a00",
        activeforeground="#000000",
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=("TkDefaultFont", 10, "bold"),
    )
    confirm_button.pack(side="right")

    window.protocol("WM_DELETE_WINDOW", lambda: on_choose("cancel"))
    window.bind("<Escape>", lambda _: on_choose("cancel"))
    window.bind("<Return>", lambda _: on_choose("confirm"))
    window.bind("<KP_Enter>", lambda _: on_choose("confirm"))
    confirm_button.focus_set()
    window.wait_window()
    return result["value"]
