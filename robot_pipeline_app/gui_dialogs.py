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
    # Keep dialogs fully on-screen so action buttons are always reachable.
    screen_w = int(window.winfo_screenwidth() or requested_width)
    screen_h = int(window.winfo_screenheight() or requested_height)

    # Never exceed the visible screen, even on short displays.
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


def show_text_dialog(
    *,
    root: Any,
    title: str,
    text: str,
    width: int = 980,
    height: int = 520,
    wrap_mode: str = "none",
) -> None:
    import tkinter as tk
    from tkinter import ttk

    dialog_bg = str(root.cget("bg") or "#0f172a")
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

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg="#0d1628",
        fg="#e5e7eb",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=8,
        pady=8,
    )
    text_widget.grid(row=0, column=0, sticky="nsew")

    y_scroll = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    text_widget.configure(yscrollcommand=y_scroll.set)

    if wrap_mode == "none":
        x_scroll = ttk.Scrollbar(body, orient="horizontal", command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(xscrollcommand=x_scroll.set)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    buttons = ttk.Frame(window, padding=(10, 0, 10, 10))
    buttons.pack(fill="x")

    def copy_text() -> None:
        root.clipboard_clear()
        root.clipboard_append(text)

    ttk.Button(buttons, text="Copy", command=copy_text).pack(side="left")
    ttk.Button(buttons, text="Close", command=window.destroy).pack(side="right")

    window.bind("<Escape>", lambda _: window.destroy())
    window.wait_window()


def ask_text_dialog(
    *,
    root: Any,
    title: str,
    text: str,
    confirm_label: str = "Continue",
    cancel_label: str = "Cancel",
    width: int = 980,
    height: int = 540,
    wrap_mode: str = "word",
) -> bool:
    import tkinter as tk
    from tkinter import ttk

    dialog_bg = str(root.cget("bg") or "#0f172a")
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

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg="#0d1628",
        fg="#e5e7eb",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=8,
        pady=8,
    )
    text_widget.grid(row=0, column=0, sticky="nsew")

    y_scroll = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    text_widget.configure(yscrollcommand=y_scroll.set)

    if wrap_mode == "none":
        x_scroll = ttk.Scrollbar(body, orient="horizontal", command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(xscrollcommand=x_scroll.set)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    buttons = ttk.Frame(window, padding=(10, 0, 10, 10))
    buttons.pack(fill="x")

    def on_confirm() -> None:
        result["value"] = True
        window.destroy()

    def on_cancel() -> None:
        result["value"] = False
        window.destroy()

    tk.Label(
        buttons,
        text="Click Confirm or Cancel (Esc = Cancel)",
        bg=dialog_bg,
        fg="#9ca3af",
    ).pack(side="left")

    cancel_button = tk.Button(
        buttons,
        text=cancel_label,
        command=on_cancel,
        width=14,
        padx=10,
        pady=8,
        bg="#1f2937",
        fg="#f3f4f6",
        activebackground="#374151",
        activeforeground="#ffffff",
        relief="raised",
        bd=1,
        highlightthickness=0,
    )
    cancel_button.pack(side="right")

    confirm_button = tk.Button(
        buttons,
        text=confirm_label,
        command=on_confirm,
        width=14,
        padx=10,
        pady=8,
        bg="#0ea5e9",
        fg="#ffffff",
        activebackground="#0284c7",
        activeforeground="#ffffff",
        relief="raised",
        bd=1,
        highlightthickness=0,
    )
    confirm_button.pack(side="right", padx=(0, 8))

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.bind("<Escape>", lambda _: on_cancel())
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

    dialog_bg = str(root.cget("bg") or "#0f172a")
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

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg="#0d1628",
        fg="#e5e7eb",
        insertbackground="#f8fafc",
        relief="flat",
        font="TkFixedFont",
        padx=8,
        pady=8,
    )
    text_widget.grid(row=0, column=0, sticky="nsew")

    y_scroll = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    text_widget.configure(yscrollcommand=y_scroll.set)

    if wrap_mode == "none":
        x_scroll = ttk.Scrollbar(body, orient="horizontal", command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(xscrollcommand=x_scroll.set)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.see("1.0")

    buttons = ttk.Frame(window, padding=(10, 0, 10, 10))
    buttons.pack(fill="x")

    def on_choose(action_id: str) -> None:
        result["value"] = action_id
        window.destroy()

    tk.Label(
        buttons,
        text="Apply a quick fix, confirm to continue, or cancel",
        bg=dialog_bg,
        fg="#9ca3af",
    ).pack(side="left")

    cancel_button = tk.Button(
        buttons,
        text=cancel_label,
        command=lambda: on_choose("cancel"),
        width=12,
        padx=8,
        pady=8,
        bg="#1f2937",
        fg="#f3f4f6",
        activebackground="#374151",
        activeforeground="#ffffff",
        relief="raised",
        bd=1,
        highlightthickness=0,
    )
    cancel_button.pack(side="right")

    confirm_button = tk.Button(
        buttons,
        text=confirm_label,
        command=lambda: on_choose("confirm"),
        width=12,
        padx=8,
        pady=8,
        bg="#0ea5e9",
        fg="#ffffff",
        activebackground="#0284c7",
        activeforeground="#ffffff",
        relief="raised",
        bd=1,
        highlightthickness=0,
    )
    confirm_button.pack(side="right", padx=(0, 8))

    for action_id, label in reversed(actions):
        tk.Button(
            buttons,
            text=label,
            command=lambda value=action_id: on_choose(value),
            padx=8,
            pady=8,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="raised",
            bd=1,
            highlightthickness=0,
        ).pack(side="right", padx=(0, 8))

    window.protocol("WM_DELETE_WINDOW", lambda: on_choose("cancel"))
    window.bind("<Escape>", lambda _: on_choose("cancel"))
    confirm_button.focus_set()
    window.wait_window()
    return result["value"]
