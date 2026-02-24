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

    max_w = max(560, int(screen_w * 0.92))
    max_h = max(420, int(screen_h * 0.88))
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

    window = tk.Toplevel(root)
    window.title(title)
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
    window.lift()
    window.focus_force()

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg="#111827",
        fg="#e5e7eb",
        insertbackground="#f8fafc",
        relief="flat",
        font=("Menlo", 10),
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

    window = tk.Toplevel(root)
    window.title(title)
    _fit_and_center_dialog(
        window=window,
        requested_width=width,
        requested_height=height,
        requested_min_width=700,
        requested_min_height=420,
    )
    window.transient(root)
    window.grab_set()

    result: dict[str, bool] = {"value": False}

    body = ttk.Frame(window, padding=10)
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)

    text_widget = tk.Text(
        body,
        wrap=wrap_mode,
        bg="#111827",
        fg="#e5e7eb",
        insertbackground="#f8fafc",
        relief="flat",
        font=("Menlo", 10),
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

    ttk.Button(buttons, text=cancel_label, command=on_cancel).pack(side="right")
    ttk.Button(buttons, text=confirm_label, style="Accent.TButton", command=on_confirm).pack(side="right", padx=(0, 8))

    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.bind("<Return>", lambda _: on_confirm())
    window.bind("<Escape>", lambda _: on_cancel())
    window.wait_window()
    return result["value"]
