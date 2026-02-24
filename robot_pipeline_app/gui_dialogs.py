from __future__ import annotations

from typing import Any


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
    window.geometry(f"{width}x{height}")
    window.minsize(700, 360)
    window.transient(root)
    window.grab_set()

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
    height: int = 560,
    wrap_mode: str = "word",
) -> bool:
    import tkinter as tk
    from tkinter import ttk

    window = tk.Toplevel(root)
    window.title(title)
    window.geometry(f"{width}x{height}")
    window.minsize(700, 420)
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
    window.bind("<Escape>", lambda _: on_cancel())
    window.wait_window()
    return result["value"]
