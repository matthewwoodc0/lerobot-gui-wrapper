from __future__ import annotations

from typing import Any

from .config_store import normalize_path, resolve_existing_directory


def _dialog_geometry(root: Any) -> str:
    try:
        root.update_idletasks()
        width = int(root.winfo_width())
        height = int(root.winfo_height())
        x = int(root.winfo_x())
        y = int(root.winfo_y())
    except Exception:
        return "1200x820+120+120"

    width = max(1100, int(width * 0.9))
    height = max(760, int(height * 0.9))
    x = max(0, x + 40)
    y = max(0, y + 40)
    return f"{width}x{height}+{x}+{y}"


def ask_directory_dialog(
    *,
    root: Any,
    filedialog: Any,
    initial_dir: str | None,
    title: str = "Select folder",
) -> str | None:
    initial = resolve_existing_directory(initial_dir)

    old_scaling = None
    boosted = None
    try:
        old_scaling = float(root.tk.call("tk", "scaling"))
        boosted = max(old_scaling, 1.35)
        if boosted != old_scaling:
            root.tk.call("tk", "scaling", boosted)
    except Exception:
        old_scaling = None
        boosted = None

    parent = None
    selected = None
    try:
        import tkinter as tk

        parent = tk.Toplevel(root)
        parent.withdraw()
        try:
            parent.transient(root)
        except Exception:
            pass
        try:
            parent.geometry(_dialog_geometry(root))
        except Exception:
            pass
        parent.update_idletasks()
        selected = filedialog.askdirectory(
            parent=parent,
            initialdir=initial,
            title=title,
        )
    finally:
        if parent is not None:
            try:
                parent.destroy()
            except Exception:
                pass
        if old_scaling is not None and boosted is not None and boosted != old_scaling:
            try:
                root.tk.call("tk", "scaling", old_scaling)
            except Exception:
                pass

    if selected:
        return normalize_path(selected)
    return None


def ask_openfilename_dialog(
    *,
    root: Any,
    filedialog: Any,
    initial_dir: str | None,
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a *file* picker dialog (as opposed to a folder picker).

    Parameters mirror :func:`ask_directory_dialog`; *filetypes* is a list of
    ``(label, pattern)`` tuples, e.g. ``[("JSON files", "*.json")]``.
    """
    if filetypes is None:
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]

    initial = resolve_existing_directory(initial_dir)

    old_scaling = None
    boosted = None
    try:
        old_scaling = float(root.tk.call("tk", "scaling"))
        boosted = max(old_scaling, 1.35)
        if boosted != old_scaling:
            root.tk.call("tk", "scaling", boosted)
    except Exception:
        old_scaling = None
        boosted = None

    parent = None
    selected = None
    try:
        import tkinter as tk

        parent = tk.Toplevel(root)
        parent.withdraw()
        try:
            parent.transient(root)
        except Exception:
            pass
        try:
            parent.geometry(_dialog_geometry(root))
        except Exception:
            pass
        parent.update_idletasks()
        selected = filedialog.askopenfilename(
            parent=parent,
            initialdir=initial,
            title=title,
            filetypes=filetypes,
        )
    finally:
        if parent is not None:
            try:
                parent.destroy()
            except Exception:
                pass
        if old_scaling is not None and boosted is not None and boosted != old_scaling:
            try:
                root.tk.call("tk", "scaling", old_scaling)
            except Exception:
                pass

    if selected:
        return normalize_path(selected)
    return None
