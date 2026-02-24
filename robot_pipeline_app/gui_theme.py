from __future__ import annotations

from typing import Any


def _pick_font(tkfont: Any, candidates: list[str], fallback: str) -> str:
    available = {name.lower(): name for name in tkfont.families()}
    for candidate in candidates:
        found = available.get(candidate.lower())
        if found:
            return found
    return fallback


def apply_gui_theme(*, root: Any, tkfont: Any, ttk: Any) -> dict[str, str]:
    default_family = tkfont.nametofont("TkDefaultFont").cget("family")
    ui_font = _pick_font(
        tkfont,
        [
            "Inter",
            "SF Pro Text",
            "Segoe UI",
            "Noto Sans",
            "Ubuntu",
            "Helvetica Neue",
            "Arial",
        ],
        str(default_family),
    )
    mono_font = _pick_font(
        tkfont,
        [
            "JetBrains Mono",
            "SF Mono",
            "Menlo",
            "Consolas",
            "DejaVu Sans Mono",
            "Courier New",
        ],
        "Courier New",
    )

    for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont"):
        try:
            tkfont.nametofont(named).configure(family=ui_font, size=10)
        except Exception:
            pass
    try:
        tkfont.nametofont("TkFixedFont").configure(family=mono_font, size=10)
    except Exception:
        pass

    colors = {
        "bg":           "#0a0a0a",
        "panel":        "#111111",
        "surface":      "#1a1a1a",
        "surface_alt":  "#252525",
        "header":       "#0d0d0d",
        "border":       "#2d2d2d",
        "border_focus": "#f0a500",
        "text":         "#eeeeee",
        "muted":        "#777777",
        "accent":       "#f0a500",
        "accent_dark":  "#c88a00",
        "running":      "#f0a500",
        "ready":        "#22c55e",
        "error":        "#ef4444",
        "success":      "#22c55e",
        "font_ui":      ui_font,
        "font_mono":    mono_font,
    }
    root.configure(bg=colors["bg"])

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure("TFrame", background=colors["bg"])
    style.configure("Panel.TFrame", background=colors["panel"])

    style.configure(
        "Section.TLabelframe",
        background=colors["panel"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Section.TLabelframe.Label",
        background=colors["panel"],
        foreground=colors["accent"],
        font=(ui_font, 11, "bold"),
    )

    style.configure("TLabel", background=colors["panel"], foreground=colors["text"], font=(ui_font, 10))
    style.configure("Field.TLabel", background=colors["panel"], foreground=colors["text"], font=(ui_font, 10))
    style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"], font=(ui_font, 10))
    style.configure("SectionTitle.TLabel", background=colors["panel"], foreground=colors["text"], font=(ui_font, 12, "bold"))

    style.configure(
        "TEntry",
        fieldbackground=colors["surface"],
        foreground=colors["text"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        insertcolor=colors["text"],
        padding=(8, 6),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", colors["border_focus"])],
        lightcolor=[("focus", colors["border_focus"])],
        fieldbackground=[("disabled", "#141414"), ("readonly", "#141414")],
        foreground=[("disabled", colors["muted"]), ("readonly", colors["muted"])],
    )

    style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"], font=(ui_font, 10))
    style.map("TCheckbutton", background=[("active", colors["panel"])], foreground=[("active", colors["text"])])

    style.configure(
        "TButton",
        background=colors["surface"],
        foreground=colors["text"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        padding=(10, 6),
        font=(ui_font, 10, "bold"),
    )
    style.map(
        "TButton",
        background=[("active", colors["surface_alt"]), ("disabled", colors["surface"])],
        foreground=[("disabled", colors["muted"])],
    )

    style.configure("TNotebook", background=colors["bg"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=colors["header"],
        foreground=colors["muted"],
        padding=(16, 9),
        font=(ui_font, 10, "bold"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["surface"]), ("active", colors["surface_alt"])],
        foreground=[("selected", colors["accent"]), ("active", colors["text"])],
    )

    style.configure(
        "Accent.TButton",
        padding=(12, 7),
        font=(ui_font, 10, "bold"),
        borderwidth=0,
    )
    style.map(
        "Accent.TButton",
        background=[("active", colors["accent_dark"]), ("!disabled", colors["accent"])],
        foreground=[("!disabled", "#000000")],
    )

    style.configure(
        "Danger.TButton",
        padding=(10, 6),
        font=(ui_font, 10, "bold"),
        borderwidth=0,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#c0392b"), ("!disabled", colors["error"])],
        foreground=[("!disabled", "#ffffff")],
    )

    style.configure(
        "Accent.Horizontal.TProgressbar",
        troughcolor=colors["surface"],
        bordercolor=colors["surface"],
        background=colors["accent"],
    )
    style.configure(
        "Time.Horizontal.TProgressbar",
        troughcolor=colors["surface"],
        bordercolor=colors["surface"],
        background=colors["accent"],
    )

    return colors
