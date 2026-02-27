"""First-run installer wizard.

A lightweight 3-step modal shown the very first time the app launches:

  Step 1 — Environment check (lerobot importable, venv active)
  Step 2 — Install desktop launcher (optional, skippable)
  Step 3 — Add shortcut to ~/Desktop (optional, skippable)

When the user clicks "Done" on the final step the wizard sets
``config["_setup_complete"] = True`` and calls *save_fn* so the
wizard never appears again.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


def show_first_run_wizard(
    root: Any,
    config: dict[str, Any],
    *,
    save_fn: Callable[[dict[str, Any]], None],
    colors: dict[str, str] | None = None,
) -> None:
    """Build and display the first-run wizard modal.

    *colors* should be the dict returned by ``apply_gui_theme()`` in
    ``gui_app.py``; if omitted the wizard resolves the theme itself.
    Blocks until the user closes the window (via Done or the X button).
    """
    import tkinter as tk
    from tkinter import ttk

    # ── Lazy imports ─────────────────────────────────────────────────────────
    from .desktop_launcher import add_desktop_shortcut, install_desktop_launcher
    from .setup_wizard import probe_setup_wizard_status

    # ── Resolve theme colors ──────────────────────────────────────────────────
    # Prefer the caller-supplied colors dict (already resolved, correct fonts).
    # Fall back to building colors ourselves only if not supplied.
    if not colors:
        try:
            import tkinter.font as tkfont
            from .gui_theme import build_theme_colors, normalize_theme_mode
            _mode = normalize_theme_mode(config.get("ui_theme_mode", "dark"))
            _ui_font = tkfont.nametofont("TkDefaultFont").cget("family") or "TkDefaultFont"
            _mono = tkfont.nametofont("TkFixedFont").cget("family") or "Courier"
            colors = build_theme_colors(ui_font=_ui_font, mono_font=_mono, theme_mode=_mode)
        except Exception:
            colors = {}

    # Unpack — these names are used throughout the function
    panel       = colors.get("panel",    "#121212")
    surface     = colors.get("surface",  "#1b1b1b")
    surface_alt = colors.get("surface_alt", "#252525")
    header_bg   = colors.get("header",   "#0e0e0e")
    accent      = colors.get("accent",   "#f0a500")
    text_col    = colors.get("text",     "#f2f2f2")
    muted       = colors.get("muted",    "#8f8f8f")
    success_col = colors.get("success",  "#22c55e")
    error_col   = colors.get("error",    "#ef4444")
    border      = colors.get("border",   "#303030")
    ui_font     = colors.get("font_ui",  "TkDefaultFont")
    mono_font   = colors.get("font_mono","TkFixedFont")
    theme_mode  = colors.get("theme_mode", "dark")

    # Accent button text should be dark on both themes (amber bg in dark, amber-brown in light)
    accent_fg = "#1a1100"

    # Secondary button (Back / Skip) styling — readable in both themes
    sec_fg   = text_col
    sec_bg   = surface_alt
    sec_abg  = surface       # active bg
    sec_afg  = text_col      # active fg

    # ── Window ───────────────────────────────────────────────────────────────
    dlg = tk.Toplevel(root)
    dlg.title("LeRobot Setup")
    dlg.configure(bg=panel)
    dlg.transient(root)
    dlg.grab_set()
    dlg.resizable(False, False)

    # Centre relative to root
    root.update_idletasks()
    w, h = 620, 500
    rx = root.winfo_x() + (root.winfo_width()  - w) // 2
    ry = root.winfo_y() + (root.winfo_height() - h) // 2
    dlg.geometry(f"{w}x{h}+{rx}+{ry}")

    # ── Step state ────────────────────────────────────────────────────────────
    TOTAL_STEPS = 3
    step_var = tk.IntVar(value=1)

    # Per-step status vars (set by action callbacks)
    env_status_var     = tk.StringVar(value="Checking…")
    launcher_status_var = tk.StringVar(value="")
    desktop_status_var  = tk.StringVar(value="")

    # ── Outer container ───────────────────────────────────────────────────────
    outer = tk.Frame(dlg, bg=panel)
    outer.pack(fill="both", expand=True)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(outer, bg=header_bg, padx=20, pady=14)
    hdr.pack(fill="x")

    tk.Label(
        hdr,
        text="Welcome to LeRobot Pipeline Manager",
        bg=header_bg, fg=accent,
        font=(ui_font, 14, "bold"),
    ).pack(anchor="w")

    subtitle_var = tk.StringVar(value="Step 1 of 3 — Environment")
    tk.Label(
        hdr,
        textvariable=subtitle_var,
        bg=header_bg, fg=muted,
        font=(ui_font, 10),
    ).pack(anchor="w", pady=(2, 0))

    # Step indicator dots
    dots_frame = tk.Frame(hdr, bg=header_bg)
    dots_frame.pack(anchor="w", pady=(8, 0))
    dot_labels: list[tk.Label] = []
    for i in range(TOTAL_STEPS):
        lbl = tk.Label(dots_frame, text="●", bg=header_bg,
                       fg=accent if i == 0 else border,
                       font=(ui_font, 10))
        lbl.pack(side="left", padx=(0, 4))
        dot_labels.append(lbl)

    tk.Frame(outer, bg=border, height=1).pack(fill="x")

    # ── Page area ─────────────────────────────────────────────────────────────
    page_area = tk.Frame(outer, bg=panel)
    page_area.pack(fill="both", expand=True, padx=24, pady=18)

    # ── Navigation bar ────────────────────────────────────────────────────────
    tk.Frame(outer, bg=border, height=1).pack(fill="x")
    nav = tk.Frame(outer, bg=panel, pady=12, padx=24)
    nav.pack(fill="x")
    nav.columnconfigure(1, weight=1)  # spacer

    def _btn(parent: Any, text: str, *, primary: bool = False, **kw: Any) -> tk.Button:
        """Create a flat button styled for the current theme."""
        if primary:
            return tk.Button(
                parent, text=text,
                bg=accent, fg=accent_fg,
                activebackground=colors.get("accent_dark", accent),
                activeforeground=accent_fg,
                relief="flat", bd=0, padx=16, pady=6,
                font=(ui_font, 10, "bold"),
                cursor="hand2",
                **kw,
            )
        return tk.Button(
            parent, text=text,
            bg=sec_bg, fg=sec_fg,
            activebackground=sec_abg, activeforeground=sec_afg,
            relief="flat", bd=0, padx=12, pady=6,
            font=(ui_font, 10),
            cursor="hand2",
            **kw,
        )

    back_btn = _btn(nav, "← Back")
    back_btn.grid(row=0, column=0, sticky="w")

    nav_right = tk.Frame(nav, bg=panel)
    nav_right.grid(row=0, column=2, sticky="e")

    skip_btn = _btn(nav_right, "Skip this step")
    skip_btn.pack(side="left", padx=(0, 8))

    next_btn = _btn(nav_right, "Next →", primary=True)
    next_btn.pack(side="left")

    # ── Page frames ───────────────────────────────────────────────────────────
    pages: list[tk.Frame] = []
    for _ in range(TOTAL_STEPS):
        f = tk.Frame(page_area, bg=panel)
        f.place(relx=0, rely=0, relwidth=1, relheight=1)
        pages.append(f)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _section_title(parent: Any, text: str) -> None:
        tk.Label(parent, text=text, bg=panel, fg=text_col,
                 font=(ui_font, 12, "bold")).pack(anchor="w")

    def _body_label(parent: Any, text: str) -> tk.Label:
        lbl = tk.Label(parent, text=text, bg=panel, fg=muted,
                       font=(ui_font, 10), justify="left", wraplength=560)
        lbl.pack(anchor="w", pady=(6, 0))
        return lbl

    def _status_label(parent: Any, var: tk.StringVar) -> tk.Label:
        lbl = tk.Label(parent, textvariable=var, bg=panel, fg=muted,
                       font=(ui_font, 10), justify="left", wraplength=560)
        lbl.pack(anchor="w", pady=(12, 0))
        return lbl

    def _action_btn(parent: Any, text: str, cmd: Callable[[], None]) -> tk.Button:
        btn = _btn(parent, text, primary=True, command=cmd)
        btn.pack(anchor="w", pady=(16, 0))
        return btn

    # ── Step 1 — Environment ──────────────────────────────────────────────────
    p1 = pages[0]
    _section_title(p1, "Environment Check")
    _body_label(p1, "Verifying that lerobot is importable using the current Python executable.")

    env_badge = tk.Label(p1, textvariable=env_status_var,
                         bg=panel, fg=muted,
                         font=(ui_font, 10, "bold"), anchor="w")
    env_badge.pack(anchor="w", pady=(16, 0))

    # Scrollable detail box
    detail_frame = tk.Frame(p1, bg=border, padx=1, pady=1)
    detail_frame.pack(fill="x", pady=(10, 0))
    env_detail_text = tk.Text(
        detail_frame,
        bg=surface, fg=muted,
        font=(mono_font, 9),
        height=9, relief="flat", bd=0,
        highlightthickness=0,
        state="disabled", wrap="word",
        selectbackground=accent, selectforeground=accent_fg,
    )
    env_detail_text.pack(fill="x")

    recheck_btn = _btn(p1, "Re-check now")
    recheck_btn.pack(anchor="w", pady=(12, 0))

    def _probe_env() -> None:
        status = probe_setup_wizard_status(config)
        if status.ready:
            env_status_var.set(f"✅  Environment ready  ({status.lerobot_import_detail})")
            env_badge.configure(fg=success_col)
        else:
            env_status_var.set("⚠️  Not ready — lerobot import failed")
            env_badge.configure(fg=error_col)

        detail_lines = [
            f"Python:           {sys.executable}",
            f"Virtual env:      {'active' if status.virtual_env_active else 'not detected'}",
            f"lerobot import:   {'OK' if status.lerobot_import_ok else 'FAILED'} ({status.lerobot_import_detail})",
        ]
        if not status.lerobot_import_ok:
            detail_lines += [
                "",
                "To fix: activate your venv and run:",
                "  pip install lerobot",
                "",
                "Then click 'Re-check now'.",
            ]
        env_detail_text.configure(state="normal")
        env_detail_text.delete("1.0", "end")
        env_detail_text.insert("1.0", "\n".join(detail_lines))
        env_detail_text.configure(state="disabled")

    recheck_btn.configure(command=_probe_env)
    dlg.after(120, _probe_env)

    # ── Step 2 — Desktop Launcher ─────────────────────────────────────────────
    p2 = pages[1]
    _section_title(p2, "Install Desktop Launcher")
    _body_label(
        p2,
        "Create a launcher entry so this app opens directly from your application menu "
        "or dock — no terminal required."
    )
    _body_label(p2, "You can always reinstall or update the launcher later from the Config tab.")

    launcher_status_lbl = _status_label(p2, launcher_status_var)

    def _do_install_launcher() -> None:
        launcher_status_var.set("Installing…")
        launcher_status_lbl.configure(fg=muted)
        dlg.update_idletasks()
        report = install_desktop_launcher(
            app_dir=Path(__file__).resolve().parents[1],
        )
        if report.ok:
            launcher_status_var.set(f"✅  {report.message}")
            launcher_status_lbl.configure(fg=success_col)
        else:
            launcher_status_var.set(f"❌  {report.message}")
            launcher_status_lbl.configure(fg=error_col)

    _action_btn(p2, "Install Launcher", _do_install_launcher)

    # ── Step 3 — Add to Desktop ───────────────────────────────────────────────
    p3 = pages[2]
    _section_title(p3, "Add to Desktop")
    _body_label(
        p3,
        "Place a shortcut on your Desktop so you can double-click to open the app, "
        "just like any other installed application."
    )
    _body_label(p3, "The desktop launcher (previous step) must be installed first.")

    desktop_status_lbl = _status_label(p3, desktop_status_var)

    def _do_add_to_desktop() -> None:
        desktop_status_var.set("Adding shortcut…")
        desktop_status_lbl.configure(fg=muted)
        dlg.update_idletasks()
        ok, msg = add_desktop_shortcut()
        if ok:
            desktop_status_var.set(f"✅  {msg}")
            desktop_status_lbl.configure(fg=success_col)
        else:
            desktop_status_var.set(f"❌  {msg}")
            desktop_status_lbl.configure(fg=error_col)

    _action_btn(p3, "Add to Desktop", _do_add_to_desktop)

    # ── Navigation logic ──────────────────────────────────────────────────────
    STEP_SUBTITLES = [
        "Step 1 of 3 — Environment",
        "Step 2 of 3 — Desktop Launcher",
        "Step 3 of 3 — Add to Desktop",
    ]

    def _show_step(n: int) -> None:
        step_var.set(n)
        subtitle_var.set(STEP_SUBTITLES[n - 1])
        for i, page in enumerate(pages):
            if i == n - 1:
                page.lift()
            else:
                page.lower()
        # Update dots
        for i, dot in enumerate(dot_labels):
            dot.configure(fg=accent if i < n else border)
        back_btn.configure(state="normal" if n > 1 else "disabled",
                           fg=sec_fg if n > 1 else muted)
        if n < TOTAL_STEPS:
            next_btn.configure(text="Next →")
            skip_btn.pack(side="left", padx=(0, 8))
        else:
            next_btn.configure(text="Done  ✓")
            skip_btn.pack_forget()

    def _on_next() -> None:
        current = step_var.get()
        if current < TOTAL_STEPS:
            _show_step(current + 1)
        else:
            _finish()

    def _on_skip() -> None:
        current = step_var.get()
        if current < TOTAL_STEPS:
            _show_step(current + 1)

    def _on_back() -> None:
        current = step_var.get()
        if current > 1:
            _show_step(current - 1)

    def _finish() -> None:
        config["_setup_complete"] = True
        try:
            save_fn(config)
        except Exception:
            pass
        dlg.destroy()

    next_btn.configure(command=_on_next)
    skip_btn.configure(command=_on_skip)
    back_btn.configure(command=_on_back)
    dlg.protocol("WM_DELETE_WINDOW", _finish)

    _show_step(1)
    dlg.wait_window()
