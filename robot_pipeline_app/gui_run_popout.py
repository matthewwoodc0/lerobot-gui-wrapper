from __future__ import annotations

import re
import time
from typing import Any, Callable


EPISODE_PATTERNS = [
    re.compile(r"[Ee]pisode\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"[Ee]pisode\s+(\d+)\s+of\s+(\d+)"),
    re.compile(r"[Ee](?:pisode|p)\s*[:#]?\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"[Ee]pisode(?:_idx)?\s*[:=]\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"\b[Ee]p\s+(\d+)\s+of\s+(\d+)\b"),
]
EPISODE_PARTIAL_PATTERN = re.compile(r"[Ee](?:pisode|p)\s*[:#]?\s*(\d+)")


def parse_episode_progress_line(line: str) -> tuple[int, int | None] | None:
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    partial = EPISODE_PARTIAL_PATTERN.search(line)
    if partial:
        return int(partial.group(1)), None
    return None


class RunControlPopout:
    def __init__(
        self,
        *,
        root: Any,
        colors: dict[str, str],
        on_send_key: Callable[[str], tuple[bool, str]],
        on_cancel: Callable[[], None],
    ) -> None:
        self.root = root
        self.colors = colors
        self.on_send_key = on_send_key
        self.on_cancel = on_cancel

        self.window: Any | None = None
        self._timer_job: str | None = None
        self._pulse_job: str | None = None

        self.mode_var: Any | None = None
        self.episode_var: Any | None = None
        self.episode_timer_var: Any | None = None
        self.key_status_var: Any | None = None
        self.episode_progressbar: Any | None = None
        self._dot_canvas: Any | None = None
        self._dot_item: Any | None = None
        self._dot_bright = True

        self._active = False
        self._total_episodes = 0
        self._episode_duration_s = 0.0
        self._current_episode = 0
        self._episode_started_at: float | None = None

    def _fmt_seconds(self, seconds: float) -> str:
        sec = max(int(seconds), 0)
        minutes, remainder = divmod(sec, 60)
        return f"{minutes:02d}:{remainder:02d}"

    def _pulse_dot(self) -> None:
        if self._pulse_job is not None:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        if not self._active or self._dot_canvas is None or self._dot_item is None:
            return
        color = self.colors.get("running", "#f0a500") if self._dot_bright else "#7a5200"
        self._dot_canvas.itemconfig(self._dot_item, fill=color, outline=color)
        self._dot_bright = not self._dot_bright
        self._pulse_job = self.root.after(600, self._pulse_dot)

    def _ensure_window(self) -> None:
        if self.window is not None and bool(self.window.winfo_exists()):
            return

        import tkinter as tk
        from tkinter import ttk

        accent = self.colors.get("accent", "#f0a500")
        panel = self.colors.get("panel", "#111111")
        surface = self.colors.get("surface", "#1a1a1a")
        border = self.colors.get("border", "#2d2d2d")
        text_col = self.colors.get("text", "#eeeeee")
        muted = self.colors.get("muted", "#777777")
        ui_font = self.colors.get("font_ui", "TkDefaultFont")
        error_col = self.colors.get("error", "#ef4444")

        self.window = tk.Toplevel(self.root)
        self.window.title("Run Controls")
        self.window.geometry("580x280")
        self.window.minsize(480, 260)
        self.window.configure(bg=panel)
        self.window.transient(self.root)
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        # ── Header bar ──────────────────────────────────────────────────────
        header = tk.Frame(self.window, bg="#0d0d0d", padx=14, pady=10)
        header.pack(fill="x")

        dot_frame = tk.Frame(header, bg="#0d0d0d")
        dot_frame.pack(side="left")

        self._dot_canvas = tk.Canvas(dot_frame, width=14, height=14, bg="#0d0d0d", highlightthickness=0)
        self._dot_canvas.pack(side="left", padx=(0, 6))
        self._dot_item = self._dot_canvas.create_oval(2, 2, 12, 12, fill=accent, outline=accent)

        self.mode_var = tk.StringVar(value="-- MODE")
        tk.Label(
            dot_frame,
            textvariable=self.mode_var,
            bg="#0d0d0d",
            fg=accent,
            font=(ui_font, 11, "bold"),
        ).pack(side="left")

        # Cancel button in header (top-right)
        tk.Button(
            header,
            text="✕  Cancel Run",
            command=self.on_cancel,
            padx=10,
            pady=6,
            bg=error_col,
            fg="#ffffff",
            activebackground="#c0392b",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=(ui_font, 10, "bold"),
        ).pack(side="right")

        # ── Thin separator ───────────────────────────────────────────────────
        tk.Frame(self.window, bg=border, height=1).pack(fill="x")

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self.window, bg=panel, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        # Episode counter
        self.episode_var = tk.StringVar(value="Episode  -- / --")
        tk.Label(
            body,
            textvariable=self.episode_var,
            bg=panel,
            fg=text_col,
            font=(ui_font, 16, "bold"),
            anchor="w",
        ).pack(fill="x")

        # Progress bar
        self.episode_progressbar = ttk.Progressbar(
            body,
            mode="determinate",
            style="Time.Horizontal.TProgressbar",
        )
        self.episode_progressbar.pack(fill="x", pady=(6, 2))

        # Timer row
        self.episode_timer_var = tk.StringVar(value="00:00 elapsed  ·  --:-- total  ·  ↤ --:--")
        tk.Label(
            body,
            textvariable=self.episode_timer_var,
            bg=panel,
            fg=muted,
            font=(ui_font, 10),
            anchor="w",
        ).pack(fill="x")

        # ── Thin separator ───────────────────────────────────────────────────
        tk.Frame(self.window, bg=border, height=1).pack(fill="x")

        # ── Control buttons ──────────────────────────────────────────────────
        controls = tk.Frame(self.window, bg=panel, padx=16, pady=10)
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        redo_btn = tk.Button(
            controls,
            text="←  Redo Run",
            command=lambda: self._send_key("left"),
            padx=16,
            pady=8,
            bg=surface,
            fg=text_col,
            activebackground=self.colors.get("surface_alt", "#252525"),
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            font=(ui_font, 11, "bold"),
        )
        redo_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        next_btn = tk.Button(
            controls,
            text="Next Run  →",
            command=lambda: self._send_key("right"),
            padx=16,
            pady=8,
            bg=surface,
            fg=text_col,
            activebackground=self.colors.get("surface_alt", "#252525"),
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            font=(ui_font, 11, "bold"),
        )
        next_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Key hints
        self.key_status_var = tk.StringVar(value="Left arrow key              Right arrow key")
        tk.Label(
            controls,
            textvariable=self.key_status_var,
            bg=panel,
            fg=muted,
            font=(ui_font, 9),
        ).grid(row=1, column=0, columnspan=2, pady=(4, 0))

        self.window.bind("<Left>", lambda _: self._send_key("left"))
        self.window.bind("<Right>", lambda _: self._send_key("right"))

    def _send_key(self, direction: str) -> None:
        ok, message = self.on_send_key(direction)
        if self.key_status_var is not None:
            self.key_status_var.set(message if message else "Left arrow key              Right arrow key")
        if not ok:
            self.root.bell()

    def _schedule_tick(self) -> None:
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
        self._timer_job = self.root.after(250, self._tick)

    def _tick(self) -> None:
        self._timer_job = None
        if not self._active:
            return

        if self._episode_started_at is None or self._episode_duration_s <= 0:
            self._schedule_tick()
            return

        elapsed = time.monotonic() - self._episode_started_at
        remaining = max(self._episode_duration_s - elapsed, 0.0)
        total = self._episode_duration_s

        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(total, 1.0))
            self.episode_progressbar["value"] = min(elapsed, total)
        if self.episode_timer_var is not None:
            self.episode_timer_var.set(
                f"{self._fmt_seconds(elapsed)} elapsed  ·  {self._fmt_seconds(total)} total  ·  ↤ {self._fmt_seconds(remaining)}"
            )

        self._schedule_tick()

    def start_run(self, run_mode: str, expected_episodes: int | None, expected_seconds: int | None) -> None:
        self._ensure_window()
        if self.window is None:
            return

        episodes = int(expected_episodes or 0)
        seconds = float(expected_seconds or 0)
        self._total_episodes = episodes
        self._episode_duration_s = (seconds / episodes) if episodes > 0 and seconds > 0 else 0.0
        self._current_episode = 1 if episodes > 0 else 0
        self._episode_started_at = time.monotonic() if self._episode_duration_s > 0 and episodes > 0 else None
        self._active = True

        if self.mode_var is not None:
            mode_label = run_mode.upper() if run_mode else "RUN"
            self.mode_var.set(f"{mode_label} MODE")

        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode  {self._current_episode} / {self._total_episodes}")
            else:
                self.episode_var.set("Episode  -- / --")

        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(self._episode_duration_s, 1.0), value=0)

        if self.episode_timer_var is not None:
            if self._episode_duration_s > 0:
                self.episode_timer_var.set(
                    f"00:00 elapsed  ·  {self._fmt_seconds(self._episode_duration_s)} total  ·  ↤ {self._fmt_seconds(self._episode_duration_s)}"
                )
            else:
                self.episode_timer_var.set("00:00 elapsed  ·  --:-- total  ·  ↤ --:--")

        if self.key_status_var is not None:
            self.key_status_var.set("Left arrow key              Right arrow key")

        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self._schedule_tick()
        self._dot_bright = True
        self._pulse_dot()

    def handle_output_line(self, line: str) -> None:
        if not self._active:
            return
        parsed = parse_episode_progress_line(line)
        if parsed is None:
            return
        current, total = parsed
        if total is not None and total > 0:
            self._total_episodes = total
        if current <= 0:
            return

        if current != self._current_episode:
            self._current_episode = current
            self._episode_started_at = time.monotonic()
            if self.episode_progressbar is not None:
                self.episode_progressbar["value"] = 0

        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode  {self._current_episode} / {self._total_episodes}")
            else:
                self.episode_var.set(f"Episode  {self._current_episode} / --")

    def hide(self) -> None:
        self._active = False
        if self._pulse_job is not None:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        if self.window is not None and bool(self.window.winfo_exists()):
            self.window.withdraw()
