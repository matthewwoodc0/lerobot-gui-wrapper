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

        self.mode_var: Any | None = None
        self.episode_var: Any | None = None
        self.episode_timer_var: Any | None = None
        self.key_status_var: Any | None = None
        self.episode_progressbar: Any | None = None

        self._active = False
        self._total_episodes = 0
        self._episode_duration_s = 0.0
        self._current_episode = 0
        self._episode_started_at: float | None = None

    def _fmt_seconds(self, seconds: float) -> str:
        sec = max(int(seconds), 0)
        minutes, remainder = divmod(sec, 60)
        return f"{minutes:02d}:{remainder:02d}"

    def _ensure_window(self) -> None:
        if self.window is not None and bool(self.window.winfo_exists()):
            return

        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(self.root)
        self.window.title("Run Controls")
        self.window.geometry("560x320")
        self.window.minsize(480, 280)
        self.window.configure(bg=self.colors.get("panel", "#0f172a"))
        self.window.transient(self.root)
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        self.mode_var = tk.StringVar(value="Run mode: --")
        self.episode_var = tk.StringVar(value="Episode: --/--")
        self.episode_timer_var = tk.StringVar(value="Episode time: --:-- / --:--")
        self.key_status_var = tk.StringVar(value="Controls ready: Left=Redo, Right=Next")

        body = ttk.Frame(self.window, style="Panel.TFrame", padding=12)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            textvariable=self.mode_var,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=(self.colors.get("font_ui", "TkDefaultFont"), 12, "bold"),
        ).pack(anchor="w")

        tk.Label(
            body,
            textvariable=self.episode_var,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=(self.colors.get("font_ui", "TkDefaultFont"), 11, "bold"),
        ).pack(anchor="w", pady=(8, 2))

        self.episode_progressbar = ttk.Progressbar(body, mode="determinate", style="Time.Horizontal.TProgressbar")
        self.episode_progressbar.pack(fill="x", pady=(0, 4))

        tk.Label(
            body,
            textvariable=self.episode_timer_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10),
        ).pack(anchor="w")

        actions = tk.Frame(body, bg=self.colors["panel"])
        actions.pack(fill="x", pady=(12, 6))

        tk.Button(
            actions,
            text="Redo Run (Left)",
            command=lambda: self._send_key("left"),
            width=16,
            padx=8,
            pady=8,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="raised",
            bd=1,
            highlightthickness=0,
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10, "bold"),
        ).pack(side="left")

        tk.Button(
            actions,
            text="Start Next (Right)",
            command=lambda: self._send_key("right"),
            width=16,
            padx=8,
            pady=8,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="raised",
            bd=1,
            highlightthickness=0,
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10, "bold"),
        ).pack(side="left", padx=(8, 0))

        tk.Button(
            actions,
            text="Cancel Run",
            command=self.on_cancel,
            width=12,
            padx=8,
            pady=8,
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            relief="raised",
            bd=1,
            highlightthickness=0,
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10, "bold"),
        ).pack(side="right")

        tk.Label(
            body,
            textvariable=self.key_status_var,
            bg=self.colors["panel"],
            fg="#93c5fd",
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10),
        ).pack(anchor="w", pady=(4, 0))

        hint = tk.Label(
            body,
            text="Use Left arrow to redo the run, Right arrow to start the next run after env reset.",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=(self.colors.get("font_ui", "TkDefaultFont"), 10),
        )
        hint.pack(anchor="w", pady=(4, 0))

        self.window.bind("<Left>", lambda _: self._send_key("left"))
        self.window.bind("<Right>", lambda _: self._send_key("right"))

    def _send_key(self, direction: str) -> None:
        ok, message = self.on_send_key(direction)
        if self.key_status_var is not None:
            self.key_status_var.set(message)
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

        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(self._episode_duration_s, 1.0))
            self.episode_progressbar["value"] = min(elapsed, self._episode_duration_s)
        if self.episode_timer_var is not None:
            self.episode_timer_var.set(
                f"Episode time: {self._fmt_seconds(elapsed)} / {self._fmt_seconds(self._episode_duration_s)}"
                f" (left {self._fmt_seconds(remaining)})"
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
            mode_label = run_mode.capitalize() if run_mode else "Run"
            self.mode_var.set(f"Run mode: {mode_label}")
        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode: {self._current_episode}/{self._total_episodes}")
            else:
                self.episode_var.set("Episode: --/--")
        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(self._episode_duration_s, 1.0), value=0)
        if self.episode_timer_var is not None:
            if self._episode_duration_s > 0:
                self.episode_timer_var.set(f"Episode time: 00:00 / {self._fmt_seconds(self._episode_duration_s)}")
            else:
                self.episode_timer_var.set("Episode time: --:-- / --:--")
        if self.key_status_var is not None:
            self.key_status_var.set("Controls ready: Left=Redo, Right=Next")

        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self._schedule_tick()

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
                self.episode_var.set(f"Episode: {self._current_episode}/{self._total_episodes}")
            else:
                self.episode_var.set(f"Episode: {self._current_episode}/--")

    def hide(self) -> None:
        self._active = False
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        if self.window is not None and bool(self.window.winfo_exists()):
            self.window.withdraw()
