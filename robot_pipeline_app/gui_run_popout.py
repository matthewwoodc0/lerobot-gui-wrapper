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
RESET_PHASE_PATTERNS = [
    re.compile(r"(left|right)\s+arrow", re.IGNORECASE),
    re.compile(r"redo.*next|next.*redo", re.IGNORECASE),
    re.compile(r"reset.*episode|next.*episode", re.IGNORECASE),
    re.compile(r"reset\s+the\s+environment", re.IGNORECASE),
]
START_PHASE_PATTERNS = [
    re.compile(r"\brecording\s+episode\b", re.IGNORECASE),
    re.compile(r"\bepisode\b.*\bstarted\b", re.IGNORECASE),
]


def parse_episode_progress_line(line: str) -> tuple[int, int | None] | None:
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    partial = EPISODE_PARTIAL_PATTERN.search(line)
    if partial:
        return int(partial.group(1)), None
    return None


def is_episode_reset_phase_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in RESET_PHASE_PATTERNS)


def is_episode_start_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in START_PHASE_PATTERNS)


def parse_outcome_tags(raw: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").split(","):
        tag = chunk.strip()
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags


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
        self.outcome_status_var: Any | None = None
        self.outcome_summary_var: Any | None = None
        self.outcome_tags_var: Any | None = None
        self.episode_progressbar: Any | None = None
        self._dot_canvas: Any | None = None
        self._dot_item: Any | None = None
        self._outcome_frame: Any | None = None
        self._outcome_controls: list[Any] = []
        self._dot_bright = True

        self._active = False
        self._run_mode = "run"
        self._total_episodes = 0
        self._episode_duration_s = 0.0
        self._current_episode = 0
        self._episode_started_at: float | None = None
        self._awaiting_next_episode = False
        self._zero_based_indexing: bool | None = None
        self._allow_outcome_marking = False
        self._episode_outcomes: dict[int, dict[str, Any]] = {}

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
        self.window.geometry("700x430")
        self.window.minsize(600, 360)
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

        tk.Label(
            header,
            text="Episode controls",
            bg="#0d0d0d",
            fg=muted,
            font=(ui_font, 9),
        ).pack(side="left", padx=(12, 0))

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
        body = tk.Frame(self.window, bg=panel, padx=18, pady=14)
        body.pack(fill="both", expand=True)

        # Episode counter
        self.episode_var = tk.StringVar(value="Episode  -- / --")
        tk.Label(
            body,
            textvariable=self.episode_var,
            bg=panel,
            fg=text_col,
            font=(ui_font, 17, "bold"),
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
        controls = tk.Frame(self.window, bg=panel, padx=18, pady=12)
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        redo_btn = tk.Button(
            controls,
            text="←  Reset Episode",
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
            text="Next Episode  →",
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
        self.key_status_var = tk.StringVar(value="← Reset episode        ·        → Next episode")
        tk.Label(
            controls,
            textvariable=self.key_status_var,
            bg=panel,
            fg=muted,
            font=(ui_font, 9),
        ).grid(row=1, column=0, columnspan=2, pady=(4, 0))

        self.window.bind("<Left>", lambda _: self._send_key("left"))
        self.window.bind("<Right>", lambda _: self._send_key("right"))

        # ── Outcome tracker (deploy mode) ───────────────────────────────────
        tk.Frame(self.window, bg=border, height=1).pack(fill="x")
        outcome = tk.Frame(self.window, bg=panel, padx=18, pady=10)
        outcome.pack(fill="x")
        self._outcome_frame = outcome

        tk.Label(
            outcome,
            text="Episode Outcome Tracker",
            bg=panel,
            fg=text_col,
            font=(ui_font, 11, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(
            outcome,
            text="Tags (optional, comma-separated):",
            bg=panel,
            fg=muted,
            font=(ui_font, 9),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.outcome_tags_var = tk.StringVar(value="")
        tags_entry = tk.Entry(
            outcome,
            textvariable=self.outcome_tags_var,
            bg=surface,
            fg=text_col,
            insertbackground=text_col,
            relief="flat",
            highlightthickness=1,
            highlightbackground=border,
            font=(ui_font, 10),
        )
        tags_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(6, 0))

        success_btn = tk.Button(
            outcome,
            text="Mark Success",
            command=lambda: self._mark_episode_outcome("success"),
            padx=12,
            pady=6,
            bg=self.colors.get("success", "#22c55e"),
            fg="#000000",
            activebackground="#16a34a",
            activeforeground="#000000",
            relief="flat",
            bd=0,
            font=(ui_font, 10, "bold"),
        )
        success_btn.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        failed_btn = tk.Button(
            outcome,
            text="Mark Failed",
            command=lambda: self._mark_episode_outcome("failed"),
            padx=12,
            pady=6,
            bg=self.colors.get("error", "#ef4444"),
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=(ui_font, 10, "bold"),
        )
        failed_btn.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        self.outcome_status_var = tk.StringVar(value="Set result after each deployment episode.")
        tk.Label(
            outcome,
            textvariable=self.outcome_status_var,
            bg=panel,
            fg=muted,
            font=(ui_font, 9),
            anchor="w",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.outcome_summary_var = tk.StringVar(value="Success: 0  |  Failed: 0  |  Rated: 0")
        tk.Label(
            outcome,
            textvariable=self.outcome_summary_var,
            bg=panel,
            fg=text_col,
            font=(ui_font, 9, "bold"),
            anchor="w",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

        outcome.columnconfigure(1, weight=1)
        self._outcome_controls = [tags_entry, success_btn, failed_btn]

    def _send_key(self, direction: str) -> None:
        ok, message = self.on_send_key(direction)
        if self.key_status_var is not None:
            self.key_status_var.set(message if message else "← Reset episode        ·        → Next episode")
        if not ok:
            self.root.bell()
        if self.window is not None and bool(self.window.winfo_exists()):
            self.window.focus_force()

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
        total = self._episode_duration_s
        if elapsed >= total:
            elapsed = total
            self._awaiting_next_episode = True
            self._episode_started_at = None

        remaining = max(total - elapsed, 0.0)

        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(total, 1.0))
            self.episode_progressbar["value"] = min(elapsed, total)
        if self.episode_timer_var is not None:
            self.episode_timer_var.set(
                f"{self._fmt_seconds(elapsed)} elapsed  ·  {self._fmt_seconds(total)} total  ·  ↤ {self._fmt_seconds(remaining)}"
            )

        self._schedule_tick()

    def _set_outcome_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._outcome_controls:
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _update_outcome_summary_label(self) -> None:
        if self.outcome_summary_var is None:
            return
        success = sum(1 for item in self._episode_outcomes.values() if item.get("result") == "success")
        failed = sum(1 for item in self._episode_outcomes.values() if item.get("result") == "failed")
        rated = len(self._episode_outcomes)
        if self._total_episodes > 0:
            summary = f"Success: {success}  |  Failed: {failed}  |  Rated: {rated}/{self._total_episodes}"
        else:
            summary = f"Success: {success}  |  Failed: {failed}  |  Rated: {rated}"
        self.outcome_summary_var.set(summary)

    def _mark_episode_outcome(self, result: str) -> None:
        if not self._allow_outcome_marking:
            if self.outcome_status_var is not None:
                self.outcome_status_var.set("Episode outcome tracking is enabled for deploy runs.")
            self.root.bell()
            return
        if self._current_episode <= 0:
            if self.outcome_status_var is not None:
                self.outcome_status_var.set("Wait for 'Recording episode ...' before marking an outcome.")
            self.root.bell()
            return

        tags = parse_outcome_tags(self.outcome_tags_var.get() if self.outcome_tags_var is not None else "")
        self._episode_outcomes[self._current_episode] = {
            "episode": self._current_episode,
            "result": result,
            "tags": tags,
            "updated_at_epoch_s": round(time.time(), 3),
        }
        self._update_outcome_summary_label()
        if self.outcome_status_var is not None:
            label = "Success" if result == "success" else "Failed"
            tags_text = ", ".join(tags) if tags else "no tags"
            self.outcome_status_var.set(f"Episode {self._current_episode} marked {label} ({tags_text}).")

    def get_episode_outcome_summary(self) -> dict[str, Any] | None:
        if not self._episode_outcomes and not self._allow_outcome_marking:
            return None
        outcomes = [self._episode_outcomes[idx] for idx in sorted(self._episode_outcomes)]
        success = sum(1 for item in outcomes if item.get("result") == "success")
        failed = sum(1 for item in outcomes if item.get("result") == "failed")
        tags: set[str] = set()
        for item in outcomes:
            for tag in item.get("tags", []):
                tags.add(str(tag))

        rated = len(outcomes)
        total = self._total_episodes
        return {
            "enabled": self._allow_outcome_marking,
            "total_episodes": total,
            "rated_count": rated,
            "success_count": success,
            "failed_count": failed,
            "unrated_count": max(total - rated, 0) if total > 0 else None,
            "tags": sorted(tags),
            "episode_outcomes": outcomes,
        }

    def start_run(self, run_mode: str, expected_episodes: int | None, expected_seconds: int | None) -> None:
        self._ensure_window()
        if self.window is None:
            return

        self._run_mode = run_mode.strip().lower() if run_mode else "run"
        self._allow_outcome_marking = self._run_mode == "deploy"
        self._episode_outcomes = {}
        episodes = int(expected_episodes or 0)
        seconds = float(expected_seconds or 0)
        self._total_episodes = episodes
        self._episode_duration_s = (seconds / episodes) if episodes > 0 and seconds > 0 else 0.0
        self._current_episode = 0
        self._episode_started_at = None
        self._awaiting_next_episode = True if episodes > 0 else False
        self._zero_based_indexing = None
        self._active = True

        if self.mode_var is not None:
            mode_label = run_mode.upper() if run_mode else "RUN"
            self.mode_var.set(f"{mode_label} MODE")

        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode  -- / {self._total_episodes}")
            else:
                self.episode_var.set("Episode  -- / --")

        if self.episode_progressbar is not None:
            self.episode_progressbar.configure(maximum=max(self._episode_duration_s, 1.0), value=0)

        if self.episode_timer_var is not None:
            if self._episode_duration_s > 0:
                self.episode_timer_var.set(
                    f"Waiting for recording...  ·  {self._fmt_seconds(self._episode_duration_s)} total"
                )
            else:
                self.episode_timer_var.set("00:00 elapsed  ·  --:-- total  ·  ↤ --:--")

        if self.key_status_var is not None:
            self.key_status_var.set("← Reset episode        ·        → Next episode")

        if self._outcome_frame is not None:
            if self._allow_outcome_marking:
                self._outcome_frame.pack(fill="x")
                if self.outcome_status_var is not None:
                    self.outcome_status_var.set("Set result after each deployment episode.")
            else:
                self._outcome_frame.pack_forget()
        self._set_outcome_controls_enabled(self._allow_outcome_marking)
        self._update_outcome_summary_label()

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
        if parsed is None and is_episode_reset_phase_line(line):
            self._awaiting_next_episode = True
            if self._episode_duration_s > 0 and self._episode_started_at is not None:
                elapsed = min(time.monotonic() - self._episode_started_at, self._episode_duration_s)
                if self.episode_progressbar is not None:
                    self.episode_progressbar.configure(maximum=max(self._episode_duration_s, 1.0))
                    self.episode_progressbar["value"] = elapsed
                if self.episode_timer_var is not None:
                    remaining = max(self._episode_duration_s - elapsed, 0.0)
                    self.episode_timer_var.set(
                        f"{self._fmt_seconds(elapsed)} elapsed  ·  {self._fmt_seconds(self._episode_duration_s)} total  ·  ↤ {self._fmt_seconds(remaining)}"
                    )
                self._episode_started_at = None
            elif self.episode_timer_var is not None and self._episode_duration_s > 0:
                self.episode_timer_var.set(
                    f"Waiting for recording...  ·  {self._fmt_seconds(self._episode_duration_s)} total"
                )
            return
        if parsed is None:
            return

        if not is_episode_start_line(line):
            return

        current, total = parsed
        if total is not None and total > 0:
            self._total_episodes = total

        if self._zero_based_indexing is None:
            self._zero_based_indexing = current == 0
        display_current = current + 1 if self._zero_based_indexing else current
        if display_current <= 0:
            display_current = 1

        if display_current != self._current_episode:
            self._current_episode = display_current
            self._episode_started_at = time.monotonic()
            self._awaiting_next_episode = False
            if self.episode_progressbar is not None:
                self.episode_progressbar["value"] = 0
            if self.outcome_status_var is not None and self._allow_outcome_marking:
                self.outcome_status_var.set(f"Episode {self._current_episode} active. Mark success or failed when done.")

        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode  {self._current_episode} / {self._total_episodes}")
            else:
                self.episode_var.set(f"Episode  {self._current_episode} / --")

    def hide(self) -> None:
        self._active = False
        self._awaiting_next_episode = False
        if self._pulse_job is not None:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        if self.window is not None and bool(self.window.winfo_exists()):
            self.window.withdraw()
