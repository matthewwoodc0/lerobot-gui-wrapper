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
ARROW_ACK_PATTERN = re.compile(r"(left|right)\s+arrow\s+key\s+pressed", re.IGNORECASE)


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
        self.reset_prompt_var: Any | None = None
        self.outcome_status_var: Any | None = None
        self.outcome_summary_var: Any | None = None
        self.outcome_tags_var: Any | None = None
        self.episode_progressbar: Any | None = None
        self._dot_canvas: Any | None = None
        self._dot_item: Any | None = None
        self._outcome_frame: Any | None = None
        self._outcome_success_button: Any | None = None
        self._outcome_failed_button: Any | None = None
        self._outcome_history_tree: Any | None = None
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
        self._pending_direction: str | None = None
        self._pending_send_job: str | None = None
        self._reset_prompt_job: str | None = None
        self._reset_prompt_tick: int = 0

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
        success_col = self.colors.get("success", "#22c55e")

        self.window = tk.Toplevel(self.root)
        self.window.title("Run Controls")
        self.window.geometry("760x560")
        self.window.minsize(680, 460)
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
        body.pack(fill="x")

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

        self.reset_prompt_var = tk.StringVar(value="")
        tk.Label(
            controls,
            textvariable=self.reset_prompt_var,
            bg=panel,
            fg=accent,
            font=(ui_font, 10, "bold"),
            anchor="center",
        ).grid(row=2, column=0, columnspan=2, pady=(6, 0))

        self.window.bind("<Left>", lambda _: self._send_key("left"))
        self.window.bind("<Right>", lambda _: self._send_key("right"))

        # ── Outcome tracker (deploy mode) ───────────────────────────────────
        tk.Frame(self.window, bg=border, height=1).pack(fill="x")
        outcome = tk.Frame(self.window, bg=panel, padx=18, pady=10)
        outcome.pack(fill="both", expand=True)
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
            bg=success_col,
            fg="#000000",
            activebackground="#16a34a",
            activeforeground="#000000",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            font=(ui_font, 10, "bold"),
        )
        success_btn.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self._outcome_success_button = success_btn

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
            highlightthickness=1,
            highlightbackground=border,
            font=(ui_font, 10, "bold"),
        )
        failed_btn.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(8, 0))
        self._outcome_failed_button = failed_btn

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

        tk.Label(
            outcome,
            text="Episode status list",
            bg=panel,
            fg=muted,
            font=(ui_font, 9, "bold"),
            anchor="w",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 4))

        style = ttk.Style(self.window)
        style.configure(
            "Outcome.Treeview",
            font=(ui_font, 9),
            rowheight=22,
            background=surface,
            foreground=text_col,
            fieldbackground=surface,
            borderwidth=0,
        )
        style.configure(
            "Outcome.Treeview.Heading",
            font=(ui_font, 9, "bold"),
            background=panel,
            foreground=accent,
        )
        style.map(
            "Outcome.Treeview",
            background=[("selected", accent)],
            foreground=[("selected", "#000000")],
        )

        history_frame = tk.Frame(outcome, bg=panel)
        history_frame.grid(row=6, column=0, columnspan=4, sticky="nsew")
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        outcome.columnconfigure(1, weight=1)
        outcome.rowconfigure(6, weight=1)

        history_tree = ttk.Treeview(
            history_frame,
            columns=("episode", "status", "tags"),
            show="headings",
            style="Outcome.Treeview",
            height=6,
        )
        history_tree.heading("episode", text="Episode")
        history_tree.heading("status", text="Status")
        history_tree.heading("tags", text="Tags")
        history_tree.column("episode", width=90, anchor="w")
        history_tree.column("status", width=120, anchor="w")
        history_tree.column("tags", width=420, anchor="w")
        history_tree.tag_configure("pending_row", foreground=muted)
        history_tree.tag_configure("active_row", foreground=accent)
        history_tree.tag_configure("success_row", foreground=success_col)
        history_tree.tag_configure("failed_row", foreground=error_col)
        history_tree.grid(row=0, column=0, sticky="nsew")
        history_scroll = ttk.Scrollbar(
            history_frame,
            orient="vertical",
            command=history_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        history_scroll.grid(row=0, column=1, sticky="ns")
        history_tree.configure(yscrollcommand=history_scroll.set)
        self._outcome_history_tree = history_tree

        self._outcome_controls = [tags_entry, success_btn, failed_btn]
        self._refresh_outcome_history_rows()
        self._refresh_outcome_button_states()

    def _stop_reset_prompt(self) -> None:
        if self._reset_prompt_job is not None:
            try:
                self.root.after_cancel(self._reset_prompt_job)
            except Exception:
                pass
            self._reset_prompt_job = None
        if self.reset_prompt_var is not None:
            self.reset_prompt_var.set("")

    def _schedule_reset_prompt(self) -> None:
        if self._reset_prompt_job is not None:
            try:
                self.root.after_cancel(self._reset_prompt_job)
            except Exception:
                pass
            self._reset_prompt_job = None
        self._reset_prompt_job = self.root.after(450, self._tick_reset_prompt)

    def _tick_reset_prompt(self) -> None:
        self._reset_prompt_job = None
        if not self._active:
            self._stop_reset_prompt()
            return
        if not self._awaiting_next_episode or self._current_episode <= 0:
            self._stop_reset_prompt()
            return

        if self.reset_prompt_var is not None:
            dot_count = (self._reset_prompt_tick % 4)
            dots = "." * dot_count
            self.reset_prompt_var.set(f"Reset the environment{dots}")
        self._reset_prompt_tick += 1
        self._schedule_reset_prompt()

    def _show_reset_prompt(self) -> None:
        if not self._active:
            return
        if self._current_episode <= 0:
            return
        self._reset_prompt_tick = 1
        if self.reset_prompt_var is not None:
            self.reset_prompt_var.set("Reset the environment...")
        self._schedule_reset_prompt()

    def _dispatch_key(self, direction: str) -> None:
        ok, message = self.on_send_key(direction)
        if self.key_status_var is not None:
            self.key_status_var.set(message if message else "← Reset episode        ·        → Next episode")
        if not ok:
            self.root.bell()
            return
        self._pending_direction = None
        if self._pending_send_job is not None:
            try:
                self.root.after_cancel(self._pending_send_job)
            except Exception:
                pass
            self._pending_send_job = None

    def _send_key(self, direction: str) -> None:
        if self._awaiting_next_episode:
            self._dispatch_key(direction)
        else:
            self._pending_direction = direction
            label = "Reset episode" if direction == "left" else "Start next episode"
            if self.key_status_var is not None:
                self.key_status_var.set(f"{label}: queued until reset phase.")
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
            self._show_reset_prompt()

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
        self._refresh_outcome_button_states()

    def _outcome_for_episode(self, episode_idx: int) -> str | None:
        if episode_idx <= 0:
            return None
        item = self._episode_outcomes.get(episode_idx)
        if not isinstance(item, dict):
            return None
        result = str(item.get("result", "")).strip().lower()
        if result in {"success", "failed"}:
            return result
        return None

    def _episode_history_indices(self) -> list[int]:
        if self._total_episodes > 0:
            return list(range(1, self._total_episodes + 1))
        indices = sorted(idx for idx in self._episode_outcomes if idx > 0)
        if self._current_episode > 0 and self._current_episode not in indices:
            indices.append(self._current_episode)
        return indices

    def _episode_status_label(self, episode_idx: int) -> str:
        result = self._outcome_for_episode(episode_idx)
        if result == "success":
            return "Success"
        if result == "failed":
            return "Failed"
        if (
            self._active
            and episode_idx == self._current_episode
            and not self._awaiting_next_episode
        ):
            return "Active"
        return "Pending"

    def _refresh_outcome_history_rows(self) -> None:
        tree = self._outcome_history_tree
        if tree is None:
            return

        wanted = self._episode_history_indices()
        wanted_ids = {f"ep_{idx}" for idx in wanted}
        for iid in tree.get_children(""):
            if iid not in wanted_ids:
                tree.delete(iid)

        for idx in wanted:
            iid = f"ep_{idx}"
            status = self._episode_status_label(idx)
            outcome = self._episode_outcomes.get(idx, {})
            tags = outcome.get("tags", []) if isinstance(outcome, dict) else []
            tag_text = ", ".join(str(tag) for tag in tags) if tags else "-"
            row_tag = "pending_row"
            if status == "Active":
                row_tag = "active_row"
            elif status == "Success":
                row_tag = "success_row"
            elif status == "Failed":
                row_tag = "failed_row"

            values = (str(idx), status, tag_text)
            if tree.exists(iid):
                tree.item(iid, values=values, tags=(row_tag,))
            else:
                tree.insert("", "end", iid=iid, values=values, tags=(row_tag,))

        if self._current_episode > 0:
            current_iid = f"ep_{self._current_episode}"
            if tree.exists(current_iid):
                tree.see(current_iid)

    def _refresh_outcome_button_states(self) -> None:
        success_btn = self._outcome_success_button
        failed_btn = self._outcome_failed_button
        if success_btn is None or failed_btn is None:
            return

        border = self.colors.get("border", "#2d2d2d")
        success_col = self.colors.get("success", "#22c55e")
        failed_col = self.colors.get("error", "#ef4444")
        enabled = self._allow_outcome_marking
        result = self._outcome_for_episode(self._current_episode)
        success_selected = enabled and result == "success"
        failed_selected = enabled and result == "failed"

        success_btn.configure(
            text="✓ Success" if success_selected else "Mark Success",
            highlightbackground=success_col if success_selected else border,
            highlightcolor=success_col if success_selected else border,
            highlightthickness=2 if success_selected else 1,
        )
        failed_btn.configure(
            text="✓ Failed" if failed_selected else "Mark Failed",
            highlightbackground=failed_col if failed_selected else border,
            highlightcolor=failed_col if failed_selected else border,
            highlightthickness=2 if failed_selected else 1,
        )

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
        self._refresh_outcome_history_rows()
        self._refresh_outcome_button_states()

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
        previous = self._outcome_for_episode(self._current_episode)
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
            if previous and previous != result:
                prev_label = "Success" if previous == "success" else "Failed"
                self.outcome_status_var.set(
                    f"Episode {self._current_episode} changed from {prev_label} to {label} ({tags_text})."
                )
            elif previous == result:
                self.outcome_status_var.set(
                    f"Episode {self._current_episode} remains {label} ({tags_text})."
                )
            else:
                self.outcome_status_var.set(
                    f"Episode {self._current_episode} marked {label} ({tags_text})."
                )

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
        self._pending_direction = None
        self._stop_reset_prompt()
        if self._pending_send_job is not None:
            try:
                self.root.after_cancel(self._pending_send_job)
            except Exception:
                pass
            self._pending_send_job = None

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
        self._stop_reset_prompt()

        if self._outcome_frame is not None:
            if self._allow_outcome_marking:
                self._outcome_frame.pack(fill="both", expand=True)
                if self.outcome_status_var is not None:
                    self.outcome_status_var.set("Set result after each deployment episode.")
            else:
                self._outcome_frame.pack_forget()
        self._set_outcome_controls_enabled(self._allow_outcome_marking)
        self._update_outcome_summary_label()
        self._refresh_outcome_history_rows()
        self._refresh_outcome_button_states()

        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self._schedule_tick()
        self._dot_bright = True
        self._pulse_dot()

    def handle_output_line(self, line: str) -> None:
        if not self._active:
            return
        if ARROW_ACK_PATTERN.search(line):
            self._pending_direction = None
            if self._pending_send_job is not None:
                try:
                    self.root.after_cancel(self._pending_send_job)
                except Exception:
                    pass
                self._pending_send_job = None
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
            self._show_reset_prompt()
            if self._pending_direction is not None:
                pending = self._pending_direction

                def send_pending() -> None:
                    self._pending_send_job = None
                    if not self._active:
                        return
                    if self._pending_direction != pending:
                        return
                    if not self._awaiting_next_episode:
                        return
                    self._dispatch_key(pending)

                if self._pending_send_job is not None:
                    try:
                        self.root.after_cancel(self._pending_send_job)
                    except Exception:
                        pass
                self._pending_send_job = self.root.after(120, send_pending)
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

        is_episode_transition = (
            display_current != self._current_episode
            or self._awaiting_next_episode
            or self._episode_started_at is None
        )
        if is_episode_transition:
            self._current_episode = display_current
            self._episode_started_at = time.monotonic()
            self._awaiting_next_episode = False
            self._stop_reset_prompt()
            self._pending_direction = None
            if self._pending_send_job is not None:
                try:
                    self.root.after_cancel(self._pending_send_job)
                except Exception:
                    pass
                self._pending_send_job = None
            if self.episode_progressbar is not None:
                self.episode_progressbar["value"] = 0
            had_previous_outcome = self._episode_outcomes.pop(self._current_episode, None) is not None
            if had_previous_outcome:
                self._update_outcome_summary_label()
            if self.outcome_status_var is not None and self._allow_outcome_marking:
                if had_previous_outcome:
                    self.outcome_status_var.set(
                        f"Episode {self._current_episode} restarted. Previous mark cleared; re-rate after this take."
                    )
                else:
                    self.outcome_status_var.set(
                        f"Episode {self._current_episode} active. Mark success or failed when done."
                    )
            self._refresh_outcome_history_rows()
            self._refresh_outcome_button_states()

        if self.episode_var is not None:
            if self._total_episodes > 0:
                self.episode_var.set(f"Episode  {self._current_episode} / {self._total_episodes}")
            else:
                self.episode_var.set(f"Episode  {self._current_episode} / --")
        self._refresh_outcome_history_rows()
        self._refresh_outcome_button_states()

    def hide(self) -> None:
        self._active = False
        self._awaiting_next_episode = False
        self._stop_reset_prompt()
        self._pending_direction = None
        if self._pulse_job is not None:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        if self._pending_send_job is not None:
            self.root.after_cancel(self._pending_send_job)
            self._pending_send_job = None
        if self.window is not None and bool(self.window.winfo_exists()):
            self.window.withdraw()
