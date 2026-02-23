from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable


EPISODE_PATTERNS = [
    re.compile(r"[Ee]pisode\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"[Ee]pisode\s+(\d+)\s+of\s+(\d+)"),
]
EPISODE_PARTIAL_PATTERN = re.compile(r"[Ee]pisode\s+(\d+)")


def parse_episode_progress_line(line: str) -> tuple[int, int | None] | None:
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    partial = EPISODE_PARTIAL_PATTERN.search(line)
    if partial:
        return int(partial.group(1)), None
    return None


class GuiLogPanel:
    def __init__(
        self,
        root: Any,
        parent: Any,
        colors: dict[str, str],
        on_cancel: Callable[[], None],
        get_last_command: Callable[[], str],
    ) -> None:
        from tkinter import filedialog, messagebox, scrolledtext, ttk

        self.root = root
        self.colors = colors
        self._filedialog = filedialog
        self._messagebox = messagebox
        self._timer_job: str | None = None
        self._start_time: float | None = None
        self._expected_seconds: float = 0.0
        self._episodes_total: int = 0
        self._is_running: Callable[[], bool] = lambda: False
        self._get_last_command = get_last_command

        self.output_panel = parent
        output_header = ttk.Frame(self.output_panel, style="Panel.TFrame")
        output_header.pack(fill="x", pady=(0, 6))
        ttk.Label(output_header, text="Terminal Output", style="SectionTitle.TLabel").pack(side="left")

        progress_wrap = ttk.Frame(output_header, style="Panel.TFrame")
        progress_wrap.pack(side="right")

        self.clear_log_button = ttk.Button(output_header, text="Clear Log", command=self.clear_log)
        self.clear_log_button.pack(side="right")
        self.save_log_button = ttk.Button(output_header, text="Save Log", command=self.save_log_to_file)
        self.save_log_button.pack(side="right", padx=(6, 0))
        self.copy_command_button = ttk.Button(output_header, text="Copy Last Command", command=self.copy_last_command)
        self.copy_command_button.pack(side="right", padx=(6, 0))
        self.cancel_run_button = ttk.Button(output_header, text="Cancel Run", command=on_cancel)
        self.cancel_run_button.pack(side="right", padx=(6, 0))
        self.cancel_run_button.configure(state="disabled")

        self.progress_frame = ttk.Frame(self.output_panel, style="Panel.TFrame")
        self.progress_frame.pack(fill="x", pady=(0, 6))

        self.episode_progress_var = self._stringvar("Episode progress: --/--")
        self.time_progress_var = self._stringvar("Run time: 00:00 / --:--")

        ttk.Label(self.progress_frame, textvariable=self.episode_progress_var, style="Field.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(self.progress_frame, textvariable=self.time_progress_var, style="Muted.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )

        self.episode_progressbar = ttk.Progressbar(self.progress_frame, mode="determinate", style="Accent.Horizontal.TProgressbar")
        self.episode_progressbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        self.time_progressbar = ttk.Progressbar(self.progress_frame, mode="determinate", style="Time.Horizontal.TProgressbar")
        self.time_progressbar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)
        self.progress_frame.columnconfigure(1, weight=1)
        self.episode_progressbar.configure(maximum=1, value=0)
        self.time_progressbar.configure(maximum=1, value=0)

        self.log_box = scrolledtext.ScrolledText(
            self.output_panel,
            height=18,
            state="disabled",
            bg="#111827",
            fg="#d4d4d4",
            insertbackground="#f8fafc",
            font=("Menlo", 11),
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.tag_configure("default", foreground="#d4d4d4")
        self.log_box.tag_configure("cmd", foreground="#fbbf24")
        self.log_box.tag_configure("error", foreground="#f87171")
        self.log_box.tag_configure("success", foreground="#4ade80")
        self.log_box.tag_configure("episode", foreground="#38bdf8")

    def _stringvar(self, value: str) -> Any:
        import tkinter as tk

        return tk.StringVar(value=value)

    def set_running_state(self, active: bool) -> None:
        self.cancel_run_button.configure(state="normal" if active else "disabled")

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self.cancel_run_button.configure(command=callback)

    def set_is_running_callback(self, callback: Callable[[], bool]) -> None:
        self._is_running = callback

    def classify_log_tag(self, line: str) -> str:
        lowered = line.lower()
        if line.startswith("$ "):
            return "cmd"
        if "exit code" in lowered and "[exit code 0]" not in lowered:
            return "error"
        if any(word in lowered for word in ("error", "failed", "traceback", "exception")):
            return "error"
        if any(word in lowered for word in ("completed", "done", "success")):
            return "success"
        if "episode" in lowered:
            return "episode"
        return "default"

    def append_log(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        tag = self.classify_log_tag(line)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {line}\n", (tag,))
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def save_log_to_file(self) -> None:
        default_name = f"lerobot_gui_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        save_path = self._filedialog.asksaveasfilename(
            title="Save Terminal Log",
            defaultextension=".log",
            initialfile=default_name,
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not save_path:
            return
        try:
            text = self.log_box.get("1.0", "end-1c")
            Path(save_path).write_text(text, encoding="utf-8")
            self.append_log(f"Saved log to {save_path}")
        except OSError as exc:
            self._messagebox.showerror("Save Log Failed", str(exc))

    def copy_last_command(self) -> None:
        cmd_text = self._get_last_command()
        if not cmd_text:
            self._messagebox.showinfo("Copy Command", "No command has been run yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd_text)
        self.append_log("Copied last command to clipboard.")

    def format_seconds(self, value: float) -> str:
        seconds = max(int(value), 0)
        minutes, sec = divmod(seconds, 60)
        return f"{minutes:02d}:{sec:02d}"

    def update_episode_progress(self, current: int, total: int | None = None) -> None:
        if total is not None and total > 0:
            self._episodes_total = total
        total_value = int(self._episodes_total or 0)
        if total_value > 0:
            self.episode_progressbar.configure(maximum=total_value)
            self.episode_progressbar["value"] = min(current, total_value)
            self.episode_progress_var.set(f"Episode progress: {min(current, total_value)} / {total_value}")
        else:
            self.episode_progressbar.configure(maximum=max(current, 1))
            self.episode_progressbar["value"] = current
            self.episode_progress_var.set(f"Episode progress: {current} / --")

    def update_progress_from_line(self, line: str) -> None:
        parsed = parse_episode_progress_line(line)
        if parsed is None:
            return
        current, total = parsed
        self.update_episode_progress(current, total)

    def _progress_tick(self) -> None:
        if not self._is_running():
            self._timer_job = None
            return
        if self._start_time is not None:
            elapsed = time.monotonic() - float(self._start_time)
            expected = float(self._expected_seconds)
            if expected > 0:
                self.time_progressbar.configure(maximum=max(expected, 1))
                self.time_progressbar["value"] = min(elapsed, expected)
                self.time_progress_var.set(f"Run time: {self.format_seconds(elapsed)} / {self.format_seconds(expected)}")
            else:
                self.time_progressbar.configure(maximum=max(elapsed, 1))
                self.time_progressbar["value"] = elapsed
                self.time_progress_var.set(f"Run time: {self.format_seconds(elapsed)} / --:--")
        self._timer_job = self.root.after(500, self._progress_tick)

    def prepare_progress(self, expected_episodes: int | None, expected_seconds: int | None) -> None:
        self.stop_progress()

        self._start_time = time.monotonic()
        self._expected_seconds = float(expected_seconds or 0)
        self._episodes_total = int(expected_episodes or 0)

        if self._episodes_total > 0:
            total = int(self._episodes_total)
            self.episode_progressbar.configure(maximum=total, value=0)
            self.episode_progress_var.set(f"Episode progress: 0 / {total}")
        else:
            self.episode_progressbar.configure(maximum=1, value=0)
            self.episode_progress_var.set("Episode progress: --/--")

        if self._expected_seconds > 0:
            self.time_progressbar.configure(maximum=max(self._expected_seconds, 1), value=0)
            self.time_progress_var.set(f"Run time: 00:00 / {self.format_seconds(self._expected_seconds)}")
        else:
            self.time_progressbar.configure(maximum=1, value=0)
            self.time_progress_var.set("Run time: 00:00 / --:--")

        self._timer_job = self.root.after(500, self._progress_tick)

    def stop_progress(self) -> None:
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
