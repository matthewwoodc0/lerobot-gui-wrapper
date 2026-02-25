from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


SubmitCallback = Callable[[str], tuple[bool, str]]
SimpleCallback = Callable[[], None]
InterruptCallback = Callable[[], tuple[bool, str]]


class GuiLogPanel:
    def __init__(
        self,
        root: Any,
        parent: Any,
        colors: dict[str, str],
        on_cancel: Callable[[], None],
        get_last_command: Callable[[], str],
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.colors = colors
        self._on_cancel = on_cancel
        self._get_last_command = get_last_command

        self._is_running: Callable[[], bool] = lambda: False
        self._submit_callback: SubmitCallback | None = None
        self._interrupt_callback: InterruptCallback | None = None
        self._toggle_terminal_callback: SimpleCallback | None = None
        self._show_history_callback: SimpleCallback | None = None
        self._open_latest_callback: SimpleCallback | None = None
        self._terminal_visible = True

        self._prompt_prefix = "▶ "
        self._input_history: list[str] = []
        self._history_index = 0

        self.output_panel = parent

        # Thin yellow left-border accent strip above the terminal header
        self._accent_strip = tk.Frame(self.output_panel, bg=colors.get("accent", "#f0a500"), height=2)
        self._accent_strip.pack(fill="x")

        output_header = ttk.Frame(self.output_panel, style="Panel.TFrame")
        output_header.pack(fill="x", pady=(4, 6))
        ttk.Label(output_header, text="Terminal", style="SectionTitle.TLabel").pack(side="left")

        self.history_button = ttk.Button(output_header, text="History", command=lambda: self._run_simple_callback(self._show_history_callback))
        self.history_button.pack(side="right")

        self.open_latest_button = ttk.Button(
            output_header,
            text="Open Latest Artifact",
            command=lambda: self._run_simple_callback(self._open_latest_callback),
        )
        self.open_latest_button.pack(side="right", padx=(6, 0))

        self.copy_cmd_button = ttk.Button(
            output_header,
            text="Copy Command",
            command=self._copy_last_command,
        )
        self.copy_cmd_button.pack(side="right", padx=(6, 0))

        self.toggle_button = ttk.Button(
            output_header,
            text="Hide Terminal",
            command=lambda: self._run_simple_callback(self._toggle_terminal_callback),
        )
        self.toggle_button.pack(side="right", padx=(6, 0))

        text_wrap = ttk.Frame(self.output_panel, style="Panel.TFrame")
        text_wrap.pack(fill="both", expand=True)
        self._text_wrap = text_wrap

        self.log_box = tk.Text(
            text_wrap,
            wrap="word",
            bg=self.colors.get("surface", "#1a1a1a"),
            fg=self.colors.get("text", "#cccccc"),
            insertbackground=self.colors.get("text", "#f8fafc"),
            font=(self.colors.get("font_mono", "TkFixedFont"), 10),
            relief="flat",
            padx=10,
            pady=10,
            undo=False,
            highlightthickness=1,
            highlightbackground=self.colors.get("border", "#2d2d2d"),
        )
        self.log_box.pack(side="left", fill="both", expand=True)

        self._configure_log_tags()

        self.log_box.bind("<KeyPress>", self._on_keypress)
        self.log_box.bind("<Button-1>", self._on_mouse_click, add="+")
        self.log_box.bind("<Up>", self._on_history_up)
        self.log_box.bind("<Down>", self._on_history_down)

        self._render_prompt("")

    def _configure_log_tags(self) -> None:
        self.log_box.tag_configure("default", foreground=self.colors.get("text", "#cccccc"))
        self.log_box.tag_configure("cmd", foreground=self.colors.get("accent", "#f0a500"))
        self.log_box.tag_configure("error", foreground=self.colors.get("error", "#f87171"))
        self.log_box.tag_configure("success", foreground=self.colors.get("success", "#4ade80"))
        self.log_box.tag_configure("timestamp", foreground=self.colors.get("muted", "#555555"))

    def apply_theme(self, updated_colors: dict[str, str]) -> None:
        self.colors = updated_colors
        try:
            self._accent_strip.configure(bg=self.colors.get("accent", "#f0a500"))
        except Exception:
            pass
        self.log_box.configure(
            bg=self.colors.get("surface", "#1a1a1a"),
            fg=self.colors.get("text", "#cccccc"),
            insertbackground=self.colors.get("text", "#f8fafc"),
            highlightbackground=self.colors.get("border", "#2d2d2d"),
            font=(self.colors.get("font_mono", "TkFixedFont"), 10),
        )
        self._configure_log_tags()

    def _run_simple_callback(self, callback: SimpleCallback | None) -> None:
        if callback is not None:
            callback()

    def set_submit_callback(self, callback: SubmitCallback) -> None:
        self._submit_callback = callback

    def set_interrupt_callback(self, callback: InterruptCallback) -> None:
        self._interrupt_callback = callback

    def set_toggle_terminal_callback(self, callback: SimpleCallback) -> None:
        self._toggle_terminal_callback = callback

    def set_show_history_callback(self, callback: SimpleCallback) -> None:
        self._show_history_callback = callback

    def set_open_latest_artifact_callback(self, callback: SimpleCallback) -> None:
        self._open_latest_callback = callback

    def set_terminal_visible(self, visible: bool) -> None:
        self._terminal_visible = bool(visible)
        self.toggle_button.configure(text="Hide Terminal" if visible else "Show Terminal")

    def set_running_state(self, active: bool) -> None:
        # Keep terminal input available during active runs so stdin can be sent.
        _ = active

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._on_cancel = callback

    def set_is_running_callback(self, callback: Callable[[], bool]) -> None:
        self._is_running = callback

    def classify_log_tag(self, line: str) -> str:
        lowered = line.lower()
        if line.startswith("$ ") or line.startswith("▶ "):
            return "cmd"
        if "exit code" in lowered and "[exit code 0]" not in lowered:
            return "error"
        if any(word in lowered for word in ("error", "failed", "traceback", "exception")):
            return "error"
        if any(word in lowered for word in ("completed", "done", "success")):
            return "success"
        return "default"

    def _current_input(self) -> str:
        return self.log_box.get("input_start", "end-1c")

    def _delete_prompt_and_input(self) -> None:
        try:
            self.log_box.delete("prompt_start", "end-1c")
        except Exception:
            pass

    def _render_prompt(self, input_text: str) -> None:
        content = self.log_box.get("1.0", "end-1c")
        if content and not content.endswith("\n"):
            self.log_box.insert("end", "\n")
        self.log_box.insert("end", self._prompt_prefix + input_text)
        line_start = self.log_box.index("end-1c linestart")
        self.log_box.mark_set("prompt_start", line_start)
        self.log_box.mark_set("input_start", f"{line_start}+{len(self._prompt_prefix)}c")
        self.log_box.mark_gravity("prompt_start", "left")
        self.log_box.mark_gravity("input_start", "left")
        self.log_box.mark_set("insert", "end-1c")
        self.log_box.see("end")

    def append_log(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        tag = self.classify_log_tag(line)
        existing_input = self._current_input()

        self._delete_prompt_and_input()

        content = self.log_box.get("1.0", "end-1c")
        if content and not content.endswith("\n"):
            self.log_box.insert("end", "\n")
        self.log_box.insert("end", f"[{timestamp}] ", ("timestamp",))
        self.log_box.insert("end", line, (tag,))

        self._render_prompt(existing_input)

    def scroll_to_first_error(self) -> None:
        match_index = self.log_box.search(
            r"(traceback|exception|error|failed)",
            "1.0",
            stopindex="end",
            regexp=True,
            nocase=True,
        )
        if match_index:
            self.log_box.see(match_index)
            self.log_box.mark_set("insert", match_index)

    def focus_input(self) -> None:
        self.log_box.focus_set()
        self.log_box.mark_set("insert", "end-1c")

    def _cursor_before_input(self) -> bool:
        return bool(self.log_box.compare("insert", "<", "input_start"))

    def _on_mouse_click(self, _: Any) -> None:
        # Keep historical output selectable but always edit at the current prompt.
        def move_if_needed() -> None:
            if self._cursor_before_input():
                self.log_box.mark_set("insert", "end-1c")

        self.root.after(0, move_if_needed)

    def _on_keypress(self, event: Any) -> str | None:
        ctrl_pressed = bool((event.state or 0) & 0x4)
        keysym = str(getattr(event, "keysym", ""))

        if ctrl_pressed and keysym.lower() == "c":
            if self.log_box.tag_ranges("sel"):
                return None
            if self._interrupt_callback is not None:
                ok, message = self._interrupt_callback()
                if message:
                    self.append_log(message)
                if not ok:
                    self.root.bell()
            return "break"

        if keysym == "Return":
            self._submit_input()
            return "break"

        if keysym == "Home":
            self.log_box.mark_set("insert", "input_start")
            return "break"

        if keysym in {"Left", "BackSpace"}:
            if self.log_box.compare("insert", "<=", "input_start"):
                return "break"
            return None

        if keysym == "Delete":
            if self.log_box.compare("insert", "<", "input_start"):
                return "break"
            return None

        if keysym in {"Up", "Down"}:
            # Handled by dedicated bindings.
            return None

        if self._cursor_before_input():
            self.log_box.mark_set("insert", "end-1c")

        return None

    def _submit_input(self) -> None:
        command = self._current_input()
        self._delete_prompt_and_input()
        content = self.log_box.get("1.0", "end-1c")
        if content and not content.endswith("\n"):
            self.log_box.insert("end", "\n")
        self.log_box.insert("end", self._prompt_prefix + command, ("cmd",))  # echo submitted command

        cleaned = command.strip()
        if cleaned:
            if not self._input_history or self._input_history[-1] != cleaned:
                self._input_history.append(cleaned)
            self._history_index = len(self._input_history)

        self._render_prompt("")

        if self._submit_callback is None:
            return

        if not command and not self._is_running():
            return

        ok, message = self._submit_callback(command)
        if message:
            self.append_log(message)
        if not ok:
            self.root.bell()

    def _replace_input_text(self, new_text: str) -> None:
        self.log_box.delete("input_start", "end-1c")
        self.log_box.insert("end", new_text)
        self.log_box.mark_set("insert", "end-1c")

    def _on_history_up(self, _: Any) -> str:
        if not self._input_history:
            return "break"
        self._history_index = max(self._history_index - 1, 0)
        self._replace_input_text(self._input_history[self._history_index])
        return "break"

    def _on_history_down(self, _: Any) -> str:
        if not self._input_history:
            return "break"
        self._history_index = min(self._history_index + 1, len(self._input_history))
        if self._history_index >= len(self._input_history):
            self._replace_input_text("")
        else:
            self._replace_input_text(self._input_history[self._history_index])
        return "break"

    def _copy_last_command(self) -> None:
        cmd = self._get_last_command().strip()
        if not cmd:
            self.append_log("No command to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.append_log("Copied last command to clipboard.")

    def open_latest_artifact(self, path: Path | None) -> None:
        if path is None:
            self.append_log("No run artifacts found yet.")
            return
        self.append_log(f"Latest artifact: {path}")
