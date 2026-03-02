from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


SubmitCallback = Callable[[str], tuple[bool, str]]
SimpleCallback = Callable[[], None]
InterruptCallback = Callable[[], tuple[bool, str]]
TerminalInputCallback = Callable[[bytes], tuple[bool, str]]


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
        self._terminal_input_callback: TerminalInputCallback | None = None
        self._toggle_terminal_callback: SimpleCallback | None = None
        self._show_history_callback: SimpleCallback | None = None
        self._open_latest_callback: SimpleCallback | None = None
        self._terminal_visible = True
        self._pending_escape = ""
        self._saved_cursor_index = "1.0"

        self.output_panel = parent

        self._accent_strip = tk.Frame(self.output_panel, bg=colors.get("accent", "#f0a500"), height=2)
        self._accent_strip.pack(fill="x")

        output_header = ttk.Frame(self.output_panel, style="Panel.TFrame")
        output_header.pack(fill="x", pady=(4, 6))
        ttk.Label(output_header, text="Output", style="SectionTitle.TLabel").pack(side="left")

        self.history_button = ttk.Button(
            output_header,
            text="History Tab",
            command=lambda: self._run_simple_callback(self._show_history_callback),
        )
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

        self.output_tabs = ttk.Notebook(text_wrap)
        self.output_tabs.pack(fill="both", expand=True)

        terminal_frame = ttk.Frame(self.output_tabs, style="Panel.TFrame")
        history_frame = ttk.Frame(self.output_tabs, style="Panel.TFrame")
        self.output_tabs.add(terminal_frame, text="Terminal")
        self.output_tabs.add(history_frame, text="Run Log")

        self.terminal_box = tk.Text(
            terminal_frame,
            wrap="none",
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
        self.terminal_box.pack(side="left", fill="both", expand=True)

        terminal_scroll_y = ttk.Scrollbar(terminal_frame, orient="vertical", command=self.terminal_box.yview)
        terminal_scroll_y.pack(side="right", fill="y")
        terminal_scroll_x = ttk.Scrollbar(terminal_frame, orient="horizontal", command=self.terminal_box.xview)
        terminal_scroll_x.pack(side="bottom", fill="x")
        self.terminal_box.configure(yscrollcommand=terminal_scroll_y.set, xscrollcommand=terminal_scroll_x.set)

        self.log_box = tk.Text(
            history_frame,
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

        log_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.log_box.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=log_scroll.set)

        self._configure_log_tags()

        self.terminal_box.bind("<KeyPress>", self._on_terminal_keypress)
        self.terminal_box.bind("<Button-1>", self._on_terminal_click, add="+")

        self._reset_terminal_buffer()

    def _configure_log_tags(self) -> None:
        self.log_box.tag_configure("default", foreground=self.colors.get("text", "#cccccc"))
        self.log_box.tag_configure("cmd", foreground=self.colors.get("accent", "#f0a500"))
        self.log_box.tag_configure("error", foreground=self.colors.get("error", "#f87171"))
        self.log_box.tag_configure("success", foreground=self.colors.get("success", "#4ade80"))
        self.log_box.tag_configure("timestamp", foreground=self.colors.get("muted", "#555555"))

    def _reset_terminal_buffer(self) -> None:
        self.terminal_box.delete("1.0", "end")
        self.terminal_box.insert("1.0", "")
        self.terminal_box.mark_set("term_cursor", "1.0")
        self.terminal_box.mark_gravity("term_cursor", "right")
        self._saved_cursor_index = "1.0"
        self._pending_escape = ""

    def _ensure_terminal_cursor(self) -> None:
        if "term_cursor" not in self.terminal_box.mark_names():
            self.terminal_box.mark_set("term_cursor", "end-1c")
            self.terminal_box.mark_gravity("term_cursor", "right")

    def apply_theme(self, updated_colors: dict[str, str]) -> None:
        self.colors = updated_colors
        try:
            self._accent_strip.configure(bg=self.colors.get("accent", "#f0a500"))
        except Exception:
            pass

        shared_cfg = {
            "bg": self.colors.get("surface", "#1a1a1a"),
            "fg": self.colors.get("text", "#cccccc"),
            "insertbackground": self.colors.get("text", "#f8fafc"),
            "highlightbackground": self.colors.get("border", "#2d2d2d"),
            "font": (self.colors.get("font_mono", "TkFixedFont"), 10),
        }
        self.terminal_box.configure(**shared_cfg)
        self.log_box.configure(**shared_cfg)
        self._configure_log_tags()

    def _run_simple_callback(self, callback: SimpleCallback | None) -> None:
        if callback is not None:
            callback()

    def set_submit_callback(self, callback: SubmitCallback) -> None:
        # Legacy callback retained for compatibility.
        self._submit_callback = callback

    def set_interrupt_callback(self, callback: InterruptCallback) -> None:
        self._interrupt_callback = callback

    def set_terminal_input_callback(self, callback: TerminalInputCallback) -> None:
        self._terminal_input_callback = callback

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

    def append_log(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        tag = self.classify_log_tag(line)

        content = self.log_box.get("1.0", "end-1c")
        if content and not content.endswith("\n"):
            self.log_box.insert("end", "\n")
        self.log_box.insert("end", f"[{timestamp}] ", ("timestamp",))
        self.log_box.insert("end", line, (tag,))
        self.log_box.see("end")

    def scroll_to_first_error(self) -> None:
        try:
            self.output_tabs.select(1)
        except Exception:
            pass
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
        self.output_tabs.select(0)
        self.terminal_box.focus_set()
        self.terminal_box.mark_set("insert", "term_cursor")

    def _on_terminal_click(self, _: Any) -> None:
        def _reset_insert() -> None:
            self.terminal_box.mark_set("insert", "term_cursor")

        self.root.after(0, _reset_insert)

    def _copy_selection_to_clipboard(self) -> bool:
        try:
            selected = self.terminal_box.get("sel.first", "sel.last")
        except Exception:
            selected = ""
        if not selected:
            return False
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
        except Exception:
            return False
        return True

    def _paste_from_clipboard(self) -> str | None:
        try:
            text = self.root.clipboard_get()
        except Exception:
            return None
        if text is None:
            return None
        return str(text)

    def _dispatch_terminal_bytes(self, payload: bytes) -> bool:
        if self._terminal_input_callback is not None:
            ok, message = self._terminal_input_callback(payload)
            if message:
                self.append_log(message)
            if not ok:
                self.root.bell()
            return ok

        if self._submit_callback is not None:
            text = payload.decode("utf-8", errors="ignore")
            ok, message = self._submit_callback(text)
            if message:
                self.append_log(message)
            if not ok:
                self.root.bell()
            return ok

        self.root.bell()
        return False

    def _on_terminal_keypress(self, event: Any) -> str | None:
        state = int(getattr(event, "state", 0) or 0)
        keysym = str(getattr(event, "keysym", ""))
        char = str(getattr(event, "char", ""))
        ctrl_pressed = bool(state & 0x4)
        meta_pressed = bool(state & 0x8)
        lower = keysym.lower()

        if (ctrl_pressed or meta_pressed) and lower == "c":
            if self._copy_selection_to_clipboard():
                return "break"
            if ctrl_pressed and self._interrupt_callback is not None:
                ok, message = self._interrupt_callback()
                if message:
                    self.append_log(message)
                if not ok:
                    self.root.bell()
                return "break"
            return "break"

        if (ctrl_pressed or meta_pressed) and lower == "v":
            pasted = self._paste_from_clipboard()
            if pasted:
                self._dispatch_terminal_bytes(pasted.encode("utf-8", errors="ignore"))
            return "break"

        special_map = {
            "Return": b"\r",
            "KP_Enter": b"\r",
            "BackSpace": b"\x7f",
            "Tab": b"\t",
            "Escape": b"\x1b",
            "Up": b"\x1b[A",
            "Down": b"\x1b[B",
            "Right": b"\x1b[C",
            "Left": b"\x1b[D",
            "Home": b"\x1b[H",
            "End": b"\x1b[F",
            "Delete": b"\x1b[3~",
            "Prior": b"\x1b[5~",
            "Next": b"\x1b[6~",
            "Insert": b"\x1b[2~",
        }
        if keysym in special_map:
            self._dispatch_terminal_bytes(special_map[keysym])
            return "break"

        if ctrl_pressed and len(keysym) == 1 and keysym.isalpha():
            code = ord(keysym.upper()) & 0x1F
            self._dispatch_terminal_bytes(bytes([code]))
            return "break"

        if ctrl_pressed and lower == "space":
            self._dispatch_terminal_bytes(b"\x00")
            return "break"

        if char:
            self._dispatch_terminal_bytes(char.encode("utf-8", errors="ignore"))
            return "break"

        return "break"

    def _cursor_line_col(self) -> tuple[int, int]:
        self._ensure_terminal_cursor()
        index = self.terminal_box.index("term_cursor")
        line_text, col_text = index.split(".", 1)
        return int(line_text), int(col_text)

    def _set_cursor(self, line: int, col: int) -> None:
        end_line = int(self.terminal_box.index("end-1c").split(".", 1)[0])
        line = max(1, min(line, max(end_line, 1)))
        line_text = self.terminal_box.get(f"{line}.0", f"{line}.end")
        col = max(0, min(col, len(line_text)))
        self.terminal_box.mark_set("term_cursor", f"{line}.{col}")

    def _terminal_carriage_return(self) -> None:
        self._ensure_terminal_cursor()
        line, _ = self._cursor_line_col()
        self.terminal_box.mark_set("term_cursor", f"{line}.0")

    def _terminal_line_feed(self) -> None:
        self._ensure_terminal_cursor()
        line, col = self._cursor_line_col()
        end_line = int(self.terminal_box.index("end-1c").split(".", 1)[0])
        next_line = line + 1
        if next_line <= end_line:
            self._set_cursor(next_line, col)
            return
        line_end = self.terminal_box.index(f"{line}.end")
        self.terminal_box.insert(line_end, "\n")
        self._set_cursor(next_line, 0)

    def _terminal_backspace(self) -> None:
        self._ensure_terminal_cursor()
        if self.terminal_box.compare("term_cursor", ">", "1.0"):
            self.terminal_box.mark_set("term_cursor", "term_cursor -1c")

    def _terminal_put_char(self, char: str) -> None:
        self._ensure_terminal_cursor()
        if self.terminal_box.compare("term_cursor", "<", "end-1c"):
            next_index = self.terminal_box.index("term_cursor +1c")
            existing = self.terminal_box.get("term_cursor", next_index)
            if existing != "\n":
                self.terminal_box.delete("term_cursor", next_index)
        self.terminal_box.insert("term_cursor", char)

    def _erase_line(self, mode: int) -> None:
        self._ensure_terminal_cursor()
        line_start = self.terminal_box.index("term_cursor linestart")
        line_end = self.terminal_box.index("term_cursor lineend")
        if mode == 1:
            self.terminal_box.delete(line_start, "term_cursor")
            return
        if mode == 2:
            self.terminal_box.delete(line_start, line_end)
            self.terminal_box.mark_set("term_cursor", line_start)
            return
        self.terminal_box.delete("term_cursor", line_end)

    def _erase_display(self, mode: int) -> None:
        self._ensure_terminal_cursor()
        if mode == 1:
            self.terminal_box.delete("1.0", "term_cursor")
            return
        if mode == 2:
            self._reset_terminal_buffer()
            return
        self.terminal_box.delete("term_cursor", "end-1c")

    def _handle_csi(self, params_text: str, command: str) -> None:
        clean = params_text.strip()
        parts = clean.split(";") if clean else []

        def _as_int(value: str, default: int) -> int:
            stripped = value.strip()
            stripped = stripped.lstrip("?")
            if not stripped:
                return default
            try:
                return int(stripped)
            except ValueError:
                return default

        count = _as_int(parts[0], 1) if parts else 1

        if command == "A":
            line, col = self._cursor_line_col()
            self._set_cursor(line - count, col)
            return
        if command == "B":
            line, col = self._cursor_line_col()
            self._set_cursor(line + count, col)
            return
        if command == "C":
            line, col = self._cursor_line_col()
            self._set_cursor(line, col + count)
            return
        if command == "D":
            line, col = self._cursor_line_col()
            self._set_cursor(line, col - count)
            return
        if command in {"H", "f"}:
            row = _as_int(parts[0], 1) if parts else 1
            col = _as_int(parts[1], 1) if len(parts) > 1 else 1
            self._set_cursor(row, col - 1)
            return
        if command == "G":
            col = _as_int(parts[0], 1)
            line, _ = self._cursor_line_col()
            self._set_cursor(line, col - 1)
            return
        if command == "K":
            mode = _as_int(parts[0], 0) if parts else 0
            self._erase_line(mode)
            return
        if command == "J":
            mode = _as_int(parts[0], 0) if parts else 0
            self._erase_display(mode)
            return
        if command == "s":
            self._saved_cursor_index = self.terminal_box.index("term_cursor")
            return
        if command == "u":
            self.terminal_box.mark_set("term_cursor", self._saved_cursor_index)
            return
        # Ignore style/bracketed-paste/private sequences we do not render.

    def _consume_escape(self, payload: str, start: int) -> int:
        if start + 1 >= len(payload):
            return 0

        lead = payload[start + 1]
        if lead == "[":
            idx = start + 2
            while idx < len(payload):
                code = payload[idx]
                if "@" <= code <= "~":
                    self._handle_csi(payload[start + 2 : idx], code)
                    return idx - start + 1
                idx += 1
            return 0

        if lead == "]":
            idx = start + 2
            while idx < len(payload):
                code = payload[idx]
                if code == "\x07":
                    return idx - start + 1
                if code == "\x1b" and idx + 1 < len(payload) and payload[idx + 1] == "\\":
                    return idx - start + 2
                idx += 1
            return 0

        return 2

    def feed_terminal_output(self, data: str) -> None:
        if not data:
            return

        self._ensure_terminal_cursor()
        payload = self._pending_escape + str(data)
        self._pending_escape = ""
        idx = 0

        while idx < len(payload):
            char = payload[idx]
            if char == "\x1b":
                consumed = self._consume_escape(payload, idx)
                if consumed == 0:
                    self._pending_escape = payload[idx:]
                    break
                idx += consumed
                continue
            if char == "\r":
                self._terminal_carriage_return()
            elif char == "\n":
                self._terminal_line_feed()
            elif char == "\b":
                self._terminal_backspace()
            elif char == "\t":
                self._terminal_put_char(" ")
                self._terminal_put_char(" ")
                self._terminal_put_char(" ")
                self._terminal_put_char(" ")
            elif char == "\x07":
                pass
            elif ord(char) >= 32:
                self._terminal_put_char(char)
            idx += 1

        self.terminal_box.see("term_cursor")

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
