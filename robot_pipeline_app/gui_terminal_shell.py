from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_run_artifacts
from .config_store import get_lerobot_dir

try:
    import pty
except Exception:
    pty = None  # type: ignore[assignment]


START_MARKER = "__RP_CMD_START__"
END_MARKER = "__RP_CMD_END__"
ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret)\s*="),
    re.compile(r"(?i)(--token|--password|--passwd|--secret|--api[_-]?key)"),
    re.compile(r"(?i)^\s*export\s+[^\n]*(token|password|secret|api[_-]?key)\s*="),
    re.compile(r"(?i)(--hf[_-]?token|--huggingface[_-]?token|--hf[_-]?api[_-]?key)"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S"),
    re.compile(r"\bhf_[A-Za-z0-9]{15,}\b"),
]


def is_sensitive_command(command: str) -> bool:
    text = str(command or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


@dataclass
class _ActiveCommand:
    command: str
    started_at: datetime
    output_lines: list[str]
    persist_history: bool
    command_id: int


class GuiTerminalShell:
    def __init__(
        self,
        *,
        root: Any,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        is_pipeline_active: Callable[[], bool],
        send_pipeline_stdin: Callable[[str], tuple[bool, str]],
        on_artifact_written: Callable[[], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.append_log = append_log
        self.is_pipeline_active = is_pipeline_active
        self.send_pipeline_stdin = send_pipeline_stdin
        self.on_artifact_written = on_artifact_written

        self._lock = threading.Lock()
        self._abort_lock = threading.Lock()
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._buffer = ""
        self._available = bool(os.name == "posix" and pty is not None)
        self._start_dir = get_lerobot_dir(config)
        self._last_known_cwd = self._start_dir
        self._command_counter = 0
        self._active_command: _ActiveCommand | None = None

    def _next_command_id(self) -> int:
        self._command_counter += 1
        return self._command_counter

    def _clean_output_line(self, line: str) -> str:
        without_ansi = ANSI_PATTERN.sub("", line)
        return without_ansi.replace("\r", "").rstrip("\n")

    def _schedule_log(self, line: str) -> None:
        try:
            self.root.after(0, self.append_log, line)
        except Exception:
            pass

    def _write_raw(self, text: str) -> bool:
        payload = text.encode("utf-8", errors="ignore")
        with self._lock:
            if self._master_fd is None:
                return False
            try:
                os.write(self._master_fd, payload)
            except OSError:
                return False
        return True

    def start(self) -> tuple[bool, str]:
        if not self._available:
            return False, "Interactive shell is unavailable on this platform."

        with self._lock:
            if self._process is not None and self._process.poll() is None and self._master_fd is not None:
                return True, "Interactive shell ready."

        self._start_dir.mkdir(parents=True, exist_ok=True)
        shell_executable = os.environ.get("SHELL") or "/bin/bash"

        master_fd = -1
        slave_fd = -1
        try:
            master_fd, slave_fd = pty.openpty()
            process = subprocess.Popen(
                [shell_executable, "-i"],
                cwd=str(self._start_dir),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
            )
        except Exception as exc:
            if slave_fd >= 0:
                try:
                    os.close(slave_fd)
                except Exception:
                    pass
            if master_fd >= 0:
                try:
                    os.close(master_fd)
                except Exception:
                    pass
            return False, f"Failed to start interactive shell ({exc.__class__.__name__}): {exc}"

        try:
            os.close(slave_fd)
        except Exception:
            pass

        with self._lock:
            self._master_fd = master_fd
            self._process = process
            self._buffer = ""

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Reduce prompt noise and shell echo in the GUI view.
        self._write_raw("stty -echo 2>/dev/null\n")
        self._write_raw("export PS1='' PROMPT='' RPROMPT='' 2>/dev/null\n")

        self._schedule_log(f"Interactive shell started in {self._start_dir}")
        return True, ""

    def _reader_loop(self) -> None:
        while True:
            with self._lock:
                process = self._process
                master_fd = self._master_fd

            if process is None or master_fd is None:
                break

            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break

            if not chunk:
                break

            self._buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in self._buffer:
                raw_line, self._buffer = self._buffer.split("\n", 1)
                self._handle_output_line(raw_line)

        remaining = self._buffer.strip()
        if remaining:
            self._handle_output_line(remaining)

        with self._lock:
            process = self._process
            self._process = None
            master_fd = self._master_fd
            self._master_fd = None

        if master_fd is not None:
            try:
                os.close(master_fd)
            except Exception:
                pass

        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

        self._abort_active_command("Interactive shell exited before command completion.", exit_code=1)
        self._schedule_log("Interactive shell exited.")

    def _handle_output_line(self, raw_line: str) -> None:
        line = self._clean_output_line(raw_line)
        if not line:
            return

        active = self._active_command
        if line.startswith(START_MARKER):
            return

        if line.startswith(END_MARKER):
            # Format: __RP_CMD_END__<id>\t<exit_code>\t<pwd>
            payload = line[len(END_MARKER) :]
            parts = payload.split("\t", 2)
            if len(parts) >= 2:
                command_id_raw = parts[0].strip()
                exit_code_raw = parts[1].strip()
                cwd_raw = parts[2].strip() if len(parts) >= 3 else ""
                try:
                    command_id = int(command_id_raw)
                except ValueError:
                    command_id = -1
                try:
                    exit_code = int(exit_code_raw)
                except ValueError:
                    exit_code = -1
                if cwd_raw:
                    cwd_path = Path(cwd_raw).expanduser()
                    if cwd_path.exists() and cwd_path.is_dir():
                        self._last_known_cwd = cwd_path
                self._finalize_command(command_id, exit_code)
            return

        if active is not None:
            active.output_lines.append(line)
        self._schedule_log(line)

    def _finalize_command(self, command_id: int, exit_code: int) -> None:
        with self._abort_lock:
            active = self._active_command
            if active is None or active.command_id != command_id:
                return
            self._active_command = None

        if not active.persist_history:
            self._schedule_log("Shell command completed (history persistence skipped: sensitive command).")
            return
        self._persist_active_command(active, exit_code)

    def _persist_active_command(self, active: _ActiveCommand, exit_code: int) -> None:
        ended_at = datetime.now(timezone.utc)

        command_argv: list[str] | None = None
        try:
            command_argv = shlex.split(active.command)
        except ValueError:
            command_argv = None

        artifact_path = write_run_artifacts(
            config=self.config,
            mode="shell",
            command=active.command,
            command_argv=command_argv,
            cwd=self._last_known_cwd,
            started_at=active.started_at,
            ended_at=ended_at,
            exit_code=exit_code,
            canceled=False,
            preflight_checks=[],
            output_lines=active.output_lines,
            source="shell",
        )
        if artifact_path is not None:
            self._schedule_log(f"Run artifacts saved: {artifact_path}")
            if self.on_artifact_written is not None:
                try:
                    self.root.after(0, self.on_artifact_written)
                except Exception:
                    pass

    def _abort_active_command(self, reason: str, exit_code: int = 1) -> None:
        with self._abort_lock:
            active = self._active_command
            if active is None:
                return
            self._active_command = None
        reason_text = str(reason).strip() or "Shell command terminated."
        active.output_lines.append(reason_text)
        self._schedule_log(reason_text)

        if not active.persist_history:
            self._schedule_log("Shell command ended (history persistence skipped: sensitive command).")
            return
        self._persist_active_command(active, exit_code)

    def is_busy(self) -> bool:
        return self._active_command is not None

    def is_available(self) -> bool:
        return self._available

    def _run_shell_command(self, command: str) -> tuple[bool, str]:
        ok, msg = self.start()
        if not ok:
            return False, msg

        if self.is_busy():
            return False, "A shell command is already running."

        command_text = str(command or "").strip()
        if not command_text:
            return True, ""

        command_id = self._next_command_id()
        command_output = [f"$ {command_text}"]
        active_command = _ActiveCommand(
            command=command_text,
            started_at=datetime.now(timezone.utc),
            output_lines=command_output,
            persist_history=not is_sensitive_command(command_text),
            command_id=command_id,
        )
        self._active_command = active_command

        wrapped_script = (
            f'printf "{START_MARKER}{command_id}\\n"\n'
            f"eval -- {shlex.quote(command_text)}\n"
            "__rp_ec=$?\n"
            f'printf "{END_MARKER}{command_id}\\t%s\\t%s\\n" "$__rp_ec" "$(pwd)"\n'
        )
        if not self._write_raw(wrapped_script):
            self._active_command = None
            return False, "Failed to write command to shell."

        return True, ""

    def run_command_from_history(self, command: str) -> tuple[bool, str]:
        if self.is_pipeline_active():
            return False, "Cannot rerun shell command while record/deploy is active."
        return self._run_shell_command(command)

    def handle_terminal_submit(self, text: str) -> tuple[bool, str]:
        line = str(text or "")
        if self.is_pipeline_active():
            return self.send_pipeline_stdin(line + "\n")

        if self.is_busy():
            if self._write_raw(line + "\n"):
                return True, ""
            return False, "Failed to send input to shell command."

        return self._run_shell_command(line)

    def send_interrupt(self) -> tuple[bool, str]:
        if self.is_pipeline_active():
            return self.send_pipeline_stdin("\x03")
        if not self.is_busy():
            return False, "No active command to interrupt."
        if self._write_raw("\x03"):
            return True, "Interrupt signal sent."
        return False, "Failed to send interrupt signal."

    def shutdown(self) -> None:
        self._abort_active_command("Shell shutdown interrupted active command.", exit_code=130)

        with self._lock:
            process = self._process
            self._process = None
            master_fd = self._master_fd
            self._master_fd = None

        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

        if master_fd is not None:
            try:
                os.close(master_fd)
            except Exception:
                pass
