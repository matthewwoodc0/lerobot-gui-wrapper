from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config_store import get_lerobot_dir
from .runner import _CANCEL_TIMEOUT_SECONDS, kill_process_tree, popen_session_kwargs, terminate_process_tree

try:
    import fcntl
    import pty
    import struct
    import termios
except Exception:
    fcntl = None  # type: ignore[assignment]
    pty = None  # type: ignore[assignment]
    struct = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]


ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_HIDDEN_INPUT_MARKERS: tuple[str, ...] = (
    "input will not be visible",
    "password (input will not be visible)",
)


def infer_terminal_status_from_output(chunk: str) -> str | None:
    text = str(chunk or "")
    lowered = text.lower()
    if any(marker in lowered for marker in _HIDDEN_INPUT_MARKERS):
        return "Command is waiting for hidden input. Paste the token, then press Enter."
    if "command not found: huggingface-cli" in lowered:
        return "huggingface-cli was not found in this shell. Use `hf auth login`, or open a new terminal tab."
    if "use 'hf auth login' instead" in lowered:
        return "Hugging Face auth is interactive here. Use `hf auth login` and follow the token prompts."
    return None


class GuiTerminalShell:
    """PTY-backed interactive shell with optional raw-terminal output streaming."""

    def __init__(
        self,
        *,
        root: Any,
        config: dict[str, Any],
        append_log: Callable[[str], None],
        is_pipeline_active: Callable[[], bool],
        send_pipeline_stdin: Callable[[str], tuple[bool, str]],
        append_terminal_output: Callable[[str], None] | None = None,
        on_artifact_written: Callable[[], None] | None = None,
        set_status_message: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.append_log = append_log
        self.is_pipeline_active = is_pipeline_active
        self.send_pipeline_stdin = send_pipeline_stdin
        self.append_terminal_output = append_terminal_output
        # on_artifact_written kept for API compatibility; not used in raw-terminal mode
        self.on_artifact_written = on_artifact_written
        self.set_status_message = set_status_message

        self._lock = threading.Lock()
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._line_buffer = ""
        self._available = bool(os.name == "posix" and pty is not None)
        self._start_dir = get_lerobot_dir(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_output_line(self, line: str) -> str:
        """Strip ANSI escape codes and carriage returns from a raw PTY line."""
        return ANSI_PATTERN.sub("", line).replace("\r", "").rstrip("\n")

    def _schedule_log(self, line: str) -> None:
        try:
            self.root.after(0, self.append_log, line)
        except Exception:
            pass

    def _schedule_terminal_output(self, chunk: str) -> None:
        callback = self.append_terminal_output
        if callback is None:
            return
        try:
            self.root.after(0, callback, chunk)
        except Exception:
            pass

    def _schedule_status_message(self, message: str) -> None:
        callback = self.set_status_message
        if callback is None:
            return
        try:
            self.root.after(0, callback, str(message))
        except Exception:
            pass

    def _write_payload(self, payload: bytes) -> bool:
        """Write *payload* directly to the PTY master file descriptor."""
        with self._lock:
            if self._master_fd is None:
                return False
            try:
                os.write(self._master_fd, payload)
            except OSError:
                return False
        return True

    def _write_raw(self, text: str) -> bool:
        return self._write_payload(text.encode("utf-8", errors="ignore"))

    def _apply_terminal_size(self, columns: int, rows: int) -> bool:
        if fcntl is None or struct is None or termios is None:
            return False
        cols = max(20, int(columns))
        line_count = max(4, int(rows))
        with self._lock:
            master_fd = self._master_fd
        if master_fd is None:
            return False
        try:
            winsize = struct.pack("HHHH", line_count, cols, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            return False
        return True

    def _shell_environment(self) -> dict[str, str]:
        """Build the environment for the embedded PTY shell."""
        env = os.environ.copy()
        if not str(env.get("TERM") or "").strip():
            # Finder / desktop launches often omit TERM entirely. zsh then redraws
            # deletions as literal spaces, which leaves stale characters on screen
            # in the embedded terminal.
            env["TERM"] = "dumb"
        venv_bin = self._get_venv_dir() / "bin"
        if venv_bin.is_dir():
            current_path = str(env.get("PATH") or "")
            path_parts = current_path.split(os.pathsep) if current_path else []
            venv_bin_text = str(venv_bin)
            if venv_bin_text not in path_parts:
                env["PATH"] = (
                    venv_bin_text + (os.pathsep + current_path if current_path else "")
                )
            env.pop("PYTHONHOME", None)
        return env

    def _get_venv_dir(self) -> Path:
        """Return the venv directory from config, falling back to the default."""
        venv_str = str(self.config.get("lerobot_venv_dir") or "").strip()
        if venv_str:
            return Path(venv_str).expanduser()
        return get_lerobot_dir(self.config) / "lerobot_env"

    def _activation_command(self) -> tuple[str | None, str]:
        """Return shell activation command and short source label."""
        custom = str(self.config.get("setup_venv_activate_cmd") or "").strip()
        if custom:
            return custom, "config:setup_venv_activate_cmd"

        venv_dir = self._get_venv_dir()
        activate_script = venv_dir / "bin" / "activate"
        if activate_script.exists():
            return f'source "{activate_script}"', "config:lerobot_venv_dir"

        if (venv_dir / "conda-meta").is_dir():
            # Common conda layout: <base>/envs/<name>.
            if venv_dir.parent.name == "envs":
                base_activate = venv_dir.parent.parent / "bin" / "activate"
                if base_activate.exists():
                    quoted_base = shlex.quote(str(base_activate))
                    quoted_prefix = shlex.quote(str(venv_dir))
                    return (
                        f"source {quoted_base} {quoted_prefix}",
                        "config:lerobot_venv_dir(conda-prefix)",
                    )

            conda_env_name = str(self.config.get("setup_conda_env_name") or "").strip()
            if conda_env_name:
                return (
                    f"conda activate {conda_env_name}",
                    "config:setup_conda_env_name",
                )
            return (
                f"conda activate {shlex.quote(str(venv_dir))}",
                "config:lerobot_venv_dir(conda-path)",
            )

        return None, f"missing activate script at {activate_script}"

    def _startup_activation_command(self) -> tuple[str | None, str]:
        custom = str(self.config.get("setup_venv_activate_cmd") or "").strip()
        if custom:
            return custom, "config:setup_venv_activate_cmd"

        venv_dir = self._get_venv_dir()
        if (venv_dir / "conda-meta").is_dir():
            return self._activation_command()

        venv_bin = venv_dir / "bin"
        if venv_bin.is_dir():
            return None, f"PATH already includes {venv_bin}"
        return self._activation_command()

    # ------------------------------------------------------------------
    # Shell lifecycle
    # ------------------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        """Start the interactive shell (no-op if already running)."""
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
                env=self._shell_environment(),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
                **popen_session_kwargs(),
            )
        except Exception as exc:
            for fd in (slave_fd, master_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
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
            self._line_buffer = ""

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        activation_cmd, activation_source = self._startup_activation_command()
        if activation_cmd:
            self._write_raw(activation_cmd + "\n")
            self._schedule_log(f"Terminal: sent activation command ({activation_source}).")
            self._schedule_status_message("Starting shell and activating the configured environment...")
        else:
            self._schedule_status_message("Interactive shell ready.")
            self._schedule_log(f"Terminal: {activation_source}.")

        return True, ""

    def activate_environment(self) -> tuple[bool, str]:
        """Ensure the shell is running, then send the configured activation command."""
        ok, message = self.start()
        if not ok:
            return False, message

        activation_cmd, activation_source = self._activation_command()
        if not activation_cmd:
            note = (
                "Terminal note: "
                + activation_source
                + ". Set 'LeRobot venv folder path' or 'setup_venv_activate_cmd' in Config."
            )
            self._schedule_log(note)
            return False, note

        if not self._write_raw(activation_cmd + "\n"):
            return False, "Failed to send activation command to shell."
        self._schedule_log(f"Terminal: sent activation command ({activation_source}).")
        self._schedule_status_message("Environment activation command sent.")
        return True, ""

    def resize_terminal(self, columns: int, rows: int) -> None:
        """Resize the PTY window so interactive shells redraw against the visible pane size."""
        self._apply_terminal_size(columns, rows)

    def _reader_loop(self) -> None:
        """Background thread: read PTY output and forward it to UI callbacks."""
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

            text_chunk = chunk.decode("utf-8", errors="ignore")
            status_message = infer_terminal_status_from_output(text_chunk)
            if status_message:
                self._schedule_status_message(status_message)
            if self.append_terminal_output is not None:
                self._schedule_terminal_output(text_chunk)
                continue

            # Fallback path: split into clean log lines when no terminal view is attached.
            self._line_buffer += text_chunk
            while "\n" in self._line_buffer:
                raw_line, self._line_buffer = self._line_buffer.split("\n", 1)
                line = self._clean_output_line(raw_line)
                if line:
                    self._schedule_log(line)

        # Flush any partial line remaining in fallback line-buffer mode.
        remaining = self._clean_output_line(self._line_buffer)
        if remaining and self.append_terminal_output is None:
            self._schedule_log(remaining)

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
            self._terminate_shell_process(process, reason="Shell stream closed.")

        self._schedule_status_message("Interactive shell exited.")
        self._schedule_log("Interactive shell exited.")

    def _terminate_shell_process(self, process: subprocess.Popen[bytes], *, reason: str) -> None:
        terminate_process_tree(process, self._schedule_log, reason=reason)
        deadline = time.monotonic() + _CANCEL_TIMEOUT_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            kill_process_tree(process, self._schedule_log, reason=f"{reason} Timeout reached.")
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Terminate the shell process on application exit."""
        with self._lock:
            process = self._process
            self._process = None
            master_fd = self._master_fd
            self._master_fd = None

        if process is not None and process.poll() is None:
            self._terminate_shell_process(process, reason="Shell shutdown requested.")

        if master_fd is not None:
            try:
                os.close(master_fd)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface (consumed by the GUI shell layers)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def is_busy(self) -> bool:
        """Raw-terminal mode never blocks pipeline runs with a busy flag."""
        return False

    def handle_terminal_input(self, payload: bytes) -> tuple[bool, str]:
        """Forward raw terminal input to active pipeline or shell PTY."""
        if self.is_pipeline_active():
            text = payload.decode("utf-8", errors="ignore")
            return self.send_pipeline_stdin(text)

        ok, msg = self.start()
        if not ok:
            return False, msg

        if not self._write_payload(payload):
            return False, "Failed to send input to shell."
        return True, ""

    def handle_terminal_submit(self, text: str) -> tuple[bool, str]:
        """Compatibility helper for line-based submits from legacy callers."""
        line = str(text or "")
        return self.handle_terminal_input((line + "\n").encode("utf-8", errors="ignore"))

    def run_command_from_history(self, command: str) -> tuple[bool, str]:
        """Replay a command from the run-history panel in the interactive shell."""
        if self.is_pipeline_active():
            return False, "Cannot rerun shell command while record/deploy is active."

        command_text = str(command or "").strip()
        return self.handle_terminal_submit(command_text)

    def send_interrupt(self) -> tuple[bool, str]:
        """Send Ctrl-C to whichever process currently owns the terminal."""
        return self.handle_terminal_input(b"\x03")
