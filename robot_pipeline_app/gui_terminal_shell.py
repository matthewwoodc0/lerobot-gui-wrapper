from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config_store import get_lerobot_dir
from .runner import _CANCEL_TIMEOUT_SECONDS, kill_process_tree, popen_session_kwargs, terminate_process_tree

try:
    import pty
except Exception:
    pty = None  # type: ignore[assignment]


ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Sentinel printed to PTY stdout once shell setup is complete.
# All output before this marker is startup noise and is silently discarded.
_SHELL_READY_MARKER = "__LEROBOT_SHELL_READY__"


class GuiTerminalShell:
    """PTY-backed interactive shell that auto-activates the configured venv.

    Commands are sent raw to the shell process; output is streamed back line
    by line with ANSI escape codes stripped (the Tkinter Text widget does not
    render them natively).
    """

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
        # on_artifact_written kept for API compatibility; not used in raw-terminal mode
        self.on_artifact_written = on_artifact_written

        self._lock = threading.Lock()
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._buffer = ""
        self._shell_ready = False  # True once startup noise has been suppressed
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

    def _write_raw(self, text: str) -> bool:
        """Write *text* directly to the PTY master file descriptor."""
        payload = text.encode("utf-8", errors="ignore")
        with self._lock:
            if self._master_fd is None:
                return False
            try:
                os.write(self._master_fd, payload)
            except OSError:
                return False
        return True

    def _get_venv_dir(self) -> Path:
        """Return the venv directory from config, falling back to the default."""
        venv_str = str(self.config.get("lerobot_venv_dir") or "").strip()
        if venv_str:
            return Path(venv_str).expanduser()
        return get_lerobot_dir(self.config) / "lerobot_env"

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
            self._buffer = ""
            self._shell_ready = False

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # --- Shell setup (processed in order before any user command) ---
        # 1. Suppress PTY echo so typed commands don't appear as output lines.
        # 2. Clear every prompt variable so the shell prompt never appears.
        # 3. Print the ready marker so the reader knows to start forwarding output.
        setup_cmds = (
            "stty -echo 2>/dev/null\n"
            "export PS1='' PS2='' PS3='' PS4='' PROMPT='' RPROMPT='' 2>/dev/null\n"
            f"printf '{_SHELL_READY_MARKER}\\n'\n"
        )
        self._write_raw(setup_cmds)

        # 4. Activate the configured venv (runs after setup, before any user input).
        venv_dir = self._get_venv_dir()
        activate_script = venv_dir / "bin" / "activate"
        if activate_script.exists():
            self._write_raw(f'source "{activate_script}" 2>/dev/null\n')
            self._schedule_log(f"Venv activated: {venv_dir.name}")
        else:
            self._schedule_log(
                f"Note: venv not found at {activate_script}. "
                "Check 'LeRobot venv folder path' in Settings."
            )

        self._schedule_log(f"Shell ready  ({self._start_dir})")
        return True, ""

    def _reader_loop(self) -> None:
        """Background thread: read PTY output and forward lines to the log panel."""
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

        # Flush any partial line remaining in the buffer.
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
            self._terminate_shell_process(process, reason="Shell stream closed.")

        self._schedule_log("Interactive shell exited.")

    def _handle_output_line(self, raw_line: str) -> None:
        # Discard everything until we see the ready marker.  This silently
        # swallows the initial shell prompt, zshrc output, stty/export echoes,
        # and any other startup noise before our setup commands finish.
        if not self._shell_ready:
            if _SHELL_READY_MARKER in raw_line:
                self._shell_ready = True
            return  # drop this line regardless

        line = self._clean_output_line(raw_line)
        if not line:
            return
        self._schedule_log(line)

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
    # Public interface (consumed by gui_app / gui_log)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def is_busy(self) -> bool:
        """Raw-terminal mode: command tracking is not available.

        Returns ``False`` so that pipeline operations (record/deploy/teleop)
        are never blocked by the shell.  The shell is always ready to accept
        new input.
        """
        return False

    def handle_terminal_submit(self, text: str) -> tuple[bool, str]:
        """Called when the user presses Enter in the terminal input box.

        While a pipeline (record/deploy) is active the text is forwarded to
        that process's stdin instead of the interactive shell.
        """
        line = str(text or "")
        if self.is_pipeline_active():
            return self.send_pipeline_stdin(line + "\n")

        ok, msg = self.start()
        if not ok:
            return False, msg

        if not self._write_raw(line + "\n"):
            return False, "Failed to send input to shell."
        return True, ""

    def run_command_from_history(self, command: str) -> tuple[bool, str]:
        """Replay a command from the run-history panel in the interactive shell."""
        if self.is_pipeline_active():
            return False, "Cannot rerun shell command while record/deploy is active."

        ok, msg = self.start()
        if not ok:
            return False, msg

        if not self._write_raw(str(command or "").strip() + "\n"):
            return False, "Failed to send command to shell."
        return True, ""

    def send_interrupt(self) -> tuple[bool, str]:
        """Send Ctrl-C to whichever process currently owns the terminal."""
        if self.is_pipeline_active():
            return self.send_pipeline_stdin("\x03")
        if self._write_raw("\x03"):
            return True, "Interrupt signal sent."
        return False, "No active shell to interrupt."
