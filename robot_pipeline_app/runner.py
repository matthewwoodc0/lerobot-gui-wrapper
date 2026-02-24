from __future__ import annotations

import os
import select
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

try:
    import pty
except Exception:
    pty = None  # type: ignore[assignment]


LineCallback = Callable[[str], None]
CompleteCallback = Callable[[int], None]
StartErrorCallback = Callable[[Exception], None]
CancelRequested = Callable[[], bool]
ProcessStartedCallback = Callable[[subprocess.Popen[object]], None]


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print("\nFull command:")
    print(format_command(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        if cmd[0] == "huggingface-cli":
            print("Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate")
        return None

    return result


def run_process_streaming(
    cmd: list[str],
    cwd: Path | None,
    on_line: LineCallback,
    on_complete: CompleteCallback,
    on_start_error: StartErrorCallback,
    cancel_requested: CancelRequested,
    on_process_started: ProcessStartedCallback | None = None,
    use_pty: bool = False,
) -> threading.Thread:
    def worker() -> None:
        spawn_env = os.environ.copy()
        if use_pty:
            # PTY mode is required for arrow-key controls; reduce rich/progress spam to lower overhead.
            spawn_env.setdefault("NO_COLOR", "1")
            spawn_env.setdefault("RICH_DISABLE", "1")
            spawn_env.setdefault("TQDM_DISABLE", "1")

        if use_pty and os.name == "posix" and pty is not None:
            master_fd = -1
            slave_fd = -1
            try:
                master_fd, slave_fd = pty.openpty()
                process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    text=False,
                    close_fds=True,
                    env=spawn_env,
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
                on_start_error(exc)
                return

            try:
                os.close(slave_fd)
            except Exception:
                pass

            setattr(process, "_rp_master_fd", master_fd)
            if on_process_started is not None:
                on_process_started(process)

            buffer = ""
            cancel_deadline: float | None = None
            while True:
                try:
                    ready, _, _ = select.select([master_fd], [], [], 0.2)
                except Exception:
                    ready = []

                if ready:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        chunk = b""
                    if chunk:
                        buffer += chunk.decode("utf-8", errors="ignore").replace("\r", "\n")
                        while "\n" in buffer:
                            raw_line, buffer = buffer.split("\n", 1)
                            on_line(raw_line.rstrip("\r"))

                if cancel_requested():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + 2.0
                        on_line("Waiting up to 2 seconds for graceful shutdown...")
                        process.terminate()
                    elif time.monotonic() >= cancel_deadline:
                        on_line("Process did not exit after terminate; killing...")
                        process.kill()

                if process.poll() is not None:
                    # Drain final PTY bytes after process exit.
                    try:
                        while True:
                            ready_now, _, _ = select.select([master_fd], [], [], 0.05)
                            if not ready_now:
                                break
                            chunk = os.read(master_fd, 4096)
                            if not chunk:
                                break
                            buffer += chunk.decode("utf-8", errors="ignore").replace("\r", "\n")
                            while "\n" in buffer:
                                raw_line, buffer = buffer.split("\n", 1)
                                on_line(raw_line.rstrip("\r"))
                    except Exception:
                        pass
                    break

            if buffer.strip():
                on_line(buffer.rstrip("\r"))

            try:
                os.close(master_fd)
            except Exception:
                pass

            return_code = process.wait()
            on_complete(return_code)
            return

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=spawn_env,
            )
        except Exception as exc:
            on_start_error(exc)
            return

        if on_process_started is not None:
            on_process_started(process)

        if process.stdout is not None:
            for raw_line in process.stdout:
                on_line(raw_line.rstrip("\n"))

        cancel_deadline: float | None = None
        while True:
            try:
                return_code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_requested():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + 2.0
                        on_line("Waiting up to 2 seconds for graceful shutdown...")
                        process.terminate()
                    elif time.monotonic() >= cancel_deadline:
                        on_line("Process did not exit after terminate; killing...")
                        process.kill()
                        return_code = process.wait()
                        break

        on_complete(return_code)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def is_huggingface_cli_command_missing(cmd: list[str], error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) and bool(cmd) and cmd[0] == "huggingface-cli"
