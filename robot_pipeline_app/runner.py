from __future__ import annotations

import os
import select
import shlex
import signal
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
ChunkCallback = Callable[[str], None]
CompleteCallback = Callable[[int], None]
StartErrorCallback = Callable[[Exception], None]
CancelRequested = Callable[[], bool]
ProcessStartedCallback = Callable[[subprocess.Popen[object]], None]
ProcessLike = subprocess.Popen[object]

_CANCEL_TIMEOUT_SECONDS = 2.0


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _consume_output_chunk(
    *,
    buffer: str,
    chunk: bytes,
    suppress_carriage_updates: bool,
) -> tuple[str, list[str], int]:
    text = chunk.decode("utf-8", errors="ignore").replace("\r\n", "\n")
    if not suppress_carriage_updates:
        merged = (buffer + text).replace("\r", "\n")
        emitted: list[str] = []
        while "\n" in merged:
            raw_line, merged = merged.split("\n", 1)
            emitted.append(raw_line.rstrip("\r"))
        return merged, emitted, 0

    emitted: list[str] = []
    dropped_carriage_updates = 0
    current = buffer
    for ch in text:
        if ch == "\r":
            if current.strip():
                dropped_carriage_updates += 1
            current = ""
            continue
        if ch == "\n":
            emitted.append(current.rstrip("\r"))
            current = ""
            continue
        current += ch
    return current, emitted, dropped_carriage_updates


def popen_session_kwargs() -> dict[str, object]:
    if os.name == "posix":
        # Isolate each command in its own session/process group so cancel can
        # target the full process tree.
        return {"start_new_session": True}
    return {}


def _process_group_id(process: ProcessLike) -> int | None:
    pid = int(getattr(process, "pid", 0) or 0)
    if pid <= 0:
        return None
    try:
        return int(os.getpgid(pid))
    except Exception:
        return pid


def terminate_process_tree(process: ProcessLike, on_line: LineCallback, *, reason: str) -> None:
    if os.name == "posix":
        pgid = _process_group_id(process)
        if pgid is not None:
            on_line(f"{reason} Sending SIGTERM to process group {pgid} (graceful shutdown; timeout {_CANCEL_TIMEOUT_SECONDS:.0f}s).")
            try:
                os.killpg(pgid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception as exc:
                on_line(f"{reason} Failed to signal process group {pgid} ({exc}); falling back to parent terminate.")

    pid = int(getattr(process, "pid", 0) or 0)
    on_line(f"{reason} Sending terminate to process {pid}.")
    try:
        process.terminate()
    except ProcessLookupError:
        return
    except Exception as exc:
        on_line(f"{reason} Terminate failed: {exc}")


def kill_process_tree(process: ProcessLike, on_line: LineCallback, *, reason: str) -> None:
    if os.name == "posix":
        pgid = _process_group_id(process)
        if pgid is not None:
            on_line(f"{reason} Sending SIGKILL to process group {pgid}.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
            except Exception as exc:
                on_line(f"{reason} Failed to kill process group {pgid} ({exc}); falling back to parent kill.")

    pid = int(getattr(process, "pid", 0) or 0)
    on_line(f"{reason} Sending kill to process {pid}.")
    try:
        process.kill()
    except ProcessLookupError:
        return
    except Exception as exc:
        on_line(f"{reason} Kill failed: {exc}")


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
            print("Make sure your LeRobot environment is active (venv or conda) before running this command.")
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
    suppress_carriage_updates: bool = True,
    on_chunk: ChunkCallback | None = None,
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
                    **popen_session_kwargs(),
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
            dropped_carriage_updates = 0
            try:
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
                            if on_chunk is not None:
                                on_chunk(chunk.decode("utf-8", errors="ignore"))
                            buffer, parsed_lines, dropped = _consume_output_chunk(
                                buffer=buffer,
                                chunk=chunk,
                                suppress_carriage_updates=suppress_carriage_updates,
                            )
                            dropped_carriage_updates += dropped
                            for parsed_line in parsed_lines:
                                on_line(parsed_line)

                    if cancel_requested():
                        if cancel_deadline is None:
                            cancel_deadline = time.monotonic() + _CANCEL_TIMEOUT_SECONDS
                            terminate_process_tree(process, on_line, reason="Cancel requested.")
                        elif time.monotonic() >= cancel_deadline:
                            if process.poll() is None:  # still running
                                kill_process_tree(process, on_line, reason="Cancel timeout reached.")
                                cancel_deadline = float("inf")
                            else:
                                break  # already exited; stop the loop

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
                                if on_chunk is not None:
                                    on_chunk(chunk.decode("utf-8", errors="ignore"))
                                buffer, parsed_lines, dropped = _consume_output_chunk(
                                    buffer=buffer,
                                    chunk=chunk,
                                    suppress_carriage_updates=suppress_carriage_updates,
                                )
                                dropped_carriage_updates += dropped
                                for parsed_line in parsed_lines:
                                    on_line(parsed_line)
                        except Exception:
                            pass
                        break

                if buffer.strip():
                    on_line(buffer.rstrip("\r"))
                if dropped_carriage_updates > 0:
                    on_line(
                        "Runtime I/O optimization: suppressed "
                        f"{dropped_carriage_updates} carriage-return progress updates."
                    )

                return_code = process.wait()
            finally:
                if master_fd >= 0:
                    try:
                        os.close(master_fd)
                    except OSError:
                        pass

            on_complete(return_code)
            return

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                env=spawn_env,
                **popen_session_kwargs(),
            )
        except Exception as exc:
            on_start_error(exc)
            return

        if on_process_started is not None:
            on_process_started(process)

        # For non-PTY mode, use non-blocking reads so cancel remains responsive
        # even when the child process emits no newline-terminated output.
        if os.name == "posix" and process.stdout is not None:
            stdout_fd = process.stdout.fileno()
            buffer = ""
            cancel_deadline: float | None = None
            dropped_carriage_updates = 0

            while True:
                try:
                    ready, _, _ = select.select([stdout_fd], [], [], 0.2)
                except Exception:
                    ready = []

                if ready:
                    try:
                        chunk = os.read(stdout_fd, 4096)
                    except OSError:
                        chunk = b""
                    if chunk:
                        if on_chunk is not None:
                            on_chunk(chunk.decode("utf-8", errors="ignore"))
                        buffer, parsed_lines, dropped = _consume_output_chunk(
                            buffer=buffer,
                            chunk=chunk,
                            suppress_carriage_updates=suppress_carriage_updates,
                        )
                        dropped_carriage_updates += dropped
                        for parsed_line in parsed_lines:
                            on_line(parsed_line)

                if cancel_requested():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + _CANCEL_TIMEOUT_SECONDS
                        terminate_process_tree(process, on_line, reason="Cancel requested.")
                    elif time.monotonic() >= cancel_deadline:
                        if process.poll() is None:
                            kill_process_tree(process, on_line, reason="Cancel timeout reached.")
                            cancel_deadline = float("inf")

                if process.poll() is not None:
                    try:
                        while True:
                            ready_now, _, _ = select.select([stdout_fd], [], [], 0.05)
                            if not ready_now:
                                break
                            chunk = os.read(stdout_fd, 4096)
                            if not chunk:
                                break
                            if on_chunk is not None:
                                on_chunk(chunk.decode("utf-8", errors="ignore"))
                            buffer, parsed_lines, dropped = _consume_output_chunk(
                                buffer=buffer,
                                chunk=chunk,
                                suppress_carriage_updates=suppress_carriage_updates,
                            )
                            dropped_carriage_updates += dropped
                            for parsed_line in parsed_lines:
                                on_line(parsed_line)
                    except Exception:
                        pass
                    break

            if buffer.strip():
                on_line(buffer.rstrip("\r"))
            if dropped_carriage_updates > 0:
                on_line(
                    "Runtime I/O optimization: suppressed "
                    f"{dropped_carriage_updates} carriage-return progress updates."
                )

            return_code = process.wait()
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception:
                pass
            on_complete(return_code)
            return

        # Fallback path for non-posix systems.
        if process.stdout is not None:
            for raw_line_bytes in process.stdout:
                line = raw_line_bytes.decode("utf-8", errors="ignore")
                on_line(line.rstrip("\n"))

        cancel_deadline: float | None = None
        while True:
            try:
                return_code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_requested():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + _CANCEL_TIMEOUT_SECONDS
                        terminate_process_tree(process, on_line, reason="Cancel requested.")
                    elif time.monotonic() >= cancel_deadline:
                        kill_process_tree(process, on_line, reason="Cancel timeout reached.")
                        cancel_deadline = float("inf")
                        return_code = process.wait()
                        break

        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass
        on_complete(return_code)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def is_huggingface_cli_command_missing(cmd: list[str], error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) and bool(cmd) and cmd[0] == "huggingface-cli"
