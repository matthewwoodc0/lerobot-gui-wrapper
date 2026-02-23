from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


LineCallback = Callable[[str], None]
CompleteCallback = Callable[[int], None]
StartErrorCallback = Callable[[Exception], None]
CancelRequested = Callable[[], bool]
ProcessStartedCallback = Callable[[subprocess.Popen[str]], None]


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
) -> threading.Thread:
    def worker() -> None:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
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
