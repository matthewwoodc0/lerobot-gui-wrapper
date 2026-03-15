from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

_TELEOP_AV1_ERROR_MARKERS = (
    "failed to get pixel format",
    "missing sequence header",
    "hardware accelerated av1 decoding",
)

_CALIBRATION_PROMPT_MARKER = "press enter to use provided calibration file associated with the id"
_CALIBRATION_PROMPT_FOLLOWUP = "type 'c' and press enter to run calibration"
_CALIBRATION_PROMPT_ID_RE = re.compile(
    r"associated with the id\s+([^\s,:]+)",
    flags=re.IGNORECASE,
)
_CALIBRATION_DIR_FLAGS = ("--robot.calibration_dir=", "--teleop.calibration_dir=")
_APPLICATION_CURSOR_MODE_ENABLE = "\x1b[?1h"
_APPLICATION_CURSOR_MODE_DISABLE = "\x1b[?1l"

_TELEOP_READY_MARKERS = (
    "teleoperate",
    "teleop started",
    "start teleoperation",
    "teleop running",
    "connected",
    "ready",
    "torque enabled",
    "fps:",
)


def is_teleop_av1_decode_error(line: str) -> bool:
    text = str(line or "").strip().lower()
    if "av1" not in text:
        return False
    if any(marker in text for marker in _TELEOP_AV1_ERROR_MARKERS):
        return True
    return "decode" in text and "hardware" in text


def is_teleop_ready_line(line: str) -> bool:
    text = str(line or "").strip().lower()
    return any(marker in text for marker in _TELEOP_READY_MARKERS)


def is_saved_calibration_prompt(text: str) -> bool:
    lower_text = str(text or "").strip().lower()
    if not lower_text:
        return False
    return _CALIBRATION_PROMPT_MARKER in lower_text and _CALIBRATION_PROMPT_FOLLOWUP in lower_text


def extract_calibration_prompt_id(text: str) -> str | None:
    match = _CALIBRATION_PROMPT_ID_RE.search(str(text or ""))
    if not match:
        return None
    robot_id = match.group(1).strip()
    return robot_id or None


def command_has_explicit_calibration_dir(cmd: list[str]) -> bool:
    for arg in cmd:
        normalized = str(arg or "").strip()
        if any(normalized.startswith(flag) for flag in _CALIBRATION_DIR_FLAGS):
            return True
    return False


@dataclass
class ProcessSessionState:
    active: bool = False
    process: Any | None = None
    cancel_requested: bool = False
    cancel_outcome: bool = False
    thread: Any | None = None
    cursor_key_mode: str = "normal"
    terminal_mode_tail: str = ""

    def has_active_process(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return True
        return bool(self.active)

    def is_running(self) -> bool:
        return bool(self.active)

    def mark_active(self) -> None:
        self.active = True

    def mark_idle(self) -> None:
        self.active = False
        self.process = None
        self.thread = None
        self.cancel_requested = False
        self.cancel_outcome = False
        self.cursor_key_mode = "normal"
        self.terminal_mode_tail = ""

    def attach_process(self, process: Any) -> None:
        self.process = process

    def attach_thread(self, thread: Any) -> None:
        self.thread = thread

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def set_cancel_outcome(self, canceled: bool) -> None:
        self.cancel_outcome = bool(canceled)

    def observe_output_chunk(self, chunk: str) -> None:
        text = str(chunk or "")
        if not text:
            return
        sample = (self.terminal_mode_tail + text)[-256:]
        enable_index = sample.rfind(_APPLICATION_CURSOR_MODE_ENABLE)
        disable_index = sample.rfind(_APPLICATION_CURSOR_MODE_DISABLE)
        if enable_index > disable_index:
            self.cursor_key_mode = "application"
        elif disable_index > enable_index:
            self.cursor_key_mode = "normal"
        self.terminal_mode_tail = sample[-32:]

    def send_input_bytes(self, payload: bytes) -> tuple[bool, str]:
        if self.process is None or self.process.poll() is not None:
            return False, "No active record/deploy process to receive input."

        pty_error: str | None = None
        master_fd = getattr(self.process, "_rp_master_fd", None)
        if isinstance(master_fd, int):
            try:
                os.write(master_fd, payload)
                return True, "Input sent to active process."
            except Exception as exc:
                pty_error = f"Failed to send PTY input ({exc})."

        stdin_handle = getattr(self.process, "stdin", None)
        if stdin_handle is not None:
            try:
                try:
                    stdin_handle.write(payload.decode("utf-8", errors="ignore"))
                except TypeError:
                    stdin_handle.write(payload)
                stdin_handle.flush()
                return True, "Input sent to active process."
            except Exception as exc:
                if pty_error:
                    return False, f"{pty_error} Failed to send stdin ({exc})."
                return False, f"Failed to send stdin ({exc})."

        return False, "Active process stdin is unavailable."

    def send_input(self, payload: str) -> tuple[bool, str]:
        return self.send_input_bytes(str(payload).encode("utf-8", errors="ignore"))

    def send_arrow_key(self, direction: str) -> tuple[bool, str]:
        action_label = "Reset episode" if direction == "left" else "Start next episode"
        if self.cursor_key_mode == "application":
            seq = b"\x1bOD" if direction == "left" else b"\x1bOC"
        else:
            seq = b"\x1b[D" if direction == "left" else b"\x1b[C"
        ok, message = self.send_input_bytes(seq)
        if not ok:
            return False, f"{action_label}: {message}"
        return True, f"{action_label}: key sent ({self.cursor_key_mode} cursor mode)."
