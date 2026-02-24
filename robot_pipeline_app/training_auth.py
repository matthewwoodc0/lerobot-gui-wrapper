from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .types import TrainingProfile

_SECRET_SERVICE_NAME = "lerobot-gui"


def _secret_attributes(profile: TrainingProfile) -> list[str]:
    return [
        "service",
        _SECRET_SERVICE_NAME,
        "protocol",
        "ssh",
        "host",
        profile.host,
        "user",
        profile.username,
        "port",
        str(profile.port),
    ]


def secret_tool_path() -> str | None:
    return shutil.which("secret-tool")


def is_secret_tool_available() -> bool:
    return secret_tool_path() is not None


def save_ssh_password(profile: TrainingProfile, password: str) -> tuple[bool, str]:
    tool = secret_tool_path()
    if not tool:
        return False, "secret-tool not found. Install libsecret tools (e.g. sudo apt install libsecret-tools)."
    value = str(password or "")
    if not value:
        return False, "Password is empty."
    if not profile.host or not profile.username:
        return False, "Profile host and username are required."

    cmd = [
        tool,
        "store",
        "--label",
        f"LeRobot SSH ({profile.name})",
        *_secret_attributes(profile),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            input=value + "\n",
            timeout=12,
        )
    except Exception as exc:
        return False, f"Unable to store password: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        return False, f"Failed to store password: {detail}"
    return True, "Password stored securely."


def load_ssh_password(profile: TrainingProfile) -> tuple[str | None, str | None]:
    tool = secret_tool_path()
    if not tool:
        return None, "secret-tool not found."
    if not profile.host or not profile.username:
        return None, "Profile host and username are required."

    cmd = [tool, "lookup", *_secret_attributes(profile)]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except Exception as exc:
        return None, f"Unable to load password: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if "No such secret item" in detail:
            return None, None
        return None, detail or "Credential not found."

    password = result.stdout.rstrip("\n")
    if not password:
        return None, None
    return password, None


def has_ssh_password(profile: TrainingProfile) -> tuple[bool, str | None]:
    password, error = load_ssh_password(profile)
    if error is not None:
        return False, error
    return bool(password), None


def delete_ssh_password(profile: TrainingProfile) -> tuple[bool, str]:
    tool = secret_tool_path()
    if not tool:
        return False, "secret-tool not found."
    if not profile.host or not profile.username:
        return False, "Profile host and username are required."

    cmd = [tool, "clear", *_secret_attributes(profile)]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except Exception as exc:
        return False, f"Unable to clear password: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        return False, f"Failed to clear password: {detail}"
    return True, "Password cleared."


def redact_sensitive_text(raw: Any) -> str:
    text = str(raw or "")
    lowered = text.lower()
    for token in ("password", "passwd", "secret", "token"):
        if token in lowered:
            return "[redacted sensitive text]"
    return text
