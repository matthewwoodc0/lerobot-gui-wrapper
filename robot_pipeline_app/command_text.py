from __future__ import annotations

import shlex


def format_command_for_dialog(cmd: list[str]) -> str:
    if not cmd:
        return "(empty command)"
    shell_safe = shlex.join(cmd)
    lines = [
        "Shell-safe command (copy/paste):",
        shell_safe,
        "",
        "Exact argv passed to subprocess (no shell quoting here):",
    ]
    for idx, arg in enumerate(cmd):
        lines.append(f"[{idx}] {arg}")
    return "\n".join(lines)


def format_command_for_editing(cmd: list[str]) -> str:
    if not cmd:
        return ""

    lines: list[str] = []
    start_idx = 0
    if len(cmd) >= 3 and cmd[1] == "-m":
        lines.append(shlex.join(cmd[:3]))
        start_idx = 3
    else:
        lines.append(shlex.join([cmd[0]]))
        start_idx = 1

    for arg in cmd[start_idx:]:
        lines.append(str(arg))
    return "\n".join(lines)


def parse_command_text(command_text: str) -> tuple[list[str] | None, str | None]:
    raw = str(command_text or "").strip()
    if not raw:
        return None, "Command is empty."
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        try:
            parts = shlex.split(lines[0])
        except ValueError as exc:
            return None, f"Unable to parse command: {exc}"
        if not parts:
            return None, "Command is empty."
        for line in lines[1:]:
            if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
                try:
                    quoted_parts = shlex.split(line)
                except ValueError as exc:
                    return None, f"Unable to parse command: {exc}"
                if len(quoted_parts) == 1:
                    parts.append(str(quoted_parts[0]))
                    continue
            parts.append(line)
        return [str(part) for part in parts], None
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        return None, f"Unable to parse command: {exc}"
    if not parts:
        return None, "Command is empty."
    return [str(part) for part in parts], None
