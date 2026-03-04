from __future__ import annotations

import re
import shlex


_RENAME_MAP_JSON_PATTERN = re.compile(
    r"(--(?:dataset\.)?rename_map=)(\{.*?\})(?=\s|$)",
    flags=re.DOTALL,
)


def _protect_rename_map_json_for_shlex(raw: str) -> str:
    """Wrap bare rename-map JSON in single quotes before shlex parsing.

    GUI users often enter:
      --rename_map={"a":"b"}
    which loses JSON quotes via shlex.
    """

    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        payload = match.group(2).replace("'", "\\'")
        return f"{prefix}'{payload}'"

    return _RENAME_MAP_JSON_PATTERN.sub(repl, raw)


def _is_lerobot_record_command(argv: list[str]) -> bool:
    module_name = ""
    for idx, part in enumerate(argv):
        if str(part) == "-m" and idx + 1 < len(argv):
            module_name = str(argv[idx + 1]).strip()
            break
    if not module_name:
        return False
    return module_name in {
        "lerobot.scripts.lerobot_record",
        "scripts.lerobot_record",
        "lerobot_record",
    }


def _normalize_lerobot_rename_map_flag(argv: list[str]) -> list[str]:
    """Normalize custom rename-map flag for LeRobot record/deploy compatibility.

    Some LeRobot versions expect ``--dataset.rename_map=...`` for lerobot_record
    and reject top-level ``--rename_map=...``.
    """
    if not _is_lerobot_record_command(argv):
        return [str(part) for part in argv]

    updated: list[str] = []
    idx = 0
    while idx < len(argv):
        part = str(argv[idx])
        if part == "--rename_map":
            value = ""
            if idx + 1 < len(argv):
                value = str(argv[idx + 1])
                idx += 1
            updated.append("--dataset.rename_map")
            if value:
                updated.append(value)
            idx += 1
            continue
        if part.startswith("--rename_map="):
            updated.append(part.replace("--rename_map=", "--dataset.rename_map=", 1))
            idx += 1
            continue
        updated.append(part)
        idx += 1

    return updated


def _normalize_flag_key(raw_key: str) -> str | None:
    key = str(raw_key or "").strip()
    if not key:
        return None
    if key.startswith("--"):
        key = key[2:]
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None
    return key


def _replace_flag_value(argv: list[str], key: str, value: str) -> list[str]:
    updated = [str(part) for part in argv]
    prefixed = f"--{key}"

    for idx in range(len(updated) - 1, -1, -1):
        current = updated[idx]
        if current.startswith(f"{prefixed}="):
            updated[idx] = f"{prefixed}={value}"
            return updated
        if current == prefixed and idx + 1 < len(updated):
            updated[idx + 1] = value
            return updated

    updated.append(f"{prefixed}={value}")
    return updated


def parse_custom_args(custom_args_raw: str) -> tuple[list[str] | None, str | None]:
    raw = str(custom_args_raw or "").strip()
    if not raw:
        return [], None
    try:
        return shlex.split(_protect_rename_map_json_for_shlex(raw)), None
    except ValueError as exc:
        return None, f"Invalid custom args: {exc}"


def apply_command_overrides(
    base_cmd: list[str],
    overrides: dict[str, str] | None = None,
    custom_args_raw: str = "",
) -> tuple[list[str] | None, str | None]:
    cmd = [str(part) for part in base_cmd]

    if overrides:
        for raw_key, raw_value in overrides.items():
            key = _normalize_flag_key(raw_key)
            if key is None:
                return None, f"Invalid advanced option key: {raw_key!r}"
            value = str(raw_value or "").strip()
            if value == "":
                continue
            cmd = _replace_flag_value(cmd, key, value)

    custom_args, custom_error = parse_custom_args(custom_args_raw)
    if custom_error is not None or custom_args is None:
        return None, custom_error or "Invalid custom args."

    cmd.extend(custom_args)
    cmd = _normalize_lerobot_rename_map_flag(cmd)
    return cmd, None


def get_flag_value(argv: list[str], key: str) -> str | None:
    normalized = _normalize_flag_key(key)
    if normalized is None:
        return None
    prefixed = f"--{normalized}"
    items = [str(part) for part in argv]

    for idx in range(len(items) - 1, -1, -1):
        current = items[idx]
        if current.startswith(f"{prefixed}="):
            return current.split("=", 1)[1]
        if current == prefixed and idx + 1 < len(items):
            return items[idx + 1]

    return None
