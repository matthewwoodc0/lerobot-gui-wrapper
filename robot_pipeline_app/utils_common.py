from __future__ import annotations

import re
from typing import Any

_NATURAL_SORT_PATTERN = re.compile(r"(\d+)")
_SKIP_DIR_NAMES = {"__pycache__", ".git"}


def parse_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def natural_sort_key(value: str) -> list[Any]:
    parts = _NATURAL_SORT_PATTERN.split(str(value))
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def is_skippable_dir_name(name: str) -> bool:
    return not name or name.startswith(".") or name in _SKIP_DIR_NAMES
