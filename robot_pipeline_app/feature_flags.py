from __future__ import annotations

from typing import Any


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def diagnostics_v2_enabled(config: dict[str, Any]) -> bool:
    return _as_bool(config.get("diagnostics_v2_enabled", True), True)


def compat_probe_enabled(config: dict[str, Any]) -> bool:
    return _as_bool(config.get("compat_probe_enabled", True), True)


def support_bundle_enabled(config: dict[str, Any]) -> bool:
    return _as_bool(config.get("support_bundle_enabled", True), True)

