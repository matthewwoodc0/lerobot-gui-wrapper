from __future__ import annotations

from typing import Any

from .utils_common import parse_bool_value


def diagnostics_v2_enabled(config: dict[str, Any]) -> bool:
    return parse_bool_value(config.get("diagnostics_v2_enabled", True), True)


def compat_probe_enabled(config: dict[str, Any]) -> bool:
    return parse_bool_value(config.get("compat_probe_enabled", True), True)


def support_bundle_enabled(config: dict[str, Any]) -> bool:
    return parse_bool_value(config.get("support_bundle_enabled", True), True)
