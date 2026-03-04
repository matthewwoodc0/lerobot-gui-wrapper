from __future__ import annotations

import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from .compat import probe_lerobot_capabilities
from .config_store import normalize_config_without_prompts
from .feature_flags import compat_probe_enabled


def _module_version(module_name: str) -> str:
    try:
        return str(importlib.metadata.version(module_name))
    except Exception:
        return "unknown"


def build_compat_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_config_without_prompts(config)
    if not compat_probe_enabled(normalized):
        return {
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "compat_policy": str(normalized.get("compat_policy", "latest_plus_n_minus_1")),
            "compat_probe_enabled": False,
            "lerobot_version": _module_version("lerobot"),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }

    capabilities = probe_lerobot_capabilities(normalized, include_flag_probe=True)
    return {
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "compat_policy": str(normalized.get("compat_policy", "latest_plus_n_minus_1")),
        "compat_probe_enabled": True,
        "record_entrypoint": capabilities.record_entrypoint,
        "teleop_entrypoint": capabilities.teleop_entrypoint,
        "calibrate_entrypoint": capabilities.calibrate_entrypoint,
        "camera_rename_flag": capabilities.active_rename_flag,
        "supported_record_flags": list(capabilities.supported_record_flags),
        "policy_path_flag": capabilities.policy_path_flag,
        "fallback_notes": list(capabilities.fallback_notes),
        "lerobot_version": capabilities.lerobot_version if capabilities.lerobot_version else _module_version("lerobot"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
