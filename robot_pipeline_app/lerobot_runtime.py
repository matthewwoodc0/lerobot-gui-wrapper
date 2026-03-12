from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .config_store import default_for_key, normalize_path

_MODULE_AVAILABLE_CACHE: dict[tuple[str, str, str], bool] = {}
_MODULE_VERSION_CACHE: dict[tuple[str, str, str], str] = {}
_PYTHON_VERSION_CACHE: dict[tuple[str, str], str] = {}


def configured_lerobot_dir(config: dict[str, Any]) -> Path | None:
    raw = str(config.get("lerobot_dir", "")).strip()
    if not raw:
        return None
    try:
        return Path(normalize_path(raw))
    except Exception:
        return None


def lerobot_runtime_cwd(config: dict[str, Any]) -> Path | None:
    lerobot_dir = configured_lerobot_dir(config)
    if lerobot_dir is None:
        return None
    try:
        if lerobot_dir.exists():
            return lerobot_dir
    except OSError:
        return None
    return None


def configured_lerobot_env_dir(config: dict[str, Any]) -> Path:
    raw = str(config.get("lerobot_venv_dir", "")).strip()
    if raw:
        return Path(normalize_path(raw))
    return Path(str(default_for_key("lerobot_venv_dir", config)))


def configured_lerobot_python_path(config: dict[str, Any]) -> Path | None:
    env_dir = configured_lerobot_env_dir(config)
    for candidate in (
        env_dir / "bin" / "python3",
        env_dir / "bin" / "python",
        env_dir / "Scripts" / "python.exe",
    ):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def resolve_lerobot_python_executable(config: dict[str, Any]) -> str:
    configured = configured_lerobot_python_path(config)
    if configured is not None:
        return str(configured)
    return sys.executable


def build_lerobot_module_command(config: dict[str, Any], module_name: str) -> list[str]:
    return [resolve_lerobot_python_executable(config), "-m", str(module_name).strip()]


def runtime_signature(config: dict[str, Any]) -> tuple[str, str]:
    python_executable = resolve_lerobot_python_executable(config)
    cwd = lerobot_runtime_cwd(config)
    return python_executable, str(cwd) if cwd is not None else ""


def runtime_module_available(config: dict[str, Any], module_name: str, *, timeout_s: float = 10.0) -> bool:
    python_executable, cwd = runtime_signature(config)
    cache_key = (python_executable, cwd, str(module_name).strip())
    cached = _MODULE_AVAILABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    script = (
        "import importlib.util, sys\n"
        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)\n"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", script, str(module_name).strip()],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            cwd=cwd or None,
        )
        available = result.returncode == 0
    except Exception:
        available = False
    _MODULE_AVAILABLE_CACHE[cache_key] = available
    return available


def detect_runtime_module_version(config: dict[str, Any], module_name: str, *, timeout_s: float = 10.0) -> str:
    python_executable, cwd = runtime_signature(config)
    cache_key = (python_executable, cwd, str(module_name).strip())
    cached = _MODULE_VERSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    script = (
        "import importlib.metadata, sys\n"
        "try:\n"
        "    print(importlib.metadata.version(sys.argv[1]), end='')\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", script, str(module_name).strip()],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            cwd=cwd or None,
        )
    except Exception:
        version = "unknown"
    else:
        version = (result.stdout or "").strip() if result.returncode == 0 else "unknown"
        if not version:
            version = "unknown"

    _MODULE_VERSION_CACHE[cache_key] = version
    return version


def detect_runtime_python_version(config: dict[str, Any], *, timeout_s: float = 10.0) -> str:
    python_executable, cwd = runtime_signature(config)
    cache_key = (python_executable, cwd)
    cached = _PYTHON_VERSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    script = "import sys; print(sys.version.split()[0], end='')\n"
    try:
        result = subprocess.run(
            [python_executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            cwd=cwd or None,
        )
    except Exception:
        version = sys.version.split()[0]
    else:
        version = (result.stdout or "").strip() if result.returncode == 0 else ""
        if not version:
            version = sys.version.split()[0]

    _PYTHON_VERSION_CACHE[cache_key] = version
    return version
