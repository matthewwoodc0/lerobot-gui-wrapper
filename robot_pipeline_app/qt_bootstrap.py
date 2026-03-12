from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_QT_BOOTSTRAP_CACHE: dict[tuple[str, str], tuple[bool, str | None]] = {}


def current_qt_platform() -> str:
    return str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()


def probe_qt_platform_support(
    *,
    python_executable: str | None = None,
    platform_name: str | None = None,
) -> tuple[bool, str | None]:
    resolved_python = str(python_executable or sys.executable)
    resolved_platform = str(platform_name or current_qt_platform() or "default").strip().lower()
    cache_key = (resolved_python, resolved_platform)
    cached = _QT_BOOTSTRAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    probe_env = dict(os.environ)
    if platform_name:
        probe_env["QT_QPA_PLATFORM"] = str(platform_name)
    script = (
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication(['qt-smoke'])\n"
        "print('ok')\n"
    )
    try:
        probe = subprocess.run(
            [resolved_python, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=probe_env,
            timeout=10,
        )
    except Exception as exc:
        result = (False, str(exc))
    else:
        ok = probe.returncode == 0 and probe.stdout.strip() == "ok"
        detail = None if ok else (probe.stderr.strip() or probe.stdout.strip() or "Qt smoke check failed")
        result = (ok, detail)

    _QT_BOOTSTRAP_CACHE[cache_key] = result
    return result


def ensure_safe_qt_bootstrap(*, python_executable: str | Path | None = None) -> None:
    platform_name = current_qt_platform()
    if platform_name not in {"offscreen", "minimal"}:
        return
    ok, reason = probe_qt_platform_support(
        python_executable=str(python_executable) if python_executable is not None else None,
        platform_name=platform_name,
    )
    if not ok:
        raise RuntimeError(reason or f"Qt bootstrap failed for platform '{platform_name}'")
