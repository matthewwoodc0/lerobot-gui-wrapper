from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_QT_BOOTSTRAP_CACHE: dict[tuple[str, str], tuple[bool, str | None]] = {}
_PREPARED = False


def current_qt_platform() -> str:
    return str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()


def _linux_qt_platform_candidates() -> tuple[str, ...]:
    session_type = str(os.environ.get("XDG_SESSION_TYPE", "")).strip().lower()
    has_wayland = bool(str(os.environ.get("WAYLAND_DISPLAY", "")).strip()) or session_type == "wayland"
    has_x11 = bool(str(os.environ.get("DISPLAY", "")).strip()) or session_type == "x11"

    ordered: list[str] = []
    if has_wayland:
        ordered.append("wayland")
    if has_x11:
        ordered.append("xcb")
    if not ordered:
        ordered.extend(["wayland", "xcb"])

    unique: list[str] = []
    for candidate in ordered:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _split_qt_env_paths(raw_value: str) -> list[str]:
    return [chunk for chunk in str(raw_value or "").split(os.pathsep) if chunk]


def _looks_like_cv2_qt_path(path_text: str) -> bool:
    normalized = str(path_text or "").replace("\\", "/").lower()
    return "/cv2/qt" in normalized or normalized.endswith("/cv2")


def _resolve_pyside6_root() -> Path | None:
    spec = importlib.util.find_spec("PySide6")
    origin = getattr(spec, "origin", None)
    if not origin:
        return None
    return Path(origin).resolve().parent


def _resolve_pyside6_plugins_dir() -> Path | None:
    package_dir = _resolve_pyside6_root()
    if package_dir is None:
        return None

    for candidate in (
        package_dir / "Qt" / "plugins",
        package_dir / "plugins",
        package_dir / "Qt6" / "plugins",
    ):
        try:
            if (candidate / "platforms").is_dir():
                return candidate
        except OSError:
            continue
    return None


def _resolve_pyside6_lib_dir() -> Path | None:
    package_dir = _resolve_pyside6_root()
    if package_dir is None:
        return None
    for candidate in (
        package_dir / "Qt" / "lib",
        package_dir / "lib",
        package_dir / "Qt6" / "lib",
    ):
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def _prepend_path_env(var_name: str, path_text: str) -> None:
    current_paths = _split_qt_env_paths(os.environ.get(var_name, ""))
    if path_text in current_paths:
        return
    os.environ[var_name] = os.pathsep.join([path_text, *current_paths]) if current_paths else path_text


def _prepend_conda_lib_dir() -> None:
    if not sys.platform.startswith("linux"):
        return
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if not conda_prefix:
        return
    lib_dir = Path(conda_prefix) / "lib"
    try:
        if not lib_dir.is_dir():
            return
    except OSError:
        return
    _prepend_path_env("LD_LIBRARY_PATH", str(lib_dir))


def prepare_qt_environment() -> None:
    """Prefer PySide6 plugin paths and strip OpenCV Qt path pollution."""
    global _PREPARED
    _prepend_conda_lib_dir()
    plugin_dir = _resolve_pyside6_plugins_dir()
    lib_dir = _resolve_pyside6_lib_dir()

    # Help macOS load Qt frameworks that platform plugins link via @rpath.
    # Linux already uses system/conda library paths; avoid mutating LD_LIBRARY_PATH
    # unless conda already needs it (handled above).
    if lib_dir is not None and sys.platform == "darwin":
        lib_text = str(lib_dir)
        _prepend_path_env("DYLD_FRAMEWORK_PATH", lib_text)
        _prepend_path_env("DYLD_LIBRARY_PATH", lib_text)

    existing_plugin_path = _split_qt_env_paths(os.environ.get("QT_PLUGIN_PATH", ""))
    filtered_plugin_path = [path for path in existing_plugin_path if not _looks_like_cv2_qt_path(path)]

    existing_platform_path = _split_qt_env_paths(os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""))
    filtered_platform_path = [path for path in existing_platform_path if not _looks_like_cv2_qt_path(path)]

    if plugin_dir is not None:
        plugin_dir_text = str(plugin_dir)
        platform_dir_text = str(plugin_dir / "platforms")
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(
            [plugin_dir_text, *[path for path in filtered_plugin_path if path != plugin_dir_text]]
        )
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.pathsep.join(
            [platform_dir_text, *[path for path in filtered_platform_path if path != platform_dir_text]]
        )
        _PREPARED = True
        return

    if filtered_plugin_path:
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(filtered_plugin_path)
    else:
        os.environ.pop("QT_PLUGIN_PATH", None)

    if filtered_platform_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.pathsep.join(filtered_platform_path)
    else:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    _PREPARED = True


def probe_qt_platform_support(
    *,
    python_executable: str | None = None,
    platform_name: str | None = None,
) -> tuple[bool, str | None]:
    prepare_qt_environment()
    resolved_python = str(python_executable or sys.executable)
    resolved_platform = str(platform_name or current_qt_platform() or "default").strip().lower()
    cache_key = (resolved_python, resolved_platform)
    cached = _QT_BOOTSTRAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    probe_env = dict(os.environ)
    # Ensure child process sees prepared plugin paths even if it was forked earlier.
    if platform_name:
        probe_env["QT_QPA_PLATFORM"] = str(platform_name)
    script = (
        "from robot_pipeline_app.qt_bootstrap import prepare_qt_environment\n"
        "prepare_qt_environment()\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication(['qt-smoke'])\n"
        "print('ok')\n"
    )
    result: tuple[bool, str | None]
    try:
        probe = subprocess.run(
            [resolved_python, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=probe_env,
            timeout=15,
        )
    except Exception as exc:
        result = (False, str(exc))
    else:
        ok = probe.returncode == 0 and "ok" in (probe.stdout or "")
        detail: str | None = None if ok else (probe.stderr.strip() or probe.stdout.strip() or "Qt smoke check failed")
        result = (ok, detail)

    _QT_BOOTSTRAP_CACHE[cache_key] = result
    return result


def clear_qt_bootstrap_cache() -> None:
    _QT_BOOTSTRAP_CACHE.clear()


def _format_qt_bootstrap_error(platform_name: str, reason: str | None) -> str:
    detail = str(reason or "Qt smoke check failed").strip()
    if "xcb-cursor0" in detail or "libxcb-cursor0" in detail:
        return (
            f"Qt could not initialize the '{platform_name}' platform plugin because the active runtime "
            "is missing the xcb-cursor library. This machine can still work without sudo if a Wayland backend "
            "is available or if the required xcb runtime is installed inside the active conda environment."
            f" Original error: {detail}"
        )
    return f"Qt could not initialize the '{platform_name}' platform plugin. {detail}"


def ensure_supported_qt_platform(*, python_executable: str | Path | None = None) -> None:
    prepare_qt_environment()
    resolved_python = str(python_executable) if python_executable is not None else sys.executable
    platform_name = current_qt_platform()

    if platform_name:
        ok, reason = probe_qt_platform_support(
            python_executable=resolved_python,
            platform_name=platform_name,
        )
        if not ok:
            raise RuntimeError(_format_qt_bootstrap_error(platform_name, reason))
        return

    if not sys.platform.startswith("linux"):
        # On macOS/Windows, let Qt pick the native platform when none is forced.
        ok, reason = probe_qt_platform_support(
            python_executable=resolved_python,
            platform_name=None,
        )
        if not ok:
            # Headless CI / SSH: fall back to offscreen then minimal.
            for candidate in ("offscreen", "minimal"):
                cand_ok, cand_reason = probe_qt_platform_support(
                    python_executable=resolved_python,
                    platform_name=candidate,
                )
                if cand_ok:
                    os.environ["QT_QPA_PLATFORM"] = candidate
                    return
            raise RuntimeError(_format_qt_bootstrap_error(platform_name or "default", reason))
        return

    failures: list[tuple[str, str | None]] = []
    for candidate in _linux_qt_platform_candidates():
        ok, reason = probe_qt_platform_support(
            python_executable=resolved_python,
            platform_name=candidate,
        )
        if ok:
            os.environ["QT_QPA_PLATFORM"] = candidate
            return
        failures.append((candidate, reason))

    default_ok, default_reason = probe_qt_platform_support(
        python_executable=resolved_python,
        platform_name=None,
    )
    if default_ok:
        return

    for candidate in ("offscreen", "minimal"):
        ok, reason = probe_qt_platform_support(
            python_executable=resolved_python,
            platform_name=candidate,
        )
        if ok:
            os.environ["QT_QPA_PLATFORM"] = candidate
            return
        failures.append((candidate, reason))

    if failures:
        first_platform, first_reason = failures[0]
        raise RuntimeError(_format_qt_bootstrap_error(first_platform, first_reason))
    raise RuntimeError(_format_qt_bootstrap_error("default", default_reason))


def ensure_safe_qt_bootstrap(*, python_executable: str | Path | None = None) -> None:
    prepare_qt_environment()
    platform_name = current_qt_platform()
    if platform_name not in {"offscreen", "minimal"}:
        return
    ok, reason = probe_qt_platform_support(
        python_executable=str(python_executable) if python_executable is not None else None,
        platform_name=platform_name,
    )
    if not ok:
        raise RuntimeError(reason or f"Qt bootstrap failed for platform '{platform_name}'")


def configure_headless_qt_for_tests() -> str:
    """Force a headless Qt platform suitable for automated GUI tests."""
    prepare_qt_environment()
    existing = current_qt_platform()
    if existing in {"offscreen", "minimal"}:
        ok, reason = probe_qt_platform_support(platform_name=existing)
        if ok:
            return existing
        raise RuntimeError(reason or f"Configured QT_QPA_PLATFORM={existing} is not usable")

    for candidate in ("offscreen", "minimal"):
        ok, _reason = probe_qt_platform_support(platform_name=candidate)
        if ok:
            os.environ["QT_QPA_PLATFORM"] = candidate
            return candidate
    raise RuntimeError(
        "PySide6 is installed but no headless Qt platform plugin (offscreen/minimal) could initialize. "
        "Reinstall PySide6 in a clean virtual environment."
    )
