from __future__ import annotations

import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

from .app_icon import find_app_icon_png


@dataclass(frozen=True)
class DesktopLauncherInstallResult:
    ok: bool
    message: str
    script_path: Path | None = None
    desktop_entry_path: Path | None = None
    icon_path: Path | None = None


def _launcher_script_content(app_dir: Path, python_executable: Path) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'APP_DIR="{app_dir}"\n'
        f'PYTHON_BIN="{python_executable}"\n\n'
        'cd "$APP_DIR"\n'
        'exec "$PYTHON_BIN" "$APP_DIR/robot_pipeline.py" gui "$@"\n'
    )


def _desktop_entry_content(script_path: Path, icon_path: Path | None = None) -> str:
    icon_line = f"Icon={icon_path}\n" if icon_path is not None else ""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        "Name=LeRobot Pipeline Manager\n"
        "Comment=Launch LeRobot record/deploy GUI\n"
        f"Exec={script_path}\n"
        f"{icon_line}"
        "Terminal=false\n"
        "Categories=Development;Science;Robotics;\n"
        "StartupNotify=true\n"
    )


def install_desktop_launcher(
    *,
    app_dir: Path,
    python_executable: Path | None = None,
    platform_name: str | None = None,
    home_dir: Path | None = None,
) -> DesktopLauncherInstallResult:
    platform_value = (platform_name or sys.platform).lower()
    if not platform_value.startswith("linux"):
        return DesktopLauncherInstallResult(
            ok=False,
            message="Desktop launcher install is currently supported on Linux only.",
        )

    resolved_app_dir = Path(app_dir).expanduser().resolve()
    entrypoint = resolved_app_dir / "robot_pipeline.py"
    if not entrypoint.exists():
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Could not find GUI entrypoint at {entrypoint}.",
        )

    python_path = Path(python_executable or Path(sys.executable)).expanduser().resolve()
    if not python_path.exists():
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Python executable not found: {python_path}",
        )

    home_path = Path(home_dir or Path.home()).expanduser().resolve()
    local_bin = home_path / ".local" / "bin"
    applications_dir = home_path / ".local" / "share" / "applications"
    icons_dir = home_path / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    script_path = local_bin / "lerobot-pipeline-manager"
    desktop_path = applications_dir / "lerobot-pipeline-manager.desktop"
    source_icon_path = find_app_icon_png(resolved_app_dir)
    installed_icon_path: Path | None = None

    try:
        local_bin.mkdir(parents=True, exist_ok=True)
        applications_dir.mkdir(parents=True, exist_ok=True)
        if source_icon_path is not None:
            icons_dir.mkdir(parents=True, exist_ok=True)
            installed_icon_path = icons_dir / "lerobot-pipeline-manager.png"
            shutil.copy2(source_icon_path, installed_icon_path)
        script_path.write_text(
            _launcher_script_content(resolved_app_dir, python_path),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        desktop_path.write_text(_desktop_entry_content(script_path, installed_icon_path), encoding="utf-8")
    except OSError as exc:
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Failed to write launcher files: {exc}",
            script_path=script_path,
            desktop_entry_path=desktop_path,
            icon_path=installed_icon_path,
        )

    return DesktopLauncherInstallResult(
        ok=True,
        message="Desktop launcher installed. Open 'LeRobot Pipeline Manager' from your app menu.",
        script_path=script_path,
        desktop_entry_path=desktop_path,
        icon_path=installed_icon_path,
    )
