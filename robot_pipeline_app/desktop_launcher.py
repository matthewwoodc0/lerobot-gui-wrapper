from __future__ import annotations

import sys
import shutil
import subprocess
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


def _macos_bundle_script_content(app_dir: Path, python_executable: Path) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -u\n\n"
        f'APP_DIR="{app_dir}"\n'
        f'PYTHON_BIN="{python_executable}"\n'
        'LOG_DIR="${HOME}/Library/Logs/LeRobot Pipeline Manager"\n'
        'LOG_FILE="${LOG_DIR}/launcher.log"\n\n'
        'mkdir -p "$LOG_DIR" 2>/dev/null || true\n'
        'export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"\n\n'
        'if ! cd "$APP_DIR"; then\n'
        '  printf "\\n[%s] App directory missing: %s\\n" "$(date \'+%Y-%m-%d %H:%M:%S\')" "$APP_DIR" >> "$LOG_FILE"\n'
        "  if command -v osascript >/dev/null 2>&1; then\n"
        '    osascript -e \'display alert "LeRobot Pipeline Manager" message "Project directory not found for launcher. Reinstall launcher from Config tab."\' >/dev/null 2>&1 || true\n'
        "  fi\n"
        "  exit 1\n"
        "fi\n\n"
        '_is_venv_python() {\n'
        '  local candidate="${1:-}"\n'
        '  if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then\n'
        "    return 1\n"
        "  fi\n"
        '  "$candidate" -c \'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)\' >/dev/null 2>&1\n'
        "}\n\n"
        '_lerobot_python="$HOME/lerobot/lerobot_env/bin/python3"\n'
        'resolved_python=""\n'
        'if _is_venv_python "$PYTHON_BIN"; then\n'
        '  resolved_python="$PYTHON_BIN"\n'
        'elif _is_venv_python "$_lerobot_python"; then\n'
        '  resolved_python="$_lerobot_python"\n'
        "fi\n\n"
        'if [ -z "$resolved_python" ]; then\n'
        '  printf "\\n[%s] No valid LeRobot venv python detected. Tried: %s\\n" "$(date \'+%Y-%m-%d %H:%M:%S\')" "$PYTHON_BIN and $_lerobot_python" >> "$LOG_FILE"\n'
        "  if command -v osascript >/dev/null 2>&1; then\n"
        '    activation_cmd=$(/usr/bin/osascript <<\'OSA\'\n'
        'try\n'
        '  set dialogResult to display dialog "Could not find a valid LeRobot virtual environment.\\n\\nEdit the activation command below if your path is different." default answer "source ~/lerobot/lerobot_env/bin/activate" buttons {"Cancel", "Launch in Terminal"} default button "Launch in Terminal"\n'
        '  text returned of dialogResult\n'
        'on error number -128\n'
        '  return ""\n'
        'end try\n'
        'OSA\n'
        ")\n"
        '    if [ -n "$activation_cmd" ]; then\n'
        '      safe_cmd="${activation_cmd//\\"/\\\\\\"}"\n'
        '      safe_app_dir="${APP_DIR//\\"/\\\\\\"}"\n'
        '      /usr/bin/osascript <<OSA\n'
        'tell application "Terminal"\n'
        "  activate\n"
        '  do script "cd \\"$safe_app_dir\\"; $safe_cmd; python3 robot_pipeline.py gui"\n'
        "end tell\n"
        "OSA\n"
        "    fi\n"
        "  fi\n"
        "  exit 1\n"
        "fi\n"
        'PYTHON_BIN="$resolved_python"\n'
        'VIRTUAL_ENV="$(cd "$(dirname "$PYTHON_BIN")/.." 2>/dev/null && pwd || true)"\n'
        'if [ -n "$VIRTUAL_ENV" ]; then\n'
        "  export VIRTUAL_ENV\n"
        '  export PATH="$VIRTUAL_ENV/bin:$PATH"\n'
        "fi\n\n"
        'printf "\\n[%s] Launching GUI with %s (VIRTUAL_ENV=%s)\\n" "$(date \'+%Y-%m-%d %H:%M:%S\')" "$PYTHON_BIN" "${VIRTUAL_ENV:-}" >> "$LOG_FILE"\n'
        '"$PYTHON_BIN" "$APP_DIR/robot_pipeline.py" gui "$@" >> "$LOG_FILE" 2>&1\n'
        "status=$?\n"
        "if [ $status -ne 0 ]; then\n"
        '  printf "\\n[%s] Launcher exited with status %s\\n" "$(date \'+%Y-%m-%d %H:%M:%S\')" "$status" >> "$LOG_FILE"\n'
        "  if command -v osascript >/dev/null 2>&1; then\n"
        '    osascript -e \'display alert "LeRobot Pipeline Manager" message "App failed to start. See ~/Library/Logs/LeRobot Pipeline Manager/launcher.log"\' >/dev/null 2>&1 || true\n'
        "  fi\n"
        "fi\n"
        "exit $status\n"
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


def _macos_info_plist_content(bundle_executable: str, *, icon_resource_name: str | None = None) -> str:
    icon_block = ""
    if icon_resource_name:
        icon_block = (
            "    <key>CFBundleIconFile</key>\n"
            f"    <string>{icon_resource_name}</string>\n"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
        ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "    <key>CFBundleIdentifier</key>\n"
        "    <string>com.lerobot.pipeline-manager</string>\n"
        "    <key>CFBundleName</key>\n"
        "    <string>LeRobot Pipeline Manager</string>\n"
        "    <key>CFBundleDisplayName</key>\n"
        "    <string>LeRobot Pipeline Manager</string>\n"
        "    <key>CFBundleVersion</key>\n"
        "    <string>1.0</string>\n"
        "    <key>CFBundleShortVersionString</key>\n"
        "    <string>1.0</string>\n"
        "    <key>CFBundlePackageType</key>\n"
        "    <string>APPL</string>\n"
        "    <key>CFBundleExecutable</key>\n"
        f"    <string>{bundle_executable}</string>\n"
        f"{icon_block}"
        "    <key>NSHighResolutionCapable</key>\n"
        "    <true/>\n"
        "    <key>LSUIElement</key>\n"
        "    <false/>\n"
        "</dict>\n"
        "</plist>\n"
    )


def _install_linux_launcher(
    *,
    resolved_app_dir: Path,
    python_path: Path,
    home_path: Path,
) -> DesktopLauncherInstallResult:
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


def _install_macos_launcher(
    *,
    resolved_app_dir: Path,
    python_path: Path,
    home_path: Path,
) -> DesktopLauncherInstallResult:
    bundle_name = "LeRobot Pipeline Manager"
    bundle_executable = bundle_name
    local_bin = home_path / ".local" / "bin"
    applications_dir = home_path / "Applications"
    bundle_path = applications_dir / f"{bundle_name}.app"
    macos_dir = bundle_path / "Contents" / "MacOS"
    contents_dir = bundle_path / "Contents"
    resources_dir = contents_dir / "Resources"
    script_path = local_bin / "lerobot-pipeline-manager"
    bundle_exec_path = macos_dir / bundle_executable
    source_icon_path = find_app_icon_png(resolved_app_dir)
    installed_icon_path: Path | None = None

    try:
        local_bin.mkdir(parents=True, exist_ok=True)
        macos_dir.mkdir(parents=True, exist_ok=True)
        contents_dir.mkdir(parents=True, exist_ok=True)
        resources_dir.mkdir(parents=True, exist_ok=True)

        # Shell script used by the CLI launcher.
        script_content = _launcher_script_content(resolved_app_dir, python_path)
        # Finder launches via this bundle executable. It logs failures + shows alerts.
        bundle_script_content = _macos_bundle_script_content(resolved_app_dir, python_path)

        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)

        bundle_exec_path.write_text(bundle_script_content, encoding="utf-8")
        bundle_exec_path.chmod(0o755)

        icon_resource_name: str | None = None
        if source_icon_path is not None:
            installed_icon_path = resources_dir / "lerobot-pipeline-manager.png"
            shutil.copy2(source_icon_path, installed_icon_path)
            icon_resource_name = installed_icon_path.name

        plist_path = contents_dir / "Info.plist"
        plist_path.write_text(
            _macos_info_plist_content(bundle_executable, icon_resource_name=icon_resource_name),
            encoding="utf-8",
        )
    except OSError as exc:
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Failed to write launcher files: {exc}",
            script_path=script_path,
            desktop_entry_path=bundle_path,
            icon_path=installed_icon_path,
        )

    return DesktopLauncherInstallResult(
        ok=True,
        message=(
            f"Launcher installed.\n"
            f"App bundle: {bundle_path}\n"
            f"You can add it to your Dock or open it from ~/Applications."
        ),
        script_path=script_path,
        desktop_entry_path=bundle_path,
        icon_path=installed_icon_path,
    )


def install_desktop_launcher(
    *,
    app_dir: Path,
    python_executable: Path | None = None,
    platform_name: str | None = None,
    home_dir: Path | None = None,
) -> DesktopLauncherInstallResult:
    platform_value = (platform_name or sys.platform).lower()

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
    if not python_path.is_file():
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Python executable is not a file: {python_path}",
        )
    if (python_path.stat().st_mode & 0o111) == 0:
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Python executable is not runnable: {python_path}",
        )
    try:
        probe = subprocess.run(
            [str(python_path), "-c", "import tkinter"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return DesktopLauncherInstallResult(
            ok=False,
            message=f"Failed to validate Python runtime: {exc}",
        )
    if probe.returncode != 0:
        stderr = (probe.stderr or "").strip()
        detail = stderr.splitlines()[-1] if stderr else f"exit code {probe.returncode}"
        return DesktopLauncherInstallResult(
            ok=False,
            message=(
                f"Selected Python runtime cannot launch Tkinter: {detail}\n"
                "Use the same virtual environment you use to run the GUI, then reinstall the launcher."
            ),
        )

    home_path = Path(home_dir or Path.home()).expanduser().resolve()

    if platform_value.startswith("linux"):
        return _install_linux_launcher(
            resolved_app_dir=resolved_app_dir,
            python_path=python_path,
            home_path=home_path,
        )

    if platform_value == "darwin":
        return _install_macos_launcher(
            resolved_app_dir=resolved_app_dir,
            python_path=python_path,
            home_path=home_path,
        )

    return DesktopLauncherInstallResult(
        ok=False,
        message="Desktop launcher install is supported on Linux and macOS only.",
    )
