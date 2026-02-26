from __future__ import annotations

import sys


def keyboard_input_help_title() -> str:
    return "Keyboard Input Setup"


def keyboard_input_help_text(platform_name: str | None = None) -> str:
    platform = str(platform_name or sys.platform).lower()
    shared = (
        "Arrow keys only work when the Run Controls window is focused.\n\n"
        "Quick checks:\n"
        "1. Click inside the Run Controls popout, then press Left/Right.\n"
        "2. Keep this app in the foreground.\n"
        "3. If input still fails, use the on-screen buttons as fallback."
    )

    if platform == "darwin":
        return (
            f"{shared}\n\n"
            "macOS permissions may block global-style keyboard hooks:\n"
            "1. Open System Settings > Privacy & Security > Input Monitoring.\n"
            "2. Enable the app that launches this GUI (Terminal, iTerm, Python, or packaged app).\n"
            "3. Also enable Accessibility for that same app if prompted.\n"
            "4. Fully quit and relaunch the approved app."
        )

    if platform.startswith("linux"):
        return (
            f"{shared}\n\n"
            "Linux notes:\n"
            "- Focused-window keys usually need no extra permission.\n"
            "- Wayland often blocks global key capture/injection by design.\n"
            "- X11 is generally less restrictive.\n"
            "- If using Flatpak/Snap, check sandbox permissions."
        )

    if platform.startswith("win"):
        return (
            f"{shared}\n\n"
            "Windows notes:\n"
            "- Focused-window keys usually work without extra setup.\n"
            "- Run from an elevated shell only if your environment policy requires it."
        )

    return shared
