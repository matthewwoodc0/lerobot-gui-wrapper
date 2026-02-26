from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ICON_CANDIDATES = (
    "lerobot-pipeline-manager-256.png",
    "lerobot-pipeline-manager-512.png",
    "lerobot-pipeline-manager-1024.png",
    "lerobot-pipeline-manager.png",
)


def _resolve_app_dir(app_dir: Path | None = None) -> Path:
    if app_dir is not None:
        return Path(app_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def find_app_icon_png(app_dir: Path | None = None) -> Path | None:
    icon_dir = _resolve_app_dir(app_dir) / "Resources" / "icons"
    for name in _ICON_CANDIDATES:
        candidate = icon_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def apply_tk_app_icon(*, root: Any, tk_module: Any, app_dir: Path | None = None) -> Path | None:
    icon_path = find_app_icon_png(app_dir)
    if icon_path is None:
        return None

    try:
        photo = tk_module.PhotoImage(file=str(icon_path))
        root.iconphoto(True, photo)
        setattr(root, "_lerobot_app_icon_photo", photo)
    except Exception:
        return None

    if sys.platform.startswith("win"):
        ico_path = icon_path.with_suffix(".ico")
        if ico_path.exists():
            try:
                root.iconbitmap(default=str(ico_path))
            except Exception:
                pass
    elif sys.platform == "darwin":
        # Best-effort Dock icon update for macOS when running from python/tkinter.
        try:
            from AppKit import NSApplication, NSImage  # type: ignore

            image = NSImage.alloc().initByReferencingFile_(str(icon_path))
            if image is not None:
                NSApplication.sharedApplication().setApplicationIconImage_(image)
        except Exception:
            pass
    return icon_path
