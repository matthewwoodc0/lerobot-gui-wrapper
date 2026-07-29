from __future__ import annotations

from pathlib import Path

from .package_paths import project_root, resolve_icon_png, resolve_resource

_ICON_CANDIDATES = (
    "lerobot-pipeline-manager-256.png",
    "lerobot-pipeline-manager-512.png",
    "lerobot-pipeline-manager-1024.png",
    "lerobot-pipeline-manager.png",
)


def _resolve_app_dir(app_dir: Path | None = None) -> Path:
    if app_dir is not None:
        return Path(app_dir).expanduser().resolve()
    root = project_root()
    if root is not None:
        return root
    return Path(__file__).resolve().parent


def find_app_icon_png(app_dir: Path | None = None) -> Path | None:
    if app_dir is not None:
        icon_dir = _resolve_app_dir(app_dir) / "Resources" / "icons"
        for name in _ICON_CANDIDATES:
            candidate = icon_dir / name
            if candidate.exists() and candidate.is_file():
                return candidate
        # Also accept icons next to an explicit app dir without Resources/.
        for name in _ICON_CANDIDATES:
            candidate = _resolve_app_dir(app_dir) / "icons" / name
            if candidate.exists() and candidate.is_file():
                return candidate

    packaged = resolve_icon_png()
    if packaged is not None:
        return packaged

    for name in _ICON_CANDIDATES:
        candidate = resolve_resource("icons", name)
        if candidate is not None and candidate.is_file():
            return candidate
    return None
