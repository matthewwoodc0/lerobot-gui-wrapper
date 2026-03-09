from __future__ import annotations

from pathlib import Path

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
