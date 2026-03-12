from __future__ import annotations

from typing import Any

from .commands import _resolve_lerobot_python_executable
from .compat import resolve_visualize_dataset_entrypoint


def build_visualize_dataset_command(
    config: dict[str, Any],
    repo_id: str,
    episode_index: int,
) -> list[str]:
    """Build command to launch lerobot dataset visualization via Rerun."""
    return [
        _resolve_lerobot_python_executable(config),
        "-m",
        resolve_visualize_dataset_entrypoint(config),
        "--repo-id",
        str(repo_id).strip(),
        "--episode-index",
        str(int(episode_index)),
    ]
