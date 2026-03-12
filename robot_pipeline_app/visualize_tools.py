from __future__ import annotations

from typing import Any

from .compat import resolve_visualize_dataset_entrypoint
from .lerobot_runtime import build_lerobot_module_command


def build_visualize_dataset_command(
    config: dict[str, Any],
    repo_id: str,
    episode_index: int,
) -> list[str]:
    """Build command to launch lerobot dataset visualization via Rerun."""
    return [
        *build_lerobot_module_command(config, resolve_visualize_dataset_entrypoint(config)),
        "--repo-id",
        str(repo_id).strip(),
        "--episode-index",
        str(int(episode_index)),
    ]
