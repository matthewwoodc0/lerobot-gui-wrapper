from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compat import resolve_edit_dataset_entrypoint
from .lerobot_runtime import build_lerobot_module_command
from .repo_utils import repo_name_from_repo_id


_ROOT_STYLE_EDIT_DATASET_MODULES = {
    "scripts.lerobot_edit_dataset",
    "lerobot.scripts.lerobot_edit_dataset",
}


def _record_data_root(config: dict[str, Any]) -> Path | None:
    raw = str(config.get("record_data_dir", "")).strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except Exception:
        return None


def _edit_dataset_uses_root_style_flags(config: dict[str, Any], entrypoint: str) -> bool:
    configured = str(config.get("lerobot_edit_dataset_entrypoint", "")).strip().lower()
    if configured:
        return configured in _ROOT_STYLE_EDIT_DATASET_MODULES

    lowered = str(entrypoint or "").strip().lower()
    if lowered in _ROOT_STYLE_EDIT_DATASET_MODULES:
        return True

    lerobot_dir_raw = str(config.get("lerobot_dir", "")).strip()
    if not lerobot_dir_raw:
        return False
    lerobot_dir = Path(lerobot_dir_raw).expanduser()
    return any(
        (lerobot_dir / relative_path).exists()
        for relative_path in (
            "src/lerobot/scripts/lerobot_edit_dataset.py",
            "lerobot/scripts/lerobot_edit_dataset.py",
            "scripts/lerobot_edit_dataset.py",
        )
    )


def _effective_local_repo_id(config: dict[str, Any], repo_id: str) -> str:
    cleaned_repo_id = str(repo_id).strip().strip("/")
    if not cleaned_repo_id or "/" not in cleaned_repo_id:
        return cleaned_repo_id

    root = _record_data_root(config)
    if root is None:
        return cleaned_repo_id
    if (root / cleaned_repo_id).exists():
        return cleaned_repo_id

    flat_repo_name = repo_name_from_repo_id(cleaned_repo_id)
    if (root / flat_repo_name).exists():
        return flat_repo_name
    return cleaned_repo_id


def normalize_dataset_repo_ids(raw_values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        cleaned = str(value).strip().strip("/")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _edit_dataset_command(
    *,
    config: dict[str, Any],
    repo_id: str,
    operation_type: str,
    episode_indices: list[int],
) -> list[str]:
    entrypoint = resolve_edit_dataset_entrypoint(config)
    indices = [int(index) for index in episode_indices]
    command = [
        *build_lerobot_module_command(config, entrypoint),
    ]
    if _edit_dataset_uses_root_style_flags(config, entrypoint):
        command.append(f"--repo_id={_effective_local_repo_id(config, repo_id)}")
        root = _record_data_root(config)
        if root is not None:
            command.append(f"--root={root}")
    else:
        command.append(f"--dataset.repo_id={str(repo_id).strip()}")
    command.extend(
        (
            f"--operation.type={operation_type}",
            f"--operation.episode_indices={json.dumps(indices)}",
        )
    )
    return command


def build_merge_datasets_command(
    config: dict[str, Any],
    output_repo_id: str,
    source_repo_ids: list[str],
) -> list[str]:
    """Build the lerobot edit_dataset command for merging datasets."""
    entrypoint = resolve_edit_dataset_entrypoint(config)
    normalized_sources = normalize_dataset_repo_ids(source_repo_ids)
    command = [
        *build_lerobot_module_command(config, entrypoint),
    ]
    if _edit_dataset_uses_root_style_flags(config, entrypoint):
        command.append(f"--repo_id={_effective_local_repo_id(config, output_repo_id)}")
        root = _record_data_root(config)
        if root is not None:
            command.append(f"--root={root}")
        source_payload = [_effective_local_repo_id(config, repo_id) for repo_id in normalized_sources]
    else:
        command.append(f"--dataset.repo_id={str(output_repo_id).strip()}")
        source_payload = normalized_sources
    command.extend(
        (
            "--operation.type=merge",
            f"--operation.repo_ids={json.dumps(source_payload)}",
        )
    )
    return command


def build_delete_episodes_command(
    config: dict[str, Any],
    repo_id: str,
    episode_indices: list[int],
) -> list[str]:
    """Build the lerobot edit_dataset command for deleting episodes."""
    return _edit_dataset_command(
        config=config,
        repo_id=repo_id,
        operation_type="delete_episodes",
        episode_indices=episode_indices,
    )


def build_keep_episodes_command(
    config: dict[str, Any],
    repo_id: str,
    episode_indices: list[int],
) -> list[str]:
    """Build the lerobot edit_dataset command for keeping only specified episodes."""
    return _edit_dataset_command(
        config=config,
        repo_id=repo_id,
        operation_type="keep_episodes",
        episode_indices=episode_indices,
    )


def dataset_local_path_candidates(
    config: dict[str, Any],
    repo_id: str,
    *,
    selected_dataset_path: str | Path | None = None,
) -> list[Path]:
    repo_name = repo_name_from_repo_id(repo_id)
    candidates: list[Path] = []
    if selected_dataset_path:
        candidates.append(Path(selected_dataset_path))

    root = Path(str(config.get("record_data_dir", "data"))).expanduser()
    candidates.append(root / repo_name)
    cleaned_repo = str(repo_id).strip().strip("/")
    if "/" in cleaned_repo:
        owner, _name = cleaned_repo.split("/", 1)
        candidates.append(root / owner / repo_name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def parse_dataset_repo_ids(raw_text: str) -> list[str]:
    values: list[str] = []
    for chunk in str(raw_text or "").replace(",", "\n").splitlines():
        cleaned = chunk.strip()
        if cleaned:
            values.append(cleaned)
    return normalize_dataset_repo_ids(values)


def find_local_dataset_episodes_file(
    config: dict[str, Any],
    repo_id: str,
    *,
    selected_dataset_path: str | Path | None = None,
) -> Path | None:
    for candidate in dataset_local_path_candidates(
        config,
        repo_id,
        selected_dataset_path=selected_dataset_path,
    ):
        episodes_path = candidate / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            return episodes_path
    return None


def collect_local_dataset_episode_indices(
    config: dict[str, Any],
    repo_id: str,
    *,
    selected_dataset_path: str | Path | None = None,
) -> tuple[list[int], str | None]:
    episodes_path = find_local_dataset_episodes_file(
        config,
        repo_id,
        selected_dataset_path=selected_dataset_path,
    )
    if episodes_path is None:
        return [], "Dataset not found locally. Download it first or check the dataset path."
    try:
        lines = episodes_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], f"Unable to read {episodes_path}: {exc}"
    return list(range(len(lines))), None
