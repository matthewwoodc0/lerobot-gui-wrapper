from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from .repo_utils import repo_name_from_repo_id


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slug_tag(value: str) -> str:
    normalized = _NON_ALNUM.sub("-", value.lower()).strip("-")
    return normalized


def _task_tag(task: str | None) -> str | None:
    if not task:
        return None
    words = [chunk for chunk in _NON_ALNUM.split(task.lower()) if chunk]
    if not words:
        return None
    compact = "-".join(words[:4])
    if len(compact) > 36:
        compact = compact[:36].rstrip("-")
    if not compact:
        return None
    return f"task-{compact}"


def default_dataset_tags(config: dict[str, Any], dataset_repo_id: str, task: str | None = None) -> list[str]:
    tags: list[str] = [
        "lerobot",
        "so101",
        "robotics",
        "teleoperation",
        "demonstrations",
    ]

    repo_name = repo_name_from_repo_id(dataset_repo_id)
    repo_tag = _slug_tag(repo_name)
    if repo_tag:
        tags.append(f"dataset-{repo_tag}")

    owner = str(dataset_repo_id).split("/", 1)[0].strip()
    configured_user = str(config.get("hf_username", "")).strip()
    if owner and owner != configured_user:
        owner_tag = _slug_tag(owner)
        if owner_tag:
            tags.append(f"owner-{owner_tag}")

    derived_task_tag = _task_tag(task)
    if derived_task_tag:
        tags.append(derived_task_tag)

    # Preserve first-seen order while de-duplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def build_dataset_card_text(
    *,
    dataset_repo_id: str,
    dataset_name: str,
    tags: list[str],
    task: str | None = None,
) -> str:
    yaml_lines = ["---", "tags:"]
    yaml_lines.extend(f"  - {tag}" for tag in tags)
    yaml_lines.append("---")
    yaml = "\n".join(yaml_lines)

    lines = [
        yaml,
        "",
        f"# {dataset_name}",
        "",
        "Auto-generated dataset card created after recording upload.",
        "",
        f"- Repo: `{dataset_repo_id}`",
        f"- Name: `{dataset_name}`",
    ]
    if task:
        lines.append(f"- Task: {task}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This dataset was uploaded via the LeRobot GUI Wrapper.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataset_card_temp(
    *,
    dataset_repo_id: str,
    dataset_name: str,
    tags: list[str],
    task: str | None = None,
) -> Path:
    content = build_dataset_card_text(
        dataset_repo_id=dataset_repo_id,
        dataset_name=dataset_name,
        tags=tags,
        task=task,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="lerobot_dataset_card_",
        suffix=".md",
        delete=False,
    ) as handle:
        handle.write(content)
        path = Path(handle.name)
    return path


def build_dataset_tag_upload_command(dataset_repo_id: str, card_path: Path) -> list[str]:
    return [
        "huggingface-cli",
        "upload",
        dataset_repo_id,
        str(card_path),
        "README.md",
        "--repo-type",
        "dataset",
    ]


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
