from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from typing import Any
from urllib import error, request


def increment_dataset_name(name: str) -> str:
    match = re.search(r"^(.*?)(\d+)$", name)
    if not match:
        return f"{name}_1"

    prefix, number = match.groups()
    next_number = int(number) + 1
    return f"{prefix}{next_number}"


def dataset_exists_on_hf(repo_id: str) -> bool | None:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = request.Request(url=url, method="GET")

    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except error.HTTPError as exc:
        if exc.code == 404:
            return False
        return None
    except error.URLError:
        return None


def suggest_dataset_name(config: dict[str, Any]) -> tuple[str, bool]:
    last_used = str(config.get("last_dataset_name", "dataset_1"))
    candidate = increment_dataset_name(last_used)

    username = str(config["hf_username"])
    checked_remote = False

    for _ in range(25):
        exists = dataset_exists_on_hf(f"{username}/{candidate}")
        if exists is None:
            return candidate, checked_remote
        checked_remote = True
        if not exists:
            return candidate, checked_remote
        candidate = increment_dataset_name(candidate)

    return candidate, checked_remote


def normalize_repo_id(username: str, dataset_name_or_repo_id: Any) -> str:
    name = str(dataset_name_or_repo_id or "").strip().strip("/")
    if not name:
        return f"{username}/dataset_1"
    if "/" in name:
        return name
    return f"{username}/{name}"


def repo_name_from_repo_id(repo_id: str) -> str:
    clean = repo_id.strip().strip("/")
    if not clean:
        return "dataset"
    parts = clean.split("/")
    return parts[-1] if parts[-1] else "dataset"


def suggest_eval_dataset_name(config: dict[str, Any], model_name: str = "") -> str:
    previous = str(config.get("last_eval_dataset_name", "")).strip()
    if previous:
        return increment_dataset_name(previous)

    clean_model = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name).strip("_")
    base = f"eval_{clean_model}" if clean_model else "eval_run"
    return f"{base}_1"


def resolve_unique_repo_id(
    username: str,
    dataset_name_or_repo_id: str,
    local_roots: list[Path] | None = None,
    max_attempts: int = 200,
    exists_fn: Callable[[str], bool | None] | None = None,
) -> tuple[str, bool, bool]:
    candidate_repo_id = normalize_repo_id(username, dataset_name_or_repo_id)
    if "/" in candidate_repo_id:
        owner, candidate_name = candidate_repo_id.split("/", 1)
    else:
        owner, candidate_name = username, candidate_repo_id

    roots = [Path(str(root)).expanduser() for root in (local_roots or [])]
    remote_exists_fn = exists_fn or dataset_exists_on_hf
    checked_remote = False
    adjusted = False

    for _ in range(max_attempts):
        local_conflict = any((root / candidate_name).exists() for root in roots)
        remote_exists = remote_exists_fn(f"{owner}/{candidate_name}")
        if remote_exists is not None:
            checked_remote = True

        if not local_conflict and remote_exists is not True:
            return f"{owner}/{candidate_name}", adjusted, checked_remote

        adjusted = True
        candidate_name = increment_dataset_name(candidate_name)

    return f"{owner}/{candidate_name}", adjusted, checked_remote
