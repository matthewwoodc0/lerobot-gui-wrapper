from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable
from typing import Any
from urllib import error, request


# Simple TTL cache so repeated HF existence checks don't block the UI.
_hf_cache: dict[str, tuple[bool | None, float]] = {}
_HF_CACHE_TTL = 60.0


def _cache_busted(repo_id: str) -> None:
    """Invalidate the cached result for a repo so the next check goes to the network."""
    _hf_cache.pop(repo_id, None)


def increment_dataset_name(name: str) -> str:
    match = re.search(r"^(.*?)(\d+)$", name)
    if not match:
        return f"{name}_1"

    prefix, number = match.groups()
    next_number = int(number) + 1
    return f"{prefix}{next_number}"


def dataset_exists_on_hf(repo_id: str) -> bool | None:
    now = time.monotonic()
    cached = _hf_cache.get(repo_id)
    if cached is not None:
        result, ts = cached
        if now - ts < _HF_CACHE_TTL:
            return result

    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = request.Request(url=url, method="GET")

    result: bool | None
    try:
        with request.urlopen(req, timeout=3) as resp:
            result = resp.status == 200
    except error.HTTPError as exc:
        if exc.code == 404:
            result = False
        else:
            return None  # transient error — don't cache
    except error.URLError:
        return None  # transient error — don't cache

    _hf_cache[repo_id] = (result, now)
    return result


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


def has_eval_prefix(repo_id: str) -> bool:
    repo_name = repo_name_from_repo_id(repo_id)
    return repo_name.startswith("eval_")


def suggest_eval_prefixed_repo_id(username: str, dataset_name_or_repo_id: Any) -> tuple[str, bool]:
    _ = username  # compatibility: caller still passes username even when preserving bare names
    raw_value = str(dataset_name_or_repo_id or "").strip().strip("/")
    if not raw_value:
        return "eval_dataset_1", True

    if "/" in raw_value:
        owner, repo_name = raw_value.split("/", 1)
        if repo_name.startswith("eval_"):
            return f"{owner}/{repo_name}", False
        return f"{owner}/eval_{repo_name}", True

    if raw_value.startswith("eval_"):
        return raw_value, False
    return f"eval_{raw_value}", True


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


def extract_dataset_repo_id_arg(command_argv: list[str]) -> tuple[int, str] | None:
    for idx, arg in enumerate(command_argv):
        text = str(arg)
        if text.startswith("--dataset.repo_id="):
            value = text.split("=", 1)[1].strip()
            if value:
                return idx, value
            return None

        if text == "--dataset.repo_id" and idx + 1 < len(command_argv):
            value = str(command_argv[idx + 1]).strip()
            if value:
                return idx, value
            return None
    return None


def replace_dataset_repo_id_arg(command_argv: list[str], repo_id: str) -> list[str]:
    updated = [str(part) for part in command_argv]
    match = extract_dataset_repo_id_arg(updated)
    if match is None:
        return updated

    idx, _ = match
    current = updated[idx]
    if current.startswith("--dataset.repo_id="):
        updated[idx] = f"--dataset.repo_id={repo_id}"
        return updated

    if current == "--dataset.repo_id" and idx + 1 < len(updated):
        updated[idx + 1] = repo_id
    return updated


def normalize_deploy_rerun_command(
    command_argv: list[str],
    username: str,
    local_roots: list[Path] | None = None,
    exists_fn: Callable[[str], bool | None] | None = None,
) -> tuple[list[str], str | None]:
    argv = [str(part) for part in command_argv]
    match = extract_dataset_repo_id_arg(argv)
    if match is None:
        return argv, None

    _, original_repo_id = match
    prefixed_repo_id, prefixed = suggest_eval_prefixed_repo_id(username, original_repo_id)
    resolved_repo_id, iterated, _ = resolve_unique_repo_id(
        username=username,
        dataset_name_or_repo_id=prefixed_repo_id,
        local_roots=local_roots,
        exists_fn=exists_fn,
    )

    if not prefixed and not iterated and resolved_repo_id == original_repo_id:
        return argv, None

    updated = replace_dataset_repo_id_arg(argv, resolved_repo_id)
    reasons: list[str] = []
    if prefixed and not has_eval_prefix(original_repo_id):
        reasons.append("added eval_ prefix")
    if iterated:
        reasons.append("iterated to avoid collision")
    if not reasons:
        reasons.append("normalized repo id")

    message = (
        "Auto-fixed deploy eval dataset for rerun: "
        f"{original_repo_id} -> {resolved_repo_id} ({', '.join(reasons)})."
    )
    return updated, message
