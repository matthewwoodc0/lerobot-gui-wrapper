from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from typing import Callable
from typing import Any
from urllib import error, request


# Simple TTL cache so repeated HF existence checks don't block the UI.
_hf_cache: dict[str, tuple[bool | None, float]] = {}
_HF_CACHE_TTL = 60.0
_hf_model_cache: dict[str, tuple[bool | None, float]] = {}
_hf_json_cache: dict[str, tuple[Any, float]] = {}
_HF_JSON_CACHE_TTL = 45.0
_HF_API_TIMEOUT_S = 4.0

_CACHE_MAX_SIZE = 400


def _evict_if_full(cache: dict, max_size: int = _CACHE_MAX_SIZE) -> None:
    if len(cache) < max_size:
        return
    # Evict the oldest quarter of entries by timestamp (second element
    # of each tuple value).
    try:
        sorted_keys = sorted(cache, key=lambda k: cache[k][1])
        for key in sorted_keys[: max(1, max_size // 4)]:
            cache.pop(key, None)
    except Exception:
        pass


def _cache_busted(repo_id: str) -> None:
    """Invalidate the cached result for a repo so the next check goes to the network."""
    _hf_cache.pop(repo_id, None)
    _hf_model_cache.pop(repo_id, None)
    clean_repo = str(repo_id).strip().strip("/")
    if not clean_repo:
        return
    for key in list(_hf_json_cache.keys()):
        if clean_repo in key:
            _hf_json_cache.pop(key, None)


def _safe_limit(limit: int, *, max_limit: int = 200) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = max_limit
    if parsed <= 0:
        return 1
    if parsed > max_limit:
        return max_limit
    return parsed


def _hf_get_json(url: str, *, cache_key: str | None = None, timeout_s: float = _HF_API_TIMEOUT_S) -> tuple[Any | None, int | None]:
    now = time.monotonic()
    if cache_key:
        cached = _hf_json_cache.get(cache_key)
        if cached is not None:
            value, ts = cached
            if now - ts < _HF_JSON_CACHE_TTL:
                return value, 200

    req = request.Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "lerobot-gui-wrapper",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read()
            data = json.loads(payload.decode("utf-8")) if payload else None
            if cache_key:
                _evict_if_full(_hf_json_cache)
                _hf_json_cache[cache_key] = (data, now)
            return data, int(getattr(resp, "status", 200))
    except error.HTTPError as exc:
        if exc.code == 404:
            return None, 404
        return None, int(exc.code)
    except (error.URLError, TimeoutError, ValueError):
        return None, None


def increment_dataset_name(name: str) -> str:
    match = re.search(r"^(.*?)(\d+)$", name)
    if not match:
        return f"{name}_1"

    prefix, number = match.groups()
    next_number = int(number) + 1
    return f"{prefix}{next_number}"


def _clean_repo_id(repo_id: str) -> str:
    return str(repo_id or "").strip().strip("/")


def _split_repo_id(repo_id: str) -> tuple[str, str] | None:
    clean_repo = _clean_repo_id(repo_id)
    if "/" not in clean_repo:
        return None
    owner, repo_name = clean_repo.split("/", 1)
    owner = owner.strip().strip("/")
    repo_name = repo_name.strip().strip("/")
    if not owner or not repo_name:
        return None
    return owner, repo_name


def _lookup_repo_exists_from_owner_list(repo_id: str, *, kind: str) -> bool | None:
    parsed = _split_repo_id(repo_id)
    if parsed is None:
        return None
    owner, _ = parsed

    rows: list[dict[str, Any]]
    error_text: str | None
    if kind == "dataset":
        rows, error_text = list_hf_datasets(owner, limit=200)
    else:
        rows, error_text = list_hf_models(owner, limit=200)
    if error_text is not None:
        return None

    clean_repo = _clean_repo_id(repo_id).lower()
    for row in rows:
        listed_repo_id = _clean_repo_id(str(row.get("repo_id", ""))).lower()
        if listed_repo_id == clean_repo:
            return True
    return False


def dataset_exists_on_hf(repo_id: str) -> bool | None:
    clean_repo = _clean_repo_id(repo_id)
    if not clean_repo:
        return None

    now = time.monotonic()
    cached = _hf_cache.get(clean_repo)
    if cached is not None:
        result, ts = cached
        if now - ts < _HF_CACHE_TTL:
            return result

    _, status = _hf_get_json(
        f"https://huggingface.co/api/datasets/{quote(clean_repo, safe='/')}",
        cache_key=f"dataset_exists:{clean_repo}",
    )
    if status == 200:
        result: bool | None = True
    elif status == 404:
        result = False
    else:
        result = _lookup_repo_exists_from_owner_list(clean_repo, kind="dataset")
    if result is None:
        return None

    _evict_if_full(_hf_cache)
    _hf_cache[clean_repo] = (result, now)
    return result


def model_exists_on_hf(repo_id: str) -> bool | None:
    clean_repo = _clean_repo_id(repo_id)
    if not clean_repo:
        return None

    now = time.monotonic()
    cached = _hf_model_cache.get(clean_repo)
    if cached is not None:
        result, ts = cached
        if now - ts < _HF_CACHE_TTL:
            return result

    _, status = _hf_get_json(
        f"https://huggingface.co/api/models/{quote(clean_repo, safe='/')}",
        cache_key=f"model_exists:{clean_repo}",
    )
    if status == 200:
        result: bool | None = True
    elif status == 404:
        result = False
    else:
        result = _lookup_repo_exists_from_owner_list(clean_repo, kind="model")
    if result is None:
        return None

    _evict_if_full(_hf_model_cache)
    _hf_model_cache[clean_repo] = (result, now)
    return result


def list_hf_datasets(owner: str, *, limit: int = 200) -> tuple[list[dict[str, Any]], str | None]:
    clean_owner = str(owner or "").strip().strip("/")
    if not clean_owner:
        return [], "Hugging Face owner is required."

    bounded_limit = _safe_limit(limit, max_limit=200)
    url = (
        "https://huggingface.co/api/datasets"
        f"?author={quote(clean_owner)}&limit={bounded_limit}&full=true"
    )
    payload, status = _hf_get_json(url, cache_key=f"list_datasets:{clean_owner}:{bounded_limit}")
    if payload is None:
        if status == 404:
            return [], None
        return [], "Unable to fetch datasets from Hugging Face."
    if not isinstance(payload, list):
        return [], "Unexpected Hugging Face response for dataset list."

    results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("id") or item.get("repoId") or "").strip().strip("/")
        if not repo_id:
            continue
        name = repo_name_from_repo_id(repo_id)
        repo_owner = repo_id.split("/", 1)[0] if "/" in repo_id else clean_owner
        results.append(
            {
                "repo_id": repo_id,
                "name": name,
                "owner": repo_owner,
                "private": bool(item.get("private", False)),
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "last_modified": item.get("lastModified"),
                "metadata": item,
            }
        )
    return results, None


def list_hf_models(owner: str, *, limit: int = 200) -> tuple[list[dict[str, Any]], str | None]:
    clean_owner = str(owner or "").strip().strip("/")
    if not clean_owner:
        return [], "Hugging Face owner is required."

    bounded_limit = _safe_limit(limit, max_limit=200)
    url = (
        "https://huggingface.co/api/models"
        f"?author={quote(clean_owner)}&limit={bounded_limit}&full=true"
    )
    payload, status = _hf_get_json(url, cache_key=f"list_models:{clean_owner}:{bounded_limit}")
    if payload is None:
        if status == 404:
            return [], None
        return [], "Unable to fetch models from Hugging Face."
    if not isinstance(payload, list):
        return [], "Unexpected Hugging Face response for model list."

    results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("id") or item.get("modelId") or "").strip().strip("/")
        if not repo_id:
            continue
        name = repo_name_from_repo_id(repo_id)
        repo_owner = repo_id.split("/", 1)[0] if "/" in repo_id else clean_owner
        results.append(
            {
                "repo_id": repo_id,
                "name": name,
                "owner": repo_owner,
                "private": bool(item.get("private", False)),
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "last_modified": item.get("lastModified"),
                "metadata": item,
            }
        )
    return results, None


def get_hf_dataset_info(repo_id: str) -> tuple[dict[str, Any] | None, str | None]:
    clean_repo_id = str(repo_id or "").strip().strip("/")
    if not clean_repo_id:
        return None, "Dataset repo id is required."

    payload, status = _hf_get_json(
        f"https://huggingface.co/api/datasets/{clean_repo_id}",
        cache_key=f"dataset_info:{clean_repo_id}",
    )
    if payload is None:
        if status == 404:
            return None, "Dataset not found on Hugging Face."
        return None, "Unable to fetch dataset metadata from Hugging Face."
    if not isinstance(payload, dict):
        return None, "Unexpected Hugging Face response for dataset metadata."
    return payload, None


def get_hf_model_info(repo_id: str) -> tuple[dict[str, Any] | None, str | None]:
    clean_repo_id = str(repo_id or "").strip().strip("/")
    if not clean_repo_id:
        return None, "Model repo id is required."

    payload, status = _hf_get_json(
        f"https://huggingface.co/api/models/{clean_repo_id}",
        cache_key=f"model_info:{clean_repo_id}",
    )
    if payload is None:
        if status == 404:
            return None, "Model not found on Hugging Face."
        return None, "Unable to fetch model metadata from Hugging Face."
    if not isinstance(payload, dict):
        return None, "Unexpected Hugging Face response for model metadata."
    return payload, None


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


def repo_name_only(name_or_repo_id: Any, *, owner: str = "") -> str:
    raw_value = str(name_or_repo_id or "").strip()
    if not raw_value:
        return ""

    had_slash = "/" in raw_value
    clean_value = raw_value.strip().strip("/")
    if not clean_value:
        return ""
    if "/" in clean_value:
        return clean_value.rsplit("/", 1)[-1].strip()

    clean_owner = str(owner or "").strip().strip("/")
    if had_slash and clean_owner and clean_value == clean_owner:
        return ""
    return clean_value


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
