from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import quote
from typing import Callable
from typing import Any
from urllib import error, request

from .workspace_provenance import build_hf_provenance_payload, write_workspace_provenance


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


def _normalize_tags(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    for item in raw_tags:
        value = str(item).strip()
        if value:
            tags.append(value)
    return tags


def _hf_result_row(item: dict[str, Any], *, kind: str, default_owner: str = "") -> dict[str, Any] | None:
    repo_id = str(item.get("id") or item.get("repoId") or item.get("modelId") or "").strip().strip("/")
    if not repo_id:
        return None
    repo_owner = repo_id.split("/", 1)[0] if "/" in repo_id else default_owner
    return {
        "repo_id": repo_id,
        "name": repo_name_from_repo_id(repo_id),
        "owner": repo_owner,
        "private": bool(item.get("private", False)),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "last_modified": item.get("lastModified"),
        "tags": _normalize_tags(item.get("tags")),
        "pipeline_tag": str(item.get("pipeline_tag", "")).strip() or None,
        "task": str(item.get("task", "")).strip() or str(item.get("pipeline_tag", "")).strip() or None,
        "metadata": item,
        "kind": kind,
    }


def search_hf_assets(
    *,
    kind: str,
    owner: str = "",
    query: str = "",
    task: str = "",
    tags: list[str] | None = None,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], str | None]:
    normalized_kind = "model" if str(kind).strip().lower() == "model" else "dataset"
    bounded_limit = _safe_limit(limit, max_limit=200)
    params = [f"limit={bounded_limit}", "full=true"]
    clean_owner = str(owner or "").strip().strip("/")
    clean_query = str(query or "").strip()
    clean_task = str(task or "").strip()
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    if clean_owner:
        params.append(f"author={quote(clean_owner)}")
    if clean_query:
        params.append(f"search={quote(clean_query)}")
    if clean_task:
        params.append(f"filter={quote(clean_task)}")
    for tag in clean_tags:
        params.append(f"filter={quote(tag)}")

    endpoint = "models" if normalized_kind == "model" else "datasets"
    cache_key = ":".join([endpoint, clean_owner, clean_query, clean_task, ",".join(clean_tags), str(bounded_limit)])
    payload, status = _hf_get_json(
        f"https://huggingface.co/api/{endpoint}?{'&'.join(params)}",
        cache_key=cache_key,
    )
    if payload is None:
        if status == 404:
            return [], None
        return [], f"Unable to search Hugging Face {endpoint}."
    if not isinstance(payload, list):
        return [], f"Unexpected Hugging Face response for {endpoint} search."

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        row = _hf_result_row(item, kind=normalized_kind, default_owner=clean_owner)
        if row is not None:
            rows.append(row)
    return rows, None


def hf_sync_target_dir(config: dict[str, Any], *, repo_id: str, kind: str) -> Path:
    normalized_kind = "model" if str(kind).strip().lower() == "model" else "dataset"
    root_key = "trained_models_dir" if normalized_kind == "model" else "record_data_dir"
    root = Path(str(config.get(root_key, "")).strip() or ".").expanduser()
    clean_repo = _clean_repo_id(repo_id)
    if "/" in clean_repo:
        owner, name = clean_repo.split("/", 1)
        return root / owner / name
    return root / repo_name_from_repo_id(clean_repo)


def build_hf_sync_plan(config: dict[str, Any], *, repo_id: str, kind: str) -> dict[str, Any]:
    target_dir = hf_sync_target_dir(config, repo_id=repo_id, kind=kind)
    return {
        "repo_id": _clean_repo_id(repo_id),
        "kind": "model" if str(kind).strip().lower() == "model" else "dataset",
        "target_dir": str(target_dir),
        "target_exists": target_dir.exists(),
        "provenance_file": str(target_dir / ".lerobot_workspace_provenance.json"),
    }


def sync_hf_asset(
    config: dict[str, Any],
    *,
    repo_id: str,
    kind: str,
    downloader: Callable[..., str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    clean_repo = _clean_repo_id(repo_id)
    if not clean_repo:
        return None, "Hugging Face repo id is required."
    plan = build_hf_sync_plan(config, repo_id=clean_repo, kind=kind)
    target_dir = Path(plan["target_dir"])
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    download_fn = downloader
    if download_fn is None:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except Exception:
            return None, "huggingface_hub is unavailable; install it to sync HF assets."
        download_fn = snapshot_download

    try:
        try:
            resolved_path = download_fn(
                repo_id=clean_repo,
                repo_type="model" if plan["kind"] == "model" else "dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            resolved_path = download_fn(
                repo_id=clean_repo,
                repo_type="model" if plan["kind"] == "model" else "dataset",
                local_dir=str(target_dir),
            )
    except Exception as exc:
        return None, f"Unable to sync {clean_repo}: {exc}"

    info_fetcher = get_hf_model_info if plan["kind"] == "model" else get_hf_dataset_info
    metadata, _ = info_fetcher(clean_repo)
    provenance_path = write_workspace_provenance(
        Path(resolved_path),
        payload=build_hf_provenance_payload(
            repo_id=clean_repo,
            asset_kind=plan["kind"],
            local_path=resolved_path,
            metadata={"hf_metadata_summary": bool(metadata)},
        ),
        prefer_meta_dir=(plan["kind"] == "dataset"),
    )
    result = {
        **plan,
        "resolved_path": str(resolved_path),
        "provenance_path": str(provenance_path) if provenance_path is not None else None,
        "metadata_cached": bool(metadata),
    }
    return result, None


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


def next_available_dataset_name(
    base_name: str,
    hf_username: str,
    dataset_root: Path | None = None,
    max_attempts: int = 99,
) -> str:
    """Return the lowest-numbered variant of *base_name* that doesn't exist locally or on HF.

    Iteration scheme: ``base`` → ``base_2`` → ``base_3`` → …
    If both checks pass (no local folder, no HF repo), returns the candidate unchanged.
    Falls back to the original name if HF check is inconclusive (None) after max_attempts.
    """
    import re as _re

    # Strip any trailing _N suffix so we always start from the bare base.
    bare = _re.sub(r"_\d+$", "", base_name)

    def candidate(n: int) -> str:
        return bare if n == 1 else f"{bare}_{n}"

    def _exists_locally(name: str) -> bool:
        if dataset_root is None:
            return False
        return (dataset_root / name).exists()

    def _exists_on_hf(name: str) -> bool:
        if not hf_username:
            return False
        repo_id = f"{hf_username}/{name}"
        result = dataset_exists_on_hf(repo_id)
        return bool(result)  # treat None (unknown) as False — don't block on network issues

    for n in range(1, max_attempts + 1):
        name = candidate(n)
        if not _exists_locally(name) and not _exists_on_hf(name):
            return name

    # Exhausted attempts — return the base name and let LeRobot handle it.
    return bare


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
    return search_hf_assets(kind="dataset", owner=owner, limit=limit)


def list_hf_models(owner: str, *, limit: int = 200) -> tuple[list[dict[str, Any]], str | None]:
    return search_hf_assets(kind="model", owner=owner, limit=limit)


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
    # Use a placeholder owner so lerobot always gets a valid "owner/name" format,
    # even when no HuggingFace username has been configured.
    effective_username = str(username or "").strip() or "local_user"
    name = str(dataset_name_or_repo_id or "").strip().strip("/")
    if not name:
        return f"{effective_username}/dataset_1"
    if "/" in name:
        return name
    return f"{effective_username}/{name}"


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


def compose_repo_id(owner: str, name: str) -> str | None:
    """Return ``owner/name`` or ``None`` if either part is empty after cleaning.

    This is the shared canonical implementation used by the active GUI paths.
    """
    clean_owner = str(owner).strip().strip("/")
    clean_name = repo_name_only(name, owner=clean_owner)
    if not clean_owner or not clean_name:
        return None
    return f"{clean_owner}/{clean_name}"


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


# ---------------------------------------------------------------------------
# HF dataset tagging helpers (merged from hf_tagging.py)
# ---------------------------------------------------------------------------

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
