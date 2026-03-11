from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .artifacts import _normalize_deploy_episode_outcomes, list_runs, normalize_deploy_result
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path
from .repo_utils import get_hf_dataset_info, get_hf_model_info, list_hf_datasets, list_hf_models
from .visualizer_metadata import looks_like_dataset_dir, visualizer_metadata_for_source

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MAX_VIDEOS_PER_SOURCE = 200
_MAX_SOURCES_PER_LIST = 500
_SKIP_DIR_NAMES = {"__pycache__", ".git"}


@dataclass(frozen=True)
class _VisualizerRefreshSnapshot:
    source: str
    deploy_root: str
    dataset_root: str
    model_root: str
    hf_owner: str


def _format_size_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(size, 0))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def _is_skippable_dir_name(name: str) -> bool:
    return not name or name.startswith(".") or name in _SKIP_DIR_NAMES


def _safe_list_dirs(path: Path) -> list[Path]:
    try:
        children = list(path.iterdir())
    except OSError:
        return []

    dirs: list[Path] = []
    for child in children:
        try:
            if child.is_dir() and not _is_skippable_dir_name(child.name):
                dirs.append(child)
        except OSError:
            continue
    return sorted(dirs)


def _looks_like_dataset_dir(path: Path) -> bool:
    return looks_like_dataset_dir(path)


def _discover_video_files(root: Path, *, limit: int = _MAX_VIDEOS_PER_SOURCE) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir() or limit <= 0:
        return []

    found: list[dict[str, Any]] = []
    for current_root, dirnames, filenames in os.walk(root_path, topdown=True):
        dirnames[:] = sorted(name for name in dirnames if not _is_skippable_dir_name(name))
        for filename in sorted(filenames):
            if len(found) >= limit:
                return found
            if Path(filename).suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            path = Path(current_root) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append(
                {
                    "path": path,
                    "relative_path": str(path.relative_to(root_path)),
                    "size_bytes": int(stat.st_size),
                    "size_text": _format_size_bytes(int(stat.st_size)),
                }
            )
    return found


def _deployment_insights(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    episodes = summary.get("episode_outcomes")
    if not isinstance(episodes, list):
        episodes = []

    parsed: list[dict[str, Any]] = []
    tag_set: set[str] = set()
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        ep = entry.get("episode")
        result = normalize_deploy_result(entry.get("result"))
        if result not in {"success", "failed"}:
            result = "unmarked"
        tags_raw = entry.get("tags")
        tags = [str(tag).strip() for tag in tags_raw] if isinstance(tags_raw, list) else []
        tags = [tag for tag in tags if tag]
        for tag in tags:
            tag_set.add(tag)
        parsed.append(
            {
                "episode": ep,
                "result": result,
                "tags": tags,
                "note": str(entry.get("note", "")).strip(),
            }
        )

    total_raw = summary.get("total_episodes")
    try:
        total = int(total_raw) if total_raw is not None else len(parsed)
    except (TypeError, ValueError):
        total = len(parsed)
    if total < len(parsed):
        total = len(parsed)
    success = int(summary.get("success_count") or 0)
    failed = int(summary.get("failed_count") or 0)
    rated = int(summary.get("rated_count") or 0)
    unmarked = summary.get("unmarked_count")
    if not isinstance(unmarked, int):
        unmarked = max(total - rated, 0)

    return {
        "enabled": bool(summary.get("enabled", True)),
        "total": total,
        "rated": rated,
        "success": success,
        "failed": failed,
        "unmarked": unmarked,
        "pending": unmarked,
        "tags": sorted(tag_set),
        "episodes": parsed,
        "overall_notes": str(metadata.get("deploy_notes_summary", "")).strip(),
    }


def _visualizer_source_row_values(source: dict[str, Any]) -> tuple[str, str]:
    scope_text = "huggingface" if str(source.get("scope", "local")) == "huggingface" else "local"
    return f"source - {scope_text}", str(source.get("name", "-"))


def _visualizer_insights_section(kind: str, resolved_metadata: dict[str, Any]) -> tuple[bool, str, list[tuple[Any, str, str, str]]]:
    if kind != "deployment":
        return False, "Deployment Insights", []
    insights = _deployment_insights(resolved_metadata)
    header = (
        f"Deployment Insights · Success {insights['success']} · Failed {insights['failed']} "
        f"· Unmarked {insights['unmarked']} · Tags {len(insights['tags'])}"
    )
    rows: list[tuple[Any, str, str, str]] = []
    for row in insights["episodes"]:
        rows.append(
            (
                row.get("episode"),
                str(row.get("result", "")).title(),
                ", ".join(row.get("tags", [])),
                row.get("note", ""),
            )
        )
    return True, header, rows


def _collect_videos_for_source(source: dict[str, Any], metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    scope = str(source.get("scope", "local")).strip() or "local"
    kind = str(source.get("kind", "")).strip()
    if scope == "local":
        source_path_raw = source.get("path")
        if source_path_raw:
            return _discover_video_files(Path(source_path_raw))
        return []
    if scope == "huggingface" and kind == "dataset":
        repo_id = str(source.get("repo_id", "")).strip()
        if repo_id and isinstance(metadata, dict):
            return _discover_hf_dataset_videos(repo_id, metadata)
    return []


def _resolve_deploy_dataset_path(dataset_repo_id: str, deploy_root: Path) -> Path | None:
    repo_id = str(dataset_repo_id or "").strip().strip("/")
    if not repo_id:
        return None

    owner = ""
    repo_name = repo_id
    if "/" in repo_id:
        owner, repo_name = repo_id.split("/", 1)

    candidates: list[Path] = []
    if owner:
        candidates.extend([deploy_root / owner / repo_name, deploy_root / repo_name])
        if deploy_root.name == owner:
            candidates.insert(0, deploy_root / repo_name)
    else:
        candidates.append(deploy_root / repo_name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return unique_candidates[0] if unique_candidates else None


def _collect_deploy_sources(config: dict[str, Any], deploy_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(deploy_root) if deploy_root is not None else get_deploy_data_dir(config)
    runs, _ = list_runs(config=config, limit=_MAX_SOURCES_PER_LIST)
    sources: list[dict[str, Any]] = []
    for item in runs:
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
        if str(item.get("mode", "")).strip().lower() != "deploy":
            continue
        run_path_raw = str(item.get("_run_path", "")).strip()
        if not run_path_raw:
            continue
        run_path = Path(run_path_raw)
        if not run_path.exists() or not run_path.is_dir():
            continue

        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        data_path = _resolve_deploy_dataset_path(dataset_repo_id, root)
        selected_path = data_path if data_path is not None else run_path
        name = str(item.get("run_id") or run_path.name)
        sources.append(
            {
                "id": f"deploy::{run_path}",
                "name": name,
                "path": selected_path,
                "run_path": run_path,
                "data_path": data_path,
                "metadata": item,
                "kind": "deployment",
                "scope": "local",
            }
        )
    return sources


def _collect_dataset_sources(config: dict[str, Any], data_root: Path | None = None) -> list[dict[str, Any]]:
    if data_root is not None:
        root = Path(data_root)
    else:
        record_root_raw = str(config.get("record_data_dir", "")).strip()
        if record_root_raw:
            root = Path(normalize_path(record_root_raw))
        else:
            lerobot_root_raw = str(config.get("lerobot_dir", "")).strip()
            root = (Path(normalize_path(lerobot_root_raw)) if lerobot_root_raw else Path.home() / "lerobot") / "data"
    if not root.exists() or not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _append_source(path: Path, name: str) -> None:
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            return
        try:
            canonical_path = path.resolve()
        except OSError:
            canonical_path = path
        if canonical_path in seen_paths:
            return
        seen_paths.add(canonical_path)
        sources.append(
            {
                "id": f"dataset::{canonical_path}",
                "name": name,
                "path": canonical_path,
                "metadata": {},
                "kind": "dataset",
                "scope": "local",
            }
        )

    if _looks_like_dataset_dir(root):
        _append_source(root, root.name or str(root))
        return sources

    for owner_dir in _safe_list_dirs(root):
        if _looks_like_dataset_dir(owner_dir):
            _append_source(owner_dir, owner_dir.name)
            continue
        for repo_dir in _safe_list_dirs(owner_dir):
            if len(sources) >= _MAX_SOURCES_PER_LIST:
                break
            if _looks_like_dataset_dir(repo_dir):
                _append_source(repo_dir, f"{owner_dir.name}/{repo_dir.name}")
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
    return sources


def _collect_model_sources(config: dict[str, Any], model_root: Path | None = None) -> list[dict[str, Any]]:
    if model_root is not None:
        root = Path(model_root)
    else:
        models_raw = str(config.get("trained_models_dir", "")).strip()
        root = Path(normalize_path(models_raw)) if models_raw else get_lerobot_dir(config) / "trained_models"
    if not root.exists() or not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for child in _safe_list_dirs(root):
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
        sources.append(
            {
                "id": f"model::{child}",
                "name": child.name,
                "path": child,
                "metadata": {},
                "kind": "model",
                "scope": "local",
            }
        )
    return sources


def _collect_hf_dataset_sources(owner: str) -> tuple[list[dict[str, Any]], str | None]:
    rows, error_text = list_hf_datasets(owner, limit=min(_MAX_SOURCES_PER_LIST, 200))
    sources: list[dict[str, Any]] = []
    for row in rows:
        repo_id = str(row.get("repo_id", "")).strip().strip("/")
        if repo_id:
            sources.append(
                {
                    "id": f"hf-dataset::{repo_id}",
                    "name": repo_id,
                    "repo_id": repo_id,
                    "metadata": row,
                    "kind": "dataset",
                    "scope": "huggingface",
                }
            )
    return sources, error_text


def _collect_hf_model_sources(owner: str) -> tuple[list[dict[str, Any]], str | None]:
    rows, error_text = list_hf_models(owner, limit=min(_MAX_SOURCES_PER_LIST, 200))
    sources: list[dict[str, Any]] = []
    for row in rows:
        repo_id = str(row.get("repo_id", "")).strip().strip("/")
        if repo_id:
            sources.append(
                {
                    "id": f"hf-model::{repo_id}",
                    "name": repo_id,
                    "repo_id": repo_id,
                    "metadata": row,
                    "kind": "model",
                    "scope": "huggingface",
                }
            )
    return sources, error_text


def _collect_sources_for_refresh(config: dict[str, Any], snapshot: _VisualizerRefreshSnapshot) -> tuple[list[dict[str, Any]], str | None, str]:
    source = str(snapshot.source).strip() or "deployments"
    if source == "deployments":
        return _collect_deploy_sources(config, deploy_root=Path(snapshot.deploy_root)), None, "deployment runs"

    owner = str(snapshot.hf_owner).strip()
    if source == "datasets":
        local_rows = _collect_dataset_sources(config, data_root=Path(snapshot.dataset_root))
        hf_rows: list[dict[str, Any]] = []
        error_text: str | None = None
        if owner:
            hf_rows, error_text = _collect_hf_dataset_sources(owner)
        return (local_rows + hf_rows)[:_MAX_SOURCES_PER_LIST], error_text, "dataset sources"

    local_rows = _collect_model_sources(config, model_root=Path(snapshot.model_root))
    hf_rows: list[dict[str, Any]] = []
    error_text = None
    if owner:
        hf_rows, error_text = _collect_hf_model_sources(owner)
    return (local_rows + hf_rows)[:_MAX_SOURCES_PER_LIST], error_text, "model sources"


def _local_path_overview(path: Path, *, limit: int = 2500) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "files_scanned": 0,
        "subdirs_scanned": 0,
        "video_files": 0,
        "total_size_bytes": 0,
        "sample_files": [],
        "truncated_scan": False,
    }
    if not path.exists() or not path.is_dir():
        return summary

    files_scanned = 0
    dirs_scanned = 0
    total_size_bytes = 0
    video_files = 0
    sample_files: list[str] = []
    truncated = False

    for current_root, dirnames, filenames in os.walk(path, topdown=True):
        dirnames[:] = sorted(name for name in dirnames if not _is_skippable_dir_name(name))
        dirs_scanned += len(dirnames)
        for filename in sorted(filenames):
            if files_scanned >= limit:
                truncated = True
                break
            files_scanned += 1
            file_path = Path(current_root) / filename
            try:
                total_size_bytes += int(file_path.stat().st_size)
            except OSError:
                pass
            if file_path.suffix.lower() in _VIDEO_EXTENSIONS:
                video_files += 1
            if len(sample_files) < 20:
                try:
                    sample_files.append(str(file_path.relative_to(path)))
                except Exception:
                    sample_files.append(str(file_path))
        if truncated:
            break

    summary["files_scanned"] = files_scanned
    summary["subdirs_scanned"] = dirs_scanned
    summary["video_files"] = video_files
    summary["total_size_bytes"] = total_size_bytes
    summary["sample_files"] = sample_files
    summary["truncated_scan"] = truncated
    return summary


def _discover_hf_dataset_videos(repo_id: str, metadata: dict[str, Any], *, limit: int = _MAX_VIDEOS_PER_SOURCE) -> list[dict[str, Any]]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        return []

    videos: list[dict[str, Any]] = []
    for entry in siblings:
        if len(videos) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        relative = str(entry.get("rfilename") or entry.get("path") or "").strip().strip("/")
        if not relative:
            continue
        if Path(relative).suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        size_raw = entry.get("size")
        try:
            size_text = _format_size_bytes(int(size_raw))
        except (TypeError, ValueError):
            size_text = "-"
        videos.append(
            {
                "relative_path": relative,
                "size_text": size_text,
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(relative, safe='/')}",
            }
        )
    return videos


def _open_path(path: Path | str) -> tuple[bool, str]:
    target = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
        elif os.name == "nt":
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception as exc:
        return False, f"Unable to open: {exc}"
    return True, "Opened"


def _build_selection_payload(source: dict[str, Any]) -> dict[str, Any]:
    metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}
    resolved_metadata: dict[str, Any] = dict(metadata)
    metadata_error: str | None = None

    if str(source.get("scope", "local")) == "huggingface":
        repo_id = str(source.get("repo_id", "")).strip()
        if str(source.get("kind", "")) == "dataset":
            resolved, metadata_error = get_hf_dataset_info(repo_id)
        else:
            resolved, metadata_error = get_hf_model_info(repo_id)
        resolved_metadata = resolved or {}

    source_path_raw = source.get("path")
    source_path = Path(source_path_raw) if source_path_raw else None
    scope = str(source.get("scope", "local")).strip() or "local"
    kind = str(source.get("kind", "source")).strip() or "source"
    repo_id = str(source.get("repo_id", "")).strip()

    meta_payload: dict[str, Any] = {
        "scope": scope,
        "kind": kind,
        "name": source.get("name"),
        "path": str(source_path) if source_path is not None else None,
        "repo_id": repo_id or None,
        "run_path": str(source.get("run_path")) if source.get("run_path") else None,
        "data_path": str(source.get("data_path")) if source.get("data_path") else None,
    }
    if scope == "local" and source_path is not None:
        meta_payload["local_overview"] = _local_path_overview(source_path)
    if metadata_error:
        meta_payload["metadata_error"] = metadata_error
    if resolved_metadata:
        meta_payload["metadata"] = resolved_metadata
    meta_payload["visualizer_metadata"] = visualizer_metadata_for_source(
        kind=kind,
        scope=scope,
        source_path=source_path,
        metadata=resolved_metadata if isinstance(resolved_metadata, dict) else {},
    )

    insights_visible, insights_header, insights_rows = _visualizer_insights_section(kind, resolved_metadata)
    videos = _collect_videos_for_source(source, resolved_metadata)
    return {
        "meta_payload": meta_payload,
        "insights_visible": insights_visible,
        "insights_header": insights_header,
        "insights_rows": insights_rows,
        "videos": videos,
    }
