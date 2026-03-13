from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import _normalize_deploy_episode_outcomes
from .model_metadata import extract_model_metadata
from .utils_common import is_skippable_dir_name

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_DATASET_MARKER_FILES = {"episodes.parquet", "episodes.jsonl", "meta.json", "stats.json"}
_DATASET_INFO_PATHS = ("meta/info.json", "meta.json")
_DATASET_STATS_PATHS = ("meta/stats.json", "stats.json")
_DATASET_TASK_PATHS = ("meta/tasks.parquet", "meta/tasks.jsonl", "tasks.parquet", "tasks.jsonl")
_DATASET_EPISODE_PATHS = ("meta/episodes.parquet", "meta/episodes.jsonl", "episodes.parquet", "episodes.jsonl")
_DATASET_LAYOUT_PREFIXES = ("meta/", "data/", "videos/")
_DATASET_CAMERA_FEATURE_PREFIXES = ("observation.images.", "observation.image.")
def _safe_list_dirs(path: Path) -> list[Path]:
    try:
        children = list(path.iterdir())
    except OSError:
        return []

    dirs: list[Path] = []
    for child in children:
        try:
            if child.is_dir() and not is_skippable_dir_name(child.name):
                dirs.append(child)
        except OSError:
            continue
    return sorted(dirs)


def _safe_list_entries(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return []


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _unique_sorted_strings(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _feature_vector_dim(feature: Any) -> int | None:
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape") or feature.get("size")
    if isinstance(shape, list) and shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    if isinstance(shape, int) and shape > 0:
        return shape
    return None


def _dataset_camera_keys_from_features(features: Any) -> list[str]:
    if not isinstance(features, dict):
        return []
    cameras: list[str] = []
    for raw_key in features:
        key = str(raw_key).strip()
        if not key:
            continue
        for prefix in _DATASET_CAMERA_FEATURE_PREFIXES:
            if key.startswith(prefix):
                camera = key[len(prefix) :].strip()
                if camera:
                    cameras.append(camera)
                break
    return _unique_sorted_strings(cameras)


def _extract_dataset_info_summary(info_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = info_payload if isinstance(info_payload, dict) else {}
    features = payload.get("features")
    feature_keys = _unique_sorted_strings([str(key).strip() for key in features.keys()]) if isinstance(features, dict) else []

    fps_raw = payload.get("fps")
    fps: int | float | None
    if isinstance(fps_raw, (int, float)) and float(fps_raw) > 0:
        fps = int(fps_raw) if float(fps_raw).is_integer() else float(fps_raw)
    else:
        fps = None

    robot_type = payload.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type.strip():
        robot_type = None

    summary: dict[str, Any] = {
        "codebase_version": str(payload.get("codebase_version", "")).strip() or None,
        "robot_type": robot_type,
        "fps": fps,
        "camera_keys": _dataset_camera_keys_from_features(features),
        "feature_keys": feature_keys,
        "action_dim": _feature_vector_dim(features.get("action")) if isinstance(features, dict) else None,
        "state_dim": _feature_vector_dim(features.get("observation.state")) if isinstance(features, dict) else None,
        "total_episodes": None,
        "total_frames": None,
        "total_videos": None,
    }

    for field in ("total_episodes", "total_frames", "total_videos"):
        raw_value = payload.get(field)
        try:
            summary[field] = int(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            summary[field] = None
    return summary


def _count_matching_files(paths: list[Path], suffixes: tuple[str, ...]) -> int:
    total = 0
    for path in paths:
        try:
            if path.is_file() and path.suffix.lower() in suffixes:
                total += 1
        except OSError:
            continue
    return total


def _count_video_files(paths: list[Path]) -> int:
    return _count_matching_files(paths, tuple(_VIDEO_EXTENSIONS))


def local_dataset_structure_summary(path: Path) -> dict[str, Any]:
    meta_dir = path / "meta"
    data_dir = path / "data"
    videos_dir = path / "videos"

    info_path = next((path / rel for rel in _DATASET_INFO_PATHS if (path / rel).is_file()), None)
    stats_path = next((path / rel for rel in _DATASET_STATS_PATHS if (path / rel).is_file()), None)
    tasks_path = next((path / rel for rel in _DATASET_TASK_PATHS if (path / rel).is_file()), None)
    episodes_file_path = next((path / rel for rel in _DATASET_EPISODE_PATHS if (path / rel).is_file()), None)
    episodes_dir = meta_dir / "episodes"

    data_chunk_dirs = [child for child in _safe_list_dirs(data_dir) if child.name.lower().startswith("chunk-")]
    legacy_chunk_dirs = [child for child in _safe_list_dirs(path) if child.name.lower().startswith("chunk-")]
    active_data_dirs = data_chunk_dirs or legacy_chunk_dirs
    data_parquet_files = _count_matching_files(
        [entry for chunk_dir in active_data_dirs for entry in _safe_list_entries(chunk_dir)],
        (".parquet",),
    ) if active_data_dirs else 0
    if data_dir.exists() and data_dir.is_dir() and not active_data_dirs:
        data_parquet_files = _count_matching_files(_safe_list_entries(data_dir), (".parquet",))

    videos_present = videos_dir.exists() and videos_dir.is_dir()
    video_chunk_dirs: list[Path] = []
    video_files = 0
    video_layout = "none"
    camera_keys: list[str] = []
    if videos_present:
        first_level_dirs = _safe_list_dirs(videos_dir)
        chunk_like_dirs = [child for child in first_level_dirs if child.name.lower().startswith("chunk-")]
        if chunk_like_dirs and len(chunk_like_dirs) == len(first_level_dirs):
            video_layout = "shared"
            video_chunk_dirs = chunk_like_dirs
            video_files = sum(_count_video_files(_safe_list_entries(chunk_dir)) for chunk_dir in chunk_like_dirs)
        else:
            video_layout = "camera-keyed"
            camera_keys = [child.name for child in first_level_dirs]
            for camera_dir in first_level_dirs:
                child_dirs = _safe_list_dirs(camera_dir)
                nested_chunks = [child for child in child_dirs if child.name.lower().startswith("chunk-")]
                if nested_chunks:
                    video_chunk_dirs.extend(nested_chunks)
                    video_files += sum(_count_video_files(_safe_list_entries(chunk_dir)) for chunk_dir in nested_chunks)
                else:
                    video_files += _count_video_files(_safe_list_entries(camera_dir))
    else:
        video_files = _count_video_files(_safe_list_entries(path))

    episodes_entry_count = 0
    if episodes_file_path is not None:
        episodes_entry_count = 1
    elif episodes_dir.exists() and episodes_dir.is_dir():
        episode_dirs = _safe_list_dirs(episodes_dir)
        episode_files = (
            [entry for chunk_dir in episode_dirs for entry in _safe_list_entries(chunk_dir)]
            if episode_dirs
            else _safe_list_entries(episodes_dir)
        )
        episodes_entry_count = _count_matching_files(episode_files, (".parquet", ".jsonl"))

    info_summary = _extract_dataset_info_summary(_read_json_dict(info_path) if info_path is not None else None)
    if not info_summary["camera_keys"] and camera_keys:
        info_summary["camera_keys"] = _unique_sorted_strings(camera_keys)

    recognized = any(
        [
            info_path is not None,
            stats_path is not None,
            tasks_path is not None,
            episodes_file_path is not None,
            episodes_entry_count > 0,
            bool(active_data_dirs),
            bool(data_parquet_files),
            videos_present,
            video_files > 0,
        ]
    )

    layout = "unknown"
    version = str(info_summary.get("codebase_version") or "").lower()
    if version.startswith("v3"):
        layout = "v3"
    elif version.startswith("v2"):
        layout = "v2.x"
    elif meta_dir.exists() or data_dir.exists() or videos_present:
        layout = "v3" if (data_dir.exists() or meta_dir.exists()) else "legacy"
    elif active_data_dirs or any((path / marker).exists() for marker in _DATASET_MARKER_FILES):
        layout = "legacy"

    return {
        "kind": "dataset",
        "format": "LeRobotDataset" if recognized else "unknown",
        "layout": layout,
        "recognized": recognized,
        "metadata_source": "meta/info.json + local dataset folders" if info_path is not None else "local dataset folders",
        "codebase_version": info_summary.get("codebase_version"),
        "robot_type": info_summary.get("robot_type"),
        "fps": info_summary.get("fps"),
        "camera_keys": info_summary.get("camera_keys"),
        "feature_keys": info_summary.get("feature_keys"),
        "action_dim": info_summary.get("action_dim"),
        "state_dim": info_summary.get("state_dim"),
        "total_episodes": info_summary.get("total_episodes"),
        "total_frames": info_summary.get("total_frames"),
        "total_videos": info_summary.get("total_videos"),
        "meta": {
            "has_info": info_path is not None,
            "has_stats": stats_path is not None,
            "tasks_file": str(tasks_path.relative_to(path)) if tasks_path is not None else None,
            "episodes_index": (
                str(episodes_file_path.relative_to(path))
                if episodes_file_path is not None
                else "meta/episodes/" if episodes_entry_count > 0
                else None
            ),
            "episode_file_count": episodes_entry_count,
        },
        "data": {
            "present": (data_dir.exists() and data_dir.is_dir()) or bool(legacy_chunk_dirs),
            "chunk_count": len(active_data_dirs),
            "parquet_file_count": data_parquet_files,
        },
        "videos": {
            "present": videos_present or video_files > 0,
            "layout": video_layout,
            "camera_keys": _unique_sorted_strings(camera_keys or list(info_summary.get("camera_keys") or [])),
            "chunk_count": len(video_chunk_dirs),
            "video_file_count": video_files,
        },
    }


def _hf_dataset_structure_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    siblings = metadata.get("siblings")
    paths: list[str] = []
    if isinstance(siblings, list):
        for entry in siblings:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("rfilename") or entry.get("path") or "").strip().strip("/")
            if value:
                paths.append(value)

    path_set = set(paths)
    data_chunks = sorted({parts[1] for parts in (path.split("/", 2) for path in paths) if len(parts) > 1 and parts[0] == "data"})
    videos_parts = [path.split("/") for path in paths if path.startswith("videos/")]
    camera_keys = sorted({parts[1] for parts in videos_parts if len(parts) > 2 and not parts[1].startswith("chunk-")})
    video_chunks = sorted({"/".join(parts[:3]) for parts in videos_parts if len(parts) > 2 and parts[2].startswith("chunk-")})
    shared_video_chunks = sorted({"/".join(parts[:2]) for parts in videos_parts if len(parts) > 1 and parts[1].startswith("chunk-")})
    videos_layout = "camera-keyed" if camera_keys else ("shared" if shared_video_chunks else "none")

    recognized = any(
        path in path_set for path in (*_DATASET_INFO_PATHS, *_DATASET_STATS_PATHS, *_DATASET_TASK_PATHS, *_DATASET_EPISODE_PATHS)
    ) or any(path.startswith(_DATASET_LAYOUT_PREFIXES) for path in paths)

    info_summary = _extract_dataset_info_summary(None)
    if not info_summary["camera_keys"] and camera_keys:
        info_summary["camera_keys"] = camera_keys

    layout = "v3" if any(path.startswith(prefix) for path in paths for prefix in _DATASET_LAYOUT_PREFIXES) else "legacy"
    if any(path == "meta/episodes.jsonl" or path == "meta/tasks.jsonl" for path in paths):
        layout = "v2.x"

    return {
        "kind": "dataset",
        "format": "LeRobotDataset" if recognized else "unknown",
        "layout": layout if recognized else "unknown",
        "recognized": recognized,
        "metadata_source": "Hugging Face repo siblings",
        "codebase_version": info_summary.get("codebase_version"),
        "robot_type": info_summary.get("robot_type"),
        "fps": info_summary.get("fps"),
        "camera_keys": info_summary.get("camera_keys"),
        "feature_keys": info_summary.get("feature_keys"),
        "action_dim": info_summary.get("action_dim"),
        "state_dim": info_summary.get("state_dim"),
        "total_episodes": info_summary.get("total_episodes"),
        "total_frames": info_summary.get("total_frames"),
        "total_videos": info_summary.get("total_videos"),
        "meta": {
            "has_info": any(path in path_set for path in _DATASET_INFO_PATHS),
            "has_stats": any(path in path_set for path in _DATASET_STATS_PATHS),
            "tasks_file": next((path for path in _DATASET_TASK_PATHS if path in path_set), None),
            "episodes_index": next((path for path in _DATASET_EPISODE_PATHS if path in path_set), None),
            "episode_file_count": sum(1 for path in paths if path.startswith("meta/episodes/") and path.endswith((".parquet", ".jsonl"))),
        },
        "data": {
            "present": any(path.startswith("data/") for path in paths),
            "chunk_count": len(data_chunks),
            "parquet_file_count": sum(1 for path in paths if path.startswith("data/") and path.endswith(".parquet")),
        },
        "videos": {
            "present": any(path.startswith("videos/") for path in paths),
            "layout": videos_layout,
            "camera_keys": camera_keys,
            "chunk_count": len(video_chunks or shared_video_chunks),
            "video_file_count": sum(1 for path in paths if Path(path).suffix.lower() in _VIDEO_EXTENSIONS),
        },
    }


def _dataset_visualizer_metadata(source_path: Path | None, metadata: dict[str, Any], *, scope: str) -> dict[str, Any]:
    if scope == "local" and source_path is not None:
        return local_dataset_structure_summary(source_path)
    return _hf_dataset_structure_summary(metadata)


def _model_visualizer_metadata(source_path: Path | None, metadata: dict[str, Any], *, scope: str) -> dict[str, Any]:
    if scope == "local" and source_path is not None:
        model_metadata = extract_model_metadata(source_path).to_dict()
        model_metadata["kind"] = "model"
        model_metadata["recognized"] = not bool(model_metadata.get("errors"))
        return model_metadata

    siblings = metadata.get("siblings")
    source_files: list[str] = []
    if isinstance(siblings, list):
        for entry in siblings:
            if not isinstance(entry, dict):
                continue
            relative = str(entry.get("rfilename") or entry.get("path") or "").strip().strip("/")
            if relative and relative.endswith(".json"):
                source_files.append(relative)
    return {
        "kind": "model",
        "recognized": bool(source_files),
        "policy_family": None,
        "policy_class": None,
        "plugin_package": None,
        "robot_type": None,
        "motor_names": [],
        "action_dim": None,
        "fps": None,
        "camera_keys": [],
        "supports_rtc": None,
        "normalization_present": None,
        "normalization_stats": None,
        "runtime_labels": [],
        "metadata_source": "Hugging Face model API",
        "source_files": _unique_sorted_strings(source_files),
        "errors": [],
    }


def _deployment_visualizer_metadata(source_path: Path | None, metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    total = int(summary.get("total_episodes") or 0)
    rated = int(summary.get("rated_count") or 0)
    payload: dict[str, Any] = {
        "kind": "deployment",
        "run_id": str(metadata.get("run_id", "")).strip() or None,
        "status": str(metadata.get("status", "")).strip() or None,
        "dataset_repo_id": str(metadata.get("dataset_repo_id", "")).strip() or None,
        "model_path": str(metadata.get("model_path", "")).strip() or None,
        "deploy_notes_summary": str(metadata.get("deploy_notes_summary", "")).strip() or None,
        "deploy_episode_outcomes": summary,
        "insights": {
            "total": total,
            "rated": rated,
            "success": int(summary.get("success_count") or 0),
            "failed": int(summary.get("failed_count") or 0),
            "unmarked": max(total - rated, 0),
            "pending": max(total - rated, 0),
            "tags": list(summary.get("tags") or []),
        },
    }
    if source_path is not None:
        dataset_summary = local_dataset_structure_summary(source_path)
        if dataset_summary.get("recognized"):
            payload["eval_dataset"] = dataset_summary
    return payload


def visualizer_metadata_for_source(
    *,
    kind: str,
    scope: str,
    source_path: Path | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if kind == "dataset":
        return _dataset_visualizer_metadata(source_path, metadata, scope=scope)
    if kind == "model":
        return _model_visualizer_metadata(source_path, metadata, scope=scope)
    if kind == "deployment":
        return _deployment_visualizer_metadata(source_path, metadata)
    return {"kind": kind, "recognized": False}


def looks_like_dataset_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return bool(local_dataset_structure_summary(path).get("recognized"))
