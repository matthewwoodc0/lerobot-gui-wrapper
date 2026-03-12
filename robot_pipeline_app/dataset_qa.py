from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .visualizer_metadata import local_dataset_structure_summary
from .workspace_provenance import read_workspace_provenance


_DATASET_INFO_PATHS = ("meta/info.json", "meta.json")
_DATASET_STATS_PATHS = ("meta/stats.json", "stats.json")
_DATASET_EPISODE_PATHS = ("meta/episodes.jsonl", "meta/episodes.parquet", "episodes.jsonl", "episodes.parquet")
_DATASET_TASK_PATHS = ("meta/tasks.parquet", "meta/tasks.jsonl", "tasks.parquet", "tasks.jsonl")
_CRITICAL_LAYOUT_KEYS = ("meta", "data")


def _read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc})"
    return payload if isinstance(payload, dict) else None, None


def _safe_line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _expected_layout_entries(layout: str) -> list[str]:
    normalized = str(layout or "").strip().lower()
    if normalized == "v3":
        return [
            "meta/info.json",
            "meta/stats.json",
            "meta/episodes.jsonl|meta/episodes.parquet",
            "data/chunk-*/*.parquet",
            "videos/",
        ]
    if normalized == "v2.x":
        return [
            "meta/info.json",
            "meta/stats.json",
            "meta/episodes.jsonl|meta/episodes.parquet",
            "meta/tasks.jsonl|meta/tasks.parquet",
        ]
    return ["meta/", "data/", "videos/"]


def _missing_layout_entries(structure: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    meta = structure.get("meta") if isinstance(structure.get("meta"), dict) else {}
    data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
    videos = structure.get("videos") if isinstance(structure.get("videos"), dict) else {}

    if not meta.get("has_info"):
        missing.append("meta/info.json")
    if not meta.get("has_stats"):
        missing.append("meta/stats.json")
    if not meta.get("episodes_index") and not isinstance(structure.get("total_episodes"), int):
        missing.append("meta/episodes.jsonl|meta/episodes.parquet")
    if not data.get("present"):
        missing.append("data/")
    if not videos.get("present"):
        missing.append("videos/")
    return missing


def _local_dataset_counts(path: Path, structure: dict[str, Any]) -> dict[str, int | None]:
    counts = {
        "episodes": structure.get("total_episodes"),
        "frames": structure.get("total_frames"),
        "videos": structure.get("total_videos"),
    }
    if not isinstance(counts["episodes"], int):
        for relative in _DATASET_EPISODE_PATHS:
            candidate = path / relative
            if candidate.suffix == ".jsonl" and candidate.is_file():
                counts["episodes"] = _safe_line_count(candidate)
                break
        if not isinstance(counts["episodes"], int):
            meta = structure.get("meta") if isinstance(structure.get("meta"), dict) else {}
            if isinstance(meta.get("episode_file_count"), int) and int(meta["episode_file_count"]) > 0:
                counts["episodes"] = int(meta["episode_file_count"])
    if not isinstance(counts["videos"], int):
        videos = structure.get("videos") if isinstance(structure.get("videos"), dict) else {}
        if isinstance(videos.get("video_file_count"), int):
            counts["videos"] = int(videos["video_file_count"])
    if not isinstance(counts["frames"], int):
        data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
        if isinstance(data.get("parquet_file_count"), int):
            counts["frames"] = int(data["parquet_file_count"])
    return counts


def _hf_dataset_counts(structure: dict[str, Any]) -> dict[str, int | None]:
    return {
        "episodes": structure.get("total_episodes") if isinstance(structure.get("total_episodes"), int) else None,
        "frames": structure.get("total_frames") if isinstance(structure.get("total_frames"), int) else None,
        "videos": structure.get("total_videos") if isinstance(structure.get("total_videos"), int) else None,
    }


def _usable_status(structure: dict[str, Any], *, corrupt_artifacts: list[str], missing_artifacts: list[str]) -> tuple[bool, str]:
    recognized = bool(structure.get("recognized"))
    if not recognized:
        return False, "Unrecognized dataset layout."
    if corrupt_artifacts:
        return False, "Dataset metadata contains corrupt artifacts."
    critical_missing = [item for item in missing_artifacts if item.split("/", 1)[0] in _CRITICAL_LAYOUT_KEYS or item.startswith("meta/info")]
    if critical_missing:
        return False, "Dataset is missing critical layout artifacts."
    return True, "Dataset looks usable."


def build_local_dataset_qa(path: Path | str) -> dict[str, Any]:
    dataset_path = Path(path)
    structure = local_dataset_structure_summary(dataset_path)
    corrupt_artifacts: list[str] = []
    warnings: list[str] = []
    for relative in (*_DATASET_INFO_PATHS, *_DATASET_STATS_PATHS):
        candidate = dataset_path / relative
        if not candidate.is_file():
            continue
        _, error_text = _read_json_file(candidate)
        if error_text:
            corrupt_artifacts.append(f"{relative}: {error_text}")

    counts = _local_dataset_counts(dataset_path, structure)
    missing_artifacts = _missing_layout_entries(structure)
    usable, summary = _usable_status(
        structure,
        corrupt_artifacts=corrupt_artifacts,
        missing_artifacts=missing_artifacts,
    )
    if not counts["episodes"]:
        warnings.append("No episode count was inferred from metadata or episode indexes.")
    if not structure.get("camera_keys"):
        warnings.append("No camera keys were detected from dataset features or video layout.")

    return {
        "source": "local",
        "dataset_path": str(dataset_path),
        "usable": usable,
        "summary": summary,
        "layout": structure.get("layout"),
        "expected_layout": _expected_layout_entries(str(structure.get("layout", ""))),
        "counts": counts,
        "camera_keys": list(structure.get("camera_keys") or []),
        "action_dim": structure.get("action_dim"),
        "state_dim": structure.get("state_dim"),
        "fps": structure.get("fps"),
        "robot_type": structure.get("robot_type"),
        "missing_artifacts": missing_artifacts,
        "corrupt_artifacts": corrupt_artifacts,
        "warnings": warnings,
        "provenance": read_workspace_provenance(dataset_path),
        "structure": structure,
    }


def build_hf_dataset_qa(*, repo_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    siblings = metadata.get("siblings")
    paths: set[str] = set()
    if isinstance(siblings, list):
        for entry in siblings:
            if not isinstance(entry, dict):
                continue
            relative = str(entry.get("rfilename") or entry.get("path") or "").strip().strip("/")
            if relative:
                paths.add(relative)
    structure = metadata.get("visualizer_metadata")
    if not isinstance(structure, dict):
        from .visualizer_metadata import visualizer_metadata_for_source

        structure = visualizer_metadata_for_source(
            kind="dataset",
            scope="huggingface",
            source_path=None,
            metadata=metadata,
        )

    missing_artifacts = _missing_layout_entries(structure)
    counts = _hf_dataset_counts(structure)
    warnings: list[str] = []
    if not paths:
        warnings.append("HF metadata did not expose repo siblings.")
    usable, summary = _usable_status(
        structure,
        corrupt_artifacts=[],
        missing_artifacts=missing_artifacts,
    )
    return {
        "source": "huggingface",
        "repo_id": str(repo_id).strip().strip("/"),
        "usable": usable,
        "summary": summary,
        "layout": structure.get("layout"),
        "expected_layout": _expected_layout_entries(str(structure.get("layout", ""))),
        "counts": counts,
        "camera_keys": list(structure.get("camera_keys") or []),
        "action_dim": structure.get("action_dim"),
        "state_dim": structure.get("state_dim"),
        "fps": structure.get("fps"),
        "robot_type": structure.get("robot_type"),
        "missing_artifacts": missing_artifacts,
        "corrupt_artifacts": [],
        "warnings": warnings,
        "provenance": {"source": "huggingface", "repo_id": str(repo_id).strip().strip("/")},
        "structure": structure,
    }
