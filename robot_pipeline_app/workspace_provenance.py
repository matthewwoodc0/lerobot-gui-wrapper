from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ROOT_PROVENANCE_FILE = ".lerobot_workspace_provenance.json"
_META_PROVENANCE_FILE = "meta/workspace_provenance.json"


def provenance_file_candidates(path: Path) -> tuple[Path, ...]:
    root = Path(path)
    return (
        root / _ROOT_PROVENANCE_FILE,
        root / _META_PROVENANCE_FILE,
    )


def read_workspace_provenance(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    root = Path(path)
    for candidate in provenance_file_candidates(root):
        try:
            if not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def write_workspace_provenance(
    path: Path | str,
    *,
    payload: dict[str, Any],
    prefer_meta_dir: bool = False,
) -> Path | None:
    root = Path(path)
    target = root / (_META_PROVENANCE_FILE if prefer_meta_dir else _ROOT_PROVENANCE_FILE)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return None
    return target


def build_hf_provenance_payload(
    *,
    repo_id: str,
    asset_kind: str,
    local_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = str(repo_id).strip().strip("/")
    return {
        "source": "huggingface",
        "repo_id": repo,
        "asset_kind": str(asset_kind).strip() or "asset",
        "local_path": str(Path(local_path)),
        "synced_at_iso": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
