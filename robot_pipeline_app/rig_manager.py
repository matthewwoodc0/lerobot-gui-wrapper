from __future__ import annotations

from typing import Any

from .config_store import normalize_config_without_prompts


_RIGS_KEY = "saved_rigs"
_ACTIVE_RIG_KEY = "active_rig_name"
_EXCLUDED_SNAPSHOT_KEYS = {
    _RIGS_KEY,
    _ACTIVE_RIG_KEY,
}


def _normalize_rig_name(value: Any) -> str:
    return str(value or "").strip()


def _snapshot_excluded_key(key: str) -> bool:
    normalized = str(key or "").strip()
    return normalized in _EXCLUDED_SNAPSHOT_KEYS or normalized.startswith("ui_")


def build_rig_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_config_without_prompts(config)
    snapshot = {
        key: value
        for key, value in normalized.items()
        if not _snapshot_excluded_key(str(key))
    }
    return snapshot


def list_named_rigs(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get(_RIGS_KEY)
    if not isinstance(raw, list):
        return []

    rigs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _normalize_rig_name(item.get("name"))
        if not name or name.lower() in seen:
            continue
        snapshot = item.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        seen.add(name.lower())
        rigs.append(
            {
                "name": name,
                "description": str(item.get("description", "")).strip(),
                "snapshot": dict(snapshot),
            }
        )
    return rigs


def active_rig_name(config: dict[str, Any]) -> str:
    return _normalize_rig_name(config.get(_ACTIVE_RIG_KEY))


def rig_names(config: dict[str, Any]) -> list[str]:
    return [str(item.get("name", "")).strip() for item in list_named_rigs(config)]


def save_named_rig(
    config: dict[str, Any],
    *,
    name: str,
    description: str = "",
) -> dict[str, Any]:
    rig_name = _normalize_rig_name(name)
    if not rig_name:
        raise ValueError("Rig name is required.")

    updated = dict(config)
    rigs = list_named_rigs(updated)
    snapshot = build_rig_snapshot(updated)
    replacement = {
        "name": rig_name,
        "description": str(description or "").strip(),
        "snapshot": snapshot,
    }

    matched = False
    next_rigs: list[dict[str, Any]] = []
    for item in rigs:
        if str(item.get("name", "")).strip().lower() == rig_name.lower():
            next_rigs.append(replacement)
            matched = True
            continue
        next_rigs.append(item)
    if not matched:
        next_rigs.append(replacement)

    updated[_RIGS_KEY] = next_rigs
    updated[_ACTIVE_RIG_KEY] = rig_name
    return updated


def delete_named_rig(config: dict[str, Any], *, name: str) -> dict[str, Any]:
    rig_name = _normalize_rig_name(name)
    updated = dict(config)
    next_rigs = [
        item
        for item in list_named_rigs(updated)
        if str(item.get("name", "")).strip().lower() != rig_name.lower()
    ]
    if next_rigs:
        updated[_RIGS_KEY] = next_rigs
    else:
        updated.pop(_RIGS_KEY, None)
    if active_rig_name(updated).lower() == rig_name.lower():
        updated.pop(_ACTIVE_RIG_KEY, None)
    return updated


def apply_named_rig(config: dict[str, Any], *, name: str) -> tuple[dict[str, Any] | None, str | None]:
    rig_name = _normalize_rig_name(name)
    if not rig_name:
        return None, "Rig name is required."

    rigs = list_named_rigs(config)
    selected = next(
        (item for item in rigs if str(item.get("name", "")).strip().lower() == rig_name.lower()),
        None,
    )
    if selected is None:
        return None, f"Unknown rig: {rig_name}"

    snapshot = selected.get("snapshot")
    if not isinstance(snapshot, dict):
        return None, f"Rig '{rig_name}' is missing a valid config snapshot."

    preserved = {
        key: value
        for key, value in config.items()
        if str(key).startswith("ui_")
    }
    updated = normalize_config_without_prompts(snapshot)
    updated.update(preserved)
    updated[_RIGS_KEY] = rigs
    updated[_ACTIVE_RIG_KEY] = str(selected.get("name", rig_name)).strip() or rig_name
    return updated, None
