from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config_store import normalize_path
from .utils_common import natural_sort_key

_CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_OBS_IMAGE_PREFIX = "observation.images."
_DEFAULT_SCHEMA_KEY = "camera_schema_json"
_DEFAULT_POLICY_MAP_KEY = "camera_policy_feature_map_json"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: int | str
    camera_type: str
    width: int
    height: int
    fps: int
    warmup_s: int


@dataclass(frozen=True)
class CameraSchemaResolution:
    specs: list[CameraSpec]
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class EditableCameraEntry:
    name: str
    source: int | str
    camera_type: str
    width: int
    height: int
    fps: int
    warmup_s: int


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _normalize_camera_name(raw_name: Any, fallback: str) -> str:
    candidate = str(raw_name or "").strip() or fallback
    if _CAMERA_NAME_PATTERN.match(candidate):
        return candidate
    return fallback


def _normalize_camera_source(raw_source: Any) -> int | str | None:
    if raw_source is None:
        return None
    if isinstance(raw_source, int):
        return raw_source
    text = str(raw_source).strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        try:
            return int(text)
        except ValueError:
            return text
    # Expand user/env vars so path-like camera sources are portable.
    if "/" in text or text.startswith("~") or "\\" in text or "$" in text:
        return normalize_path(text)
    return text


def _legacy_camera_specs(config: dict[str, Any], *, width: int, height: int, fps: int, warmup_s: int) -> list[CameraSpec]:
    laptop_name = _normalize_camera_name(config.get("camera_laptop_name"), "laptop")
    phone_name = _normalize_camera_name(config.get("camera_phone_name"), "phone")
    laptop_idx = int(config.get("camera_laptop_index", 0))
    phone_idx = int(config.get("camera_phone_index", 1))
    return [
        CameraSpec(
            name=laptop_name,
            source=laptop_idx,
            camera_type="opencv",
            width=width,
            height=height,
            fps=fps,
            warmup_s=warmup_s,
        ),
        CameraSpec(
            name=phone_name,
            source=phone_idx,
            camera_type="opencv",
            width=width,
            height=height,
            fps=fps,
            warmup_s=warmup_s,
        ),
    ]


def _parse_schema_payload(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    return json.loads(text)


def resolve_camera_schema(config: dict[str, Any]) -> CameraSchemaResolution:
    width_default = _positive_int(config.get("camera_default_width"), 640)
    height_default = _positive_int(config.get("camera_default_height"), 480)
    fps_default = _positive_int(config.get("camera_fps"), 30)
    warmup_default = _positive_int(config.get("camera_warmup_s"), 5)
    warnings: list[str] = []
    errors: list[str] = []

    raw_schema = config.get(_DEFAULT_SCHEMA_KEY)
    if raw_schema in (None, ""):
        return CameraSchemaResolution(
            specs=_legacy_camera_specs(
                config,
                width=width_default,
                height=height_default,
                fps=fps_default,
                warmup_s=warmup_default,
            ),
            warnings=warnings,
            errors=errors,
        )

    try:
        payload = _parse_schema_payload(raw_schema)
    except json.JSONDecodeError as exc:
        warnings.append(f"{_DEFAULT_SCHEMA_KEY} is not valid JSON ({exc}); using legacy camera_laptop/phone settings.")
        return CameraSchemaResolution(
            specs=_legacy_camera_specs(
                config,
                width=width_default,
                height=height_default,
                fps=fps_default,
                warmup_s=warmup_default,
            ),
            warnings=warnings,
            errors=errors,
        )

    entries: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        entries = [(str(name), spec) for name, spec in payload.items()]
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append(f"{_DEFAULT_SCHEMA_KEY}[{idx}] must be an object.")
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                errors.append(f"{_DEFAULT_SCHEMA_KEY}[{idx}] is missing 'name'.")
                continue
            entries.append((name, item))
    else:
        warnings.append(f"{_DEFAULT_SCHEMA_KEY} must be a JSON object or array; using legacy camera_laptop/phone settings.")
        return CameraSchemaResolution(
            specs=_legacy_camera_specs(
                config,
                width=width_default,
                height=height_default,
                fps=fps_default,
                warmup_s=warmup_default,
            ),
            warnings=warnings,
            errors=errors,
        )

    specs: list[CameraSpec] = []
    seen_names: set[str] = set()
    for idx, (raw_name, raw_spec) in enumerate(entries):
        name = _normalize_camera_name(raw_name, f"camera{idx + 1}")
        if name in seen_names:
            errors.append(f"Duplicate camera name '{name}' in {_DEFAULT_SCHEMA_KEY}.")
            continue
        if not isinstance(raw_spec, dict):
            errors.append(f"Camera '{raw_name}' must map to an object.")
            continue
        source_value = raw_spec.get("index_or_path", raw_spec.get("index", raw_spec.get("path")))
        source = _normalize_camera_source(source_value)
        if source is None:
            errors.append(f"Camera '{name}' is missing a valid 'index_or_path'.")
            continue

        camera_type = str(raw_spec.get("type", "opencv")).strip() or "opencv"
        width = _positive_int(raw_spec.get("width"), width_default)
        height = _positive_int(raw_spec.get("height"), height_default)
        fps = _positive_int(raw_spec.get("fps"), fps_default)
        warmup_s = _positive_int(raw_spec.get("warmup_s"), warmup_default)

        specs.append(
            CameraSpec(
                name=name,
                source=source,
                camera_type=camera_type,
                width=width,
                height=height,
                fps=fps,
                warmup_s=warmup_s,
            )
        )
        seen_names.add(name)

    if specs:
        return CameraSchemaResolution(specs=specs, warnings=warnings, errors=errors)

    warnings.append(f"{_DEFAULT_SCHEMA_KEY} did not produce any usable cameras; using legacy camera_laptop/phone settings.")
    return CameraSchemaResolution(
        specs=_legacy_camera_specs(
            config,
            width=width_default,
            height=height_default,
            fps=fps_default,
            warmup_s=warmup_default,
        ),
        warnings=warnings,
        errors=errors,
    )


def camera_schema_entries_for_editor(config: dict[str, Any]) -> list[EditableCameraEntry]:
    resolution = resolve_camera_schema(config)
    return [
        EditableCameraEntry(
            name=spec.name,
            source=spec.source,
            camera_type=spec.camera_type,
            width=int(spec.width),
            height=int(spec.height),
            fps=int(spec.fps),
            warmup_s=int(spec.warmup_s),
        )
        for spec in resolution.specs
    ]


def _unique_camera_name(name: str, seen: set[str], fallback_index: int) -> str:
    base = _normalize_camera_name(name, f"camera{fallback_index}")
    if base not in seen:
        return base
    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in seen and _CAMERA_NAME_PATTERN.match(candidate):
            return candidate
        suffix += 1


def normalize_camera_schema_entries(
    entries: list[EditableCameraEntry | dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[EditableCameraEntry]:
    width_default = _positive_int(config.get("camera_default_width"), 640)
    height_default = _positive_int(config.get("camera_default_height"), 480)
    fps_default = _positive_int(config.get("camera_fps"), 30)
    warmup_default = _positive_int(config.get("camera_warmup_s"), 5)

    normalized: list[EditableCameraEntry] = []
    seen_names: set[str] = set()
    for idx, raw_entry in enumerate(entries, start=1):
        if isinstance(raw_entry, EditableCameraEntry):
            payload = {
                "name": raw_entry.name,
                "source": raw_entry.source,
                "camera_type": raw_entry.camera_type,
                "width": raw_entry.width,
                "height": raw_entry.height,
                "fps": raw_entry.fps,
                "warmup_s": raw_entry.warmup_s,
            }
        else:
            payload = dict(raw_entry or {})

        name = _unique_camera_name(str(payload.get("name", "")).strip(), seen_names, idx)
        seen_names.add(name)

        source = _normalize_camera_source(payload.get("source", payload.get("index_or_path", idx - 1)))
        if source is None:
            source = idx - 1

        camera_type = str(payload.get("camera_type", payload.get("type", "opencv"))).strip() or "opencv"
        width = _positive_int(payload.get("width"), width_default)
        height = _positive_int(payload.get("height"), height_default)
        fps = _positive_int(payload.get("fps"), fps_default)
        warmup_s = _positive_int(payload.get("warmup_s"), warmup_default)

        normalized.append(
            EditableCameraEntry(
                name=name,
                source=source,
                camera_type=camera_type,
                width=width,
                height=height,
                fps=fps,
                warmup_s=warmup_s,
            )
        )
    return normalized


def format_camera_schema_json(entries: list[EditableCameraEntry | dict[str, Any]], *, config: dict[str, Any]) -> str:
    normalized = normalize_camera_schema_entries(entries, config=config)
    width_default = _positive_int(config.get("camera_default_width"), 640)
    height_default = _positive_int(config.get("camera_default_height"), 480)
    fps_default = _positive_int(config.get("camera_fps"), 30)
    warmup_default = _positive_int(config.get("camera_warmup_s"), 5)

    payload: dict[str, dict[str, Any]] = {}
    for entry in normalized:
        item: dict[str, Any] = {"index_or_path": entry.source}
        if entry.camera_type != "opencv":
            item["type"] = entry.camera_type
        if entry.width != width_default:
            item["width"] = entry.width
        if entry.height != height_default:
            item["height"] = entry.height
        if entry.fps != fps_default:
            item["fps"] = entry.fps
        if entry.warmup_s != warmup_default:
            item["warmup_s"] = entry.warmup_s
        payload[entry.name] = item
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def apply_camera_schema_entries_to_config(
    config: dict[str, Any],
    entries: list[EditableCameraEntry | dict[str, Any]],
) -> dict[str, Any]:
    normalized = normalize_camera_schema_entries(entries, config=config)
    updated = dict(config)
    updated["camera_schema_json"] = format_camera_schema_json(normalized, config=updated)

    if normalized:
        first = normalized[0]
        updated["camera_laptop_name"] = first.name
        if isinstance(first.source, int):
            updated["camera_laptop_index"] = int(first.source)
    if len(normalized) > 1:
        second = normalized[1]
        updated["camera_phone_name"] = second.name
        if isinstance(second.source, int):
            updated["camera_phone_index"] = int(second.source)
    return updated


def runtime_camera_keys(config: dict[str, Any]) -> set[str]:
    return {spec.name for spec in resolve_camera_schema(config).specs}


def normalize_camera_feature_key(raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    if value.startswith(_OBS_IMAGE_PREFIX):
        return value[len(_OBS_IMAGE_PREFIX) :].strip()
    return value


def _parse_policy_map(raw_map: Any) -> dict[str, str] | None:
    if raw_map in (None, ""):
        return None
    payload: Any
    if isinstance(raw_map, dict):
        payload = raw_map
    else:
        text = str(raw_map).strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    normalized: dict[str, str] = {}
    for raw_key, raw_val in payload.items():
        key = normalize_camera_feature_key(raw_key)
        val = normalize_camera_feature_key(raw_val)
        if not key or not val:
            continue
        normalized[key] = val
    return normalized or None


def resolve_camera_feature_mapping(
    *,
    config: dict[str, Any],
    runtime_keys: set[str],
    model_keys: set[str],
) -> tuple[dict[str, str] | None, str | None]:
    if not runtime_keys:
        return None, "runtime camera schema is empty"
    if not model_keys:
        return None, "model camera metadata did not expose camera/image keys"
    if len(runtime_keys) != len(model_keys):
        return None, (
            f"camera count mismatch: runtime has {len(runtime_keys)} keys {sorted(runtime_keys)}, "
            f"model expects {len(model_keys)} keys {sorted(model_keys)}"
        )

    explicit_map = _parse_policy_map(config.get(_DEFAULT_POLICY_MAP_KEY))
    if explicit_map is not None:
        resolved: dict[str, str] = {}
        for runtime_key in sorted(runtime_keys, key=natural_sort_key):
            mapped = explicit_map.get(runtime_key)
            if mapped is None:
                if runtime_key in model_keys:
                    mapped = runtime_key
                else:
                    return None, (
                        f"{_DEFAULT_POLICY_MAP_KEY} is missing an entry for runtime key '{runtime_key}'"
                    )
            if mapped not in model_keys:
                return None, (
                    f"{_DEFAULT_POLICY_MAP_KEY} maps '{runtime_key}' -> '{mapped}', "
                    f"but '{mapped}' is not present in model keys {sorted(model_keys)}"
                )
            resolved[runtime_key] = mapped
        if set(resolved.values()) != set(model_keys):
            return None, (
                f"{_DEFAULT_POLICY_MAP_KEY} does not cover all model keys; "
                f"mapped={sorted(set(resolved.values()))}, model={sorted(model_keys)}"
            )
        return resolved, None

    runtime_sorted = sorted(runtime_keys, key=natural_sort_key)
    model_sorted = sorted(model_keys, key=natural_sort_key)
    if runtime_sorted == model_sorted:
        return {name: name for name in runtime_sorted}, None

    resolved: dict[str, str] = {}
    used_models: set[str] = set()

    if "laptop" in runtime_keys and "camera1" in model_keys:
        resolved["laptop"] = "camera1"
        used_models.add("camera1")
    if "phone" in runtime_keys and "camera2" in model_keys and "phone" not in resolved:
        resolved["phone"] = "camera2"
        used_models.add("camera2")

    remaining_runtime = [name for name in runtime_sorted if name not in resolved]
    remaining_model = [name for name in model_sorted if name not in used_models]
    if len(remaining_runtime) != len(remaining_model):
        return None, "unable to automatically pair runtime/model camera keys"

    for runtime_name, model_name in zip(remaining_runtime, remaining_model):
        resolved[runtime_name] = model_name

    if set(resolved.values()) != set(model_keys):
        return None, "automatic camera key pairing did not produce a complete bijection"
    return resolved, None


def build_observation_rename_map(runtime_to_model: dict[str, str]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for runtime_name, model_name in sorted(runtime_to_model.items(), key=lambda item: natural_sort_key(item[0])):
        if runtime_name == model_name:
            continue
        rename_map[f"{_OBS_IMAGE_PREFIX}{runtime_name}"] = f"{_OBS_IMAGE_PREFIX}{model_name}"
    return rename_map


def format_observation_rename_map(runtime_to_model: dict[str, str]) -> str:
    return json.dumps(build_observation_rename_map(runtime_to_model), separators=(",", ":"), sort_keys=True)
