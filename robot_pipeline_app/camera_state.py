from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .camera_schema import apply_camera_schema_entries_to_config, camera_schema_entries_for_editor, resolve_camera_schema

_MAX_REASONABLE_REPORTED_FPS = 240.0
DEFAULT_LIVE_PREVIEW_FPS_CAP = 15


def normalize_scan_limit(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 14
    return max(1, min(value, 64))


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def sanitize_reported_fps(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if fps <= 0 or fps > _MAX_REASONABLE_REPORTED_FPS:
        return None
    return fps


def compute_capture_fps(timestamps: list[float], reported_fps: float | None = None) -> float | None:
    if len(timestamps) < 2:
        return sanitize_reported_fps(reported_fps)

    elapsed_s = float(timestamps[-1] - timestamps[0])
    if elapsed_s <= 0:
        return sanitize_reported_fps(reported_fps)

    observed_fps = float(len(timestamps) - 1) / elapsed_s
    fallback_fps = sanitize_reported_fps(reported_fps)

    if fallback_fps is not None and observed_fps > (fallback_fps * 1.5):
        return fallback_fps
    return observed_fps


def normalize_live_preview_fps_cap(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = DEFAULT_LIVE_PREVIEW_FPS_CAP
    return max(1, min(value, 30))


def live_preview_interval_ms(fps_cap: int) -> int:
    cap = max(1, int(fps_cap))
    return max(25, int(round(1000.0 / float(cap))))


def camera_indices(config: dict[str, Any]) -> dict[str, int]:
    return {
        "laptop": int(config.get("camera_laptop_index", 0)),
        "phone": int(config.get("camera_phone_index", 1)),
    }


def camera_source_map(config: dict[str, Any]) -> dict[str, int | str]:
    schema = resolve_camera_schema(config)
    return {spec.name: spec.source for spec in schema.specs}


def choose_alternative_index(detected_indices: list[int], disallow_index: int) -> int | None:
    for idx in detected_indices:
        if idx != disallow_index:
            return idx
    return None


@dataclass(frozen=True)
class CameraRoleAssignment:
    ok: bool
    updated_config: dict[str, Any]
    messages: tuple[str, ...]


def assign_camera_role(
    *,
    config: dict[str, Any],
    detected_indices: list[int],
    detected_frame_sizes: dict[int, tuple[int, int]],
    role: str,
    index: int,
    fingerprint: str | None = None,
) -> CameraRoleAssignment:
    role_key = f"camera_{role}_index"
    other_role = "phone" if role == "laptop" else "laptop"
    other_key = f"camera_{other_role}_index"

    updated_config = dict(config)
    previous_role_index = int(updated_config.get(role_key, -1))
    previous_other_index = int(updated_config.get(other_key, -1))

    updated_config[role_key] = index

    if previous_other_index == index:
        fallback = choose_alternative_index(detected_indices, index)
        if fallback is None and previous_role_index != index:
            fallback = previous_role_index
        if fallback is None:
            return CameraRoleAssignment(
                ok=False,
                updated_config=dict(config),
                messages=("Could not assign role: laptop/phone must use two different ports.",),
            )
        updated_config[other_key] = fallback

    messages: list[str] = []
    detected_size = detected_frame_sizes.get(index)
    if detected_size is not None:
        width, height = detected_size
        messages.append(f"Mapped {role} camera {index} (detected frame {width}x{height}).")

    if fingerprint:
        updated_config[f"camera_{role}_fingerprint"] = fingerprint
        messages.append(f"Saved {role} camera fingerprint.")

    messages.append(
        f"Set roles: laptop={updated_config.get('camera_laptop_index')}, phone={updated_config.get('camera_phone_index')}"
    )
    return CameraRoleAssignment(ok=True, updated_config=updated_config, messages=tuple(messages))


def camera_mapping_summary(config: dict[str, Any]) -> str:
    mapping = camera_source_map(config)
    if not mapping:
        indices = camera_indices(config)
        return f"laptop={indices['laptop']} phone={indices['phone']}"
    return " ".join(f"{name}={source}" for name, source in mapping.items())


def camera_input_fps_summary(config: dict[str, Any], detected_input_fps: dict[int, float]) -> str:
    sources = camera_source_map(config)
    parts: list[str] = []
    for name, source in sources.items():
        fps = detected_input_fps.get(source) if isinstance(source, int) else None
        if fps is None:
            parts.append(f"{name}=n/a")
        else:
            parts.append(f"{name}={fps:.1f}")
    return "input fps " + " ".join(parts)


def assign_named_camera_source(
    *,
    config: dict[str, Any],
    detected_indices: list[int],
    detected_frame_sizes: dict[int, tuple[int, int]],
    camera_name: str,
    index: int,
    fingerprint: str | None = None,
) -> CameraRoleAssignment:
    entries = [entry for entry in camera_schema_entries_for_editor(config)]
    target_index = next((idx for idx, entry in enumerate(entries) if entry.name == camera_name), None)
    if target_index is None:
        return CameraRoleAssignment(
            ok=False,
            updated_config=dict(config),
            messages=(f"Could not assign camera '{camera_name}': it is not present in the runtime camera schema.",),
        )

    target_entry = entries[target_index]
    previous_source = target_entry.source
    conflicting_index = next(
        (
            idx
            for idx, entry in enumerate(entries)
            if idx != target_index and isinstance(entry.source, int) and int(entry.source) == index
        ),
        None,
    )

    updated_entries = [entry for entry in entries]
    messages: list[str] = []

    if conflicting_index is not None:
        used_indices = {
            int(entry.source)
            for idx, entry in enumerate(updated_entries)
            if idx not in {target_index, conflicting_index} and isinstance(entry.source, int)
        }
        fallback: int | None = None
        if isinstance(previous_source, int) and previous_source != index and previous_source not in used_indices:
            fallback = previous_source
        if fallback is None:
            fallback = choose_alternative_index(
                [candidate for candidate in detected_indices if candidate not in used_indices],
                index,
            )
        if fallback is None:
            return CameraRoleAssignment(
                ok=False,
                updated_config=dict(config),
                messages=(f"Could not assign camera '{camera_name}': no alternative detected source is available.",),
            )
        conflict_entry = updated_entries[conflicting_index]
        updated_entries[conflicting_index] = type(conflict_entry)(
            name=conflict_entry.name,
            source=fallback,
            camera_type=conflict_entry.camera_type,
            width=conflict_entry.width,
            height=conflict_entry.height,
            fps=conflict_entry.fps,
            warmup_s=conflict_entry.warmup_s,
        )
        messages.append(f"Reassigned camera '{conflict_entry.name}' -> {fallback} to keep camera sources unique.")

    updated_entries[target_index] = type(target_entry)(
        name=target_entry.name,
        source=index,
        camera_type=target_entry.camera_type,
        width=target_entry.width,
        height=target_entry.height,
        fps=target_entry.fps,
        warmup_s=target_entry.warmup_s,
    )

    updated_config = apply_camera_schema_entries_to_config(config, updated_entries)
    detected_size = detected_frame_sizes.get(index)
    if detected_size is not None:
        width, height = detected_size
        messages.append(f"Mapped {camera_name} camera {index} (detected frame {width}x{height}).")
    else:
        messages.append(f"Mapped {camera_name} camera {index}.")

    if fingerprint:
        updated_config[f"camera_{camera_name}_fingerprint"] = fingerprint
        messages.append(f"Saved {camera_name} camera fingerprint.")

    messages.append(f"Set camera mapping: {camera_mapping_summary(updated_config)}")
    return CameraRoleAssignment(ok=True, updated_config=updated_config, messages=tuple(messages))


@dataclass(frozen=True)
class CameraPreviewSnapshot:
    detected_indices: list[int]
    detected_frame_sizes: dict[int, tuple[int, int]]
    detected_input_fps: dict[int, float]
    status_text: str
    detected_ports_text: str
    scan_limit: str
    live_fps_cap: str
    live_enabled: bool
    pause_on_run: bool
    run_active: bool
    live_paused_for_run: bool


def export_camera_preview_state(
    *,
    detected_indices: list[int],
    detected_frame_sizes: dict[int, tuple[int, int]],
    detected_input_fps: dict[int, float],
    status_text: str,
    detected_ports_text: str,
    scan_limit: str,
    live_fps_cap: str,
    live_enabled: bool,
    pause_on_run: bool,
    run_active: bool,
    live_paused_for_run: bool,
) -> dict[str, Any]:
    return {
        "detected_indices": list(detected_indices),
        "detected_frame_sizes": dict(detected_frame_sizes),
        "detected_input_fps": dict(detected_input_fps),
        "status_text": str(status_text),
        "detected_ports_text": str(detected_ports_text),
        "scan_limit": str(scan_limit),
        "live_fps_cap": str(live_fps_cap),
        "live_enabled": bool(live_enabled),
        "pause_on_run": bool(pause_on_run),
        "run_active": bool(run_active),
        "live_paused_for_run": bool(live_paused_for_run),
    }


def restore_camera_preview_state(state: dict[str, Any] | None) -> CameraPreviewSnapshot:
    snapshot = dict(state or {})
    detected = [int(index) for index in snapshot.get("detected_indices", [])]
    frame_sizes = {
        int(index): (int(size[0]), int(size[1]))
        for index, size in dict(snapshot.get("detected_frame_sizes", {})).items()
        if isinstance(size, (tuple, list)) and len(size) == 2
    }
    input_fps = {
        int(index): float(fps)
        for index, fps in dict(snapshot.get("detected_input_fps", {})).items()
        if fps is not None
    }

    return CameraPreviewSnapshot(
        detected_indices=detected,
        detected_frame_sizes=frame_sizes,
        detected_input_fps=input_fps,
        status_text=str(snapshot.get("status_text", "Preview idle.")),
        detected_ports_text=str(snapshot.get("detected_ports_text", "Detected open camera ports: (scan to detect)")),
        scan_limit=str(snapshot.get("scan_limit", "14")),
        live_fps_cap=str(snapshot.get("live_fps_cap", DEFAULT_LIVE_PREVIEW_FPS_CAP)),
        live_enabled=bool(snapshot.get("live_enabled", False)),
        pause_on_run=bool(snapshot.get("pause_on_run", True)),
        run_active=bool(snapshot.get("run_active", False)),
        live_paused_for_run=bool(snapshot.get("live_paused_for_run", False)),
    )
