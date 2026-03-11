from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_CAMERA_NAME_BLOCKLIST = {
    "width",
    "height",
    "fps",
    "shape",
    "dtype",
    "type",
    "index",
    "path",
    "device",
    "name",
    "mean",
    "std",
    "low",
    "high",
    "channels",
    "color_space",
    "normalization",
}
_POLICY_DISPLAY_ALIASES = {
    "pi0-fast": "Pi0-FAST",
    "pi0fast": "Pi0-FAST",
    "wall-x": "Wall-X",
    "wallx": "Wall-X",
    "x-vla": "X-VLA",
    "xvla": "X-VLA",
    "sarm": "SARM",
}
_BUILTIN_POLICY_PACKAGES = {"lerobot"}
_RTC_KEYWORDS = {"rtc", "real_time", "real_time_control"}
_RUNTIME_LABEL_TOKENS = {
    "envhub": "EnvHub",
    "isaaclab": "IsaacLab",
    "isaac_lab": "IsaacLab",
    "rtc": "RTC",
}


@dataclass(frozen=True)
class ModelMetadata:
    policy_family: str | None = None
    policy_class: str | None = None
    plugin_package: str | None = None
    robot_type: str | None = None
    motor_names: tuple[str, ...] = ()
    action_dim: int | None = None
    fps: float | None = None
    camera_keys: tuple[str, ...] = ()
    supports_rtc: bool | None = None
    normalization_present: bool | None = None
    normalization_stats: dict[str, Any] | None = None
    runtime_labels: tuple[str, ...] = ()
    metadata_source: str = ""
    source_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_family": self.policy_family,
            "policy_class": self.policy_class,
            "plugin_package": self.plugin_package,
            "robot_type": self.robot_type,
            "motor_names": list(self.motor_names),
            "action_dim": self.action_dim,
            "fps": self.fps,
            "camera_keys": list(self.camera_keys),
            "supports_rtc": self.supports_rtc,
            "normalization_present": self.normalization_present,
            "normalization_stats": self.normalization_stats,
            "runtime_labels": list(self.runtime_labels),
            "metadata_source": self.metadata_source,
            "source_files": list(self.source_files),
            "errors": list(self.errors),
        }


def _extract_top_level_fps(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("fps", "control_fps"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _extract_robot_type(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    robot_type = payload.get("robot_type")
    if isinstance(robot_type, str) and robot_type.strip():
        return robot_type.strip()
    for section_key in ("robot", "env", "config"):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        robot_type = section.get("robot_type") or section.get("type")
        if isinstance(robot_type, str) and robot_type.strip():
            return robot_type.strip()
    return None


def _extract_motor_info(payload: Any) -> tuple[list[str] | None, int | None]:
    if not isinstance(payload, dict):
        return None, None

    motor_names: list[str] | None = None
    action_dim: int | None = None

    for key in ("motor_names", "motors", "joint_names", "actuator_names"):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            motor_names = [str(item) for item in value]
            break

    for shapes_key in ("output_shapes", "action_space", "features"):
        shapes = payload.get(shapes_key)
        if not isinstance(shapes, dict):
            continue
        entry = shapes.get("action")
        if isinstance(entry, dict):
            shape = entry.get("shape") or entry.get("size")
            if isinstance(shape, list) and shape:
                try:
                    action_dim = int(shape[0])
                except (TypeError, ValueError):
                    action_dim = None
            elif isinstance(shape, int) and shape > 0:
                action_dim = shape
        elif isinstance(entry, list) and entry:
            try:
                action_dim = int(entry[0])
            except (TypeError, ValueError):
                action_dim = None
        if action_dim is not None:
            break

    return motor_names, action_dim


def _collect_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned} if cleaned else set()
    if isinstance(value, list):
        items: set[str] = set()
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.add(cleaned)
        return items
    return set()


def _looks_like_camera_name(name: str) -> bool:
    normalized = name.strip()
    if not normalized:
        return False
    if normalized.lower() in _CAMERA_NAME_BLOCKLIST:
        return False
    return _CAMERA_NAME_PATTERN.match(normalized) is not None


def _collect_camera_keys_from_json(payload: Any, out: set[str]) -> None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).lower()
            if key in {"camera_keys", "image_keys", "observation_image_keys", "observation_camera_keys"}:
                out.update({item for item in _collect_strings(value) if _looks_like_camera_name(item)})
            elif key in {"cameras", "images"} and isinstance(value, dict):
                out.update({str(name) for name in value.keys() if _looks_like_camera_name(str(name))})
            _collect_camera_keys_from_json(value, out)
        return

    if isinstance(payload, list):
        for item in payload:
            _collect_camera_keys_from_json(item, out)


def _extract_normalization_present(payload: Any) -> bool | None:
    if not isinstance(payload, dict):
        return None
    for stats_key in ("stats", "normalization_stats", "norm_stats"):
        stats = payload.get(stats_key)
        if not isinstance(stats, dict):
            continue
        if stats:
            return True
    return None


def _extract_normalization_stats(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for stats_key in ("stats", "normalization_stats", "norm_stats"):
        stats = payload.get(stats_key)
        if not isinstance(stats, dict):
            continue
        for item_key in ("observation.state", "action", "state"):
            item = stats.get(item_key)
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("min"), list) and isinstance(item.get("max"), list):
                return item
    return None


def _normalize_policy_family(raw_value: str) -> str:
    cleaned = str(raw_value or "").strip()
    lowered = (
        cleaned.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("/", "")
        .replace("\\", "")
    )
    for token, display in _POLICY_DISPLAY_ALIASES.items():
        if token.replace("-", "") in lowered:
            return display
    return cleaned


def _derive_plugin_package(candidate: str) -> str | None:
    cleaned = str(candidate or "").strip()
    if not cleaned or "." not in cleaned:
        return None
    root = cleaned.split(".", 1)[0]
    if not root or root in _BUILTIN_POLICY_PACKAGES:
        return None
    if root.startswith("transformers"):
        return None
    return root


def _extract_policy_fields(payload: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, None

    policy_family: str | None = None
    policy_class: str | None = None
    plugin_package: str | None = None

    for key in ("policy_family", "policy_type", "policy_name", "architecture", "model_type"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            policy_family = _normalize_policy_family(value)
            break

    for key in ("policy_class", "policy_cls", "class_name", "_class_name", "_target_", "target"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            policy_class = value.strip()
            plugin_package = _derive_plugin_package(policy_class)
            if policy_family is None:
                policy_family = _normalize_policy_family(policy_class.split(".")[-1].replace("Policy", ""))
            break

    if plugin_package is None:
        value = payload.get("plugin_package")
        if isinstance(value, str) and value.strip():
            plugin_package = value.strip()

    return policy_family, policy_class, plugin_package


def _extract_runtime_labels(payload: Any, labels: set[str]) -> bool | None:
    supports_rtc: bool | None = None
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).strip().lower()
            compact_key = key.replace("-", "_")
            for token, label in _RUNTIME_LABEL_TOKENS.items():
                if token in compact_key:
                    labels.add(label)
            if compact_key in _RTC_KEYWORDS:
                if isinstance(value, bool):
                    supports_rtc = value
                elif isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "yes", "1", "enabled"}:
                        supports_rtc = True
                    elif lowered in {"false", "no", "0", "disabled"}:
                        supports_rtc = False
            nested = _extract_runtime_labels(value, labels)
            if nested is not None:
                supports_rtc = nested
        return supports_rtc

    if isinstance(payload, list):
        for item in payload:
            nested = _extract_runtime_labels(item, labels)
            if nested is not None:
                supports_rtc = nested
        return supports_rtc

    if isinstance(payload, str):
        lowered = payload.strip().lower()
        for token, label in _RUNTIME_LABEL_TOKENS.items():
            if token in lowered:
                labels.add(label)
        if "real time" in lowered or lowered == "rtc":
            labels.add("RTC")
            return True
    return None


def extract_model_metadata(model_path: Path) -> ModelMetadata:
    if not model_path.exists() or not model_path.is_dir():
        return ModelMetadata(errors=(f"model path not readable: {model_path}",))

    try:
        json_files = sorted(path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    except PermissionError as exc:
        return ModelMetadata(errors=(f"cannot list model folder (permission denied): {exc}",))
    except OSError as exc:
        return ModelMetadata(errors=(f"cannot list model folder: {exc}",))

    if not json_files:
        return ModelMetadata(errors=("no JSON metadata files found in model payload",))

    policy_family: str | None = None
    policy_class: str | None = None
    plugin_package: str | None = None
    robot_type: str | None = None
    motor_names: list[str] | None = None
    action_dim: int | None = None
    fps: float | None = None
    camera_keys: set[str] = set()
    supports_rtc: bool | None = None
    normalization_present: bool | None = None
    normalization_stats: dict[str, Any] | None = None
    runtime_labels: set[str] = set()
    source_files: list[str] = []

    for json_path in json_files[:16]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        changed = False

        if fps is None:
            fps = _extract_top_level_fps(payload)
            changed = changed or fps is not None

        if robot_type is None:
            robot_type = _extract_robot_type(payload)
            changed = changed or robot_type is not None

        extracted_motor_names, extracted_action_dim = _extract_motor_info(payload)
        if motor_names is None and extracted_motor_names is not None:
            motor_names = extracted_motor_names
            changed = True
        if action_dim is None and extracted_action_dim is not None:
            action_dim = extracted_action_dim
            changed = True

        extracted_policy_family, extracted_policy_class, extracted_plugin_package = _extract_policy_fields(payload)
        if policy_family is None and extracted_policy_family is not None:
            policy_family = extracted_policy_family
            changed = True
        if policy_class is None and extracted_policy_class is not None:
            policy_class = extracted_policy_class
            changed = True
        if plugin_package is None and extracted_plugin_package is not None:
            plugin_package = extracted_plugin_package
            changed = True

        before_camera_count = len(camera_keys)
        _collect_camera_keys_from_json(payload, camera_keys)
        if len(camera_keys) > before_camera_count:
            changed = True

        extracted_normalization = _extract_normalization_present(payload)
        if normalization_present is None and extracted_normalization is not None:
            normalization_present = extracted_normalization
            changed = True
        extracted_stats = _extract_normalization_stats(payload)
        if normalization_stats is None and extracted_stats is not None:
            normalization_stats = extracted_stats
            changed = True

        before_label_count = len(runtime_labels)
        extracted_rtc = _extract_runtime_labels(payload, runtime_labels)
        if extracted_rtc is not None:
            supports_rtc = extracted_rtc
            changed = True
        if len(runtime_labels) > before_label_count:
            changed = True

        if changed and json_path.name not in source_files:
            source_files.append(json_path.name)

    if not source_files:
        return ModelMetadata(errors=("could not extract structured model metadata from JSON files",))

    if supports_rtc is None and "RTC" in runtime_labels:
        supports_rtc = True

    return ModelMetadata(
        policy_family=policy_family,
        policy_class=policy_class,
        plugin_package=plugin_package,
        robot_type=robot_type,
        motor_names=tuple(motor_names or ()),
        action_dim=action_dim,
        fps=fps,
        camera_keys=tuple(sorted(camera_keys)),
        supports_rtc=supports_rtc,
        normalization_present=normalization_present,
        normalization_stats=normalization_stats,
        runtime_labels=tuple(sorted(runtime_labels)),
        metadata_source=f"detected via: {', '.join(source_files[:4])}",
        source_files=tuple(source_files),
        errors=(),
    )


def format_model_metadata_summary(
    model_path: Path | None,
    *,
    deploy_payload: Path | None = None,
) -> str:
    if model_path is None or not model_path.exists() or not model_path.is_dir():
        return "No model selected."

    try:
        entries = sorted(model_path.iterdir(), key=lambda item: item.name.lower())
    except PermissionError as exc:
        return "\n".join(
            [
                f"Selected path: {model_path}",
                f"Access error: permission denied ({exc})",
                "Fix: grant read+execute permissions to this model folder, then refresh.",
            ]
        )
    except OSError as exc:
        return "\n".join(
            [
                f"Selected path: {model_path}",
                f"Access error: {exc}",
                "Fix: verify the path exists and is readable, then refresh.",
            ]
        )

    metadata = extract_model_metadata(deploy_payload or model_path)
    child_names = [item.name for item in entries[:8]]
    lines = [
        f"Selected path: {model_path}",
        f"Deploy payload: {deploy_payload or model_path}",
    ]
    if metadata.errors:
        lines.append(f"Model metadata: {metadata.errors[0]}")
    else:
        lines.append(f"Policy family/class: {metadata.policy_family or 'unknown'} / {metadata.policy_class or 'unknown'}")
        lines.append(f"Plugin package: {metadata.plugin_package or 'built-in / not declared'}")
        lines.append(
            f"Robot type: {metadata.robot_type or 'unknown'}  |  Action dim: {metadata.action_dim if metadata.action_dim is not None else 'unknown'}"
        )
        lines.append(
            f"FPS: {int(metadata.fps) if metadata.fps is not None and float(metadata.fps).is_integer() else metadata.fps or 'unknown'}"
            f"  |  Cameras: {list(metadata.camera_keys) if metadata.camera_keys else 'unknown'}"
        )
        lines.append(
            "Normalization: "
            + (
                "present"
                if metadata.normalization_present is True
                else "not detected" if metadata.normalization_present is False else "unknown"
            )
            + "  |  RTC: "
            + (
                "supported"
                if metadata.supports_rtc is True
                else "not declared" if metadata.supports_rtc is False else "unknown"
            )
        )
        if metadata.runtime_labels:
            lines.append(f"Runtime labels: {', '.join(metadata.runtime_labels)}")
        lines.append(f"Metadata source: {metadata.metadata_source}")
    lines.append(f"Contents: {', '.join(child_names) if child_names else '(empty)'}")
    return "\n".join(lines)
