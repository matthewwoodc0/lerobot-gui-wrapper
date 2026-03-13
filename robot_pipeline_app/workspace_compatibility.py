from __future__ import annotations

from pathlib import Path
from typing import Any

from .camera_schema import resolve_camera_feature_mapping, runtime_camera_keys
from .model_metadata import extract_model_metadata


def _issue(level: str, name: str, detail: str) -> dict[str, str]:
    return {
        "level": level,
        "name": name,
        "detail": detail,
    }


def _dataset_keys(dataset_qa: dict[str, Any] | None) -> set[str]:
    if not isinstance(dataset_qa, dict):
        return set()
    return {str(item) for item in dataset_qa.get("camera_keys", []) if str(item).strip()}


def _dataset_value(dataset_qa: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(dataset_qa, dict):
        return None
    return dataset_qa.get(key)


def _model_value(model_meta: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(model_meta, dict):
        return None
    return model_meta.get(key)


def build_workspace_compatibility_summary(
    *,
    config: dict[str, Any],
    dataset_qa: dict[str, Any] | None = None,
    model_meta: dict[str, Any] | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    resolved_model_meta = dict(model_meta or {})
    if not resolved_model_meta and model_path:
        resolved_model_meta = extract_model_metadata(Path(model_path)).to_dict()

    issues: list[dict[str, str]] = []
    runtime_keys = runtime_camera_keys(config)
    model_keys = {str(item) for item in resolved_model_meta.get("camera_keys", []) if str(item).strip()}
    dataset_keys = _dataset_keys(dataset_qa)

    if model_keys:
        mapping, error_text = resolve_camera_feature_mapping(
            config=config,
            runtime_keys=runtime_keys,
            model_keys=model_keys,
        )
        if error_text:
            issues.append(_issue("FAIL", "Runtime/model camera keys", error_text))
        elif mapping and any(runtime != model for runtime, model in mapping.items()):
            issues.append(
                _issue(
                    "WARN",
                    "Runtime/model rename map",
                    f"Runtime cameras {sorted(runtime_keys)} require rename mapping to model keys {sorted(model_keys)}.",
                )
            )

    if dataset_keys and model_keys:
        if len(dataset_keys) != len(model_keys):
            issues.append(
                _issue(
                    "FAIL",
                    "Dataset/model camera keys",
                    f"Dataset exposes {sorted(dataset_keys)} but model expects {sorted(model_keys)}.",
                )
            )
        elif dataset_keys != model_keys:
            issues.append(
                _issue(
                    "WARN",
                    "Dataset/model camera keys",
                    f"Dataset cameras {sorted(dataset_keys)} differ from model cameras {sorted(model_keys)}; review rename mapping.",
                )
            )

    dataset_action_dim = _dataset_value(dataset_qa, "action_dim")
    model_action_dim = _model_value(resolved_model_meta, "action_dim")
    config_action_dim = config.get("follower_robot_action_dim")
    if isinstance(dataset_action_dim, int) and isinstance(model_action_dim, int) and dataset_action_dim != model_action_dim:
        issues.append(
            _issue(
                "FAIL",
                "Dataset/model action dim",
                f"Dataset action_dim={dataset_action_dim} but model action_dim={model_action_dim}.",
            )
        )
    if isinstance(config_action_dim, int) and isinstance(model_action_dim, int) and config_action_dim != model_action_dim:
        issues.append(
            _issue(
                "FAIL",
                "Robot/model action dim",
                f"Configured robot action_dim={config_action_dim} but model action_dim={model_action_dim}.",
            )
        )

    dataset_robot = str(_dataset_value(dataset_qa, "robot_type") or "").strip()
    model_robot = str(_model_value(resolved_model_meta, "robot_type") or "").strip()
    config_robot = str(config.get("follower_robot_type", "")).strip()
    if dataset_robot and model_robot and dataset_robot != model_robot:
        issues.append(
            _issue(
                "FAIL",
                "Dataset/model robot type",
                f"Dataset robot_type={dataset_robot} but model robot_type={model_robot}.",
            )
        )
    if config_robot and model_robot and config_robot != model_robot:
        issues.append(
            _issue(
                "WARN",
                "Runtime/model robot type",
                f"Configured robot_type={config_robot} while model robot_type={model_robot}.",
            )
        )

    dataset_fps = _dataset_value(dataset_qa, "fps")
    model_fps = _model_value(resolved_model_meta, "fps")
    config_fps = config.get("camera_fps")
    if isinstance(dataset_fps, (int, float)) and isinstance(model_fps, (int, float)) and float(dataset_fps) != float(model_fps):
        issues.append(
            _issue(
                "WARN",
                "Dataset/model FPS",
                f"Dataset FPS={dataset_fps} while model FPS={model_fps}.",
            )
        )
    if isinstance(config_fps, int) and isinstance(model_fps, (int, float)) and float(config_fps) != float(model_fps):
        issues.append(
            _issue(
                "WARN",
                "Runtime/model FPS",
                f"Configured camera_fps={config_fps} while model FPS={model_fps}.",
            )
        )

    normalization_present = resolved_model_meta.get("normalization_present")
    if normalization_present is False:
        issues.append(_issue("FAIL", "Model normalization", "Model metadata explicitly indicates normalization is missing."))
    elif normalization_present is None and resolved_model_meta:
        issues.append(_issue("WARN", "Model normalization", "Model metadata does not expose normalization statistics."))

    status = "PASS"
    if any(item["level"] == "FAIL" for item in issues):
        status = "FAIL"
    elif issues:
        status = "WARN"
    return {
        "status": status,
        "issues": issues,
    }
