from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .commands import build_lerobot_calibrate_command
from .deploy_diagnostics import find_nested_model_candidates, is_runnable_model_path
from .diagnostics import checks_to_events
from .model_metadata import format_model_metadata_summary
from .repo_utils import compose_repo_id, model_exists_on_hf, repo_name_only
from .runner import format_command
from .types import CheckResult, DiagnosticEvent


@dataclass(frozen=True)
class ModelBrowserNode:
    path: Path
    label: str
    kind: str
    tag: str
    children: tuple["ModelBrowserNode", ...] = ()


def first_model_payload_candidate(checks: list[CheckResult]) -> str | None:
    for _level, name, detail in checks:
        if name.strip().lower() != "model payload candidates":
            continue
        candidate = detail.split(",", 1)[0].strip()
        return candidate or None
    return None


def build_calibration_command(config: dict[str, Any]) -> str:
    return format_command(build_lerobot_calibrate_command(config, role="follower"))


def camera_rename_map_suggestion(checks: list[CheckResult]) -> str | None:
    for _level, name, detail in checks:
        if name.strip().lower() == "camera rename map suggestion":
            value = detail.strip()
            return value or None
    return None


def quick_actions_from_diagnostics(events: list[DiagnosticEvent]) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    action_labels = {
        "fix_eval_prefix": "Apply eval_ Prefix",
        "fix_model_payload": "Use Suggested Model Payload",
        "apply_rename_map": "Apply Camera Rename Map",
        "browse_follower_calib": "Browse Follower Calibration",
        "browse_leader_calib": "Browse Leader Calibration",
        "show_calib_cmd": "Show Recalibration Command",
    }
    actions: list[tuple[str, str]] = []
    context_by_action: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    def add(action_id: str, label: str, context: dict[str, Any] | None = None) -> None:
        if action_id in seen:
            return
        seen.add(action_id)
        actions.append((action_id, label))
        if context:
            context_by_action[action_id] = dict(context)

    for event in events:
        action_id = str(event.quick_action_id or "").strip()
        if not action_id:
            continue
        context = dict(event.context or {})
        if action_id == "fix_camera_fps":
            suggested_fps = context.get("suggested_fps")
            if isinstance(suggested_fps, int) and suggested_fps > 0:
                scoped_action = f"fix_camera_fps:{suggested_fps}"
                add(scoped_action, f"Set camera_fps -> {suggested_fps} Hz (match training)", context)
            continue
        label = action_labels.get(action_id)
        if label:
            add(action_id, label, context)

    return actions, context_by_action


def resolve_payload_path(path: Path) -> Path:
    if is_runnable_model_path(path):
        return path
    candidates = find_nested_model_candidates(path)
    return candidates[0] if candidates else path


def quick_actions_from_checks(checks: list[CheckResult]) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    return quick_actions_from_diagnostics(checks_to_events(checks))


def model_tree_node_kind(path: Path) -> tuple[str, str]:
    if is_runnable_model_path(path):
        return "Model", "model_root"

    candidates = find_nested_model_candidates(path, max_depth=3, limit=1)
    if candidates:
        if "checkpoint" in path.name.lower():
            return "Checkpoint -> model", "resolved"
        return "Contains model", "resolved"

    try:
        has_subdirs = any(item.is_dir() for item in path.iterdir())
    except OSError:
        has_subdirs = False

    if has_subdirs and "checkpoint" in path.name.lower():
        return "Checkpoint", "checkpoint"
    return ("Folder", "folder") if has_subdirs else ("", "folder")


def build_model_browser_tree(root_path: Path, *, max_depth: int = 4) -> list[ModelBrowserNode]:
    if not root_path.exists() or not root_path.is_dir():
        return []

    def build_node(path: Path, depth: int) -> ModelBrowserNode:
        kind, tag = model_tree_node_kind(path)
        children: list[ModelBrowserNode] = []
        if depth < max_depth:
            try:
                subdirs = sorted(
                    (item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")),
                    key=lambda item: item.name.lower(),
                )
            except OSError:
                subdirs = []
            children = [build_node(subdir, depth + 1) for subdir in subdirs]
        return ModelBrowserNode(path=path, label=path.name, kind=kind, tag=tag, children=tuple(children))

    try:
        top_dirs = sorted(
            (item for item in root_path.iterdir() if item.is_dir() and not item.name.startswith(".")),
            key=lambda item: item.name.lower(),
        )
    except OSError:
        return []
    return [build_node(item, 1) for item in top_dirs]


def summarize_model_info(model_path: Path | None) -> str:
    deploy_payload = resolve_payload_path(model_path)
    return format_model_metadata_summary(model_path, deploy_payload=deploy_payload)


def split_model_selection(root_path: Path, selected_path: Path) -> tuple[str, str]:
    try:
        rel = selected_path.relative_to(root_path)
    except ValueError:
        return selected_path.name, ""
    parts = rel.parts
    model_folder = parts[0] if parts else selected_path.name
    checkpoint = str(Path(*parts[1:])) if len(parts) > 1 else ""
    return model_folder, checkpoint


def model_hf_parity_detail(exists: bool | None, repo_id: str) -> tuple[str, str]:
    if exists is True:
        return "WARN", f"Remote model already exists: {repo_id}"
    if exists is False:
        return "PASS", f"Remote model not found yet: {repo_id}"
    return "WARN", f"Unable to confirm if remote model exists: {repo_id}"


def build_model_upload_request(
    *,
    local_model_raw: str,
    owner_raw: str,
    repo_name_raw: str,
) -> tuple[dict[str, Any] | None, str | None]:
    local_model = Path(str(local_model_raw or "").strip()).expanduser()
    if not local_model.exists() or not local_model.is_dir():
        return None, f"Local model folder not found: {local_model}"

    cleaned_repo_name = repo_name_only(repo_name_raw, owner=owner_raw)
    repo_id = compose_repo_id(owner_raw, cleaned_repo_name)
    if repo_id is None:
        return None, "Hugging Face owner and model name are required."

    hf_cli = shutil.which("huggingface-cli")
    if hf_cli is None:
        return None, "huggingface-cli not found in PATH."

    upload_cmd = [
        "huggingface-cli",
        "upload",
        repo_id,
        str(local_model),
        "--repo-type",
        "model",
    ]

    exists = model_exists_on_hf(repo_id)
    parity_level, parity_detail = model_hf_parity_detail(exists, repo_id)
    checks: list[CheckResult] = [
        ("PASS", "Local model folder", str(local_model)),
        ("PASS", "Target model repo", repo_id),
        ("PASS", "huggingface-cli", hf_cli),
        (parity_level, "Parity", parity_detail),
    ]
    return {
        "local_model": local_model,
        "repo_id": repo_id,
        "repo_name": cleaned_repo_name,
        "upload_cmd": upload_cmd,
        "remote_exists": exists,
        "parity_detail": parity_detail,
        "checks": checks,
    }, None
