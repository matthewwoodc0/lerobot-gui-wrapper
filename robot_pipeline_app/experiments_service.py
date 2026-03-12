from __future__ import annotations

import ast
import json
import os
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import _normalize_deploy_episode_outcomes, coerce_diagnostic_events, list_runs
from .command_overrides import get_flag_value
from .deploy_diagnostics import find_nested_model_candidates, is_runnable_model_path
from .diagnostics import attribution_label, normalize_attribution
from .failure_inspector import first_failure_from_metadata
from .model_metadata import extract_model_metadata

_EXPERIMENT_RUN_MODES = {"train", "deploy", "sim_eval"}
_CHECKPOINT_STEP_PATTERN = re.compile(r"(?:checkpoint|ckpt|step|iter|epoch)[-_]?(\d+)", re.IGNORECASE)
_TRAILING_NUMBER_PATTERN = re.compile(r"(\d+)$")
_METRIC_KV_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_./%-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_THROUGHPUT_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*(samples/s|steps/s|it/s|iters/s|frames/s|hz)", re.IGNORECASE)
_STEP_PATTERN = re.compile(r"\bstep(?:_id)?\s*[:=]\s*(\d+)", re.IGNORECASE)
_WANDB_URL_PATTERN = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[^/\s?#]+)",
    re.IGNORECASE,
)
_WANDB_RUN_DIR_PATTERN = re.compile(r"run-\d{8}_\d{6}-(?P<run_id>[A-Za-z0-9]+)")
_TRAIN_METRIC_ALIASES = {
    "grdn": "grad_norm",
    "gradnorm": "grad_norm",
    "update_s": "update_s",
    "updt_s": "update_s",
    "data_s": "dataloading_s",
    "dataloading_s": "dataloading_s",
    "pc_success": "pc_success",
    "avg_sum_reward": "avg_sum_reward",
    "eval_s": "eval_s",
    "eval_ep_s": "eval_ep_s",
}
_INTERESTING_METRICS = {
    "loss",
    "grad_norm",
    "lr",
    "update_s",
    "dataloading_s",
    "pc_success",
    "avg_sum_reward",
    "eval_s",
    "eval_ep_s",
    "step",
}
_METRIC_FILE_NAMES = {
    "trainer_state.json",
    "eval_info.json",
    "metrics.json",
    "metrics.jsonl",
    "train_metrics.json",
    "summary.json",
    "results.json",
    "wandb-summary.json",
}
_WANDB_FILE_NAMES = {"wandb-summary.json", "wandb-metadata.json"}


def _normalize_metric_key(raw_key: str) -> str:
    lowered = (
        str(raw_key or "")
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_per_")
        .replace("-", "_")
        .replace(".", "_")
    )
    normalized = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    return _TRAIN_METRIC_ALIASES.get(normalized, normalized)


def _safe_json_read(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_output_dir(output_dir: str | Path | None, *, cwd: str | Path | None = None) -> str:
    raw = str(output_dir or "").strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if cwd:
        return str(Path(cwd).expanduser() / candidate)
    return str(candidate)


def _safe_read_recent_lines(run_path: Path | None, *, limit: int = 4000) -> list[str]:
    if run_path is None:
        return []
    log_path = Path(run_path) / "command.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if limit <= 0:
        return lines
    return lines[-limit:]


def _safe_walk_files(root: Path, *, max_depth: int = 5, limit: int = 200) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    files: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(files) < limit:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.lower(), reverse=True)
        except OSError:
            continue
        for child in children:
            try:
                if child.is_file():
                    files.append(child)
                    if len(files) >= limit:
                        break
                elif child.is_dir() and depth < max_depth and not child.name.startswith("."):
                    stack.append((child, depth + 1))
            except OSError:
                continue
    files.sort(key=lambda item: str(item).lower())
    return files


def _extract_step_from_path(path: Path) -> int | None:
    parts = path.parts
    best: int | None = None
    for idx, part in enumerate(parts):
        match = _CHECKPOINT_STEP_PATTERN.search(part)
        if match:
            value = int(match.group(1))
            best = value if best is None else max(best, value)
            continue
        if part.lower() in {"checkpoint", "checkpoints", "ckpt", "ckpts"} and idx + 1 < len(parts):
            trailing = _TRAILING_NUMBER_PATTERN.search(parts[idx + 1])
            if trailing:
                value = int(trailing.group(1))
                best = value if best is None else max(best, value)
    return best


def _classify_checkpoint_kind(path: Path, *, root: Path) -> str:
    relative = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    lowered = relative.lower()
    if "best" in lowered:
        return "best"
    if "last" in lowered or "latest" in lowered:
        return "latest"
    if "final" in lowered:
        return "final"
    if _extract_step_from_path(path) is not None:
        return "checkpoint"
    return "artifact"


def _nearest_train_config(path: Path, *, root: Path) -> Path | None:
    candidates = [
        path / "train_config.json",
        path.parent / "train_config.json",
        path.parent.parent / "train_config.json",
        root / "train_config.json",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def discover_checkpoint_artifacts(output_dir: str | Path | None) -> list[dict[str, Any]]:
    if output_dir is None:
        return []
    root = Path(output_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    payloads: list[Path] = []
    if is_runnable_model_path(root):
        payloads.append(root)
    payloads.extend(find_nested_model_candidates(root, max_depth=5, limit=120))

    seen: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for payload in payloads:
        key = str(payload.resolve()) if payload.exists() else str(payload)
        if key in seen:
            continue
        seen.add(key)
        metadata = extract_model_metadata(payload)
        train_config_path = _nearest_train_config(payload, root=root)
        try:
            relative = str(payload.relative_to(root))
        except ValueError:
            relative = payload.name
        artifacts.append(
            {
                "label": relative or payload.name,
                "path": str(payload),
                "relative_path": relative or payload.name,
                "kind": _classify_checkpoint_kind(payload, root=root),
                "step": _extract_step_from_path(payload),
                "is_deployable": True,
                "train_config_path": str(train_config_path) if train_config_path is not None else None,
                "policy_family": metadata.policy_family,
                "runtime_labels": list(metadata.runtime_labels),
                "metadata_errors": list(metadata.errors),
            }
        )

    if artifacts:
        artifacts.sort(
            key=lambda item: (
                {"best": 0, "latest": 1, "final": 2, "checkpoint": 3, "artifact": 4}.get(str(item.get("kind")), 9),
                -(int(item.get("step")) if isinstance(item.get("step"), int) else -1),
                str(item.get("relative_path", "")).lower(),
            )
        )
        return artifacts

    for candidate in _safe_walk_files(root, max_depth=4, limit=80):
        if candidate.name != "train_config.json":
            continue
        try:
            relative = str(candidate.relative_to(root))
        except ValueError:
            relative = candidate.name
        artifacts.append(
            {
                "label": relative,
                "path": str(candidate),
                "relative_path": relative,
                "kind": "config",
                "step": _extract_step_from_path(candidate),
                "is_deployable": False,
                "train_config_path": str(candidate),
                "policy_family": None,
                "runtime_labels": [],
                "metadata_errors": [],
            }
        )
    return artifacts


def _flatten_numeric_scalars(payload: Any, *, prefix: str = "", limit: int = 40) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if limit <= 0:
        return flattened
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_scalars(value, prefix=child_prefix, limit=limit - len(flattened)))
            if len(flattened) >= limit:
                break
        return flattened
    if isinstance(payload, list):
        return flattened
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flattened[prefix] = float(payload)
    return flattened


def _extract_metrics_from_python_dict_lines(lines: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in lines:
        stripped = str(line).strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            payload = ast.literal_eval(stripped)
        except Exception:
            continue
        metrics.update(_flatten_numeric_scalars(payload))
    return metrics


def _extract_metrics_from_output_lines(lines: Sequence[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    best_loss: float | None = None
    for line in lines:
        step_match = _STEP_PATTERN.search(line)
        if step_match:
            metrics["step"] = int(step_match.group(1))
        for raw_key, raw_value in _METRIC_KV_PATTERN.findall(line):
            key = _normalize_metric_key(raw_key)
            if key not in _INTERESTING_METRICS and "loss" not in key:
                continue
            value = float(raw_value)
            metrics[key] = value
            if key == "loss":
                best_loss = value if best_loss is None else min(best_loss, value)
        throughput_match = _THROUGHPUT_PATTERN.search(line)
        if throughput_match:
            metrics["throughput"] = float(throughput_match.group(1))
            metrics["throughput_unit"] = throughput_match.group(2).lower()
    if best_loss is not None:
        metrics["best_loss"] = best_loss

    python_dict_metrics = _extract_metrics_from_python_dict_lines(lines)
    if "avg_sum_reward" not in metrics:
        for key in ("overall.avg_sum_reward", "avg_sum_reward"):
            if key in python_dict_metrics:
                metrics["avg_sum_reward"] = python_dict_metrics[key]
                break
    if "pc_success" not in metrics:
        for key in ("overall.pc_success", "pc_success"):
            if key in python_dict_metrics:
                metrics["pc_success"] = python_dict_metrics[key]
                break
    if "eval_s" not in metrics:
        for key in ("overall.eval_s", "eval_s"):
            if key in python_dict_metrics:
                metrics["eval_s"] = python_dict_metrics[key]
                break
    return metrics


def _extract_training_artifact_metrics(output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    sources: list[str] = []
    for path in _safe_walk_files(output_dir, max_depth=5, limit=120):
        if path.name not in _METRIC_FILE_NAMES:
            continue
        try:
            relative = str(path.relative_to(output_dir))
        except ValueError:
            relative = path.name
        if path.name == "trainer_state.json":
            payload = _safe_json_read(path)
            if isinstance(payload, dict):
                global_step = payload.get("global_step")
                if isinstance(global_step, int):
                    metrics["step"] = global_step
                log_history = payload.get("log_history")
                if isinstance(log_history, list):
                    for item in reversed(log_history):
                        if not isinstance(item, dict):
                            continue
                        if "loss" in item and isinstance(item["loss"], (int, float)):
                            metrics.setdefault("loss", float(item["loss"]))
                            break
                    for item in reversed(log_history):
                        if not isinstance(item, dict):
                            continue
                        if "learning_rate" in item and isinstance(item["learning_rate"], (int, float)):
                            metrics.setdefault("lr", float(item["learning_rate"]))
                            break
                sources.append(relative)
            continue
        if path.name == "eval_info.json":
            payload = _safe_json_read(path)
            if isinstance(payload, dict):
                overall = payload.get("overall")
                if isinstance(overall, dict):
                    for key in ("avg_sum_reward", "pc_success", "eval_s", "eval_ep_s"):
                        value = overall.get(key)
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)
                    video_paths = overall.get("video_paths")
                    if isinstance(video_paths, list):
                        metrics["video_paths"] = [str(item) for item in video_paths]
                sources.append(relative)
            continue
        if path.name.endswith(".json"):
            payload = _safe_json_read(path)
            if payload is not None:
                flattened = _flatten_numeric_scalars(payload)
                for key, value in flattened.items():
                    normalized = _normalize_metric_key(key)
                    if normalized in _INTERESTING_METRICS and normalized not in metrics:
                        metrics[normalized] = value
                sources.append(relative)
            continue
        if path.name.endswith(".jsonl"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            parsed_rows: list[Any] = []
            for line in lines[-50:]:
                try:
                    parsed_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if parsed_rows:
                flattened = _flatten_numeric_scalars(parsed_rows[-1])
                for key, value in flattened.items():
                    normalized = _normalize_metric_key(key)
                    if normalized in _INTERESTING_METRICS and normalized not in metrics:
                        metrics[normalized] = value
                sources.append(relative)
    return metrics, sources


def extract_training_metrics(
    output_lines: Sequence[str] | str | None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    lines = output_lines.splitlines() if isinstance(output_lines, str) else [str(line) for line in output_lines or []]
    metrics = _extract_metrics_from_output_lines(lines)
    sources: list[str] = ["stdout"] if metrics else []

    output_path = Path(output_dir).expanduser() if output_dir else None
    if output_path is not None and output_path.exists() and output_path.is_dir():
        artifact_metrics, artifact_sources = _extract_training_artifact_metrics(output_path)
        for key, value in artifact_metrics.items():
            metrics.setdefault(key, value)
        if artifact_sources:
            sources.extend(artifact_sources)

    metrics["source"] = " + ".join(sources) if sources else "none"
    return metrics


def extract_sim_eval_metrics(
    output_lines: Sequence[str] | str | None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    lines = output_lines.splitlines() if isinstance(output_lines, str) else [str(line) for line in output_lines or []]
    metrics = _extract_metrics_from_output_lines(lines)
    sources: list[str] = ["stdout"] if metrics else []

    output_path = Path(output_dir).expanduser() if output_dir else None
    if output_path is not None and output_path.exists() and output_path.is_dir():
        eval_info_path = output_path / "eval_info.json"
        payload = _safe_json_read(eval_info_path)
        if isinstance(payload, dict):
            overall = payload.get("overall")
            if isinstance(overall, dict):
                for key in ("avg_sum_reward", "pc_success", "eval_s", "eval_ep_s"):
                    value = overall.get(key)
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
                video_paths = overall.get("video_paths")
                if isinstance(video_paths, list):
                    metrics["video_paths"] = [str(item) for item in video_paths]
            task_groups = [str(key) for key, value in payload.items() if key != "overall" and isinstance(value, dict)]
            if task_groups:
                metrics["task_groups"] = sorted(task_groups)
            sources.append("eval_info.json")

    metrics["source"] = " + ".join(sources) if sources else "none"
    return metrics


def _extract_wandb_numeric_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("loss", "train/loss", "eval/pc_success", "eval/avg_sum_reward", "pc_success", "avg_sum_reward", "_step"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            summary[str(key)] = value
    return summary


def extract_wandb_run_metadata(
    *,
    output_lines: Sequence[str] | str | None,
    output_dir: str | Path | None,
    enabled: bool,
    project: str,
    entity: str = "",
    job_name: str = "",
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "entity": str(entity or "").strip() or None,
        "project": str(project or "").strip() or None,
        "run_id": None,
        "run_name": str(job_name or "").strip() or None,
        "run_url": None,
        "summary": {},
        "source": "disabled" if not enabled else "config",
    }
    if not enabled:
        return info

    lines = output_lines.splitlines() if isinstance(output_lines, str) else [str(line) for line in output_lines or []]
    for line in lines:
        match = _WANDB_URL_PATTERN.search(line)
        if not match:
            continue
        info["entity"] = match.group("entity")
        info["project"] = match.group("project")
        info["run_id"] = match.group("run_id")
        info["run_url"] = match.group(0)
        info["source"] = "stdout"
        break

    output_path = Path(output_dir).expanduser() if output_dir else None
    if output_path is None:
        return info
    wandb_root = output_path / "wandb"
    if not wandb_root.exists() or not wandb_root.is_dir():
        return info

    for path in _safe_walk_files(wandb_root, max_depth=4, limit=80):
        if path.name not in _WANDB_FILE_NAMES:
            continue
        if path.name == "wandb-summary.json":
            payload = _safe_json_read(path)
            summary = _extract_wandb_numeric_summary(payload)
            if summary:
                info["summary"] = summary
                info["source"] = "local"
        elif path.name == "wandb-metadata.json":
            payload = _safe_json_read(path)
            if isinstance(payload, dict):
                for key in ("display_name", "name", "run_name"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        info["run_name"] = value.strip()
                        break
                for key in ("run_id", "runId", "id"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        info["run_id"] = value.strip()
                        break
                project_value = payload.get("project")
                if isinstance(project_value, str) and project_value.strip():
                    info["project"] = project_value.strip()
                entity_value = payload.get("entity")
                if isinstance(entity_value, str) and entity_value.strip():
                    info["entity"] = entity_value.strip()
                info["source"] = "local"

        for part in path.parts:
            match = _WANDB_RUN_DIR_PATTERN.fullmatch(part)
            if match:
                info["run_id"] = info.get("run_id") or match.group("run_id")

    if not info.get("run_url") and info.get("entity") and info.get("project") and info.get("run_id"):
        info["run_url"] = (
            f"https://wandb.ai/{info['entity']}/{info['project']}/runs/{info['run_id']}"
        )
    return info


def _wandb_credentials_available() -> bool:
    api_key = str(os.environ.get("WANDB_API_KEY", "")).strip()
    if api_key:
        return True
    netrc_path = Path.home() / ".netrc"
    try:
        text = netrc_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "wandb.ai" in text or "api.wandb.ai" in text


def fetch_wandb_remote_snapshot(wandb_info: Mapping[str, Any]) -> dict[str, Any]:
    entity = str(wandb_info.get("entity", "")).strip()
    project = str(wandb_info.get("project", "")).strip()
    run_id = str(wandb_info.get("run_id", "")).strip()
    if not entity or not project or not run_id or not _wandb_credentials_available():
        return {}
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception:
        return {}

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        return {"remote_error": str(exc)}

    summary_payload = dict(run.summary) if getattr(run, "summary", None) is not None else {}
    config_payload = dict(run.config) if getattr(run, "config", None) is not None else {}
    return {
        "remote_summary": _extract_wandb_numeric_summary(summary_payload),
        "remote_config": config_payload,
        "run_name": getattr(run, "name", None) or wandb_info.get("run_name"),
        "run_url": getattr(run, "url", None) or wandb_info.get("run_url"),
    }


def format_training_metrics_summary(metrics: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if isinstance(metrics.get("step"), (int, float)):
        parts.append(f"step {int(metrics['step'])}")
    if isinstance(metrics.get("loss"), (int, float)):
        parts.append(f"loss {float(metrics['loss']):.4f}")
    if isinstance(metrics.get("throughput"), (int, float)):
        unit = str(metrics.get("throughput_unit", "it/s"))
        parts.append(f"{float(metrics['throughput']):.2f} {unit}")
    elif isinstance(metrics.get("update_s"), (int, float)):
        parts.append(f"update {float(metrics['update_s']):.3f}s")
    if isinstance(metrics.get("pc_success"), (int, float)):
        parts.append(f"eval success {float(metrics['pc_success']) * 100:.1f}%")
    if isinstance(metrics.get("avg_sum_reward"), (int, float)):
        parts.append(f"eval reward {float(metrics['avg_sum_reward']):.3f}")
    return " | ".join(parts) if parts else "No parsed training metrics yet."


def extract_deploy_analytics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    total = int(summary.get("total_episodes") or 0)
    success = int(summary.get("success_count") or 0)
    failed = int(summary.get("failed_count") or 0)
    unmarked_raw = summary.get("unmarked_count")
    unmarked = int(unmarked_raw) if isinstance(unmarked_raw, int) else max(total - success - failed, 0)

    category_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    for event in coerce_diagnostic_events(metadata.get("runtime_diagnostics")):
        category_counts[normalize_attribution(event.attribution)] += 1
        if event.code:
            code_counts[event.code] += 1
    first_failure = first_failure_from_metadata(dict(metadata))
    if first_failure is not None and not category_counts:
        category_counts[normalize_attribution(first_failure.attribution)] += 1
        if first_failure.code:
            code_counts[first_failure.code] += 1

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "unmarked": unmarked,
        "success_rate": (success / total) if total > 0 else None,
        "tags": list(summary.get("tags") or []),
        "notes": str(metadata.get("deploy_notes_summary", "")).strip(),
        "failure_categories": [
            {"category": category, "label": attribution_label(category), "count": count}
            for category, count in sorted(category_counts.items())
        ],
        "failure_codes": [
            {"code": code, "count": count}
            for code, count in sorted(code_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def build_train_metadata_extra(
    *,
    context: Mapping[str, Any],
    output_lines: Sequence[str] | str,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = str(context.get("output_dir", "")).strip()
    resolved_output_dir = _resolve_output_dir(output_dir, cwd=cwd)
    wandb_project = str(context.get("wandb_project", "")).strip()
    wandb_entity = str(context.get("wandb_entity", "")).strip()
    wandb_enabled = bool(context.get("wandb_enabled"))
    job_name = str(context.get("job_name", "")).strip()
    return {
        "policy_type": str(context.get("policy_type", "")).strip() or None,
        "output_dir": output_dir or None,
        "output_dir_resolved": resolved_output_dir or None,
        "device": str(context.get("device", "")).strip() or None,
        "job_name": job_name or None,
        "resume_from": str(context.get("resume_from", "")).strip() or None,
        "wandb": extract_wandb_run_metadata(
            output_lines=output_lines,
            output_dir=resolved_output_dir or output_dir,
            enabled=wandb_enabled,
            project=wandb_project,
            entity=wandb_entity,
            job_name=job_name,
        ),
        "train_metrics": extract_training_metrics(output_lines, output_dir=resolved_output_dir or output_dir),
        "checkpoint_artifacts": discover_checkpoint_artifacts(resolved_output_dir or output_dir),
    }


def build_sim_eval_metadata_extra(
    *,
    context: Mapping[str, Any],
    output_lines: Sequence[str] | str,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = str(context.get("output_dir", "")).strip()
    resolved_output_dir = _resolve_output_dir(output_dir, cwd=cwd)
    model_path = str(context.get("model_path", "")).strip()
    payload = {
        "output_dir": output_dir or None,
        "output_dir_resolved": resolved_output_dir or None,
        "device": str(context.get("device", "")).strip() or None,
        "job_name": str(context.get("job_name", "")).strip() or None,
        "policy_type": str(context.get("policy_type", "")).strip() or None,
        "sim_eval": {
            "env_type": str(context.get("env_type", "")).strip() or None,
            "task": str(context.get("task", "")).strip() or None,
            "benchmark": str(context.get("benchmark", "")).strip() or None,
            "episodes": context.get("episodes"),
            "batch_size": context.get("batch_size"),
            "seed": context.get("seed"),
        },
        "sim_eval_metrics": extract_sim_eval_metrics(output_lines, output_dir=resolved_output_dir or output_dir),
    }
    if model_path:
        payload["model_path"] = model_path
    return payload


def _command_argv_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
    command_argv = metadata.get("command_argv")
    if isinstance(command_argv, list):
        return [str(part) for part in command_argv if str(part)]
    command_text = str(metadata.get("command", "")).strip()
    if not command_text:
        return []
    try:
        return shlex.split(command_text)
    except ValueError:
        return []


def _notes_and_tags(metadata: Mapping[str, Any]) -> tuple[str, list[str]]:
    notes = str(metadata.get("deploy_notes_summary", "")).strip()
    tags: list[str] = []
    summary = _normalize_deploy_episode_outcomes(metadata.get("deploy_episode_outcomes"))
    raw_tags = summary.get("tags")
    if isinstance(raw_tags, list):
        tags = [str(tag) for tag in raw_tags if str(tag).strip()]
    return notes, tags


def _run_output_location(metadata: Mapping[str, Any], mode: str) -> str:
    if mode in {"train", "sim_eval"}:
        value = str(metadata.get("output_dir_resolved", "")).strip() or str(metadata.get("output_dir", "")).strip()
        if value:
            return value
    return str(metadata.get("_run_path", "")).strip()


def _model_checkpoint_label(model_path: str) -> str:
    if not model_path:
        return "-"
    candidate = Path(model_path)
    if candidate.name:
        return candidate.name
    return model_path


def _build_record(metadata: dict[str, Any], *, include_wandb_remote: bool) -> dict[str, Any]:
    mode = str(metadata.get("mode", "")).strip().lower()
    command_argv = _command_argv_from_metadata(metadata)
    notes, tags = _notes_and_tags(metadata)
    model_path = str(metadata.get("model_path", "")).strip()
    policy_type = str(metadata.get("policy_type", "")).strip()
    device = str(metadata.get("device", "")).strip()

    if not policy_type:
        policy_type = str(get_flag_value(command_argv, "policy.type") or "").strip()

    if not policy_type and model_path:
        model_metadata = extract_model_metadata(Path(model_path))
        policy_type = model_metadata.policy_family or ""

    if not device:
        if mode == "sim_eval":
            device = str(metadata.get("device", "")).strip() or str(get_flag_value(command_argv, "policy.device") or "")
        else:
            device = str(get_flag_value(command_argv, "policy.device") or "")

    dataset_or_env = str(metadata.get("dataset_repo_id", "")).strip()
    metric_summary = ""
    checkpoints = metadata.get("checkpoint_artifacts")
    if not isinstance(checkpoints, list):
        checkpoints = []
    wandb_info = metadata.get("wandb") if isinstance(metadata.get("wandb"), dict) else {}
    if mode == "train":
        output_dir = str(metadata.get("output_dir_resolved", "")).strip() or str(metadata.get("output_dir", "")).strip()
        recent_lines = _safe_read_recent_lines(Path(str(metadata.get("_run_path")))) if metadata.get("_run_path") else []
        train_metrics = metadata.get("train_metrics")
        if not isinstance(train_metrics, dict):
            train_metrics = extract_training_metrics(recent_lines, output_dir=output_dir)
        if not checkpoints:
            checkpoints = discover_checkpoint_artifacts(output_dir)
        if not isinstance(wandb_info, dict):
            wandb_info = {}
        if not wandb_info:
            wandb_info = extract_wandb_run_metadata(
                output_lines=recent_lines,
                output_dir=output_dir,
                enabled=(str(get_flag_value(command_argv, "wandb.enable") or "").strip().lower() == "true"),
                project=str(get_flag_value(command_argv, "wandb.project") or ""),
                job_name=str(metadata.get("job_name", "")).strip(),
            )
        if include_wandb_remote and wandb_info:
            wandb_info = {**wandb_info, **fetch_wandb_remote_snapshot(wandb_info)}
        metric_summary = format_training_metrics_summary(train_metrics)
        return {
            "run_id": str(metadata.get("run_id", metadata.get("_run_path", ""))),
            "mode": mode,
            "status": str(metadata.get("status", "")).strip() or "unknown",
            "started_at_iso": str(metadata.get("started_at_iso", "")).strip(),
            "duration_s": metadata.get("duration_s"),
            "dataset_or_env": str(metadata.get("dataset_repo_id", "")).strip() or "-",
            "policy": policy_type or "-",
            "checkpoint": checkpoints[0]["label"] if checkpoints else "-",
            "command": str(metadata.get("command", "")).strip(),
            "command_argv": command_argv,
            "device": device or "-",
            "notes": notes,
            "tags": tags,
            "output_location": _run_output_location(metadata, mode),
            "metrics": train_metrics,
            "metrics_summary": metric_summary,
            "checkpoints": checkpoints,
            "wandb": wandb_info,
            "model_path": model_path or None,
            "record": metadata,
        }

    if mode == "deploy":
        analytics = extract_deploy_analytics(metadata)
        metric_summary = (
            f"success {analytics['success']}/{analytics['total']} | "
            f"failed {analytics['failed']} | unmarked {analytics['unmarked']}"
            if analytics["total"]
            else "No episode outcomes saved yet."
        )
        return {
            "run_id": str(metadata.get("run_id", metadata.get("_run_path", ""))),
            "mode": mode,
            "status": str(metadata.get("status", "")).strip() or "unknown",
            "started_at_iso": str(metadata.get("started_at_iso", "")).strip(),
            "duration_s": metadata.get("duration_s"),
            "dataset_or_env": str(metadata.get("dataset_repo_id", "")).strip() or "-",
            "policy": policy_type or "-",
            "checkpoint": _model_checkpoint_label(model_path),
            "command": str(metadata.get("command", "")).strip(),
            "command_argv": command_argv,
            "device": device or "-",
            "notes": notes,
            "tags": tags,
            "output_location": _run_output_location(metadata, mode),
            "metrics": analytics,
            "metrics_summary": metric_summary,
            "checkpoints": [],
            "wandb": {},
            "model_path": model_path or None,
            "record": metadata,
        }

    recent_lines = _safe_read_recent_lines(Path(str(metadata.get("_run_path")))) if metadata.get("_run_path") else []
    sim_eval_metrics = metadata.get("sim_eval_metrics")
    if not isinstance(sim_eval_metrics, dict):
        sim_eval_metrics = extract_sim_eval_metrics(
            recent_lines,
            output_dir=str(metadata.get("output_dir_resolved", "")).strip() or metadata.get("output_dir"),
        )
    sim_eval = metadata.get("sim_eval") if isinstance(metadata.get("sim_eval"), dict) else {}
    env_type = str(sim_eval.get("env_type", "")).strip()
    benchmark = str(sim_eval.get("benchmark", "")).strip()
    task = str(sim_eval.get("task", "")).strip()
    dataset_or_env = benchmark or env_type or "-"
    if task:
        dataset_or_env = f"{dataset_or_env} · {task}" if dataset_or_env != "-" else task
    metric_summary = []
    if isinstance(sim_eval_metrics.get("pc_success"), (int, float)):
        metric_summary.append(f"success {float(sim_eval_metrics['pc_success']) * 100:.1f}%")
    if isinstance(sim_eval_metrics.get("avg_sum_reward"), (int, float)):
        metric_summary.append(f"reward {float(sim_eval_metrics['avg_sum_reward']):.3f}")
    if isinstance(sim_eval_metrics.get("eval_s"), (int, float)):
        metric_summary.append(f"eval {float(sim_eval_metrics['eval_s']):.2f}s")
    return {
        "run_id": str(metadata.get("run_id", metadata.get("_run_path", ""))),
        "mode": mode,
        "status": str(metadata.get("status", "")).strip() or "unknown",
        "started_at_iso": str(metadata.get("started_at_iso", "")).strip(),
        "duration_s": metadata.get("duration_s"),
        "dataset_or_env": dataset_or_env,
        "policy": policy_type or "-",
        "checkpoint": _model_checkpoint_label(model_path),
        "command": str(metadata.get("command", "")).strip(),
        "command_argv": command_argv,
        "device": device or "-",
        "notes": notes,
        "tags": tags,
        "output_location": _run_output_location(metadata, mode),
        "metrics": sim_eval_metrics,
        "metrics_summary": " | ".join(metric_summary) if metric_summary else "No parsed sim eval metrics yet.",
        "checkpoints": [],
        "wandb": {},
        "model_path": model_path or None,
        "record": metadata,
    }


def collect_experiment_runs(
    config: dict[str, Any],
    *,
    limit: int = 0,
    include_wandb_remote: bool = False,
) -> dict[str, Any]:
    runs, warning_count = list_runs(config=config, limit=limit)
    records: list[dict[str, Any]] = []
    for item in runs:
        mode = str(item.get("mode", "")).strip().lower()
        if mode not in _EXPERIMENT_RUN_MODES:
            continue
        records.append(_build_record(dict(item), include_wandb_remote=include_wandb_remote))

    stats = {
        "total": len(records),
        "train": sum(1 for item in records if item["mode"] == "train"),
        "deploy": sum(1 for item in records if item["mode"] == "deploy"),
        "sim_eval": sum(1 for item in records if item["mode"] == "sim_eval"),
    }
    return {
        "records": records,
        "warning_count": warning_count,
        "stats": stats,
    }


def build_experiment_comparison_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        tags = ", ".join(str(tag) for tag in record.get("tags", []))
        notes = str(record.get("notes", "")).strip()
        notes_and_tags = " | ".join(part for part in (notes, tags) if part)
        duration_raw = record.get("duration_s")
        try:
            duration_text = f"{float(duration_raw):.1f}s"
        except (TypeError, ValueError):
            duration_text = "-"
        wandb_info = record.get("wandb") if isinstance(record.get("wandb"), dict) else {}
        wandb_label = str(wandb_info.get("run_name") or wandb_info.get("run_id") or "").strip()
        if wandb_info.get("remote_summary"):
            wandb_label = f"{wandb_label} (remote)" if wandb_label else "remote summary"
        rows.append(
            {
                "run_id": record.get("run_id"),
                "values": (
                    str(record.get("mode", "")).replace("_", " ").title(),
                    str(record.get("status", "")).title(),
                    str(record.get("dataset_or_env", "-")),
                    str(record.get("policy", "-")),
                    str(record.get("checkpoint", "-")),
                    str(record.get("device", "-")),
                    duration_text,
                    notes_and_tags or "-",
                    str(record.get("output_location", "-")),
                    str(record.get("metrics_summary", "-")),
                    wandb_label or "-",
                ),
            }
        )

    summary = {
        "total": len(rows),
        "modes": dict(Counter(str(record.get("mode", "")) for record in records)),
        "statuses": dict(Counter(str(record.get("status", "")) for record in records)),
    }
    return {
        "headers": [
            "Type",
            "Status",
            "Dataset / Env",
            "Policy",
            "Checkpoint",
            "Device",
            "Duration",
            "Notes / Tags",
            "Output",
            "Metrics",
            "WandB",
        ],
        "rows": rows,
        "summary": summary,
    }


def build_experiment_details_text(record: Mapping[str, Any]) -> str:
    lines = [
        f"Run ID: {record.get('run_id', '-')}",
        f"Mode: {str(record.get('mode', '')).replace('_', ' ').title()}",
        f"Status: {str(record.get('status', '')).title()}",
        f"Started: {record.get('started_at_iso', '-')}",
        f"Duration (s): {record.get('duration_s', '-')}",
        f"Dataset / Env: {record.get('dataset_or_env', '-')}",
        f"Policy: {record.get('policy', '-')}",
        f"Checkpoint: {record.get('checkpoint', '-')}",
        f"Device: {record.get('device', '-')}",
        f"Output: {record.get('output_location', '-')}",
    ]
    notes = str(record.get("notes", "")).strip()
    if notes:
        lines.append(f"Notes: {notes}")
    tags = record.get("tags")
    if isinstance(tags, list) and tags:
        lines.append(f"Tags: {', '.join(str(tag) for tag in tags)}")

    metrics = record.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.extend(["", "Metrics"])
        for key in (
            "step",
            "loss",
            "best_loss",
            "throughput",
            "throughput_unit",
            "pc_success",
            "avg_sum_reward",
            "eval_s",
            "eval_ep_s",
        ):
            if key not in metrics:
                continue
            lines.append(f"- {key}: {metrics[key]}")
        if record.get("mode") == "deploy":
            categories = metrics.get("failure_categories")
            if isinstance(categories, list) and categories:
                lines.append("- failure categories: " + ", ".join(f"{item['label']}={item['count']}" for item in categories))

    checkpoints = record.get("checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        lines.extend(["", "Discovered Checkpoints"])
        for checkpoint in checkpoints[:12]:
            parts = [str(checkpoint.get("label", checkpoint.get("path", "-")))]
            if checkpoint.get("kind"):
                parts.append(f"kind={checkpoint['kind']}")
            if isinstance(checkpoint.get("step"), int):
                parts.append(f"step={checkpoint['step']}")
            lines.append("- " + " | ".join(parts))

    wandb_info = record.get("wandb")
    if isinstance(wandb_info, dict) and wandb_info.get("enabled"):
        lines.extend(["", "WandB"])
        for key in ("entity", "project", "run_id", "run_name", "run_url"):
            value = wandb_info.get(key)
            if value:
                lines.append(f"- {key}: {value}")
        summary = wandb_info.get("remote_summary") or wandb_info.get("summary")
        if isinstance(summary, dict) and summary:
            lines.append("- summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        remote_error = str(wandb_info.get("remote_error", "")).strip()
        if remote_error:
            lines.append(f"- remote_error: {remote_error}")

    first_failure = first_failure_from_metadata(dict(record.get("record", {}) if isinstance(record.get("record"), dict) else {}))
    if first_failure is not None:
        lines.extend(
            [
                "",
                "First Failure",
                f"- {first_failure.code or 'UNSPECIFIED'} {first_failure.name}",
                f"- detail: {first_failure.detail}",
                f"- attribution: {attribution_label(normalize_attribution(first_failure.attribution))}",
            ]
        )
        if first_failure.fix:
            lines.append(f"- fix: {first_failure.fix}")

    command = str(record.get("command", "")).strip()
    if command:
        lines.extend(["", "Command", command])
    return "\n".join(lines)
