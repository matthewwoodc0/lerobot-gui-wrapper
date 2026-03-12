from __future__ import annotations

from pathlib import Path
from typing import Any

from .workspace_provenance import read_workspace_provenance


def _dataset_key(repo_id: str) -> str:
    return f"dataset:{str(repo_id).strip().strip('/')}"


def _model_key(model_path: str) -> str:
    return f"model:{str(model_path).strip()}"


def _run_key(run_id: str) -> str:
    return f"run:{str(run_id).strip()}"


def build_lineage_graph(runs: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, **payload: Any) -> None:
        existing = nodes.get(node_id, {})
        existing.update({key: value for key, value in payload.items() if value is not None})
        nodes[node_id] = existing

    def add_edge(source: str, target: str, relation: str) -> None:
        edges.append({"source": source, "target": target, "relation": relation})

    for item in runs:
        run_id = str(item.get("run_id") or item.get("_run_path") or "").strip()
        if not run_id:
            continue
        mode = str(item.get("mode", "run")).strip().lower() or "run"
        run_node = _run_key(run_id)
        add_node(run_node, id=run_node, kind="run", mode=mode, label=run_id, path=item.get("_run_path"))

        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        if dataset_repo_id:
            dataset_node = _dataset_key(dataset_repo_id)
            add_node(dataset_node, id=dataset_node, kind="dataset", label=dataset_repo_id, repo_id=dataset_repo_id)
            if mode == "record":
                add_edge(run_node, dataset_node, "produces_dataset")
            else:
                add_edge(dataset_node, run_node, "uses_dataset")

        output_dir = str(item.get("output_dir_resolved", "")).strip() or str(item.get("output_dir", "")).strip()
        if output_dir:
            output_node = _model_key(output_dir)
            add_node(output_node, id=output_node, kind="model_group", label=Path(output_dir).name or output_dir, path=output_dir)
            add_edge(run_node, output_node, "produces_model_group")

        checkpoints = item.get("checkpoint_artifacts")
        if isinstance(checkpoints, list):
            for checkpoint in checkpoints:
                if not isinstance(checkpoint, dict):
                    continue
                checkpoint_path = str(checkpoint.get("path", "")).strip()
                if not checkpoint_path:
                    continue
                checkpoint_node = _model_key(checkpoint_path)
                add_node(
                    checkpoint_node,
                    id=checkpoint_node,
                    kind="checkpoint",
                    label=str(checkpoint.get("label", "")).strip() or Path(checkpoint_path).name,
                    path=checkpoint_path,
                )
                add_edge(run_node, checkpoint_node, "produces_checkpoint")

        model_path = str(item.get("model_path", "")).strip()
        if model_path:
            model_node = _model_key(model_path)
            add_node(model_node, id=model_node, kind="model", label=Path(model_path).name or model_path, path=model_path)
            add_edge(model_node, run_node, "used_by_run")

        resume_from = str(item.get("resume_from", "")).strip()
        if resume_from:
            resume_node = _model_key(resume_from)
            add_node(resume_node, id=resume_node, kind="checkpoint", label=Path(resume_from).name or resume_from, path=resume_from)
            add_edge(resume_node, run_node, "resumed_into")

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def lineage_rows_for_selection(
    *,
    selection: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kind = str(selection.get("kind", "")).strip()
    scope = str(selection.get("scope", "local")).strip() or "local"
    repo_id = str(selection.get("repo_id", "")).strip()
    path = str(selection.get("path", "")).strip()
    run_path = str(selection.get("run_path", "")).strip()
    run_id = str(selection.get("run_id", "")).strip() or (Path(run_path).name if run_path else "")

    if scope == "local" and path:
        provenance = read_workspace_provenance(Path(path))
        if provenance:
            repo = str(provenance.get("repo_id", "")).strip()
            if repo:
                url = f"https://huggingface.co/datasets/{repo}" if kind == "dataset" else f"https://huggingface.co/{repo}"
                rows.append({"relation": "HF source", "label": repo, "target": url})

    if kind == "dataset":
        dataset_id = repo_id or str(selection.get("name", "")).strip()
        for item in runs:
            if str(item.get("dataset_repo_id", "")).strip() != dataset_id:
                continue
            rows.append(
                {
                    "relation": f"{str(item.get('mode', 'run')).strip()} run",
                    "label": str(item.get("run_id", "")).strip() or str(item.get("_run_path", "")),
                    "target": str(item.get("_run_path", "")).strip(),
                }
            )
            checkpoints = item.get("checkpoint_artifacts")
            if isinstance(checkpoints, list):
                for checkpoint in checkpoints[:3]:
                    if isinstance(checkpoint, dict) and checkpoint.get("path"):
                        rows.append(
                            {
                                "relation": "Produced checkpoint",
                                "label": str(checkpoint.get("label", "")).strip() or Path(str(checkpoint["path"])).name,
                                "target": str(checkpoint["path"]),
                            }
                        )
    elif kind == "model":
        for item in runs:
            model_path = str(item.get("model_path", "")).strip()
            checkpoint_match = False
            checkpoints = item.get("checkpoint_artifacts")
            if isinstance(checkpoints, list):
                checkpoint_match = any(str(entry.get("path", "")).strip() == path for entry in checkpoints if isinstance(entry, dict))
            produced_group = str(item.get("output_dir_resolved", "")).strip() or str(item.get("output_dir", "")).strip()
            if model_path == path or checkpoint_match or produced_group == path:
                rows.append(
                    {
                        "relation": f"{str(item.get('mode', 'run')).strip()} run",
                        "label": str(item.get("run_id", "")).strip() or str(item.get("_run_path", "")),
                        "target": str(item.get("_run_path", "")).strip(),
                    }
                )
        for item in runs:
            model_path = str(item.get("model_path", "")).strip()
            if model_path != path:
                continue
            dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
            if dataset_repo_id:
                rows.append({"relation": "Eval dataset", "label": dataset_repo_id, "target": f"https://huggingface.co/datasets/{dataset_repo_id}"})
    elif run_id:
        for item in runs:
            current_id = str(item.get("run_id", "")).strip() or Path(str(item.get("_run_path", ""))).name
            if current_id != run_id:
                continue
            dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
            if dataset_repo_id:
                rows.append({"relation": "Dataset", "label": dataset_repo_id, "target": f"https://huggingface.co/datasets/{dataset_repo_id}"})
            model_path = str(item.get("model_path", "")).strip()
            if model_path:
                rows.append({"relation": "Model", "label": Path(model_path).name or model_path, "target": model_path})
            checkpoints = item.get("checkpoint_artifacts")
            if isinstance(checkpoints, list):
                for checkpoint in checkpoints[:5]:
                    if isinstance(checkpoint, dict) and checkpoint.get("path"):
                        rows.append(
                            {
                                "relation": "Checkpoint",
                                "label": str(checkpoint.get("label", "")).strip() or Path(str(checkpoint["path"])).name,
                                "target": str(checkpoint["path"]),
                            }
                        )
            break

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("relation", "")), str(row.get("label", "")), str(row.get("target", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
