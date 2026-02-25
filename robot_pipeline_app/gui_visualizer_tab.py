from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .artifacts import list_runs
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path, save_config
from .repo_utils import get_hf_dataset_info, get_hf_model_info, list_hf_datasets, list_hf_models

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MAX_VIDEOS_PER_SOURCE = 200
_MAX_SOURCES_PER_LIST = 500
_SKIP_DIR_NAMES = {"__pycache__", ".git"}
_DATASET_MARKER_FILES = {"episodes.parquet", "meta.json", "stats.json"}


@dataclass
class VisualizerTabHandles:
    refresh: Callable[[], None]


def _wheel_units(event: Any) -> int:
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    try:
        delta = float(getattr(event, "delta", 0.0))
    except (TypeError, ValueError):
        return 0
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        units = int(-delta / 120)
        if units != 0:
            return units
    return -1 if delta > 0 else 1


def _bind_tree_wheel_scroll(tree_widget: Any) -> None:
    def on_wheel(event: Any) -> str | None:
        units = _wheel_units(event)
        if units == 0:
            return None
        before = tree_widget.yview()
        tree_widget.yview_scroll(units, "units")
        after = tree_widget.yview()
        if before != after:
            return "break"
        return None

    tree_widget.bind("<MouseWheel>", on_wheel, add="+")
    tree_widget.bind("<Button-4>", on_wheel, add="+")
    tree_widget.bind("<Button-5>", on_wheel, add="+")


def _format_size_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(size, 0))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def _is_skippable_dir_name(name: str) -> bool:
    return not name or name.startswith(".") or name in _SKIP_DIR_NAMES


def _safe_list_dirs(path: Path) -> list[Path]:
    try:
        children = list(path.iterdir())
    except OSError:
        return []

    dirs: list[Path] = []
    for child in children:
        try:
            if child.is_dir() and not _is_skippable_dir_name(child.name):
                dirs.append(child)
        except OSError:
            continue
    return sorted(dirs)


def _looks_like_dataset_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False

    child_dirs = _safe_list_dirs(path)
    for child in child_dirs:
        name = child.name.lower()
        if name.startswith("chunk-") or "video" in name:
            return True

    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        try:
            if not child.is_file():
                continue
        except OSError:
            continue
        if child.name in _DATASET_MARKER_FILES or child.suffix.lower() in _VIDEO_EXTENSIONS:
            return True
    return False


def _discover_video_files(root: Path, *, limit: int = _MAX_VIDEOS_PER_SOURCE) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir() or limit <= 0:
        return []

    found: list[dict[str, Any]] = []
    for current_root, dirnames, filenames in os.walk(root_path, topdown=True):
        dirnames[:] = sorted(name for name in dirnames if not _is_skippable_dir_name(name))
        for filename in sorted(filenames):
            if len(found) >= limit:
                return found
            if Path(filename).suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            path = Path(current_root) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append(
                {
                    "path": path,
                    "relative_path": str(path.relative_to(root_path)),
                    "size_bytes": int(stat.st_size),
                    "size_text": _format_size_bytes(int(stat.st_size)),
                }
            )
    return found


def _normalize_deploy_result(value: Any) -> str:
    result = str(value or "").strip().lower()
    if result in {"success", "failed"}:
        return result
    if result in {"pending", "unmarked"}:
        return "unmarked"
    return "unmarked"


def _deployment_insights(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = metadata.get("deploy_episode_outcomes")
    if not isinstance(summary, dict):
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "unmarked": 0,
            "pending": 0,
            "tags": [],
            "episodes": [],
            "overall_notes": str(metadata.get("deploy_notes_summary", "")).strip(),
        }

    episodes = summary.get("episode_outcomes")
    if not isinstance(episodes, list):
        episodes = []

    parsed: list[dict[str, Any]] = []
    tag_set: set[str] = set()
    success = 0
    failed = 0
    unmarked = 0
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        ep = entry.get("episode")
        result = _normalize_deploy_result(entry.get("result"))
        if result == "success":
            success += 1
        elif result == "failed":
            failed += 1
        else:
            unmarked += 1
            result = "unmarked"
        tags_raw = entry.get("tags")
        tags = [str(tag).strip() for tag in tags_raw] if isinstance(tags_raw, list) else []
        tags = [tag for tag in tags if tag]
        for tag in tags:
            tag_set.add(tag)
        parsed.append(
            {
                "episode": ep,
                "result": result,
                "tags": tags,
                "note": str(entry.get("note", "")).strip(),
            }
        )

    total_raw = summary.get("total_episodes")
    try:
        total = int(total_raw)
    except (TypeError, ValueError):
        total = len(parsed)
    if total < len(parsed):
        total = len(parsed)

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "unmarked": max(total - success - failed, 0),
        "pending": max(total - success - failed, 0),
        "tags": sorted(tag_set),
        "episodes": parsed,
        "overall_notes": str(metadata.get("deploy_notes_summary", "")).strip(),
    }


def _resolve_deploy_dataset_path(dataset_repo_id: str, deploy_root: Path) -> Path | None:
    repo_id = str(dataset_repo_id or "").strip().strip("/")
    if not repo_id:
        return None

    owner = ""
    repo_name = repo_id
    if "/" in repo_id:
        owner, repo_name = repo_id.split("/", 1)

    candidates: list[Path] = []
    if owner:
        candidates.extend(
            [
                deploy_root / owner / repo_name,
                deploy_root / repo_name,
            ]
        )
        if deploy_root.name == owner:
            candidates.insert(0, deploy_root / repo_name)
    else:
        candidates.append(deploy_root / repo_name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return unique_candidates[0] if unique_candidates else None


def _collect_deploy_sources(config: dict[str, Any], deploy_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(deploy_root) if deploy_root is not None else get_deploy_data_dir(config)
    runs, _ = list_runs(config=config, limit=_MAX_SOURCES_PER_LIST)
    sources: list[dict[str, Any]] = []
    for item in runs:
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
        if str(item.get("mode", "")).strip().lower() != "deploy":
            continue
        run_path_raw = str(item.get("_run_path", "")).strip()
        if not run_path_raw:
            continue
        run_path = Path(run_path_raw)
        if not run_path.exists() or not run_path.is_dir():
            continue

        dataset_repo_id = str(item.get("dataset_repo_id", "")).strip()
        data_path = _resolve_deploy_dataset_path(dataset_repo_id, root)
        selected_path = data_path if data_path is not None else run_path
        name = str(item.get("run_id") or run_path.name)
        sources.append(
            {
                "id": f"deploy::{run_path}",
                "name": name,
                "path": selected_path,
                "run_path": run_path,
                "data_path": data_path,
                "metadata": item,
                "kind": "deployment",
            }
        )
    return sources


def _collect_dataset_sources(config: dict[str, Any], data_root: Path | None = None) -> list[dict[str, Any]]:
    if data_root is not None:
        root = Path(data_root)
    else:
        record_root_raw = str(config.get("record_data_dir", "")).strip()
        if record_root_raw:
            root = Path(normalize_path(record_root_raw))
        else:
            lerobot_root_raw = str(config.get("lerobot_dir", "")).strip()
            if lerobot_root_raw:
                root = Path(normalize_path(lerobot_root_raw)) / "data"
            else:
                root = Path.home() / "lerobot" / "data"
    if not root.exists() or not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _append_source(path: Path, name: str) -> None:
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            return
        try:
            canonical_path = path.resolve()
        except OSError:
            canonical_path = path
        if canonical_path in seen_paths:
            return
        seen_paths.add(canonical_path)
        sources.append({"id": f"dataset::{canonical_path}", "name": name, "path": canonical_path, "metadata": {}, "kind": "dataset"})

    if _looks_like_dataset_dir(root):
        _append_source(root, root.name or str(root))
        return sources

    for owner_dir in _safe_list_dirs(root):
        if _looks_like_dataset_dir(owner_dir):
            _append_source(owner_dir, owner_dir.name)
            continue
        for repo_dir in _safe_list_dirs(owner_dir):
            if len(sources) >= _MAX_SOURCES_PER_LIST:
                break
            if _looks_like_dataset_dir(repo_dir):
                _append_source(repo_dir, f"{owner_dir.name}/{repo_dir.name}")
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
    return sources


def _collect_model_sources(config: dict[str, Any], model_root: Path | None = None) -> list[dict[str, Any]]:
    if model_root is not None:
        root = Path(model_root)
    else:
        models_raw = str(config.get("trained_models_dir", "")).strip()
        if models_raw:
            root = Path(normalize_path(models_raw))
        else:
            root = get_lerobot_dir(config) / "trained_models"
    if not root.exists() or not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for child in _safe_list_dirs(root):
        if len(sources) >= _MAX_SOURCES_PER_LIST:
            break
        sources.append(
            {
                "id": f"model::{child}",
                "name": child.name,
                "path": child,
                "metadata": {},
                "kind": "model",
                "scope": "local",
            }
        )
    return sources


def _collect_hf_dataset_sources(owner: str) -> tuple[list[dict[str, Any]], str | None]:
    rows, error_text = list_hf_datasets(owner, limit=min(_MAX_SOURCES_PER_LIST, 200))
    sources: list[dict[str, Any]] = []
    for row in rows:
        repo_id = str(row.get("repo_id", "")).strip().strip("/")
        if not repo_id:
            continue
        sources.append(
            {
                "id": f"hf-dataset::{repo_id}",
                "name": repo_id,
                "repo_id": repo_id,
                "metadata": row,
                "kind": "dataset",
                "scope": "huggingface",
            }
        )
    return sources, error_text


def _collect_hf_model_sources(owner: str) -> tuple[list[dict[str, Any]], str | None]:
    rows, error_text = list_hf_models(owner, limit=min(_MAX_SOURCES_PER_LIST, 200))
    sources: list[dict[str, Any]] = []
    for row in rows:
        repo_id = str(row.get("repo_id", "")).strip().strip("/")
        if not repo_id:
            continue
        sources.append(
            {
                "id": f"hf-model::{repo_id}",
                "name": repo_id,
                "repo_id": repo_id,
                "metadata": row,
                "kind": "model",
                "scope": "huggingface",
            }
        )
    return sources, error_text


def _local_path_overview(path: Path, *, limit: int = 2500) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "files_scanned": 0,
        "subdirs_scanned": 0,
        "video_files": 0,
        "total_size_bytes": 0,
        "sample_files": [],
        "truncated_scan": False,
    }
    if not path.exists() or not path.is_dir():
        return summary

    files_scanned = 0
    dirs_scanned = 0
    total_size_bytes = 0
    video_files = 0
    sample_files: list[str] = []
    truncated = False

    for current_root, dirnames, filenames in os.walk(path, topdown=True):
        dirnames[:] = sorted(name for name in dirnames if not _is_skippable_dir_name(name))
        dirs_scanned += len(dirnames)
        for filename in sorted(filenames):
            if files_scanned >= limit:
                truncated = True
                break
            files_scanned += 1
            file_path = Path(current_root) / filename
            try:
                total_size_bytes += int(file_path.stat().st_size)
            except OSError:
                pass
            if file_path.suffix.lower() in _VIDEO_EXTENSIONS:
                video_files += 1
            if len(sample_files) < 20:
                try:
                    sample_files.append(str(file_path.relative_to(path)))
                except Exception:
                    sample_files.append(str(file_path))
        if truncated:
            break

    summary["files_scanned"] = files_scanned
    summary["subdirs_scanned"] = dirs_scanned
    summary["video_files"] = video_files
    summary["total_size_bytes"] = total_size_bytes
    summary["sample_files"] = sample_files
    summary["truncated_scan"] = truncated
    return summary


def _discover_hf_dataset_videos(repo_id: str, metadata: dict[str, Any], *, limit: int = _MAX_VIDEOS_PER_SOURCE) -> list[dict[str, Any]]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        return []

    videos: list[dict[str, Any]] = []
    for entry in siblings:
        if len(videos) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        relative = str(entry.get("rfilename") or entry.get("path") or "").strip().strip("/")
        if not relative:
            continue
        suffix = Path(relative).suffix.lower()
        if suffix not in _VIDEO_EXTENSIONS:
            continue
        size_raw = entry.get("size")
        try:
            size_value = int(size_raw)
            size_text = _format_size_bytes(size_value)
        except (TypeError, ValueError):
            size_text = "-"
        videos.append(
            {
                "relative_path": relative,
                "size_text": size_text,
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(relative, safe='/')}",
            }
        )
    return videos


def _open_path(path: Path | str) -> tuple[bool, str]:
    target = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
        elif os.name == "nt":
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception as exc:
        return False, f"Unable to open: {exc}"
    return True, "Opened"


def setup_visualizer_tab(*, root: Any, visualizer_tab: Any, config: dict[str, Any], colors: dict[str, str], log_panel: Any, messagebox: Any) -> VisualizerTabHandles:
    import tkinter as tk
    from tkinter import ttk

    from .gui_file_dialogs import ask_directory_dialog

    frame = ttk.Frame(visualizer_tab, style="Panel.TFrame")
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=2)
    frame.columnconfigure(1, weight=3)
    frame.rowconfigure(1, weight=1)

    source_var = tk.StringVar(value="deployments")
    scope_var = tk.StringVar(value="local")
    hf_owner_var = tk.StringVar(value=str(config.get("hf_username", "")).strip())

    dataset_root_default = normalize_path(str(config.get("record_data_dir", get_lerobot_dir(config) / "data")))
    deploy_root_default = normalize_path(str(config.get("deploy_data_dir", get_deploy_data_dir(config))))
    model_root_default = normalize_path(str(config.get("trained_models_dir", get_lerobot_dir(config) / "trained_models")))
    dataset_root_var = tk.StringVar(value=dataset_root_default)
    deploy_root_var = tk.StringVar(value=deploy_root_default)
    model_root_var = tk.StringVar(value=model_root_default)
    root_label_var = tk.StringVar(value="Deployments root")
    root_var = tk.StringVar(value=deploy_root_var.get())

    toolbar = ttk.Frame(frame, style="Panel.TFrame")
    toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    ttk.Label(toolbar, text="Source", style="Field.TLabel").pack(side="left")
    ttk.Radiobutton(toolbar, text="Deployments", value="deployments", variable=source_var, style="TRadiobutton").pack(side="left", padx=(8, 6))
    ttk.Radiobutton(toolbar, text="Datasets", value="datasets", variable=source_var, style="TRadiobutton").pack(side="left")
    ttk.Radiobutton(toolbar, text="Models", value="models", variable=source_var, style="TRadiobutton").pack(side="left", padx=(6, 0))

    ttk.Label(toolbar, text="Location", style="Field.TLabel").pack(side="left", padx=(18, 6))
    scope_local_radio = ttk.Radiobutton(toolbar, text="Local", value="local", variable=scope_var, style="TRadiobutton")
    scope_local_radio.pack(side="left", padx=(0, 6))
    scope_hf_radio = ttk.Radiobutton(toolbar, text="Hugging Face", value="huggingface", variable=scope_var, style="TRadiobutton")
    scope_hf_radio.pack(side="left")

    root_controls = ttk.Frame(toolbar, style="Panel.TFrame")
    ttk.Label(root_controls, textvariable=root_label_var, style="Field.TLabel").pack(side="left", padx=(18, 6))
    root_entry = ttk.Entry(root_controls, textvariable=root_var, width=42)
    root_entry.pack(side="left", fill="x", expand=True)
    browse_root_button = ttk.Button(root_controls, text="Browse")
    browse_root_button.pack(side="left", padx=(6, 0))

    hf_controls = ttk.Frame(toolbar, style="Panel.TFrame")
    ttk.Label(hf_controls, text="HF owner", style="Field.TLabel").pack(side="left", padx=(18, 6))
    hf_owner_entry = ttk.Entry(hf_controls, textvariable=hf_owner_var, width=24)
    hf_owner_entry.pack(side="left")

    refresh_button = ttk.Button(toolbar, text="Refresh")
    refresh_button.pack(side="right")

    source_list = ttk.Treeview(frame, columns=("scope", "name"), show="headings", style="History.Treeview", selectmode="browse")
    source_list.heading("scope", text="Source")
    source_list.heading("name", text="Run / Dataset")
    source_list.column("scope", width=130, anchor="w")
    source_list.column("name", width=360, anchor="w")
    source_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
    _bind_tree_wheel_scroll(source_list)
    source_scroll = ttk.Scrollbar(
        frame,
        orient="vertical",
        command=source_list.yview,
        style="Dark.Vertical.TScrollbar",
    )
    source_list.configure(yscrollcommand=source_scroll.set)
    source_scroll.grid(row=1, column=0, sticky="nse")

    right = ttk.Frame(frame, style="Panel.TFrame")
    right.grid(row=1, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=2)
    right.rowconfigure(3, weight=2)

    meta_title = ttk.Label(right, text="Selection Details", style="SectionTitle.TLabel")
    meta_title.grid(row=0, column=0, sticky="w")
    meta_wrap = ttk.Frame(right, style="Panel.TFrame")
    meta_wrap.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
    meta_wrap.columnconfigure(0, weight=1)
    meta_wrap.rowconfigure(0, weight=1)
    meta_text = tk.Text(
        meta_wrap,
        height=8,
        wrap="word",
        bg=colors.get("surface", "#1a1a1a"),
        fg=colors.get("text", "#eeeeee"),
        insertbackground=colors.get("text", "#eeeeee"),
        font=(colors.get("font_mono", "TkFixedFont"), 10),
        relief="flat",
        padx=8,
        pady=8,
    )
    meta_scroll = ttk.Scrollbar(
        meta_wrap,
        orient="vertical",
        command=meta_text.yview,
        style="Dark.Vertical.TScrollbar",
    )
    meta_text.configure(yscrollcommand=meta_scroll.set)
    meta_text.grid(row=0, column=0, sticky="nsew")
    meta_scroll.grid(row=0, column=1, sticky="ns")

    insights_title = ttk.Label(right, text="Deployment Insights", style="SectionTitle.TLabel")
    insights_title.grid(row=2, column=0, sticky="w")
    insights_tree = ttk.Treeview(right, columns=("episode", "result", "tags", "note"), show="headings", height=6, style="History.Treeview")
    for key, heading, width in (("episode", "Episode", 70), ("result", "Result", 90), ("tags", "Tags", 180), ("note", "Notes", 320)):
        insights_tree.heading(key, text=heading)
        insights_tree.column(key, width=width, anchor="w")
    insights_tree.grid(row=3, column=0, sticky="nsew", pady=(4, 8))
    _bind_tree_wheel_scroll(insights_tree)

    videos_title = ttk.Label(right, text="Video Feed", style="SectionTitle.TLabel")
    videos_title.grid(row=4, column=0, sticky="w")
    video_tree = ttk.Treeview(right, columns=("file", "size"), show="headings", height=6, style="History.Treeview")
    video_tree.heading("file", text="Video")
    video_tree.heading("size", text="Size")
    video_tree.column("file", width=460, anchor="w")
    video_tree.column("size", width=100, anchor="e")
    video_tree.grid(row=5, column=0, sticky="nsew")
    _bind_tree_wheel_scroll(video_tree)

    current_sources: dict[str, dict[str, Any]] = {}
    current_videos: dict[str, dict[str, Any]] = {}

    def _clear_tree(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _active_source_scope() -> tuple[str, str]:
        source = source_var.get().strip() or "deployments"
        if source == "deployments":
            return "deployments", "local"
        scope = scope_var.get().strip() or "local"
        if scope not in {"local", "huggingface"}:
            scope = "local"
        return source, scope

    def _set_insights_visible(visible: bool) -> None:
        if visible:
            insights_title.grid()
            insights_tree.grid()
            right.rowconfigure(1, weight=2)
            right.rowconfigure(3, weight=2)
        else:
            insights_title.grid_remove()
            insights_tree.grid_remove()
            right.rowconfigure(1, weight=4)
            right.rowconfigure(3, weight=0)

    def _render_meta(payload: dict[str, Any]) -> None:
        meta_text.configure(state="normal")
        meta_text.delete("1.0", "end")
        meta_text.insert("1.0", json.dumps(payload, indent=2, default=str))
        meta_text.see("1.0")
        meta_text.configure(state="disabled")

    def _set_active_root(value: str, *, persist: bool) -> None:
        source, scope = _active_source_scope()
        if scope != "local":
            return
        cleaned = normalize_path(value.strip()) if value.strip() else ""
        if source == "deployments":
            final_value = cleaned or deploy_root_default
            deploy_root_var.set(final_value)
            root_var.set(final_value)
            config["deploy_data_dir"] = final_value
        elif source == "datasets":
            final_value = cleaned or dataset_root_default
            dataset_root_var.set(final_value)
            root_var.set(final_value)
            config["record_data_dir"] = final_value
        else:
            final_value = cleaned or model_root_default
            model_root_var.set(final_value)
            root_var.set(final_value)
            config["trained_models_dir"] = final_value
        if persist:
            save_config(config, quiet=True)

    def _sync_toolbar_for_mode() -> None:
        source, scope = _active_source_scope()
        if source == "deployments" and scope_var.get() != "local":
            scope_var.set("local")
            scope = "local"

        if source == "deployments":
            scope_local_radio.configure(state="disabled")
            scope_hf_radio.configure(state="disabled")
            source_list.heading("name", text="Deployment Run")
        else:
            scope_local_radio.configure(state="normal")
            scope_hf_radio.configure(state="normal")
            source_list.heading("name", text="Dataset" if source == "datasets" else "Model")

        if scope == "local":
            if hf_controls.winfo_manager():
                hf_controls.pack_forget()
            if not root_controls.winfo_manager():
                root_controls.pack(side="left", fill="x", expand=True, padx=(8, 0))
            browse_root_button.configure(state="normal")
            if source == "deployments":
                root_label_var.set("Deployments root")
                root_var.set(deploy_root_var.get().strip() or deploy_root_default)
            elif source == "datasets":
                root_label_var.set("Datasets root")
                root_var.set(dataset_root_var.get().strip() or dataset_root_default)
            else:
                root_label_var.set("Models root")
                root_var.set(model_root_var.get().strip() or model_root_default)
            return

        if root_controls.winfo_manager():
            root_controls.pack_forget()
        if not hf_controls.winfo_manager():
            hf_controls.pack(side="left", padx=(8, 0))

    def _render_empty_state(reason: str) -> None:
        _render_meta({"message": reason})
        _set_insights_visible(False)
        _clear_tree(insights_tree)
        _clear_tree(video_tree)
        current_videos.clear()
        videos_title.configure(text="Video Feed (0 found)")

    def _resolve_hf_metadata(source: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        repo_id = str(source.get("repo_id", "")).strip()
        if not repo_id:
            return None, "Hugging Face repo id is missing."
        kind = str(source.get("kind", "")).strip()
        if kind == "dataset":
            return get_hf_dataset_info(repo_id)
        if kind == "model":
            return get_hf_model_info(repo_id)
        return None, "Unsupported Hugging Face source type."

    def _render_videos_for_source(source: dict[str, Any], metadata: dict[str, Any] | None) -> None:
        _clear_tree(video_tree)
        current_videos.clear()

        scope = str(source.get("scope", "local")).strip() or "local"
        kind = str(source.get("kind", "")).strip()
        videos: list[dict[str, Any]] = []
        if scope == "local":
            source_path_raw = source.get("path")
            if source_path_raw:
                source_path = Path(source_path_raw)
                videos = _discover_video_files(source_path)
        elif scope == "huggingface" and kind == "dataset":
            repo_id = str(source.get("repo_id", "")).strip()
            if repo_id and isinstance(metadata, dict):
                videos = _discover_hf_dataset_videos(repo_id, metadata)

        for idx, item in enumerate(videos):
            iid = f"video-{idx}"
            current_videos[iid] = item
            video_tree.insert("", "end", iid=iid, values=(item.get("relative_path", "-"), item.get("size_text", "-")))
        if len(videos) >= _MAX_VIDEOS_PER_SOURCE:
            videos_title.configure(text=f"Video Feed ({len(videos)} shown, cap {_MAX_VIDEOS_PER_SOURCE})")
        else:
            videos_title.configure(text=f"Video Feed ({len(videos)} found)")

    def _render_selection(source: dict[str, Any]) -> None:
        metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}
        resolved_metadata = metadata
        metadata_error: str | None = None

        if str(source.get("scope", "local")) == "huggingface":
            resolved, error_text = _resolve_hf_metadata(source)
            if resolved is not None:
                resolved_metadata = resolved
            else:
                resolved_metadata = {}
            metadata_error = error_text

        source_path_raw = source.get("path")
        source_path = Path(source_path_raw) if source_path_raw else None
        run_path = source.get("run_path")
        data_path = source.get("data_path")
        repo_id = str(source.get("repo_id", "")).strip()
        scope = str(source.get("scope", "local")).strip() or "local"
        kind = str(source.get("kind", "")).strip() or "source"

        meta_payload: dict[str, Any] = {
            "scope": scope,
            "kind": kind,
            "name": source.get("name"),
            "path": str(source_path) if source_path is not None else None,
            "repo_id": repo_id or None,
            "run_path": str(run_path) if isinstance(run_path, Path) else run_path,
            "data_path": str(data_path) if isinstance(data_path, Path) else data_path,
            "url": (
                f"https://huggingface.co/datasets/{repo_id}"
                if scope == "huggingface" and kind == "dataset" and repo_id
                else (
                    f"https://huggingface.co/{repo_id}"
                    if scope == "huggingface" and kind == "model" and repo_id
                    else None
                )
            ),
        }
        if source_path is not None and scope == "local":
            meta_payload["local_overview"] = _local_path_overview(source_path)
        if metadata_error:
            meta_payload["metadata_error"] = metadata_error
        if resolved_metadata:
            meta_payload["metadata"] = resolved_metadata
        _render_meta(meta_payload)

        _clear_tree(insights_tree)
        insights = _deployment_insights(resolved_metadata) if kind == "deployment" else None
        if insights is None:
            _set_insights_visible(False)
            insights_title.configure(text="Deployment Insights")
        else:
            _set_insights_visible(True)
            insights_title.configure(
                text=(
                    f"Deployment Insights · Success {insights['success']} · Failed {insights['failed']} "
                    f"· Unmarked {insights['unmarked']} · Tags {len(insights['tags'])}"
                )
            )
            for row in insights["episodes"]:
                insights_tree.insert(
                    "",
                    "end",
                    values=(row.get("episode"), str(row.get("result", "")).title(), ", ".join(row.get("tags", [])), row.get("note", "")),
                )

        _render_videos_for_source(source, resolved_metadata if isinstance(resolved_metadata, dict) else None)

    def _collect_sources() -> tuple[list[dict[str, Any]], str | None, str]:
        source, scope = _active_source_scope()
        if source == "deployments":
            _set_active_root(root_var.get(), persist=False)
            deploy_root = Path(deploy_root_var.get().strip() or deploy_root_default)
            rows = _collect_deploy_sources(config, deploy_root=deploy_root)
            for row in rows:
                row["scope"] = "local"
            return rows, None, "deployment runs"

        if source == "datasets" and scope == "local":
            _set_active_root(root_var.get(), persist=False)
            dataset_root = Path(dataset_root_var.get().strip() or dataset_root_default)
            rows = _collect_dataset_sources(config, data_root=dataset_root)
            for row in rows:
                row["scope"] = "local"
            return rows, None, "datasets"

        if source == "models" and scope == "local":
            _set_active_root(root_var.get(), persist=False)
            model_root = Path(model_root_var.get().strip() or model_root_default)
            rows = _collect_model_sources(config, model_root=model_root)
            return rows, None, "models"

        owner = hf_owner_var.get().strip()
        if source == "datasets":
            rows, error_text = _collect_hf_dataset_sources(owner)
            return rows, error_text, f"Hugging Face datasets for {owner or '(owner missing)'}"
        rows, error_text = _collect_hf_model_sources(owner)
        return rows, error_text, f"Hugging Face models for {owner or '(owner missing)'}"

    def refresh() -> None:
        _clear_tree(source_list)
        current_sources.clear()

        sources, error_text, source_kind = _collect_sources()
        if error_text and not sources:
            _render_empty_state(error_text)
            return

        for idx, src in enumerate(sources):
            iid = f"source-{idx}"
            current_sources[iid] = src
            scope_text = "Hugging Face" if str(src.get("scope", "local")) == "huggingface" else "Local"
            kind_text = str(src.get("kind", "source")).strip().title()
            source_list.insert("", "end", iid=iid, values=(f"{scope_text} {kind_text}", src.get("name", "-")))

        if sources:
            source_list.selection_set("source-0")
            _render_selection(sources[0])
            return
        hint = "Try refreshing or changing the root/owner settings."
        _render_empty_state(f"No {source_kind} found. {hint}")

    def _open_selected_video() -> None:
        selected = video_tree.selection()
        if not selected:
            messagebox.showinfo("Visualizer", "Select a video first.")
            return
        video_item = current_videos.get(selected[0])
        if video_item is None:
            return
        target = video_item.get("path") or video_item.get("url")
        if not target:
            return
        ok, msg = _open_path(target)
        if ok:
            log_panel.append_log(f"Visualizer opened video: {target}")
        else:
            messagebox.showerror("Visualizer", msg)

    def _open_selected_source() -> None:
        selected = source_list.selection()
        if not selected:
            messagebox.showinfo("Visualizer", "Select a source first.")
            return
        source = current_sources.get(selected[0])
        if source is None:
            return
        scope = str(source.get("scope", "local")).strip() or "local"
        kind = str(source.get("kind", "")).strip()
        if scope == "huggingface":
            repo_id = str(source.get("repo_id", "")).strip()
            if not repo_id:
                messagebox.showerror("Visualizer", "Selected Hugging Face source is missing repo id.")
                return
            if kind == "dataset":
                target = f"https://huggingface.co/datasets/{repo_id}"
            else:
                target = f"https://huggingface.co/{repo_id}"
            ok, msg = _open_path(target)
            if ok:
                log_panel.append_log(f"Visualizer opened source: {target}")
            else:
                messagebox.showerror("Visualizer", msg)
            return

        source_path_raw = source.get("path")
        target_path = Path(source_path_raw) if source_path_raw else None
        if target_path is None or not target_path.exists():
            run_path = source.get("run_path")
            if isinstance(run_path, Path) and run_path.exists():
                target_path = run_path
        if target_path is None:
            messagebox.showerror("Visualizer", "No local path is available for this source.")
            return
        ok, msg = _open_path(target_path)
        if ok:
            log_panel.append_log(f"Visualizer opened source: {target_path}")
        else:
            messagebox.showerror("Visualizer", msg)

    def on_source_selected(_: Any) -> None:
        selected = source_list.selection()
        if not selected:
            return
        source = current_sources.get(selected[0])
        if source is None:
            return
        _render_selection(source)

    def on_root_enter(_: Any = None) -> None:
        _set_active_root(root_var.get(), persist=True)
        refresh()

    def choose_root() -> None:
        _, scope = _active_source_scope()
        if scope != "local":
            return
        from tkinter import filedialog as _fd

        chosen = ask_directory_dialog(
            root=root,
            filedialog=_fd,
            initial_dir=root_var.get().strip() or str(Path.home()),
            title="Choose visualizer root",
        )
        if chosen:
            _set_active_root(str(chosen), persist=True)
            refresh()

    def on_mode_changed(*_: Any) -> None:
        _sync_toolbar_for_mode()
        refresh()

    action_row = ttk.Frame(right, style="Panel.TFrame")
    action_row.grid(row=6, column=0, sticky="w", pady=(8, 0))
    ttk.Button(action_row, text="Open Selected Source", command=_open_selected_source).pack(side="left")
    ttk.Button(action_row, text="Open Selected Video", command=_open_selected_video).pack(side="left", padx=(6, 0))

    source_list.bind("<<TreeviewSelect>>", on_source_selected)
    root_entry.bind("<Return>", on_root_enter)
    hf_owner_entry.bind("<Return>", lambda *_: refresh())
    video_tree.bind("<Double-1>", lambda *_: _open_selected_video())
    source_var.trace_add("write", on_mode_changed)
    scope_var.trace_add("write", on_mode_changed)
    browse_root_button.configure(command=choose_root)
    refresh_button.configure(command=refresh)

    _sync_toolbar_for_mode()
    _set_insights_visible(source_var.get() == "deployments")
    refresh()
    return VisualizerTabHandles(refresh=refresh)
