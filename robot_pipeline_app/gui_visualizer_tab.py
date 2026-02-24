from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import list_runs
from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path, save_config

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MAX_VIDEOS_PER_SOURCE = 200
_MAX_SOURCES_PER_LIST = 500
_SKIP_DIR_NAMES = {"__pycache__", ".git"}
_DATASET_MARKER_FILES = {"episodes.parquet", "meta.json", "stats.json"}


@dataclass
class VisualizerTabHandles:
    refresh: Callable[[], None]


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


def _open_path(path: Path) -> tuple[bool, str]:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        return False, f"Unable to open: {exc}"
    return True, "Opened"


def setup_visualizer_tab(*, root: Any, visualizer_tab: Any, config: dict[str, Any], colors: dict[str, str], log_panel: Any, messagebox: Any) -> VisualizerTabHandles:
    import tkinter as tk
    from tkinter import ttk

    frame = ttk.Frame(visualizer_tab, style="Panel.TFrame")
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=2)
    frame.columnconfigure(1, weight=3)
    frame.rowconfigure(1, weight=1)

    source_var = tk.StringVar(value="deployments")
    dataset_root_default = normalize_path(str(config.get("record_data_dir", get_lerobot_dir(config) / "data")))
    deploy_root_default = normalize_path(str(config.get("deploy_data_dir", get_deploy_data_dir(config))))
    dataset_root_var = tk.StringVar(value=dataset_root_default)
    deploy_root_var = tk.StringVar(value=deploy_root_default)
    root_label_var = tk.StringVar(value="Deployments root")
    root_var = tk.StringVar(value=deploy_root_var.get())

    toolbar = ttk.Frame(frame, style="Panel.TFrame")
    toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    ttk.Label(toolbar, text="Source", style="Field.TLabel").pack(side="left")
    ttk.Radiobutton(toolbar, text="Deployments", value="deployments", variable=source_var).pack(side="left", padx=(8, 6))
    ttk.Radiobutton(toolbar, text="Datasets", value="datasets", variable=source_var).pack(side="left")

    ttk.Label(toolbar, textvariable=root_label_var, style="Field.TLabel").pack(side="left", padx=(20, 6))
    root_entry = ttk.Entry(toolbar, textvariable=root_var, width=42)
    root_entry.pack(side="left", fill="x", expand=True)

    def _sync_root_controls_for_source() -> None:
        if source_var.get() == "deployments":
            root_label_var.set("Deployments root")
            root_var.set(deploy_root_var.get().strip() or deploy_root_default)
            return
        root_label_var.set("Datasets root")
        root_var.set(dataset_root_var.get().strip() or dataset_root_default)

    def _set_active_root(value: str, *, persist: bool) -> None:
        cleaned = normalize_path(value.strip()) if value.strip() else ""
        if source_var.get() == "deployments":
            final_value = cleaned or deploy_root_default
            deploy_root_var.set(final_value)
            root_var.set(final_value)
            config["deploy_data_dir"] = final_value
            if persist:
                save_config(config, quiet=True)
            return

        final_value = cleaned or dataset_root_default
        dataset_root_var.set(final_value)
        root_var.set(final_value)
        config["record_data_dir"] = final_value
        if persist:
            save_config(config, quiet=True)

    def choose_dataset_root() -> None:
        try:
            from tkinter import filedialog as _fd

            from .gui_file_dialogs import ask_directory_dialog

            chosen = ask_directory_dialog(
                root=root,
                filedialog=_fd,
                initial_dir=root_var.get().strip() or str(Path.home()),
                title="Choose deployments root" if source_var.get() == "deployments" else "Choose dataset root",
            )
        except Exception:
            chosen = None
        if chosen:
            _set_active_root(str(chosen), persist=True)
            refresh()

    ttk.Button(toolbar, text="Browse", command=choose_dataset_root).pack(side="left", padx=(6, 0))
    ttk.Button(toolbar, text="Refresh", command=lambda: refresh()).pack(side="left", padx=(6, 0))

    source_list = ttk.Treeview(frame, columns=("kind", "path"), show="headings", style="History.Treeview")
    source_list.heading("kind", text="Type")
    source_list.heading("path", text="Run / Dataset")
    source_list.column("kind", width=110, anchor="w")
    source_list.column("path", width=360, anchor="w")
    source_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

    right = ttk.Frame(frame, style="Panel.TFrame")
    right.grid(row=1, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=2)
    right.rowconfigure(3, weight=2)

    meta_title = ttk.Label(right, text="Selection Details", style="SectionTitle.TLabel")
    meta_title.grid(row=0, column=0, sticky="w")
    meta_text = tk.Text(right, height=8, wrap="word", bg=colors.get("surface", "#1a1a1a"), fg=colors.get("text", "#eeeeee"))
    meta_text.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

    insights_title = ttk.Label(right, text="Deployment Insights", style="SectionTitle.TLabel")
    insights_title.grid(row=2, column=0, sticky="w")
    insights_tree = ttk.Treeview(right, columns=("episode", "result", "tags", "note"), show="headings", height=6, style="History.Treeview")
    for key, heading, width in (("episode", "Episode", 70), ("result", "Result", 90), ("tags", "Tags", 180), ("note", "Notes", 320)):
        insights_tree.heading(key, text=heading)
        insights_tree.column(key, width=width, anchor="w")
    insights_tree.grid(row=3, column=0, sticky="nsew", pady=(4, 8))

    videos_title = ttk.Label(right, text="Video Feed", style="SectionTitle.TLabel")
    videos_title.grid(row=4, column=0, sticky="w")
    video_tree = ttk.Treeview(right, columns=("file", "size"), show="headings", height=6, style="History.Treeview")
    video_tree.heading("file", text="Video")
    video_tree.heading("size", text="Size")
    video_tree.column("file", width=460, anchor="w")
    video_tree.column("size", width=100, anchor="e")
    video_tree.grid(row=5, column=0, sticky="nsew")

    current_sources: dict[str, dict[str, Any]] = {}
    current_videos: dict[str, dict[str, Any]] = {}

    def _clear_tree(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def render_empty_state(reason: str) -> None:
        meta_text.configure(state="normal")
        meta_text.delete("1.0", "end")
        meta_text.insert("1.0", reason)
        meta_text.configure(state="disabled")
        insights_title.configure(text="Deployment Insights (deployments only)")
        _clear_tree(insights_tree)
        _clear_tree(video_tree)
        current_videos.clear()
        videos_title.configure(text="Video Feed (0 found)")

    def render_selection(source: dict[str, Any]) -> None:
        source_path = Path(source["path"])
        metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}
        run_path = source.get("run_path")
        data_path = source.get("data_path")

        meta_payload = {
            "kind": source.get("kind"),
            "name": source.get("name"),
            "path": str(source_path),
            "run_path": str(run_path) if isinstance(run_path, Path) else run_path,
            "data_path": str(data_path) if isinstance(data_path, Path) else data_path,
            "dataset_repo_id": metadata.get("dataset_repo_id"),
            "model_path": metadata.get("model_path"),
            "started_at_iso": metadata.get("started_at_iso"),
            "ended_at_iso": metadata.get("ended_at_iso"),
            "status": metadata.get("status"),
            "run_id": metadata.get("run_id"),
        }

        meta_text.configure(state="normal")
        meta_text.delete("1.0", "end")
        meta_text.insert("1.0", json.dumps(meta_payload, indent=2, default=str))
        meta_text.configure(state="disabled")

        _clear_tree(insights_tree)

        insights = _deployment_insights(metadata) if source.get("kind") == "deployment" else None
        if insights is None:
            insights_title.configure(text="Deployment Insights (deployments only)")
        else:
            insights_title.configure(
                text=(
                    f"Deployment Insights · Success {insights['success']} · Failed {insights['failed']} "
                    f"· Unmarked {insights['unmarked']} · Tags: {', '.join(insights['tags']) or '-'}"
                )
            )
            for row in insights["episodes"]:
                insights_tree.insert(
                    "",
                    "end",
                    values=(row.get("episode"), str(row.get("result", "")).title(), ", ".join(row.get("tags", [])), row.get("note", "")),
                )

        _clear_tree(video_tree)
        current_videos.clear()
        videos = _discover_video_files(source_path)
        for idx, item in enumerate(videos):
            iid = f"video-{idx}"
            current_videos[iid] = item
            video_tree.insert("", "end", iid=iid, values=(item["relative_path"], item["size_text"]))
        if len(videos) >= _MAX_VIDEOS_PER_SOURCE:
            videos_title.configure(text=f"Video Feed ({len(videos)} shown, cap {_MAX_VIDEOS_PER_SOURCE})")
        else:
            videos_title.configure(text=f"Video Feed ({len(videos)} found)")

    def refresh() -> None:
        _clear_tree(source_list)
        current_sources.clear()

        if source_var.get() == "deployments":
            _set_active_root(root_var.get(), persist=False)
            deploy_root = Path(deploy_root_var.get().strip() or deploy_root_default)
            sources = _collect_deploy_sources(config, deploy_root=deploy_root)
        else:
            _set_active_root(root_var.get(), persist=False)
            dataset_root = Path(dataset_root_var.get().strip() or dataset_root_default)
            sources = _collect_dataset_sources(config, data_root=dataset_root)

        for idx, src in enumerate(sources):
            iid = f"source-{idx}"
            current_sources[iid] = src
            source_list.insert("", "end", iid=iid, values=(src.get("kind", "-"), src.get("name", "-")))

        if sources:
            source_list.selection_set("source-0")
            render_selection(sources[0])
            return
        source_kind = "deployment runs" if source_var.get() == "deployments" else "datasets"
        render_empty_state(f"No {source_kind} found. Try refreshing or changing the source root path.")

    def on_source_selected(_: Any) -> None:
        selected = source_list.selection()
        if not selected:
            return
        source = current_sources.get(selected[0])
        if source is None:
            return
        render_selection(source)

    def open_selected_video() -> None:
        selected = video_tree.selection()
        if not selected:
            messagebox.showinfo("Visualizer", "Select a video first.")
            return
        video_item = current_videos.get(selected[0])
        if video_item is None:
            return
        ok, msg = _open_path(Path(video_item["path"]))
        if ok:
            log_panel.append_log(f"Visualizer opened video: {video_item['path']}")
        else:
            messagebox.showerror("Visualizer", msg)

    def open_selected_source() -> None:
        selected = source_list.selection()
        if not selected:
            messagebox.showinfo("Visualizer", "Select a source first.")
            return
        source = current_sources.get(selected[0])
        if source is None:
            return
        target = Path(source["path"])
        if not target.exists():
            run_path = source.get("run_path")
            if isinstance(run_path, Path) and run_path.exists():
                target = run_path
        ok, msg = _open_path(target)
        if ok:
            log_panel.append_log(f"Visualizer opened source: {target}")
        else:
            messagebox.showerror("Visualizer", msg)

    action_row = ttk.Frame(right, style="Panel.TFrame")
    action_row.grid(row=6, column=0, sticky="w", pady=(8, 0))
    ttk.Button(action_row, text="Open Selected Source", command=open_selected_source).pack(side="left")
    ttk.Button(action_row, text="Open Selected Video", command=open_selected_video).pack(side="left", padx=(6, 0))

    def on_root_enter(_: Any = None) -> None:
        _set_active_root(root_var.get(), persist=True)
        refresh()

    def on_source_changed(*_: Any) -> None:
        _sync_root_controls_for_source()
        refresh()

    source_list.bind("<<TreeviewSelect>>", on_source_selected)
    root_entry.bind("<Return>", on_root_enter)
    video_tree.bind("<Double-1>", lambda *_: open_selected_video())
    source_var.trace_add("write", on_source_changed)

    _sync_root_controls_for_source()
    refresh()
    return VisualizerTabHandles(refresh=refresh)
