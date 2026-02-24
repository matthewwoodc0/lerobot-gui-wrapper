from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import list_runs
from .config_store import get_lerobot_dir, normalize_path, save_config

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MAX_VIDEOS_PER_SOURCE = 200


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


def _discover_video_files(root: Path, *, limit: int = _MAX_VIDEOS_PER_SOURCE) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    found: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("*")):
        if len(found) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
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


def _deployment_insights(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = metadata.get("deploy_episode_outcomes")
    if not isinstance(summary, dict):
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
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
    pending = 0
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        ep = entry.get("episode")
        result = str(entry.get("result", "pending")).strip().lower()
        if result == "success":
            success += 1
        elif result == "failed":
            failed += 1
        else:
            pending += 1
            result = "pending"
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
        "pending": max(total - success - failed, 0),
        "tags": sorted(tag_set),
        "episodes": parsed,
        "overall_notes": str(metadata.get("deploy_notes_summary", "")).strip(),
    }


def _collect_deploy_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs, _ = list_runs(config=config, limit=500)
    sources: list[dict[str, Any]] = []
    for item in runs:
        if str(item.get("mode", "")).strip().lower() != "deploy":
            continue
        run_path = Path(str(item.get("_run_path", "")).strip())
        if not run_path.exists() or not run_path.is_dir():
            continue
        name = str(item.get("run_id") or run_path.name)
        sources.append({"id": f"deploy::{run_path}", "name": name, "path": run_path, "metadata": item, "kind": "deployment"})
    return sources


def _collect_dataset_sources(config: dict[str, Any], data_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(data_root) if data_root is not None else (get_lerobot_dir(config) / "data")
    if not root.exists() or not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        child_dirs = [p for p in path.iterdir() if p.is_dir()]
        if child_dirs and all(("videos" in p.name.lower() or p.name.startswith("chunk-")) for p in child_dirs):
            sources.append({"id": f"dataset::{path}", "name": path.name, "path": path, "metadata": {}, "kind": "dataset"})
            continue
        for repo_dir in child_dirs:
            if repo_dir.is_dir():
                sources.append({"id": f"dataset::{repo_dir}", "name": f"{path.name}/{repo_dir.name}", "path": repo_dir, "metadata": {}, "kind": "dataset"})
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
    root_var = tk.StringVar(value=str(get_lerobot_dir(config) / "data"))

    toolbar = ttk.Frame(frame, style="Panel.TFrame")
    toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    ttk.Label(toolbar, text="Source", style="Field.TLabel").pack(side="left")
    ttk.Radiobutton(toolbar, text="Deployments", value="deployments", variable=source_var).pack(side="left", padx=(8, 6))
    ttk.Radiobutton(toolbar, text="Datasets", value="datasets", variable=source_var).pack(side="left")

    ttk.Label(toolbar, text="Dataset root", style="Field.TLabel").pack(side="left", padx=(20, 6))
    root_entry = ttk.Entry(toolbar, textvariable=root_var, width=42)
    root_entry.pack(side="left", fill="x", expand=True)

    def choose_dataset_root() -> None:
        try:
            from .gui_file_dialogs import ask_directory_dialog

            chosen = ask_directory_dialog(initialdir=root_var.get().strip() or str(Path.home()), title="Choose dataset root")
        except Exception:
            chosen = None
        if chosen:
            root_var.set(str(chosen))
            config["record_data_dir"] = str(chosen)
            save_config(config)
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

    def render_selection(source: dict[str, Any]) -> None:
        source_path = Path(source["path"])
        metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}

        meta_payload = {
            "kind": source.get("kind"),
            "name": source.get("name"),
            "path": str(source_path),
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

        for item in insights_tree.get_children():
            insights_tree.delete(item)

        insights = _deployment_insights(metadata) if source.get("kind") == "deployment" else None
        if insights is None:
            insights_title.configure(text="Deployment Insights (deployments only)")
        else:
            insights_title.configure(
                text=(
                    f"Deployment Insights · Success {insights['success']} · Failed {insights['failed']} "
                    f"· Pending {insights['pending']} · Tags: {', '.join(insights['tags']) or '-'}"
                )
            )
            for row in insights["episodes"]:
                insights_tree.insert(
                    "",
                    "end",
                    values=(row.get("episode"), str(row.get("result", "")).title(), ", ".join(row.get("tags", [])), row.get("note", "")),
                )

        for item in video_tree.get_children():
            video_tree.delete(item)
        current_videos.clear()
        videos = _discover_video_files(source_path)
        for idx, item in enumerate(videos):
            iid = f"video-{idx}"
            current_videos[iid] = item
            video_tree.insert("", "end", iid=iid, values=(item["relative_path"], item["size_text"]))
        videos_title.configure(text=f"Video Feed ({len(videos)} found)")

    def refresh() -> None:
        for item in source_list.get_children():
            source_list.delete(item)
        current_sources.clear()

        if source_var.get() == "deployments":
            sources = _collect_deploy_sources(config)
        else:
            sources = _collect_dataset_sources(config, data_root=Path(normalize_path(root_var.get().strip() or str(get_lerobot_dir(config) / "data"))))

        for idx, src in enumerate(sources):
            iid = f"source-{idx}"
            current_sources[iid] = src
            source_list.insert("", "end", iid=iid, values=(src.get("kind", "-"), src.get("name", "-")))

        if sources:
            source_list.selection_set("source-0")
            render_selection(sources[0])

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

    action_row = ttk.Frame(right, style="Panel.TFrame")
    action_row.grid(row=6, column=0, sticky="w", pady=(8, 0))
    ttk.Button(action_row, text="Open Selected Video", command=open_selected_video).pack(side="left")

    source_list.bind("<<TreeviewSelect>>", on_source_selected)
    source_var.trace_add("write", lambda *_: refresh())

    refresh()
    return VisualizerTabHandles(refresh=refresh)
