from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.gui_visualizer_tab import (
    _collect_deploy_sources,
    _collect_dataset_sources,
    _deployment_insights,
    _discover_video_files,
)


class GuiVisualizerTabHelpersTest(unittest.TestCase):
    def test_discover_video_files_finds_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            (root / "b.txt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "c.webm").write_text("x", encoding="utf-8")

            items = _discover_video_files(root)

        self.assertEqual([item["relative_path"] for item in items], ["a.mp4", "nested/c.webm"])

    def test_discover_video_files_skips_hidden_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hidden = root / ".cache"
            hidden.mkdir(parents=True, exist_ok=True)
            (hidden / "ignored.mp4").write_text("x", encoding="utf-8")
            visible = root / "videos"
            visible.mkdir(parents=True, exist_ok=True)
            (visible / "keep.mp4").write_text("x", encoding="utf-8")

            items = _discover_video_files(root)

        self.assertEqual([item["relative_path"] for item in items], ["videos/keep.mp4"])

    def test_collect_dataset_sources_handles_org_repo_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset = data_root / "alice" / "dataset_1"
            dataset.mkdir(parents=True, exist_ok=True)
            (dataset / "videos").mkdir(parents=True, exist_ok=True)

            items = _collect_dataset_sources(config={}, data_root=data_root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "alice/dataset_1")

    def test_collect_dataset_sources_filters_non_dataset_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset = data_root / "alice" / "dataset_1"
            dataset.mkdir(parents=True, exist_ok=True)
            (dataset / "videos").mkdir(parents=True, exist_ok=True)
            notes = data_root / "alice" / "notes"
            notes.mkdir(parents=True, exist_ok=True)
            (notes / "readme.txt").write_text("n/a", encoding="utf-8")

            items = _collect_dataset_sources(config={}, data_root=data_root)

        self.assertEqual([item["name"] for item in items], ["alice/dataset_1"])

    def test_collect_dataset_sources_supports_direct_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_1"
            dataset_root.mkdir(parents=True, exist_ok=True)
            (dataset_root / "chunk-000").mkdir(parents=True, exist_ok=True)

            items = _collect_dataset_sources(config={}, data_root=dataset_root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "dataset_1")

    def test_collect_deploy_sources_skips_entries_without_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "deploy_run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            runs = [
                {"mode": "deploy", "_run_path": str(run_dir), "run_id": "deploy_run_1"},
                {"mode": "deploy", "_run_path": "", "run_id": "missing"},
                {"mode": "deploy", "run_id": "missing2"},
                {"mode": "record", "_run_path": str(run_dir), "run_id": "record_run"},
            ]
            with patch("robot_pipeline_app.gui_visualizer_tab.list_runs", return_value=(runs, None)):
                items = _collect_deploy_sources(config={})

        self.assertEqual([item["name"] for item in items], ["deploy_run_1"])

    def test_deployment_insights_counts_success_failure_pending_and_tags(self) -> None:
        metadata = {
            "deploy_notes_summary": "Overall stable.",
            "deploy_episode_outcomes": {
                "total_episodes": 4,
                "episode_outcomes": [
                    {"episode": 1, "result": "success", "tags": ["smooth"], "note": "good"},
                    {"episode": 2, "result": "failed", "tags": ["collision"], "note": "bumped"},
                    {"episode": 3, "result": "pending", "tags": []},
                ],
            },
        }

        insights = _deployment_insights(metadata)

        self.assertEqual(insights["success"], 1)
        self.assertEqual(insights["failed"], 1)
        self.assertEqual(insights["unmarked"], 2)
        self.assertEqual(insights["pending"], 2)
        self.assertEqual(insights["tags"], ["collision", "smooth"])
        self.assertEqual(insights["overall_notes"], "Overall stable.")


if __name__ == "__main__":
    unittest.main()
