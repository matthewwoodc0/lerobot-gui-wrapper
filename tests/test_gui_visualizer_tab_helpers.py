from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.visualizer_utils import (
    _VisualizerRefreshSnapshot,
    _build_selection_payload,
    _collect_sources_for_refresh,
    _collect_videos_for_source,
    _collect_deploy_sources,
    _collect_dataset_sources,
    _deployment_insights,
    _discover_video_files,
    _visualizer_insights_section,
    _visualizer_source_row_values,
)


class GuiVisualizerTabHelpersTest(unittest.TestCase):
    def test_build_selection_payload_includes_visualizer_dataset_metadata_for_v3_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "alice" / "dataset_v3"
            (dataset / "meta").mkdir(parents=True, exist_ok=True)
            (dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "videos" / "front" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "meta" / "info.json").write_text(
                (
                    '{"codebase_version":"v3.0","robot_type":"so101_follower","fps":30,'
                    '"total_episodes":12,"features":{"action":{"shape":[6]},'
                    '"observation.state":{"shape":[7]},'
                    '"observation.images.front":{"dtype":"video"}}}\n'
                ),
                encoding="utf-8",
            )
            (dataset / "meta" / "stats.json").write_text('{"action":{"mean":[0.0]}}\n', encoding="utf-8")
            (dataset / "meta" / "tasks.parquet").write_text("stub\n", encoding="utf-8")
            (dataset / "data" / "chunk-000" / "episode_000000.parquet").write_text("stub\n", encoding="utf-8")
            (dataset / "videos" / "front" / "chunk-000" / "episode_000000.mp4").write_bytes(b"x")

            payload = _build_selection_payload(
                {
                    "scope": "local",
                    "kind": "dataset",
                    "name": "alice/dataset_v3",
                    "path": str(dataset),
                    "metadata": {},
                }
            )

        meta = payload["meta_payload"]["visualizer_metadata"]
        self.assertEqual(meta["format"], "LeRobotDataset")
        self.assertEqual(meta["layout"], "v3")
        self.assertEqual(meta["codebase_version"], "v3.0")
        self.assertEqual(meta["robot_type"], "so101_follower")
        self.assertEqual(meta["fps"], 30)
        self.assertEqual(meta["camera_keys"], ["front"])
        self.assertEqual(meta["action_dim"], 6)
        self.assertEqual(meta["state_dim"], 7)
        self.assertTrue(meta["meta"]["has_info"])
        self.assertEqual(meta["data"]["chunk_count"], 1)
        self.assertEqual(meta["videos"]["camera_keys"], ["front"])

    def test_build_selection_payload_infers_hf_dataset_visualizer_layout_from_siblings(self) -> None:
        with patch(
            "robot_pipeline_app.visualizer_utils.get_hf_dataset_info",
            return_value=(
                {
                    "siblings": [
                        {"rfilename": "meta/info.json"},
                        {"rfilename": "meta/stats.json"},
                        {"rfilename": "meta/tasks.parquet"},
                        {"rfilename": "data/chunk-000/episode_000000.parquet"},
                        {"rfilename": "videos/front/chunk-000/episode_000000.mp4"},
                    ]
                },
                None,
            ),
        ):
            payload = _build_selection_payload(
                {
                    "scope": "huggingface",
                    "kind": "dataset",
                    "name": "alice/dataset_v3",
                    "repo_id": "alice/dataset_v3",
                    "metadata": {},
                }
            )

        meta = payload["meta_payload"]["visualizer_metadata"]
        self.assertEqual(meta["format"], "LeRobotDataset")
        self.assertEqual(meta["layout"], "v3")
        self.assertTrue(meta["meta"]["has_info"])
        self.assertEqual(meta["data"]["chunk_count"], 1)
        self.assertEqual(meta["videos"]["camera_keys"], ["front"])
        self.assertEqual(meta["videos"]["video_file_count"], 1)

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
            (dataset / "meta").mkdir(parents=True, exist_ok=True)
            (dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}\n', encoding="utf-8")

            items = _collect_dataset_sources(config={}, data_root=data_root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "alice/dataset_1")

    def test_collect_dataset_sources_filters_non_dataset_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset = data_root / "alice" / "dataset_1"
            (dataset / "meta").mkdir(parents=True, exist_ok=True)
            (dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}\n', encoding="utf-8")
            notes = data_root / "alice" / "notes"
            notes.mkdir(parents=True, exist_ok=True)
            (notes / "readme.txt").write_text("n/a", encoding="utf-8")

            items = _collect_dataset_sources(config={}, data_root=data_root)

        self.assertEqual([item["name"] for item in items], ["alice/dataset_1"])

    def test_collect_dataset_sources_supports_direct_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_1"
            (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
            (dataset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset_root / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}\n', encoding="utf-8")

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
            with patch("robot_pipeline_app.visualizer_utils.list_runs", return_value=(runs, None)):
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

    def test_visualizer_source_row_values_formats_source_tag(self) -> None:
        self.assertEqual(
            _visualizer_source_row_values({"scope": "huggingface", "kind": "dataset", "name": "alice/ds"}),
            ("source - huggingface", "alice/ds"),
        )
        self.assertEqual(
            _visualizer_source_row_values({"scope": "local", "kind": "deployment", "name": "run_1"}),
            ("source - local", "run_1"),
        )

    def test_visualizer_insights_section_only_enabled_for_deployments(self) -> None:
        visible, header, rows = _visualizer_insights_section("dataset", {"deploy_episode_outcomes": {}})
        self.assertFalse(visible)
        self.assertEqual(header, "Deployment Insights")
        self.assertEqual(rows, [])

        visible_dep, header_dep, rows_dep = _visualizer_insights_section(
            "deployment",
            {
                "deploy_episode_outcomes": {
                    "total_episodes": 2,
                    "episode_outcomes": [
                        {"episode": 1, "result": "success", "tags": ["smooth"], "note": "ok"},
                        {"episode": 2, "result": "failed", "tags": ["collision"], "note": "bumped"},
                    ],
                }
            },
        )
        self.assertTrue(visible_dep)
        self.assertIn("Success 1", header_dep)
        self.assertIn("Failed 1", header_dep)
        self.assertEqual(rows_dep[0], (1, "Success", "smooth", "ok"))
        self.assertEqual(rows_dep[1], (2, "Failed", "collision", "bumped"))

    def test_build_selection_payload_includes_model_and_deploy_visualizer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(
                (
                    '{"policy_family":"pi0-fast","policy_class":"acme.pi0.Policy","plugin_package":"acme",'
                    '"robot_type":"unitree_g1","fps":20,"camera_keys":["front"],'
                    '"output_shapes":{"action":{"shape":[29]}}}\n'
                ),
                encoding="utf-8",
            )
            dataset = root / "eval_dataset"
            (dataset / "meta").mkdir(parents=True, exist_ok=True)
            (dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "videos" / "head" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dataset / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}\n', encoding="utf-8")

            model_payload = _build_selection_payload(
                {"scope": "local", "kind": "model", "name": "model", "path": str(model_dir), "metadata": {}}
            )
            deploy_payload = _build_selection_payload(
                {
                    "scope": "local",
                    "kind": "deployment",
                    "name": "deploy_1",
                    "path": str(dataset),
                    "metadata": {
                        "run_id": "deploy_1",
                        "status": "success",
                        "dataset_repo_id": "alice/eval_demo",
                        "model_path": str(model_dir),
                        "deploy_notes_summary": "Stable",
                        "deploy_episode_outcomes": {
                            "enabled": True,
                            "total_episodes": 3,
                            "episode_outcomes": [
                                {"episode": 1, "result": "success", "tags": ["smooth"]},
                                {"episode": 2, "result": "failed", "tags": ["collision"]},
                            ],
                        },
                    },
                }
            )

        model_meta = model_payload["meta_payload"]["visualizer_metadata"]
        self.assertEqual(model_meta["policy_family"], "Pi0-FAST")
        self.assertEqual(model_meta["plugin_package"], "acme")
        self.assertEqual(model_meta["action_dim"], 29)

        deploy_meta = deploy_payload["meta_payload"]["visualizer_metadata"]
        self.assertEqual(deploy_meta["deploy_episode_outcomes"]["success_count"], 1)
        self.assertEqual(deploy_meta["deploy_episode_outcomes"]["failed_count"], 1)
        self.assertEqual(deploy_meta["deploy_episode_outcomes"]["unmarked_count"], 1)
        self.assertEqual(deploy_meta["insights"]["unmarked"], 1)
        self.assertEqual(deploy_meta["eval_dataset"]["layout"], "v3")

    def test_collect_videos_for_source_uses_local_and_hf_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vid.mp4").write_text("x", encoding="utf-8")
            local_videos = _collect_videos_for_source(
                {"scope": "local", "kind": "dataset", "path": str(root)},
                metadata=None,
            )
        self.assertEqual([item["relative_path"] for item in local_videos], ["vid.mp4"])

        with patch("robot_pipeline_app.visualizer_utils._discover_hf_dataset_videos", return_value=[{"relative_path": "clip.mp4"}]) as mocked:
            hf_videos = _collect_videos_for_source(
                {"scope": "huggingface", "kind": "dataset", "repo_id": "alice/repo"},
                metadata={"siblings": []},
            )
        self.assertEqual(hf_videos, [{"relative_path": "clip.mp4"}])
        mocked.assert_called_once_with("alice/repo", {"siblings": []})

        self.assertEqual(
            _collect_videos_for_source({"scope": "huggingface", "kind": "model", "repo_id": "alice/model"}, metadata={"siblings": []}),
            [],
        )

    def test_collect_sources_for_refresh_combines_local_and_hf_datasets(self) -> None:
        snapshot = _VisualizerRefreshSnapshot(
            source="datasets",
            deploy_root="/tmp/deploy-root",
            dataset_root="/tmp/dataset-root",
            model_root="/tmp/model-root",
            hf_owner="alice",
        )
        with (
            patch("robot_pipeline_app.visualizer_utils._collect_dataset_sources", return_value=[{"name": "local_ds", "scope": "local"}]) as mocked_local,
            patch("robot_pipeline_app.visualizer_utils._collect_hf_dataset_sources", return_value=([{"name": "alice/ds", "scope": "huggingface"}], None)) as mocked_hf,
        ):
            rows, error_text, source_kind = _collect_sources_for_refresh(config={}, snapshot=snapshot)
        mocked_local.assert_called_once_with({}, data_root=Path("/tmp/dataset-root"))
        mocked_hf.assert_called_once_with("alice")
        self.assertEqual(rows, [{"name": "local_ds", "scope": "local"}, {"name": "alice/ds", "scope": "huggingface"}])
        self.assertIsNone(error_text)
        self.assertEqual(source_kind, "dataset sources")

    def test_collect_sources_for_refresh_skips_hf_when_owner_missing(self) -> None:
        snapshot = _VisualizerRefreshSnapshot(
            source="datasets",
            deploy_root="/tmp/deploy-root",
            dataset_root="/tmp/dataset-root",
            model_root="/tmp/model-root",
            hf_owner="",
        )
        with (
            patch("robot_pipeline_app.visualizer_utils._collect_dataset_sources", return_value=[{"name": "local_ds", "scope": "local"}]) as mocked_local,
            patch("robot_pipeline_app.visualizer_utils._collect_hf_dataset_sources") as mocked_hf,
        ):
            rows, error_text, source_kind = _collect_sources_for_refresh(config={}, snapshot=snapshot)
        mocked_local.assert_called_once_with({}, data_root=Path("/tmp/dataset-root"))
        mocked_hf.assert_not_called()
        self.assertEqual(rows, [{"name": "local_ds", "scope": "local"}])
        self.assertIsNone(error_text)
        self.assertEqual(source_kind, "dataset sources")

    def test_collect_sources_for_refresh_combines_local_and_hf_models(self) -> None:
        snapshot = _VisualizerRefreshSnapshot(
            source="models",
            deploy_root="/tmp/deploy-root",
            dataset_root="/tmp/dataset-root",
            model_root="/tmp/model-root",
            hf_owner="alice",
        )
        with (
            patch("robot_pipeline_app.visualizer_utils._collect_model_sources", return_value=[{"name": "m-local", "scope": "local"}]) as mocked_local,
            patch("robot_pipeline_app.visualizer_utils._collect_hf_model_sources", return_value=([{"name": "alice/m1", "scope": "huggingface"}], None)) as mocked_hf,
        ):
            rows, error_text, source_kind = _collect_sources_for_refresh(config={}, snapshot=snapshot)
        mocked_local.assert_called_once_with({}, model_root=Path("/tmp/model-root"))
        mocked_hf.assert_called_once_with("alice")
        self.assertEqual(rows, [{"name": "m-local", "scope": "local"}, {"name": "alice/m1", "scope": "huggingface"}])
        self.assertIsNone(error_text)
        self.assertEqual(source_kind, "model sources")


if __name__ == "__main__":
    unittest.main()
