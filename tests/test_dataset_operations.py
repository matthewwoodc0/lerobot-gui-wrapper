from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_pipeline_app.dataset_operations import (
    DatasetEditPlan,
    DatasetOperationError,
    DatasetOperationService,
    MergeDatasetsPlan,
    ReplayLaunchPlan,
    ReplaySelectionPlan,
)
from robot_pipeline_app.hardware_replay import ReplayRequest, ReplaySupport


class DatasetOperationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DatasetOperationService()

    def test_command_dataset_repo_id_prefers_repo_id_flag(self) -> None:
        repo_id = self.service._command_dataset_repo_id(
            ["python", "--dataset.repo_id=owner/fallback", "--repo_id=owner/edited"],
            "owner/original",
        )

        self.assertEqual(repo_id, "owner/edited")

    def test_run_dataset_edit_operation_returns_validation_error_without_repo(self) -> None:
        result = self.service._run_dataset_edit_operation(
            config={},
            repo_id="",
            selected_indices=[1],
            operation_name="Delete Episodes",
            command_builder=lambda *_args: ["python", "-m", "edit"],
        )

        self.assertIsInstance(result, DatasetOperationError)
        assert isinstance(result, DatasetOperationError)
        self.assertEqual(result.title, "Dataset Required")

    def test_run_dataset_edit_operation_returns_plan(self) -> None:
        result = self.service._run_dataset_edit_operation(
            config={"record_data_dir": "/tmp/data"},
            repo_id="alice/demo_dataset",
            selected_indices=[2, 4],
            operation_name="Delete Episodes",
            command_builder=lambda _config, repo_id, indices: ["python", repo_id, ",".join(str(i) for i in indices)],
        )

        self.assertIsInstance(result, DatasetEditPlan)
        assert isinstance(result, DatasetEditPlan)
        self.assertEqual(result.repo_id, "alice/demo_dataset")
        self.assertEqual(result.selected_indices, (2, 4))
        self.assertEqual(result.command_argv, ["python", "alice/demo_dataset", "2,4"])

    def test_merge_datasets_returns_validation_error_without_sources(self) -> None:
        result = self.service.merge_datasets(
            config={},
            output_repo_id="alice/merged",
            source_repo_ids=["alice/source_one"],
        )

        self.assertIsInstance(result, DatasetOperationError)
        assert isinstance(result, DatasetOperationError)
        self.assertEqual(result.title, "Source Datasets Required")

    def test_merge_datasets_rejects_duplicate_sources_after_normalization(self) -> None:
        result = self.service.merge_datasets(
            config={},
            output_repo_id="alice/merged",
            source_repo_ids=["alice/source_one", " alice/source_one/ "],
        )

        self.assertIsInstance(result, DatasetOperationError)
        assert isinstance(result, DatasetOperationError)
        self.assertEqual(result.title, "Source Datasets Required")

    def test_merge_datasets_returns_plan(self) -> None:
        with patch(
            "robot_pipeline_app.dataset_operations.build_merge_datasets_command",
            return_value=["python", "-m", "merge"],
        ) as build_command:
            result = self.service.merge_datasets(
                config={"record_data_dir": "/tmp/data"},
                output_repo_id="alice/merged",
                source_repo_ids=["alice/source_one", " alice/source_two "],
            )

        self.assertIsInstance(result, MergeDatasetsPlan)
        assert isinstance(result, MergeDatasetsPlan)
        build_command.assert_called_once_with(
            {"record_data_dir": "/tmp/data"},
            "alice/merged",
            ["alice/source_one", "alice/source_two"],
        )
        self.assertEqual(result.source_repo_ids, ("alice/source_one", "alice/source_two"))
        self.assertEqual(result.command_argv, ["python", "-m", "merge"])

    def test_build_replay_selection_returns_plan(self) -> None:
        with patch(
            "robot_pipeline_app.dataset_operations.discover_replay_episodes",
            return_value=SimpleNamespace(episode_indices=(3, 7), scan_error=None),
        ):
            result = self.service.build_replay_selection(
                config={},
                dataset_repo_id="alice/demo_dataset",
                dataset_path="/tmp/demo_dataset",
                selected_episode=7,
            )

        self.assertIsInstance(result, ReplaySelectionPlan)
        assert isinstance(result, ReplaySelectionPlan)
        self.assertEqual(result.episode_choices, ["3", "7"])
        self.assertEqual(result.selected_value, "7")

    def test_replay_dataset_episode_returns_plan(self) -> None:
        request = ReplayRequest(
            dataset_repo_id="alice/demo_dataset",
            dataset_path=Path("/tmp/demo_dataset"),
            episode_index=5,
            robot_type="so101_follower",
            robot_port="/dev/ttyUSB0",
            robot_id="robot-1",
            calibration_dir="/tmp/calibration",
        )
        support = ReplaySupport(
            available=True,
            entrypoint="lerobot.replay",
            detail="Replay ready.",
            supported_flags=("dataset.repo_id", "dataset.episode"),
            dataset_flag="dataset.repo_id",
            dataset_root_flag=None,
            dataset_path_flag=None,
            episode_flag="dataset.episode",
            robot_type_flag="robot.type",
            robot_port_flag="robot.port",
            robot_id_flag="robot.id",
            calibration_dir_flag="robot.calibration_dir",
            used_fallback_flags=False,
        )
        with (
            patch(
                "robot_pipeline_app.dataset_operations.build_replay_request_and_command",
                return_value=(request, ["python", "-m", "replay"], support, None),
            ),
            patch(
                "robot_pipeline_app.dataset_operations.build_replay_preflight_checks",
                return_value=[("PASS", "Replay episode", "Episode 5 exists locally.")],
            ),
            patch(
                "robot_pipeline_app.dataset_operations.build_replay_readiness_summary",
                return_value="Replay summary",
            ),
        ):
            result = self.service.replay_dataset_episode(
                config={"record_data_dir": "/tmp/data"},
                dataset_repo_id="alice/demo_dataset",
                episode_raw="5",
                dataset_path="/tmp/demo_dataset",
            )

        self.assertIsInstance(result, ReplayLaunchPlan)
        assert isinstance(result, ReplayLaunchPlan)
        self.assertEqual(result.command_argv, ["python", "-m", "replay"])
        self.assertEqual(result.artifact_context["replay_episode"], 5)
        self.assertIn("Replay summary", result.preflight_text)


if __name__ == "__main__":
    unittest.main()
