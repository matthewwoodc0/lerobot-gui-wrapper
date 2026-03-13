from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .command_overrides import get_flag_value
from .dataset_tools import build_merge_datasets_command
from .hardware_workflows import (
    ReplayRequest,
    build_replay_preflight_checks,
    build_replay_readiness_summary,
    build_replay_request_and_command,
    discover_replay_episodes,
)

DatasetEditCommandBuilder = Callable[[dict[str, Any], str, list[int]], list[str]]


@dataclass(frozen=True)
class DatasetOperationError:
    title: str
    text: str
    log_message: str


@dataclass(frozen=True)
class DatasetEditPlan:
    operation_name: str
    repo_id: str
    selected_indices: tuple[int, ...]
    command_argv: list[str]
    artifact_context: dict[str, Any]
    start_log: str


@dataclass(frozen=True)
class MergeDatasetsPlan:
    output_repo_id: str
    source_repo_ids: tuple[str, ...]
    command_argv: list[str]
    artifact_context: dict[str, Any]
    start_log: str


@dataclass(frozen=True)
class ReplaySelectionPlan:
    repo_id: str
    dataset_path: str
    episode_choices: list[str]
    selected_value: str
    helper_text: str


@dataclass(frozen=True)
class ReplayLaunchPlan:
    request: ReplayRequest
    command_argv: list[str]
    preflight_checks: list[tuple[str, str, str]]
    preflight_text: str
    artifact_context: dict[str, Any]
    start_log: str


class DatasetOperationService:
    def _command_dataset_repo_id(self, argv: Sequence[str], fallback: str = "") -> str:
        return (
            get_flag_value(list(argv), "repo_id")
            or get_flag_value(list(argv), "dataset.repo_id")
            or str(fallback).strip()
        )

    def _run_dataset_edit_operation(
        self,
        *,
        config: dict[str, Any],
        repo_id: str,
        selected_indices: Sequence[int],
        operation_name: str,
        command_builder: DatasetEditCommandBuilder,
    ) -> DatasetEditPlan | DatasetOperationError:
        cleaned_repo_id = str(repo_id).strip()
        if not cleaned_repo_id:
            return DatasetOperationError(
                title="Dataset Required",
                text="Enter a dataset repo id before running dataset tools.",
                log_message="Visualizer dataset tools launch failed validation.",
            )

        normalized_indices = tuple(int(index) for index in selected_indices)
        if not normalized_indices:
            return DatasetOperationError(
                title="No Episodes Selected",
                text="Select at least one episode first.",
                log_message="Visualizer dataset tools launch skipped with no selected episodes.",
            )

        return DatasetEditPlan(
            operation_name=operation_name,
            repo_id=cleaned_repo_id,
            selected_indices=normalized_indices,
            command_argv=command_builder(config, cleaned_repo_id, list(normalized_indices)),
            artifact_context={"dataset_repo_id": cleaned_repo_id},
            start_log=f"Visualizer {operation_name.lower()} starting for {cleaned_repo_id}.",
        )

    def merge_datasets(
        self,
        *,
        config: dict[str, Any],
        output_repo_id: str,
        source_repo_ids: Sequence[str],
    ) -> MergeDatasetsPlan | DatasetOperationError:
        cleaned_output_repo_id = str(output_repo_id).strip()
        if not cleaned_output_repo_id:
            return DatasetOperationError(
                title="Output Dataset Required",
                text="Enter the output dataset repo id before merging.",
                log_message="Visualizer merge-datasets launch failed validation.",
            )

        normalized_sources = tuple(str(repo_id).strip() for repo_id in source_repo_ids if str(repo_id).strip())
        if len(normalized_sources) < 2:
            return DatasetOperationError(
                title="Source Datasets Required",
                text="Enter at least two source dataset repo ids to merge.",
                log_message="Visualizer merge-datasets launch failed validation.",
            )

        return MergeDatasetsPlan(
            output_repo_id=cleaned_output_repo_id,
            source_repo_ids=normalized_sources,
            command_argv=build_merge_datasets_command(config, cleaned_output_repo_id, list(normalized_sources)),
            artifact_context={"dataset_repo_id": cleaned_output_repo_id},
            start_log=f"Visualizer merge datasets starting for {cleaned_output_repo_id}.",
        )

    def build_replay_selection(
        self,
        *,
        config: dict[str, Any],
        dataset_repo_id: str,
        dataset_path: str = "",
        selected_episode: int = 0,
    ) -> ReplaySelectionPlan | DatasetOperationError:
        repo_id = str(dataset_repo_id).strip()
        if not repo_id:
            return DatasetOperationError(
                title="Dataset Required",
                text="Enter a dataset repo id before replaying an episode on hardware.",
                log_message="Visualizer replay launch failed validation.",
            )

        discovery = discover_replay_episodes(config, repo_id, dataset_path_raw=dataset_path)
        return ReplaySelectionPlan(
            repo_id=repo_id,
            dataset_path=str(dataset_path).strip(),
            episode_choices=[str(index) for index in discovery.episode_indices[:500]] or ["0"],
            selected_value=str(int(selected_episode)),
            helper_text=discovery.scan_error
            or "Use a discovered local episode when possible. Manual override is available if the list is incomplete.",
        )

    def replay_dataset_episode(
        self,
        *,
        config: dict[str, Any],
        dataset_repo_id: str,
        episode_raw: str,
        dataset_path: str = "",
    ) -> ReplayLaunchPlan | DatasetOperationError:
        request, cmd, support, error = build_replay_request_and_command(
            config=config,
            dataset_repo_id=dataset_repo_id,
            episode_raw=episode_raw,
            dataset_path_raw=dataset_path,
        )
        if error or request is None or cmd is None:
            return DatasetOperationError(
                title="Replay Unavailable",
                text=error or support.detail,
                log_message="Visualizer replay launch failed validation.",
            )

        checks = build_replay_preflight_checks(config=config, request=request, support=support)
        return ReplayLaunchPlan(
            request=request,
            command_argv=cmd,
            preflight_checks=checks,
            preflight_text=build_replay_readiness_summary(config=config, request=request, support=support)
            + "\n\n"
            + "\n".join(f"[{level}] {name}: {detail}" for level, name, detail in checks)
            + "\n\nClick Confirm to continue, or Cancel to stop.",
            artifact_context={
                "dataset_repo_id": request.dataset_repo_id,
                "dataset_path": str(request.dataset_path) if request.dataset_path is not None else "",
                "replay_episode": request.episode_index,
            },
            start_log=(
                f"Visualizer replay launch starting for {request.dataset_repo_id} "
                f"episode {request.episode_index}."
            ),
        )
