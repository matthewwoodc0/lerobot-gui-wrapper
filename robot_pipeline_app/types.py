from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

CheckResult = tuple[str, str, str]
AppConfig = dict[str, Any]
GuiCompleteCallback = Callable[[int, bool], None]


class GuiRunProcessAsync(Protocol):
    def __call__(
        self,
        cmd: list[str],
        cwd: Path | None,
        complete_callback: GuiCompleteCallback | None = None,
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
        run_mode: str = "run",
        preflight_checks: list[CheckResult] | None = None,
        artifact_context: dict[str, Any] | None = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class RecordRequest:
    dataset_repo_id: str
    dataset_name: str
    dataset_root: Path
    num_episodes: int
    episode_time_s: int
    task: str
    upload_after_record: bool


@dataclass(frozen=True)
class DeployRequest:
    model_path: Path
    eval_repo_id: str
    eval_num_episodes: int
    eval_duration_s: int
    eval_task: str


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None
    canceled: bool
    output_lines: list[str]
    artifact_path: Path | None


@dataclass(frozen=True)
class PreflightReport:
    checks: list[CheckResult]
    pass_count: int
    warn_count: int
    fail_count: int

    @property
    def has_failures(self) -> bool:
        return self.fail_count > 0
