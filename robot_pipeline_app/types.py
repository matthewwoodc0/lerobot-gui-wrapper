from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CheckResult = tuple[str, str, str]
AppConfig = dict[str, Any]


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
