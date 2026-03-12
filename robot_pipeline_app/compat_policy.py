from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ValidatedLeRobotTrack:
    key: str
    label: str
    version_spec: str
    status_date: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "version_spec": self.version_spec,
            "status_date": self.status_date,
        }


@dataclass(frozen=True)
class PythonCompatibility:
    requirement: str
    status: str
    detail: str
    hard_fail: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "requirement": self.requirement,
            "status": self.status,
            "detail": self.detail,
            "hard_fail": self.hard_fail,
        }


VALIDATED_LEROBOT_TRACKS: tuple[ValidatedLeRobotTrack, ...] = (
    ValidatedLeRobotTrack(
        key="current",
        label="validated current track",
        version_spec="0.5.x",
        status_date="2026-03-10",
    ),
    ValidatedLeRobotTrack(
        key="n_minus_1",
        label="validated N-1 track",
        version_spec="0.4.x",
        status_date="2026-03-10",
    ),
)

WORKFLOW_PASS_GATE_NOTE = (
    "CI verifies compatibility probing and tooling only; workflow PASS status is granted only after the GA manual hardware gate."
)
WRAPPER_PYTHON_BASELINE = (3, 12)
WRAPPER_PYTHON_BASELINE_TEXT = "3.12+"

TRAIN_REQUIRED_FLAGS: tuple[str, ...] = (
    "policy.path",
    "policy.input_features",
    "policy.output_features",
    "dataset.repo_id",
    "batch_size",
    "steps",
    "output_dir",
    "job_name",
    "policy.device",
    "wandb.enable",
    "policy.push_to_hub",
    "save_freq",
)

TRAINING_COMMAND_LABEL = "LeRobot training command"
TRAINING_COMMAND_EXAMPLE = "python -m lerobot.train"
TRAINING_COMMAND_NOTE = "The generated command uses the configured LeRobot runtime and detected train entrypoint for your environment."


def _parse_numeric_version_parts(raw_version: str, *, limit: int) -> tuple[int, ...] | None:
    matches = re.findall(r"\d+", str(raw_version or "").strip())
    if not matches:
        return None
    return tuple(int(item) for item in matches[:limit])


def _format_python_version(parts: tuple[int, ...]) -> str:
    if not parts:
        return "unknown"
    return ".".join(str(part) for part in parts)


def lerobot_requires_python_3_12(version: str) -> bool:
    parts = _parse_numeric_version_parts(version, limit=2)
    if parts is None or len(parts) < 2:
        return False
    return parts[0] > 0 or (parts[0] == 0 and parts[1] >= 5)


def python_requirement_for_lerobot(version: str) -> str:
    _ = version
    return f"Python {WRAPPER_PYTHON_BASELINE_TEXT}"


def evaluate_python_compatibility(
    lerobot_version: str,
    python_version_info: tuple[int, int, int] | None,
) -> PythonCompatibility:
    requirement = python_requirement_for_lerobot(lerobot_version)
    if python_version_info is None:
        return PythonCompatibility(
            requirement=requirement,
            status="WARN",
            detail=f"Unable to detect the active Python version. Expected {requirement}.",
            hard_fail=False,
        )

    active_text = _format_python_version(python_version_info)
    minimum_text = _format_python_version(WRAPPER_PYTHON_BASELINE)
    if python_version_info >= WRAPPER_PYTHON_BASELINE:
        return PythonCompatibility(
            requirement=requirement,
            status="PASS",
            detail=f"Active Python {active_text} satisfies the wrapper baseline ({minimum_text}).",
            hard_fail=False,
        )

    if lerobot_requires_python_3_12(lerobot_version):
        return PythonCompatibility(
            requirement=requirement,
            status="FAIL",
            detail=(
                f"LeRobot {lerobot_version} requires Python {minimum_text}+; "
                f"active Python is {active_text}."
            ),
            hard_fail=True,
        )

    if str(lerobot_version or "").strip() and str(lerobot_version).strip() != "unknown":
        detail = (
            f"Active Python {active_text} is below the wrapper baseline ({minimum_text}+). "
            f"LeRobot {lerobot_version} may still run, but this wrapper is only validated on {minimum_text}+."
        )
    else:
        detail = (
            f"Active Python {active_text} is below the wrapper baseline ({minimum_text}+). "
            "Detected LeRobot version is unknown, so 0.5.x compatibility cannot be confirmed."
        )
    return PythonCompatibility(
        requirement=requirement,
        status="WARN",
        detail=detail,
        hard_fail=False,
    )


def compatibility_policy_display(raw_policy: str) -> str:
    normalized = str(raw_policy or "").strip()
    if normalized in {"", "latest_plus_n_minus_1"}:
        return "validated current + N-1"
    return normalized


def validated_tracks_payload() -> list[dict[str, str]]:
    return [track.to_dict() for track in VALIDATED_LEROBOT_TRACKS]


def validated_tracks_summary() -> str:
    return ", ".join(
        f"{track.label}={track.version_spec} ({track.status_date})"
        for track in VALIDATED_LEROBOT_TRACKS
    )


def match_validated_track(version: str) -> ValidatedLeRobotTrack | None:
    raw_version = str(version or "").strip()
    if not raw_version or raw_version == "unknown":
        return None

    for track in VALIDATED_LEROBOT_TRACKS:
        prefix = track.version_spec.removesuffix(".x")
        if raw_version == prefix or raw_version.startswith(prefix + "."):
            return track
    return None
