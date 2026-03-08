from __future__ import annotations

from dataclasses import dataclass


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


VALIDATED_LEROBOT_TRACKS: tuple[ValidatedLeRobotTrack, ...] = (
    ValidatedLeRobotTrack(
        key="current",
        label="validated current track",
        version_spec="0.4.x",
        status_date="2026-03-07",
    ),
    ValidatedLeRobotTrack(
        key="n_minus_1",
        label="validated N-1 track",
        version_spec="0.3.x",
        status_date="2026-03-07",
    ),
)

WORKFLOW_PASS_GATE_NOTE = (
    "CI verifies compatibility probing and tooling only; workflow PASS status is granted only after the GA manual hardware gate."
)

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
TRAINING_COMMAND_NOTE = "The generated command uses the detected LeRobot train entrypoint for your environment."


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
