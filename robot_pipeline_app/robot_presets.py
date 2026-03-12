from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobotPreset:
    label: str
    follower_robot_type: str
    leader_robot_type: str
    action_dim: int
    description: str = ""

    def apply_payload(self) -> dict[str, Any]:
        return {
            "follower_robot_type": self.follower_robot_type,
            "leader_robot_type": self.leader_robot_type,
            "follower_robot_action_dim": int(self.action_dim),
        }


_ROBOT_PRESETS: tuple[RobotPreset, ...] = (
    RobotPreset(
        label="SO-100",
        follower_robot_type="so100_follower",
        leader_robot_type="so100_leader",
        action_dim=6,
        description="SO-100 leader/follower pair with 6-DoF action output.",
    ),
    RobotPreset(
        label="SO-101",
        follower_robot_type="so101_follower",
        leader_robot_type="so101_leader",
        action_dim=6,
        description="SO-101 leader/follower pair with 6-DoF action output.",
    ),
    RobotPreset(
        label="Unitree G1 (29 DOF)",
        follower_robot_type="unitree_g1_29dof",
        leader_robot_type="unitree_g1_29dof",
        action_dim=29,
        description="Unitree G1 full-body preset with 29-DoF action output.",
    ),
    RobotPreset(
        label="Unitree G1 (23 DOF)",
        follower_robot_type="unitree_g1_23dof",
        leader_robot_type="unitree_g1_23dof",
        action_dim=23,
        description="Unitree G1 reduced-body preset with 23-DoF action output.",
    ),
)


def robot_presets() -> tuple[RobotPreset, ...]:
    return _ROBOT_PRESETS


def robot_preset_labels() -> list[str]:
    return [preset.label for preset in _ROBOT_PRESETS]


def robot_preset_payload(label: str) -> dict[str, Any] | None:
    selected = str(label or "").strip()
    for preset in _ROBOT_PRESETS:
        if preset.label == selected:
            return preset.apply_payload()
    return None


def robot_type_options(*values: str) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for preset in _ROBOT_PRESETS:
        for candidate in (preset.follower_robot_type, preset.leader_robot_type):
            if candidate in seen:
                continue
            seen.add(candidate)
            options.append(candidate)
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        options.append(normalized)
    return options
