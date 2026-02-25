from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .probes import parse_frame_dimensions, probe_camera_capture


def _resolve_camera_dimensions(
    config: dict[str, Any],
    role: str,
    index: int,
    default_width: int,
    default_height: int,
) -> tuple[int, int]:
    _ = (config, role)
    opened, detail = probe_camera_capture(index, default_width, default_height)
    parsed = parse_frame_dimensions(detail)
    if opened and parsed is not None:
        return parsed

    return default_width, default_height


def camera_arg(config: dict[str, Any]) -> str:
    laptop = int(config["camera_laptop_index"])
    phone = int(config["camera_phone_index"])
    warmup = int(config.get("camera_warmup_s", 5))
    width = 640
    height = 360
    fps = int(config.get("camera_fps", 30))
    laptop_width, laptop_height = _resolve_camera_dimensions(config, "laptop", laptop, width, height)
    phone_width, phone_height = _resolve_camera_dimensions(config, "phone", phone, width, height)
    cameras = {
        "laptop": {
            "type": "opencv",
            "index_or_path": laptop,
            "width": laptop_width,
            "height": laptop_height,
            "fps": fps,
            "warmup_s": warmup,
        },
        "phone": {
            "type": "opencv",
            "index_or_path": phone,
            "width": phone_width,
            "height": phone_height,
            "fps": fps,
            "warmup_s": warmup,
        },
    }
    return json.dumps(cameras, separators=(",", ":"))


def build_lerobot_record_command(
    config: dict[str, Any],
    dataset_repo_id: str,
    num_episodes: int,
    task: str,
    episode_time: int,
    policy_path: Path | None = None,
    include_warmup_time_s: bool | None = None,
) -> list[str]:
    warmup_s = int(config.get("camera_warmup_s", 5))
    if include_warmup_time_s is None:
        include_warmup_time_s = policy_path is None
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type=so101_follower",
        f"--robot.port={config['follower_port']}",
        "--robot.id=red4",
        f"--robot.cameras={camera_arg(config)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={config['leader_port']}",
        "--teleop.id=white",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time}",
    ]
    if include_warmup_time_s:
        cmd.append(f"--warmup_time_s={warmup_s}")
    if policy_path is not None:
        cmd.append(f"--policy.path={policy_path}")
    return cmd


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _resolve_teleop_entrypoint(config: dict[str, Any]) -> tuple[str, bool]:
    lerobot_dir_value = str(config.get("lerobot_dir", "")).strip()
    lerobot_dir = Path(lerobot_dir_value).expanduser() if lerobot_dir_value else None

    # Source checkout layout (works when cwd is the LeRobot root).
    if lerobot_dir is not None:
        if (lerobot_dir / "scripts" / "lerobot_teleoperate.py").exists():
            return "scripts.lerobot_teleoperate", False
        if (lerobot_dir / "lerobot" / "scripts" / "lerobot_teleoperate.py").exists():
            return "lerobot.scripts.lerobot_teleoperate", False

    # Installed package layouts.
    if _module_available("lerobot.teleoperate"):
        return "lerobot.teleoperate", False
    if _module_available("lerobot.scripts.lerobot_teleoperate"):
        return "lerobot.scripts.lerobot_teleoperate", False

    # Legacy LeRobot fallback.
    if _module_available("lerobot.scripts.control_robot"):
        return "lerobot.scripts.control_robot", True

    if lerobot_dir is not None:
        if (lerobot_dir / "lerobot" / "scripts" / "control_robot.py").exists():
            return "lerobot.scripts.control_robot", True
        if (lerobot_dir / "scripts" / "control_robot.py").exists():
            return "scripts.control_robot", True

    # Default to modern package entrypoint; preflight/setup will surface missing lerobot installs.
    return "lerobot.teleoperate", False


def build_lerobot_teleop_command(
    config: dict[str, Any],
    *,
    follower_robot_id: str = "red4",
    leader_robot_id: str = "white",
    control_fps: int | None = None,
) -> list[str]:
    module_name, use_legacy_control = _resolve_teleop_entrypoint(config)
    cmd = [
        sys.executable,
        "-m",
        module_name,
    ]
    if use_legacy_control:
        cmd.append("--control.type=teleoperate")
    cmd.extend(
        [
            "--robot.type=so101_follower",
            f"--robot.port={config['follower_port']}",
            f"--robot.cameras={camera_arg(config)}" if use_legacy_control else "--robot.cameras={}",
            f"--robot.id={follower_robot_id}",
            "--teleop.type=so101_leader",
            f"--teleop.port={config['leader_port']}",
            f"--teleop.id={leader_robot_id}",
        ]
    )
    if use_legacy_control and control_fps is not None:
        cmd.append(f"--control.fps={control_fps}")
    return cmd
