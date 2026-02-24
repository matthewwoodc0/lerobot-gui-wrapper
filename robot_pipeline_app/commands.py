from __future__ import annotations

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
) -> list[str]:
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
    if policy_path is not None:
        cmd.append(f"--policy.path={policy_path}")
    return cmd
