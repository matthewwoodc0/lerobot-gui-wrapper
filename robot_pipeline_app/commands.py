from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def camera_arg(config: dict[str, Any]) -> str:
    laptop = int(config["camera_laptop_index"])
    phone = int(config["camera_phone_index"])
    warmup = int(config.get("camera_warmup_s", 5))
    width = int(config.get("camera_width", 640))
    height = int(config.get("camera_height", 360))
    fps = int(config.get("camera_fps", 30))
    cameras = {
        "laptop": {
            "type": "opencv",
            "index_or_path": laptop,
            "width": width,
            "height": height,
            "fps": fps,
            "warmup_s": warmup,
        },
        "phone": {
            "type": "opencv",
            "index_or_path": phone,
            "width": width,
            "height": height,
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
