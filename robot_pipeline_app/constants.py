from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_LEROBOT_DIR = Path.home() / "lerobot"
DEFAULT_LEROBOT_VENV_DIR = DEFAULT_LEROBOT_DIR / "lerobot_env"
APP_ROOT_DIR = Path(__file__).resolve().parent.parent
LEGACY_DEFAULT_RUNS_DIR = Path.home() / ".robot_pipeline_runs"
DEFAULT_RUNS_DIR = APP_ROOT_DIR / ".robot_pipeline_runs"
PRIMARY_CONFIG_PATH = Path.home() / ".robot_config.json"
DEFAULT_SECONDARY_CONFIG_PATH = DEFAULT_LEROBOT_DIR / ".robot_config.json"
LEGACY_CONFIG_PATH = Path.home() / ".robot_pipeline_config.json"
DEFAULT_HF_USERNAME = ""

DEFAULT_TASK = "Pick up the white block and place it in the bin"

_DEPLOY_DATA_DIR_FALLBACK = "lerobot_datasets"


def default_deploy_data_dir(hf_username: Any) -> Path:
    owner = str(hf_username or "").strip()
    if owner:
        return Path.home() / ".cache" / "huggingface" / "lerobot" / owner
    return Path.home() / ".cache" / "huggingface" / "lerobot" / _DEPLOY_DATA_DIR_FALLBACK


DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "lerobot_dir": str(DEFAULT_LEROBOT_DIR),
    "lerobot_venv_dir": str(DEFAULT_LEROBOT_VENV_DIR),
    "runs_dir": str(DEFAULT_RUNS_DIR),
    "record_data_dir": str(DEFAULT_LEROBOT_DIR / "data"),
    "deploy_data_dir": str(default_deploy_data_dir(DEFAULT_HF_USERNAME)),
    "trained_models_dir": str(DEFAULT_LEROBOT_DIR / "trained_models"),
    "hf_username": DEFAULT_HF_USERNAME,
    "last_dataset_name": "dataset_1",
    "follower_port": "/dev/ttyACM1",
    "leader_port": "/dev/ttyACM0",
    "follower_robot_id": "red4",
    "leader_robot_id": "white",
    "camera_laptop_index": 4,
    "camera_phone_index": 6,
    "camera_warmup_s": 5,
    "camera_fps": 30,
    "eval_num_episodes": 10,
    "eval_duration_s": 20,
    "eval_task": DEFAULT_TASK,
    "last_eval_dataset_name": "",
    "last_model_name": "",
    "follower_calibration_path": "",
    "leader_calibration_path": "",
    "ui_theme_mode": "dark",
}

CONFIG_FIELDS = [
    {"key": "lerobot_dir", "prompt": "LeRobot folder path", "type": "path"},
    {"key": "lerobot_venv_dir", "prompt": "LeRobot venv folder path", "type": "path"},
    {"key": "runs_dir", "prompt": "Run artifacts folder", "type": "path"},
    {
        "key": "record_data_dir",
        "prompt": "Local dataset save folder",
        "type": "path",
    },
    {
        "key": "deploy_data_dir",
        "prompt": "Deploy dataset cache folder",
        "type": "path",
    },
    {
        "key": "trained_models_dir",
        "prompt": "Local trained models folder",
        "type": "path",
    },
    {"key": "hf_username", "prompt": "HuggingFace username", "type": "str"},
    {"key": "follower_port", "prompt": "Follower port", "type": "str"},
    {"key": "leader_port", "prompt": "Leader port", "type": "str"},
    {"key": "follower_robot_id", "prompt": "Follower robot id", "type": "str"},
    {"key": "leader_robot_id", "prompt": "Leader robot id", "type": "str"},
    {"key": "camera_laptop_index", "prompt": "Laptop camera index", "type": "int"},
    {"key": "camera_phone_index", "prompt": "Phone camera index", "type": "int"},
    {"key": "camera_warmup_s", "prompt": "Camera warmup (s)", "type": "int"},
    {"key": "camera_fps", "prompt": "Camera FPS", "type": "int"},
    {"key": "eval_num_episodes", "prompt": "Deploy eval episodes", "type": "int"},
    {"key": "eval_duration_s", "prompt": "Deploy eval episode time (s)", "type": "int"},
    {"key": "eval_task", "prompt": "Deploy eval task", "type": "str"},
    {
        "key": "follower_calibration_path",
        "prompt": "Follower calibration file (.json) — leave empty for auto-detect",
        "type": "path",
    },
    {
        "key": "leader_calibration_path",
        "prompt": "Leader calibration file (.json) — leave empty for auto-detect",
        "type": "path",
    },
    {"key": "ui_theme_mode", "prompt": "UI theme mode (dark/light)", "type": "str"},
]
