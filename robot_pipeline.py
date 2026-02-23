#!/usr/bin/env python3
"""LeRobot Local Pipeline Manager for SO-101 data recording and deployment."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_LEROBOT_DIR = Path.home() / "lerobot"
PRIMARY_CONFIG_PATH = Path.home() / ".robot_config.json"
DEFAULT_SECONDARY_CONFIG_PATH = DEFAULT_LEROBOT_DIR / ".robot_config.json"
LEGACY_CONFIG_PATH = Path.home() / ".robot_pipeline_config.json"
SFTP_BATCH_PATH = Path("/tmp/sftp_batch.txt")

DEFAULT_TASK = "Pick up the white block and place it in the bin"

DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "lerobot_dir": str(DEFAULT_LEROBOT_DIR),
    "record_data_dir": str(DEFAULT_LEROBOT_DIR / "data"),
    "trained_models_dir": str(DEFAULT_LEROBOT_DIR / "trained_models"),
    "hf_username": "matthewwoodc0",
    "last_dataset_name": "matthew_20",
    "follower_port": "/dev/ttyACM1",
    "leader_port": "/dev/ttyACM0",
    "camera_laptop_index": 4,
    "camera_phone_index": 6,
    "olympus_user": "matthew.woodc0",
    "olympus_host": "olympus.hprc.tamu.edu",
    "olympus_scratch": "/mnt/shared-scratch/Shakkottai_S/matthewwoodc0/lerobot",
    "last_model_name": "",
    "last_checkpoint_steps": "100000",
}

CONFIG_FIELDS = [
    {"key": "lerobot_dir", "prompt": "LeRobot folder path", "type": "path"},
    {
        "key": "record_data_dir",
        "prompt": "Local dataset save folder",
        "type": "path",
    },
    {
        "key": "trained_models_dir",
        "prompt": "Local trained models folder",
        "type": "path",
    },
    {"key": "hf_username", "prompt": "HuggingFace username", "type": "str"},
    {"key": "last_dataset_name", "prompt": "Last dataset name", "type": "str"},
    {"key": "follower_port", "prompt": "Follower port", "type": "str"},
    {"key": "leader_port", "prompt": "Leader port", "type": "str"},
    {"key": "camera_laptop_index", "prompt": "Laptop camera index", "type": "int"},
    {"key": "camera_phone_index", "prompt": "Phone camera index", "type": "int"},
    {"key": "olympus_user", "prompt": "Olympus username", "type": "str"},
    {"key": "olympus_host", "prompt": "Olympus host", "type": "str"},
    {"key": "olympus_scratch", "prompt": "Olympus scratch path", "type": "str"},
    {
        "key": "last_model_name",
        "prompt": "Last model name (optional)",
        "type": "str",
    },
    {
        "key": "last_checkpoint_steps",
        "prompt": "Last checkpoint steps",
        "type": "str",
    },
]


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalize_path(value: str | Path) -> str:
    return str(Path(value).expanduser())


def prompt_text(label: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return str(default)
        print("Please enter a value.")


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = prompt_text(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def pick_directory(initial_dir: str | None = None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("Folder picker is unavailable. Enter a path manually.")
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    selected = filedialog.askdirectory(
        initialdir=initial_dir or str(Path.home()),
        title="Select folder",
    )
    root.destroy()

    if selected:
        return normalize_path(selected)
    return None


def prompt_path(label: str, default: str) -> str:
    while True:
        prompt = f"{label} [{default}] (Enter=default, b=browse): "
        raw = input(prompt).strip()
        if not raw:
            return normalize_path(default)

        if raw.lower() in {"b", "browse"}:
            chosen = pick_directory(default)
            if chosen:
                print(f"Selected folder: {chosen}")
                return chosen
            print("No folder selected. Try again.")
            continue

        return normalize_path(raw)


def prompt_yes_no(label: str, default: str = "y") -> bool:
    value = prompt_text(f"{label} (y/n)", default).lower()
    return value in {"y", "yes"}



def get_secondary_config_path(config: dict[str, Any]) -> Path:
    lerobot_dir = normalize_path(config.get("lerobot_dir", DEFAULT_LEROBOT_DIR))
    return Path(lerobot_dir) / ".robot_config.json"



def load_raw_config() -> tuple[dict[str, Any], Path | None]:
    candidate_paths = [
        PRIMARY_CONFIG_PATH,
        DEFAULT_SECONDARY_CONFIG_PATH,
        LEGACY_CONFIG_PATH,
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data, path
            except (json.JSONDecodeError, OSError):
                pass

    return {}, None



def save_config(config: dict[str, Any]) -> None:
    payload = json.dumps(config, indent=2) + "\n"

    PRIMARY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRIMARY_CONFIG_PATH.write_text(payload, encoding="utf-8")

    secondary_path = get_secondary_config_path(config)
    try:
        secondary_path.parent.mkdir(parents=True, exist_ok=True)
        secondary_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write secondary config file {secondary_path}: {exc}")

    print(f"Saved config to {PRIMARY_CONFIG_PATH}")
    print("Done! ✓")



def default_for_key(key: str, config: dict[str, Any]) -> Any:
    if key == "record_data_dir":
        lerobot_dir = normalize_path(config.get("lerobot_dir", DEFAULT_CONFIG_VALUES["lerobot_dir"]))
        return str(Path(lerobot_dir) / "data")

    if key == "trained_models_dir":
        lerobot_dir = normalize_path(config.get("lerobot_dir", DEFAULT_CONFIG_VALUES["lerobot_dir"]))
        return str(Path(lerobot_dir) / "trained_models")

    return DEFAULT_CONFIG_VALUES[key]



def ensure_config(config: dict[str, Any], force_prompt_all: bool = False) -> dict[str, Any]:
    updated = False

    for field in CONFIG_FIELDS:
        key = field["key"]
        field_type = field["type"]
        default = default_for_key(key, config)

        has_value = key in config and config[key] not in (None, "")
        if force_prompt_all or not has_value:
            if field_type == "int":
                config[key] = prompt_int(field["prompt"], int(default))
            elif field_type == "path":
                config[key] = prompt_path(field["prompt"], normalize_path(str(default)))
            else:
                config[key] = prompt_text(field["prompt"], str(default))
            updated = True
            continue

        if field_type == "int":
            try:
                config[key] = int(config[key])
            except (TypeError, ValueError):
                config[key] = prompt_int(field["prompt"], int(default))
                updated = True
        elif field_type == "path":
            config[key] = normalize_path(str(config[key]))
        else:
            config[key] = str(config[key])

    if updated:
        save_config(config)

    return config



def increment_dataset_name(name: str) -> str:
    match = re.search(r"^(.*?)(\d+)$", name)
    if not match:
        return f"{name}_1"

    prefix, number = match.groups()
    next_number = int(number) + 1
    return f"{prefix}{next_number}"



def dataset_exists_on_hf(repo_id: str) -> bool | None:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = request.Request(url=url, method="GET")

    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except error.HTTPError as exc:
        if exc.code == 404:
            return False
        return None
    except error.URLError:
        return None



def suggest_dataset_name(config: dict[str, Any]) -> tuple[str, bool]:
    last_used = str(config.get("last_dataset_name", "dataset_1"))
    candidate = increment_dataset_name(last_used)

    username = str(config["hf_username"])
    checked_remote = False

    for _ in range(25):
        exists = dataset_exists_on_hf(f"{username}/{candidate}")
        if exists is None:
            return candidate, checked_remote
        checked_remote = True
        if not exists:
            return candidate, checked_remote
        candidate = increment_dataset_name(candidate)

    return candidate, checked_remote



def camera_arg(config: dict[str, Any]) -> str:
    laptop = int(config["camera_laptop_index"])
    phone = int(config["camera_phone_index"])
    return (
        "{laptop: {type: opencv, index_or_path: "
        + str(laptop)
        + ", width: 640, height: 480, fps: 30}, "
        "phone:{type: opencv, index_or_path: "
        + str(phone)
        + ", width: 640, height: 480, fps: 30}}"
    )



def show_command(cmd: list[str]) -> None:
    print("\nFull command:")
    print(shlex.join(cmd))



def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    show_command(cmd)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        if cmd[0] == "huggingface-cli":
            print("Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate")
        elif cmd[0] == "sftp":
            print("Install OpenSSH client tools and make sure 'sftp' is available.")
        return None

    return result



def get_lerobot_dir(config: dict[str, Any]) -> Path:
    return Path(normalize_path(config["lerobot_dir"]))



def run_record_mode(config: dict[str, Any]) -> None:
    print_section("=== 🎬 RECORD MODE ===")

    suggested_name, checked_remote = suggest_dataset_name(config)
    if checked_remote:
        print(f"Suggested next dataset name: {suggested_name}")
    else:
        print(f"Suggested dataset name (local increment): {suggested_name}")

    dataset_name = prompt_text("Dataset name", suggested_name)
    username = str(config["hf_username"])

    dataset_root = Path(
        prompt_path("Local dataset save folder", str(config["record_data_dir"]))
    )
    config["record_data_dir"] = str(dataset_root)

    remote_exists = dataset_exists_on_hf(f"{username}/{dataset_name}")
    if remote_exists is True:
        print(f"Warning: {username}/{dataset_name} already exists on HuggingFace.")
        if not prompt_yes_no("Continue with this dataset name?", "n"):
            print("Cancelled.")
            return

    num_episodes = prompt_int("Number of episodes", 20)
    episode_time = prompt_int("Episode duration in seconds", 20)
    task = prompt_text("Task description", DEFAULT_TASK)

    lerobot_dir = get_lerobot_dir(config)
    cmd = [
        "python",
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type=so101_follower",
        f"--robot.port={config['follower_port']}",
        "--robot.id=red4",
        f"--robot.cameras={camera_arg(config)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={config['leader_port']}",
        "--teleop.id=white",
        f"--dataset.repo_id={username}/{dataset_name}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={episode_time}",
    ]

    print("\nSummary:")
    print(f"- Dataset: {username}/{dataset_name}")
    print(f"- Episodes: {num_episodes}")
    print(f"- Episode time (s): {episode_time}")
    print(f"- Task: {task}")
    print(f"- Local dataset folder: {dataset_root}")
    show_command(cmd)

    if not prompt_yes_no("Run this recording command now?", "y"):
        print("Cancelled.")
        return

    result = run_command(cmd, cwd=lerobot_dir)
    if result is None:
        return
    if result.returncode != 0:
        print(f"Recording failed with exit code {result.returncode}.")
        return

    source_dataset = lerobot_dir / "data" / dataset_name
    target_dataset = dataset_root / dataset_name
    active_dataset_path = source_dataset

    if source_dataset.exists() and source_dataset.resolve() != target_dataset.resolve():
        target_dataset.parent.mkdir(parents=True, exist_ok=True)
        if target_dataset.exists():
            print(f"Warning: target dataset folder already exists: {target_dataset}")
            print("Keeping recorded data at original location.")
        else:
            shutil.move(str(source_dataset), str(target_dataset))
            active_dataset_path = target_dataset
            print(f"Moved dataset to: {target_dataset}")
            print("Done! ✓")
    elif target_dataset.exists():
        active_dataset_path = target_dataset

    print("Recording completed.")
    print("Done! ✓")

    config["last_dataset_name"] = dataset_name
    save_config(config)

    if not prompt_yes_no("Upload to HuggingFace now?", "n"):
        return

    upload_path = active_dataset_path if active_dataset_path.exists() else target_dataset
    upload_cmd = [
        "huggingface-cli",
        "upload",
        f"{username}/{dataset_name}",
        str(upload_path),
        "--repo-type",
        "dataset",
    ]

    upload_result = run_command(upload_cmd, cwd=lerobot_dir)
    if upload_result is None:
        return
    if upload_result.returncode != 0:
        print(f"Upload failed with exit code {upload_result.returncode}.")
        return

    print("Upload completed.")
    print("Done! ✓")



def run_deploy_mode(config: dict[str, Any]) -> None:
    print_section("=== 🚀 DEPLOY MODE ===")

    last_model = str(config.get("last_model_name", "")).strip()
    if last_model:
        model_name = prompt_text("Model name on Olympus", last_model)
    else:
        model_name = prompt_text("Model name on Olympus", None)

    last_steps = str(config.get("last_checkpoint_steps", "100000"))
    checkpoint_steps = prompt_text("Checkpoint step number", last_steps)

    models_root = Path(
        prompt_path("Local model save folder", str(config["trained_models_dir"]))
    )
    models_root.mkdir(parents=True, exist_ok=True)
    config["trained_models_dir"] = str(models_root)

    olympus_path = (
        f"{config['olympus_scratch'].rstrip('/')}/outputs/train/"
        f"{model_name}/checkpoints/{checkpoint_steps}/pretrained_model"
    )
    local_destination = models_root / f"{model_name}_{checkpoint_steps}"
    local_destination.mkdir(parents=True, exist_ok=True)

    batch_contents = "\n".join(
        [
            f"cd {olympus_path}",
            f"lcd {local_destination}",
            "get -r pretrained_model .",
            "bye",
            "",
        ]
    )
    SFTP_BATCH_PATH.write_text(batch_contents, encoding="utf-8")

    print("\nSFTP batch file:")
    print(batch_contents.rstrip())

    sftp_cmd = [
        "sftp",
        "-b",
        str(SFTP_BATCH_PATH),
        f"{config['olympus_user']}@{config['olympus_host']}",
    ]

    sftp_result = run_command(sftp_cmd, capture_output=True)
    if sftp_result is None:
        return

    if sftp_result.returncode != 0:
        print("SFTP download failed.")
        if sftp_result.stderr:
            print(sftp_result.stderr.strip())
        print("Suggestion: check VPN/SSH access and Olympus credentials.")
        return

    if sftp_result.stdout:
        print(sftp_result.stdout.strip())

    print(f"Model download completed at {local_destination}")
    print("Done! ✓")

    config["last_model_name"] = model_name
    config["last_checkpoint_steps"] = checkpoint_steps
    save_config(config)

    if not prompt_yes_no("Run deployment now?", "y"):
        return

    lerobot_dir = get_lerobot_dir(config)
    eval_cmd = [
        "python",
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={local_destination}",
        "--robot.type=so101_follower",
        f"--robot.port={config['follower_port']}",
        f"--robot.cameras={camera_arg(config)}",
    ]

    eval_result = run_command(eval_cmd, cwd=lerobot_dir)
    if eval_result is None:
        return
    if eval_result.returncode != 0:
        print(f"Deployment command failed with exit code {eval_result.returncode}.")
        return

    print("Deployment command completed.")
    print("Done! ✓")



def run_config_mode(config: dict[str, Any]) -> dict[str, Any]:
    print_section("=== ⚙️ CONFIG MODE ===")
    print("Press Enter to keep defaults shown in brackets.")

    for field in CONFIG_FIELDS:
        key = field["key"]
        default = default_for_key(key, config)
        current = config.get(key, default)

        if field["type"] == "int":
            config[key] = prompt_int(field["prompt"], int(current))
        elif field["type"] == "path":
            config[key] = prompt_path(field["prompt"], normalize_path(str(current)))
        else:
            config[key] = prompt_text(field["prompt"], str(current))

    save_config(config)
    return config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeRobot local pipeline manager for recording and deployment."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("record", help="Record teleoperated demos and optionally upload.")
    subparsers.add_parser("deploy", help="Download model from Olympus and run deployment.")
    subparsers.add_parser("config", help="Update saved config values.")

    return parser.parse_args()



def main() -> int:
    args = parse_args()

    raw_config, source = load_raw_config()
    first_run = source is None

    if first_run:
        print_section("=== 🛠️ FIRST-TIME SETUP ===")
        print(f"Config not found. Creating one at {PRIMARY_CONFIG_PATH}")
        print("You can type a path or enter 'b' to browse folders in Finder/File Manager.")

    config = ensure_config(raw_config, force_prompt_all=first_run)

    if args.mode == "config":
        run_config_mode(config)
        return 0

    if args.mode == "record":
        run_record_mode(config)
        return 0

    if args.mode == "deploy":
        run_deploy_mode(config)
        return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting safely.")
        raise SystemExit(1)
