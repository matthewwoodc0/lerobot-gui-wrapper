#!/usr/bin/env python3
"""LeRobot local pipeline manager for SO-101 recording and local deployment."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_LEROBOT_DIR = Path.home() / "lerobot"
DEFAULT_RUNS_DIR = Path.home() / ".robot_pipeline_runs"
PRIMARY_CONFIG_PATH = Path.home() / ".robot_config.json"
DEFAULT_SECONDARY_CONFIG_PATH = DEFAULT_LEROBOT_DIR / ".robot_config.json"
LEGACY_CONFIG_PATH = Path.home() / ".robot_pipeline_config.json"

DEFAULT_TASK = "Pick up the white block and place it in the bin"

DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "lerobot_dir": str(DEFAULT_LEROBOT_DIR),
    "runs_dir": str(DEFAULT_RUNS_DIR),
    "record_data_dir": str(DEFAULT_LEROBOT_DIR / "data"),
    "trained_models_dir": str(DEFAULT_LEROBOT_DIR / "trained_models"),
    "hf_username": "matthewwoodc0",
    "last_dataset_name": "matthew_20",
    "follower_port": "/dev/ttyACM1",
    "leader_port": "/dev/ttyACM0",
    "camera_laptop_index": 4,
    "camera_phone_index": 6,
    "camera_warmup_s": 5,
    "camera_width": 640,
    "camera_height": 360,
    "camera_fps": 30,
    "eval_num_episodes": 10,
    "eval_duration_s": 20,
    "eval_task": DEFAULT_TASK,
    "last_eval_dataset_name": "",
    "last_model_name": "",
}

CONFIG_FIELDS = [
    {"key": "lerobot_dir", "prompt": "LeRobot folder path", "type": "path"},
    {"key": "runs_dir", "prompt": "Run artifacts folder", "type": "path"},
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
    {"key": "camera_warmup_s", "prompt": "Camera warmup (s)", "type": "int"},
    {"key": "camera_width", "prompt": "Camera width", "type": "int"},
    {"key": "camera_height", "prompt": "Camera height", "type": "int"},
    {"key": "camera_fps", "prompt": "Camera FPS", "type": "int"},
    {"key": "eval_num_episodes", "prompt": "Deploy eval episodes", "type": "int"},
    {"key": "eval_duration_s", "prompt": "Deploy eval episode time (s)", "type": "int"},
    {"key": "eval_task", "prompt": "Deploy eval task", "type": "str"},
    {"key": "last_eval_dataset_name", "prompt": "Last deploy eval dataset name", "type": "str"},
    {
        "key": "last_model_name",
        "prompt": "Last model name (optional)",
        "type": "str",
    },
]

CheckResult = tuple[str, str, str]


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


def normalize_repo_id(username: str, dataset_name_or_repo_id: str) -> str:
    name = dataset_name_or_repo_id.strip().strip("/")
    if "/" in name:
        return name
    return f"{username}/{name}"


def repo_name_from_repo_id(repo_id: str) -> str:
    clean = repo_id.strip().strip("/")
    if not clean:
        return "dataset"
    parts = clean.split("/")
    return parts[-1] if parts[-1] else "dataset"


def probe_module_import(module_name: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"exit code {result.returncode}"


def summarize_probe_error(raw_message: str) -> str:
    lines = [line.strip() for line in raw_message.splitlines() if line.strip()]
    if not lines:
        return "unknown error"
    return lines[-1]


def probe_camera_capture(index: int, width: int, height: int) -> tuple[bool, str]:
    script = (
        "import sys\n"
        "idx=int(sys.argv[1]); width=int(sys.argv[2]); height=int(sys.argv[3])\n"
        "import cv2\n"
        "cap=cv2.VideoCapture(idx)\n"
        "if cap is None or not cap.isOpened():\n"
        "    print('camera not opened')\n"
        "    raise SystemExit(2)\n"
        "cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)\n"
        "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)\n"
        "ok, frame = cap.read()\n"
        "cap.release()\n"
        "if not ok or frame is None:\n"
        "    print('camera opened but no frame')\n"
        "    raise SystemExit(3)\n"
        "h, w = frame.shape[:2]\n"
        "print(f'frame={w}x{h}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(index), str(width), str(height)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, (result.stdout or "").strip() or "camera opened"
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"camera probe failed with exit code {result.returncode}"


def suggest_eval_dataset_name(config: dict[str, Any], model_name: str = "") -> str:
    previous = str(config.get("last_eval_dataset_name", "")).strip()
    if previous:
        return increment_dataset_name(previous)

    clean_model = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name).strip("_")
    base = f"eval_{clean_model}" if clean_model else "eval_run"
    return f"{base}_1"


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
        return None

    return result



def get_lerobot_dir(config: dict[str, Any]) -> Path:
    return Path(normalize_path(config["lerobot_dir"]))


def ensure_runs_dir(config: dict[str, Any]) -> Path:
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def build_run_id(mode: str) -> str:
    safe_mode = re.sub(r"[^a-zA-Z0-9_-]+", "_", mode.strip() or "run")
    return f"{safe_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _check_counts(checks: list[CheckResult]) -> tuple[int, int, int]:
    pass_count = sum(1 for level, _, _ in checks if level == "PASS")
    warn_count = sum(1 for level, _, _ in checks if level == "WARN")
    fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
    return pass_count, warn_count, fail_count


def summarize_checks(checks: list[CheckResult], title: str = "Checks") -> str:
    pass_count, warn_count, fail_count = _check_counts(checks)
    lines = [title]
    lines.extend(f"[{level:4}] {name}: {detail}" for level, name, detail in checks)
    lines.append("")
    lines.append(f"Summary: PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return "\n".join(lines)


def has_failures(checks: list[CheckResult]) -> bool:
    return any(level == "FAIL" for level, _, _ in checks)


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current == current.parent:
            return None
        current = current.parent
    return current


def _run_common_preflight_checks(config: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def add(level: str, name: str, detail: str) -> None:
        checks.append((level, name, detail))

    lerobot_dir = get_lerobot_dir(config)
    add(
        "PASS" if lerobot_dir.exists() else "FAIL",
        "LeRobot folder",
        str(lerobot_dir),
    )

    lerobot_ok, lerobot_msg = probe_module_import("lerobot")
    add(
        "PASS" if lerobot_ok else "FAIL",
        "Python module: lerobot",
        "import ok" if lerobot_ok else summarize_probe_error(lerobot_msg),
    )

    follower_port = str(config.get("follower_port", "")).strip()
    leader_port = str(config.get("leader_port", "")).strip()
    add(
        "PASS" if follower_port and Path(follower_port).exists() else "WARN",
        "Follower port",
        follower_port or "(empty)",
    )
    add(
        "PASS" if leader_port and Path(leader_port).exists() else "WARN",
        "Leader port",
        leader_port or "(empty)",
    )

    laptop_idx = int(config.get("camera_laptop_index", 0))
    phone_idx = int(config.get("camera_phone_index", 1))
    add(
        "PASS" if laptop_idx != phone_idx else "WARN",
        "Camera indices",
        f"laptop={laptop_idx}, phone={phone_idx}",
    )

    cv2_ok, cv2_msg = probe_module_import("cv2")
    add(
        "PASS" if cv2_ok else "WARN",
        "Python module: cv2",
        "import ok" if cv2_ok else summarize_probe_error(cv2_msg),
    )
    if cv2_ok:
        width = int(config.get("camera_width", 640))
        height = int(config.get("camera_height", 360))
        laptop_open, laptop_detail = probe_camera_capture(laptop_idx, width, height)
        phone_open, phone_detail = probe_camera_capture(phone_idx, width, height)
        add(
            "PASS" if laptop_open else "WARN",
            f"Camera {laptop_idx} probe",
            laptop_detail,
        )
        add(
            "PASS" if phone_open else "WARN",
            f"Camera {phone_idx} probe",
            phone_detail,
        )

    return checks


def run_preflight_for_record(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
) -> list[CheckResult]:
    checks = _run_common_preflight_checks(config)

    if dataset_root.exists() and not dataset_root.is_dir():
        checks.append(("FAIL", "Dataset root folder", f"Not a directory: {dataset_root}"))
    else:
        probe_path = dataset_root if dataset_root.exists() else _nearest_existing_parent(dataset_root)
        if probe_path is None:
            checks.append(("FAIL", "Dataset root folder", f"No existing parent for path: {dataset_root}"))
        elif not os.access(str(probe_path), os.W_OK):
            checks.append(("FAIL", "Dataset root folder", f"No write permission for: {probe_path}"))
        else:
            checks.append(("PASS", "Dataset root folder", f"Writable path available via: {probe_path}"))

    if upload_enabled:
        hf_cli = shutil.which("huggingface-cli")
        checks.append(
            (
                "PASS" if hf_cli else "FAIL",
                "huggingface-cli",
                hf_cli or "not found in PATH",
            )
        )

    return checks


def run_preflight_for_deploy(config: dict[str, Any], model_path: Path) -> list[CheckResult]:
    checks = _run_common_preflight_checks(config)
    checks.append(
        (
            "PASS" if model_path.exists() and model_path.is_dir() else "FAIL",
            "Model folder",
            str(model_path),
        )
    )
    return checks


def write_run_artifacts(
    config: dict[str, Any],
    mode: str,
    command: list[str] | str,
    cwd: Path | str | None,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int | None,
    canceled: bool,
    preflight_checks: list[CheckResult] | None,
    output_lines: list[str] | str,
    dataset_repo_id: str | None = None,
    model_path: Path | str | None = None,
    run_id: str | None = None,
) -> Path | None:
    try:
        runs_dir = ensure_runs_dir(config)
    except OSError:
        return None

    created_run_id = run_id or build_run_id(mode)
    run_path = runs_dir / created_run_id
    suffix = 1
    while run_path.exists():
        run_path = runs_dir / f"{created_run_id}_{suffix}"
        suffix += 1

    try:
        run_path.mkdir(parents=True, exist_ok=False)
    except OSError:
        return None

    command_text = shlex.join(command) if isinstance(command, list) else str(command)
    cwd_text = str(cwd) if cwd is not None else ""

    if isinstance(output_lines, list):
        log_text = "\n".join(output_lines)
    else:
        log_text = str(output_lines)
    if log_text and not log_text.endswith("\n"):
        log_text += "\n"

    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=timezone.utc)

    pass_count, warn_count, fail_count = _check_counts(preflight_checks or [])
    metadata = {
        "run_id": run_path.name,
        "mode": mode,
        "command": command_text,
        "cwd": cwd_text,
        "started_at_iso": started_at.isoformat(),
        "ended_at_iso": ended_at.isoformat(),
        "duration_s": round(max((ended_at - started_at).total_seconds(), 0.0), 3),
        "exit_code": exit_code,
        "canceled": bool(canceled),
        "preflight_fail_count": fail_count,
        "preflight_warn_count": warn_count,
        "preflight_pass_count": pass_count,
        "dataset_repo_id": dataset_repo_id,
        "model_path": str(model_path) if model_path is not None else None,
    }

    try:
        (run_path / "command.log").write_text(log_text, encoding="utf-8")
        (run_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return None

    return run_path


def list_runs(config: dict[str, Any], limit: int = 15) -> tuple[list[dict[str, Any]], int]:
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    if not runs_dir.exists() or not runs_dir.is_dir():
        return [], 0

    warning_count = 0
    runs: list[dict[str, Any]] = []

    def parse_iso(raw: Any) -> datetime:
        text = str(raw or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warning_count += 1
            continue

        if not isinstance(data, dict):
            warning_count += 1
            continue

        data["_metadata_path"] = str(metadata_path)
        data["_run_path"] = str(metadata_path.parent)
        runs.append(data)

    runs.sort(key=lambda item: parse_iso(item.get("started_at_iso")), reverse=True)

    if limit > 0:
        runs = runs[:limit]

    return runs, warning_count


def run_history_mode(config: dict[str, Any], limit: int = 15) -> None:
    print_section("=== 📜 HISTORY MODE ===")
    runs, warning_count = list_runs(config, limit=limit)

    if not runs:
        print("No run artifacts found yet.")
        runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
        print(f"Runs folder: {runs_dir}")
        if warning_count:
            print(f"Skipped unreadable metadata files: {warning_count}")
        return

    print("Time                | Mode    | Exit      | Duration | Hint")
    print("-" * 88)
    for item in runs:
        started = str(item.get("started_at_iso", "")).replace("T", " ")[:19]
        mode = str(item.get("mode", "run"))[:7].ljust(7)
        canceled = bool(item.get("canceled", False))
        exit_code = item.get("exit_code")
        exit_text = "CANCELED" if canceled else f"{exit_code}"
        duration = f"{float(item.get('duration_s', 0.0)):.1f}s"
        hint = str(item.get("dataset_repo_id") or item.get("model_path") or "-")
        if hint not in {"-", ""} and "/" in hint and not hint.count("/") == 1:
            hint = Path(hint).name
        print(f"{started:19} | {mode:7} | {exit_text:9} | {duration:8} | {hint}")

    if warning_count:
        print(f"\nSkipped unreadable metadata files: {warning_count}")


def collect_doctor_checks(config: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def add(level: str, name: str, detail: str) -> None:
        checks.append((level, name, detail))

    add("PASS", "Python", f"{sys.executable} ({sys.version.split()[0]})")

    lerobot_dir = get_lerobot_dir(config)
    add(
        "PASS" if lerobot_dir.exists() else "FAIL",
        "LeRobot folder",
        str(lerobot_dir),
    )

    record_data_dir = Path(normalize_path(config.get("record_data_dir", "")))
    add(
        "PASS" if record_data_dir.exists() else "WARN",
        "Record data dir",
        str(record_data_dir),
    )

    models_dir = Path(normalize_path(config.get("trained_models_dir", "")))
    add(
        "PASS" if models_dir.exists() else "WARN",
        "Trained models dir",
        str(models_dir),
    )

    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    add(
        "PASS" if runs_dir.exists() else "WARN",
        "Runs artifacts dir",
        str(runs_dir),
    )

    lerobot_ok, lerobot_msg = probe_module_import("lerobot")
    add(
        "PASS" if lerobot_ok else "FAIL",
        "Python module: lerobot",
        "import ok" if lerobot_ok else summarize_probe_error(lerobot_msg),
    )

    huggingface_cli = shutil.which("huggingface-cli")
    add(
        "PASS" if huggingface_cli else "WARN",
        "huggingface-cli",
        huggingface_cli or "not found in PATH",
    )

    follower_port = str(config.get("follower_port", "")).strip()
    leader_port = str(config.get("leader_port", "")).strip()
    add(
        "PASS" if follower_port and Path(follower_port).exists() else "WARN",
        "Follower port",
        follower_port or "(empty)",
    )
    add(
        "PASS" if leader_port and Path(leader_port).exists() else "WARN",
        "Leader port",
        leader_port or "(empty)",
    )

    laptop_idx = int(config.get("camera_laptop_index", 0))
    phone_idx = int(config.get("camera_phone_index", 1))
    add(
        "PASS" if laptop_idx != phone_idx else "WARN",
        "Camera indices",
        f"laptop={laptop_idx}, phone={phone_idx}",
    )

    cv2_ok, cv2_msg = probe_module_import("cv2")
    add(
        "PASS" if cv2_ok else "WARN",
        "Python module: cv2",
        "import ok" if cv2_ok else summarize_probe_error(cv2_msg),
    )

    if cv2_ok:
        width = int(config.get("camera_width", 640))
        height = int(config.get("camera_height", 360))
        laptop_open, laptop_detail = probe_camera_capture(laptop_idx, width, height)
        phone_open, phone_detail = probe_camera_capture(phone_idx, width, height)
        add(
            "PASS" if laptop_open else "WARN",
            f"Camera {laptop_idx} probe",
            laptop_detail,
        )
        add(
            "PASS" if phone_open else "WARN",
            f"Camera {phone_idx} probe",
            phone_detail,
        )

    return checks


def run_doctor_mode(config: dict[str, Any]) -> None:
    print_section("=== 🩺 DOCTOR MODE ===")
    checks = collect_doctor_checks(config)

    for level, name, detail in checks:
        print(f"[{level:4}] {name}: {detail}")

    fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
    warn_count = sum(1 for level, _, _ in checks if level == "WARN")
    pass_count = sum(1 for level, _, _ in checks if level == "PASS")

    print("\nSummary:")
    print(f"- PASS: {pass_count}")
    print(f"- WARN: {warn_count}")
    print(f"- FAIL: {fail_count}")
    if fail_count == 0:
        print("Doctor completed. No hard blockers found.")
    else:
        print("Doctor found blockers. Resolve FAIL items first.")



def run_record_mode(config: dict[str, Any]) -> None:
    print_section("=== 🎬 RECORD MODE ===")

    suggested_name, checked_remote = suggest_dataset_name(config)
    if checked_remote:
        print(f"Suggested next dataset name: {suggested_name}")
    else:
        print(f"Suggested dataset name (local increment): {suggested_name}")

    dataset_input = prompt_text("Dataset name (or full repo id)", suggested_name)
    username = str(config["hf_username"])
    dataset_repo_id = normalize_repo_id(username, dataset_input)
    dataset_name = repo_name_from_repo_id(dataset_repo_id)

    dataset_root = Path(
        prompt_path("Local dataset save folder", str(config["record_data_dir"]))
    )
    config["record_data_dir"] = str(dataset_root)

    remote_exists = dataset_exists_on_hf(dataset_repo_id)
    if remote_exists is True:
        print(f"Warning: {dataset_repo_id} already exists on HuggingFace.")
        if not prompt_yes_no("Continue with this dataset name?", "n"):
            print("Cancelled.")
            return

    num_episodes = prompt_int("Number of episodes", 20)
    episode_time = prompt_int("Episode duration in seconds", 20)
    task = prompt_text("Task description", DEFAULT_TASK)

    lerobot_dir = get_lerobot_dir(config)
    cmd = build_lerobot_record_command(
        config=config,
        dataset_repo_id=dataset_repo_id,
        num_episodes=num_episodes,
        task=task,
        episode_time=episode_time,
    )

    print("\nSummary:")
    print(f"- Dataset: {dataset_repo_id}")
    print(f"- Episodes: {num_episodes}")
    print(f"- Episode time (s): {episode_time}")
    print(f"- Task: {task}")
    print(f"- Local dataset folder: {dataset_root}")
    show_command(cmd)

    upload_after_record = prompt_yes_no("Upload to HuggingFace after recording?", "n")

    preflight_checks = run_preflight_for_record(
        config=config,
        dataset_root=dataset_root,
        upload_enabled=upload_after_record,
    )
    print("\n" + summarize_checks(preflight_checks, title="Preflight"))
    if has_failures(preflight_checks):
        if not prompt_yes_no("Continue despite preflight FAILs?", "n"):
            print("Cancelled.")
            return

    if not prompt_yes_no("Run this recording command now?", "y"):
        print("Cancelled.")
        return

    run_started = datetime.now(timezone.utc)
    run_output_lines = ["$ " + shlex.join(cmd)]
    try:
        result = run_command(cmd, cwd=lerobot_dir, capture_output=True)
    except KeyboardInterrupt:
        run_ended = datetime.now(timezone.utc)
        run_output_lines.append("Interrupted by user.")
        artifact_path = write_run_artifacts(
            config=config,
            mode="record",
            command=cmd,
            cwd=lerobot_dir,
            started_at=run_started,
            ended_at=run_ended,
            exit_code=None,
            canceled=True,
            preflight_checks=preflight_checks,
            output_lines=run_output_lines,
            dataset_repo_id=dataset_repo_id,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        print("Interrupted by user.")
        return
    run_ended = datetime.now(timezone.utc)
    if result is None:
        run_output_lines.append(f"Command not found: {cmd[0]}")
        artifact_path = write_run_artifacts(
            config=config,
            mode="record",
            command=cmd,
            cwd=lerobot_dir,
            started_at=run_started,
            ended_at=run_ended,
            exit_code=-1,
            canceled=False,
            preflight_checks=preflight_checks,
            output_lines=run_output_lines,
            dataset_repo_id=dataset_repo_id,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        return
    if result.stdout:
        print(result.stdout, end="")
        run_output_lines.extend(result.stdout.splitlines())
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
        run_output_lines.extend(result.stderr.splitlines())
    run_output_lines.append(f"[exit code {result.returncode}]")

    artifact_path = write_run_artifacts(
        config=config,
        mode="record",
        command=cmd,
        cwd=lerobot_dir,
        started_at=run_started,
        ended_at=run_ended,
        exit_code=result.returncode,
        canceled=False,
        preflight_checks=preflight_checks,
        output_lines=run_output_lines,
        dataset_repo_id=dataset_repo_id,
    )
    if artifact_path is not None:
        print(f"Run artifacts saved: {artifact_path}")

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

    if not upload_after_record:
        return

    upload_path = active_dataset_path if active_dataset_path.exists() else target_dataset
    upload_cmd = [
        "huggingface-cli",
        "upload",
        dataset_repo_id,
        str(upload_path),
        "--repo-type",
        "dataset",
    ]

    upload_started = datetime.now(timezone.utc)
    upload_output_lines = ["$ " + shlex.join(upload_cmd)]
    try:
        upload_result = run_command(upload_cmd, cwd=lerobot_dir, capture_output=True)
    except KeyboardInterrupt:
        upload_ended = datetime.now(timezone.utc)
        upload_output_lines.append("Interrupted by user.")
        artifact_path = write_run_artifacts(
            config=config,
            mode="upload",
            command=upload_cmd,
            cwd=lerobot_dir,
            started_at=upload_started,
            ended_at=upload_ended,
            exit_code=None,
            canceled=True,
            preflight_checks=[],
            output_lines=upload_output_lines,
            dataset_repo_id=dataset_repo_id,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        print("Interrupted by user.")
        return
    upload_ended = datetime.now(timezone.utc)
    if upload_result is None:
        upload_output_lines.append(f"Command not found: {upload_cmd[0]}")
        artifact_path = write_run_artifacts(
            config=config,
            mode="upload",
            command=upload_cmd,
            cwd=lerobot_dir,
            started_at=upload_started,
            ended_at=upload_ended,
            exit_code=-1,
            canceled=False,
            preflight_checks=[],
            output_lines=upload_output_lines,
            dataset_repo_id=dataset_repo_id,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        return
    if upload_result.stdout:
        print(upload_result.stdout, end="")
        upload_output_lines.extend(upload_result.stdout.splitlines())
    if upload_result.stderr:
        print(upload_result.stderr, end="", file=sys.stderr)
        upload_output_lines.extend(upload_result.stderr.splitlines())
    upload_output_lines.append(f"[exit code {upload_result.returncode}]")

    artifact_path = write_run_artifacts(
        config=config,
        mode="upload",
        command=upload_cmd,
        cwd=lerobot_dir,
        started_at=upload_started,
        ended_at=upload_ended,
        exit_code=upload_result.returncode,
        canceled=False,
        preflight_checks=[],
        output_lines=upload_output_lines,
        dataset_repo_id=dataset_repo_id,
    )
    if artifact_path is not None:
        print(f"Run artifacts saved: {artifact_path}")

    if upload_result.returncode != 0:
        print(f"Upload failed with exit code {upload_result.returncode}.")
        return

    print("Upload completed.")
    print("Done! ✓")



def run_deploy_mode(config: dict[str, Any]) -> None:
    print_section("=== 🚀 DEPLOY MODE ===")

    models_root = Path(
        prompt_path("Local model save folder", str(config["trained_models_dir"]))
    )
    models_root.mkdir(parents=True, exist_ok=True)
    config["trained_models_dir"] = str(models_root)

    available_models = sorted(p.name for p in models_root.iterdir() if p.is_dir())
    if available_models:
        print("Available local model folders:")
        for name in available_models:
            print(f"- {name}")
    else:
        print("No model folders found in that directory yet.")

    last_model = str(config.get("last_model_name", "")).strip()
    default_model_path = str(models_root / last_model) if last_model else str(models_root)
    model_path = Path(prompt_path("Local model folder to deploy", default_model_path))
    if not model_path.exists() or not model_path.is_dir():
        print(f"Model folder does not exist: {model_path}")
        return

    eval_dataset_name = prompt_text(
        "Eval dataset name (or full repo id)",
        suggest_eval_dataset_name(config, model_path.name),
    )
    eval_repo_id = normalize_repo_id(str(config["hf_username"]), eval_dataset_name)
    eval_num_episodes = prompt_int(
        "Deploy eval episodes",
        int(config.get("eval_num_episodes", 10)),
    )
    eval_duration = prompt_int(
        "Deploy eval episode duration (s)",
        int(config.get("eval_duration_s", 20)),
    )
    eval_task = prompt_text(
        "Deploy eval task description",
        str(config.get("eval_task", DEFAULT_TASK)),
    )

    remote_exists = dataset_exists_on_hf(eval_repo_id)
    if remote_exists is True:
        print(f"Warning: {eval_repo_id} already exists on HuggingFace.")
        if not prompt_yes_no("Continue and append new eval episodes anyway?", "n"):
            print("Cancelled.")
            return

    config["last_model_name"] = model_path.name
    config["eval_num_episodes"] = eval_num_episodes
    config["eval_duration_s"] = eval_duration
    config["eval_task"] = eval_task
    config["last_eval_dataset_name"] = eval_repo_id.split("/", 1)[1]
    save_config(config)

    lerobot_dir = get_lerobot_dir(config)
    eval_cmd = build_lerobot_record_command(
        config=config,
        dataset_repo_id=eval_repo_id,
        num_episodes=eval_num_episodes,
        task=eval_task,
        episode_time=eval_duration,
        policy_path=model_path,
    )

    print("\nDeploy summary:")
    print(f"- Model path: {model_path}")
    print(f"- Eval dataset: {eval_repo_id}")
    print(f"- Episodes: {eval_num_episodes}")
    print(f"- Episode time (s): {eval_duration}")
    print(f"- Task: {eval_task}")

    preflight_checks = run_preflight_for_deploy(config=config, model_path=model_path)
    print("\n" + summarize_checks(preflight_checks, title="Preflight"))
    if has_failures(preflight_checks):
        if not prompt_yes_no("Continue despite preflight FAILs?", "n"):
            print("Cancelled.")
            return

    if not prompt_yes_no("Run deployment now?", "y"):
        return

    eval_started = datetime.now(timezone.utc)
    eval_output_lines = ["$ " + shlex.join(eval_cmd)]
    try:
        eval_result = run_command(eval_cmd, cwd=lerobot_dir, capture_output=True)
    except KeyboardInterrupt:
        eval_ended = datetime.now(timezone.utc)
        eval_output_lines.append("Interrupted by user.")
        artifact_path = write_run_artifacts(
            config=config,
            mode="deploy",
            command=eval_cmd,
            cwd=lerobot_dir,
            started_at=eval_started,
            ended_at=eval_ended,
            exit_code=None,
            canceled=True,
            preflight_checks=preflight_checks,
            output_lines=eval_output_lines,
            dataset_repo_id=eval_repo_id,
            model_path=model_path,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        print("Interrupted by user.")
        return
    eval_ended = datetime.now(timezone.utc)
    if eval_result is None:
        eval_output_lines.append(f"Command not found: {eval_cmd[0]}")
        artifact_path = write_run_artifacts(
            config=config,
            mode="deploy",
            command=eval_cmd,
            cwd=lerobot_dir,
            started_at=eval_started,
            ended_at=eval_ended,
            exit_code=-1,
            canceled=False,
            preflight_checks=preflight_checks,
            output_lines=eval_output_lines,
            dataset_repo_id=eval_repo_id,
            model_path=model_path,
        )
        if artifact_path is not None:
            print(f"Run artifacts saved: {artifact_path}")
        return
    if eval_result.stdout:
        print(eval_result.stdout, end="")
        eval_output_lines.extend(eval_result.stdout.splitlines())
    if eval_result.stderr:
        print(eval_result.stderr, end="", file=sys.stderr)
        eval_output_lines.extend(eval_result.stderr.splitlines())
    eval_output_lines.append(f"[exit code {eval_result.returncode}]")

    artifact_path = write_run_artifacts(
        config=config,
        mode="deploy",
        command=eval_cmd,
        cwd=lerobot_dir,
        started_at=eval_started,
        ended_at=eval_ended,
        exit_code=eval_result.returncode,
        canceled=False,
        preflight_checks=preflight_checks,
        output_lines=eval_output_lines,
        dataset_repo_id=eval_repo_id,
        model_path=model_path,
    )
    if artifact_path is not None:
        print(f"Run artifacts saved: {artifact_path}")

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



def normalize_config_without_prompts(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)

    for field in CONFIG_FIELDS:
        key = field["key"]
        field_type = field["type"]
        default = default_for_key(key, normalized)
        value = normalized.get(key, default)
        if value in (None, ""):
            value = default

        if field_type == "int":
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                normalized[key] = int(default)
        elif field_type == "path":
            normalized[key] = normalize_path(str(value))
        else:
            normalized[key] = str(value)

    return normalized



def run_gui_mode(raw_config: dict[str, Any]) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception as exc:
        print("Tkinter GUI is unavailable on this device.")
        print(f"Details: {exc}")
        return

    if "LEROBOT_DISABLE_CAMERA_PREVIEW" in os.environ:
        cv2_probe_ok, cv2_probe_error = False, "Disabled by LEROBOT_DISABLE_CAMERA_PREVIEW"
    else:
        cv2_probe_ok, cv2_probe_error = probe_module_import("cv2")

    config = normalize_config_without_prompts(raw_config)
    if not raw_config:
        save_config(config)

    root = tk.Tk()
    root.title("LeRobot Pipeline Manager")
    root.geometry("1240x900")
    root.minsize(1080, 760)

    colors = {
        "bg": "#0b1220",
        "panel": "#111a2e",
        "header": "#111827",
        "border": "#273449",
        "text": "#e2e8f0",
        "muted": "#9ca3af",
        "accent": "#0ea5e9",
        "accent_dark": "#0284c7",
        "running": "#f59e0b",
        "ready": "#22c55e",
        "error": "#ef4444",
    }
    root.configure(bg=colors["bg"])

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Panel.TFrame", background=colors["bg"])
    style.configure("Section.TLabelframe", background=colors["panel"], bordercolor=colors["border"])
    style.configure(
        "Section.TLabelframe.Label",
        background=colors["panel"],
        foreground=colors["text"],
        font=("Helvetica", 11, "bold"),
    )
    style.configure("Field.TLabel", background=colors["panel"], foreground=colors["text"], font=("Helvetica", 10))
    style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"], font=("Helvetica", 10))
    style.configure("SectionTitle.TLabel", background=colors["bg"], foreground=colors["text"], font=("Helvetica", 12, "bold"))
    style.configure("TNotebook", background=colors["bg"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=colors["panel"],
        foreground=colors["muted"],
        padding=(14, 8),
        font=("Helvetica", 11, "bold"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["accent"]), ("active", colors["accent_dark"])],
        foreground=[("selected", "#ffffff"), ("active", "#ffffff")],
    )
    style.configure("Accent.TButton", padding=(12, 6), font=("Helvetica", 10, "bold"))
    style.map(
        "Accent.TButton",
        background=[("active", colors["accent_dark"]), ("!disabled", colors["accent"])],
        foreground=[("!disabled", "#ffffff")],
    )
    style.configure("Accent.Horizontal.TProgressbar", troughcolor="#1f2937", bordercolor="#1f2937", background=colors["accent"])
    style.configure("Time.Horizontal.TProgressbar", troughcolor="#1f2937", bordercolor="#1f2937", background="#34d399")

    status_var = tk.StringVar(value="Ready.")
    header_subtitle_var = tk.StringVar()
    hf_var = tk.StringVar()
    running_state: dict[str, Any] = {"active": False, "process": None, "cancel_requested": False}
    action_buttons: list[ttk.Button] = []
    last_command_state: dict[str, str] = {"value": ""}
    progress_state: dict[str, Any] = {
        "timer_job": None,
        "start_time": None,
        "expected_seconds": 0.0,
        "episodes_total": 0,
    }

    header_bar = tk.Frame(root, bg=colors["header"], padx=14, pady=10)
    header_bar.pack(fill="x")

    title_frame = tk.Frame(header_bar, bg=colors["header"])
    title_frame.pack(side="left", fill="x", expand=True)
    tk.Label(
        title_frame,
        text="LeRobot Pipeline Manager",
        fg=colors["text"],
        bg=colors["header"],
        font=("Helvetica", 20, "bold"),
    ).pack(anchor="w")
    tk.Label(
        title_frame,
        textvariable=header_subtitle_var,
        fg=colors["muted"],
        bg=colors["header"],
        font=("Helvetica", 10),
    ).pack(anchor="w")

    status_frame = tk.Frame(header_bar, bg=colors["header"])
    status_frame.pack(side="right", anchor="e")
    status_dot_canvas = tk.Canvas(status_frame, width=16, height=16, bg=colors["header"], highlightthickness=0)
    status_dot_canvas.grid(row=0, column=0, padx=(0, 6))
    status_dot = status_dot_canvas.create_oval(2, 2, 14, 14, fill=colors["ready"], outline=colors["ready"])
    tk.Label(status_frame, textvariable=status_var, fg=colors["text"], bg=colors["header"], font=("Helvetica", 11, "bold")).grid(
        row=0, column=1, sticky="w"
    )
    tk.Label(status_frame, textvariable=hf_var, fg=colors["muted"], bg=colors["header"], font=("Helvetica", 9)).grid(
        row=1, column=0, columnspan=2, sticky="e"
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=12, pady=(10, 8))
    record_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
    deploy_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
    config_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
    notebook.add(record_tab, text="Record")
    notebook.add(deploy_tab, text="Deploy")
    notebook.add(config_tab, text="Config")

    output_panel = tk.Frame(root, bg=colors["bg"], padx=12, pady=(0, 12))
    output_panel.pack(fill="both", expand=False)

    output_header = tk.Frame(output_panel, bg=colors["bg"])
    output_header.pack(fill="x", pady=(0, 6))
    ttk.Label(output_header, text="Terminal Output", style="SectionTitle.TLabel").pack(side="left")

    progress_wrap = tk.Frame(
        output_panel,
        bg=colors["panel"],
        highlightthickness=1,
        highlightbackground=colors["border"],
        padx=10,
        pady=8,
    )
    progress_wrap.pack(fill="x", pady=(0, 6))

    episode_progress_var = tk.StringVar(value="Episode progress: --/--")
    time_progress_var = tk.StringVar(value="Run time: 00:00 / --:--")
    tk.Label(progress_wrap, textvariable=episode_progress_var, fg=colors["text"], bg=colors["panel"], font=("Helvetica", 10)).grid(
        row=0, column=0, sticky="w"
    )
    tk.Label(progress_wrap, textvariable=time_progress_var, fg=colors["muted"], bg=colors["panel"], font=("Helvetica", 10)).grid(
        row=0, column=1, sticky="e"
    )
    episode_progressbar = ttk.Progressbar(progress_wrap, mode="determinate", style="Accent.Horizontal.TProgressbar")
    episode_progressbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
    time_progressbar = ttk.Progressbar(progress_wrap, mode="determinate", style="Time.Horizontal.TProgressbar")
    time_progressbar.grid(row=2, column=0, columnspan=2, sticky="ew")
    progress_wrap.columnconfigure(0, weight=1)
    progress_wrap.columnconfigure(1, weight=1)
    episode_progressbar.configure(maximum=1, value=0)
    time_progressbar.configure(maximum=1, value=0)

    log_box = scrolledtext.ScrolledText(
        output_panel,
        height=18,
        state="disabled",
        bg="#111827",
        fg="#d4d4d4",
        insertbackground="#f8fafc",
        font=("Menlo", 11),
        relief="flat",
        padx=10,
        pady=10,
    )
    log_box.pack(fill="both", expand=True)
    log_box.tag_configure("default", foreground="#d4d4d4")
    log_box.tag_configure("cmd", foreground="#fbbf24")
    log_box.tag_configure("error", foreground="#f87171")
    log_box.tag_configure("success", foreground="#4ade80")
    log_box.tag_configure("episode", foreground="#38bdf8")

    def clear_log() -> None:
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")

    def save_log_to_file() -> None:
        default_name = f"lerobot_gui_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        save_path = filedialog.asksaveasfilename(
            title="Save Terminal Log",
            defaultextension=".log",
            initialfile=default_name,
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not save_path:
            return
        try:
            text = log_box.get("1.0", "end-1c")
            Path(save_path).write_text(text, encoding="utf-8")
            append_log(f"Saved log to {save_path}")
        except OSError as exc:
            messagebox.showerror("Save Log Failed", str(exc))

    def copy_last_command() -> None:
        cmd_text = last_command_state["value"]
        if not cmd_text:
            messagebox.showinfo("Copy Command", "No command has been run yet.")
            return
        root.clipboard_clear()
        root.clipboard_append(cmd_text)
        append_log("Copied last command to clipboard.")

    def cancel_active_run() -> None:
        if not running_state.get("active"):
            messagebox.showinfo("Cancel", "No active command is running.")
            return
        process = running_state.get("process")
        if process is None or process.poll() is not None:
            append_log("Cancel requested, but no active process handle was available.")
            return
        running_state["cancel_requested"] = True
        append_log("Cancel requested. Sending terminate signal...")
        try:
            process.terminate()
        except Exception as exc:
            append_log(f"Terminate failed: {exc}")

    clear_log_button = ttk.Button(output_header, text="Clear Log", command=clear_log)
    clear_log_button.pack(side="right")
    save_log_button = ttk.Button(output_header, text="Save Log", command=save_log_to_file)
    save_log_button.pack(side="right", padx=(6, 0))
    copy_command_button = ttk.Button(output_header, text="Copy Last Command", command=copy_last_command)
    copy_command_button.pack(side="right", padx=(6, 0))
    cancel_run_button = ttk.Button(output_header, text="Cancel Run", command=cancel_active_run)
    cancel_run_button.pack(side="right", padx=(6, 0))
    cancel_run_button.configure(state="disabled")

    def refresh_header_subtitle() -> None:
        header_subtitle_var.set(
            "Follower {follower} | Leader {leader} | Cameras: laptop idx {laptop}, phone idx {phone} @ {w}x{h} {fps}fps".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                w=config.get("camera_width", 640),
                h=config.get("camera_height", 360),
                fps=config.get("camera_fps", 30),
            )
        )
        hf_var.set(f"Hugging Face: {config['hf_username']}")

    def set_status_dot(color: str) -> None:
        status_dot_canvas.itemconfig(status_dot, fill=color, outline=color)

    def classify_log_tag(line: str) -> str:
        lowered = line.lower()
        if line.startswith("$ "):
            return "cmd"
        if "exit code" in lowered and "[exit code 0]" not in lowered:
            return "error"
        if any(word in lowered for word in ("error", "failed", "traceback", "exception")):
            return "error"
        if any(word in lowered for word in ("completed", "done", "success")):
            return "success"
        if "episode" in lowered:
            return "episode"
        return "default"

    def append_log(line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        tag = classify_log_tag(line)
        log_box.configure(state="normal")
        log_box.insert("end", f"[{timestamp}] {line}\n", (tag,))
        log_box.see("end")
        log_box.configure(state="disabled")

    def format_seconds(value: float) -> str:
        seconds = max(int(value), 0)
        minutes, sec = divmod(seconds, 60)
        return f"{minutes:02d}:{sec:02d}"

    episode_patterns = [
        re.compile(r"[Ee]pisode\s+(\d+)\s*/\s*(\d+)"),
        re.compile(r"[Ee]pisode\s+(\d+)\s+of\s+(\d+)"),
    ]
    episode_partial_pattern = re.compile(r"[Ee]pisode\s+(\d+)")

    def update_episode_progress(current: int, total: int | None = None) -> None:
        if total is not None and total > 0:
            progress_state["episodes_total"] = total
        total_value = int(progress_state["episodes_total"] or 0)
        if total_value > 0:
            episode_progressbar.configure(maximum=total_value)
            episode_progressbar["value"] = min(current, total_value)
            episode_progress_var.set(f"Episode progress: {min(current, total_value)} / {total_value}")
        else:
            episode_progressbar.configure(maximum=max(current, 1))
            episode_progressbar["value"] = current
            episode_progress_var.set(f"Episode progress: {current} / --")

    def update_progress_from_line(line: str) -> None:
        for pattern in episode_patterns:
            match = pattern.search(line)
            if match:
                update_episode_progress(int(match.group(1)), int(match.group(2)))
                return
        partial = episode_partial_pattern.search(line)
        if partial:
            update_episode_progress(int(partial.group(1)))

    def confirm_preflight_in_gui(title: str, checks: list[CheckResult]) -> bool:
        summary = summarize_checks(checks, title=title)
        if has_failures(checks):
            return messagebox.askyesno(
                "Preflight Failures",
                summary + "\n\nFAIL items detected. Continue anyway?",
            )
        messagebox.showinfo("Preflight", summary)
        return True

    def progress_tick() -> None:
        if not running_state["active"]:
            progress_state["timer_job"] = None
            return
        start_time = progress_state.get("start_time")
        if start_time is not None:
            elapsed = time.monotonic() - float(start_time)
            expected = float(progress_state.get("expected_seconds", 0.0))
            if expected > 0:
                time_progressbar.configure(maximum=max(expected, 1))
                time_progressbar["value"] = min(elapsed, expected)
                time_progress_var.set(f"Run time: {format_seconds(elapsed)} / {format_seconds(expected)}")
            else:
                time_progressbar.configure(maximum=max(elapsed, 1))
                time_progressbar["value"] = elapsed
                time_progress_var.set(f"Run time: {format_seconds(elapsed)} / --:--")

        progress_state["timer_job"] = root.after(500, progress_tick)

    def prepare_progress(expected_episodes: int | None, expected_seconds: int | None) -> None:
        timer_job = progress_state.get("timer_job")
        if timer_job is not None:
            root.after_cancel(timer_job)
            progress_state["timer_job"] = None

        progress_state["start_time"] = time.monotonic()
        progress_state["expected_seconds"] = float(expected_seconds or 0)
        progress_state["episodes_total"] = int(expected_episodes or 0)

        if progress_state["episodes_total"] > 0:
            total = int(progress_state["episodes_total"])
            episode_progressbar.configure(maximum=total, value=0)
            episode_progress_var.set(f"Episode progress: 0 / {total}")
        else:
            episode_progressbar.configure(maximum=1, value=0)
            episode_progress_var.set("Episode progress: --/--")

        expected = float(progress_state["expected_seconds"])
        if expected > 0:
            time_progressbar.configure(maximum=max(expected, 1), value=0)
            time_progress_var.set(f"Run time: 00:00 / {format_seconds(expected)}")
        else:
            time_progressbar.configure(maximum=1, value=0)
            time_progress_var.set("Run time: 00:00 / --:--")

        progress_state["timer_job"] = root.after(500, progress_tick)

    def set_running(active: bool, status_text: str | None = None, is_error: bool = False) -> None:
        running_state["active"] = active
        if active:
            status_var.set(status_text or "Running command...")
            set_status_dot(colors["running"])
            cancel_run_button.configure(state="normal")
        else:
            if is_error:
                status_var.set(status_text or "Last command failed.")
                set_status_dot(colors["error"])
            else:
                status_var.set(status_text or "Ready.")
                set_status_dot(colors["ready"])
            timer_job = progress_state.get("timer_job")
            if timer_job is not None:
                root.after_cancel(timer_job)
                progress_state["timer_job"] = None
            running_state["process"] = None
            running_state["cancel_requested"] = False
            cancel_run_button.configure(state="disabled")

        for button in action_buttons:
            button.configure(state="disabled" if active else "normal")

    def run_process_async(
        cmd: list[str],
        cwd: Path | None,
        complete_callback: Any | None = None,
        expected_episodes: int | None = None,
        expected_seconds: int | None = None,
        run_mode: str = "run",
        preflight_checks: list[CheckResult] | None = None,
        artifact_context: dict[str, Any] | None = None,
    ) -> None:
        if running_state["active"]:
            messagebox.showinfo("Busy", "Another command is already running.")
            return

        checks = preflight_checks or []
        context = artifact_context or {}
        prepare_progress(expected_episodes, expected_seconds)
        running_state["cancel_requested"] = False
        set_running(True, "Running command...")
        command_text = shlex.join(cmd)
        last_command_state["value"] = command_text
        append_log("$ " + command_text)
        run_id = build_run_id(run_mode)
        run_started = datetime.now(timezone.utc)
        run_output_lines: list[str] = [f"$ {command_text}"]

        def persist_artifacts(exit_code: int | None, canceled: bool) -> None:
            run_ended = datetime.now(timezone.utc)
            artifact_path = write_run_artifacts(
                config=config,
                mode=run_mode,
                command=cmd,
                cwd=cwd,
                started_at=run_started,
                ended_at=run_ended,
                exit_code=exit_code,
                canceled=canceled,
                preflight_checks=checks,
                output_lines=run_output_lines,
                dataset_repo_id=context.get("dataset_repo_id"),
                model_path=context.get("model_path"),
                run_id=run_id,
            )
            if artifact_path is not None:
                root.after(0, append_log, f"Run artifacts saved: {artifact_path}")

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                root.after(0, append_log, f"Command not found: {cmd[0]}")
                run_output_lines.append(f"Command not found: {cmd[0]}")
                if cmd[0] == "huggingface-cli":
                    root.after(
                        0,
                        append_log,
                        "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate",
                    )
                    run_output_lines.append(
                        "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate"
                    )
                persist_artifacts(exit_code=-1, canceled=False)
                root.after(0, set_running, False, "Command failed to start.", True)
                root.after(0, messagebox.showerror, "Command Error", str(exc))
                return

            running_state["process"] = process
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\n")
                    run_output_lines.append(line)
                    root.after(0, append_log, line)
                    root.after(0, update_progress_from_line, line)

            cancel_deadline: float | None = None
            while True:
                try:
                    return_code = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if running_state.get("cancel_requested"):
                        if cancel_deadline is None:
                            cancel_deadline = time.monotonic() + 2.0
                            root.after(0, append_log, "Waiting up to 2 seconds for graceful shutdown...")
                            run_output_lines.append("Waiting up to 2 seconds for graceful shutdown...")
                        elif time.monotonic() >= cancel_deadline:
                            root.after(0, append_log, "Process did not exit after terminate; killing...")
                            run_output_lines.append("Process did not exit after terminate; killing...")
                            try:
                                process.kill()
                            except Exception as exc:
                                root.after(0, append_log, f"Kill failed: {exc}")
                                run_output_lines.append(f"Kill failed: {exc}")
                            return_code = process.wait()
                            break

            canceled = bool(running_state.get("cancel_requested") and return_code != 0)
            if canceled:
                root.after(0, append_log, "Command canceled by user.")
                run_output_lines.append("Command canceled by user.")
            root.after(0, append_log, f"[exit code {return_code}]")
            run_output_lines.append(f"[exit code {return_code}]")
            if return_code != 0:
                if any(arg.startswith("--policy.path=") for arg in cmd):
                    root.after(
                        0,
                        append_log,
                        "Deploy hint: check model path, serial ports, camera indices, and camera width/height in Config.",
                    )
                    run_output_lines.append(
                        "Deploy hint: check model path, serial ports, camera indices, and camera width/height in Config."
                    )
                if any(arg.startswith("--robot.cameras=") for arg in cmd):
                    root.after(
                        0,
                        append_log,
                        "Camera hint: preview cameras in the tab and verify camera_height/camera_width match your device.",
                    )
                    run_output_lines.append(
                        "Camera hint: preview cameras in the tab and verify camera_height/camera_width match your device."
                    )

            persist_artifacts(exit_code=return_code, canceled=canceled)

            def complete() -> None:
                if complete_callback is not None:
                    complete_callback(return_code)
                else:
                    set_running(False, "Ready." if return_code == 0 else "Command failed.", return_code != 0)

            root.after(0, complete)

        threading.Thread(target=worker, daemon=True).start()

    def choose_folder(var: Any) -> None:
        selected = filedialog.askdirectory(
            initialdir=normalize_path(str(var.get() or Path.home())),
            title="Select folder",
        )
        if selected:
            var.set(normalize_path(selected))

    class DualCameraPreview:
        def __init__(self, parent: Any, title: str) -> None:
            self.frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
            self.frame.pack(fill="x", pady=(10, 0))
            self.running = False
            self.stop_event = threading.Event()
            self.thread: threading.Thread | None = None
            self.captures: dict[str, Any] = {"laptop": None, "phone": None}
            self.photos: dict[str, Any] = {}
            self.status_preview_var = tk.StringVar(value="Preview stopped.")
            self.cv2_module: Any | None = None

            controls = ttk.Frame(self.frame, style="Panel.TFrame")
            controls.pack(fill="x", pady=(0, 8))
            self.toggle_button = ttk.Button(controls, text="Preview Cameras", command=self.toggle)
            self.toggle_button.pack(side="left")
            ttk.Label(controls, textvariable=self.status_preview_var, style="Muted.TLabel").pack(side="left", padx=(10, 0))

            feeds = ttk.Frame(self.frame, style="Panel.TFrame")
            feeds.pack(fill="x")
            self.camera_labels: dict[str, Any] = {}
            self.canvases: dict[str, Any] = {}
            for col, key in enumerate(("laptop", "phone")):
                pane = ttk.Frame(feeds, style="Panel.TFrame")
                pane.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
                feeds.columnconfigure(col, weight=1)
                label_var = tk.StringVar(value="")
                self.camera_labels[key] = label_var
                ttk.Label(pane, textvariable=label_var, style="Field.TLabel").pack(anchor="w", pady=(0, 4))
                canvas = tk.Canvas(
                    pane,
                    width=320,
                    height=240,
                    bg="#111827",
                    highlightthickness=1,
                    highlightbackground=colors["border"],
                )
                canvas.pack(anchor="w")
                self.canvases[key] = canvas
                self._draw_placeholder(key, "Preview stopped")

            self.refresh_labels()

        def _camera_indices(self) -> dict[str, int]:
            return {
                "laptop": int(config.get("camera_laptop_index", 0)),
                "phone": int(config.get("camera_phone_index", 1)),
            }

        def _camera_shape(self) -> tuple[int, int, int]:
            width = max(int(config.get("camera_width", 640)), 160)
            height = max(int(config.get("camera_height", 360)), 120)
            fps = max(int(config.get("camera_fps", 30)), 1)
            return width, height, fps

        def _draw_placeholder(self, key: str, text: str) -> None:
            canvas = self.canvases[key]
            canvas.delete("all")
            canvas.create_rectangle(0, 0, 320, 240, fill="#111827", outline="")
            canvas.create_text(160, 120, text=text, fill="#9ca3af", width=290)

        def refresh_labels(self) -> None:
            indices = self._camera_indices()
            self.camera_labels["laptop"].set(f"Laptop camera (index {indices['laptop']})")
            self.camera_labels["phone"].set(f"Phone camera (index {indices['phone']})")

        def _render_frame(self, key: str, frame_rgb: Any) -> None:
            if self.cv2_module is None:
                self._draw_placeholder(key, "Preview unavailable")
                return
            cv2_mod = self.cv2_module
            ok, encoded = cv2_mod.imencode(".png", cv2_mod.cvtColor(frame_rgb, cv2_mod.COLOR_RGB2BGR))
            if not ok:
                self._draw_placeholder(key, "Frame encode failed")
                return
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            photo = tk.PhotoImage(data=data)

            canvas = self.canvases[key]
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            self.photos[key] = photo

        def _capture_loop(self) -> None:
            if self.cv2_module is None:
                return
            cv2_mod = self.cv2_module
            indices = self._camera_indices()
            while not self.stop_event.is_set():
                saw_frame = False
                for key, cap in self.captures.items():
                    if cap is None:
                        continue
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    saw_frame = True
                    frame = cv2_mod.resize(frame, (320, 240), interpolation=cv2_mod.INTER_AREA)
                    cv2_mod.putText(
                        frame,
                        f"{key.title()} idx {indices[key]}",
                        (8, 20),
                        cv2_mod.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2_mod.LINE_AA,
                    )
                    frame_rgb = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)
                    root.after(0, self._render_frame, key, frame_rgb)
                time.sleep(0.03 if saw_frame else 0.12)

        def start(self) -> None:
            if self.running:
                return
            self.refresh_labels()
            if not cv2_probe_ok:
                self.status_preview_var.set("OpenCV unavailable for this macOS/Python.")
                self._draw_placeholder("laptop", "OpenCV unavailable")
                self._draw_placeholder("phone", "OpenCV unavailable")
                reason = summarize_probe_error(cv2_probe_error) if cv2_probe_error else "incompatible module"
                append_log(f"Camera preview disabled: {reason}")
                return

            if self.cv2_module is None:
                try:
                    import cv2 as cv2_loaded  # type: ignore[import-not-found]
                except Exception as exc:
                    self.status_preview_var.set("OpenCV import failed.")
                    self._draw_placeholder("laptop", "OpenCV import failed")
                    self._draw_placeholder("phone", "OpenCV import failed")
                    append_log(f"Camera preview unavailable: {exc}")
                    return
                self.cv2_module = cv2_loaded

            cv2_mod = self.cv2_module
            if cv2_mod is None:
                self.status_preview_var.set("OpenCV unavailable.")
                self._draw_placeholder("laptop", "OpenCV unavailable")
                self._draw_placeholder("phone", "OpenCV unavailable")
                return

            width, height, fps = self._camera_shape()
            indices = self._camera_indices()
            self.stop_event.clear()
            self.running = True
            self.toggle_button.configure(text="Stop Preview")
            self.status_preview_var.set("Preview running.")
            self.captures = {"laptop": None, "phone": None}

            for key, index in indices.items():
                cap = cv2_mod.VideoCapture(index)
                if cap is not None and cap.isOpened():
                    cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2_mod.CAP_PROP_FPS, fps)
                    self.captures[key] = cap
                else:
                    if cap is not None:
                        cap.release()
                    self.captures[key] = None
                    self._draw_placeholder(key, f"Camera index {index} unavailable")

            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

        def stop(self) -> None:
            if not self.running:
                return
            self.stop_event.set()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for cap in self.captures.values():
                if cap is not None:
                    cap.release()
            self.captures = {"laptop": None, "phone": None}
            self.thread = None
            self.running = False
            self.toggle_button.configure(text="Preview Cameras")
            self.status_preview_var.set("Preview stopped.")
            self._draw_placeholder("laptop", "Preview stopped")
            self._draw_placeholder("phone", "Preview stopped")

        def toggle(self) -> None:
            if self.running:
                self.stop()
            else:
                self.start()

        def close(self) -> None:
            self.stop()

    # ----------------------------- Record Tab ---------------------------------
    record_container = ttk.Frame(record_tab, style="Panel.TFrame")
    record_container.pack(fill="both", expand=True)

    suggested_dataset, _ = suggest_dataset_name(config)
    record_dataset_var = tk.StringVar(value=suggested_dataset)
    record_episodes_var = tk.StringVar(value="20")
    record_duration_var = tk.StringVar(value="20")
    record_task_var = tk.StringVar(value=DEFAULT_TASK)
    record_dir_var = tk.StringVar(value=str(config["record_data_dir"]))
    record_upload_var = tk.BooleanVar(value=False)

    record_form = ttk.LabelFrame(record_container, text="Recording Setup", style="Section.TLabelframe", padding=12)
    record_form.pack(fill="x")
    record_form.columnconfigure(1, weight=1)

    ttk.Label(record_form, text="Dataset name (or repo id)", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(record_form, textvariable=record_dataset_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(
        record_form,
        text="Suggest Next",
        command=lambda: record_dataset_var.set(suggest_dataset_name(config)[0]),
    ).grid(row=0, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Label(record_form, text="Local dataset save folder", style="Field.TLabel").grid(
        row=1, column=0, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(record_form, textvariable=record_dir_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(record_form, text="Browse", command=lambda: choose_folder(record_dir_var)).grid(
        row=1, column=2, sticky="w", padx=(6, 0), pady=4
    )

    ttk.Label(record_form, text="Episodes", style="Field.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_episodes_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Episode time (seconds)", style="Field.TLabel").grid(
        row=3, column=0, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(record_form, textvariable=record_duration_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(record_form, text="Task description", style="Field.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_form, textvariable=record_task_var, width=52).grid(row=4, column=1, sticky="ew", pady=4)

    ttk.Checkbutton(record_form, text="Upload to Hugging Face after recording", variable=record_upload_var).grid(
        row=5, column=1, sticky="w", pady=(8, 8)
    )

    record_buttons = ttk.Frame(record_form, style="Panel.TFrame")
    record_buttons.grid(row=6, column=1, sticky="w", pady=(8, 0))
    preview_record_button = ttk.Button(record_buttons, text="Preview Command")
    preview_record_button.pack(side="left")
    run_record_button = ttk.Button(record_buttons, text="Run Record", style="Accent.TButton")
    run_record_button.pack(side="left", padx=(10, 0))
    action_buttons.extend([preview_record_button, run_record_button])

    record_summary_var = tk.StringVar(value="")
    record_summary_panel = ttk.LabelFrame(record_container, text="Current Robot Snapshot", style="Section.TLabelframe", padding=10)
    record_summary_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(record_summary_panel, textvariable=record_summary_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    record_camera_preview = DualCameraPreview(record_container, "Record Camera Preview")

    def refresh_record_summary() -> None:
        record_summary_var.set(
            "Follower port: {follower} | Leader port: {leader}\n"
            "Laptop camera idx: {laptop} | Phone camera idx: {phone}\n"
            "Camera stream: {w}x{h} @ {fps}fps (warmup {warmup}s)".format(
                follower=config["follower_port"],
                leader=config["leader_port"],
                laptop=config["camera_laptop_index"],
                phone=config["camera_phone_index"],
                w=config.get("camera_width", 640),
                h=config.get("camera_height", 360),
                fps=config.get("camera_fps", 30),
                warmup=config["camera_warmup_s"],
            )
        )

    def build_record_from_gui() -> tuple[list[str], str, str, Path, int, int] | None:
        dataset_input = record_dataset_var.get().strip()
        if not dataset_input:
            messagebox.showerror("Validation Error", "Dataset name is required.")
            return None

        try:
            episodes = int(record_episodes_var.get().strip())
            episode_time = int(record_duration_var.get().strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Episodes and episode time must be integers.")
            return None

        if episodes <= 0 or episode_time <= 0:
            messagebox.showerror("Validation Error", "Episodes and episode time must be greater than zero.")
            return None

        task = record_task_var.get().strip() or DEFAULT_TASK
        dataset_root = Path(normalize_path(record_dir_var.get().strip() or str(config["record_data_dir"])))
        config["record_data_dir"] = str(dataset_root)
        dataset_repo_id = normalize_repo_id(str(config["hf_username"]), dataset_input)
        dataset_name = repo_name_from_repo_id(dataset_repo_id)

        cmd = build_lerobot_record_command(
            config=config,
            dataset_repo_id=dataset_repo_id,
            num_episodes=episodes,
            task=task,
            episode_time=episode_time,
        )
        return cmd, dataset_name, dataset_repo_id, dataset_root, episodes, episode_time

    def preview_record() -> None:
        built = build_record_from_gui()
        if built is None:
            return
        cmd, _, _, _, _, _ = built
        last_command_state["value"] = shlex.join(cmd)
        append_log("Preview record command:")
        append_log(last_command_state["value"])
        messagebox.showinfo("Record Command", last_command_state["value"])

    def run_record_from_gui() -> None:
        built = build_record_from_gui()
        if built is None:
            return
        cmd, dataset_local_name, dataset_repo_id, dataset_root, episodes, episode_time = built

        exists = dataset_exists_on_hf(dataset_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{dataset_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not messagebox.askyesno("Confirm Record", shlex.join(cmd)):
            return

        preflight_checks = run_preflight_for_record(
            config=config,
            dataset_root=dataset_root,
            upload_enabled=record_upload_var.get(),
        )
        if not confirm_preflight_in_gui("Record Preflight", preflight_checks):
            return

        def after_record(return_code: int) -> None:
            was_canceled = bool(running_state.get("cancel_requested"))
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Record canceled.")
                    messagebox.showinfo("Canceled", "Record command was canceled.")
                else:
                    set_running(False, "Recording failed.", True)
                    messagebox.showerror("Record Failed", f"Recording failed with exit code {return_code}.")
                return

            lerobot_dir = get_lerobot_dir(config)
            source_dataset = lerobot_dir / "data" / dataset_local_name
            target_dataset = dataset_root / dataset_local_name
            active_dataset = source_dataset

            if source_dataset.exists() and source_dataset.resolve() != target_dataset.resolve():
                target_dataset.parent.mkdir(parents=True, exist_ok=True)
                if target_dataset.exists():
                    append_log(f"Target already exists, kept source dataset: {target_dataset}")
                else:
                    shutil.move(str(source_dataset), str(target_dataset))
                    active_dataset = target_dataset
                    append_log(f"Moved dataset to: {target_dataset}")
            elif target_dataset.exists():
                active_dataset = target_dataset

            config["last_dataset_name"] = dataset_local_name
            save_config(config)
            refresh_record_summary()
            refresh_header_subtitle()

            if was_canceled:
                set_running(False, "Record canceled.")
                messagebox.showinfo("Canceled", "Record command was canceled. Upload was skipped.")
                return

            if not record_upload_var.get():
                set_running(False, "Record completed.")
                messagebox.showinfo("Done", "Recording completed.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                dataset_repo_id,
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]

            set_running(False, "Record completed. Starting upload...")

            def after_upload(upload_code: int) -> None:
                if upload_code != 0:
                    set_running(False, "Upload failed.", True)
                    messagebox.showerror("Upload Failed", f"Upload failed with exit code {upload_code}.")
                else:
                    set_running(False, "Record + upload completed.")
                    messagebox.showinfo("Done", "Recording and upload completed.")

            run_process_async(
                upload_cmd,
                cwd=get_lerobot_dir(config),
                complete_callback=after_upload,
                run_mode="upload",
                artifact_context={"dataset_repo_id": dataset_repo_id},
            )

        run_process_async(
            cmd,
            cwd=get_lerobot_dir(config),
            complete_callback=after_record,
            expected_episodes=episodes,
            expected_seconds=episodes * episode_time,
            run_mode="record",
            preflight_checks=preflight_checks,
            artifact_context={"dataset_repo_id": dataset_repo_id},
        )

    preview_record_button.configure(command=preview_record)
    run_record_button.configure(command=run_record_from_gui)

    # ----------------------------- Deploy Tab ---------------------------------
    deploy_container = ttk.Frame(deploy_tab, style="Panel.TFrame")
    deploy_container.pack(fill="both", expand=True)

    deploy_root_var = tk.StringVar(value=str(config["trained_models_dir"]))
    default_model_path = (
        str(Path(config["trained_models_dir"]) / config["last_model_name"])
        if str(config.get("last_model_name", "")).strip()
        else str(config["trained_models_dir"])
    )
    deploy_model_var = tk.StringVar(value=default_model_path)
    deploy_eval_dataset_var = tk.StringVar(
        value=str(config.get("last_eval_dataset_name", "")).strip()
        or suggest_eval_dataset_name(config, str(config.get("last_model_name", "")))
    )
    deploy_eval_episodes_var = tk.StringVar(value=str(config.get("eval_num_episodes", 10)))
    deploy_eval_duration_var = tk.StringVar(value=str(config.get("eval_duration_s", 20)))
    deploy_eval_task_var = tk.StringVar(value=str(config.get("eval_task", DEFAULT_TASK)))

    deploy_form = ttk.LabelFrame(deploy_container, text="Deploy / Eval Setup", style="Section.TLabelframe", padding=12)
    deploy_form.pack(fill="x")
    deploy_form.columnconfigure(1, weight=1)

    ttk.Label(deploy_form, text="Local model root folder", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_root_var, width=52).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_form, text="Browse", command=lambda: choose_folder(deploy_root_var)).grid(
        row=0, column=2, sticky="w", padx=(6, 0), pady=4
    )

    ttk.Label(deploy_form, text="Model folder to deploy", style="Field.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_model_var, width=52).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_form, text="Browse", command=lambda: choose_folder(deploy_model_var)).grid(
        row=1, column=2, sticky="w", padx=(6, 0), pady=4
    )

    ttk.Label(deploy_form, text="Eval dataset name (or repo id)", style="Field.TLabel").grid(
        row=2, column=0, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_dataset_var, width=52).grid(row=2, column=1, sticky="ew", pady=4)

    ttk.Label(deploy_form, text="Eval episodes", style="Field.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_episodes_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval episode time (seconds)", style="Field.TLabel").grid(
        row=4, column=0, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(deploy_form, textvariable=deploy_eval_duration_var, width=20).grid(row=4, column=1, sticky="w", pady=4)

    ttk.Label(deploy_form, text="Eval task description", style="Field.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_form, textvariable=deploy_eval_task_var, width=52).grid(row=5, column=1, sticky="ew", pady=4)

    deploy_buttons = ttk.Frame(deploy_form, style="Panel.TFrame")
    deploy_buttons.grid(row=6, column=1, sticky="w", pady=(8, 0))
    preview_deploy_button = ttk.Button(deploy_buttons, text="Preview Command")
    preview_deploy_button.pack(side="left")
    run_deploy_button = ttk.Button(deploy_buttons, text="Run Deploy", style="Accent.TButton")
    run_deploy_button.pack(side="left", padx=(10, 0))
    action_buttons.extend([preview_deploy_button, run_deploy_button])

    model_section = ttk.LabelFrame(deploy_container, text="Local Models", style="Section.TLabelframe", padding=10)
    model_section.pack(fill="x", pady=(10, 0))
    model_section.columnconfigure(0, weight=1)
    model_listbox = tk.Listbox(
        model_section,
        height=8,
        bg="#111827",
        fg="#e5e7eb",
        selectbackground=colors["accent"],
        selectforeground="#ffffff",
        relief="flat",
        highlightthickness=1,
        highlightbackground=colors["border"],
    )
    model_listbox.grid(row=0, column=0, sticky="ew")
    refresh_models_button = ttk.Button(model_section, text="Refresh Model List")
    refresh_models_button.grid(row=0, column=1, sticky="n", padx=(8, 0))
    action_buttons.append(refresh_models_button)

    model_info_var = tk.StringVar(value="No model selected.")
    model_info_panel = ttk.LabelFrame(deploy_container, text="Selected Model Info", style="Section.TLabelframe", padding=10)
    model_info_panel.pack(fill="x", pady=(10, 0))
    ttk.Label(model_info_panel, textvariable=model_info_var, style="Muted.TLabel", justify="left").pack(anchor="w")

    deploy_camera_preview = DualCameraPreview(deploy_container, "Deploy Camera Preview")

    auto_eval_hint = {"value": deploy_eval_dataset_var.get().strip()}

    def update_model_info(model_path: Path | None) -> None:
        if model_path is None or not model_path.exists() or not model_path.is_dir():
            model_info_var.set("No model selected.")
            return
        entries = sorted(model_path.iterdir())
        child_names = [p.name for p in entries[:8]]
        checkpoints = [p.name for p in entries if p.is_dir() and "checkpoint" in p.name.lower()]
        has_config = any((model_path / name).exists() for name in ("config.json", "model_config.json"))
        info_lines = [
            f"Path: {model_path}",
            f"Items: {len(entries)} | Config file: {'yes' if has_config else 'no'}",
            f"Checkpoint-like folders: {', '.join(checkpoints[:4]) if checkpoints else 'none'}",
            f"Sample contents: {', '.join(child_names) if child_names else '(empty)'}",
        ]
        model_info_var.set("\n".join(info_lines))

    def refresh_local_models() -> None:
        model_listbox.delete(0, "end")
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        if not root_path.exists():
            return
        for folder in sorted(p.name for p in root_path.iterdir() if p.is_dir()):
            model_listbox.insert("end", folder)

    def on_model_select(_: Any) -> None:
        selected = model_listbox.curselection()
        if not selected:
            return
        root_path = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        folder_name = model_listbox.get(selected[0])
        model_path = root_path / folder_name
        deploy_model_var.set(str(model_path))
        update_model_info(model_path)
        current_eval_name = deploy_eval_dataset_var.get().strip()
        if not current_eval_name or current_eval_name == auto_eval_hint["value"]:
            suggested = suggest_eval_dataset_name(config, folder_name)
            deploy_eval_dataset_var.set(suggested)
            auto_eval_hint["value"] = suggested

    model_listbox.bind("<<ListboxSelect>>", on_model_select)
    refresh_models_button.configure(command=refresh_local_models)

    def build_deploy_from_gui() -> tuple[list[str], Path, str, int, int] | None:
        models_root = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        model_path = Path(normalize_path(deploy_model_var.get().strip()))
        if not model_path.is_absolute():
            model_path = models_root / model_path

        if not model_path.exists() or not model_path.is_dir():
            messagebox.showerror("Validation Error", f"Model folder not found:\n{model_path}")
            return None

        eval_dataset_input = deploy_eval_dataset_var.get().strip()
        if not eval_dataset_input:
            messagebox.showerror("Validation Error", "Eval dataset name is required.")
            return None

        try:
            eval_episodes = int(deploy_eval_episodes_var.get().strip())
            eval_duration = int(deploy_eval_duration_var.get().strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Eval episodes and duration must be integers.")
            return None

        if eval_episodes <= 0 or eval_duration <= 0:
            messagebox.showerror("Validation Error", "Eval episodes and duration must be greater than zero.")
            return None

        eval_task = deploy_eval_task_var.get().strip() or DEFAULT_TASK
        eval_repo_id = normalize_repo_id(str(config["hf_username"]), eval_dataset_input)

        config["trained_models_dir"] = str(models_root)
        config["last_model_name"] = model_path.name
        config["eval_num_episodes"] = eval_episodes
        config["eval_duration_s"] = eval_duration
        config["eval_task"] = eval_task
        config["last_eval_dataset_name"] = eval_repo_id.split("/", 1)[1]

        cmd = build_lerobot_record_command(
            config=config,
            dataset_repo_id=eval_repo_id,
            num_episodes=eval_episodes,
            task=eval_task,
            episode_time=eval_duration,
            policy_path=model_path,
        )
        return cmd, model_path, eval_repo_id, eval_episodes, eval_duration

    def preview_deploy() -> None:
        built = build_deploy_from_gui()
        if built is None:
            return
        cmd, _, _, _, _ = built
        last_command_state["value"] = shlex.join(cmd)
        append_log("Preview deploy command:")
        append_log(last_command_state["value"])
        messagebox.showinfo("Deploy Command", last_command_state["value"])

    def run_deploy_from_gui() -> None:
        built = build_deploy_from_gui()
        if built is None:
            return
        cmd, model_path, eval_repo_id, eval_episodes, eval_duration = built

        exists = dataset_exists_on_hf(eval_repo_id)
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{eval_repo_id} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not messagebox.askyesno("Confirm Deploy", shlex.join(cmd)):
            return

        preflight_checks = run_preflight_for_deploy(config=config, model_path=model_path)
        if not confirm_preflight_in_gui("Deploy Preflight", preflight_checks):
            return

        save_config(config)
        refresh_header_subtitle()

        def after_deploy(return_code: int) -> None:
            was_canceled = bool(running_state.get("cancel_requested"))
            if return_code != 0:
                if was_canceled:
                    set_running(False, "Deploy canceled.")
                    messagebox.showinfo("Canceled", "Deploy command was canceled.")
                else:
                    set_running(False, "Deploy failed.", True)
                    messagebox.showerror("Deploy Failed", f"Deploy failed with exit code {return_code}.")
            else:
                set_running(False, "Deploy completed.")
                messagebox.showinfo(
                    "Done",
                    f"Deployment completed.\nModel: {model_path}\nEval dataset: {eval_repo_id}",
                )

        run_process_async(
            cmd,
            cwd=get_lerobot_dir(config),
            complete_callback=after_deploy,
            expected_episodes=eval_episodes,
            expected_seconds=eval_episodes * eval_duration,
            run_mode="deploy",
            preflight_checks=preflight_checks,
            artifact_context={"dataset_repo_id": eval_repo_id, "model_path": str(model_path)},
        )

    preview_deploy_button.configure(command=preview_deploy)
    run_deploy_button.configure(command=run_deploy_from_gui)
    refresh_local_models()
    update_model_info(Path(deploy_model_var.get()) if deploy_model_var.get().strip() else None)

    # ----------------------------- Config Tab ---------------------------------
    config_vars: dict[str, Any] = {}
    path_keys = {"lerobot_dir", "runs_dir", "record_data_dir", "trained_models_dir"}
    field_lookup = {field["key"]: field for field in CONFIG_FIELDS}
    group_layout = [
        ("Paths", ["lerobot_dir", "runs_dir", "record_data_dir", "trained_models_dir"]),
        ("Robot Ports", ["follower_port", "leader_port"]),
        (
            "Cameras",
            [
                "camera_laptop_index",
                "camera_phone_index",
                "camera_warmup_s",
                "camera_width",
                "camera_height",
                "camera_fps",
            ],
        ),
        (
            "Hugging Face + Defaults",
            [
                "hf_username",
                "last_dataset_name",
                "eval_num_episodes",
                "eval_duration_s",
                "eval_task",
                "last_eval_dataset_name",
                "last_model_name",
            ],
        ),
    ]

    def add_config_group(parent: Any, title: str, keys: list[str]) -> None:
        frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        for row, key in enumerate(keys):
            field = field_lookup[key]
            ttk.Label(frame, text=field["prompt"], style="Field.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=4
            )
            current = config.get(key, default_for_key(key, config))
            value_var = tk.StringVar(value=str(current))
            config_vars[key] = value_var
            ttk.Entry(frame, textvariable=value_var, width=52).grid(row=row, column=1, sticky="ew", pady=4)
            if key in path_keys:
                ttk.Button(frame, text="Browse", command=lambda var=value_var: choose_folder(var)).grid(
                    row=row, column=2, sticky="w", padx=(6, 0), pady=4
                )

    for group_title, keys in group_layout:
        add_config_group(config_tab, group_title, keys)

    diagnostics_frame = ttk.LabelFrame(config_tab, text="Diagnostics", style="Section.TLabelframe", padding=10)
    diagnostics_frame.pack(fill="both", expand=True, pady=(0, 10))
    diagnostics_controls = ttk.Frame(diagnostics_frame, style="Panel.TFrame")
    diagnostics_controls.pack(fill="x", pady=(0, 6))

    doctor_report_var = tk.StringVar(value="")
    doctor_text = scrolledtext.ScrolledText(
        diagnostics_frame,
        height=9,
        state="disabled",
        bg="#111827",
        fg="#d4d4d4",
        insertbackground="#f8fafc",
        font=("Menlo", 10),
        relief="flat",
        padx=8,
        pady=8,
    )
    doctor_text.tag_configure("pass", foreground="#4ade80")
    doctor_text.tag_configure("warn", foreground="#fbbf24")
    doctor_text.tag_configure("fail", foreground="#f87171")
    doctor_text.tag_configure("default", foreground="#d4d4d4")
    doctor_text.pack(fill="both", expand=True)

    def build_config_preview_from_vars() -> dict[str, Any] | None:
        preview = dict(config)
        for field in CONFIG_FIELDS:
            key = field["key"]
            raw_value = config_vars[key].get().strip()
            if field["type"] == "int":
                try:
                    preview[key] = int(raw_value)
                except ValueError:
                    messagebox.showerror("Validation Error", f"{field['prompt']} must be an integer.")
                    return None
            elif field["type"] == "path":
                preview[key] = normalize_path(raw_value)
            else:
                preview[key] = raw_value
        return preview

    def render_doctor_report(checks: list[CheckResult]) -> None:
        summary = summarize_checks(checks, title="Doctor Report")
        doctor_report_var.set(summary)
        doctor_text.configure(state="normal")
        doctor_text.delete("1.0", "end")
        for line in summary.splitlines():
            tag = "default"
            if line.startswith("[PASS"):
                tag = "pass"
            elif line.startswith("[WARN"):
                tag = "warn"
            elif line.startswith("[FAIL"):
                tag = "fail"
            doctor_text.insert("end", line + "\n", (tag,))
        doctor_text.see("end")
        doctor_text.configure(state="disabled")

    def run_doctor_from_gui() -> None:
        preview = build_config_preview_from_vars()
        if preview is None:
            return
        checks = collect_doctor_checks(preview)
        render_doctor_report(checks)
        append_log("Ran Doctor from Config tab.")

    def copy_doctor_report() -> None:
        report_text = doctor_report_var.get()
        if not report_text.strip():
            messagebox.showinfo("Copy Doctor Report", "No doctor report available yet.")
            return
        root.clipboard_clear()
        root.clipboard_append(report_text)
        append_log("Copied doctor report to clipboard.")

    run_doctor_button = ttk.Button(diagnostics_controls, text="Run Doctor", command=run_doctor_from_gui)
    run_doctor_button.pack(side="left")
    copy_doctor_button = ttk.Button(diagnostics_controls, text="Copy Doctor Report", command=copy_doctor_report)
    copy_doctor_button.pack(side="left", padx=(8, 0))
    action_buttons.extend([run_doctor_button, copy_doctor_button])

    def save_config_from_gui() -> None:
        for field in CONFIG_FIELDS:
            key = field["key"]
            raw_value = config_vars[key].get().strip()
            if field["type"] == "int":
                try:
                    config[key] = int(raw_value)
                except ValueError:
                    messagebox.showerror("Validation Error", f"{field['prompt']} must be an integer.")
                    return
            elif field["type"] == "path":
                config[key] = normalize_path(raw_value)
            else:
                config[key] = raw_value

        save_config(config)
        record_dir_var.set(str(config["record_data_dir"]))
        deploy_root_var.set(str(config["trained_models_dir"]))
        deploy_eval_episodes_var.set(str(config.get("eval_num_episodes", 10)))
        deploy_eval_duration_var.set(str(config.get("eval_duration_s", 20)))
        deploy_eval_task_var.set(str(config.get("eval_task", DEFAULT_TASK)))
        refresh_local_models()
        refresh_record_summary()
        refresh_header_subtitle()
        record_camera_preview.refresh_labels()
        deploy_camera_preview.refresh_labels()
        messagebox.showinfo("Saved", "Configuration saved.")

    save_config_button = ttk.Button(config_tab, text="Save Config", style="Accent.TButton", command=save_config_from_gui)
    save_config_button.pack(anchor="w", pady=(2, 0))
    action_buttons.append(save_config_button)

    def on_tab_changed(_: Any) -> None:
        selected = notebook.select()
        if selected != str(record_tab):
            record_camera_preview.stop()
        if selected != str(deploy_tab):
            deploy_camera_preview.stop()

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    def on_close() -> None:
        record_camera_preview.close()
        deploy_camera_preview.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_header_subtitle()
    refresh_record_summary()
    record_camera_preview.refresh_labels()
    deploy_camera_preview.refresh_labels()
    append_log("GUI ready. Configure tabs, preview cameras, then run record/deploy.")
    root.mainloop()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeRobot local pipeline manager for recording and local deployment."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("record", help="Record teleoperated demos and optionally upload.")
    subparsers.add_parser("deploy", help="Run deployment/eval using a local model folder.")
    subparsers.add_parser("config", help="Update saved config values.")
    history_parser = subparsers.add_parser("history", help="Show recent run artifacts.")
    history_parser.add_argument("--limit", type=int, default=15, help="Maximum number of runs to show.")
    subparsers.add_parser("doctor", help="Run local diagnostics for env, ports, and cameras.")
    subparsers.add_parser("gui", help="Launch desktop GUI for config, record, and deploy.")

    return parser.parse_args()



def main() -> int:
    args = parse_args()

    raw_config, source = load_raw_config()
    first_run = source is None

    if args.mode == "gui":
        run_gui_mode(raw_config)
        return 0

    if args.mode == "doctor":
        config = normalize_config_without_prompts(raw_config)
        run_doctor_mode(config)
        return 0

    if args.mode == "history":
        config = normalize_config_without_prompts(raw_config)
        run_history_mode(config, limit=max(int(args.limit), 1))
        return 0

    if first_run:
        print_section("=== 🛠️ FIRST-TIME SETUP ===")
        print(f"Config not found. Creating one at {PRIMARY_CONFIG_PATH}")
        print("You can type a path or enter 'b' to browse folders in Finder/File Manager.")

    if args.mode == "config":
        run_config_mode(raw_config)
        return 0

    config = ensure_config(raw_config, force_prompt_all=first_run)

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
