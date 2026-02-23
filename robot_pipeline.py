#!/usr/bin/env python3
"""LeRobot local pipeline manager for SO-101 recording and local deployment."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_LEROBOT_DIR = Path.home() / "lerobot"
PRIMARY_CONFIG_PATH = Path.home() / ".robot_config.json"
DEFAULT_SECONDARY_CONFIG_PATH = DEFAULT_LEROBOT_DIR / ".robot_config.json"
LEGACY_CONFIG_PATH = Path.home() / ".robot_pipeline_config.json"

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
    "camera_warmup_s": 5,
    "last_model_name": "",
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
    {"key": "camera_warmup_s", "prompt": "Camera warmup (s)", "type": "int"},
    {
        "key": "last_model_name",
        "prompt": "Last model name (optional)",
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
    warmup = int(config.get("camera_warmup_s", 5))
    cameras = {
        "laptop": {
            "type": "opencv",
            "index_or_path": laptop,
            "width": 640,
            "height": 480,
            "fps": 30,
            "warmup_s": warmup,
        },
        "phone": {
            "type": "opencv",
            "index_or_path": phone,
            "width": 640,
            "height": 480,
            "fps": 30,
            "warmup_s": warmup,
        },
    }
    return json.dumps(cameras, separators=(",", ":"))



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

    config["last_model_name"] = model_path.name
    save_config(config)

    if not prompt_yes_no("Run deployment now?", "y"):
        return

    lerobot_dir = get_lerobot_dir(config)
    eval_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={model_path}",
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

    config = normalize_config_without_prompts(raw_config)
    if not raw_config:
        save_config(config)

    root = tk.Tk()
    root.title("LeRobot Local Pipeline Manager")
    root.geometry("1020x780")

    status_var = tk.StringVar(value="Ready.")
    running_state = {"active": False}
    action_buttons: list[ttk.Button] = []

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    record_tab = ttk.Frame(notebook, padding=12)
    deploy_tab = ttk.Frame(notebook, padding=12)
    config_tab = ttk.Frame(notebook, padding=12)
    notebook.add(record_tab, text="Record")
    notebook.add(deploy_tab, text="Deploy")
    notebook.add(config_tab, text="Config")

    log_box = scrolledtext.ScrolledText(root, height=12, state="disabled")
    log_box.pack(fill="both", expand=False, padx=10, pady=(0, 8))

    status_label = ttk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(fill="x", padx=10, pady=(0, 10))

    def append_log(line: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", line + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def set_running(active: bool, status_text: str | None = None) -> None:
        running_state["active"] = active
        if status_text is None:
            status_var.set("Running..." if active else "Ready.")
        else:
            status_var.set(status_text)
        for button in action_buttons:
            button.configure(state="disabled" if active else "normal")

    def run_process_async(
        cmd: list[str],
        cwd: Path | None,
        complete_callback: Any | None = None,
    ) -> None:
        if running_state["active"]:
            messagebox.showinfo("Busy", "Another command is already running.")
            return

        set_running(True, "Running command...")
        append_log("$ " + shlex.join(cmd))

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
                if cmd[0] == "huggingface-cli":
                    root.after(
                        0,
                        append_log,
                        "Make sure you're in your lerobot env: source ~/lerobot/lerobot_env/bin/activate",
                    )
                root.after(0, set_running, False, "Ready.")
                root.after(0, messagebox.showerror, "Command Error", str(exc))
                return

            if process.stdout is not None:
                for line in process.stdout:
                    root.after(0, append_log, line.rstrip("\n"))

            return_code = process.wait()
            root.after(0, append_log, f"[exit code {return_code}]")

            if complete_callback is not None:
                root.after(0, complete_callback, return_code)
            else:
                root.after(0, set_running, False, "Ready.")

        threading.Thread(target=worker, daemon=True).start()

    def choose_folder(var: Any) -> None:
        selected = filedialog.askdirectory(
            initialdir=normalize_path(str(var.get() or Path.home())),
            title="Select folder",
        )
        if selected:
            var.set(normalize_path(selected))

    # ----------------------------- Record Tab ---------------------------------
    suggested_dataset, _ = suggest_dataset_name(config)
    record_dataset_var = tk.StringVar(value=suggested_dataset)
    record_episodes_var = tk.StringVar(value="20")
    record_duration_var = tk.StringVar(value="20")
    record_task_var = tk.StringVar(value=DEFAULT_TASK)
    record_dir_var = tk.StringVar(value=str(config["record_data_dir"]))
    record_upload_var = tk.BooleanVar(value=False)

    ttk.Label(record_tab, text="Dataset name").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_tab, textvariable=record_dataset_var, width=55).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(
        record_tab,
        text="Suggest Next",
        command=lambda: record_dataset_var.set(suggest_dataset_name(config)[0]),
    ).grid(row=0, column=2, sticky="w", padx=(6, 0), pady=4)

    ttk.Label(record_tab, text="Local dataset save folder").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_tab, textvariable=record_dir_var, width=55).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(record_tab, text="Browse", command=lambda: choose_folder(record_dir_var)).grid(
        row=1, column=2, sticky="w", padx=(6, 0), pady=4
    )

    ttk.Label(record_tab, text="Episodes").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_tab, textvariable=record_episodes_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(record_tab, text="Episode time (seconds)").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_tab, textvariable=record_duration_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(record_tab, text="Task description").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(record_tab, textvariable=record_task_var, width=55).grid(row=4, column=1, sticky="ew", pady=4)

    ttk.Checkbutton(record_tab, text="Upload to Hugging Face after recording", variable=record_upload_var).grid(
        row=5, column=1, sticky="w", pady=(8, 8)
    )

    record_tab.columnconfigure(1, weight=1)

    def build_record_from_gui() -> tuple[list[str], str, Path] | None:
        dataset_name = record_dataset_var.get().strip()
        if not dataset_name:
            messagebox.showerror("Validation Error", "Dataset name is required.")
            return None

        try:
            episodes = int(record_episodes_var.get().strip())
            episode_time = int(record_duration_var.get().strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Episodes and episode time must be integers.")
            return None

        task = record_task_var.get().strip() or DEFAULT_TASK
        dataset_root = Path(normalize_path(record_dir_var.get().strip() or str(config["record_data_dir"])))
        config["record_data_dir"] = str(dataset_root)

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
            f"--dataset.repo_id={config['hf_username']}/{dataset_name}",
            f"--dataset.num_episodes={episodes}",
            f"--dataset.single_task={task}",
            f"--dataset.episode_time_s={episode_time}",
        ]
        return cmd, dataset_name, dataset_root

    def preview_record() -> None:
        built = build_record_from_gui()
        if built is None:
            return
        cmd, _, _ = built
        append_log("Preview record command:")
        append_log(shlex.join(cmd))
        messagebox.showinfo("Record Command", shlex.join(cmd))

    def run_record_from_gui() -> None:
        built = build_record_from_gui()
        if built is None:
            return
        cmd, dataset_name, dataset_root = built

        exists = dataset_exists_on_hf(f"{config['hf_username']}/{dataset_name}")
        if exists is True:
            proceed = messagebox.askyesno(
                "Dataset Exists",
                f"{config['hf_username']}/{dataset_name} already exists on Hugging Face.\nContinue anyway?",
            )
            if not proceed:
                return

        if not messagebox.askyesno("Confirm Record", shlex.join(cmd)):
            return

        def after_record(return_code: int) -> None:
            if return_code != 0:
                set_running(False, "Ready.")
                messagebox.showerror("Record Failed", f"Recording failed with exit code {return_code}.")
                return

            lerobot_dir = get_lerobot_dir(config)
            source_dataset = lerobot_dir / "data" / dataset_name
            target_dataset = dataset_root / dataset_name
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

            config["last_dataset_name"] = dataset_name
            save_config(config)

            if not record_upload_var.get():
                set_running(False, "Record completed.")
                messagebox.showinfo("Done", "Recording completed.")
                return

            upload_cmd = [
                "huggingface-cli",
                "upload",
                f"{config['hf_username']}/{dataset_name}",
                str(active_dataset),
                "--repo-type",
                "dataset",
            ]

            def after_upload(upload_code: int) -> None:
                set_running(False, "Ready.")
                if upload_code != 0:
                    messagebox.showerror("Upload Failed", f"Upload failed with exit code {upload_code}.")
                else:
                    messagebox.showinfo("Done", "Recording and upload completed.")

            run_process_async(upload_cmd, cwd=get_lerobot_dir(config), complete_callback=after_upload)

        run_process_async(cmd, cwd=get_lerobot_dir(config), complete_callback=after_record)

    preview_record_button = ttk.Button(record_tab, text="Preview Command", command=preview_record)
    preview_record_button.grid(row=6, column=1, sticky="w", pady=(8, 0))
    action_buttons.append(preview_record_button)

    run_record_button = ttk.Button(record_tab, text="Run Record", command=run_record_from_gui)
    run_record_button.grid(row=6, column=1, sticky="w", padx=(130, 0), pady=(8, 0))
    action_buttons.append(run_record_button)

    # ----------------------------- Deploy Tab ---------------------------------
    deploy_root_var = tk.StringVar(value=str(config["trained_models_dir"]))
    default_model_path = (
        str(Path(config["trained_models_dir"]) / config["last_model_name"])
        if str(config.get("last_model_name", "")).strip()
        else str(config["trained_models_dir"])
    )
    deploy_model_var = tk.StringVar(value=default_model_path)

    ttk.Label(deploy_tab, text="Local model root folder").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_tab, textvariable=deploy_root_var, width=55).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_tab, text="Browse", command=lambda: choose_folder(deploy_root_var)).grid(
        row=0, column=2, sticky="w", padx=(6, 0), pady=4
    )

    ttk.Label(deploy_tab, text="Model folder to deploy").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(deploy_tab, textvariable=deploy_model_var, width=55).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(deploy_tab, text="Browse", command=lambda: choose_folder(deploy_model_var)).grid(
        row=1, column=2, sticky="w", padx=(6, 0), pady=4
    )

    model_listbox = tk.Listbox(deploy_tab, height=10)
    model_listbox.grid(row=2, column=1, sticky="nsew", pady=(6, 6))
    deploy_tab.columnconfigure(1, weight=1)
    deploy_tab.rowconfigure(2, weight=1)

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
        deploy_model_var.set(str(root_path / model_listbox.get(selected[0])))

    model_listbox.bind("<<ListboxSelect>>", on_model_select)

    def build_deploy_from_gui() -> tuple[list[str], Path] | None:
        models_root = Path(normalize_path(deploy_root_var.get().strip() or str(config["trained_models_dir"])))
        model_path = Path(normalize_path(deploy_model_var.get().strip()))
        if not model_path.is_absolute():
            model_path = models_root / model_path

        if not model_path.exists() or not model_path.is_dir():
            messagebox.showerror("Validation Error", f"Model folder not found:\n{model_path}")
            return None

        config["trained_models_dir"] = str(models_root)
        config["last_model_name"] = model_path.name

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            f"--policy.path={model_path}",
            "--robot.type=so101_follower",
            f"--robot.port={config['follower_port']}",
            f"--robot.cameras={camera_arg(config)}",
        ]
        return cmd, model_path

    def preview_deploy() -> None:
        built = build_deploy_from_gui()
        if built is None:
            return
        cmd, _ = built
        append_log("Preview deploy command:")
        append_log(shlex.join(cmd))
        messagebox.showinfo("Deploy Command", shlex.join(cmd))

    def run_deploy_from_gui() -> None:
        built = build_deploy_from_gui()
        if built is None:
            return
        cmd, model_path = built
        if not messagebox.askyesno("Confirm Deploy", shlex.join(cmd)):
            return

        save_config(config)

        def after_deploy(return_code: int) -> None:
            set_running(False, "Ready.")
            if return_code != 0:
                messagebox.showerror("Deploy Failed", f"Deploy failed with exit code {return_code}.")
            else:
                messagebox.showinfo("Done", f"Deployment completed using:\n{model_path}")

        run_process_async(cmd, cwd=get_lerobot_dir(config), complete_callback=after_deploy)

    ttk.Button(deploy_tab, text="Refresh Model List", command=refresh_local_models).grid(
        row=2, column=2, sticky="nw", padx=(6, 0), pady=(6, 0)
    )

    preview_deploy_button = ttk.Button(deploy_tab, text="Preview Command", command=preview_deploy)
    preview_deploy_button.grid(row=3, column=1, sticky="w", pady=(8, 0))
    action_buttons.append(preview_deploy_button)

    run_deploy_button = ttk.Button(deploy_tab, text="Run Deploy", command=run_deploy_from_gui)
    run_deploy_button.grid(row=3, column=1, sticky="w", padx=(130, 0), pady=(8, 0))
    action_buttons.append(run_deploy_button)

    refresh_local_models()

    # ----------------------------- Config Tab ---------------------------------
    config_vars: dict[str, Any] = {}
    path_keys = {"lerobot_dir", "record_data_dir", "trained_models_dir"}

    for row, field in enumerate(CONFIG_FIELDS):
        key = field["key"]
        ttk.Label(config_tab, text=field["prompt"]).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=4)

        current = config.get(key, default_for_key(key, config))
        value_var = tk.StringVar(value=str(current))
        config_vars[key] = value_var

        ttk.Entry(config_tab, textvariable=value_var, width=55).grid(row=row, column=1, sticky="ew", pady=4)
        if key in path_keys:
            ttk.Button(config_tab, text="Browse", command=lambda var=value_var: choose_folder(var)).grid(
                row=row, column=2, sticky="w", padx=(6, 0), pady=4
            )

    config_tab.columnconfigure(1, weight=1)

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
        refresh_local_models()
        messagebox.showinfo("Saved", "Configuration saved.")

    ttk.Button(config_tab, text="Save Config", command=save_config_from_gui).grid(
        row=len(CONFIG_FIELDS), column=1, sticky="w", pady=(10, 0)
    )

    append_log("GUI ready. Use tabs to configure, record, and deploy local models.")
    root.mainloop()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeRobot local pipeline manager for recording and local deployment."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("record", help="Record teleoperated demos and optionally upload.")
    subparsers.add_parser("deploy", help="Run deployment/eval using a local model folder.")
    subparsers.add_parser("config", help="Update saved config values.")
    subparsers.add_parser("gui", help="Launch desktop GUI for config, record, and deploy.")

    return parser.parse_args()



def main() -> int:
    args = parse_args()

    raw_config, source = load_raw_config()
    first_run = source is None

    if args.mode == "gui":
        run_gui_mode(raw_config)
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
