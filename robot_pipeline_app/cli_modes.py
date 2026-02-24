from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .artifacts import run_history_mode
from .checks import collect_doctor_checks, has_failures, run_preflight_for_deploy, run_preflight_for_record, summarize_checks
from .commands import build_lerobot_record_command
from .config_store import (
    ensure_config,
    get_lerobot_dir,
    load_raw_config,
    normalize_config_without_prompts,
    print_section,
    prompt_int,
    prompt_path,
    prompt_text,
    prompt_yes_no,
    save_config,
)
from .constants import CONFIG_FIELDS, DEFAULT_TASK, PRIMARY_CONFIG_PATH
from .deploy_diagnostics import validate_model_path
from .repo_utils import (
    dataset_exists_on_hf,
    normalize_repo_id,
    repo_name_from_repo_id,
    suggest_dataset_name,
    suggest_eval_dataset_name,
)
from .workflows import execute_command_with_artifacts, move_recorded_dataset, upload_dataset_with_artifacts


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

    dataset_root = Path(prompt_path("Local dataset save folder", str(config["record_data_dir"])))
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

    run_result = execute_command_with_artifacts(
        config=config,
        mode="record",
        cmd=cmd,
        cwd=lerobot_dir,
        preflight_checks=preflight_checks,
        dataset_repo_id=dataset_repo_id,
    )
    if run_result.canceled or run_result.exit_code is None:
        return
    if run_result.exit_code != 0:
        print(f"Recording failed with exit code {run_result.exit_code}.")
        return

    active_dataset_path = move_recorded_dataset(
        lerobot_dir=lerobot_dir,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    print("Recording completed.")
    print("Done! ✓")

    config["last_dataset_name"] = dataset_name
    save_config(config)

    if not upload_after_record:
        return

    upload_result = upload_dataset_with_artifacts(
        config=config,
        dataset_repo_id=dataset_repo_id,
        upload_path=active_dataset_path if active_dataset_path.exists() else (dataset_root / dataset_name),
    )
    if upload_result.canceled or upload_result.exit_code is None:
        return
    if upload_result.exit_code != 0:
        print(f"Upload failed with exit code {upload_result.exit_code}.")
        return

    print("Upload completed.")
    print("Done! ✓")


def run_deploy_mode(config: dict[str, Any]) -> None:
    print_section("=== 🚀 DEPLOY MODE ===")

    models_root = Path(prompt_path("Local model save folder", str(config["trained_models_dir"])))
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
    valid_model_path, model_detail, _ = validate_model_path(model_path)
    if not valid_model_path:
        print(model_detail)
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

    eval_result = execute_command_with_artifacts(
        config=config,
        mode="deploy",
        cmd=eval_cmd,
        cwd=lerobot_dir,
        preflight_checks=preflight_checks,
        dataset_repo_id=eval_repo_id,
        model_path=model_path,
    )
    if eval_result.canceled or eval_result.exit_code is None:
        return
    if eval_result.exit_code != 0:
        print(f"Deployment command failed with exit code {eval_result.exit_code}.")
        return

    print("Deployment command completed.")
    print("Done! ✓")


def run_config_mode(config: dict[str, Any]) -> dict[str, Any]:
    print_section("=== ⚙️ CONFIG MODE ===")
    print("Press Enter to keep defaults shown in brackets.")

    from .config_store import default_for_key

    for field in CONFIG_FIELDS:
        key = field["key"]
        default = default_for_key(key, config)
        current = config.get(key, default)

        if field["type"] == "int":
            config[key] = prompt_int(field["prompt"], int(current))
        elif field["type"] == "path":
            config[key] = prompt_path(field["prompt"], str(current))
        else:
            config[key] = prompt_text(field["prompt"], str(current))

    save_config(config)
    return config


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
        from .gui_app import run_gui_mode

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
