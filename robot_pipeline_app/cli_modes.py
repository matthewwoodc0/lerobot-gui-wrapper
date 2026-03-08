from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import run_history_mode
from .compat import compatibility_checks, probe_lerobot_capabilities
from .checks import (
    collect_doctor_checks,
    collect_doctor_events,
    has_failures,
    run_preflight_for_deploy,
    run_preflight_for_record,
    summarize_checks,
)
from .commands import build_lerobot_record_command
from .config_store import (
    ensure_config,
    get_deploy_data_dir,
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
from .desktop_launcher import install_desktop_launcher
from .deploy_diagnostics import validate_model_path
from .feature_flags import compat_probe_enabled, diagnostics_v2_enabled, support_bundle_enabled
from .repo_utils import (
    dataset_exists_on_hf,
    normalize_repo_id,
    repo_name_from_repo_id,
    resolve_unique_repo_id,
    suggest_dataset_name,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .profile_io import export_profile, import_profile
from .setup_wizard import build_setup_status_summary, build_setup_wizard_guide, probe_setup_wizard_status
from .support_bundle import create_support_bundle
from .workflows import (
    execute_command_with_artifacts,
    move_recorded_dataset,
    tag_uploaded_dataset_with_artifacts,
    upload_dataset_with_artifacts,
)


def run_doctor_mode(config: dict[str, Any], *, json_output: bool = False) -> None:
    if not diagnostics_v2_enabled(config):
        checks = collect_doctor_checks(config)
        pass_count = sum(1 for level, _, _ in checks if level == "PASS")
        warn_count = sum(1 for level, _, _ in checks if level == "WARN")
        fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
        if json_output:
            payload = {
                "generated_at_iso": datetime.now(timezone.utc).isoformat(),
                "diagnostic_version": "v1",
                "summary": {
                    "pass_count": pass_count,
                    "warn_count": warn_count,
                    "fail_count": fail_count,
                },
                "checks": [
                    {"level": level, "name": name, "detail": detail}
                    for level, name, detail in checks
                ],
            }
            print(json.dumps(payload, indent=2))
            return

        print_section("=== 🩺 DOCTOR MODE ===")
        print("Diagnostics V2 is disabled by config (diagnostics_v2_enabled=false).")
        print(summarize_checks(checks, title="Doctor"))
        if fail_count == 0:
            print("Doctor completed. No hard blockers found.")
        else:
            print("Doctor found blockers. Resolve FAIL items first.")
        return

    events = collect_doctor_events(config)

    fail_count = sum(1 for event in events if event.level == "FAIL")
    warn_count = sum(1 for event in events if event.level == "WARN")
    pass_count = sum(1 for event in events if event.level == "PASS")

    if json_output:
        payload = {
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "diagnostic_version": "v2",
            "summary": {
                "pass_count": pass_count,
                "warn_count": warn_count,
                "fail_count": fail_count,
            },
            "events": [event.to_dict() for event in events],
        }
        print(json.dumps(payload, indent=2))
        return

    print_section("=== 🩺 DOCTOR MODE ===")

    for event in events:
        print(f"[{event.level:4}] {event.code} {event.name}: {event.detail}")
        if event.level in {"WARN", "FAIL"}:
            if event.fix:
                print(f"          Fix: {event.fix}")
            if event.docs_ref:
                print(f"          Docs: {event.docs_ref}")

    print("\nSummary:")
    print(f"- PASS: {pass_count}")
    print(f"- WARN: {warn_count}")
    print(f"- FAIL: {fail_count}")
    if fail_count == 0:
        print("Doctor completed. No hard blockers found.")
    else:
        print("Doctor found blockers. Resolve FAIL items first.")


def run_support_bundle_mode(config: dict[str, Any], *, run_id: str, output: str) -> int:
    if not support_bundle_enabled(config):
        print("Support bundle export is disabled by config (support_bundle_enabled=false).")
        return 1

    output_path = Path(output).expanduser()
    result = create_support_bundle(config=config, run_id=run_id, output_path=output_path)
    print(result.message)
    if result.bundle_path is not None:
        print(f"- bundle: {result.bundle_path}")
    if result.run_id:
        print(f"- run id: {result.run_id}")
    return 0 if result.ok else 1


def run_compat_mode(config: dict[str, Any], *, json_output: bool = False, refresh: bool = False) -> int:
    if not compat_probe_enabled(config):
        if json_output:
            payload = {
                "enabled": False,
                "reason": "compat_probe_enabled=false",
                "checks": [],
            }
            print(json.dumps(payload, indent=2))
        else:
            print("Compatibility probe is disabled by config (compat_probe_enabled=false).")
        return 1

    capabilities = probe_lerobot_capabilities(
        config=config,
        include_flag_probe=True,
        force_refresh=refresh,
    )
    checks = compatibility_checks(config, include_flag_probe=True)

    if json_output:
        payload = {
            "capabilities": capabilities.to_dict(),
            "checks": [
                {"level": level, "name": name, "detail": detail}
                for level, name, detail in checks
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print_section("=== 🔧 COMPAT MODE ===")
    print("Detected capabilities:")
    print(f"- LeRobot version: {capabilities.lerobot_version}")
    print(f"- record entrypoint: {capabilities.record_entrypoint}")
    print(f"- train entrypoint: {capabilities.train_entrypoint}")
    print(f"- teleop entrypoint: {capabilities.teleop_entrypoint}")
    print(f"- calibrate entrypoint: {capabilities.calibrate_entrypoint}")
    policy_flag = capabilities.policy_path_flag or "none"
    print(f"- policy path flag: --{policy_flag}" if policy_flag != "none" else "- policy path flag: unavailable")
    print(f"- active rename flag: --{capabilities.active_rename_flag}")
    if capabilities.fallback_notes:
        print("- fallback notes:")
        for note in capabilities.fallback_notes:
            print(f"  - {note}")
    print("\nCompatibility checks:")
    for level, name, detail in checks:
        print(f"[{level:4}] {name}: {detail}")
    return 0


def run_profile_export_mode(
    config: dict[str, Any],
    *,
    output: str,
    name: str = "",
    description: str = "",
    include_paths: bool = False,
) -> int:
    result = export_profile(
        config=config,
        output_path=Path(output).expanduser(),
        name=name,
        description=description,
        include_paths=include_paths,
    )
    print(result.message)
    if result.output_path is not None:
        print(f"- profile: {result.output_path}")
    return 0 if result.ok else 1


def run_profile_import_mode(
    config: dict[str, Any],
    *,
    input_path: str,
    apply_paths: bool = False,
) -> int:
    result = import_profile(
        config=config,
        input_path=Path(input_path).expanduser(),
        apply_paths=apply_paths,
    )
    print(result.message)
    if not result.ok or result.updated_config is None:
        return 1
    save_config(result.updated_config, quiet=True)
    if result.applied_keys:
        print(f"- applied keys: {', '.join(result.applied_keys)}")
    if result.skipped_keys:
        print(f"- skipped keys: {', '.join(result.skipped_keys)}")
        print("  (use --apply-paths on import to apply profile path fields)")
    return 0


def run_record_mode(config: dict[str, Any]) -> None:
    print_section("=== 🎬 RECORD MODE ===")

    suggested_name, checked_remote = suggest_dataset_name(config)
    if checked_remote:
        print(f"Suggested next dataset name: {suggested_name}")
    else:
        print(f"Suggested dataset name (local increment): {suggested_name}")

    dataset_input = prompt_text("Dataset name (or full repo id)", suggested_name)
    username = str(config["hf_username"])

    dataset_root = Path(prompt_path("Local dataset save folder", str(config["record_data_dir"])))
    config["record_data_dir"] = str(dataset_root)
    lerobot_dir = get_lerobot_dir(config)

    dataset_repo_id, dataset_adjusted, _ = resolve_unique_repo_id(
        username=username,
        dataset_name_or_repo_id=dataset_input,
        local_roots=[dataset_root, lerobot_dir / "data"],
    )
    dataset_name = repo_name_from_repo_id(dataset_repo_id)
    if dataset_adjusted:
        print(f"Auto-iterated dataset to avoid existing target: {dataset_repo_id}")

    remote_exists = dataset_exists_on_hf(dataset_repo_id)
    if remote_exists is True:
        print(f"Warning: {dataset_repo_id} already exists on HuggingFace.")
        if not prompt_yes_no("Continue with this dataset name?", "n"):
            print("Cancelled.")
            return

    num_episodes = prompt_int("Number of episodes", 20)
    episode_time = prompt_int("Episode duration in seconds", 20)
    task = prompt_text("Task description", DEFAULT_TASK)

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
        episode_time_s=episode_time,
        dataset_repo_id=dataset_repo_id,
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
    config["last_dataset_repo_id"] = dataset_repo_id
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

    tag_result, tags, tag_detail = tag_uploaded_dataset_with_artifacts(
        config=config,
        dataset_repo_id=dataset_repo_id,
        task=task,
    )
    tag_ok = bool((not tag_result.canceled) and tag_result.exit_code == 0)

    print("Upload completed.")
    print(f"- Uploaded name: {dataset_name}")
    print(f"- Hugging Face repo: {dataset_repo_id}")
    print(f"- Tagging: {'success' if tag_ok else 'failed'}")
    print(f"- Tags: {', '.join(tags)}")
    print(f"- Tagging details: {tag_detail}")
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

    username = str(config["hf_username"]).strip()
    eval_dataset_name = prompt_text(
        "Eval dataset repo id (owner/eval_name)",
        normalize_repo_id(username, suggest_eval_dataset_name(config, model_path.name)),
    )
    suggested_eval_repo_id, requires_quick_fix = suggest_eval_prefixed_repo_id(
        username=username,
        dataset_name_or_repo_id=eval_dataset_name,
    )
    suggested_eval_repo_id = normalize_repo_id(username, suggested_eval_repo_id)
    if requires_quick_fix:
        print(
            "Eval dataset naming convention requires dataset names to start with 'eval_'.\n"
            f"Suggested quick fix: {suggested_eval_repo_id}"
        )
        if not prompt_yes_no(f"Apply quick fix and use {suggested_eval_repo_id}?", "y"):
            print("Cancelled.")
            return
        eval_dataset_name = suggested_eval_repo_id

    lerobot_dir = get_lerobot_dir(config)
    deploy_data_dir = get_deploy_data_dir(config)
    eval_repo_id, eval_adjusted, _ = resolve_unique_repo_id(
        username=username,
        dataset_name_or_repo_id=eval_dataset_name,
        local_roots=[deploy_data_dir, lerobot_dir / "data"],
    )
    if eval_adjusted:
        print(f"Auto-iterated eval dataset to avoid existing target: {eval_repo_id}")
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

    preflight_checks = run_preflight_for_deploy(
        config=config,
        model_path=model_path,
        eval_repo_id=eval_repo_id,
        command=eval_cmd,
    )
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

    config["last_dataset_repo_id"] = eval_repo_id
    save_config(config)
    print("Deployment command completed.")
    print("Done! ✓")


def run_config_mode(config: dict[str, Any]) -> dict[str, Any]:
    print_section("=== ⚙️ CONFIG MODE ===")
    print("Press Enter to keep defaults shown in brackets.")
    setup_status = probe_setup_wizard_status(config)
    print("\nSetup readiness check:")
    print(build_setup_status_summary(setup_status))
    if setup_status.needs_bootstrap:
        print("\nFirst-time setup guidance:")
        print(build_setup_wizard_guide(setup_status))
        if not prompt_yes_no("Continue to config prompts?", "y"):
            return config

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
    doctor_parser = subparsers.add_parser("doctor", help="Run local diagnostics for env, ports, and cameras.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics JSON.")
    compat_parser = subparsers.add_parser("compat", help="Probe LeRobot entrypoint/flag compatibility.")
    compat_parser.add_argument("--json", action="store_true", help="Emit compatibility probe as JSON.")
    compat_parser.add_argument("--refresh", action="store_true", help="Bypass cache and re-probe capabilities.")
    subparsers.add_parser("gui", help="Launch desktop GUI for config, record, and deploy.")
    subparsers.add_parser("gui-qt", help="Launch the Qt preview GUI while Tk remains the default.")
    subparsers.add_parser("install-launcher", help="Install a desktop launcher for the GUI (Linux: app menu entry; macOS: .app bundle).")
    support_bundle_parser = subparsers.add_parser("support-bundle", help="Export a local support bundle for a run.")
    support_bundle_parser.add_argument("--run-id", default="latest", help="Run id to export (or 'latest').")
    support_bundle_parser.add_argument("--output", required=True, help="Output zip path for support bundle.")
    profile_parser = subparsers.add_parser("profile", help="Import/export community profile templates.")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_mode", required=True)
    profile_export = profile_subparsers.add_parser("export", help="Export current config as a community profile.")
    profile_export.add_argument("--output", required=True, help="Output profile path (.yaml).")
    profile_export.add_argument("--name", default="", help="Optional profile display name.")
    profile_export.add_argument("--description", default="", help="Optional profile description.")
    profile_export.add_argument(
        "--include-paths",
        action="store_true",
        help="Include local path values (off by default for portability).",
    )
    profile_import = profile_subparsers.add_parser("import", help="Import a community profile into config.")
    profile_import.add_argument("--input", required=True, help="Profile file path (.yaml).")
    profile_import.add_argument(
        "--apply-paths",
        action="store_true",
        help="Apply profile path values during import (off by default).",
    )

    return parser.parse_args()


def _require_venv_on_macos() -> None:
    """On macOS, exit early with a clear message when no virtual environment is active."""
    if sys.platform != "darwin":
        return

    in_venv = (
        sys.prefix != sys.base_prefix
        or bool(os.environ.get("VIRTUAL_ENV"))
        or bool(os.environ.get("CONDA_PREFIX"))
        or bool(os.environ.get("CONDA_DEFAULT_ENV"))
    )
    if in_venv:
        return

    # Try to find a venv in the project directory to show a concrete command.
    project_dir = Path(__file__).resolve().parents[1]
    activate_cmd: str | None = None
    for candidate in (".venv", "venv", "env"):
        activate = project_dir / candidate / "bin" / "activate"
        if activate.exists():
            activate_cmd = f"source {project_dir}/{candidate}/bin/activate"
            break

    print("---")
    print("No virtual environment detected.")
    print()
    print("LeRobot Pipeline Manager must be run from inside a Python virtual")
    print("environment on macOS. Activate your venv first, then retry:")
    print()
    if activate_cmd:
        print(f"    {activate_cmd}")
    else:
        print("    source /path/to/your/venv/bin/activate")
    args_hint = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "gui"
    print(f"    python3 -m robot_pipeline_app {args_hint}")
    print()
    print("If you haven't set up a venv yet, see the README for instructions.")
    print("---")
    sys.exit(1)


def main() -> int:
    _require_venv_on_macos()
    args = parse_args()

    raw_config, source = load_raw_config()
    first_run = source is None

    if args.mode == "gui":
        from .gui_app import run_gui_mode

        run_gui_mode(raw_config)
        return 0

    if args.mode == "gui-qt":
        from .gui_qt_app import run_gui_qt_mode

        run_gui_qt_mode(raw_config)
        return 0

    if args.mode == "doctor":
        config = normalize_config_without_prompts(raw_config)
        run_doctor_mode(config, json_output=bool(getattr(args, "json", False)))
        return 0

    if args.mode == "compat":
        config = normalize_config_without_prompts(raw_config)
        return run_compat_mode(
            config,
            json_output=bool(getattr(args, "json", False)),
            refresh=bool(getattr(args, "refresh", False)),
        )

    if args.mode == "support-bundle":
        config = normalize_config_without_prompts(raw_config)
        return run_support_bundle_mode(
            config,
            run_id=str(getattr(args, "run_id", "latest")),
            output=str(getattr(args, "output", "")).strip(),
        )

    if args.mode == "profile":
        config = normalize_config_without_prompts(raw_config)
        profile_mode = str(getattr(args, "profile_mode", "")).strip()
        if profile_mode == "export":
            return run_profile_export_mode(
                config,
                output=str(getattr(args, "output", "")).strip(),
                name=str(getattr(args, "name", "")).strip(),
                description=str(getattr(args, "description", "")).strip(),
                include_paths=bool(getattr(args, "include_paths", False)),
            )
        if profile_mode == "import":
            return run_profile_import_mode(
                config,
                input_path=str(getattr(args, "input", "")).strip(),
                apply_paths=bool(getattr(args, "apply_paths", False)),
            )
        print("Unknown profile mode. Use: profile export|import")
        return 1

    if args.mode == "history":
        config = normalize_config_without_prompts(raw_config)
        run_history_mode(config, limit=max(int(args.limit), 1))
        return 0

    if args.mode == "install-launcher":
        launcher_config = normalize_config_without_prompts(raw_config)
        launcher_venv_dir = Path(str(launcher_config.get("lerobot_venv_dir", ""))).expanduser()
        install_result = install_desktop_launcher(
            app_dir=Path(__file__).resolve().parents[1],
            python_executable=Path(sys.executable),
            venv_dir=launcher_venv_dir,
        )
        print(install_result.message)
        if install_result.script_path is not None:
            print(f"- launcher script: {install_result.script_path}")
        if install_result.desktop_entry_path is not None:
            print(f"- desktop entry: {install_result.desktop_entry_path}")
        if install_result.icon_path is not None:
            print(f"- icon: {install_result.icon_path}")
        return 0 if install_result.ok else 1

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
