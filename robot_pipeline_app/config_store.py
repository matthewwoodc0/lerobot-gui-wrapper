from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import (
    CONFIG_FIELDS,
    DEFAULT_CONFIG_VALUES,
    DEFAULT_LEROBOT_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SECONDARY_CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    PRIMARY_CONFIG_PATH,
)


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


def save_config(config: dict[str, Any], quiet: bool = False) -> None:
    payload = json.dumps(config, indent=2) + "\n"

    PRIMARY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRIMARY_CONFIG_PATH.write_text(payload, encoding="utf-8")

    secondary_path = get_secondary_config_path(config)
    try:
        secondary_path.parent.mkdir(parents=True, exist_ok=True)
        secondary_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write secondary config file {secondary_path}: {exc}")

    if not quiet:
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


def get_lerobot_dir(config: dict[str, Any]) -> Path:
    return Path(normalize_path(config["lerobot_dir"]))


def ensure_runs_dir(config: dict[str, Any]) -> Path:
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir
