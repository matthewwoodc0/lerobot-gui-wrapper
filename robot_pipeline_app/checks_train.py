from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

from .checks_common import _activation_config_check, _configured_env_dir, _nearest_existing_parent
from .checks_deploy import _probe_torch_accelerator
from .compat import resolve_train_entrypoint
from .config_store import normalize_path
from .probes import probe_module_import, summarize_probe_error
from .repo_utils import normalize_repo_id
from .types import CheckResult

_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _configured_train_python(config: dict[str, Any]) -> Path | None:
    configured_env_dir = _configured_env_dir(config)
    for candidate in (
        configured_env_dir / "bin" / "python3",
        configured_env_dir / "bin" / "python",
        configured_env_dir / "Scripts" / "python.exe",
    ):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _repo_id_candidate(config: dict[str, Any], raw_value: str) -> str:
    return normalize_repo_id(str(config.get("hf_username", "")), raw_value)


def _entrypoint_resolvable(config: dict[str, Any], entrypoint: str) -> bool:
    try:
        if importlib.util.find_spec(entrypoint) is not None:
            return True
    except Exception:
        pass

    raw_lerobot_dir = str(config.get("lerobot_dir", "")).strip()
    if not raw_lerobot_dir:
        return False
    lerobot_dir = Path(raw_lerobot_dir).expanduser()
    checkout_paths = {
        "lerobot.train": lerobot_dir / "lerobot" / "train.py",
        "scripts.train": lerobot_dir / "scripts" / "train.py",
        "lerobot.scripts.train": lerobot_dir / "lerobot" / "scripts" / "train.py",
        "scripts.lerobot_train": lerobot_dir / "scripts" / "lerobot_train.py",
        "lerobot.scripts.lerobot_train": lerobot_dir / "lerobot" / "scripts" / "lerobot_train.py",
    }
    candidate = checkout_paths.get(entrypoint)
    return bool(candidate and candidate.exists())


def run_preflight_for_train(config: dict[str, Any], form_values: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    activation_level, activation_detail = _activation_config_check(config)
    checks.append((activation_level, "Environment activation", activation_detail))

    configured_env_dir = _configured_env_dir(config)
    checks.append(
        (
            "PASS" if configured_env_dir.exists() else "FAIL",
            "LeRobot venv folder",
            str(configured_env_dir),
        )
    )

    train_python = _configured_train_python(config)
    checks.append(
        (
            "PASS" if train_python is not None else "WARN",
            "LeRobot venv python",
            str(train_python) if train_python is not None else "python executable not found under configured venv folder",
        )
    )

    lerobot_ok, lerobot_msg = probe_module_import("lerobot")
    checks.append(
        (
            "PASS" if lerobot_ok else "FAIL",
            "Python module: lerobot",
            "import ok" if lerobot_ok else summarize_probe_error(lerobot_msg),
        )
    )

    train_entrypoint = resolve_train_entrypoint(config)
    checks.append(
        (
            "PASS" if _entrypoint_resolvable(config, train_entrypoint) else "FAIL",
            "Train entrypoint",
            train_entrypoint,
        )
    )

    dataset_text = str(form_values.get("dataset_repo_id", "")).strip()
    if not dataset_text:
        checks.append(("FAIL", "Dataset", "dataset is required"))
    else:
        dataset_path = Path(normalize_path(dataset_text))
        if dataset_path.exists():
            if dataset_path.is_dir():
                checks.append(("PASS", "Dataset", f"local dataset path: {dataset_path}"))
            else:
                checks.append(("FAIL", "Dataset", f"not a directory: {dataset_path}"))
        else:
            repo_id = _repo_id_candidate(config, dataset_text)
            if _HF_REPO_ID_PATTERN.fullmatch(repo_id):
                checks.append(("PASS", "Dataset", f"Hugging Face repo id: {repo_id}"))
            else:
                checks.append(("FAIL", "Dataset", f"not a valid local dataset path or repo id: {dataset_text}"))

    output_dir_raw = str(form_values.get("output_dir", "")).strip() or str(config.get("trained_models_dir", "outputs/train"))
    output_dir = Path(normalize_path(output_dir_raw))
    if output_dir.exists() and not output_dir.is_dir():
        checks.append(("FAIL", "Output directory", f"not a directory: {output_dir}"))
    else:
        probe_path = output_dir if output_dir.exists() else _nearest_existing_parent(output_dir)
        if probe_path is None:
            checks.append(("FAIL", "Output directory", f"no existing parent for path: {output_dir}"))
        elif not os.access(str(probe_path), os.W_OK):
            checks.append(("FAIL", "Output directory", f"no write permission for: {probe_path}"))
        else:
            checks.append(("PASS", "Output directory", f"writable path available via: {probe_path}"))

    device = str(form_values.get("device", "")).strip().lower()
    if not device:
        checks.append(("PASS", "Training device", "auto-detect"))
    elif device == "cuda":
        accelerator, accelerator_detail = _probe_torch_accelerator()
        checks.append(
            (
                "PASS" if accelerator == "cuda" else "WARN",
                "Training device",
                accelerator_detail if accelerator == "cuda" else f"CUDA requested, but {accelerator_detail.lower()}",
            )
        )
    elif device == "mps":
        if sys.platform != "darwin":
            checks.append(("WARN", "Training device", "MPS requested on a non-macOS host"))
        else:
            accelerator, accelerator_detail = _probe_torch_accelerator()
            checks.append(
                (
                    "PASS" if accelerator == "mps" else "WARN",
                    "Training device",
                    accelerator_detail if accelerator == "mps" else f"MPS requested, but {accelerator_detail.lower()}",
                )
            )
    else:
        checks.append(("PASS", "Training device", device))

    resume_from_raw = str(form_values.get("resume_from", "")).strip()
    if resume_from_raw:
        resume_path = Path(normalize_path(resume_from_raw))
        checks.append(
            (
                "PASS" if resume_path.exists() else "FAIL",
                "Resume checkpoint",
                str(resume_path) if resume_path.exists() else f"checkpoint path not found: {resume_path}",
            )
        )

    policy_type = str(form_values.get("policy_type", "")).strip()
    checks.append(
        (
            "PASS" if policy_type else "FAIL",
            "Policy type",
            policy_type or "(empty)",
        )
    )
    return checks
