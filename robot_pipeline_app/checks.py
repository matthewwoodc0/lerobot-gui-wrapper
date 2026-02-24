from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .config_store import get_lerobot_dir, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .deploy_diagnostics import validate_model_path
from .probes import probe_camera_capture, probe_module_import, summarize_probe_error
from .repo_utils import increment_dataset_name, repo_name_from_repo_id, suggest_eval_dataset_name
from .types import CheckResult, PreflightReport

CommonChecksFn = Callable[[dict[str, Any]], list[CheckResult]]
WhichFn = Callable[[str], Optional[str]]


def _check_counts(checks: list[CheckResult]) -> tuple[int, int, int]:
    pass_count = sum(1 for level, _, _ in checks if level == "PASS")
    warn_count = sum(1 for level, _, _ in checks if level == "WARN")
    fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
    return pass_count, warn_count, fail_count


def build_preflight_report(checks: list[CheckResult]) -> PreflightReport:
    pass_count, warn_count, fail_count = _check_counts(checks)
    return PreflightReport(
        checks=checks,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
    )


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


def _probe_policy_path_support() -> CheckResult:
    timeout_s = 20
    cmd_variants = (
        [sys.executable, "-m", "lerobot.scripts.lerobot_record", "--help"],
        [sys.executable, "-m", "lerobot.scripts.lerobot_record", "-h"],
    )
    saw_timeout = False
    for cmd in cmd_variants:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            saw_timeout = True
            continue
        except Exception as exc:
            return ("WARN", "lerobot_record policy flag", f"Unable to probe help output: {exc}")

        text = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
        if "--policy.path" in text:
            return ("PASS", "lerobot_record policy flag", "--policy.path supported")
        if "modulenotfounderror" in text and "lerobot" in text:
            return ("FAIL", "lerobot_record policy flag", "lerobot module not importable in active env")

    if saw_timeout:
        return (
            "WARN",
            "lerobot_record policy flag",
            f"Help probe timed out after {timeout_s}s; skipping policy flag verification (non-blocking).",
        )

    return (
        "WARN",
        "lerobot_record policy flag",
        "Could not confirm '--policy.path' in lerobot_record help output (non-blocking).",
    )


def run_preflight_for_record(
    config: dict[str, Any],
    dataset_root: Path,
    upload_enabled: bool,
    common_checks_fn: CommonChecksFn | None = None,
    which_fn: WhichFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)
    which = which_fn or shutil.which

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
        hf_cli = which("huggingface-cli")
        checks.append(
            (
                "PASS" if hf_cli else "FAIL",
                "huggingface-cli",
                hf_cli or "not found in PATH",
            )
        )

    return checks


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)
    is_valid_model, detail, candidates = validate_model_path(model_path)
    checks.append(("PASS" if model_path.exists() and model_path.is_dir() else "FAIL", "Model folder", str(model_path)))
    checks.append(("PASS" if is_valid_model else "FAIL", "Model payload", detail))
    if candidates:
        checks.append(
            (
                "WARN",
                "Model payload candidates",
                ", ".join(str(path) for path in candidates[:3]),
            )
        )
    checks.append(_probe_policy_path_support())
    return checks


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

    next_record_name = increment_dataset_name(repo_name_from_repo_id(str(config.get("last_dataset_name", "dataset_1"))))
    record_targets = [record_data_dir / next_record_name, lerobot_dir / "data" / next_record_name]
    existing_record_targets = [str(path) for path in record_targets if path.exists()]
    add(
        "WARN" if existing_record_targets else "PASS",
        "Next record dataset collision",
        ", ".join(existing_record_targets)
        if existing_record_targets
        else f"{next_record_name} is free in record/lerobot data targets",
    )

    next_eval_name = repo_name_from_repo_id(
        suggest_eval_dataset_name(config, str(config.get("last_model_name", "")))
    )
    eval_target = lerobot_dir / "data" / next_eval_name
    add(
        "WARN" if eval_target.exists() else "PASS",
        "Next eval dataset collision",
        str(eval_target) if eval_target.exists() else f"{next_eval_name} is free in {lerobot_dir / 'data'}",
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
