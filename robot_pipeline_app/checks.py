from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .config_store import get_deploy_data_dir, get_lerobot_dir, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .deploy_diagnostics import validate_model_path
from .probes import (
    camera_fingerprint,
    in_virtual_env,
    parse_frame_dimensions,
    probe_camera_capture,
    probe_module_import,
    serial_port_fingerprint,
    summarize_probe_error,
)
from .repo_utils import (
    has_eval_prefix,
    increment_dataset_name,
    normalize_repo_id,
    repo_name_from_repo_id,
    suggest_eval_dataset_name,
    suggest_eval_prefixed_repo_id,
)
from .types import CheckResult, PreflightReport

CommonChecksFn = Callable[[dict[str, Any]], list[CheckResult]]
WhichFn = Callable[[str], Optional[str]]

_CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_CAMERA_NAME_BLOCKLIST = {
    "width",
    "height",
    "fps",
    "shape",
    "dtype",
    "type",
    "index",
    "path",
    "device",
    "name",
    "mean",
    "std",
    "low",
    "high",
    "channels",
    "color_space",
    "normalization",
}
_HEAVY_MODEL_PATTERNS = (
    ("smolvlm", "SmolVLM"),
    ("vision_language", "vision-language"),
    ("vision-language", "vision-language"),
    ("video-instruct", "video-instruct"),
    ("vlm", "VLM"),
)


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



def _dialout_membership() -> tuple[bool | None, str]:
    if not sys.platform.startswith("linux"):
        return None, "dialout group check is only applicable on Linux."

    try:
        import grp
        import pwd
    except Exception as exc:
        return None, f"Unable to import group modules: {exc}"

    try:
        dialout_group = grp.getgrnam("dialout")
    except KeyError:
        return None, "Group 'dialout' not found on this system."

    try:
        user_name = pwd.getpwuid(os.getuid()).pw_name
        user_groups = set(os.getgroups())
    except Exception as exc:
        return None, f"Unable to inspect current user groups: {exc}"

    in_group = dialout_group.gr_gid in user_groups or user_name in dialout_group.gr_mem
    if in_group:
        return True, f"User '{user_name}' is in dialout."

    return (
        False,
        (
            f"User '{user_name}' is not in dialout. "
            "Fix: sudo usermod -a -G dialout $USER (then log out/in or run newgrp dialout)."
        ),
    )


def _serial_lock_check(ports: list[str]) -> CheckResult:
    unique_ports = sorted({str(port).strip() for port in ports if str(port).strip()})
    if not unique_ports:
        return ("WARN", "Serial port lock", "No accessible serial ports available for lock inspection.")

    lsof_path = shutil.which("lsof")
    if not lsof_path:
        return ("WARN", "Serial port lock", "lsof not found in PATH; cannot detect active serial lock holders.")

    try:
        result = subprocess.run(
            [lsof_path, *unique_ports],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return ("WARN", "Serial port lock", f"Unable to run lsof on configured serial ports: {exc}")

    stdout = result.stdout or ""
    if result.returncode == 1 and not stdout.strip():
        return ("PASS", "Serial port lock", "No active process holds configured serial ports.")

    if result.returncode == 0:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        holders = lines[1:] if len(lines) > 1 else []
        holder_preview = "; ".join(holders[:3]) if holders else "one or more lock holders detected"
        ports_text = " ".join(unique_ports)
        return (
            "FAIL",
            "Serial port lock",
            (
                f"Configured serial ports are currently in use ({holder_preview}). "
                f"Stop the holder first. Quick reset: lsof {ports_text}; sudo fuser -k {ports_text}"
            ),
        )

    detail = summarize_probe_error(result.stderr or stdout or f"exit code {result.returncode}")
    return ("WARN", "Serial port lock", f"Could not determine serial lock state: {detail}")


def _append_fingerprint_check(
    *,
    add: Callable[[str, str, str], None],
    check_name: str,
    baseline_value: str,
    current_value: str | None,
    missing_detail: str,
) -> None:
    baseline = str(baseline_value or "").strip()
    if current_value is None:
        add("WARN", check_name, missing_detail)
        return

    if baseline and baseline == current_value:
        add("PASS", check_name, "matches saved baseline")
        return

    if baseline and baseline != current_value:
        add("FAIL", check_name, f"saved={baseline}; current={current_value} (mapping drift detected)")
        return

    add("WARN", check_name, "no baseline saved yet; scan/assign once to lock mapping.")


def _inferred_role_from_fingerprint(fingerprint: str | None) -> str | None:
    if not fingerprint:
        return None
    lowered = fingerprint.lower()
    has_leader = "leader" in lowered
    has_follower = "follower" in lowered
    if has_leader and not has_follower:
        return "leader"
    if has_follower and not has_leader:
        return "follower"
    return None


def _append_robot_port_checks(config: dict[str, Any], add: Callable[[str, str, str], None]) -> None:
    follower_port = str(config.get("follower_port", "")).strip()
    leader_port = str(config.get("leader_port", "")).strip()
    named_ports = (("Follower port", "follower", follower_port), ("Leader port", "leader", leader_port))

    resolved_paths: dict[str, str] = {}
    accessible_ports: list[str] = []
    fingerprints: dict[str, str | None] = {}

    for label, role_key, port in named_ports:
        if not port:
            add("FAIL", label, "(empty)")
            continue

        port_path = Path(port)
        if not port_path.exists():
            add("FAIL", label, f"{port} (not found)")
            continue

        add("PASS", label, port)
        try:
            resolved_paths[role_key] = str(port_path.resolve())
        except OSError:
            resolved_paths[role_key] = port

        has_rw = os.access(port, os.R_OK | os.W_OK)
        add(
            "PASS" if has_rw else "FAIL",
            f"{label} access",
            "read/write ok" if has_rw else f"permission denied for {port}",
        )
        if has_rw:
            accessible_ports.append(port)

        fingerprint = serial_port_fingerprint(port)
        fingerprints[role_key] = fingerprint
        _append_fingerprint_check(
            add=add,
            check_name=f"{label} fingerprint",
            baseline_value=str(config.get(f"{role_key}_port_fingerprint", "")),
            current_value=fingerprint,
            missing_detail=f"could not fingerprint {port}; use stable /dev/serial/by-id path when available.",
        )

        inferred_role = _inferred_role_from_fingerprint(fingerprint)
        if inferred_role is None:
            add("WARN", f"{label} role inference", "could not infer role from serial fingerprint text.")
        elif inferred_role == role_key:
            add("PASS", f"{label} role inference", f"fingerprint label suggests '{role_key}'.")
        else:
            add(
                "FAIL",
                f"{label} role inference",
                f"fingerprint label suggests '{inferred_role}', not configured role '{role_key}'.",
            )

    if follower_port and leader_port:
        same_path = follower_port == leader_port
        same_resolved = (
            resolved_paths.get("follower") is not None
            and resolved_paths.get("leader") is not None
            and resolved_paths["follower"] == resolved_paths["leader"]
        )
        add(
            "FAIL" if same_path or same_resolved else "PASS",
            "Leader/Follower port uniqueness",
            (
                "leader and follower point to the same serial device"
                if same_path or same_resolved
                else "leader and follower ports are distinct"
            ),
        )

    follower_fp = fingerprints.get("follower")
    leader_fp = fingerprints.get("leader")
    if follower_fp and leader_fp:
        add(
            "FAIL" if follower_fp == leader_fp else "PASS",
            "Leader/Follower port identity",
            (
                "both role ports fingerprint to the same hardware device"
                if follower_fp == leader_fp
                else "role ports fingerprint to different hardware devices"
            ),
        )

    linux_acm_ports = [
        port
        for port in (follower_port, leader_port)
        if port and "ttyacm" in port.lower()
    ]
    if linux_acm_ports:
        in_dialout, dialout_detail = _dialout_membership()
        if in_dialout is True:
            add("PASS", "dialout group", dialout_detail)
        elif in_dialout is False:
            add("FAIL", "dialout group", dialout_detail)
        else:
            add("WARN", "dialout group", dialout_detail)

    lock_level, lock_name, lock_detail = _serial_lock_check(accessible_ports)
    add(lock_level, lock_name, lock_detail)


def _append_camera_checks(config: dict[str, Any], add: Callable[[str, str, str], None]) -> None:
    try:
        laptop_idx = int(config.get("camera_laptop_index", 0))
    except (TypeError, ValueError):
        laptop_idx = 0
    try:
        phone_idx = int(config.get("camera_phone_index", 1))
    except (TypeError, ValueError):
        phone_idx = 1

    add(
        "PASS" if laptop_idx != phone_idx else "FAIL",
        "Camera indices",
        f"laptop={laptop_idx}, phone={phone_idx}",
    )

    cv2_ok, cv2_msg = probe_module_import("cv2")
    add(
        "PASS" if cv2_ok else "FAIL",
        "Python module: cv2",
        "import ok" if cv2_ok else summarize_probe_error(cv2_msg),
    )
    if not cv2_ok:
        return

    width = 640
    height = 360
    role_specs = (
        (
            "laptop",
            "Laptop",
            laptop_idx,
            width,
            height,
        ),
        (
            "phone",
            "Phone",
            phone_idx,
            width,
            height,
        ),
    )

    role_fingerprints: dict[str, str | None] = {}
    for role, role_label, index, target_width, target_height in role_specs:
        opened, probe_detail = probe_camera_capture(index, target_width, target_height)
        if opened:
            add("PASS", f"Camera {index} probe", probe_detail)
        else:
            detail = summarize_probe_error(probe_detail)
            add(
                "FAIL",
                f"Camera {index} probe",
                f"{detail}; verify index {index} exists and camera is connected.",
            )

        actual = parse_frame_dimensions(probe_detail)
        if actual is not None:
            actual_width, actual_height = actual
            if (actual_width, actual_height) == (target_width, target_height):
                add(
                    "PASS",
                    f"{role_label} camera resolution",
                    f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}",
                )
            else:
                add(
                    "WARN",
                    f"{role_label} camera resolution",
                    (
                        f"configured={target_width}x{target_height}; detected={actual_width}x{actual_height}; "
                        "runtime will auto-detect camera frame size when building the command."
                    ),
                )
        elif opened:
            add("WARN", f"{role_label} camera resolution", "opened camera but could not parse detected frame size.")

        fingerprint = camera_fingerprint(index)
        role_fingerprints[role] = fingerprint
        _append_fingerprint_check(
            add=add,
            check_name=f"{role_label} camera fingerprint",
            baseline_value=str(config.get(f"camera_{role}_fingerprint", "")),
            current_value=fingerprint,
            missing_detail=f"could not fingerprint camera index {index} on this platform.",
        )

    laptop_fp = role_fingerprints.get("laptop")
    phone_fp = role_fingerprints.get("phone")
    if laptop_fp and phone_fp:
        add(
            "FAIL" if laptop_fp == phone_fp else "PASS",
            "Laptop/Phone camera identity",
            (
                "both roles map to the same camera fingerprint"
                if laptop_fp == phone_fp
                else "camera role fingerprints are distinct"
            ),
        )


def _collect_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned} if cleaned else set()
    if isinstance(value, list):
        items: set[str] = set()
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.add(cleaned)
        return items
    return set()


def _looks_like_camera_name(name: str) -> bool:
    normalized = name.strip()
    if not normalized:
        return False
    if normalized.lower() in _CAMERA_NAME_BLOCKLIST:
        return False
    return _CAMERA_NAME_PATTERN.match(normalized) is not None


def _collect_camera_keys_from_json(payload: Any, out: set[str]) -> None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).lower()
            if key in {"camera_keys", "image_keys", "observation_image_keys", "observation_camera_keys"}:
                out.update({item for item in _collect_strings(value) if _looks_like_camera_name(item)})
            elif key in {"cameras", "images"} and isinstance(value, dict):
                out.update({name for name in value.keys() if _looks_like_camera_name(str(name))})
            _collect_camera_keys_from_json(value, out)
        return

    if isinstance(payload, list):
        for item in payload:
            _collect_camera_keys_from_json(item, out)


def _extract_model_camera_keys(model_path: Path) -> tuple[set[str] | None, str]:
    if not model_path.exists() or not model_path.is_dir():
        return None, f"model path is not readable: {model_path}"

    json_files = sorted(path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    if not json_files:
        return None, "no JSON metadata files found in model payload"

    discovered: set[str] = set()
    source_files: list[str] = []

    for json_path in json_files[:12]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        before = len(discovered)
        _collect_camera_keys_from_json(payload, discovered)
        if len(discovered) > before:
            source_files.append(json_path.name)

    if not discovered:
        return None, "could not infer camera keys from model metadata JSON files"

    return discovered, f"detected via: {', '.join(source_files[:4])}"


def _collect_local_dataset_names(roots: list[Path]) -> set[str]:
    names: set[str] = set()
    for root in roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
            for child in root.iterdir():
                if child.is_dir():
                    names.add(child.name)
        except OSError:
            continue
    return names


def _is_numbered_neighbor(a: str, b: str) -> bool:
    try:
        return increment_dataset_name(a) == b or increment_dataset_name(b) == a
    except Exception:
        return False


def _possible_dataset_typo(dataset_repo_id: str, known_names: set[str]) -> str | None:
    dataset_name = repo_name_from_repo_id(dataset_repo_id)
    if not dataset_name or dataset_name in known_names:
        return None

    candidates = sorted(name for name in known_names if name and name != dataset_name)
    if not candidates:
        return None

    close_matches = difflib.get_close_matches(dataset_name, candidates, n=1, cutoff=0.8)
    if not close_matches:
        return None

    closest = close_matches[0]
    if _is_numbered_neighbor(dataset_name, closest):
        return None

    return (
        f"'{dataset_name}' is very close to existing dataset '{closest}'. "
        "Confirm this isn't a typo before recording/upload."
    )


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

    add(
        "PASS" if in_virtual_env() else "WARN",
        "Python environment",
        f"executable={sys.executable}",
    )

    _append_robot_port_checks(config, add)
    _append_camera_checks(config, add)

    return checks


def _probe_torch_accelerator() -> tuple[str, str]:
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'imported': False, 'error': str(exc)}))\n"
        "    raise SystemExit(0)\n"
        "cuda = bool(torch.cuda.is_available())\n"
        "mps_backend = getattr(torch.backends, 'mps', None)\n"
        "mps = bool(mps_backend and mps_backend.is_available())\n"
        "print(json.dumps({'imported': True, 'cuda': cuda, 'mps': mps, 'torch': getattr(torch, '__version__', '')}))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return "unknown", f"Unable to probe torch runtime: {exc}"

    payload = (result.stdout or "").strip()
    if not payload:
        return "unknown", "torch probe returned no output."

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "unknown", summarize_probe_error(payload)

    if not isinstance(data, dict):
        return "unknown", "torch probe returned invalid payload."

    if not bool(data.get("imported")):
        return "unknown", f"torch import unavailable: {data.get('error', 'unknown error')}"

    torch_version = str(data.get("torch", "")).strip()
    suffix = f" (torch {torch_version})" if torch_version else ""
    if bool(data.get("cuda")):
        return "cuda", f"CUDA available{suffix}"
    if bool(data.get("mps")):
        return "mps", f"MPS available{suffix}"
    return "cpu", f"CPU-only runtime{suffix}"


def _infer_model_runtime_risk(model_path: Path) -> str | None:
    lowered_name = model_path.name.lower()
    for token, label in _HEAVY_MODEL_PATTERNS:
        if token in lowered_name:
            return f"{label} hint from model path name"

    try:
        json_files = [path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() == ".json"]
    except OSError:
        return None

    for json_path in json_files[:14]:
        try:
            text = json_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for token, label in _HEAVY_MODEL_PATTERNS:
            if token in text:
                return f"{label} hint from {json_path.name}"
    return None


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
    episode_time_s: int | None = None,
    dataset_repo_id: str | None = None,
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

    if episode_time_s is not None:
        if episode_time_s <= 0:
            checks.append(("FAIL", "Episode duration", f"{episode_time_s}s is invalid; must be greater than zero."))
        elif episode_time_s < 8:
            checks.append(
                (
                    "WARN",
                    "Episode duration",
                    f"{episode_time_s}s is likely too short and may truncate tasks (recommended: 8-120s).",
                )
            )
        elif episode_time_s > 180:
            checks.append(
                (
                    "WARN",
                    "Episode duration",
                    f"{episode_time_s}s is long and may add dead time (recommended: 8-120s).",
                )
            )
        else:
            checks.append(("PASS", "Episode duration", f"{episode_time_s}s"))

    if dataset_repo_id:
        known_names = {
            str(config.get("last_dataset_name", "")).strip(),
            str(config.get("last_eval_dataset_name", "")).strip(),
        }
        known_names.update(_collect_local_dataset_names([dataset_root, get_lerobot_dir(config) / "data"]))
        typo_warning = _possible_dataset_typo(dataset_repo_id, {name for name in known_names if name})
        if typo_warning:
            checks.append(("WARN", "Dataset repo typo risk", typo_warning))

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


def run_preflight_for_teleop(
    config: dict[str, Any],
    control_fps: int | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)

    if control_fps is not None:
        if control_fps <= 0:
            checks.append(("FAIL", "Teleop control FPS", f"{control_fps} is invalid; must be greater than zero."))
        elif control_fps > 120:
            checks.append(("WARN", "Teleop control FPS", f"{control_fps} is high and may be unstable on CPU-bound hosts."))
        else:
            checks.append(("PASS", "Teleop control FPS", str(control_fps)))

    return checks


def run_preflight_for_deploy(
    config: dict[str, Any],
    model_path: Path,
    eval_repo_id: str | None = None,
    common_checks_fn: CommonChecksFn | None = None,
) -> list[CheckResult]:
    checks_fn = common_checks_fn or _run_common_preflight_checks
    checks = checks_fn(config)

    username = str(config.get("hf_username", "")).strip()
    eval_repo = str(eval_repo_id or "").strip()
    if not eval_repo:
        fallback_dataset = str(config.get("last_eval_dataset_name", "")).strip()
        eval_repo = normalize_repo_id(username, fallback_dataset)
    suggested_eval_repo, _ = suggest_eval_prefixed_repo_id(username, eval_repo)
    has_prefix = has_eval_prefix(eval_repo)

    checks.append(
        (
            "PASS" if has_prefix else "FAIL",
            "Eval dataset naming",
            eval_repo
            if has_prefix
            else f"Eval dataset repo must begin with 'eval_' (dataset part). Suggested quick fix: {suggested_eval_repo}",
        )
    )

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

    expected_camera_keys = {"laptop", "phone"}
    model_camera_keys, camera_key_detail = _extract_model_camera_keys(model_path)
    if model_camera_keys is None:
        checks.append(("WARN", "Model camera keys", camera_key_detail))
    elif model_camera_keys == expected_camera_keys:
        checks.append(("PASS", "Model camera keys", f"matches runtime keys {sorted(expected_camera_keys)}"))
    else:
        checks.append(
            (
                "FAIL",
                "Model camera keys",
                (
                    f"model={sorted(model_camera_keys)}; runtime={sorted(expected_camera_keys)}; "
                    "camera key mismatch between training and deployment"
                ),
            )
        )

    checks.append(_probe_policy_path_support())

    fps = int(config.get("camera_fps", 30))
    accelerator, accel_detail = _probe_torch_accelerator()
    accel_level = "PASS" if accelerator in {"cuda", "mps"} else "WARN"
    checks.append((accel_level, "Compute accelerator", accel_detail))

    if accelerator == "cpu" and fps >= 25:
        checks.append(
            (
                "WARN",
                "Deploy loop performance risk",
                (
                    f"camera_fps={fps} with CPU-only runtime. "
                    "Target 30Hz often drops to single-digit Hz during policy inference. "
                    "Consider camera_fps=8-15, smaller model, or GPU/MPS acceleration."
                ),
            )
        )

    model_risk = _infer_model_runtime_risk(model_path)
    if model_risk:
        checks.append(
            (
                "WARN",
                "Model inference load",
                f"{model_risk}. VLM-style policies are commonly slower than 30Hz without acceleration.",
            )
        )
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

    deploy_data_dir = get_deploy_data_dir(config)
    add(
        "PASS" if deploy_data_dir.exists() else "WARN",
        "Deploy data dir",
        str(deploy_data_dir),
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
    eval_targets = [deploy_data_dir / next_eval_name, lerobot_dir / "data" / next_eval_name]
    existing_eval_targets = [str(path) for path in eval_targets if path.exists()]
    add(
        "WARN" if existing_eval_targets else "PASS",
        "Next eval dataset collision",
        ", ".join(existing_eval_targets)
        if existing_eval_targets
        else f"{next_eval_name} is free in deploy/lerobot data targets",
    )

    huggingface_cli = shutil.which("huggingface-cli")
    add(
        "PASS" if huggingface_cli else "WARN",
        "huggingface-cli",
        huggingface_cli or "not found in PATH",
    )

    common_checks = _run_common_preflight_checks(config)
    for level, name, detail in common_checks:
        if name == "LeRobot folder":
            continue
        add(level, name, detail)

    return checks
