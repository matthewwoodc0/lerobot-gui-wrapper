from __future__ import annotations

import json
import re
from pathlib import Path


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_CONFIG_MARKERS = ("config", "policy_config", "model_config")
_CHECKPOINT_TOKEN_PATTERN = re.compile(r"(?:checkpoint|ckpt|step|iter|epoch)[-_]?(\d+)")
_TRAILING_NUMBER_PATTERN = re.compile(r"(\d+)$")
_CHECKPOINT_CONTAINER_NAMES = {"checkpoint", "checkpoints", "ckpt", "ckpts"}
_PREFERRED_PAYLOAD_NAMES = {"pretrained_model", "final", "latest", "last", "best_model", "best"}
_LOOP_SLOW_PATTERN = re.compile(
    r"record loop is running slower\s*\(([\d.]+)\s*hz\).*target fps\s*\(([\d.]+)\s*hz\)",
    re.IGNORECASE,
)
_SUPPRESSED_PROGRESS_PATTERN = re.compile(
    r"suppressed\s+(\d+)\s+carriage-return progress updates",
    re.IGNORECASE,
)
_ROBOT_CAMERAS_PREFIX = "--robot.cameras="
_MOTOR_ID_PATTERN = re.compile(r"id_?=\s*(\d+)", re.IGNORECASE)
_TTY_PATH_PATTERN = re.compile(r"(/dev/tty[\w.-]+)", re.IGNORECASE)
_MODULE_NOT_FOUND_PATTERN = re.compile(r"no module named '([^']+)'", re.IGNORECASE)

# Install hints for common ML/VLM dependencies that may be missing from a
# LeRobot environment when optional extras were not installed.
_ML_DEP_INSTALL_HINTS: dict[str, str] = {
    "transformers": (
        "pip install transformers  "
        "(or reinstall LeRobot with VLM extras: cd ~/lerobot && pip install -e '.[smolvla]')"
    ),
    "peft": "pip install peft  (required for LoRA/adapter-based policies)",
    "diffusers": "pip install diffusers",
    "accelerate": "pip install accelerate",
    "torch": "pip install torch  (follow the PyTorch install guide for your CUDA version)",
    "torchvision": "pip install torchvision",
    "torchaudio": "pip install torchaudio",
    "huggingface_hub": "pip install huggingface_hub",
    "timm": "pip install timm",
    "einops": "pip install einops",
    "tokenizers": "pip install tokenizers",
    "safetensors": "pip install safetensors",
    "sentencepiece": "pip install sentencepiece",
}


def _file_markers(path: Path) -> tuple[bool, bool]:
    has_weights = False
    has_config = False

    try:
        entries = list(path.iterdir())
    except OSError:
        return False, False

    for entry in entries:
        try:
            is_file = entry.is_file()
        except OSError:
            continue
        if not is_file:
            continue
        lower_name = entry.name.lower()
        if lower_name.endswith(_WEIGHT_SUFFIXES):
            has_weights = True
        if any(marker in lower_name for marker in _CONFIG_MARKERS):
            has_config = True
        if has_weights and has_config:
            break

    return has_weights, has_config


def is_runnable_model_path(model_path: Path) -> bool:
    if not model_path.exists() or not model_path.is_dir():
        return False
    has_weights, has_config = _file_markers(model_path)
    return has_weights and has_config


def _extract_checkpoint_step(parts: tuple[str, ...]) -> int:
    highest = -1
    for idx, part in enumerate(parts):
        lowered = part.lower()
        token_match = _CHECKPOINT_TOKEN_PATTERN.search(lowered)
        if token_match:
            highest = max(highest, int(token_match.group(1)))
            continue

        if lowered in _CHECKPOINT_CONTAINER_NAMES and idx + 1 < len(parts):
            next_part = parts[idx + 1].lower()
            if next_part.isdigit():
                highest = max(highest, int(next_part))
                continue
            trailing_match = _TRAILING_NUMBER_PATTERN.search(next_part)
            if trailing_match:
                highest = max(highest, int(trailing_match.group(1)))

    return highest


def _candidate_sort_key(root_path: Path, candidate_path: Path) -> tuple[int, int, int, int, str]:
    try:
        rel = candidate_path.relative_to(root_path)
        parts = rel.parts
    except ValueError:
        parts = candidate_path.parts

    lowered_parts = tuple(part.lower() for part in parts)
    leaf = lowered_parts[-1] if lowered_parts else candidate_path.name.lower()
    if leaf == "pretrained_model":
        payload_priority = 3
    elif "pretrained" in leaf and "model" in leaf:
        payload_priority = 2
    elif leaf in _PREFERRED_PAYLOAD_NAMES:
        payload_priority = 1
    else:
        payload_priority = 0

    path_priority = 1 if any(part in _PREFERRED_PAYLOAD_NAMES for part in lowered_parts) else 0
    checkpoint_step = _extract_checkpoint_step(parts)
    depth = len(parts)
    lex = "/".join(lowered_parts)
    return (-payload_priority, -path_priority, -checkpoint_step, depth, lex)


def find_nested_model_candidates(model_path: Path, max_depth: int = 4, limit: int = 12) -> list[Path]:
    if not model_path.exists() or not model_path.is_dir():
        return []

    candidates: list[Path] = []
    stack: list[tuple[Path, int]] = [(model_path, 0)]
    visited: set[Path] = set()

    while stack:
        current, depth = stack.pop()
        try:
            resolved = current.resolve()
        except OSError:
            resolved = current
        if resolved in visited:
            continue
        visited.add(resolved)

        if depth > 0 and is_runnable_model_path(current):
            candidates.append(current)

        if depth >= max_depth:
            continue

        children: list[Path] = []
        try:
            raw_children = list(current.iterdir())
        except OSError:
            continue
        for child in raw_children:
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if not is_dir or child.name.startswith("."):
                continue
            children.append(child)
        children.sort(key=lambda p: p.name, reverse=True)
        for child in children:
            stack.append((child, depth + 1))

    ordered = sorted(candidates, key=lambda path: _candidate_sort_key(model_path, path))
    if limit <= 0:
        return ordered
    return ordered[:limit]


def validate_model_path(model_path: Path) -> tuple[bool, str, list[Path]]:
    if not model_path.exists() or not model_path.is_dir():
        return False, f"Model folder not found: {model_path}", []

    if is_runnable_model_path(model_path):
        return True, f"Runnable model payload found: {model_path}", []

    candidates = find_nested_model_candidates(model_path)
    if not candidates:
        return (
            False,
            "Model folder does not look deployable (missing config + weight files in same folder).",
            [],
        )

    top = "\n".join(f"- {candidate}" for candidate in candidates[:3])
    return (
        False,
        "Selected folder is not directly deployable. Choose a nested model payload folder instead.\n"
        f"Examples:\n{top}",
        candidates,
    )


def explain_deploy_failure(output_lines: list[str], model_path: Path | None = None) -> list[str]:
    joined = "\n".join(output_lines[-240:]).lower()
    hints: list[str] = []

    def add(msg: str) -> None:
        if msg not in hints:
            hints.append(msg)

    missing_mods = _MODULE_NOT_FOUND_PATTERN.findall(joined)
    for mod in missing_mods:
        root = mod.split(".")[0]
        if root == "lerobot":
            add("Deploy environment error: 'lerobot' module is missing in the active Python env.")
            add("Activate your env before running: source ~/lerobot/lerobot_env/bin/activate")
        else:
            hint = _ML_DEP_INSTALL_HINTS.get(root)
            if hint:
                add(f"Missing Python dependency '{root}': not installed in the active env.")
                add(f"Fix: {hint}")
            else:
                add(f"Missing Python module '{mod}': not installed in the active env.")
                add("Fix: install the missing package or activate the correct virtual environment.")

    if (
        "policy.path" in joined
        and (
            "no such file or directory" in joined
            or "filenotfounderror" in joined
            or "not a directory" in joined
        )
    ):
        add("Policy path error: verify '--policy.path' points to a folder containing config + weight files.")
    elif model_path is not None and str(model_path).lower() in joined and "no such file or directory" in joined:
        add("Model path error: selected model folder was not readable at runtime.")

    if "unrecognized arguments" in joined or "could not override" in joined:
        add("CLI argument error: your installed LeRobot version may not match expected lerobot_record flags.")
        add("Run the previewed command manually with '--help' to verify supported options.")

    if "permission denied" in joined and ("/dev/tty" in joined or "ttyacm" in joined):
        add("Serial permission error: check access to follower/leader ports (e.g. /dev/ttyACM*).")

    motor_error_signals = (
        "motor",
        "servo",
        "joint",
        "not responding",
        "not respond",
        "timed out",
        "timeout",
        "over current",
        "overcurrent",
    )
    if any(signal in joined for signal in motor_error_signals):
        add("Motor/servo communication issue: stop the process and keep hands clear of pinch points.")
        add("Clean reset: release load, power-cycle arms for 5-10s, unplug/replug USB, then confirm /dev/ttyACM* re-enumerates.")
        add("If ports are stuck, run: lsof /dev/ttyACM0 /dev/ttyACM1 then sudo fuser -k /dev/ttyACM0 /dev/ttyACM1")
        add("Validate with a short warmup + short episode before full deployment.")

    if "can't open camera by index" in joined or "camera index out of range" in joined:
        add("Camera open error: verify camera indices and resolution in Config, then refresh camera scan.")

    if "failed to set capture_height" in joined or "failed to set capture_width" in joined:
        add("Camera resolution negotiation failed: selected camera likely enforces a different native frame size.")
        add("Re-scan cameras and re-assign laptop/phone roles, then retry (camera size is auto-detected at runtime).")

    if "cuda out of memory" in joined or "mps backend out of memory" in joined:
        add("GPU memory error: reduce camera resolution/fps or use a smaller model/checkpoint.")

    if not hints:
        add("Deploy failed. Open the latest run artifact command.log and find the first traceback/error line.")

    return hints


def _extract_flag_value(command: list[str] | None, flag_name: str) -> str:
    if not command:
        return ""
    prefix = f"--{flag_name}="
    for arg in command:
        text = str(arg or "").strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""


def explain_runtime_failure(
    output_lines: list[str],
    command: list[str] | None = None,
    run_mode: str | None = None,
) -> list[str]:
    joined_raw = "\n".join(output_lines[-320:])
    joined = joined_raw.lower()
    mode = str(run_mode or "").strip().lower()
    hints: list[str] = []

    def add(msg: str) -> None:
        if msg and msg not in hints:
            hints.append(msg)

    follower_port = _extract_flag_value(command, "robot.port")
    leader_port = _extract_flag_value(command, "teleop.port")
    known_ports = [port for port in (follower_port, leader_port) if port]
    matched_ports = sorted({match.group(1) for match in _TTY_PATH_PATTERN.finditer(joined_raw)})
    all_ports = sorted({*known_ports, *matched_ports})
    port_summary = ", ".join(all_ports)

    missing_mods = _MODULE_NOT_FOUND_PATTERN.findall(joined)
    for mod in missing_mods:
        root = mod.split(".")[0]
        if root == "lerobot":
            add("Environment error: 'lerobot' module is missing in the active Python environment.")
            add("Fix: activate your env and relaunch the GUI from that shell.")
        else:
            hint = _ML_DEP_INSTALL_HINTS.get(root)
            if hint:
                add(f"Missing Python dependency '{root}': not installed in the active env.")
                add(f"Fix: {hint}")
            else:
                add(f"Missing Python module '{mod}': not installed in the active env.")
                add("Fix: install the missing package or activate the correct virtual environment.")

    serial_error_signals = (
        "serialexception",
        "could not open port",
        "permission denied",
        "device or resource busy",
        "input/output error",
    )
    if any(signal in joined for signal in serial_error_signals):
        if port_summary:
            add(f"Serial port access error on: {port_summary}.")
        else:
            add("Serial port access error detected.")
        add("Fix: unplug/replug USB, close any other app using the ports, then re-scan and reapply follower/leader ports.")
        add("If on Linux, verify user serial permissions (dialout/uucp group) and reconnect the device after permission changes.")

    if "there is no status packet" in joined or ("txrxresult" in joined and "status packet" in joined):
        id_match = _MOTOR_ID_PATTERN.search(joined_raw)
        if id_match:
            add(
                f"Motor bus timeout: no status packet from motor id {id_match.group(1)}."
            )
        else:
            add("Motor bus timeout: no status packet from one or more motors.")
        add("Fix: verify arm power and daisy-chain motor cables, then power-cycle the arm(s) for 5-10 seconds.")
        add("Fix: confirm robot/teleop IDs and calibration files match the connected physical arms.")
        add("Fix: rerun calibration for each arm if IDs or wiring changed since last successful run.")

    if (
        "mismatch between calibration values in the motor and the calibration file" in joined
        or "no calibration file found" in joined
    ):
        add("Calibration mismatch detected between motor EEPROM values and selected calibration files.")
        add("Fix: keep calibration_dir + robot.id paired to the same arm profile, or rerun calibration for current hardware.")

    if "unrecognized arguments" in joined or "could not override" in joined:
        add("CLI argument mismatch: your installed LeRobot version may use different flags.")
        add("Fix: run the generated command with '--help' and align advanced overrides to supported options.")

    if "can't open camera by index" in joined or "camera index out of range" in joined:
        add("Camera open failure: one or more configured camera indices are unavailable.")
        add("Fix: rescan cameras and reassign laptop/phone roles before rerunning.")

    if "failed to set capture_height" in joined or "failed to set capture_width" in joined:
        add("Camera resolution negotiation failed for the active camera backend.")
        add("Fix: lower camera FPS or reassign to a camera that supports the requested mode.")

    if "cuda out of memory" in joined or "mps backend out of memory" in joined:
        add("GPU memory exhausted during run.")
        add("Fix: lower camera FPS/resolution or use a smaller model checkpoint.")

    if mode in {"teleop", "record", "deploy"}:
        add("Always rerun preflight after applying fixes to confirm ports, calibration, and env state are green.")

    if not hints:
        add("Run failed without a known signature. Use run artifacts command.log and start at the first traceback/error line.")

    return hints


def _extract_camera_specs_from_command(command: list[str] | None) -> list[tuple[str, int, int, int]]:
    if not command:
        return []
    cameras_raw = ""
    for arg in command:
        text = str(arg)
        if text.startswith(_ROBOT_CAMERAS_PREFIX):
            cameras_raw = text[len(_ROBOT_CAMERAS_PREFIX) :]
            break
    if not cameras_raw:
        return []

    try:
        payload = json.loads(cameras_raw)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    specs: list[tuple[str, int, int, int]] = []
    for name, item in payload.items():
        if not isinstance(item, dict):
            continue
        try:
            width = int(item.get("width", 0))
            height = int(item.get("height", 0))
            fps = int(item.get("fps", 0))
        except Exception:
            continue
        if width <= 0 or height <= 0 or fps <= 0:
            continue
        specs.append((str(name), width, height, fps))
    return specs


def summarize_camera_command_load(command: list[str] | None) -> str | None:
    specs = _extract_camera_specs_from_command(command)
    if not specs:
        return None

    details = ", ".join(f"{name}={width}x{height}@{fps}" for name, width, height, fps in specs)
    total_mpix_per_s = sum((width * height * fps) / 1_000_000.0 for _, width, height, fps in specs)
    return f"Record camera load: {details} ({total_mpix_per_s:.1f} MPix/s aggregate)."


def _extract_suppressed_progress_updates(output_lines: list[str]) -> int:
    total = 0
    for line in output_lines[-600:]:
        match = _SUPPRESSED_PROGRESS_PATTERN.search(str(line))
        if not match:
            continue
        try:
            total += int(match.group(1))
        except Exception:
            continue
    return total


def explain_runtime_slowdown(output_lines: list[str], command: list[str] | None = None) -> list[str]:
    slowed: list[tuple[float, float]] = []
    for line in output_lines[-600:]:
        match = _LOOP_SLOW_PATTERN.search(str(line))
        if not match:
            continue
        try:
            actual_hz = float(match.group(1))
            target_hz = float(match.group(2))
        except (TypeError, ValueError):
            continue
        if actual_hz > 0 and target_hz > 0:
            slowed.append((actual_hz, target_hz))

    if not slowed:
        return []

    min_hz = min(actual for actual, _ in slowed)
    max_hz = max(actual for actual, _ in slowed)
    target = slowed[0][1]
    ratio = min_hz / target if target > 0 else 0.0
    command_specs = _extract_camera_specs_from_command(command)
    aggregate_mpix_per_s = sum((width * height * fps) / 1_000_000.0 for _, width, height, fps in command_specs)
    suppressed_updates = _extract_suppressed_progress_updates(output_lines)

    hints: list[str] = []
    hints.append(
        f"Observed control loop slowdown: {min_hz:.1f}-{max_hz:.1f} Hz vs target {target:.0f} Hz."
    )
    if command_specs:
        command_summary = ", ".join(f"{name}={width}x{height}@{fps}" for name, width, height, fps in command_specs)
        hints.append(
            f"Command camera load: {command_summary} ({aggregate_mpix_per_s:.1f} MPix/s aggregate)."
        )
    if suppressed_updates > 0:
        hints.append(
            "UI output overhead mitigated: "
            f"suppressed {suppressed_updates} carriage-return progress updates."
        )
    if ratio < 0.8 and aggregate_mpix_per_s >= 30.0:
        hints.append(
            "Likely bottleneck: camera capture + video encode/disk I/O throughput from current command camera flags."
        )
    if ratio < 0.25:
        hints.append(
            "Likely bottleneck is policy inference compute (CPU/GPU), not GUI rendering."
        )
    hints.append(
        "Common fixes: reduce camera_fps to 8-15, use a lighter policy checkpoint, or run with CUDA/MPS acceleration."
    )
    hints.append(
        "If using VLM-style policies, single-digit Hz is common on CPU-only systems."
    )
    return hints
