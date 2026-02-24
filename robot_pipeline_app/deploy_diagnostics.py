from __future__ import annotations

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


def _file_markers(path: Path) -> tuple[bool, bool]:
    has_weights = False
    has_config = False

    try:
        entries = list(path.iterdir())
    except OSError:
        return False, False

    for entry in entries:
        if not entry.is_file():
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

        try:
            children = [p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            continue
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

    if "modulenotfounderror" in joined and "lerobot" in joined:
        add("Deploy environment error: 'lerobot' module is missing in the active Python env.")
        add("Activate your env before running: source ~/lerobot/lerobot_env/bin/activate")

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


def explain_runtime_slowdown(output_lines: list[str]) -> list[str]:
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

    hints: list[str] = []
    hints.append(
        f"Observed control loop slowdown: {min_hz:.1f}-{max_hz:.1f} Hz vs target {target:.0f} Hz."
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
