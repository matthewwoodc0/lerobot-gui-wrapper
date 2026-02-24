from __future__ import annotations

from pathlib import Path


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_CONFIG_MARKERS = ("config", "policy_config", "model_config")


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


def find_nested_model_candidates(model_path: Path, max_depth: int = 4, limit: int = 12) -> list[Path]:
    if not model_path.exists() or not model_path.is_dir():
        return []

    candidates: list[Path] = []
    stack: list[tuple[Path, int]] = [(model_path, 0)]
    visited: set[Path] = set()

    while stack:
        current, depth = stack.pop()
        resolved = current.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)

        if depth > 0 and is_runnable_model_path(current):
            candidates.append(current)
            if len(candidates) >= limit:
                break

        if depth >= max_depth:
            continue

        try:
            children = [p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            continue
        children.sort(key=lambda p: p.name, reverse=True)
        for child in children:
            stack.append((child, depth + 1))

    return sorted(candidates, key=lambda p: (len(p.parts), p.name))


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

    if "can't open camera by index" in joined or "camera index out of range" in joined:
        add("Camera open error: verify camera indices and resolution in Config, then refresh camera scan.")

    if "cuda out of memory" in joined or "mps backend out of memory" in joined:
        add("GPU memory error: reduce camera resolution/fps or use a smaller model/checkpoint.")

    if not hints:
        add("Deploy failed. Open the latest run artifact command.log and find the first traceback/error line.")

    return hints
