from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def in_virtual_env() -> bool:
    """Return True when running inside a venv or conda environment."""
    if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
        return True
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return True
    # Conda environments do not always expose a differing base_prefix.
    # Treat a prefix containing conda metadata as an active managed env.
    return (Path(sys.prefix) / "conda-meta").is_dir()


FRAME_SIZE_PATTERN = re.compile(r"frame=(\d+)x(\d+)")


def _safe_resolve(path: Path) -> str | None:
    try:
        return str(path.resolve())
    except Exception:
        return None


def probe_module_import(module_name: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"exit code {result.returncode}"


def summarize_probe_error(raw_message: str) -> str:
    lines = [line.strip() for line in raw_message.splitlines() if line.strip()]
    if not lines:
        return "unknown error"
    return lines[-1]


def parse_frame_dimensions(message: str) -> tuple[int, int] | None:
    match = FRAME_SIZE_PATTERN.search(message)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def probe_camera_capture(index: int, width: int, height: int) -> tuple[bool, str]:
    script = (
        "import sys\n"
        "idx=int(sys.argv[1]); width=int(sys.argv[2]); height=int(sys.argv[3])\n"
        "import cv2\n"
        "cap=cv2.VideoCapture(idx)\n"
        "if cap is None or not cap.isOpened():\n"
        "    print('camera not opened')\n"
        "    raise SystemExit(2)\n"
        "cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)\n"
        "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)\n"
        "ok, frame = cap.read()\n"
        "cap.release()\n"
        "if not ok or frame is None:\n"
        "    print('camera opened but no frame')\n"
        "    raise SystemExit(3)\n"
        "h, w = frame.shape[:2]\n"
        "print(f'frame={w}x{h}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(index), str(width), str(height)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, (result.stdout or "").strip() or "camera opened"
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"camera probe failed with exit code {result.returncode}"


def camera_fingerprint(index: int) -> str | None:
    parts: list[str] = []
    node = Path(f"/dev/video{index}")
    resolved_node = _safe_resolve(node) if node.exists() else None
    if resolved_node is not None:
        parts.append(f"node={resolved_node}")

    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.exists():
        try:
            for link in sorted(by_id_dir.iterdir()):
                if not link.is_symlink():
                    continue
                resolved = _safe_resolve(link)
                if resolved is None:
                    continue
                if resolved.endswith(f"/video{index}"):
                    parts.append(f"id={link.name}")
        except OSError:
            pass

    sys_device = Path(f"/sys/class/video4linux/video{index}/device")
    if sys_device.exists():
        resolved_sys = _safe_resolve(sys_device)
        if resolved_sys is not None:
            parts.append(f"sys={resolved_sys}")

    if not parts:
        return None
    return "|".join(sorted(set(parts)))


def serial_port_fingerprint(port: str) -> str | None:
    cleaned = str(port or "").strip()
    if not cleaned:
        return None

    port_path = Path(cleaned)
    if not port_path.exists():
        return None

    parts: list[str] = []
    resolved_port = _safe_resolve(port_path)
    if resolved_port is not None:
        parts.append(f"node={resolved_port}")

    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.exists():
        try:
            for link in sorted(by_id_dir.iterdir()):
                if not link.is_symlink():
                    continue
                resolved_link = _safe_resolve(link)
                if resolved_link is None or resolved_port is None:
                    continue
                if resolved_link == resolved_port:
                    parts.append(f"id={link.name}")
        except OSError:
            pass

    if not parts:
        return None
    return "|".join(sorted(set(parts)))
