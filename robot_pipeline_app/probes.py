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


def probe_module_import(
    module_name: str,
    *,
    timeout_s: float = 10.0,
    python_executable: str | None = None,
    cwd: str | Path | None = None,
) -> tuple[bool, str]:
    resolved_cwd = str(cwd) if cwd is not None else None
    try:
        result = subprocess.run(
            [str(python_executable or sys.executable), "-c", f"import {module_name}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            env=_camera_probe_env(),
            cwd=resolved_cwd,
        )
    except subprocess.TimeoutExpired:
        return False, f"module import timed out after {float(timeout_s):.1f}s"
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
    priority_markers = (
        "access has been denied",
        "permission denied",
        "timed out",
        "camera not opened",
        "no frame",
        "failed to properly initialize",
    )
    lowered_lines = [line.lower() for line in lines]
    for marker in priority_markers:
        for idx, lowered in enumerate(lowered_lines):
            if marker in lowered:
                return lines[idx]
    return lines[-1]


def parse_frame_dimensions(message: str) -> tuple[int, int] | None:
    match = FRAME_SIZE_PATTERN.search(message)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def probe_camera_capture(
    index_or_path: int | str,
    width: int,
    height: int,
    *,
    timeout_s: float = 4.0,
    backend_name: str | None = None,
) -> tuple[bool, str]:
    backend_arg = str(backend_name or "")
    script = (
        "import sys\n"
        "import time\n"
        "source_raw=sys.argv[1]; width=int(sys.argv[2]); height=int(sys.argv[3]); backend_name=sys.argv[4]\n"
        "try:\n"
        "    source=int(source_raw)\n"
        "except Exception:\n"
        "    source=source_raw\n"
        "import cv2\n"
        "backend=getattr(cv2, backend_name, None) if backend_name else None\n"
        "cap=cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)\n"
        "if cap is None or not cap.isOpened():\n"
        "    print('camera not opened')\n"
        "    raise SystemExit(2)\n"
        "cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)\n"
        "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)\n"
        "deadline=time.monotonic() + (1.2 if sys.platform == 'darwin' else 0.5)\n"
        "frame=None\n"
        "while time.monotonic() < deadline:\n"
        "    ok, candidate = cap.read()\n"
        "    if ok and candidate is not None:\n"
        "        frame=candidate\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        "cap.release()\n"
        "if frame is None:\n"
        "    print('camera opened but no frame')\n"
        "    raise SystemExit(3)\n"
        "h, w = frame.shape[:2]\n"
        "print(f'frame={w}x{h}')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(index_or_path), str(width), str(height), backend_arg],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(0.5, float(timeout_s)),
            env=_camera_probe_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"camera probe timed out after {float(timeout_s):.1f}s"
    if result.returncode == 0:
        return True, (result.stdout or "").strip() or "camera opened"
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"camera probe failed with exit code {result.returncode}"


def _camera_probe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    env.setdefault("QT_LOGGING_RULES", "*.debug=false;*.info=false")
    return env


def camera_fingerprint(index_or_path: int | str) -> str | None:
    parts: list[str] = []
    source_text = str(index_or_path).strip()
    if not source_text:
        return None

    index: int | None = None
    try:
        index = int(source_text)
    except ValueError:
        index = None

    if index is None:
        source_path = Path(source_text).expanduser()
        resolved_source = _safe_resolve(source_path) if source_path.exists() else None
        if resolved_source is not None:
            parts.append(f"node={resolved_source}")
        return "|".join(sorted(set(parts))) if parts else None

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
