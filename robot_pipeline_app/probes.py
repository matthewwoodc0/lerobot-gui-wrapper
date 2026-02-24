from __future__ import annotations

import re
import subprocess
import sys


FRAME_SIZE_PATTERN = re.compile(r"frame=(\d+)x(\d+)")


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
