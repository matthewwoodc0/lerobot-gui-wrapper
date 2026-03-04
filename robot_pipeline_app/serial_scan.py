from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _natural_sort_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", str(value))
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _linux_sysfs_metadata(dev_name: str) -> tuple[str, str]:
    tty_device = Path("/sys/class/tty") / dev_name / "device"
    if not tty_device.exists():
        return "", ""
    probe_roots: list[Path] = [tty_device]
    try:
        probe_roots.extend(list(tty_device.parents)[:4])
    except Exception:
        pass
    manufacturer = ""
    product = ""
    for root in probe_roots:
        if not manufacturer:
            manufacturer = _read_text(root / "manufacturer")
        if not product:
            product = _read_text(root / "product")
        if manufacturer and product:
            break
    return manufacturer, product


def _scan_busy_ports(candidates: list[str]) -> dict[str, str]:
    lsof_path = shutil.which("lsof")
    if not lsof_path or not candidates:
        return {}
    try:
        result = subprocess.run(
            [lsof_path, *candidates],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {}
    output = (result.stdout or "").strip()
    if result.returncode != 0 or not output:
        return {}

    busy: dict[str, str] = {}
    for line in output.splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        cmd = parts[0]
        pid = parts[1]
        for token in reversed(parts):
            if token.startswith("/dev/"):
                busy[token] = f"{cmd} (pid {pid})"
                break
    return busy


def _likely_motor_controller(
    *,
    path: str,
    by_id: list[str],
    manufacturer: str,
    product: str,
) -> bool:
    haystack = " ".join([path, *by_id, manufacturer, product]).lower()
    keywords = (
        "feetech",
        "scservo",
        "servo",
        "usb serial",
        "wch",
        "ch340",
        "cp210",
        "ftdi",
        "arduino",
        "stm32",
    )
    return any(word in haystack for word in keywords) or "/dev/ttyacm" in path.lower()


def scan_robot_serial_ports() -> list[dict[str, Any]]:
    """Scan common robot serial ports and return metadata-rich entries."""
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/tty.usbserial*",
        "/dev/cu.usbserial*",
        "/dev/tty.usbmodem*",
        "/dev/cu.usbmodem*",
    ]
    candidates: set[str] = set()
    for pattern in patterns:
        for path in Path("/").glob(pattern.lstrip("/")):
            candidates.add(str(path))

    by_id_map: dict[str, list[str]] = {}
    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.exists():
        try:
            for link in by_id_dir.iterdir():
                if not link.is_symlink():
                    continue
                resolved = str(link.resolve())
                by_id_map.setdefault(resolved, []).append(link.name)
                candidates.add(resolved)
        except Exception:
            pass

    ordered = sorted(candidates, key=_natural_sort_key)
    busy_map = _scan_busy_ports(ordered)
    rows: list[dict[str, Any]] = []
    for path in ordered:
        dev_name = Path(path).name
        manufacturer = ""
        product = ""
        if path.startswith("/dev/tty"):
            manufacturer, product = _linux_sysfs_metadata(dev_name)
        by_id = sorted(by_id_map.get(path, []), key=_natural_sort_key)
        rows.append(
            {
                "path": path,
                "by_id": by_id,
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "busy": path in busy_map,
                "busy_detail": busy_map.get(path, ""),
                "manufacturer": manufacturer,
                "product": product,
                "likely_motor_controller": _likely_motor_controller(
                    path=path,
                    by_id=by_id,
                    manufacturer=manufacturer,
                    product=product,
                ),
            }
        )
    return rows


def suggest_follower_leader_ports(
    entries: list[dict[str, Any]],
    *,
    current_follower: str = "",
    current_leader: str = "",
) -> tuple[str | None, str | None]:
    """Suggest (follower, leader) ports from scanned entries."""
    def _preferred_path(item: dict[str, Any]) -> str:
        path = str(item.get("path", "")).strip()
        by_id = item.get("by_id") or []
        if by_id:
            first = str(by_id[0]).strip()
            if first:
                candidate = f"/dev/serial/by-id/{first}"
                if Path(candidate).exists():
                    return candidate
        return path

    normalized_entries: list[dict[str, Any]] = []
    for item in entries:
        preferred = _preferred_path(item)
        if not preferred:
            continue
        normalized_entries.append({**item, "preferred_path": preferred})

    raw_paths = [str(item.get("path", "")).strip() for item in normalized_entries if str(item.get("path", "")).strip()]
    paths = [str(item.get("preferred_path", "")).strip() for item in normalized_entries if str(item.get("preferred_path", "")).strip()]
    if not paths:
        return None, None
    available = set(paths) | set(raw_paths)
    follower = current_follower.strip()
    leader = current_leader.strip()
    if follower in available and leader in available and follower != leader:
        return follower, leader

    likely_entries = [item for item in normalized_entries if item.get("likely_motor_controller")]
    candidate_entries = likely_entries if len(likely_entries) >= 2 else normalized_entries
    if len(candidate_entries) == 1:
        only = str(candidate_entries[0].get("preferred_path", "")).strip()
        return only, None

    def _index(path: str) -> int:
        match = re.search(r"(\d+)$", Path(path).name)
        if match:
            return int(match.group(1))
        return 10**9

    def _entry_index(item: dict[str, Any]) -> int:
        raw = str(item.get("path", "")).strip()
        preferred = str(item.get("preferred_path", "")).strip()
        raw_index = _index(raw)
        preferred_index = _index(preferred)
        return raw_index if raw_index != 10**9 else preferred_index

    ordered_by_index = sorted(candidate_entries, key=_entry_index)
    ordered_paths = [str(item.get("preferred_path", "")).strip() for item in ordered_by_index]
    ordered_paths = [path for path in ordered_paths if path]
    if len(ordered_paths) == 1:
        return ordered_paths[0], None

    # Typical LeRobot convention:
    # leader on lower-numbered ACM/USB port, follower on higher-numbered port.
    leader_guess = ordered_paths[0]
    follower_guess = ordered_paths[-1]
    if follower_guess == leader_guess and len(ordered_paths) >= 2:
        follower_guess = ordered_paths[1]
    return follower_guess, leader_guess


def format_robot_port_scan(entries: list[dict[str, Any]]) -> str:
    walkthrough = (
        "Port assignment walkthrough (to avoid ACM0/ACM1 confusion):\n"
        "1. Leave both robots connected and click Scan Robot Ports.\n"
        "2. Note the currently listed /dev/ttyACM* (or /dev/ttyUSB*) ports.\n"
        "3. Unplug ONE robot USB cable, then scan again.\n"
        "4. The port that disappeared belongs to the robot you unplugged.\n"
        "5. Plug it back in, unplug the other robot, and scan again to confirm.\n"
        "6. Set Leader/Follower based on your physical labels (not ACM number).\n"
        "Tip: /dev/serial/by-id names are more stable than ACM0/ACM1 across reboots."
    )

    if not entries:
        return "No candidate serial robot ports found.\n\n" + walkthrough

    lines = ["Detected serial robot port candidates:"]
    for item in entries:
        path = str(item.get("path", ""))
        rw = f"R={'Y' if item.get('readable') else 'N'} W={'Y' if item.get('writable') else 'N'}"
        busy = f"busy={item.get('busy_detail')}" if item.get("busy") else "busy=no"
        likely = "likely motor controller" if item.get("likely_motor_controller") else "generic serial device"
        meta_parts = [part for part in (item.get("manufacturer"), item.get("product")) if str(part).strip()]
        meta = " | ".join(meta_parts) if meta_parts else "metadata unavailable"
        lines.append(f"- {path}  ({rw}, {busy}, {likely})")
        by_id = item.get("by_id") or []
        if by_id:
            lines.append(f"  by-id: {', '.join(str(v) for v in by_id)}")
        lines.append(f"  device: {meta}")
    lines.append("")
    lines.append(walkthrough)
    return "\n".join(lines)
