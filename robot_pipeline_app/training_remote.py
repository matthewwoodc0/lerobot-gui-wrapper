from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config_store import normalize_path
from .types import TrainingProfile


def expect_wrapper_path() -> Path:
    return Path(__file__).resolve().parent / "scripts" / "ssh_secret_expect.tcl"


def _ssh_destination(profile: TrainingProfile) -> str:
    return f"{profile.username}@{profile.host}"


def _remote_path_for_shell(path: str) -> str:
    raw = str(path or "").strip()
    if raw == "~":
        return '"$HOME"'
    if raw.startswith("~/"):
        suffix = raw[2:]
        if not suffix:
            return '"$HOME"'
        return f'"$HOME"/{shlex.quote(suffix)}'
    return shlex.quote(raw)


def _remote_path_for_rsync(path: str) -> str:
    raw = str(path or "").strip()
    if raw == "~":
        return "$HOME"
    if raw.startswith("~/"):
        suffix = raw[2:]
        if not suffix:
            return "$HOME"
        return f"$HOME/{shlex.quote(suffix)}"
    return shlex.quote(raw)


def _ssh_base_args(profile: TrainingProfile) -> list[str]:
    known_hosts = str((Path.home() / ".ssh" / "known_hosts").resolve())
    args = [
        "-p",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]
    if profile.auth_mode == "ssh_key":
        args.extend(["-o", "BatchMode=yes"])
        if profile.identity_file:
            args.extend(["-i", profile.identity_file])
    else:
        args.extend(["-o", "BatchMode=no"])
    return args


def _wrap_with_expect(profile: TrainingProfile, raw_cmd: list[str]) -> list[str]:
    return [
        "expect",
        str(expect_wrapper_path()),
        "--host",
        profile.host,
        "--user",
        profile.username,
        "--port",
        str(profile.port),
        "--",
        *raw_cmd,
    ]


def build_remote_launch_command(profile: TrainingProfile, remote_command: str) -> list[str]:
    raw = [
        "ssh",
        *_ssh_base_args(profile),
        _ssh_destination(profile),
        str(remote_command),
    ]
    if profile.auth_mode == "password":
        return _wrap_with_expect(profile, raw)
    return raw


def _rsync_ssh_transport(profile: TrainingProfile) -> str:
    known_hosts = str((Path.home() / ".ssh" / "known_hosts").resolve())
    parts = [
        "ssh",
        "-p",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]
    if profile.auth_mode == "ssh_key" and profile.identity_file:
        parts.extend(["-i", profile.identity_file])
    return shlex.join(parts)


def command_uses_binary(cmd: list[str], binary: str) -> bool:
    needle = str(binary or "").strip()
    if not needle:
        return False
    for part in cmd:
        piece = str(part).strip()
        if not piece:
            continue
        if Path(piece).name == needle:
            return True
    return False


def build_pull_command(
    profile: TrainingProfile,
    remote_path: str,
    local_path: Path,
    prefer_rsync: bool = True,
) -> list[str]:
    use_rsync = bool(prefer_rsync and shutil.which("rsync"))
    if use_rsync:
        remote_target = f"{profile.username}@{profile.host}:{_remote_path_for_rsync(str(remote_path))}"
        raw = [
            "rsync",
            "-az",
            "--partial",
            "--info=progress2",
            "-e",
            _rsync_ssh_transport(profile),
            remote_target,
            str(local_path),
        ]
        if profile.auth_mode == "password":
            return _wrap_with_expect(profile, raw)
        return raw

    fallback, _ = build_sftp_pull_command(profile, remote_path, local_path)
    return fallback


def build_sftp_pull_command(
    profile: TrainingProfile,
    remote_path: str,
    local_path: Path,
) -> tuple[list[str], Path]:
    target = Path(normalize_path(str(local_path)))
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    destination_name = target.name

    batch_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="lerobot_sftp_", suffix=".txt", delete=False)
    batch_path = Path(batch_file.name)
    try:
        batch_file.write(f"lcd {str(target_parent)}\n")
        batch_file.write(f"get -r {str(remote_path)} {destination_name}\n")
    finally:
        batch_file.close()

    raw = [
        "sftp",
        "-P",
        str(profile.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={str((Path.home() / '.ssh' / 'known_hosts').resolve())}",
        "-b",
        str(batch_path),
        _ssh_destination(profile),
    ]
    if profile.auth_mode == "ssh_key" and profile.identity_file:
        raw[1:1] = ["-i", profile.identity_file]

    if profile.auth_mode == "password":
        return _wrap_with_expect(profile, raw), batch_path
    return raw, batch_path


def ensure_host_trusted(profile: TrainingProfile, messagebox: Any) -> tuple[bool, str]:
    if not profile.host:
        return False, "Profile host is empty."

    ssh_dir = Path.home() / ".ssh"
    known_hosts = ssh_dir / "known_hosts"
    try:
        ssh_dir.mkdir(parents=True, exist_ok=True)
        known_hosts.touch(exist_ok=True)
    except OSError as exc:
        return False, f"Unable to prepare ~/.ssh/known_hosts: {exc}"

    host_refs = [profile.host]
    if profile.port != 22:
        host_refs.insert(0, f"[{profile.host}]:{profile.port}")
    for ref in host_refs:
        try:
            check = subprocess.run(
                ["ssh-keygen", "-F", ref, "-f", str(known_hosts)],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception as exc:
            return False, f"Unable to check known_hosts: {exc}"
        if check.returncode == 0 and (check.stdout or "").strip():
            return True, "Host key is already trusted."

    keyscan_cmd = ["ssh-keyscan", "-p", str(profile.port), profile.host]
    try:
        scan = subprocess.run(
            keyscan_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, f"Unable to run ssh-keyscan: {exc}"

    if scan.returncode != 0 or not (scan.stdout or "").strip():
        detail = (scan.stderr or scan.stdout or "ssh-keyscan returned no key").strip()
        return False, f"Failed to fetch host key: {detail}"

    key_lines = [line.strip() for line in (scan.stdout or "").splitlines() if line.strip()]
    preview = "\n".join(key_lines[:2])
    approved = messagebox.askyesno(
        "Trust SSH Host?",
        (
            f"Host {profile.host}:{profile.port} is not in known_hosts.\n\n"
            f"Key preview:\n{preview}\n\n"
            "Trust and save this host key?"
        ),
    )
    if not approved:
        return False, "Host trust was not approved."

    try:
        with known_hosts.open("a", encoding="utf-8") as handle:
            for line in key_lines:
                handle.write(line + "\n")
    except OSError as exc:
        return False, f"Unable to save host key: {exc}"

    return True, "Host key saved to known_hosts."


def list_remote_dirs(profile: TrainingProfile, remote_path: str) -> tuple[list[str] | None, str | None]:
    target = str(remote_path or "").strip()
    if not target:
        return None, "Remote path is empty."
    quoted = _remote_path_for_shell(target)
    remote_cmd = (
        "set -e; "
        f"target={quoted}; "
        'if [ ! -d "$target" ]; then echo "__RP_MISSING_DIR__"; exit 44; fi; '
        'find "$target" -mindepth 1 -maxdepth 1 -type d | sed "s#^.*/##" | sort'
    )
    cmd = build_remote_launch_command(profile, remote_cmd)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return None, f"Failed to list remote directories: {exc}"

    output = (result.stdout or "").strip()
    if "__RP_MISSING_DIR__" in output:
        return None, f"Remote path not found: {target}"
    if result.returncode != 0:
        detail = (result.stderr or output or f"exit code {result.returncode}").strip()
        return None, f"Failed to list remote path: {detail}"

    names = [line.strip() for line in output.splitlines() if line.strip()]
    return names, None


def remote_path_exists(profile: TrainingProfile, remote_path: str) -> tuple[bool, str | None]:
    target = str(remote_path or "").strip()
    if not target:
        return False, "Remote path is empty."
    quoted = _remote_path_for_shell(target)
    remote_cmd = f'test -d {quoted} && echo "__RP_EXISTS__"'
    cmd = build_remote_launch_command(profile, remote_cmd)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return False, f"Failed to probe remote path: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, detail or "Remote path check failed."
    return "__RP_EXISTS__" in (result.stdout or ""), None


def run_pull_with_fallback(
    profile: TrainingProfile,
    remote_path: str,
    local_path: Path,
    run_fn: Callable[[list[str]], int],
) -> tuple[str, int]:
    primary = build_pull_command(profile, remote_path, local_path, prefer_rsync=True)
    primary_mode = "rsync" if command_uses_binary(primary, "rsync") else "sftp"
    if primary_mode == "sftp":
        code = int(run_fn(primary))
        return "sftp", code

    code = int(run_fn(primary))
    if code == 0:
        return "rsync", 0

    fallback, batch_file = build_sftp_pull_command(profile, remote_path, local_path)
    try:
        fallback_code = int(run_fn(fallback))
    finally:
        try:
            batch_file.unlink()
        except OSError:
            pass
    return "sftp", fallback_code
