from __future__ import annotations

import json
import os
import platform
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import list_runs
from .checks import summarize_checks
from .compat_snapshot import build_compat_snapshot
from .config_store import normalize_config_without_prompts, normalize_path
from .constants import DEFAULT_RUNS_DIR
from .diagnostics import checks_to_events, summarize_events
from .utils_common import parse_bool_value

_SENSITIVE_KEY_PATTERN = re.compile(r"(token|secret|password|api[_-]?key|cookie)", flags=re.IGNORECASE)
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{16,}")
_ENV_EXPORT_PATTERN = re.compile(
    r"(?P<key>HF_TOKEN|HUGGINGFACE_HUB_TOKEN|AWS_SECRET_ACCESS_KEY|OPENAI_API_KEY)\s*=\s*(?P<value>\S+)",
    flags=re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_PATTERN = re.compile(
    r"(?P<key>\b[A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|API[_-]?KEY|COOKIE)\b)\s*=\s*(?P<value>\S+)",
    flags=re.IGNORECASE,
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?P<prefix>(?:[?&]|(?<=\s)|(?<=^))(?:access_token|refresh_token|token|api[_-]?key|apikey|signature|sig)=)"
    r"(?P<value>[^&\s\"']+)",
    flags=re.IGNORECASE,
)
_BEARER_TOKEN_PATTERN = re.compile(
    r"(?P<prefix>\bBearer\s+)(?P<token>[A-Za-z0-9._~+/=-]{12,})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SupportBundleResult:
    ok: bool
    bundle_path: Path | None
    message: str
    run_id: str | None = None


def _sanitize_text(text: str, *, home_dir: str, redact_paths: bool, redact_env: bool) -> str:
    sanitized = str(text or "")
    if redact_paths and home_dir:
        sanitized = sanitized.replace(home_dir, "~")
    if redact_env:
        sanitized = _HF_TOKEN_PATTERN.sub("hf_***REDACTED***", sanitized)
        sanitized = _ENV_EXPORT_PATTERN.sub(lambda m: f"{m.group('key')}=***REDACTED***", sanitized)
        sanitized = _QUERY_SECRET_PATTERN.sub(lambda m: f"{m.group('prefix')}***REDACTED***", sanitized)
        sanitized = _GENERIC_SECRET_ASSIGN_PATTERN.sub(lambda m: f"{m.group('key')}=***REDACTED***", sanitized)
        sanitized = _BEARER_TOKEN_PATTERN.sub(lambda m: f"{m.group('prefix')}***REDACTED***", sanitized)
    return sanitized


def _sanitize_value(
    value: Any,
    *,
    home_dir: str,
    redact_paths: bool,
    redact_env: bool,
) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if redact_env and _SENSITIVE_KEY_PATTERN.search(key_text):
                sanitized[key_text] = "***REDACTED***"
                continue
            sanitized[key_text] = _sanitize_value(
                item,
                home_dir=home_dir,
                redact_paths=redact_paths,
                redact_env=redact_env,
            )
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_value(item, home_dir=home_dir, redact_paths=redact_paths, redact_env=redact_env)
            for item in value
        ]
    if isinstance(value, str):
        return _sanitize_text(value, home_dir=home_dir, redact_paths=redact_paths, redact_env=redact_env)
    return value


def _resolve_run_path(config: dict[str, Any], run_id: str) -> tuple[Path | None, str | None]:
    normalized_run_id = str(run_id or "latest").strip() or "latest"
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))

    if normalized_run_id == "latest":
        runs, _ = list_runs(config=config, limit=1)
        if not runs:
            return None, None
        latest = runs[0]
        run_path_raw = latest.get("_run_path")
        if not run_path_raw:
            return None, None
        return Path(str(run_path_raw)), str(latest.get("run_id") or Path(str(run_path_raw)).name)

    return runs_dir / normalized_run_id, normalized_run_id


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.expanduser().resolve(strict=False)
        resolved_root = root.expanduser().resolve(strict=False)
    except Exception:
        return False
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _extract_preflight_checks(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    raw_checks = metadata.get("preflight_checks")
    checks: list[tuple[str, str, str]] = []
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            level = str(item.get("level", "")).strip()
            name = str(item.get("name", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if not level or not name:
                continue
            checks.append((level, name, detail))
    return checks


def _check_counts(checks: list[tuple[str, str, str]]) -> tuple[int, int, int]:
    pass_count = sum(1 for level, _, _ in checks if str(level).upper() == "PASS")
    warn_count = sum(1 for level, _, _ in checks if str(level).upper() == "WARN")
    fail_count = sum(1 for level, _, _ in checks if str(level).upper() == "FAIL")
    return pass_count, warn_count, fail_count


def build_compatibility_snapshot(config: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = build_compat_snapshot(config)
    metadata_snapshot = metadata.get("compat_snapshot") if isinstance(metadata, dict) else None
    if isinstance(metadata_snapshot, dict):
        snapshot["metadata_compat_snapshot"] = metadata_snapshot
    return snapshot


def build_environment_probe(redact_env: bool) -> dict[str, Any]:
    env_keys = [
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PATH",
        "PYTHONPATH",
        "SHELL",
        "LANG",
        "TERM",
        "HF_HOME",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ]
    raw_env = {key: os.environ.get(key, "") for key in env_keys if os.environ.get(key, "")}
    if redact_env:
        env_payload = {
            key: ("***REDACTED***" if _SENSITIVE_KEY_PATTERN.search(key) else value)
            for key, value in raw_env.items()
        }
    else:
        env_payload = raw_env

    probe = {
        "captured_at_iso": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": env_payload,
    }
    try:
        probe["uid"] = os.getuid()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        probe["gid"] = os.getgid()  # type: ignore[attr-defined]
    except Exception:
        pass
    return probe


def create_support_bundle(
    *,
    config: dict[str, Any],
    run_id: str,
    output_path: Path,
) -> SupportBundleResult:
    runs_dir = Path(normalize_path(config.get("runs_dir", DEFAULT_RUNS_DIR)))
    run_path, resolved_run_id = _resolve_run_path(config, run_id)
    if run_path is None or resolved_run_id is None:
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message="No run artifacts were found. Run record/deploy once before exporting a support bundle.",
        )
    if not _is_path_within(run_path, runs_dir):
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Invalid run id '{resolved_run_id}': path traversal is not allowed.",
            run_id=resolved_run_id,
        )
    if not run_path.exists() or not run_path.is_dir():
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Run artifacts not found for run id '{resolved_run_id}'.",
            run_id=resolved_run_id,
        )

    metadata_path = run_path / "metadata.json"
    command_log_path = run_path / "command.log"
    if not metadata_path.exists() or not command_log_path.exists():
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Run '{resolved_run_id}' is missing metadata.json or command.log.",
            run_id=resolved_run_id,
        )

    try:
        metadata = _load_json(metadata_path)
    except Exception as exc:
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Unable to read metadata.json for run '{resolved_run_id}': {exc}",
            run_id=resolved_run_id,
        )

    try:
        command_log_text = command_log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Unable to read command.log for run '{resolved_run_id}': {exc}",
            run_id=resolved_run_id,
        )

    redact_paths = parse_bool_value(config.get("support_bundle_redact_paths", True), True)
    redact_env = parse_bool_value(config.get("support_bundle_redact_env", True), True)
    home_dir = str(Path.home())

    preflight_checks = _extract_preflight_checks(metadata)
    preflight_events = checks_to_events(preflight_checks)
    pass_count, warn_count, fail_count = _check_counts(preflight_checks)
    preflight_report = {
        "diagnostic_version": "v2",
        "summary": {
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
        },
        "checks": [
            {"level": level, "name": name, "detail": detail}
            for level, name, detail in preflight_checks
        ],
        "events": [event.to_dict() for event in preflight_events],
    }
    preflight_text = (
        summarize_events(preflight_events, title="Preflight Diagnostics")
        if preflight_events
        else summarize_checks(preflight_checks, title="Preflight")
    )

    normalized_config = normalize_config_without_prompts(config)
    config_snapshot = _sanitize_value(
        normalized_config,
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )
    metadata_snapshot = _sanitize_value(
        metadata,
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )
    compatibility_snapshot = _sanitize_value(
        build_compatibility_snapshot(config, metadata=metadata),
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )
    environment_probe = _sanitize_value(
        build_environment_probe(redact_env=redact_env),
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )
    sanitized_command_log = _sanitize_text(
        command_log_text,
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )

    output = output_path.expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Unable to create output folder for support bundle: {exc}",
            run_id=resolved_run_id,
        )

    manifest = {
        "support_bundle_version": "v1",
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "run_id": resolved_run_id,
        "source_run_path": str(run_path),
        "redaction": {
            "redact_paths": redact_paths,
            "redact_env": redact_env,
        },
    }
    manifest = _sanitize_value(
        manifest,
        home_dir=home_dir,
        redact_paths=redact_paths,
        redact_env=redact_env,
    )

    try:
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            archive.writestr("metadata.json", json.dumps(metadata_snapshot, indent=2) + "\n")
            archive.writestr("command.log", sanitized_command_log if sanitized_command_log.endswith("\n") else sanitized_command_log + "\n")
            archive.writestr("preflight_report.json", json.dumps(preflight_report, indent=2) + "\n")
            archive.writestr("preflight_report.txt", preflight_text if preflight_text.endswith("\n") else preflight_text + "\n")
            archive.writestr("config_snapshot.json", json.dumps(config_snapshot, indent=2) + "\n")
            archive.writestr("compatibility_snapshot.json", json.dumps(compatibility_snapshot, indent=2) + "\n")
            archive.writestr("environment_probe.json", json.dumps(environment_probe, indent=2) + "\n")
    except OSError as exc:
        return SupportBundleResult(
            ok=False,
            bundle_path=None,
            message=f"Unable to write support bundle: {exc}",
            run_id=resolved_run_id,
        )

    return SupportBundleResult(
        ok=True,
        bundle_path=output,
        message=f"Support bundle created for run '{resolved_run_id}'.",
        run_id=resolved_run_id,
    )
