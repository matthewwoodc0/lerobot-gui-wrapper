from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import coerce_diagnostic_events
from .diagnostics import attribution_label, default_attribution_for_code, first_failure_event, normalize_attribution
from .types import DiagnosticEvent

_STATUS_LABELS = {
    "success": "Success",
    "failed": "Failed",
    "canceled": "Canceled",
}
_ATTRIBUTION_EXPLANATIONS = {
    "wrapper": "Likely source: GUI wrapper / local process management.",
    "lerobot": "Likely source: LeRobot CLI or upstream compatibility, not the GUI wrapper.",
    "model": "Likely source: model runtime or checkpoint behavior, not the GUI wrapper.",
    "environment": "Likely source: Python environment or dependency state, not the GUI wrapper.",
    "hardware": "Likely source: hardware, device access, or OS-level runtime state, not the GUI wrapper.",
    "unknown": "Likely source: unknown. Start from the first traceback or error line in the raw transcript.",
}


def all_diagnostic_events(metadata: dict[str, Any]) -> list[DiagnosticEvent]:
    return [
        *coerce_diagnostic_events(metadata.get("preflight_diagnostics")),
        *coerce_diagnostic_events(metadata.get("runtime_diagnostics")),
    ]


def first_failure_from_metadata(metadata: dict[str, Any]) -> DiagnosticEvent | None:
    events = all_diagnostic_events(metadata)
    first = first_failure_event(events)
    if first is not None:
        return first

    code = str(metadata.get("first_failure_code", "")).strip()
    name = str(metadata.get("first_failure_name", "")).strip()
    detail = str(metadata.get("first_failure_detail", "")).strip()
    if not code and not name and not detail:
        return None

    return DiagnosticEvent(
        level="FAIL",
        code=code or "UNSPECIFIED",
        name=name or "First failure",
        detail=detail or "Structured failure details were not stored for this run.",
        fix=str(metadata.get("first_failure_fix", "")).strip(),
        docs_ref=str(metadata.get("first_failure_docs_ref", "")).strip(),
        attribution=normalize_attribution(
            str(metadata.get("first_failure_attribution", "")).strip() or default_attribution_for_code(code)
        ),
    )


def has_failure_details(metadata: dict[str, Any]) -> bool:
    return first_failure_from_metadata(metadata) is not None


def raw_transcript_text(run_path: Path | None) -> str:
    if run_path is None:
        return "No raw transcript is available for this run."
    log_path = Path(run_path) / "command.log"
    if not log_path.exists():
        return f"Raw transcript is missing: {log_path}"
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Unable to read raw transcript: {exc}"
    return text if text else "(empty command log)"


def build_run_summary_text(metadata: dict[str, Any]) -> str:
    status = _STATUS_LABELS.get(str(metadata.get("status", "")).strip().lower(), "Ready")
    mode = str(metadata.get("mode", "run")).strip() or "run"
    lines = [
        f"Status: {status}",
        f"Mode: {mode}",
    ]

    first = first_failure_from_metadata(metadata)
    if first is None:
        if status == "Success":
            lines.append("Structured diagnostics: none recorded.")
        else:
            lines.append("Structured diagnostics: none recorded for this run.")
        return "\n".join(lines)

    attribution = normalize_attribution(first.attribution)
    lines.extend(
        [
            f"Likely source: {attribution_label(attribution)}",
            _ATTRIBUTION_EXPLANATIONS[attribution],
            f"First failure: {first.code or 'UNSPECIFIED'} {first.name}",
            f"Detail: {first.detail}",
        ]
    )
    if first.fix:
        lines.append(f"Fix: {first.fix}")
    if first.docs_ref:
        lines.append(f"Docs: {first.docs_ref}")

    compat_lines = _compat_summary_lines(metadata, first)
    if compat_lines:
        lines.append("")
        lines.append("Compatibility Snapshot")
        lines.extend(compat_lines)

    events = all_diagnostic_events(metadata)
    remaining = [
        event
        for event in events
        if not (
            event.code == first.code
            and event.name == first.name
            and event.detail == first.detail
        )
    ]
    if remaining:
        lines.append("")
        lines.append("Other Diagnostics")
        for event in remaining[:6]:
            lines.append(f"- {event.level} {event.code}: {event.detail}")
        if len(remaining) > 6:
            lines.append(f"- ... {len(remaining) - 6} more")
    return "\n".join(lines)


def build_failure_explanation_text(metadata: dict[str, Any], *, run_path: Path | None = None) -> str:
    lines = [build_run_summary_text(metadata)]
    transcript_path = Path(run_path) / "command.log" if run_path is not None else None
    if transcript_path is not None:
        lines.extend(["", f"Raw transcript: {transcript_path}"])
    command = str(metadata.get("command", "")).strip()
    if command:
        lines.extend(["", "Command", command])
    return "\n".join(lines)


def _compat_summary_lines(metadata: dict[str, Any], first: DiagnosticEvent) -> list[str]:
    if normalize_attribution(first.attribution) != "lerobot":
        return []
    snapshot = metadata.get("compat_snapshot")
    if not isinstance(snapshot, dict):
        return []

    mode = str(metadata.get("mode", "run")).strip().lower()
    lines: list[str] = []
    lerobot_version = str(snapshot.get("lerobot_version", "")).strip()
    if lerobot_version:
        lines.append(f"- LeRobot version: {lerobot_version}")

    entrypoint_key = "teleop_entrypoint" if mode == "teleop" else "record_entrypoint"
    entrypoint = str(snapshot.get(entrypoint_key, "")).strip()
    if entrypoint:
        lines.append(f"- Active entrypoint: {entrypoint}")

    policy_flag = str(snapshot.get("policy_path_flag", "")).strip()
    if policy_flag:
        lines.append(f"- Policy flag: --{policy_flag}")

    rename_flag = str(snapshot.get("camera_rename_flag", "")).strip()
    if rename_flag:
        lines.append(f"- Rename flag: --{rename_flag}")

    fallback_notes = snapshot.get("fallback_notes")
    if isinstance(fallback_notes, list):
        for note in [str(item).strip() for item in fallback_notes if str(item).strip()][:3]:
            lines.append(f"- Fallback: {note}")

    supported_flags = snapshot.get("supported_record_flags")
    if isinstance(supported_flags, list) and supported_flags:
        items = [f"--{str(flag).strip()}" for flag in supported_flags if str(flag).strip()]
        preview = ", ".join(items[:12])
        if len(items) > 12:
            preview += f", ... ({len(items)} total)"
        lines.append(f"- Supported record flags: {preview}")
    return lines
