from __future__ import annotations

import re
from typing import Any, Sequence

from .types import CheckResult, DiagnosticEvent

DOCS_ERROR_CATALOG_REF = "Resources/error-catalog.md"
VALID_ATTRIBUTIONS = {"wrapper", "lerobot", "model", "environment", "hardware", "unknown"}
_ATTRIBUTION_LABELS = {
    "wrapper": "GUI wrapper",
    "lerobot": "LeRobot",
    "model": "Model runtime",
    "environment": "Environment",
    "hardware": "Hardware / OS",
    "unknown": "Unknown",
}

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_FIX_PATTERN = re.compile(r"\bFix:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_SUGGESTED_REPO_PATTERN = re.compile(r"Suggested quick fix:\s*([^\s,;]+)", flags=re.IGNORECASE)
_NEXT_AVAILABLE_NAME_PATTERN = re.compile(r"Next available name:\s*'([^']+)'", flags=re.IGNORECASE)
_TRAINING_FPS_PATTERN = re.compile(r"model trained at\s+(\d+)\s*hz", flags=re.IGNORECASE)


def normalize_level(level: str) -> str:
    normalized = str(level or "").strip().upper()
    if normalized in {"PASS", "WARN", "FAIL", "INFO"}:
        return normalized
    return "WARN"


def normalize_attribution(attribution: str) -> str:
    normalized = str(attribution or "").strip().lower()
    if normalized in VALID_ATTRIBUTIONS:
        return normalized
    return "unknown"


def attribution_label(attribution: str) -> str:
    normalized = normalize_attribution(attribution)
    return _ATTRIBUTION_LABELS.get(normalized, _ATTRIBUTION_LABELS["unknown"])


def default_attribution_for_code(code: str) -> str:
    prefix = str(code or "").split("-", 1)[0].upper()
    if prefix == "ENV":
        return "environment"
    if prefix in {"SER", "CAM", "CAL"}:
        return "hardware"
    if prefix == "MODEL":
        return "model"
    if prefix in {"CLI", "COMPAT"}:
        return "lerobot"
    return "unknown"


def _slugify(value: str) -> str:
    slug = _NON_ALNUM_PATTERN.sub("_", str(value or "").strip().lower()).strip("_")
    return slug or "unspecified"


def _code_prefix_for_name(name: str, detail: str = "") -> str:
    text = f"{name} {detail}".lower()

    if "calibration" in text or "motor" in text:
        return "CAL"
    if "camera" in text:
        return "CAM"
    if "serial" in text or "dialout" in text or "tty" in text or "port" in text:
        return "SER"
    if "dataset" in text or "huggingface-cli" in text:
        return "DATA"
    if "model" in text or "policy" in text or "action dim" in text:
        return "MODEL"
    if "flag" in text or "argument" in text or "entrypoint" in text:
        return "CLI"
    if "training vs deploy" in text or "compat" in text:
        return "COMPAT"
    if "python" in text or "environment" in text or "folder" in text or " dir" in text:
        return "ENV"
    return "ENV"


def code_for_check(name: str, detail: str = "") -> str:
    prefix = _code_prefix_for_name(name, detail)
    return f"{prefix}-{_slugify(name).upper()}"


def _docs_ref_for_code(code: str) -> str:
    prefix = str(code or "").split("-", 1)[0].upper()
    anchor_by_prefix = {
        "ENV": "env",
        "SER": "ser",
        "CAM": "cam",
        "CAL": "cal",
        "CLI": "cli",
        "MODEL": "model",
        "DATA": "data",
        "COMPAT": "compat",
    }
    anchor = anchor_by_prefix.get(prefix, "taxonomy")
    return f"{DOCS_ERROR_CATALOG_REF}#{anchor}"


def _extract_fix(detail: str) -> str:
    match = _FIX_PATTERN.search(str(detail or ""))
    if not match:
        return ""
    return str(match.group(1)).strip()


def _fallback_fix(name: str, detail: str, level: str) -> str:
    if level == "PASS":
        return ""

    lowered = f"{name} {detail}".lower()
    if "calibration" in lowered:
        return "Run calibration and select valid calibration files in Config, then rerun preflight."
    if "camera" in lowered:
        return "Validate camera mapping/schema and rerun camera scan before retrying."
    if "serial" in lowered or "port" in lowered or "dialout" in lowered:
        return "Verify serial ports exist and permissions are correct, then rerun doctor."
    if "dataset" in lowered:
        return "Check dataset repo id/path naming and write permissions, then rerun preflight."
    if "model" in lowered or "policy" in lowered:
        return "Use a runnable model payload folder (config + weights) that matches runtime features."
    if "flag" in lowered or "argument" in lowered:
        return "Run the relevant LeRobot command with --help and update flags to match your installed version."
    return "Review the detail, apply the suggested fix, then rerun preflight/doctor."


def _quick_action_id(name: str, detail: str, level: str) -> str | None:
    if level == "PASS":
        return None

    lowered_name = str(name or "").lower()
    lowered_detail = str(detail or "").lower()
    if lowered_name == "eval dataset naming":
        return "fix_eval_prefix"
    if lowered_name == "eval dataset already exists":
        return "fix_eval_name"
    if lowered_name == "model payload" and "nested model payload" in lowered_detail:
        return "fix_model_payload"
    if lowered_name == "model payload candidates":
        return "fix_model_payload"
    if "camera rename map suggestion" in lowered_name:
        return "apply_rename_map"
    if lowered_name == "model camera keys" and "require --" in lowered_detail:
        return "apply_rename_map"
    if lowered_name == "training vs deploy fps":
        return "fix_camera_fps"
    if "follower" in lowered_name and "calibration" in lowered_name:
        return "browse_follower_calib"
    if "leader" in lowered_name and "calibration" in lowered_name:
        return "browse_leader_calib"
    if "calibration" in lowered_name and level == "FAIL":
        return "show_calib_cmd"
    return None


def _event_context(name: str, detail: str, quick_action_id: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {}
    text = str(detail or "")

    if quick_action_id in {"fix_eval_prefix", "fix_eval_name"}:
        match = _SUGGESTED_REPO_PATTERN.search(text) or _NEXT_AVAILABLE_NAME_PATTERN.search(text)
        if match:
            context["suggested_eval_repo_id"] = match.group(1).strip()

    if quick_action_id == "fix_camera_fps":
        match = _TRAINING_FPS_PATTERN.search(text)
        if match:
            context["suggested_fps"] = int(match.group(1))

    if quick_action_id == "apply_rename_map":
        trimmed = text.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            context["rename_map_suggestion"] = trimmed

    if str(name).lower() == "model payload candidates":
        candidate = next((part.strip() for part in text.split(",") if part.strip()), "")
        if candidate:
            context["model_candidate"] = candidate

    return context


def check_result_to_event(check: CheckResult) -> DiagnosticEvent:
    level, name, detail = check
    normalized_level = normalize_level(level)
    code = code_for_check(name, detail)
    fix = _extract_fix(detail) or _fallback_fix(name, detail, normalized_level)
    quick_action_id = _quick_action_id(name, detail, normalized_level)
    context = _event_context(name, detail, quick_action_id)
    return DiagnosticEvent(
        level=normalized_level,
        code=code,
        name=str(name),
        detail=str(detail),
        fix=fix,
        docs_ref=_docs_ref_for_code(code),
        attribution=default_attribution_for_code(code),
        quick_action_id=quick_action_id,
        context=context or None,
    )


def checks_to_events(checks: Sequence[CheckResult]) -> list[DiagnosticEvent]:
    return [check_result_to_event(check) for check in checks]


def events_to_checks(events: Sequence[DiagnosticEvent]) -> list[CheckResult]:
    return [event.as_check_result() for event in events]


def first_failure_event(events: Sequence[DiagnosticEvent]) -> DiagnosticEvent | None:
    for event in events:
        if normalize_level(event.level) == "FAIL":
            return event
    return None


def summarize_events(events: Sequence[DiagnosticEvent], title: str = "Diagnostics") -> str:
    pass_count = sum(1 for event in events if normalize_level(event.level) == "PASS")
    warn_count = sum(1 for event in events if normalize_level(event.level) == "WARN")
    fail_count = sum(1 for event in events if normalize_level(event.level) == "FAIL")

    lines = [title]
    for event in events:
        code = event.code or "UNSPECIFIED"
        lines.append(f"[{normalize_level(event.level):4}] {code} {event.name}: {event.detail}")
    lines.append("")
    lines.append(f"Summary: PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return "\n".join(lines)


def diagnostic_event_from_runtime(
    *,
    level: str,
    code: str,
    name: str,
    detail: str,
    fix: str = "",
    docs_ref: str | None = None,
    attribution: str | None = None,
    quick_action_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> DiagnosticEvent:
    resolved_docs_ref = docs_ref if docs_ref is not None else _docs_ref_for_code(code)
    resolved_attribution = normalize_attribution(attribution or default_attribution_for_code(code))
    return DiagnosticEvent(
        level=normalize_level(level),
        code=str(code).strip() or code_for_check(name, detail),
        name=str(name),
        detail=str(detail),
        fix=str(fix).strip(),
        docs_ref=str(resolved_docs_ref).strip(),
        attribution=resolved_attribution,
        quick_action_id=quick_action_id,
        context=context or None,
    )
