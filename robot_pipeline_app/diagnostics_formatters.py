from __future__ import annotations

import html as _html

from .diagnostics import checks_to_events
from .types import CheckResult, DiagnosticEvent, PreflightReport

_LEVEL_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
_LEVEL_COLORS = {"PASS": "#4caf50", "WARN": "#f0a500", "FAIL": "#f44336"}


def _check_counts(checks: list[CheckResult]) -> tuple[int, int, int]:
    pass_count = sum(1 for level, _, _ in checks if level == "PASS")
    warn_count = sum(1 for level, _, _ in checks if level == "WARN")
    fail_count = sum(1 for level, _, _ in checks if level == "FAIL")
    return pass_count, warn_count, fail_count


def _grouped_checks_with_events(
    checks: list[CheckResult],
) -> list[tuple[str, str, str, object]]:
    events = checks_to_events(checks)
    combined = [
        (str(level).strip().upper(), name, detail, event)
        for (level, name, detail), event in zip(checks, events)
    ]
    return sorted(combined, key=lambda x: _LEVEL_ORDER.get(x[0], 99))


def build_preflight_report(checks: list[CheckResult]) -> PreflightReport:
    pass_count, warn_count, fail_count = _check_counts(checks)
    return PreflightReport(
        checks=checks,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        diagnostics=checks_to_events(checks),
    )


def summarize_checks(checks: list[CheckResult], title: str = "Checks") -> str:
    pass_count, warn_count, fail_count = _check_counts(checks)
    lines = [title]
    for level, name, detail, event in _grouped_checks_with_events(checks):
        lines.append(f"[{level:4}] {event.code} {name}: {detail}")
    lines.append("")
    lines.append(f"Summary: PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return "\n".join(lines)


def summarize_checks_html(checks: list[CheckResult], title: str = "Checks") -> str:
    pass_count, warn_count, fail_count = _check_counts(checks)
    lines = [f"<b>{_html.escape(title)}</b>"]
    for level, name, detail, event in _grouped_checks_with_events(checks):
        color = _LEVEL_COLORS.get(level, "#ffffff")
        label = f'<span style="color:{color}">[{_html.escape(level):4}]</span>'
        lines.append(
            f"{label} {_html.escape(str(event.code))} {_html.escape(name)}: {_html.escape(detail)}"
        )
    lines.append("")
    pass_html = f'<span style="color:{_LEVEL_COLORS["PASS"]}">PASS={pass_count}</span>'
    warn_html = f'<span style="color:{_LEVEL_COLORS["WARN"]}">WARN={warn_count}</span>'
    fail_html = f'<span style="color:{_LEVEL_COLORS["FAIL"]}">FAIL={fail_count}</span>'
    lines.append(f"Summary: {pass_html} {warn_html} {fail_html}")
    return "<br>".join(lines)


def has_failures(checks: list[CheckResult]) -> bool:
    return any(level == "FAIL" for level, _, _ in checks)


def diagnostics_from_checks(checks: list[CheckResult]) -> list[DiagnosticEvent]:
    return checks_to_events(checks)
