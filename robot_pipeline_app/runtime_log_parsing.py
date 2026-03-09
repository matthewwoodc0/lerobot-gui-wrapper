from __future__ import annotations

import re

EPISODE_PATTERNS = [
    re.compile(r"[Ee]pisode\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"[Ee]pisode\s+(\d+)\s+of\s+(\d+)"),
    re.compile(r"[Ee](?:pisode|p)\s*[:#]?\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"[Ee]pisode(?:_idx)?\s*[:=]\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"\b[Ee]p\s+(\d+)\s+of\s+(\d+)\b"),
]
EPISODE_PARTIAL_PATTERN = re.compile(r"[Ee](?:pisode|p)\s*[:#]?\s*(\d+)")
RESET_PHASE_PATTERNS = [
    re.compile(r"(left|right)\s+arrow", re.IGNORECASE),
    re.compile(r"redo.*next|next.*redo", re.IGNORECASE),
    re.compile(r"reset.*episode|next.*episode", re.IGNORECASE),
    re.compile(r"reset\s+the\s+environment", re.IGNORECASE),
]
START_PHASE_PATTERNS = [
    re.compile(r"\brecording\s+episode\b", re.IGNORECASE),
    re.compile(r"\bepisode\b.*\bstarted\b", re.IGNORECASE),
]


def parse_episode_progress_line(line: str) -> tuple[int, int | None] | None:
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    partial = EPISODE_PARTIAL_PATTERN.search(line)
    if partial:
        return int(partial.group(1)), None
    return None


def is_episode_reset_phase_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in RESET_PHASE_PATTERNS)


def is_episode_start_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in START_PHASE_PATTERNS)


def parse_outcome_tags(raw: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").split(","):
        tag = chunk.strip()
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags
