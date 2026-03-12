from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowQueueItem:
    queue_id: int
    recipe_type: str
    title: str
    step_labels: list[str]
    payload: dict[str, Any]
    status: str = "queued"
    current_step_index: int = 0
    current_command: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error_text: str = ""
    current_artifact_path: Path | None = None
    current_artifact_metadata: dict[str, Any] | None = None
    resume_required: bool = False

    @property
    def current_step_label(self) -> str:
        if 0 <= self.current_step_index < len(self.step_labels):
            return self.step_labels[self.current_step_index]
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "recipe_type": self.recipe_type,
            "title": self.title,
            "status": self.status,
            "current_step_index": self.current_step_index,
            "current_step_label": self.current_step_label,
            "total_steps": len(self.step_labels),
            "step_labels": list(self.step_labels),
            "current_command": list(self.current_command),
            "artifacts": list(self.artifacts),
            "error_text": self.error_text,
            "log_text": "\n".join(self.log_lines[-400:]),
            "resume_required": self.resume_required,
        }

    def to_persisted_payload(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "recipe_type": self.recipe_type,
            "title": self.title,
            "step_labels": list(self.step_labels),
            "payload": dict(self.payload),
            "status": self.status,
            "current_step_index": self.current_step_index,
            "current_step_label": self.current_step_label,
            "current_command": list(self.current_command),
            "artifacts": list(self.artifacts),
            "error_text": self.error_text,
            "log_text": "\n".join(self.log_lines[-400:]),
        }

    @classmethod
    def from_persisted_payload(cls, payload: dict[str, Any]) -> "WorkflowQueueItem | None":
        try:
            queue_id = int(payload.get("queue_id"))
        except (TypeError, ValueError):
            return None
        step_labels = payload.get("step_labels")
        payload_map = payload.get("payload")
        artifacts = payload.get("artifacts")
        if not isinstance(step_labels, list) or not isinstance(payload_map, dict):
            return None
        item = cls(
            queue_id=queue_id,
            recipe_type=str(payload.get("recipe_type", "")).strip(),
            title=str(payload.get("title", "")).strip() or f"Workflow #{queue_id}",
            step_labels=[str(label) for label in step_labels],
            payload=dict(payload_map),
            status=str(payload.get("status", "queued")).strip() or "queued",
            current_step_index=max(0, int(payload.get("current_step_index", 0) or 0)),
            current_command=[str(part) for part in list(payload.get("current_command", []))],
            artifacts=[dict(entry) for entry in artifacts] if isinstance(artifacts, list) else [],
            error_text=str(payload.get("error_text", "")).strip(),
            log_lines=[line for line in str(payload.get("log_text", "")).splitlines() if line],
            resume_required=False,
        )
        return item
