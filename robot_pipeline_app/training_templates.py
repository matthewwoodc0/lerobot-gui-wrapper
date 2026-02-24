from __future__ import annotations

import string
from typing import Any


def default_training_templates() -> list[dict[str, str]]:
    return [
        {
            "id": "srun_tmux",
            "name": "srun + tmux",
            "description": "Start a tmux session that runs training under srun.",
            "template": (
                "{env_activate_cmd} && "
                "cd {remote_project_root} && "
                "tmux kill-session -t {tmux_session} 2>/dev/null || true && "
                "tmux new-session -d -s {tmux_session} "
                "'{srun_prefix} \"{train_command}\"' && "
                "tmux ls"
            ),
        },
        {
            "id": "tmux_custom",
            "name": "tmux custom",
            "description": "Run a custom command in a tmux session.",
            "template": (
                "{env_activate_cmd} && "
                "cd {remote_project_root} && "
                "tmux kill-session -t {tmux_session} 2>/dev/null || true && "
                "tmux new-session -d -s {tmux_session} '{train_command}' && "
                "tmux ls"
            ),
        },
        {
            "id": "custom",
            "name": "custom command",
            "description": "Run a direct custom command.",
            "template": "{env_activate_cmd} && cd {remote_project_root} && {train_command}",
        },
    ]


def template_variables(template_text: str) -> list[str]:
    variables: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(str(template_text or "")):
        if field_name and field_name not in variables:
            variables.append(field_name)
    return variables


def render_template(template_text: str, variables: dict[str, str]) -> tuple[str | None, str | None]:
    text = str(template_text or "").strip()
    if not text:
        return None, "Template is empty."

    required = template_variables(text)
    missing = [name for name in required if not str(variables.get(name, "")).strip()]
    if missing:
        return None, f"Missing template values: {', '.join(missing)}"

    payload: dict[str, Any] = {name: str(value) for name, value in variables.items()}
    try:
        rendered = text.format(**payload).strip()
    except KeyError as exc:
        return None, f"Missing template value: {exc}"
    except Exception as exc:
        return None, f"Invalid template: {exc}"

    if not rendered:
        return None, "Rendered command is empty."
    return rendered, None
