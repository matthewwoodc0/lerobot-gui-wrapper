from __future__ import annotations

import re
import uuid
from typing import Any

from .types import TrainingProfile

_PROFILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _default_profile() -> TrainingProfile:
    return TrainingProfile(
        id="olympus",
        name="Olympus",
        host="olympus.ece.tamu.edu",
        port=22,
        username="",
        auth_mode="password",
        identity_file="",
        remote_models_root="~/lerobot/trained_models",
        remote_project_root="~/lerobot",
        env_activate_cmd="source ~/lerobot/lerobot_env/bin/activate",
        default_tmux_session="lerobot_train",
        default_srun_prefix="srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash -lc",
    )


def _safe_profile_id(raw: Any) -> str:
    text = str(raw or "").strip()
    if _PROFILE_ID_PATTERN.match(text):
        return text
    return f"profile_{uuid.uuid4().hex[:12]}"


def _to_int_port(raw: Any) -> int:
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return 22
    if port <= 0 or port > 65535:
        return 22
    return port


def _normalize_auth_mode(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"password", "ssh_key"}:
        return text
    return "password"


def _normalize_profile(raw: dict[str, Any]) -> TrainingProfile:
    base = _default_profile()
    return TrainingProfile(
        id=_safe_profile_id(raw.get("id", base.id)),
        name=str(raw.get("name", base.name)).strip() or base.name,
        host=str(raw.get("host", base.host)).strip(),
        port=_to_int_port(raw.get("port", base.port)),
        username=str(raw.get("username", base.username)).strip(),
        auth_mode=_normalize_auth_mode(raw.get("auth_mode", base.auth_mode)),
        identity_file=str(raw.get("identity_file", base.identity_file)).strip(),
        remote_models_root=str(raw.get("remote_models_root", base.remote_models_root)).strip() or base.remote_models_root,
        remote_project_root=str(raw.get("remote_project_root", base.remote_project_root)).strip() or base.remote_project_root,
        env_activate_cmd=str(raw.get("env_activate_cmd", base.env_activate_cmd)).strip() or base.env_activate_cmd,
        default_tmux_session=str(raw.get("default_tmux_session", base.default_tmux_session)).strip()
        or base.default_tmux_session,
        default_srun_prefix=str(raw.get("default_srun_prefix", base.default_srun_prefix)).strip()
        or base.default_srun_prefix,
    )


def profile_to_dict(profile: TrainingProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "host": profile.host,
        "port": profile.port,
        "username": profile.username,
        "auth_mode": profile.auth_mode,
        "identity_file": profile.identity_file,
        "remote_models_root": profile.remote_models_root,
        "remote_project_root": profile.remote_project_root,
        "env_activate_cmd": profile.env_activate_cmd,
        "default_tmux_session": profile.default_tmux_session,
        "default_srun_prefix": profile.default_srun_prefix,
    }


def load_training_profiles(config: dict[str, Any]) -> tuple[list[TrainingProfile], str | None]:
    raw_profiles = config.get("training_profiles")
    profiles: list[TrainingProfile] = []
    if isinstance(raw_profiles, list):
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            profile = _normalize_profile(item)
            if any(existing.id == profile.id for existing in profiles):
                continue
            profiles.append(profile)

    if not profiles:
        profiles = [_default_profile()]

    active_id = str(config.get("training_active_profile_id", "")).strip() or None
    if active_id is None or not any(profile.id == active_id for profile in profiles):
        active_id = profiles[0].id if profiles else None

    return profiles, active_id


def save_training_profiles(
    config: dict[str, Any],
    profiles: list[TrainingProfile],
    active_profile_id: str | None,
) -> None:
    if not profiles:
        profiles = [_default_profile()]
    active = str(active_profile_id or "").strip()
    if not any(profile.id == active for profile in profiles):
        active = profiles[0].id
    config["training_profiles"] = [profile_to_dict(profile) for profile in profiles]
    config["training_active_profile_id"] = active
