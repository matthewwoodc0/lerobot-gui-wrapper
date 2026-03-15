from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RemoteExistsFn = Callable[[str], bool | None]
LocalExistsFn = Callable[[str], bool]

_NON_ALNUM = re.compile(r"[^A-Za-z0-9_]+")
_NUMBERING_ROOT = re.compile(r"(?:_)?\d+$")


@dataclass(frozen=True)
class NameResolution:
    raw_value: str
    owner: str
    requested_name: str
    resolved_name: str
    display_value: str
    repo_id: str | None
    iterated: bool
    prefix_applied: bool
    owner_displayed: bool
    checked_remote: bool
    occupied: bool
    occupied_sources: tuple[str, ...]


def increment_name(name: str) -> str:
    match = re.search(r"^(.*?)(\d+)$", str(name or ""))
    if not match:
        return f"{str(name or '').strip()}_1"
    prefix, number = match.groups()
    return f"{prefix}{int(number) + 1}"


def repo_name_from_value(value: Any) -> str:
    text = str(value or "").strip().strip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].strip()


def sanitize_name_token(value: Any, *, fallback: str = "run") -> str:
    text = _NON_ALNUM.sub("_", str(value or "").strip()).strip("_")
    return text or fallback


def numbering_root(value: str) -> str:
    clean_value = str(value or "").strip()
    if not clean_value:
        return ""
    return _NUMBERING_ROOT.sub("", clean_value).rstrip("_")


def build_train_job_name_base(dataset_input: str, policy_type: str) -> str:
    dataset_name = sanitize_name_token(repo_name_from_value(dataset_input), fallback="")
    policy_name = sanitize_name_token(policy_type, fallback="")
    parts = [part for part in (dataset_name, policy_name) if part]
    return "_".join(parts) or "train_run"


def train_job_name_seed(config: dict[str, Any], dataset_input: str, policy_type: str) -> str:
    family = build_train_job_name_base(dataset_input, policy_type)
    last_job_name = str(config.get("last_train_job_name", "")).strip()
    if last_job_name and numbering_root(last_job_name) == family:
        return last_job_name
    return f"{family}_1"


def record_dataset_seed(config: dict[str, Any]) -> str:
    return (
        str(config.get("last_dataset_repo_id", "")).strip()
        or str(config.get("last_dataset_name", "")).strip()
        or "dataset_1"
    )


def deploy_eval_seed(
    config: dict[str, Any],
    *,
    model_name: str = "",
    prefer_model_seed: bool = False,
) -> str:
    if not prefer_model_seed:
        previous = str(config.get("last_eval_dataset_name", "")).strip()
        if previous:
            return previous
    clean_model = sanitize_name_token(model_name, fallback="run")
    if clean_model.startswith("eval_"):
        return f"{clean_model}_1"
    return f"eval_{clean_model}_1"


def resolve_available_name(
    raw_value: str,
    *,
    default_owner: str = "",
    required_prefix: str = "",
    prefer_owner_display: bool | None = None,
    local_exists_fn: LocalExistsFn | None = None,
    remote_exists_fn: RemoteExistsFn | None = None,
    force_occupied: str | None = None,
    max_attempts: int = 200,
) -> NameResolution:
    clean_value = str(raw_value or "").strip().strip("/")
    raw_owner = ""
    requested_name = clean_value
    if "/" in clean_value:
        raw_owner, requested_name = clean_value.split("/", 1)
    owner = raw_owner.strip().strip("/") or str(default_owner or "").strip().strip("/")
    requested_name = requested_name.strip().strip("/")
    prefix_applied = bool(required_prefix) and requested_name != "" and not requested_name.startswith(required_prefix)
    if prefix_applied:
        requested_name = f"{required_prefix}{requested_name}"
    owner_displayed = bool(owner) if prefer_owner_display is None else bool(prefer_owner_display)
    repo_id = f"{owner}/{requested_name}" if owner and requested_name else None
    if not requested_name:
        return NameResolution(
            raw_value=clean_value,
            owner=owner,
            requested_name="",
            resolved_name="",
            display_value=repo_id or "",
            repo_id=repo_id,
            iterated=False,
            prefix_applied=False,
            owner_displayed=owner_displayed,
            checked_remote=False,
            occupied=False,
            occupied_sources=(),
        )

    checked_remote = False
    forced_name = str(force_occupied or "").strip()

    def candidate_taken(candidate_name: str) -> tuple[bool, tuple[str, ...]]:
        nonlocal checked_remote
        sources: list[str] = []
        if forced_name and candidate_name == forced_name:
            sources.append("forced")
        if local_exists_fn is not None and local_exists_fn(candidate_name):
            sources.append("local")
        if remote_exists_fn is not None and owner:
            remote_result = remote_exists_fn(f"{owner}/{candidate_name}")
            if remote_result is not None:
                checked_remote = True
            if remote_result is True:
                sources.append("remote")
        return bool(sources), tuple(sources)

    candidate = requested_name
    occupied, occupied_sources = candidate_taken(candidate)
    attempts = 0
    while occupied and attempts < max_attempts:
        candidate = increment_name(candidate)
        occupied, _ = candidate_taken(candidate)
        attempts += 1

    resolved_repo_id = f"{owner}/{candidate}" if owner else None
    display_value = resolved_repo_id if owner and owner_displayed else candidate
    return NameResolution(
        raw_value=clean_value,
        owner=owner,
        requested_name=requested_name,
        resolved_name=candidate,
        display_value=display_value,
        repo_id=resolved_repo_id,
        iterated=(candidate != requested_name),
        prefix_applied=prefix_applied,
        owner_displayed=owner_displayed,
        checked_remote=checked_remote,
        occupied=bool(occupied_sources),
        occupied_sources=occupied_sources,
    )


def resolve_record_dataset_name(
    raw_value: str,
    *,
    config: dict[str, Any],
    dataset_root_raw: str = "",
    force_occupied: str | None = None,
) -> NameResolution:
    from .config_store import get_lerobot_dir
    from .repo_utils import dataset_exists_on_hf

    roots: list[Path] = []
    dataset_root_text = str(dataset_root_raw or "").strip() or str(config.get("record_data_dir", "")).strip()
    if dataset_root_text:
        roots.append(Path(dataset_root_text).expanduser())
    try:
        lerobot_data_dir = get_lerobot_dir(config) / "data"
    except Exception:
        lerobot_data_dir = None
    if lerobot_data_dir is not None and lerobot_data_dir not in roots:
        roots.append(lerobot_data_dir)

    def _exists_locally(name: str) -> bool:
        return any((root / name).exists() for root in roots)

    return resolve_available_name(
        raw_value,
        default_owner=str(config.get("hf_username", "")).strip(),
        local_exists_fn=_exists_locally,
        remote_exists_fn=dataset_exists_on_hf,
        force_occupied=force_occupied,
    )


def resolve_deploy_eval_name(
    raw_value: str,
    *,
    config: dict[str, Any],
    force_occupied: str | None = None,
) -> NameResolution:
    from .config_store import get_deploy_data_dir, get_lerobot_dir
    from .repo_utils import dataset_exists_on_hf

    roots: list[Path] = []
    try:
        roots.append(get_deploy_data_dir(config))
    except Exception:
        pass
    try:
        lerobot_data_dir = get_lerobot_dir(config) / "data"
    except Exception:
        lerobot_data_dir = None
    if lerobot_data_dir is not None and lerobot_data_dir not in roots:
        roots.append(lerobot_data_dir)

    def _exists_locally(name: str) -> bool:
        return any((root / name).exists() for root in roots)

    return resolve_available_name(
        raw_value,
        default_owner=str(config.get("hf_username", "")).strip(),
        required_prefix="eval_",
        local_exists_fn=_exists_locally,
        remote_exists_fn=dataset_exists_on_hf,
        force_occupied=force_occupied,
    )


def resolve_train_job_name(
    raw_value: str,
    *,
    config: dict[str, Any],
    dataset_input: str,
    policy_type: str,
    output_dir_raw: str = "",
    force_occupied: str | None = None,
) -> NameResolution:
    from .repo_utils import model_exists_on_hf

    output_dir_text = str(output_dir_raw or "").strip() or str(config.get("trained_models_dir", "outputs/train")).strip()
    output_dir = Path(output_dir_text).expanduser()
    requested_value = str(raw_value or "").strip() or train_job_name_seed(config, dataset_input, policy_type)

    def _exists_locally(name: str) -> bool:
        run_dir = output_dir / name
        if not run_dir.exists():
            return False
        try:
            return any(run_dir.iterdir())
        except OSError:
            return True

    return resolve_available_name(
        requested_value,
        default_owner=str(config.get("hf_username", "")).strip(),
        prefer_owner_display=False,
        local_exists_fn=_exists_locally,
        remote_exists_fn=model_exists_on_hf,
        force_occupied=force_occupied,
    )
