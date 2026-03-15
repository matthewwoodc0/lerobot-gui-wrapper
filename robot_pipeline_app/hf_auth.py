from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


_HF_TOKEN_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def huggingface_token_paths(*, env: Mapping[str, str] | None = None, home: Path | None = None) -> tuple[Path, ...]:
    env_map = env if env is not None else os.environ
    home_dir = home if home is not None else Path.home()
    candidates: list[Path] = []

    raw_token_path = str(env_map.get("HF_TOKEN_PATH", "")).strip()
    if raw_token_path:
        candidates.append(Path(os.path.expandvars(raw_token_path)).expanduser())

    raw_hf_home = str(env_map.get("HF_HOME", "")).strip()
    if raw_hf_home:
        hf_home = Path(os.path.expandvars(raw_hf_home)).expanduser()
    else:
        raw_xdg_cache = str(env_map.get("XDG_CACHE_HOME", "")).strip()
        cache_root = Path(os.path.expandvars(raw_xdg_cache)).expanduser() if raw_xdg_cache else home_dir / ".cache"
        hf_home = cache_root / "huggingface"
    candidates.append(hf_home / "token")
    candidates.append(home_dir / ".huggingface" / "token")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return tuple(unique_candidates)


def has_huggingface_auth_token(*, env: Mapping[str, str] | None = None, home: Path | None = None) -> bool:
    env_map = env if env is not None else os.environ
    for key in _HF_TOKEN_ENV_KEYS:
        if str(env_map.get(key, "")).strip():
            return True
    for token_path in huggingface_token_paths(env=env_map, home=home):
        try:
            if token_path.is_file() and token_path.read_text(encoding="utf-8").strip():
                return True
        except OSError:
            continue
    return False
