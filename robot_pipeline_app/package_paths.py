from __future__ import annotations

from pathlib import Path


def package_dir() -> Path:
    """Return the installed `robot_pipeline_app` package directory."""
    return Path(__file__).resolve().parent


def project_root() -> Path | None:
    """Return the source checkout root when present, else None.

    Installed wheels also ship ``robot_pipeline.py`` in site-packages, so a lone
    module file is not enough. Require checkout-only layout markers.
    """
    candidate = package_dir().parent
    # Source checkouts keep docs/resources and tests beside the package tree.
    if (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "Resources").is_dir()
        and (candidate / "tests").is_dir()
        and (candidate / "robot_pipeline_app").is_dir()
    ):
        return candidate
    return None


def bundled_root() -> Path:
    return package_dir() / "bundled"


def resolve_resource(*parts: str) -> Path | None:
    """Resolve a resource path for installed wheels and source checkouts.

    Preference order:
    1. Package-bundled data under ``robot_pipeline_app/bundled/``
    2. Source checkout ``Resources/`` or ``schema/`` siblings
    """
    if not parts:
        return None

    relative = Path(*parts)
    bundled = bundled_root() / relative
    try:
        if bundled.exists():
            return bundled
    except OSError:
        pass

    # Map common top-level names used by the checkout layout.
    root = project_root()
    if root is not None:
        checkout_candidates = [
            root.joinpath(*parts),
            root / "Resources" / relative,
            root / "schema" / relative,
        ]
        # When caller already includes Resources/ or schema/, try as-is only.
        if parts[0] in {"Resources", "schema", "docs"}:
            checkout_candidates = [root.joinpath(*parts)]
        for candidate in checkout_candidates:
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
    return None


def resolve_icon_png() -> Path | None:
    names = (
        "lerobot-pipeline-manager-256.png",
        "lerobot-pipeline-manager-512.png",
        "lerobot-pipeline-manager-1024.png",
        "lerobot-pipeline-manager.png",
    )
    for name in names:
        candidate = resolve_resource("icons", name)
        if candidate is not None and candidate.is_file():
            return candidate
        # checkout Resources/icons
        candidate = resolve_resource("Resources", "icons", name)
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def resolve_schema_file(name: str) -> Path | None:
    for parts in (
        ("schema", name),
        ("Resources", "schema", name),
        (name,),
    ):
        candidate = resolve_resource(*parts)
        if candidate is not None and candidate.is_file():
            return candidate
    return None
