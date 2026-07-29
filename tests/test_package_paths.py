from __future__ import annotations

from pathlib import Path

from robot_pipeline_app.package_paths import bundled_root, resolve_icon_png, resolve_schema_file


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_resources_match_checkout_sources() -> None:
    mappings = (
        (bundled_root() / "guides", ROOT / "Resources"),
        (bundled_root() / "icons", ROOT / "Resources" / "icons"),
        (bundled_root() / "schema", ROOT / "schema"),
    )

    for bundled_dir, source_dir in mappings:
        for bundled_path in sorted(path for path in bundled_dir.iterdir() if path.is_file()):
            source_path = source_dir / bundled_path.name
            assert source_path.is_file(), f"Bundled resource has no checkout source: {bundled_path}"
            assert bundled_path.read_bytes() == source_path.read_bytes(), (
                f"Bundled resource is stale: {bundled_path}"
            )


def test_runtime_resource_resolvers_find_packaged_files() -> None:
    icon_path = resolve_icon_png()
    schema_path = resolve_schema_file("diagnostic_event.v1.json")

    assert icon_path is not None and icon_path.is_file()
    assert schema_path is not None and schema_path.is_file()
    assert bundled_root() in icon_path.parents
    assert bundled_root() in schema_path.parents
