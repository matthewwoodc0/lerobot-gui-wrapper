from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repository root is importable for editable and source runs.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Prepare a headless Qt platform before any GUI test module imports Qt."""
    config.addinivalue_line("markers", "gui_qt: Qt GUI tests that require a usable PySide6 platform plugin")
    # Force a headless backend for CI and local automation unless the user
    # already selected a platform (including cocoa for interactive debugging).
    if not str(os.environ.get("QT_QPA_PLATFORM", "")).strip():
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    try:
        from robot_pipeline_app.qt_bootstrap import configure_headless_qt_for_tests, prepare_qt_environment

        prepare_qt_environment()
        # Only force headless when the selected/default platform cannot start.
        # configure_headless_qt_for_tests keeps offscreen/minimal when usable.
        if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() in {"", "offscreen", "minimal"}:
            try:
                configure_headless_qt_for_tests()
            except RuntimeError as exc:
                # Defer failure to individual GUI tests / suite guard.
                config._qt_bootstrap_error = str(exc)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - import-time environment issues
        config._qt_bootstrap_error = str(exc)  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Annotate GUI tests so the suite guard can detect unexpected mass skips."""
    for item in items:
        path = str(getattr(item, "fspath", "") or getattr(item, "path", "") or "")
        name = item.nodeid
        if "test_gui_qt" in path or "test_gui_qt" in name:
            item.add_marker(pytest.mark.gui_qt)


@pytest.fixture(autouse=True)
def isolate_config_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from reading or writing a user's real configuration files."""
    from robot_pipeline_app import config_store

    config_dir = tmp_path / "config"
    primary_path = config_dir / ".robot_config.json"
    secondary_path = config_dir / "lerobot" / ".robot_config.json"
    legacy_path = config_dir / ".robot_pipeline_config.json"

    monkeypatch.setattr(config_store, "PRIMARY_CONFIG_PATH", primary_path)
    monkeypatch.setattr(config_store, "DEFAULT_SECONDARY_CONFIG_PATH", secondary_path)
    monkeypatch.setattr(config_store, "LEGACY_CONFIG_PATH", legacy_path)
    monkeypatch.setattr(config_store, "get_secondary_config_path", lambda _config: secondary_path)
