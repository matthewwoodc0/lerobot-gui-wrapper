from __future__ import annotations

import unittest

from robot_pipeline_app.gui_app import _apply_runtime_theme_to_components


class _ThemeSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def apply_theme(self, colors: dict[str, str]) -> None:
        self.calls.append(colors)


class _NoTheme:
    pass


class GuiAppThemePropagationTests(unittest.TestCase):
    def test_apply_runtime_theme_propagates_to_all_theme_aware_components(self) -> None:
        colors = {"theme_mode": "dark", "accent": "#f0a500"}
        log_panel = _ThemeSink()
        record_preview = _ThemeSink()
        deploy_preview = _ThemeSink()
        teleop_preview = _ThemeSink()
        training_handles = _ThemeSink()
        config_handles = _ThemeSink()
        visualizer_handles = _ThemeSink()
        history_handles = _ThemeSink()

        _apply_runtime_theme_to_components(
            colors=colors,
            log_panel=log_panel,
            preview_handles={
                "record": record_preview,
                "deploy": deploy_preview,
                "teleop": teleop_preview,
            },
            training_handles_ref={"handles": training_handles},
            config_tab_handles={"handles": config_handles},
            visualizer_handles_ref={"handles": visualizer_handles},
            history_handles_ref={"handles": history_handles},
        )

        for sink in (
            log_panel,
            record_preview,
            deploy_preview,
            teleop_preview,
            training_handles,
            config_handles,
            visualizer_handles,
            history_handles,
        ):
            self.assertEqual(len(sink.calls), 1)
            self.assertIs(sink.calls[0], colors)

    def test_apply_runtime_theme_skips_missing_or_non_theme_targets(self) -> None:
        colors = {"theme_mode": "light"}
        log_panel = _ThemeSink()
        record_preview = _ThemeSink()

        _apply_runtime_theme_to_components(
            colors=colors,
            log_panel=log_panel,
            preview_handles={
                "record": record_preview,
                "deploy": None,
                "teleop": _NoTheme(),
            },
            training_handles_ref={"handles": _NoTheme()},
            config_tab_handles={"handles": None},
            visualizer_handles_ref={"handles": _NoTheme()},
            history_handles_ref={"handles": None},
        )

        self.assertEqual(len(log_panel.calls), 1)
        self.assertEqual(len(record_preview.calls), 1)


if __name__ == "__main__":
    unittest.main()
