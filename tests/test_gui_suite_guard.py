from __future__ import annotations

import importlib.util
import os
import unittest

from robot_pipeline_app.qt_bootstrap import prepare_qt_environment, probe_qt_platform_support


class GuiSuiteGuardTests(unittest.TestCase):
    """Fail the suite when PySide6 is installed but GUI tests would mass-skip."""

    def test_pyside6_headless_platform_is_usable_when_installed(self) -> None:
        if importlib.util.find_spec("PySide6") is None:
            self.skipTest("PySide6 is not installed in this environment")

        prepare_qt_environment()
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        ok_offscreen, reason_offscreen = probe_qt_platform_support(platform_name="offscreen")
        ok_minimal, reason_minimal = probe_qt_platform_support(platform_name="minimal")
        self.assertTrue(
            ok_offscreen or ok_minimal,
            msg=(
                "PySide6 is installed but neither offscreen nor minimal Qt platforms could initialize. "
                "GUI tests must not mass-skip when PySide6 is present. "
                f"offscreen={reason_offscreen!r}; minimal={reason_minimal!r}. "
                "Reinstall PySide6 in a clean virtual environment."
            ),
        )


if __name__ == "__main__":
    unittest.main()
