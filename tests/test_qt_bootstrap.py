from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.qt_bootstrap import ensure_supported_qt_platform, prepare_qt_environment


class QtBootstrapTest(unittest.TestCase):
    def test_prepare_qt_environment_prepends_conda_lib_dir_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conda_lib = Path(tmpdir) / "lib"
            conda_lib.mkdir(parents=True, exist_ok=True)
            with patch("robot_pipeline_app.qt_bootstrap.sys.platform", "linux"), patch(
                "robot_pipeline_app.qt_bootstrap._resolve_pyside6_plugins_dir",
                return_value=None,
            ), patch.dict(
                os.environ,
                {"CONDA_PREFIX": tmpdir, "LD_LIBRARY_PATH": "/tmp/existing/lib"},
                clear=False,
            ):
                prepare_qt_environment()

                self.assertEqual(
                    os.environ["LD_LIBRARY_PATH"],
                    os.pathsep.join([str(conda_lib), "/tmp/existing/lib"]),
                )

    def test_prepare_qt_environment_prefers_pyside6_plugins_over_cv2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "PySide6" / "Qt" / "plugins"
            platform_dir = plugin_dir / "platforms"
            platform_dir.mkdir(parents=True, exist_ok=True)

            with patch("robot_pipeline_app.qt_bootstrap._resolve_pyside6_plugins_dir", return_value=plugin_dir), patch.dict(
                os.environ,
                {
                    "QT_PLUGIN_PATH": f"/tmp/cv2/qt/plugins{os.pathsep}/tmp/custom/plugins",
                    "QT_QPA_PLATFORM_PLUGIN_PATH": f"/tmp/cv2/qt/plugins/platforms{os.pathsep}/tmp/custom/platforms",
                },
                clear=False,
            ):
                prepare_qt_environment()

                self.assertEqual(
                    os.environ["QT_PLUGIN_PATH"],
                    os.pathsep.join([str(plugin_dir), "/tmp/custom/plugins"]),
                )
                self.assertEqual(
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"],
                    os.pathsep.join([str(platform_dir), "/tmp/custom/platforms"]),
                )

    def test_prepare_qt_environment_strips_cv2_paths_without_pyside6_dir(self) -> None:
        with patch("robot_pipeline_app.qt_bootstrap._resolve_pyside6_plugins_dir", return_value=None), patch.dict(
            os.environ,
            {
                "QT_PLUGIN_PATH": f"/tmp/cv2/qt/plugins{os.pathsep}/tmp/custom/plugins",
                "QT_QPA_PLATFORM_PLUGIN_PATH": "/tmp/cv2/qt/plugins/platforms",
            },
            clear=False,
        ):
            prepare_qt_environment()

            self.assertEqual(os.environ["QT_PLUGIN_PATH"], "/tmp/custom/plugins")
            self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ)

    def test_ensure_supported_qt_platform_prefers_wayland_when_available(self) -> None:
        def fake_probe(*, python_executable: str | None = None, platform_name: str | None = None) -> tuple[bool, str | None]:
            _ = python_executable
            return (platform_name == "wayland", None if platform_name == "wayland" else "unsupported")

        with patch("robot_pipeline_app.qt_bootstrap.sys.platform", "linux"), patch.dict(
            os.environ,
            {"QT_QPA_PLATFORM": "", "XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"},
            clear=False,
        ), patch("robot_pipeline_app.qt_bootstrap.probe_qt_platform_support", side_effect=fake_probe):
            os.environ.pop("QT_QPA_PLATFORM", None)
            ensure_supported_qt_platform()
            self.assertEqual(os.environ["QT_QPA_PLATFORM"], "wayland")

    def test_ensure_supported_qt_platform_raises_user_space_error_for_missing_xcb_cursor(self) -> None:
        def fake_probe(*, python_executable: str | None = None, platform_name: str | None = None) -> tuple[bool, str | None]:
            _ = python_executable
            if platform_name == "wayland":
                return False, "wayland unavailable"
            if platform_name == "xcb":
                return False, "From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed"
            return False, "default failed"

        with patch("robot_pipeline_app.qt_bootstrap.sys.platform", "linux"), patch.dict(
            os.environ,
            {"QT_QPA_PLATFORM": "", "DISPLAY": ":0"},
            clear=False,
        ), patch("robot_pipeline_app.qt_bootstrap.probe_qt_platform_support", side_effect=fake_probe):
            os.environ.pop("QT_QPA_PLATFORM", None)
            with self.assertRaises(RuntimeError) as ctx:
                ensure_supported_qt_platform()

        self.assertIn("missing the xcb-cursor library", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
