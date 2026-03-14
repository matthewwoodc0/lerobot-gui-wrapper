from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.qt_bootstrap import prepare_qt_environment


class QtBootstrapTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
