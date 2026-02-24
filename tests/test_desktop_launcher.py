from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.desktop_launcher import install_desktop_launcher


class DesktopLauncherTest(unittest.TestCase):
    def test_install_desktop_launcher_writes_linux_launcher_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app_dir = root / "LeRobot GUI Wrapper"
            app_dir.mkdir(parents=True, exist_ok=True)
            (app_dir / "robot_pipeline.py").write_text("print('ok')\n", encoding="utf-8")

            python_bin = root / "venv" / "bin" / "python3"
            python_bin.parent.mkdir(parents=True, exist_ok=True)
            python_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python_bin.chmod(0o755)

            home_dir = root / "home"
            home_dir.mkdir(parents=True, exist_ok=True)

            result = install_desktop_launcher(
                app_dir=app_dir,
                python_executable=python_bin,
                platform_name="linux",
                home_dir=home_dir,
            )

            self.assertTrue(result.ok)
            self.assertIsNotNone(result.script_path)
            self.assertIsNotNone(result.desktop_entry_path)
            assert result.script_path is not None
            assert result.desktop_entry_path is not None
            self.assertTrue(result.script_path.exists())
            self.assertTrue(result.desktop_entry_path.exists())

            script_text = result.script_path.read_text(encoding="utf-8")
            self.assertIn(f'APP_DIR="{app_dir.resolve()}"', script_text)
            self.assertIn(f'PYTHON_BIN="{python_bin.resolve()}"', script_text)
            self.assertIn('"$APP_DIR/robot_pipeline.py" gui "$@"', script_text)

            desktop_text = result.desktop_entry_path.read_text(encoding="utf-8")
            self.assertIn(f"Exec={result.script_path}", desktop_text)
            self.assertIn("Name=LeRobot Pipeline Manager", desktop_text)

    def test_install_desktop_launcher_rejects_non_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            (app_dir / "robot_pipeline.py").write_text("print('ok')\n", encoding="utf-8")
            result = install_desktop_launcher(
                app_dir=app_dir,
                platform_name="darwin",
            )
            self.assertFalse(result.ok)
            self.assertIn("Linux only", result.message)


if __name__ == "__main__":
    unittest.main()
