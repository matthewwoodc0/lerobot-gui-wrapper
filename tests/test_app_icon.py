from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.app_icon import find_app_icon_png


class AppIconTests(unittest.TestCase):
    def test_find_app_icon_png_prefers_known_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            icon_dir = app_dir / "Resources" / "icons"
            icon_dir.mkdir(parents=True, exist_ok=True)
            (icon_dir / "lerobot-pipeline-manager-512.png").write_bytes(b"png")

            found = find_app_icon_png(app_dir)

        self.assertEqual(found, (icon_dir / "lerobot-pipeline-manager-512.png").resolve())


if __name__ == "__main__":
    unittest.main()
