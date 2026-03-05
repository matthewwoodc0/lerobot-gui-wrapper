from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.app_icon import apply_tk_app_icon, find_app_icon_png


class _FakeTk:
    def __init__(self) -> None:
        self.loaded_paths: list[str] = []

    def PhotoImage(self, *, file: str):  # noqa: N802 - mimic tkinter API
        self.loaded_paths.append(file)
        return {"file": file}


class _FakeRoot:
    def __init__(self) -> None:
        self.iconphoto_calls: list[tuple[bool, object]] = []

    def iconphoto(self, default: bool, image: object) -> None:
        self.iconphoto_calls.append((default, image))


class AppIconTests(unittest.TestCase):
    def test_find_app_icon_png_prefers_known_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            icon_dir = app_dir / "Resources" / "icons"
            icon_dir.mkdir(parents=True, exist_ok=True)
            (icon_dir / "lerobot-pipeline-manager-512.png").write_bytes(b"png")

            found = find_app_icon_png(app_dir)

        self.assertEqual(found, (icon_dir / "lerobot-pipeline-manager-512.png").resolve())

    def test_apply_tk_app_icon_sets_iconphoto_and_keeps_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            icon_dir = app_dir / "Resources" / "icons"
            icon_dir.mkdir(parents=True, exist_ok=True)
            icon_path = icon_dir / "lerobot-pipeline-manager-256.png"
            icon_path.write_bytes(b"png")

            root = _FakeRoot()
            tk = _FakeTk()
            with patch("robot_pipeline_app.app_icon.sys.platform", "linux"):
                applied = apply_tk_app_icon(root=root, tk_module=tk, app_dir=app_dir)

        self.assertEqual(applied, icon_path.resolve())
        self.assertEqual(len(tk.loaded_paths), 1)
        self.assertEqual(Path(tk.loaded_paths[0]).resolve(), icon_path.resolve())
        self.assertEqual(len(root.iconphoto_calls), 1)
        self.assertTrue(hasattr(root, "_lerobot_app_icon_photo"))

    def test_apply_tk_app_icon_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir)
            root = _FakeRoot()
            tk = _FakeTk()
            applied = apply_tk_app_icon(root=root, tk_module=tk, app_dir=app_dir)

        self.assertIsNone(applied)
        self.assertEqual(tk.loaded_paths, [])
        self.assertEqual(root.iconphoto_calls, [])


if __name__ == "__main__":
    unittest.main()
