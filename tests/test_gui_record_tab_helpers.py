from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import robot_pipeline_app.gui_record_tab as gui_record_tab
from robot_pipeline_app.gui_record_tab import (
    _build_v30_convert_command,
    _compose_repo_id,
    _hf_parity_detail,
    _list_local_dataset_dirs,
    _record_action_button_specs,
)


class GuiRecordTabHelpersTest(unittest.TestCase):
    def test_compose_repo_id(self) -> None:
        self.assertEqual(_compose_repo_id("alice", "demo_1"), "alice/demo_1")
        self.assertEqual(_compose_repo_id("/alice/", "/demo_1/"), "alice/demo_1")
        self.assertEqual(_compose_repo_id("alice", "alice/demo_1"), "alice/demo_1")
        self.assertIsNone(_compose_repo_id("alice", "alice/"))
        self.assertIsNone(_compose_repo_id("", "demo_1"))
        self.assertIsNone(_compose_repo_id("alice", ""))

    def test_hf_parity_detail(self) -> None:
        self.assertEqual(_hf_parity_detail(True, "alice/demo_1")[0], "WARN")
        self.assertEqual(_hf_parity_detail(False, "alice/demo_1")[0], "PASS")
        self.assertEqual(_hf_parity_detail(None, "alice/demo_1")[0], "WARN")

    def test_build_v30_convert_command(self) -> None:
        cmd = _build_v30_convert_command("alice/demo_1", python_bin="/usr/bin/python3")
        self.assertEqual(
            cmd,
            [
                "/usr/bin/python3",
                "-m",
                "lerobot.datasets.v30.convert_dataset_v21_to_v30",
                "--repo-id=alice/demo_1",
            ],
        )

    def test_list_local_dataset_dirs_merges_roots_and_skips_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record_root = root / "record_data"
            lerobot_dir = root / "lerobot"
            lerobot_data = lerobot_dir / "data"
            record_root.mkdir(parents=True, exist_ok=True)
            lerobot_data.mkdir(parents=True, exist_ok=True)

            (record_root / "demo_a").mkdir()
            (record_root / ".hidden").mkdir()
            (lerobot_data / "demo_b").mkdir()

            items = _list_local_dataset_dirs(record_root, lerobot_dir)
            names = [item.name for item in items]

        self.assertEqual(names, ["demo_a", "demo_b"])

    def test_setup_record_tab_binds_dataset_browser_scroll_widgets(self) -> None:
        source_path = Path(gui_record_tab.__file__)
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        setup_record_tab = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_record_tab"
        )
        bound_names = {
            node.args[0].id
            for node in ast.walk(setup_record_tab)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bind_yview_wheel_scroll"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }

        self.assertIn("dataset_tree", bound_names)
        self.assertIn("dataset_meta_text", bound_names)

    def test_record_action_button_specs_make_run_record_primary_and_first(self) -> None:
        specs = _record_action_button_specs()

        self.assertEqual(specs[0], ("run", "Run Record", "Accent.TButton"))
        self.assertEqual(specs[1][0], "preview")
        self.assertEqual(specs[1][2], "Secondary.TButton")
        self.assertTrue(all(style != "Accent.TButton" for _, _, style in specs[1:]))


if __name__ == "__main__":
    unittest.main()
