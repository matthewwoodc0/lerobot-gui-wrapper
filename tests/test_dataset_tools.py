from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.compat import resolve_edit_dataset_entrypoint, resolve_visualize_dataset_entrypoint
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.dataset_tools import (
    build_delete_episodes_command,
    build_keep_episodes_command,
    build_merge_datasets_command,
)
from robot_pipeline_app.visualize_tools import build_visualize_dataset_command


class DatasetToolsTest(unittest.TestCase):
    def test_build_delete_episodes_command_uses_root_style_flags_for_lerobot_edit_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "lerobot_env"
            python_path = venv_dir / "bin" / "python3"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(venv_dir)
            config["record_data_dir"] = str(Path(tmpdir) / "data")

            with patch(
                "robot_pipeline_app.dataset_tools.resolve_edit_dataset_entrypoint",
                return_value="lerobot.scripts.lerobot_edit_dataset",
            ):
                cmd = build_delete_episodes_command(config, "alice/demo_dataset", [3, 7, 12])

        self.assertEqual(
            cmd,
            [
                str(python_path),
                "-m",
                "lerobot.scripts.lerobot_edit_dataset",
                "--repo_id=alice/demo_dataset",
                f"--root={Path(tmpdir) / 'data'}",
                "--operation.type=delete_episodes",
                "--operation.episode_indices=[3, 7, 12]",
            ],
        )

    def test_build_keep_episodes_command_produces_correct_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "lerobot_env"
            python_path = venv_dir / "bin" / "python3"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(venv_dir)
            config["lerobot_dir"] = ""
            config["record_data_dir"] = ""

            with patch("robot_pipeline_app.dataset_tools.resolve_edit_dataset_entrypoint", return_value="lerobot.edit_dataset"):
                cmd = build_keep_episodes_command(config, "alice/demo_dataset", [0, 1, 2])

        self.assertEqual(
            cmd,
            [
                str(python_path),
                "-m",
                "lerobot.edit_dataset",
                "--dataset.repo_id=alice/demo_dataset",
                "--operation.type=keep_episodes",
                "--operation.episode_indices=[0, 1, 2]",
            ],
        )

    def test_build_merge_datasets_command_produces_correct_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "lerobot_env"
            python_path = venv_dir / "bin" / "python3"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(venv_dir)
            config["record_data_dir"] = str(Path(tmpdir) / "data")

            with patch(
                "robot_pipeline_app.dataset_tools.resolve_edit_dataset_entrypoint",
                return_value="lerobot.scripts.lerobot_edit_dataset",
            ):
                cmd = build_merge_datasets_command(
                    config,
                    "alice/merged_dataset",
                    ["alice/train_a", "alice/train_b"],
                )

        self.assertEqual(
            cmd,
            [
                str(python_path),
                "-m",
                "lerobot.scripts.lerobot_edit_dataset",
                "--repo_id=alice/merged_dataset",
                f"--root={Path(tmpdir) / 'data'}",
                "--operation.type=merge",
                '--operation.repo_ids=["alice/train_a", "alice/train_b"]',
            ],
        )

    def test_build_visualize_dataset_command_produces_correct_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "lerobot_env"
            python_path = venv_dir / "bin" / "python3"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_venv_dir"] = str(venv_dir)

            with patch(
                "robot_pipeline_app.visualize_tools.resolve_visualize_dataset_entrypoint",
                return_value="lerobot.scripts.visualize_dataset",
            ):
                cmd = build_visualize_dataset_command(config, "alice/demo_dataset", 0)

        self.assertEqual(
            cmd,
            [
                str(python_path),
                "-m",
                "lerobot.scripts.visualize_dataset",
                "--repo-id",
                "alice/demo_dataset",
                "--episode-index",
                "0",
            ],
        )

    def test_resolve_edit_dataset_entrypoint_returns_module_string(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["lerobot_dir"] = ""

        with patch(
            "robot_pipeline_app.compat._lerobot_module_available",
            side_effect=lambda _config, name: name == "lerobot.edit_dataset",
        ):
            entrypoint = resolve_edit_dataset_entrypoint(config)

        self.assertIsInstance(entrypoint, str)
        self.assertEqual(entrypoint, "lerobot.edit_dataset")

    def test_resolve_visualize_dataset_entrypoint_returns_module_string(self) -> None:
        config = dict(DEFAULT_CONFIG_VALUES)
        config["lerobot_dir"] = ""

        with patch(
            "robot_pipeline_app.compat._lerobot_module_available",
            side_effect=lambda _config, name: name == "lerobot.visualize_dataset",
        ):
            entrypoint = resolve_visualize_dataset_entrypoint(config)

        self.assertIsInstance(entrypoint, str)
        self.assertEqual(entrypoint, "lerobot.visualize_dataset")

    def test_resolve_edit_dataset_entrypoint_detects_src_checkout_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot"
            script_path = lerobot_dir / "src" / "lerobot" / "scripts" / "lerobot_edit_dataset.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)

            entrypoint = resolve_edit_dataset_entrypoint(config)

        self.assertEqual(entrypoint, "lerobot.scripts.lerobot_edit_dataset")

    def test_resolve_visualize_dataset_entrypoint_detects_src_checkout_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lerobot_dir = Path(tmpdir) / "lerobot"
            script_path = lerobot_dir / "src" / "lerobot" / "scripts" / "lerobot_dataset_viz.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("", encoding="utf-8")

            config = dict(DEFAULT_CONFIG_VALUES)
            config["lerobot_dir"] = str(lerobot_dir)

            entrypoint = resolve_visualize_dataset_entrypoint(config)

        self.assertEqual(entrypoint, "lerobot.scripts.lerobot_dataset_viz")


if __name__ == "__main__":
    unittest.main()
