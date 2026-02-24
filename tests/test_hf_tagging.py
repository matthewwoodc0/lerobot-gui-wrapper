from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.hf_tagging import (
    build_dataset_card_text,
    build_dataset_tag_upload_command,
    default_dataset_tags,
    safe_unlink,
    write_dataset_card_temp,
)


class HfTaggingTest(unittest.TestCase):
    def test_default_dataset_tags_include_expected_values(self) -> None:
        tags = default_dataset_tags(
            config={"hf_username": "alice"},
            dataset_repo_id="alice/demo_2",
            task="Pick up the block and place it",
        )
        self.assertIn("lerobot", tags)
        self.assertIn("so101", tags)
        self.assertIn("dataset-demo-2", tags)
        self.assertTrue(any(tag.startswith("task-") for tag in tags))

    def test_build_dataset_card_text_contains_tags_and_name(self) -> None:
        card = build_dataset_card_text(
            dataset_repo_id="alice/demo_3",
            dataset_name="demo_3",
            tags=["lerobot", "dataset-demo-3"],
            task="Move block",
        )
        self.assertIn("tags:", card)
        self.assertIn("dataset-demo-3", card)
        self.assertIn("Name: `demo_3`", card)

    def test_write_card_and_build_upload_command(self) -> None:
        path = write_dataset_card_temp(
            dataset_repo_id="alice/demo_4",
            dataset_name="demo_4",
            tags=["lerobot"],
            task=None,
        )
        try:
            self.assertTrue(path.exists())
            cmd = build_dataset_tag_upload_command("alice/demo_4", path)
            self.assertEqual(cmd[0], "huggingface-cli")
            self.assertIn("README.md", cmd)
        finally:
            safe_unlink(path)

    def test_safe_unlink_ignores_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.md"
            safe_unlink(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
