from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app import config_store as cs


class ConfigStoreTest(unittest.TestCase):
    def test_load_raw_config_uses_precedence_primary_then_secondary_then_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            primary = base / "primary.json"
            secondary = base / "secondary.json"
            legacy = base / "legacy.json"

            secondary.write_text('{"source":"secondary"}', encoding="utf-8")
            legacy.write_text('{"source":"legacy"}', encoding="utf-8")

            with patch.object(cs, "PRIMARY_CONFIG_PATH", primary), patch.object(
                cs,
                "DEFAULT_SECONDARY_CONFIG_PATH",
                secondary,
            ), patch.object(cs, "LEGACY_CONFIG_PATH", legacy):
                cfg, source = cs.load_raw_config()
                self.assertEqual(cfg["source"], "secondary")
                self.assertEqual(source, secondary)

                primary.write_text('{"source":"primary"}', encoding="utf-8")
                cfg2, source2 = cs.load_raw_config()
                self.assertEqual(cfg2["source"], "primary")
                self.assertEqual(source2, primary)

    def test_normalize_config_without_prompts_coerces_ints_and_paths(self) -> None:
        raw = {
            "lerobot_dir": "~/lerobot",
            "runs_dir": "~/runs",
            "camera_laptop_index": "5",
            "camera_phone_index": "6",
            "camera_width": "800",
            "camera_height": "450",
            "camera_fps": "24",
            "hf_username": "alice",
        }
        normalized = cs.normalize_config_without_prompts(raw)
        self.assertIsInstance(normalized["camera_laptop_index"], int)
        self.assertEqual(normalized["camera_width"], 800)
        self.assertTrue(str(normalized["lerobot_dir"]).startswith(str(Path.home())))


if __name__ == "__main__":
    unittest.main()
