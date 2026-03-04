from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES, PLATFORM_PORT_DEFAULTS


class ReadmeConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.readme_text = (cls.repo_root / "README.md").read_text(encoding="utf-8")
        cls.error_catalog_text = (cls.repo_root / "docs" / "error-catalog.md").read_text(encoding="utf-8")

    def test_readme_has_persona_first_paths(self) -> None:
        self.assertIn("## Quick Start by Persona", self.readme_text)
        self.assertIn("### Quick Start (macOS venv)", self.readme_text)
        self.assertIn("### Quick Start (Linux sudo)", self.readme_text)
        self.assertIn("### Quick Start (Conda / No sudo)", self.readme_text)
        self.assertIn("### Lab Maintainer", self.readme_text)

    def test_readme_defaults_block_matches_constants(self) -> None:
        start_marker = "<!-- README_DEFAULTS_START -->"
        end_marker = "<!-- README_DEFAULTS_END -->"
        start = self.readme_text.index(start_marker) + len(start_marker)
        end = self.readme_text.index(end_marker)
        payload = self.readme_text[start:end]
        payload = payload.replace("```json", "").replace("```", "").strip()
        data = json.loads(payload)

        self.assertEqual(data["camera_default_width"], DEFAULT_CONFIG_VALUES["camera_default_width"])
        self.assertEqual(data["camera_default_height"], DEFAULT_CONFIG_VALUES["camera_default_height"])
        self.assertEqual(data["camera_fps"], DEFAULT_CONFIG_VALUES["camera_fps"])
        self.assertEqual(data["camera_warmup_s"], DEFAULT_CONFIG_VALUES["camera_warmup_s"])
        self.assertEqual(data["follower_robot_id_default"], DEFAULT_CONFIG_VALUES["follower_robot_id"])
        self.assertEqual(data["leader_robot_id_default"], DEFAULT_CONFIG_VALUES["leader_robot_id"])
        self.assertEqual(data["ports"]["linux"], PLATFORM_PORT_DEFAULTS["linux"])
        self.assertEqual(data["ports"]["darwin"], PLATFORM_PORT_DEFAULTS["darwin"])

    def test_readme_has_no_stale_linux_default_warning(self) -> None:
        self.assertNotIn("defaults are Linux-style and will not work on macOS", self.readme_text)

    def test_fix_messages_are_mapped_in_error_catalog(self) -> None:
        fix_messages: set[str] = set()
        pattern = re.compile(r"Fix:\s*([^\"]+)")
        for source_name in ("checks.py", "deploy_diagnostics.py"):
            source_text = (self.repo_root / "robot_pipeline_app" / source_name).read_text(encoding="utf-8")
            for line in source_text.splitlines():
                if "Fix:" not in line:
                    continue
                match = pattern.search(line)
                if match is None:
                    continue
                message = match.group(1).strip()
                message = message.rstrip('",')
                if message:
                    fix_messages.add(message)

        catalog_normalized = " ".join(self.error_catalog_text.split())
        missing = [
            message
            for message in sorted(fix_messages)
            if " ".join(message.split()) not in catalog_normalized
        ]
        self.assertEqual(missing, [], f"Missing fix mappings in docs/error-catalog.md: {missing}")


if __name__ == "__main__":
    unittest.main()
