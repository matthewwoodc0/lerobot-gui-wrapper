from __future__ import annotations

import re
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"


class ReadmeQuickStartTests(unittest.TestCase):
    """Prove Quick Start is copy-pasteable for a first-run install→launch path."""

    def test_quick_start_clones_before_editable_install_and_uses_console_entry(self) -> None:
        text = _README.read_text(encoding="utf-8")
        match = re.search(r"## Quick Start\n+(.*?)(?:\n## |\Z)", text, re.S)
        self.assertIsNotNone(match, "README must contain a Quick Start section")
        assert match is not None
        section = match.group(1)

        self.assertIn("git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git", section)
        self.assertIn("cd lerobot-gui-wrapper", section)
        self.assertIn('pip install -e ".[gui]"', section)
        self.assertIn("lerobot-pipeline-manager gui", section)

        # Do not claim a PyPI package name that is not published.
        self.assertNotIn('pip install "lerobot-gui-wrapper', section)
        self.assertNotIn("pip install lerobot-gui-wrapper", section)

        clone_i = section.index("git clone")
        cd_i = section.index("cd lerobot-gui-wrapper")
        pip_i = section.index('pip install -e ".[gui]"')
        launch_i = section.index("lerobot-pipeline-manager gui")
        self.assertLess(clone_i, cd_i)
        self.assertLess(cd_i, pip_i)
        self.assertLess(pip_i, launch_i)

    def test_platform_setup_primary_launch_uses_console_entry(self) -> None:
        text = _README.read_text(encoding="utf-8")
        self.assertIn("## Platform Setup", text)

        macos = text.split("### macOS", 1)[1].split("### Linux", 1)[0]
        self.assertIn("lerobot-pipeline-manager gui", macos)
        primary_launch = macos.index("lerobot-pipeline-manager gui")
        # Developer fallback may still mention robot_pipeline.py after the primary path.
        if "robot_pipeline.py" in macos:
            self.assertLess(primary_launch, macos.index("robot_pipeline.py"))

        linux = text.split("### Linux", 1)[1].split("### Linux — Shared", 1)[0]
        self.assertIn("lerobot-pipeline-manager gui", linux)
        self.assertNotIn(
            "```bash\npython3 robot_pipeline.py gui\n```",
            linux,
            "Linux Platform Setup must not present robot_pipeline.py as the primary launch block",
        )


if __name__ == "__main__":
    unittest.main()
