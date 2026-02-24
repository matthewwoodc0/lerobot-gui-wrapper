from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_pipeline_app.gui_training_tab import (
    _default_local_destination,
    _increment_destination,
    _remote_join,
    _remote_parent,
)


class GuiTrainingTabHelpersTest(unittest.TestCase):
    def test_default_local_destination_uses_remote_basename(self) -> None:
        config = {"trained_models_dir": "/tmp/models"}
        destination = _default_local_destination(config, "~/checkpoints/model_a")
        self.assertEqual(destination, Path("/tmp/models/model_a"))

    def test_remote_parent(self) -> None:
        self.assertEqual(_remote_parent("~/models/a"), "~/models")
        self.assertEqual(_remote_parent("/"), "/")

    def test_remote_join(self) -> None:
        self.assertEqual(_remote_join("~/models", "run1"), "~/models/run1")
        self.assertEqual(_remote_join("/tmp", "x"), "/tmp/x")

    def test_increment_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            base.mkdir()
            second = Path(tmpdir) / "model_2"
            second.mkdir()
            candidate = _increment_destination(base)
        self.assertEqual(candidate.name, "model_3")


if __name__ == "__main__":
    unittest.main()
