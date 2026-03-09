from __future__ import annotations

import unittest

from robot_pipeline_app.gui_teleop_tab import _teleop_action_button_specs


class GuiTeleopTabHelpersTest(unittest.TestCase):
    def test_teleop_action_button_specs_make_run_teleop_primary_and_first(self) -> None:
        specs = _teleop_action_button_specs()

        self.assertEqual(specs[0], ("run", "Run Teleop", "Accent.TButton"))
        self.assertEqual(specs[1][0], "preview")
        self.assertEqual(specs[1][2], "Secondary.TButton")
        self.assertTrue(all(style != "Accent.TButton" for _, _, style in specs[1:]))


if __name__ == "__main__":
    unittest.main()
