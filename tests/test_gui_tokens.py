import unittest

from robot_pipeline_app.gui_tokens import build_theme_colors, normalize_theme_mode


class GuiTokensTests(unittest.TestCase):
    def test_normalize_theme_mode_defaults_to_dark(self):
        self.assertEqual(normalize_theme_mode(None), "dark")
        self.assertEqual(normalize_theme_mode("unknown"), "dark")

    def test_build_theme_colors_dark_and_light(self):
        dark = build_theme_colors(ui_font="A", mono_font="B", theme_mode="dark")
        light = build_theme_colors(ui_font="A", mono_font="B", theme_mode="light")
        self.assertEqual(dark["theme_mode"], "dark")
        self.assertEqual(light["theme_mode"], "light")
        self.assertNotEqual(dark["bg"], light["bg"])


if __name__ == "__main__":
    unittest.main()
