from __future__ import annotations

import unittest

from robot_pipeline_app.gui_window import fit_window_to_screen


class _FakeWindow:
    def __init__(self, *, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.min_size: tuple[int, int] | None = None
        self.geometry_value: str | None = None

    def winfo_screenwidth(self) -> int:
        return self.screen_w

    def winfo_screenheight(self) -> int:
        return self.screen_h

    def minsize(self, width: int, height: int) -> None:
        self.min_size = (width, height)

    def geometry(self, value: str) -> None:
        self.geometry_value = value


class GuiWindowTests(unittest.TestCase):
    def test_fit_window_to_screen_caps_dimensions_and_centers(self) -> None:
        window = _FakeWindow(screen_w=1280, screen_h=720)
        final_w, final_h = fit_window_to_screen(
            window=window,
            requested_width=1500,
            requested_height=900,
            requested_min_width=1200,
            requested_min_height=800,
            margin_x=24,
            margin_y=48,
            center=True,
        )

        self.assertEqual((final_w, final_h), (1256, 672))
        self.assertEqual(window.min_size, (1200, 672))
        self.assertEqual(window.geometry_value, "1256x672+12+24")

    def test_fit_window_to_screen_non_center_uses_plain_geometry(self) -> None:
        window = _FakeWindow(screen_w=1920, screen_h=1080)
        final_w, final_h = fit_window_to_screen(
            window=window,
            requested_width=900,
            requested_height=700,
            requested_min_width=600,
            requested_min_height=500,
            center=False,
        )

        self.assertEqual((final_w, final_h), (900, 700))
        self.assertEqual(window.min_size, (600, 500))
        self.assertEqual(window.geometry_value, "900x700")

    def test_fit_window_to_screen_enforces_baseline_minimum_visible_area(self) -> None:
        window = _FakeWindow(screen_w=200, screen_h=200)
        final_w, final_h = fit_window_to_screen(
            window=window,
            requested_width=1000,
            requested_height=1000,
            requested_min_width=1000,
            requested_min_height=1000,
            margin_x=24,
            margin_y=48,
            center=True,
        )

        self.assertEqual((final_w, final_h), (360, 280))
        self.assertEqual(window.min_size, (360, 280))
        self.assertEqual(window.geometry_value, "360x280+8+8")


if __name__ == "__main__":
    unittest.main()
