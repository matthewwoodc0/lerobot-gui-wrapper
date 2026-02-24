from __future__ import annotations

import unittest

from robot_pipeline_app.gui_run_popout import is_episode_reset_phase_line, parse_episode_progress_line


class GuiLogParsingTest(unittest.TestCase):
    def test_parse_episode_progress_line_with_total(self) -> None:
        self.assertEqual(parse_episode_progress_line("Episode 3/10 started"), (3, 10))
        self.assertEqual(parse_episode_progress_line("episode 4 of 12"), (4, 12))
        self.assertEqual(parse_episode_progress_line("Episode: 5/20"), (5, 20))
        self.assertEqual(parse_episode_progress_line("ep 6 of 20"), (6, 20))
        self.assertEqual(parse_episode_progress_line("episode_idx=7/20"), (7, 20))

    def test_parse_episode_progress_line_partial(self) -> None:
        self.assertEqual(parse_episode_progress_line("Episode 5"), (5, None))

    def test_parse_episode_progress_line_no_match(self) -> None:
        self.assertIsNone(parse_episode_progress_line("no episode info"))

    def test_is_episode_reset_phase_line_matches_common_prompts(self) -> None:
        self.assertTrue(is_episode_reset_phase_line("Press left arrow to reset episode, right arrow for next episode"))
        self.assertTrue(is_episode_reset_phase_line("redo run / next run"))

    def test_is_episode_reset_phase_line_ignores_unrelated_lines(self) -> None:
        self.assertFalse(is_episode_reset_phase_line("Episode 2/10 started"))


if __name__ == "__main__":
    unittest.main()
