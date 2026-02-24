from __future__ import annotations

import unittest

from robot_pipeline_app.training_templates import default_training_templates, render_template, template_variables


class TrainingTemplatesTest(unittest.TestCase):
    def test_default_training_templates_include_expected_ids(self) -> None:
        templates = default_training_templates()
        ids = {template["id"] for template in templates}
        self.assertIn("srun_tmux", ids)
        self.assertIn("tmux_custom", ids)
        self.assertIn("custom", ids)

    def test_template_variables_returns_unique_ordered_names(self) -> None:
        names = template_variables("{a} {b} {a} {c}")
        self.assertEqual(names, ["a", "b", "c"])

    def test_render_template_success(self) -> None:
        rendered, error = render_template(
            "{env_activate_cmd} && {train_command}",
            {"env_activate_cmd": "source env/bin/activate", "train_command": "python train.py"},
        )
        self.assertIsNone(error)
        self.assertIn("python train.py", str(rendered))

    def test_render_template_missing_value_errors(self) -> None:
        rendered, error = render_template("{a} {b}", {"a": "one"})
        self.assertIsNone(rendered)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("missing", error.lower())


if __name__ == "__main__":
    unittest.main()
