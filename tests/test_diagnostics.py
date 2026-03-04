from __future__ import annotations

import unittest

from robot_pipeline_app.diagnostics import check_result_to_event, checks_to_events, first_failure_event


class DiagnosticsTest(unittest.TestCase):
    def test_check_result_to_event_extracts_fix_and_docs(self) -> None:
        event = check_result_to_event(
            (
                "FAIL",
                "Python module: scservo_sdk",
                "ModuleNotFoundError. Fix: pip install feetech-servo-sdk",
            )
        )
        self.assertEqual(event.level, "FAIL")
        self.assertTrue(event.code.startswith("ENV-"))
        self.assertEqual(event.fix, "pip install feetech-servo-sdk")
        self.assertIn("docs/error-catalog.md#", event.docs_ref)

    def test_quick_action_mapping_for_eval_prefix(self) -> None:
        event = check_result_to_event(
            (
                "FAIL",
                "Eval dataset naming",
                "Eval dataset repo must begin with 'eval_'. Suggested quick fix: alice/eval_demo_2",
            )
        )
        self.assertEqual(event.quick_action_id, "fix_eval_prefix")
        self.assertEqual((event.context or {}).get("suggested_eval_repo_id"), "alice/eval_demo_2")

    def test_quick_action_mapping_for_training_fps(self) -> None:
        event = check_result_to_event(
            (
                "FAIL",
                "Training vs deploy FPS",
                "model trained at 15 Hz but camera_fps=30; mismatch",
            )
        )
        self.assertEqual(event.quick_action_id, "fix_camera_fps")
        self.assertEqual((event.context or {}).get("suggested_fps"), 15)

    def test_model_payload_candidates_maps_to_quick_action(self) -> None:
        event = check_result_to_event(
            (
                "WARN",
                "Model payload candidates",
                "/tmp/model_a, /tmp/model_b",
            )
        )
        self.assertEqual(event.quick_action_id, "fix_model_payload")
        self.assertEqual((event.context or {}).get("model_candidate"), "/tmp/model_a")

    def test_first_failure_event_returns_first_fail(self) -> None:
        events = checks_to_events(
            [
                ("PASS", "A", "ok"),
                ("FAIL", "B", "bad"),
                ("FAIL", "C", "also bad"),
            ]
        )
        first = first_failure_event(events)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.name, "B")


if __name__ == "__main__":
    unittest.main()
