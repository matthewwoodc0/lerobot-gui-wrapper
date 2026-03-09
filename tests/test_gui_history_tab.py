from __future__ import annotations

import unittest
from typing import Callable

from robot_pipeline_app.gui_history_tab import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _cancel_debounce_job,
    _command_from_item,
    _derive_status,
    _normalize_deploy_episode_outcomes,
    _parse_tags_csv,
    _schedule_debounce_job,
)


class GuiHistoryTabHelpersTest(unittest.TestCase):
    def test_derive_status_prefers_explicit_status(self) -> None:
        self.assertEqual(_derive_status({"status": "success", "exit_code": 1}), "success")

    def test_derive_status_from_exit_code_and_cancel(self) -> None:
        self.assertEqual(_derive_status({"canceled": True, "exit_code": None}), "canceled")
        self.assertEqual(_derive_status({"exit_code": 0}), "success")
        self.assertEqual(_derive_status({"exit_code": 2}), "failed")

    def test_command_from_item_prefers_command_argv(self) -> None:
        cmd, error = _command_from_item({"command_argv": ["python3", "-V"], "command": "ignored"})
        self.assertIsNone(error)
        self.assertEqual(cmd, ["python3", "-V"])

    def test_command_from_item_parses_legacy_command(self) -> None:
        cmd, error = _command_from_item({"command": "python3 -m unittest"})
        self.assertIsNone(error)
        self.assertEqual(cmd, ["python3", "-m", "unittest"])

    def test_command_from_item_handles_missing_command(self) -> None:
        cmd, error = _command_from_item({})
        self.assertIsNone(cmd)
        self.assertIsNotNone(error)

    def test_history_mode_values_hide_training_modes(self) -> None:
        self.assertNotIn("train_sync", HISTORY_MODE_VALUES)
        self.assertNotIn("train_launch", HISTORY_MODE_VALUES)
        self.assertNotIn("train_attach", HISTORY_MODE_VALUES)

    def test_parse_tags_csv_dedupes_and_trims(self) -> None:
        tags = _parse_tags_csv(" vertical, horizontal ,vertical ,, ")
        self.assertEqual(tags, ["vertical", "horizontal"])

    def test_normalize_deploy_episode_outcomes_counts_success_failed_and_pending(self) -> None:
        summary = _normalize_deploy_episode_outcomes(
            {
                "total_episodes": 4,
                "episode_outcomes": [
                    {"episode": 1, "result": "success", "tags": ["a"]},
                    {"episode": 2, "result": "failed", "tags": ["b"], "note": "dropped"},
                    {"episode": 3, "result": "pending", "tags": ["c"], "note": "interesting behavior"},
                ],
            }
        )
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["rated_count"], 2)
        self.assertEqual(summary["unmarked_count"], 2)
        self.assertEqual(summary["unrated_count"], 2)
        self.assertEqual(summary["tags"], ["a", "b", "c"])

    def test_build_history_refresh_payload_filters_and_dedupes_rows(self) -> None:
        runs = [
            {
                "run_id": "same",
                "mode": "deploy",
                "status": "success",
                "started_at_iso": "2026-02-25T10:00:00",
                "duration_s": 2.1,
                "dataset_repo_id": "alice/eval_a",
                "command": "python deploy --dataset alice/eval_a",
            },
            {
                "run_id": "same",
                "mode": "deploy",
                "status": "failed",
                "started_at_iso": "2026-02-25T10:02:00",
                "duration_s": 2.2,
                "model_path": "/tmp/model",
                "command": "python deploy --model /tmp/model",
            },
            {
                "run_id": "record-1",
                "mode": "record",
                "status": "success",
                "started_at_iso": "2026-02-25T10:04:00",
                "duration_s": 3.5,
                "dataset_repo_id": "alice/demo_a",
                "command": "python record --dataset alice/demo_a",
            },
        ]
        payload = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=3,
            mode_filter="deploy",
            status_filter="all",
            query="model",
        )

        rows = payload["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["iid"], "same")
        self.assertEqual(rows[0]["values"][2], "deploy")
        self.assertEqual(payload["warning_count"], 3)
        self.assertEqual(payload["stats"]["total"], 1)
        self.assertEqual(payload["stats"]["failed"], 1)
        self.assertEqual(payload["stats"]["success"], 0)
        payload_dedup = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=0,
            mode_filter="deploy",
            status_filter="all",
            query="",
        )
        self.assertEqual([row["iid"] for row in payload_dedup["rows"]], ["same", "same_1"])

    def test_build_history_refresh_payload_hides_training_rows(self) -> None:
        payload = _build_history_refresh_payload_from_runs(
            runs=[
                {
                    "run_id": "train-1",
                    "mode": "train_launch",
                    "status": "success",
                    "started_at_iso": "2026-02-25T12:00:00",
                    "duration_s": 4.0,
                    "command": "python train.py",
                },
                {
                    "run_id": "deploy-1",
                    "mode": "deploy",
                    "status": "success",
                    "started_at_iso": "2026-02-25T12:02:00",
                    "duration_s": 1.0,
                    "command": "python deploy.py",
                },
            ],
            warning_count=0,
            mode_filter="all",
            status_filter="all",
            query="",
        )

        self.assertEqual([row["iid"] for row in payload["rows"]], ["deploy-1"])

    def test_build_history_refresh_payload_query_matches_command_and_hint(self) -> None:
        runs = [
            {
                "run_id": "deploy-1",
                "mode": "deploy",
                "status": "success",
                "started_at_iso": "2026-02-25T11:00:00",
                "duration_s": 1.5,
                "dataset_repo_id": "alice/eval_pick_place",
                "command": "python deploy --dataset alice/eval_pick_place",
            },
            {
                "run_id": "record-1",
                "mode": "record",
                "status": "success",
                "started_at_iso": "2026-02-25T11:02:00",
                "duration_s": 2.5,
                "dataset_repo_id": "alice/demo_pick_place",
                "command": "python record --dataset alice/demo_pick_place",
            },
        ]
        payload = _build_history_refresh_payload_from_runs(
            runs=runs,
            warning_count=0,
            mode_filter="all",
            status_filter="success",
            query="eval_pick_place",
        )

        self.assertEqual([row["iid"] for row in payload["rows"]], ["deploy-1"])
        self.assertEqual(payload["stats"]["success"], 1)
        self.assertEqual(payload["stats"]["failed"], 0)

    def test_debounce_helpers_cancel_previous_and_run_latest(self) -> None:
        class _FakeRoot:
            def __init__(self) -> None:
                self.jobs: dict[str, Callable[[], None]] = {}
                self.cancelled: list[str] = []
                self._counter = 0

            def after(self, _delay: int, callback):
                self._counter += 1
                job_id = f"job-{self._counter}"
                self.jobs[job_id] = callback
                return job_id

            def after_cancel(self, job_id: str) -> None:
                self.cancelled.append(job_id)
                self.jobs.pop(job_id, None)

        root = _FakeRoot()
        state: dict[str, object] = {"id": None}
        fired: list[str] = []

        first_id = _schedule_debounce_job(root=root, job_state=state, callback=lambda: fired.append("first"), delay_ms=220)
        second_id = _schedule_debounce_job(root=root, job_state=state, callback=lambda: fired.append("second"), delay_ms=220)

        self.assertNotEqual(first_id, second_id)
        self.assertIn(first_id, root.cancelled)
        self.assertEqual(state["id"], second_id)

        root.jobs[str(second_id)]()
        self.assertEqual(fired, ["second"])
        self.assertIsNone(state["id"])

        _cancel_debounce_job(root, state, "id")
        self.assertIsNone(state["id"])


if __name__ == "__main__":
    unittest.main()
