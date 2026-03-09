from __future__ import annotations

import unittest
from typing import Callable

from robot_pipeline_app.history_utils import (
    HISTORY_MODE_VALUES,
    _build_history_refresh_payload_from_runs,
    _command_from_item,
    _derive_status,
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

if __name__ == "__main__":
    unittest.main()
