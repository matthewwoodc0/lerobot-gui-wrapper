from __future__ import annotations

import unittest

from robot_pipeline_app.background_jobs import LatestJobRunner


class BackgroundJobsTests(unittest.TestCase):
    def test_bump_and_is_current_track_versions(self) -> None:
        runner = LatestJobRunner(max_workers=1)
        self.addCleanup(runner.shutdown)

        v1 = runner.bump("datasets")
        self.assertTrue(runner.is_current("datasets", v1))

        v2 = runner.bump("datasets")
        self.assertFalse(runner.is_current("datasets", v1))
        self.assertTrue(runner.is_current("datasets", v2))

    def test_submit_returns_future_result_for_latest_job(self) -> None:
        runner = LatestJobRunner(max_workers=1)
        self.addCleanup(runner.shutdown)

        version, future = runner.submit("models", lambda: 7)

        self.assertTrue(runner.is_current("models", version))
        self.assertEqual(future.result(timeout=1.0), 7)


if __name__ == "__main__":
    unittest.main()
