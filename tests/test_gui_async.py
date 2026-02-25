import unittest

from robot_pipeline_app.gui_async import UiBackgroundJobs


class _FakeRoot:
    def after(self, _ms, callback):
        callback()
        return "after-id"


class GuiAsyncTests(unittest.TestCase):
    def test_submit_applies_latest_result_only(self):
        root = _FakeRoot()
        jobs = UiBackgroundJobs(root, max_workers=1)
        self.addCleanup(jobs.shutdown)

        applied: list[int] = []
        stale: list[bool] = []

        first = jobs.bump("k")
        jobs.submit(
            "k",
            lambda: 1,
            on_success=lambda value: applied.append(value),
            on_complete=lambda is_stale: stale.append(is_stale),
        )
        self.assertGreater(first, 0)

        self.assertEqual(applied, [1])
        self.assertEqual(stale, [False])

    def test_is_current_tracks_versions(self):
        jobs = UiBackgroundJobs(_FakeRoot(), max_workers=1)
        self.addCleanup(jobs.shutdown)

        v1 = jobs.bump("source")
        self.assertTrue(jobs.is_current("source", v1))
        v2 = jobs.bump("source")
        self.assertFalse(jobs.is_current("source", v1))
        self.assertTrue(jobs.is_current("source", v2))


if __name__ == "__main__":
    unittest.main()
