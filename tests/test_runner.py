from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

from robot_pipeline_app.runner import run_process_streaming


class RunnerStreamingTest(unittest.TestCase):
    def test_run_process_streaming_nonpty_captures_output(self) -> None:
        lines: list[str] = []
        errors: list[Exception] = []
        return_codes: list[int] = []

        thread = run_process_streaming(
            cmd=[sys.executable, "-c", "print('hello');print('world')"],
            cwd=Path("/tmp"),
            on_line=lines.append,
            on_complete=return_codes.append,
            on_start_error=errors.append,
            cancel_requested=lambda: False,
            use_pty=False,
        )
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(return_codes, [0])
        self.assertIn("hello", lines)
        self.assertIn("world", lines)

    def test_run_process_streaming_nonpty_cancel_terminates_process(self) -> None:
        lines: list[str] = []
        errors: list[Exception] = []
        return_codes: list[int] = []
        started = time.monotonic()

        def cancel_requested() -> bool:
            return time.monotonic() - started >= 0.2

        thread = run_process_streaming(
            cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=Path("/tmp"),
            on_line=lines.append,
            on_complete=return_codes.append,
            on_start_error=errors.append,
            cancel_requested=cancel_requested,
            use_pty=False,
        )
        thread.join(timeout=6)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(return_codes)
        self.assertNotEqual(return_codes[0], 0)
        self.assertTrue(any("graceful shutdown" in line.lower() for line in lines))


if __name__ == "__main__":
    unittest.main()
