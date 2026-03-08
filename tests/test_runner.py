from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app import runner as runner_module
from robot_pipeline_app.runner import (
    kill_process_tree,
    popen_session_kwargs,
    run_command,
    run_process_streaming,
    terminate_process_tree,
)


class _FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


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

    def test_run_process_streaming_nonpty_suppresses_carriage_return_progress_spam(self) -> None:
        lines: list[str] = []
        errors: list[Exception] = []
        return_codes: list[int] = []
        script = (
            "import sys\n"
            "for i in range(200):\n"
            "    sys.stdout.write(f'progress {i}\\r')\n"
            "    sys.stdout.flush()\n"
            "print('done')\n"
        )

        thread = run_process_streaming(
            cmd=[sys.executable, "-u", "-c", script],
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
        self.assertIn("done", lines)
        self.assertTrue(any("suppressed" in line.lower() for line in lines))
        self.assertFalse(any("progress 199" in line for line in lines))

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

    @unittest.skipUnless(sys.platform != "win32" and runner_module.pty is not None, "PTY cancel test requires posix PTY support")
    def test_run_process_streaming_pty_cancel_terminates_process_tree(self) -> None:
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
            use_pty=True,
        )
        thread.join(timeout=6)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(return_codes)
        self.assertNotEqual(return_codes[0], 0)
        self.assertTrue(any("sigterm" in line.lower() for line in lines))

    @patch("robot_pipeline_app.runner.os.name", "posix")
    def test_popen_session_kwargs_uses_new_session_on_posix(self) -> None:
        self.assertEqual(popen_session_kwargs(), {"start_new_session": True})

    @patch("robot_pipeline_app.runner.os.name", "nt")
    def test_popen_session_kwargs_empty_on_non_posix(self) -> None:
        self.assertEqual(popen_session_kwargs(), {})

    @patch("robot_pipeline_app.runner.os.killpg")
    @patch("robot_pipeline_app.runner.os.getpgid", return_value=9001)
    @patch("robot_pipeline_app.runner.os.name", "posix")
    def test_terminate_process_tree_uses_sigterm_on_process_group(self, getpgid: object, killpg: object) -> None:
        lines: list[str] = []
        process = _FakeProcess(pid=2222)
        terminate_process_tree(process, lines.append, reason="Cancel requested.")
        getpgid.assert_called_once_with(2222)  # type: ignore[attr-defined]
        killpg.assert_called_once()  # type: ignore[attr-defined]
        args = killpg.call_args.args  # type: ignore[attr-defined]
        self.assertEqual(args[0], 9001)
        self.assertEqual(int(args[1]), 15)  # SIGTERM
        self.assertFalse(process.terminated)
        self.assertTrue(any("SIGTERM" in line for line in lines))
        self.assertTrue(any("process group" in line.lower() for line in lines))

    @patch("robot_pipeline_app.runner.os.name", "nt")
    def test_kill_process_tree_falls_back_to_parent_kill_on_non_posix(self) -> None:
        lines: list[str] = []
        process = _FakeProcess(pid=3333)
        kill_process_tree(process, lines.append, reason="Cancel timeout reached.")
        self.assertTrue(process.killed)
        self.assertTrue(any("kill to process" in line.lower() for line in lines))

    @patch("robot_pipeline_app.runner.subprocess.run")
    def test_run_command_uses_argv_without_shell(self, mocked_run: object) -> None:
        run_command(["python3", "-V"], cwd=Path("/tmp"), capture_output=True)
        mocked_run.assert_called_once()  # type: ignore[attr-defined]
        args = mocked_run.call_args.args  # type: ignore[attr-defined]
        kwargs = mocked_run.call_args.kwargs  # type: ignore[attr-defined]
        self.assertEqual(args[0], ["python3", "-V"])
        self.assertNotIn("shell", kwargs)


if __name__ == "__main__":
    unittest.main()
