from __future__ import annotations

import unittest

from robot_pipeline_app.run_state import (
    ProcessSessionState,
    command_has_explicit_calibration_dir,
    extract_calibration_prompt_id,
    is_saved_calibration_prompt,
    is_teleop_av1_decode_error,
    is_teleop_ready_line,
)


class _StdinRecorder:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: object) -> int:
        text = str(data)
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        return


class _ProcessStub:
    def __init__(self, stdin: _StdinRecorder | None = None) -> None:
        self.stdin = stdin
        self._poll_value: int | None = None

    def poll(self) -> int | None:
        return self._poll_value


class RunStateTests(unittest.TestCase):
    def test_teleop_output_markers_detect_expected_lines(self) -> None:
        self.assertTrue(is_teleop_av1_decode_error("Failed to get pixel format for AV1 stream"))
        self.assertTrue(is_teleop_ready_line("Teleop running and connected"))
        self.assertFalse(is_teleop_ready_line("calibrating camera"))

    def test_calibration_prompt_helpers_detect_prompt_and_id(self) -> None:
        prompt = (
            "Press ENTER to use provided calibration file associated with the id grey_follower, "
            "or type 'c' and press ENTER to run calibration: "
        )
        self.assertTrue(is_saved_calibration_prompt(prompt))
        self.assertEqual(extract_calibration_prompt_id(prompt), "grey_follower")

    def test_command_has_explicit_calibration_dir_checks_known_flags(self) -> None:
        self.assertTrue(command_has_explicit_calibration_dir(["python3", "--robot.calibration_dir=/tmp/cal"]))
        self.assertTrue(command_has_explicit_calibration_dir(["python3", "--teleop.calibration_dir=/tmp/cal"]))
        self.assertFalse(command_has_explicit_calibration_dir(["python3", "--dataset.repo_id=alice/demo"]))

    def test_process_session_state_tracks_running_and_input(self) -> None:
        state = ProcessSessionState()
        self.assertFalse(state.has_active_process())

        recorder = _StdinRecorder()
        process = _ProcessStub(stdin=recorder)
        state.mark_active()
        state.attach_process(process)

        self.assertTrue(state.has_active_process())
        ok, message = state.send_input("hello")
        self.assertTrue(ok)
        self.assertIn("Input sent", message)
        self.assertEqual(recorder.writes, ["hello"])

        ok_arrow, arrow_message = state.send_arrow_key("left")
        self.assertTrue(ok_arrow)
        self.assertIn("Reset episode", arrow_message)

    def test_process_session_state_returns_clear_message_without_active_process(self) -> None:
        state = ProcessSessionState()
        ok, message = state.send_input("x")
        self.assertFalse(ok)
        self.assertIn("No active", message)


if __name__ == "__main__":
    unittest.main()
