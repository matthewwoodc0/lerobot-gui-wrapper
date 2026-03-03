from __future__ import annotations

import unittest

from robot_pipeline_app.serial_scan import format_robot_port_scan, suggest_follower_leader_ports


class SerialScanTest(unittest.TestCase):
    def test_suggest_follower_leader_prefers_index_convention(self) -> None:
        entries = [
            {"path": "/dev/ttyACM0"},
            {"path": "/dev/ttyACM1"},
        ]
        follower, leader = suggest_follower_leader_ports(entries)
        self.assertEqual(follower, "/dev/ttyACM1")
        self.assertEqual(leader, "/dev/ttyACM0")

    def test_suggest_respects_existing_when_valid(self) -> None:
        entries = [
            {"path": "/dev/ttyACM0"},
            {"path": "/dev/ttyACM1"},
        ]
        follower, leader = suggest_follower_leader_ports(
            entries,
            current_follower="/dev/ttyACM0",
            current_leader="/dev/ttyACM1",
        )
        self.assertEqual(follower, "/dev/ttyACM0")
        self.assertEqual(leader, "/dev/ttyACM1")

    def test_format_scan_handles_empty(self) -> None:
        self.assertIn("No candidate serial robot ports", format_robot_port_scan([]))


if __name__ == "__main__":
    unittest.main()
