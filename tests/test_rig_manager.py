from __future__ import annotations

import unittest

from robot_pipeline_app.rig_manager import (
    active_rig_name,
    apply_named_rig,
    build_rig_snapshot,
    delete_named_rig,
    list_named_rigs,
    save_named_rig,
)


class RigManagerTests(unittest.TestCase):
    def test_save_named_rig_captures_snapshot_without_ui_keys(self) -> None:
        config = {
            "follower_port": "/dev/ttyUSB0",
            "leader_port": "/dev/ttyUSB1",
            "hf_username": "alice",
            "ui_theme_mode": "dark",
        }

        updated = save_named_rig(config, name="Rig A", description="Bench robot")

        self.assertEqual(active_rig_name(updated), "Rig A")
        rigs = list_named_rigs(updated)
        self.assertEqual(len(rigs), 1)
        self.assertEqual(rigs[0]["description"], "Bench robot")
        snapshot = rigs[0]["snapshot"]
        self.assertEqual(snapshot["follower_port"], "/dev/ttyUSB0")
        self.assertEqual(snapshot["leader_port"], "/dev/ttyUSB1")
        self.assertEqual(snapshot["hf_username"], "alice")
        self.assertNotIn("ui_theme_mode", snapshot)
        self.assertNotIn("saved_rigs", snapshot)
        self.assertNotIn("active_rig_name", snapshot)

        rebuilt = build_rig_snapshot(updated)
        self.assertNotIn("ui_theme_mode", rebuilt)

    def test_apply_named_rig_restores_snapshot_and_preserves_ui_preferences(self) -> None:
        base = {
            "follower_port": "/dev/ttyUSB0",
            "leader_port": "/dev/ttyUSB1",
            "ui_sidebar_collapsed": True,
            "ui_theme_mode": "light",
        }
        with_rig = save_named_rig(base, name="Rig A")

        mutated = dict(with_rig)
        mutated["follower_port"] = "/dev/ttyUSB9"
        mutated["ui_theme_mode"] = "dark"

        updated, error = apply_named_rig(mutated, name="Rig A")

        self.assertIsNone(error)
        assert updated is not None
        self.assertEqual(updated["follower_port"], "/dev/ttyUSB0")
        self.assertEqual(updated["leader_port"], "/dev/ttyUSB1")
        self.assertEqual(updated["ui_sidebar_collapsed"], True)
        self.assertEqual(updated["ui_theme_mode"], "dark")
        self.assertEqual(active_rig_name(updated), "Rig A")

    def test_delete_named_rig_clears_active_name_when_needed(self) -> None:
        config = save_named_rig({"follower_port": "/dev/ttyUSB0"}, name="Rig A")

        updated = delete_named_rig(config, name="Rig A")

        self.assertEqual(list_named_rigs(updated), [])
        self.assertEqual(active_rig_name(updated), "")


if __name__ == "__main__":
    unittest.main()
