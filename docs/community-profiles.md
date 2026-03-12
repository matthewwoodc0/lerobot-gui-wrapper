# Community Profiles

Community profiles stay user-portable; named rigs stay machine-local.

## Rules

- Put reusable robot defaults, paths that can be safely rewritten, and profile-level metadata in community profiles.
- Keep bench-specific ports, live queue state, and run-history state out of profiles.
- When rig handling changes, update this file and [`Resources/config-tab-guide.md`](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/config-tab-guide.md).

## Rigs vs Profiles

- Profiles are import/export artifacts for sharing defaults.
- Named rigs are saved snapshots of the active local hardware state.
- If motor setup changes the active config and that diverges from the active rig snapshot, save the rig again after confirming the hardware state is correct.
