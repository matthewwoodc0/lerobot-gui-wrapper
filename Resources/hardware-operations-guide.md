# Hardware Operations Guide

This guide covers the day-to-day hardware workflows added on top of the existing record, deploy, train, history, and visualizer surfaces.

## What Is New

- `Replay` replays recorded dataset episodes on hardware with command review, preflight, live output, cancel, and saved run artifacts.
- `Motor Setup` handles first-time servo bring-up with port selection, port scan, editable command review, preflight, live output, cancel, and result logging.
- Named rigs let one machine keep multiple saved hardware states and switch between them quickly.
- `Queue` runs small local sequential recipes without introducing a second scheduler.

## Replay Workflow

Ways to launch replay:
- Open the dedicated `Replay` section.
- In `Visualizer`, browse a dataset and click `Replay on Hardware`.
- In `History`, select a relevant run and click `Replay Selected`.

Replay behavior:
- Resolves the configured LeRobot replay entrypoint if one exists.
- Builds the command from dataset repo id, local dataset path/root when available, episode index, follower robot defaults, and calibration info.
- Uses discovered local episodes first and keeps a manual episode fallback when discovery is incomplete or unavailable.
- Shows a replay readiness summary at the point of action so dataset presence, episode availability, follower config, and compatibility warnings are visible before launch.
- Shows an editable command dialog before launch.
- Runs preflight checks against local dataset availability and episode presence.
- Streams live output through the normal run controller.
- Supports cancel.
- Saves artifacts in normal run history with dataset path and replay episode context.

If replay is unavailable:
- the UI explains that no replay entrypoint was detected
- no custom replay backend is invented as a fallback

## Motor Setup Workflow

Use `Motor Setup` for:
- first-time servo bring-up
- confirming the right port before record/deploy/teleop
- runtime-supported ID or baudrate changes

What the page does:
- lets you pick `Follower` or `Leader`
- loads the matching robot type, port, and robot id defaults
- supports `Scan Robot Ports`
- shows an editable command dialog
- runs a preflight summary before launch
- streams live output and supports cancel
- writes normal run history artifacts

Result handling:
- successful bring-up writes the selected role, port, robot type, and id context into artifacts
- successful runs feed updated port/id/type values back into config state
- success output now summarizes which config fields changed, whether ID/baudrate changes were actually applied by upstream flags, and whether the next step should be `Save Rig` or `Open Teleop`
- if the active config now differs from the active named-rig snapshot, the UI shows a non-blocking reminder to update that rig snapshot

Fallback behavior:
- if a dedicated setup entrypoint is missing but calibration exists, the page falls back to calibration-oriented bring-up
- in that mode the UI warns that new ID / baudrate changes may not actually be applied upstream

## Named Rigs

Named rigs are lightweight saved config snapshots for multi-rig labs using one machine.

Where to use them:
- `Config` has `Save Rig`, `Switch Rig`, and `Delete Rig`
- the main header shows the active rig and exposes quick `Save Rig` / `Switch Rig` controls

What is saved:
- hardware-facing config such as ports, robot ids/types, paths, and related runtime defaults
- UI-only keys are excluded from rig snapshots

Safety behavior:
- rig switching is blocked while a run is active
- rig switching is blocked while the local workflow queue still has queued or running work

## Local Queue / Recipes

Queue scope is intentionally small:
- `Record -> Upload`
- `Train -> Sim Eval`
- `Train -> Deploy Eval`

What the queue guarantees:
- sequential execution on the same machine
- shared use of the normal run controller
- normal history/artifact writing for each step
- workflow linkage through queue id, recipe type, step label, and previous run id metadata
- queue state persistence at `runs_dir/queue_state.json`
- restart recovery that marks any previously running item as `interrupted`
- explicit operator actions for `Resume Pending`, `Retry Interrupted Step`, and `Clear Finished/Interrupted`

Restart behavior:
- queued work restored from disk never auto-resumes
- interrupted work never auto-resumes
- `Resume Pending` only resumes clean queued items restored from disk
- `Retry Interrupted Step` reruns the current interrupted step from the beginning

What it does not try to do:
- cluster scheduling
- distributed execution
- simultaneous multi-robot orchestration

## Recommended Lab Flow

1. Save each bench or robot as a named rig.
2. Switch to the right rig before touching hardware.
3. Run `Motor Setup` for first-time bring-up or port confirmation.
4. Use `Record` or `Deploy` as usual.
5. Use `Replay` after recording to verify data quality directly on hardware.
6. Queue follow-up work when you want local sequential automation without babysitting each handoff.
