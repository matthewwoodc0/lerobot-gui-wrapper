# LeRobot GUI Wrapper

Qt-only desktop wrapper for local LeRobot record, replay, deploy, teleop, motor setup, train, queue, experiments, config, history, and visualizer workflows.

## Requirements

- Python 3.12+ environment with the project dependencies installed
- `PySide6`
- OpenCV (`cv2`) for camera previews and in-app video rendering

LeRobot `0.5.x` is the validated current track for this wrapper. `0.4.x` remains the validated N-1 track.

## Launch

```bash
python3 -m robot_pipeline_app gui
```

`gui` and `gui-qt` both launch the same Qt application.

## Current UI Surface

- `Record`
- `Replay`
- `Deploy`
- `Teleop`
- `Motor Setup`
- `Train`
- `Queue`
- `Experiments`
- `Config`
- `Visualizer`
- `History`

The legacy alternate GUI path has been removed. The app now ships and runs through the Qt shell only.

## Notes

- If Qt imports fail, verify the active environment can import `PySide6`.
- If camera previews or video tiles fail, verify the active environment can import `cv2`.
- `Visualizer` is now the research workspace surface: local assets plus HF browse/search, dataset QA, in-context compatibility warnings, sync-to-local roots, and lineage links.
- `History` now exposes lineage links and compatibility context alongside rerun and deploy-note editing.
- `History` and `Visualizer` can launch hardware replay directly against local dataset episodes when the configured LeRobot runtime exposes a replay entrypoint.
- `Config` now supports GUI profile import/export, portable lab presets, and named rig save/switch flows that update robot defaults, camera schema, rename-map hints, and setup guidance.
- `Motor Setup` handles first-time servo bring-up from the GUI with port scan, command review, preflight, live output, and result logging.
- `Queue` stays intentionally local and sequential: `record -> upload`, `train -> sim-eval`, and `train -> deploy eval` all reuse the normal run controller and artifact history.
- Rig switching is visible in the main header and blocked while an active run or queued local workflow is in progress.
- Generated LeRobot commands and compatibility/help probes use the configured `lerobot_venv_dir` runtime when that interpreter exists; otherwise they fall back to the wrapper's current Python.
- Replay and motor setup remain upstream-entrypoint-driven. If the configured LeRobot runtime does not expose those commands, the UI explains that clearly and degrades without inventing a second hardware backend.
- Training resume is only accepted when the detected LeRobot train entrypoint exposes a real checkpoint/config-path flag (for example `--config_path`). The UI will now block unsupported resume requests instead of silently sending `--resume=true`.
- `Experiments` turns saved train, deploy, and sim-eval runs into one comparison surface with checkpoint discovery, parsed metrics, deploy/sim-eval handoff, and optional WandB links.
- WandB remains optional. Local experiment views still work without it; when a run exposes WandB metadata and credentials are available, the app can add deep links plus remote summary/config context.
- Simulation eval is compatibility-driven. The wrapper probes the installed LeRobot runtime for a supported eval entrypoint and its `--help` flags before enabling checkpoint-launched sim-eval workflows.
- In headless sandboxes, Qt offscreen bootstrap is smoke-checked before creating `QApplication` so test runs skip cleanly instead of aborting the interpreter.

## Guides

- [Hardware Operations Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/hardware-operations-guide.md)
- [Config Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/config-tab-guide.md)
- [History Tab Guide](/Users/matthewwoodcock/Desktop/Projects/LeRobot%20GUI%20Wrapper/Resources/history-tab-guide.md)
