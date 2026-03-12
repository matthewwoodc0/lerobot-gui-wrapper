# LeRobot GUI Wrapper

Qt-only desktop wrapper for local LeRobot record, deploy, teleop, train, config, history, and visualizer workflows.

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
- `Deploy`
- `Teleop`
- `Train`
- `Config`
- `Visualizer`
- `History`

The legacy alternate GUI path has been removed. The app now ships and runs through the Qt shell only.

## Notes

- If Qt imports fail, verify the active environment can import `PySide6`.
- If camera previews or video tiles fail, verify the active environment can import `cv2`.
- Generated LeRobot commands and compatibility/help probes use the configured `lerobot_venv_dir` runtime when that interpreter exists; otherwise they fall back to the wrapper's current Python.
- Training resume is only accepted when the detected LeRobot train entrypoint exposes a real checkpoint/config-path flag (for example `--config_path`). The UI will now block unsupported resume requests instead of silently sending `--resume=true`.
- In headless sandboxes, Qt offscreen bootstrap is smoke-checked before creating `QApplication` so test runs skip cleanly instead of aborting the interpreter.
