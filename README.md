# LeRobot GUI Wrapper

Qt-only desktop wrapper for local LeRobot record, deploy, teleop, config, history, and visualizer workflows.

## Requirements

- Python environment with the project dependencies installed
- `PySide6`
- OpenCV (`cv2`) for camera previews and in-app video rendering

## Launch

```bash
python3 -m robot_pipeline_app gui
```

`gui` and `gui-qt` both launch the same Qt application.

## Current UI Surface

- `Record`
- `Deploy`
- `Teleop`
- `Config`
- `Visualizer`
- `History`

The legacy alternate GUI path has been removed. The app now ships and runs through the Qt shell only.

## Notes

- If Qt imports fail, verify the active environment can import `PySide6`.
- If camera previews or video tiles fail, verify the active environment can import `cv2`.
- In headless sandboxes, Qt may still fail to bootstrap a platform plugin even when `PySide6` is installed.
