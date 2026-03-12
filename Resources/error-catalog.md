# Error Catalog

This catalog maps runtime and preflight diagnostics to actionable fixes.

## Taxonomy

- `ENV-*`: Python/virtualenv/module activation issues
- `SER-*`: serial port access/permission/device contention
- `CAM-*`: camera index/device/resolution issues
- `CAL-*`: calibration mismatch or missing calibration state
- `COMPAT-*`: wrapper/LeRobot compatibility or fallback behavior
- `CLI-*`: LeRobot flag or command interface mismatch
- `MODEL-*`: model payload or performance constraints
- `DATA-*`: dataset naming/storage/upload issues

## Current Fix Messages (Phase 1 Sync)

The following `Fix:` messages are emitted by diagnostics today and are intentionally mirrored here.

### ENV

- `Fix: install the missing package or activate the correct virtual environment.`
- `Fix: activate your env and relaunch the GUI from that shell.`
- `Fix: source {activate_script}  or conda activate <env> before launching GUI.`
- `Fix: {hint}`
- `Fix: {fix_hint}`

### SER

- `Fix: sudo usermod -a -G dialout $USER (then log out/in or run newgrp dialout).`
- `Fix: unplug/replug USB, close any other app using the ports, then re-scan and reapply follower/leader ports.`

### CAM

- `Fix: rescan cameras and reassign laptop/phone roles before rerunning.`
- `Fix: lower camera FPS or reassign to a camera that supports the requested mode.`

### CAL

- `Fix: pip install feetech-servo-sdk  (required for Feetech SO-100 arms)`
- `Fix: verify arm power and daisy-chain motor cables, then power-cycle the arm(s) for 5-10 seconds.`
- `Fix: confirm robot/teleop IDs and calibration files match the connected physical arms.`
- `Fix: rerun calibration for each arm if IDs or wiring changed since last successful run.`
- `Fix: keep calibration_dir + robot.id paired to the same arm profile, or rerun calibration for current hardware.`

### CLI

- `Fix: run the generated command with '--help' and align advanced overrides to supported options.`

### MODEL

- `Fix: lower camera FPS/resolution or use a smaller model checkpoint.`

### DATA

- `Fix: Check dataset repo id/path naming and write permissions, then rerun preflight.`

### COMPAT

- `Fix: run the relevant LeRobot command with --help and update flags to match your installed version.`
- `Fix: run preflight again after applying compatibility-related quick fixes.`

## How To Use This Catalog

1. Find the first failing diagnostic in preflight or runtime output.
2. Apply the matching fix above.
3. Re-run preflight before retrying the full workflow.
