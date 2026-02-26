# Teleop Tab Guide

This guide explains the lightweight `Teleop` tab, including command generation, preflight checks, and camera preview behavior.

## What This Tab Is For

Use `Teleop` when you want to quickly launch teleoperation with minimal setup:
- follower serial port
- leader serial port
- follower/leader robot IDs
- camera scan/preview tools

## Main UI Areas

## 1) Teleop Setup

- `Follower port`
- `Leader port`
- `Follower robot id` (default `red4`)
- `Leader robot id` (default `white`)

Buttons:
- `Preview Command`
- `Run Teleop`

## 2) Teleop Snapshot

Shows live summary of:
- selected follower/leader ports
- follower/leader IDs
- camera mapping from config (`camera_laptop_index`, `camera_phone_index`)

## 3) Teleop Camera Preview

Supports:
- `Scan Camera Ports`
- `Refresh Camera Preview`
- assigning detected ports to laptop/phone roles

Role updates persist into config camera indices.

## Command Behavior

Teleop command uses module auto-detection:
- `lerobot.teleoperate`
- `lerobot.scripts.lerobot_teleoperate`
- legacy fallback: `lerobot.scripts.control_robot`

For modern entrypoints, command includes:
- `--robot.cameras={}`

For legacy fallback, command includes full camera JSON and optional control flags.

Typical command shape:

```bash
python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras={} \
  --robot.id=red4 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=white
```

## What Happens When You Click Run Teleop

1. App validates required ports.
2. App builds teleop command with selected IDs.
3. App saves changed follower/leader ports to config.
4. Teleop preflight runs (same common environment/ports/camera checks).
5. Command runs in `lerobot_dir`.
6. The **Teleop Controls** popout appears. A **Starting up…** progress bar counts up for up to ~12 seconds while the robot arm initializes and establishes communication. This is normal — the bar disappears automatically once the teleop process signals it is ready.
7. Status updates reflect completed/canceled/failed result.

## Example Workflow

1. Open `Teleop`.
2. Enter or verify follower/leader serial ports.
3. Set robot IDs if needed.
4. Optional: scan camera ports and verify laptop/phone assignments.
5. Click `Preview Command`.
6. Click `Run Teleop`.
7. If preflight is clean or acceptable, continue and teleoperate.

## What You Might See

Validation:
- `Validation Error: Follower port is required.`
- `Validation Error: Leader port is required.`

Preflight dialog title:
- `Teleop Preflight`

Log/status examples:
- `Saved teleop connection defaults to config.`
- `Running teleop session...`
- `Teleop completed.`
- `Teleop failed.`

## Port Fingerprint Warnings

During preflight you may see warnings like:

```
[WARN] Follower port fingerprint: no baseline saved yet; scan/assign once to lock mapping.
[WARN] Follower port role inference: could not infer role from serial fingerprint text.
```

**What is a fingerprint?** It is the unique serial ID that a USB device exposes. The app reads it to verify the correct physical robot arm is on the correct port — even if port names change after a reboot.

**These are warnings, not errors.** Teleop will still work without them. To suppress them permanently:

1. Connect both arms.
2. Click **Scan Ports / Assign Roles** in the Teleop tab.
3. The app reads each device's serial ID and saves it as the baseline.

After saving once, preflight will confirm hardware identity on every future run.

## Notes

- This tab is intentionally simple.
- Most hardware/environment failures are surfaced in preflight before launch.
- If teleop starts but the arms do not respond, check the terminal log — a calibration prompt (press Enter to use defaults) may be waiting for input. After accepting once, calibration is saved and the prompt will not appear again.
