# Teleop Tab Guide

This guide explains the lightweight `Teleop` tab, including command generation, preflight checks, and first-time hardware bring-up.

## What This Tab Is For

Use `Teleop` when you want to quickly launch teleoperation with minimal setup:
- follower serial port
- leader serial port
- follower/leader robot IDs
- a fast real-hardware check before trying Record or Deploy

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

## 3) Teleop Camera Workspace

The `Teleop` tab includes the shared camera workspace used across the hardware pages.

Use it to:
- scan integer camera ports
- refresh live previews
- assign detected ports to configured camera names
- confirm the current runtime camera mapping before you switch to Record or Deploy

## 4) Teleop Helper

The helper window is runtime-focused rather than episode-focused.

It shows:
- elapsed session time
- live console output from the teleop process
- connection/readiness status
- an `End Teleop` action for intentionally stopping the session

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
6. The **Teleop Helper** opens with elapsed runtime, live logs, and readiness status.
7. Use `End Teleop` when you want to stop the live session intentionally.
8. Status updates reflect completed/ended/canceled/failed result.

## Example Workflow

1. Open `Teleop`.
2. Click `Scan Robot Ports`.
3. Enter or verify follower/leader serial ports.
4. Set robot IDs if needed.
5. Click `Preview Command`.
6. Click `Run Teleop`.
7. Watch the helper for readiness and console output.
8. If preflight flags calibration problems, open the terminal view in the output panel and run calibration there.
9. Use the Teleop camera workspace or switch to `Record` to verify camera routing.

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

**These are warnings, not errors.** Teleop can still work with them present.

Use them as an identity hint:
- confirm the ports shown by `Scan Robot Ports` match the physical follower/leader arms
- prefer stable `/dev/serial/by-id/...` device names on Linux when possible
- treat successful Teleop startup as the real acceptance test

## Notes

- This tab is intentionally simple.
- Most hardware/environment failures are surfaced in preflight before launch.
- If teleop starts but the arms do not respond, check the terminal log in the output panel. A calibration prompt may be waiting for input.
- For a new machine, use Teleop first. Do not start with Record or Deploy.
- Teleop now includes the shared camera workspace, but Record and Deploy still give the fuller camera validation flow.
