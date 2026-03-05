# First-Time Setup Guide

Use this guide on a new machine, a new robot pair, or after USB ports / calibration files have changed.

Goal:
- confirm the Python environment is valid
- confirm follower/leader ports are correct
- make sure calibration files exist and match the connected robots
- verify teleop works before trying to record or deploy
- verify camera indices separately before recording

## Recommended Order

1. Launch the app from an active LeRobot environment:

```bash
python3 robot_pipeline.py gui
```

2. Open `Config`.
3. Click `Run Setup Check`.
4. Fix any `FAIL` items before continuing.
5. Save the core machine defaults:
   - `lerobot_dir`
   - `follower_port`
   - `leader_port`
   - `follower_robot_id`
   - `leader_robot_id`
   - `camera_laptop_index`
   - `camera_phone_index`
   - `record_data_dir`
   - `deploy_data_dir`
   - `trained_models_dir`
6. Click `Run Doctor`.
7. Resolve `FAIL` items there too.

## Teleop First

Use `Teleop` as the first real hardware test. It is the fastest way to prove that:
- ports are correct
- IDs line up with the connected arms
- calibration is usable
- the app can actually start a LeRobot session

### Step 1: Find the correct ports

1. Open `Teleop`.
2. Click `Scan Robot Ports`.
3. If the app suggests follower/leader assignments, apply them if they match your physical setup.
4. If not, identify ports manually by unplugging one arm at a time and scanning again.

Tip:
- On Linux, prefer `/dev/serial/by-id/...` paths when available.
- On macOS, expect `/dev/tty.*` or `/dev/cu.*` device names.

### Step 2: Check robot IDs

Set `Follower robot id` and `Leader robot id` to match the calibration/profile you intend to use.

If you already have calibration JSON files and their filenames match the real robot IDs, the app can often infer IDs from those filenames. If you are unsure, enter the IDs explicitly.

### Step 3: Run Teleop

1. Click `Preview Command` if you want to inspect the generated command.
2. Click `Run Teleop`.
3. Review the preflight dialog.

If teleop starts successfully, your environment, ports, and core robot setup are in good shape.

## If Teleop Fails on Calibration

Calibration failures are common on:
- first boot on a new machine
- remapped USB ports
- swapped follower/leader arms
- new robot IDs

When preflight shows a calibration-related `FAIL`:

1. Open the terminal view in the bottom output panel.
2. Run follower and leader calibration commands using the exact values from Config:

```bash
python3 -m lerobot.calibrate --robot.type=<follower_robot_type> --robot.port=<follower_port> --robot.id=<follower_robot_id>
python3 -m lerobot.calibrate --robot.type=<leader_robot_type>   --robot.port=<leader_port>   --robot.id=<leader_robot_id>
```

3. Accept prompts in the terminal view as needed.
4. Save the resulting calibration paths in `Config` if you want explicit overrides.
5. Run `Teleop` again.

You should not move on to `Record` or `Deploy` until teleop can start cleanly.

## Verify Cameras Separately

`Teleop` verifies robot communication. Camera verification is better done in `Record` or `Deploy`, where the preview UI lives.

1. Open `Record`.
2. Use `Scan Camera Ports`.
3. Confirm the laptop/phone mapping is correct.
4. If needed, assign the correct roles and let the app save them.
5. Refresh preview and confirm both expected views are visible.

If camera preview is unreliable, fix that before attempting a real recording run.

## First Short Record

After teleop and camera checks both work:

1. Stay in `Record`.
2. Use a small test run:
   - `2` episodes
   - short duration
   - simple task string
3. Run preflight.
4. Record a short sample dataset.

This confirms the full teleop-plus-camera path is working, not just teleop by itself.

## First Deploy / Eval

Only try `Deploy` after:
- teleop starts cleanly
- camera mapping is confirmed
- at least one short record succeeds

Then:

1. Open `Deploy`.
2. Select a valid local model.
3. Use a small eval dataset and short episode count.
4. Run preflight.
5. Start the eval run.

## Minimum “Ready” Definition

Treat a machine as ready for normal use only after all of these are true:
- `Run Setup Check` has no unresolved `FAIL`
- `Run Doctor` has no unresolved `FAIL`
- `Teleop` starts successfully
- calibration files exist and match the intended robot IDs
- camera preview is correct in `Record`
- one short record completes

At that point, the machine is in a good state for community use.
