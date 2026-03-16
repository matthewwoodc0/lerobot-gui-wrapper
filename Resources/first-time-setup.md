# First-Time Setup Walkthrough

Use this guide on a new machine, a new robot pair, or after USB ports or calibration files have changed. It walks every step from launching the app for the first time to completing a verified record run.

**Goal:** By the end, you have confirmed that your Python environment, hardware ports, calibration, cameras, and recording pipeline all work together.

**Time:** 15–30 minutes depending on how many things need fixing.

---

## Overview

The recommended order is:

1. Launch the app and open Config
2. Set core paths and fields
3. Run diagnostics (Setup Check and Doctor)
4. Verify hardware with Teleop
5. Verify cameras in Record
6. Complete a short test record run

Do not skip ahead. Teleop and cameras must both work before you attempt a full record or deploy run.

---

## Step 1 — Launch the App

From inside the wrapper folder with the conda environment active:

```bash
conda activate lerobot
python3 robot_pipeline.py gui
```

If the app doesn't start, see the platform-specific steps in [README.md](../README.md#platform-setup) for xcb/Qt errors on Linux.

---

## Step 2 — Open Config and Set Core Fields

Click the **Config** tab.

Click **Apply Path Defaults** first — this auto-fills `lerobot_venv_dir` and several data paths from whatever you set in `lerobot_dir`.

Then fill in the following fields. Each one matters.

### Required fields

| Field | What to set | Example |
|---|---|---|
| `lerobot_dir` | Path to your LeRobot checkout | `~/lerobot` |
| `lerobot_venv_dir` | LeRobot virtual environment | auto-filled after `lerobot_dir` |
| `follower_port` | Serial device for follower arm | `/dev/ttyACM1` or `/dev/cu.usbmodem1` |
| `leader_port` | Serial device for leader arm | `/dev/ttyACM0` or `/dev/cu.usbmodem0` |
| `follower_robot_id` | ID matching calibration profile | `red4` |
| `leader_robot_id` | ID matching calibration profile | `white` |
| `camera_laptop_index` | USB index for the laptop camera | `0`, `2`, or `4` |
| `camera_phone_index` | USB index for the phone camera | `2`, `4`, or `6` |
| `record_data_dir` | Local folder for saved datasets | `~/datasets` |
| `trained_models_dir` | Folder where trained models live | `~/lerobot/trained_models` |
| `hf_username` | Hugging Face username | `your_username` |

> **Finding your ports on macOS:** Plug in each arm one at a time and run `ls /dev/cu.*` in terminal before and after to identify which device appeared.

> **Finding your ports on Linux:** Run `ls /dev/serial/by-id/` for stable names, or `ls /dev/ttyACM*` for index-based names. Prefer the by-id paths — they don't change between reboots.

> **Finding camera indices:** The app can scan for you. See Step 5 below. For now, try `0` and `2` as defaults; you'll verify them in Step 5.

### Robot IDs

The robot ID must match the ID used when calibration files were created. Common IDs in the SO-101 community are `red4` (follower) and `white` (leader), but yours may differ. Check the filenames of any existing calibration JSON files in `~/.cache/huggingface/lerobot/calibration/` — the filename stem is the ID.

### Click Save Config

After filling in all fields, click **Save Config**. Fields are persisted to `~/.robot_config.json`.

---

## Step 3 — Run Diagnostics

### Run Setup Check

Click **Run Setup Check**.

This checks:
- whether `lerobot_dir` exists and contains a valid LeRobot checkout
- whether `lerobot_venv_dir` exists and the interpreter is functional
- whether LeRobot imports are available in that environment
- whether `ffmpeg` is accessible

**Expected output:** All checks pass or show `WARN`. Fix any `FAIL` items before continuing.

Common first-run failures and fixes:

| Error | Fix |
|---|---|
| `lerobot` module not found | Activate the correct conda env and ensure LeRobot is installed: `pip show lerobot` |
| `ffmpeg` not found | `conda install ffmpeg -y` |
| `feetech-servo-sdk` missing | `pip install feetech-servo-sdk` |
| `lerobot_dir` not found | Set the correct path and click Save Config |

### Run Doctor

Click **Run Doctor**.

Doctor runs a broader diagnostic: serial ports, camera access, calibration file presence, environment health, and compatibility probes.

**Expected output:** All items are `PASS` or `WARN`. Fix any `FAIL` items.

Common doctor failures:

| Error code | What it means | Fix |
|---|---|---|
| `SER-*` | Serial port inaccessible | On Linux: `sudo usermod -aG dialout $USER`, then log out/in |
| `CAL-*` | Calibration file missing or mismatched | Run calibration for the arm (see below) |
| `ENV-*` | Python or venv issue | Reactivate conda env; check `lerobot_venv_dir` points to the right place |
| `CAM-*` | Camera not accessible | Check camera is plugged in; try different indices |

See the [Error Catalog](./error-catalog.md) for the full list.

---

## Step 4 — Verify Hardware with Teleop

Open the **Teleop** tab.

Teleop is the fastest way to confirm the full hardware chain — ports, IDs, calibration, and communication — before you commit to a record session.

### Find and set ports

Click **Scan Robot Ports**. The app scans available serial devices and may suggest follower/leader assignments based on fingerprint matching. If the suggestions match your physical setup, apply them. If not, assign manually.

Verify the port assignments match your physical wiring. If in doubt: unplug one arm, scan again, and note which port disappeared — that's the unplugged arm's port.

### Set robot IDs

Confirm `Follower robot id` and `Leader robot id` match the IDs used during calibration.

### Click Run Teleop

1. Click **Preview Command** to review the generated command.
2. Click **Run Teleop**.
3. Review the preflight dialog. If there are warnings, read them. If there are failures, fix them first.
4. The Teleop Helper window opens.

**What success looks like:** The Teleop Helper shows a running session timer and live log output. Move the leader arm — the follower arm should mirror it within about one second.

**If the arms don't respond:** Check the live log in the Teleop Helper. If you see a calibration prompt waiting for input, that needs to be handled. See below.

---

## If Teleop Fails on Calibration

Calibration failures are common on:
- first boot on a new machine
- USB ports remapped after a reboot
- swapped follower/leader arms
- changed robot IDs

When preflight flags a calibration-related `FAIL`:

1. Open the terminal panel at the bottom of the app (or use a separate terminal in your conda env).
2. Run calibration for each arm using the **exact same** robot type, port, and ID values from your Config:

```bash
# Follower
python3 -m lerobot.calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=red4

# Leader
python3 -m lerobot.calibrate \
  --robot.type=so101_leader \
  --robot.port=/dev/ttyACM0 \
  --robot.id=white
```

3. Follow all prompts in the terminal. Calibration typically requires moving each joint through its range.
4. After calibration completes, close that terminal and return to the app.
5. Run **Run Doctor** again to confirm calibration is now detected.
6. Try **Run Teleop** again.

Do not proceed to Record or Deploy until Teleop starts cleanly and arms respond.

---

## Step 5 — Verify Cameras

Camera verification is done in the **Record** tab where the full camera preview UI lives.

1. Open the **Record** tab.
2. In the **Camera** section, click **Scan Camera Ports**.
3. The app lists detected integer USB camera indices.
4. If the detected camera doesn't match the expected role (laptop vs phone), click **Set Laptop** or **Set Phone** to assign it.
5. Click **Refresh Camera Preview**.
6. Confirm both camera views show what you expect.

**If a camera doesn't appear:**
- Verify the physical connection.
- Try a different USB port.
- On Linux, the USB index can shift depending on which devices are plugged in. Rescan after reconnecting.
- If preview is blank, try index `0`, `2`, `4`, or `6` until you find the active camera.

**If only one camera is showing:** Check that `camera_phone_index` in Config is set to a different value than `camera_laptop_index` and that the second camera is physically connected.

Click **Save Config** after assigning camera roles so the assignments persist.

---

## Step 6 — Record a Short Test Dataset

Still in the **Record** tab.

1. Leave the dataset name on its auto-managed value (e.g. `yourname_1`).
2. Set:
   - **Episodes:** `2`
   - **Episode time (seconds):** `10`
   - **Task description:** something short, like `test`
3. Click **Preview Command** to confirm the generated command looks right.
4. Click **Run Record**.
5. Review and accept the preflight dialog.
6. Complete two short episodes using the leader arm.
7. Wait for the run to complete.

**What success looks like:** The run completes, the app shows a success dialog, and a new dataset folder appears in your `record_data_dir`.

**If the run fails immediately:** Check the error in the run output panel. Common causes are missing calibration (go back to Step 4), camera not found (go back to Step 5), or write permission issues on `record_data_dir`.

---

## Minimum "Ready" Definition

The machine is ready for normal use when all of these are true:

- `Run Setup Check` has no unresolved `FAIL`
- `Run Doctor` has no unresolved `FAIL`
- Teleop starts successfully and arms respond
- Calibration files exist and match the active robot IDs
- Camera preview is correct in Record
- One short test record run completes

---

## After Setup — What's Next

With setup verified, here's where to go:

| What you want to do | Where to start |
|---|---|
| Collect training data | [Record Tab Guide](./record-tab-guide.md) |
| Run a trained model | [Deploy Tab Guide](./deploy-tab-guide.md) |
| Save this machine's hardware config | Config → Save Rig |
| Share this config with a teammate | [Community Profiles Guide](./community-profiles.md) |
| Set up a second robot on the same machine | Config → Save Rig, then set up the second robot and save as a second rig |

---

## Troubleshooting Reference

### App launches but Config tab is blank

The config file may be corrupted. Back up `~/.robot_config.json`, then delete it. Re-launch and reconfigure.

### Teleop starts but arms are stiff or don't mirror

Power-cycle both arms (unplug USB and power for 5–10 seconds). Re-run Doctor after reconnecting.

### Record run fails with "dataset already exists"

The auto-managed name detected a collision locally or on Hugging Face. Click the increment button next to the dataset name to advance to the next number, or clear the HF namespace by using a different `hf_username`.

### Camera preview flickers or goes blank

On Linux, this can be a USB bandwidth issue. Try using different USB ports (ideally on different controllers). Set a lower camera resolution in Config.

### Doctor passes but teleop immediately errors

This usually means the arms were powered on after the doctor check ran. Re-run Doctor after confirming both arms have power.
