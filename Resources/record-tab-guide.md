# Record Tab Guide

Use this tab to collect teleoperated demonstration datasets, optionally upload them to Hugging Face, and browse your local and remote dataset library.

---

## Table of Contents

1. [What This Tab Is For](#what-this-tab-is-for)
2. [Before You Record](#before-you-record)
3. [Full Walkthrough — First Recording Session](#full-walkthrough--first-recording-session)
4. [UI Reference](#ui-reference)
5. [Dataset Naming](#dataset-naming)
6. [Uploading to Hugging Face](#uploading-to-hugging-face)
7. [Dataset Browser](#dataset-browser)
8. [Command Shape](#command-shape)
9. [What You Might See](#what-you-might-see)
10. [Troubleshooting](#troubleshooting)

---

## What This Tab Is For

- Build and run a `lerobot_record` command for teleoperated data collection.
- Optionally upload the completed dataset to Hugging Face immediately after recording.
- Manually upload an existing local dataset folder.
- Browse local and Hugging Face datasets side by side.

---

## Before You Record

Record requires a working hardware setup. If you're starting from scratch, complete [First-Time Setup](./first-time-setup.md) first.

Minimum requirements before hitting Run Record:

- Teleop starts and arms respond (confirms ports, IDs, calibration)
- Camera preview shows correct views in this tab
- `record_data_dir` is writable and has enough disk space (~500MB per 20-episode dataset)
- `ffmpeg` is in PATH (required for video encoding)

---

## Full Walkthrough — First Recording Session

This section walks a complete recording session from opening the tab to having a dataset saved locally.

### 1. Open Record

Click the **Record** tab in the sidebar.

### 2. Set the dataset name

The dataset name field starts in **auto-managed mode**. It generates a name like `yourname_1` based on your `hf_username` and advances the number automatically after each successful run. Leave it in auto-managed mode for your first session — it handles collision detection for you.

If you want a specific name, type it in. The field exits auto-managed mode once you type. It returns to auto-managed after a successful run.

The full repo id format is `hf_username/dataset_name`. If you type just `dataset_name`, the app prepends your `hf_username` automatically.

### 3. Set episodes and timing

| Field | Recommended first run | Notes |
|---|---|---|
| Episodes | `5–10` | Each episode is one demonstration attempt |
| Episode time (seconds) | `15–30` | Total time per episode including warmup |
| Task description | `"Pick up the block"` | Plain language; stored in dataset metadata |

For test runs, use 2 episodes and 10 seconds.

### 4. Check the camera preview

In the **Camera** section, click **Refresh Camera Preview**. Both camera views should show your workspace.

If cameras are blank or misassigned:
1. Click **Scan Camera Ports** to detect connected cameras.
2. Click **Set Laptop** or **Set Phone** to assign the correct roles.
3. Click **Refresh Camera Preview** again.

### 5. Verify the robot snapshot

Below the main form, the **Current Robot Snapshot** shows:
- follower/leader serial ports
- robot IDs
- camera indices and FPS

If anything looks wrong, open Config, fix the value, and Save Config. The snapshot updates on next tab open.

### 6. Preview the command

Click **Preview Command**. A dialog shows the exact `lerobot_record` command that will run. Verify:
- `--robot.port` and `--teleop.port` match your physical arms
- `--robot.id` and `--teleop.id` match your calibration profiles
- `--dataset.repo_id` has the right name
- Camera JSON has both cameras with the expected indices

If anything looks wrong, close the dialog, fix it in the form or Config, and preview again.

### 7. Run Record

Click **Run Record**.

1. A confirmation dialog shows the command. Click **Confirm**.
2. Preflight runs. Review any warnings. If there are failures, fix them and re-run.
3. The record session starts in the terminal panel.
4. Use the leader arm to demonstrate the task. The follower arm mirrors your movements and all cameras record.
5. Each episode ends automatically when the episode timer expires, or you can press the configured key to end it early.
6. After all episodes complete, the command exits.

**What success looks like:** The terminal shows `Episode X complete` for each episode and then exits with code 0. The app shows a success dialog.

### 8. Find your dataset

Your dataset is saved in `record_data_dir/<hf_username>/<dataset_name>/`. It contains:
- `data/` — episode parquet files
- `videos/` — encoded camera videos
- `meta/` — metadata files

### 9. (Optional) Upload to Hugging Face

If **Upload to Hugging Face after recording** was checked, the upload runs automatically after the record completes. See [Uploading to Hugging Face](#uploading-to-hugging-face) below.

---

## UI Reference

### Recording Setup

| Control | What it does |
|---|---|
| Dataset name / repo id | Accepts `name` or `owner/name`; auto-managed by default |
| Local dataset save folder | Override for where the dataset lands locally |
| Episodes | `--dataset.num_episodes` |
| Episode time (seconds) | `--dataset.episode_time_s` |
| Task description | `--dataset.single_task` |
| Upload to HF after recording | Runs upload immediately after a successful record |

### Advanced Command Options

Expand this section to override specific flags:

| Flag | Override |
|---|---|
| `--robot.port` | Follower serial port |
| `--teleop.port` | Leader serial port |
| `--dataset.repo_id` | Full dataset repo id |
| `--dataset.num_episodes` | Episode count |
| `--dataset.episode_time_s` | Episode time |
| Custom args (raw) | Appended verbatim to the end of the command |

### Action Buttons

| Button | What it does |
|---|---|
| Preview Command | Shows the command without running it |
| Run Record | Starts record with preflight confirmation |
| Upload Local Dataset | Opens upload dialog for an existing local dataset |

---

## Dataset Naming

### Auto-managed mode

The app starts in auto-managed mode. The name is seeded from your last successful record run and advances the trailing number. For example, after `your_name_5` succeeds, the next auto-managed name is `your_name_6`.

Before each preview/preflight/run, the app checks for local and Hugging Face collisions and advances the number if needed.

### Manual mode

Type any name to exit auto-managed mode. The app preserves your name until the next successful run, at which point it reseeds.

### Collision detection

The app checks:
- Whether a folder with that name already exists in `record_data_dir`
- Whether a dataset with that repo id already exists on Hugging Face (with a 60-second cache to avoid repeated API calls)

If a collision is detected during auto-management, the name advances. If you're in manual mode, you'll see a warning and a chance to continue or cancel.

---

## Uploading to Hugging Face

### Post-record upload

Check **Upload to Hugging Face after recording** before starting a run. After a successful record, the app runs:

```bash
huggingface-cli upload <owner/dataset_name> <local_dataset_path> --repo-type dataset
```

After a successful upload, the app tags the dataset card with `lerobot` and other provenance tags.

Requirements:
- You must have run `huggingface-cli login` (or `hf auth login`) at least once.
- `huggingface-cli` must be in PATH.
- `hf_username` in Config must match your Hugging Face account.

### Manual upload (Upload Local Dataset)

Use this to push an existing local dataset folder that wasn't uploaded right after recording.

1. Click **Upload Local Dataset**.
2. Choose a local dataset folder.
3. Set the HF owner and dataset name.
4. Review the parity check — the app warns you if:
   - No HF login is detected
   - The remote repo already exists
   - The local dataset already has HF provenance for a different repo id
5. Click **Preview Upload** to see the exact command.
6. Click **Run Upload** to execute.

Upload runs appear in History as `upload` mode entries.

---

## Dataset Browser

The browser shows local and Hugging Face datasets side by side.

### Left pane — Local datasets

A tree rooted at `record_data_dir`. Shows owner folders and recognized dataset folders inside them. Click **Refresh Local** to rescan.

### Right pane — Hugging Face datasets

Owner-scoped list of datasets. Defaults to `hf_username`. Change the owner field and click **Refresh HF** to browse a different owner's datasets.

### Actions

| Button | What it does |
|---|---|
| Use Selected in Record | Copies the selected dataset's repo id into the Record form |
| Upload Local Dataset | Opens manual upload dialog for the selected local dataset |
| Refresh Local | Rescans `record_data_dir` |
| Refresh HF | Fetches latest datasets for the current HF owner |

---

## Command Shape

A typical generated record command:

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=red4 \
  --robot.cameras='{"laptop":{"type":"opencv","index_or_path":4,"width":640,"height":360,"fps":30,"warmup_s":5},"phone":{"type":"opencv","index_or_path":6,"width":640,"height":360,"fps":30,"warmup_s":5}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=white \
  --dataset.repo_id=matthewwoodc0/my_dataset_1 \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the white block and place it in the bin" \
  --dataset.episode_time_s=20 \
  --warmup_time_s=5
```

Key notes:
- Camera JSON is auto-built from your configured indices and detected frame sizes.
- `--warmup_time_s` is a global warmup before recording starts (Record-only; not added for Deploy).

---

## What You Might See

### Validation messages

| Message | Meaning |
|---|---|
| `Validation Error: Episodes and episode time must be integers.` | Non-numeric value entered |
| `Dataset Exists: <repo> already exists on Hugging Face. Continue anyway?` | Remote collision in manual mode |
| `Record Failed: Recording failed with exit code <n>.` | Non-zero subprocess exit |
| `Done: Recording completed.` | Success |

### Dataset browser status

| Status | Meaning |
|---|---|
| `Local datasets in <path>` | Datasets found and listed |
| `No local datasets detected in <path>` | Folder exists but no recognized datasets inside |
| `Local dataset root not found: <path>` | `record_data_dir` doesn't exist |
| `Hugging Face datasets for <owner>` | Remote list loaded successfully |

### HF upload popup status

| Status | Meaning |
|---|---|
| `Local dataset provenance already points to: <owner/dataset>` | Dataset was previously uploaded; shown as a warning |
| `Remote dataset already exists: <owner/dataset>` | Remote already has data at that path |
| `Dataset upload completed: <owner/dataset>` | Success |

---

## Troubleshooting

### Record starts but cameras show black frames

The camera warmup period may be too short. Increase `camera_warmup_s` in Config (try `8–10`). If only one camera is black, verify the camera index in Config and rescan.

### Episode time warning in preflight

- `< 8s` — very short; may not capture enough data per episode
- `> 180s` — very long; may cause timeout or memory issues

These are warnings, not failures. You can proceed, but consider adjusting.

### Record exits early with "device not found"

A camera was disconnected or the serial port dropped. Check USB connections and re-run. On Linux, use `/dev/serial/by-id/...` stable paths to prevent port reassignment between runs.

### Upload fails with authentication error

Run `huggingface-cli login` (or `hf auth login`) in your conda environment to refresh credentials.

### Auto-managed name keeps advancing past expected number

This happens when the app detects a collision at the current number. Check `record_data_dir` and your HF namespace for folders/repos at the expected names. Delete or rename the collision if it was a test artifact.

### Record completes but no dataset folder appears

Check `record_data_dir` in Config — the path may not be what you expect. Also check that the LeRobot process actually had write permission to that directory.
