# Record Tab Guide

This guide explains exactly what the `Record` tab does, how each control maps to runtime behavior, and what messages you might see while using it.

## What This Tab Is For

Use `Record` to:
- Build a `lerobot_record` command for teleoperated data collection.
- Run local recording on your device.
- Optionally upload the recorded dataset to Hugging Face.
- Browse local and Hugging Face datasets from the same panel.

## Main UI Areas

## 1) Recording Setup

- `Dataset name (or repo id)`
  - Accepts either `dataset_name` or full `owner/dataset_name`.
  - Starts in auto-managed mode and advances from your last successful numbered dataset name.
  - If you type your own name, the field stops auto-overwriting it until the next successful record run reseeds it.
- `Local dataset save folder`
  - Where completed dataset folders are expected to end up.
- `Episodes`
  - Maps to `--dataset.num_episodes`.
- `Episode time (seconds)`
  - Maps to `--dataset.episode_time_s`.
- `Task description`
  - Maps to `--dataset.single_task`.
- `Upload to Hugging Face after recording`
  - Enables post-record upload flow and upload options panel.

## 2) Upload Options (only shown when upload is enabled)

- `Hugging Face username`
- `Dataset name on Hugging Face`
- `Use Dataset Field`
  - Copies repo name from dataset field.
- `Run LeRobot v3.0 conversion/tagging after upload`
  - Runs:
  - `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=<owner/dataset>`

## 3) Advanced command options

- `Advanced command options` reveals full flag override inputs.
- You can override specific `lerobot_record` flags like:
  - `--robot.port`
  - `--teleop.port`
  - `--dataset.repo_id`
  - `--dataset.num_episodes`
  - `--dataset.episode_time_s`
- `Custom args (raw)` appends raw args to the generated command.

## 4) Action buttons

- `Preview Command`
  - Shows the final command without running it.
- `Run Record`
  - Starts the record run after confirmation and preflight checks.
- `Deploy Dataset to Hugging Face...`
  - Opens a dedicated upload popup for local datasets.

## 5) Current Robot Snapshot + Camera Preview

- Snapshot shows:
  - follower/leader serial ports
  - laptop/phone camera indices
  - camera FPS and warmup seconds
- Camera panel supports:
  - `Scan Camera Ports`
  - `Refresh Camera Preview`
  - role assignment (`Set Laptop`, `Set Phone`) for detected cameras

## 6) Dataset Browser

Left pane:
- Dataset list with source toggle:
  - `Local`
  - `Hugging Face`

Right pane:
- Metadata JSON for selected dataset.

Toolbar behavior:
- In `Local` mode, you set/browse local dataset root.
- In `Hugging Face` mode, you set HF owner and fetch remote datasets.
- `Use Selected in Record` copies selection into record inputs.

## What Happens When You Click Run Record

1. The app builds a `lerobot_record` command using your current form values.
2. If upload is enabled, repo id is forced to `owner/repo_name` from upload fields.
3. The app revalidates auto-managed dataset names before preview/preflight/run and advances them when local or HF collisions are detected.
4. You confirm the command in a dialog.
5. Record preflight runs (ports, cameras, `lerobot`, `cv2`, dataset root writable, duration sanity, etc.).
6. The command executes in `lerobot_dir`.
7. On success, the app moves/normalizes dataset location into configured record root.
8. If upload is enabled, it runs:
   - `huggingface-cli upload <repo_id> <local_dataset> --repo-type dataset`
9. If conversion is enabled, it runs v3.0 conversion/tagging command.
10. Config is updated (`last_dataset_name`, optional `hf_username`, upload conversion preference).

## Command Shape You Should Expect

Record command (example):

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=red4 \
  --robot.cameras='{"laptop":{"type":"opencv","index_or_path":4,"width":640,"height":360,"fps":30,"warmup_s":5},"phone":{"type":"opencv","index_or_path":6,"width":640,"height":360,"fps":30,"warmup_s":5}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=white \
  --dataset.repo_id=matthewwoodc0/jeffrey_20 \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the white block and place it in the bin" \
  --dataset.episode_time_s=20 \
  --warmup_time_s=5
```

## Example Workflow

1. Open `Record`.
2. Set dataset name, or leave the auto-managed value in place.
3. Set episodes/time/task.
4. Check camera preview and port assignments.
5. Click `Preview Command`.
6. Click `Run Record`.
7. Review confirm dialog.
8. Review preflight dialog and continue.
9. If uploading, monitor upload completion and optional conversion result.

## What You Might See

Validation/popups:
- `Validation Error: Episodes and episode time must be integers.`
- `Dataset Exists: <repo> already exists on Hugging Face. Continue anyway?`
- `Record Failed: Recording failed with exit code <n>.`
- `Done: Recording completed.`

Dataset browser status lines:
- `Local datasets in <path>`
- `No local datasets detected in the configured record roots.`
- `Hugging Face datasets for <owner>`

HF upload popup status examples:
- `Upload local dataset to Hugging Face with local/remote parity checks.`
- `Remote dataset already exists. Upload skipped.`
- `HF upload completed.`

## Notes

- If upload is enabled, `huggingface-cli` must be in PATH.
- `Episode time` warnings appear in preflight when very short (<8s) or very long (>180s).
- Camera JSON is auto-built from your configured camera indices and detected frame sizes.
