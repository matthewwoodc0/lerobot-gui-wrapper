# Record Tab Guide

This guide explains exactly what the `Record` tab does, how each control maps to runtime behavior, and what messages you might see while using it.

## What This Tab Is For

Use `Record` to:
- Build a `lerobot_record` command for teleoperated data collection.
- Run local recording on your device.
- Optionally upload the recorded dataset to Hugging Face after a run.
- Manually upload an existing local dataset folder to Hugging Face.
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
  - Runs a post-record dataset upload with the same repo id as the record command.

## 2) Advanced command options

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
- `Upload Local Dataset`
  - Opens a dedicated upload popup for an existing local dataset folder.

## 4) Current Robot Snapshot + Camera Preview

- Snapshot shows:
  - follower/leader serial ports
  - laptop/phone camera indices
  - camera FPS and warmup seconds
- Camera panel supports:
  - `Scan Camera Ports`
  - `Refresh Camera Preview`
  - role assignment (`Set Laptop`, `Set Phone`) for detected cameras

## 5) Dataset Browser

Left pane:
- `Local datasets`
  - Tree view rooted at the current `Dataset root`.
  - Uses the same explorer-style look as the Deploy model browser.
  - Shows owner folders and recognized dataset folders.
  - `Refresh Local` rescans the current record root.

Right pane:
- `Hugging Face datasets`
  - Owner-scoped dataset list.
  - Defaults to your saved `hf_username`.
  - `Refresh HF` fetches the latest datasets for that owner.

Bottom actions:
- `Use Selected in Record`
  - Local selection uses HF provenance when present, otherwise falls back to `<hf_username>/<folder_name>` or just the folder name.
  - HF selection copies the selected repo id directly into the main Record dataset field.
- `Upload Local Dataset`
  - Opens the manual upload dialog for the selected or chosen local dataset folder.
  - The dialog previews the upload command and warns before upload when:
  - no HF login is detected
  - the target repo already exists
  - the local dataset already carries HF provenance for the same or a different repo

## What Happens When You Click Run Record

1. The app builds a `lerobot_record` command using your current form values.
2. The app revalidates auto-managed dataset names before preview/preflight/run and advances them when local or HF collisions are detected.
3. If you used `Use Selected in Record`, the selected dataset repo id stays in the main dataset field and participates in the same validation flow.
4. You confirm the command in a dialog.
5. Record preflight runs (ports, cameras, `lerobot`, `cv2`, dataset root writable, duration sanity, etc.).
6. The command executes in `lerobot_dir`.
7. On success, the app moves/normalizes dataset location into configured record root.
8. If post-record upload is enabled, it runs:
   - `huggingface-cli upload <repo_id> <local_dataset> --repo-type dataset`
9. The local dataset browser refreshes after successful record completion.
10. Config is updated (`last_dataset_name`, `last_dataset_repo_id`, `record_data_dir`, optional `hf_username`).

## What Happens When You Click Upload Local Dataset

1. The app requires an existing Hugging Face login and tells you to run `hf auth login` if it cannot find one.
2. You choose or confirm a local dataset folder, HF owner, and HF dataset name.
3. The app validates the local folder shape, checks for local HF provenance, and checks whether the remote repo already exists.
4. If the local dataset is already linked to Hugging Face or the target repo exists remotely, the app shows a warning and requires explicit confirmation.
5. You review a preflight summary and the exact `huggingface-cli upload ... --repo-type dataset` command before launch.
6. The upload runs through the same shared run controller and is saved into run history as an `upload` run.

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
- `No local datasets detected in <path>`
- `Local dataset root not found: <path>`
- `Hugging Face datasets for <owner>`

HF upload popup status examples:
- `Choose a local dataset folder, then preview or run the upload.`
- `Local dataset provenance already points to: <owner/dataset>`
- `Remote dataset already exists: <owner/dataset>`
- `Dataset upload completed: <owner/dataset>`

## Notes

- If upload is enabled, `huggingface-cli` must be in PATH.
- Manual uploads also require a valid Hugging Face login (`hf auth login`).
- `Episode time` warnings appear in preflight when very short (<8s) or very long (>180s).
- Camera JSON is auto-built from your configured camera indices and detected frame sizes.
