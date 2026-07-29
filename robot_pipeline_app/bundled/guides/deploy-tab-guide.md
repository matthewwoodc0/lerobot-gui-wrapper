# Deploy Tab Guide

Use this tab to evaluate a trained policy on hardware: select a model, configure an eval run, run preflight checks, and review results.

---

## Table of Contents

1. [What This Tab Is For](#what-this-tab-is-for)
2. [Before You Deploy](#before-you-deploy)
3. [Full Walkthrough — First Deploy Run](#full-walkthrough--first-deploy-run)
4. [UI Reference](#ui-reference)
5. [Model Selection](#model-selection)
6. [Eval Dataset Naming](#eval-dataset-naming)
7. [Preflight and Fix Center](#preflight-and-fix-center)
8. [After the Run — Results](#after-the-run--results)
9. [Command Shape](#command-shape)
10. [Uploading a Model to Hugging Face](#uploading-a-model-to-hugging-face)
11. [Troubleshooting](#troubleshooting)

---

## What This Tab Is For

- Select a local trained model and checkpoint folder.
- Configure an evaluation dataset name, episode count, and task description.
- Run preflight checks that validate model payload, compute device, camera keys, and flag support.
- Prefer `lerobot-rollout` when the installed LeRobot runtime provides it (0.6.x current track).
- Fall back to a `lerobot_record` command with `--policy.path` on supported older runtimes so the robot can run autonomously while recording eval episodes.
- Review results in **History** and **Experiments**.

---

## Before You Deploy

Deploy requires the same hardware setup as Record, plus a trained model. Confirm:

- Teleop starts and arms respond.
- Camera preview is correct in this tab.
- A trained model folder exists in `trained_models_dir`.
- The model contains a valid policy config (`config.json` or equivalent) and a checkpoint.

If you just finished training, open the **Experiments** tab to browse checkpoints and launch Deploy directly from there.

---

## Full Walkthrough — First Deploy Run

### 1. Open Deploy

Click the **Deploy** tab in the sidebar.

### 2. Select your model

In the **Model Selection** panel on the right:

1. The **Root** path defaults to `trained_models_dir`. Change it if your model lives elsewhere.
2. The tree view shows model and checkpoint folders. Expand a top-level model folder to see its checkpoints.
3. Click a folder to select it.

**Reading the tree colors:**
- **Green rows** — folder is a directly runnable model payload (has a config + weights).
- **Amber/yellow rows** — folder resolves to a nested payload inside (the app will find it).
- **Muted rows** — regular folder with no policy payload detected.

Click a green or amber row. The **Selected Model Info** panel below the tree updates to show:
- the resolved payload path
- whether it's directly runnable
- the config presence
- detected checkpoint folders

If nothing in the tree is green or amber, your model folder structure may not match what the app expects. Use **Browse Model...** to navigate directly to any folder, or check [Model Selection](#model-selection) below.

### 3. Set eval dataset name

The eval dataset name starts in **auto-managed mode**, similar to Record. The name always requires an `eval_` prefix (e.g. `eval_my_model_1`).

If the name is missing the prefix, a **Quick Fix eval_** button appears — click it to prepend `eval_` automatically.

Leave the name in auto-managed mode for your first run. Collision handling then follows `Config -> Name Iteration`:
- `manual` keeps the current name and only surfaces collisions
- `auto` advances only auto-managed names
- `always` advances colliding names even after manual edits

### 4. Set episodes and timing

| Field | Recommended first run | Notes |
|---|---|---|
| Eval episodes | `5–10` | Each episode is one autonomous policy run |
| Eval episode time (seconds) | `20–30` | Match your training episode duration |
| Eval task description | Same as training task | Stored in eval dataset metadata |

For a quick smoke test, use 3 episodes and 15 seconds.

### 5. No Hugging Face account?

Deploy commands automatically include `--dataset.push_to_hub=false`. Your eval dataset is saved locally only. No HF account is required.

### 6. Preview the command

Click **Preview Command**. Verify:
- On rollout: `--policy.path` and strategy flags point to the selected model
- On legacy deploy: `--policy.path` is attached to the record entrypoint
- Camera JSON matches your current camera setup
- Dataset name has the `eval_` prefix

### 7. Run Deploy

Click **Run Deploy**.

1. The app validates the model payload and enforces the `eval_` naming convention.
2. A confirmation dialog shows the command. Click **Confirm**.
3. Deploy preflight runs. If fixable issues are found, the **Deploy Preflight Fix Center** appears with one-click fixes. Apply them, then confirm the updated command.
4. The deploy session starts. The robot runs autonomously — the policy controls the follower arm, no human input is needed.
5. Each episode ends when the timer expires.
6. After all episodes, the command exits.

**What success looks like:** Terminal shows `Episode X complete` for each episode and exits with code 0. A success dialog appears with a link to open History.

### 8. Review results

Click **Open History** in the success dialog, or navigate to the History tab.

In History:
- Find the deploy run by mode filter (`deploy`).
- Click **Edit Outcomes** to mark each episode as success or failed.
- Add tags and notes.
- Export `episode_outcomes.csv` for analysis.

Open **Experiments** to compare this run against other deploy and training runs.

---

## UI Reference

### Deploy / Eval Setup

| Control | What it does |
|---|---|
| Eval dataset name / repo id | Eval dataset (must have `eval_` prefix) |
| Eval episodes | `--dataset.num_episodes` |
| Eval episode time (seconds) | `--dataset.episode_time_s` |
| Eval task description | `--dataset.single_task` |
| Quick Fix eval_ | Prepends `eval_` if missing |
| Preview Command | Shows command without running |
| Run Deploy | Starts eval with preflight confirmation |

### Advanced Command Options

| Flag | Override |
|---|---|
| `--policy.path` | Direct path to model payload (auto-set from tree selection) |
| `--robot.port` | Follower serial port |
| `--teleop.port` | Leader serial port |
| `--dataset.repo_id` | Full eval dataset repo id |
| Custom args (raw) | Appended verbatim to the command |

> Note: The `dataset.repo_id` in Advanced Options syncs with the main eval dataset field. If it shows a stale name, toggle Advanced Options off and back on.

---

## Model Selection

### Tree view

The tree is rooted at `trained_models_dir`. It shows two levels: top-level model folders and their immediate subdirectories (checkpoints). The app tags each row based on whether it detects a runnable policy payload.

**Use Browse Model...** to navigate to any folder outside `trained_models_dir`. The browsed path is applied directly and the tree updates.

**Use Browse Root** to change `trained_models_dir` and rescan the whole tree.

### Model payload detection

The app looks for a `config.json` (or equivalent) plus weight files. If your model folder contains a `checkpoint-NNNN/` subfolder with the actual payload, select the parent — the app resolves the nested payload automatically (shown as amber in the tree).

### Selected Model Info

After selecting a folder, this panel shows:
- **Selected path** — what you clicked
- **Resolved deploy payload** — the actual path the app will pass to `--policy.path`
- **Directly runnable** — yes/no
- **Config present** — whether a policy config was found
- **Checkpoint folders** — any `checkpoint-*` subfolders discovered
- **Top-level contents** — first-level file listing

If "Resolved deploy payload" is blank or shows an error, the selected folder doesn't contain a valid model. Try a subdirectory.

---

## Eval Dataset Naming

The eval dataset name is auto-managed in the same way as Record. Key differences:

- Must have an `eval_` prefix. The Quick Fix button adds it.
- Selecting a different model updates the eval name only while it's still auto-managed.
- The name advances on open, preflight, launch, success, cancel, or failure whenever the active `Name Iteration` policy allows it.
- Local and HF collisions are checked before preview/preflight/run.
- Local collision checks include both flat folders and owner-qualified paths such as `deploy_data_dir/<owner>/<eval_name>`.

Best practice: keep eval datasets completely separate from training datasets. The `eval_` prefix enforces this visually and in your HF namespace.

---

## Preflight and Fix Center

Deploy preflight checks:

| Check | What it validates |
|---|---|
| Eval dataset naming | Requires `eval_` prefix |
| Model payload | Valid payload path resolves and config is present |
| Model payload candidates | If parent folder selected, finds nested payload |
| Model camera keys | Config's camera keys (`laptop`, `phone`) match policy feature map |
| lerobot_record policy flag | `--policy.path` flag is supported by installed LeRobot |
| Compute accelerator | CUDA/MPS/CPU availability |
| Deploy loop performance | Warning when using CPU + high camera FPS (may not keep up) |

**Fix Center:** If one or more fixes are available (e.g. camera key remapping, eval name fix), the Fix Center dialog shows them as one-click actions. After applying fixes, the app shows the updated command for a final confirmation.

---

## After the Run — Results

### History tab

Each deploy run writes artifacts to `runs_dir`. Open History to:
- View the full transcript
- Mark individual episodes as success/failed/unmarked
- Add tags and free-text notes
- Export `episode_outcomes.csv` and `notes.md`
- Launch a replay of any episode

### Experiments tab

Experiments aggregates deploy runs with training runs and sim-eval runs. Use it to:
- Compare success rates across different checkpoints
- View parsed metrics from eval output
- Launch a new deploy run directly from a checkpoint

---

## Command Shape

A typical generated deploy command:

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=red4 \
  --robot.cameras='{"laptop":{"type":"opencv","index_or_path":4,"width":640,"height":360,"fps":30,"warmup_s":5},"phone":{"type":"opencv","index_or_path":6,"width":640,"height":360,"fps":30,"warmup_s":5}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=white \
  --dataset.repo_id=your_username/eval_my_model_1 \
  --dataset.num_episodes=10 \
  --dataset.single_task="Grasp a lego block and put it in the bin." \
  --dataset.episode_time_s=20 \
  --dataset.push_to_hub=false \
  --policy.path=/home/you/lerobot/trained_models/my_model
```

Key differences from a record command:
- `--policy.path` — points to your trained model
- `--dataset.push_to_hub=false` — eval datasets are local only by default
- No global `--warmup_time_s` — warmup is per-camera in the cameras JSON

---

## Uploading a Model to Hugging Face

Use **Deploy Model to Hugging Face...** to back up or share a trained model.

1. Click **Deploy Model to Hugging Face...** in the Model Selection panel.
2. Choose or confirm the local model folder.
3. Set the HF owner and model name.
4. The parity check shows whether the remote already exists.
5. Optionally enable **skip if exists** to avoid re-uploading unchanged models.
6. Click **Preview** to review the exact command.
7. Click **Run** to execute.

Upload command shape:

```bash
huggingface-cli upload <owner/model_repo> <local_model_folder> --repo-type model
```

This is for backup/sharing only. You do not need to upload a model to HF to run deploy locally.

---

## Troubleshooting

### "eval_ prefix required" dialog keeps appearing

The auto-managed name lost its prefix. Click **Quick Fix eval_** and it reappears. If you're in manual mode, edit the name directly.

### "not enough values to unpack" on second deploy run

This means the eval dataset repo id ended up without a `username/` prefix. Set `hf_username` in Config (a placeholder like `local_user` works if you don't have HF). The app normalizes the name automatically when `hf_username` is set.

### Policy crashes with upload/push error

Deploy runs always include `--dataset.push_to_hub=false`. If you still see this error, your LeRobot version may not support the flag. Add `--dataset.push_to_hub=false` explicitly to the **Custom args** field as a workaround, or check your LeRobot version against the [Compatibility Matrix](./compatibility-matrix.md).

### Advanced Options `dataset.repo_id` shows a stale name

Toggle Advanced Options off and back on. The field re-seeds from the current eval dataset name.

### Model selected but "Resolved deploy payload" is blank

The folder doesn't contain a recognized policy payload. Check that the folder has a `config.json` (or `pretrained_model_name_or_path` pointer) and weight files. Try selecting a checkpoint subfolder directly.

### Preflight warns about camera key mismatch

The policy was trained with different camera names than your current Config. Either:
- Use the Fix Center to remap camera keys before running.
- Update your Config camera names to match the policy's expected keys.
- Add a `--policy.camera_feature_map` override in Custom args.

### Arms don't move during deploy

The policy may need a warmup period. The camera warmup (`warmup_s` in the cameras JSON) should give enough time. If arms stay still for the full episode, check the terminal output for errors — the policy may have crashed silently or the camera feature map may be wrong.
