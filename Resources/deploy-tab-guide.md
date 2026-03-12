# Deploy Tab Guide

This guide covers the `Deploy` tab for local model evaluation/deployment, model selection, preflight fixes, and optional model upload to Hugging Face.

## What This Tab Is For

Use `Deploy` to:
- Select a local model payload.
- Generate and run an eval/deploy command.
- Enforce eval dataset naming (`eval_...`).
- Run preflight checks before deployment.
- Optionally upload local model folders to Hugging Face.

## Main UI Areas

## 1) Deploy / Eval Setup

- `Eval dataset name (or repo id)`
  - Accepts `dataset_name` or `owner/dataset_name`.
  - `Quick Fix eval_` prepends `eval_` when missing.
- `Eval episodes`
  - Maps to `--dataset.num_episodes`.
- `Eval episode time (seconds)`
  - Maps to `--dataset.episode_time_s`.
- `Eval task description`
  - Maps to `--dataset.single_task`.
- Buttons:
  - `Preview Command`
  - `Run Deploy`

## 2) Advanced command options

- `Advanced command options` reveals explicit flag overrides.
- Includes `--policy.path` override and full robot/dataset flags.
- `Custom args (raw)` appends raw arguments.
- The `dataset.repo_id` field in Advanced Options stays in sync with the main **Eval dataset name** field automatically. If you had it open during a previous run, toggle it off and back on (or just change the main eval dataset field) to refresh.

> **No HuggingFace account?** Deploy commands automatically include `--dataset.push_to_hub=false`, so evaluation datasets are only saved locally. You do not need an HF account to run deploy.

## 3) Model Selection

- `Root` path controls the models base directory.
- Tree view shows model/checkpoint folders.
- Bottom actions:
  - `Refresh`
  - `Browse Model...`
  - `Deploy Model to Hugging Face...`

Color/tag meaning in tree:
- Green-style model rows: folder is directly runnable.
- Accent/yellow-style rows: folder resolves to nested payload.
- Muted rows: regular folder/spacer.

## 4) Selected Model Info

Shows:
- selected path
- resolved deploy payload
- direct-runnable yes/no
- config presence
- checkpoint-like folders
- top-level contents preview

## 5) Deploy Camera Preview

Same scan/refresh/role assignment mechanics as other camera preview panels.

## What Happens When You Click Run Deploy

1. App validates selected model directory and payload shape.
2. App enforces eval naming convention (`eval_...`), offering quick-fix dialog.
3. App resolves unique eval dataset id against local and HF collisions.
4. You confirm command in a dialog.
5. Deploy preflight runs. If fixable issues are found, `Deploy Preflight Fix Center` appears with quick actions.
6. If fixes changed the command, app asks for one final confirm.
7. Config updates are persisted (model selection and eval defaults).
8. Deploy command runs in `lerobot_dir`.
9. On success, use `Experiments` for cross-run comparison and `History` for episode outcome editing and notes.

## Deploy Command Shape You Should Expect

Deploy command still uses `lerobot_record` with `--policy.path`:

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

Important:
- Deploy command uses per-camera `warmup_s` inside `--robot.cameras`.
- Deploy path currently does not add global `--warmup_time_s` (record-only default behavior).

## Preflight Checks You Might See

- `Eval dataset naming` (requires `eval_` dataset prefix)
- `Model payload` validation
- `Model payload candidates` (if parent folder selected)
- `Model camera keys` vs runtime keys (`laptop`, `phone`)
- `lerobot_record policy flag` support (`--policy.path`)
- `Compute accelerator` (cuda/mps/cpu)
- `Deploy loop performance risk` warning on CPU + high FPS

## Hugging Face Model Upload Popup

`Deploy Model to Hugging Face...` popup includes:
- local model folder picker
- local model candidate list
- HF owner + model name
- parity check (remote exists / missing / unknown)
- skip-if-exists option
- command preview
- confirm-and-run upload

Upload command shape:

```bash
huggingface-cli upload <owner/model_repo> <local_model_folder> --repo-type model
```

## Example Workflow

1. Open `Deploy`.
2. Pick model root and select a model/checkpoint in tree.
3. Confirm `Selected Model Info` shows a valid deploy payload.
4. Set eval dataset/episodes/time/task.
5. Click `Preview Command`.
6. Click `Run Deploy`.
7. Apply preflight quick fixes if offered.
8. After completion, open `Experiments` to compare against train/sim-eval runs, then open `History` to annotate outcomes.

## What You Might See

Dialogs:
- `Eval Dataset Prefix Required`
- `Deploy Preflight Fix Center`
- `Confirm Updated Deploy Command`
- `Deploy Failed: Deploy failed with exit code <n>.`
- `Done: Deployment completed. ... Open History ...`

Status/log examples:
- `Applied eval dataset quick fix: owner/eval_name`
- `Auto-iterated eval dataset to avoid existing target: owner/eval_name_2`
- `Model selected: <resolved_payload_path>`

## Troubleshooting Deploy Issues

**smolVLA or other policy crashes with upload/push error:** Deploy runs always include `--dataset.push_to_hub=false`. If you still see an error, check that your lerobot installation supports this flag. Add `--dataset.push_to_hub=false` explicitly to the **Custom args** field as a workaround.

**Second deploy run fails with `ValueError: not enough values to unpack`:** This means the eval dataset repo id ended up without a `username/` prefix. The app now normalizes this automatically. Set `hf_username` in Config (even a placeholder like `local_user` works) and the error should not recur.

**Advanced Options `dataset.repo_id` shows a stale name from a previous run:** Toggle Advanced Options off and back on to re-seed from the current eval dataset field.

## Notes

- If selected folder is a parent run/checkpoint folder, app may suggest nested payload path.
- For shared/public naming hygiene, keep eval datasets separate from record datasets using `eval_...`.
- Deploy datasets are saved locally only by default (`--dataset.push_to_hub=false`). You can override this in Advanced Options if you want to push eval results to HuggingFace.
