# Training Tab Guide

This tab is a command generator. It does not run training jobs directly.

## What This Tab Is For

Use `Training` to:
- Generate an editable `lerobot_train` command.
- Optionally wrap that command with `srun`.
- Copy/paste command into your own terminal/session.
- Save reusable generator defaults.

## Main UI Areas

## 1) Training Command Generator

Core train fields:
- `Policy path` -> `--policy.path`
- `Dataset repo id` -> `--dataset.repo_id`
- `Output dir` -> `--output_dir`
- `Job name` -> `--job_name`
- `Device` -> `--policy.device`
- `Batch size` -> `--batch_size`
- `Steps` -> `--steps`
- `Save freq` -> `--save_freq`
- `Policy input features` -> `--policy.input_features`
- `Policy output features` -> `--policy.output_features`
- `Python binary` (first token in generated command)
- `Extra train args` (raw append)

`srun` wrapper fields (used when `Wrap with srun` is enabled):
- `srun partition` -> `-p`
- `srun queue` -> `-q`
- `srun cpus/task` -> `--cpus-per-task`
- `srun gres` -> `--gres`
- `srun job name` -> `-J`
- `srun extra args` (raw append before python command)

Guidance-only fields:
- `Project root`
- `Env activate cmd`

Toggles:
- `W&B enabled` -> `--wandb.enable=true/false`
- `Push to hub` -> `--policy.push_to_hub=true/false`
- `Wrap with srun`

Buttons:
- `Generate Command`
- `Copy Command`
- `Preview Guidance`
- `Save Defaults`

## 2) Generated Command (Editable)

- Dark themed mini-editor.
- Final command text can be edited manually before copy/paste.
- `Copy Command` copies current editor contents.

## Command Shape You Should Expect

Default style (with `srun` enabled):

```bash
srun -p gpu-research --cpus-per-task=8 --gres=gpu:a100:1 -J smolvla_b16_jeffrey_20 -q olympus-research-gpu --pty \
python -m lerobot.scripts.lerobot_train \
--policy.path=lerobot/smolvla_base \
--policy.input_features=null \
--policy.output_features=null \
--dataset.repo_id=matthewwoodc0/jeffrey_20 \
--batch_size=16 \
--steps=50000 \
--output_dir=outputs/train/smolvla_b16_jeffrey_20 \
--job_name=smolvla_b16_jeffrey_20 \
--policy.device=cuda \
--wandb.enable=true \
--policy.push_to_hub=false \
--save_freq=5000
```

## What Happens When You Click Buttons

`Generate Command`:
1. Validates numeric fields (`batch`, `steps`, `save_freq`, and optional `srun cpus/task`).
2. Validates required text fields.
3. Builds base train command.
4. Optionally prepends `srun`.
5. Writes result into editor and saves defaults.

`Copy Command`:
1. Uses editor contents (or generates if empty).
2. Copies to clipboard.
3. Saves current generator settings.

`Preview Guidance`:
1. Opens a dialog with shell flow:
   - `cd <project_root>`
   - `<env_activate_cmd>`
   - paste generated command
2. Shows expected model output path:
   - `<project_root>/<output_dir>/checkpoints/last/pretrained_model`

`Save Defaults`:
- Persists all generator fields and editor command to config.

## Example Workflow

1. Open `Training`.
2. Set dataset repo id, output dir, and job name.
3. Choose whether `srun` wrapping is needed.
4. Click `Generate Command`.
5. Optionally edit generated command in editor.
6. Click `Copy Command`.
7. In your terminal/session, run the command manually.

## What You Might See

Validation messages:
- `Batch size must be an integer.`
- `Steps must be an integer.`
- `Save freq must be an integer.`
- `srun cpus-per-task must be an integer.`
- `Policy path is required.`

Status examples:
- `Generated command. Copy and paste into your terminal. Expected model path: ...`
- `Copied command to clipboard. Paste it into your terminal.`
- `Saved training generator defaults.`

## Notes

- This tab intentionally does not SSH, SFTP, attach tmux, or launch remote jobs.
- It is focused on reproducible command text generation plus quick manual edits.
