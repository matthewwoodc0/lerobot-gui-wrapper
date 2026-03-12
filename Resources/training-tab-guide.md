# Training Tab Guide

This tab is currently dedicated to **Human Intervention Learning (HIL)** workflows.

Current runtime behavior:
- Generated train commands run with the configured LeRobot runtime (`lerobot_venv_dir`) when available.
- Resume only works when the detected train entrypoint exposes a real checkpoint/config-path flag. If the current LeRobot build only offers a boolean resume stub or no resume-path flag at all, the UI will reject the resume request instead of silently starting a fresh run.
- If you point resume at a checkpoint folder that contains `train_config.json`, the wrapper will resolve that file automatically before building the command.

## What This Tab Is For

Use `Training` to:
- Prepare a short, incremental HIL adaptation run.
- Keep adaptation runs isolated with `_hil` output/job naming.
- Copy/paste the resulting HIL command into your own terminal/session.
- Save reusable HIL defaults.

## Main UI Areas

## 1) Human Intervention Learning (HIL)

Core train fields (used to build the HIL adaptation command):
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

Guidance fields:
- `Project root`
- `Env activate cmd`
- `HIL intervention repo` (dataset repo id that contains human-correction episodes)
- `HIL base model path` (existing model checkpoint/hub id used as adaptation starting point)

Toggles:
- `W&B enabled` -> `--wandb.enable=true/false`
- `Push to hub` -> `--policy.push_to_hub=true/false`
- `Wrap with srun`

Buttons:
- `Apply HIL Preset`
- `Copy HIL Command`
- `Save HIL Defaults`

## 2) Generated Command (Editable)

- Dark themed mini-editor.
- Final command text can be edited manually before copy/paste.
- `Copy HIL Command` copies current editor contents.

## Human Intervention Learning (HIL) Quick Adaptation

`Apply HIL Preset` sets the tab into short adaptation mode:
- `Batch size` -> `8`
- `Steps` -> `3000`
- `Save freq` -> `300`
- Appends `_hil` to `Output dir` and `Job name`.
- Regenerates command text and opens a step-by-step HIL workflow dialog.

Recommended loop:
1. Capture teleop corrections for specific failure modes.
2. Merge/push those episodes into `HIL intervention repo`.
3. Point `Policy path` (or `HIL base model path`) to the last successful model.
4. Click `Apply HIL Preset`, run the generated command, and validate.
5. Repeat only on new intervention slices.

## Notes

- This tab intentionally does not SSH, SFTP, attach tmux, or launch remote jobs.
- It is currently focused on HIL adaptation-only workflows.
