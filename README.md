# LeRobot GUI Wrapper

Local-first pipeline manager for SO-101 + LeRobot.

This project now works **only with on-device datasets and models**. It does not access Olympus or any remote cluster.

## What You Get

- Desktop GUI app (`python3 robot_pipeline.py gui`)
- CLI modes (`record`, `deploy`, `config`, `doctor`, `history`)
- First-time setup wizard via config prompts
- Persistent settings in `~/.robot_config.json`
- Teleop dataset recording with command preview
- Optional Hugging Face upload after recording
- Local model deployment/eval via `lerobot_record --policy.path=...`
- Built-in diagnostics command for local env/ports/cameras
- Preflight safety checks before record/deploy execution
- Run artifacts (command log + metadata) and history listing

## Internal Architecture

`robot_pipeline.py` is now a compatibility shim that preserves:
- `python3 robot_pipeline.py <mode>`
- `import robot_pipeline as rp`

Implementation modules live in `robot_pipeline_app/`:
- `constants.py`: defaults and config field definitions
- `types.py`: internal typed dataclasses for requests/results/reports
- `config_store.py`: config loading/saving/normalization and prompt helpers
- `repo_utils.py`: dataset/repo naming + Hugging Face existence checks
- `commands.py`: `lerobot_record` command construction
- `probes.py`: module/camera probing helpers
- `checks.py`: preflight + doctor checks
- `artifacts.py`: run artifact persistence + history listing
- `runner.py`: shared sync/async command execution
- `workflows.py`: shared record/deploy/upload execution helpers
- `cli_modes.py`: CLI mode handlers and main dispatcher
- `gui_app.py`: GUI composition/orchestration
- `gui_runner.py`: shared async run state, cancellation, and artifact lifecycle for GUI actions
- `gui_record_tab.py`, `gui_deploy_tab.py`, `gui_config_tab.py`: per-tab UI builders and callbacks
- `gui_camera.py`, `gui_log.py`, `gui_forms.py`: reusable GUI camera/log/form helpers

## Install / Clone

```bash
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
source ~/lerobot/lerobot_env/bin/activate
```

## Run As App (GUI)

```bash
python3 robot_pipeline.py gui
```

GUI tabs:
- `Record`: dataset/repo name, episodes, task, camera preview, scan open camera ports, assign laptop/phone camera roles, run recording, optional upload
- `Deploy`: pick local model folder, set eval dataset/episodes/task/time, camera preview, scan open camera ports, assign laptop/phone camera roles, run deployment
- `Config`: edit and save grouped settings

Output area:
- large dark terminal with timestamps and color highlighting
- episode progress + run-time progress bars
- `Copy Last Command`, `Save Log`, `Clear Log`, and `Cancel Run` tools

Path fields support:
- manual typing
- browse/select folder button

Camera role controls in preview panels:
- `Scan Camera Ports` to detect open camera indices on the device
- select a detected index and set it as `Laptop` or `Phone`
- `Swap Laptop/Phone` to toggle camera roles quickly
- role changes auto-update and auto-save `camera_laptop_index` and `camera_phone_index`

## CLI Modes

```bash
python3 robot_pipeline.py record
python3 robot_pipeline.py deploy
python3 robot_pipeline.py config
python3 robot_pipeline.py doctor
python3 robot_pipeline.py history
python3 robot_pipeline.py history --limit 30
python3 robot_pipeline.py --help
```

Quick local test run:

```bash
PYTHONPYCACHEPREFIX=/tmp python3 -m unittest discover -s tests -p 'test_*.py'
```

## Config Explained

Saved at `~/.robot_config.json` (and mirrored to `<lerobot_dir>/.robot_config.json`):

- `lerobot_dir`: local LeRobot root used as the working directory
- `runs_dir`: folder where run artifacts are saved (`command.log` + `metadata.json`)
- `record_data_dir`: where recorded datasets should end up
- `trained_models_dir`: where local trained model folders live
- `hf_username`: Hugging Face username for dataset repo IDs
- `last_dataset_name`: used to suggest the next dataset name
- `follower_port`: follower arm serial port (e.g. `/dev/ttyACM1`)
- `leader_port`: teleop leader serial port (e.g. `/dev/ttyACM0`)
- `camera_laptop_index`: workspace camera index
- `camera_phone_index`: wrist/phone camera index
- `camera_warmup_s`: camera warmup in seconds used in `--robot.cameras` for record/deploy
- `camera_width`: camera capture width used in `--robot.cameras`
- `camera_height`: camera capture height used in `--robot.cameras`
- `camera_fps`: camera FPS used in `--robot.cameras`
- `eval_num_episodes`: default deploy/eval episode count
- `eval_duration_s`: default deploy/eval episode duration
- `eval_task`: default deploy/eval task
- `last_eval_dataset_name`: used to suggest next eval dataset name
- `last_model_name`: last local model folder name used for deploy

## Teleop / Recording Workflow

1. Run `record` mode (GUI or CLI).
2. Confirm dataset name, episodes, duration, task.
3. Review full `lerobot_record` command.
4. Run recording.
5. Preflight checks run and report PASS/WARN/FAIL items before launch.
6. Script uses `--robot.cameras` JSON with `warmup_s` for laptop and phone cameras.
7. Script moves dataset into `record_data_dir` if needed.
8. Optional: upload to Hugging Face.

## Training Workflow

Training is external to this script. Typical local loop:

1. Record demonstrations (`record`)
2. Train your policy using your LeRobot training command(s)
3. Save resulting model folder into `trained_models_dir`
4. Deploy that local model with `deploy`

The script does not launch or manage training jobs.

## Deployment Workflow (Local Only)

1. Open `deploy` mode.
2. Choose local model root folder (`trained_models_dir`).
3. Select the specific local model folder to run.
4. Choose eval dataset name/repo, episodes, task, and duration.
5. Review full `lerobot_record` command with `--policy.path=<local model>`.
6. Preflight checks run and report PASS/WARN/FAIL items before launch.
7. Script uses the same `--robot.cameras` JSON with `warmup_s`.
8. Run deployment/eval on-device.

No SFTP, no Olympus, no remote model fetch.

## Troubleshooting

### Folder picker does not appear
Use manual path entry (common on headless systems).

### `huggingface-cli` not found
Activate your env:

```bash
source ~/lerobot/lerobot_env/bin/activate
```

### Bad camera index / serial ports
Run `python3 robot_pipeline.py config` and update the values.

### Where are run logs?
Run artifacts are stored under `runs_dir` (default `~/.robot_pipeline_runs`), one folder per run:
- `command.log`
- `metadata.json`

Quick view:

```bash
python3 robot_pipeline.py history
python3 robot_pipeline.py history --limit 30
```

### macOS abort on GUI startup mentioning required macOS version
This usually comes from a mismatched OpenCV wheel in your Python env.
Use diagnostics and/or disable preview:

```bash
python3 robot_pipeline.py doctor
LEROBOT_DISABLE_CAMERA_PREVIEW=1 python3 robot_pipeline.py gui
```

## Recommended Daily Loop

1. `python3 robot_pipeline.py gui`
2. Record new data
3. Train locally
4. Deploy local model
5. Repeat
