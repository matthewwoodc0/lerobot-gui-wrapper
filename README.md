# LeRobot GUI Wrapper

Local-first pipeline manager for SO-101 + LeRobot, now with optional remote training sync/launch from Linux hosts (for example Olympus).

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
- Deploy artifacts include structured notes and spreadsheet exports (`notes.md`, `episode_outcomes.csv`, `episode_outcomes_summary.csv`)
- Training tab for SSH profile management, remote model pull (`rsync`/`sftp`), and remote launch templates (`srun` + `tmux` / custom)

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
- `gui_theme.py`: shared GUI font/theme/style setup
- `gui_runner.py`: shared async run state, cancellation, and artifact lifecycle for GUI actions
- `gui_terminal_shell.py`: persistent interactive shell manager and shell history artifact logging
- `gui_history_tab.py`: history table/filter/details/rerun UI
- `gui_record_tab.py`, `gui_deploy_tab.py`, `gui_config_tab.py`, `gui_training_tab.py`: per-tab UI builders and callbacks
- `gui_camera.py`, `gui_log.py`, `gui_forms.py`: reusable GUI camera/log/form helpers
- `training_profiles.py`, `training_auth.py`, `training_remote.py`, `training_templates.py`: training profile/auth/remote/template helpers

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
- `Record`: dataset/repo name, episodes, task, camera snapshots, scan open camera ports, assign laptop/phone camera roles, run recording, optional upload
- `Deploy`: pick local model folder, set eval dataset/episodes/task/time, camera snapshots, scan open camera ports, assign laptop/phone camera roles, quick-fix `eval_` prefix, run deployment
- `Training`: manage SSH profiles, securely store SSH passwords via Linux `secret-tool`, browse remote model folders, pull remote checkpoints/models into local `trained_models_dir`, and launch remote training commands from templates
- `Config`: edit and save grouped settings

Output area:
- interactive dark terminal with timestamps and direct typing
- terminal can be hidden/shown and starts hidden by default
- terminal auto-opens on command failures and jumps to first error line
- quick access actions: `History`, `Open Latest Artifact`, and `Hide/Show Terminal`
- during `record`/`deploy`, Run Controls popout provides `Redo Run (Left)` / `Start Next (Right)` and per-episode countdown progress
- deploy preflight includes a `Fix Center` with quick actions before confirmation

Path fields support:
- manual typing
- browse/select folder button

History:
- new `History` tab with mode/status/search filters
- inspect stored command metadata and logs
- `Open Artifact Folder`, open `command.log`, copy command, and rerun with confirmation
- includes pipeline and shell commands (sensitive commands still execute but are not persisted)
- deploy reruns auto-normalize eval dataset IDs (`eval_` prefix + collision iteration)

Camera role controls in preview panels:
- `Scan Camera Ports` to detect open camera indices on the device
- `Refresh Camera Preview` to fetch one still frame per detected port (no continuous video stream)
- detected port cards are shown in a scrollable grid so all found ports remain reachable
- use `Set Laptop` / `Set Phone` on each detected port card
- active assignments are shown as `Laptop (Active)` and `Phone (Active)`
- role changes auto-update and auto-save `camera_laptop_index` and `camera_phone_index`
- deploy preflight compares configured role resolution vs detected frame size and offers one-click camera dimension fixes

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
- `camera_fps`: camera FPS used in `--robot.cameras`
- `gui_terminal_visible` (internal): remembers whether terminal pane is hidden/shown
- `eval_num_episodes`: default deploy/eval episode count
- `eval_duration_s`: default deploy/eval episode duration
- `eval_task`: default deploy/eval task
- `last_eval_dataset_name`: used to suggest next eval dataset name
- `last_model_name`: last local model folder name used for deploy
- `training_profiles` / `training_active_profile_id` / `training_last_remote_path` / `training_last_local_destination`: internal training-tab state

## Teleop / Recording Workflow

1. Run `record` mode (GUI or CLI).
2. Confirm dataset name, episodes, duration, task.
3. If the chosen dataset name already exists remotely or in local target folders, the app auto-iterates to the next name.
4. Review full `lerobot_record` command.
5. Run recording.
6. Preflight checks run and report PASS/WARN/FAIL items before launch.
7. Script uses `--robot.cameras` JSON with `warmup_s` for laptop and phone cameras.
8. Script moves dataset into `record_data_dir` if needed.
9. Optional: upload to Hugging Face.

## Training Workflow (GUI Training Tab)

1. Open `Training` tab.
2. Create/select an SSH profile (host/user/auth/root paths).
3. For password auth, store SSH password securely with `secret-tool` (Linux secret service).
4. Browse remote model/checkpoint directories and pull selected folder to local `trained_models_dir`.
5. Deploy tab refreshes automatically and can select the pulled model immediately.
6. Optionally launch remote training commands with template presets (`srun + tmux`, `tmux custom`, `custom command`).

## Deployment Workflow (Local Only)

1. Open `deploy` mode.
2. Choose local model root folder (`trained_models_dir`).
3. Select the specific local model payload folder to run (must contain config + weights in the same folder).
4. If you pick a parent run/checkpoint directory, deploy validation will block launch and suggest nested candidate paths.
5. Choose eval dataset name/repo, episodes, task, and duration (auto-iterated if a collision is detected). Deploy requires eval dataset names to start with `eval_`, with quick-fix actions in CLI and GUI.
6. Review full `lerobot_record` command with `--policy.path=<local model>`.
7. Preflight checks run and report PASS/WARN/FAIL items before launch.
8. Script uses the same `--robot.cameras` JSON with `warmup_s`.
9. Run deployment/eval on-device.

Remote sync/launch uses `ssh`, `rsync`, and `sftp` with strict host key checking.

## Linux Credential Storage

For password-based training profiles, install:

```bash
sudo apt install libsecret-tools expect
```

The GUI stores credentials in Linux Secret Service via `secret-tool`. Passwords are not written to config files or run artifacts.

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

### Deploy fails with exit code but no obvious cause
Deploy now logs `Deploy diagnostics:` hints after failures. Common causes include:
- invalid `--policy.path` (selected folder is not a runnable model payload)
- missing `lerobot` module in active environment
- camera index / serial port access failures
- LeRobot CLI flag mismatch with installed version

### Where are run logs?
Run artifacts are stored under `runs_dir` (default `~/.robot_pipeline_runs`), one folder per run:
- `command.log`
- `metadata.json`

`metadata.json` now includes optional additive fields used by GUI history (`status`, `command_argv`, `source`) while preserving existing keys.

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
