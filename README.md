# LeRobot GUI Wrapper

Local-first pipeline manager for SO-101 + LeRobot.

This project now works **only with on-device datasets and models**. It does not access Olympus or any remote cluster.

## What You Get

- Desktop GUI app (`python3 robot_pipeline.py gui`)
- CLI modes (`record`, `deploy`, `config`)
- First-time setup wizard via config prompts
- Persistent settings in `~/.robot_config.json`
- Teleop dataset recording with command preview
- Optional Hugging Face upload after recording
- Local model deployment/eval (no remote download)

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
- `Record`: dataset name, episodes, task, save path, run recording, optional upload
- `Deploy`: pick local model folder, preview command, run deployment
- `Config`: edit and save all core settings

Path fields support:
- manual typing
- browse/select folder button

## CLI Modes

```bash
python3 robot_pipeline.py record
python3 robot_pipeline.py deploy
python3 robot_pipeline.py config
python3 robot_pipeline.py --help
```

## Config Explained

Saved at `~/.robot_config.json` (and mirrored to `<lerobot_dir>/.robot_config.json`):

- `lerobot_dir`: local LeRobot root used as the working directory
- `record_data_dir`: where recorded datasets should end up
- `trained_models_dir`: where local trained model folders live
- `hf_username`: Hugging Face username for dataset repo IDs
- `last_dataset_name`: used to suggest the next dataset name
- `follower_port`: follower arm serial port (e.g. `/dev/ttyACM1`)
- `leader_port`: teleop leader serial port (e.g. `/dev/ttyACM0`)
- `camera_laptop_index`: workspace camera index
- `camera_phone_index`: wrist/phone camera index
- `last_model_name`: last local model folder name used for deploy

## Teleop / Recording Workflow

1. Run `record` mode (GUI or CLI).
2. Confirm dataset name, episodes, duration, task.
3. Review full `lerobot_record` command.
4. Run recording.
5. Script moves dataset into `record_data_dir` if needed.
6. Optional: upload to Hugging Face.

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
4. Review full `lerobot_eval` command.
5. Run deployment/eval on-device.

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

## Recommended Daily Loop

1. `python3 robot_pipeline.py gui`
2. Record new data
3. Train locally
4. Deploy local model
5. Repeat
