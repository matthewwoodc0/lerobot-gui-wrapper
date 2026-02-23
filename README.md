# LeRobot GUI Wrapper: `robot_pipeline.py`

A beginner-friendly local pipeline helper for SO-101 + LeRobot.

It wraps the repetitive steps for:
- Teleop recording (`record`)
- Pulling trained checkpoints from Olympus + running eval/deploy (`deploy`)
- Managing saved settings (`config`)

Repository: `https://github.com/matthewwoodc0/lerobot-gui-wrapper`

## What This Script Covers

- First-time setup wizard (auto-prompts if config is missing)
- Persistent config in `~/.robot_config.json`
- Dataset name auto-increment (`matthew_20` -> `matthew_21`)
- Optional check against Hugging Face dataset names
- Optional dataset upload to Hugging Face after recording
- SFTP download from Olympus for trained checkpoints
- Local deployment/eval command execution
- Manual path entry or folder browser (`b`) for save locations

## What It Does Not Cover

- Training job submission itself (you still run training commands on Olympus)
- Hardware setup/USB permissions
- Building LeRobot from scratch

## Repo Layout

```text
lerobot-gui-wrapper/
  robot_pipeline.py
  Resources/
```

Runtime files created by the script:

```text
~/.robot_config.json
~/lerobot/.robot_config.json   # secondary mirror copy
/tmp/sftp_batch.txt
```

## Prerequisites

1. LeRobot is installed on your inference/control machine.
2. Your environment is activated before running script commands:

```bash
source ~/lerobot/lerobot_env/bin/activate
```

3. Tools available in PATH:
- `python3`
- `huggingface-cli` (only needed for upload)
- `sftp` (for Olympus download)

4. Accounts/access:
- Hugging Face login (`huggingface-cli login`) if uploading datasets
- Olympus VPN/SSH access for model downloads

## Clone On Your Device

```bash
git clone https://github.com/matthewwoodc0/lerobot-gui-wrapper.git
cd lerobot-gui-wrapper
source ~/lerobot/lerobot_env/bin/activate
```

## First-Time Setup Wizard

Run either `record`, `deploy`, or `config`. If config is missing, the setup wizard starts automatically.

Recommended first command:

```bash
python3 robot_pipeline.py config
```

Path prompts support:
- `Enter` = use shown default
- Type path directly
- Type `b` = open folder picker (Finder/File Manager)

If running headless (no GUI), use typed paths.

## Config Fields Explained

Saved in `~/.robot_config.json`:

- `lerobot_dir`: Root LeRobot folder used as working directory for LeRobot module commands.
- `record_data_dir`: Where finished datasets should live locally.
- `trained_models_dir`: Where downloaded model checkpoints should be stored.
- `hf_username`: Hugging Face username for dataset repo IDs.
- `last_dataset_name`: Last dataset base name used to suggest next name.
- `follower_port`: SO-101 follower serial port (example: `/dev/ttyACM1`).
- `leader_port`: Leader/teleop serial port (example: `/dev/ttyACM0`).
- `camera_laptop_index`: OpenCV camera index for workspace/laptop camera.
- `camera_phone_index`: OpenCV camera index for wrist/phone camera.
- `olympus_user`: Olympus username.
- `olympus_host`: Olympus host used for SFTP.
- `olympus_scratch`: Olympus scratch base path containing training outputs.
- `last_model_name`: Last model run name used in deploy prompts.
- `last_checkpoint_steps`: Last checkpoint step value used in deploy prompts.

## Teleop Data Recording (`record`)

```bash
python3 robot_pipeline.py record
```

Flow:
1. Suggests next dataset name (auto-increment; checks Hugging Face when possible).
2. Prompts for save folder, episodes, episode duration, task description.
3. Prints full `lerobot_record` command before running.
4. Runs recording.
5. Moves dataset from `<lerobot_dir>/data/<name>` to your chosen `record_data_dir` when needed.
6. Prompts optional Hugging Face upload.

### Upload behavior

If you choose upload, it runs:

```bash
huggingface-cli upload <hf_user>/<dataset_name> <local_dataset_path> --repo-type dataset
```

If `huggingface-cli` is missing, script prints environment guidance.

## Training Workflow (Olympus)

Training is currently **manual** (outside this script).

Typical flow:
1. Record/upload dataset with `record` mode.
2. On Olympus, run your LeRobot training command using that dataset.
3. Ensure checkpoints end up at:

```text
<olympus_scratch>/outputs/train/<model_name>/checkpoints/<steps>/pretrained_model
```

That exact path structure is what `deploy` expects.

Use your known-good Olympus training commands from your LeRobot setup docs/scripts (for example, from `Resources/Lerobot Commands.pdf`).

## Deployment / Inference (`deploy`)

```bash
python3 robot_pipeline.py deploy
```

Flow:
1. Prompts model name and checkpoint step number.
2. Prompts local model save folder.
3. Generates `/tmp/sftp_batch.txt` and runs:

```bash
sftp -b /tmp/sftp_batch.txt <olympus_user>@<olympus_host>
```

4. Downloads the remote `pretrained_model` folder into:

```text
<trained_models_dir>/<model_name>_<steps>
```

5. Prompts to run deploy/eval command locally.
6. Runs LeRobot eval with your follower port + camera config.

If SFTP fails, it prints stderr and suggests checking VPN/SSH access.

## Common Commands

```bash
# Show CLI usage
python3 robot_pipeline.py --help

# Reconfigure everything
python3 robot_pipeline.py config

# Record teleop dataset
python3 robot_pipeline.py record

# Download checkpoint + deploy/eval
python3 robot_pipeline.py deploy
```

## Troubleshooting

### `python: command not found`
Use `python3` explicitly:

```bash
python3 robot_pipeline.py record
```

### Folder picker does not open
Usually means no desktop GUI is available. Enter path manually at prompt.

### Hugging Face upload fails
- Activate environment:
  - `source ~/lerobot/lerobot_env/bin/activate`
- Log in:
  - `huggingface-cli login`

### SFTP fails
- Confirm VPN and SSH access to Olympus
- Confirm `olympus_user`, `olympus_host`, and `olympus_scratch` in config
- Verify checkpoint path exists remotely

## Suggested End-to-End Loop

1. `python3 robot_pipeline.py record`
2. Train on Olympus (manual)
3. `python3 robot_pipeline.py deploy`
4. Iterate
