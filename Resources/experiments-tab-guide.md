# Experiments Tab Guide

This tab is the experiment console for comparing train, deploy, and sim-eval runs in one place.

## What This Tab Is For

Use `Experiments` to:
- inspect saved training metadata and parsed metrics
- discover checkpoints and model artifacts under train output folders
- compare train, deploy, and sim-eval runs side by side
- launch deploy eval directly from a selected checkpoint
- launch simulation eval directly from a selected checkpoint
- open local output folders, command logs, and WandB runs

Data source:
- reads existing run artifacts from `runs_dir`
- reuses `metadata.json`, `command.log`, deploy notes/outcomes, and train/sim-eval output folders
- does not create a second experiment database

## Main UI Areas

## 1) Experiment Runs

Filters:
- `All runs`, `Train`, `Deploy`, `Sim eval`
- status filter
- free-text search over dataset/env, policy, checkpoint, and command

The run table summarizes:
- start time
- run type
- status
- dataset or env target
- policy
- checkpoint/model label
- parsed metrics summary

## 2) Comparison

Select multiple rows and click `Compare Selected`.

Comparison columns include:
- type
- status
- dataset or env
- policy
- checkpoint
- device
- duration
- notes or tags
- output location
- key metrics or outcomes
- WandB label when available

This is the main place to compare:
- training runs against other training runs
- training runs against downstream deploy runs
- simulation eval runs against local deploy results

## 3) Experiment Details

Selecting a row shows:
- full command
- output location
- parsed metrics
- deploy analytics or sim-eval summaries when present
- WandB local metadata and remote summary/config when available

If `command.log` exists, the raw transcript is available in the output panel.

## 4) Checkpoints

Training rows expose discovered checkpoint/model artifacts as first-class objects.

For each discovered checkpoint, the table shows:
- checkpoint label
- kind (`best`, `latest`, `final`, `checkpoint`, `artifact`)
- parsed step when detectable
- policy family
- path

Actions:
- `Open Checkpoint Folder`
- `Open train_config.json`

## 5) Deploy From Checkpoint

With a deployable checkpoint selected, `Launch Deploy Eval` reuses the existing deploy/eval machinery.

You provide:
- eval dataset
- episodes
- duration
- task

The wrapper then:
- builds the normal deploy command
- runs deploy preflight checks
- records the resulting deploy run so it can be compared immediately in `Experiments`

## 6) Sim Eval From Checkpoint

With a deployable checkpoint selected, `Launch Sim Eval` uses the detected LeRobot eval entrypoint only when compatibility probing confirms support.

Available fields are passed only when the installed runtime exposes matching flags, such as:
- model/checkpoint path
- env type
- task
- benchmark
- episodes
- batch size
- seed
- device
- output dir
- job name
- `--trust_remote_code`

If the installed LeRobot build does not expose a supported sim-eval entrypoint or required flags, the page reports that clearly instead of guessing.

## Metrics and Analytics

Training metrics are parsed pragmatically from:
- stdout
- `trainer_state.json`
- `metrics.json` / `metrics.jsonl`
- `wandb-summary.json`
- `eval_info.json` when train outputs include eval summaries

Simulation eval metrics are parsed from:
- stdout
- `eval_info.json`

Deploy analytics summarize:
- success / failed / unmarked episode counts
- success rate
- notes and tags
- grouped failure categories
- grouped diagnostic codes

## WandB Behavior

WandB is optional.

When disabled or unavailable:
- experiment views still work from local artifacts only

When enabled and discoverable:
- local metadata captures project, entity, run id, run name, and run URL
- experiment rows can deep-link to the matching WandB run
- remote summary/config data is added when credentials are available

## Typical Workflow

1. Launch a train run from `Train`.
2. Open `Experiments` and select the new train run.
3. Inspect parsed metrics and discovered checkpoints.
4. Compare the run against prior train runs.
5. Select a checkpoint and launch `Deploy Eval` or `Sim Eval`.
6. Refresh `Experiments` and compare the downstream results against the source train run.
7. Open `History` only when you need to edit deploy episode outcomes or rerun a raw command.
