# History Tab Guide

This tab is the run log browser and deploy outcome editor.

Use `Experiments` when you want cross-run comparison, checkpoint browsing, or checkpoint-launched deploy/sim-eval actions. Use `History` when you need the raw run log, rerun controls, or deploy outcome editing.

## What This Tab Is For

Use `History` to:
- Filter and inspect past runs.
- Open run artifacts quickly.
- Copy/rerun prior commands.
- Edit deployment episode outcomes and notes.

Data source:
- Reads run artifacts from configured `runs_dir` (default `~/.robot_pipeline_runs`).
- Each run folder typically includes:
  - `command.log`
  - `metadata.json`
  - deploy-only generated files when edited: `notes.md`, `episode_outcomes.csv`, `episode_outcomes_summary.csv`

## Main UI Areas

## 1) Filters + Stats

Filters:
- `Mode` (`all`, `record`, `deploy`, `teleop`, `upload`, `shell`, `doctor`, etc.)
- `Status` (`all`, `success`, `failed`, `canceled`)
- `Search` (free-text match over run fields)

Stats strip:
- `Showing`
- `Success`
- `Failed`
- `Canceled`

## 2) Run Table

Columns:
- `Time`
- `Duration`
- `Mode`
- `Status`
- `Hint` (dataset repo id or model path)
- `Command` (truncated preview)

Selecting a row fills the details panel.

## 3) Selected History Entry

Shows rich metadata:
- run id
- mode/status/exit/canceled
- start/end/duration
- source
- dataset/model path
- cwd
- artifact path
- full command

For deploy runs, details also show artifact paths for:
- `notes.md`
- `episode_outcomes.csv`
- `episode_outcomes_summary.csv`

## 4) History Actions

- `Open Artifact Folder`
- `Open command.log`
- `Copy Command`
- `Rerun`

Rerun behavior:
- Confirms with command preview dialog.
- Replays shell history entries through shell rerun path.
- Replays pipeline runs through pipeline rerun path.

## 5) Deploy Outcome + Notes Editor

This panel only appears for deploy rows.

Fields:
- `Episode` selector
- `Status` (`success`, `failed`, `unmarked`)
- `Tags` (CSV input)
- `Episode note`
- `Deployment overall notes` (multiline)

Buttons:
- `Save Episode Edit`
- `Save Deployment Notes`
- `Open notes.md`

## What Saving Deploy Edits Does

When you save deploy edits, app writes:
1. updated `metadata.json`
2. generated/updated `notes.md`
3. generated/updated:
   - `episode_outcomes.csv`
   - `episode_outcomes_summary.csv`

Normalization behavior:
- Missing episodes up to `total_episodes` are represented as `unmarked`.
- `pending` is normalized to `unmarked`.
- Tags are normalized/deduplicated.

## Example Workflow: Annotating a Deploy Run

1. Open `History`.
2. Filter `Mode = deploy`.
3. Select latest deploy run.
4. In `Deploy Outcome + Notes Editor`:
   - choose episode number
   - set status (`success`, `failed`, `unmarked`)
   - add tags and note if needed
   - click `Save Episode Edit`
5. Add global summary in `Deployment overall notes`.
6. Click `Save Deployment Notes`.
7. Click `Open notes.md` to inspect rendered notes file.

## What You Might See

Info/errors:
- `Select a history row first.`
- `Selected history entry has no command text.`
- `Could not load metadata for this run.`
- `Episode must be an integer.`
- `Episode must be greater than zero.`
- `Status must be success, failed, or unmarked.`
- `Saved episode <n>: success.`
- `Saved deployment notes.`

Rerun dialog title:
- `Confirm Rerun`

Deploy-only info:
- `Notes file is only available for deploy runs.`

## Notes

- If unreadable metadata files exist, History logs a warning and skips them.
- Rerun uses command argv when available; otherwise parses legacy command text.
