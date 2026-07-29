# Transition and Community Upgrade Plan

## Purpose

This plan defines two separate bodies of work:

1. Finish the PySide6 and current-LeRobot transition.
2. Upgrade LeRobot Pipeline Manager into a useful community operations tool.

The transition is the first release gate. Community features must not hide an
unreliable installation, an untested GUI, or an outdated LeRobot interface.

This is a living plan. Update the status and evidence after each completed
milestone.

## Product Direction

LeRobot Pipeline Manager is a local-first research and operations companion to
LeRobot. It is not a replacement for LeLab.

LeLab is the primary beginner interface for SO-101 setup, teleoperation,
recording, training, and deployment. Pipeline Manager should focus on repeated
lab work:

- named rig configuration
- preflight diagnostics
- safe command review
- recoverable workflow queues
- dataset quality checks
- reproducible training jobs
- experiment comparison
- rollout failure capture
- artifact lineage
- support bundles

The core product loop is:

```text
Dataset QA -> Train -> Simulate -> Rollout -> Label failures
     ^                                           |
     +------------ Corrections / HIL ------------+
```

## Observed Baseline

Review date: 2026-07-29.

- The source migration from Tkinter to PySide6 is mostly complete.
- The transition branch runs the Qt tests with an offscreen platform.
- The final isolated full test run reports `594 passed` and no skipped tests.
- The compatibility policy now covers LeRobot `0.6.x` and `0.5.x`.
- LeRobot `0.6.0` is the current stable release at this review.
- The transition branch adds the `lerobot-pipeline-manager` installed command.
- The wheel includes the required icons, schemas, guides, and the `py.typed`
  marker.
- An isolated wheel install runs the installed CLI and starts the offscreen GUI
  from outside the source checkout.
- A fake runtime covers command preview, cancellation, record, train, rollout,
  and artifact history.
- The fake runtime does not yet cover simulation evaluation, queue recovery, or
  support-bundle inspection.
- The repository remains an alpha. The software release-candidate checks and
  the real hardware checks are not complete.
- Real hardware validation is incomplete.

Do not treat this baseline as permanent. Confirm the current upstream release
before each compatibility update.

## Status Terms

Use these terms in milestone updates:

- `NOT STARTED`: No implementation evidence exists.
- `IN PROGRESS`: Implementation exists, but one or more exit checks remain.
- `BLOCKED`: A named external dependency or user action prevents progress.
- `DONE`: All listed checks pass and evidence is recorded.

## Workstream A: Finish the Transition

### A1. Make the PySide6 Runtime Reliable

Priority: P0.

Status: `IN PROGRESS`.

Required work:

- Reproduce the Qt platform initialization failure in a clean environment.
- Fix the root cause or provide a supported environment repair.
- Launch and close the GUI on macOS.
- Run headless GUI tests with a supported Qt backend.
- Fail CI when the expected GUI tests are skipped.
- Test page navigation, resize behavior, dialogs, terminal handling,
  cancellation, and clean shutdown.
- Remove dead Tkinter modules and unused migration aliases.
- Preserve required public constructors and launch entry points.
- Finish the shared dialog and form migrations listed in
  `Resources/ui-layout-style-standard.md`.

Exit checks:

- PySide6 imports and initializes in a clean supported environment.
- The GUI opens and closes without a Qt plugin error.
- The GUI test suite executes instead of using a mass skip.
- No active Tkinter runtime path remains.
- Functional checks pass before visual cleanup is called complete.

### A2. Support the Current LeRobot Release

Priority: P0.

Status: `IN PROGRESS`.

Required work:

- Confirm the current stable LeRobot release from official sources.
- Update the validated current and N-1 tracks.
- Add current-release help fixtures for each supported command.
- Probe installed entry points and flags instead of relying only on version
  numbers.
- Support current commands for:
  - calibration
  - teleoperation
  - recording
  - replay
  - training
  - dataset visualization
  - simulation evaluation
  - policy rollout
- Use `lerobot-rollout` for current deployment when available.
- Keep a tested fallback for the supported older deployment path.
- Update command builders, preflight checks, queue recipes, artifacts, History,
  Experiments, CI, and documentation together.

Exit checks:

- Current and N-1 capability probes pass in isolated environments.
- Generated commands match the installed `--help` output.
- Unsupported flags are omitted or reported.
- Current rollout metadata reaches History and Experiments.
- The compatibility matrix states the tested limits.

### A3. Make Installation and Launch Self-Contained

Priority: P0.

Status: `IN PROGRESS`.

Required work:

- Add an installed command such as `lerobot-pipeline-manager`.
- Keep `python3 robot_pipeline.py` as a source-checkout compatibility path.
- Include icons, schemas, and required resources in the wheel.
- Remove runtime assumptions that `.git`, `Resources/`, or
  `robot_pipeline.py` must be beside the installed package.
- Make macOS and Linux launchers use the installed package entry point.
- Detect missing LeRobot feature extras and provide exact fix commands.
- Add a zero-hardware demo mode for evaluation and contributor work.
- Test the wheel from a directory outside the repository.

Exit checks:

- A fresh editable install launches.
- A built wheel installs and launches in a clean environment.
- The installed command works outside the checkout.
- Launcher installation does not depend on a movable source path.
- First-run diagnostics identify missing dependencies clearly.

### A4. Prove the Core Software Workflow

Priority: P0.

Status: `IN PROGRESS`.

Build a safe fake-runtime integration test. It must use the same services as the
GUI.

Required sequence:

1. Install and launch.
2. Create or load a configuration.
3. Run setup checks and Doctor.
4. Detect LeRobot capabilities.
5. Preview commands.
6. Start and cancel a fake teleop run.
7. Complete a fake record run.
8. Complete a fake training run.
9. Complete a fake simulation evaluation.
10. Complete a fake rollout or supported deployment fallback.
11. Write artifacts and lineage metadata.
12. Display runs in History and Experiments.
13. Recover an interrupted queue.
14. Create and inspect a support bundle.

Exit checks:

- The complete sequence passes in CI.
- Tests use temporary directories and sanitized fake data.
- The test does not create a separate demonstration-only implementation.
- No test can move real hardware.

### A5. Complete Release Documentation and Evidence

Priority: P0.

Status: `IN PROGRESS`.

Required work:

- Rewrite Quick Start from a fresh-user test.
- Update the installation dependency model.
- Separate software evidence from hardware evidence.
- Update the compatibility matrix and relevant workflow guides.
- Update screenshots only after the final layout is stable.
- Record exact validation commands and results.
- Use a release-candidate label until the hardware gate passes.

Transition exit decision:

The transition is `DONE` only when:

- clean installation passes
- wheel installation passes
- GUI launch and shutdown pass
- GUI tests execute
- the full test suite passes
- Ruff passes
- targeted mypy checks pass
- Git diff checks pass
- current and N-1 compatibility smoke tests pass
- the fake-runtime workflow passes
- documentation matches observed behavior

## Workstream B: Upgrade the Community Tool

Workstream B begins after the Workstream A software gates pass. Small design
preparation can happen earlier, but broad implementation must not delay the
transition.

### B1. Dataset Health and Provenance

Priority: P0.

Community value: Very high.

Build a Dataset Health Report that can answer:

- Are video files present and readable?
- Do frame counts and timestamps agree?
- Are control frequencies stable?
- Are camera features missing or inconsistent?
- Do state and action features match the selected policy?
- Are joints stuck, constant, or outliers?
- Are episode lengths and task labels credible?
- Which episodes need review or removal?
- Which exact Hub revision will training use?

Required outputs:

- pass, warning, and failure checks
- episode-level results
- camera thumbnails
- action and state plots
- schema comparison
- revision and provenance record
- safe copy-on-write repair options
- exportable JSON report

Do not duplicate upstream dataset edit commands. Add inspection, safety,
explanation, and provenance around them.

### B2. Managed Training Jobs

Priority: P0.

Community value: Very high.

Replace the concept of "a command that ran" with a durable training job.

Each job must capture:

- dataset repository and immutable revision
- policy and resolved configuration
- LeRobot and Pipeline Manager versions
- source Git commit when available
- Python and dependency snapshot
- device and VRAM
- seeds
- checkpoints
- metrics
- final state and failure category
- downstream evaluation links

Planned job backends:

1. local subprocess
2. Hugging Face Jobs
3. Slurm
4. optional SSH-managed execution

Planned operations:

- resource and VRAM preflight
- short smoke or overfit run
- start
- attach
- stream logs
- cancel
- retry
- resume from a real checkpoint
- retain or prune checkpoints
- run bounded parameter comparisons
- promote a checkpoint to evaluation

The current Training guide and Qt behavior must use one consistent contract.

### B3. Rollout, HIL, and Failure Capture

Priority: P0.

Community value: Very high.

Expose supported upstream rollout strategies as clear workflows:

- base
- episodic
- sentry
- highlight
- DAgger or other human-intervention flow
- real-time chunking when supported

Connect each rollout to:

- source training run
- exact checkpoint
- rig snapshot
- camera configuration
- output dataset
- success and failure labels
- intervention segments
- support bundle
- follow-up training recipe

Safety requirements:

- explicit arm and disarm state
- physical-action confirmation
- port lock checks
- calibration snapshot comparison
- model and camera feature compatibility
- clear cancel behavior
- no automatic hardware restart after interruption

### B4. Simulation and Benchmark Manager

Priority: P1.

Community value: High when connected to real experiments.

Do not build a simulator. Integrate supported LeRobot evaluation tools,
benchmark packages, EnvHub environments, and official containers.

Required work:

- discover installed evaluation backends
- report missing dependencies
- provide tested installation recipes
- launch supported `lerobot-eval` jobs
- normalize success, reward, duration, and failure metrics
- compare checkpoints across seeds
- save videos and environment configuration
- connect simulation results to training and hardware rollout results

Suggested sequence:

1. Add one fast smoke benchmark.
2. Add one meaningful manipulation benchmark.
3. Add a descriptor contract for more benchmark adapters.

Simulation work is complete only when results are comparable and reproducible.
A generic command form alone is not enough.

### B5. Headless CLI and Recipe Runner

Priority: P1.

Community value: High.

Expose the same application services through a stable command-line interface.

Target commands:

```text
lpm doctor --json
lpm dataset inspect owner/dataset
lpm train run recipe.yaml
lpm jobs list
lpm jobs logs JOB_ID --follow
lpm jobs cancel JOB_ID
lpm eval run CHECKPOINT --benchmark BENCHMARK
lpm rollout preview recipe.yaml
lpm queue status
lpm experiments compare RUN_A RUN_B
lpm support-bundle latest
```

Requirements:

- stable exit codes
- JSON output
- non-interactive operation
- the same validation and execution services as the GUI
- safe recipe validation before execution
- no duplicate business logic

### B6. Monitoring TUI

Priority: P2.

Community value: Moderate.

Do not build a full TUI before the headless CLI and job event model are stable.

A later TUI can cover:

- queue state
- job status
- live logs
- resource use
- cancel and retry
- recent failures

Do not duplicate camera setup, large configuration forms, or experiment charts
in the TUI. Keep those tasks in the desktop GUI.

### B7. Capability-Driven Extensions

Priority: P2.

Community value: High after the core contracts stabilize.

Support descriptors for:

- robots
- teleoperators
- cameras and sensors
- policies
- simulation benchmarks
- job backends
- workflow recipes

Each descriptor should declare:

- required fields
- upstream entry points
- supported flags
- dependency extras
- preflight checks
- safety requirements
- artifact metadata

Community profiles should be reviewable data. Do not execute arbitrary profile
code by default.

## Planned Release Sequence

### Version 0.2: Reliable Current Runtime

- complete PySide6 transition
- support current LeRobot
- add installed command
- prove clean installation
- add demo mode

### Version 0.3: Research Data and Training

- add Dataset Health Report
- add durable training jobs
- support Hugging Face Jobs
- strengthen experiment lineage

### Version 0.4: Rollout and Improvement Loop

- support current rollout strategies
- add intervention and failure capture
- link corrections to follow-up training

### Version 0.5: Simulation and Comparison

- add benchmark manager
- compare checkpoints across seeds
- compare simulation and hardware results

### Version 0.6: Lab Operations

- add headless recipe CLI
- add Slurm or SSH job management
- add a small monitoring TUI if evidence supports it

### Version 1.0: Community Release

- complete the real hardware matrix
- stabilize extension contracts
- publish supported installation artifacts
- publish complete compatibility and validation evidence

## Hardware Validation Gate

Software validation cannot prove hardware safety or usefulness.

Before a community release, validate:

- Doctor with no unresolved failure
- teleop start, response, cancel, and shutdown
- camera preview
- one short record run
- replay of a known episode
- one safe policy rollout
- History and Experiments linkage
- support-bundle export

Run the applicable matrix across:

- macOS and Linux
- one and multiple cameras
- current and N-1 LeRobot tracks
- each claimed robot and teleoperator combination

Do not claim a row as passed when only command generation or CI probes ran.

## Scope Controls

Do not:

- compete with LeLab on basic onboarding
- build a custom trainer
- build a custom simulator
- add a full TUI before the CLI contract exists
- support every policy or robot with hard-coded forms
- rewrite broad architecture without a measured blocker
- let cosmetic cleanup delay runtime correctness
- claim hardware validation from fake data or offscreen media

## Evidence Required for Every Milestone

Record:

- branch and commit
- changed behavior
- exact test commands and results
- GUI test pass and skip counts
- LeRobot versions tested
- installation proof
- screenshots when relevant
- hardware proof or the exact missing gate
- remaining known debt

A green backend test suite alone is not proof that the desktop tool works.
