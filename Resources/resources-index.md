# Resources Index

All documentation for the LeRobot Pipeline Manager. Organized by what you're trying to do.

---

## I'm setting up for the first time

Start here. Follow in order.

1. **[First-Time Setup Walkthrough](./first-time-setup.md)**
   Full walkthrough from a cold machine to a verified working stack: environment check, config, teleop, cameras, and a first record run. Covers common failures at each step.

2. **[Config Tab Guide](./config-tab-guide.md)**
   Every config field explained. Named rigs for multi-robot labs. Community profile import/export. Desktop launcher install.

3. **[Teleop Guide](./teleop-tab-guide.md)**
   Use Teleop as your first real hardware test. This guide explains what to look for, what port fingerprint warnings mean, and how to recover from calibration failures.

4. **[Compatibility Matrix](./compatibility-matrix.md)**
   Which LeRobot versions are validated and what falls back gracefully.

---

## I'm collecting training data

- **[Record Tab Guide](./record-tab-guide.md)**
  Dataset naming (auto-managed, collision detection), camera setup, HF upload, manual upload for existing datasets, and dataset browser. Includes a full end-to-end walkthrough and a real command example.

- **[Hardware Operations Guide](./hardware-operations-guide.md)**
  Named rigs, Replay (verify data quality on hardware), Motor Setup, and local workflow queues (Record → Upload, etc.).

---

## I'm training and evaluating a policy

- **[Training Tab Guide](./training-tab-guide.md)**
  HIL adaptation from a base model + intervention dataset. srun cluster wrapping. Checkpoint discovery and how results flow to Experiments.

- **[Deploy Tab Guide](./deploy-tab-guide.md)**
  Model and checkpoint selection, eval naming (`eval_` convention), preflight diagnostics, and how to read results in History and Experiments.

- **[Experiments Tab Guide](./experiments-tab-guide.md)**
  Cross-run comparison for train, deploy, and sim-eval runs. Parsed metrics, checkpoint browsing, WandB deep-links, and launching follow-up evals from a checkpoint.

---

## I'm reviewing and annotating past runs

- **[History Tab Guide](./history-tab-guide.md)**
  Filter runs by mode/status, view transcripts, replay from history, annotate deploy episode outcomes (success/failed), add tags and notes, export `episode_outcomes.csv`.

---

## I need to troubleshoot something

- **[Error Catalog](./error-catalog.md)**
  Every preflight diagnostic code (`ENV-*`, `SER-*`, `CAM-*`, `CAL-*`, `CLI-*`, etc.) mapped to an actionable fix.

- **[Support Bundle Guide](./support-bundle.md)**
  How to export a redacted diagnostics archive from the Config tab for filing bug reports.

- **[Upstream Bridge Guide](./upstream-bridge.md)**
  How the wrapper integrates with LeRobot — entrypoint detection, version fallback behavior, and how to handle edge cases when your LeRobot version doesn't match the expected API.

---

## I'm managing hardware or a lab

- **[Hardware Operations Guide](./hardware-operations-guide.md)**
  Named rigs, Motor Setup, Replay, and local workflow queues. Includes the recommended lab setup flow.

- **[Community Profiles Guide](./community-profiles.md)**
  Share portable config profiles across machines or team members.

---

## I'm contributing to the project

See the **[Developer Guide](../docs/DEVELOPER.md)** for architecture, subsystem ownership, test commands, and the definition of done.

---

## All Guides

| Guide | Description |
|---|---|
| [First-Time Setup](./first-time-setup.md) | Cold-machine-to-working walkthrough |
| [Compatibility Matrix](./compatibility-matrix.md) | Validated LeRobot versions and fallback behavior |
| [Config Tab Guide](./config-tab-guide.md) | Config fields, rigs, profiles, launcher |
| [Teleop Guide](./teleop-tab-guide.md) | Teleop workflow and hardware verification |
| [Record Tab Guide](./record-tab-guide.md) | Dataset recording and HF upload |
| [Deploy Tab Guide](./deploy-tab-guide.md) | Model eval and results |
| [Training Tab Guide](./training-tab-guide.md) | HIL training and checkpoints |
| [Experiments Tab Guide](./experiments-tab-guide.md) | Cross-run comparison |
| [History Tab Guide](./history-tab-guide.md) | Run log and outcome annotation |
| [Hardware Operations Guide](./hardware-operations-guide.md) | Replay, Motor Setup, named rigs, workflows |
| [Error Catalog](./error-catalog.md) | Preflight error codes and fixes |
| [Community Profiles](./community-profiles.md) | Portable config sharing |
| [Support Bundle Guide](./support-bundle.md) | Debug export for bug reports |
| [Upstream Bridge Guide](./upstream-bridge.md) | LeRobot integration details |
