# LeRobot GUI Wrapper Compatibility Matrix

This project reads its track policy from `robot_pipeline_app/compat_policy.py`. The docs are checked against that source in the test suite.

Automated tests verify compatibility probing and tooling only. Workflow PASS on real hardware needs the manual hardware gate in `ga-validation.md`.

## Validated Tracks

| Track | CI probe/tooling | Workflow PASS status | Status date | Notes |
|---|---|---|---|---|
| validated current track (`0.6.x`) | PASS (software) | Requires manual hardware gate | 2026-07-29 | Primary target. Prefer `lerobot-rollout` for deploy. |
| validated N-1 track (`0.5.x`) | PASS (software) | Requires manual hardware gate | 2026-07-29 | Supported with runtime entrypoint and flag detection. Legacy deploy uses record + `--policy.path`. |

## Validation Process

1. Run CI workflow `.github/workflows/compat-smoke.yml` (quality matrix + validated-track smoke).
2. Use Python 3.12+ for wrapper validation.
3. Run `lerobot-pipeline-manager doctor` in a real LeRobot environment.
4. Run `lerobot-pipeline-manager compat` to capture entrypoint and flag capability output.
5. Validate command generation for record, teleop, train, replay, calibrate, dataset visualization, evaluation, and rollout/deploy against installed `--help` output.
6. Mark workflow PASS only after the manual hardware matrix is complete.
7. Log compatibility deltas and update this matrix.

## Capability Probe

```bash
lerobot-pipeline-manager compat --json
# developer fallback:
python3 robot_pipeline.py compat --json
```

This reports:

- detected LeRobot version
- record, train, teleop, calibrate, replay, and rollout entrypoints
- preferred deploy path (`rollout` or `record_policy_path`)
- supported flags for record, train, sim-eval, and rollout when help is available
- fallback notes when configured flags are unsupported

## Install Extras (LeRobot 0.6)

| Extra | Typical use |
|---|---|
| `core_scripts` | record, replay, calibrate, teleoperate |
| `training` | train policies |
| `feetech` | SO-101 Feetech motors |
| `dataset_viz` / `viz` | dataset visualization |

Example:

```bash
pip install 'lerobot[core_scripts,training,feetech]'
```

## Out-of-Range Versions

Versions older than N-1 are best-effort only. If you must run them:

1. Run Doctor first.
2. Use command preview and compare flags against installed module `--help` output.
3. Capture failures with artifacts and include them in issue reports.
