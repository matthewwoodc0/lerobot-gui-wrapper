# LeRobot GUI Wrapper Compatibility Matrix

This project reads its track policy from `robot_pipeline_app/compat_policy.py`, and the docs are checked against that source in the test suite.

CI verifies compatibility probing and tooling only; workflow PASS status is granted only after the GA manual hardware gate.

## Validated Tracks

| Track | CI probe/tooling | Workflow PASS status | Status date | Notes |
|---|---|---|---|---|
| validated current track (`0.5.x`) | PASS | Requires GA manual hardware gate | 2026-03-10 | Primary validation target for current upstream releases. |
| validated N-1 track (`0.4.x`) | PASS | Requires GA manual hardware gate | 2026-03-10 | Supported via entrypoint and flag fallback logic. |

## Validation Process

1. Run CI workflow `.github/workflows/compat-smoke.yml` (quality matrix + validated-track smoke).
2. Use Python 3.12+ for wrapper validation and LeRobot `0.5.x` smoke runs.
3. Run `python3 robot_pipeline.py doctor` in a real LeRobot environment.
4. Run `python3 robot_pipeline.py compat` to capture entrypoint + flag capability probe output.
5. Validate command generation for record/train/deploy/teleop against `--help` in the target LeRobot version.
6. Mark workflow PASS only after the GA manual hardware matrix is complete.
7. Log any compatibility deltas in issue tracker and update this matrix.

## Capability Probe

Use machine-readable probe output for release checks and bug triage:

```bash
python3 robot_pipeline.py compat --json
```

This reports:
- detected LeRobot version
- selected record/train/teleop/calibrate entrypoints
- supported record flags (including policy-path and rename-map forms)
- supported train flags and any missing required train flags
- fallback behavior notes when configured flags are unsupported

## Out-of-Range Versions

Versions older than N-1 are best-effort. If you must run older versions:

1. Run `doctor` first.
2. Use command preview and compare flags against your installed module `--help` output.
3. Capture failures with artifacts and include them in issue reports.
