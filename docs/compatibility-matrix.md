# LeRobot GUI Wrapper Compatibility Matrix

This project targets a **latest + N-1** LeRobot compatibility policy.

## Known Good Matrix

| LeRobot version | Record | Deploy | Teleop | Status date | Notes |
|---|---|---|---|---|---|
| `0.3.x` (latest) | PASS | PASS | PASS | 2026-03-04 | Primary supported track. |
| `0.2.x` (N-1) | PASS | PASS | PASS | 2026-03-04 | Supported via entrypoint and flag fallback logic. |

## Validation Process

1. Run CI workflow `.github/workflows/compat-smoke.yml` (unit tests + latest/N-1 smoke).
2. Run `python3 robot_pipeline.py doctor` in a real LeRobot environment.
3. Run `python3 robot_pipeline.py compat` to capture entrypoint + flag capability probe output.
4. Validate command generation for record/deploy/teleop against `--help` in the target LeRobot version.
5. Log any compatibility deltas in issue tracker and update this matrix.

## Capability Probe

Use machine-readable probe output for release checks and bug triage:

```bash
python3 robot_pipeline.py compat --json
```

This reports:
- detected LeRobot version
- selected record/teleop/calibrate entrypoints
- supported record flags (including policy-path and rename-map forms)
- fallback behavior notes when configured flags are unsupported

## Out-of-Range Versions

Versions older than N-1 are best-effort. If you must run older versions:

1. Run `doctor` first.
2. Use command preview and compare flags against your installed module `--help` output.
3. Capture failures with artifacts and include them in issue reports.
