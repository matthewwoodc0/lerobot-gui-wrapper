# GA Validation Guide

Use this guide before calling a release "community GA".

CI verifies compatibility probing and tooling only; workflow PASS status is granted only after the GA manual hardware gate.

## 1) CI Status (Validated Current + N-1)

The repository now includes CI checks for the validated current track (`0.5.x`) and validated N-1 track (`0.4.x`):

- workflow: `.github/workflows/compat-smoke.yml`
- jobs:
  - quality matrix (`pytest`, `ruff`, `mypy` on Ubuntu/macOS and Python 3.12)
  - LeRobot validated-track smoke (resolves the exact patch versions for the configured tracks, then runs `compat --json` and `doctor --json`)

GA gate:

1. CI green on `main`.
2. Smoke reports uploaded (`compat-<version>.json`, `doctor-<version>.json`).
3. CI results are treated as probe/tooling evidence only, not workflow PASS evidence.

## 2) Rollout Flags

Set these in `~/.robot_config.json` for staged rollout:

```json
{
  "diagnostics_v2_enabled": true,
  "compat_probe_enabled": true,
  "support_bundle_enabled": true
}
```

Recommended GA setting: all `true`.

## 3) Manual Hardware Matrix

Run at least one pass for each row:

| OS | Cameras | Robot Layout |
|---|---|---|
| macOS | 1 camera | single follower + leader |
| macOS | 2 cameras | standard lab |
| Linux | 2 cameras | standard lab |
| Linux | 3+ cameras | multi-camera lab |

For each row:

1. Set config (ports, robot IDs, camera schema, calibration paths).
2. Run:
   - `python3 robot_pipeline.py doctor --json > doctor.json`
   - `python3 robot_pipeline.py compat --json > compat.json`
3. Validate:
   - `doctor.summary.fail_count == 0`
   - compatibility probe resolves expected entrypoints/flags
4. Execute one real workflow:
   - `teleop` session start/stop
   - one short `record`
   - one `deploy` eval run
5. Export support bundle for traceability:
   - `python3 robot_pipeline.py support-bundle --run-id latest --output ./support-bundle.zip`

## 4) Release Decision Checklist

Mark release ready only when all are true:

1. CI workflow green (quality matrix + validated-track smoke).
2. Manual hardware matrix complete with no blocker failures.
3. `doctor --json` FAIL count is zero on validation machines.
4. Support bundle generation succeeds on failed and successful runs.
5. Compatibility matrix updated with date and notes.
