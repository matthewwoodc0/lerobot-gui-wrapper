# GA Validation Guide

Use this guide before calling a release community GA.

Automated tests verify compatibility probing and tooling only. Workflow PASS on real hardware needs this manual hardware gate.

## Software-validated release candidate

Until the hardware matrix below is complete, call the build a **software-validated release candidate**. Do not claim GA.

Software evidence includes:

- full unit/integration suite (including Qt offscreen GUI tests)
- fake-runtime end-to-end operator loop
- current + N-1 LeRobot smoke (`0.6.x` and `0.5.x`)
- clean-venv wheel install and `lerobot-pipeline-manager` launch outside a source checkout

## 1) CI Status (Validated Current + N-1)

CI checks the validated current track (`0.6.x`) and validated N-1 track (`0.5.x`):

- workflow: `.github/workflows/compat-smoke.yml`
- jobs:
  - quality matrix (`pytest` with `QT_QPA_PLATFORM=offscreen`, `ruff`, `mypy` on Ubuntu/macOS and Python 3.12)
  - LeRobot validated-track smoke (resolves patch versions, then runs `compat --json` and `doctor --json`)
  - GUI skip guard: fails if too many tests skip while PySide6 is installed

GA gate:

1. CI green on `main`.
2. Smoke reports uploaded (`compat-<version>.json`, `doctor-<version>.json`).
3. CI results are probe/tooling evidence only, not hardware workflow PASS evidence.

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
   - `lerobot-pipeline-manager doctor --json > doctor.json`
   - `lerobot-pipeline-manager compat --json > compat.json`
3. Validate:
   - `doctor.summary.fail_count == 0`
   - compatibility probe resolves expected entrypoints/flags
   - preferred deploy path is `rollout` on 0.6.x when available
4. Execute one real workflow (get explicit user approval before any physical motion):
   - `teleop` session start/stop and clean shutdown
   - camera preview
   - one short `record`
   - one safe `rollout` or legacy deploy fallback
5. Export support bundle for traceability:
   - `lerobot-pipeline-manager support-bundle --run-id latest --output ./support-bundle.zip`

## 4) Release Decision Checklist

Mark release ready only when all are true:

1. CI workflow green (quality matrix + validated-track smoke).
2. Manual hardware matrix complete with no blocker failures.
3. `doctor --json` FAIL count is zero on validation machines.
4. Support bundle generation succeeds on failed and successful runs.
5. Compatibility matrix updated with date and notes.
