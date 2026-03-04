# Upstream Bridge Guide

Use this workflow when you find LeRobot compatibility deltas that should be reported upstream.

## Goal

Create reproducible, privacy-safe artifacts so upstream maintainers can verify wrapper/LeRobot behavior differences quickly.

## Bridge Workflow

1. Reproduce the failure with the wrapper.
2. Export support bundle:
   - `python3 robot_pipeline.py support-bundle --run-id latest --output ./support-bundle.zip`
3. Capture capability probe:
   - `python3 robot_pipeline.py compat --json`
4. Capture diagnostics:
   - `python3 robot_pipeline.py doctor --json`
5. File an upstream issue with:
   - LeRobot version and Python version
   - failing command (from bundle `metadata.json`)
   - first failure code/details
   - compatibility probe output
   - sanitized support bundle attachment (or key excerpts)

## Suggested Issue Template Fields

- environment (`os`, `python`, `lerobot_version`)
- expected behavior
- actual behavior
- exact failing CLI command
- `--help` output differences for relevant entrypoint
- wrapper compatibility fallback notes (if any)

## Notes

- No telemetry is collected automatically.
- Bundle sharing is explicit and local-first.
- Redaction defaults hide home paths and token-like values.
