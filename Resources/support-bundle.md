# Support Bundle Guide

This document defines what a support bundle should contain for reproducible community bug reports.

## Goal

Enable maintainers and contributors to reproduce failures without requiring access to a specific lab setup.

## Minimum Contents

- `metadata.json` from run artifacts
- `command.log` from run artifacts
- preflight report (`preflight_report.json` + `preflight_report.txt`)
- sanitized config snapshot
- compatibility snapshot (`compatibility_snapshot.json`)
- environment probe output (`environment_probe.json`)

## CLI Usage

Generate a bundle for the latest run:

```bash
python3 robot_pipeline.py support-bundle --run-id latest --output ./support-bundle-latest.zip
```

Generate a bundle for a specific run id:

```bash
python3 robot_pipeline.py support-bundle --run-id deploy_20260304_120011 --output ./support-bundle-deploy.zip
```

## Privacy Defaults

- redact home-directory prefixes where possible
- do not include tokens or secrets
- do not include private datasets or model weights

## Reporting Checklist

1. Include the exact wrapper commit hash.
2. Include the LeRobot version (`pip show lerobot` or module metadata).
3. Include OS and Python runtime details.
4. Include the first traceback/error line from `command.log`.
