# Community Profiles

Community profiles are portable hardware/runtime templates that make multi-lab setup faster and safer.

## Profile Scope

- robot role defaults (follower/leader type and IDs)
- serial port naming conventions and guidance
- camera schema
- camera feature mapping guidance for policy compatibility
- optional notes for calibration source layout

Machine-specific ports and calibration file paths are treated as local path fields.
They are only exported with `--include-paths` and only applied on import with
`--apply-paths`.

## CLI Usage

## GUI Usage

- Open `Config`.
- Use `Profiles + Portable Presets`.
- `Import Profile` applies a community profile file into the current GUI config.
- `Export Profile` writes the current GUI config back out as a portable profile.
- `Apply Portable Preset` loads a built-in lab/hardware preset and refreshes robot defaults, camera schema, and setup guidance immediately.

The GUI stores the active profile name in config so Config snapshot/setup guidance can show which lab template is currently shaping robot and camera behavior.

Export a portable profile:

```bash
python3 robot_pipeline.py profile export --output ./lab-profile.yaml
```

Export including local path fields:

```bash
python3 robot_pipeline.py profile export --output ./lab-profile.yaml --include-paths
```

Import a profile (path fields skipped by default):

```bash
python3 robot_pipeline.py profile import --input ./lab-profile.yaml
```

Import and apply path fields:

```bash
python3 robot_pipeline.py profile import --input ./lab-profile.yaml --apply-paths
```

## Camera Schema Example

```json
{
  "camera_schema_json": {
    "wrist": {"index_or_path": 0, "type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5},
    "overhead": {"index_or_path": 1, "type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5},
    "side": {"index_or_path": 2, "type": "opencv", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}
  }
}
```

## Model Feature Rename Example

Use this when runtime camera keys differ from training feature keys.

```json
{
  "rename_map": {
    "observation.images.wrist": "observation.images.camera1",
    "observation.images.overhead": "observation.images.camera2",
    "observation.images.side": "observation.images.camera3"
  }
}
```

## Schema

- schema id: `community_profile.v1`
- reference: `schema/community_profile.v1.yaml`
- parser behavior:
  - accepts JSON-formatted YAML by default
  - accepts full YAML when `pyyaml` is installed
  - rejects unsupported keys with explicit validation errors
