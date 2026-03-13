# Qt UI Layout and Style Standard

This document is the canonical reference for the Qt UI in this repository. Any future work in `robot_pipeline_app/gui_qt_*.py` should be checked against this file before adding, restructuring, or restyling a module.

The goal is not to invent a new design system. The goal is to describe the one that already exists, make it explicit, and reduce future drift.

## Scope

This standard covers:

- window layout and page composition
- color and typography choices
- spacing, card structure, and button hierarchy
- shared UI architecture and code patterns
- user experience rules
- UI mapping from sidebar section to implementation module
- current inconsistencies already present in the codebase

## Canonical Source Files

These files define the current UI contract more than any others:

- `robot_pipeline_app/gui_qt_app.py`
- `robot_pipeline_app/gui_qt_theme.py`
- `robot_pipeline_app/app_theme.py`
- `robot_pipeline_app/gui_qt_common.py` - shared canonical Qt card and form-layout helpers
- `robot_pipeline_app/gui_qt_ops_base.py`
- `robot_pipeline_app/gui_qt_page_base.py`
- `robot_pipeline_app/gui_qt_output.py`
- `robot_pipeline_app/gui_qt_dialogs.py`
- `robot_pipeline_app/gui_qt_camera.py`
- `robot_pipeline_app/dataset_operations.py` - canonical non-Qt dataset replay/edit/merge service used by visualizer workflows

`_build_card()` and `_InputGrid` are sourced from `robot_pipeline_app/gui_qt_common.py`; the base classes consume those shared helpers rather than defining their own copies.

## Shell Layout

The application shell is built in `robot_pipeline_app/gui_qt_app.py` and is intentionally stable:

```text
QMainWindow
  root HBoxLayout
    Sidebar (full) or SidebarRail (collapsed)
    ContentSurface
      Vertical QSplitter
        WorkspaceWindow
          PaneHeader
            section stage/title/subtitle
            compact Hugging Face auth status
          QStackedWidget
            per-section page wrapped in QScrollArea
        TerminalWindow
          PaneHeader
          terminal tabs
```

### Main shell rules

- The sidebar is the primary navigation surface.
- The workspace header always communicates current context.
- The page body is scrollable and vertically stacked.
- The terminal is first-class, not secondary. It is a peer surface under the workspace via a vertical splitter.
- Sidebar collapse and terminal visibility are treated as shell-state, not page-state.

### Internal shell helpers

- `QtPreviewWindow` remains the composition root for the shell, section registry, theme switching, sidebar visibility, and run-controller wiring.
- `_WorkspacePulseController` owns the workspace eyebrow pulse timer and the pulse reset/tick behavior for active runs.
- `_HuggingFaceStatusController` owns Hugging Face status presentation shaping plus the chip/summary refresh path used by the workspace header.
- `_TerminalTabManager` owns terminal-tab add/close/rename bookkeeping and emits tab-change / close-request events back to `QtPreviewWindow`.
- These helpers stay in `robot_pipeline_app/gui_qt_app.py` so shell behavior remains local to the shell module even when `QtPreviewWindow` delegates the details.

### Fixed shell dimensions currently in use

- Full sidebar width: `280`
- Collapsed sidebar rail width: `56`
- Shell outer margin: `16`
- Main content surface padding: `16`
- Workspace and terminal pane padding: `18`
- Splitter handle width: `14`

These values are part of the current visual language and should not be changed casually.

## UI Mapping

The section registry lives in `robot_pipeline_app/gui_qt_app.py` as `_QT_SECTIONS`. The user-visible mapping is:

| Sidebar Section | Primary Class | Type | Main UI Shape |
| --- | --- | --- | --- |
| Record | `RecordOpsPanel` | core workflow | form card + camera workspace + output |
| Replay | `ReplayOpsPanel` | core workflow | form card + output |
| Deploy | `DeployOpsPanel` | core workflow | form card + model browser + camera workspace + helper dialog + output |
| Teleop | `TeleopOpsPanel` | core workflow | snapshot card + camera workspace + form card + output |
| Motor Setup | `MotorSetupOpsPanel` | core workflow | form card + output |
| Train | `TrainOpsPanel` | core workflow | form card + output |
| Workflows | `QtWorkflowsPage` | secondary page | stacked recipe cards + workflow table + output |
| Experiments | `QtExperimentsPage` | secondary page | filter card + tables + launch cards + output |
| Config | `QtConfigPage` | secondary page | grouped config cards + camera schema editor + actions + output |
| Visualizer | `QtVisualizerPage` | secondary page | source browser panel + video gallery panel + dataset tools + selection details + output |
| History | `QtHistoryPage` | secondary page | filters + run table + workspace links + notes editor + output |

## Color Standard

The app uses one warm accent family across both themes. Gold/orange is the identity color, not blue and not purple.

### Dark palette

| Token | Value | Use |
| --- | --- | --- |
| `bg` | `#090909` | window background |
| `panel` | `#121212` | outer shell panels |
| `surface` | `#1b1b1b` | cards, inputs, terminal tabs |
| `surface_alt` | `#252525` | scrollbar handles / minor contrast |
| `surface_elevated` | `#2a2a2a` | reserved elevated surface |
| `scrollbar_handle` | `#3a3a3a` | scrollbar thumb fill |
| `header` | `#0e0e0e` | terminal output area / darker emphasis |
| `border` | `#303030` | all standard borders |
| `text` | `#f2f2f2` | primary text |
| `muted` | `#8f8f8f` | helper text |
| `accent` | `#f0a500` | primary action and emphasis |
| `accent_soft` | `#4d390e` | status chip fill / soft highlight |
| `running_dim` | `#7a5200` | pulse dim state for active-run indicators |
| `error` | `#ef4444` | destructive state |
| `success` / `ready` | `#22c55e` | success / ready state |

### Light palette

| Token | Value | Use |
| --- | --- | --- |
| `bg` | `#f4f5f7` | window background |
| `panel` | `#ffffff` | outer shell panels |
| `surface` | `#eef1f5` | cards and fields |
| `surface_alt` | `#e3e8ef` | minor contrast |
| `surface_elevated` | `#d8e0ea` | reserved elevated surface |
| `scrollbar_handle` | `#b8c2cf` | scrollbar thumb fill |
| `header` | `#e9edf2` | terminal and darkened panel substitute |
| `border` | `#c7d0db` | borders |
| `text` | `#12161d` | primary text |
| `muted` | `#526070` | secondary text |
| `accent` | `#ca7a00` | primary action and emphasis |
| `accent_soft` | `#f2dcc1` | status chip fill / soft highlight |
| `running_dim` | `#e1b36e` | pulse dim state for active-run indicators |
| `error` | `#cc3c3c` | destructive state |
| `success` / `ready` | `#1f9d55` | success / ready state |

### Color usage rules

- Primary actions use `AccentButton`.
- Destructive actions use `DangerButton`.
- Low-importance helper text uses `MutedLabel`.
- Section headers and shell eyebrows use the accent color.
- Selection backgrounds use the accent color with black text for contrast.
- Avoid adding new one-off highlight colors unless the shared palette is extended centrally in `app_theme.py`.

## Typography Standard

The UI font is `Inter`. The mono font is `JetBrains Mono`. All widgets inherit the UI font unless they are output/code fields.

| Object / Surface | Style |
| --- | --- |
| base widgets | `10pt` |
| `BrandLabel` | `22pt`, bold, accent |
| `PaneEyebrow` | `8.5pt`, bold, uppercase, accent |
| `PaneTitle` | `17pt`, bold |
| `PaneSubtitle` | `9.5pt`, muted |
| `DialogTitle` | `16pt`, bold |
| `PageTitle` | `19pt`, bold |
| `FormLabel` | `9pt`, bold, muted |
| `SectionMeta` | `9pt`, bold, uppercase, accent |
| nav title | `11.5pt`, extra bold |
| nav meta | `8.5pt`, semibold |
| output / logs | `JetBrains Mono`, `10pt` |

### Typography rules

- Use object names, not ad hoc font overrides.
- Use `FormLabel` for field labels.
- Use `SectionMeta` for card headings and small section headers.
- Use `PaneTitle` and `PaneSubtitle` only for major shell headers.
- Commands, transcripts, and JSON-like output belong in `QPlainTextEdit` with the mono font.

### Button and status interaction states

- `AccentButton` uses `accent_dark` on hover and a darker pressed fill.
- `DangerButton` uses progressively darker red on hover and pressed states.
- `StatusChip` uses a `state` dynamic property (`running`, `error`, `success`) to change its fill, border, and text color contextually.

## Shape and Surface Standard

The UI relies on rounded rectangular surfaces with low-contrast borders.

| Surface | Radius |
| --- | --- |
| `ContentSurface` | `18` |
| `Sidebar` / `SidebarRail` | `16` |
| `WorkspaceWindow` / `TerminalWindow` | `16` |
| `DialogPanel` | `16` |
| `SectionCard` | `14` |
| standard buttons | `10` |
| nav items | `10` |
| status chip | `9` |

### Surface rules

- Major shell surfaces use `panel`.
- Interior panes and cards use `surface`.
- The terminal output area uses `header` to read as denser and more technical.
- Borders are always present and subtle.
- The workspace header uses a compact Hugging Face status block: chip + one-line summary, with longer guidance in a tooltip.
- `NavItem` shows a soft accent tint on hover and full accent fill when selected.
- First-class surfaces should feel restrained and dense enough to scan quickly, not bubbly or oversized.
- Do not introduce flat borderless floating widgets into the main shell unless they are intentionally transient.

## Spacing Standard

The codebase uses a small set of repeated spacing values. These should be treated as tiers:

| Use | Standard |
| --- | --- |
| shell/page outer spacing | `16` |
| major pane padding | `18` |
| compact utility/dialog padding | `12` |
| form horizontal gap | `12` |
| card internal gap | `10` |
| dense control row gap | `8` |
| dense metadata stack gap | `2` to `6` |

### Practical spacing rules

- Default page stacks use `16` between cards.
- Section cards use `16` internal padding when they are primary content.
- Compact helper dialogs or utility panels may use `12` internal padding, but that is the exception, not the default.
- Button rows and inline field groups use `8`.
- Avoid introducing new arbitrary values unless there is a clear reason.

## Shared Architecture and Code Decisions

The UI architecture is intentionally helper-driven.

### Page construction

- Core workflow pages inherit from `_CoreOpsPanel` in `robot_pipeline_app/gui_qt_ops_base.py`.
- Secondary analysis/config pages inherit from `_PageWithOutput` in `robot_pipeline_app/gui_qt_page_base.py`.
- Both patterns produce vertically stacked cards and a hidden-until-needed output area.

### Reusable building blocks

- `_InputGrid` is the standard form layout helper.
- `_build_card()` is the standard card factory.
- `QtRunOutputPanel` is the standard summary/raw-output component.
- `gui_qt_dialogs.py` is the standard dialog scaffolding.
- `QtCameraWorkspace` is the standard camera-preview surface.
- `dataset_operations.py` is the canonical shared service module for dataset replay planning and dataset mutation command preparation.

### Styling contract

- Styling is global and centralized in `build_qt_stylesheet()` in `robot_pipeline_app/gui_qt_theme.py`.
- Widget `objectName` values are the styling API.
- New UI modules should not use inline `setStyleSheet()` calls for normal theme behavior.
- If a new visual primitive is needed, add a named style target to the shared stylesheet rather than styling one widget in isolation.

### Separation of concerns

- Widget code should orchestrate UI, not own business logic.
- Command building belongs in helper modules like `gui_forms.py` and workflow helpers.
- Preflight logic belongs in `checks*.py`.
- History, visualizer, compatibility, and workflow shaping belong in their non-Qt helpers.
- Dataset mutation and replay preparation for the visualizer belong in `robot_pipeline_app/dataset_operations.py`, not inline in page classes.
- Runtime execution flows through `ManagedRunController` and shared hooks, not custom per-page subprocess code.
- Canonical spacing and radius constants are defined in `app_theme.py` as `SPACING_*` and `RADIUS_*` module-level constants. Widget code should import these instead of using magic numbers.
- Long workflow launch methods should prefer a same-file runner helper such as `gui_qt_deploy.py`'s `_DeployWorkflowRunner`: the widget stays focused on form state and UI wiring, while the runner receives explicit inputs/callbacks and owns the step-by-step execution flow.

## UX Standard

The current UX model is operational and local-first. New modules should fit that model.

### Workflow pattern

Use this order when a page launches work:

1. gather inputs
2. allow preview or command inspection
3. run preflight or validation
4. confirm launch
5. stream raw output live
6. summarize the result in-page
7. preserve artifacts in shared history

### Interaction rules

- Keep destructive or cancel actions visible near the primary action row.
- Hide output cards until there is something meaningful to show.
- Use the page output for summaries and structured feedback.
- Use the terminal for raw transcript and shell interaction.
- Persist lightweight UI filters and shell preferences when that improves continuity.
- Prefer word-wrapped explanatory text and no-wrap command/log text.
- Cancel buttons for active runs use `DangerButton`. Dialog dismiss buttons before launch stay neutral.
- The workspace header should answer Hugging Face auth with a compact logged-in / not-logged-in status first, and push setup detail into tooltips.
- The workspace eyebrow label pulses between `accent` and `running_dim` every `600ms` while any workflow run is active, then resets when the run ends.

### Tooltip standard

- Every icon-only button must have a tooltip explaining its action.
- Nav items show their section subtitle as a tooltip.
- Action buttons that are contextually enabled, like `Explain Failure`, should explain when they become useful.
- Tooltips should be short, imperative, and lowercase except for proper nouns.

### User-facing hierarchy

- Sidebar answers "where am I?"
- Workspace header answers "what mode am I in?"
- Card headings answer "what sub-task is this?"
- Output answers "what just happened?"
- Terminal answers "what is happening right now?"

## Standard for New Modules

Any new UI module should follow this checklist:

1. Register the new section in the shell routing layer only once.
2. Choose `_CoreOpsPanel` for run-oriented workflows or `_PageWithOutput` for analysis/config surfaces.
3. Build the page from `SectionCard`s instead of custom unframed widget stacks.
4. Use `_InputGrid` for labeled forms.
5. Use `AccentButton` for the primary call to action.
6. Use `DangerButton` for destructive actions and explicit cancel actions.
7. Reuse `QtRunOutputPanel` or the shared output card pattern instead of inventing a new result surface.
8. Route command building, validation, and runtime execution through the existing helper layers.
9. Persist only meaningful state, and keep it in the existing config patterns.
10. Add any new styles through `gui_qt_theme.py`, not widget-local CSS.
11. Dataset mutation operations belong in `dataset_operations.py`, not inline in page classes.

## Current Discrepancies From This Standard

These inconsistencies already exist in the codebase and should be treated as debt, not precedent.

### 3. Not all dialogs use the shared dialog shell

- `robot_pipeline_app/gui_qt_dialogs.py` defines the canonical dialog panel builder.
- `robot_pipeline_app/gui_qt_deploy.py:65-162` (`_QtModelUploadDialog`) uses its own raw `QDialog` layout.
- `robot_pipeline_app/gui_qt_runtime_helpers.py:24-147` (`QtRunHelperDialog`) also bypasses the shared dialog panel wrapper.
- Result: dialog chrome, padding tiers, and visual framing are not fully consistent. Partially addressed: TODOs mark the remaining migrations.

### 4. Some forms skip `_InputGrid` and `FormLabel` styling

- `robot_pipeline_app/gui_qt_experiments_page.py:212-263` builds the sim-eval checkpoint form with plain `QLabel`s in a raw `QGridLayout`.
- `robot_pipeline_app/gui_qt_visualizer_page.py:152-201` builds the source-browser control row with plain labels instead of shared field labeling.
- `robot_pipeline_app/gui_qt_page_base.py:506-523` uses plain labels in the camera schema controls row.
- Result: label typography and alignment are less consistent than the rest of the app. Partially addressed: TODOs mark the remaining layout migrations.

The standard should be: explicit cancellation and destructive actions use `DangerButton`.

### 7. Compact spacing is used in places that read like first-class surfaces

- `robot_pipeline_app/gui_qt_deploy.py:83-85` uses `14/10` padding and spacing for the model upload dialog.
- `robot_pipeline_app/gui_qt_runtime_helpers.py:50-52` uses `14/12` for a large helper window.

That is acceptable for dense utilities, but these are substantial user-facing panels. The current shell standard is visually clearer when first-class surfaces use the shared dialog/card treatment.

## Recommended Interpretation Going Forward

When there is a conflict between existing code and this document, use this order:

1. shared theme object names and shared layout helpers
2. shell layout conventions in `gui_qt_app.py`
3. helper-driven page architecture in `_CoreOpsPanel` and `_PageWithOutput`
4. this discrepancy list as debt to avoid copying

If you need a new pattern, add it centrally and document it here rather than introducing one more local exception.
