# Qt Parity Matrix

Status meanings:
- `Done`: feature exists in the Qt path with comparable behavior.
- `Partial`: feature exists, but the original Tk interaction model or workflow is still incomplete.
- `Missing`: no meaningful Qt equivalent yet.

## Shell
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Main app shell | Header, status pulse, notebook + output pane | `Partial` | Qt shell now has artifact shortcuts and an interactive lower pane, but a few richer runtime affordances are still simpler than Tk. |
| Terminal toggle | Header toggle for terminal panel | `Done` | Visible toggle now exists in Qt. |
| PTY terminal shell | Interactive terminal shell with activation/send flow | `Done` | Qt lower pane now uses the PTY-backed shell with activation, send, and Ctrl-C support. |
| Open latest artifact | Global shell action | `Done` | Qt shell now exposes an `Open Latest Artifact` shortcut. |

## Shared dialogs and popouts
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Read-only text dialog | `show_text_dialog` | `Done` | Qt dialog foundation added. |
| Editable command dialog | `ask_editable_command_dialog` | `Done` | Qt editable command dialog added. |
| Confirm / cancel text dialog | `ask_text_dialog` | `Done` | Qt confirm dialog added. |
| Action quick-fix dialog | `ask_text_dialog_with_actions` | `Done` | Foundation added; not yet wired to all flows. |
| Run popouts | Deploy outcome tracker, runtime helpers | `Done` | Qt now has deploy/teleop helper dialogs with live progress and controls, and deploy tracker edits are persisted back into the run artifact. |

## Record
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Preview command dialog | Modal preview dialog | `Done` | Qt preview now opens a modal dialog. |
| Editable launch confirmation | Editable command dialog before run | `Done` | Qt record run now uses editable confirmation. |
| Preflight review dialog | Modal preflight confirmation | `Done` | Qt record run now uses modal preflight review. |
| Scan robot ports | Scan + apply detected ports | `Done` | Qt record now scans and can apply detected follower/leader defaults. |
| HF dataset deploy/upload flow | Preview, parity, run, tagging | `Partial` | Inline upload follow-up exists; full parity dialog flow is missing. |
| Advanced/raw args | Advanced options panel | `Done` | Advanced record overrides and raw flag entry are exposed in Qt. |

## Deploy
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Preview command dialog | Modal preview dialog | `Done` | Qt preview now opens a modal dialog. |
| Editable launch confirmation | Editable command dialog before run | `Done` | Qt deploy run now uses editable confirmation. |
| Preflight review dialog | Modal preflight confirmation | `Done` | Qt deploy run now uses modal preflight review. |
| Quick-fix center | Deploy preflight action dialog | `Partial` | Qt deploy now supports eval-prefix/model/calibration/FPS/rename-map fixes, but richer upload/browser parity is still missing. |
| Model browser / tree | Local model browser with details | `Partial` | Qt deploy now has a selectable local model tree and info panel, but it is simpler than the Tk browser. |
| HF model upload popout | Local/remote parity + upload flow | `Partial` | Qt deploy now has a dedicated upload dialog with parity/preview/run, but it still lacks some Tk polish. |
| Advanced/raw args | Advanced overrides panel | `Done` | Advanced deploy overrides and raw flag entry are exposed in Qt. |

## Teleop
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Preview command dialog | Modal preview dialog | `Done` | Qt preview now opens a modal dialog. |
| Editable launch confirmation | Editable command dialog before run | `Done` | Qt teleop run now uses editable confirmation. |
| Preflight review dialog | Modal preflight confirmation | `Done` | Qt teleop run now uses modal preflight review. |
| Scan robot ports | Scan + apply detected ports | `Done` | Qt teleop now scans and can apply detected ports directly into the form. |
| Teleop snapshot/help | Snapshot pane + help affordances | `Done` | Qt now exposes teleop help plus a live session helper dialog with arrow-key controls. |
| Runtime helper popout | Dedicated session helper surface | `Done` | Qt teleop now has a modeless runtime helper dialog. |

## Secondary pages
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Config grouped coverage | Full grouped config editor | `Done` | Qt config now exposes grouped fields across paths, robot defaults, camera mapping, deploy defaults, calibration, and hub settings. |
| First-time setup wizard | Wizard popout + terminal actions | `Done` | Qt now has a setup wizard dialog with activate/custom-source/re-check flows and update/restart support. |
| Training HIL builder | Editable command buffer + HIL plan dialog | `Done` | Qt now supports generate, edit, copy, save-defaults, and HIL workflow dialog flows. |
| Visualizer split view | Root browse + richer details/insights | `Done` | Qt now has browse-root, dedicated metadata/details, deployment insights, and video/source actions. |
| History rerun/open flows | Rerun + open log/folder | `Done` | Basic history actions exist. |
| Deploy outcome + notes editor | Post-run notes/outcome tooling | `Done` | Qt History now edits deploy outcomes/notes and writes metadata, notes, and CSV exports. |

## Camera and runtime
| Area | Tk baseline | Qt status | Notes |
| --- | --- | --- | --- |
| Camera preview | Live preview/pause-resume | `Done` | Qt record/deploy pages now expose live camera preview with scan, refresh, live mode, and pause-on-run behavior. |
| Camera role assignment | Rename/mapping support | `Done` | Qt preview cards can assign Laptop/Phone camera roles and save the mapping. |
| Deploy outcome tracker | Runtime popout | `Done` | Qt deploy helper now tracks episodes live and persists marked outcomes back into the artifact metadata/CSV files. |
