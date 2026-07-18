# MieWorkbench per-feature guide

Terse, code-derived reference for each GUI surface and the surrounding
system tooling. For the engine's physics model and authoring contract see
[`docs/RAYTRACER.md`](../RAYTRACER.md); for the whole-workbench tour see
the top-level [`README.md`](../../README.md). Reachable in-app from
**Help →** (mainwindow.py).

Screenshots live in `img/`; a page missing one either has none needed or
the shot is deferred (see `scripts/tools/capture_docs_screenshots.py`'s
shot list and `img/manifest.json` for current status).

## GUI

| Page | Covers |
|---|---|
| [viewport-3d.md](viewport-3d.md) | Central 3D scene view: navigation, face picking, scale bar, face indicators, train ghosting/linkage lines |
| [outliner.md](outliner.md) | Scene Elements list: select/copy/paste/delete by name, role/status badges |
| [inspector.md](inspector.md) | Single-element view and the primary face-selection surface |
| [element-editor.md](element-editor.md) | Optical properties, the Active Properties per-face assignment table, parameter sheet |
| [transform.md](transform.md) | Position/Orientation panel: translate/rotate operations, reference points |
| [train-editor.md](train-editor.md) | LDE-style optical train tree: chain/anchor, ports, folds, flip, expression cells |
| [variables.md](variables.md) | `miewb_vars` global variables, expression grammar, sweeps |
| [compare.md](compare.md) | Sweep-comparison viewer (metrics, plots, gallery, diffs) |
| [optimize.md](optimize.md) | Merit-function optimizer: variables, operands, algorithms, convergence |
| [tolerance.md](tolerance.md) | Sensitivity + Monte-Carlo tolerancing, compensators, yield |
| [results.md](results.md) | Case browser: galleries, power tab, analysis, monitor mode |
| [library-browser.md](library-browser.md) | Element/property library dock + the element wizard |
| [property-library-editor.md](property-library-editor.md) | Registry table editor: schema tooltips, status line, validation |
| [run-and-validate.md](run-and-validate.md) | Configuration matrix, pre-run dialog, estimate, dry run, validation |
| [animation.md](animation.md) | Tracer-bead ray animation, incl. bead-opacity mode |
| [console-and-problems.md](console-and-problems.md) | Pipeline log console, Python console, Problems pane |

## System

| Page | Covers |
|---|---|
| [pipeline-cli.md](pipeline-cli.md) | `run_pipeline.py` presets/flags — condensed, links to the full CLI reference |
| [file-formats.md](file-formats.md) | `.MieWB` / `.MieSim` / `.FCStd`, `miewb_tool.py`, `sniff()` |
| [headless-remote.md](headless-remote.md) | Running the pipeline/tool without the GUI, locking, testing |
| [authoring.md](authoring.md) | Pointer page: new primitives / new property entries |
| [demo-gallery.md](demo-gallery.md) | The 38-demo gallery overview + the every-demo-ships-configured Optimize/Tolerance story |

## Walkthroughs

One page per optimize/tolerance showcase demo — load, what the
pre-configured panes show, run a short optimize, run tolerance
sensitivity, interpret the result. Index + table:
[walkthroughs/README.md](walkthroughs/README.md).

| Page | System |
|---|---|
| [walkthroughs/camera-triplet.md](walkthroughs/camera-triplet.md) | Cooke triplet — `spot_rms`, MC-unstable decenter ranking (honest caveat) |
| [walkthroughs/schmidt-cassegrain.md](walkthroughs/schmidt-cassegrain.md) | C8 catadioptric — `spot_rms`, secondary despace magnification |
| [walkthroughs/double-gauss.md](walkthroughs/double-gauss.md) | Symmetric double-Gauss — `spot_rms`, symmetry-breaking decenters |
| [walkthroughs/fiber-coupling-doublet.md](walkthroughs/fiber-coupling-doublet.md) | Achromat → fiber — `detected_power`, decenter vs. despace |
