# MieWorkbench

MieWorkbench is a PySide6 GUI and orchestration layer built around an
existing FreeCAD-driven, physically-based coherent Monte-Carlo optical ray
tracer (referred to below as **the engine**). The engine itself — the
authoring contract for tagged `.FCStd` scenes, the physics model and its
honest limits, the optical-property CSV schemas, the full stage-by-stage
command reference, the 24-scene validation catalog, and troubleshooting —
is documented in **[docs/RAYTRACER.md](docs/RAYTRACER.md)**. This README
covers the workbench built on top of it: the desktop GUI, the headless
tools, the file formats it introduces (`.MieWB` / `.MieSim`), and how the
whole thing fits together.

See also: **[INSTALL.md](INSTALL.md)** (setup from a clone) and
**[CUSTOMIZE.md](CUSTOMIZE.md)** (authoring new optical elements and
property entries).

---

## 1. What this is

The engine is a four-stage pipeline (permute → extract → trace → post →
viz) that turns a tagged FreeCAD `PartDesign` optical bench into detector
irradiance images, spectra, Stokes/polarization maps, an energy audit, and
3D/ParaView visualizations. It is driven entirely by `scripts/run_pipeline.py`
and a handful of stage scripts, each pinned to the right interpreter
(FreeCAD's embedded Python, a numpy/scipy/torch "optics env", ParaView's
`pvpython`, or plain system `python3`) — see docs/RAYTRACER.md §1/§2 for
the full picture.

MieWorkbench wraps that pipeline with:

- **A desktop GUI** (`mieworkbench/`, PySide6 + VTK) for building and
  editing tagged scenes visually — a 3D optical-train view with face
  picking, a library of parametric elements, a properties/tagging editor,
  a transform panel, a run-configuration dialog auto-generated from the
  real CLI, and a results viewer with ParaView handoff and live monitor
  mode.
- **Two new archive formats**, `.MieWB` (a portable, editable "workbench":
  scene + project property library + run configuration) and `.MieSim` (a
  self-contained, re-runnable result: the exact workbench used + its
  results), so a project can be handed to someone else — or to a headless
  server — as a single file.
- **A headless tool**, `scripts/miewb_tool.py`, that packs/unpacks/runs
  these archives without any GUI, for remote or CI use.
- **New parametric primitives** (`primitives/*.FCStd` + `.meta.json`, built
  by `scripts/primitivelib.py`) that the GUI's "Library" pane and "Add
  element" wizards use to drop pre-tagged lenses, mirrors, gratings,
  sources, etc. into a scene without hand-authoring FreeCAD geometry.

Everything the engine does — the physics, the tagging contract, the
optical-property registries — is unchanged; MieWorkbench only adds a UI
and some packaging around it. Command-line users who only want the
original engine pipeline can ignore this repo's GUI entirely and follow
docs/RAYTRACER.md directly.

---

## 2. Quick start

Install first — see [INSTALL.md](INSTALL.md). Once `env/` exists and the
tool paths in `scripts/common.py` (or your `MIEWB_*` overrides / Settings
dialog) are correct for your machine:

```bash
# launch the GUI
env/bin/python -m mieworkbench
# or, if the launcher script has been installed (see INSTALL.md):
bin/mieworkbench

# open a specific file directly
env/bin/python -m mieworkbench example.FCStd
bin/mieworkbench example.FCStd
```

From the GUI: **File → Open…** and pick `example.FCStd` (a
divergent+collimated two-laser bench with a BK7 lens, a glass sphere, and
three detector screens — see docs/RAYTRACER.md §1). Then either:

- **Simulation → Run Pipeline…** to open the configuration matrix (§3.9
  below), pick the `quick` preset, and press **Run**; or
- **Simulation → Dry Run** for a fast estimate-only pass; or, from the
  command line, the same thing the GUI ultimately launches:

```bash
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
```

This runs extract → trace → post → viz into `results/example/quick/` and
prints a summary table. Open the **Results** pane (or `File → Open…` a
finished case directory / `.MieSim`) to see detector images, spectra,
plots, and the energy-closure audit; the **Console** pane's stage chips
turn green as each stage completes. `docs/RAYTRACER.md` §4 documents the
exact output tree.

---

## 3. The UI tour

`mieworkbench/mainwindow.py`'s `MainWindow` is a Zemax-inspired, multi-dock
shell: a central 3D optical-train view surrounded by dockable panes, a
menu/toolbar for file, simulation and view actions, and three file kinds
it can open — a bare `.FCStd` model (live FreeCAD session, edited in
place), a `.MieWB` workbench (exploded into a scratch workspace under
`var/work/`; **Save** re-packs the archive), or a `.MieSim` result (viewed
read-only, or opened via its embedded workbench for editing/rerun — a
successful rerun replaces the `.MieSim` in place).

### 3.1 Central 3D optical-train viewport (`panes/scene3d.py`)

The `QMainWindow`'s central widget. Shows every body in the scene in one
shared 3D view. Toolbar: **Fit** (reframe the whole scene), four
axis-view buttons (**+X/−X/+Y/+Z**), and a checkable **Rays** toggle
(shows/hides the loaded ray overlay). Clicking a face resolves to
`(body, face_id)`; clicking a different body always replaces the
selection, plain vs. Ctrl-click within the same body follows the usual
select/toggle convention. Standard VTK trackball-camera mouse controls
(drag to rotate, scroll to zoom).

Bodies are colored by **role**, resolved from their tags: a body with
both `power` and `lambdac` set is a **source** (red-ish, opaque); a body
with `material == "detector"` is a **detector** (gray-blue, translucent);
everything else is an **optic** (glassy light blue, translucent). A
selected face is highlighted orange with edges shown. (This color mapping
exists as a style table in the code; there is no separate on-screen
legend widget.)

### 3.2 Element Inspector (`panes/inspector3d.py`)

Dock **"Element Inspector"** — a single-element 3D view showing only the
currently selected body, centered and camera-fit. This is the primary
surface for building up a face selection to hand to the Element
Properties pane (the central viewport can also pick faces, but this pane
is where multi-face selections are expected to be built). Buttons:
**Select all faces**, **Clear**. Plain click replaces the face selection;
Ctrl+click toggles membership. Rotating is the standard VTK trackball
interactor, not a dedicated control.

### 3.3 Element Properties (`panes/element_editor.py`)

Dock **"Element Properties"** — edits the selected body's tagging
contract, per-face assignments, and parameter-sheet aliases, entirely
through the in-memory Project (never talking to FreeCAD directly). Three
sections:

- **Optical properties (Base tags)** — one row per non-internal custom
  property on the body (internal `miewb_*` bookkeeping tags are hidden),
  plus **Add property…**. The full contract property set is `material`,
  `power`, `lambdac`, `lambdamin`, `lambdamax`, `coherent`, `polarization`,
  `coating`, `roughness`, `filter`, `polarizer`, `polarizer_axis`,
  `crystal_axis`, `grating`, `surface_override`, `mirror`, `absorbance` —
  see §5 below and docs/RAYTRACER.md §5.1 for full semantics of each.
- **Per-face assignments** — assigns one of `coating` / `roughness` /
  `grating` / `surface_override` onto the current face selection (these
  four support per-face maps like `'Face3=MgF2;Face5=x'`); a table lists
  every current per-face entry across all four properties.
- **Element parameters** — the parameter-sheet ("dim spreadsheet") editor:
  one row per aliased cell (`Alias` / `Value` / `Unit`), parsed from and
  recomposed back into the raw `"=<value> <unit>"` cell form. If the body
  is a library primitive (carries a `miewb_primitive` tag), committing an
  edit also triggers a rebuild of its geometry from the primitive builder
  (see CUSTOMIZE.md — geometry parameters can change topology, so they
  aren't driven by ordinary FreeCAD expressions).

### 3.4 Position / Orientation (`panes/transform_panel.py`)

Dock **"Position / Orientation"** — translate and rotate the selected
element with repeatable operations. Reference points resolve *live* at
apply time, so "Apply again" after other moves keeps its meaning (e.g.
"toward the lens"). Reference-point kinds: **Origin**, a **fixed point**
you type in, or one of three element-relative points (**optical center**,
**center of mass**, **bbox center**, **point on face normal**).

- **Translate**: X/Y/Z offset + **Apply translation**; or pick a
  reference point, a distance (default 10 mm, negative moves away), and
  **Move toward**.
- **Rotate**: axis (Global X/Y/Z, a custom vector, or the selected
  element's own optical axis) + angle (±360°) + an "About:" reference
  point + **Apply rotation**.
- **Apply again** re-applies the last operation, re-resolving any
  reference points at their current positions; a log lists every
  operation applied this session. If a body's placement is itself
  spreadsheet-driven, the panel says so and routes the move through the
  driving expression instead of silently fighting it.

### 3.5 Library (`panes/library.py`)

Dock **"Library"**, three tabs:

- **Elements** — a tree of `primitives/*.FCStd`, grouped by category
  (Sources, Lenses, Mirrors, …). **Refresh** rescans the directory (drop a
  hand-authored `.FCStd` in and it appears); **Add to scene** (or
  double-click) prompts for a label and adds it.
- **Project library** / **System library** — per-category row counts and
  names read from the optical-property registries, with **Open in
  editor** launching the Property Library Editor (CUSTOMIZE.md).

Two libraries live on disk: the **system library** is `<repo>/opticalproperties/`
and `<repo>/primitives/` (read by default; only written to by an explicit,
validated "promote to system" action); the **project library** is
`<project>/opticalproperties/` inside a `.MieWB` workspace — a
possibly-partial copy holding just the registry rows and table/nk files a
given model actually uses, so a project directory (or a `.MieWB`) is
self-contained and can be traced elsewhere with
`--optical-properties <project>/opticalproperties`.

### 3.6 Console, stage chips, progress (bottom dock, `panes/console.py`)

One colored pill per pipeline stage (`extract`/`trace`/`post`/`viz`: blue
= running, green = done/estimated, red = failed, gray = not yet run), an
overall progress bar, and a dark, monospace console log fed line-by-line
from the running pipeline subprocess (an in-memory ring buffer of the last
20000 lines). A stage filter combo and a **Clear** button (does not stop a
running pipeline); lines are colorized by stage and severity (errors red,
notices orange). Internal `@MIEWB {json}` progress lines are consumed to
drive the chips/progress bar rather than being printed raw.

### 3.7 Results (`panes/results.py`)

Dock **"Results"** — browse a completed (or in-progress) case: `report.json`
headline numbers, the energy-closure audit ("OK ✓" / "FAILED ✗" / "n/a"),
and thumbnail galleries for `images/`, `spectra/`, `plots/`, `viz/`. A
**Summary** tab tables per-detector power/peak irradiance/pixel
size/fringe visibility. **Open in ParaView** launches interactive ParaView
on the case's `.vtp` ray/detector data (enabled once viz output exists).
**Monitor mode**: opening a case that is currently locked by a live run
polls `progress.json` and new images once a second and shows live stage
progress in the title bar — this pane never writes anything while
monitoring; editing/rerun affordances are the main window's job to
disable.

### 3.8 Problems (`panes/problems.py`)

Dock **"Problems"** — pre-run validation, click-to-locate. **Validate
scene** runs pure Python checks (missing tags, bad registry references,
inconsistent per-face maps, …) against the live scene, the active
property library, and the current run configuration. **Deep check**
additionally runs FreeCAD-side geometry checks (recompute errors, open
solids, overlaps). Findings are listed with a severity icon; double-click
selects the offending body in the scene. Errors block **Run** (with a
blocking dialog); warnings prompt "Run anyway?".

### 3.9 Run Pipeline dialog — the configuration matrix (`panes/config_matrix.py`)

**Simulation → Run Pipeline…** opens a dialog embedding `ConfigMatrix`, a
form **auto-generated from the real CLI**: it introspects
`cli_specs.build_parser("pipeline")` (the same parser `run_pipeline.py`
itself uses) and builds one widget per option, grouped exactly as the
parser's own argument groups — so a new `--option` added to
`scripts/cli_specs.py` shows up here automatically, with no GUI code to
keep in sync. (`--help`, `--models`, and `--print-only` are never
rendered; `--preset` gets its own dedicated combo.) Widget choice follows
the option's argparse action: a checkbox for `store_true`, a combo for
`choices` (blank = "let the preset/default decide" when the parser's own
default is `None`), a semicolon-separated line edit for `append` options,
a spin box for plain integers (0 = "unset, fall back to preset"), and a
validated line edit for floats/strings (empty = unset). Only values that
differ from the parser's own default are ever forwarded as flags, so the
form can never accidentally override a default the pipeline would have
picked anyway. A **Preset** combo and an **Estimate runtime** button sit
above the form.

### 3.10 Estimate Runtime

Available from the Simulation menu, the toolbar, and the configuration
matrix itself. Resolves the current widget values (falling back to the
active preset) into `common.estimate()`'s inputs (rays, resolution,
nlambda, backend, etc.) and shows a message box with estimated trace time,
gather time, total time, and accumulator memory (GB) — a pure computed
estimate; nothing is run.

### 3.11 Dry Run

**Simulation → Dry Run** saves and validates the scene as usual, then
launches the pipeline with `--dry-run` appended: the trace stage builds
its estimates but does not actually trace, and post/viz are then skipped
for that model. Useful as a fast end-to-end sanity check of a
configuration before committing to a real run.

### 3.12 Export Run Script

**File → Export Run Script…** packs the current model into a `.MieWB`
(alongside a `.MieSim` sibling name it will produce) and writes a small,
`chmod +x` POSIX shell script that a machine with just a repo clone (and
the pinned tools, or `MIEWB_*` overrides) can run headlessly:

```sh
#!/bin/sh
set -e
python3 <repo>/scripts/miewb_tool.py run <the>.MieWB -o <the>.MieSim
```

The script contains no simulation logic itself — it is a thin, portable
wrapper around `miewb_tool.py run` (§5.9), intended for handing a
configured job to a remote/CI machine.

---

## 4. File formats

### 4.1 `.FCStd` — the scene

An ordinary FreeCAD document. The tagging contract every model must
follow (body/face `App::Property*` custom properties: `material`,
`power`/`lambdac`, `coating`, `roughness`, `filter`, `polarizer` +
`polarizer_axis`, `crystal_axis`, `grating`, `surface_override`, `mirror`,
`absorbance`, plus the `dim`-labeled parameter spreadsheet and the
GUI-internal `miewb_primitive`/`miewb_group` tags) is fully specified in
**docs/RAYTRACER.md §5**. A quick-reference summary is in
[CUSTOMIZE.md](CUSTOMIZE.md).

### 4.2 `.MieWB` — a portable workbench

A ZIP archive, built and read by `scripts/miewb_tool.py`:

```
manifest.json          {"format":"MieWB","version":1,"created":...,
                         "app":..., "fcstd":"model.FCStd", "model_stem":...}
model.FCStd             the scene (stored, not deflated — .FCStd is itself a zip)
opticalproperties/**    the project property library
simparams.json          run_pipeline.py option values (from the configuration matrix)
project.json            optional GUI/session metadata
```

**Open**: the GUI unpacks it into a scratch workspace under
`var/work/<name>-<hash>/`, opens the exploded model, and loads
`simparams.json` into the configuration matrix; the library manager points
at the workspace's `opticalproperties/` as the *project* library. **Save**
re-packs the whole archive from the current workspace state (a full
`pack_miewb()` call to the same path, atomically replacing it) —
"repacking" is not an incremental patch, it is a fresh pack each time.
From the command line:

```bash
python3 scripts/miewb_tool.py pack model.FCStd -o project.MieWB \
    [--optical-properties DIR] [--simparams params.json]
python3 scripts/miewb_tool.py unpack project.MieWB -d some/dir
python3 scripts/miewb_tool.py info project.MieWB
```

### 4.3 `.MieSim` — a self-contained result

Also a ZIP archive:

```
manifest.json           {"format":"MieSim","version":1,"created":...,
                          "source_miewb":..., "model":<stem>, "case":<case>,
                          "status":..., "purged_intermediates":bool}
input.MieWB              the EXACT workbench used for this run (stored)
geometry/<stem>/**       the extracted contract (model.json + face STLs)
results/<stem>/<case>/** everything the pipeline wrote (never includes .lock.json)
```

Opening a `.MieSim` in the GUI shows results. If the case is currently
locked by a live process, it opens **read-only in monitor mode**
(§3.7). Otherwise you're asked whether to just view results or open the
embedded `input.MieWB` for editing — **a successful rerun replaces
`input.MieWB` and every result member of the same `.MieSim`, in place**.
Short of a rerun, the only mutation a `.MieSim` supports is pulling its
embedded workbench back out ("save as `.MieWB`"):

```bash
python3 scripts/miewb_tool.py run project.MieWB -o result.MieSim [--workdir DIR] [--keep]
python3 scripts/miewb_tool.py pack-sim -d workdir -o result.MieSim --miewb project.MieWB \
    [--model-stem STEM] [--case CASE] [--purge-intermediates]
python3 scripts/miewb_tool.py extract-miewb result.MieSim -o project.MieWB
```

`--purge-intermediates` drops the bulky, regenerable-from-kept-outputs
files (`rays.npy`, `viz/*`, `log.*`, per-face `.stl` meshes) while keeping
`detectors/*.h5`, `case.json`, and `model.json` — a disk-space option for
archiving finished runs, exposed only via the CLI (the GUI's own
rerun-and-repack path does not purge).

`miewb_tool.py`'s `sniff()` tells `.MieWB`/`.MieSim`/bare-`.FCStd` apart by
manifest content, not by file extension.

### 4.4 Optical property files (`opticalproperties/`)

The property library uses self-describing extensions; **the content is
still plain CSV**, and every loader falls back to a same-stem legacy
`.csv` file if the new-style file isn't present (so an old all-`.csv`
library keeps working, with a one-line `NOTE:` to stderr). Every registry
row requires a non-empty `reference` (citation) column — loaders hard-fail
on a missing one.

| File | Category | Required columns |
|---|---|---|
| `materials.miemat` | bulk n(λ)/k(λ) database | `name,class,model,p1..p6,nk_file,density_kg_m3,transmission_um_min,transmission_um_max,notes,reference` |
| `nk/*.mienk` | tabulated n,k spectra (metals, water, TiO2, …) | `wavelength_nm,n,k` |
| `coating/coatings.miecoat` (+ `coating/tables/*.mietab`) | TMM stacks **or** measured Rs/Rp/Ts/Tp tables | registry: `name,layers,table,aoi_deg,reference`; table: `wavelength_nm,Rs,Rp,Ts,Tp` |
| `polarizer/polarizers.miepol` (+ `polarizer/tables/*.mietab`) | linear/circular diattenuators | registry: `name,type,table_csv,retardance_waves,reference`; table: `wavelength_nm,T_parallel,T_perpendicular` |
| `filter/filters.miefilt` (+ `filter/tables/*.mietab`) | bulk spectral filters (Beer-Lambert) | registry: `name,table_csv,ref_thickness_mm,reference`; table: `wavelength_nm,transmittance_internal` |
| `grating/gratings.miegrat` (+ `grating/tables/*.mietab`) | lamellar/Kogelnik/Dammann/table registry | registry: `name,model,lines_per_mm,params,table_csv,reference`; table: `wavelength_nm,order,eta_s,eta_p` |
| `birefringence/uniaxial.miebrf` | calcite/quartz/sapphire o/e crystal pairs | `name,n_o_material,n_e_material,reference` (+`notes`) |

Full schema semantics, the citation policy, and the physics each category
feeds into are in docs/RAYTRACER.md §7. Loader: `scripts/raytracer/optprops.py`
(polarizer/filter/grating/birefringence) and `scripts/raytracer/materials.py`
(materials/coatings). See [CUSTOMIZE.md](CUSTOMIZE.md) for adding entries.

---

## 5. The scripts

All CLI options below are read from each script's own `--help` (or, for
the three FreeCAD-only scripts, their argparse source — the FreeCAD
AppImage's `-c` batch mode does not reliably print `--help` output).

### 5.1 `run_pipeline.py` — the orchestrator (system `python3`)

```
run_pipeline.py --models FCSTD [FCSTD ...] [--preset {quick,normal,detailed}]
                 [--tag TAG] [--steps LIST] [--var VAR --min MIN --max MAX --n N]
                 [--dry-run] [--rays R] [--resolution N] [--nlambda N] [...physics options...]
                 [--keep-going] [--print-only]
```

Composes and launches each pinned stage command as a subprocess; imports
nothing beyond the standard library. `--steps extract,trace,post,viz`
picks a subset (fixed order); `--var/--min/--max/--n` (repeatable, paired
in order) sweep spreadsheet aliases through `permute_model.py` before
extraction; `--print-only` prints the composed commands without running
anything. Presets fill in rays/resolution/nlambda/spectral-bins/viz-rays:
`quick` = 1e5/512/5/16, `normal` = 1e6/2048/9/16, `detailed` =
1e7/4096/17/32 (`common.PRESETS`).

### 5.2 `run_trace.py` — the solver (optics env python)

```
run_trace.py --model-json MODEL_JSON --case-dir CASE_DIR
             [--rays R] [--resolution N] [--nlambda N] [--spectral-bins N]
             [--backend {auto,torch,numpy}] [--seeds N] [--save-fields]
             [--viz-pattern SPEC] [--ray-differentials] [--gather-occlusion] [...]
```

`--viz-pattern 'rings:dr=<mm>:nper=<N>[:nrings=<K>]'` replaces the random
viz-ray sample with a deterministic layout (one central ray plus
concentric rings every `dr` mm, `nper` rays per ring, out to the emit
face's rim or `nrings` rings if given) — **visualization only; it never
affects the physics** (traced in a separate viz-only pass; a dedicated
test pins that detector cubes are bit-identical with and without it). One
writer per case — see §6.

### 5.3 `extract_geometry.py` — FreeCAD headless

```
/home3/freecad/FreeCAD.AppImage -c scripts/extract_geometry.py -- \
    --models example.FCStd [--outdir geometry] [--strict] < /dev/null
```

Reads each `PartDesign::Body`'s Base-group tags, classifies it as
source/detector/optic/ignored, and writes `geometry/<stem>/model.json` +
`geometry/<stem>/faces/*.stl`. `--strict` hard-fails (instead of warning)
on any face that falls back to mesh-only representation.

### 5.4 `permute_model.py` — parameter sweeps (FreeCAD headless)

```
/home3/freecad/FreeCAD.AppImage -c scripts/permute_model.py -- \
    --model example.FCStd --var lenspos --min -5 --max 5 --n 2 \
    [--outdir basemodels] [--unit mm] < /dev/null
```

`--var` accepts a bare spreadsheet alias (`lenspos`) or a
**sheet-qualified** name `<sheet_label>.<alias>` (e.g. `dim_Lens1.ct`) to
address a specific element's own parameter sheet — MieWorkbench primitives
each carry a `dim_<label>` sheet, so this is how a sweep targets one
element's geometry parameter rather than a scene-global alias. `--n 0` →
`[min]` only, `--n 1` → `[min, max]`, `--n>1` → `n+1` evenly spaced
values; `--var`/`--min`/`--max`/`--n` counts must match (repeatable,
paired in order). If a swept sheet is a `dim_*` primitive sheet, every
`PartDesign::Body` tagged with the matching `miewb_group` is rebuilt from
its primitive builder afterward (topology-changing edits can't be done by
FreeCAD expressions alone — see CUSTOMIZE.md).

### 5.5 `post_process.py` — rendering/analysis (optics env python)

```
post_process.py --case-dir CASE_DIR --model-json MODEL_JSON [--viz-generations N]
```

Rerunnable without re-tracing. `--viz-generations N` declutters
`rays_xy.png` to reconstructed-generation ≤ N segments only.

### 5.6 `make_viz.py` — 3D visualization (ParaView `pvpython`)

```
/home3/paraview/.../bin/pvpython --force-offscreen-rendering scripts/make_viz.py \
    --case-dir CASE_DIR --model-json MODEL_JSON [--views v1,v2,...] \
    [--resolution WIDTHxHEIGHT] [--out DIR] [--smoke] [--skip-vtkexport]
```

`--views` picks a subset of the view registry (`overview3d`, `top`, `side`,
`detector_closeup`, `turntable`, `rays_polmode`, …); `--smoke` renders only
`overview3d` at 800×600 for a fast end-to-end check; `--skip-vtkexport`
skips the optics-env `raytracer.vtkexport` prep sub-step if `.vtp` files
already exist. `run_pipeline.py`'s internal viz step only ever forwards
`--case-dir`/`--model-json`, so to pick views/resolution/smoke you invoke
`make_viz.py` directly on an already-completed case (docs/RAYTRACER.md
§4.2).

### 5.7 `sweep_variants.py` — batch jobs (system `python3`)

```
sweep_variants.py [--jobs jobs.json | --job k=v,k=v [--job ...]]
                   [--models ...] [--preset ...] [...defaults for every job...]
                   [--keep-going] [--no-compare] [--compare-out DIR]
```

Runs several `run_pipeline.py` jobs back-to-back (a `--jobs` JSON file of
per-job option dicts, or repeatable inline `--job k=v,k=v` jobs; common
`--models`/`--preset`/etc. seed every job unless a job overrides them),
then automatically overlays the finished cases with `compare_runs.py`
unless `--no-compare` is given.

### 5.8 `compare_runs.py` — overlay finished cases (optics env python)

```
compare_runs.py --cases DIR [DIR ...] [--out OUT]
```

Overlays the detector results of several finished `results/<model>/<case>`
directories; default output is `results/comparisons/<case names>`.

### 5.9 `miewb_tool.py` — the headless/remote path (system `python3`)

```
miewb_tool.py pack model.FCStd -o X.MieWB [--optical-properties DIR] [--simparams params.json]
miewb_tool.py unpack X.MieWB -d DEST
miewb_tool.py info X.MieWB
miewb_tool.py run X.MieWB -o X.MieSim [--workdir DIR] [--keep] [-- extra run_pipeline.py args]
miewb_tool.py pack-sim -d WORKDIR -o X.MieSim --miewb X.MieWB [--model-stem S] [--case C] [--purge-intermediates]
miewb_tool.py extract-miewb X.MieSim -o X.MieWB
```

The `run` subcommand is the full headless flow: unpack `.MieWB` into an
isolated workspace, run `run_pipeline.py` there (extra args after a bare
`--` are forwarded), and pack the result into `.MieSim` — the same code
path the GUI's "Export Run Script" and rerun-from-`.MieSim` flows use.
This is the tool a machine with just a repo clone (plus FreeCAD/optics-env/
ParaView, or `MIEWB_*` overrides pointing elsewhere) needs to run a
workbench with no GUI at all.

### 5.10 `make_primitives.py` — generate the element library (FreeCAD headless)

```
/home3/freecad/FreeCAD.AppImage -c scripts/make_primitives.py -- \
    [--outdir primitives] [--kind <name>|all] < /dev/null
```

Builds `primitives/*.FCStd` + `.meta.json` sidecars from
`scripts/primitivelib.py`'s `PRIMITIVES` registry (see CUSTOMIZE.md).
`--kind` builds a single primitive by name instead of the whole library.

### 5.11 `make_test_scenes.py` — validation scene catalog (FreeCAD headless)

```
/home3/freecad/FreeCAD.AppImage -c scripts/make_test_scenes.py -- \
    [--outdir DIR] [--scene NAME|all] < /dev/null
```

Authors the 24+ FreeCAD validation scenes cataloged in docs/RAYTRACER.md
§10 (polarizers, birefringent crystals, filters, coatings, aspheres, a
deliberately non-analytic mesh face, `doubleslit.FCStd`, …); also supplies
the geometry-helper functions (`lens_meridian`, `revolve_body`,
`new_body_pad`, …) that `primitivelib.py` reuses to build primitives.

### 5.12 `cli_specs.py` / `common.py` — shared infrastructure

`scripts/cli_specs.py` is the single source of truth for the `pipeline`
(`run_pipeline.py`), `trace` (`run_trace.py`), `post` (`post_process.py`),
and `viz` (`make_viz.py`) argument parsers (`build_parser(stage)`); every
stage script and the GUI's configuration matrix (§3.9) build their parser
from here, so they can never drift apart. Self-check: `python3
scripts/cli_specs.py`.

`scripts/common.py` is the stdlib-only hub every interpreter stack
imports: pinned tool paths (env-overridable, see below), fidelity
presets, the spec parsers for face/grating/roughness/polarization/axis/
particle/viz-pattern option values, the `model.json` contract validator,
sweep-name/case-name helpers, the runtime/memory estimator, and case
locking (§6). Self-check (verifies the three interpreter paths exist,
`materials.miemat` exists, and a battery of pure-math invariants):

```bash
python3 scripts/common.py
```

### 5.13 Environment variables

All tool paths and data directories are overridable — either by exporting
these before launching anything, or via the GUI's **File → Settings…**
dialog (which persists the same values through `QSettings` and layers
them onto every pipeline subprocess it launches):

| Variable | Overrides | Default |
|---|---|---|
| `MIEWB_FREECAD` | FreeCAD AppImage path | `/home3/freecad/FreeCAD.AppImage` |
| `MIEWB_OPTICS_PYTHON` | optics-env Python interpreter | `/home3/optics/env/bin/python` |
| `MIEWB_PVPYTHON` | ParaView's `pvpython` | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython` |
| `MIEWB_GEOMETRY_DIR` | `geometry/` output root | `<repo>/geometry` |
| `MIEWB_RESULTS_DIR` | `results/` output root | `<repo>/results` |
| `MIEWB_OPTPROPS_DIR` | `opticalproperties/` library root | `<repo>/opticalproperties` |
| `MIEWB_PROGRESS` | when `1`, stages also print `@MIEWB {json}` progress lines to stdout | unset (progress.json heartbeat is always written regardless) |

---

## 6. Concurrency and locking

Exactly one writer is allowed per case directory. `run_trace.py` calls
`common.acquire_case_lock(case_dir)` before tracing, which atomically
creates `<case_dir>/.lock.json` (`{pid, host, started, cmdline}`). If a
fresh lock already exists, the trace **refuses and exits with code 4**:

```
[trace] REFUSED: case is locked by pid <PID> on <HOST> since <TIMESTAMP>
(rerun when it finishes, or remove <case_dir>/.lock.json if you are sure it is dead)
```

A lock is considered stale (safe to steal) once its heartbeat
(`progress.json` or the lock file's own mtime) is more than 120s old *and*
its recorded pid is no longer alive; the lock is released in a `finally`
block so both success and failure paths clean up. `.lock.json` is never
included when an archive is packed into a `.MieSim` (§4.3).

In the GUI, opening a case/`.MieSim` that is currently locked opens it
**read-only in monitor mode** (§3.7) instead of racing the live run —
a `QTimer` polls `progress.json` and the growing image galleries once a
second.

---

## 7. Testing

Two independent test suites, run under two different interpreters —
never cross-import between them:

```bash
# the engine (pure Python + numpy/scipy/torch; no FreeCAD, no Qt)
/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -q
# (slow end-to-end cases, e.g. test_gather.py/test_doubleslit_e2e.py, are
#  marked `slow`: add -m "not slow" to skip them for a fast loop)

# the GUI (PySide6 + VTK; runs headless via Qt's offscreen platform plugin)
QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q

# + FreeCAD integration tests (slower; drives the real fc_server worker)
MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q
```

GUI tests marked `@pytest.mark.freecad` are auto-skipped unless
`MIEWB_RUN_FREECAD=1` is set (`mieworkbench/tests/conftest.py`); tests
marked `needs_gl` are skipped when running offscreen (no real OpenGL
context). Both suites currently collect on the order of 200+ tests each;
run with `--collect-only -q` to see the current count on your checkout.
