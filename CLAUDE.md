# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

A git repository (main branch `master`) holding **MieWorkbench**: a PySide6
GUI + orchestration layer wrapped around a physically-based **coherent
Monte-Carlo optical ray tracer** driven by annotated FreeCAD models. Tagged
`.FCStd` scenes go through extract → trace → post → viz to produce detector
irradiance images (real interference/diffraction), 2D/3D ray renders,
spectra, Stokes maps, and an energy audit. Full polarization physics
(Jones-vector rays, birefringence, TMM coatings, gratings, aspheres, mesh
BVH). The engine is a baseline copy of `~/Documents/cad/opticalraytracer`
with GUI-integration extensions; **`docs/RAYTRACER.md` is the authoritative
engine reference** (authoring contract, physics model + honest limits,
property schemas, CLI reference, validation results). Top-level `README.md`
documents the workbench/GUI; `INSTALL.md` and `CUSTOMIZE.md` cover setup and
extension authoring. This file is the terse operator map.

Layout: `scripts/` engine + pipeline + tools · `scripts/fcserver/` headless
FreeCAD worker · `mieworkbench/` the GUI package (`core/` logic, `panes/`
dock widgets, `widgets/` VTK, `tests/`) · `opticalproperties/` property
library · `primitives/` parametric element library · `basemodels/` test
scenes · `demos/` ten classic-system `.MieWB` galleries (built by
`scripts/make_demos.py` through the fcclient op path; `demos/README.md`
has prescriptions+citations, `demos/UXNOTES.md` the shakedown log) ·
`env/` GUI venv (gitignored) · `var/` workspaces/caches (gitignored).

## Pinned interpreters — always use the right one (never cross-import)

| Stack | Interpreter | Runs |
|---|---|---|
| FreeCAD embedded | `/home3/freecad/FreeCAD.AppImage -c <script> -- <args> < /dev/null` | `extract_geometry.py`, `permute_model.py`, `make_test_scenes.py`, `make_primitives.py`, `fcserver/` |
| optics env (numpy/scipy/torch-CUDA/miepython/h5py) | `/home3/optics/env/bin/python` | `run_trace.py`, `post_process.py`, `compare_runs.py`, all `scripts/raytracer/`, engine pytest |
| ParaView 6.1.1 | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython --force-offscreen-rendering` | `make_viz.py` |
| system `python3` | stdlib only | `run_pipeline.py`, `sweep_variants.py`, `miewb_tool.py`, `common.py`, `cli_specs.py` |
| GUI venv | `env/bin/python` (PySide6 6.11 + vtk 9.6 + numpy/scipy/h5py) | `python -m mieworkbench`, GUI pytest |

All tool paths/dirs are env-overridable: `MIEWB_FREECAD`,
`MIEWB_OPTICS_PYTHON`, `MIEWB_PVPYTHON`, `MIEWB_GEOMETRY_DIR`,
`MIEWB_RESULTS_DIR`, `MIEWB_OPTPROPS_DIR` (defaults in `common.py` are this
machine's pins). `MIEWB_PROGRESS=1` makes stages emit `@MIEWB {json}`
progress lines; every stage also heartbeats `<case>/progress.json`.

## One-command flows

```bash
# CLI pipeline (unchanged from the original engine)
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
# presets: quick=1e5 rays/512²/5λ, normal=1e6/2048²/9λ, detailed=1e7/4096²/17λ
# --max-reflections N raises the 6-bounce cap (the fiber demo needs ~200)
# sweeps: --var lenspos --min -5 --max 5 --n 2   (sheet-qualified vars work:
#         --var dim_Lens1.ct — primitive groups are rebuilt per variant)
# visual-only overlay rays: --viz-pattern 'rings:dr=1:nper=12[:nrings=K]'
#                        or 'fan[:n=K]' (center + edge midpoints; GUI preview)

# GUI
env/bin/python -m mieworkbench [model.FCStd|X.MieWB|X.MieSim]   # or bin/mieworkbench

# headless/remote (only needs a repo clone + tools)
python3 scripts/miewb_tool.py pack model.FCStd -o X.MieWB --simparams p.json
python3 scripts/miewb_tool.py run X.MieWB -o X.MieSim    # unpack→pipeline→pack
```

Tests:
```bash
/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -q   # engine (~250; -m "not slow" for loops)
QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q          # GUI, fast
MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python -m pytest mieworkbench/tests -q  # + FreeCAD integration
```

## Architecture of the GUI layer

- **`scripts/fcserver/`** — the GUI cannot import FreeCAD; a persistent
  worker (`fc_server.py`, run under `AppImage -c`) speaks newline JSON over
  stdin/stdout, every protocol line prefixed `@FCJSON` (FreeCAD noise is
  discarded; the server emits a LEADING newline per protocol line and the
  client finds the prefix mid-line — FreeCAD progress noise lacks trailing
  newlines and used to glue onto responses = silent 300s timeouts). Ops in
  `fcops.py`: open/new/get_structure/tessellate (per-face STL, **body-local
  metres**)/set_property/remove_property/set_spreadsheet/set_placement/
  import_primitive/rebuild_primitive/delete_element (optional pre-image
  stash)/import_bodies (verbatim restore)/duplicate_element/save/save_as/
  save_copy/check. `fc_batch.py` is the one-shot fallback
  (`FC_REQUEST_FILE`/`FC_RESPONSE_FILE`).
- **`mieworkbench/core/`** — `fcclient.py` (Qt-free client, edit journal +
  relaunch-and-replay crash recovery; new_document is tracked like an
  open), `geomcache.py` (STL cache keyed by quantized
  placement-independent shape fingerprints), `project.py` (THE session
  object: routes worker mutations into `bodiesReshaped` = re-tessellate vs
  `bodiesMoved` = transform-only signals; every public mutation is an
  undoable Command with pre-image capture — undo/redo call private `_do_*`
  bodies), `undostack.py` (~20-level command stack + macros; delete
  stashes `.FCStd` pre-images under `<workspace>/undo/`; mid-stack failure
  clears the stack and reports), `selection.py` (shared SelectionModel
  syncing 3D picks/outliner/problems), `transforms.py` (placements,
  rotate-about-point, reference resolver, `element_bounds`), `wizards.py`
  (thick-lens solvers, oracle-tested), `validation.py` (pre-run check
  registry; deep check emits an explicit success INFO), `runner.py`
  (QProcess around `run_pipeline.py`, parses `@MIEWB`), `raypreview.py`
  (live-fan preview chain: save_copy → AppImage extract → optics-env
  `scripts/preview_rays.py` → rays.vtp, all via QProcess — never
  cross-imports), `units.py` (display units, completeness pinned vs the
  contract), `librarymgr.py`/`proplib.py` (system vs project property
  libraries), `paraview_launcher.py`.
- **Shell features**: File→New (.MieWB workspace packed immediately, or
  bare .FCStd; never creates .MieSim); outliner pane (select/copy/paste/
  delete elements by name; paste offsets +x past occupied AABBs); Results
  pane with lightbox galleries + per-element Power tab (in/out/absorbed/
  detected via `common.element_power_table`); adaptive mm/µm scale bar;
  ray overlays auto-load after runs and grey "stale" on geometry edits.
- **Formats** (`scripts/miewb_tool.py`, stdlib): `.MieWB` = ZIP {manifest,
  model.FCStd (stored), opticalproperties/, simparams.json}; `.MieSim` =
  ZIP {manifest, input.MieWB (the exact workbench used), geometry/,
  results/<model>/<case>/}. Rerun replaces members; `--purge-intermediates`
  drops rays.npy/viz/logs/face STLs but keeps the physics. `sniff()`
  distinguishes MieWB/MieSim/FCStd by content.
- **Locking**: one writer per case — `acquire_case_lock()` in `common.py`;
  `run_trace` REFUSES a locked case (exit 4). The GUI opens live cases
  read-only in monitor mode (polls `progress.json`).
- **Primitives** (`scripts/primitivelib.py` + `primitives/*.FCStd` +
  `.meta.json`): 54 catalog elements (sources/detectors/fiber optics/
  lenses/beamsplitters/filters/polarization/prisms & mirrors/apertures/
  diffusers — incl. `fiber_optic` core+cladding analytic cylinders,
  NA 0.22 via the `fiber_core_na22` material row, and `mirror_annular`
  perforated SCT-style primary); geometry params live in a `dim`
  spreadsheet; edits go through **rebuild-on-edit** (`rebuild_primitive`
  op re-runs the builder — constraint expressions can't change topology).
  Hand-authored primitives with real cell expressions use the normal
  `set_spreadsheet` path instead. Conventions: aperture params are
  `diameter`/`width` (LEGACY_ALIASES in `read_params` migrates old
  radius/half sheets ×2), `round_flag` picks circular vs rectangular on
  plates and sources, `derived_props` lets builder-owned body properties
  track sheet params through rebuilds (iris `blackness` → `absorbance`;
  `surface_override` on lens_asphere/mirror_parabolic — preserving a
  stale override string trips the extractor's <1 µm asphere gate), and
  face lookups must use the SIGN-AWARE `_find_face_by_signed_normal`
  (the abs-dot helper can't tell front from back caps). BS ratio / ND
  density / diffuser grit are swapped via the coating/filter/diffuser
  property rows (`bs_XXYY_vis_45`, `nd_odXX`, `@dg_XXX`), generated by
  `scripts/tools/gen_registry_rows.py`.
- **Round-C GUI features**: `core/facemaps.py` (pure per-face assignment
  model behind the assignment-centric "Active Properties" table + its
  right-click apply/remove menu tree; grating never collapses to the
  bare whole-body form), Shift=extend/Ctrl=toggle face picking,
  `widgets/faceindicators.py` (red half-disc emit/detector, blue/green
  +x dots; View-menu toggle, QSettings-persisted),
  `Project.opticsChanged` (miewb_*-filtered) + `core/previewscheduler.py`
  (1 s debounce, queue-one-more) drive the auto ray-preview; stale rays
  grey the ACTORS via `VtkSceneView.set_overlay_stale`. PySide6 trap:
  never retrieve a submenu via `QAction.menu()` (ownership transfers to
  Python and the GC deletes the C++ menu — use `menu.property_submenus`).

## Body-tagging contract (details docs/RAYTRACER.md §5)

`App::Property` on each `PartDesign::Body`, group "Base":
- `material` (string): materials registry row | uniaxial crystal name |
  `detector` | absent/`none` → ignored
- sources: `power` (mW) + `lambdac` (nm) [+ `lambdamin`/`lambdamax` nm,
  `coherent` bool, `polarization` = `unpolarized` | `linear:<deg>` |
  `circular:left|right` | `elliptical:<psi>:<chi>`]
- optic extras (stackable): `coating` (whole-body or `Face3=MgF2;...`),
  `roughness`, `diffuser` (`grit:120`|`slope:0.08`|`@dg_600`, per-face ok;
  NEVER with roughness on one face — deep-rough Beckmann limit, §5.4.1),
  `filter`, `polarizer` + `polarizer_axis`, `crystal_axis`,
  `grating` (`Face2=600:v:orders=-1..1` or `Face2=@vbg_1800`),
  `surface_override` (asphere, verified <1 µm; the verifier needs the
  vertex inside the retained face — off-axis parabola segments are
  structurally unverifiable, use `mirror_parabolic` on-axis),
  `mirror`, `absorbance`
- dimensions in `Spreadsheet::Sheet` aliased `=<val> mm` cells; extract
  echoes the primary `dim` sheet FLAT plus every sheet namespaced
  `<sheetlabel>.<alias>`
- GUI-internal tags: `miewb_primitive` (builder kind), `miewb_group`
  (bodies of one multi-body element)

**Aperture scenes:** fill slit/hole openings with thin `material=air` bodies.

## Optical component library

`opticalproperties/` uses self-describing extensions (content is still CSV):
`materials.miemat`, `nk/*.mienk`, `coating/coatings.miecoat`,
`polarizer/polarizers.miepol`, `filter/filters.miefilt`,
`grating/gratings.miegrat`, `birefringence/uniaxial.miebrf`, per-item
tables `*/tables/*.mietab`. Loaders prefer the new names and **fall back to
legacy `.csv`** (external all-.csv libraries keep working). `reference`
(citation) column is REQUIRED everywhere; loaders hard-validate
(`raytracer/optprops.py`). Override root: `--optical-properties DIR`.
A `.MieWB`'s embedded library is the **project library**; the GUI can
promote entries to the repo (system) library.

## Quirks / traps (each cost real debugging time)

- **FreeCAD `-c`**: bare `--` before script args; script runs TWICE per
  invocation (writes must be idempotent; `fc_server` kills pass 2 with
  `os._exit(0)` after its serve loop); NO `if __name__=="__main__"` guard;
  `print()` can drop (log via `FreeCAD.Console` too); `sys.exit` swallowed →
  `os._exit`; batch runs `< /dev/null` (the SERVER deliberately keeps stdin
  open as its request channel). `.FCStd` = zip: `unzip -p X.FCStd
  Document.xml` for recon.
- **Spreadsheet aliases must not look like cell addresses** — FreeCAD
  rejects `R1`/`A2` etc. ("Invalid alias"); use `R_front`, `R_back`.
- **Placements can be expression-bound** (`.Placement.Base.y =
  <<dim>>.lenspos` — note the LEADING DOT in ExpressionEngine paths);
  writing such a placement is silently undone on recompute. `fcops`
  refuses and names the driving alias; the GUI routes the move through it.
- **Shape fingerprints are quantized absolutely** (0.1 µm CoM, `-0.0`
  folded): `%g`-style relative formatting turns 1e-15 recompute noise into
  spurious cache invalidations. Placement moves must NOT invalidate the
  tessellation cache; body-local translations MUST.
- **Modal dialogs in teardown paths hang offscreen test runs** — guard
  `closeEvent`-style prompts on `isVisible()`.
- **Coherent gather at a focus needs rays**: a tightly focused coherent
  spot reconstructed via Huygens gather from too few samples loses power
  to phase noise (looks like "detector sees 0.5%"). Geometric focus checks
  should use `coherent=false` sources (direct deposit) or many more rays.
- **`orientation_outward` contract semantics**: the flag describes the
  CANONICAL normal derived from stored analytic params, NOT FreeCAD's
  `normalAt()` — don't "simplify" the extractor probe.
- **Detector grid basis is arbitrary**: always read `xhat`/`yhat` from the
  detector `.h5` attrs.
- **Emit/detector face auto-pick = closest face centroid to the WORLD
  origin** — on a rotated off-axis detector (folded telescope eyepiece)
  that's a thin EDGE face and the run silently detects 0 mW. Pin
  `--detector-face` (simparams `detector_face` list) in folded scenes;
  the demos do.
- **`set_placement` resolves miewb_group BEFORE label** (an imported
  multi-body element's primary body carries the element label itself —
  label-first lookup moved only that body and tore elements apart).
- **Multi-edge planar trim wires used to self-cross** (OrderedEdges
  doesn't flip reversed edges' point sequences) — the even-odd containment
  then killed ~half of every pad rectangle/triangle face (dead
  half-faces, phantom transmission, the wollaston scene's detected-power
  anomaly). Fixed in `extract_geometry.trim_polylines_xyz` (head-to-tail
  chain orientation); geometry/ caches from before the fix are stale —
  re-extract.
- **Table coatings past the critical angle now TIR honestly** (tracer
  folds the table's T into R); before, they emitted a grazing ghost
  "transmitted" child booked as seam loss.
- **A sheet param must never live in the body Placement** —
  rebuild_element preserves the PRE-rebuild placement, silently reverting
  it (the prism's `rotation` is baked into sketch vertices for this
  reason). Related: **rebuilds renumber FaceN**, so preserved face-mapped
  props (a grating_plate's `Face1=...`) can land on an edge face after a
  size edit — face indices are only trustworthy for the geometry they
  were authored against.
- **Gather keys are (source, λ-stratum, POL-stratum)**: budget rays
  accordingly or the `GatherError: undersampled` gate trips.
- **Optically-contacted solids don't exist** — but **proper NESTING does**
  (one solid strictly inside another; extractor classifies it
  `validation.nested_solids`, the tracer's LIFO medium stack handles it).
  Model cemented interfaces either with a ~5 µm air gap (achromats — fine
  at near-normal incidence) or, when the gap would TIR (45° internal
  beamsplitter interfaces: past BK7's 41.2° critical angle), as a thin
  coated plate NESTED in a single solid (how bs_cube/pbs_cube work now;
  the old two-prism gap build lost ~1/3 of the power to TIR/seam loss).
- **Detected power is a diagnostic, not a closure bucket**; the ledger
  partitions LOSSES only, gates at 1e-3. Detector maps are UNBIASED with
  zero-mean negative MC noise — clip only for display.
- **Mesh faces trace but phase doesn't** — keep phase-critical surfaces
  analytic. Asphere authoring: BSpline through exact sag points + matching
  `surface_override` (extractor verifies <1 µm or dies).
- **CUDA OOM in the gather**: reduce `pixel_chunk`/`sample_chunk` in
  `gather.points_torch`; occlusion mask slices sample-columns-FIRST.
- miepython 3.x: `efficiencies_mx(m,x)` wants m = n − ik. numpy 2.x:
  `np.trapezoid`.
- Disk: `/` is chronically ~97% full; envs/tools/workspaces live on
  `/home3` (`env/`, `var/`); `--save-fields` off by default.

## Physics invariants pinned by tests (don't break them)

Fresnel R+T=1 (1e-12); Brewster; TIR phase; TMM λ/4 MgF2 + half-wave
absentee; thick-lens focus vs lensmaker (0.5%) for PCX/DCX/PCV/DCV/ball/
rod/cylinder; double-slit fringe pitch λL/d ±1px + visibility >0.85
end-to-end from FCStd; Malus 1%; calcite walk-off 6.226°@45°/590nm; Kogelnik
η=1 at ν=π/2; Dammann Parseval; Igehy differentials vs finite differences;
BVH == brute force; Mie Qext vs Wiscombe; energy closure <1e-3 in EVERY
scene; torch/numpy gather <5e-3. New: `--viz-pattern` detector cubes are
BIT-identical with vs without the overlay (rings AND fan); wizard lens
designs reproduce the SCENES oracle focal lengths; fc-worker no-op save
round-trips the extraction contract exactly; element boundary-flux
tallies satisfy in − out = absorbed (diagnostic side-table, zero RNG use,
never a closure bucket); undo torture walk (add/move/edit/duplicate/
delete → undo to empty → redo to tip) compares worker structures equal.

## Everything else

`docs/RAYTRACER.md`: full engine reference (authoring contract syntax,
physics model + honest limits, schemas, CLI, 24-scene catalog + validation
table, troubleshooting). `README.md`: workbench/GUI tour + scripts +
formats. `INSTALL.md`: 22.04/24.04 setup from a clone + headless-server
install. `CUSTOMIZE.md`: authoring new primitives and property entries.
`future.md`: engine roadmap seams (exact uniaxial Fresnel, optical contact,
RCWA, curved detectors, GRIN, fluorescence).
