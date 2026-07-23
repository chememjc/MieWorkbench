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
scenes · `demos/` 42 classic-system `.MieWB` galleries — benchmark set is
11 scenes, 21 designs have committed baselines (coherent-diffraction
characterization demos like `airy_singleslit`/`bladed_iris_star` are gated
by their pattern tests, not the placement/power oracle) — incl.
`fizeau_flats`/`fs_shg_spectrogram`/`quartz_rotator`/`speckle_mie_combo`
(coherent ghost fringes / fs SHG+dispersion time products / asserted
natural optical-activity rotation / coherent Mie speckle) — (built by
`scripts/make_demos.py` through the Project/chain op path; `demos/README.md`
has prescriptions+citations, `demos/UXNOTES.md` the consolidated
open-UX-friction list (per-round shakedown logs pruned into it 2026-07-19);
`demos/baselines/` committed placement+power oracles
for `scripts/run_demo_equivalence.py`) · `demos/library_tests/`
nine library-validation template scenes + automated sweep runner
(`scripts/make_library_tests.py`, `scripts/run_library_tests.py`) ·
`library_data/` sourced data for staged library entries · `env/` GUI venv
(gitignored) · `var/` workspaces/caches (gitignored).

## Pinned interpreters — always use the right one (never cross-import)

| Stack | Interpreter | Runs |
|---|---|---|
| FreeCAD embedded | `/home3/freecad/FreeCAD.AppImage -c <script> -- <args> < /dev/null` | `extract_geometry.py`, `permute_model.py` (+ `train_fcstd.py` it imports), `make_test_scenes.py`, `make_primitives.py`, `fcserver/` |
| optics env (numpy/scipy/torch-CUDA/miepython/h5py) | `/home3/optics/env/bin/python` | `run_trace.py`, `post_process.py`, `compare_runs.py`, `compare_sweep.py`, all `scripts/raytracer/`, engine pytest |
| ParaView 6.1.1 | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython --force-offscreen-rendering` | `make_viz.py` |
| system `python3` | stdlib only | `run_pipeline.py`, `sweep_variants.py`, `miewb_tool.py`, `common.py`, `cli_specs.py`, `train_solver.py` (pure stdlib BY CONTRACT — shared with FreeCAD's numpy-less python) |
| GUI venv | `env/bin/python` (PySide6 6.11 + vtk 9.6 + numpy/scipy/h5py) | `python -m mieworkbench`, GUI pytest, `make_demos.py` + `run_demo_equivalence.py` (they drive a full Project session; NO LONGER system python3) |

Machine paths live in ONE place: gitignored `<repo>/miewb.env` (created by
`scripts/setup_env.sh`, shell-loaded by `source scripts/miewb_env.sh`, parsed
directly by `common.py` — exported `MIEWB_*` env vars win over file entries;
required tools missing both = import-time error; empty value = "configured
absent"). The literal paths in the table above are THIS machine's miewb.env
contents — never hardcode them in code or docs; use the `MIEWB_FREECAD` /
`MIEWB_OPTICS_PYTHON` / `MIEWB_PVPYTHON` / `MIEWB_NVCC` / `MIEWB_CUDA_ARCH`
keys (+ `MIEWB_GEOMETRY_DIR`/`MIEWB_RESULTS_DIR`/`MIEWB_OPTPROPS_DIR`/
`MIEWB_GUI_PYTHON`/`MIEWB_CENGINE` optional overrides). `MIEWB_PROGRESS=1`
makes stages emit `@MIEWB {json}` progress lines; every stage also
heartbeats `<case>/progress.json`.

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
# detector outputs: --photometric (lux maps) --spectrometer (λ-vs-x profiles);
#   QE-weighted photocurrent has NO flag — set qe_curve on the detector body
# time domain (pulsed round): --time-products pulse,spectrogram,streak,cube
#   (pulsed source auto-enables pulse,spectrogram; 'none' suppresses);
#   --time-bins preset-scaled 128/256/512; --time-window ns; --time-envelope
#   analytic|histogram; --gdd-budget = per-element GD/GDD/TOD table (free
#   when time products run; on CW forces group-delay tracking). time_products
#   and gdd_budget are PORTED to the C engine (P7 tranche 1: gopl/gdd
#   accumulators + arrival records + the bulk-path tally; dispersion resolved
#   Python-side, finalize_time/build_gdd_budget untouched — a crystal+time
#   scene still Python-routes via time_directional_index). ray_differentials +
#   the NLO bulk effects saturable/tpa/kerr are PORTED (P7 tranche 2:
#   intensity-dependent bulk alpha + Kerr opl phase; per-ray intensity
#   (p/dA)·κ from the ported differentials dA, else the source flat-top area).
#   Only the chi2 `nonlinear` token still Python-routes — SHG harmonic-child
#   strata + the Pockels index-shift split are a later tranche.
# scattering samples (samples-instruments): body's `sample` property ->
#   sample/samples.miesamp row (particle pop + optional S(q)/T-matrix
#   spheroid); --conical/--conical-fan/--conical-delta = biaxial internal
#   conical refraction (off by default); --ring-profile/--reference-case
#   (post stage) = log-annular sizer readout / UV-Vis absorbance
# traced DLS (dynamic light scattering, samples-instruments):
/home3/optics/env/bin/python scripts/run_dls.py \
    --model-json geometry/<stem>/model.json \
    --case-dir results/<stem>/<case> --frames 200 --dt-ms 1.0
/home3/optics/env/bin/python scripts/dls_correlate.py --case-dir results/<stem>/<case>
#   (both optics-env python; needs one EXPLICIT-mode `sample` body + a
#   coherent source; run_dls.py persists dls/frames.h5, dls_correlate.py
#   is offline/re-runnable -> g1/g2/D/hydrodynamic-diameter, docs
#   RAYTRACER.md §8.7)

# GUI
env/bin/python -m mieworkbench [model.FCStd|X.MieWB|X.MieSim]   # or bin/mieworkbench

# headless/remote (only needs a repo clone + tools)
python3 scripts/miewb_tool.py pack model.FCStd -o X.MieWB --simparams p.json
python3 scripts/miewb_tool.py run X.MieWB -o X.MieSim    # unpack→pipeline→pack
```

Tests:
```bash
/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -q   # engine (~1336; -m "not slow" for loops)
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
  `.meta.json`): 80 catalog elements (sources incl. 6 pulsed/SC
  lasers (P12: maitai_800/erfiber_1560/ndyag_1064/sc_superk/laser_pulsed/
  fiber_nonlinear_output)/detectors/fiber optics/
  lenses/beamsplitters/filters/polarization/prisms & mirrors/apertures/
  diffusers — incl. 8 LED monochromatic sources, `fiber_optic` core+cladding
  analytic cylinders, NA 0.22 via the `fiber_core_na22` material row,
  `mirror_annular` perforated SCT-style primary, and `iris_bladed` (P8:
  N-blade true-polygon aperture stop -> N-fold coherent diffraction star);
  samples-instruments adds 10: `cuvette_square`/`cuvette_capillary`/
  `flow_cell` (nested wall+liquid cells), `vial_cylindrical`/
  `vat_cylindrical` (DLS vial / decalin index-matching bath),
  `sample_region` (bare air anchor for unwalled clouds), `tungsten_halogen`/
  `d2_lamp`/`hg_calibration` (lamp sources), `source_image` (Lambertian
  USAF-style image emitter); geometry params live in a
  `dim` spreadsheet; edits go through **rebuild-on-edit** (`rebuild_primitive`
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
  Preview config is ONE dialog (`panes/previewdialog.py`, opened by
  "Live ray preview…"/Settings pointers): pattern (fan/rings widget +
  synced advanced spec text), trace ENGINE (`sequential` fast path vs
  `full` = forced MC subprocess with Fresnel ghosts; per-document
  `{"spec","engine"}` via `Project.set_preview_config`, QSettings
  fallback, default `full`), extinction (incl. `log` dB mode,
  `ray_dimming_range_db`; full-trace + Off auto-selects log), and the
  bead-anim keys (the old Settings Defaults tab is a pointer page).

## Body-tagging contract (details docs/RAYTRACER.md §5)

`App::Property` on each `PartDesign::Body`, group "Base":
- `material` (string): materials registry row | uniaxial crystal name |
  `detector` | absent/`none` → ignored
- detectors: `qe_curve` (string): detector-QE registry row → post reports
  photocurrent_A/coverage_frac; absent → no QE block (power report unchanged)
- sources: `power` (mW) + `lambdac` (nm) [+ `lambdamin`/`lambdamax` nm,
  `coherent` bool, `polarization` = `unpolarized` | `linear:<deg>` |
  `circular:left|right` | `elliptical:<psi>:<chi>`, `spectrum` =
  `emission/emitters.miesrc` row = tabulated/synthesized emission SPD
  (`kind` continuous/blackbody/lines; supersedes lambdamin/max,
  inverse-CDF equal-power strata — `lines` places strata at the line
  centers via Hamilton apportionment), `image` = `image/images.mieimg`
  row = per-position radiance bitmap (alias-method density sampling at
  equal per-ray power, Lambertian by default; optional `image_cone_deg`
  (0,90] cone restriction; excludes `beam`/`apodization`)]
- pulsed sources (P3/P6): `power` XOR `pulse_energy` (µJ, needs
  `rep_rate` Hz); `pulse_duration` (ps FWHM; on a power-only source =
  virtual pulse); derived {P_pk=0.94E/τ, κ} echoed in case.json
  source_pulse; power_mW=0.0 is the "unset" extractor sentinel. `spm` =
  `phimax:<rad>` | `gamma:<W⁻¹km⁻¹>:length:<m>` (source-side SPM:
  exact FFT spectrum installed as SPD + S-curve chirp via stratum
  birth-time offsets)
- optic extras (stackable): `coating` (whole-body or `Face3=MgF2;...`),
  `roughness`, `diffuser` (`grit:120`|`slope:0.08`|`@dg_600`, per-face ok;
  NEVER with roughness on one face — deep-rough Beckmann limit, §5.4.1),
  `filter`, `polarizer` + `polarizer_axis`, `crystal_axis`,
  `grating` (`Face2=600:v:orders=-1..1` or `Face2=@vbg_1800`),
  `surface_override` (asphere, verified <1 µm; the verifier needs the
  vertex inside the retained face — off-axis parabola segments are
  structurally unverifiable, use `mirror_parabolic` on-axis),
  `figure_error` (P8: per-face Zernike SURFACE figure error, `figure/
  figures.miefig` names -> a surfaces.PerturbedSurface sag perturbation at
  scene build; the CAD is the UNPERTURBED shape by design so the <1 µm gate
  checks base-vs-CAD only; Python-routed), `edge_blackened` (P8 bool: blacken
  the lens CYLINDER barrel = per-face absorbance on cylinder faces, immune to
  FaceN renumbering; Python-routed), `mirror`, `absorbance`, `sample`
  (samples-instruments: `sample/samples.miesamp` row binding a particle
  population to THIS body's interior — body's own `material` is the host
  medium, body shape bounds the cloud; optional S(q) structure factor +
  T-matrix spheroid shape; continuum mode C-ported, explicit/lattice mode
  stays Python-routed)
- NLO extras (pulsed round): `nonlinear` = nonlinear.mienlo row
  (chi2_process → per-segment SHG transfer: incoherent λ/2 child,
  stratum id n_λ+parent, η clamped 0.5; pockels rows + `pockels_voltage`/
  `pockels_gap_mm` = transverse EO via shifted-index proxies;
  chi2_tensor rows are authoring-only, hard error on a body),
  `saturable` (`@row`), `tpa_beta` (cm/GW), `kerr_n2` (`@n2_row`) —
  intensity uses (p/dA)·κ, ray differentials preferred (flat-top
  fallback warns; Kerr NEEDS --ray-differentials)
- dimensions in `Spreadsheet::Sheet` aliased `=<val> mm` cells; extract
  echoes the primary `dim` sheet FLAT plus every sheet namespaced
  `<sheetlabel>.<alias>`
- GUI-internal tags: `miewb_primitive` (builder kind), `miewb_group`
  (bodies of one multi-body element)

**Aperture scenes:** fill slit/hole openings with thin `material=air` bodies.

## Optical train / chain model (object-placer round)

Elements are **anchored** (absolute pose, the classic default) or
**chained** ("d mm down-beam of element X's port"). ONE solver —
`scripts/train_solver.py`, pure stdlib BY CONTRACT — is used by the GUI
(`mieworkbench/core/train.py` TrainModel + Project chain API) AND by
`permute_model.py` per variant (via `train_fcstd.py`), pinned to 1e-9 by
the parity oracle `mieworkbench/tests/test_train_parity.py`. Chained
placements are BAKED (files stay plain-FreeCAD editable); the recipe
lives in dynamic props (FreeCAD group `MieTrain`, names `miewb_train_*`
on the element's PRIMARY body): mode/ref/port/distance/decenter_x,y/
tilt_rx,ry,rz (expressions over the globals allowed)/rot_order/
pos_rot_order/pivot/flip/fold/folded/fold_deviation/fold_azimuth.
Distances are VERTEX-TO-VERTEX along the beam (exit vertex → entry
vertex); per-kind local port geometry = `primitivelib.port_frames`
(exact, param-derived, never FaceN). Ports: `out`/`transmit`
(pass-through: NEVER redirects the train, even for tilted/decentered
elements), `reflect` (the element's actual placed mirror plane),
`deviate` (explicit `fold_deviation`/`fold_azimuth` — gratings/prisms;
non-specular, wins as the default port when set). Default-port
heuristic: pure mirrors (entry==exit + reflect plane) default to
`reflect`. `flip` = beam-side surface is the local exit (the "turn the
lens around" affordance). Folds: `fold=True` elements toggle
folded/unfolded (Project.set_fold_state / set_folds_all /
insert_fold_mirror / fold_about_surface): unfolding re-collinearizes
the downstream chain (pass-through frame at the SAME port origin, so
distances keep meaning), stashes poses, and sets `miewb_exclude` on the
mirror's bodies (extract_geometry classifies them `ignored`); refold is
a bit-exact re-solve. Downstream ALWAYS follows rigidly (one undo macro;
`Project.move_element`/`sync_chain_from_pose` re-derive edge fields from
a spatial drag).

**Global variables**: a `miewb_vars` Spreadsheet (UNITLESS value cells
aliased `<name>`; sweep meta `<name>__min/__max/__n/__on`). Expressions
(`+ - * /`, constant `pi`, functions sin cos tan asin acos atan atan2
sqrt abs radians degrees — **trig is DEGREES-native** (matches tilt
fields; `sinr`/`cosr`/… take radians); cycle-checked with the full path
named; `train_solver.EXPR_HELP` is the one grammar string) usable in
chain fields, dim cells (FreeCAD expr `=<<miewb_vars>>.name * 1mm` — the
`* 1mm` is REQUIRED, and dim cells use FreeCAD's OWN expression engine,
not this grammar), float body props via `miewb_expr_<prop>`
(baked by GUI and permute), and ANCHORED element world poses via
`miewb_expr_pos_x/_y/_z` + `miewb_expr_rot_rx/_ry/_rz` (group `MieTrain`
on the primary body; `train_solver.place_anchored` bakes them inside the
shared `solve_chain`, so a goniometer detector at `pos_x=R*cos(theta)`
sweeps — anchored-only, a chained element with one is refused; a spatial
drag clears them to a literal pose like a chain drag re-derives its edge;
GUI editor = transform_panel "Pose expressions"). Editing a variable
rebuilds every primitive
whose dim sheet references `miewb_vars` (GUI:
`Project.apply_variable_cells`; headless:
`permute_model.extend_touched_for_miewb_vars`). Sweeps: `--sweep-mode
product|zip` (`common.sweep_combos` is the ONE combination-order
authority); the GUI writes `results/<stem>/sweep-<case>.manifest.json`
(RunController.write_sweep_manifest) consumed by
`scripts/compare_sweep.py` + the Compare pane.

**GUI surfaces**: Train editor dock (LDE-style indented tree; port
combo, fold/flip toggles, expression cells showing `expr (= value)`),
Variables dock (table + product/zip + pre-sweep summary dialog — ALWAYS
shown), Compare dock (metric-vs-variable plots, gallery, signed diffs,
scrub; also arbitrary-case comparisons), viewport ghosting
(`set_excluded_bodies`) + dotted chain/fold linkage lines
(`set_chain_links`), outliner badges (`set_train_info`), File→Export
FCStd (current fold state, `save_copy`).

**Demo equivalence**: `demos/baselines/*.json` are the pre-rebuild
oracles; `env/bin/python scripts/run_demo_equivalence.py` rebuilds every
demo and gates placements (≤1 µm; axis direction ≤0.01°, spin about a
symmetric element's own axis allowed+reported) and 3-seed power
(±max(3σ, 1%)) + michelson fringe visibility.

## Optical component library

`opticalproperties/` uses self-describing extensions (content is still CSV):
`materials.miemat` (849 rows, incl. `decalin` + `dye_solution_kmno4`),
`nk/*.mienk` (18 tables), `coating/coatings.miecoat`
(39), `polarizer/polarizers.miepol` (17), `filter/filters.miefilt` (56),
`grating/gratings.miegrat` (9), `birefringence/uniaxial.miebrf` (13 uniaxial
crystals) + `birefringence/biaxial.mibiax` (4 biaxial),
`nonlinear/nonlinear.mienlo` (14: chi2 tensors/processes, pockels, n2,
saturable), `diffuser/diffusers.miedif` (4), `scatter/bsdf.miebsdf` (4),
`instrument/instruments.mieinst` (4 classes shipped — camera/powermeter/
spectrometer/`diode_array`, the last a P12 physical linear-array readout;
+3 schema-defined placeholder classes with no rows),
`detector/detectors.miedet` (detector QE curves, 4 entries: hamamatsu_s1223
+ toshiba_tcd1304ap/sony_ilx511b/hamamatsu_s3904 linear-CCD rows),
`emission/emitters.miesrc` (tabulated/synthesized source emission spectra,
5 entries: led_white_2733k + sc_superk continuous, bb_halogen_3000k
blackbody, d2_uv_approx continuous, hg_penlamp lines; `kind` ∈
{continuous, blackbody, lines}),
`figure/figures.miefig` (P8: Zernike SURFACE figure-error sets — Noll
`j:rms_nm` coeffs + `r_norm_mm`, 4 entries: defocus/astig/trefoil/lambda10),
`sample/samples.miesamp` (samples-instruments: 7 scattering-sample rows —
particle material/size-distribution/loading + optional S(q) structure
factor (Percus-Yevick/Baxter/Teixeira-fractal/paracrystal/tabulated) or
T-matrix spheroid shape — bound to a scene via a body's `sample` property),
`image/images.mieimg` (samples-instruments: 1 extended image-source row,
`usaf_style_target`, bound via a source's `image` property),
per-item tables `*/tables/*.mietab`. Loaders prefer the new names and **fall back to
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
  that's a thin EDGE face and the run silently detects 0 mW. The
  authoring-time fix is the **`detector_face` body property** (bare
  `FaceN` or full `Body.Tip.FaceN`): it REPLACES the detector's primary
  face at extract time (no extra screen), so the scene stays
  C-engine-routable — unlike the additive CLI `--detector-face`
  (simparams `detector_face` list), whose `extra_detector_faces` screen
  is NOT in the C engine's ported set and silently forces Python. The
  demos bake `detector_face` pins (`make_demos.resolve_detector_pins`).
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
- **`set_placement` must be addressed by the element GROUP** for
  multi-body elements: `Project._flush_placement` resolves the body's
  `miewb_group` and sends THAT (op_set_placement group-matches first).
  Passing a member's internal name falls to the single-body path and
  TEARS the element apart (found by the achromat: no member carries the
  element label, the flint stayed behind while the crown moved 10 mm).
- **FreeCAD cross-sheet expressions need a unit multiplication**:
  `=<<miewb_vars>>.gap * 1mm` — bare `=<<sheet>>.alias mm` fails to
  parse and silently becomes a literal STRING cell (leading `'`).
- **Primitives' local origin is the FRONT vertex** (x=0) and SOURCES
  EMIT FROM local x=0 (the body extends toward -x) — chain distances
  from a source measure from its position, not position+length.
- **Never show an unguarded modal in a pane code path** — a
  `QMessageBox` in an offscreen test run blocks forever and looks like
  a hang (found via faulthandler in the snap-error path; guard on
  `isVisible()`, write to the status label otherwise).
- **train_solver.py must stay pure stdlib** (FreeCAD's python has no
  numpy) and `reflect_matrix` (det=-1) must NEVER touch a Placement —
  fold placements use the proper `fold_rotation` about the fold line.
- GUI features are verified interactively via `scripts/tools/gui_verify.py`
  (`xvfb-run`) — screenshots per scenario; run it before closing GUI work.
- **`fresnel.cos_theta_t`'s branch rule needs a radiation-condition
  carve-out** (`Re(n2·cos_t)>=0`) for the effectively-propagating regime —
  the unconditional `Im(n2·cos_t)>=0` decay rule flips a genuinely
  propagating root into a spurious evanescent one whenever the INCIDENT
  medium carries trace absorption (water, k~1e-8) into an exactly lossless
  far medium, exploding closure by `O(1e16)` on a curved nested interface
  (fixed; Python engine only, the C engine's rule already matched).
- **`Tracer.run`'s termination valve must be a PER-LINEAGE hop cap, never
  a shared/global pop budget** — a global counter consumed by `batch_size`
  chunk splits silently truncates live-eligible rays on a many-interface
  stack well before it should (found losing 37.8% of emitted power at 60k
  rays on a depth-4 nested scene; fixed by carrying each batch's ancestry
  step count through chunk splits unchanged).

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

## C engine (`cengine/`)

Compiled OpenMP+CUDA trace/gather core behind `--engine {auto,python,c}`
(default auto: C when every scene feature is in `PORTED` in
`scripts/raytracer/cengine.py`, else Python — the PERMANENT reference,
never behaviorally modified). Build: `cd cengine && ./build.sh` (gcc,
cmake, ninja; CUDA 13 at /usr/local/cuda-13 — the system nvcc is 11.5,
CMake pins the right one). Binary override: `MIEWB_CENGINE`. Parity:
`test_cengine_parity.py` (side-by-side scenes, root fuzz, TLAS==linear,
thread invariance). RNG is lineage-keyed Philox, C-engine-only
(bit-identical across thread counts); the Python engine uses numpy
default_rng and agrees statistically only — parity bar is 1e-9
deterministic / 2% statistical, single-seed (1e-12 on emitted_W only).
Engine+reason recorded in case.json; C failures
fall back to Python under auto. `--workers` is Python-only (C threads
internally). Benchmarks: `cengine/BENCHMARKS.md`;
docs: `docs/RAYTRACER.md` §13, `cengine/README.md` (incl. torch-gather
sunset roadmap). samples-instruments: `image_source` (extended
image-emitting source) and `sample_body` (continuum-mode `sample` body
property, region-gated by the medium stack, zero C-side S(q) logic) are
now PORTED; `conical` (biaxial internal conical refraction) and
`sample_explicit` (explicit/lattice-mode sample realization) stay
Python-only routing tokens.

## Everything else

`docs/RAYTRACER.md`: full engine reference (authoring contract syntax,
physics model + honest limits, schemas, CLI, 24-scene catalog + validation
table, troubleshooting). `README.md`: workbench/GUI tour + scripts +
formats. `INSTALL.md`: 22.04/24.04 setup from a clone + headless-server
install. `CUSTOMIZE.md`: authoring new primitives and property entries.
`future.md`: engine roadmap seams — open items only (optical contact,
curved detectors, GRIN, fluorescence, off-axis gyration, χ² C-port;
landed items are pruned each docs pass). `CHANGELOG.md`: per-round
changelog (c-engine round onward). `docs/README.md`: reader-facing doc
map. `docs/archive/`: historical design ledgers
(engine.md/engine2.md/engine3.md, UI_COORDINATE_PROPOSAL.md) — cited
for provenance, not maintained.

**Doc hygiene**: every round updates `CHANGELOG.md` (one per-round entry)
and the affected docs BEFORE merging — doc drift is a bug.
