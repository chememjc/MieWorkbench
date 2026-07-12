# future.md — staged follow-ups with exact starting points

Change-management notes in the pcbsim style: what to run/extend next and
where the seams are. Nothing here blocks current use. This is now the
**single roadmap document**: the untracked `lowhanging.md` (highest-impact/
lowest-effort backlog + the named-analysis-products and biaxial/stress
birefringence design studies) was merged in here on 2026-07-11 and deleted.

## Delivered since the last pass (kept here only as a pointer to the flag/module)

- **Ray-differential dA tracking** — shipped behind `--ray-differentials`
  (`scripts/raytracer/differentials.py`, Igehy 1999 method; README §6.4).
  Lost (NaN, falls back to source-referenced area) through grating orders,
  roughness/particle scattering lobes, and birefringent o/e splits — those
  three transport paths are the natural next increment if exact dA is
  needed everywhere.
- **Polarization-resolved explicit Mie azimuth** — shipped as the default
  scattering-azimuth sampler in `particles.py` (samples the true polarized
  differential cross-section `|S1|^2|Es|^2 + |S2|^2|Ep|^2`); the old
  uniform-azimuth-with-rescaling approximation is kept as an opt-in via
  `--no-pol-scatter` for A/B comparison (README §6.2/§9).
- **Gather occlusion** — shipped behind `--gather-occlusion`
  (`gather.py`'s two-level AABB-prefilter + tile-shadow-ray implementation;
  README §6.5). Still tile-quantized (default 16 px) and models every
  occluder as fully opaque (deliberate, to avoid double-counting a
  refracted field that already re-anchors past the occluder) — a smaller
  `occlusion["tile"]` (down to 1) is today's only way to sharpen a shadow
  edge; a true per-pixel default would need a cost/quality tradeoff
  decision, not new math.
- **Mesh-face tracing** — shipped by default (`mesh.py`'s BVH +
  Möller–Trumbore; README §5.8/§6.1). `--strict-analytic` restores the old
  hard-error behavior. Coherent phase through a mesh face is still
  explicitly out of scope (sag error >> lambda) — mesh faces remain
  incoherent-power-only; a source or detector face still cannot be
  mesh-type.
- **Polarization state, strata, and birefringence** — shipped in full:
  source `polarization` property (unpolarized/linear/circular/elliptical),
  `polarizer`/`polarizer_axis`/`filter` body properties, uniaxial
  `crystal_axis` double refraction, gather keys extended to `(source,
  lam_stratum, pol_stratum)` (README §5.2/§5.3/§5.6).
- **Gratings beyond scalar lamellar** — shipped: `bragg_kogelnik`
  (Kogelnik thin-hologram transmission, polarization-resolved), `dammann`
  (exact Fourier-order), `table` (measured per-order eta_s/eta_p), plus a
  registry (`opticalproperties/grating/gratings.miegrat`, `@name` syntax).
  RCWA is still explicitly out of scope (README §5.5).
- **Aspheres** — shipped: `surface_override='FaceN=asphere:...'`,
  extract-time verification against the real FreeCAD face to 1um,
  bracket-guarded Newton intersection (README §5.7).
- **`--save-fields` (complex Ex/Ey field export + Stokes/DOP maps)** —
  shipped end-to-end: `run_trace.py --save-fields` writes
  `detectors/<label>.h5`'s `fields/<key>/{Ex,Ey}` groups (seed 0 only),
  and `post_process.py`'s `render_stokes_maps()`/`stokes_from_jones()`
  render `stokes_<label>_<key>.png` + `dop_<label>.png` from them
  (README §6.5). As of the `lowhanging-improvements` round it also has a
  prominent, always-visible GUI checkbox ("Save coherent fields (enables
  Stokes/PSF/MTF)" in the Run Pipeline config matrix) — still opt-in, not
  default-on. See below for the named-analysis-products this unlocked.

### `lowhanging-improvements` round (landed 2026-07-10) — named analysis products, CSV export, biaxial, and more

Twelve of the 15 ranked items in the former `lowhanging.md` §1 landed this
round (two more — photometric units, glass-catalog import — had already
landed 2026-07-09 in the `library-expansion` round; one — the fold operator
— in the `object-placer` round). Compact record, STATUS facts only (design
prose is in git history / superseded by the shipped code):

| # | Item | STATUS | Code seam |
|--|--|--|--|
|1|`--save-fields` GUI surfacing|**partial** — prominent checkbox, still opt-in|`run_trace.py` save gate; `mieworkbench/panes/config_matrix.py`|
|2|Photometric units (lux/lm/cd via CIE V(λ))|**done** (`library-expansion`, 2026-07-09)|`post_process.py` post-multiply of `spectral_cube_mean`, `--photometric`|
|3|Per-(source,detector) detected power|**done** (2026-07-10)|`case.json["detected"]` (coherent+incoherent) → `report.json` `per_source` + `data/source_detector.csv` + Results "Sources" tab|
|4|Unified CSV data export|**done** (2026-07-10)|`post_process.CsvEmitter`, `--emit-csv`, `data/index.csv` (every library-derived CSV carries its registry `reference` column)|
|5|PSF + FFT-MTF + encircled/ensquared energy|**done** (2026-07-10, needs `--save-fields`, seed-0 only)|`raytracer/analysis_field.py` + `post_process.render_field_analysis`|
|6|Public glass-catalog import (Schott/Ohara Sellmeier)|**done** (`library-expansion`, 2026-07-09; 168 materials incl. 41 Schott/Ohara)|`opticalproperties/materials.miemat`|
|7|Spot diagram + transverse/OPD ray fans|**done** (2026-07-10, needs `--export-rays`)|`run_trace.write_rays_full` → `rays_full.npz`; `post_process.render_spot_diagram`/`render_ray_fans`|
|8|Gaussian-beam source + apodization|**done** (2026-07-10)|`scripts/raytracer/sources.py` (`beam_waist`/`m2`/`apodization`) + `element_wizard.py`|
|9|Ghost/stray-light analysis|**done** (2026-07-10)|`RayBatch.refl_hist` → `post_process.render_ghost_analysis`, `--ghost-analysis` (implies `--export-rays`)|
|10|BSDF/ABg measured-scatter import|**done** (2026-07-10), v1 BRDF-only|`raytracer/scatter.py` + `opticalproperties/scatter/bsdf.miebsdf`; BTDF (transmitted-side) not built|
|11|Fold operator + relative optical-train chaining|**done** (`object-placer` round)|`mieworkbench/core/project.py`/`train.py`, `scripts/train_solver.py`|
|12|Multi-process ray sharding (`--workers`)|**done** (2026-07-10)|`run_trace._run_sharded`, `SeedSequence.spawn`; `--workers 1` bit-identical, `N>1` statistically equivalent|
|13|Zernike wavefront + Strehl|**done, v1 pupil model** (2026-07-10)|`raytracer/analysis.py` + `post_process.render_wavefront`; pupil is **source-referenced** (exact for collimated benches), not a true exit pupil — see backlog|
|14|Biaxial-crystal birefringence|**done** (2026-07-10) — KTP/KTA/LBO/BiBO|`raytracer/birefringence.py` biaxial extension + `birefringence/biaxial.mibiax`; conical refraction not modeled (honest limit, README §5.6b)|
|15|Stress/spatially-varying birefringence|**deferred** — see Backlog (a)|—|

Also landed this round: curved (sphere/cylinder) detectors, incoherent path
only (`CurvedDetectorGrid`, per-pixel `pixel_area_map`; coherent Huygens
gather on a curved screen still raises — see Backlog (a)).

**Reference obligation carried forward:** every `data/*.csv` emitted by
`--emit-csv` that contains library-derived values (n/k, R/T, coating,
filter, BSDF rows) MUST include the `reference` column copied from the
source registry row — this is enforced project-wide, not just for the new
CSVs.

### C engine round (branch `c-engine`, in progress)

A compiled OpenMP+CUDA trace/gather core (`cengine/`, binary
`miewb-trace`) behind `--engine {auto,python,c}` (default `auto`: routes to
C only when every scene feature is in `PORTED` in
`scripts/raytracer/cengine.py`, else falls back to Python — the Python
engine remains the PERMANENT reference, never behaviorally modified).
Benchmark (`cengine/BENCHMARKS.md`, `scripts/bench_engines.py`, git
`293ceb1`, RTX 4090 Laptop + 32-core CPU, 11 scenes, 1e6 rays/2048²/9λ):
**8.3x wall-clock geomean, 10.6x trace+gather-stage geomean** vs the
Python engine (`--workers auto` + torch-CUDA gather baseline); per-scene
speedups range 2.8x (czerny_turner) to 24.8x (microscope_objective).
Feature phases A-I are all **done** (plane/sphere/cylinder/cone/torus/
asphere, TMM+table coatings, polarizers, scene-wide TLAS+mesh BLAS,
coherent Huygens gather (CUDA+CPU), gratings, Beckmann roughness/
diffusers, ABg scatter, uniaxial birefringence, continuum-mode particle
clouds, export-rays/ghost-analysis/viz-pattern, `--importance-aim`) — see
the `PORTED` feature table in `cengine/README.md` §Feature status for the
authoritative per-phase status and what still Python-routes (biaxial
birefringence, explicit particle realizations, `--ray-differentials`).
`--workers` is Python-only; the C engine threads internally via OpenMP.
Docs: `docs/RAYTRACER.md` §13, `cengine/README.md`, CLAUDE.md's C-engine
section. **Torch-gather sunset roadmap** (`cengine/README.md` §Sunset
roadmap): (1) through shakedown/merge, torch gather stays as the 100%
fallback and three-way parity reference (numpy/torch/CUDA); (2) after a
post-merge shakedown period, retire the torch gather backend and its ~5 GB
dependency from the optics env; (3) eventually sunset the Python compute
paths for day-to-day use — the numpy engine remains indefinitely as the
slow, readable parity reference.

## Backlog

### (a) Near-term, carried forward from `lowhanging.md` §6

- **Stress/spatially-varying birefringence** (was lowhanging.md §4.2/§15,
  effort L-XL). Photoelastic/injection-molded/thermally-stressed
  birefringence is a 2-D (or 3-D) retardance field that varies across the
  element — reuses the *uniaxial* o/e machinery (physics per point is
  still a small index anisotropy) but breaks the constant-per-body
  assumption `scene.uniaxial_indices(body, lam)`/`body.crystal_axis` rely
  on. Needed: index tensor and optic axis become functions of the hit
  point `p` (`n(p)`, `c(p)` from a stress-optic model `Δn = C·(σ1−σ2)` with
  a per-body stress field); the trace loop must re-evaluate the tensor per
  segment (today fetched once per body — no mechanism carries a
  spatially-varying tensor through the loop); needs a stress-field input
  (analytic parametric map, or an imported FEA field coupling to the STOP
  gap in `features.md` §7.8). **Recommended first cut** (closer to L):
  an analytic parametric retardance map with **constant optic-axis
  orientation, position-dependent Δn only** (e.g. radial/edge-load
  formula) — demonstrates photoelastic fringes without the curved-ray
  complication; hold the axis constant to avoid a GRIN-like curved walk-off
  on a first pass. Full stress-optic + FEA-import path is XL. Hook points:
  make `scene.uniaxial_indices`/`crystal_axis` accept `p` (or add
  `scene.local_birefringence(body, p, lam)`), thread `p` into
  `_birefringent_children`, add a `stress_optic` body property + a
  stress-field model module; `birefringence.py`'s per-point math is reused
  as-is. Acceptance target: the `photoelastic_stress` demo
  (`demosystems.md` §3.8) — `source_broadband` (white, isochromatic color
  fringes) or monochromatic (dark fringes), unpolarized, through
  `polarizer_plate` → stressed window (new `stress_optic` property, e.g.
  edge/point load) → crossed `polarizer_plate` (a circular-polariscope
  variant adds `waveplate`s), imaged by `detector_plane` on the sample —
  demonstrates position-dependent retardance + polariscope fringes; also
  ties to `biaxial_conoscopy` (§3.7, already landed) as the sibling
  crystal-optics demo.
- **Exit-pupil/chief-ray search stage.** The Zernike/Strehl wavefront
  analysis (`raytracer/analysis.py`, README §6.10) shipped with a
  source-referenced pupil (each ray's normalized birth position on the
  emitting face) — exact for the collimated/laser benches this tracer
  models, but not a true exit pupil for finite-conjugate, field-point
  imaging. A real exit-pupil/chief-ray search (reference sphere centered
  on a field point's image, locate the chief ray, sample the pupil
  relative to it) would also unlock a PSF-peak-ratio Strehl (measured PSF
  peak vs. a diffraction-limited reference — needs the same missing
  reference-sphere concept, cross-checking `--save-fields` against
  `--export-rays`).
- **BTDF (transmitted-side) measured scatter.** The shipped `scatter`
  property (`raytracer/scatter.py`, README §5.4.2) is reflected-side
  (BRDF) only, v1; a scattering lens/window exit face currently transmits
  its Fresnel/TMM child unmodified.
- **Coherent gather on curved detectors.** `CurvedDetectorGrid` (README
  §5.12) shipped incoherent-only as scoped — the planar Huygens gather
  kernel assumes a flat aperture; `add_gather_samples()` raises on a
  curved screen. A curved-aperture gather needs per-pixel normals/
  obliquity terms threaded through. Acceptance target: `curved_focal_surface`
  demo (`demosystems.md` §3.9) — a fast wide-field singlet/Schmidt with
  deliberate field curvature, several tilted `laser_collimated` field
  angles @550, imaged onto a spherical `detector_plane` matching the
  Petzval surface vs. a flat one for comparison.
- **Materials `dn/dT` (thermo-optic) hook.** Noted since
  `library-expansion`; needs a small temperature parameter threaded
  through `materials.py`'s dispersion evaluation — data already compiled
  in `library.md`. Shared with the thermal-lensing item below (§b) and
  `features.md` §7.8 (STOP).
- **`--save-fields` caps + estimator wiring** — **DONE
  (design-usability round, 2026-07-11)**: `--save-fields-detectors
  LABEL[,...]` restricts field writes per detector (hard error on unknown
  labels; a real subset Python-routes since the C engine writes all), and
  `common.estimate()`'s `fields_h5_GB` now reflects save-fields state,
  the detector subset, and (source, lambda, pol) key count. One follow-up:
  incoherent-only scenes now WARN that fields groups will be empty.
- **White-LED/blackbody/lamp tabulated spectra** — continuous tabulated
  spectra + the CIE 015:2018 LED-B1 white LED **LANDED (design-usability
  round)**: `spectrum` body property → `opticalproperties/emission/`
  registry, equal-power quantile strata in `sources.wavelength_strata`
  (zero C-engine changes, scenes stay C-routable), `led_white` primitive.
  Line-spectrum and blackbody/lamp source kinds are still open (the
  loader rejects those `kind`s with "needs engine support").

### (a2) Placement/authoring affordances (design-usability round findings)

From `demos/UXNOTES_ROUND3.md` — the pain points too large for that
round's fix loop (each names its seam):

- **Expressions/variables for ANCHORED placements.** Chain edges accept
  the full expression grammar; anchored poses are literal xyz/quat only
  (`Project.apply_operation` / `transform_panel`). The moment a pose
  isn't a beam relationship (a 90° side-scatter detector, a field-source
  fan) you drop to hand-computed literals that can't sweep. Natural
  shape: `miewb_expr_*`-style expression baking for placement fields, or
  a polar/spherical place-about-point operation in `core/transforms.py`.
- **A `--particles` cloud is not a chain-referenceable body** — no way to
  chain a detector "40 mm at 90° from the cloud center" (nephelometer
  ring). Needs a lightweight non-solid "region anchor" element the train
  solver can reference (`train_solver` port_frames + an extractor-ignored
  marker body, or a virtual element in the recipe).
- **Field-angle source fan helper.** N collimated sources at a common
  pivot overlap as solids; they must be spread on an arc by hand
  (y = L·tanθ). A wizard ("fan of field angles: N, ±θ, pivot") placing
  them chained/anchored would remove the trig (`element_wizard.py`).
- **Co-located transparent detectors** overlap-fail extraction — no
  authoring path for "measure the same plane two ways" (needs either
  zero-thickness detector sheets or an extractor exemption for
  detector-detector overlap).
- **Coherent-gather ray-budget preflight.** Aperture-diffraction scenes
  have an implicit `rays >> (beam/aperture)² · 1000` requirement the
  presets don't know about; the GatherError names the fix only AFTER a
  failed trace. `core/validation.py` could estimate transmitted-fraction
  × rays against the M_eff gate at check time (a coarse aperture-area
  ratio suffices).
- **`--particles` target-optical-depth knob.** phi is opaque (the
  aerosol demo needed 4 orders of magnitude off the spec'd value to make
  τ visible); `parse_particles_spec` + the Mie ensemble tables could
  accept `tau=1.0` and solve phi for the box length.
- **"Span N Airy zeros" detector-sizing intent** and other
  diffraction-scale insert-values for the right-click menu
  (`core/opticalvalues.py` — needs aperture+distance context).

### (b) Higher-fidelity physics (still open)

- **Exact uniaxial Fresnel at a birefringent interface.** The current
  model decomposes the incident field into the o/e eigenbasis and applies
  each channel's own *isotropic*-effective-index Fresnel coefficients
  (`n_o`/`n(theta)`) rather than solving the true anisotropic boundary-
  value problem; energy still closes exactly via the ledger, but per-
  channel phase/amplitude is an approximation, worst near grazing
  incidence (`tracer._birefringent_children`, README §5.6). A rigorous
  fix solves the 4-wave (2 incident + 2 reflected, or +2 transmitted)
  boundary-matching problem directly.
- **Optical activity / chiral media (e.g. quartz's rotary power along its
  own optic axis).** Not modeled at all today — `birefringence.py`'s
  header explicitly scopes this out. Needed for a physically complete
  quartz-along-axis scene (Babinet-Soleil compensators, saccharimetry).
- **Biaxial crystals — conical refraction only.** Biaxial birefringence
  itself LANDED 2026-07-10 (`raytracer/birefringence.py`
  `refract_in_biaxial()`/`biaxial_modes_for_k()`, quartic normal-surface
  root solve via companion-matrix eigenvalues; `birefringence/
  biaxial.mibiax` registry; KTP/KTA/LBO/BiBO ship, 15 tests in
  `test_biaxial.py`). The remaining open limit is **conical refraction
  near an optic axis** — degenerate eigenvectors there return an
  arbitrary transverse basis; documented as an honest limit rather than
  solved (README §5.6b). A conical-refraction validation scene would be
  the natural next increment.
- **Absorbing (dichroic) uniaxial crystals.** `Im(n_o)`/`Im(n_e)` are
  currently ignored for geometry (real indices only); the o-ray's index
  stands in for bulk absorption of both modes. Needed for tourmaline-like
  or intentionally-doped-crystal scenes.
- **Reflection-geometry Kogelnik gratings.** `bragg_kogelnik` only
  implements the thin-hologram *transmission* coupled-wave solution;
  reflection VBGs need the tanh/sinh reflection-geometry solution instead
  of the sin-based transmission formula (`grating.py`'s own header names
  this gap explicitly).
- **RCWA (rigorous coupled-wave analysis).** Still out of scope for every
  grating model; would replace/supplement `lamellar`/`bragg_kogelnik`/
  `dammann` for sub-wavelength or highly non-sinusoidal groove profiles
  where the current closed-form/thin-element models break down.
- **Ray-differential transport through gratings/scatter/birefringence.**
  The differential machinery exists and is correct for reflection/
  refraction/free-space transfer; extending it through a diffraction
  order, a Beckmann-scattered lobe, or an o/e split is the natural next
  increment (`tracer._kill_differentials`'s call sites are the exact
  insertion points).
- **Tile-quantized / opaque-only gather occlusion.** A per-pixel
  (`tile=1`) mode already exists as an opt-in; a translucent/partial
  occluder model (rather than always-fully-opaque) would need a
  coupled-transmission accounting scheme to avoid double-counting the
  tracer's own refracted-field samples.
- **Stress birefringence.** See Backlog (a) — carried forward with a
  first-cut design now specified.
- **GRIN (gradient-index) media.** Not modeled — every medium is currently
  homogeneous between interfaces; a GRIN element needs curved-ray
  propagation inside the bulk (a genuinely different integration scheme
  from the current straight-segment-between-hits loop; also named in
  `features.md` §7.4/§7.8).
- **Fluorescence / phosphors.** No wavelength-shifting absorption-then-
  reemission event exists; would need a new emission event type
  (isotropic, incoherent, at a shifted wavelength stratum) triggered by
  bulk or surface absorption in a fluorescent material.
- **Measured BSDF scatter — BTDF half.** See Backlog (a); the BRDF half
  landed 2026-07-10.
- **Ghost-image analysis mode.** LANDED as `--ghost-analysis`
  (`RayBatch.refl_hist` → `post_process.render_ghost_analysis`,
  2026-07-10) — groups generation->=2 purely-specular detector hits by
  ordered face-id path signature, ranks by summed detected power, emits a
  top-12 bar chart + top-3 footprint images + `data/ghost_table_<label>.csv`.
- **Thermal lensing.** No temperature-dependent index/absorption coupling
  exists; would need a coupled thermal (absorbed-power -> local
  temperature -> `dn/dT` -> refractive index) model, likely iterative.
  Shares the `dn/dT` hook in Backlog (a).
- **Curved detector faces.** Incoherent path LANDED 2026-07-10
  (`CurvedDetectorGrid`, sphere/cylinder, auto-chosen by face surface
  type, per-pixel metric area map). Coherent Huygens gather on a curved
  screen is still open — see Backlog (a).

### (c) Capability gaps (hard errors today, by design)

- **Explicit particle clouds > `MAX_BRUTE` (200,000) spheres** are capped
  (brute-force chunked collision; `--particle-threshold` default is now
  aligned to this same cap, README §9). A numba DDA/uniform-grid traversal
  removes the cap; grid build is already cell-hashed in
  `ExplicitRealization._place`. Note: the C engine's continuum-mode
  particle-cloud phase (G, `cengine/README.md`) is done for the
  Mie-ensemble-table path; explicit realizations above the cap still
  Python-route regardless of engine.
- **Mesh-type source/detector faces** still hard-error unconditionally
  (README §5.8/§5.11) — both need a UV parameterization the incoherent/
  coherent paths don't have yet; the ordinary-optic mesh path (BVH tracing)
  is otherwise fully shipped.
- **Aspherical particles** (user goal): T-matrix (e.g. `pytmatrix`) drop-in
  behind the `MieEvaluator` interface — `efficiencies()` and `amplitudes()`
  are the only two entry points `particles.py` uses. Still open.

Multi-process tracing (`--workers`) and multi-process ray sharding are no
longer gaps — both landed 2026-07-10 (`run_trace._run_sharded`,
`SeedSequence.spawn` + linear ledger/cube merge). The C engine round layers
on top of that: `--engine c` gets its parallelism from OpenMP threading
internally (`--threads`, 0 = all cores) and supersedes most of the
raw-throughput story for `PORTED` scenes (8.3x wall geomean, see above);
`--workers` remains the Python-engine-only sharding path for scenes that
still Python-route.

### (d) Big-roadmap acceptance-target demos (`demosystems.md` §4)

Specified so the roadmap has concrete acceptance targets; each names the
`features.md` gap it needs. `folded_periscope` (a straight relay, then
one-click "insert fold mirror" twice → periscope, downstream train
reflects rigidly) is **not** listed here — the fold operator landed
(§11 above) and the demo itself is being built in the design-usability
round.

- **auto_designed_lens** (optimizer, `features.md` §7.2 — *the biggest
  categorical gap*). Start from a poor doublet; a merit-function optimize
  (spot RMS / encircled energy) + glass substitution converges to a
  corrected design. Pragmatic path: wrap the existing FreeCAD
  spreadsheet-parameter sweep (`permute_model.py`/`--var`) as an
  optimization loop — scipy.optimize (least_squares/differential_evolution)
  or nevergrad/CMA over a merit function built from the named analysis
  products (spot RMS, encircled energy, detected power); start with a
  headless `scripts/optimize.py`, add a GUI merit-function panel later.
  Per-iteration cost is a full FreeCAD rebuild -> extract -> trace via
  `permute_model.py` plus the coherent gather — mitigate with geometry
  caching for unchanged bodies and a geometric-only fast mode
  (`coherent=false`, direct deposit) for the inner loop, refining
  coherently at the end. Effort: headless optimizer L; GUI M; global XL.
- **tolerance_yield** (tolerancing, `features.md` §7.3). Take
  `camera_triplet`; a Monte-Carlo tolerance run over radius/thickness/
  decenter/tilt with a focus compensator -> yield histogram. Pragmatic
  path: `scripts/tolerance.py` perturbs the FreeCAD model per a tolerance
  table, runs the (geometric-fast) pipeline N times using the existing
  `--seeds` + `permute_model.py` machinery, aggregates a merit-metric
  distribution + sensitivity ranking; compensators = a nested §7.2
  optimize call per draw. Effort: sensitivity M; MC tolerancing L;
  compensators L.
- **cad_import_scene** (CAD import, `features.md` §7.4). A STEP-imported
  lens barrel + baffles traced as optomechanics (stray light) around an
  existing optical train. FreeCAD already imports STEP/IGES; the missing
  piece is the GUI exposing "Import STEP as element" through the
  fc_server worker (`import_bodies`/`import_primitive` ops) and the
  extractor canonicalizing imported faces (falling back to the existing
  mesh-BVH path for non-canonical ones — already shipped, incoherent-
  only). Effort: M for mesh-import; L for analytic-face recovery.
- **freeform_illuminator** (illumination design, `features.md` §7.9). An
  LED + freeform reflector/TIR lens optimized to a prescribed irradiance
  target with photometric units. Non-imaging design (freeform tailoring)
  rides on the §7.2 optimizer above; photometric units (lux/lumen/candela
  via CIE V(λ)) already landed (item #2 above). Effort: L (depends on
  §7.2 landing first).

## Operational

- Estimator calibration: `results/.calibration.json` self-improves per run;
  seed it with a few normal-preset runs so `--dry-run` predictions tighten.
  It also does not yet fold in the extra cost of `--ray-differentials` or
  `--gather-occlusion` (README §12) — both add real wall time/memory beyond
  the base `gather_ops` model.
- `sweep_variants.py --jobs jobs.json` + `compare_runs.py` are wired; a
  worked multi-job example lives in README §sweeps.
- Detector `.h5` at detailed preset (4096^2 x 32 bins) ≈ 2 GB/detector; mind
  `/` at ~97% (outputs can be pointed at /home3 via --case-dir).
  `--save-fields` adds two more full-resolution complex float64 arrays per
  `(source, lam, pol)` key on top of that (seed 0 only) — budget
  accordingly for polarization-heavy, high-resolution `--save-fields` runs.
- `run_pipeline.py` forwarding of `--views`/`--smoke`/`--viz-generations`
  — **DONE (design-usability round)**: all three now flow through the
  pipeline; only `--resolution`/`--out`/`--skip-vtkexport` remain
  direct-call-only (README §4.2/§8).
- `--save-fields-detectors` (design-usability round) caps field writes
  to named detectors; per-KEY capping is still all-or-nothing.

## Known cosmetics

- `wavelength_rgb` saturates deep red/violet ends (CIE tails); purely
  cosmetic in the sRGB detector images.
- `rays_polmode`'s ParaView view silently skips (with a warning) if
  `viz/rays.vtp` predates the `pol_mode` array (a stale `viz/` combined
  with `--skip-vtkexport`) — rerun without `--skip-vtkexport` to refresh it.

## Scene-suite findings (test_scenes_e2e.py xfails — follow-up work)

Four strict xfails remain in `test_scenes_e2e.py` (wollaston, previously
xfail'd, now PASSES after the trim-loop head-to-tail orientation fix);
the tracer is faithful in each case, the follow-up is scene authoring /
model extension:

- **lens_asphere design math — FIXED (design-usability round)**: the
  front profile was re-solved for the COMPLETE lens (k=-1 + A4, exact
  meridional trace + Nelder-Mead); the scene now beats the spherical
  control 17x and the test is un-xfailed. `wizards.solve_asphere` and the
  `lens_asphere` primitive (new `A4_mm3` param — `A4` is a spreadsheet
  CELL ADDRESS, rejected as an alias) carry the corrected design.
- **prism_equilateral geometry — FIXED (design-usability round)**: the
  rotation had the wrong SIGN (-19.399 deg gives the 49.4 deg
  minimum-deviation entry); deviation now matches delta_min analytics
  (38.76 vs 38.80 deg) with blue>red ordering, the detector sits face-on,
  and the test is un-xfailed. Two strict xfails remain in the suite
  (pol_circular, pbs_cube — below).
- **pol_circular is a GENERATOR, not an analyzer**: the polarizer model
  applies linear diattenuator -> retarder in propagation order, which
  cannot discriminate incident handedness (left/right transmit equally).
  Analyze circular light with a quartz waveplate body + linear polarizer
  body (fully supported, validated by waveplate_quartz). A future
  `orientation` column in polarizers.miepol could flip the stage order. Open.
- **pbs_cube air gap**: without optically-contacted solids, the 5 um gap
  Fresnel-splits (PBS loses ~35% to seam/ghosts; reflected-arm detector
  also edge-on — see the detector_face item below). Real fix =
  cemented/contact interface support: paired coincident faces treated as
  a single material-to-material boundary (no ambient hop). Until then the
  scene documents the physics of an *air-gapped* assembly. (The wollaston
  scene's related anomaly turned out to be the multi-edge trim-loop bug,
  fixed in `extract_geometry.trim_polylines_xyz` — its e2e test now
  passes clean; nested/coated-plate builds are the shipped workaround for
  45° cemented interfaces, see bs_cube/pbs_cube in CLAUDE.md.) Open.
- **Rotated off-axis detector bodies — ADDRESSED (design-usability
  round)**: the `detector_face` body property pins the recording face at
  authoring time and (unlike the CLI flag, which adds an extra screen)
  keeps scenes C-engine-routable; demo pins are BAKED by make_demos, a
  pre-run validation check warns on rotated unpinned detectors, and the
  GUI exposes a face combo. The e2e prism/pbs scenes could now be
  migrated from direction-based readouts to pinned faces (small
  follow-up).

## Pulsed-optics round follow-ups (2026-07-12)

Engine seams deliberately deferred by the pulsed/fs round (each is
documented as an honest limit in docs/RAYTRACER.md §5.2.1/§6.11/§6.12):

- **Split-step NLSE propagation**: the SPM transform is source-side and
  quasi-classical (one FFT, single-time-per-frequency chirp). A split-step
  Fourier propagator would give real intra-train nonlinear evolution
  (SPM+GVD interplay, soliton dynamics) at the cost of a per-segment field
  model.
- **Mid-train SPM**: an `spm` property on a fiber/waveguide BODY (not the
  source) breaks the per-source wavelength-strata bookkeeping — needs
  stratum re-quantization at the element, same machinery an OPO/Raman
  element would need.
- **Depleted-pump coupled-wave SHG**: the bulk event clamps the undepleted
  quadratic η at 0.5. The full coupled-amplitude tanh² solution would
  extend validity to strong conversion (and enable back-conversion).
- **Harmonic walk-off + exact uniaxial SHG**: the harmonic child is
  collinear with equal s/p split; real type-I/II geometry puts it in the
  e/o eigenpolarization with Poynting walk-off (needs the exact uniaxial
  Fresnel work already listed above).
- **Cascaded/coherent harmonics**: children are incoherent and never
  re-convert (no THG via cascade, no phase-sensitive pump-harmonic
  interplay). A coherent-harmonic mode would gather the 2ω population
  with its own phase ledger.
- **Raman / fluorescence-style inelastic transfer**: the SHG event is the
  template (stratum id extension + ledger transfer); a Stokes-shift bulk
  event would reuse the same plumbing with a gain spectrum row.
- **C-engine port of the round's tokens**: `time_products`, `gdd_budget`,
  `nonlinear`, `saturable`, `tpa`, `kerr` all Python-route today
  (`cengine.detect_features`). The arrival-record buffer + per-segment
  alpha hooks are the natural first ports; the SHG child spawn needs the
  C children queue to learn stratum extension.
- **Fringe-resolved timing**: the coherent population records its
  GEOMETRIC arrival power — interference within a time bin is not
  resolved (a coherent time-domain gather would need per-record complex
  amplitudes at 100× the record cost).
- **Angular-dispersion group-index term**: e-ray group delay uses the
  frozen directional n_g and neglects dθ/dλ (calcite oracle bounds the
  error); gratings/prisms get geometric GDD only through traced arrival
  times (correct) — the analytic GDD-budget table stays material-only.
- **Per-source time cubes / .h5 growth**: time products bin all sources
  together (profile has a by-source split; the cube does not) — a
  per-source cube would multiply memory by n_sources.
- **SuperK SPD tail vs material tables**: the sc_superk table spans
  400–2400 nm; benches whose materials aren't tabulated that far must
  clip the SPD (documented in the primitive tooltip + registry notes).
