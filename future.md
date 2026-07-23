# future.md — staged follow-ups with exact starting points

Change-management notes in the pcbsim style: what to run/extend next and
where the seams are. Nothing here blocks current use. This is now the
**single roadmap document**: the untracked `lowhanging.md` (highest-impact/
lowest-effort backlog + the named-analysis-products and biaxial/stress
birefringence design studies) was merged in here on 2026-07-11 and deleted.

## Rating legend (effort × impact)

Every open item below carries a **`[effort · impact]`** tag, unifying the ad-hoc
effort sizes that used to be scattered through this doc. Same effort tiers as
`features.md` §7.

- **Effort:** **S** ≤1 wk · **M** ~1 mo · **L** ~1 quarter · **XL** multi-quarter/research.
- **Impact:** **High** = removes a categorical gap that most/all competitors win
  (see `features.md` §4) or a broad user-pain point · **Med** = matches some
  competitors or a solid quality-of-life win · **Low** = niche audience,
  moat-widening on an already-won axis, or a corner-case fidelity fix.

The consolidated **Roadmap rating index** (after the Backlog, before Operational)
rates *every* open item in one scannable table; the inline items keep their
narrative + code seams.

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
  RCWA landed as a **precompute-and-interpolate** path:
  `scripts/tools/gen_rcwa_table.py` tabulates complex per-order amplitudes
  via meent (Li inverse-rule RCWA) on a (λ,θ,φ) grid with
  Rayleigh/Wood-anomaly-aware refinement, interpolated at trace time
  (`grating._v2_amplitudes`, `grating_table_v2` feature — Python-routed,
  not yet C-ported). A true per-ray RCWA solve remains out of scope
  (README §5.5).
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
|10|BSDF/ABg measured-scatter import|**done** (2026-07-10), v1 BRDF-only|`raytracer/scatter.py` + `opticalproperties/scatter/bsdf.miebsdf`; BTDF (transmitted-side) not yet built at the time — LANDED SINCE (P2/P2.5, `engine3 overhaul` round, `btdf`/`btdf_A`/`btdf_B`/`btdf_g`/`btdf_tis_cap` columns)|
|11|Fold operator + relative optical-train chaining|**done** (`object-placer` round)|`mieworkbench/core/project.py`/`train.py`, `scripts/train_solver.py`|
|12|Multi-process ray sharding (`--workers`)|**done** (2026-07-10)|`run_trace._run_sharded`, `SeedSequence.spawn`; `--workers 1` bit-identical, `N>1` statistically equivalent|
|13|Zernike wavefront + Strehl|**done, v1 pupil model** (2026-07-10)|`raytracer/analysis.py` + `post_process.render_wavefront`; pupil is **source-referenced** (exact for collimated benches), not a true exit pupil — see backlog|
|14|Biaxial-crystal birefringence|**done** (2026-07-10) — KTP/KTA/LBO/BiBO|`raytracer/birefringence.py` biaxial extension + `birefringence/biaxial.mibiax`; internal conical refraction now modeled behind `--conical` (`samples-instruments` round, below), off by default (docs/RAYTRACER.md §5.6b)|
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

### Design-apparatus round (landed 2026-07-13) — the v1 design tools

The round that closed the former optimization/tolerancing ❌-sweep in
`features.md` §4. STATUS facts only:

| Item | STATUS | Code seam |
|--|--|--|
|Weighted-merit optimizer (spot-RMS/EE/MTF50/power)|**done, v1** — scipy Nelder-Mead **local** (demo-backed end-to-end) + nevergrad CMA-ES **global** (unit-tested on analytic bowls only — the `auto_designed_lens` demo exercises the local path); NO named-operand library, NO glass substitution, NO multi-config|`scripts/optimize.py`; Optimize GUI dock; `auto_designed_lens` demo|
|Persistent-worker fast evaluator|**done** — ~10× geometry-stage speedup, fingerprint-cache, crash-recovery, parity-oracle-verified (the shared optimizer/tolerancer inner loop)|`scripts/fast_eval.py`|
|Sensitivity + Monte-Carlo yield tolerancing|**done, v1** — sensitivity ranking + MC yield histogram + a **focus compensator** (single, nested optimize call); NO distribution library, NO compensator chains, NO fast-differential|`scripts/tolerance.py`; Tolerance GUI dock; `tolerance_yield` demo|
|dn/dT thermo-optic index + AGF catalog import|**done** — Schott TIE-19 `n(λ,T)` + `--temperature`; **168→847 materials** (Schott/Ohara AGF importer)|`materials.py` TIE-19 term; `scripts/tools/import_agf.py`; `opticalproperties/materials.miemat`|
|True exit-pupil / chief-ray + PSF-peak Strehl|**done** — exit-pupil reference sphere; PSF-peak-ratio Strehl (was Maréchal-only)|`scripts/raytracer/analysis_imaging.py`, `--wavefront-pupil exit_pupil`|
|Partial-coherence + image simulation|**done, v1** — coherent/incoherent/partial image-sim, **single space-invariant PSF convolution** (NOT field-varying, NOT a VCZ projector)|`analysis_field.image_*`|
|In-app Python console bound to `Project`|**done** — real interactive scripting (not a 4-mode SDK)|`mieworkbench/panes/py_console.py`|

**What this does NOT include** (the v1-maturation gaps, now the top of the
priority list below): named-operand library + true DLS local (I1/I2 depth),
glass substitution (I4), multi-config optimization + a config editor
(I5/M1–M3), compensator chains (J3), fast-differential tolerancing (J5),
directed global synthesis (I6), annular/multi-basis Zernike (B6 depth),
field-varying image-sim + VCZ partial coherence (F10/B8 depth).

### `samples-instruments` round (landed 2026-07-23) — scattering samples, S(q), T-matrix spheroids, image sources, DLS

STATUS facts only (full narrative in `CHANGELOG.md`):

| Item | STATUS | Code seam |
|--|--|--|
|Biaxial internal conical refraction|**done** — perturbed two-sheet fan, off by default (`--conical`/`--conical-fan`/`--conical-delta`); closes Backlog (b)'s "conical refraction" gap|`raytracer/birefringence.py` (`biaxial_optic_axes`/`cone_half_angle`/`conical_fan`); docs/RAYTRACER.md §5.6b|
|S(q) structure factors + explicit lattice realizations|**done** — exact Percus-Yevick + exact Baxter sticky-sphere + Teixeira fractal + powder paracrystal + tabulated; wired into the continuum ensemble AND real fcc/bcc/sc lattice site placement|`raytracer/structure.py`; `particles.py` (`ParticleCloud(sq=)`, `ExplicitRealization`)|
|Body-bound sample media (`sample` property)|**done** — exact-containment medium-stack binding, host-material override, phi XOR tau; `MEDIUM_STACK_DEPTH` 4→8|`raytracer/particles.py` (`BodyParticleMedium`); `sample/samples.miesamp` (7 rows)|
|Aspherical-particle T-matrix spheroids|**done** — closes Backlog (c)'s open T-matrix item|`raytracer/tmatrix.py` (pytmatrix, optics-env-only soft dep); `shape`/`aspect_ratio` sample columns|
|Extended image-emitting source (`image` property)|**done, v1** — Lambertian/cone emission, alias-method density sampling; C-ported|`raytracer/sources.py`; `image/images.mieimg`; `cengine/src/trace.c` `sample_image_pos_dir`|
|Blackbody + line emission kinds|**done** — closes the design-usability round's stated open item ("line-spectrum and blackbody/lamp source kinds... needs engine support")|`raytracer/optprops.py` (`EMISSION_KINDS`), `sources.wavelength_strata`'s `lines` regime|
|Diode-array instrument class + absorbance + ring profile|**done, v1**|`post_process.render_diode_array`/`render_absorbance`/`render_ring_profile`; `instrument/instruments.mieinst` `diode_array` class|
|Traced dynamic light scattering (DLS)|**done, v1** — dilute well-separated explicit clouds only (shared-RNG dense-cloud desync, see Backlog below)|`scripts/run_dls.py` + `scripts/dls_correlate.py`|
|Co-located transparent detectors|**done** — closes the a2 backlog item below|`extract_geometry.py` (`validation.detector_overlap`)|
|`--particles` cloud as a chain-referenceable anchor|**done** — closes the a2 backlog item below|`sample_region` primitive (air anchor cube) + its `port_frames` entry|
|Anchored-placement expressions|**done** — closes the a2 backlog item below|`miewb_expr_pos_x/_y/_z`/`miewb_expr_rot_rx/_ry/_rz`; `train_solver.place_anchored`|
|10 new primitives (catalog 70→80)|**done**|`primitivelib.py` — cuvette/vial/vat/sample_region/lamp/image-source builders|
|Two Python-engine bug fixes (Fresnel weakly-absorbing-incident branch; trace hop cap)|**done**|`raytracer/fresnel.py` `cos_theta_t`; `raytracer/tracer.py` `Tracer.run`|

## Priority ranking — impact × leverage (2026-07-13, post design-apparatus round)

The former "Must-Have" list (optimizer, tolerancer, dn/dT, exit-pupil,
console) is **DELIVERED** (above). This is the rebuilt single ordered
priority list, **interleaving v1-maturation of the new design apparatus with
the still-open categorical gaps**, ranked by **(impact on closing a
competitor-won `features.md` line) × (leverage over existing code)**. Effort ·
impact per the legend. Each item's narrative + code seam lives in its own
Backlog/§ below (or `features.md` §7).

1. **Glass substitution (I4)** — **[M · High]**. Discrete catalog search
   wrapping the shipped `optimize.py` over the 847-row material registry;
   all of Zemax/CODE V/QUADOA win it and it is central to real lens design.
   Highest leverage-to-impact: the optimizer and the catalog already exist.
2. **More + named merit operands (I1 depth)** — **[S-M · High]**. Extend the
   weighted merit in `optimize.py` with a named-operand set (EFL, RMS
   wavefront, MTF@freq, edge/boundary constraints). Pure extension of shipped
   code; moves I1 🟡→toward ✅.
3. **Multi-config optimization + config-table editor (I5 · M1–M3)** —
   **[M · High]**. Wrap the `--var` sweep + Variables dock as a named-config
   table the optimizer iterates; all four suites win M and I5. Reuses the
   sweep machinery + fast evaluator.
4. **CAD (STEP/IGES) import as traceable elements (D9/D10 · P2)** —
   **[M / L(analytic) · High]**. FreeCAD already imports STEP/IGES; expose
   via the fc_server worker, fall back to the shipped incoherent mesh-BVH.
   Removes a hard ❌ and unlocks optomechanical/stray-light scenes.
5. **Compensator chains (J3 full)** — **[M · Med-High]**. Generalize the
   shipped single focus compensator to N chained compensators (nested
   optimize over the compensator set per MC draw); moves J3 🟡→✅.
6. **Annular / multi-basis Zernike + exit-pupil polish (B6 depth)** —
   **[M · Med]**. Add annular + Standard/Fringe bases on top of the shipped
   exit-pupil stage; moves B6 🟡→✅. High leverage (extends analysis_imaging).
7. **Ray-aiming to a stop + measured source files (E6/E7)** — **[M · Med]**.
   `sources.py` already samples faces; iterate emission to hit a named
   aperture (E6), add an IES/TM-25/rayfile importer (E7). The last source
   gaps vs Zemax/OSLO/QUADOA.
8. **Analytic Q-type (Forbes) / XY-Zernike freeform with coherent phase
   (D3/D4)** — **[M-L · Med]**. Extend the `surfaces.py` asphere machinery
   (Newton-intersect + `<1µm` verify) to Qbfs/Qcon + freeform sag; matters
   for freeform/AR-VR.
9. **Fast-differential wavefront tolerancing (J5)** — **[L · Med]**. Finite-
   difference perturbation of the exit-pupil Zernike vector (the exit-pupil
   stage it needs is now shipped), cheap enough for in-loop desensitization;
   CODE V's unique bar.
10. **Directed global synthesis / multi-start (I6)** — **[XL · Med]**. Stretch
    on `optimize.py`: surface many distinct minima per run (CODE V Global-
    Synthesis-style). Lower leverage (research-grade), CODE V-unique.
11. **Fuller stray-light report (H6)** — **[M · Med]**. BTDF itself
    LANDED since this list was written (P2/P2.5, `engine3 overhaul`
    round, beside the shipped BRDF ABg sampler); what remains is a
    Path-Analysis-style report on the shipped ghost ranking —
    MieWorkbench already wins the scatter *physics*.
12. **Field-varying image-sim + VCZ partial coherence (F10/B8 depth)** —
    **[L · Med]**. Upgrade the shipped space-invariant image-sim to
    field-varying, add a Van Cittert–Zernike projector; rides on the exit-pupil
    field pipeline. Depth polish on a shipped v1.
13. **GRIN media (D8)** — **[XL · High]**. High impact (every design suite
    except 3DOptix has it) but **low leverage** — genuinely new curved-ray
    eikonal (Runge–Kutta) integration replacing the straight-segment loop in
    `tracer.py`. Impact-worthy but expensive, hence mid-list by impact×leverage.
14. **Nestable assemblies + cross-platform packaging (O7 · N7/R3)** —
    **[M / L · Med]**. First-class assembly object over `miewb_group`;
    Windows/Mac build (blocker is the FreeCAD/optics-env/ParaView stack, not
    the GUI).
15. **Gridded POP / beamlet propagator (B3/B10/B11)** — **[L-XL · Low]**.
    Propagate a gridded field surface-to-surface on the existing gather kernel;
    the coherent gather already covers most cases, so lowest priority.

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
- ~~**BTDF (transmitted-side) measured scatter.**~~ LANDED (P2/P2.5,
  `engine3 overhaul` round): the `scatter` property (§5.4.2) now supports
  an optional transmitted-side lobe via the `btdf`/`btdf_A`/`btdf_B`/
  `btdf_g`/`btdf_tis_cap` columns on a `scatter/bsdf.miebsdf` row (each
  defaulting to its reflected-side counterpart), split from the specular
  transmitted remainder the same way the BRDF splits the reflected one;
  ported to the C engine with importance-sampling support
  (`raytracer/cengine.py`, `test_scatter_importance_btdf.py`). This row
  was stale — this note is the only remaining stale copy as of the
  `samples-instruments` docs pass.
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
  ~~Line-spectrum and blackbody/lamp source kinds are still open~~ LANDED
  (`samples-instruments` round): `EMISSION_KINDS` now includes
  `blackbody` (Planck synthesis at load) and `lines` (Hamilton-
  apportioned per-line strata); `tungsten_halogen`/`d2_lamp`/
  `hg_calibration` are the new lamp primitives.

### (a2) Placement/authoring affordances (design-usability round findings)

From the design-usability-round shakedown (now consolidated into
`demos/UXNOTES.md`) — the pain points too large for that round's fix
loop (each names its seam):

- ~~**Expressions/variables for ANCHORED placements.**~~ LANDED
  (`samples-instruments`): anchored elements carry `miewb_expr_pos_x/_y/_z`
  and `miewb_expr_rot_rx/_ry/_rz` expression props (group `MieTrain` on the
  primary body); `train_solver.place_anchored` bakes them from `miewb_vars`
  inside the SHARED `solve_chain` (GUI ripple + headless `train_fcstd`
  bake, pinned by `test_train_parity.py` across a θ sweep). Anchored-only
  (a chained element with one is refused); a spatial drag clears them to a
  literal pose (chain-drag convention). GUI: transform_panel "Pose
  expressions (variables)" per-field editor with the `expr (= value)`
  affordance. The one-shot polar place-about-point (`place_about_point`)
  remains for computing a literal pose without persisting expressions.
- ~~**A `--particles` cloud is not a chain-referenceable body**~~ LANDED
  (`samples-instruments`): the `sample_region` primitive (a bare
  `material=air` anchor cube) carries a `port_frames` pass-through entry,
  so a body-bound sample cloud (§5.13's `sample` property, bound to any
  host body's interior, not just a CLI `--particles` box) can now anchor a
  chained detector "N mm at θ° from the cloud center" like any other
  element.
- **Field-angle source fan helper.** N collimated sources at a common
  pivot overlap as solids; they must be spread on an arc by hand
  (y = L·tanθ). A wizard ("fan of field angles: N, ±θ, pivot") placing
  them chained/anchored would remove the trig (`element_wizard.py`).
- ~~**Co-located transparent detectors** overlap-fail extraction~~ LANDED
  (`samples-instruments`, T16a): detector-detector solid overlap now
  classifies into the informational `validation.detector_overlap` list
  instead of the fatal `overlapping_solids` — "measure the same plane two
  ways" is a physically well-defined stack of transparent screens
  (`extract_geometry.py`).
- ~~**Coherent-gather ray-budget preflight.**~~ This row was STALE — the
  preflight check already existed (`design-usability` round,
  `mieworkbench/core/validation.py`'s `check_gather_preflight`, a coarse
  aperture-area-ratio estimate against the M_eff gate at check time, run
  BEFORE a trace). No open item remains here.
- ~~**`--particles` target-optical-depth knob.**~~ This row was STALE —
  `common.parse_particles_spec` already accepts `tau=<F>` (mutually
  exclusive with `phi`) and resolves it to an equivalent `phi` via
  `ParticleCloud` (§9 of docs/RAYTRACER.md). The `sample` registry
  (§5.13) also exposes `tau` directly on every sample row. No open item
  remains here.
- **"Span N Airy zeros" detector-sizing intent** and other
  diffraction-scale insert-values for the right-click menu
  (`core/opticalvalues.py` — needs aperture+distance context).

### (b) Higher-fidelity physics (still open)

- **Exact uniaxial Fresnel at a birefringent interface — LANDED (P6,
  Python engine).** The interface amplitudes now solve the exact uniaxial
  boundary-value problem (Lekner 1991, a 4×4 tangential-E/H match per ray,
  `bir.uniaxial_interface_in`/`_out`): full o/e transmission split plus the
  reflected s/p Jones *including* the `r_sp`/`r_ps` cross terms, Poynting-
  flux normalized (unitary to 1e-10 for calcite/quartz). The legacy
  effective-index approximation stays behind `--biref-approx`. Two seams
  remain open: (i) the **C engine** still carries the effective-index form
  (`birefk.h`), so exact-uniaxial scenes Python-route via the `biref_exact`
  feature token — a shared CPU/CUDA port of the complex 4×4 solve +
  Poynting normalization is the follow-up; (ii) **exit internal reflection**
  is still mode-preserving in geometry (its total power is exact via
  `R = 1 − T`, but o↔e mode conversion on internal reflection and its
  reflected-mode split are not propagated). The full-anisotropy Berreman
  4×4 (biaxial/absorbing/gyrotropic) remains the final option (§7.4-2).
- **Optical activity / chiral media (e.g. quartz's rotary power along its
  own optic axis).** **PARTIALLY LANDED**: near-axis scene-level rotation
  is wired (`tracer._apply_optical_activity`: a gyrotropic uniaxial body
  on its isotropic n_o path rotates the Jones vector by ρ·ds; validated by
  `quartz_rotator`, sin²(ρd) to 0.0% error) and the full-anisotropy
  Berreman 4×4 module (`berreman.py`) independently reproduces
  ρ=21.77°/mm (ORACLE 3). Still open: off-axis/elliptical-eigenmode
  gyration, C-engine port (`gyration` is a Python-only routing token), and
  scene-level coupling of Berreman's full tensor beyond the near-axis
  uniaxial case.
- ~~**Biaxial crystals — conical refraction only.**~~ Biaxial birefringence
  itself LANDED 2026-07-10 (`raytracer/birefringence.py`
  `refract_in_biaxial()`/`biaxial_modes_for_k()`, quartic normal-surface
  root solve via companion-matrix eigenvalues; `birefringence/
  biaxial.mibiax` registry; KTP/KTA/LBO/BiBO ship, 15 tests in
  `test_biaxial.py`). **Internal conical refraction near an optic axis
  LANDED (`samples-instruments` round, 2026-07-23)** behind `--conical`
  (off by default, still an arbitrary transverse basis at the degeneracy
  when off): a perturbed two-sheet fan reproduces Hamilton's internal cone
  (`birefringence.py` `biaxial_optic_axes`/`cone_half_angle`/
  `conical_fan`; docs/RAYTRACER.md §5.6b). Still open: **external**
  conical refraction (the emergent double-ring from a point source outside
  the crystal) is not modeled — only the internal cone the fan reproduces
  — and the fan is Python-only (biaxial scenes already Python-route
  regardless). **A C-engine port of `--conical` is NOT currently
  planned**: biaxial birefringence itself (`biaxial`/`berreman` tokens)
  is already Python-only, so the fan rides on a physics path that has no
  C-side counterpart to port INTO yet — porting `--conical` alone, ahead
  of a biaxial C port, would be a seam with nothing on the other side.
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
- ~~**Measured BSDF scatter — BTDF half.**~~ LANDED (P2/P2.5,
  `engine3 overhaul` round) — see the note at Backlog (a)'s item #10
  above; the BRDF half landed 2026-07-10.
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
  Python-route regardless of engine. **`sample_explicit` C port
  (samples-instruments round, new)**: the body-bound `sample` registry's
  EXPLICIT/lattice mode (§5.13) hits the exact same seam — it emits the
  `sample_explicit` Python-only routing token for the identical reason
  (brute-force collision, no DDA/grid traversal on the C side yet). A
  future numba/C DDA implementation would close both gaps together, not
  as two separate ports.
- **Mesh-type source/detector faces** still hard-error unconditionally
  (README §5.8/§5.11) — both need a UV parameterization the incoherent/
  coherent paths don't have yet; the ordinary-optic mesh path (BVH tracing)
  is otherwise fully shipped.
- ~~**Aspherical particles** (user goal): T-matrix drop-in~~ LANDED
  (`samples-instruments` round, 2026-07-23): `raytracer/tmatrix.py`'s
  `TMatrixEvaluator` (pytmatrix, optics-env-only soft dependency) drops in
  behind the exact `MieEvaluator.efficiencies()`/`amplitudes()` interface
  this item named, orientation-averaged at the volume-equivalent-sphere
  radius, wired into both continuum and explicit sample media via the
  `sample` registry's `shape`/`aspect_ratio` columns (docs/RAYTRACER.md
  §5.13). Physics caveat documented at the code site: pytmatrix's own
  orientation averaging is exact for Qext but ~15% low for Qsca vs
  independent random orientations — worked around by deriving Qsca/g/S1/S2
  from the phase matrix Z instead.

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

- **auto_designed_lens** (optimizer) — **DELIVERED (design-apparatus round,
  2026-07-13).** `scripts/optimize.py` (weighted spot-RMS/EE/MTF50/power
  merit, scipy Nelder-Mead local + nevergrad CMA-ES global, on the
  `fast_eval.py` persistent worker) + an Optimize GUI dock + the demo all
  shipped. The v1 excludes glass substitution / multi-config / a named
  operand library — now the top of the priority list.
- **tolerance_yield** (tolerancing) — **DELIVERED (design-apparatus round,
  2026-07-13).** `scripts/tolerance.py` (sensitivity ranking + Monte-Carlo
  yield histogram + a focus compensator via a nested optimize call) + a
  Tolerance GUI dock + the demo all shipped. The v1 excludes a distribution
  library, compensator chains, and CODE V-style fast-differential
  tolerancing — priority-list follow-ons.
- **cad_import_scene** (CAD import, `features.md` §7.4). A STEP-imported
  lens barrel + baffles traced as optomechanics (stray light) around an
  existing optical train. FreeCAD already imports STEP/IGES; the missing
  piece is the GUI exposing "Import STEP as element" through the
  fc_server worker (`import_bodies`/`import_primitive` ops) and the
  extractor canonicalizing imported faces (falling back to the existing
  mesh-BVH path for non-canonical ones — already shipped, incoherent-
  only). Effort: M for mesh-import; L for analytic-face recovery.
- **freeform_illuminator** (illumination design, `features.md` §7.9) — now
  **UNBLOCKED** (the optimizer it rides on landed 2026-07-13). An LED +
  freeform reflector/TIR lens optimized to a prescribed irradiance target
  with photometric units. Non-imaging freeform tailoring wraps the shipped
  `optimize.py`; photometric units (lux/lumen/candela via CIE V(λ)) already
  landed. Effort: L.

### (e) `samples-instruments` round follow-ons (new)

- **DLS dense-cloud RNG desync.** `run_dls.py`/`dls_correlate.py`
  (docs/RAYTRACER.md §8.7) are validated for a DILUTE, well-separated
  explicit sample cloud only: the shared Monte-Carlo RNG stream's draw
  order depends on the exact ray-particle collision set, so a dense
  cloud's collisions change frame-to-frame and the field goes
  delta-correlated instead of decaying at the physical `D·q^2` rate — not
  a bug in the correlator (independently validated against synthetic
  fields), but a real limitation of sharing one RNG stream across a
  changing collision set. A collision-order-independent (or per-particle-
  keyed) RNG scheme would remove the dilute-cloud requirement. Related
  frame-sequence extensions still open: multi-angle detectors already
  work (`gamma_vs_q2.png` is multi-angle by construction), but there is no
  polydisperse-population decomposition (a mixed-size sample's g1 is a
  single effective cumulant fit, not a resolved distribution) and no
  concentration/viscosity sweep automation (each is a separate `run_dls.py`
  invocation today).
- **Field-varying PSF image simulation.** Explicitly EXCLUDED from the
  `image`-source work this round (§5.14) — `render_image_traced`/
  `--image-sim` both still use a single space-invariant PSF convolution
  (`analysis_field.image_*`); an extended `image` source makes this gap
  more visible (a resolution-target image traced end-to-end has real
  field-dependent aberrations the space-invariant sim cannot reproduce),
  but closing it is the same field-varying-image-sim item already tracked
  under Backlog (b)/priority list #12 (F10/B8 depth), not a new item —
  cross-referenced here because the `image` source is the natural
  acceptance-test scene for it.
- **C-engine trace iteration valve is still a heuristic, not the exact
  per-lineage cap.** The Python engine's hop-cap fix this round (this
  file's `samples-instruments` round-summary table, `raytracer/tracer.py`
  `Tracer.run`) replaced a shared/global pop budget with an exact
  per-lineage step count. `cengine/src/trace.c`'s `max_iter` valve
  (`trace.c:2121-2123`) instead scales the same style of global budget by
  a `split_factor` — a "worst-case children-per-primary" headroom
  multiplier chosen empirically (documented at the code site, from an
  earlier `microscope_objective`-benchmark fix) — which is HEADROOM, not
  an exact bound: a sufficiently pathological many-interface C-engine
  scene could in principle still exhaust it before a lineage is actually
  done, the same failure mode the Python-side exact cap now categorically
  rules out. Porting the exact per-lineage accounting into `trace.c`
  (replacing the heuristic `split_factor` scale-up with a per-batch
  ancestry counter, mirroring the Python fix) would close this asymmetry;
  not yet started.

### (f) samples-instruments demo-shakedown findings (2026-07-23)

Two engine seams surfaced by the new demos' shakedown, LOUD failures
(the closure gate aborts the run), documented in demos/UXNOTES.md:

- **C-engine continuum-scatter divergence in TIR-trapping shells**
  **[M · Med]**. A continuum sample INSIDE a closed glass cell (cuvette/
  vial/vat walls) diverges the C engine's closure (1e48+) — scattered
  children trapped by TIR at the glass shell cascade through the
  C iteration valve (heuristic split_factor budget; the Python engine's
  new exact per-lineage hop cap truncates the same trap honestly with a
  warning). The demos work around it with bare `sample_region` volumes.
  Seams: cengine/src/trace.c iteration valve (align with the Python
  per-lineage cap — carry a per-ray hop counter, e.g. extend event_ctr
  gating) + investigate whether the C fresnel branch amplifies at
  grazing incidence from weakly absorbing media (the Python twin of
  that bug was fixed this round in fresnel.cos_theta_t; check
  kernels/fresnel.h's branch rule against the same water->lossless
  case).
- **`ambient_material` is hardcoded to air** **[S · Med]**. extract's
  model.json always writes ambient_material=air; a DLS bench wants a
  water ambient (index-matching the sample region) and today nothing —
  no body property, no simparam — can set it. Expose it (body-less
  model-level property or CLI override) and the dls_goniometer
  demonstrator can graduate to a Gamma = D q^2 gated oracle.

## Roadmap rating index (every open item, 2-axis)

One row per open item, grouped by the section it lives in. Effort/impact per the
legend at the top. This is the uniform rating the design comparison asked for; the
narrative + exact code seam for each stays in its own section above (or in
`features.md` §7 for the design-apparatus items). Landed items are omitted (see
"Delivered").

### Backlog (a) — near-term
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Stress/spatially-varying birefringence|L-XL|Med|first cut (constant axis, position-dependent Δn) closer to L; full stress-optic + FEA XL; niche photoelastic|
|Exit-pupil / chief-ray search stage|—|—|**DONE (design-apparatus round, 2026-07-13)** — `analysis_imaging.py`, `--wavefront-pupil exit_pupil`; PSF-peak Strehl + partial image-sim shipped. Depth follow-on (annular Zernike, field-varying image-sim) in the priority list #6/#12|
|BTDF (transmitted-side) measured scatter|✅|—|**LANDED (P2/P2.5, `engine3 overhaul` round)** — this row was stale; `btdf`/`btdf_A`/`btdf_B`/`btdf_g`/`btdf_tis_cap` columns beside the shipped BRDF ABg sampler. Fuller stray-light report (H6) remains — priority list #11|
|Coherent gather on curved detectors|L|Med|curved-aperture obliquity terms; `curved_focal_surface` demo|
|Materials dn/dT (thermo-optic) hook|—|—|**DONE (design-apparatus round, 2026-07-13)** — Schott TIE-19 `n(λ,T)` + `--temperature` + 847-glass AGF import (`materials.py`, `import_agf.py`)|
|Line-spectrum + blackbody/lamp sources|M|Med|continuous-tabulated + white LED landed; discrete-line & Planck kinds remain|

### Backlog (a2) — placement/authoring affordances
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Expressions/variables for anchored placements|M|Med|**DONE (samples-instruments)** — `miewb_expr_pos_*/rot_*` baked by `train_solver.place_anchored` in the shared solve; GUI + headless, parity-pinned|
|`--particles` cloud as chain-referenceable anchor|—|—|**DONE (samples-instruments)** — `sample_region` primitive + its `port_frames` pass-through entry; a body-bound `sample` cloud (§5.13) is chain-referenceable like any element|
|Field-angle source-fan wizard|S|Med|removes hand-computed `y=L·tanθ` placement|
|Co-located transparent detectors|—|—|**DONE (samples-instruments, T16a)** — `validation.detector_overlap` informational classification replaces the fatal `overlapping_solids`|
|Coherent-gather ray-budget preflight|—|—|**ALREADY DONE (design-usability round)** — this row was stale; `core/validation.py`'s `check_gather_preflight` already estimates the M_eff gate at check time|
|`--particles` target-optical-depth knob|—|—|**ALREADY DONE** — this row was stale; `common.parse_particles_spec` already accepts `tau=<F>` and resolves it to `phi`; the `sample` registry (§5.13) also exposes `tau` directly|
|"Span N Airy zeros" detector-sizing intent|S|Low|diffraction-scale insert-values in the right-click menu|

### Backlog (b) — higher-fidelity physics
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Exact uniaxial Fresnel at a birefringent interface|✅|—|**LANDED (P6, Python):** Lekner-1991 4×4 boundary-match; C-engine port + exit mode-conversion remain|
|Optical activity / chiral media (off-axis/elliptical + full-tensor scene coupling)|M|Low|**partially landed** — near-axis scene-level rotation + Berreman module both ship; remaining scope is off-axis gyration and a C-engine port|
|Biaxial conical refraction (internal)|✅|—|**LANDED (samples-instruments)**: perturbed two-sheet fan behind `--conical`, off by default. **External** conical refraction (point-source double-ring) and a C-engine port remain open, corner-cases of a MieWorkbench-unique win (C5)|
|Absorbing (dichroic) uniaxial crystals|M|Low|`Im(n_o)/Im(n_e)` currently ignored|
|Reflection-geometry Kogelnik gratings|M|Low|tanh/sinh reflection VBG solution|
|RCWA|XL|Low|Zemax-only among the six; closed-form models suffice (`features.md` §7.15)|
|Ray-differential transport through gratings/scatter/birefringence|L|Low|three transport paths currently NaN the differential|
|Translucent (non-opaque) gather occlusion|M|Low|per-pixel mode already opt-in; partial occluders need transmission accounting|
|GRIN (gradient-index) media|XL|**High**|every design suite except 3DOptix has it (`features.md` D8/§7.4); curved-ray eikonal integration|
|Fluorescence / phosphors|L|Low|new wavelength-shifting emission event|
|Thermal lensing|L|Low|coupled absorbed-power→ΔT→dn/dT; shares the dn/dT hook|

### Backlog (c) — capability gaps (hard errors today)
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Explicit particle clouds > 200k spheres|L|Low|numba DDA/grid traversal removes the cap; `sample_explicit` (samples-instruments) hits the identical seam|
|Mesh-type source/detector faces|M|Low|needs a UV parameterization the paths lack|
|Aspherical particles (T-matrix)|✅|—|**DONE (samples-instruments)** — `tmatrix.py` (pytmatrix) drops in behind `MieEvaluator`|

### Design-apparatus v1-maturation follow-ons (NEW — the top of the priority list)
| Item | Closes | Effort | Impact | Note |
|--|--|:--:|:--:|--|
|Glass substitution|I4|M|**High**|discrete catalog search wrapping `optimize.py` over the 847-row registry; all of Zemax/CODE V/QUADOA win it (priority #1)|
|Named-operand merit library + true DLS local|I1/I2 depth|S-M|**High**|extend the weighted merit in `optimize.py`; EFL/RMS-wavefront/MTF@freq/edge operands (priority #2)|
|Multi-config optimization + config-table editor|I5 · M1–M3|M|**High**|wrap `--var` sweep + Variables dock as a named-config table the optimizer iterates (priority #3)|
|Compensator chains|J3 full|M|Med-High|generalize the shipped single focus compensator to N chained compensators (priority #5)|
|Annular / multi-basis Zernike|B6 depth|M|Med|add annular + Standard/Fringe bases on the shipped exit-pupil stage; B6 🟡→✅ (priority #6)|
|Fast-differential wavefront tolerancing|J5|L|Med|finite-difference perturbation of the (now-shipped) exit-pupil Zernike vector; CODE V's unique bar (priority #9)|
|Directed global synthesis / multi-start|I6|XL|Med|stretch on `optimize.py`; surface many distinct minima per run (priority #10)|
|Field-varying image-sim + VCZ partial coherence|F10/B8 depth|L|Med|upgrade the shipped space-invariant image-sim; add a VCZ projector (priority #12)|

### Backlog (d) — big-roadmap acceptance-target demos
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|`auto_designed_lens` (optimizer)|—|—|**DONE (design-apparatus round, 2026-07-13)** — `optimize.py` (Nelder-Mead + CMA-ES) + demo; v1-maturation (glass-sub/multi-config/operands) in the follow-ons table above|
|`tolerance_yield` (tolerancing)|—|—|**DONE (design-apparatus round, 2026-07-13)** — `tolerance.py` (sensitivity + MC yield + focus compensator) + demo; compensator-chains/J5 in the follow-ons table above|
|`cad_import_scene` (CAD import)|M/L|Med|FreeCAD already imports STEP; expose as element (`features.md` §7.4; priority list #4)|
|`freeform_illuminator` (illumination design)|L|Med|rides on the optimizer (now shipped); photometric units already landed|

### Roadmap (e) — `samples-instruments` round follow-ons
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Conical refraction C port|—|—|**NOT PLANNED** — biaxial birefringence itself is Python-only; nothing to port `--conical` into yet|
|`sample_explicit` C port (DDA/grid traversal)|L|Low|same seam as the `>200k`-sphere explicit-cloud cap above; a shared DDA/grid implementation would close both|
|DLS dense-cloud RNG desync|L|Med|shared-RNG collision-set desync limits DLS to dilute clouds; a collision-order-independent RNG scheme would remove it|
|Field-varying PSF image simulation|L|Med|cross-references the existing F10/B8-depth item (priority list #12); the new `image` source is its natural acceptance scene|
|C-engine exact per-lineage hop cap|M|Low|`trace.c`'s `split_factor` headroom valve vs the Python engine's now-exact per-lineage cap; a pathological C-engine scene could still exhaust the heuristic|

### Pulsed-optics round follow-ups (moat-widening on the S axis MieWorkbench uniquely owns)
| Item | Effort | Impact | Note |
|--|:--:|:--:|--|
|Split-step NLSE propagation|L-XL|Med|real intra-train SPM+GVD/soliton dynamics|
|Mid-train SPM (body `spm` property)|M|Low|needs stratum re-quantization at the element|
|Depleted-pump coupled-wave SHG|M|Med|tanh² solution → strong conversion + back-conversion|
|Harmonic walk-off / exact uniaxial SHG|L|Med|type-I/II e/o geometry; needs exact-uniaxial Fresnel (b)|
|Cascaded/coherent harmonics|L|Low|THG via cascade, phase-sensitive pump-harmonic interplay|
|Raman / fluorescence inelastic transfer|M|Low|Stokes-shift bulk event reusing the SHG plumbing|
|C-engine port of nonlinear (χ²) tokens|M|Med|`nonlinear` (SHG/Pockels) is the only pulsed/NLO token still Python-routed — `time_products`/`gdd_budget`/`ray_differentials`/`saturable`/`tpa`/`kerr` are C-ported (P7)|
|Fringe-resolved timing|L|Low|per-record complex amplitudes at ~100× record cost|
|Angular-dispersion group-index term|M|Low|e-ray group delay neglects dθ/dλ|
|Per-source time cubes|S|Low|cube currently bins all sources together|
|SuperK SPD tail vs material tables|S|Low|clip SPD where materials aren't tabulated|
|Deferred demos (prism_compressor, wideangle_retrofocus)|S-M|Low|physics validated; only gallery scenes missing|
|Transmission-grating truncated-order booking|S|Med|closure-gate correctness (reflective branch already books it)|
|Dominant-cluster auto time window|M|Med|cluster records so fs pulses resolve without a hand-pinned window|

## Additional supportable features (surfaced by the 6-package comparison)

> **✅ DESIGN-APPARATUS ROUND LANDED (2026-07-13).** The core "design apparatus" gap this
> section named is now closed on `master`: the **headless optimizer** (`scripts/optimize.py`,
> scipy + nevergrad CMA), **Monte-Carlo tolerancing + sensitivity + compensator**
> (`scripts/tolerance.py`), the **persistent-worker fast evaluator** (`scripts/fast_eval.py`,
> the shared inner loop with crash-recovery), **dn/dT + 847 Schott/Ohara glasses**
> (`scripts/tools/import_agf.py` + Schott TIE-19 in `materials.py`), the **exit-pupil / PSF-peak
> Strehl** verification + **partial-coherence & image-simulation** (`analysis_field.image_*`), and
> the **in-app Python console** (`panes/py_console.py`) all shipped with full-parity GUI docks,
> demos, and physics oracles. The remaining rows below (GRIN, Q-type/freeform, CAD import, POP/BSP,
> ray-aiming, measured source files, multi-config editor, cross-platform packaging) are the
> genuinely-still-open items.

The `features.md` refresh (now vs Zemax, CODE V, OSLO, QUADOA, 3DOptix) tells a two-part
story: MieWorkbench closed the *analysis-product* gap (PSF/MTF/EE/spot/fans/Zernike/Strehl,
photometry, ghost analysis) in earlier rounds, and the 2026-07-13 design-apparatus round
then landed a **v1 of the design apparatus itself** (optimizer, tolerancer, dn/dT, exit-pupil
Strehl, image-sim, console — see the banner above). What the table below now tracks is
therefore **(a) the DELIVERED design-apparatus rows** (marked DONE, kept for traceability)
and **(b) the still-open items** — both the *v1-maturation* follow-ons (glass substitution,
multi-config, compensator chains, fast-differential, directed synthesis, operand library —
ranked in the priority list up top) and the *categorical* gaps (GRIN, Q-type/freeform, CAD
import, POP/BSP, ray-aiming, measured source files, cross-platform packaging). Each carries a
2-axis rating, the exact code seam, and the `features.md` line it closes. (Items already in the
Backlog above are cross-referenced, not repeated.)

| Feature | Closes | Effort | Impact | Seam / path |
|--|--|:--:|:--:|--|
|**Headless optimization loop** (merit function + local/global)|I1🟡/I2🟡/I3🟡|**DONE v1**|—|**DELIVERED** `scripts/optimize.py` (Nelder-Mead local, demo-backed; CMA-ES global, bowl-tested only) on `fast_eval.py`; maturation (I1 operands / I4 glass-sub / I5 multi-config) = priority list #1–3|
|**Directed global synthesis** (many distinct minima, CODE V-style)|I6|XL|Med|OPEN — stretch goal on the shipped optimizer; surfaces multiple design forms per run (priority #10)|
|**Sensitivity + Monte-Carlo tolerancing**|J1✅/J2✅/J4✅|**DONE v1**|—|**DELIVERED** `scripts/tolerance.py`; sensitivity ranking + MC yield + focus compensator (J3🟡). Chains/J5 = priority #5/#9|
|**Fast differential wavefront tolerancing** (CODE V Wavefront-Differential)|J5|L|Med|OPEN — finite-difference perturbation of the (now-shipped) exit-pupil Zernike vector, for in-loop desensitization (priority #9)|
|**True exit-pupil / chief-ray search**|B6🟡/B7✅, B8🟡, F10🟡|**DONE v1**|—|**DELIVERED** `analysis_imaging.py` (`--wavefront-pupil exit_pupil`) + PSF-peak Strehl + partial image-sim. Annular-Zernike/field-varying depth = priority #6/#12|
|**dn/dT + expanded glass/dispersion catalogs**|G1🟡/G3✅, K1🟡|**DONE**|—|**DELIVERED** Schott TIE-19 `n(λ,T)` + `--temperature` + 847-glass AGF import (`materials.py`, `import_agf.py`). More catalog breadth = `features.md` §7.6|
|**GRIN media**|D8|XL|**High**|OPEN — Runge–Kutta eikonal curved-ray integration in `tracer.py`. Backlog (b) (priority #13)|
|**In-app Python console** bound to `Project`|P3🟡|**DONE**|—|**DELIVERED** `mieworkbench/panes/py_console.py` — real interactive scripting bound to the live Project (not a 4-mode SDK)|
|**CAD (STEP/IGES) import as traceable element**|D9/D10|M / L(analytic)|Med|FreeCAD imports natively; expose via fc_server `import_bodies`, fall back to mesh-BVH. Backlog (d)|
|**Analytic Q-type (Forbes) / XY-Zernike freeform** with coherent phase|D3/D4|M-L|Med|extend `surfaces.py` asphere machinery (Newton-intersect + `<1µm` verify) to Qbfs/Qcon + freeform sag|
|**Ray-aiming to a real pupil**|E6|M|Med|iterate emission direction to hit a named aperture body in `sources.py`|
|**Measured source-file import** (IES/TM-25/rayfile)|E7|M|Low|weighted-ray-set importer; 3DOptix/Zemax win this|
|**Config-table multi-configuration editor**|M1–M3|M|Med|named-config table wrapping the `--var` sweep, overlay via `compare_runs.py`|
|**Gridded POP / beamlet propagator** (Zemax POP / CODE V BSP class)|B3/B10|L-XL|Low|propagate a gridded field surface-to-surface on top of the existing gather kernel; the coherent gather already covers most cases|
|**Partial-coherence imaging + image simulation**|B8🟡/F10🟡|**DONE v1**|—|**DELIVERED** `analysis_field.image_*` — coherent/incoherent/partial image-sim (space-invariant PSF convolution). Field-varying + VCZ projector depth = priority #12|
|**Multi-GPU gather**|Q3|L|Med|merge detector cubes/ledgers (linear accumulators); after `--workers`|
|**Nestable assemblies / grouping**|O7|M|Low|first-class assembly object over `miewb_group`; QUADOA-style|
|**Cross-platform (Windows/Mac) packaging**|N7/R3|L|Med|PySide6+VTK are portable; blocker is the FreeCAD/optics-env/ParaView stack — bundle as installer/container|

**Deliberate non-goals** (documented, not chased — `features.md` §7.15): full per-ray RCWA
solve (Zemax-only here), Mueller-matrix formalism (only QUADOA claims it), off-axis/elliptical
gyration + C-engine gyration port, coating needle-synthesis (nobody here has it), native cloud
compute (conflicts with the data-locality/ITAR value proposition), a macro *language*
(redundant given a Python console).

## Partial features — behavioral differences vs commercial tools

MieWorkbench ships several features at 🟡 (`features.md` §4). This section states, for each,
**what MieWorkbench actually does**, **how the commercial tools behave differently** (the
ones that rate it higher), and **what full parity needs**. This is the "if partially
implemented, explain the behavioral difference" contract — framed against the tools a user
would compare to.

| Partial feature | What MieWorkbench does | How the commercial tools differ | Path to parity |
|--|--|--|--|
|**Zernike / Strehl** (B6/B7 🟡)|Fits Noll+Fringe Zernike (jmax=15) on a **source-referenced** pupil (each ray's normalized birth position); Strehl via the **Maréchal** approximation from residual RMS. Exact for collimated/laser benches.|Zemax/CODE V/OSLO/QUADOA reference a **true exit pupil** at a field point's image and report a **PSF-peak-ratio** Strehl — correct for finite-conjugate, off-axis field imaging.|Exit-pupil/chief-ray search stage (Backlog a / `features.md` §7.3) — **[L·High]**|
|**Curved detectors** (F3 🟡)|Sphere/cylinder detector grids with a per-pixel metric-area map, **incoherent path only** — a coherent Huygens gather on a curved screen raises `NotImplementedError`.|Zemax curved/annular detector objects accept the full (coherent) field.|Per-pixel normals/obliquity through the gather kernel (Backlog a) — **[L·Med]**|
|**Measured (tabulated) coatings** (C7)|P2: table coatings now accept OPTIONAL Zemax-TABLE-style `ars_deg/arp_deg/ats_deg/atp_deg` phase columns (`materials.py` `phase_valid`; branch-cut-safe complex interpolation, `optprops.interp_phase_deg`); a phase-carrying table forces Python routing (`coating_phase` cengine token — not yet C-ported) and a pre-run/CLI warning fires when a coherent scene uses a phase-invalid table. `scripts/tools/import_zemax_coating.py` converts real Zemax TABLE files. Tables WITHOUT phase columns (most of the shipped 39-row library — one demo row, `bs_5050_vis_45_ph`, carries phase) still borrow the bare-interface Fresnel phase, same as before.|Zemax/OSLO's measured-coating tables can carry phase as standard practice and their catalogs are far larger; MieWorkbench's own library still ships phase for only one illustrative row (real vendor phase curves are rarely published).|Populate more of the shipped library's phase columns from vendor/TMM data as it becomes available; C-engine table-coating phase support — **[S·Low]**|
|**Measured BSDF scatter** (H2 🟡)|ABg model, **BRDF + optional BTDF (transmitted-side)** since P2/P2.5 (`engine3 overhaul` round), single-scatter, isotropic; 4 shipped rows (one, `lightly_ground_glass_window`, exercises BTDF), flagged UNVERIFIED.|Zemax imports full BSDF (BRDF+BTDF), anisotropic, with importance sampling; the scatter physics MieWorkbench *does* have (Mie/volume) beats them, but the measured-import tooling and anisotropy are narrower.|Anisotropic (per-azimuth) fit — **[M·Med]**|
|**Grating efficiency** (C9 🟡 for CODE V/OSLO comparison)|Four closed-form models with real efficiency (Kogelnik VBG, Dammann, measured table); lamellar/Dammann are polarization-blind; **no RCWA**; reflection VBGs not modeled.|Zemax uses rigorous **RCWA** (exact for sub-wavelength/non-sinusoidal grooves). CODE V/OSLO are scalar/efficiency-limited — MieWorkbench actually *leads* those two on closed-form efficiency.|RCWA is a deliberate non-goal (§7.15); reflection-Kogelnik is the pragmatic increment (Backlog b) — **[M·Low]**|
|**Biaxial birefringence** (C5 🟡)|Validated two-sheet quartic solver (KTP/KTA/LBO/BiBO, `<1e-9`); the **only** biaxial in the field. **Internal conical refraction** is now modeled behind `--conical` (samples-instruments round, off by default); the entry interface uses the exact Berreman 4×4, but internal reflections are still **sheet-preserving** and the exit interface still uses an **effective-index Fresnel** approximation.|No competitor here has biaxial at all, so MieWorkbench is strictly ahead — the "partial" is vs the rigorous ideal, not vs a competitor.|Exact anisotropic Fresnel at the exit interface + cross-sheet internal-reflection coupling (Backlog b) — **[L·Med]**|
|**Curved-vs-flat, multi-config, ghost** (M1 ⚠️, F9 ✅/🟡)|Multi-config is CLI-sweep + Variables dock + Compare pane (no named editor). Ghost is a specular **path ranking** (top multi-bounce paths by detected power).|Zemax/CODE V/OSLO/QUADOA have named multi-config editors (12–unlimited configs) and Zemax's Path Analysis / Critical Ray Tracer is a fuller stray-light workflow than a ranked list.|Config-table GUI (§7.10) — **[M·Med]**; fuller stray-light report on the ghost ranking (§7.9) — **[M·Med]**|
|**Photocurrent / QE** (F2 partial)|`qe_curve` body property → photocurrent_A + coverage_frac, but only **1** QE curve ships and there's no CLI flag.|Zemax/vendor tools ship large detector-QE libraries.|Add QE-curve library rows + a CLI/GUI surface — **[S·Low]**|

## Partial features — behavioral gaps vs the physical ideal

The same partials, framed the way this doc has always framed them: **honest limits vs the
rigorous physics**, independent of any competitor. (Kept distinct from the section above so
the "vs ideal" breadcrumbs stay intact.) Each names its approximation and the seam that
would make it exact.

| Partial feature | Approximation today | Rigorous ideal | Seam |
|--|--|--|--|
|**Uniaxial interface Fresnel** (C4)|**EXACT (P6, Python engine):** Lekner-1991 4×4 tangential-E/H boundary match per ray — full o/e split + reflected s/p Jones incl. `r_sp`/`r_ps` cross terms, Poynting-flux-normalized (unitary to 1e-10). Legacy effective-index kept behind `--biref-approx`.|C-engine port of the 4×4 solve (exact path currently Python-routes via `biref_exact`); propagate o↔e conversion on exit internal reflection.|`bir.uniaxial_interface_in`/`_out`; `tracer._birefringent_children` (README §5.6)|
|**SHG / χ² conversion** (S2)|Undepleted Boyd quadratic, η clamped ≤0.5; **no walk-off, no angular detuning, equal s/p harmonic split, no cascaded re-conversion**.|Depleted coupled-wave tanh²; type-I/II e/o geometry with Poynting walk-off; cascade.|bulk SHG event (README §6.12); needs exact-uniaxial Fresnel|
|**Self-phase modulation** (S4)|**Source-side only**, quasi-classical single-time-per-frequency; exact FFT spectrum installed as an SPD.|Split-step NLSE with real intra-train SPM+GVD interplay.|`sources.py` SPM transform (README §5.2.1); split-step propagator|
|**GDD budget** (S3)|**Material dispersion only**; geometric GDD (gratings/prisms/angular chirp) shows up in the traced time products instead, not the analytic table.|Unified analytic GD/GDD/TOD incl. angular-dispersion dθ/dλ term.|GDD-budget table (README §6.11)|
|**Mesh optical faces** (D9)|Traced for geometry/**incoherent power only** — a tessellated sag error ≫ λ makes coherent optical-path phase meaningless.|Analytic or wavelength-accurate faces carry coherent phase.|keep phase-critical surfaces analytic (README §5.8)|
|**Continuum particle scattering** (H4)|**Incoherent by construction** — contributes power, never fringe structure.|Coherent multiple-scattering transport.|continuum medium path (README §6.2) — deliberate scope choice|
|**Gather occlusion** (default off)|No occlusion test between a gather sample and the pixel unless `--gather-occlusion`; then tile-quantized + fully-opaque occluders.|Per-pixel, translucency-aware occlusion.|`gather.py` two-level AABB + tile-shadow (README §6.5)|
|**Diffuser / ground-glass depolarization** (H1)|Single-scatter Beckmann microfacet — no shadowing/masking, no subsurface transport; real ground glass depolarizes more.|Multiple-scatter + subsurface transport.|`roughness`/diffuser sampler (README §5.4.1)|

## Performance & C-rewrite opportunities

Where wall-clock could improve later, with an emphasis on **moving hot Python paths
into the C engine** (`cengine/`, `scripts/raytracer/cengine.py` `PORTED` set). The
Python engine stays the PERMANENT reference — these are *additive* C ports gated by
`--engine`, never behavioral changes. Ordered roughly by leverage. Effort/impact per
the legend above.

| Opportunity | Effort | Impact | Where / why |
|--|:--:|:--:|--|
|**Optimizer/tolerancer inner-loop evaluator** (design tools)|L|**High**|The dominant new cost is a full FreeCAD rebuild + extract + trace *per evaluation*. First mitigation is the persistent-worker + fingerprint-cache fast evaluator (`scripts/fast_eval.py`, planned); the next step is a **resident C incoherent trace-only evaluator** that skips subprocess spawn + JSON round-trip per eval (compute the merit scalar directly from the in-memory trace result). This is the single biggest win for making optimization/tolerancing interactive.|
|**C-port the remaining NLO token**|S|Med|Only `nonlinear` (χ² SHG/Pockels) still Python-routes; the rest shipped in P7 (`cengine.PORTED`). The SHG child-spawn needs the C children queue to learn stratum extension.|
|**C-port the remaining unported trace features**|M-L|Med|Per `cengine.py`, these force Python and could each be ported: **biaxial birefringence**, **explicit-realization particle clouds** (numba DDA/uniform-grid traversal removes the 200k cap too — Backlog (c)), **`--ray-differentials`** transport, **curved detectors**, **ABg `g≠2`**, **`rough_fresnel=macro`**, extra CLI detector faces. Each shrinks the "Python-routed" scene set.|
|**Thermo-optic index term in the C index path**|S-M|Med|The planned dn/dT feature routes any `--temperature≠T0` run to Python (temperature deliberately kept out of `PORTED`). Replicating the `n(λ,T)` thermo-optic term in the C engine's `n_complex`/`medium_index` mirror keeps thermal scenes C-routable.|
|**FFT-heavy post-process (PSF/MTF/image-sim)**|M|Med|`analysis_field`/`analysis_imaging`/`post_process` PSF/MTF/Zernike + the planned partial-coherence/image-sim convolution are pure numpy; large grids would benefit from FFTW/`cupy`/a C-CUDA FFT pass. Post-process only, engine-agnostic — a self-contained accel target.|
|**Per-pixel gather occlusion**|M|Low|`--gather-occlusion` is tile-quantized + fully-opaque today (`gather.py`); a per-pixel C/CUDA shadow-ray pass (with a translucency-aware transmission accounting) sharpens shadow edges without the tile tradeoff.|
|**Incremental FreeCAD extract / tessellation cache**|M|Med|The extract stage (`extract_geometry.py`, FreeCAD-Python) re-tessellates every body per variant. The fast evaluator's persistent worker + shape-fingerprint cache (mirroring `mieworkbench/core/geomcache.py`) skips unchanged bodies — the same cache could back the ordinary sweep/pipeline path, not just the optimizer.|
|**Torch-gather sunset**|—|—|Already on the cengine roadmap (`cengine/README.md` §Sunset): once the C-CUDA gather is the default and shaken out, retire the torch gather backend + its ~5 GB dependency from the optics env; keep the numpy gather as the slow readable reference.|
|**Mesh source/detector faces in C**|M|Low|When the mesh-face UV parameterization lands (Backlog (c)), implement it directly in the C mesh path rather than Python-only.|

## Operational

- Estimator calibration: `results/.calibration.json` self-improves per run;
  seed it with a few normal-preset runs so `--dry-run` predictions tighten.
  It also does not yet fold in the extra cost of `--ray-differentials` or
  `--gather-occlusion` (README §12) — both add real wall time/memory beyond
  the base `gather_ops` model.
- `sweep_variants.py --jobs jobs.json` + `compare_runs.py` are wired; a
  worked multi-job example lives in README §sweeps.
- Detector `.h5` at detailed preset (4096^2 x 32 bins) ≈ 2 GB/detector; mind
  `/` at ~97% (outputs can be pointed at a roomier disk via `--case-dir`,
  or by setting `MIEWB_RESULTS_DIR` in miewb.env).
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
- **C-engine port of the remaining NLO token**: `nonlinear` (χ² SHG/Pockels)
  is the last Python-only token from the pulsed/NLO round —
  `time_products`, `gdd_budget`, `ray_differentials`, `saturable`, `tpa`,
  `kerr` shipped as C registrations in P7 (`cengine.PORTED`). The SHG
  child-spawn needs the C children queue to learn stratum extension.
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
- **Deferred demos (P13 scope call)**: `prism_compressor` (two-prism
  Fork pair — needs the min-deviation central-ray trace helpers factored
  out of demo_prism_spectrometer + a mirrored-prism placement solve) and
  `wideangle_retrofocus` (negative-front retrofocus + field fan +
  --imaging-products distortion). Both have their physics fully
  validated engine-side (material GDD budget, angular dispersion via
  traced arrivals, imaging products on imaging_analysis); only the
  gallery scenes are missing.
- **Transmission-grating truncated-order booking**: a bk7
  `grating_plate` with `orders=-1..1` leaks the truncated lamellar
  orders (~8% at 800 nm/600 g/mm) past the closure gate — the
  REFLECTIVE branch books the remainder into absorbed_surface exactly
  (see `demos/UXNOTES.md` #11). The transmission branch needs the same
  remainder credit.
- **Dominant-cluster auto time window**: the auto window spans ALL
  arrivals incl. double-bounce ghost echoes ~60 ps out — a 100 fs pulse
  then lands in one 2 ps bin. Cluster the records (or window on the
  p0.1–p99.9 power span) so fs pulses resolve without a hand-pinned
  --time-window.
