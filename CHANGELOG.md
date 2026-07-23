# Changelog

Per-round changelog, maintained from the c-engine round onward. Earlier history (2026-07-06 through 2026-07-09, Phases A-C through object-placer) is in `git log`, not reconstructed here.

## samples-instruments round — 2026-07-23

Scattering-instrument bench work: liquid/particle samples with real structure
factors, T-matrix spheroids, extended image sources, blackbody/line emitters,
a physical diode-array readout + absorbance + ring-profile CLI, and a traced
dynamic-light-scattering (DLS) workflow — plus two real Python-engine bug
fixes found by the new nested-cell scenes.

- **Two bug fixes (both Python engine; the C engine never had either):**
  - **Weakly-absorbing-incident Fresnel branch flip.** `fresnel.cos_theta_t`'s
    unconditional `Im(n2·cos_t) >= 0` decay rule negated a genuinely
    *propagating* root whenever the incident medium carried trace absorption
    (water, k~1e-8) into an exactly lossless far medium: numerical dust gave
    `Im(cos_t) < 0`, the flip turned `cos_t=+0.9999` into `-0.9999`, the
    near-cancelling Fresnel denominator spiked `|rs|~15`, and a nested-cylinder
    chord loop amplified every bounce — `vial_cylindrical`'s default
    glass↔water interface exploded closure from 1 mW in to ~1e8 W booked
    (3.7e16 in the worst reproduction). Fixed with a radiation-condition
    branch (`Re(n2·cos_t) >= 0`) for the effectively-propagating regime,
    reserving the decay rule for genuinely evanescent roots (`|Im| <=
    1e-9·|q|` gates which rule applies). Vial-scene closure 3.7e16 → 1.85e-13;
    all 48 Fresnel/TIR/TMM invariant tests + parity spots stay green.
  - **Trace hop-cap silently truncating live rays.** `Tracer.run`'s
    termination valve was a *global* POP budget (`64*(max_reflections+2)`)
    shared across every source and consumed by `batch_size` chunk splits —
    on a many-interface stack at scale it dumped still-eligible rays into
    `truncated_generation` well before they were actually done (found by the
    `nested4` depth-4 spike: 37.8% of emitted power lost at 60k rays,
    bit-reproducible). Replaced with a **per-lineage hop cap**: each batch
    carries its ancestry step count, chunk splits inherit it unchanged
    (splitting is now budget-neutral), the queue drains fully, and only a
    genuinely exhausted 512-segment lineage truncates (with a named-power
    warning).
- **Internal conical refraction** (`--conical` / `--conical-fan N` (default
  16) / `--conical-delta RAD` (default 1e-4)): `birefringence.py` gains
  closed-form biaxial optic axes and Hamilton cone half-angle (Born & Wolf;
  KTP 1.61°/KTA 1.25°/LBO 1.40°/BiBO 3.69° @1064nm, both verified to
  <1e-8/<1e-10 vs numeric limits), a per-ray axis-proximity dispatch, and a
  perturbed two-sheet fan that reproduces Hamilton's internal cone (2N
  coherent children, Poggendorff double-ring, φ/2 polarization half-turn) as
  the perturbation angle shrinks. Off (default) is unchanged behavior — an
  arbitrary orthonormal transverse basis at the degeneracy, now also tallied
  into `conical_guard`; on, fanned rays are tallied into `conical_fanned`
  (both in `audit.json`/`case.json`). Fixed a pre-existing Berreman qz
  mode-collision bug along the way: forward-mode assignment could double-map
  both sheets onto one mode exactly at the degeneracy (46% closure violation
  for an on-axis beam through a biaxial body) — now forced distinct. Still
  Python-only (biaxial scenes always Python-route).
- **S(q) structure factors + explicit lattice realizations**
  (`structure.py`): exact Percus-Yevick hard-sphere (Wertheim/Thiele) and
  exact Baxter sticky-hard-sphere (delta-shell PY, SasView-verified, replaces
  an earlier flagged approximation), Teixeira fractal/OZ gel, powder
  paracrystal (fcc/bcc/sc peak positions + selection rules), and tabulated
  S(q); wired into the continuum ensemble tables (`<S>_p`-scaled `mu_sca`,
  decoupling approximation for polydispersity — an honest limit) and into
  `ExplicitRealization` for real fcc/bcc/sc lattice site placements
  (`mode=explicit`, Gaussian jitter `sigma=g*a`) for coherent Bragg/speckle
  work.
- **Body-bound sample media** (`sample` body property + `sample/
  samples.miesamp`, 7 shipped rows): a particle population bound to a
  liquid-fill solid's interior via exact containment (the medium-stack top
  == the host body), host material overriding the Mie contrast/density/OPL
  vs the real solvent; `MEDIUM_STACK_DEPTH` 4→8 (both engines) after
  cuvette-in-bath-in-vat nesting hit the old cap exactly. `depth-4 nesting`
  validated end-to-end (`basemodels/nested4`, cuvette-in-bath-in-vat, 6
  pairwise `nested_solids`, closure <1e-12, detected power within 0.11% of
  the Fresnel × Beer-Lambert oracle).
- **T-matrix spheroids** (`tmatrix.py`, via `pytmatrix`, optics-env-only soft
  dependency): orientation-averaged non-spherical particles behind the same
  `MieEvaluator` interface (`efficiencies()`/`amplitudes()`), volume-
  equivalent radius convention, disk-cached builds. Physics catch documented
  in the module: pytmatrix's own coherent-S orientation averaging is exact
  for Qext (optical theorem) but ~15% low for Qsca vs independent random
  orientations — Qsca/g/|S1|/|S2| are instead derived from the (correctly
  bilinear-averaging) phase matrix Z, verified <0.02% against pytmatrix's own
  slow reference integrals. Wired into both continuum and explicit sample
  media via `shape`/`aspect_ratio` sample-registry columns.
- **Extended image-emitting source** (`image` + `image_cone_deg` body
  properties + `image/images.mieimg` + the `usaf_style_target` bitmap,
  MIL-STD-150A-style-alike, generated by `scripts/tools/gen_usaf_target.py`):
  a per-position radiance map on a source face, alias-method density
  sampling at equal per-ray power, Lambertian emission by default (an
  imaging bench needs each object point to fill the aperture) with an
  optional cone restriction as a variance optimization. Excludes `beam`/
  `apodization`. Ported to the C engine (`image_source`, T18a): the SAME
  `sources._build_alias_table` builds the alias table serialized into the
  request; `trace.c`'s `sample_image_pos_dir` does the alias draw + in-pixel
  jitter + Lambertian/cone emission, with two reserved RNG event slots
  keeping every other stream thread-invariant.
- **Blackbody + line emission kinds** (`emission/emitters.miesrc` gains
  `params`/`lines` columns; loader now validates `kind ∈ {continuous,
  blackbody, lines}`): Planck synthesis at load time (Wien-verified) for
  `blackbody`, per-line strata with Hamilton largest-remainder apportionment
  + finite non-overlapping `linewidth_nm` bands for `lines` (equal-power
  warning when `n_lambda < n_lines`). Emission registry 2→5 rows:
  `bb_halogen_3000k`, `d2_uv_approx` (tabulated UV continuum), `hg_penlamp`
  (NIST-cited 11-line row).
- **10 new primitives** (catalog 70→80): `cuvette_square`/
  `cuvette_capillary`/`flow_cell` (nested wall+liquid rectangular cells),
  `vial_cylindrical` (DLS vial), `vat_cylindrical` (decalin index-matching
  bath), `sample_region` (bare air anchor cube for unwalled clouds),
  `tungsten_halogen`/`d2_lamp`/`hg_calibration` (the three new lamp emission
  rows as sources), `source_image` (the USAF-style Lambertian image
  emitter). All nested pairs follow the `bs_cube` exact-containment pattern
  (probe-verified volumes, zero overlaps).
- **Instruments: physical diode-array readout, absorbance, log-annular ring
  profile.** New `diode_array` instrument class (`pixel_height_um`/`n_px`
  columns) + `tcd1304_array` row (TCD1304 datasheet geometry, USB4000
  as-operated full-well/read-noise/ADC): `render_diode_array` bins the ideal
  detector-plane cube onto the array's real pixel geometry with QE-exact
  per-bin electrons and the existing shot/read-noise + full-well/ADC
  conventions. `--reference-case DIR` renders `A(λ) = -log10(I/I0)` against a
  blank case's matching instrument product (instrument/absorbance CSV+PNG).
  `--ring-profile 'n=N:rmin_mm=..:rmax_mm=..[:center=peak|chief|X,Y]'`
  (laser-diffraction-sizer style): log-spaced annular power bins with exact
  closure including inside/outside remainders (`analysis_field.
  log_annular_power`, analysis/rings CSV+PNG).
- **Traced-dynamics DLS** (`scripts/run_dls.py` + `scripts/
  dls_correlate.py`): `run_dls.py` builds the scene once off a single
  EXPLICIT-mode sample body, pre-generates a sequential Brownian frame
  sequence (Stokes-Einstein D per particle, reflective-wall BC), traces
  frames embarrassingly-parallel (`SeedSequence.spawn`, CUDA-safe), and
  persists per-frame RAW coherent detector fields to `dls/frames.h5`.
  `dls_correlate.py` is a fully offline, re-runnable correlator: FFT g1(τ)
  per incoherent channel, Siegert g2 = 1+β|g1|², a weighted cumulant fit for
  Γ → D → hydrodynamic diameter. Validated correlator-first (synthetic OU
  field, Γ to 2%) and end-to-end (Γ = 2.0×Dq² at the accepted factor-3 bar).
  Honest limits: dilute well-separated explicit clouds only (the shared
  Monte-Carlo RNG stream desyncs on a dense cloud's changing collision set,
  collapsing g1 to delta-correlated noise), frozen radii, no hydrodynamic
  interactions or sedimentation/flow.
- **C engine**: `image_source` and `sample_body` (continuum-mode) both
  ported this round — S(q) needs zero C-side logic (the corrected
  mu_ext/albedo + size-averaged inverse-CDF table are pre-resolved
  Python-side and serialized as plain tables). `MEDIUM_STACK_DEPTH` 8 on
  both engines. `conical` and `sample_explicit` (the EXPLICIT/lattice
  sample realization) stay Python-only routing tokens.
- **Co-located transparent detectors**: detector-detector solid overlap now
  classifies into the informational `validation.detector_overlap` list
  instead of the fatal `overlapping_solids` — "measure the same plane two
  ways" is a physically well-defined stack of transparent screens.
- **Expression-driven ANCHORED placements**: `miewb_expr_pos_x/_y/_z` +
  `miewb_expr_rot_rx/_ry/_rz` (group `MieTrain`, `train_solver.
  place_anchored`, degrees-native `miewb_vars` grammar) let an anchored
  element's world pose itself be an expression (e.g. a goniometer detector
  swept by `pos_x=R*cos(theta)`) — baked inside the same `solve_chain` GUI
  and headless permute share, pinned by a θ-sweep parity oracle.
- **Library**: materials 847→849 (`decalin`, Sigma-Aldrich n20/D dispersion;
  `dye_solution_kmno4`, a cited aqueous KMnO4 UV/Vis absorbance standard);
  detector QE curves 1→4 (`toshiba_tcd1304ap`, `sony_ilx511b`,
  `hamamatsu_s3904`); emission 2→5; new `sample`/`image` registries (7 + 1
  rows respectively).
- GUI: samples/images Library tabs + `sample`/`image`/`image_cone_deg`
  property editors (the same registry-combo + tooltip pattern as `spectrum`);
  a Results "dls" gallery; `libschema` current for every new/changed
  registry column.

## docs round — 2026-07-19

- Full documentation-currency audit of every doc against the code at 4832abb: 9 parallel section audits, all numeric claims re-measured (coatings 39, gratings 9, 42 demos / 21 baselined, 935 engine + 1219 GUI tests, 70 primitives, 33 validation scenes).
- Physics docs corrected to post-P6/P9 reality: exact Lekner/Berreman are the DEFAULT interface amplitudes (effective-index is the legacy `--biref-approx` path); scene-level natural optical activity documented as shipped+asserted (new RAYTRACER.md §6.12b); C-engine ported/Python-routed token lists completed.
- Preview round (engine selector, log-dB extinction, Preview Configuration dialog) documented in README.md and the UI test checklist; sequential/Optiland framed everywhere as preview/evaluation aid, never a co-equal analysis engine.
- New RAYTRACER.md §7 registry schemas (emission/figure/diffuser/nonlinear) + §8 flag coverage (--engine, --viz-pattern, --resume/--extend, --imaging-products, compare_sweep.py); CUSTOMIZE.md gains emission/instrument/figure-error authoring sections.
- Historical design ledgers (engine.md/engine2.md/engine3.md, UI_COORDINATE_PROPOSAL.md) archived to docs/archive/; the four UXNOTES shakedown logs consolidated into one open-items demos/UXNOTES.md; future.md pruned of landed items (RCWA, Lekner, near-axis optical activity, P7 C-ports).
- This CHANGELOG.md and the docs/README.md doc-set index introduced; CLAUDE.md gains the standing doc-hygiene rule (every round updates CHANGELOG.md + affected docs before merge).
- Full docs/guide screenshot set recaptured via the offscreen capture tool; gui_verify xvfb pass rerun.

## preview-config round — 2026-07-19 (4832abb)

- Live-preview trace engine selector: `sequential` (fast Optiland path) vs `full` (forced MC subprocess, Fresnel ghosts); per-document via `Project.set_preview_config`, default `full`.
- Log-dB extinction mode for ray dimming (`ray_dimming_range_db`); full-trace + Off extinction auto-selects log.
- New unified Preview Configuration dialog (pattern, engine, extinction, bead-animation keys) replacing scattered settings entry points.

## polish round — 2026-07-18 (merged 7ce0f7d)

- Optimize/Tolerance panes: qualified `miewb_vars` names, persisted configs, surface-failure handling, plot inspection, Apply-optimum, toolbar contrast fix.
- Property-library editor: full column schema (tooltips/status/validation), extended to biaxial/figures/nonlinear/scatter/instruments rows.
- docs/guide per-feature docs + screenshot tooling + Help menu; `gui_verify.py` xvfb screenshot harness now required before closing GUI work.
- Opt-in bead-opacity animation (log-dB power + leading-wavefront); sequential preview emits OPL so beads animate on on-axis systems.
- Element-level selection (whole-element picks, member sub-selection, clear); demo gallery gains optimize/tolerance studies + 4 showcase demos.
- Fix-round follow-ups: quartz_rotator asserts natural optical activity; C-parity retarder scene moved to MgF2; gyration token declared PYTHON_ONLY.

## engine3 overhaul round (P0-P9) — 2026-07-16 (close-out 2026-07-17)

- P0: quick-win display/gather fixes, corrected estimator law, MIT LICENSE.
- P1: chunked run contract (checkpoint/resume/extend) on the C tiled gather; 11.5x tile-factorized gather; per-run RunDialog; opt-in NUFFT gather route (off by default).
- P2/P2.5: coating phase columns, measured-scatter BTDF, Forbes Q-bfs/Q-con surfaces, parallel-transport Q matrix (honest retardance/diattenuation); virtual instrument layer + bench-comparison contract.
- P3 (core-v3): interaction-registry rewrite replaces the trace if-chain; persistent C-engine worker (`--serve`) with CUDA buffer pool.
- P4a/P4b: Optiland parity oracle; deterministic sequential mode (DLS lens design); later unified so live preview shares the run's physics.
- P5: prescription-primary data model — prescription is truth, CAD is a view.
- P6: RCWA grating tables via meent (pinned 0.12.0); exact Lekner-1991 uniaxial interface amplitudes, default on.
- P7: time_products/gdd_budget and Igehy ray differentials ported to the C engine; saturable/TPA/Kerr NLO ported. chi2 `nonlinear` (SHG/Pockels split) stays Python-routed.
- P8: bladed-iris N-fold aperture, per-face Zernike figure error, edge-blackened lens barrels (Python-routed).
- P9: Berreman 4x4 full-anisotropy module; biaxial/absorbing/gyrotropic interfaces default-routed through it.

## design-tools round — 2026-07-13

- Schott thermo-optic model (TIE-19) + per-body operating temperature through Scene/C-engine; Schott+Ohara AGF catalogs imported (168 -> 847 materials).
- In-app Python console bound to the live session.
- Partial-coherence imaging + image simulation, with an Imaging results gallery.
- Fast merit evaluator (in-place extract/permute + persistent-worker fingerprint cache); merit-function optimizer (scipy NM + nevergrad CMA) with Optimize pane.
- Sensitivity + Monte-Carlo yield tolerancing with nested focus compensator, Tolerance pane; GUI gains central tabbed 3D/Optimize/Tolerance/Results area.
- Documentation currency sweep + competitive re-analysis vs Zemax/QUADOA/3DOptix/CODE V/OSLO, adversarially reviewed.

## pulsed-optics round — 2026-07-11 to 2026-07-12

- Time core: per-ray group-delay accumulators/path tally; dispersion group-index/GDD/TOD derivative API.
- Pulsed sources: `power` XOR `pulse_energy` contract; 6 new pulsed/supercontinuum laser primitives.
- GDD budget, SPM source model, bulk SHG event; chi2 registry (d_il tensors, d_eff/phase-matching).
- NLO elements: transverse Pockels EO, saturable/TPA absorption, Kerr lens.
- Time-binned detector + 4 time products (pulse/spectrogram/streak/cube); GUI pulsed/time-product integration.
- Imaging exit-pupil/chief-ray stage + 4 field products; 7 pulsed-optics demo benches.

## design-usability round — 2026-07-11

- Paraxial ABCD engine for element cardinals + train system summary, surfaced in the element/train editors.
- `opticalvalues` model + full-lens asphere design in the wizard/primitive; lens_asphere/prism_equilateral scene fixes.
- Tabulated emission-spectrum sources; `detector_face` body property (pins the recorded face, C-routable).
- Pipeline forwarding (`--views`/`--smoke`/`--viz-generations`, `--save-fields-detectors`); expression grammar (whitelisted math, degrees-native trig, `pi`).
- 8 new demo galleries plus MieTrain-preserving rebuild fixes and an offscreen GUI shakedown pass.

## c-engine round — 2026-07-10 to 2026-07-11 (merged efe445e)

- Compiled OpenMP+CUDA trace/gather core (`cengine/`) behind `--engine {auto,python,c}`; auto-routes to C when every used feature is ported, else falls back to Python (permanent reference).
- Phases A-I: OpenMP trace core, TLAS BVH + mesh BLAS, fused-CUDA coherent Huygens gather, gratings/roughness/diffusers/ABg scatter, uniaxial birefringence, continuum particle clouds, export-rays/ghost analysis, opt-in `--importance-aim`.
- Parity oracle: side-by-side scenes, root fuzz, TLAS==linear-BVH, thread invariance; lineage-keyed Philox RNG (bit-identical) vs Python's numpy RNG (statistical parity).
- GUI Simulation Settings dialog, persisted into `.MieWB`; michelson-family benchmarks 4.8-6.5x at 2e5 rays.
