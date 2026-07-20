# Changelog

Per-round changelog, maintained from the c-engine round onward. Earlier history (2026-07-06 through 2026-07-09, Phases A-C through object-placer) is in `git log`, not reconstructed here.

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
