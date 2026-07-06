# future.md — staged follow-ups with exact starting points

Change-management notes in the pcbsim style: what to run/extend next and
where the seams are. Nothing here blocks current use.

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
  registry (`opticalproperties/grating/gratings.csv`, `@name` syntax).
  RCWA is still explicitly out of scope (README §5.5).
- **Aspheres** — shipped: `surface_override='FaceN=asphere:...'`,
  extract-time verification against the real FreeCAD face to 1um,
  bracket-guarded Newton intersection (README §5.7).
- **`--save-fields` (complex Ex/Ey field export + Stokes/DOP maps)** —
  shipped end-to-end: `run_trace.py --save-fields` writes
  `detectors/<label>.h5`'s `fields/<key>/{Ex,Ey}` groups (seed 0 only),
  and `post_process.py`'s `render_stokes_maps()`/`stokes_from_jones()`
  render `stokes_<label>_<key>.png` + `dop_<label>.png` from them
  (README §6.5). Not yet done: no CLI flag caps `--save-fields` to a
  subset of detectors/keys, and `common.estimate()`'s `fields_h5_GB`
  dry-run prediction isn't wired to the actual `--save-fields`/pol-strata
  state (README §12) — both are cheap follow-ups if disk budgeting for
  `--save-fields` runs becomes a recurring pain point.

## Higher-fidelity physics (still open)

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
- **Biaxial crystals and gyrotropy.** Also explicitly out of scope in
  `birefringence.py`; would need a genuinely different (non-uniaxial)
  normal-surface solver.
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
  See "delivered" above — the differential machinery exists and is
  correct for reflection/refraction/free-space transfer; extending it
  through a diffraction order, a Beckmann-scattered lobe, or an o/e split
  is the natural next increment (`tracer._kill_differentials`'s call
  sites are the exact insertion points).
- **Tile-quantized / opaque-only gather occlusion.** See "delivered"
  above — a per-pixel (`tile=1`) mode already exists as an opt-in; a
  translucent/partial occluder model (rather than always-fully-opaque)
  would need a coupled-transmission accounting scheme to avoid
  double-counting the tracer's own refracted-field samples.
- **Stress birefringence.** Induced (not intrinsic) birefringence from
  mechanical stress in an otherwise-isotropic material — would reuse the
  existing uniaxial o/e machinery with a stress-optic-coefficient-derived
  `n_o`/`n_e`/axis per body rather than a fixed crystal lookup.
- **GRIN (gradient-index) media.** Not modeled — every medium is currently
  homogeneous between interfaces; a GRIN element needs curved-ray
  propagation inside the bulk (a genuinely different integration scheme
  from the current straight-segment-between-hits loop).
- **Fluorescence / phosphors.** No wavelength-shifting absorption-then-
  reemission event exists; would need a new emission event type
  (isotropic, incoherent, at a shifted wavelength stratum) triggered by
  bulk or surface absorption in a fluorescent material.
- **Measured BSDF scatter.** Roughness today is Beckmann-microfacet only;
  a tabulated BRDF/BTDF (e.g. from a goniophotometer) would be a new
  scattering-lobe sampler alongside `roughness.beckmann_sample`.
- **Ghost-image analysis mode.** No dedicated tool exists to isolate and
  rank specific reflection paths (e.g. Nth-surface ghosts in a lens
  stack) by detected power; would consume the existing per-ray generation/
  medium-stack history already carried on `RayBatch`, mostly a
  post-processing feature rather than new physics.
- **Thermal lensing.** No temperature-dependent index/absorption coupling
  exists; would need a coupled thermal (absorbed-power -> local
  temperature -> `dn/dT` -> refractive index) model, likely iterative.
- **Curved detector faces.** `DetectorGrid` still hard-errors on a
  non-Plane surface (README §5.11/§6.2); UV-parameterized grids on
  spheres/cylinders are straightforward for the incoherent path, but the
  coherent gather needs per-pixel normals and a non-planar obliquity term.

## Capability gaps (hard errors today, by design)

- **Explicit particle clouds > `MAX_BRUTE` (200,000) spheres** are capped
  (brute-force chunked collision; `--particle-threshold` default is now
  aligned to this same cap, README §9). A numba DDA/uniform-grid traversal
  removes the cap; grid build is already cell-hashed in
  `ExplicitRealization._place`.
- **Multi-process tracing** (`--workers`): the loop is single-process
  vectorized numpy; shard primary rays via `SeedSequence.spawn` and merge
  ledgers/detector cubes (all accumulators already add linearly).
- **Mesh-type source/detector faces** still hard-error unconditionally
  (README §5.8/§5.11) — both need a UV parameterization the incoherent/
  coherent paths don't have yet; the ordinary-optic mesh path (BVH tracing)
  is otherwise fully shipped.
- **Aspherical particles** (user goal): T-matrix (e.g. `pytmatrix`) drop-in
  behind the `MieEvaluator` interface — `efficiencies()` and `amplitudes()`
  are the only two entry points `particles.py` uses. Still open.

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
- `run_pipeline.py` does not forward `--views`/`--smoke` (make_viz.py) or
  `--viz-generations` (post_process.py) — call those scripts directly to
  use them (documented in README §4.2/§8).
- No CLI flag caps `--save-fields` to a subset of detectors or gather
  keys; it is currently all-or-nothing per completed trace.

## Known cosmetics

- `wavelength_rgb` saturates deep red/violet ends (CIE tails); purely
  cosmetic in the sRGB detector images.
- `rays_polmode`'s ParaView view silently skips (with a warning) if
  `viz/rays.vtp` predates the `pol_mode` array (a stale `viz/` combined
  with `--skip-vtkexport`) — rerun without `--skip-vtkexport` to refresh it.

## Scene-suite findings (test_scenes_e2e.py xfails — follow-up work)

Five documented xfails in `test_scenes_e2e.py`; the tracer is faithful in
each case, the follow-up is scene authoring / model extension:

- **lens_asphere design math**: `k = -n^2` on the convex front makes only
  that surface stigmatic in-glass; the flat exit re-adds spherical
  aberration and the full lens over-corrects (best-focus RMS ~3x worse
  than the spherical control instead of >=5x better). Re-solve the conic
  (or add A4/A6 terms) for the complete lens and update `SCENES`.
- **prism_equilateral geometry**: `prism_rotation_deg=19.4` puts the beam
  at ~10 deg AOI (not the intended 49.4 deg minimum-deviation entry), so
  the exit face TIRs; the rotated detector's auto-picked screen face is
  also edge-on. Fix the rotation + detector normal.
- **pol_circular is a GENERATOR, not an analyzer**: the polarizer model
  applies linear diattenuator -> retarder in propagation order, which
  cannot discriminate incident handedness (left/right transmit equally).
  Analyze circular light with a quartz waveplate body + linear polarizer
  body (fully supported, validated by waveplate_quartz). A future
  `orientation` column in polarizers.csv could flip the stage order.
- **pbs_cube / wollaston air gaps**: without optically-contacted solids,
  the 5 um gaps Fresnel-split (PBS loses ~35% to seam/ghosts; the
  Wollaston double-refracts at the gap into 4 beams instead of 2, split
  angle off accordingly). Real fix = cemented/contact interface support:
  paired coincident faces treated as a single material-to-material
  boundary (no ambient hop). Until then the scenes document the physics
  of *air-gapped* assemblies.
- **Rotated off-axis detector bodies** (prism/pbs/hot_mirror reflected
  arms) can auto-pick an edge-on screen face; the e2e tests read the
  deviated beams from ray directions instead. A `detector_face` body
  property (explicit screen-face override at authoring time, mirroring
  the CLI --detector-face) would remove the ambiguity.
