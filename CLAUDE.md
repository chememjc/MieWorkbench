# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

A git repository (main branch `master`) holding a physically-based **coherent
Monte-Carlo optical ray tracer** driven by annotated FreeCAD models: tagged
`.FCStd` scenes go through extract → trace → post → viz to produce detector
irradiance images (with real interference/diffraction — the double-slit scene
yields λL/d fringes), 2D/3D ray renders, spectra, Stokes polarization maps,
and an energy audit. Full polarization physics: Jones-vector rays, incoherent
polarization strata, polarizers, uniaxial birefringence (calcite/quartz/
sapphire o/e splitting with walk-off), spectral filters, tabulated + TMM
coatings, Kogelnik/Dammann/table gratings, aspheres, mesh BVH tracing.
**`README.md` is the authoritative human reference** (authoring contract,
physics model + honest limits, opticalproperties CSV schemas, full command
reference, validation results, troubleshooting); this file is a terse
operator map. Conventions follow `~/Documents/cfdsim/SimulationsGuide.md`.

## Pinned interpreters — always use the right one (never cross-import)

| Stack | Interpreter | Scripts |
|---|---|---|
| FreeCAD embedded | `/home3/freecad/FreeCAD.AppImage -c <script> -- <args> < /dev/null` | `extract_geometry.py`, `permute_model.py`, `make_test_scenes.py` |
| optics env (numpy/scipy/torch-CUDA/miepython/h5py) | `/home3/optics/env/bin/python` | `run_trace.py`, `post_process.py`, `compare_runs.py`, all of `scripts/raytracer/`, pytest |
| ParaView 6.1.1 | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython --force-offscreen-rendering` | `make_viz.py` |
| system `python3` | stdlib only | `run_pipeline.py`, `sweep_variants.py`, `common.py` (self-check: `python3 scripts/common.py`) |

## One-command pipeline

```bash
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
# stages extract→trace→post→viz into results/<model>/<preset>[-<tag>]/
# --print-only / --dry-run / --keep-going / --steps ... ; presets:
# quick=1e5 rays/512^2/5λ, normal=1e6/2048^2/9λ, detailed=1e7/4096^2/17λ
# fidelity flags (all mirrored from run_trace): --ray-differentials,
# --gather-occlusion, --save-fields (Stokes maps), --rough-fresnel
# micro|macro, --no-pol-scatter, --mesh-flat-normals, --strict-analytic,
# --viz-density RAYS_PER_MM2 (default 1.0; --viz-rays = absolute override)
# permutation: --var lenspos --min -5 --max 5 --n 2
```

Tests: `/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -q`
(~230 tests; `test_gather.py`, `test_doubleslit_e2e.py`, `test_allflags_e2e.py`
take minutes — deselect with `-m "not slow"` plus ignores for quick loops).
24 element test scenes: `make_test_scenes.py --scene all` (FreeCAD), metadata
in its importable `SCENES` dict; e2e physics gates in `test_scenes_e2e.py`.

## Body-tagging contract (details README §5)

`App::Property` on each `PartDesign::Body`, group "Base":
- `material` (string): materials.csv row | uniaxial crystal name from
  `birefringence/uniaxial.csv` (calcite/quartz/sapphire) | `detector` |
  absent/`none` → ignored
- sources: `power` (mW) + `lambdac` (nm) [+ `lambdamin`/`lambdamax` nm,
  `coherent` bool, `polarization` = `unpolarized` | `linear:<deg>` |
  `circular:left|right` | `elliptical:<psi>:<chi>`]
- optic extras (stackable): `coating` (`MgF2` whole-body or
  `Face3=MgF2;Face5=pbs_visible_45`), `roughness` (float whole-body nm RMS
  or string `Face1=200:lcorr=5`), `filter` (bulk spectral, filters.csv row),
  `polarizer` + `polarizer_axis` (`x,y,z` body-local, default 0,0,1),
  `crystal_axis` (`x,y,z` body-local, default 1,0,0 — uniaxial c-axis),
  `grating` (`Face2=600:v:orders=-1..1` or `Face2=@vbg_1800` registry),
  `surface_override` (`FaceN=asphere:R=25.0;k=-0.6;A4=1.2e-6;r_max=10`,
  mm units, verified <1 µm against the authored face or extraction dies),
  `mirror`, `absorbance`
- dimensions in a `Spreadsheet::Sheet` with aliased `=<val> mm` cells

**Aperture scenes:** fill slit/hole openings with thin `material=air` bodies —
re-anchors coherent-gather wavefront samples AT the aperture.

## Optical component library

`opticalproperties/`: `materials.csv` at the root (Sellmeier/cauchy/constant/
tabulated + `nk/` tables), `coating/coatings.csv` (TMM `layers` XOR measured
`table` = Rs/Rp/Ts/Tp CSV + `aoi_deg`), `polarizer/`, `filter/`, `grating/`,
`birefringence/uniaxial.csv` — each registry references per-item CSVs under
`<category>/tables/`. `reference` (citation) column is REQUIRED everywhere;
loaders hard-validate at startup (`raytracer/optprops.py`). Path override:
`--optical-properties DIR`.

## Quirks / traps (each cost real debugging time)

- **FreeCAD `-c`**: bare `--` before script args; script runs TWICE per
  invocation (writes must be idempotent); NO `if __name__=="__main__"` guard;
  `print()` can drop (log via `FreeCAD.Console` too); `sys.exit` swallowed →
  `os._exit`; always `< /dev/null`. `.FCStd` = zip: `unzip -p X.FCStd
  Document.xml` for recon without FreeCAD.
- **`orientation_outward` contract semantics**: the flag describes the
  CANONICAL normal the tracer derives from the stored analytic params, NOT
  FreeCAD's orientation-corrected `normalAt()` — probing the wrong vector
  inverted every plane crossing once (all power → `seam_loss`). The extractor
  probes the canonical normal; don't "simplify" this.
- **Detector grid basis is arbitrary**: always read `xhat`/`yhat` from the
  detector `.h5` attrs; never assume grid x = global y.
- **Gather keys are (source, λ-stratum, POL-stratum)**: unpolarized sources
  emit two mutually-incoherent populations, halving samples per key; add a
  birefringent element and o/e splits halve again — budget ~4× rays or the
  `GatherError: undersampled` gate trips (message says how much to raise).
- **Polarization strata are load-bearing**: co-located co-polarized fields in
  different pol strata must NOT interfere; don't "optimize" the per-key
  accumulators away. Circular polarizer elements act in GENERATOR orientation
  (linear stage then retarder); to ANALYZE handedness, stack a quartz
  waveplate body + a linear polarizer body.
- **Optically-contacted solids don't exist**: two touching solids = overlap
  error or seam-leak chaos. Model cemented interfaces (PBS halves, Wollaston
  wedges, achromat elements) as a small AIR GAP (~5 µm); README documents the
  Fresnel-fidelity tradeoff. Wollaston with a gap double-refracts at the gap
  (4 beams) — a real contacted Wollaston needs the future contact feature.
- **`seam_loss` audit bucket**: rays crossing exactly through a face-face seam
  whose trim tests disagree are killed and accounted there. ~1e-5 of power is
  normal; large values mean broken geometry (or a contacted-solid attempt).
- **Detected power is a diagnostic, not a closure bucket** — detectors are
  transparent screens; the ledger partitions LOSSES only, gates at 1e-3.
  New buckets: `polarizer_absorbed` (dichroic rejection).
- **Gather noise floor**: stored detector maps are UNBIASED with zero-mean
  negative MC noise; clip only for display. `norm_factor_dimensionless` ~O(1)
  means the dA model is consistent (with `--ray-differentials` it is, except
  at caustics where differential patch areas legitimately collapse).
- **Mesh faces trace but phase doesn't**: BVH mesh tracing is
  power/geometry-faithful; coherent OPL through a tessellated face carries
  sag error ≫ λ (loud warning at Scene build). Keep phase-critical surfaces
  analytic (spheres, cones, `surface_override` aspheres).
- **Asphere authoring**: revolve a `Part.BSplineCurve` through exact sag
  points AND set `surface_override`; the extractor verifies to <1 µm and
  DIES on mismatch (never silently corrupts phase). Coeff units are mm-based:
  `A_n_SI = A_n_mm · 10^(3(n-1))`.
- **phi is a MASS fraction vs the ambient medium** (particle clouds): dense
  particles in air need surprisingly large phi.
- **CUDA OOM in the gather**: reduce `pixel_chunk`/`sample_chunk` in
  `gather.points_torch`; with `--gather-occlusion` the mask is sliced
  sample-columns-FIRST (row-first fancy indexing OOMed at ~9 GB — caught by
  `test_allflags_e2e.py`; don't reorder).
- miepython 3.x API: `efficiencies_mx(m,x)`, `S1_S2(m,x,mu)`; it wants
  m = n − ik (conjugate of our n + ik). numpy 2.x: `np.trapezoid`.
- ParaView offscreen: cosmetic warnings harmless; body-face STLs merged per
  role in `viz_common.py`. New view: `rays_polmode` (o/e coloring).
- Disk: `/` is chronically ~97% full; envs/tools live on `/home3`;
  `--save-fields` off by default (complex Ex/Ey per gather key is large);
  animations to scratch space only.

## Physics invariants pinned by tests (don't break them)

Fresnel R+T=1 (1e-12); Brewster; TIR phase; TMM λ/4 MgF2 on BK7 and
half-wave absentee; thick-lens focus vs lensmaker (0.5%) for
PCX/DCX/PCV/DCV/ball/rod/cylinder scenes; double-slit fringe pitch λL/d ±1px
+ visibility >0.85 END-TO-END from FCStd; Malus's law 1%; crossed polarizers
→ `polarizer_absorbed`; calcite walk-off 6.226° @45°/590nm vs 6.23°
published; quartz opposite-sign walk-off; e-ray normal-surface residual
<1e-12; slab o/e round-trip parallelism + displacement; multi-order quartz
HWP between crossed analyzers (gathered, coherent o/e recombination);
Kogelnik η=1 at ν=π/2 Bragg; Dammann Parseval; polarized-Mie azimuth χ² +
θ-marginal invariance; Igehy differentials vs finite differences + r² law +
R/2 and n₂R/(n₂−n₁) foci; asphere Newton vs sphere/parabola closed forms;
BVH == brute force; occlusion off-path is bit-identical + torch==numpy
masks; Mie Qext vs Wiscombe; medium-stack push/pop; energy closure <1e-3 in
EVERY scene incl. the all-flags composite; torch/numpy gather <5e-3.

## Everything else

README.md: authoring contract with per-property syntax examples,
opticalproperties schemas + citation policy, physics model + honest limits
(§6: effective-index anisotropic Fresnel, tabulated-coating phase, mesh OPL,
tile-quantized occlusion, thin-hologram Kogelnik, nonlinear optics & optical
activity out of scope), full CLI reference, 24-scene catalog + validation
table, troubleshooting. future.md: delivered items struck, open seams
(exact uniaxial Fresnel, optical contact, reflection-Kogelnik, RCWA, curved
detectors, GRIN, fluorescence).
