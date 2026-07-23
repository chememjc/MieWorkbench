# FreeCAD → Coherent Monte-Carlo Ray Tracer → Detector Images / 3D Viz

> This is the optical-engine reference. For the MieWorkbench GUI and project-level docs see the top-level [README.md](../README.md).

> Numbers marked "measured" come from real completed runs in this repo's
> `results/` tree (see §11/§12). Numbers marked "estimate" come from
> `common.estimate()` / `case.json["estimates"]` — a self-calibrating linear
> model driven by `results/.calibration.json` — and should be treated as a
> rough guide, not a promise. Sections state which is which.

A Python pipeline that turns a tagged FreeCAD `PartDesign` optical bench
(lenses, mirrors, gratings, polarizers, waveplates/crystals, filters,
apertures, particle clouds, detector screens) into a physically-based,
coherent Monte-Carlo ray trace and a set of detector images, spectra,
Stokes/polarization maps, energy audits, and 3D/ParaView visualizations —
end to end from one command. `example.FCStd` (a divergent+collimated
two-laser bench with a BK7 lens, a glass sphere, and three detector
screens) and `doubleslit.FCStd` (a Young's double-slit wave-optics
validation scene) ship in the repo root with completed runs under
`results/` that this README's "measured" numbers are drawn from;
`scripts/make_test_scenes.py` can additionally author 32 more FreeCAD test
scenes on demand (§10) covering polarizers, birefringent crystals, filters,
coatings, aspheres, and a deliberately non-analytic mesh face.

---

## 1. What this is

The pipeline has four stages, each with its own pinned Python interpreter,
chained by `scripts/run_pipeline.py`:

```
 .FCStd  --[FreeCAD -c]-->  model.json + faces/*.stl  --[optics env]-->  rays.npy + detectors/*.h5
(tagged bodies)              (extract_geometry.py)         (run_trace.py)
                                                                  |
                                                                  v
                                                     images/spectra/plots + report.json
                                                        (post_process.py, optics env)
                                                                  |
                                                                  v
                                                     viz/*.png + *.pvsm  (make_viz.py, pvpython)
```

| Stage | Script | Interpreter | Reads | Writes |
|---|---|---|---|---|
| permute (optional) | `permute_model.py` | FreeCAD AppImage (`-c`) | `<model>.FCStd` | `basemodels/<stem>-<var><val>....FCStd` |
| extract | `extract_geometry.py` | FreeCAD AppImage (`-c`) | `<model>.FCStd` | `geometry/<stem>/model.json` + `geometry/<stem>/faces/*.stl` |
| trace | `run_trace.py` | optics env python | `geometry/<stem>/model.json`, `opticalproperties/*.mie*` (+ `*/tables/*.mietab`) | `results/<stem>/<case>/{case.json,audit.json,rays.npy,detectors/*.h5}` |
| post | `post_process.py` | optics env python | the trace outputs above + `model.json` | `results/<stem>/<case>/{images,spectra,plots}/*.png` + `report.json` |
| viz | `make_viz.py` | ParaView pvpython | `rays.npy`/`detectors/*.h5` (via a `raytracer.vtkexport` sub-step under the optics env) + body STLs | `results/<stem>/<case>/viz/*.png` + `<case>.pvsm` |

`run_pipeline.py` itself runs under plain system `python3` (stdlib only)
and never imports FreeCAD/numpy/torch/paraview — it only composes argv
lists and launches each stage's pinned interpreter as a subprocess.
`scripts/common.py` is the shared, stdlib-only contract hub every stage
imports: pinned paths, fidelity presets, CLI spec parsers (face/grating/
roughness/polarization/axis/particle specs), the `model.json` contract
validator (schema v2), sweep semantics, and the runtime/memory estimator.

**The physical model this pipeline traces**: geometric ray propagation
(reflection/refraction/absorption/gratings/roughness/particle scattering/
polarization/birefringence) plus a coherent Huygens–Fresnel "final gather"
at each detector screen — see §6 for what is exact and what is an
approximation.

---

## 2. Environment (pinned)

| Tool | Path | Notes |
|---|---|---|
| FreeCAD | `/home3/freecad/FreeCAD.AppImage` | FreeCAD 1.1.1, headless `-c` console mode |
| optics env python | `/home3/optics/env/bin/python` | numpy/scipy/torch/miepython/h5py/matplotlib |
| ParaView / pvpython | `/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython` | ParaView 6.1.1 |
| GUI venv python | `env/bin/python` | PySide6 GUI + the Optiland *sequential* preview/optimizer-evaluation bridge (not a pipeline stage) |
| orchestrators (`run_pipeline.py`, `sweep_variants.py`) | system `python3` | stdlib only — never imports FreeCAD/numpy/paraview |

These paths are the **env-overridable defaults** in `scripts/common.py`:
`FREECAD_APPIMAGE` (`MIEWB_FREECAD`), `OPTICS_PYTHON`
(`MIEWB_OPTICS_PYTHON`), `PVPYTHON` (`MIEWB_PVPYTHON`), and `GUI_PYTHON`
(`MIEWB_GUI_PYTHON`, default `env/bin/python`) — each reads its
environment variable and falls back to this machine's pin. `GUI_PYTHON`
hosts the PySide6 workbench and its Optiland sequential engine (used for
the interactive ray preview and as the optimizer's paraxial evaluator);
it is **not** one of the four pipeline stages — the nonsequential coherent
Monte-Carlo trace above is the physics engine for every run. Run `python3
scripts/common.py` for a self-check (verifies the interpreter paths
exist, `opticalproperties/materials.miemat` exists, and a battery of
pure-math invariants — sweep semantics, face/grating/roughness/
polarization/axis/particle spec parsing, the runtime estimator).
**Always use the pinned interpreter for each stage** — the three stacks
(FreeCAD's embedded Python, the optics env, ParaView's bundled Python)
share nothing but the standard library, which is why `common.py` is
deliberately stdlib-only. `run_pipeline.py` composes and launches the
exact pinned command for each stage; you never need to type these paths
yourself when driving the pipeline through it (§8).

This repository is also a git repository (`origin` points at a private
Gitea remote); nothing about that changes how the pipeline is invoked.

---

## 3. Repository layout

```
opticalraytracer/
├── example.FCStd, doubleslit.FCStd     # example tagged models (§5)
├── opticalproperties/                  # editable optical component library (§7)
│   ├── materials.miemat                #   bulk n(lambda)/k(lambda) database
│   ├── nk/*.mienk                      #   tabulated n,k spectra (metals, water, TiO2)
│   ├── coating/coatings.miecoat (+tables/*.mietab) # TMM stacks AND measured Rs/Rp/Ts/Tp tables
│   ├── polarizer/polarizers.miepol (+tables/)  # linear/circular diattenuator tables
│   ├── filter/filters.miefilt (+tables/)   #   bulk spectral filters (Beer-Lambert)
│   ├── grating/gratings.miegrat (+tables/) #   lamellar/Kogelnik/Dammann/table registry
│   ├── birefringence/uniaxial.miebrf   #   uniaxial o/e crystal pairs (§5.6)
│   └── birefringence/biaxial.mibiax    #   biaxial n_x/n_y/n_z crystal triples (§5.6b)
│   (content is still plain CSV under self-describing extensions; a legacy
│    all-.csv library loads unchanged via the same-stem fallback, §7)
├── scripts/
│   ├── common.py            # paths, PRESETS, CLI spec parsers, model.json validator,
│   │                        #   sweep semantics, runtime/memory estimator
│   ├── extract_geometry.py  # FreeCAD headless: .FCStd -> geometry/<stem>/model.json + STLs
│   ├── permute_model.py     # FreeCAD headless: sweep spreadsheet alias(es) -> basemodels/*.FCStd
│   ├── make_test_scenes.py  # FreeCAD headless: authors 33 validation FCStd scenes (§10)
│   ├── run_trace.py         # optics env: model.json -> rays.npy + detectors/*.h5 (the solver)
│   ├── post_process.py      # optics env: trace outputs -> images/spectra/plots + report.json
│   ├── run_pipeline.py      # system python3: orchestrates permute/extract/trace/post/viz
│   ├── sweep_variants.py    # system python3: N run_pipeline.py jobs + a compare_runs.py overlay
│   ├── compare_runs.py      # optics env: overlay several finished cases' detector results
│   ├── make_viz.py          # pvpython: batch ParaView renders driven by viz_configs.VIEWS
│   ├── viz_common.py        # pvpython: shared reader/camera/colormap/render helpers
│   ├── viz_configs.py       # pure-Python declarative view registry (no paraview import)
│   └── raytracer/           # the physics engine package (§6) + tests/ (§11)
├── geometry/<stem>/          # extract_geometry.py output: model.json + faces/<face_id>.stl
├── basemodels/               # permute_model.py output + make_test_scenes.py scenes
└── results/<stem>/<case>/    # run_trace.py + post_process.py + make_viz.py output (§4)
```

`<case>` is the case directory name, by default `<preset>[-<tag>]`
(e.g. `quick`, `normal-hires`), computed by `common.case_name()` and used
identically by every stage so trace/post/viz always agree on where a
run's files live.

---

## 4. Quickstart

```bash
python3 scripts/run_pipeline.py --models example.FCStd --preset quick
```

This runs extract → trace → post → viz in order, in
`results/example/quick/`, and prints a summary table (model, case, status,
detected power per detector). Expected output tree after a completed run:

```
results/example/quick/
├── case.json               # options echo + status ("estimated" -> "completed") + diagnostics
├── audit.json               # per-seed energy ledger (closure gated at 1e-3, §6/§11)
├── rays.npy                 # viz polylines (N,13): source_id, lam_m, power_W, x0..z0, x1..z1,
│                            #   pol_mode (0=isotropic/ordinary, 1=uniaxial extraordinary,
│                            #   2/3=biaxial slow/fast, §5.6b), rel_power (power/birth_power in
│                            #   [0,1] — drives --dim-rays), opl0_m, opl1_m (optical path Σn·ds at
│                            #   segment start/end; t = opl/c drives the tracer-bead animation;
│                            #   escaped-ray stubs get a synthetic opl1 = opl0 + n·0.25 m matching
│                            #   the drawn stub)
├── rays_full.npz            # only if --export-rays/--ghost-analysis: per-detector namespaced
│                            #   seed-0 ray records (§8.2) + JSON meta (grid basis, seed, cap,
│                            #   face_labels if --ghost-analysis)
├── detectors/
│   └── <safe_label>.h5      # spectral cube (mean [+std if --seeds>1]) + grid basis (§5.11)
│                            #   + optional fields/<key>/{Ex,Ey} complex maps if --save-fields
│                            #   + optional pixel_area_map + curved-detector attrs (§5.12) if the
│                            #   detector face is a Sphere/Cylinder
├── data/                    # only if --emit-csv: one *.csv per chart below + index.csv
│                            #   (file -> entity, chart, units, provenance, image), §8.3
├── analysis/                # only if --export-rays (spot_/fan_/ghost_*) and/or --save-fields
│                            #   (psf_/mtf_/ee_*), plus wavefront_* when both a coherent key has
│                            #   enough rays (§6.10, §8.3)
├── images/
│   ├── det_<label>.png              # wavelength-colored sRGB irradiance
│   ├── det_<label>_lin.png          # linear grayscale irradiance (colorbar, W/m^2)
│   ├── det_<label>_log.png          # log10 grayscale irradiance
│   ├── det_<label>_profiles.png     # horizontal/vertical cuts through the peak pixel
│   ├── stokes_<label>_<key>.png     # 2x2 S0/S1/S2/S3 panel — only if --save-fields (§6.7)
│   └── dop_<label>.png              # summed degree-of-polarization map — only if --save-fields
├── spectra/
│   └── spectrum_<label>.png         # detected power per spectral bin
├── plots/
│   ├── rays_xy.png                  # XY cross-section, rays colored by wavelength, body outlines
│   │                                #   (o/e-split birefringent rays get a distinct marker/color;
│   │                                #   --viz-generations N declutters to reconstructed-gen <= N)
│   ├── materials_nk.png             # n(lambda)/k(lambda) of every material used in the scene
│   ├── coating_reflectance.png      # R(lambda) of every coating used (if any)
│   ├── polarizer_<name>.png         # T_parallel(lambda)/T_perpendicular(lambda) — per polarizer used
│   ├── filter_<name>.png            # internal transmittance at ref thickness vs lambda — per filter used
│   ├── grating_<name>.png           # per-order eta_s/eta_p vs lambda — per registry grating used
│   └── energy_audit.png             # stacked-bar power ledger per source, closure error annotated
├── viz/
│   ├── overview3d.png, top.png, side.png   # ParaView renders (§4.2, §8.3)
│   ├── rays_polmode.png        # rays colored by pol_mode (ordinary vs extraordinary) instead of wavelength
│   ├── detector_closeup_<label>.png
│   ├── turntable_frame0..7.png
│   ├── rays.vtp, det_<label>.vtp, det_<label>.json   # intermediate VTK/metadata
│   └── <case>.pvsm            # combined interactive ParaView scene
└── report.json                # per-detector total_power_W / peak_irradiance / profile_visibility
```

`safe_label` is the detector face id with `.` replaced by `_` (e.g.
`Body003.Pad001.Face5` → `Body003_Pad001_Face5.h5`). The Stokes/DOP images
and `fields/` HDF5 groups only appear when the trace was run with
`--save-fields` (§6.7, §8.2) — every other output above is unconditional.

### 4.1 Individual stages

```bash
# 1. geometry extraction (note the bare "--" before extract_geometry.py's own flags)
/home3/freecad/FreeCAD.AppImage -c scripts/extract_geometry.py -- --models example.FCStd

# 2. the ray trace itself
/home3/optics/env/bin/python scripts/run_trace.py \
    --model-json geometry/example/model.json --case-dir results/example/quick \
    --rays 1e5 --resolution 512 --nlambda 5 --spectral-bins 16

# 3. rendering/analysis (rerunnable without re-tracing)
/home3/optics/env/bin/python scripts/post_process.py \
    --case-dir results/example/quick --model-json geometry/example/model.json

# 4. 3D visualization
/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64/bin/pvpython \
    --force-offscreen-rendering scripts/make_viz.py \
    --case-dir results/example/quick --model-json geometry/example/model.json
```

### 4.2 Note: `run_pipeline.py`'s viz stage still uses some of make_viz.py's defaults

`run_pipeline.py`'s internal `viz_cmd()` builder forwards `--case-dir`,
`--model-json`, `--dim-rays`/`--dim-rays-floor`, `--views`, and `--smoke`
to `make_viz.py` — it does **not** forward `--resolution`/`--out`/
`--skip-vtkexport`, so driving the pipeline through `run_pipeline.py`
always renders at 1920×1080 (2048×2048 for `detector_closeup`) into
`<case-dir>/viz/` and always reruns the vtkexport prep step (unless
`--smoke` narrows the render to `overview3d` at 800×600). `post_cmd()`
forwards `--dim-rays`/`--dim-rays-floor` (attenuation dimming: segment
opacity = P/P_birth, linear or sqrt curve, optional percent floor —
applies to `rays_xy.png` and the 3D ray renders), `--photometric`,
`--spectrometer`, `--emit-csv`, `--wavefront-point`, and
`--viz-generations`. To pick a different resolution or skip the vtkexport
rerun, invoke `make_viz.py`/`post_process.py` directly (§8) on an
already-completed case directory.

---

## 5. Model authoring contract

This is the user-facing contract every `.FCStd` model must follow for
`extract_geometry.py` to accept it (enforced by
`common.validate_model()`, called both at extract time and by every
downstream stage that loads `model.json`). The extractor always emits
`schema_version: 2`, which adds — on top of the v1 contract —
`polarizer`/`polarizer_axis`/`filter`/`crystal_axis`(+`crystal_axis2` for
biaxial)/per-face `coating`+`roughness`+`grating`+`scatter`+`figure_error`
maps/`diffuser`/`asphere` surfaces, the detector extras
`detector_face`/`qe_curve`/`instrument`, the per-body `temperature`, and
the source extras `spectrum`/`apodization`/`beam_waist`+`m2` plus the
pulsed-optics `pulse_energy`/`rep_rate`/`pulse_duration`/`spm` blocks;
`validate_model()` accepts both versions.

### 5.1 Body tagging (group "Base" custom properties)

`extract_geometry.py`'s `classify_body()` reads these `App::Property*`
fields on each `PartDesign::Body`:

| Property | Type | Meaning |
|---|---|---|
| `lambdac` (nm) + (`power` (mW) XOR `pulse_energy`) | Float | `lambdac` present **and** at least one of `power`/`pulse_energy` present marks the body a **source** (checked first, before `material`; `scene.py` then enforces the `power`/`pulse_energy` XOR) |
| `pulse_energy` (µJ) + `rep_rate` (Hz) | Float | source-only, pulsed optics: energy-specified pulse train (alternative to `power`; needs `rep_rate`). Derived {P_pk=0.94·E/τ, κ} echoed in `case.json` `source_pulse` (§5.2) |
| `pulse_duration` (ps FWHM) | Float, optional | source-only: pulse FWHM; on a `power`-only source it authors a *virtual* pulse (§5.2). Auto-enables the `pulse,spectrogram` time products (§8.1) |
| `spm` | String, optional | source-only self-phase modulation: `'phimax:<rad>'` or `'gamma:<W⁻¹km⁻¹>:length:<m>'` — installs the exact-FFT SPM spectrum as the source SPD plus an S-curve chirp via stratum birth-time offsets (§5.2). Python-engine-routed |
| `material` | String | a row name in `opticalproperties/materials.miemat`, a crystal name in `birefringence/uniaxial.miebrf` (§5.6) or `birefringence/biaxial.mibiax` (§5.6b), `"detector"`, or `"none"`/absent |
| `coating` | String, optional | a coating name (whole-body) or a per-face map `'Face3=MgF2;Face5=x'` (§5.3) |
| `polarizer` | String, optional | a row name in `opticalproperties/polarizer/polarizers.miepol` (§5.3) |
| `polarizer_axis` | String `'x,y,z'`, optional | body-local transmission-axis vector, default `0,0,1` (§5.3) |
| `filter` | String, optional | a row name in `opticalproperties/filter/filters.miefilt` (§5.3) |
| `sample` | String, optional | optic-only: a row name in `opticalproperties/sample/samples.miesamp` (or an inline spec) binding a particle population to THIS body's interior — the body's own `material` is the host medium, its shape bounds the cloud (§5.13) |
| `crystal_axis` | String `'x,y,z'`, optional | body-local optic-axis vector for a uniaxial birefringent `material` (§5.6), default `+x`; the X principal axis for a biaxial `material` (§5.6b) |
| `crystal_axis2` | String `'x,y,z'`, optional | body-local Y principal axis; REQUIRED alongside `crystal_axis` when `material` is a biaxial crystal (§5.6b), otherwise ignored with a warning |
| `scatter` | String, optional | a per-face map (or whole-body) of `opticalproperties/scatter/bsdf.miebsdf` registry names (§5.4.2) — mutually exclusive with `roughness`/`diffuser` on the same face |
| `scatter_targets` | String, optional | comma-separated detector labels this body's measured scatter aims at under `--importance-scatter` (§8.1; absent → every detector). Names only, resolved at trace; never affects the physics, only variance reduction |
| `surface_override` | String, optional | per-face `'FaceN=asphere:R=..;k=..;A4=..;...;r_max=..'` (§5.7) |
| `mirror` | Float, optional, [0,1] | achromatic partial-reflector fraction (§5.3 precedence) |
| `absorbance` | Float, optional, [0,1] | fraction of the physical (non-mirror) remainder absorbed (whole-body). Per-face absorbance is set indirectly by `edge_blackened` |
| `edge_blackened` | Bool, optional | lens/optic stray-light suppression: blacken the CYLINDRICAL barrel/edge (fully absorbing) while the refracting caps stay clear. The extractor turns it into a per-face absorbance on every cylinder face (identified by surface TYPE, so it survives FaceN renumbering). Convenience flag on the simple-lens primitives (pcx/dcx/pcv/dcv/meniscus). Python-engine-routed (per-face absorbance) |
| `figure_error` | String, optional | Zernike SURFACE figure error (§5.9): a per-face map (or whole-body) of `opticalproperties/figure/figures.miefig` registry names. Applies a Zernike sag perturbation to the analytic surface at scene build (deterministic wavefront error between the ideal surface and statistical roughness). Python-engine-routed |
| `temperature` | Float, optional | per-body operating temperature in °C; shifts glasses carrying a thermo-optic model (§ materials, Schott TIE-19). Overrides the scene-global `--temperature`. Routes the run to the Python engine. |
| `roughness` | Float or String, optional | whole-body RMS nm, or a per-face map `'Face1=200:lcorr=5;Face2=50'` (§5.4) |
| `diffuser` | String, optional | ground glass: `'grit:120'` \| `'slope:0.08'` \| `'@dg_600'`, whole-body or per-face map `'Face2=@dg_600'` (§5.4.1) — mutually exclusive with `roughness` on the same face |
| `grating` | String, optional | a per-face map `'Face2=600:v:orders=-1..1'` or `'Face2=@registryname'` (§5.5) — must name specific faces, not the whole body |
| `detector_face` | String, optional | detector-only: pin the recorded PRIMARY face — a bare `'FaceN'` (this body) or a full `'Body.Tip.FaceN'` id — overriding the closest-to-origin auto-pick (§5.2). Replaces the primary face in place (no extra screen), so the scene stays C-engine-routable, unlike the additive CLI `--detector-face` |
| `qe_curve` | String, optional | detector-only: a row name in `opticalproperties/detector/detectors.miedet` — `post_process.py` reports a QE-weighted photocurrent block (§7.8) |
| `instrument` | String, optional | detector-only: `'row'` or `'row:mode'` (mode `ideal`\|`full`, default `full`) naming a row in `opticalproperties/instrument/instruments.mieinst` — `post_process.py`'s virtual instrument layer (§7.11) reads that detector's ideal spectral cube through the named instrument profile |
| `polarization` | String, optional | source-only: `'unpolarized'` (default) `'linear:<deg>'` `'circular:left|right'` `'elliptical:<psi>:<chi>'` (§5.2) |
| `lambdamin`, `lambdamax` (nm, optional), `coherent` (bool, default False) | — | source-only, see §5.2 |
| `spectrum` | String, optional | source-only: `emission/emitters.miesrc` registry row naming a tabulated emission spectrum (§5.2); supersedes `lambdamin`/`lambdamax` |
| `apodization` | String, optional | source-only: `'gaussian:w0=<mm>[:order=<n>]'` transverse field-amplitude taper across the emitting face (§5.2) |
| `beam_waist` (mm) + `m2` (optional, default 1.0) | Float | source-only: Gaussian-beam mode — position sampled from the waist intensity profile plus a per-ray angular divergence (§5.2); requires a planar emitting face |
| `image` | String, optional | source-only: a row name in `opticalproperties/image/images.mieimg` naming a per-position radiance bitmap (§5.14); mutually exclusive with `beam`/`apodization`; requires a planar emitting face |
| `image_cone_deg` | Float, optional, (0,90] | source-only: emission-cone HALF-angle restriction for an `image` source (default: full Lambertian, §5.14) |

Classification (`classify_body`, in this priority order):
0. `miewb_exclude` truthy (a bool set by the GUI on unfolded fold
   mirrors, or any user-excluded element) → **ignored**. The body stays
   in the document — it is invisible to the physics only. All other
   `miewb_*` properties (the GUI's optical-train metadata, group
   `MieTrain`) are never read by the extractor at all.
1. `lambdac` present **and** at least one of `power`/`pulse_energy`
   present → **source** (`scene.py` then enforces the `power`/`pulse_energy`
   XOR the OR-gate deliberately lets through).
2. `material` missing/empty or `"none"` (case-insensitive) → **ignored**
   (skipped entirely; extractor logs `"[Label] is ignored"`).
3. `material == "detector"` (case-insensitive) → **detector**.
4. Otherwise → **optic**, and `material` must be a row in
   `opticalproperties/materials.miemat` **or** a crystal name in
   `opticalproperties/birefringence/uniaxial.miebrf` (or
   `birefringence/biaxial.mibiax`; a name that resolves to none is a hard
   error at trace time when the `Scene` is built).

`mirror`/`absorbance` values outside [0,1] are **not** rejected — they are
capped to [0,1] with a loud warning (`capped01()`), and `model.json`
contract validation (`common.validate_model`) then hard-errors if a body
still shows an out-of-range value after that capping (an extractor bug,
not a user error). `polarizer`/`filter`/`crystal_axis`/`crystal_axis2`/`apodization`/
`beam_waist` set on a non-`optic` (or non-`source`, for the source-only
properties) body are ignored with a warning (they are only meaningful on
their intended role); `crystal_axis` is **always emitted** for every optic
body (default local `+x` if the property is absent) since the tracer's own
local-+x default would otherwise be ambiguous without the body's
Placement rotation applied. `crystal_axis2` is emitted only when authored
(there is no default second axis) — the `Scene` loader raises if a body's
`material` resolves to a biaxial crystal (§5.6b) but `crystal_axis2` is
absent or (near-)parallel to `crystal_axis`. A model needs **at least one
source** and **at least one detector**, or extraction/validation fails
outright.

Extract-time validation resolves only geometry-shape properties
(`surface_override`'s asphere declaration, §5.7); it does **not** check
that a `polarizer`/`filter`/`grating` name actually exists in its
registry — those "unknown polarizer/filter/registry" errors fire later,
at trace time, when `Scene.__init__` builds the optical-property tables
(§13).

### 5.2 Sources: emission, wavelength, and polarization

A source body's emitting face is auto-detected at extract time as the
face **closest to the global origin** (`extract_faces()`'s
`closest_dist` tracking) — same rule extract_geometry.py uses to
auto-pick a detector body's recorded face. Override on the trace stage
with `--source-face Body.Feature.FaceN` (retargets that source's
`emit_face`; matched by owning body, hard error if the face doesn't
belong to a source body or doesn't exist on it). `--detector-face
Body.Feature.FaceN` adds an *extra*, transparent zero-effect detector
screen on any face (including an optical element's own face) without
disturbing its physical interaction — the field is recorded, then the
ray continues to interact with the surface normally.

To retarget a detector body's own recorded PRIMARY face (rather than add
an extra screen), set the `detector_face` **body property** (§5.1):
a bare `'FaceN'` or a full `'Body.Tip.FaceN'` id, resolved and validated
against that body's extracted faces at extract time (`autodetected:false`
in `model.json`). This is the authoring-time fix for the "auto-pick lands
on a thin edge face → detects 0 mW" trap on rotated/off-axis detectors
(the folded-telescope eyepiece). Unlike the additive `--detector-face`
CLI flag, it does not add a screen (`extra_detector_faces` is NOT in the
C engine's ported feature set — it silently forces the Python engine),
so a scene pinned via the body property stays C-engine-routable.

**Emission direction** (`sources.py`):
- **flat emitting face** → collimated beam along the face normal, with
  the sign chosen once (from the mean "toward-origin" vector across the
  sampled points) so the whole beam heads into the origin hemisphere —
  this is what makes `example.FCStd`'s Laser/DivergentLaser bodies behave
  as a laser aimed at the bench.
- **curved emitting face** (sphere/cylinder) → each sampled point emits
  along its own **local surface normal**, sign chosen **per sample**
  toward the origin. If a face's normals straddle the origin direction (a
  mixed-sign face), the against-origin samples are dropped and their
  power is credited to the `emission_clipped` audit bucket (with a
  `RuntimeWarning`) — a fully backward-facing curved emitter (100%
  clipped) instead just flips its whole normal field, no warning.

**Wavelength distribution** (`sources.wavelength_strata()`, from
`lambdac_nm` plus optional `lambdamin_nm`/`lambdamax_nm`):
- **neither bound given** → monochromatic, a single stratum at `lambdac`.
- **both bounds given** → asymmetric Gaussian: `sigma- = lambdac -
  lambdamin`, `sigma+ = lambdamax - lambdac`, each half a half-normal, with
  the left/right side chosen per stratum with probability
  `sigma / (sigma- + sigma+)`.
- **exactly one bound given** → uniform on `[lambdac - w, lambdac + w]`
  where `w` is the given half-width (whichever of `lambdamin`/`lambdamax`
  was set).
- **`spectrum` set** (`emission/emitters.miesrc` registry row) → inverse-CDF
  of the tabulated piecewise-linear PDF: each stratum sits at a quantile
  center `(k+0.5)/n` so every stratum carries equal power (the strata cluster
  where the spectrum is bright). The table's own `[min, max]` sets the source
  wavelength range (detector spectral bins cover it); `spectrum` supersedes
  `lambdamin`/`lambdamax` (both dropped with a warning if also present), but
  `lambdac` must still be present (it marks the body a source and is the
  power-weighted mean λ). **Honest limit:** only `kind=continuous` tables are
  supported — `blackbody` (analytic Planck) and `line` (discrete line lamps)
  rows are staged in the library but rejected at load with a "needs engine
  support" error until their source models land.

Sampling is **stratified**: `--nlambda` equal-probability wavelength
strata, one deterministic wavelength per stratum, equal power weight per
stratum; each ray belongs to exactly one wavelength stratum, and the
coherent gather keeps separate per-(source, wavelength stratum,
polarization stratum) accumulators (§6) — different optical frequencies,
and mutually-incoherent polarization populations, never interfere
stationarily. `coherent` (bool, default False) controls phase: coherent
sources start every ray at phase 0 on the emitting surface (the surface
itself is the reference wavefront, `opl = 0` there); incoherent sources
get a uniform random phase per ray (fringe visibility ≈ 0).

**Polarization** (`sources.py`, `common.parse_polarization_spec`): the
optional `polarization` body property, source-only, string-valued:

| Value | Meaning |
|---|---|
| `unpolarized` (default) | two mutually-incoherent orthogonal populations, `pol_stratum` 0/1 |
| `linear:<deg>` | linear polarization at `<deg>` from the reference axis |
| `circular:left` / `circular:right` | circular polarization (handedness convention below) |
| `elliptical:<psi_deg>:<chi_deg>` | orientation angle `psi`, ellipticity angle `chi` in `[-45, 45]` |

**Reference frame**: per emitted ray, `e_ref` = global **+z** projected
transverse to the emission direction (falls back to global **+y** when
the direction is parallel to z), `e_perp = dir × e_ref`; `(Es, Ep)` is the
Jones vector in that `(e_ref, e_perp)` basis (`s_hat = e_ref` at
emission). `linear:<deg>` rotates from `e_ref` toward `e_perp` by `<deg>`.
**Circular handedness** follows the optics/Hecht convention: `right`
means the E-vector rotates clockwise as seen by an observer facing the
**oncoming** beam (`Jones = (1, +i)/sqrt(2)` in the `(e_ref, e_perp)`
basis with the field convention `Re[E exp(-i w t)]`) — verified
numerically in `test_circular_handedness_numeric`. Elliptical uses the
standard orientation/ellipticity-angle parameterization: `E = (cos(psi)
cos(chi) - i sin(psi) sin(chi), sin(psi) cos(chi) + i cos(psi) sin(chi))`.

**`unpolarized` is modeled as two incoherent orthogonal populations**
(pure `e_ref` and pure `e_perp`), not as one 45°-linear Jones vector —
this is exact for polarizer/retarder chains (Malus's law etc.), unlike an
older equal-power-split single-Jones-vector approximation would be. The
gather keys are `(source, lam_stratum, pol_stratum)`, so these two
populations, like wavelength strata, never interfere. Because each
`(source, lambda)` combination's ray budget is split across `n_pol`
polarization strata (`run_trace.py`'s `rays_per_key = rays / (n_strata *
n_pol)`), **an unpolarized source gets half the rays-per-key of an
explicitly polarized source** for the same total `--rays` — raise `--rays`
accordingly if the gather's `M_eff` gate trips (§6.3). Uniaxial
birefringence (§5.6) does not add its own gather-key dimension: an o/e
split inherits the parent ray's existing `pol_stratum` and is tracked
instead via a per-ray `pol_mode`/`n_eff` state, so more rays are needed
for a birefringent scene only insofar as the o/e power split dilutes the
samples already present at each `(source, lambda, pol)` key — there is no
separate fixed multiplier in the code for this.

**Gaussian-beam mode (`beam_waist` + `m2`, source-only, `sources.py`).**
Setting `beam_waist` (mm, required `> 0`) switches position sampling from
uniform-over-the-face to a rejection-sampled 2D Gaussian about the face
center (`_sample_beam_points`): `dx, dy ~ N(0, w0/2)` in the face tangent
frame, redrawing candidates that land outside the physical aperture
(rejection, not clamping, so the truncated-Gaussian shape is exact and
each ray keeps uniform power `P/N` — position density alone carries the
intensity profile `I(r) ~ exp(-2 r^2/w0^2)`). Each ray additionally gets a
per-ray angular divergence: half-angle `theta0 = M2*lambda/(pi*w0)`
(evaluated at the ray's own sampled wavelength) with independent per-axis
tilt `N(0, theta0/2)` — the `/2` (not `/sqrt(2)`) is required so the
near-field position spread and the far-field angular spread jointly
satisfy `w(z)^2 = w0^2 + (theta0 z)^2` at every `z` (documented and
derived in the `sources.py` docstring). `m2` (beam quality factor,
default 1.0, must be `>= 1.0`) scales the diffraction-limited divergence;
`m2=1` is a true TEM00 Gaussian mode. **Requires a planar emitting face**
(`NotImplementedError` on a curved one).

**Apodization (`apodization`, source-only, any face shape).** A
*different* mechanism from `beam_waist`: it reweights each sampled ray's
**power** by a transverse field-amplitude taper `exp(-2 (r/w0)^(2*order))`
(`r` from the face center) and renormalizes so the total still equals the
source's exact `power_mW` — position sampling itself is untouched (so it
composes with any face geometry, not just planar). `order=1` is a plain
Gaussian; `order>1` is a super-Gaussian (flatter core, steeper falloff).
Grammar (`common.parse_apodization_spec`): `'gaussian:w0=<mm>[:order=<n>]'`
(only `kind=gaussian` is implemented; `w0` required `>0`; `order` defaults
to 1, must be `>=1`). A source can carry `beam_waist`/`m2` and
`apodization` simultaneously (independent knobs — position spread from
one, power taper from the other) but this is an uncommon combination.

### 5.2.1 Pulsed sources (`pulse_energy` / `pulse_duration` / `rep_rate`)

Three optional float body properties turn a source pulsed
(`raytracer.scene._parse_pulse_source` resolves them at scene build):

- **`power` XOR `pulse_energy`** — exactly one defines the source's
  average power. `power` (mW) is the classic CW spec; `pulse_energy`
  (µJ) **requires `rep_rate`** (Hz) and derives
  `power = energy × rep_rate`. Both set (nonzero) is a hard error —
  which average-power definition wins would be ambiguous. `power = 0.0`
  is the extractor's "not authored" sentinel for pulse_energy-only
  sources (`common.validate_model` requires the key to exist as a
  float).
- **`pulse_duration`** (ps, Gaussian FWHM). With an energy spec this is
  a real pulsed source; with `power` alone it is a *virtual-pulse*
  annotation (a CW source whose time products render as one τ₀ pulse).
  Either way, whenever any pulse property is present the source dict
  carries a `pulse` block `{energy_J, duration_s, rep_rate_Hz,
  peak_power_W, avg_power_W, derived, kappa}` — echoed per source into
  `case.json` `source_pulse`. `peak_power_W = 0.94·E/τ` (Gaussian shape
  factor); `kappa = P_pk/P_avg` is the peak-to-average ratio that
  scales every nonlinear element's intensity estimate (§6.12).
- **Everything downstream still trades in average watts.** The ledger,
  detector cubes, and closure gate are untouched — per-pulse quantities
  are reporting/NLO scalars, and *one pulse per rep period* is the
  model (no inter-pulse interference).

A pulsed source (nonzero `pulse_duration`) auto-enables the
`pulse,spectrogram` time products (§6.11); `--time-products none`
suppresses that.

**`spm` (source-only, string).** Source-side self-phase-modulation
transform (`sources.install_spm`): `'phimax:<rad>'` or
`'gamma:<W⁻¹km⁻¹>:length:<m>'` (γ from a fiber datasheet;
`φ_max = γ·P_pk·L_eff` from the derived peak power). At scene build the
EXACT pure-SPM spectrum (one 4096-pt FFT of
`E(t) = √I(t)·e^{iφ_max·I(t)/I₀}`) is installed as the source's
tabulated SPD (superseding `lambdamin`/`lambdamax`), and each wavelength
stratum gets a birth-time offset from the analytic instantaneous-
frequency curve evaluated on its central monotonic branch — the
spectrogram shows the SPM S-curve with the physical tilt (leading edge
red, trailing blue). **Honest limits:** a quasi-classical
single-time-per-frequency approximation (the spectral wings recur at
two times; they are folded onto the branch ends); the transform is
source-side only — mid-train SPM would break the stratum bookkeeping
(future.md). Resolved `φ_max` is echoed into `case.json` `source_spm`.

### 5.3 Surface interaction: mirror/absorbance, coatings, polarizers, filters

Documented in `tracer.py`'s module header and dispatched per hit face as
follows (grating faces and birefringent-crystal faces are dedicated paths
that bypass the ordinary optic path entirely — see §5.5/§5.6):

1. **Mirror/absorbance/Fresnel-or-coating split.** `mirror` r is an
   idealized achromatic partial reflector layered **on top of** the
   physical interaction; `absorbance` a eats a fraction of what mirror
   doesn't claim; whatever's left interacts through Fresnel (bare
   interface), a TMM layer stack, or a measured coating table (below). Per
   polarization, the reflected child's amplitude is `sqrt(r + (1-r)(1-a)
   |r_F|^2) * exp(i * arg(r_F))` and the transmitted child's is
   `sqrt((1-r)(1-a)) * t_F` — power-exact, with phase taken from the
   physical (Fresnel/TMM) coefficient. Surface absorption is computed as
   the exact power difference `p_in - p_accounted`, so the energy ledger
   closes by construction. `r=0, a=0` (the defaults) reduce identically to
   plain Fresnel/TMM, phases included. A **detector body's non-detector
   faces** are strict pass-through no-ops (no refraction, no medium
   change) — a detector solid is treated as an ideal thin measurement
   screen.
2. **Coatings** (`coating` body property, whole-body string or a per-face
   map `'Face3=MgF2;Face5=pbs_visible_45'` — the same generic per-face
   grammar as roughness/grating, `common.parse_facemap_spec`: a bare value
   applies to every face, `FaceN=value` entries need `Body.Feature`
   context at extract time, fully-qualified `Body.Feature.FaceN=value`
   passes straight through). Each named coating resolves to **either** a
   TMM layer stack **or** a measured `Rs/Rp/Ts/Tp` table (exactly one of
   `layers`/`table` may be set per `coatings.miecoat` row, §7.3): TMM stacks
   carry real phase from the characteristic-matrix method; tabulated
   coatings carry **no phase** — their amplitude coefficients borrow the
   bare-interface Fresnel phase (a documented approximation: use a TMM
   stack when coating phase matters coherently). Roughness's microfacet
   Fresnel re-evaluation (§5.4) only re-evaluates TMM stacks at the local
   angle; measured coating tables keep their macro (nominal-angle) values
   under scattering.
3. **Polarizer** (`polarizer` + `polarizer_axis` body properties, optics
   only). Applied once per traversal, on the **entry** face only, to the
   already-Fresnel-split transmitted child. `polarizer_axis` (default
   `0,0,1`, body-local, rotated to global by the body's Placement) is
   projected transverse to the ray direction to get the local
   transmission axis; the incident Jones vector is rotated into that
   basis and scaled by `sqrt(T_parallel(lambda))`/`sqrt(T_perpendicular
   (lambda))` from the registry table (§7.4). Rays exactly (anti)parallel
   to the axis (no transverse component) fall back to attenuating both
   channels by the mean of `T_parallel`/`T_perpendicular` ("the film looks
   isotropic edge-on"). A `circular_left`/`circular_right` polarizer
   additionally applies an ideal retarder Jones matrix (fast axis at
   ±45° from the transmission axis, retardance `2*pi*retardance_waves`,
   default a quarter-wave) after the diattenuator stage. The rejected
   power (`p` just before the polarizer minus `p` just after) is credited
   to its own **`polarizer_absorbed`** ledger bucket, not
   `absorbed_surface` — a crossed-polarizer scene parks the great
   majority of emitted power there by design.
   **Orientation caveat:** the stages always apply in GENERATOR order
   (linear diattenuator, then retarder) regardless of which face the ray
   enters, so a circular-polarizer element creates circular light but
   cannot *analyze* handedness (left and right inputs transmit equally).
   To analyze circular light, stack two bodies: a quartz waveplate
   (`crystal_axis` at 45°) followed by a linear polarizer — both fully
   modeled, validated by the `waveplate_quartz` scene.
4. **Bulk spectral filter** (`filter` body property, optics only). At
   scene-build time the filter's internal-transmittance-vs-wavelength
   table (§7.5) is converted to an additive absorption coefficient
   `alpha_per_m(lambda) = -ln(T_internal(lambda)) / d_ref` (`d_ref` = the
   registry's `ref_thickness_mm`); at trace time this `alpha_add` is added
   to the medium's own `4*pi*Im(n)/lambda` bulk absorption before
   `exp(-alpha * seg)` is applied over the **actually traversed** path
   length `seg` — so any thickness (not just `d_ref`) scales correctly
   through ordinary Beer-Lambert law. Loss is credited to
   `absorbed_bulk`, the same bucket ordinary medium absorption uses.

### 5.4 Roughness

Two independent ways to declare roughness, both feeding the same
`sigma_nm[:lcorr_um=]` value grammar (`common.parse_rough_value`, default
`lcorr_um=10.0`):
- **Body property** `roughness`: a plain float (whole-body RMS nm,
  legacy) or a per-face map string, e.g. `'Face1=200:lcorr=5;Face2=50'`
  (schema v2, same generic per-face grammar as coating/grating).
- **CLI** `--rough Body.Feature.FaceN:sigma_nm[:lcorr=um]` (repeatable),
  a separate parser (`common.parse_rough_spec`) that always needs the
  fully-qualified face id. CLI specs win over a body's per-face entries,
  which win over a body's whole-body float.

Either way, roughness attenuates the specular amplitude by `sqrt(A)` with
`A = exp(-(4*pi*sigma*cos_i/lambda)^2)` (Davies/Bennett–Porteus TIS) and
redistributes the remaining `(1-A)` power into Beckmann-microfacet-
scattered lobes (RMS microfacet slope `sqrt(2)*sigma/lcorr`). Scattered
children keep their deterministic OPL phase (so a fixed geometry
realization produces physical speckle; `--seeds N` averages realizations)
but are flagged `scattered=True` so the coherent gather treats their
incoherent pedestal as **real intensity**, not Monte-Carlo noise to
subtract (§6.2).

**`--rough-fresnel {micro,macro}`** (default `micro`) controls how the
Fresnel/TMM coefficients feeding the scattered lobes are evaluated:
`micro` (physical) re-evaluates the coefficients at each microfacet's own
local incidence angle and rotates the Jones vector into that microfacet's
own (s,p) basis before applying per-polarization scattered amplitudes;
legacy `macro` reuses the single nominal-angle (macroscopic specular)
coefficient as a scalar average across the whole lobe, kept only for A/B
comparison against the old behavior.

### 5.4.1 Ground-glass diffusers

The `diffuser` body property (same whole-body-or-per-face grammar as
roughness; value grammar `common.parse_diffuser_value`) declares a
ground-glass surface:

```
'grit:120'      catalog grit number (calibration below)
'slope:0.08'    RMS microfacet slope directly (dimensionless)
'@dg_600'       opticalproperties/diffuser/diffusers.miedif registry row
'Face2=@dg_600' per-face map (put the ground surface on the EXIT face)
```

**Model: the deep-rough limit of §5.4's Beckmann machinery, not a new
scatter path.** The slope resolves to an equivalent (σ = 20 µm,
l_corr = √2·σ/m) roughness entry: σ ≫ λ drives the Davies/Bennett–Porteus
specular retention to exactly 0.0 in float, so every ray scatters through
Beckmann-sampled microfacets with the full `rough_fresnel=micro`
treatment — per-microfacet Fresnel at the local angle, Jones rotation
into each microfacet's own (s,p) basis, TIR suppression, grazing-NaN
power folded into `absorbed_surface` by the exact difference, children
flagged `scattered=True` (the coherent gather treats their pedestal as
real intensity). Depolarization therefore EMERGES from the ensemble of
rotated microfacet bases; there is no ad-hoc depolarizer.

Grit calibration (`roughness.GRIT_FWHM_DEG`, log-log interpolated):
approximate published DG-series transmitted FWHM at 633 nm
(120→12°, 220→9°, 600→5°, 1500→2.5°) inverted through the small-angle
model FWHM ≈ 2√(2 ln 2)·(n−1)·m_rms at n = 1.515. `test_diffuser.py`
pins the traced spot width to this model within 15%.

Honest limits: **single scatter** (one microfacet per interface event —
no shadowing/masking, no multiple scattering, no subsurface transport),
so depolarization is real but weak (measured: a slope-0.17 diffuser
leaks ~4×10⁻⁶ of the incident power through a crossed polarizer, ~m³
scaling); real ground glass depolarizes more. Declaring `diffuser` and
`roughness` on the same face is a ContractError.

### 5.4.2 Measured scatter / BSDF (`scatter`)

The `scatter` body property (same whole-body-or-per-face grammar as
roughness/coating/grating — names only, no value grammar, like `coating`)
declares a **measured, tabulated BRDF** on a face, using the empirical
**ABg model** (`opticalproperties/scatter/bsdf.miebsdf`, §7.9; see
Pfisterer, "Approximated Scatter Models for Stray Light Analysis," *Optics
& Photonics News* (2011), and Harvey's Harvey-Shack formulation, *Opt.
Eng.* 51, 013402 (2012)):

```
'polished_bk7_glass'          whole-body registry name
'Face2=diamond_turned_aluminum' per-face map
```

**Model.** The scattered lobe follows `BSDF(u) = A / (B + u^g)`,
`u` the direction-cosine offset of a scattered ray from the specular
direction (`scatter.py` module header derives the closed-form radial
total-integrated-scatter integral for `g==2`, the case every shipped
registry row uses; `g != 2` falls back to a per-call numeric inverse-CDF,
exercised by tests but unused by any shipped entry). At each hit,
`scatter.abg_tis(A, B, g, cos_i)` gives the fraction of the reflected
power that leaves the specular direction at the ray's own incidence
angle; the specular child keeps amplitude `sqrt(1 - TIS)` and the
remainder is emitted as ABg-lobe-sampled scattered children (flagged
`scattered=True`, same coherent-gather treatment as roughness/diffuser
pedestals, §5.4). An optional per-row `tis_cap` ceilings the computed TIS
(a measured total-scatter number the ABg fit may over-integrate at
grazing incidence) — every registry entry is validated at load time so
`TIS(normal incidence) <= 1` (energy).

**BTDF (transmitted-side scatter, optional).** A registry row may set a
`btdf` flag (plus optional `btdf_A`/`btdf_B`/`btdf_g`/`btdf_tis_cap`,
each defaulting to the reflected value): the **transmitted** child is then
also split into a specular remainder `sqrt(1 - TIS_t)` and an ABg lobe
about the **refracted** direction, where `TIS_t = abg_tis` at the
refracted-angle cosine. Reflected specular + reflected lobes + transmitted
specular + transmitted lobes + absorbed == the parent power exactly
(closure gate). Rows without `btdf` behave exactly as before —
reflected-side only. `lightly_ground_glass_window` demonstrates a
BRDF+BTDF surface.

**Importance sampling (`--importance-scatter`, off by default).** ABg
lobes spread over a hemisphere, so a full-lobe sampler almost never lands
a ray on a small far detector — BRO's "46,000-year" stray-light problem.
With the flag on, each scatter event ALSO emits a child aimed straight at
every detector (or the labels in a body's `scatter_targets` property) by
**importance area sampling**: draw a point `q` uniformly on the detector
face and weight the child `w = BSDF * cos(theta_s) * dOmega * (TIS/INT
BSDF)`, `dOmega = A_face*cos(psi)/r^2` — the exact one-sample next-event
estimator of the power onto that detector (unbiased for any detector
size/obliquity, no cone approximation), so every aimed ray hits (variance
drops ~100-1000x). Closure stays **exact**: the full-lobe remainder
carries the deterministic complement `TIS - sum_c w_c` (aimed rays that
would double-count are rejection-resampled out of the remainder), and
`--importance-limit F` (0<F<=1, default 1) caps the aimed fraction at
`F*TIS` so a full-sphere remainder always survives and `p_accounted`
never exceeds `p_in`. Both BTDF and importance sampling are
**Python-engine features** (tokens `scatter_btdf` / `scatter_importance`
route `--engine auto` to Python).

Honest limits: **single scatter** (no multiple-bounce lobe transport);
the tabulated A/B/g fit is isotropic (a real diamond-turned or ground
surface can be azimuthally anisotropic — out of scope). **Mutually
exclusive with `roughness`/`diffuser` on the same face**
(`Scene.__init__` hard-errors naming the face) — a face is either a
Beckmann-microfacet roughness/diffuser surface or a measured-ABg surface,
never both.

### 5.5 Gratings

`grating` body property, a **per-face map only** — it must name specific
faces (`FaceN=...`), never a whole-body value:
```
'Face2=600:v:orders=-1..1'                        (explicit lamellar spec)
'Face2=600:v:orders=-1..1:eff=0.1,0.8,0.1'         (explicit per-order efficiencies)
'Face2=@vbg_1800'                                  (opticalproperties/grating registry)
'Face2=@vbg_1800:orders=0..1'                      (registry, orders overridden)
```
`groove` is `u`/`v` (the face's own local tangent frame) or an explicit
`x,y,z` direction projected into the local tangent plane. `--grating
Body.Feature.FaceN:<same grammar>` (CLI, repeatable) is the fully
face-qualified equivalent and takes precedence over the body property.
A registry name (`@name`) is resolved against `opticalproperties/
grating/gratings.miegrat` (§7.5) at scene-build time; an unrecognized name is
a hard error naming every loaded registry entry.

Diffraction directions always come from the exact vector grating equation
(`grating.order_directions`), which at order `m=0` reduces algebraically
to ordinary Snell refraction/specular reflection (tested to 1e-12,
independent of groove direction, and conical-mount invariant). Per-order
power comes from one of four models (`spec["model"]`, default
`lamellar`), selected explicitly (`table`/`bragg_kogelnik`/`dammann`) or
via a registry row:

| Model | Description | Polarization | Validated special case |
|---|---|---|---|
| `lamellar` | Idealized binary-phase/duty-cycle grooves (`eta_m` closed form in the groove duty cycle) | scalar (`eta_s == eta_p`) | duty=0.5 reduces to the textbook `eta_{+-1} = 4/pi^2`, `eta_even = 0` |
| `bragg_kogelnik` | Kogelnik (1969) thin-hologram transmission coupled-wave theory (volume Bragg gratings/VBGs); params `thickness_um`, `dn`, `slant_deg` | polarization-resolved (s/TE vs p/TM detuning factor) | exact Bragg incidence (`nu = pi/2`) gives `eta_1 = 1.0`, `nu = pi/4` gives `eta_1 = 0.5` |
| `dammann` | Exact Fourier-order computation of a binary +-pi phase profile from its `transitions` list (fraction of period) | scalar | Parseval (`sum|c_m|^2 = 1`); a single transition at 0.5 reduces exactly to lamellar duty=0.5; a real 1x5 equal-intensity Dammann design matches its published transition points to <0.1% uniformity |
| `table` (v1) | Per-order `eta_s`/`eta_p` interpolated at the ray wavelength from a CSV (§7.5); `cos_i`/azimuth ignored | measured, polarization-resolved (real) | exact linear-interpolation roundtrip; hard error outside the tabulated range; an order absent from the table reads 0 |
| `table` (v2, RCWA) | Per-order **complex amplitudes** `amp_s`/`amp_p` (re/im columns) interpolated multilinearly over `(lambda, theta, phi)`; `|amp|^2` = order efficiency, `arg(amp)` = the coherent diffracted-order phase (§7.5) | measured/RCWA, polarization- and **phase**-resolved | node-exact + analytic-midpoint interpolation; carries real diffracted-order phase into the coherent gather; generated by `scripts/tools/gen_rcwa_table.py` (meent RCWA) |

Every model rotates the incident Jones vector into the grating face's own
local (s,p) interface basis (the same `fr.pol_basis`/`fr.rotate_jones`
machinery the ordinary Fresnel path uses) **before** scaling each
diffracted child's amplitude by the per-order amplitude
(`order_amplitudes`: `sqrt(eta_s)`/`sqrt(eta_p)` — real — for the
analytic/legacy models, or the interpolated **complex** `amp_s`/`amp_p`
for a v2 RCWA table), so children stay power-exact against the incident
field to 1e-12 in the test suite. Analytic/legacy models inherit OPL
continuously with no modeled inter-order phase offset; **v2 RCWA tables
inject the real diffracted-order phase** (`arg(amp)`) on top of the
inherited OPL, so inter-order coherence at a Treacy pair or an
interferometer arm is now physical. Ray differentials (§6.4) are not
tracked through a grating ("order transport unimplemented"); the gather
falls back to the source-referenced patch area for diffracted rays.

**RCWA (rigorous coupled-wave analysis):** per-ray RCWA is out of scope
(an `O(N^3)` eigen-solve per hit — CPU-hours per surface), but the
industry-standard **precompute-and-interpolate** path is supported (what
Zemax/Lumerical do): `gen_rcwa_table.py` runs Lifeng-Li-inverse-rule RCWA
(via **meent** 0.12.0, MIT) over a `(lambda, theta, phi)` grid with
**adaptive refinement near the analytic Rayleigh/Wood anomaly loci** and
ships a v2 complex-amplitude table; the tracer interpolates the complex
components at trace time. `bragg_kogelnik` remains the closed-form model
for thick weakly-modulated near-Bragg VBGs (the meent table agrees with
it to <1% in the SVEA regime, `scripts/tools/rcwa_kogelnik_crosscheck.py`);
it is a thin-hologram transmission approximation only (reflection-geometry
VBGs, which need the tanh/sinh reflection coupled-wave solution, are not
modeled). Interpolated RCWA tables remain **weakest right at the
Rayleigh/Wood anomalies** — adaptive refinement mitigates, never
eliminates. The C engine bakes only the lambda-only real table; a v2 table
emits the `grating_table_v2` feature token and routes to the Python engine.

### 5.6 Uniaxial birefringence (`crystal_axis`)

Setting `material` to a crystal name in `opticalproperties/birefringence/
uniaxial.miebrf` (13 crystals: `calcite`, `quartz`, `sapphire`, `linbo3`,
`litao3`, `yvo4`, `bbo`, `alpha_bbo`, `kdp`, `adp`, `rutile`, `teo2`,
`mgf2` — **not** the underlying `<name>_o`/`<name>_e` `materials.miemat`
row names, which the registry resolves internally) marks a body
birefringent
(`MaterialDB.is_birefringent`). `crystal_axis` (body-local `x,y,z`,
default `+x`, normalized, rotated to global by the Placement) sets the
optic axis. `birefringence.py`'s model (Born & Wolf / Yariv):

- The **ordinary** wave sees an isotropic sphere `|k| = n_o k0` (plain
  Snell with `n_o`).
- The **extraordinary** wave sees an ellipsoidal normal surface: its
  phase index depends on the angle `theta` between the wavevector and the
  optic axis, `1/n(theta)^2 = cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2`
  (`n(0)=n_o`, `n(90deg)=n_e`), and its **ray** (Poynting/group-velocity)
  direction is *not* parallel to the wavevector — it walks off by an angle
  `rho`. Both the normal-surface residual and the o/e tangential-
  wavevector continuity at an interface are validated to 1e-12; the
  **calcite walk-off at 45° internal incidence, 590nm, matches the
  literature value of 6.23° to within 0.05°** (`test_calcite_walkoff_
  45deg`, negative-uniaxial sign convention). Internal reflections are
  mode-preserving (the wavevector reflects specularly; the e-ray
  direction/index is recomputed from the reflected wavevector); exit
  refraction uses the same wavevector tangential-continuity method and
  reduces exactly to ordinary Fresnel refraction for the o-mode.

At an interface the amplitudes are computed **exactly** (default) by
solving the uniaxial boundary-value problem Lekner (*J. Phys.: Condens.
Matter* **3**, 6121, 1991) factors in closed form — continuity of
tangential **E** and **H** across the interface, a 4×4 linear system per
ray (`bir.uniaxial_interface_in` / `_out`, ~Fresnel cost). This gives the
full o/e transmission split **and** the reflected s/p Jones matrix
*including* the cross terms `r_sp`/`r_ps` that the older approximation
dropped — those are odd in the optic-axis azimuth, vanish at 0°/90° and
peak near 45° (cross-checked against Lekner, *JOSA A* **40**, 722, 2023).
Amplitudes are **Poynting-flux normalized** (the e-wave's walk-off makes
S_z differ from a `k_z`-weighting), so the flux-normalized scattering
matrix is unitary and `|R_ss|²+|R_ps|²+T_o+T_e = 1` to 1e-10 for calcite
and quartz (positive and negative birefringence). The exit transmission
into the isotropic medium is likewise exact; the mode-preserving internal
reflection carries the exact remaining power `R = 1 − T` (energy
conservation — the crystal-side incident/reflected fields interfere, so a
self-flux split is not clean). Non-absorbing uniaxial media only; the
transmitted-into-lower-index wave goes evanescent honestly under TIR.
See `birefringence.py`'s header for conventions and the transmitted-e
branch choice; oracles in `test_birefringence.py`.

The legacy **effective-index Fresnel approximation** (`n_o` for the
o-channel, angle-dependent `n(theta)` for the e-channel, cross terms
dropped into `absorbed_surface`) is retained behind **`--biref-approx`**
for A/B comparison and C-engine parity. Its error vs the exact path is
≈0 at azimuth 0°/90° and O(1%) near 45° (quantified in
`test_uniaxial_approx_error_finding`). **Routing:** the exact path is
Python-only (the C engine's `birefk.h` still carries the effective-index
form), so a uniaxial scene emits the unported `biref_exact` feature token
and honestly routes to the Python engine; under `--biref-approx` it emits
the ported `birefringence` token and the C engine (matching Python) is
available. `mirror`/`absorbance` still apply to birefringent bodies;
**coating and roughness are not modeled on a birefringent face** (bare
interface amplitudes are used, with a one-time warning).

o/e rays carry a per-ray `pol_mode` (0=isotropic/ordinary, 1=
extraordinary) and `n_eff` (the e-ray's fixed direction-dependent phase
index, cached at the entry interface); ordinary bulk-OPL accumulation
then just uses `n_eff` in place of the medium's scalar index for e-mode
rays. There is **no dedicated birefringence code in `gather.py`** — by
the time o/e rays reach a detector they are ordinary coherent samples with
an `opl` that already encodes the `n_o`/`n_e` path-length difference, so
waveplate retardance and beam-displacer/Wollaston splitting both emerge
purely from that accumulated OPL and the standard Huygens sum, with no
special-cased retardance formula. **Wollaston/Glan-type prisms** are not a
dedicated code path either — they are built from two ordinary
birefringent bodies cemented with a thin air gap and orthogonal
`crystal_axis` values (`make_test_scenes.py`'s `wollaston`/
`calcite_displacer`/`waveplate_quartz` scenes, §10); the beam-splitting
angle in a Wollaston comes from the `n_o != n_e` index step at the
internal wedge interface, not from walk-off (walk-off vanishes exactly
when the optic axis lies in the interface plane).

Honest limits: absorbing crystals are out of scope (`Im(n_o)`/`Im(n_e)`
are ignored; geometry uses real indices only, and the o-ray's index is
used for bulk absorption of both modes — a documented approximation);
natural optical activity of a gyrotropic uniaxial crystal **is** modeled
on the uniaxial run path (α-quartz carries a cited rotatory-power datum,
`gyration_deg_per_mm = 21.77 @589.3 nm`, in `uniaxial.miebrf`): near the
optic axis, where the o/e indices are degenerate and circular
birefringence is the sole anisotropy, rays route through the isotropic
n_o path and `Tracer._apply_optical_activity` rotates the polarization
plane by ρ·ds per bulk segment (§6.12b) — reciprocal (a retro
double-pass cancels, unlike Faraday rotation) and unitary. A gyrotropic
scene emits the unported `gyration` feature token and honestly routes
`--engine auto` to Python. The **Berreman 4×4** (§5.6b, `berreman.py`,
McClain-1993 frozen `g = G·k̂`) is the exact ORACLE validating that
rotation (`test_berreman.py`) and the formulation that also covers
absorbing/dichroic anisotropic media (complex principal permittivities;
the `.mibiax` registry accepts optional `k_x/k_y/k_z` extinction
columns) — those remain module-level only, never invoked in a scene
trace. Off-axis (&gt;5° from the optic axis) rays keep the exact o/e
Lekner split with gyration neglected — the general gyrotropic Lekner
amplitudes are unmodeled, and there is no continuous ρ(θ) crossover.
Biaxial crystals **are**
now modeled — §5.6b); an e-mode ray that hits a **non-birefringent** face
(e.g. a body nested inside a crystal) is silently downgraded to ordinary-index propagation
with a one-time warning ("documented approximation"); ray differentials
are not tracked through an o/e split (falls back to source-referenced
patch area, like gratings). The closed-form birefringence math
(`test_birefringence.py`, 14 tests) is fully pinned, but the calcite-
displacer/Wollaston/waveplate FreeCAD scenes in `make_test_scenes.py`
have no dedicated end-to-end pytest gate the way the double-slit scene
does (§10/§11) — treat them as manually-validated reference scenes.

### 5.6b Biaxial birefringence (`crystal_axis` + `crystal_axis2`)

Setting `material` to a crystal name in `opticalproperties/birefringence/
biaxial.mibiax` (currently `ktp`, `kta`, `lbo`, `bibo`, §7.7) marks a body
**biaxial** (`MaterialDB.is_biaxial`) instead of uniaxial. A biaxial
crystal has three distinct principal indices `n_x != n_y != n_z` and
**two** optic axes, so a single `crystal_axis` vector can no longer pin
the crystal's orientation: the body needs a **full principal frame**.
`crystal_axis` is the X principal axis and `crystal_axis2` (new,
body-local `x,y,z`, no default) is the Y axis; the `Scene` loader
Gram-Schmidt-orthogonalizes Y against X and derives Z = X × Y
(`body.crystal_frame`, a 3×3 rotation matrix, rows = principal axes in
global coordinates) — a missing `crystal_axis2`, or one (near-)parallel
to `crystal_axis`, is a hard error at scene-build time.

The o/e binary split becomes a **slow/fast two-sheet split** of the
biaxial Fresnel wave-normal equation (`birefringence.py`, Born & Wolf):

```
H(K) = |K|^2 * P - Q + eps_x*eps_y*eps_z = 0
P = sum_i eps_i K_i^2,   Q = sum_i eps_i*(eps_j + eps_k)*K_i^2
```

(`K` the wavevector in k0 units expressed in the crystal principal frame,
`eps_i = n_i^2`). At an interface, substituting the conserved tangential
wavevector `K = t_vec + s*n_hat` turns this into a **quartic in `s`**,
solved batched via companion-matrix eigenvalues (`refract_in_biaxial`);
the (at most) two real inward roots are the **slow** sheet (larger phase
index) and **fast** sheet. D-field eigenvectors for each sheet come from
a closed-form 2×2 symmetric eigenproblem of the inverse-permittivity
tensor projected transverse to `K` (`biaxial_modes_for_k`), robust
everywhere the uniaxial `k_i/(1/n^2 - eps_i)` closed form would hit a 0/0
(the principal planes). The ray (Poynting) direction for each sheet comes
from `grad_K H` (`biaxial_ray_from_k`) — like the uniaxial e-wave, ray and
wavevector differ (walk-off), and because the biaxial ray→k inversion has
**no closed form**, the tracer carries the internal unit wavevector
explicitly in a per-ray `k_dir` field (allocated only inside a biaxial
medium) rather than reconstructing it from the ray direction.

Rays carry `pol_mode` 2 (slow) / 3 (fast) inside a biaxial body (0/1 stay
reserved for isotropic/uniaxial-extraordinary); as with uniaxial e-mode
rays, `n_eff`/OPL bookkeeping is ordinary from the gather's point of view
— there is no dedicated biaxial code in `gather.py`. The **entry** interface
(outside → crystal) amplitudes are the **exact Berreman 4×4** boundary-value
solution by default (`berreman.py`, §7.4-2): the full reflected Jones
including the `r_sp`/`r_ps` cross terms plus the two transmitted-sheet
couplings, flux-normalized so `|amp|²` is the true power fraction. Under
**`--biref-approx`** the legacy **effective-index approximation** is used
instead (each sheet's own `n_phase` fed into isotropic Fresnel formulas per
channel — the same tier the uniaxial path uses under that flag). The **exit**
interface (crystal → outside) still uses the effective-index Fresnel (a
documented follow-on to the entry Berreman-ization). Whatever cross-term or
flux difference either path drops is absorbed into `absorbed_surface` via the
exact power difference, so **energy closure holds by construction on both
paths** (`test_berreman.test_biaxial_e2e_energy_closure_both_paths`, <1e-3).
The finding — the reflected-Jones change vs effective-index — is exact
(machine-zero) at principal alignment / near-normal and grows to O(1%) at
steep oblique off-principal incidence (`test_berreman.
test_biaxial_effective_index_finding`), the biaxial analogue of the uniaxial
azimuth finding. **Routing:** the exact biaxial path emits the `berreman`
feature token (a C-registry seam stub — hard-errors under forced `--engine c`,
routes to Python under `auto`), alongside the always-Python `biaxial` token.
**KTP @ 1064 nm** (Kato & Takaoka 2002) is the validation oracle: the quartic
normal-surface residual, D-eigenvector orthonormality, and the
uniaxial/isotropic degenerate limits are all pinned to <1e-9 relative
(`test_biaxial.py`, 15 tests); the Berreman 4×4 itself reduces to the P6
Lekner uniaxial amplitudes and to isotropic Fresnel to ~1e-15, closes
Poynting flux to 1e-13, and reproduces α-quartz optical activity 21.77 deg/mm
@589.3 nm and 4H-SiC reststrahlen reflectivity (`test_berreman.py`).

Honest limits (mirror the uniaxial ones, plus two biaxial-specific ones):
- **Internal conical refraction is modeled behind `--conical`, off by
  default.** Near an optic axis the two sheets meet and their eigenvectors
  become numerically degenerate. By **default** (`--conical` absent) the
  solver still returns an arbitrary orthonormal transverse pair there — a
  degenerate ray just passes through on an unphysical single direction
  rather than the true internal-conical-refraction cone; those rays are
  tallied into `conical_guard` (`case.json`/`audit.json`) so a scene that
  grazes an optic axis is at least visible in the diagnostics. With
  `--conical`, a ray landing within `--conical-delta` (radians, default
  1e-4) of an optic axis instead fans into `2*--conical-fan` (default 16)
  coherent children — a perturbed two-sheet solve at closely-spaced
  transverse azimuths that reproduces Hamilton's internal cone (opening
  angle `tan A = sqrt((n2^2-n1^2)(n3^2-n2^2))/(n1 n3)`, Born & Wolf, verified
  <1e-8 vs the numeric ray limit) as the perturbation shrinks, including the
  classic φ/2 polarization half-turn around the Poggendorff ring; those rays
  are tallied into `conical_fanned` instead. **External** conical refraction
  (the emergent double-ring cone from a point source outside the crystal)
  is not modeled — only the internal cone the fan reproduces. `--conical`
  is Python-engine-only (`birefringence.py`; biaxial scenes already
  Python-route regardless, §13).
- **Internal reflections are sheet-preserving** (`reflect_internal_
  biaxial`): a slow-sheet ray reflects to another slow-sheet ray (same for
  fast); real cross-sheet mode coupling at an internal reflection is
  ignored (the same tier of approximation as the effective-index
  interface Fresnel above). A ray with no same-sheet returning root (a
  conical-corner case) drops its reflected share into `absorbed_surface`.
- Absorbing crystals (`Im(n_x)`/`Im(n_y)`/`Im(n_z)`), optical activity,
  and χ² nonlinear conversion are out of scope for biaxial crystals —
  unlike uniaxial, which now models near-axis natural optical activity
  (§5.6, §6.12b).
- Coating and roughness/scatter are not modeled on a biaxial face (bare
  Fresnel used instead, one-time warning) — same restriction as uniaxial.

### 5.7 Aspheres (`surface_override`)

`surface_override` (per-face map, same generic grammar as coating/
roughness/grating) declares an analytic asphere on a revolved face in
place of the mesh/canonicalized fallback:
```
surface_override='Face1=asphere:R=25.0;k=-0.6;A4=1.2e-6;A6=...;r_max=10'
```
(`R`, `r_max` in mm; `A4`, `A6`, ... in `mm^-3`, `mm^-5`, ... — contiguous
even orders starting at `A4`). The sag convention is the standard even
asphere: `z(r) = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2)) + sum_i coeffs[i] *
r^(4+2i)`, `c = 1/R`. **At extract time**, `build_asphere_surface()`
recovers the face's revolution vertex/axis from the actual FreeCAD
surface, samples a ~15×14 `(u,v)` grid of real points, and computes the
sag residual between the declared analytic formula and the actual
geometry at every sample; the extractor **dies** (`os._exit(1)`, never
silently trusting a bad declaration) if the maximum residual exceeds
`ASPHERE_TOL_M = 1e-6 m` (**1 micron**), naming the worst sample's radius,
actual sag, declared sag, and residual. A face's declared `k`/`R`
combination that makes the sag formula unphysical (`1 - (1+k) c^2 r^2 <=
0` within `r_max`) also dies immediately.

At trace time the asphere is intersected analytically: `Asphere.intersect
()` (`surfaces.py`) brackets each ray with a cylinder+axial-slab interval,
samples the implicit sag equation to bracket sign changes, then refines
with bracket-guarded bisection followed by Newton iterations (converging
to `|f(t)| < 1e-13`) — not a single closed-form root, since the general
asphere equation with polynomial terms has none. Validated against a
tangent sphere (k=0, no coefficients reduces to the sphere's near-cap
intersection to <1e-10), a closed-form parabola (k=-1), an independent
`scipy.brentq` root for a general polynomial asphere, and finite-
difference normals/shape-operators for every analytic primitive
(<1e-5 relative). The `lens_asphere` test scene (§10) is a
diffraction-limited plano-convex singlet whose conic constant
`k = -n^2` eliminates on-axis spherical aberration for a collimated
beam, with `lens_sphere_control` as its identical-aperture spherical
twin for an RMS-spot-size A/B comparison.

### 5.8 Mesh (non-analytic) faces

A face that cannot be canonicalized to plane/sphere/cylinder/cone/torus/
asphere extracts as `surface.type == "mesh"` (tessellated triangles). By
default the tracer now traces these with a real BVH (median-split
axis-aligned bounding-box tree, batched Möller–Trumbore intersection,
duck-type compatible with the analytic-face `.intersect`/
`normal_out_of_solid` interface) instead of hard-erroring. Normals default
to angle-weighted smoothed vertex normals (barycentric-interpolated
across a triangle); `--mesh-flat-normals` uses the raw per-triangle
winding normal instead. `--strict-analytic` restores the old v1 behavior
of a hard `NotImplementedError` on any mesh-type face.

**Honest limit, stated loudly in code and repeated here**: a tessellated
face carries a sag error on the order of the mesh's linear deflection
(tens of microns for a typical export), which is **far larger than an
optical wavelength**. Coherent optical-path phase through a mesh face is
therefore meaningless — every `MeshFace` warns once at construction.
Mesh faces are for **incoherent power accounting / geometry-limited
faces only** (e.g. a deliberately non-analytic freeform element you only
need to block/redirect rays, not interfere coherently through). Two
narrower cases still hard-error unconditionally regardless of
`--strict-analytic`: a mesh-type **source emitting face**, and a
mesh-type **detector screen face** (detector grids need an analytic
plane, §5.11) — both need a UV parameterization the incoherent/coherent
paths don't have yet. The `mesh_freeform` test scene (§10, a prolate
ellipsoid revolved from a BSpline meridian) exercises exactly this
fallback path end to end.

### 5.8b Surface figure error (`figure_error`) + edge blackening

**Figure error** is the deterministic middle ground between an ideal surface
and statistical micro-roughness (§5.4): a slowly-varying deviation of the real
polished surface from its nominal prescription, decomposed into **Zernike**
terms (Noll indexing, nm-RMS coefficients). It dominates real optical PSFs.

Author it with a `figure_error` body property — a per-face map (or whole-body)
of names in `opticalproperties/figure/figures.miefig`. Each registry row gives
`coeffs` (a `';'`-separated list of `j:rms_nm` terms — the SURFACE sag RMS in
nm at Noll index `j≥2`; a mirror's WAVEFRONT error is 2× this and falls out of
the tracer's OPL) and `r_norm_mm` (the pupil radius the coefficients are
referenced to). At scene build the analytic base surface is wrapped in a
`raytracer.surfaces.PerturbedSurface`, which adds the Zernike sag along the
base normal (nm displacement — the base hit barely moves, but its phase and
its normal, tilted by the transverse sag gradient, both change; the normal
tilt is what refocuses the beam). Validated gates (`test_figure_error.py`): a
λ/4 PV Z4 defocus shifts the reflected focus by the analytic 1/(4a); balanced
Z6 astigmatism makes two orthogonal opposite-sign line foci; the Maréchal
Strehl matches the direct PSF-peak Strehl for small RMS (<λ/14).

**Design decision (binding):** the CAD is the UNPERTURBED shape by design, so
the extractor's <1 µm asphere/prescription verification checks the base surface
against the CAD only — it never sees the figure perturbation. The perturbation
lives entirely in the analytic surface at trace time. Frame derivation is exact
for a flat surface at normal incidence (the validated case) and Asphere/vertex
frames; a curved base uses the standard near-normal thin-figure-error
approximation. `figure_error` routes the run to the **Python engine** (a C port
is later work — the `PerturbedSurface` class and an explicit `figure_error`
token are both absent from `cengine.PORTED`).

**Edge blackening** (`edge_blackened` bool, or the flag on the simple-lens
primitives) blackens a lens's cylindrical barrel for ghost/stray-light
suppression: the extractor marks every CYLINDER face of the body fully
absorbing via a per-face absorbance map (identified by surface TYPE, immune to
the FaceN renumbering a rebuild causes), while the refracting caps stay clear.
Per-face absorbance is Python-engine-routed.

### 5.9 Spreadsheet-driven dimensions + permutation

A `Spreadsheet::Sheet` object drives geometry via aliased cells holding
`=<value> mm` expressions. `extract_geometry.py` echoes **every**
discovered alias/value from the **first** `Spreadsheet::Sheet` object
found in the document into `model.json["spreadsheet"]` (no label
filtering at extract time — `example.FCStd`'s sheet has 17 such aliases:
`lenspos`, `sphered`, `laserpos`, `red`, `green`, etc). `permute_model.py`,
by contrast, looks specifically for a sheet **labeled `dim`** (falling
back to the first `Spreadsheet::Sheet` found, with a warning, if none is
labeled `dim`) and sweeps named aliases on it:

```bash
/home3/freecad/FreeCAD.AppImage -c scripts/permute_model.py -- \
    --model example.FCStd \
    --var lenspos --min -5 --max 5 --n 2 \
    --var sphered --min 20 --max 20 --n 0
```

Value-count semantics (`common.sweep_values`, project-wide law, identical
to the antenna-project convention): `n=0` → `[min]` only (`max` ignored);
`n=1` → `[min, max]`; `n>1` → `n+1` evenly spaced values `min +
i*(max-min)/n` for `i` in `0..n`. The output set is the **cross-product**
across all swept `--var`s, named by chaining `common.variant_name()` once
per variable in `--var` order (decimal point → `p`, minus sign → `m`):
the repo's `basemodels/` directory holds three real examples —
`example-lenspos0-sphered20.FCStd`, `example-lenspos5-sphered20.FCStd`,
`example-lensposm5-sphered20.FCStd` (from a `--var lenspos --min -5 --max
5 --n 2 --var sphered --min 20 --max 20 --n 0` sweep). Feed
`basemodels/*.FCStd` back into `run_pipeline.py --models` as an ordinary
model batch, or drive the whole sweep in one call via `run_pipeline.py
--var/--min/--max/--n` (§8.1), which runs `permute_model.py` once and then
predicts the resulting variant filenames itself
(`variant_output_names()`) rather than parsing stdout.

### 5.10 The air-filler aperture trick

`make_test_scenes.py`'s `doubleslit.FCStd` scene demonstrates the pattern
for any scene with an opaque plate + aperture(s): the slit openings are
not left empty — they are filled with thin `material=air` bodies
(`SlitFill0`/`SlitFill1`). An `n=1 → n=1` crossing is optically a no-op
for the ray's power/direction, but it **re-anchors the coherent-gather
wavefront samples at the aperture plane** (`opl=0` gets reset to the
sample's position at that plane), which is what makes the
diffraction/interference calculation exact — without the filler, Huygens
wavelets would radiate from the *source* plane and simply ignore the
opaque plate's shadowing, producing the wrong pattern downstream. **Use
this trick in any aperture scene** (slits, irises, stops): fill every
open aperture in an opaque element with a thin `material=air` body of the
same aperture shape.

### 5.11 Detector grid basis (recorded in `.h5` attrs)

`DetectorGrid` builds a deterministic in-plane basis per detector face:
`xhat` = the global axis most orthogonal to the face normal, projected
into the plane and normalized; `yhat = normal × xhat`. The pixel
rectangle is the trim polygon's bounding box in that `(xhat, yhat)` frame;
pixels outside the trim are masked (`mask` array in the `.h5`). Every
`detectors/<label>.h5` records `label, H, W, pixel_m, lam_lo_m, lam_hi_m,
xhat, yhat, normal, x_lo, y_lo, seeds` as HDF5 attributes precisely so
that **any downstream analysis maps pixel (row, col) to world coordinates
via the recorded basis, never by assuming a fixed global (x, y)
convention** — `xhat`/`yhat` are scene-dependent (e.g. in the
`doubleslit` scene, `xhat = (0,0,1)` and `yhat = (0,-1,0)`, i.e. the grid's
horizontal axis is global z and vertical axis is global −y, because the
detector screen's own local frame happened to come out that way). See
`post_process.py`'s `render_detector()` and `test_doubleslit_e2e.py` for
worked examples of reading this basis correctly instead of assuming a
convention.

Planar detectors use `DetectorGrid`; a detector body whose recorded face
is a `Sphere` or `Cylinder` instead automatically gets `CurvedDetectorGrid`
(§5.12) — chosen by face surface type, no body-property toggle needed.
`Scene.__init__` still raises `NotImplementedError` if a detector's
recorded face is mesh-type (§5.8) — freeform detector screens remain
unsupported.

### 5.12 Curved detectors (sphere/cylinder, incoherent only)

A detector body whose recorded face canonicalizes to a `Sphere` or
`Cylinder` builds a `CurvedDetectorGrid` (`detector.py`) instead of the
planar `DetectorGrid` — same `run_trace.py`/`post_process.py` pipeline,
no new CLI flag. Pixels are a regular grid in the surface's own canonical
`(u, v)` parameterization (`surfaces.py`'s `to_uv`): for a sphere, `u` =
azimuth, `v` = latitude, per-pixel metric area `R^2 * cos(v) * du * dv`;
for a cylinder, `u` = azimuth, `v` = axial coordinate, per-pixel area
`R * du * dv` (constant). `resolution` sizes pixels along the longer
metric span so pixels stay square-ish regardless of aspect ratio. Power
is splatted (bilinear) into the same `inc` accumulator the planar grid
uses — the *only* difference is `to_grid` mapping world hits through
`surf.to_uv` instead of an in-plane projection — so detected-power tallies
and the energy-audit booking are byte-identical to the planar path;
`post_process.py` divides by the recorded **true metric per-pixel area
map** (`pixel_area_map`, an extra dataset in `detectors/<label>.h5`, only
present for curved detectors so planar `.h5` files stay byte-compatible)
to get irradiance, so total detected power is grid-geometry-independent.
Curved-detector `.h5` files carry extra attrs: `surface_type` (`"sphere"`
or `"cylinder"`), `radius_m`, `u_lo`/`u_hi`/`v_lo`/`v_hi` (the trimmed
parameter range).

**Honest limit: incoherent path only.** `CurvedDetectorGrid.
add_gather_samples()` raises `NotImplementedError` — the coherent Huygens
gather kernel (§6) assumes a flat aperture (obliquity/free-space
propagation math is written for a planar reference); a curved detector
must be fed by `coherent=false` sources only. `--export-rays`/
`--save-fields` still populate a nominal in-plane `xhat`/`yhat`/`normal`
frame at the arc center (for the ray-export meta and spot-diagram
machinery, §8.2) but that frame is **not** used by the splat itself.

### 5.13 Body-bound scattering samples & S(q) structure factors (`sample`)

An optic body's `sample` property (string) names a row in
`opticalproperties/sample/samples.miesamp` (§7.16) and binds a **particle
population to that body's interior**: the body's own `material` is the
HOST medium (the real solvent — controls the Mie contrast/density/OPL
against it), and the body's shape is the containing region
(`raytracer/particles.py`'s `BodyParticleMedium`, `Scene.point_inside_body`
does exact marching containment against the body's own faces). This is the
authoring-time route to a liquid-fill sample cell (a cuvette, vial, or bath
primitive, §6 catalog additions below); the CLI `--particles` world-box
spec (§9) is unchanged and coexists — `sample` is the only route to a
registry-driven **S(q) structure factor** or an **explicit lattice
realization**, neither of which `--particles` exposes.

**Mode** (`mode` column: `auto`/`continuum`/`explicit`, same threshold
convention as §9's hybrid mode) picks between a continuum participating
medium (deterministic-splitting estimator, §6.2) and a frozen discrete
sphere/lattice realization (brute-force collision, exact Mie per hit).
`phi` (mass fraction) XOR `tau` (target Beer-Lambert optical depth along
the body's own AABB x-extent — the resolution basis for a body-bound
cloud, since there is no CLI box length to target) is required, same
semantics as §9.

**S(q)** (`sq_model` ∈ `none`/`py`/`baxter`/`fractal`/`paracrystal`/`table`,
`sq_params` `;`-separated `key:val` pairs — full grammar in §7.16) applies
an inter-particle structure factor on top of the independent-scatterer Mie
physics, via `structure.sq_evaluate()` (§7.16's model list):

- **Continuum mode**: the per-λ scattering-angle PDF becomes
  `P_ens(μ)·S(q(μ))` (`q = 4π·n_host/λ_μm · sqrt((1-μ)/2)`, the metres→1/µm
  conversion happens at the call site); `mu_sca` is corrected by `<S>_p`
  (the phase-function-weighted mean of `S(q)` over the shared angular
  grid), `mu_abs` is untouched, so albedo — and therefore Beer-Lambert
  transmission and the coherent/incoherent child split — stays energy-exact
  **by construction**; azimuth sampling is untouched (`S` is θ-only). This
  is a **decoupling approximation** for a polydisperse population (the
  size distribution and the structure factor are averaged independently,
  not jointly) — an honest limit, not exact liquid-state theory.
- **Explicit/lattice mode**: `sq_model=paracrystal` with `mode=explicit`
  places a REAL fcc/bcc/sc lattice realization (`ExplicitRealization`,
  conventional-cell sites with Gaussian positional jitter `sigma = g·a`,
  clipped to the region including real body interiors) instead of an
  independent-sphere dart-throw — for coherent Bragg/speckle work where
  the actual site correlations (not just the ensemble-averaged S(q))
  matter. `count` in the registry row is overridden to the kept-site count
  (both echoed in diagnostics); a KDTree overlap check warns.

**Honest limits**: continuum-mode S(q) is a decoupling approximation for
polydispersity (above); a continuum-medium scattered child is still
incoherent by construction regardless of S(q) (§6.2's continuum-scattering
note is unchanged — S(q) reshapes *where* power scatters, not whether the
scattered child carries phase); `tau` resolves along the body's own AABB
x-extent, which is only exactly the beam path length for an on-axis body
with the beam along local x (an off-axis or non-rectangular cell's `tau`
is therefore approximate — use `phi` directly for an exact loading in that
case).

**Shape / T-matrix spheroids**: a sample row's `shape` (`sphere`/
`spheroid`) + `aspect_ratio` columns select the per-particle scattering
evaluator via `tmatrix.make_evaluator` — `sphere` is byte-identical to the
ordinary `MieEvaluator` path; `spheroid` builds an orientation-averaged
`TMatrixEvaluator` (pytmatrix, an OPTICS-ENV-ONLY soft dependency —
`ImportError` at first use names the install path, INSTALL.md §3.4) at the
**volume-equivalent-sphere radius** convention. `efficiencies()`/
`amplitudes()` are the only two entry points `particles.py` needs, so
`TMatrixEvaluator` subclasses `MieEvaluator` and inherits its
phase-function/sampler machinery unchanged. **Physics caveat** (verified,
not assumed): pytmatrix's own `orient_averaged_fixed` averages the
scattering amplitude S coherently across orientations — exact for `Qext`
(optical theorem) but measured ~15% LOW for `Qsca` vs independent random
orientations, so `Qsca`/`g`/`|S1|`/`|S2|` are instead derived from the
(correctly bilinear-averaging) phase matrix Z, cross-checked <0.02% against
pytmatrix's own slow `sca_xsect`/`asym` reference integrals.
`amplitudes()` returns `|S1|`,`|S2|` **magnitudes only** (the ensemble
phase is orientation-random, so downstream gather/Mie-amplitude consumers
only ever use magnitudes for a spheroid population). Disk-cached under
`var/cache/tmatrix` keyed on resolved optical params.

**Routing**: continuum-mode `sample` bodies are C-engine-ported
(`sample_body`, §13) — the corrected `mu_ext`/albedo and the
size-averaged S(q)-corrected inverse-CDF table are pre-resolved
Python-side and serialized as plain tables, so the C side needs zero S(q)
or host-material logic. Explicit/lattice-mode sample rows emit the
Python-only `sample_explicit` token regardless of engine.

See `opticalproperties/sample/samples.miesamp` for the 7 shipped rows
(§7.16) and the `cuvette_square`/`cuvette_capillary`/`flow_cell`/
`vial_cylindrical`/`vat_cylindrical`/`sample_region` primitives (§ catalog
list) for ready-made host bodies.

### 5.14 Extended image-emitting sources (`image`, `image_cone_deg`)

A source body's `image` property (string) names a row in
`opticalproperties/image/images.mieimg` (§7.17): a greyscale bitmap
(PNG/JPG/TIFF/BMP or a 2-D `.npy`) that replaces the source's uniform emit
face with a **per-position radiance map** — the authoring path for a test
target (USAF-1951-style resolution chart), a structured illumination
pattern, or any other spatially-varying extended source. `sources.py`
loads the pixel data once at scene build (`load_image_gray`, BT.601 luma
collapse for color inputs; an all-zero bitmap is rejected), builds a Vose
alias table (`_build_alias_table`, O(1) per-ray draw) over the pixel
values, and samples emission positions with **density proportional to
pixel value at EQUAL per-ray power** (in-pixel jitter; row 0 of the bitmap
= the picture's TOP = the maximum-value row, so an image loads
right-side-up) — trim rejection against the emit face's real aperture
still applies on top, exactly like an ordinary source face.

**Emission directions default Lambertian** (`sqrt(1-u1)` cosine-weighted
sampling) — deliberate: an imaging bench needs every object point to fill
the whole aperture, not emit a narrow beam. `image_cone_deg` (0,90],
optional) restricts emission to a uniform-solid-angle cone about the
signed emit normal instead, as a variance-reduction optimization when the
downstream aperture subtends a known small angle (**not** a physical
statement that the object emits into a narrower cone). `image` is
mutually exclusive with `beam`/`apodization` (a hard error at scene build)
and requires a planar emitting face.

**End-to-end products**: `post_process.render_image_traced` (§8.3)
publishes the traced detector image into `imaging/` whenever the source
carried `image` — auto-surfaces in the Results "Imaging" gallery; if
`--image-sim` also ran on the same detector, a traced-vs-convolution-sim
side-by-side PNG plus an NCC (normalized cross-correlation) agreement
metric is added, evaluated at both direct and 180°-rotated orientations
(a real imaging bench inverts the image; the space-invariant sim is
object-oriented) with the better-agreeing orientation reported.

**Honest limits**: source-side only — there is no field-varying PSF
imaging simulation (the existing `--image-sim` convolution stays a single
space-invariant PSF, §6.10); a coherent `image` source warns (an extended
incoherent object is the physically sensible default for this kind of
target).

**Routing**: `image_source` is C-engine-ported (§13) — the request
serializes the same Vose alias table (`sources._build_alias_table`, one
implementation, zero drift) plus the face UV bbox and cone half-angle;
`trace.c`'s `sample_image_pos_dir` alias-draws a pixel, jitters in-pixel,
and emits Lambertian/cone about the signed emit normal, using two reserved
RNG event slots so every other stream stays bit-identical with/without an
image source present.

See `opticalproperties/image/images.mieimg` for the shipped
`usaf_style_target` row and the `source_image` primitive.

---

## 6. Physics engine

Two-part model, in `scripts/raytracer/`:

1. **Geometric propagation** (`tracer.py`, `surfaces.py`, `fresnel.py`,
   `thinfilm.py`, `grating.py`, `roughness.py`, `birefringence.py`,
   `mesh.py`, `particles.py`): rays propagate through analytic surfaces
   (plane/sphere/cylinder/cone/torus/asphere — exact float64
   ray-quadric/Newton intersection, not tessellated geometry, because
   coherent interference needs optical-path accuracy `<< lambda/10`), or
   through tessellated mesh faces via a BVH (incoherent-only, §5.8),
   splitting at each hit into reflected/transmitted/diffracted/o-e
   children per Fresnel, TMM, a measured coating table, the grating
   models, or uniaxial double refraction, attenuating through polarizers
   and bulk filters, scattering at rough surfaces, and colliding with
   particle clouds. Bulk (Beer–Lambert, including any additive filter
   alpha) absorption and phase accumulate continuously along each
   segment. Every child below `power_floor * birth_power` or past
   `max_reflections` (default 6, counts reflections only — transmissions
   don't increment it) is killed with its power credited to the ledger
   (§6.6).
2. **Coherent Huygens–Fresnel "final gather"** (`gather.py`): every
   coherent ray reaching a detector is treated as a wavelet sample taken
   at its **last interaction point**. The complex field at each detector
   pixel is
   ```
   E(p) = sum_i  E_i * sqrt(dA_i) * K_i * exp(i k (opl_i + n_amb r_ip)) / r_ip * (1 / i*lambda)
   ```
   with obliquity `K = clamp(0.5*(cos(theta_prop) + cos(theta_det)), 0, 1)`
   (no back-radiation). Interference (double-slit fringes, focal Airy
   structure) emerges purely from the `k*opl_i` phase differing between
   sample paths. Accumulation is keyed per `(source, wavelength stratum,
   polarization stratum)` — populations that share a key interfere;
   populations in different keys never do.

### 6.1 What is exact

- Analytic surface intersection (plane/sphere/cylinder/cone/torus/
  asphere) in float64 — nanometre-accurate over 0.1 m paths (torus
  intersection is validated against `np.roots` on the same quartic to
  1e-7; the asphere's bracketed Newton solve is validated against a
  tangent-sphere reduction, a closed-form parabola, and an independent
  `scipy.brentq` root, §5.7/§9). SurfaceOfRevolution faces are
  canonicalized to native sphere/cylinder/asphere where geometrically
  possible; anything else falls back to mesh tracing (§5.8) or, with
  `--strict-analytic`, a hard error.
- Fresnel/TMM coefficients (complex index, any polarization, any angle,
  including TIR and absorbing metals) — closed-form validated to 1e-10 to
  1e-12 (§11).
- Uniaxial double refraction (o/e wavevector geometry, walk-off,
  tangential continuity) validated to 1e-12 (calcite walk-off vs
  literature to 0.05°, §5.6); interface amplitudes are the EXACT
  Lekner-1991 boundary solution by default — reduces to Fresnel to 1e-12,
  Poynting-flux closure to 1e-10, azimuth cross-term parity to 1e-14
  (§5.6, `test_birefringence.py`). The legacy effective-index per-channel
  approximation is retained under `--biref-approx`.
- The Kogelnik thin-hologram and Dammann exact-Fourier grating models'
  closed-form special cases (Bragg-peak `eta=1` at `nu=pi/2`; Parseval
  sum; reduction to the lamellar duty=0.5 case) — validated (§5.5/§11).
- Polarizer diattenuation and circular-retarder Jones construction —
  Malus's law, crossed-polarizer extinction, and circular handedness all
  validated against their closed forms (§11).
- Energy bookkeeping: every watt lost is credited to exactly one ledger
  bucket at the moment of loss (`audit.py`), so closure is an invariant of
  the trace loop, gated at 1e-3 relative per source.
- The vector grating equation (any model) and Beckmann roughness
  statistics (closed-form validated, §11).

### 6.2 Approximations (be honest about these)

- **Scalar grating efficiencies (lamellar/Dammann).** Both models are
  polarization-blind (`eta_s == eta_p`); `bragg_kogelnik` and `table` are
  polarization-resolved. Analytic/legacy models carry no relative
  inter-order phase offset beyond the continuously-inherited OPL; **v2
  RCWA tables carry the diffracted-order phase** and are polarization- and
  phase-resolved (§5.5). v2 tables model the diagonal (co-polarized) s/p
  response — **cross-polarization at a conical mount (phi != 0) is booked
  as loss**, not scattered into the orthogonal channel. Per-ray RCWA
  remains out of scope; precompute-and-interpolate RCWA tables are the
  supported path (§5.5).
- **Roughness Fresnel at the microfacet-local angle is the default but
  the legacy macro-angle mode still exists** (`--rough-fresnel macro`),
  which evaluates the reflectance/transmittance feeding the scattered
  lobes at the nominal (specular) incidence angle rather than the true
  microfacet-local angle — kept only for A/B comparison (§5.4).
- **Tabulated (measured) coatings carry no phase.** Their amplitude
  coefficients borrow the bare-interface Fresnel phase; only TMM layer
  stacks carry real coherent coating phase (§5.3).
- **Effective-index Fresnel is the legacy `--biref-approx` path, not the
  default.** By default the uniaxial interface uses the exact Lekner-1991
  amplitudes (entry AND exit, including the r_sp/r_ps cross terms,
  flux-normalized) and the biaxial ENTRY uses the exact Berreman 4×4; the
  biaxial EXIT is still effective-index (documented follow-on, §5.6b).
  Under `--biref-approx` both crystal classes fall back to isotropic
  Fresnel formulas with each channel's own phase index — the dropped
  cross-terms/flux land in `absorbed_surface` so energy still closes
  exactly, but per-channel phase/amplitude off principal alignment is not
  claimed exact on that path. Natural optical activity of a gyrotropic
  uniaxial crystal (quartz's rotary power along its own optic axis) IS
  modeled as a near-axis bulk ρ·ds polarization rotation (§5.6, §6.12b);
  absorbing (dichroic) uniaxial/biaxial crystals are not modeled in a
  scene trace (Berreman-oracle-only); biaxial internal conical refraction
  is modeled as a perturbed-fan approximation behind `--conical` (off by
  default — an arbitrary transverse basis at the degeneracy otherwise),
  and cross-sheet internal-reflection coupling is still not modeled at all
  (a same-sheet-only reflection, on or off the cone) (§5.6b).
- **Measured scatter (`scatter`, §5.4.2) is reflected-side (BRDF) only,
  single-scatter, isotropic.** The transmitted child at a scattering face
  is untouched; multiple-bounce lobe transport and azimuthal anisotropy
  (e.g. a real turned-metal surface's groove pattern) are not modeled.
- **No occlusion test between a gather sample and the detector pixel
  being evaluated, unless `--gather-occlusion` is passed.** By default
  the Huygens sum in `gather.py` propagates every sample to every pixel by
  straight-line free-space propagation with only an obliquity/
  back-radiation cutoff (fine for the bench scenes; the air-filler trick,
  §5.10, handles apertures). With `--gather-occlusion` (§6.5), a
  tile-quantized shadow test is applied instead, itself with its own
  documented approximations.
- **Source-referenced sample-area normalization, unless
  `--ray-differentials` is passed.** By default, exact wavefront patch
  areas are not tracked through refraction; samples instead carry the
  *source*-referenced patch area `A_source / N_rays`, and the finished
  per-(source, stratum) intensity map is renormalized so its integral
  equals the geometrically detected power (`norm_factor_applied` in
  `case.json["gather"]`). `--ray-differentials` (§6.4) tracks exact
  per-sample `dA` instead, wherever the transport is implemented.
- **MC noise floor is reported, not hidden.** The cross-estimator used for
  "smooth" (non-scattered) sample populations is *unbiased*: its
  zero-mean speckle noise can push individual pixels negative. The stored
  spectral cube is **not** clipped (so sums/spectra/profiles stay
  unbiased); only the PNG renderer clips at zero for display.
  `gather_diags["noise_floor_W_per_px"]` (`sqrt(mean(negative_pixels^2))`)
  is written to `case.json` per (detector, source, stratum) as the
  estimated per-pixel MC noise level.
- **Continuum particle scattering is incoherent by construction.** A
  continuum-medium scattered child is flagged non-coherent (random phase)
  — physically correct for a disordered medium, but it means continuum
  particle scattering never contributes fringe structure, only power.
- **Explicit-particle Mie scattering azimuth is polarized by default,
  with a legacy flattening mode.** `--no-pol-scatter` reverts to sampling
  the scattering azimuth uniformly and rescaling to conserve `albedo *
  P_in` exactly (preserves the S1/S2 ratio and phase only on average over
  an ensemble of azimuths, not per-collision); the default instead samples
  azimuth from the true polarized differential cross-section `|S1|^2|Es|^2
  + |S2|^2|Ep|^2` given the incoming Jones vector.
- **Detector screens are planar or curved-incoherent only.** Planar
  screens support the full coherent gather; sphere/cylinder screens
  (`CurvedDetectorGrid`, §5.12) support incoherent power/irradiance only
  — `add_gather_samples()` hard-errors, so a curved detector must be fed
  by `coherent=false` sources. Any other surface type (mesh, freeform) is
  still an unconditional hard error for a detector face.
- **Mesh (non-analytic) faces carry a sag error far larger than a
  wavelength** — coherent phase through a mesh face is meaningless; they
  are for incoherent power accounting only (§5.8). A mesh-type source or
  detector face is still an unconditional hard error.

### 6.3 Sampling gate

With the random-jittered sampling `sources.py` uses (never a regular
grid — a regular grid would silently re-enable coherent aliasing),
undersampling shows up as an incoherent speckle pedestal of relative power
`~ 1/M_eff`, where `M_eff = (sum|a|)^2 / sum|a|^2` is the effective sample
count. `gather.render_coherent()` hard-gates on `M_eff >=
min_eff_samples` (CLI `--min-eff-samples`, default 1000.0, i.e. pedestal
`<= 0.1%`) and raises `GatherError` naming the detector/source/(lambda,
pol)-stratum key and the effective-samples deficit, with a dynamically
computed `--rays` multiplier suggestion, if it trips (`--no-gather-gate`
disables the check, e.g. for smoke-testing at low ray counts). Because
gather keys are now `(source, lam_stratum, pol_stratum)`, an unpolarized
source's two polarization populations and every wavelength stratum each
need their own `M_eff >= 1000` — raise `--rays` accordingly for
polarization-heavy scenes (§5.2).

### 6.4 Ray differentials (`--ray-differentials`)

Off by default (`+96 B/ray` when on, per-ray `dPdx/dDdx/dPdy/dDdy`
float64 (N,3) arrays). Implements the Igehy (1999) ray-differential
method: each ray additionally carries two abstract wavefront-parameter
derivatives (`x`,`y`) of both its position and direction; propagating
those exactly through free-space transfer, reflection, and refraction
(closed-form, cross-checked against finite differences for
sphere/cylinder/asphere to <1e-5 relative) lets the gather compute the
true local wavefront-patch area `dA = |dPdx_perp x dPdy_perp|` per sample
instead of the source-referenced approximation (§6.2) — `case.json`'s
`norm_factor_applied` then stays close to O(1) wherever differentials
survived. Differentials are **lost (set to NaN) whenever a ray takes a
branch whose differential transport isn't implemented**: grating-order
diffraction, roughness/particle scattering lobes, and birefringent o/e
splits all explicitly kill them; the gather then falls back per-sample to
the source-referenced patch area for exactly those rays (mixed batches
carry both kinds of sample simultaneously, tracked by `case.json["gather"]
[...]["n_differential_dA"]`).

### 6.5 Gather occlusion (`--gather-occlusion`)

Off by default (zero overhead, numerically bit-identical to omitting the
flag when off). When enabled, every gather sample→detector-pixel segment
is shadow-tested against the rest of the scene, in two conservative
levels:

1. **Level-1 prefilter**: a provably conservative axis-aligned
   bounding-box test — a face's world AABB is intersected against the
   convex-hull AABB of every possible shadow segment (sample points ∪
   detector-tile centers); a face whose AABB misses that hull cannot
   occlude anything and is dropped before any ray is cast. Only planar,
   polygon-trimmed faces get a cheap AABB; anything else (curved,
   untrimmed) is conservatively kept.
2. **Level-2 tile shadow rays**: the detector is subdivided into `tile ×
   tile` pixel tiles (default 16, down to 1 = a genuine per-pixel shadow
   ray); one shadow ray per surviving `(sample, tile-center)` pair is cast
   through each remaining face via the same `.intersect()` interface
   analytic and mesh faces share, and blocked pairs get their obliquity
   zeroed. The boolean `(n_tiles, M)` visibility mask is built once in
   numpy and shared by both the numpy and torch render kernels, so the
   two backends apply an identical mask.

Documented, deliberate approximations: **tile quantization** (a shadow
edge resolves only to `tile` pixels — use a smaller `tile` for sharp
shadow geometry); **opaque occluders** (any face blocks fully, even a
clear glass lens face — deliberate, because the tracer's own refracted
field already re-anchors as fresh gather samples at the lens exit, so
letting pre-lens samples shine straight through the lens as well would
double-count that path); a detector's own face never occludes itself.
Memory scales as `n_tiles * M` bits per detector key; cost scales with
`n_active_faces * M * n_tiles`.

### 6.6 Backend

`--backend {auto,torch,numpy}`. `auto` uses CUDA via torch if
`torch.cuda.is_available()`, else falls back to numpy; requesting
`torch` explicitly without CUDA available is a hard error. Both backends
compute `r`/phase in float64 and reduce phase mod 2π **before** any
float32 trig (path lengths are `~1e5`–`1e6` waves; float32 phase directly
would inject O(1) rad errors and destroy fringes), then accumulate in
complex64 — validated against a brute-force float64 reference to
`<1e-4` relative and cross-validated numpy-vs-torch to `<5e-3` relative,
including with `--gather-occlusion` enabled (§11).
`common.record_calibration()` logs every completed trace's and gather's
throughput to `results/.calibration.json`, and
`common.calibrated_rate()` (median of matching-kind entries) feeds
subsequent `--dry-run` estimates.

### 6.7 Detector normalization diagnostics, seams, and sub-pixel refinement

Beyond what's in §6.2: a `seam_loss` ledger bucket (§13) catches the rare
case where a ray's tracked medium stack disagrees with the direction it's
crossing an interface — a symptom of two trimmed faces disagreeing at
their shared edge; the ray is killed and its power accounted visibly
rather than silently corrupting the medium stack. Sub-pixel hot-spot
refinement (`gather._hot_pixels`/`_subgrid_factor`) re-samples
geometrically concentrated power (e.g. a tight focal spot) on a finer
sub-grid so a focus falling between pixel centers isn't silently
under-resolved; the applied sub-grid factor per detector is diagnostic
(`case.json["gather"][...]["populations"]["smooth"]["subgrid"]`).

**`--save-fields`** (off by default) additionally writes, per
`(source, lam_stratum, pol_stratum)` key, the complex transverse
field maps `Ex`/`Ey` into `detectors/<label>.h5`'s `fields/<key>/{Ex,Ey}`
groups (seed 0 only; large files at high resolution). `post_process.py`
then renders, per key, a 2×2 Stokes panel `stokes_<label>_<key>.png`
(`S0=|Ex|^2+|Ey|^2`, `S1=|Ex|^2-|Ey|^2`, `S2=2Re(Ex Ey*)`,
`S3=-2Im(Ex Ey*)`), plus a single incoherently-summed degree-of-
polarization map `dop_<label>.png` (`sqrt(S1^2+S2^2+S3^2)/S0`, summed
across every key — valid precisely because different `(source, lam,
pol)` keys never interfere, the same gather contract §6 relies on
throughout). Without `--save-fields` no `fields/` group is written and
both renderers are a silent no-op — `render_stokes_maps()`'s docstring
states the on-disk layout it expects so the reader and the (already-
shipped) writer can evolve independently.

### 6.8 Energy ledger buckets (`audit.py`)

Every watt a ray loses is credited to exactly one of:

| Bucket | Meaning |
|---|---|
| `absorbed_surface` | mirror/absorbance fraction + metal/film (Fresnel/TMM/birefringent-channel) absorption |
| `absorbed_bulk` | Beer–Lambert bulk absorption while traversing a medium, including any additive bulk-filter alpha |
| `particle_absorbed` | Mie albedo losses in a particle cloud |
| `escaped` | left the scene without hitting anything |
| `truncated_generation` | killed by the reflection-generation cap |
| `truncated_power` | killed by the relative power floor |
| `emission_clipped` | source samples whose emission hemisphere was clipped (§5.2) |
| `polarizer_absorbed` | dichroic rejection at a `polarizer` element (§5.3) — a crossed-polarizer scene parks nearly all power here |
| `seam_loss` | killed crossing a face-face seam whose trim tests disagreed (rare; large values indicate broken geometry) |

Power **arriving** at a detector is a per-detector diagnostic
(`detected_geometric`, `by_surface`), **not** a loss bucket — a ray
passing through two detector screens must not be double-counted against
closure. `PowerLedger.report()`'s `closure_error` per source is
`|1 - sum(buckets)/emitted|`, gated at `1e-3` (`closure_ok`).

### 6.9 Multi-process trace sharding (`--workers`)

Off by default in effect (`--workers auto` still means "use it"; pass
`--workers 1` for the historical single-process path). `resolve_workers()`
turns `auto` into `max(1, cpu_count - 2)`; any explicit integer is clamped
to `>= 1`. **`--workers 1` is bit-identical** to the pre-sharding code
path (same RNG stream, same results) — sharding only activates for
`workers > 1` **and** `--rays > 1` (`run_trace._run_sharded`).

With `N > 1`, the total primary-ray budget per source is split as evenly
as possible across `N` spawned processes (`multiprocessing`, `"spawn"`
context — required to keep CUDA state out of forked children), each given
its own child `SeedSequence` (`np.random.SeedSequence(seed).spawn(N)`) so
shards draw independent, non-overlapping RNG streams rather than
re-running the same seed `N` times. Each shard rebuilds the `Scene` from
scratch (cheap; no shared torch/CUDA state crosses the process boundary)
and traces its share of rays through the geometric-propagation loop only;
only worker 0 records viz rays (so the viz-ray budget doesn't multiply by
`N`). The parent process then merges every shard's ray records/ledger
tallies (linear accumulators — trivial to merge) and runs the
**coherent Huygens gather once, single-process, in the parent** (torch-CUDA
when available) — the gather itself is never sharded. Because the RNG
draws differ from the `N=1` path, **`N>1` results are statistically
equivalent to `N=1` (same closure, same expected detector image within MC
noise) but not bit-identical** — `test_workers.py` pins this contract
(merged ledgers/detector cubes agree with single-process runs within
Monte-Carlo tolerance, and `N=1` reproduces the pre-sharding path exactly).
`run_pipeline.py`/`post_process.py` also accept `--workers` for symmetry
with the CLI-introspecting GUI config matrix, but only `run_trace.py`'s
trace-loop stage actually shards.

### 6.10 Named analysis products (PSF/MTF/EE/Zernike/Strehl, spot/ray-fan)

Two independent post-processing families, both driven by opt-in trace
flags and both **seed-0 only** (neither averages across `--seeds`):

- **Field-based (needs `--save-fields`, §6.7):** `analysis_field.py` +
  `post_process.render_field_analysis()` build, per coherent gather key
  `(source, lam_stratum, pol_stratum)` **plus** a synthetic `"all"` row
  (the incoherent sum of every key's PSF — valid because different keys
  never interfere, §6), the **PSF** (the coherent irradiance image itself,
  peak-normalized + log-stretched + a radial profile), the **2D FFT-MTF**
  (`MTF = |FFT(PSF)| / FFT(PSF)|_0`, plus tangential/sagittal 1D slices and
  the `MTF50` frequency), and **encircled energy** (radial cumulative sum,
  reporting the 50/80/90% radii). Headline `report.json` scalars
  (`detectors.<label>.analysis.{psf_peak_W_m2, mtf50_tan/sag_cy_mm,
  ee_r50/r80/r90_um}`) come from the dominant-power **physical** key
  (never the synthetic `"all"` row); every key's numbers (including
  `"all"`) live in `analysis.keys`. Outputs: `analysis/psf_<label>.png`,
  `analysis/mtf_<label>.png`, `analysis/ee_<label>.png` (+ CSVs under
  `--emit-csv`, §8.3).
- **Ray-based (needs `--export-rays`, §8.2/§8.3):** `render_ray_analysis()`
  builds a **spot diagram** (per `(source, lam_stratum)` panel: landing
  scatter about the centroid, RMS + geometric/100% radius) and **ray
  fans** (tangential/sagittal transverse-aberration fans plus a
  chief-referenced **OPD fan**, using each ray's `birth_pos` on the source
  face as its pupil coordinate) into `analysis/spot_<label>.png` /
  `analysis/fan_<label>.png`. `render_wavefront()` then Zernike-fits the
  same per-key OPD (`analysis.py`: Noll-indexed, unit-RMS-normalized over
  the unit disc, `jmax=15`, both Noll and Fringe indices reported) and
  reports the **Maréchal-approximation Strehl** `exp(-(2*pi*sigma/lambda)^2)`
  from the piston/tip/tilt-removed residual RMS, into
  `analysis/wavefront_<label>.png` and `report.json`'s
  `detectors.<label>.wavefront` block (per-key, plus the dominant-power
  key's `strehl_marechal`/`rms_waves`/`pv_waves` promoted to the top
  level). **Pupil model, stated honestly**: this is a **source-referenced**
  pupil (each ray's normalized transverse birth position on the emitting
  face), exact for the collimated/laser benches this tracer models but
  **not a true exit pupil** for finite-conjugate, field-point imaging — a
  real exit-pupil/chief-ray search is future work (`lowhanging.md`). A
  key needs more than `MIN_WAVEFRONT_RAYS = 200` landing rays to get a
  wavefront fit; `--wavefront-point X_MM,Y_MM` overrides the default
  power-weighted landing centroid as the OPD reference point.
  **Future work, explicitly out of scope today**: a PSF-peak-ratio Strehl
  (measured PSF peak vs. a diffraction-limited reference PSF) that would
  let `--save-fields` and `--export-rays` cross-check each other — this
  needs a reference (Airy/aberration-free) PSF model this module does not
  build yet.

### 6.11 Time domain: group delay, time products, and the GDD budget

**The time model.** Whenever any time product (or `--gdd-budget`) is
active, every ray carries two extra accumulators beside `opl`
(`RayBatch.alloc_time`, tracer `cfg.track_time`): `gopl = Σ n_g·ds`
(GROUP optical path — arrival time is `t = gopl/c`; ambient counts
exactly 1.0, so an air path's group delay is its geometric length) and
`gdd_acc = Σ (φ₂/L)·ds` (accumulated material GDD, s²). Directional
group indices for crystal e-rays ride `n_g_eff` exactly like `n_eff`
does for phase. Strictly additive instrumentation: nothing touches
power/phase/ledger/RNG, and detector cubes and `opl` are bit-identical
with tracking on vs off (pinned by `test_time_core.py`).

**Time products** (`--time-products pulse,spectrogram,streak,cube` /
`all` / `none`; a pulsed source auto-enables `pulse,spectrogram`).
Detectors buffer compact **arrival records** (t, fx, fy, λ, power, gdd,
stratum — both the incoherent deposit and the coherent population at
its GEOMETRIC arrival; fringe-resolved timing is documented out of
scope) and bin them once at the end (`finalize_time`):

- `time_profile` (n_t,) [+ `time_profile_by_source`] — I(t), W/s
- `time_spectrogram` (spectral_bins, n_t)
- `time_streak` (n_t, W) — x vs t
- `time_cube` (n_t, H', W') float32, spatially capped at
  `--time-cube-res` (default 256)

All stored as arrival-power **densities** [W/s]: integrating any
product over the window reproduces the in-window detected power
(`time_total_W` attr; `time_excluded_W` books out-of-window power).
`.h5` attrs: `t_lo_s/t_hi_s/time_bins/time_dt_s/time_envelope/
time_products/t_p001_s/t_p999_s` (+ cube binning factors). Window:
auto = exact arrival span padded 3× the widest kernel, or explicit
`--time-window T0,T1` (ns; clipped kernels renormalize over in-window
bins — energy conserving).

**Envelopes.** `--time-envelope analytic` (default): each record splats
a Gaussian of FWHM `sqrt(τ0² + (2√(2ln2)·|gdd_acc|·Δω_stratum)²)` —
zero GDD gives exactly the transform-limited pulse, and the traced
FWHM matches the textbook `τ(φ₂) = τ0·√(1+(4ln2·φ₂/τ0²)²)` at any
`--nlambda` (2% gate, `test_gdd_budget.py`); it also kills MC shot
noise at `quick`. `histogram` is a plain weighted arrival histogram
(assumption-free cross-check; CW/rangefinder mode). Sub-bin kernels
(σ < dt/6) deposit as single-bin deltas — evaluating a fs kernel on
ps bins underflows (NaN via 0·inf; regression-pinned).

**GDD budget** (`--gdd-budget`, or free whenever time tracking runs).
`case.json['gdd_budget']`: per traversed body, the power-weighted mean
bulk path `L̄ = path_tally/flux_in` and its MATERIAL dispersion at the
reference source's λc — n_g, GD, GDD, TOD (crystal bodies use the
o-material; biaxial the principal mean) — plus totals and a per-pulsed-
source broadening annotation τ0 → τ_out at its own λc. post_process
renders `images/gdd_budget.png` + CSV + `report.json`. **Honest
limit:** MATERIAL dispersion only — geometric GDD (gratings, prisms,
angular chirp) shows up in the traced time products instead, and the
e-ray group index neglects the dθ/dλ angular term (calcite oracle
bounds it). The flag on a CW scene forces group-delay tracking (Python
engine, token `gdd_budget`).

### 6.12 χ² / nonlinear elements (SHG, Pockels, saturable, TPA, Kerr)

All intensity-dependent elements share one convention: the local PEAK
intensity `I_pk = (p_ray/dA)·κ_pulse` (`nlo.ray_intensity`) — dA from
ray differentials when `--ray-differentials` is on, else a flat-top
source-area fallback (warned once; the Kerr lens *needs* differentials
to be physical). κ comes from the source's pulse block (§5.2.1); a CW
source has κ = 1.

- **SHG bulk transfer** (`nonlinear` body property naming a
  `chi2_process` registry row). Deterministic per-segment conversion —
  zero RNG use — with the undepleted Boyd plane-wave efficiency
  `η = 8π²d_eff²L²I/(n₁²n₂ε₀cλ₁²)·sinc²(ΔkL/2)`, clamped at 0.5 with a
  warning (pump depletion makes the quadratic growth unphysical past
  that). The row supplies `d_eff` and the design pump wavelength
  (exactly phase-matched there); spectral detuning uses the BODY
  material's scalar index. Parent amplitudes deplete by `√(1−η)`; an
  **incoherent** child at λ/2 with stratum id `n_lambda + parent`
  carries `η·p` — a pure transfer (ledger closes with no new bucket;
  children are never gathered, so coherent budget gates ignore them),
  inherits `gopl`/`gdd_acc` (arrival rides the pump group delay), and
  refracts/Fresnels out through the exit face at λ/2. The detector
  spectral range extends to cover λ/2 automatically;
  `case.json['harmonic_strata']` maps child↔parent ids and
  `shg_converted_W` tallies the transfer per body. **Honest limits:**
  no walk-off, no angular detuning, equal s/p harmonic split, no
  cascaded re-conversion, `chi2_tensor` rows are authoring-side only
  (derive a process row via `nlo.d_eff_tensor`/`phase_match_angle`).
- **Pockels** (`nonlinear` = a `pockels` row + `pockels_voltage`/
  `pockels_gap_mm`): transverse geometry only — Δn = −½n³rE wrapped as
  shifted-index material proxies, so retardance/dispersion/group delay
  all follow from the existing crystal physics.
- **Saturable absorption** (`saturable` = `@row` or inline) and **TPA**
  (`tpa_beta`, cm/GW): intensity-dependent bulk absorption in the same
  Beer-Lambert `alpha_add` hook as spectral filters — energy lands in
  `absorbed_bulk`, closure-free. Evaluated once per segment at entry
  intensity.
- **Kerr lens** (`kerr_n2` = `@n2_row` or a value): thin-element phase
  `Δopl = n₂·I(r)·L` added to coherent rays' `opl` — the gather then
  shows the self-focusing (oracle `f_K = w²/(4n₂I₀L)` at 5%). Coherent
  benches only (warned otherwise).

`saturable`/`tpa`/`kerr` are PORTED to the C engine (P7 tranche 2:
intensity-dependent bulk alpha on the `alpha_add` hook + Kerr opl phase,
per-ray intensity from the ported differentials' dA). Only the χ² token
`nonlinear` (SHG harmonic-child strata + the Pockels index-shift split)
still forces the Python engine (§13).

### 6.12b Natural optical activity (gyrotropic uniaxial crystals)

A uniaxial crystal whose `uniaxial.miebrf` row carries
`gyration_deg_per_mm`/`gyration_ref_nm` (currently α-quartz,
ρ = 21.77 deg/mm @589.3 nm, citation required by the loader) is flagged
gyrotropic (`body.gyration`, `scene.py`). Rays within 5° of the optic
axis (`GYRO_AXIS_COS`) — where n_o = n_e and circular birefringence is
the sole remaining anisotropy — are routed as a single full-Jones
isotropic-n_o child instead of an o/e split, and their polarization
plane is rotated in bulk by δ = ρ·ds per segment
(`Tracer._apply_optical_activity`), a UNITARY Jones step: R+T are
unchanged and energy closure holds with no new ledger bucket. The
rotation sense is right-handed about +k and flips with `sign(k·axis)`,
so a retro double-pass CANCELS — natural activity is reciprocal, unlike
Faraday rotation. The scene-level gate is asserted end-to-end by the
`quartz_rotator` demo (crossed analyzer at 43.5°, measured
sin²(43.5°) = 0.475).

Honest limits: (i) off-axis (&gt;5°) rays keep the exact o/e Lekner split
with gyration neglected — there is no continuous ρ(θ) crossover;
(ii) dispersion of ρ is not modeled (single registry datum at its
reference λ); (iii) the Berreman 4×4 `g = G·k̂` formulation
(`berreman.py`, McClain-1993) is the validating oracle only, never
invoked in a scene trace; (iv) biaxial and absorbing-anisotropic
activity remain out of scene scope. The `gyration` feature token is
deliberately unported — a gyrotropic scene routes `--engine auto` to
Python (§13).

---

## 7. Optical properties library (`opticalproperties/`)

`scripts/raytracer/optprops.py` (loaders) plus `scripts/raytracer/
materials.py` (materials + coatings) load the whole tree in one call,
`optprops.load_optical_properties(root, db)`, used by `run_trace.py`;
individual `load_*` functions are importable directly for tests. Every
registry hard-validates its referenced table CSVs at load time and
requires a non-empty `reference` citation column — the same policy as
`materials.miemat` — and table interpolation **never extrapolates**
(`interp_hard()` raises `MaterialError` outside the tabulated range).
Coatings/polarizers/filters/gratings/birefringence (uniaxial and biaxial)/
scatter are all **optional**: a trimmed library missing any of these CSVs
loads that category as an empty dict rather than failing. `--optical-properties PATH` (run_trace.py/
run_pipeline.py) overrides the library root; there is no per-category
accessor beyond indexing the loaded dicts directly (e.g.
`props.polarizers["lp_test"]`) except for materials, which get
`MaterialDB.get(name)` (case-insensitive, `KeyError` listing every
available name).

### 7.1 `materials.miemat`

One row per material. Columns: `name, class, model, p1..p6, nk_file,
density_kg_m3, transmission_um_min, transmission_um_max, notes,
reference`.

- `class` ∈ `{gas, glass, liquid, polymer, metal, oxide, film, special}`
  (`VALID_CLASSES`) — organizational only, not cross-validated against
  any electrical role (this project has none).
- `model` ∈ `{sellmeier, schott, cauchy, constant, tabulated}` (`VALID_MODELS`):
  - `sellmeier`: `n^2 = 1 + sum_{j=1..3} p_j * lam_um^2 / (lam_um^2 -
    p_{j+3})`; each `C_j = p_{j+3}` need only be **finite** (a negative or
    zero `C` is a well-behaved fit — no real pole; the old `>0` requirement
    was relaxed so genuine catalog fits load). AGF formula code 2.
  - `schott`: legacy power series `n^2 = p1 + p2*lam_um^2 + p3*lam_um^-2 +
    p4*lam_um^-4 + p5*lam_um^-6 + p6*lam_um^-8` (`p1..p6 = a0..a5`), used by
    many older glass-catalog rows. AGF formula code 1.
  - `cauchy`: `n = p1 + p2/lam_um^2 + p3/lam_um^4`.
  - `constant`: `n = p1` (and `k = p2` if given, else 0 unless a
    tabulated `nk_file` supplies k separately).
  - `tabulated`: requires a non-empty `nk_file`; `n` **and** `k` both come
    from the table (log-linear interpolation for `k` when all-positive,
    else linear; `n` linear).
- `density_kg_m3` must be `> 0` **except** `vacuum`/`detector`
  (`ZERO_DENSITY_OK`), which may be 0 (used by the particle-cloud mass-
  fraction math, §9, and as sentinels).
- `transmission_um_min`/`_max` are advisory only for parametric models:
  evaluating `n_complex()` outside that window emits a `UserWarning` (does
  **not** raise); a tabulated material's `nk_file` range, by contrast, is
  a **hard** `MaterialError` if exceeded — no silent extrapolation either
  way.
- `reference` is **required** on every row (`MaterialError` if blank) —
  every material must cite where its optical constants came from.
- **Thermo-optic (optional)**: the columns `thermo_d0,thermo_d1,thermo_d2,
  thermo_e0,thermo_e1,thermo_lambda_tk` (+ optional `thermo_t_ref_c`,
  default 20 °C) carry the Schott TIE-19 model. When a run sets a
  temperature (§ `--temperature`), the real index is shifted by
  `dn_abs(lam,T) = (n^2-1)/(2n) * (D0*dT + D1*dT^2 + D2*dT^3 +
  (E0*dT + E1*dT^2)/(lam_um^2 - lambda_tk^2))`, `dT = T - t_ref`. Absent
  (or `T` unset / `T == t_ref`) → no shift, so the columns are fully
  backward-compatible. Populated in bulk by `scripts/tools/import_agf.py`
  from a catalog's AGF `TD` line.
- `opticalproperties/nk/*.mienk` tables (`wavelength_nm, n, k`, strictly
  increasing wavelength, 18 total) back the `tabulated` model: metals
  (aluminum, chromium, copper, gold, nickel, platinum, silver, titanium,
  tungsten), semiconductors (gaas, germanium, silicon, sic), IR materials
  (kbr, nacl), and water. **Birefringent crystal materials are not selected directly** —
  a body sets `material=calcite` (the crystal name in `uniaxial.miebrf`, §7.6),
  which resolves to the `calcite_o`/`calcite_e` pair internally.

849 materials ship today: a 170-row hand-curated core (24 originals + the
`library-expansion` round + `decalin`/`dye_solution_kmno4` from the
samples-instruments round, §5.13's sample-cell host/index-match media)
plus 679 Schott + Ohara optical glasses imported from
the vendor Zemax AGF catalogs by `scripts/tools/import_agf.py` (formula code
1→`schott`, 2→`sellmeier`; unsupported formulas skipped, never approximated),
carrying Schott TIE-19 dn/dT where the catalog provides it. The curated core:
optical glasses (41 Schott/Ohara crowns/flints), metals/semiconductors/IR windows
(17), polymers/liquids/gases/biological (35), coating-film materials (5), crystals
with o/e pairs (46 uniaxial axis rows), plus foundational `vacuum`, `air`, `bk7`,
`fused_silica`, `sapphire_o/e`, `water`, `glass`, `polystyrene`, `latex`, `pmma`,
`polycarbonate`, `tio2`, `mgf2`, `sio2_film`, `detector`, `calcite`, `quartz`,
`sf5`, and `fiber_core_na22`. Every row carries a required `reference`; the
imported glasses' provenance is `library_data/agf/` (with a preservation guardrail,
`scripts/tools/verify_miemat_preserved.py`, proving the import never altered a
pre-existing row). Curated entries are spot-checked against authoritative sources
(NIST, peer-reviewed publications, manufacturer datasheets) per §7.10.

**To add a material**: append a row with a unique `name`; pick `class`
descriptively; pick `model` and supply the required parameters for it
(Sellmeier's three `C` values need only be finite; tabulated needs `nk_file`);
set `density_kg_m3 > 0`; fill `reference` (required); optionally note a
**spot-check** in `notes` — the existing rows follow the pattern "verified
n(lambda)=X vs target Y, matches within Z" (e.g. `calcite_o`'s row:
"verified n(590nm)=1.65830 vs target 1.658"). This is the project's
**spot-check policy**: every parametric fit should be validated against
at least one literature/catalog reference point, and that check should be
pinned in `scripts/raytracer/tests/test_materials.py` (22 tests spot-
check bk7, fused_silica, mgf2, water — including its ~975nm absorption
shoulder — sapphire, polystyrene, aluminum's visible-range n/k and its
monotonic-into-IR k rise, plus hard-error behavior for out-of-range
tabulated lookups, unknown materials, malformed CSV rows, and missing
references).

### 7.2 `coating/coatings.miecoat` (+ `coating/tables/*.mietab`)

Columns `name, layers, table, aoi_deg, reference` — **exactly one** of
`layers`/`table` must be set per row. `layers` is a `;`-separated ordered
stack from the incident side toward the substrate,
`material:thickness_spec`, where `thickness_spec` is either a literal
nanometre thickness (e.g. `100.0`) or `qw@<lam0_nm>` (quarter-wave at that
design wavelength, resolved dispersively at trace time via `n(lambda0)` —
**not** baked in once, because a dispersive coating's quarter-wave
thickness genuinely depends on which wavelength you design for). `table`
names a measured `wavelength_nm,Rs,Rp,Ts,Tp` CSV in `coating/tables/`
(hard-validated: strictly increasing wavelength, values in [0,1], and
**`Rs+Ts<=1`, `Rp+Tp<=1` per row**). 39 coatings ship (expanded from 10):
TMM stacks (AR: MgF2 at 550/633/1064nm, quarter-quarter V-coats at 532/633/1064nm,
3-layer QHQ W-coats at 550nm; HR: dielectric 11–15 layer stacks at 532/633/1064nm),
measured table models (protected mirrors, dichroic/laser elements, standard
45°-AOI elements). TMM stacks apply via `thinfilm.py`'s Macleod-formulation
characteristic-matrix method (any number of layers, any angle, complex indices);
zero layers reduces identically to bare Fresnel (tested to 1e-12). See §5.3
for the phase caveat on measured tables.

### 7.3 `polarizer/polarizers.miepol` (+ `polarizer/tables/*.mietab`)

Columns `name, type, table_csv, retardance_waves, reference`, `type` ∈
`{linear, circular_left, circular_right}`. Tables are
`wavelength_nm, T_parallel, T_perpendicular` (power-transmission
fractions), hard-validated to both lie in `(0,1]` and
**`T_perpendicular < T_parallel` everywhere** ("otherwise the
transmission axis is mislabeled"); `retardance_waves` defaults to `0.25`
(quarter-wave) if blank. 17 polarizers ship (expanded from 5): linear sheet
polarizers (Glan-Thompson, Glan-Taylor, Polaroid HN22/HN38, Moxtek visible,
Moxtek KRS-5 IR wire-grids), circular polarizers at 488/633/780 nm, plus
validation-test `ideal_linear` and `thorlabs_*` measured references.
See §5.3 for how the diattenuator/retarder Jones stage is built from these
tables.

### 7.4 `filter/filters.miefilt` (+ `filter/tables/*.mietab`)

Columns `name, table_csv, ref_thickness_mm, reference`. Tables are
`wavelength_nm, transmittance_internal`, values must lie in `(0,1]` (a
floor like `1e-6` is required instead of an exact `0` in stopbands). 56 filters
ship (expanded from 3): Schott colored-glass series (OG/RG/BG/GG/KG/UG/NG families,
31 entries verified against datasheets), interference bandpass filters (6 entries),
and foundational heat-absorbing/bandpass examples. See §5.3 for the Beer-Lambert
thickness-scaling math.

### 7.5 `grating/gratings.miegrat` (+ `grating/tables/*.mietab`)

Columns `name, model, lines_per_mm, params, table_csv, reference`,
`model` ∈ `{lamellar, bragg_kogelnik, dammann, table}`. `params` is a
`;`-separated `key=value` field (e.g. `thickness_um=3000;dn=0.0005;
slant_deg=0` for `bragg_kogelnik`, `transitions=0.03863,0.39084` for
`dammann` — validated strictly increasing, each in `(0,1)`). 9 gratings ship
(expanded from 3): lamellar ruled gratings (1200 l/mm first entry exercising
the `lamellar` model), Bragg/VPH (volume Bragg grating, VPH Kogelnik, ESO)
Dammann diffractive optics, transmission gratings, echelle, and ruled
blazed gratings.

`table`-model entries reference a `.mietab` in `grating/tables/`; two
on-disk formats are supported, distinguished by the **first line**:

- **v1 (legacy)** — no marker; header `wavelength_nm, order, eta_s, eta_p`.
  REAL per-order power efficiencies, interpolated on wavelength only
  (`cos_i`/azimuth ignored). Validated so summed per-wavelength,
  per-polarization order efficiencies never exceed 1. Existing tables and
  external all-`.csv` libraries keep working unchanged.
- **v2 (RCWA)** — first line `# mietab grating v2 [side=transmission|
  reflection]`; header `wavelength_nm, theta_deg, phi_deg, order, amp_s_re,
  amp_s_im, amp_p_re, amp_p_im`. COMPLEX per-order amplitudes on a full
  regular `(lambda, theta, phi)` grid (`|amp|^2` = co-polarized order
  efficiency, `arg(amp)` = diffracted-order phase). Interpolated multi-
  linearly over the complex components (the Zemax/Lumerical approach,
  sidestepping phase unwrapping). Loader validates the grid is complete and
  the co-polarized order sum `<= 1`. Generate with
  `scripts/tools/gen_rcwa_table.py` (meent RCWA + adaptive Wood-anomaly
  refinement). `rcwa_fs_600_v2` ships as a 600 l/mm binary fused-silica
  transmission example.

See §5.5 for the model formulas and the RCWA-table contract.

### 7.6 `birefringence/uniaxial.miebrf`

Columns `name, n_o_material, n_e_material, reference, notes` plus the
OPTIONAL natural-optical-activity triple `gyration_deg_per_mm,
gyration_ref_nm, gyration_reference` (P9). 13 uniaxial
crystals ship (expanded from 3): calcite, quartz, sapphire, plus LiNbO3,
LiTaO3, YVO4, β-BBO, α-BBO, KDP, ADP, rutile TiO2, TeO2, MgF2 (e-ray
addition enabling waveplate/Rochon prism designs), each pointing at a
`materials.miemat` o/e pair (§7.1) with spot-checked birefringence values.
When `gyration_deg_per_mm` is present the row carries the measured rotatory
power (deg/mm) at `gyration_ref_nm` and REQUIRES a `gyration_reference`
citation; absent → non-gyrotropic (backward compatible). α-quartz ships the
datum (ρ = 21.77 deg/mm at 589.3 nm) that drives the scene-level activity
model of §5.6/§6.12b. See §5.6 for the physics model.

### 7.7 `birefringence/biaxial.mibiax`

Columns `name, n_x_material, n_y_material, n_z_material, reference,
notes`. 4 biaxial crystals ship: `ktp`, `kta`, `lbo`, `bibo`, each
pointing at three `materials.miemat` principal-index rows (`<name>_nx`/
`_ny`/`_nz`, §7.1) — e.g. `ktp` resolves to `ktp_nx`/`ktp_ny`/`ktp_nz`.
All four are spot-checked HIGH-confidence rows (Kato & Takaoka 2002 for
KTP; see `library.md` for the other three's citations). A body sets
`material=ktp` plus **both** `crystal_axis` (X) and `crystal_axis2` (Y)
to use one — see §5.6b for the physics model and the principal-frame
contract.

### 7.8 `detector/detectors.miedet` (+ `detector/tables/*.mietab`)

Columns `name, table_csv, reference, notes`. Tables are `wavelength_nm, qe`
(quantum efficiency, values in (0,1]). 4 detector QE curves ship:
`hamamatsu_s1223` (Si PIN photodiode, 4 datasheet points, 660–960 nm),
and three samples-instruments-round linear-CCD rows —
`toshiba_tcd1304ap`/`sony_ilx511b` (relative response digitized from
datasheet plots, an assumed 55% peak absolute scale, flagged) and
`hamamatsu_s3904` (absolute A/W from the 2024 datasheet, QE derived
honestly via `QE=R·1240/λ_nm`, high confidence) — the same
`toshiba_tcd1304ap` curve backs `tcd1304_array`'s `diode_array`
instrument row (§7.11). When a detector body carries a
`qe_curve` property naming a registered curve, `post_process.py` reports a
`qe` block per detector: `photocurrent_A` (R(λ)=QE·qλ/hc weighting of the
spectral cube), `qe_weighted_power_W`, and `coverage_frac` (the fraction of
detected power whose bin centers lie inside the QE table's range — QE is
zero-filled outside the table rather than extrapolated, and coverage_frac
makes that truncation visible). No CLI flag; it is driven entirely by the
body property. The separate `--photometric` flag produces lux maps using the
CIE Ȳ=V(λ) table in `raytracer/detector.py`.

### 7.9 `scatter/bsdf.miebsdf` (+ measured ABg surfaces)

Columns `name, model, A, B, g, tis_cap, reference, notes` plus the
OPTIONAL transmitted-side (BTDF) block `btdf, btdf_A, btdf_B, btdf_g,
btdf_tis_cap`. `model` is
always `abg` today (`SCATTER_MODELS = ("abg",)`). `A`, `B`, `g` must each
be `> 0`; `tis_cap` is optional and, if given, must lie in `(0, 1]`. When
the `btdf` flag column is truthy (`1`/`true`/`yes`/`on`) the transmitted
child is ALSO split into a specular remainder + a scattered lobe about the
REFRACTED direction using the `btdf_*` ABg triple (each `btdf_*` field
defaults to the corresponding reflected `A`/`B`/`g` when left blank; a row
with no `btdf` column — or a falsey one — is reflected-side only, exactly
as before). Every
row is validated at load time: `scatter.abg_tis(A, B, g, cos_i=1)` (the
widest, normal-incidence total-integrated-scatter integral) must not
exceed 1 — a fit that would scatter more power than it receives is a
load-time `MaterialError`, not a silent energy leak discovered at trace
time. 4 surfaces ship: `polished_fused_silica` (TIS ~0.09% near normal),
`polished_bk7_glass` (TIS ~2.2%, standard 60-40 scratch-dig polish),
`diamond_turned_aluminum` (raw TIS ~11.6%, `tis_cap=0.1` pins the split to
a plausible measured 10% total scatter), and `lightly_ground_glass_window`
(a combined BRDF+BTDF demo surface — a transmitted scatter lobe with
`btdf_tis_cap=0.3`) — all rows are flagged
**UNVERIFIED** in `notes` (representative ABg fits per Pfisterer 2011's
form, not transcribed from a specific vendor/published measurement); spot
this before citing a scatter result as measured. See §5.4.2 for the
physics model and the `roughness`/`diffuser` mutual-exclusion contract.

### 7.10 Citation policy

Every row in every registry above requires a non-empty `reference`
column (`MaterialError` if blank, enforced identically in `materials.py`
and `optprops.py`). Where a real vendor datasheet or peer-reviewed
dispersion equation exists, `reference` cites it directly (e.g.
`schott_kg3`'s UQG datasheet PDF, Ghosh 1999 for calcite/quartz,
Johnson & Christy 1972 for gold/silver). Where a vendor's interactive
curve graph could not be machine-fetched (most coating/polarizer/filter
tables modeling a specific commercial part), the `reference` column says
so explicitly and states which numbers *are* real (peak transmission,
extinction ratio, cut-on wavelength, band-average R/T, etc., taken from
the product page's stated specs) versus which curve *shape* is an
engineering approximation anchored to those numbers rather than a pixel
digitization. Treat any such row as "physically plausible, not
vendor-exact" unless its `reference` says otherwise.

### 7.11 `instrument/instruments.mieinst` (+ `instrument/tables/*.mietab`) — virtual instrument layer

A **post-process layer over an ideal detector plane** (never a tracer
change): a detector body carrying an `instrument` property (`'row'` or
`'row:mode'`, mode `ideal`\|`full`, default `full` — §5.1) gets its
already-computed spectral cube read through the named
`instruments.mieinst` row by `post_process.py`'s `render_instrument`
dispatcher, and a `report.json` `detectors.<label>.instrument` block is
added (`row`, `class`, `mode`, plus per-class fields below). `ideal` mode
is deterministic response only (QE/responsivity weighting, integration,
resolution convolution, stray-light floor, full-well saturation
clipping, bit-depth/ADC quantization — no rng draw, no `seed` key
reported); `full` mode additionally draws a reproducible noise chain
seeded from `sha256(case_seed | row_name | detector_label)` (recorded as
`seed` in the block, so a run's instrument view can be reproduced
bit-for-bit from `case.json`'s own seed). `--instruments {on,off}`
(default `on`) is a run-time override — `off` skips the layer even when
assigned; there is no CLI way to render it on a detector that carries no
`instrument` property, matching `qe_curve`'s posture.

One wide CSV header (`name, class, reference, notes` + every class's own
columns, sparse per row — the same shape as `nonlinear/nonlinear.mienlo`)
covers four shipped classes and three schema-defined **placeholder**
classes with no shipped rows (`polarimeter`, `wavefront_sensor`,
`autocorrelator` — gear the owner does not have yet; their columns are
still hard-validated the moment a row is authored):

| class | key columns | outputs |
|---|---|---|
| `camera` | `pixel_pitch_um, width_px, height_px, fill_factor, qe_table, full_well_e, read_noise_e, dark_current_e_per_s, bit_depth, adc_gain_e_per_dn, integration_time_s_default` | `<case>/instrument/instr_<label>_camera_<mode>.png` + `..._counts.npy` (a full-well-clipped, bit-depth-quantized counts image on the SAME (H,W) grid as the ideal detector plane — see `detector.spectral_cube_to_electrons`'s docstring for why the counts image is not resampled to the camera's native pixel count); report fields `integration_time_s, saturation_fraction, mean_counts, max_counts, snr_estimate` |
| `powermeter` | EXACTLY ONE of `responsivity_table` / `flat_responsivity_a_w`, `aperture_mm, nep_w_per_sqrthz, bandwidth_hz, display_digits` | report fields `power_reported_W, power_reported_display` (sig-fig rounded), `lam_ref_nm` (the cube's own power-weighted mean wavelength — exact for a monochromatic source), `responsivity_a_w_at_ref` |
| `spectrometer` | `lam_lo_nm, lam_hi_nm, resolution_fwhm_nm, slit_um, stray_light_floor, detector_qe_table` | `<case>/spectra/instrument_<label>_spectrum_<mode>.png` + `.csv` under `--emit-csv` (Gaussian-convolved to `resolution_fwhm_nm`, QE-weighted, stray-light floor added, clipped to `[lam_lo_nm, lam_hi_nm]`); report fields `lam_lo_nm, lam_hi_nm, resolution_fwhm_nm, peak_power_W, stray_light_floor` |
| `diode_array` (samples-instruments) | `pixel_pitch_um, pixel_height_um, n_px` (physical array geometry) + the `camera`-style electron chain (`full_well_e, read_noise_e, bit_depth, adc_gain_e_per_dn, integration_time_s_default`) + the `spectrometer`-style `stray_light_floor, detector_qe_table` (no `lam_lo_nm`/`lam_hi_nm`/`resolution_fwhm_nm`/`slit_um` — the array's own `n_px*pixel_pitch_um` extent and real pixel geometry replace them) | `render_diode_array` (§8.3): bins the ideal detector-plane cube onto the array's REAL `n_px`-pixel geometry (dispersion axis auto-picked via the same `_axis_wavelength_fit` the spectrometer centroid uses, `pixel_height_um` integrated across the perpendicular axis) instead of `spectrometer`'s continuous resolution-convolved curve — `<case>/instrument/instr_<label>_diode_array_<mode>.png` + `.csv`, QE-exact per-pixel electrons, the same ideal/full noise split (Poisson shot + read noise) and full-well-clip + ADC-DN chain as `camera` |

Four starter rows ship, each citing a real datasheet:
`camera_generic` (Sony IMX264 Pregius global-shutter CMOS, 3.45 µm,
2448×2048, per FLIR Blackfly S BFS-U3-51S5M-C's technical reference),
`powermeter_generic` (Thorlabs S130C Si photodiode sensor, the sensor
behind the PM16-130 power meter), `spectrometer_generic` (Ocean
Optics/Ocean Insight USB4000, Toshiba TCD1304AP CCD, VIS-NIR
configuration), `tcd1304_array` (`diode_array`: the SAME Toshiba
TCD1304DG/AP CCD as `spectrometer_generic`, but read as a real 8 µm ×
200 µm 3648-pixel linear array rather than a continuous curve — the
`--reference-case`-paired absorbance workflow below is validated against
this row). GUI: a detector's instrument assignment lands beside
`qe_curve` in its property panel; the Results pane's "Instrument" tab
flattens the report block into a table above a thumbnail gallery of
`<case>/instrument/*.png` (the spectrometer's own PNG stays in the
existing Spectra tab, prefixed `instrument_`, rather than duplicating
into a second gallery).

**Absorbance** (`--reference-case DIR`, §8.1/§8.3): given a blank
(reference) case directory using the SAME instrument row, `post_process.py`
recomputes that case's raw spectrometer/diode-array product from its own
`.h5` and renders `A(λ) = -log10(I/I0)` against the current case
(`instrument/absorbance_<label>.csv` + PNG) — a hard `SystemExit` if the
reference case's instrument row or pixel grid doesn't match. Pairs
naturally with a `dye_solution_kmno4`-filled `cuvette_square` (§5.13) and
`tcd1304_array`/`spectrometer_generic` for a UV-Vis-style absorbance bench.

### 7.12 `emission/emitters.miesrc` (+ `emission/tables/*.mietab`)

Columns `name, kind, table_csv, params, lines, reference, notes`; `kind` ∈
`{continuous, blackbody, lines}` (`EMISSION_KINDS` — samples-instruments
round: `blackbody` and `lines` were staged/rejected in earlier rounds,
both now load). A source body's `spectrum` property (§5.1/§5.2) names any
row regardless of kind; the tabulated/synthesized SPD supersedes
`lambdamin`/`lambdamax` in every case.

- **`continuous`**: `table_csv` names a per-emitter table
  `wavelength_nm, relative_power` giving the RELATIVE spectral power
  density of a source's emission — only the SHAPE matters
  (`sources.wavelength_strata` normalizes it to a PDF and places
  equal-power quantile strata). Validated: `relative_power >= 0`
  everywhere, positive integral, `>= 2` rows.
- **`blackbody`**: `params` = `'temp_k:<K>;lam_lo_nm:<lo>;lam_hi_nm:<hi>'`.
  The Planck curve (`optprops.planck_relative_power`, peak-normalized;
  Wien's-law-verified) is synthesized to a dense table **AT LOAD TIME**
  over `[lam_lo_nm, lam_hi_nm]`, so every downstream consumer sees exactly
  the same tabulated-SPD shape a `continuous` row would produce — no
  separate code path anywhere else in the pipeline.
- **`lines`**: `lines` = `'wavelength_nm:intensity;...'` (strictly
  increasing, unique wavelengths, intensities relative — shape only, `>= 1`
  line), optional `params` = `'linewidth_nm:<w>'` (floored at
  `MIN_LINEWIDTH_NM` so every stratum keeps a finite wavelength edge for
  `stratum_domega`/time-product bookkeeping; adjacent line bands must not
  overlap once widened — a hard error naming the colliding pair). A `lines`
  row has NO `lam_um`/`relative_power` keys, so a consumer that mistakenly
  treats a line source as continuous fails loudly instead of
  mis-sampling. `sources.wavelength_strata`'s `lines` regime places strata
  AT the line centers using Hamilton largest-remainder apportionment
  (proportional to line intensity, `>= 1` stratum per kept line) and warns
  explicitly when `n_lambda < n_lines` (proportions become
  unrepresentable below one stratum per line).

5 emitters ship: `led_white_2733k` (phosphor-converted white LED SPD,
CCT ~2733 K, CIE 015:2018 illuminant LED-B1, `continuous`), `sc_superk`
(NKT SuperK EXTREME supercontinuum SPD, 400–2400 nm, `continuous`),
`bb_halogen_3000k` (`blackbody`, tungsten-halogen 3000 K approximation),
`d2_uv_approx` (`continuous`, tabulated deuterium-lamp UV continuum
approximation — the Balmer-series line structure is explicitly flagged
omitted, not modeled as `lines`), `hg_penlamp` (`lines`, an 11-line
NIST-cited mercury pen-lamp spectrum, the classic UV-Vis wavelength
calibration source).

### 7.13 `figure/figures.miefig`

Zernike SURFACE-figure-error registry (P8). Columns `name, coeffs,
r_norm_mm, reference, notes`. `coeffs` is a `;`-separated set of Noll
`j:rms_nm` terms (SURFACE sag RMS in nm — a mirror's WAVEFRONT error is 2×
this and falls out of the traced OPL; piston `j=1` is rejected as a
meaningless constant offset). `r_norm_mm` (`> 0`) is the pupil radius the
coefficients are referenced to. Applied at scene-build time as a
`raytracer.surfaces.PerturbedSurface` sag perturbation over the transverse
pupil (the CAD carries the UNPERTURBED shape by design, so the <1 µm
asphere gate checks base-vs-CAD only; Python-engine-routed, §13). 4 sets
ship: `fig_lambda4_defocus_633`, `fig_astig_633`, `fig_trefoil_633`,
`fig_lambda10_typical`. Named by a body's `figure_error` property
(§5.1/§5.9).

### 7.14 `diffuser/diffusers.miedif`

Ground-glass diffuser registry. Columns `name, grit, slope_rms,
reference`. Each row supplies EITHER a catalog `grit` number (mapped to an
RMS microfacet slope by `roughness.slope_for_grit` at scene build) or an
explicit `slope_rms` in `(0, 1)`; if BOTH are present they must agree with
the mapping to within 20 % (a mislabeled row fails loudly at load). 4
diffusers ship: `dg_120`, `dg_220`, `dg_600`, `dg_1500` (approximate
Thorlabs DG-series grits, ~12/9/5/2.5° FWHM at 633 nm). Referenced by a
body's `diffuser` property as `@dg_600` (§5.1/§5.4.1).

### 7.15 `nonlinear/nonlinear.mienlo` (pulsed-optics NLO registry)

One wide sparse CSV (the same shape as `instruments.mieinst`, §7.11) whose
leading `kind` column discriminates row types; every row also carries a
required `reference` and optional `notes`, and full-line `#` comments are
allowed and skipped. 14 rows ship across five kinds:

| `kind` | key columns | use |
|---|---|---|
| `chi2_tensor` | `crystal, point_group, d_il_pm_V` ((3,6) `d`-tensor), `kleinman`, `lam_ref_nm` | authoring-only reference tensors (5 rows: linbo3/bbo/ktp/lbo/kdp) — a hard error if set as a body's `nonlinear` |
| `chi2_process` | `crystal, process, lam_pump_nm, theta_deg, phi_deg, d_eff_pm_V` | phase-matched SHG per-segment transfer (2 rows: ktp type-II @1064, bbo type-I @800) |
| `pockels` | `crystal, r_coeffs_pm_V, geometry` | transverse EO index shift via `pockels_voltage`/`pockels_gap_mm` (2 rows: kdp Q-switch, linbo3 EO) |
| `n2` | `material, n2_m2_W, lam_ref_nm` | Kerr `kerr_n2` (`@row`) self-phase index (4 rows: fused_silica/sapphire/yag/bk7) |
| `saturable` | `I_sat_W_cm2, T0, tau_recovery_s, alpha0_per_mm` (optional) | saturable-absorber bulk α (1 row: sam_1550_16_2ps SESAM) |

`crystal` names on `chi2_*`/`pockels` rows are cross-checked against the
uniaxial/biaxial registries when loaded via `load_optical_properties`;
`n2` `material` is resolved LAZILY at Kerr use time (staged rows may
precede their `materials.miemat` index row). See §5.1's NLO extras and
§6.12/§6.12b for the physics; `nonlinear`/`pockels` are Python-routed
(§13).

### 7.16 `sample/samples.miesamp` (+ `sample/tables/*.mietab`)

The scattering-sample registry (samples-instruments round) named by a
body's `sample` property (§5.13). 16 columns: `name, particle_material,
dist, median_um, gsd, phi, tau, mode, count, sq_model, sq_params, shape,
aspect_ratio, solvent_visc_pas, reference, notes`.

- `particle_material` must exist in `materials.miemat`; `dist` (`mono` or
  a log-normal-over-radius kind), `median_um` (median DIAMETER, the same
  `--particles` convention, §9), `gsd` (defaults 1.6, forced to 1.0 for
  `dist=mono`).
- `phi` (mass fraction, `(0,1)`) XOR `tau` (target optical depth, `> 0`) —
  exactly one required, same semantics as §9/§5.13.
- `mode` ∈ `auto`/`continuum`/`explicit`; `count` (optional int `> 0`)
  pins an explicit-mode site count directly instead of deriving one from
  `phi`/`tau`.
- `sq_model` ∈ `none`/`py`/`baxter`/`fractal`/`paracrystal`/`table`
  (`SQ_MODELS`); `sq_params` is `;`-separated `key:val`, validated per
  model (§5.13's model list gives the physics; the params each needs):
  `py`/`baxter` take optional `phi_hs`/`r_hs_um` (default: derived from
  the sample's own `phi`/median radius at trace time) plus `baxter`'s
  required `tau_stick > 0`; `fractal` requires `xi_um > 0`, `df` in
  `(1, 3]`, optional `r0_um`; `paracrystal` requires `lattice` ∈
  `fcc`/`bcc`/`sc`, `a_um > 0`, `g` in `(0, 1)`; `table` requires
  `table:<name>` resolving to `sample/tables/<name>.mietab`
  (`q_per_um, s` columns, `q` strictly increasing `>= 0`, `s > 0`).
- `shape` ∈ `sphere`/`spheroid` (default `sphere`); `aspect_ratio`
  (default 1.0, must be `1.0` when `shape=sphere` — a nonzero value there
  is a hard error naming the fix) selects the T-matrix path (§5.13) when
  `shape=spheroid`.
- `solvent_visc_pas` (optional): the host solvent's dynamic viscosity
  (Pa·s), consumed by `run_dls.py`'s Stokes-Einstein diffusion
  coefficient (§8.7) — required for a sample used in a DLS run, unused
  otherwise.

7 rows ship: `latex_100nm_water` (the DLS standard reference sample —
100 nm polystyrene latex spheres in water), `glass_beads_10um_water`
(Insitec/ISO 13320-class calibration beads), `hard_sphere_py` (Percus-
Yevick S(q), Pusey & van Megen's colloidal hard-sphere system),
`sticky_sphere_baxter` (Baxter S(q)), `silica_gel_fractal` (Teixeira
fractal, Schaefer/Teixeira `df=2.1` silica gel), `colloidal_crystal_fcc`
(explicit fcc lattice, a synthetic opal — (111) Bragg peak ~537 nm in
water), `spheroid_hematite` (T-matrix prolate spheroid, aspect 1.8).

### 7.17 `image/images.mieimg`

The extended image-source registry (samples-instruments round) named by a
source body's `image` property (§5.14). Columns `name, file, reference,
notes` — `file` names a bitmap stored NEXT TO the registry CSV (extensions
`.png`/`.jpg`/`.jpeg`/`.tif`/`.tiff`/`.bmp`/`.npy`, `IMAGE_EXTENSIONS`);
files live inside `opticalproperties/` so a `.MieWB` project library
carries its targets with it. The loader validates existence + extension
and the `reference` citation contract only — pixel data is loaded once by
the engine at scene build (§5.14), not here.

1 row ships: `usaf_style_target` (512×512 8-bit, MIL-STD-150A-style-alike
3-bar resolution groups, generated by `scripts/tools/gen_usaf_target.py` —
NOT a licensed reproduction of the real USAF-1951 chart; bright-emits
convention with an asymmetric top-left orientation mark for end-to-end
flip/orientation tests). Used by the `source_image` primitive.

---

## 8. Command reference

Every flag below is read from each script's actual `parse_args()`, not
from memory.

### 8.1 `run_pipeline.py` (system `python3`)

```
python3 scripts/run_pipeline.py --models FCSTD [FCSTD ...]
    [--preset {quick,normal,detailed}] [--tag TAG]
    [--steps extract,trace,post,viz]
    [--var ALIAS --min F --max F --n N]...   (repeatable group, paired positionally)
    [--sweep-mode {product,zip}]
    [--dry-run] [--resume] [--extend RAYS] [--seeds N] [--rays F]
    [--resolution N] [--nlambda N]
    [--spectral-bins N] [--max-reflections N]
    [--viz-rays N] [--viz-density F] [--viz-pattern SPEC]
    [--backend {auto,torch,numpy}] [--engine {auto,python,c}]
    [--importance-aim] [--importance-scatter] [--importance-limit FRAC]
    [--rough-fresnel {micro,macro}] [--biref-approx]
    [--ray-differentials] [--gather-occlusion] [--gather-exact] [--gather-nufft]
    [--no-pol-scatter] [--mesh-flat-normals] [--temperature DEG_C]
    [--save-fields]
    [--save-fields-detectors LABEL[,LABEL...]] [--strict-analytic]
    [--optical-properties PATH]
    [--source-face SPEC]... [--detector-face SPEC]...
    [--grating SPEC]... [--rough SPEC]... [--particles SPEC]
    [--particle-threshold F] [--suppress-body NAME]...
    [--conical] [--conical-fan N] [--conical-delta RAD]
    [--photometric] [--spectrometer] [--instruments {on,off}]
    [--ring-profile SPEC] [--reference-case DIR]
    [--time-products LIST] [--time-bins N] [--time-window T0,T1]
    [--time-cube-res N] [--time-envelope {analytic,histogram}]
    [--gdd-budget]
    [--dim-rays {off,linear,sqrt}] [--dim-rays-floor PCT]
    [--emit-csv] [--export-rays] [--export-rays-max N] [--ghost-analysis]
    [--pol-transport] [--wavefront-point X,Y]
    [--wavefront-pupil {source,exit_pupil}] [--imaging-products LIST]
    [--image-sim PATH] [--image-sim-coherence {incoherent,coherent,partial}]
    [--image-sim-sigma F]
    [--viz-generations N] [--views v1,v2,...] [--smoke]
    [--keep-going] [--print-only] [--workers N]
```

**Time-domain flags** (forwarded to **trace**; §6.11): `--time-products`
is a comma list of `pulse,spectrogram,streak,cube` (`all`/`none`) —
default `pulse,spectrogram` when the scene has a pulsed source, nothing
otherwise, `none` suppresses the auto-rule. `--time-bins` is
preset-scaled (quick 128 / normal 256 / detailed 512,
`cli_specs.TIME_BINS_PRESET`) when not given. `--time-window` is in ns.
Any active product tracks per-ray group delay and forces the Python
engine. `--gdd-budget` emits the per-element dispersion table (§6.11)
into `case.json`/`report.json` + `images/gdd_budget.png`; on a CW scene
it forces group-delay tracking (Python engine).

**Engine / sweep / variance-reduction flags** (forwarded to **trace**
except `--sweep-mode`): `--engine {auto,python,c}` selects the trace
engine (default `auto` = C when the binary exists and every scene feature
is ported, else Python; §13). `--sweep-mode {product,zip}` sets how
multiple `--var` groups combine (`product` cartesian default, `zip`
advances them together; `common.sweep_combos`, §5.9). `--resume` restarts
an interrupted C-engine trace from its checkpoint and `--extend RAYS`
raises a COMPLETED C-engine case to `RAYS` and continues (both C-engine
only; the stateful numpy RNG makes them refuse on the Python engine, §14).
`--importance-aim` birth-culls source samples that would immediately
escape (unbiased, C engine); `--importance-scatter` aims measured-scatter
(ABg) children at the detectors' solid angles with `--importance-limit
FRAC` the bias/variance knob (default 1.0; §5.4.2). `--biref-approx`
selects the legacy isotropic effective-index Fresnel at uniaxial
interfaces instead of the default exact Lekner amplitudes (§6.1).
`--conical` (default off) turns on the internal-conical-refraction fan at
biaxial optic axes; `--conical-fan N` (default 16) and `--conical-delta
RAD` (default `1e-4`) are its azimuth count and optic-axis dispatch
radius (§5.6b). `--ring-profile SPEC` (forwarded to **post**) integrates
each detector image into log-spaced annular power bins about the optical
axis (`'n=32:rmin_mm=0.05:rmax_mm=10[:center=peak|chief|X,Y]'`,
laser-diffraction-sizer style; §5.13/§8.3). `--reference-case DIR`
(forwarded to **post**) renders `A(λ) = -log10(I/I0)` absorbance against
a blank case's matching spectrometer/diode-array product (§7.11/§8.3).
`--gather-exact` forces the bit-exact fp64 gather kernel and
`--gather-nufft` opts into the EXPERIMENTAL NUFFT angular-spectrum fast
path (both C engine only; §12). `--viz-pattern SPEC` lays out viz rays
deterministically (`rings:dr=<mm>:nper=<N>[:nrings=<K>]` or `fan[:n=<K>]`
— visualization only, physics unaffected). `--pol-transport` (forwarded
to **trace**, implies `--export-rays`, seed 0 only, Python engine)
parallel-transports each ray's geometric frame + cumulative Jones matrix
so **post** renders honest per-detector retardance/diattenuation/fast-axis
maps. `--wavefront-pupil {source,exit_pupil}` (forwarded to **post**)
picks render_wavefront's pupil model (default `source`; `exit_pupil` runs
the chief-ray/exit-pupil search, falling back with a note when it
degenerates). `--imaging-products LIST` (forwarded to **post**, requires
`--export-rays` and a multi-source field fan) renders
`distortion,vignetting,field_curves,telecentricity` (or `all`) analysis
PNGs + `report.json` `imaging` blocks.

`--models` accepts globs and multiple files (bare names also resolve
under `basemodels/`). `--steps` always executes in the fixed canonical
order `extract,trace,post,viz` regardless of the order you list them.
`--var`/`--min`/`--max`/`--n` must each appear the same number of times;
if given, `permute_model.py` runs once first and its predicted output
filenames (§5.9) replace the input model list for the rest of the batch.
Physics options are forwarded to the **trace** stage only, verbatim
(including every new flag above); `--preset` fills
`rays`/`resolution`/`nlambda`/`spectral-bins`/`viz-rays` from
`common.PRESETS` unless explicitly overridden. `--dry-run` makes trace
build estimates only (`case.json["status"]` stays `"estimated"`), which
generically skips post/viz for that model with a NOTICE (gated on
`case.json["status"] == "completed"`, so this also covers any other
reason trace didn't finish, not just `--dry-run`). `--photometric`
generates per-detector lux maps (CIE photometric, using the Ȳ=V(λ) luminosity
function) and reports `photometric.{luminous_flux_lm, peak_illuminance_lux,
mean_illuminance_lux}` in `report.json`. `--spectrometer` generates
wavelength-centroid and λ-vs-position profile maps and reports
`spectrometer.{lambda_min_nm, lambda_max_nm, dispersion_nm_per_mm, fit_r2}`
(wavelength resolution quantized by `--spectral-bins`). `--instruments`
(default `on`) is an override on the data-driven virtual instrument layer
(§7.11) — `off` skips it even when a detector body carries an `instrument`
property (a fast-preview escape hatch); there is no way to turn it *on*
from the CLI beyond assigning the property, same posture as `qe_curve`.
`--emit-csv`
(forwarded to **post**) writes `results/<case>/data/*.csv` beside every
PNG chart plus a `data/index.csv` (§8.3). `--export-rays`/
`--export-rays-max` (forwarded to **trace**) capture per-detector ray
landing records into `rays_full.npz` (§8.2), which **post** then turns
into spot diagrams, ray/OPD fans, and (with enough coherent rays per key)
a Zernike/Strehl wavefront fit under `results/<case>/analysis/` (§6.10).
`--ghost-analysis` (forwarded to **trace**, implies `--export-rays`'s
behavior and its own `--export-rays-max` forwarding) additionally tracks
each ray's face-id reflection history so **post** can rank multi-bounce
stray-light ("ghost") paths by detected power (§8.2/§8.3). `--wavefront-
point X_MM,Y_MM` (forwarded to **post**) overrides the wavefront fit's
default power-weighted-centroid image point. `--image-sim PATH`
(forwarded to **post**; §5.2's imaging-analysis round) convolves an input
greyscale image with the amplitude PSF taken from the dominant coherent
gather key's saved detector field — REQUIRES `--save-fields` and a
coherent source; `--image-sim-coherence {incoherent,coherent,partial}`
picks the illumination model (default `incoherent`, intensities convolve
with `|h|^2`) and `--image-sim-sigma F` sets the partial-coherence
source-radius factor (default 0.5). `--save-fields-detectors
LABEL[,LABEL...]` (forwarded to **trace**; no effect without
`--save-fields`) restricts the Ex/Ey field-map writes to those detector
labels instead of every detector — an unknown label is a hard error
naming the scene's available ones, and it forces the Python engine (the
C engine doesn't support a per-detector subset yet; see §8.2/§13).
`--viz-generations N` (forwarded to **post**) declutters `rays_xy.png`
to reconstructed-generation `<= N` segments only. `--views v1,v2,...`
and `--smoke` (forwarded to **viz**) pick a view subset / the fast
`overview3d`-at-800×600 smoke render — `--resolution`/`--out`/
`--skip-vtkexport` remain reachable only by invoking `make_viz.py`
directly (§4.2). `--workers N` (forwarded to **trace**; default `auto`)
shards the trace loop across `N` spawned processes (§6.9) — `1`
reproduces the exact single-process path. `--keep-going` turns a stage
failure into a `FAILED: <tag>` notice and a skip to the next model
(process still exits nonzero if anything failed). `--print-only` composes
and prints every stage command **without running anything**. Extract runs
**once** for the whole model batch (one FreeCAD launch handles every
model); trace/post/viz then loop **sequentially** per model (a single
trace can already saturate every core/GPU). Logs: `results/log.extract`
(batch), `results/log.permute-<stem>` (if swept),
`results/<stem>/<case>/log.{trace,post,viz}` (per model/stage).

### 8.2 `run_trace.py` (optics env python — the solver)

```
/home3/optics/env/bin/python scripts/run_trace.py
    --model-json PATH --case-dir DIR
    [--rays F=1e5] [--nlambda N=5] [--resolution N=512] [--spectral-bins N=16]
    [--max-reflections N=6] [--power-floor F=1e-4]
    [--seeds N=1] [--seed0 N=42] [--backend {auto,torch,numpy}=auto]
    [--engine {auto,python,c}=auto]
    [--importance-aim] [--importance-scatter] [--importance-limit FRAC=1.0]
    [--workers N=auto]
    [--viz-rays N] [--viz-density F=1.0] [--viz-rays-max N=20000]
    [--viz-pattern SPEC]
    [--ray-differentials] [--no-pol-scatter] [--rough-fresnel {micro,macro}=micro]
    [--biref-approx]
    [--conical] [--conical-fan N=16] [--conical-delta RAD=1e-4]
    [--temperature DEG_C]
    [--source-face SPEC]... [--detector-face SPEC]...
    [--grating SPEC]... [--rough SPEC]...
    [--particles SPEC] [--particle-threshold F=2e5]
    [--suppress-body NAME]...
    [--min-eff-samples F=1000.0] [--no-gather-gate]
    [--save-fields] [--save-fields-detectors LABEL[,LABEL...]]
    [--gather-occlusion] [--gather-exact] [--gather-nufft]
    [--optical-properties PATH]
    [--strict-analytic] [--mesh-flat-normals] [--dry-run]
    [--resume] [--extend RAYS]
    [--time-products LIST] [--time-bins N=256] [--time-window T0,T1]
    [--time-cube-res N=256] [--time-envelope {analytic,histogram}=analytic]
    [--gdd-budget]
    [--export-rays] [--export-rays-max N=2000000] [--ghost-analysis]
    [--pol-transport]
```

`--rays` is primary rays **per source**. `--seeds N` re-traces with
`N` different seeds (`seed0 + s`) and writes both the per-seed mean and
(if `N>1`) the std-dev spectral cube per detector; useful for estimating
Monte-Carlo/speckle noise directly rather than relying only on the
reported noise floor. `--viz-rays` is an absolute per-source viz-ray cap
that overrides the default density-driven budget: `--viz-density`
(default 1.0 rays/mm² of source emit area, visualization only — physics
is unaffected) capped at `--viz-rays-max` (default 20000) per source.
`--ray-differentials`, `--no-pol-scatter`, `--rough-fresnel`,
`--save-fields`, `--gather-occlusion`, `--strict-analytic`,
`--mesh-flat-normals` are documented in §6/§5; all default off (or
`micro` for `--rough-fresnel`) except where noted. `--conical`/
`--conical-fan`/`--conical-delta` (§5.6b) default off/16/`1e-4`. `--temperature
DEG_C` shifts glasses carrying a thermo-optic model via Schott TIE-19
dn/dT (§5.1, §7.1); default is each material's reference temperature
(no shift), and a per-body `temperature` property overrides the scene-
global value. It forces the Python engine. `--save-fields-
detectors LABEL[,LABEL...]` restricts `--save-fields`' complex Ex/Ey
field-map writes (`detectors/<label>.h5`'s `fields/` groups) to the named
detector labels (comma-separated face ids, e.g. `Body001.Pad.Face3`,
matching `--detector-face`/`DetectorGrid.label`); omitted or empty means
every detector (bare `--save-fields`' pre-existing behavior, unchanged),
and an unknown label is a hard error naming the scene's available ones —
never a silent no-op. It has no effect without `--save-fields`, and
forces the Python engine when set together with `--save-fields` (the C
engine always writes fields for every detector; §13). `--particle-threshold`
default `2e5` is deliberately aligned with `ExplicitRealization.
MAX_BRUTE` (§9) so the default can never land in the dead zone between
"explicit mode selected" and "over the brute-force cap." `--optical-
properties PATH` overrides the `opticalproperties/` library root (§7).
`--suppress-body NAME` (name or label) treats a body as if it were
`ignored`, without editing the `.FCStd`. `--dry-run` writes `case.json`
with `estimates` and `status="estimated"`, then stops (no trace/gather/
detector output). Exit code is `0` on closure-OK, `3` if any source's
energy closure gate failed (still writes all outputs; this is a
warning-level failure, not a crash).

`--workers N` (default `auto` = `max(1, cpu_count-2)`; `1` = the exact
pre-sharding single-process path) shards the trace loop across spawned
processes; the coherent gather always runs single-process in the parent
(§6.9). `--export-rays` captures seed-0 per-detector ray landing records
(`pos, dir, opl, lam, source_id, lam_stratum, pol_stratum, generation,
pol_mode, power, scattered, coherent, birth_pos`) into `results/<case>/
rays_full.npz`, namespaced `<safe_label>/<field>` per detector plus a JSON
`meta` array (grid basis, seed, cap, kept fraction, model name);
`--export-rays-max` (default 2000000) caps the per-detector count above
which a uniform-random subset (seeded by `--seed0`) is kept, with the kept
fraction recorded in the meta. **Diagnostic only** — the splat/gather math
and the `rays.npy` viz contract are completely untouched by either flag.
`--ghost-analysis` additionally allocates `RayBatch.refl_hist` (an `(N,
HIST_DEPTH=8)` int32 face-id history, one slot per reflection generation,
generation `>= HIST_DEPTH-1` reusing the last slot) and implies
`--export-rays`'s behavior (a bare `--export-rays` does **not** track
history) — the npz then also carries `<safe_label>/refl_hist` per detector
and a global `face_labels` list (face index → `"<element>.<FaceN>"`) so
`post_process.py` can map a reflection-path signature to element names
without reconstructing the scene. Both flags are **seed 0 only**, like
`--save-fields`.

### 8.3 `post_process.py` / `compare_runs.py` / `make_viz.py` (rendering)

```
/home3/optics/env/bin/python scripts/post_process.py \
    --case-dir DIR --model-json PATH [--viz-generations N]
    [--dim-rays {off,linear,sqrt}] [--dim-rays-floor PCT]
    [--photometric] [--spectrometer] [--instruments {on,off}]
    [--ring-profile SPEC] [--reference-case DIR]
    [--emit-csv] [--wavefront-point X_MM,Y_MM]
    [--wavefront-pupil {source,exit_pupil}] [--imaging-products LIST]
    [--image-sim PATH] [--image-sim-coherence {incoherent,coherent,partial}]
    [--image-sim-sigma F]
```
Requires `case.json["status"] == "completed"`; fully rerunnable without
re-tracing (reads only `case.json`/`audit.json`/`rays.npy`/
`detectors/*.h5`/`rays_full.npz` (if present) + `model.json`).
`--viz-generations N` declutters `plots/rays_xy.png` to
reconstructed-generation `<= N` segments only (default: every generation,
unchanged behavior) — useful for scenes with many reflection/diffraction/
o-e-split generations where the 2D plot would otherwise be an unreadable
tangle. `--dim-rays` switches `rays_xy.png`'s segment alpha from the
default ensemble 95th-percentile scaling to each segment's `rel_power`
(power relative to its own ray's power at the source — linear, or sqrt
for a perceptual curve; `--dim-rays-floor` sets a minimum opacity
percent); falls back with a warning on a 10-column `rays.npy` predating
the `rel_power` column. Unconditionally also renders (when the
relevant body properties are present in the model) `polarizer_<name>.png`/
`filter_<name>.png`/`grating_<name>.png` per-element response-curve plots
and (when `--save-fields` produced `fields/` groups, §6.7)
`stokes_<label>_<key>.png` + `dop_<label>.png` polarization maps — none of
these have their own CLI toggle; they are silent no-ops when the relevant
inputs aren't present.

**`--emit-csv`** writes `results/<case>/data/*.csv` alongside essentially
every chart this stage renders (detector irradiance/profile/spectrum/
photometric/spectrometer, material n/k, coating/grating/polarizer/filter
response curves, per-source spectrum/polarization/ledger/closure,
per-element power/boundary-flux/per-face power, system energy-ledger/
power-flow, plus every `--export-rays`/`--save-fields` product below) and
a single `data/index.csv` mapping `file -> entity, chart, units,
provenance, image` (the PNG each CSV's data backs, where one exists).
Every CSV containing library-derived numbers (n/k, R/T, coating, filter)
carries the source registry row's `reference` column. Off by default: no
`data/` directory is created at all, and every renderer runs exactly as it
did before `--emit-csv` existed. The per-(source, detector) detected-power
promotion from `case.json["detected"]` into `report.json`
(`detectors.<label>.per_source`) and the `--export-rays`-driven analysis
renders (spot/fan/PSF/MTF/EE/wavefront/ghost) run **regardless** of
`--emit-csv` — only the CSV/`data/` side of those same products is gated
on the flag.

**`--export-rays` follow-on** (no-op unless the trace stage wrote
`rays_full.npz`, §8.2): `render_ray_analysis()` writes
`analysis/spot_<label>.png` + `analysis/fan_<label>.png` (spot diagram,
transverse ray fans, chief-referenced OPD fan; §6.10) per detector, and,
when the npz also carries `refl_hist` (i.e. the trace ran with
`--ghost-analysis`), `render_ghost_analysis()` additionally writes
`analysis/ghost_table_<label>.png` (top-12 multi-bounce specular paths by
detected power, as an ordered `element.FaceN -> element.FaceN -> ...`
signature) and `analysis/ghost_footprint_<label>_<k>.png` (detector-frame
2D histograms for the top 3 paths), plus
`report.json`'s `detectors.<label>.ghosts` block (`total_detected_W`,
`ghost_detected_W`, `ghost_fraction`, `n_paths`, `top` rows). A "ghost" is
defined as a detected ray with reflection generation `>= 2` that is
**purely specular** (`scattered=False` — roughness/diffuser/scatter lobes
are excluded as a continuous BSDF pedestal, not a discrete ghost).
`render_wavefront()` (also gated on `rays_full.npz`, needing
`> MIN_WAVEFRONT_RAYS=200` coherent rays in a key) writes
`analysis/wavefront_<label>.png` (per-key OPD map + Zernike-coefficient
bar chart) and `report.json`'s `detectors.<label>.wavefront` block;
`--wavefront-point X_MM,Y_MM` overrides its default power-weighted
landing-centroid image point (detector-grid-frame mm, same `u=pos.xhat,
v=pos.yhat` convention as the spot/fan renders); ignored without
`rays_full.npz`. `--wavefront-pupil {source,exit_pupil}` picks the pupil
model — `source` (default, normalized birth position on the emitting face)
or `exit_pupil` (an `analysis_imaging.py` chief-ray/exit-pupil search over
the field bundles, falling back to `source` with a report note when the
solve degenerates on a single field point / telecentric image side).
`--imaging-products LIST` (`distortion,vignetting,field_curves,
telecentricity`, or `all`) **hard-requires** `rays_full.npz` and a
multi-source field fan (e.g. the field-angle fan wizard): each writes
`analysis/imaging_*.png` + `report.json` `detectors.<label>.imaging`
blocks (+ `data/*.csv` under `--emit-csv`). See §6.10 for the physics.

**`--save-fields` follow-on** (no-op per-detector unless that `.h5` has a
populated `fields/` group, §6.7): `render_field_analysis()` writes
`analysis/psf_<label>.png`/`analysis/mtf_<label>.png`/
`analysis/ee_<label>.png` (one panel per coherent gather key plus a
synthetic incoherent-summed `"all"` panel) and `report.json`'s
`detectors.<label>.analysis` block. See §6.10 for the physics.

Curved (sphere/cylinder) detectors (§5.12) render through the same
`render_detector()` path as planar ones; `post_process.py` divides by the
`.h5`'s extra `pixel_area_map` dataset (true per-pixel metric area,
present only for curved detectors) instead of the fixed planar pixel
area, so irradiance stays correct regardless of screen curvature.

**Instruments follow-ons** (samples-instruments round; §7.11): a detector
carrying an `instrument` property naming a `diode_array` row (e.g.
`tcd1304_array`) gets `render_diode_array()`'s physical linear-array
readout instead of `spectrometer`'s continuous curve — same
`instrument/instr_<label>_diode_array_<mode>.png`/`.csv`/report-block
shape as every other instrument class. `--reference-case DIR` runs
`render_absorbance()`: it recomputes `DIR`'s own raw spectrometer/
diode-array product from that case's `.h5` (same instrument row, same
mode) and writes `A(λ) = -log10(I/I0)` against the current case's product
(`instrument/absorbance_<label>.csv` + PNG, `report.json`
`detectors.<label>.absorbance` block, with stray-light-floor masking near
`I≈0`); a mismatched instrument row or pixel grid between the two cases
is a hard `SystemExit` naming the mismatch, not a silently wrong curve.
`--ring-profile SPEC` (§8.1) runs `render_ring_profile()`
(`analysis_field.log_annular_power` + `parse_ring_spec`): log-spaced
annular power bins about the optical axis on the detector's UNCLIPPED
(possibly zero-mean-negative) power image, `center` resolved to `peak`
(brightest pixel), `chief`/unset (the image power centroid — every other
radial `analysis_field` function's own default), or an explicit `X,Y` mm
in the detector grid frame; writes `analysis/rings_<label>.csv` + PNG with
an exact closure accounting (incl. inside/outside remainders) so the ring
sum plus the two remainders equals the detector's total power exactly.
When the source carried an `image` body property (§5.14),
`render_image_traced()` (no CLI flag — runs automatically whenever the
scene has one) publishes the traced end-to-end detector image into
`imaging/`, auto-surfacing in the Results "Imaging" gallery; if
`--image-sim` also ran on the same detector, it additionally writes a
traced-vs-convolution-sim side-by-side PNG plus an NCC agreement metric,
evaluated at direct AND 180°-rotated orientations (a real imaging bench
inverts; the space-invariant sim is object-oriented) with the
better-agreeing orientation reported.

```
/home3/optics/env/bin/python scripts/compare_runs.py \
    --cases DIR [DIR ...] [--out DIR]
```
Overlays several finished cases' detector results: one
`profile_<label>.png` + `spectrum_<label>.png` per detector label present
in any case (Okabe-Ito CVD-safe palette, one color per `--cases` input
position, consistent across every plot), plus a single `compare.csv`
(`case, detector, total_power_W, peak_irradiance, profile_visibility`)
and a printed summary table. `--out` defaults to
`results/comparisons/<case1>_vs_<case2>...` (or `<first>_vs_N_more` if
that name would exceed 120 characters).

```
/home3/optics/env/bin/python scripts/compare_sweep.py \
    (--manifest PATH | --cases DIR [DIR ...]) [--out DIR] [--ref REF]
```
The sweep-comparison companion to `compare_runs.py`. `--manifest`
consumes a `results/<model>/sweep-<case>.manifest.json` (written by the
GUI's sweep controller, §5.9) to plot metric-vs-variable curves over the
swept axis; `--cases` instead takes arbitrary case directories with no
variable axis (exactly like `compare_runs.py`). `--ref` names the
reference variant stem / case label (default: the first variant/case),
and `--out` defaults per mode.

```
pvpython make_viz.py --case-dir DIR --model-json PATH
    [--views geometry-view-names] [--resolution WIDTHxHEIGHT=1920x1080]
    [--out DIR] [--smoke] [--skip-vtkexport]
    [--dim-rays {off,linear,sqrt}] [--dim-rays-floor PCT]
```
Registered views (`viz_configs.VIEWS`): `overview3d` (three-quarter
perspective, all bodies+rays+detectors), `top`/`side` (parallel
projection), `detector_closeup` (one PNG per detector, camera zoomed
face-on, forced 2048×2048), `turntable` (8 frames, fixed elevation,
azimuth sweep), and **`rays_polmode`** (rays colored by the `rays.npy`
`pol_mode` column — ordinary/isotropic vs. extraordinary o/e-split —
instead of by wavelength; skipped with a warning if `rays.vtp` predates
the `pol_mode` array, e.g. a stale `viz/` combined with
`--skip-vtkexport`). `--smoke` forces just `overview3d` at 800×600
(ignores `--views`/`--resolution`) for a fast end-to-end render check.
`--skip-vtkexport` reuses an existing `viz/rays.vtp`/`viz/det_*.vtp`
(skips the `raytracer.vtkexport` sub-step, which itself always runs under
`OPTICS_PYTHON` as a subprocess — pvpython never imports the `raytracer`
package directly). `--dim-rays` fades ray segments by attenuation
(opacity = P/P_birth from the `rel_power` column, linear or sqrt curve,
`--dim-rays-floor` percent minimum): the flags are forwarded to the
vtkexport prep step, which bakes an `rgba` cell array (wavelength rgb +
power alpha) that ParaView renders as direct RGBA — deliberately NOT a
pvpython ProgrammableFilter, which leaks `numpy_interface` names into
`__main__` and shadows builtins. Falls back to undimmed rgb with a
warning if `rays.vtp` lacks the array (`--skip-vtkexport` on a viz/
exported without the flag, or a pre-dimming trace).

### 8.4 `sweep_variants.py` (system `python3` — batch jobs + auto-compare)

```
python3 scripts/sweep_variants.py
    [--jobs jobs.json] [--job "k=v,k=v"]...
    [--models FCSTD ...] [--preset P] [--tag T] [--steps S] [--seeds N]
    [--rays F] [--resolution N] [--nlambda N] [--spectral-bins N]
    [--viz-rays N] [--backend B] [--particles SPEC] [--particle-threshold F]
    [--dry-run] [--keep-going] [--no-compare] [--compare-out DIR] [--print-only]
```
Two ways to specify jobs: `--jobs jobs.json` (a JSON list of per-job
dicts, keys = `run_pipeline.py` option names without leading dashes,
dashes → underscores; list-valued flags like `source_face`/`var` may be
JSON lists) or repeatable `--job "k=v,k=v"` (scalar-only shorthand). Flags
given directly on the `sweep_variants.py` command line seed every job as
defaults; a per-job entry overrides them. Jobs run **sequentially**
through `run_pipeline.py` (imported directly, reusing its exact
model-glob/variant-naming/case-naming logic so the case directories this
script predicts never drift out of sync with what `run_pipeline.py`
itself writes). After the batch, every job that produced a
`report.json` is overlaid with one `compare_runs.py` call (skip with
`--no-compare`; redirect with `--compare-out`).

### 8.5 `extract_geometry.py` / `permute_model.py` / `make_test_scenes.py` (FreeCAD headless)

```
/home3/freecad/FreeCAD.AppImage -c scripts/extract_geometry.py -- \
    [--models FCSTD ...] [--outdir DIR] [--strict]
```
No `--models` scans `PROJECT_DIR/*.FCStd` + `BASEMODELS_DIR/*.FCStd`.
`--strict` turns "face falls back to mesh representation" from a warning
into a hard error at extract time — use it when authoring a new analytic
scene to catch un-canonicalizable surfaces immediately (independent of
the trace-time `--strict-analytic` flag, §5.8, which controls whether the
*tracer* accepts an already-extracted mesh face).

```
/home3/freecad/FreeCAD.AppImage -c scripts/permute_model.py -- \
    --model FCSTD --var ALIAS --min F --max F --n N [--var ...]...
    [--outdir DIR] [--unit mm]
```
`--var`/`--min`/`--max`/`--n` are all `action="append"`, required, and
must appear the same number of times (paired positionally, §5.9).

```
/home3/freecad/FreeCAD.AppImage -c scripts/make_test_scenes.py -- \
    [--outdir DIR] [--scene NAME|all]
```
`--scene` (default `doubleslit`) accepts either a single scene name or
the literal `all` (builds every scene, §10); it does **not** accept a
comma-separated list. An unrecognized name is an error naming every known
scene.

### 8.6 `optimize.py` / `tolerance.py` (optics env python — design tools)

Both tools drive spreadsheet cell aliases (the same addressing
`permute_model.py --var`/`fast_eval.py` use: bare `alias` on the default
`dim` sheet, or `sheetlabel.alias`) through `fast_eval.py`, the shared
merit evaluator: it maps a dict of design parameters to a dict of merit
scalars pulled from `report.json`, backed by either a fresh
extract→trace→post per evaluation (`--eval-backend full`, the reference)
or a persistent headless FreeCAD worker that applies only the changed
cells and re-extracts in place (`--eval-backend worker`, the default,
much faster; parity pinned by `test_fast_eval.py`). `cli_specs.STAGES`
now lists six parsers (`pipeline, trace, post, viz, optimize, tolerance`,
up from the original four pipeline stages in §1).

```
/home3/optics/env/bin/python scripts/optimize.py --model FCSTD \
    --var NAME:START:LO:HI [--var ...]... \
    --operand OPERAND[@DETECTOR]:TARGET:WEIGHT [--operand ...]... \
    [--algorithm {local,global}] [--budget N] [--tol F] \
    [--preset {quick,normal,detailed}] [--eval-backend {worker,full}] \
    [--no-final-coherent] [--config JSON] [--out DIR]
```
Merit-function optimizer: `--algorithm local` (default) is scipy
Nelder-Mead within bounds; `global` is nevergrad CMA-ES (needs the
`nevergrad`/`cma` packages, §3.2 of INSTALL.md). Operands: `spot_rms`/
`focus` (detector spot RMS radius, needs `--export-rays`, added
automatically), `encircled_energy`/`mtf50` (coherent field analysis,
needs `--save-fields`, slow), `detected_power`, or any raw flattened
`report.json` merit key. Runs the inner loop incoherently for speed, then
re-evaluates the best design once with coherence as authored (skip with
`--no-final-coherent`). The `auto_designed_lens` scene
(`scripts/make_test_scenes.py`, §10; not part of the `demos/` gallery)
is its validation fixture.

```
/home3/optics/env/bin/python scripts/tolerance.py --model FCSTD \
    --tolerance NAME:NOMINAL:DIST:BAND [--tolerance ...]... \
    --operand OPERAND[@DETECTOR]:TARGET:WEIGHT [--operand ...]... \
    [--draws N] [--merit-threshold X] [--compensator VAR:LO:HI] \
    [--skip-sensitivity] [--preset {quick,normal,detailed}] \
    [--eval-backend {worker,full}] [--config JSON] [--out DIR]
```
Sensitivity analysis (per-tolerance finite-difference merit impact,
ranked table; skip with `--skip-sensitivity`) plus Monte-Carlo yield
tolerancing: `--draws` random perturbation sets (`normal`/`uniform`
distributions) report the merit distribution and, with
`--merit-threshold`, the pass/fail yield fraction; `--compensator`
nests a focus-recovery `optimize.py` local-engine call per draw. The
`tolerance_lens` scene (`scripts/make_test_scenes.py`, §10; not part of
the `demos/` gallery) is its validation fixture, built on `camera_triplet`-
style radius/thickness/decenter perturbations.

Both share `--preset`/`--rays`/`--resolution`/`--nlambda`/`--seeds`/
`--seed0` (fed to the fast_eval evaluator the same way `run_pipeline.py`
fills them from `common.PRESETS`) and `--config JSON` (keys mirror the
CLI dests; explicit flags win over the file).

### 8.7 `run_dls.py` / `dls_correlate.py` (optics env python — dynamic light scattering)

Traced-dynamics DLS: a real Brownian-motion frame sequence off a
body-bound EXPLICIT-mode `sample` (§5.13), each frame independently
traced and gathered into a coherent speckle field, so the frame-to-frame
field decorrelation is genuine particle-displacement physics rather than
an analytic model.

```
/home3/optics/env/bin/python scripts/run_dls.py
    --model-json PATH --case-dir DIR
    --frames N --dt-ms F
    [--temp-k F=293.15] [--rays F=1e5] [--nlambda N=1]
    [--detectors LABEL,...] [--resolution N=32]
    [--workers auto|N] [--seed N=12345] [--max-gb F=2.0]
    [--optical-properties PATH] [--particle-threshold F=2e5]
    [--max-reflections N=6] [--power-floor F=1e-4]
```

Requires exactly one EXPLICIT-mode `sample` body in the scene and at
least one `coherent=true` source. Flow: (1) build the `Scene` **once**
and locate the sampled body's `BodyParticleMedium`; (2) frame 0 = the
sample row's frozen explicit realization; pre-generate `--frames` further
positions SEQUENTIALLY via per-particle Stokes-Einstein diffusion
(`D_i = kB·T / (6π·η·r_i)`, `η` = the sample row's `solvent_visc_pas`,
`T` = `--temp-k`; per-axis step `sigma_i = sqrt(2·D_i·dt)`), a REFLECTIVE
body-wall boundary condition enforced by rejection-redraw against
`Scene.point_inside_body` (a particle holds still that frame after
`_REFLECT_TRIES` failed redraws); radii are frozen at frame 0 (no size
evolution). (3) Trace all frames EMBARRASSINGLY PARALLEL over a spawned
process pool (`numpy.random.SeedSequence.spawn`, the same CUDA-safe
convention `run_trace._run_sharded` uses); each worker reconstructs the
RAW (unnormalized, ungated) coherent field per frame with
`gather.points_numpy` directly — DLS needs frame-to-frame PHASE evolution,
not a power-calibrated image, so `render_coherent`'s per-population power
renormalization and its `M_eff` sampling gate are deliberately bypassed
(numpy also keeps torch/CUDA out of the spawned workers entirely). (4)
Persist `<case>/dls/frames.h5` + `manifest.json`.

**`frames.h5` schema**: `/positions` `(N, n_p, 3)` float32 (Brownian
positions, m); `/radii` `(n_p,)` float32 (frozen); `/dt_s` (), `/temp_k`
() scalars; `/detectors/<safe>/frames` `(N, nkeys, 2, H, W)` complex64 —
per-frame RAW coherent field, axis 1 = gather key (mutually incoherent
populations — correlations sum, fields don't), axis 2 = `[Ex, Ey]`
detector-frame Jones components; `/detectors/<safe>/q_vector` `(3,)`
float64 (`k_s - k_i`, 1/m); group attrs `label, H, W, pixel_m, xhat,
yhat, normal, x_lo, y_lo, q_magnitude_per_m, lam_lo_m, lam_hi_m,
keys_json`; root attrs `seed, engine, sample, body, host_material,
solvent_visc_pas, n_particles, rays, nlambda, frames, dt_ms,
sample_row_json`.

```
/home3/optics/env/bin/python scripts/dls_correlate.py
    --case-dir DIR [--aperture-px K] [--emit-csv]
```

Fully OFFLINE and re-runnable (reads only `frames.h5`, never re-traces).
Per detector: the complex field summed over a central `K×K` aperture
(default: the full grid; smaller `K` trades a higher coherence factor
`beta` for lower SNR) per (source, λ-stratum, pol-stratum) gather key and
Jones component gives an FFT field autocorrelation `g1(tau)` (per-channel
autocorrelations summed at the CORRELATION level — the physically correct
combination for mutually-incoherent channels — then normalized to
`g1(0)=1`); the Siegert relation `g2(tau) = 1 + beta·|g1(tau)|^2`
reconstructs the intensity correlation, `beta` fitted from the measured
`tau->0` aperture-intensity-fluctuation intercept; a weighted second-order
cumulant fit `ln|g1| = -Gamma·tau + mu2·tau^2/2` over the contiguous
`|g1| > 0.1` window gives the decay rate `Gamma`, diffusion coefficient
`D = Gamma / q^2` (`q` from the h5's `q_magnitude_per_m`), and the
hydrodynamic diameter `d_H = kB·T / (3·pi·eta·D)` (Stokes-Einstein).
Outputs under `<case>/dls/`: `g2_<label>.csv`, `correlogram.png`
(`|g1|(tau)` per detector, log-x), `gamma_vs_q2.png` (multi-angle
through-origin fit, slope = `D`), `report.json` (per-detector `D, d_H,
beta, Gamma`). GUI: Results pane "dls" gallery.

**Honest limits (shared)**: **single-scattering only** — keep the sample
optically thin (`tau` from §5.13's `sample` row, target a few×`1e-3`);
frozen radii (no size evolution); no hydrodynamic interactions and no
structure-factor collective slow-down of `D` (dilute Stokes-Einstein);
no sedimentation/flow (drift-free Brownian only). **Sparse-cloud
requirement**: the explicit medium samples each scatter event with a
SHARED Monte-Carlo RNG stream whose draw order depends on the exact set
of ray-particle collisions — in a DENSE cloud a nm-scale particle move
flips a collision somewhere, desynchronizing the whole stream, so the
speckle field goes delta-correlated frame-to-frame instead of decaying at
the physical `D·q^2` rate. For a clean, quantitative `g1` decay the
sample must be DILUTE (tens of well-separated spheres, not thousands) so
the collision set stays stable across a frame — dense suspensions still
run and persist, but their `Gamma`/`D`/`d_H` are unreliable (the
correlator math itself is validated independently against synthetic
fields, `test_dls.py`). Index-match the ambient to the solvent (e.g. the
`vat_cylindrical` decalin bath, §5.13's catalog additions) so scattered
rays are not TIR-trapped at cell walls.

---

## 9. Particle clouds

This section covers the CLI `--particles` world-box spec. A body-bound
sample cloud authored via the `sample` property (registry-driven S(q)
structure factors, explicit lattice realizations, T-matrix spheroids) is a
separate, coexisting route — §5.13.

`--particles` spec (`common.parse_particles_spec`):
```
box=x0,y0,z0:dx,dy,dz;material=NAME;(phi=F|tau=F);median_um=F;gsd=F[;seed=N]
```
`phi` (mass fraction, below) and `tau` (target Beer–Lambert optical depth
along the box's along-beam length `dx`) are **mutually exclusive — supply
exactly one**. `tau` is only *parsed* by the stdlib-only `common.py`; it
is resolved to an equivalent `phi` inside
`raytracer.particles.ParticleCloud` (which needs the material's `Qext`,
hence numpy/scipy), and the resolved `phi` plus the `tau` it came from are
echoed back through `ParticleCloud.diagnostics()['tau_resolved']` into
`case.json`.
`box` corner+size are in **project CAD units (mm)**, converted to SI
internally; omit `box=` for the default `10×20×20 mm` box centered on the
x-axis just before the origin (`corner=(-12,-10,-10)mm`,
`size=(10,20,20)mm`). `median_um` is the **median diameter** (halved
internally to a median radius for the underlying log-normal-over-radius
distribution). `gsd` (geometric standard deviation, `>= 1`) defaults to
1.6 if omitted, matching realistic aerosol/droplet-cloud polydispersity.

**`phi` is a MASS fraction, not a volume fraction or a particle count** —
this is the single most common source of "surprising" particle counts.
`mie.number_density()` converts it to volume fraction
`f_v = (phi/rho_p) / (phi/rho_p + (1-phi)/rho_h)` and then to a number
density `N = f_v / E[V_particle]`. Worked example (water droplets, median
diameter 10 µm, `gsd=1.6`, in air, the default 10×20×20mm box):

| `phi` (mass fraction) | volume fraction `f_v` | number density `N` [/m³] | count in default box |
|---|---|---|---|
| `1e-4` | ≈1.2e-7 | ≈8.5e7 | ≈341 |

Because water is ~800× denser than air, even a *tiny* mass fraction
already implies hundreds of droplets — and conversely, achieving a
*large* explicit particle count (thousands+) for dense particles in a
light host medium requires a `phi` that looks alarmingly large (the
test suite's `test_explicit_realization` uses `phi=0.45` for ~1000
50 µm water droplets, `f_v` still only `~1e-3`). If your particle count
comes out unexpectedly low or the constructor raises "particle cloud is
empty," the fix is almost always to raise `phi`, not to misread it as a
volume fraction.

**Hybrid mode threshold**: `count = N * box_volume` compared against
`--particle-threshold` (default `2e5`, deliberately aligned with
`ExplicitRealization.MAX_BRUTE = 200_000` so the default never lands in
the dead zone between "explicit mode selected" and "over the brute-force
cap"). `count <= threshold` → **explicit** mode: a frozen
non-overlapping-sphere realization (dart-throwing placement, rejected
against overlaps and optic-body interiors), rays collide via brute-force
AABB-culled sphere intersection against an **extinction radius**
`r * sqrt(Qext)` (the extinction paradox makes `sigma_ext` up to 2× the
geometric cross-section), and scattering applies exact complex Mie
S1/S2 amplitudes per collision, with the scattering azimuth sampled from
the true polarized differential cross-section by default (`--no-pol-
scatter` reverts to the legacy uniform-azimuth approximation, §6.2) —
deterministic per realization, so a fixed seed produces physical speckle;
`MAX_BRUTE = 200_000` is a hard cap (raise `--particle-threshold` to
force continuum mode instead, or reduce `phi`/box size), with a
performance warning above 50,000 spheres. `count > threshold` →
**continuum** mode: the box becomes a participating medium with a
deterministic-splitting estimator per traversal (ballistic Beer-Lambert
amplitude decay on the parent + one incoherent scattered child per
crossing ray, §6.2).

**`--seeds N` statistics**: each seed redraws the RNG (`seed0 + s`) for
both ray sampling and — in explicit mode — a **fresh, independent sphere
realization** each seed (a new `ParticleCloud`/`ExplicitRealization` is
constructed inside `run_one_seed()` per seed), so `--seeds N > 1` on a
particle scene gives you N independent speckle realizations to average,
exactly like `--seeds` on a rough-surface scene (§5.4).

---

## 10. Test scene catalog (`make_test_scenes.py`)

```bash
/home3/freecad/FreeCAD.AppImage -c scripts/make_test_scenes.py -- --scene all
python3 scripts/run_pipeline.py --models basemodels/lens_pcx.FCStd --preset quick
```

(For MULTI-element showcase systems — telescopes, a Cooke-triplet camera,
a Michelson interferometer, spectrometers, a ball-lens fiber coupler —
see the top-level `demos/` gallery instead: ten packed `.MieWB` scenes
with cited prescriptions, built by `scripts/make_demos.py`. The scenes
below stay single-element by design: they are the physics validation
fixtures.)

33 buildable scenes total: `doubleslit` (the original scene, documented
in §5.10, still built with unchanged parameters — d=0.5mm slit
separation, 633nm, L=99mm plate-to-screen gap, fringe pitch λL/d=125.3µm)
plus 32 more registered in the `SCENES` metadata dict, spanning lenses,
polarization, birefringence (uniaxial + biaxial), gratings,
filters/coatings, Gaussian beams, Fresnel ghosts, measured scatter, curved
detectors, optimizer/tolerancing demos, and a deliberately non-analytic
mesh face:

| Scene | Setup | Validates | Expected value |
|---|---|---|---|
| `lens_pcx` | Plano-convex BK7 singlet, R1=25mm, t=5mm, collimated 633nm Ø10mm beam | thick-lens EFL/BFL vs. lensmaker's equation | n(633)=1.51508, EFL=48.536mm, BFL=45.236mm |
| `lens_dcx` | Symmetric biconvex BK7, R=±40mm, t=6mm, collimated 633nm | symmetric biconvex EFL | EFL=39.845mm |
| `lens_pcv` | Plano-concave BK7, R2=25mm concave, t=3mm, diverging | negative-EFL thick-lens formula | EFL=−48.536mm (virtual focus) |
| `lens_dcv` | Symmetric biconcave BK7, R=∓40mm, t=3mm, diverging | biconcave diverging EFL | EFL=−38.340mm |
| `lens_achromat` | Cemented BK7/SF5 achromatic doublet (5µm air gap for the cemented interface), achromatized at F/C | achromat design vs. Abbe-number chromatic-focus-shift theory | EFL=50.0mm, V_bk7=64.14, V_sf5=32.24 |
| `lens_sphere_control` | Plano-convex spherical BK7, f≈40mm, Ø20 aperture, k=0 | spherical-aberration spot size, control twin of `lens_asphere` | EFL=40.0mm |
| `lens_asphere` | Plano-convex BK7 asphere, conic `k=-n^2` (eliminates on-axis SA), `surface_override` verified <1µm | asphere sag reconstruction + diffraction-limited spot vs. spherical control | EFL=40.0mm, k=−2.29547 |
| `lens_cyl_pos` | Plano-convex cylinder lens (axis z), R=25mm, collimated 633nm | 1D cylindrical focusing, native Cylinder-face extraction | EFL=48.536mm |
| `lens_cyl_neg` | Plano-concave cylinder lens, R=−25mm, diverging | diverging cylinder-lens EFL | EFL=−48.536mm |
| `axicon_pcx` | Plano-convex BK7 axicon, 10° base angle, Ø22, conical front (native Cone face) | axicon Bessel-ring formation, conical-surface extraction | ring radius = z·tan((n−1)α) = 2.7043mm at z=30mm |
| `lens_ball` | Full 8mm BK7 sphere, collimated 587.6nm Ø4 beam | ball-lens BFL/EFL formula, native Sphere-face extraction | BFL=1.870mm, EFL=5.870mm |
| `lens_rod` | 8mm-diameter BK7 rod (cylinder), collimated 587.6nm | rod/cylindrical-lens focusing, padded-circle→native-Cylinder path | n_d only (no target EFL given) |
| `lens_fresnel` | Collapsed plano-convex Fresnel lens, f≈50mm, 8 annular conical facets | faceted/Fresnel-surface handling, focusing accuracy | EFL=50.0mm, ≥8 Cone faces |
| `prism_equilateral` | 60° equilateral BK7 prism near minimum deviation for 550nm, broadband 420-680nm source | prismatic dispersion / minimum-deviation geometry | n(550)=1.51852, min deviation=38.798°, entrance AOI=49.399° |
| `pol_linear` | Thorlabs LPVISE100-A linear polarizer on 2mm PMMA, `linear:30` source, spreadsheet-driven rotation | Malus's law + substrate Fresnel loss + real extinction-ratio curve | Malus factor cos²(30°)=0.75 at polangle=0 |
| `pol_crossed` | Two `ideal_linear` polarizers, crossed axes, unpolarized 550nm | crossed-polarizer leakage (finite extinction ratio) | transmission ≈5e-7 |
| `pol_circular` | Thorlabs CP1L532 left-handed circular polarizer, BK7 substrate, 532nm | circular-polarization GENERATION (linear-then-retarder stage order) | output is circular with the stated handedness; handedness *analysis* needs a waveplate+linear stack (see §5.3 note; xfail documents this in `test_scenes_e2e.py`) |
| `waveplate_quartz` | Multi-order (m=30) half-wave quartz plate (589nm) between crossed `ideal_linear` polarizers at ±45° | birefringent retardance / waveplate action | n_o=1.54422, n_e=1.55332, thickness=1.9740mm, retardance=30.5 waves, transmission factor 1.0 |
| `waveplate_mgf2` | Multi-order (m=30) half-wave MgF2 plate (589nm) between crossed `ideal_linear` polarizers at ±45°, `crystal_axis '0,0,1'` | uniaxial retardance with C-ENGINE PARITY — MgF2 carries no gyration data (unlike gyrotropic quartz, which reference-routes), so the scene stays C-routable | n_o=1.37772, n_e=1.38953, Δn=0.011812, t=1.5209mm, retardance=30.5 waves, transmission factor 1.0 |
| `pbs_cube` | 20mm BK7 PBS cube from two 45° prisms, `pbs_visible_45` coating on the hypotenuse, unpolarized 550nm | coated-interface polarization splitting (s/p separation) | qualitative: transmitted arm ~p-pol, reflected arm ~s-pol |
| `calcite_displacer` | 10mm calcite slab, `crystal_axis` at 45° in x-z, unpolarized 590nm Ø0.5mm beam | birefringent walk-off (o/e spatial displacement) | n_o=1.65830, n_e=1.48611, walk-off=6.232°, displacement=1.0919mm |
| `wollaston` | Two 30° calcite wedges, orthogonal optic axes, 5µm air gap, 590nm unpolarized | birefringent-wedge polarization beam-splitting angle | split angle = 2(n_o−n_e)tan(30°) = 11.392° |
| `filter_bandpass` | `bp_550_40` bandpass filter on a 3.5mm BK7 slab, broadband 450-650nm source | wavelength-dependent filter transmission | qualitative (center 550nm, band 450-650nm) |
| `hot_mirror` | BK7 plate at 45° AOI, `hot_mirror_45` coating, broadband 450-1000nm, two detectors | angle-dependent dichroic spectral splitting | qualitative visible/IR split (center 700nm) |
| `ktp_walkoff` | 15mm KTP biaxial plate, X principal axis at 45° in x-z (`'0.70711,0,0.70711'`), Y principal = global y; 633nm unpolarized Ø0.3 beam in the X-Z principal plane (max walk-off) | biaxial walk-off (y-sheet straight at n=n_y, in-plane sheet walks off in z → two spots) | two-spot separation = solver-predicted in-plane-sheet transverse displacement (`biaxial_ray_from_k`; `test_biaxial._expected_walkoff_dz`) |
| `gaussian_bench` | Gaussian-beam source (`beam_waist` 50µm at the face, M²=1.0) propagating 62mm (=5 Rayleigh ranges) through air to a screen | Gaussian-beam expansion, incoherent direct-deposit beam mode (`coherent=False`) | w(z)=w0·√(1+(z/zR)²), zR=πw0²/λ≈12.4mm → ~5× waist |
| `ghost_doublet` | Two uncoated N-BK7 flat windows (4mm thick, 8mm gap) in a collimated 633nm incoherent beam; downstream screen | Fresnel ghost enumeration | dominant gen-2 ghost power = direct·R² (normal-incidence air/BK7 Fresnel product) |
| `scatter_plate` | Flat BK7 window at 45° with measured ABg scatter (`scatter=polished_bk7_glass`) on its front face; collimated 633nm | measured-scatter specular+diffuse split (reflected side conserves R; scattered rays flagged `scattered=True`) | qualitative: specular spot + diffuse lobe on the +y screen |
| `curved_focal` | Collimated 633nm Ø10 → plano-convex BK7 lens (R=25, f≈48.5) → CONCAVE cylindrical detector hugging the focus at x≈50mm | curved-detector irradiance (per-pixel metric-area division, §5.12) | EFL=48.536mm, BFL=45.236mm; curved screen catches >90% of focused power |
| `auto_designed_lens` | `lens_pcx` singlet with spreadsheet-driven axial position (`dim.lenspos`, expression-bound Placement); detector 4mm past the lenspos=0 focal plane | optimizer demo: spot-minimizing lens position | expected focus at lenspos=+4mm (collimated: focus translates 1:1 with the lens) |
| `tolerance_lens` | The `auto_designed_lens` singlet with THREE spreadsheet DOFs — `dim.lenspos` (axial), `dim.lensdy` (decenter), `dim.detpos` (detector axial); nominal design in focus | tolerancing demo | lenspos defocuses 1:1, lensdy mostly translates the spot (RMS first-order insensitive), detpos is the refocus compensator |
| `mesh_freeform` | Prolate ellipsoid of revolution (revolved BSpline meridian) — deliberately non-canonicalizable | mesh-fallback code path (extractor WARN, `--strict` incompatible) | `expects_mesh_fallback: True` |

Usage is the same one-liner as any other model:
`python3 scripts/run_pipeline.py --models basemodels/<scene>.FCStd
--preset quick` after `--scene all` has populated `basemodels/`. None of
the birefringence/polarization/filter/coating scenes above have a
dedicated end-to-end pytest gate the way `doubleslit` does (§11) — they
are manually-validated reference scenes exercising the closed-form-tested
physics modules against a real extracted FreeCAD geometry.

---

## 11. Validation

`scripts/raytracer/tests/` (935 tests total — see the actual count with
`--collect-only -q`; the table below covers the physics-pinning core, not
every file; run with
`/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -v`;
`test_gather.py` and `test_doubleslit_e2e.py` take minutes):

| Test file | Physics pinned | Tolerance |
|---|---|---|
| `test_kernels.py` (18) | Snell angles; Fresnel energy conservation; Brewster angle; TIR magnitude + Fresnel-rhomb phase; absorbing-metal Fresnel branch; reflect/pol-basis unitarity + Jones-rotation energy conservation; TMM reduces to bare Fresnel at 0 layers; quarter-wave MgF2-on-BK7 TMM reflectance; half-wave "absentee" layer; TMM energy conservation (oblique, 2-layer stack); sphere/cylinder/cone exact intersection; torus intersection vs `np.roots` reference; trimmed spherical cap, full-sphere-untrimmed, polar-cap-band, plane+hole trim, face exclude/eps guard | 1e-9 to 1e-12 (closed-form); torus vs `np.roots` 1e-7 |
| `test_asphere.py` (7) | Asphere reduces to a tangent sphere (k=0, no coeffs); parabola (k=-1) closed form; general polynomial asphere vs independent `scipy.brentq` root; analytic normal vs finite-difference; shape-operator (`normal_derivative`) vs finite difference for every analytic primitive; miss/grazing cases produce no NaN; trimmed asphere cap containment | <1e-10 (sphere/parabola reduction); <1e-10 (brentq); <1e-7 (normal FD); <1e-5 relative (shape operator, curved primitives) |
| `test_birefringence.py` (21) | Uniaxial `n(theta)` endpoints/monotonicity; eigenbasis orthonormality; isotropic limit reduces to Fresnel; **calcite walk-off at 45° internal incidence, 590nm, matches 6.23° to <0.05°**; quartz's opposite (positive-uniaxial) walk-off sign; e-wave normal-surface residual; tangential wavevector continuity at an interface; exit refraction reduces to Fresnel for the o-mode; beam-displacer normal-incidence walk-off + displacement; Wollaston geometry (c in interface plane -> zero walk-off); plane-parallel slab o/e roundtrip; internal e-ray TIR; exact `ray_from_k`/`k_from_ray` inversion | 1e-12 (closed-form); **0.05° (calcite/quartz walk-off vs literature)**; <0.5% (displacement) |
| `test_grating_models.py` (11) | Kogelnik efficiency formula (exact Bragg `nu=pi/2` gives `eta=1`, `nu=pi/4` gives `eta=0.5`); full geometric Bragg-peak construction with energy conservation; Dammann Parseval sum + symmetry; Dammann reduces to lamellar duty=0.5; a real 1x5 equal-intensity Dammann design vs its published transition points; table-model exact interpolation roundtrip + out-of-range hard error + missing-order zero; polarized table dispatch (pure s/p stays pure, exact `eta_s`/`eta_p` power, energy-exact children+absorbed); m=0 lamellar still reduces to Snell | 1e-9 to 1e-12 (closed-form Kogelnik/energy); <0.1% (Dammann 1x5 uniformity, published tol 2%) |
| `test_grating_roughness.py` (10) | Vector grating equation angles; m=0 reduces exactly to Snell/Fresnel (any incidence, conical mount, groove component invariance); lamellar efficiency closed form + energy partition; roughness TIS closed form; Beckmann slope statistics + reflected-energy clustering; groove-vector frame construction | 1e-9 to 1e-15 (closed-form); Beckmann RMS slope within 3%, reflected energy clustering >0.99 mean/>0.9 min |
| `test_mie_particles.py` (13) | Mie efficiencies vs Wiscombe MIEV0 canonical values (m=1.5, x=10 → Qext=2.8820; x=100 → Qext=2.0944); extinction-paradox limit (Qext→2); Rayleigh limit closed form; absorbing-particle albedo; phase-function normalization + forward-peaking + CDF sampling fidelity; monodisperse number-density/mu_ext exact closed forms; log-normal mean volume vs Monte Carlo; continuum Beer-Lambert energy split; explicit-mode non-overlapping placement + energy conservation | abs 2e-3 (MIEV0); rel 1–5% (statistical); exact to 1e-9/1e-12 (closed-form) |
| `test_pol_scatter.py` (7) | Polarized-azimuth Mie sampler: theta-marginal invariance (analytic + numeric) between polarized and legacy uniform-azimuth sampling; chi-squared goodness-of-fit for linear-polarized and circular/unpolarized incidence; per-scattering-event energy exactness; legacy `--no-pol-scatter` uniform-azimuth mode still available; general sampler-vs-PDF agreement | chi2 goodness-of-fit; exact energy per event |
| `test_polarization.py` (9) | Reference-frame (`e_ref`,`e_perp`) orthonormality + z-axis fallback to y; unit power for every polarization kind/stratum; `linear:0/90/45` Jones values; unpolarized strata orthogonality; elliptical limits (chi=0 -> linear, chi=45 -> circular); circular handedness matches the Hecht convention numerically; polarization strata do not interfere in the gather (same-stratum fringes >0.7 visibility, cross-stratum <0.35); end-to-end `sample_source` polarization states (skipped unless `doubleslit` is extracted) | exact (Jones unit power); numeric handedness check; visibility >0.7 same-stratum / <0.35 cross-stratum |
| `test_polarizer_filter.py` (9) | Malus's law through an ideal linear polarizer; crossed-polarizer floor + `polarizer_absorbed` bucket (>0.85 of power parked there); unpolarized half-transmission; bulk filter Beer-Lambert thickness scaling (`T^(d/d_ref)`); calcite double-spot separation + polarization-selected single spot; circular-polarizer output handedness | Malus 1% or better; energy-exact bucket accounting |
| `test_opticalproperties.py` (53) | Full opticalproperties tree loading; birefringence/coating/polarizer/filter/grating registry schema validation (each hard-error path: bad type, swapped T_par/T_perp axes, zero transmittance, over-unity order efficiencies, R+T>1, missing reference, unknown crystal material); trimmed-library optional-category handling; per-face/polarization/axis/grating-value spec parsers (`common.py`); schema-v2 `validate_model()` additions (polarizer/crystal_axis/coating-dict/roughness_faces/grating/polarization) and their rejections; asphere surface-dict schema validation | hard MaterialError/ContractError on every malformed input |
| `test_materials.py` (22) | Dispersion spot-checks (bk7, fused_silica, mgf2, water, sapphire, polystyrene, aluminum n/k incl. IR k rise and water's ~975nm absorption shoulder); tabulated-range hard error; unknown-material error message; vectorized evaluation; coating quarter-wave resolution; malformed-CSV rejections (bad model, non-positive Sellmeier C, missing reference); coating-references-unknown-material rejection; detector/vacuum sentinels; case-insensitive lookup | abs 1e-4 to 2e-4 vs literature/refractiveindex.info |
| `test_mesh_bvh.py` (6) | MeshFace intersection vs analytic Sphere (tessellation-sag error bound); smoothed vs flat normal modes; BVH traversal bit-identical to brute-force Möller–Trumbore; binary/ASCII STL reader agreement; self-hit `t_eps` guard; degenerate (zero-area) triangles dropped with a warning | >98% hit/miss agreement away from silhouette; bit-identical BVH-vs-bruteforce |
| `test_ray_differentials.py` (9) | Free-space spherical-wave patch area grows as r²; collimated patch area invariant; `init_curved` direction derivative vs finite difference; reflect/refract differentials vs finite-difference oracle for sphere/cylinder/asphere; concave-mirror patch area minimizes at f=R/2; single-surface refraction patch area minimizes at f=n2 R/(n2-n1); TIR/grazing produces NaN cleanly (no exception) | <1e-5 relative (finite-difference cross-checks) |
| `test_gather_occlusion.py` (7) | `occlusion=None` bit-identical to omitting the argument; empty-faces identity; an opaque plate fully shadows the geometrically blocked half (<1% of unoccluded peak) while leaving the rest exactly rescaled; an off-axis plate is culled by the Level-1 AABB prefilter; a detector's own face never self-occludes; torch/numpy backends apply an identical mask; rendered shadow-edge position within `tile+1.5` px of the geometric projection | <1% leakage; bit-identical torch/numpy mask; shadow edge within `tile+1.5` px |
| `test_gather.py` (7) | Gather kernel vs brute-force reference; point-source 1/r² + spherical phase; double-slit fringe pitch + visibility; single-slit first-zero position; incoherent source shows no fringes; undersampling gate raises; torch vs numpy backend agreement | kernel rel 1e-4; point-source rel 1e-5; double-slit pitch rel 1%, visibility >0.9; single-slit zero rel 2%; incoherent visibility <0.35; torch/numpy rel 5e-3 |
| `test_integration.py` (5) | `Scene` built from the real extracted `example.FCStd` contract; wavelength-strata semantics (monochromatic + asymmetric-Gaussian bounding); full-trace energy closure; traced focal position vs the thick-lens lensmaker equation | closure <1e-3; **focal position vs lensmaker equation: rel 5e-3 (<0.5%)** |
| `test_doubleslit_e2e.py` (1) | **The** end-to-end wave-optics validation: the real `doubleslit.FCStd` scene, extracted and traced through the full engine, must produce Young's fringes with the correct pitch, visibility, and Fraunhofer-envelope shape — unchanged by the polarization/birefringence/grating feature work | gate: correlation with analytic Fraunhofer pattern >0.75, pitch within 1.5 detector pixels of `lambda*L/d`, visibility >0.85, closure <1e-3 |
| `test_dispersion.py` | Group index / GDD / TOD derivative API vs literature: fused silica n_g(800nm)=1.467145, GVD 36.16 fs²/mm; BK7 @1064; stencil clamping at table edges | 1e-4 (n_g), 1% (GVD) |
| `test_time_core.py` | Slab group delay `(n_g−1)L/c`; CW impulse response; calcite o/e group split; **bit-identity of opl + detector cubes with time tracking on vs off**; per-body path tally | closed-form/bit-identical |
| `test_pulsed_source.py` | Power XOR pulse_energy contract (all error paths); derived pulse block (P_pk=0.94E/τ, κ); case.json `source_pulse` echo | exact |
| `test_time_products.py` (21) | ∫I(t)dt == detected power for every product × both envelopes (coherent population via geometric power); marginal consistency (spectrogram/streak/cube vs profile); auto vs explicit window + clipped-kernel conservation; analytic smoother than histogram; CW delta at d/c; **traced FWHM vs τ(φ₂) broadening through 20mm fused silica**; auto-enable rule; engine routing; sub-bin-kernel splat regression (fs kernels on ns windows stayed finite) | conservation 1e-12; broadening 5% (products), 2% (the locked `test_gdd_budget` gate) |
| `test_gdd_budget.py` (5) | Budget GDD == `gdd_per_length·L̄` (the table's own numbers, 1e-3); τ_out formula exactness; **traced FWHM matches the budget's τ_out at 2% (THE locked gate)**; CW `--gdd-budget` forces tracking + Python routing; pipeline forwarding | 1e-3 / 2% |
| `test_spm.py` (8) | SPM RMS spectral broadening vs Agrawal `√(1+(4/3√3)φ_max²)`; multi-peak count ≈ φ_max/π+1; chirp tilt sign (red first) + monotonic t0; SPD normalization/supersession; grammar + γ·P_pk·L_eff derivation; sc_superk registry row | 10% (RMS); exact peak windows |
| `test_nlo.py` | d_eff tensor contraction (3m closed form 7e-16); KTP II 3.246 pm/V vs published 3.2; BBO θ_pm=29.211° vs 29.2°; registry load/validation | closed-form to 7e-16; published values 1–2% |
| `test_nlo_elements.py` (21) | Pockels crossed-polarizer `sin²(πV/2V_π)`; saturable α(I) curve + TPA law vs `exp(-α(I_in)L)`; Kerr focal shift vs `f_K=w²/(4n₂I₀L)`; chi2_process→shg_spec resolution; chi2_tensor rejection | sin² 1%; Kerr 5% |
| `test_shg_event.py` (7) | Vectorized η == scalar (incl. clamp); **η ∝ I·L² at Δk=0**; sinc² detuning sweep vs closed form; transfer closure (harmonic ≤ tally, total detected conserved); harmonic at λ/2 in the right spectral bin + `harmonic_strata` map; **child gopl continuity** (harmonic arrival rides the pump group delay); NLO bodies force Python routing | 1e-12 (vec); 1–2% (scaling/detuning); closure <1e-3 |
| `test_imaging.py` | Exit pupil vs paraxial stop image; singlet distortion; clipped-stop vignetting; Petzval field curvature; telecentric CRA≈0; Strehl PSF-peak vs Maréchal | per-product gates (see file header) |

**Measured double-slit result** (real completed run,
`results/doubleslit/dev/`, 400,000 rays, 1024×1024 detector, computed
directly from the saved detector cube): fringe pitch **117.2 µm** vs the
analytic `lambda*L/d = 633e-9 * 0.099 / 0.5e-3 = 125.3 µm`
(**≈0.7 detector pixels** off, at 11.7 µm/px — well inside the test's
1.5-pixel gate), **visibility 0.95**, and **correlation 0.89** with the
analytic Fraunhofer pattern `cos²(pi*d*y/(lambda*L)) * sinc²(a*y/
(lambda*L))` — matching the numbers this README was asked to ground.
Energy closure on that same run: source `Laser`, `closure_error ≈
2e-7` (gated at 1e-3). This regression is unchanged by every feature
described in §5.2-§5.8 above.

**Lensmaker focal validation** (`test_traced_focus_matches_lensmaker`,
real extracted `example.FCStd` BK7 lens, paraxial collimated 633nm
bundle): the traced axis-crossing position matches the thick-lens
equation `f = 1/((n-1)(1/R1 - 1/R2 + (n-1)d/(n R1 R2)))` (back focal
distance from the exit vertex) to **rel 5e-3, i.e. <0.5%**.

---

## 12. Performance and presets

`common.PRESETS`:

| Preset | rays (per source) | resolution | nlambda | spectral_bins | viz_rays |
|---|---|---|---|---|---|
| `quick` | 1e5 | 512 | 5 | 16 | 500 |
| `normal` | 1e6 | 2048 | 9 | 16 | 2000 |
| `detailed` | 1e7 | 4096 | 17 | 32 | 5000 |

`--preset` (default `quick` on `run_pipeline.py`; `run_trace.py` itself
defaults to the `quick` values individually if invoked directly without
any of `--rays`/`--resolution`/etc.) fills these; any of `--rays`,
`--resolution`, `--nlambda`, `--spectral-bins`, `--viz-rays` given
explicitly overrides just that one field.

**The dominant cost is the coherent gather, not the trace**, and it scales
roughly as `rays * resolution^2 * spr` (surviving coherent samples per
primary ray) — `common.estimate()`'s `gather_pairs`. The gather bills one
pass over (detector pixels × surviving samples) and PARTITIONS the
survivors across the (source, λ-stratum, pol-stratum) keys, so `nlambda`,
`n_pol_strata`, and `n_coherent_sources` do NOT multiply the gather cost —
they do multiply the accumulator/field MEMORY (§6.7) and, via the
per-key `M_eff` floor, the ray budget you need (§6.3; `n_pol_strata` is 2
for any scene containing an unpolarized source, 1 if every source
declares an explicit polarization, §5.2). A real `--dry-run`
estimate for `example.FCStd` at `normal` (1e6 rays, 2048², 9 wavelengths,
2 coherent sources) predicted **trace ≈ 306 s + gather ≈ 22,176 s (≈6.2
hours)** — `results/example/normal/case.json` (status stuck at
`"estimated"`, i.e. this estimate was never run to completion in this
repo). By contrast, a real **completed** `quick`-scale run at 60,000
rays/512²/3 wavelengths (`results/example/devsmoke/`) measured **trace
197.5 s + gather 67.3 s** wall time (`case.json["timing"]`) — illustrating
why `detailed`/`normal` at full 2048–4096 resolution is a serious
commitment and `quick` (or an explicit low `--resolution`) is the right
default for iterating on a scene. `--ray-differentials` (+96 B/ray) and
`--gather-occlusion` (an extra `n_tiles * M`-scale shadow-mask pass per
detector key) both add real cost beyond the base gather estimate above;
neither is currently folded into `common.estimate()`'s model, so budget
extra headroom when either flag is on.

**Backend**: `--backend auto` picks CUDA via torch when available. This
machine's GPU accelerates the gather step; both backends compute
`r`/phase in **float64** and reduce phase mod 2π before any float32 trig,
accumulating the field itself in **complex64** (fp32) — precision notes
in §6.6. `results/.calibration.json` in this repo records per-backend
`gather_pairs_per_s_<backend>` entries (recent medians: c_cuda ≈ 1.5e9,
torch and c_cpu ≈ 1.5e8, numpy ≈ 4e6 pairs/s) alongside per-scene
`trace_rps_*` throughput spanning ~2e4–1.5e5 rays/s (the wide spread
reflects very different scene/particle/roughness configurations, not a
single stable number — `common.calibrated_rate()` takes the **median**
of recorded entries of a given kind, falling back to
`FALLBACK_TRACE_RAYS_PER_S=2e5` or `FALLBACK_GATHER_PAIRS_PER_S=
{"torch": 1.1e9, "numpy": 5e7, "c_cuda": 6.7e9, "c_cpu": 3.3e8}` before
any real run has completed).

**Disk**: `detectors/*.h5` dominates a case's disk footprint — a single
512×512×16-bin detector cube in this repo's `results/example/devsmoke/`
is **~33 MB** per detector (`spectral_cube_mean` at float64, plus `mask`);
three detectors in that one run is ~100 MB. Every completed trace writes
the full spectral cube for every active detector regardless of
`--save-fields`; plan that part of the budget as `H * W * spectral_bins *
8 bytes * n_detectors * (2 if --seeds > 1 else 1)` (not itself estimated
in `case.json["estimates"]` — only the `--save-fields` addition below is).
`--save-fields` itself (§6.7) writes two more float64-complex (16-byte)
`H×W` arrays per `(source, lambda-stratum, pol-stratum)` gather key, on
top of the spectral cube, seed 0 only, for every detector eligible under
`--save-fields-detectors` (all of them if omitted) — budget accordingly
for polarization-heavy, high-resolution `--save-fields` runs.
`case.json["estimates"]["fields_h5_GB"]` in the dry-run estimate predicts
exactly this: `common.estimate()` is wired to the actual `save_fields`/
`--save-fields-detectors`-resolved detector count/gather-key count
(`n_coherent_sources * nlambda * n_pol_strata`)/resolution (see its
`field_bytes` comment for the exact formula) and is `0` whenever
`--save-fields` is off; treat it as a ~2x disk-budget aid, not a
byte-exact prediction (it ignores per-detector resolution/spectral-bin
differences and gather-occlusion overhead).

---

## 13. The C engine (`cengine/`)

A compiled OpenMP + CUDA implementation of the trace + coherent-gather
core, selected per case by `--engine {auto,python,c}` (default `auto`).
The Python engine in `scripts/raytracer/` is the PERMANENT REFERENCE:
`auto` routes a case to the C engine only when every feature the scene
uses has passed the side-by-side parity gates
(`scripts/raytracer/tests/test_cengine_parity.py`); anything else runs on
Python, and the choice + reason are logged and recorded in `case.json`
(`engine`, `engine_reason`). No feature is ever lost — unported features
simply keep their Python path.

Build (optional; without the binary everything runs on Python):

    cd cengine && ./build.sh        # cmake+ninja, gcc, OpenMP, CUDA 13

Ported (routing source of truth: `PORTED` in `scripts/raytracer/
cengine.py`): all analytic surfaces + mesh faces, trimmed geometry, the
scene-wide BVH (an acceleration the Python engine lacks), Fresnel,
TMM/table coatings, polarizers, spectral filters, the medium stack,
gratings (all models), Beckmann roughness + diffusers, ABg scatter
(g = 2), uniaxial birefringence (the effective-index kernel — used only
under `--biref-approx`; the default exact Lekner amplitudes route to
Python, §5.6), continuum-mode particle clouds, the
coherent Huygens gather (fused CUDA kernel + CPU twin; same fp64-phase
precision contract as the torch backend), gather occlusion, save-fields
(every detector — `--save-fields-detectors` is a Python-engine-only
subset restriction, §8.2), export-rays, ghost analysis, viz-pattern
overlays, multi-seed, the opt-in `--importance-aim` variance
reduction (unbiased birth culling), time products + the GDD budget
(P7 tranche 1), Igehy ray differentials (P7 tranche 2, oracle-pinned
1e-5 vs finite differences), and the intensity-dependent bulk NLO trio
saturable/TPA/Kerr (P7 tranche 2).

Still Python-routed (auto fallback; every token `detect_features()` can
emit that is absent from `PORTED`):

- **exact uniaxial birefringence** (`biref_exact`) — the DEFAULT
  Lekner-1991 amplitudes; the ported kernel serves only `--biref-approx`
- **natural optical activity** (`gyration`, e.g. the quartz rotator,
  §6.12b) and **biaxial crystals** (`biaxial`) with their exact
  **Berreman 4×4** entry (`berreman`)
- **χ² nonlinear optics** — SHG / Pockels (`nonlinear`, §6.12)
- **thermo-optic index shift** (`temperature` — any body off its
  material's reference temperature with a dn/dT model)
- **RCWA v2 grating tables** (`grating_table_v2`, complex per-order
  (λ,θ,φ) amplitudes)
- **Forbes Q-type aspheres** (`surface:qforbes`) and **Zernike
  figure-error surfaces** (`surface:perturbedsurface`, `figure_error`) —
  and any other `surface:*` token not in `PORTED`
- **beam / apodization sources** (`beam`, `apodization`)
- **scatter variants**: ABg g ≠ 2 (`scatter_g_ne_2`), transmissive BTDF
  (`scatter_btdf`), importance-sampled scatter (`scatter_importance`)
- **phase-carrying table coatings** (`coating_phase`)
- **extra CLI detector faces** (`--detector-face` →
  `extra_detector_faces`; the `detector_face` BODY property stays
  C-routable, §5.2) and **curved detectors** (`curved_detector`)
- **explicit-realization particle clouds** (`particles_explicit`)
- **parallel-transport polarization analysis** (`--pol-transport`)
- **time products through a crystal** (`time_directional_index`,
  directional group index)
- **legacy macro-angle rough Fresnel** (`rough_fresnel_macro`)

Determinism: the C engine's RNG is counter-based Philox4x32-10 (C-engine-only) keyed by
ray lineage — results are bit-identical across thread counts; the Python engine uses
numpy default_rng and agrees statistically only. It does
NOT reproduce numpy's streams; parity is deterministic (1e-9) for
non-random physics and statistical (2%) for MC
aggregates, single-seed; 1e-12 applies to emitted_W only. This is the same bar the demo-equivalence gate uses.

Performance: see `cengine/BENCHMARKS.md` (committed table). Headlines at
production settings: trace-bound scenes ~13x per stage; gather-bound
coherent scenes ~6.6x on the gather (fused CUDA vs torch).

SUNSET ROADMAP (project decision): through the post-merge shakedown the
torch gather remains the 100% fallback and three-way parity reference
(numpy/torch/CUDA). Once the C engine has proven itself in day-to-day
use, the torch backend (and its ~5 GB dependency) will be retired, and
eventually the Python compute paths sunset for routine use — the numpy
engine stays indefinitely as the slow, readable reference that the
parity suite runs against. Errors in the C engine never present as bare
segfaults: every failure carries context, a log
(`<case>/cengine/cengine.log`), typed exit codes, and crash backtraces.

## 14. Troubleshooting

- **FreeCAD `-c` needs a bare `--` before the script's own flags.**
  Without it, FreeCAD's own argument parser tries to interpret
  `--models`/`--model`/`--var` itself and aborts:
  `FreeCAD.AppImage -c script.py -- --flag value`.
- **FreeCAD `-c` mode runs the whole script TWICE per invocation** — a
  silent headless pass, then a GUI-spinup pass — before exiting 0. Every
  write in `extract_geometry.py`/`permute_model.py`/`make_test_scenes.py`
  is idempotent (deterministic tessellation, no wall-clock timestamps in
  the JSON), so this roughly doubles wall time but produces
  byte-identical output on rerun — don't be alarmed to see "wrote..."
  lines twice in the log.
- **No `if __name__ == "__main__":` guard in the FreeCAD scripts** — `-c`
  console mode sets `__name__` to the script's basename, not `"__main__"`,
  which would silently skip `main()` if guarded; these three scripts call
  `main()` unconditionally at module scope and use `os._exit()` (not
  `sys.exit()`, which FreeCAD's `-c` mode swallows) to force real exit
  codes.
- **`gather undersampled on <detector> for source/stratum <key>`
  (`GatherError`)**: `M_eff` (effective sample count) fell below
  `--min-eff-samples` (default 1000). The message reports the actual
  `M_eff`, the resulting speckle pedestal fraction, and an approximate
  `--rays` multiplier to fix it. The gather key is now `(source,
  lam_stratum, pol_stratum)` — an **unpolarized source needs roughly
  double the `--rays`** of an explicitly polarized one to reach the same
  `M_eff` per key, since its two polarization populations split the ray
  budget (§5.2/§6.3); a scene with several wavelength strata multiplies
  this further. Raise `--rays`, or pass `--no-gather-gate` if you
  deliberately want a fast, noisy smoke-test render.
- **`unknown polarizer %r` / `unknown filter %r` / `unknown coating %r` /
  grating `unknown registry entry %r`**: the name in the body's
  `polarizer`/`filter`/`coating`/`grating` property doesn't match any row
  in the corresponding `opticalproperties/` registry (`.miepol`/`.miefilt`/
  `.miecoat`/`.miegrat`). These checks fire at
  **trace time** (`Scene.__init__`), not at extract time — the error
  message lists every name the registry actually loaded, so diff it
  against your body property string and against the registry file (§7).
- **Asphere `surface_override` declaration mismatch dies at extract
  time**: `surface_override=asphere declaration does not match the
  actual face geometry (max residual ... tolerance 1.0 um)` means the
  declared `R`/`k`/`A4...` doesn't reproduce the FreeCAD-authored face's
  actual sag to within 1 micron — re-derive the coefficients from the
  same design equations used to build the face, or loosen `r_max` if the
  mismatch is only at the aperture edge. This check runs in
  `extract_geometry.py`, well before any trace-time error would surface.
- **A detector reports 0 mW (or a tiny fraction of the expected power)
  on a rotated / off-axis detector**: the recorded face is auto-picked as
  the one whose centroid is closest to the world origin, which on a
  tilted or decentered detector (a folded-telescope eyepiece screen) is a
  thin *edge* face nearly parallel to the beam — so almost nothing lands
  on it. Pin the real sensing face with the **`detector_face` body
  property** (a bare `'FaceN'` or a full `'Body.Tip.FaceN'` id, §5.1/§5.2):
  it replaces the primary face at extract time, needs no extra screen, and
  keeps the scene C-engine-routable. The additive CLI `--detector-face`
  works too but forces the Python engine (`extra_detector_faces` is not a
  ported feature). The demos that fold (`newtonian`, `dobsonian`, …) bake
  a `detector_face` pin via `make_demos.resolve_detector_pins`.
- **Seam warnings on cemented interfaces (e.g. an achromat doublet or a
  Wollaston prism)**: this project's cemented-interface scenes model the
  cement layer as a thin (typically 5 µm) air (or matched-index) gap
  between two solids rather than a single fused body — if you see
  `seam_loss` or trim-disagreement warnings around such an interface,
  check that the gap is present and non-zero (a literal zero-thickness
  gap makes the two faces coincide, which is exactly the degenerate case
  §13's `seam_loss` bucket exists to catch) and that both sides' trims
  describe the same footprint. **The gap convention fails wherever the
  internal incidence exceeds the critical angle** (a 45° beamsplitter
  interface in BK7: critical angle 41.2° — the gap TIRs the transmitted
  arm; there is no frustrated-TIR physics). For those interfaces use
  **proper nesting** instead: one solid strictly inside another is
  legal (the extractor classifies it `validation.nested_solids`; the
  tracer's LIFO medium stack handles enter/exit), so a thin coated
  plate nested inside a single cube gives a glass-glass interface where
  a measured split table applies exactly — this is how the catalog
  `bs_cube`/`pbs_cube` are built. PARTIAL overlap is still rejected.
  Related trace-time honesty rule: a TABLE coating evaluated past the
  critical angle now folds its transmitted power into the reflected
  side (TIR) instead of emitting a degenerate grazing child.
- **`seam_loss` in `audit.json`/`report.json`**: a ray's tracked medium
  stack disagreed with the direction it crossed a face — one face's trim
  rejected a true hit while a neighboring face's trim accepted a grazing
  one. Expected to be near-zero (it's rare by construction); a
  *substantial* `seam_loss` fraction of a source's emitted power means two
  adjacent trimmed faces don't share a boundary cleanly — check the
  originating FreeCAD geometry for a hairline gap or overlap at that seam
  (see the cemented-interface note above for one common cause).
- **Model validation / closure failures at trace time**: `case.json`
  and `audit.json`'s `closure_ok`/`closure_error` per source are gated at
  1e-3 — a failure prints a WARNING at trace time and `run_trace.py`
  exits code `3` (not a crash; outputs are still written). Check
  `by_surface_W`/`by_body_W` in `audit.json` for which body/face is
  eating unaccounted power, and re-verify any custom `--grating`/`--rough`
  spec you passed (both feed their own `absorbed_surface` credits — a
  spec typo elsewhere can silently zero out a bucket you expected); a
  crossed-polarizer or narrow-band filter scene will legitimately park
  most power in `polarizer_absorbed`/`absorbed_bulk` rather than
  indicating a bug.
- **Astronomically large detected power (e.g. `1e8` W out of a 1 mW
  source) on a nested-cylinder / curved-glass-to-liquid interface, Python
  engine only** — FIXED (samples-instruments round), but the failure
  signature is worth recognizing if you're on an older checkout or a
  from-scratch reimplementation of `fresnel.cos_theta_t`: a weakly
  absorbing INCIDENT medium (e.g. water, k~1e-8) into an exactly lossless
  far medium (e.g. decalin/glass) picks up a `~1e-13` numerical-dust
  imaginary part on the transmitted-cosine radicand; the old unconditional
  `Im(n2·cos_t) >= 0` decay-branch rule flipped that dust into a spurious
  evanescent classification, and the near-cancelling Fresnel denominator
  this produces gives `|rs| ~ 15` — squared and looped around a curved
  interface, that is an `O(1e16)` closure explosion in a handful of
  bounces. The fix applies the radiation condition
  (`Re(n2·cos_t) >= 0`) instead, whenever the imaginary part is at
  numerical-dust scale relative to the whole quantity (`fresnel.py`,
  `cos_theta_t`). The C engine's branch rule already matched the fix, so
  this only ever showed up under `--engine python`/`--biref-approx`-free
  Python fallback — `--engine auto` silently masked it on any scene the C
  engine could route.
- **A many-interface stack (deep nesting, several particle-medium
  crossings) loses a large, oddly-specific fraction of emitted power into
  `truncated_generation` well before `--max-reflections` should matter,
  Python engine only** — FIXED (samples-instruments round, found by the
  `nested4` depth-4 spike at 37.8% loss at 60k rays). The old termination
  valve was a fixed **global** pop budget (`64*(max_reflections+2)`)
  shared across every source and consumed by `batch_size` chunk splits, so
  a scene with enough live interfaces could exhaust it while rays were
  still legitimately eligible to continue. It is now a **per-lineage hop
  cap**: each batch carries its own ancestry step count and chunk splits
  inherit it unchanged, so splitting the work never changes the outcome —
  a truncation today means a lineage genuinely exhausted 512 segments, and
  it is reported by name with the exact power lost, not silently folded
  into the ordinary ledger.
- **`[trace] REFUSED: … .lock.json` (exit code `4`)**: a second writer on
  a case that already has a live run is refused rather than corrupting it
  (one writer per case, `common.acquire_case_lock()`). Rerun when the
  first run finishes, or remove `<case>/.lock.json` if its owner is dead;
  `--resume`/`--extend` steal only a dead-owner lock. The GUI opens live
  cases read-only in monitor mode instead of writing.
- **Particle count surprises**: remember **`phi` is a MASS fraction**
  (§9) — "too few particles" or a `ValueError: particle cloud is empty`
  almost always means `phi` needs to go up, especially for dense
  particles (e.g. water, TiO2) suspended in a light host medium (air).
- **CUDA OOM on the gather step**: `accumulate_torch`/`points_torch`
  chunk both the pixel dimension (`pixel_chunk`, default 16384) and the
  sample dimension (`sample_chunk`, default 8192) specifically to bound
  live GPU memory to a handful of `(pixel_chunk × sample_chunk)` float64
  intermediates at a time — these aren't currently exposed as CLI flags,
  so if you hit OOM on an unusually large scene, the fix today is either
  `--backend numpy` (slower, host RAM instead) or lowering `--resolution`/
  `--rays` for that case (`--gather-occlusion`'s shadow mask adds its own
  `n_tiles * M`-scale memory on top, §6.5); editing the hardcoded
  `pixel_chunk`/`sample_chunk` defaults in `gather.py` is the escape
  hatch if neither is acceptable.
- **Visible noise/graininess in an image at low ray counts**: this is the
  MC estimator noise floor (§6.2), not a bug — check
  `case.json["gather"][...]["noise_floor_W_per_px"]` for the detector/
  source/stratum in question and compare it against the peak irradiance
  in `report.json`; if the ratio is uncomfortably large, raise `--rays`
  (which raises `M_eff` and lowers this floor) rather than trying to
  post-process the noise away.
- **`.FCStd` files are zip archives** — inspect one without opening
  FreeCAD: `unzip -p example.FCStd Document.xml` (objects, spreadsheet
  cells, body properties, expressions).
- **`AttributeError`/`FileNotFoundError` in `post_process.py` or
  `make_viz.py`** almost always means the upstream stage hasn't completed
  yet (or completed under a different case name than you're pointing at)
  — check `case.json["status"]` (`"estimated"` = `--dry-run`'d or
  interrupted; `"completed"` = safe to post-process/visualize) before
  debugging further.
- **`--strict` at extract time**: pass it when authoring a new scene to
  turn "face fell back to mesh representation" from a warning into a hard
  error immediately. This is distinct from the trace-time
  `--strict-analytic` flag (§5.8), which controls whether the tracer
  accepts an already-extracted mesh face (default: yes, incoherent BVH
  tracing) or hard-errors on it (v1 behavior) — use extract-time
  `--strict` to catch a surface that *should* have canonicalized to an
  analytic primitive, and trace-time `--strict-analytic` to forbid mesh
  faces outright even when the extractor accepted them (e.g. the
  deliberately-mesh `mesh_freeform` test scene, §10).
