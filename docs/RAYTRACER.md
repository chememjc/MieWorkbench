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
`scripts/make_test_scenes.py` can additionally author 24 more FreeCAD test
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
| trace | `run_trace.py` | optics env python | `geometry/<stem>/model.json`, `opticalproperties/*.csv` | `results/<stem>/<case>/{case.json,audit.json,rays.npy,detectors/*.h5}` |
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
| orchestrators (`run_pipeline.py`, `sweep_variants.py`) | system `python3` | stdlib only — never imports FreeCAD/numpy/paraview |

These paths are hardcoded in `scripts/common.py`
(`FREECAD_APPIMAGE`, `OPTICS_PYTHON`, `PVPYTHON`). Run `python3
scripts/common.py` for a self-check (verifies the three interpreter paths
exist, `opticalproperties/materials.csv` exists, and a battery of
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
│   ├── materials.csv                   #   bulk n(lambda)/k(lambda) database
│   ├── nk/*.csv                        #   tabulated n,k spectra (metals, water, TiO2)
│   ├── coating/coatings.csv (+tables/) #   TMM stacks AND measured Rs/Rp/Ts/Tp tables
│   ├── polarizer/polarizers.csv (+tables/)  # linear/circular diattenuator tables
│   ├── filter/filters.csv (+tables/)   #   bulk spectral filters (Beer-Lambert)
│   ├── grating/gratings.csv (+tables/) #   lamellar/Kogelnik/Dammann/table registry
│   └── birefringence/uniaxial.csv      #   calcite/quartz/sapphire o/e crystal pairs
├── scripts/
│   ├── common.py            # paths, PRESETS, CLI spec parsers, model.json validator,
│   │                        #   sweep semantics, runtime/memory estimator
│   ├── extract_geometry.py  # FreeCAD headless: .FCStd -> geometry/<stem>/model.json + STLs
│   ├── permute_model.py     # FreeCAD headless: sweep spreadsheet alias(es) -> basemodels/*.FCStd
│   ├── make_test_scenes.py  # FreeCAD headless: authors 25 validation FCStd scenes (§10)
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
│                            #   pol_mode (0=isotropic/ordinary, 1=extraordinary o/e-split ray),
│                            #   rel_power (power/birth_power in [0,1] — drives --dim-rays),
│                            #   opl0_m, opl1_m (optical path Σn·ds at segment start/end; t = opl/c
│                            #   drives the tracer-bead animation; escaped-ray stubs get a synthetic
│                            #   opl1 = opl0 + n·0.25 m matching the drawn stub)
├── detectors/
│   └── <safe_label>.h5      # spectral cube (mean [+std if --seeds>1]) + grid basis (§5.11)
│                            #   + optional fields/<key>/{Ex,Ey} complex maps if --save-fields
├── images/
│   ├── det_<label>.png              # wavelength-colored sRGB irradiance
│   ├── det_<label>_lin.png          # linear grayscale irradiance (colorbar, W/m^2)
│   ├── det_<label>_log.png          # log10 grayscale irradiance
│   ├── det_<label>_profiles.png     # horizontal/vertical cuts through the peak pixel
│   ├── stokes_<label>_<key>.png     # 2x2 S0/S1/S2/S3 panel — only if --save-fields (§6.5)
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
`--save-fields` (§6.5, §8.2) — every other output above is unconditional.

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

### 4.2 Note: `run_pipeline.py`'s viz stage always uses make_viz.py's defaults

`run_pipeline.py`'s internal `viz_cmd()` builder passes only `--case-dir`
and `--model-json` to `make_viz.py` — it does **not** forward
`--views`/`--resolution`/`--smoke`. Driving the pipeline through
`run_pipeline.py` therefore always renders all six views
(`overview3d`, `top`, `side`, `detector_closeup`, `turntable`,
`rays_polmode`) at 1920×1080 (2048×2048 for `detector_closeup`). The only
display options `post_cmd()`/`viz_cmd()` forward beyond
`--case-dir`/`--model-json` are `--dim-rays`/`--dim-rays-floor`
(attenuation dimming: segment opacity = P/P_birth, linear or sqrt curve,
optional percent floor — applies to `rays_xy.png` and the 3D ray
renders); `post_process.py`'s `--viz-generations` remains reachable only
by invoking that script directly. To pick a subset of views, a different
resolution/smoke test, or a decluttered ray plot, invoke
`make_viz.py`/`post_process.py` directly (§8) on an already-completed
case directory.

---

## 5. Model authoring contract

This is the user-facing contract every `.FCStd` model must follow for
`extract_geometry.py` to accept it (enforced by
`common.validate_model()`, called both at extract time and by every
downstream stage that loads `model.json`). The extractor always emits
`schema_version: 2`, which adds `polarizer`/`polarizer_axis`/`filter`/
`crystal_axis`/per-face `coating`+`roughness`+`grating` maps/`asphere`
surfaces on top of the v1 contract; `validate_model()` accepts both
versions.

### 5.1 Body tagging (group "Base" custom properties)

`extract_geometry.py`'s `classify_body()` reads these `App::Property*`
fields on each `PartDesign::Body`:

| Property | Type | Meaning |
|---|---|---|
| `power` (mW) + `lambdac` (nm) | Float | presence of **both** marks the body a **source** (checked first, before `material`) |
| `material` | String | a row name in `opticalproperties/materials.csv`, a crystal name in `birefringence/uniaxial.csv` (§5.6), `"detector"`, or `"none"`/absent |
| `coating` | String, optional | a coating name (whole-body) or a per-face map `'Face3=MgF2;Face5=x'` (§5.3) |
| `polarizer` | String, optional | a row name in `opticalproperties/polarizer/polarizers.csv` (§5.3) |
| `polarizer_axis` | String `'x,y,z'`, optional | body-local transmission-axis vector, default `0,0,1` (§5.3) |
| `filter` | String, optional | a row name in `opticalproperties/filter/filters.csv` (§5.3) |
| `crystal_axis` | String `'x,y,z'`, optional | body-local optic-axis vector for a birefringent `material` (§5.6), default `+x` |
| `surface_override` | String, optional | per-face `'FaceN=asphere:R=..;k=..;A4=..;...;r_max=..'` (§5.7) |
| `mirror` | Float, optional, [0,1] | achromatic partial-reflector fraction (§5.3 precedence) |
| `absorbance` | Float, optional, [0,1] | fraction of the physical (non-mirror) remainder absorbed |
| `roughness` | Float or String, optional | whole-body RMS nm, or a per-face map `'Face1=200:lcorr=5;Face2=50'` (§5.4) |
| `diffuser` | String, optional | ground glass: `'grit:120'` \| `'slope:0.08'` \| `'@dg_600'`, whole-body or per-face map `'Face2=@dg_600'` (§5.4.1) — mutually exclusive with `roughness` on the same face |
| `grating` | String, optional | a per-face map `'Face2=600:v:orders=-1..1'` or `'Face2=@registryname'` (§5.5) — must name specific faces, not the whole body |
| `polarization` | String, optional | source-only: `'unpolarized'` (default) `'linear:<deg>'` `'circular:left|right'` `'elliptical:<psi>:<chi>'` (§5.2) |
| `lambdamin`, `lambdamax` (nm, optional), `coherent` (bool, default False) | — | source-only, see §5.2 |

Classification (`classify_body`, in this priority order):
1. `power` **and** `lambdac` both present → **source**.
2. `material` missing/empty or `"none"` (case-insensitive) → **ignored**
   (skipped entirely; extractor logs `"[Label] is ignored"`).
3. `material == "detector"` (case-insensitive) → **detector**.
4. Otherwise → **optic**, and `material` must be a row in
   `opticalproperties/materials.csv` **or** a crystal name in
   `opticalproperties/birefringence/uniaxial.csv` (a name that resolves to
   neither is a hard error at trace time when the `Scene` is built).

`mirror`/`absorbance` values outside [0,1] are **not** rejected — they are
capped to [0,1] with a loud warning (`capped01()`), and `model.json`
contract validation (`common.validate_model`) then hard-errors if a body
still shows an out-of-range value after that capping (an extractor bug,
not a user error). `polarizer`/`filter`/`crystal_axis` set on a
non-`optic` body are ignored with a warning (they are only meaningful on
optics); `crystal_axis` is **always emitted** for every optic body
(default local `+x` if the property is absent) since the tracer's own
local-+x default would otherwise be ambiguous without the body's
Placement rotation applied. A model needs **at least one source** and
**at least one detector**, or extraction/validation fails outright.

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
   `layers`/`table` may be set per `coatings.csv` row, §7.3): TMM stacks
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
grating/gratings.csv` (§7.6) at scene-build time; an unrecognized name is
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
| `table` | Per-order `eta_s`/`eta_p` interpolated at the ray wavelength from a CSV (§7.6) | measured, polarization-resolved | exact linear-interpolation roundtrip; hard error outside the tabulated range; an order absent from the table reads 0 |

Every model rotates the incident Jones vector into the grating face's own
local (s,p) interface basis (the same `fr.pol_basis`/`fr.rotate_jones`
machinery the ordinary Fresnel path uses) **before** scaling each
diffracted child's amplitude by `sqrt(eta_s)`/`sqrt(eta_p)`, so children
stay power-exact against the incident field to 1e-12 in the test suite.
Diffracted orders inherit OPL continuously — there is no modeled
inter-order grating phase offset beyond that. Ray differentials (§6.4)
are not tracked through a grating ("order transport unimplemented"); the
gather falls back to the source-referenced patch area for diffracted
rays. **RCWA (rigorous coupled-wave analysis) is out of scope** — these
four closed-form/tabulated models are the complete set; `bragg_kogelnik`
is a thin-hologram transmission approximation only (reflection-geometry
volume Bragg gratings, which need the tanh/sinh reflection coupled-wave
solution, are not modeled).

### 5.6 Uniaxial birefringence (`crystal_axis`)

Setting `material` to a crystal name in `opticalproperties/birefringence/
uniaxial.csv` (currently `calcite`, `quartz`, `sapphire` — **not** the
underlying `<name>_o`/`<name>_e` materials.csv row names, which the
registry resolves internally) marks a body birefringent
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

At an interface, the incident Jones vector is decomposed **first** into
the o/e eigenbasis (`bir.eigenbasis()`), and each channel's own R/T is
applied using an **effective-index Fresnel approximation** (`n_o` for the
o-channel, the angle-dependent `n(theta)` for the e-channel) — the true
anisotropic-boundary Fresnel solution is not solved. This decompose-first
order matters: applying a single Fresnel R/T to the whole field instead
(rather than per-channel) broke the energy ledger's closure at the ~1e-2
level during development; decomposing first and applying each channel's
own coefficients keeps R+T=1 per channel, with any dropped cross-terms
absorbed into `absorbed_surface` via the exact power difference (so
overall energy closure is still exact, though the per-channel phase is an
approximation without a stated numeric error bound beyond that). `mirror`/
`absorbance` still apply to birefringent bodies; **coating and roughness
are not modeled on a birefringent face** (bare Fresnel is used instead,
with a one-time warning).

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
biaxial crystals, optical activity, and gyrotropy are not modeled; an
e-mode ray that hits a **non-birefringent** face (e.g. a body nested
inside a crystal) is silently downgraded to ordinary-index propagation
with a one-time warning ("documented approximation"); ray differentials
are not tracked through an o/e split (falls back to source-referenced
patch area, like gratings). The closed-form birefringence math
(`test_birefringence.py`, 14 tests) is fully pinned, but the calcite-
displacer/Wollaston/waveplate FreeCAD scenes in `make_test_scenes.py`
have no dedicated end-to-end pytest gate the way the double-slit scene
does (§10/§11) — treat them as manually-validated reference scenes.

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

Detectors are **planar only** — `DetectorGrid.__init__` raises
`NotImplementedError` for a non-`Plane` surface (§6.2), and (separately)
`Scene.__init__` raises `NotImplementedError` if a detector's recorded
face is mesh-type (§5.8) — curved or freeform detector screens are not
supported.

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
  literature to 0.05°, §5.6); the effective-index Fresnel amplitude
  approximation used per o/e channel is documented in §5.6, not claimed
  exact.
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
  polarization-blind (`eta_s == eta_p`); only `bragg_kogelnik` and `table`
  are polarization-resolved. Diffracted orders carry no relative
  inter-order phase offset beyond the continuously-inherited OPL. RCWA is
  out of scope entirely (§5.5).
- **Roughness Fresnel at the microfacet-local angle is the default but
  the legacy macro-angle mode still exists** (`--rough-fresnel macro`),
  which evaluates the reflectance/transmittance feeding the scattered
  lobes at the nominal (specular) incidence angle rather than the true
  microfacet-local angle — kept only for A/B comparison (§5.4).
- **Tabulated (measured) coatings carry no phase.** Their amplitude
  coefficients borrow the bare-interface Fresnel phase; only TMM layer
  stacks carry real coherent coating phase (§5.3).
- **Effective-index Fresnel approximation for birefringent interfaces.**
  Uniaxial o/e Fresnel amplitudes use isotropic Fresnel formulas with
  `n_o`/`n(theta)` per channel rather than the true anisotropic-boundary
  solution; energy still closes exactly via the ledger, but per-channel
  phase/amplitude near-normal vs. grazing incidence is not claimed exact
  (§5.6). Absorbing (dichroic) uniaxial crystals, biaxial crystals,
  optical activity, and gyrotropy (e.g. quartz's rotary power along its
  own optic axis) are **not modeled at all**.
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
- **Detector screens are planar only** (`DetectorGrid` hard-errors on a
  non-Plane surface).
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

---

## 7. Optical properties library (`opticalproperties/`)

`scripts/raytracer/optprops.py` (loaders) plus `scripts/raytracer/
materials.py` (materials + coatings) load the whole tree in one call,
`optprops.load_optical_properties(root, db)`, used by `run_trace.py`;
individual `load_*` functions are importable directly for tests. Every
registry hard-validates its referenced table CSVs at load time and
requires a non-empty `reference` citation column — the same policy as
`materials.csv` — and table interpolation **never extrapolates**
(`interp_hard()` raises `MaterialError` outside the tabulated range).
Coatings/polarizers/filters/gratings/birefringence are all **optional**:
a trimmed library missing any of these CSVs loads that category as an
empty dict rather than failing. `--optical-properties PATH` (run_trace.py/
run_pipeline.py) overrides the library root; there is no per-category
accessor beyond indexing the loaded dicts directly (e.g.
`props.polarizers["lp_test"]`) except for materials, which get
`MaterialDB.get(name)` (case-insensitive, `KeyError` listing every
available name).

### 7.1 `materials.csv`

One row per material. Columns: `name, class, model, p1..p6, nk_file,
density_kg_m3, transmission_um_min, transmission_um_max, notes,
reference`.

- `class` ∈ `{gas, glass, liquid, polymer, metal, oxide, film, special}`
  (`VALID_CLASSES`) — organizational only, not cross-validated against
  any electrical role (this project has none).
- `model` ∈ `{sellmeier, cauchy, constant, tabulated}` (`VALID_MODELS`):
  - `sellmeier`: `n^2 = 1 + sum_{j=1..3} p_j * lam_um^2 / (lam_um^2 -
    p_{j+3})`; each `C_j = p_{j+3}` must be `> 0`.
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
- `opticalproperties/nk/*.mienk` tables (`wavelength_nm, n, k`, strictly
  increasing wavelength, 18 total) back the `tabulated` model: metals
  (aluminum, chromium, copper, gold, nickel, platinum, silver, titanium,
  tungsten), semiconductors (gaas, germanium, silicon, sic), IR materials
  (kbr, nacl), and water. **Birefringent crystal materials are not selected directly** —
  a body sets `material=calcite` (the crystal name in `uniaxial.miebrf`, §7.6),
  which resolves to the `calcite_o`/`calcite_e` pair internally.

168 materials ship today (expanded from 24): optical glasses (41 Schott/Ohara
crowns/flints), metals/semiconductors/IR windows (17), polymers/liquids/gases/
biological (35), coating-film materials (5), crystals with o/e pairs (46 uniaxial
axis rows), plus foundational `vacuum`, `air`, `bk7`, `fused_silica`, `sapphire_o/e`,
`water`, `glass`, `polystyrene`, `latex`, `pmma`, `polycarbonate`, `tio2`, `mgf2`,
`sio2_film`, `detector`, `calcite`, `quartz`, `sf5`, and `fiber_core_na22`. All
entries are spot-checked against authoritative sources (NIST, peer-reviewed
publications, manufacturer datasheets) per §7.8 citation policy.

**To add a material**: append a row with a unique `name`; pick `class`
descriptively; pick `model` and supply the required parameters for it
(Sellmeier's three `C` values must be `>0`; tabulated needs `nk_file`);
set `density_kg_m3 > 0`; fill `reference` (required); optionally note a
**spot-check** in `notes` — the existing rows follow the pattern "verified
n(lambda)=X vs target Y, matches within Z" (e.g. `calcite_o`'s row:
"verified n(590nm)=1.65830 vs target 1.658"). This is the project's
**spot-check policy**: every parametric fit should be validated against
at least one literature/catalog reference point, and that check should be
pinned in `scripts/raytracer/tests/test_materials.py` (21 tests spot-
check bk7, fused_silica, mgf2, water — including its ~975nm absorption
shoulder — sapphire, polystyrene, aluminum's visible-range n/k and its
monotonic-into-IR k rise, plus hard-error behavior for out-of-range
tabulated lookups, unknown materials, malformed CSV rows, and missing
references).

### 7.2 `coating/coatings.csv` (+ `coating/tables/*.csv`)

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
**`Rs+Ts<=1`, `Rp+Tp<=1` per row**). 38 coatings ship (expanded from 10):
TMM stacks (AR: MgF2 at 550/633/1064nm, quarter-quarter V-coats at 532/633/1064nm,
3-layer QHQ W-coats at 550nm; HR: dielectric 11–15 layer stacks at 532/633/1064nm),
measured table models (protected mirrors, dichroic/laser elements, standard
45°-AOI elements). TMM stacks apply via `thinfilm.py`'s Macleod-formulation
characteristic-matrix method (any number of layers, any angle, complex indices);
zero layers reduces identically to bare Fresnel (tested to 1e-12). See §5.3
for the phase caveat on measured tables.

### 7.3 `polarizer/polarizers.csv` (+ `polarizer/tables/*.csv`)

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
`dammann` — validated strictly increasing, each in `(0,1)`). 8 gratings ship
(expanded from 3): lamellar ruled gratings (1200 l/mm first entry exercising
the `lamellar` model), Bragg/VPH (volume Bragg grating, VPH Kogelnik, ESO)
Dammann diffractive optics, transmission gratings, echelle, and ruled
blazed gratings. Table-model entries are `wavelength_nm, order, eta_s, eta_p`
columns, validated so summed per-wavelength, per-polarization order efficiencies
never exceed 1. See §5.5 for the model formulas.

### 7.6 `birefringence/uniaxial.miebrf`

Columns `name, n_o_material, n_e_material, reference, notes`. 13 uniaxial
crystals ship (expanded from 3): calcite, quartz, sapphire, plus LiNbO3,
LiTaO3, YVO4, β-BBO, α-BBO, KDP, ADP, rutile TiO2, TeO2, MgF2 (e-ray
addition enabling waveplate/Rochon prism designs), each pointing at a
`materials.miemat` o/e pair (§7.1) with spot-checked birefringence values.
See §5.6 for the physics model.

### 7.7 `detector/detectors.miedet` (+ `detector/tables/*.mietab`)

Columns `name, table_csv, reference, notes`. Tables are `wavelength_nm, qe`
(quantum efficiency, values in (0,1]). One detector QE curve ships: `hamamatsu_s1223`
(Si PIN photodiode, 4 datasheet points, 660–960 nm). When a detector body carries a
`qe_curve` property naming a registered curve, `post_process.py` reports a
`qe` block per detector: `photocurrent_A` (R(λ)=QE·qλ/hc weighting of the
spectral cube), `qe_weighted_power_W`, and `coverage_frac` (the fraction of
detected power whose bin centers lie inside the QE table's range — QE is
zero-filled outside the table rather than extrapolated, and coverage_frac
makes that truncation visible). No CLI flag; it is driven entirely by the
body property. The separate `--photometric` flag produces lux maps using the
CIE Ȳ=V(λ) table in `raytracer/detector.py`.

### 7.8 Citation policy

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
    [--dry-run] [--seeds N] [--rays F] [--resolution N] [--nlambda N]
    [--spectral-bins N] [--max-reflections N]
    [--viz-rays N] [--viz-density F] [--backend {auto,torch,numpy}]
    [--rough-fresnel {micro,macro}] [--ray-differentials] [--gather-occlusion]
    [--no-pol-scatter] [--mesh-flat-normals] [--save-fields] [--strict-analytic]
    [--optical-properties PATH]
    [--source-face SPEC]... [--detector-face SPEC]...
    [--grating SPEC]... [--rough SPEC]... [--particles SPEC]
    [--particle-threshold F] [--suppress-body NAME]...
    [--photometric] [--spectrometer]
    [--dim-rays {off,linear,sqrt}] [--dim-rays-floor PCT]
    [--keep-going] [--print-only]
```

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
(wavelength resolution quantized by `--spectral-bins`). `--keep-going`
turns a stage failure into a `FAILED: <tag>` notice and a skip to the next
model (process still exits nonzero if anything failed). `--print-only` composes
and prints every stage command **without running anything**. Extract runs
**once** for the whole model batch (one FreeCAD launch handles every
model); trace/post/viz then loop **sequentially** per model (a single
trace can already saturate every core/GPU). **`post_cmd()`/`viz_cmd()`
forward nothing beyond `--case-dir`/`--model-json`** — `post_process.py`'s
`--viz-generations` and `make_viz.py`'s `--views`/`--resolution`/`--out`/
`--smoke`/`--skip-vtkexport` are unreachable through `run_pipeline.py`;
invoke those two scripts directly for that (§4.2, §8.3). Logs:
`results/log.extract` (batch), `results/log.permute-<stem>` (if swept),
`results/<stem>/<case>/log.{trace,post,viz}` (per model/stage).

### 8.2 `run_trace.py` (optics env python — the solver)

```
/home3/optics/env/bin/python scripts/run_trace.py
    --model-json PATH --case-dir DIR
    [--rays F=1e5] [--nlambda N=5] [--resolution N=512] [--spectral-bins N=16]
    [--max-reflections N=6] [--power-floor F=1e-4]
    [--seeds N=1] [--seed0 N=42] [--backend {auto,torch,numpy}=auto]
    [--viz-rays N] [--viz-density F=1.0] [--viz-rays-max N=20000]
    [--ray-differentials] [--no-pol-scatter] [--rough-fresnel {micro,macro}=micro]
    [--source-face SPEC]... [--detector-face SPEC]...
    [--grating SPEC]... [--rough SPEC]...
    [--particles SPEC] [--particle-threshold F=2e5]
    [--suppress-body NAME]...
    [--min-eff-samples F=1000.0] [--no-gather-gate]
    [--save-fields] [--gather-occlusion] [--optical-properties PATH]
    [--strict-analytic] [--mesh-flat-normals] [--dry-run]
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
`micro` for `--rough-fresnel`) except where noted. `--particle-threshold`
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

### 8.3 `post_process.py` / `compare_runs.py` / `make_viz.py` (rendering)

```
/home3/optics/env/bin/python scripts/post_process.py \
    --case-dir DIR --model-json PATH [--viz-generations N]
    [--dim-rays {off,linear,sqrt}] [--dim-rays-floor PCT]
```
Requires `case.json["status"] == "completed"`; fully rerunnable without
re-tracing (reads only `case.json`/`audit.json`/`rays.npy`/
`detectors/*.h5` + `model.json`). `--viz-generations N` declutters
`plots/rays_xy.png` to reconstructed-generation `<= N` segments only
(default: every generation, unchanged behavior) — useful for scenes with
many reflection/diffraction/o-e-split generations where the 2D plot would
otherwise be an unreadable tangle. `--dim-rays` switches `rays_xy.png`'s
segment alpha from the default ensemble 95th-percentile scaling to each
segment's `rel_power` (power relative to its own ray's power at the
source — linear, or sqrt for a perceptual curve; `--dim-rays-floor` sets
a minimum opacity percent); falls back with a warning on a 10-column
`rays.npy` predating the `rel_power` column. Unconditionally also renders (when the
relevant body properties are present in the model) `polarizer_<name>.png`/
`filter_<name>.png`/`grating_<name>.png` per-element response-curve plots
and (when `--save-fields` produced `fields/` groups, §6.5)
`stokes_<label>_<key>.png` + `dop_<label>.png` polarization maps — none of
these have their own CLI toggle; they are silent no-ops when the relevant
inputs aren't present.

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

---

## 9. Particle clouds

`--particles` spec (`common.parse_particles_spec`):
```
box=x0,y0,z0:dx,dy,dz;material=NAME;phi=F;median_um=F;gsd=F[;seed=N]
```
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

25 buildable scenes total: `doubleslit` (the original scene, documented
in §5.10, still built with unchanged parameters — d=0.5mm slit
separation, 633nm, L=99mm plate-to-screen gap, fringe pitch λL/d=125.3µm)
plus 24 more registered in the `SCENES` metadata dict, spanning lenses,
polarization, birefringence, gratings, filters/coatings, and a
deliberately non-analytic mesh face:

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
| `pbs_cube` | 20mm BK7 PBS cube from two 45° prisms, `pbs_visible_45` coating on the hypotenuse, unpolarized 550nm | coated-interface polarization splitting (s/p separation) | qualitative: transmitted arm ~p-pol, reflected arm ~s-pol |
| `calcite_displacer` | 10mm calcite slab, `crystal_axis` at 45° in x-z, unpolarized 590nm Ø0.5mm beam | birefringent walk-off (o/e spatial displacement) | n_o=1.65830, n_e=1.48611, walk-off=6.232°, displacement=1.0919mm |
| `wollaston` | Two 30° calcite wedges, orthogonal optic axes, 5µm air gap, 590nm unpolarized | birefringent-wedge polarization beam-splitting angle | split angle = 2(n_o−n_e)tan(30°) = 11.392° |
| `filter_bandpass` | `bp_550_40` bandpass filter on a 3.5mm BK7 slab, broadband 450-650nm source | wavelength-dependent filter transmission | qualitative (center 550nm, band 450-650nm) |
| `hot_mirror` | BK7 plate at 45° AOI, `hot_mirror_45` coating, broadband 450-1000nm, two detectors | angle-dependent dichroic spectral splitting | qualitative visible/IR split (center 700nm) |
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

`scripts/raytracer/tests/` (173 tests total; run with
`/home3/optics/env/bin/python -m pytest scripts/raytracer/tests/ -v`;
`test_gather.py` and `test_doubleslit_e2e.py` take minutes):

| Test file | Physics pinned | Tolerance |
|---|---|---|
| `test_kernels.py` (18) | Snell angles; Fresnel energy conservation; Brewster angle; TIR magnitude + Fresnel-rhomb phase; absorbing-metal Fresnel branch; reflect/pol-basis unitarity + Jones-rotation energy conservation; TMM reduces to bare Fresnel at 0 layers; quarter-wave MgF2-on-BK7 TMM reflectance; half-wave "absentee" layer; TMM energy conservation (oblique, 2-layer stack); sphere/cylinder/cone exact intersection; torus intersection vs `np.roots` reference; trimmed spherical cap, full-sphere-untrimmed, polar-cap-band, plane+hole trim, face exclude/eps guard | 1e-9 to 1e-12 (closed-form); torus vs `np.roots` 1e-7 |
| `test_asphere.py` (7) | Asphere reduces to a tangent sphere (k=0, no coeffs); parabola (k=-1) closed form; general polynomial asphere vs independent `scipy.brentq` root; analytic normal vs finite-difference; shape-operator (`normal_derivative`) vs finite difference for every analytic primitive; miss/grazing cases produce no NaN; trimmed asphere cap containment | <1e-10 (sphere/parabola reduction); <1e-10 (brentq); <1e-7 (normal FD); <1e-5 relative (shape operator, curved primitives) |
| `test_birefringence.py` (14) | Uniaxial `n(theta)` endpoints/monotonicity; eigenbasis orthonormality; isotropic limit reduces to Fresnel; **calcite walk-off at 45° internal incidence, 590nm, matches 6.23° to <0.05°**; quartz's opposite (positive-uniaxial) walk-off sign; e-wave normal-surface residual; tangential wavevector continuity at an interface; exit refraction reduces to Fresnel for the o-mode; beam-displacer normal-incidence walk-off + displacement; Wollaston geometry (c in interface plane -> zero walk-off); plane-parallel slab o/e roundtrip; internal e-ray TIR; exact `ray_from_k`/`k_from_ray` inversion | 1e-12 (closed-form); **0.05° (calcite/quartz walk-off vs literature)**; <0.5% (displacement) |
| `test_grating_models.py` (10) | Kogelnik efficiency formula (exact Bragg `nu=pi/2` gives `eta=1`, `nu=pi/4` gives `eta=0.5`); full geometric Bragg-peak construction with energy conservation; Dammann Parseval sum + symmetry; Dammann reduces to lamellar duty=0.5; a real 1x5 equal-intensity Dammann design vs its published transition points; table-model exact interpolation roundtrip + out-of-range hard error + missing-order zero; polarized table dispatch (pure s/p stays pure, exact `eta_s`/`eta_p` power, energy-exact children+absorbed); m=0 lamellar still reduces to Snell | 1e-9 to 1e-12 (closed-form Kogelnik/energy); <0.1% (Dammann 1x5 uniformity, published tol 2%) |
| `test_grating_roughness.py` (10) | Vector grating equation angles; m=0 reduces exactly to Snell/Fresnel (any incidence, conical mount, groove component invariance); lamellar efficiency closed form + energy partition; roughness TIS closed form; Beckmann slope statistics + reflected-energy clustering; groove-vector frame construction | 1e-9 to 1e-15 (closed-form); Beckmann RMS slope within 3%, reflected energy clustering >0.99 mean/>0.9 min |
| `test_mie_particles.py` (11) | Mie efficiencies vs Wiscombe MIEV0 canonical values (m=1.5, x=10 → Qext=2.8820; x=100 → Qext=2.0944); extinction-paradox limit (Qext→2); Rayleigh limit closed form; absorbing-particle albedo; phase-function normalization + forward-peaking + CDF sampling fidelity; monodisperse number-density/mu_ext exact closed forms; log-normal mean volume vs Monte Carlo; continuum Beer-Lambert energy split; explicit-mode non-overlapping placement + energy conservation | abs 2e-3 (MIEV0); rel 1–5% (statistical); exact to 1e-9/1e-12 (closed-form) |
| `test_pol_scatter.py` (7) | Polarized-azimuth Mie sampler: theta-marginal invariance (analytic + numeric) between polarized and legacy uniform-azimuth sampling; chi-squared goodness-of-fit for linear-polarized and circular/unpolarized incidence; per-scattering-event energy exactness; legacy `--no-pol-scatter` uniform-azimuth mode still available; general sampler-vs-PDF agreement | chi2 goodness-of-fit; exact energy per event |
| `test_polarization.py` (9) | Reference-frame (`e_ref`,`e_perp`) orthonormality + z-axis fallback to y; unit power for every polarization kind/stratum; `linear:0/90/45` Jones values; unpolarized strata orthogonality; elliptical limits (chi=0 -> linear, chi=45 -> circular); circular handedness matches the Hecht convention numerically; polarization strata do not interfere in the gather (same-stratum fringes >0.7 visibility, cross-stratum <0.35); end-to-end `sample_source` polarization states (skipped unless `doubleslit` is extracted) | exact (Jones unit power); numeric handedness check; visibility >0.7 same-stratum / <0.35 cross-stratum |
| `test_polarizer_filter.py` (7) | Malus's law through an ideal linear polarizer; crossed-polarizer floor + `polarizer_absorbed` bucket (>0.85 of power parked there); unpolarized half-transmission; bulk filter Beer-Lambert thickness scaling (`T^(d/d_ref)`); calcite double-spot separation + polarization-selected single spot; circular-polarizer output handedness | Malus 1% or better; energy-exact bucket accounting |
| `test_opticalproperties.py` (24) | Full opticalproperties tree loading; birefringence/coating/polarizer/filter/grating registry schema validation (each hard-error path: bad type, swapped T_par/T_perp axes, zero transmittance, over-unity order efficiencies, R+T>1, missing reference, unknown crystal material); trimmed-library optional-category handling; per-face/polarization/axis/grating-value spec parsers (`common.py`); schema-v2 `validate_model()` additions (polarizer/crystal_axis/coating-dict/roughness_faces/grating/polarization) and their rejections; asphere surface-dict schema validation | hard MaterialError/ContractError on every malformed input |
| `test_materials.py` (21) | Dispersion spot-checks (bk7, fused_silica, mgf2, water, sapphire, polystyrene, aluminum n/k incl. IR k rise and water's ~975nm absorption shoulder); tabulated-range hard error; unknown-material error message; vectorized evaluation; coating quarter-wave resolution; malformed-CSV rejections (bad model, non-positive Sellmeier C, missing reference); coating-references-unknown-material rejection; detector/vacuum sentinels; case-insensitive lookup | abs 1e-4 to 2e-4 vs literature/refractiveindex.info |
| `test_mesh_bvh.py` (6) | MeshFace intersection vs analytic Sphere (tessellation-sag error bound); smoothed vs flat normal modes; BVH traversal bit-identical to brute-force Möller–Trumbore; binary/ASCII STL reader agreement; self-hit `t_eps` guard; degenerate (zero-area) triangles dropped with a warning | >98% hit/miss agreement away from silhouette; bit-identical BVH-vs-bruteforce |
| `test_ray_differentials.py` (9) | Free-space spherical-wave patch area grows as r²; collimated patch area invariant; `init_curved` direction derivative vs finite difference; reflect/refract differentials vs finite-difference oracle for sphere/cylinder/asphere; concave-mirror patch area minimizes at f=R/2; single-surface refraction patch area minimizes at f=n2 R/(n2-n1); TIR/grazing produces NaN cleanly (no exception) | <1e-5 relative (finite-difference cross-checks) |
| `test_gather_occlusion.py` (7) | `occlusion=None` bit-identical to omitting the argument; empty-faces identity; an opaque plate fully shadows the geometrically blocked half (<1% of unoccluded peak) while leaving the rest exactly rescaled; an off-axis plate is culled by the Level-1 AABB prefilter; a detector's own face never self-occludes; torch/numpy backends apply an identical mask; rendered shadow-edge position within `tile+1.5` px of the geometric projection | <1% leakage; bit-identical torch/numpy mask; shadow edge within `tile+1.5` px |
| `test_gather.py` (7) | Gather kernel vs brute-force reference; point-source 1/r² + spherical phase; double-slit fringe pitch + visibility; single-slit first-zero position; incoherent source shows no fringes; undersampling gate raises; torch vs numpy backend agreement | kernel rel 1e-4; point-source rel 1e-5; double-slit pitch rel 1%, visibility >0.9; single-slit zero rel 2%; incoherent visibility <0.35; torch/numpy rel 5e-3 |
| `test_integration.py` (5) | `Scene` built from the real extracted `example.FCStd` contract; wavelength-strata semantics (monochromatic + asymmetric-Gaussian bounding); full-trace energy closure; traced focal position vs the thick-lens lensmaker equation | closure <1e-3; **focal position vs lensmaker equation: rel 5e-3 (<0.5%)** |
| `test_doubleslit_e2e.py` (1) | **The** end-to-end wave-optics validation: the real `doubleslit.FCStd` scene, extracted and traced through the full engine, must produce Young's fringes with the correct pitch, visibility, and Fraunhofer-envelope shape — unchanged by the polarization/birefringence/grating feature work | gate: correlation with analytic Fraunhofer pattern >0.75, pitch within 1.5 detector pixels of `lambda*L/d`, visibility >0.85, closure <1e-3 |

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
roughly as `rays * resolution^2 * n_coherent_sources * nlambda *
n_pol_strata` (`common.estimate()`'s `gather_ops`) — `n_pol_strata` is 2
for any scene containing an unpolarized source (the default), 1 if every
source declares an explicit polarization (§5.2). A real `--dry-run`
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
in §6.6. `results/.calibration.json` in this repo shows real recorded
`gather_torch` throughput of roughly **6e8–9e8 gather-ops/s**, alongside
recorded `trace` throughput ranging from ~500 to ~36,000 rays/s
(the wide spread reflects very different scene/particle/roughness
configurations across the recorded runs, not a single stable number —
`common.calibrated_rate()` takes the **median** of all recorded entries
of a given `kind`, falling back to `FALLBACK_TRACE_RAYS_PER_S=2e5` or
`FALLBACK_GATHER_OPS_PER_S={"torch": 2e10, "numpy": 1e9}` before any real
run has completed).

**Disk**: `detectors/*.h5` dominates a case's disk footprint — a single
512×512×16-bin detector cube in this repo's `results/example/devsmoke/`
is **~33 MB** per detector (`spectral_cube_mean` at float64, plus `mask`);
three detectors in that one run is ~100 MB. Every completed trace writes
the full spectral cube for every active detector regardless of
`--save-fields`; plan disk budget as `H * W * spectral_bins * 8 bytes *
n_detectors * (2 if --seeds > 1 else 1)`
(`case.json["estimates"]["fields_h5_GB"]` in the dry-run estimate covers
this — note the estimator's `save_fields`/`n_pol_strata` parameters exist
in `common.estimate()` but `run_pipeline.py`'s dry-run call does not yet
thread `--save-fields`/actual polarization strata through to it, so
treat `fields_h5_GB` as covering the always-written spectral cube, not
an accurate prediction of `--save-fields`'s additional complex-field
storage). `--save-fields` itself (§6.5) writes two more float64 complex
(i.e. 16-byte) `H×W` arrays per `(source, lam, pol)` key on top of the
spectral cube, seed 0 only — budget accordingly for polarization-heavy,
high-resolution `--save-fields` runs.

---

## 13. Troubleshooting

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
  in the corresponding `opticalproperties/*/*.csv`. These checks fire at
  **trace time** (`Scene.__init__`), not at extract time — the error
  message lists every name the registry actually loaded, so diff it
  against your body property string and against the CSV (§7).
- **Asphere `surface_override` declaration mismatch dies at extract
  time**: `surface_override=asphere declaration does not match the
  actual face geometry (max residual ... tolerance 1.0 um)` means the
  declared `R`/`k`/`A4...` doesn't reproduce the FreeCAD-authored face's
  actual sag to within 1 micron — re-derive the coefficients from the
  same design equations used to build the face, or loosen `r_max` if the
  mismatch is only at the aperture edge. This check runs in
  `extract_geometry.py`, well before any trace-time error would surface.
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
