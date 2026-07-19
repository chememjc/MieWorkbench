# Demo gallery

`demos/` ships 42 optical systems as self-contained `.MieWB` workbenches
(+ the bare `.FCStd` scene): eleven classic instruments, five
single-physics benches, four analysis/scattering benches, four
physics-showcase benches, the telephoto pair + a folded periscope, seven
pulsed-optics/time-domain benches, four coherence/anisotropy/scattering
benches (see "Beyond sequential codes" below), and two dedicated
optimize/tolerance showcase objectives (`double_gauss`,
`fiber_coupling_doublet`) alongside extended versions of
`camera_triplet`/`schmidt_cassegrain`. Every demo is
assembled as a real **optical train** through `scripts/make_demos.py`
(the GUI's own Project/chain op path — vertex-to-vertex chained
distances, live unfoldable fold mirrors, deviate ports for prisms/
gratings) and ships a curated set of global **variables** (arm lengths,
air gaps, iris opening, …) in the [Variables](variables.md) dock,
sweep-ready with min/max/steps.

```bash
env/bin/python scripts/make_demos.py --demo <name>          # or 'all'
python3 scripts/miewb_tool.py run demos/<name>.MieWB -o /tmp/<name>.MieSim
env/bin/python -m mieworkbench demos/<name>.MieWB            # open in the GUI
```

Every demo completes on the `quick` preset with the energy ledger closing
below 1e-3; the train rebuild is gated against a pre-train baseline by
`scripts/run_demo_equivalence.py` (placements ≤1 µm/0.01°, detected power
within Monte-Carlo bounds — committed oracles in `demos/baselines/`).
`demos/README.md` is the authoritative per-demo prescription + citation
list; `demos/UXNOTES.md`/`UXNOTES_ROUND2.md` record the friction (and
real bugs) found building these through the interface.
[`demos/gallery/`](../../demos/gallery/) has representative detector
renders (irradiance/PSF/Stokes/etc. — not 3D scene views, see
[viewport-3d.md](viewport-3d.md)'s screenshot note) for eleven of the
design-usability demos.

## Beyond sequential codes

Four demos exercise physics no sequential (Optiland-style) lens-design
code models at all — each is its own single-physics showcase, not an
imaging system:

- **`fizeau_flats`** — coherent ghost fringes off a thin wedged air gap
  between two flats.
- **`fs_shg_spectrogram`** — femtosecond SHG plus dispersion time
  products (a Mai Tai pulse, stretched then frequency-doubled).
- **`quartz_rotator`** — gyrotropic polarization rotation through a
  z-cut crystal, resolved via the full-anisotropy Berreman 4×4 solver.
- **`speckle_mie_combo`** — coherent laser speckle formed by exact Mie
  scattering off a ground-glass diffuser.

See `demos/README.md` for the full prescription + citation of each.

## Every demo ships pre-configured Optimize/Tolerance panes

Since the optimize/tolerance round, **every demo opens with the
[Optimize](optimize.md) and [Tolerance](tolerance.md) panes' tables
already populated** — nothing has been *run*, but the variable/operand/
tolerance tables are there the moment the scene loads
(`scripts/make_demos.py` attaches configs through the same
`Project.set_optimize_config`/`set_tolerance_config` the GUI itself uses;
they persist as JSON on the `miewb_vars` sheet and travel with the
`.FCStd`/`.MieWB`, so a reopened scene's panes stay populated on their
own — see `mainwindow._load_pane_configs`).

- **Optimize**: where a merit is meaningful, each demo's own global
  variables (air gaps, spacings, working distances) drive `spot_rms` for
  imagers, `detected_power` for coupling/throughput benches, or `mtf50`
  where the imaging-analysis products run. Pattern-characterization
  benches (`airy_singleslit`, `bladed_iris_star`, `gaussian_bench`,
  `diffuser_speckle`, `aerosol_mie`) and pure no-variable scenes ship no
  config — optimization is meaningless there. Airspace-only studies route
  to the deterministic **sequential** (Optiland) backend; power/MTF
  merits need the **worker** (Monte-Carlo) backend.
- **Tolerance**: `Demo.auto_tolerances()` walks every chained element and
  emits commercial-precision rows — despace ±0.1 mm, decenter ±0.05 mm
  (refractive elements/apertures/detectors/fibers), tilt ±0.1°
  (mirrors/beamsplitters/gratings/prisms/lenses), surface-radius ±0.1%
  and center-thickness ±0.1 mm on lenses (quoted values are 2σ; the
  stored `normal` distribution band is 1σ = half). Tolerance studies
  always run on the **worker (MC)** backend — the sequential path models
  an axisymmetric system and reports decenter/tilt impact as identically
  zero, only a real 3D trace honors actual perturbed geometry.

Many-row studies (a full auto-generated table runs ~15–30 rows) run end
to end thanks to variant-scratch-directory name hashing
(`common.shorten_variant`) — past ~140 characters a variant name is
hash-shortened so the filesystem's 255-char limit is never hit.

## Showcase walkthroughs

Four demos are the dedicated optimize/tolerance showcase set
(`scripts/run_demo_equivalence.py`'s `SHOWCASE` — also the four the gate
smoke-runs end to end on every equivalence check): **camera_triplet**,
**schmidt_cassegrain**, **double_gauss**, **fiber_coupling_doublet**. Each
has a full walkthrough — load, what ships configured, run a short
optimize, run tolerance sensitivity, interpret the result (including the
one showcase whose textbook result is honestly MC-unstable) — in
[walkthroughs/](walkthroughs/README.md).
