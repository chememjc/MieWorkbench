# UX friction log — consolidated open items

Consolidated 2026-07-19 (docs round). This file supersedes the four
per-round shakedown logs (round-1 `UXNOTES.md`, `UXNOTES_ROUND2.md`,
`UXNOTES_ROUND3.md`, `UXNOTES_PULSED.md`): every observation in those
files was re-verified against the current code; everything not listed
below was confirmed FIXED (with the fixing code located), obsolete, or
non-actionable historical color, and was dropped. Git history keeps the
originals.

## Placement / transform panel

1. **No axis+angle rotation readout.** The Absolute-pose group in
   `transform_panel.py` only shows intrinsic XYZ Euler angles at 2
   decimal places — a sub-mrad tilt (e.g. a Michelson fringe-tilt of
   0.158 mrad ≈ 0.009°) rounds away and can't be visually verified.
   Wants an axis+angle display with 4+ decimals.
2. **Anchored (absolute) placement fields are literal-only.** "Set
   position"/"Set orientation" in the Absolute group are plain
   `QDoubleSpinBox`es — no expression/variable support, unlike the chain
   edge cells and the newer polar "Place about point" tool (which DOES
   support expressions and a genuine polar form). A side-scatter
   detector or field-source fan still can't be swept parametrically
   from an anchored pose.
3. **`--particles` clouds are not chain-referenceable.** No way to say
   "detector 40 mm at 90° from the cloud center" (nephelometer-ring
   authoring) without hand-computed literals — needs a lightweight
   region-anchor/virtual element the reference resolver can target.
4. **Co-located transparent detectors overlap-fail extraction.** No
   authoring path exists for "measure the same plane two ways."

## Catalog / wizard gaps

5. **Achromat wizard can build an unbuildable lens.** `solve_achromat`/
   `design_lens("achromat", …)` never maps or clamps `aperture` against
   the solved interface radius — a small-f solve against the default
   18 mm aperture still dies with a bare "math domain error" instead of
   a clamp or warning.
6. **No "slit lamp" catalog preset.** A divergent+broadband source is
   still assembled by hand from `laser_divergent` + two properties.
7. **No `grating_mirror` catalog primitive.** A reflective grating
   (Czerny-Turner-style) still needs three manual switches on
   `grating_plate` (`mirror=1.0`, explicit periodicity vector, face
   pin).

## Rebuild-on-edit

8. **Face-map re-resolution after rebuild is unimplemented.**
   Rebuilding a primitive still renumbers `FaceN` indices with no
   geometric old→new matching, silently orphaning coating/grating/
   surface_override face-map properties. The Czerny-Turner demo still
   works around this by importing un-rebuilt geometry.

## Pulsed-optics round

9. **`pockels_switch` demo is still dropped.** The coherent o/e-gather
   recombination that would turn EO retardance into an observable
   detected-power modulation is phase-noise-dominated at gallery scale
   (0.6% observed vs 61% predicted by sin²); needs either a much larger
   coherent ray budget or a new incoherent-path observable (e.g. a
   Stokes/DOP readout between circular polarizers). Not currently
   tracked in future.md.
10. **Auto time-window sizing covers echo-train ghosts** instead of
    clustering around the dominant pulse (e.g. a ~1 ns window for a
    100 fs pulse when a double-bounce ghost arrives ~60 ps later).
    Tracked in future.md as "Dominant-cluster auto time window" but
    still unimplemented.
11. **Transmission-grating truncated diffraction orders leak ~8%** past
    the closure gate — only the reflective/aluminum booking branch
    accounts for the truncated remainder correctly. Tracked in
    future.md as "Transmission-grating truncated-order booking" but
    still open.

## Minor / unverified

12. **Small doc/naming gaps** (open-unverified): `particle_threshold`'s
    "explicit if count ≤ threshold" direction reads backwards;
    `laser_collimated` divergence semantics aren't documented on the
    params; LED primitive name suffixes are bin nicknames while the
    actual CWL is in `lambdac`; `model.json`/`rays_full.npz` units are
    metres (a 1000× foot-gun) with no doc callout.

## Resolved (samples-instruments round, 2026-07-23)

- Item **3** (`--particles` clouds not chain-referenceable): the
  `sample_region` primitive (a bare `material=air` anchor cube) carries a
  `port_frames` pass-through entry, so a body-bound `sample` cloud is now
  chain-referenceable like any other element.
- Item **4** (co-located transparent detectors overlap-fail extraction):
  detector-detector solid overlap now classifies into the informational
  `validation.detector_overlap` list instead of the fatal
  `overlapping_solids`.

## Discovered building the samples-instruments demo galleries (2026-07-23)

Seven demos (`conical_refraction`, `colloidal_crystal`, `goniometer_bath`,
`uvvis_spectrometer`, `forward_scatter_diffraction_sizer`, `imaging_bench`, `dls_goniometer`)
were built; several hit real engine/scale limits that forced documented,
physics-preserving substitutions (all in `scripts/make_demos.py` docstrings
+ `demos/README.md`):

1. **Continuum particle scatter inside a GLASS cell diverges the C engine.**
   A wide-angle continuum-scattered ray hitting a water/glass/air window
   TIR-traps in the cell, and the C-engine continuum-sample pop accounting
   then DIVERGES (closure blows to ~1e48–1e58; the trace itself prints
   `pop accounting diverged; investigate`). Reproduced with `flow_cell`
   (forward-scatter-diffraction), the decalin `vat_cylindrical`/`vial_cylindrical` bath
   (goniometer), and `cuvette_square` (colloidal). The Python reference
   engine does not diverge but runs away (multi-minute at a few thousand
   rays). WORKAROUND in every case: put the sample in a bare
   `sample_region` (air host) so wide-angle scatter escapes — an air region
   has no refraction anyway, the sim-equivalent of the index-matching bath.
   The intended cell/bath/vial primitives could not be showcased WITH an
   active scattering sample. Worth an engine follow-up (the divergence is a
   bug, not just a ray-budget cap).

2. **Explicit paracrystal lattices are intractable at macroscopic scale.**
   `colloidal_crystal_fcc` (mode=explicit) places a real jittered FCC
   lattice; filling a millimetre-scale cell = ~1e12 sites and numpy tries
   to allocate **136 TiB**. Added a demo-specific `colloidal_fcc_continuum`
   row (mode=continuum, same paracrystal S(q) as a structure factor) so the
   wavelength-selective Bragg backscatter survives. Even so the SHARP
   single-bin (111) Bragg peak does not resolve above the λ⁻⁴ background at
   the fixed 16-bin detector cube — the gate checks the tractable
   wavelength-selective backscatter (blue/red reflectance > 1.8), not a
   sharp Bragg line. A macroscopic coherent colloidal-crystal Bragg demo
   needs either the explicit lattice made tractable or finer spectral cube
   resolution.

3. **`sample=` count controls: the row `count` column is ignored** (only
   `phi`/`tau` set the explicit count) and `phi` is a MASS fraction vs the
   HOST — so the same row gives wildly different counts in a water vs air
   host, and a physical 100 nm concentration is astronomically many
   spheres. A `tau=`-based row (optical-depth target) is host-independent
   and far easier to tune to a sparse explicit realization.

4. **DLS: 100 nm PSL is below the traced-speckle scatter floor, and the
   resolvable geometry needs a water AMBIENT the extractor can't set.**
   `run_dls` reconstructs a coherent speckle field per frame; 100 nm PSL
   scatters ~9 orders too weakly for a discrete-ray gather (the detector
   field comes out identically 0). The proven `test_dls.py` recipe is a
   HANDFUL of LARGE (3 µm), well-separated spheres (a dense/small cloud
   desynchronises the shared scatter RNG and g1 collapses in one frame) —
   used here as `psl_dls_3um`. But a RESOLVABLE decay needs LARGE-angle
   (large-q) detectors, and those need the scattered rays NOT to TIR at the
   sample/air boundary — the test achieves this with
   `model["ambient_material"] = "water"` (index-matched immersion), which
   `extract_geometry` HARDCODES to `"air"` (no body property or simparam
   exposes it). Forward small-angle detectors avoid the TIR but give a q so
   small the decay (τ_c ~ seconds) never resolves in a sane frame budget.
   `dls_goniometer` therefore ships as a DEMONSTRATOR (the builder + a
   populated near-forward speckle field via `run_dls`), not a baseline
   oracle; a full Γ-vs-D·q² gate would need a scene-settable ambient
   material. Worth exposing `ambient_material` (e.g. a scene/source
   property or a `--ambient` simparam).

5. **The `tcd1304_array` diode-array saturates full-well across the band**
   at ordinary lamp powers, so a `--reference-case` absorbance A=-log10(I/I0)
   collapses to 0 (both sample and blank clip to 4095 counts) — and
   reducing the lamp power did not de-saturate it in testing. The
   `uvvis_spectrometer` gate instead forms A(λ) from the two Array
   detectors' RAW spectral cubes (unsaturated dispersed power), which
   recovers the KMnO₄ band cleanly (A peaks **3.24 at 528 nm** vs
   Beer-Lambert 3.34). The diode-array instrument's electron/full-well/ADC
   chain may need an auto-exposure or a documented linear-range note.
