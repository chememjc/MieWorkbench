# MieWorkbench demo gallery

Thirty-four optical systems (eleven classic instruments, five
single-physics benches, four analysis/scattering benches, four
physics-showcase benches, the telephoto pair, a folded periscope, and
seven pulsed-optics/time-domain benches),
each a self-contained `.MieWB` workbench archive (double-click-open in
the GUI, or run headlessly) plus the bare `.FCStd` scene. All are assembled as **optical trains** by
`scripts/make_demos.py` (the GUI's own Project/chain op path): every
element chains a vertex-to-vertex distance down the beam from its
reference, fold mirrors are live unfoldable folds, prisms/gratings are
deviate ports, and each demo ships a curated set of **global variables**
(arm lengths, air gaps, the iris opening, fringe tilt — see the
Variables dock, sweep-ready with min/max/steps). Rebuild any of them:

```bash
env/bin/python scripts/make_demos.py --demo <name>   # or 'all'
python3 scripts/miewb_tool.py run demos/<name>.MieWB -o /tmp/<name>.MieSim
env/bin/python -m mieworkbench demos/<name>.MieWB    # open in the GUI
```

Every demo completes on the `quick` preset (1e5 rays, 512², 5λ) with the
energy ledger closing below 1e-3, and the train rebuild is gated against
the pre-train gallery by `scripts/run_demo_equivalence.py` (placements
≤1 µm / 0.01°, detected power within Monte-Carlo bounds — the committed
oracles live in `demos/baselines/`). `demos/UXNOTES.md` and
`demos/UXNOTES_ROUND2.md` record the friction found while building these
through the interface (and the real bugs each exercise caught).

| Demo | System | What it shows | Detected (of 5 mW, quick preset) |
|---|---|---|---|
| `beam_expander` | 3× Keplerian expander: BK7 PCX f=50 + f=150, spacing f1+f2, convex sides out | Collimation preserved, 3× beam diameter; loss = the four uncoated Fresnel surfaces (0.96⁴ ≈ 0.85) | **4.23 mW** |
| `newtonian` | 150 mm f/6: parabolic primary (rfl 900), 45° round diagonal, folded focus, WHITE-LIGHT star (450–650 nm) | Exact parabolic focus, 90° fold, central-obstruction shadow; loss = two Al bounces + obstruction | **3.74 mW** |
| `dobsonian` | 200 mm f/5 Newtonian optics, white-light star (a Dobsonian is the same telescope on an alt-az mount) | Faster, larger variant of the above | **3.68 mW** |
| `michelson` | 25 mm 50:50 **plate** beamsplitter at 45°, 60 mm arms, one mirror tilted 0.158 mrad (the `m1_tilt` variable) | Coherent two-beam interference: straight fringes (measured visibility 0.90) across the detector at 633 nm, pitch λ/2θ | **1.09 mW** at the fringe port (seed-sensitive: coherent fringe power varies ±0.45 mW across seeds at this preset — compare runs seed-matched) |
| `michelson_folded` | The michelson with its transmit arm **Z-folded** by two extra 45° aluminum mirrors (equal optical path, 45×15 mm dogleg instead of a straight 60 mm arm) | Live fold workflow: unfold both folds in the train editor and the arm re-collinearizes into the plain michelson EXACTLY (measured: unfolded run reproduces 1.088 mW); the `ideal_folds` variable A/Bs the mirror losses | **0.857 mW** folded (the 4 extra Al bounces cost 21%); **1.084 mW** with `ideal_folds=1`; **1.088 mW** unfolded |
| `prism_spectrometer` | 25 mm equilateral SF5 prism at minimum deviation (550 nm), f=100 camera lens | Chromatic dispersion: 450–650 nm spread ~2.3° → a ~4 mm spectrum (the honest prism-vs-grating tradeoff) | **3.74 mW** (was 0.60 before the round-2 trim-loop extractor fix resurrected the prism's dead half-faces) |
| `czerny_turner` | Crossed CT: divergent slit source, R=200 collimator, 600 g/mm reflective grating (`mirror=1.0`), R=200 camera mirror | Grating dispersion + off-axis mirror folding; 400–700 nm across ~25 mm, first order | **0.08 mW** (slit + overfill + order efficiency) |
| `camera_triplet` | Cooke triplet ~50 mm EFL (published design rescaled), iris stop ~f/5.6, 36×24 mm sensor, white-light scene (450–650 nm) | A real multi-element photographic objective; detected ≈ the f/5.6 pupil fraction of the 14 mm input beam | **1.26 mW** |
| `microscope_objective` | Lister-type: two air-spaced achromats (f=25 + f=50, 10 mm apart), finite conjugates, white-light point source | Aberration-corrected imaging of a point source | **2.79 mW** |
| `fiber_coupler` | 650 nm laser → 2 mm BK7 ball lens (BFL 0.47 mm) → 75 mm of 200 µm/0.22 NA fiber | TIR guiding down the fiber core (~60 bounces; `max_reflections` simparam) | **3.95 mW** at the exit face |
| `schmidt_cassegrain` | C8-class 203 mm f/10: quartic Schmidt corrector (hand-authored asphere), perforated spherical primary R 812.8, spherical secondary R 231.07, white-light star | Catadioptric folding: corrector → primary → secondary → focus through the primary's hole; loss ≈ two Al bounces + 11 % obstruction | **3.42 mW** |

### New-physics benches (Phase 12)

Five deliberately minimal scenes, each isolating one physics feature that
landed this round (biaxial crystals, Gaussian beams, Fresnel ghosts,
measured ABg scatter, curved detectors). "As simple as possible but
physically real": solids with real materials, one source, one screen.

| Demo | System | What it shows | Detected (of 5 mW, quick preset) |
|---|---|---|---|
| `ktp_walkoff` | 15 mm biaxial KTP plate, X principal axis at 45° in the layout plane (`crystal_axis` + `crystal_axis2`), 633 nm narrow unpolarized beam | Biaxial double refraction: the in-plane sheet walks off ~0.85 mm in z while the out-of-plane (n_y) sheet goes straight → two spots | **4.25 mW** (both spots; ~4 uncoated KTP-face Fresnel losses) |
| `gaussian_bench` | 50 µm-waist (M²=1.0) 633 nm Gaussian source (`beam_waist` + `m2`, incoherent beam mode), 62 mm of empty air | TEM₀₀ diffraction: the beam expands ~5× (5 Rayleigh ranges) per w(z)=w₀√(1+(z/z_R)²) | **5.00 mW** (lossless propagation) |
| `ghost_doublet` | Two uncoated N-BK7 windows (4 mm thick, 8 mm spacing), collimated 633 nm, `--ghost-analysis` on | Natural double-bounce Fresnel ghosts: the strongest gen-2 path carries direct·R² (R≈4.2 %); enumerated by the ghost report | **4.25 mW** direct + ghosts |
| `scatter_plate` | BK7 window at 45° with a measured ABg finish (`scatter=polished_bk7_glass`); the reflection folds to +y | Diffuse stray light: the reflected arm catches the specular spot plus the ABg scatter lobe (TIS ~2 %), reflected split conserving R | **0.48 mW** in the reflected arm |
| `curved_focal` | 633 nm Ø10 beam → BK7 PCX lens (f≈48.5) → cylindrical detector (`material=detector`, axis ‖ z) at the focus | A curved focal-surface screen: the cylindrical (`CurvedDetectorGrid`) face catches >90 % of the focused cone | **4.59 mW** (92 % of the focus) |

### Analysis & scattering benches (design-usability round)

Four demos exercising the exact-Mie particle clouds, ground-glass
diffuser/speckle, coherent aperture diffraction, and the named
imaging-analysis products (PSF/MTF/Strehl/Zernike/EE/spot/fans). All four
route to the **C engine** on the quick preset and close the energy ledger
(<1e-3). Several carry deliberate deviations from the original
`demosystems.md §3` prescription so the physics is actually visible at the
quick budget — each is noted in the demo docstring and below.

| Demo | System | What it shows | Detected (of 5 mW, quick preset) |
|---|---|---|---|
| `aerosol_mie` | 532 nm coherent Ø2 probe through a 40 mm cube of log-normal water droplets (2 µm median, `--particles` continuum mode), forward + 90° side detectors | Exact-Mie aerosol scattering: Beer–Lambert extinction of the ballistic beam (τ≈1.14 → ~32% transmitted) plus a forward halo, and the single-scatter phase-function lobe on the side detector | **3.19 mW** forward (attenuated ballistic + forward halo) + **0.057 mW** side scatter |
| `diffuser_speckle` | 633 nm coherent Ø4 beam through a 600-grit ground-glass diffuser (`@dg_600`, ~5° cone), far-field screen 150 mm on, `--save-fields` | Ground-glass scatter cone + coherent speckle; the Stokes/DOP map shows the diffuser's partial depolarization | **4.60 mW** (scatter cone within the 80 mm screen) |
| `airy_singleslit` | 633 nm coherent Ø0.6 beam floods a Ø0.2 mm pinhole (air-filled plug), screen 250 mm on | Circular-aperture Fraunhofer diffraction: the central Airy disk forms at the right scale (EE-83.8% radius ~1.3 mm vs the ideal 1.22 λL/D = **0.966 mm** first null); the ring nulls are NOT resolved at the quick coherent-gather budget (smooth profile) | **0.28 mW** through the pinhole |
| `imaging_analysis` | Cooke triplet (reused from `camera_triplet`) fed an on-axis coherent 550 nm Ø5 wavefront, 0.5 mm sensor at paraxial focus, `--save-fields --export-rays --emit-csv` | Every named analysis product renders: PSF + FFT-MTF + encircled energy (from fields) and Strehl + Zernike wavefront + spot/OPD fans (from exported rays). The BK7/SF5-substituted triplet is aberrated (RMS ~38 waves, Strehl ~0, MTF50 ~10.5 cy/mm, EE50 ~18 µm) — the bench demonstrates the analysis **pipeline** on a real imperfect lens | **3.69 mW** at the sensor |

Deliberate deviations (physics-vs-preset, all in the docstrings):
`aerosol_mie` raises `phi` from the prescribed 1e-6 to **2e-2** (at 1e-6
the optical depth is ~5e-5 — forward unattenuated, side detector 0 mW; the
headline Mie physics is invisible). `airy_singleslit` shrinks the beam from
Ø6 to **Ø0.6** (Ø6 passes only ~0.1% of the rays through the Ø0.2 pinhole
→ the coherent gather hard-errors `undersampled M_eff=28`); it also sets
`blackness=1.0` (opaque screen) and `nlambda=1` (one coherent-gather
stratum). `imaging_analysis` uses a Ø5 input pupil and a 0.5 mm square
sensor rather than `camera_triplet`'s Ø14 / 36×24 mm frame (Ø14 is
spherical-aberration-dominated — 2.8 mm spot, PV ~2000 waves; a non-square
sensor also crashes the field-analysis MTF plotter). These are unlisted in
`run_demo_equivalence` (lighter gating this round, by decision).

Every run closes the energy ledger (<1e-3). The five imaging demos use
WHITE-LIGHT (450–650 nm) sources: their preview fans emit a red/green/
blue bundle from every fan point and the traced overlays carry per-ray
wavelength colors, so chromatic behavior reads directly off the rays.
(The Michelson uses a PLATE beamsplitter for historical reasons; the
`bs_cube` primitive itself now splits correctly too — it was rebuilt as
a single cube with a nested coated plate after the round-2 gap/TIR
investigation.) The folded systems'
`simparams.json` pins `detector_face` (and the CT's `--grating` face)
resolved at build time from the shipped file's extraction — FaceN indices
are not stable across rebuilds or save/reload, see UXNOTES.md.

### Physics-showcase benches (design-usability round 2)

Four more demos covering photometry + per-source reporting, full-Stokes
polarimetry, biaxial conoscopy, and a curved (Petzval) focal surface. All
close the energy ledger (<1e-3). Two route to the **C engine**
(multiled_photometry, stokes_polarimeter); the other two use unported
features and route to the **Python** reference engine as expected
(biaxial_conoscopy → `unported: biaxial`; curved_focal_surface →
`unported: curved_detector` — the engine + reason are recorded in
`case.json`).

| Demo | System | What it shows | Engine · detected/closure |
|---|---|---|---|
| `multiled_photometry` | Four incoherent LEDs at staggered heights (royal-blue 452 nm 3 mW, green 527 nm 4 mW, red 625 nm 5 mW catalog primitives + a `source_broadband` wearing the tabulated **CIE 015:2018 LED-B1** white spectrum `led_white_2733k` 6 mW) → BK7 PCX condenser → `@dg_600` ground-glass mixer → photometric target. `--photometric --emit-csv` | Per-source / per-detector charts + CSV (`data/source_detector.csv` lists all four sources, incoherent direct-deposit path) and photometric units: **4.28 lm**, peak **1.95e5 lux**, mean **1.52e3 lux** on TargetLux | **C** · TargetLux **15.2 mW** of 18 mW in (blue 2.5 / green 3.4 / red 4.2 / white 5.1); closes |
| `stokes_polarimeter` | 550 nm coherent linear:45 diverging cone (~±8°) through a 3 mm multi-order quartz A-plate; full-Stokes detector 150 mm on, `--save-fields` | Angle-varying retardance → the coherent gather maps field angle to detector radius, so the Stokes S1/S2/S3 + DOP maps carry genuine spatial structure (S1/S2/S3 span [−1,+1] across the figure, DOP ≈ 1.0 — a pure retarder) | **C** · **4.56 mW**; stokes/dop PNGs render; closes |
| `biaxial_conoscopy` | 589 nm unpolarized diverging cone (~±18°) → input polarizer (+z) → 2 mm **KTP** biaxial plate (acute bisectrix Z along the beam, optic axial plane spun 45°) → crossed analyzer (+y) → figure screen 50 mm on | Conoscopic interference figure between crossed polarizers: a 512² figure with clear 4-fold azimuthal isogyre/isochrome structure (the acute-bisectrix biaxial figure) | **Python** (biaxial) · **1.95 mW** (figure bright — mean crossed-analyzer transmission ≈ 0.78 over the multi-order field; azimuthal isogyre contrast ≈ 0.96); closes |
| `curved_focal_surface` | Three 550 nm field angles (0°, ±16°, chief rays through the lens centre) through an AR-coated BK7 DCX singlet (f≈80, Ø14) onto TWO detectors: a **concave** spherical recorder (`mirror_concave` tagged `detector`, R = n·f ≈ 121 mm = the Petzval radius) + a flat screen | The concave (Petzval-matched) surface holds a tighter spot than the flat screen at every field angle — measured clean-bundle spot-RMS ratio concave/flat ≈ **0.83 / 0.90 / 0.90** (0° / +16° / −16°) | **Python** (curved_detector) · both detectors record 512² images; closes |

Deliberate deviations (all in the docstrings + `UXNOTES` round 2):
- `multiled_photometry` OMITS the optional pellicle + "MonitorTap" second
  detector — the required deliverables (photometric block, four-source
  `per_source` table, `source_detector.csv`, closure) are all met by the
  single-detector scene, which stays cleanly C-routable; a 45° pellicle
  fold arm adds risk for no new required coverage.
- `stokes_polarimeter` drops the crossed analyzer the brief named and
  varies retardance by beam **divergence**, not a plate tilt. A tilt does
  not vary retardance across a *collimated* aperture, and a *full* crossed
  analyzer projects every ray onto its axis → spatially FLAT S1/S2 (only S0
  structured). Letting the detector be the Stokes analyzer, fed an
  angle-varying field, is what actually yields S1/S2/S3 structure.
  Retarder thickness is capped ~3 mm by the coherent gather (a thicker
  plate's fast OPL variation collapses the Huygens reconstruction to phase
  noise — 0 mW displayed, zero field map).
- `curved_focal_surface` uses `mirror_concave` NOT the brief's `lens_ball`.
  A positive lens's Petzval field sags TOWARD the lens (concave-toward-beam
  best focus); a solid ball presents a CONVEX near face (wrong sign) that is
  provably worse than a flat screen. The concave surface is the correct
  spherical detector. Field curvature is small (~2 mm sag) and partly masked
  by the singlet's own coma at 16°, so the improvement is modest but present
  at every field angle.

These four are also unlisted in `run_demo_equivalence` (lighter gating this
round, by decision).

### The telephoto pair + folded periscope (design-usability round 3)

Three demos whose POINT is the no-side-math workflow: every prescription
number is solved live in the demo function by `core/paraxial` +
`core/wizards` (asserted against the frozen design study), and the
geometry is driven entirely by global variables + chain expressions. All
three route to the **C engine** and close the ledger (< 1e-3).

| Demo | System | What it shows | Engine · detected/closure |
|---|---|---|---|
| `telephoto` | Classic achromatized **200 mm f/4** telephoto (Kingslake layout): front positive achromat (`solve_achromat(120)`, Ø56) + iris stop + CEMENTED negative rear doublet (equiconcave BK7 crown R∓30.85 + SF5 flint, back radius +75 = the chromatic tuning DOF), telephoto ratio **0.739** (147.8 mm front-vertex→focus ≪ 200). A divergent source sits at the FRONT FOCAL POINT, its distance the expression `efl × 2.914` — change the `efl` variable (150…300) and *every* radius/thickness/airspace/aperture rescales via dim-cell expressions while the source tracks the focus; `stop_d` (44.98 = exactly f/4 by marginal-ray solve) sets the equivalent f-number = efl/pupil | Whole-prescription variable scaling, source-at-focus expression tracking, equivalent f/# from the paraxial engine, `importance_aim` (96% acceptance on the overfilled pupil), secondary spectrum dBFL(F−C) = −0.09 mm — real achromat territory | **C** · Collimated **1.29 mW**; closure 2.5e-12 |
| `telephoto_zoom` | The SAME two groups, prescriptions fixed; only the airspace `z` moves (zoom variable, 84…98 → **EFL 258…159, 1.62× zoom**). The sensor is chained at the expression `-(pA+qA·z)/(pC+qC·z)` — BFL as a rational function of the zoom gap, coefficients from the two group ABCD matrices — so dragging `z` moves the rear group AND the sensor exactly like a mechanically-compensated zoom | The chain expression grammar carrying a real optical design law (focus tracking without re-anchoring anything) | **C** · Sensor **3.31 mW**; closure 6.7e-13 |
| `folded_periscope` | Unit-magnification afocal relay (two f=75 PCX, spacing = 2×thick-lens BFL — NOT the thin-lens f1+f2) folded TWICE at 90° by proper train folds: up the `arm` variable, back to horizontal. The L2 spacing self-corrects through `relay − 25 − arm` | The fold operator end-to-end: Unfold All → flat bench, Refold All → **bit-exact** restore (asserted at build time); throughput = 4 uncoated Fresnel surfaces × 2 aluminum mirrors ≈ 0.69 | **C** · Exit **3.44 mW** (5 mW × 0.688 predicted); closure 5.1e-13 |

Verification sweeps (quick preset, `--var miewb_vars.<name>`, C engine;
plots in `gallery/telephoto_*_sweep.png`):

- **`stop_d` 11.25 → 45 mm** (telephoto, efl=200): the collimated-beam
  diameter on the detector is linear in `stop_d` (Deq@20% = 4.0 / 8.3 /
  13.3 / 18.3 mm, R² = 0.999), and detected power tracks the pupil **area**
  — √(P/P₀) = 1.00 / 2.01 / 3.00 / 3.95 vs `stop_d` ratio 1 / 2 / 3 / 4
  (P = 0.083 / 0.335 / 0.745 / 1.29 mW). Equivalent f-number f/16 → f/4.
- **`efl` 150 → 300 mm** (telephoto, stop_d=44.98 fixed): every element x
  is exactly linear in `efl` (R² = 1.00000) about the fixed Star anchor —
  FrontGroup = 2.95·efl − 700 (−110 mm at efl=200), Stop/RearGroup/
  Collimated all scale; the beam stays collimated (Deq ≈ 16–18 mm). With
  `stop_d` held fixed the f-number grows with efl, so detected power falls
  monotonically (1.96 / 1.29 / 0.84 / 0.59 mW) rather than staying flat —
  the demo's stated behavior (`stop_d` sets f/# = efl/pupil).
- **`z` 84 → 98 mm** (telephoto_zoom, EFL 258 → 159, 1.6× zoom): the
  rational-BFL chain keeps the sensor at focus — spot EE50 radius is
  constant at 41.4 / 41.5 / 41.5 µm across the zoom, detected power ≈ 3.3 mW
  throughout.

Hard-won notes baked into these three (full log: `UXNOTES_ROUND3.md`):
- **Never anchor a source at the world origin** — emit directions choose
  the "toward the origin" hemisphere, which degenerates AT the origin
  (rays spray backwards; the telephoto detected 0 mW until Star moved to
  x=−700).
- The rear negative doublet MUST be cemented (achromat primitive with a
  negative prescription): an air-spaced dcv+dcx pair at aperture 30
  overlaps solids (concave rim sag ~3.6 mm) — the same reason real
  negative doublets are cemented.
- The afocal relay spacing is `bfl₁+bfl₂` (thick-lens), not `f₁+f₂`.

### Pulsed-optics / time-domain benches (pulsed round)

Seven demos exercising the round's physics: per-ray group delay + the
four time products (`--time-products`), the GDD budget, the pulsed-source
contract, source-side SPM, the χ² SHG transfer, and the transverse
Pockels cell. Every number below is measured from the shipped
`var/evaluation/*.MieSim` run (quick preset unless noted); every run
closes the energy ledger <1e-3. Time/NLO features route to the PYTHON
engine by design (feature tokens). The shakedown log is
`UXNOTES_PULSED.md` — it includes the two engine bugs these demos
caught (the grating child polarization-frame leak and the absorbed-path
tally bias).

| Demo | System | What it shows | Verified numbers |
|---|---|---|---|
| `sc_spectrogram` | SuperK EXR-20 (tabulated 400–2400 nm SPD, 5 ps/80 MHz) → 100 mm SF11 → screen; `pulse,spectrogram` + `--gdd-budget` | Time-of-flight spectroscopy: the spectrogram ridge IS the material group-delay curve t(λ)=[air+n_g(λ)L]/c (~37 ps ramp) | ridge matches n_g(λ)L/c to **≤1.0%** across 474–1514 nm |
| `erfiber_spm` | Er-fiber fs oscillator through 2 cm HNLF (`spm=gamma:11.5:length:0.02`, φ_max = 9.46 rad) → screen; 129 λ-strata | Source-side SPM: the classic M-shaped multi-peak spectrum + the S-curve chirp (leading edge red) | strong outer lobes at ~1441/1698 nm; spectrogram tilt d⟨t⟩/dλ < 0 (red first) |
| `fs_lens_telescope` | Mai Tai (100 fs, 800 nm) → BK7 PCX pair (f=50+100) + 40 mm SF11 block → screen; `--gdd-budget`, pinned ±2 ps window | The refractive half of the GDD contrast pair: material dispersion stretches the pulse; the budget table PREDICTS the traced profile | budget φ₂ **8029 fs²** → τ_out 244.0 fs; traced FWHM **244.5 fs** (0.2%) |
| `fs_oap_telescope` | Same pulse through an all-reflective 2× expander (concave f=50+100, periscope Z-fold) | The reflective half: zero glass in the train ⇒ no material GDD, pulse stays transform-limited | traced FWHM **101.3 fs** (kernel-limited 100 fs); GDD budget empty |
| `tof_rangefinder` | 0.5 ns/10 µJ pulses → 50:50 plate BS → target mirror at 600 mm; start pulse on DetRef (−y), return on DetReturn (+y); histogram envelope, 1024 bins | Time-of-flight ranging off two one-pulse profiles | Δt = **4.0225 ns** → range 603 mm; the +3 mm over 2d/c = the BS glass transits (a real rangefinder systematic, ~5.9 ps per 3 mm/45° pass) |
| `shg_green_bench` | 2 µJ/10 ps/1 kHz at 1064 nm → 5 mm KTP (`ktp_shg_1064_type2`, d_eff 3.2 pm/V) → 805 nm shortpass dichroic → DetGreen (532 straight) + DetIR (1064 folded); `--ray-differentials` | The χ² SHG transfer: incoherent λ/2 children, `harmonic_strata` map, `shg_converted_W` tally, dichroic color split | η = **6.0%** (0.120 mW of 2 mW avg converted); DetGreen 0.096 mW at 532 nm (exit-Fresnel losses), DetIR 1.54 mW; closure 3e-13 |
| `treacy_compressor` | 600 g/mm reflective grating pair at normal incidence (m=−1 then m=+1), 70 mm slant → screen; 17 strata, 1024 bins | ALL-GEOMETRIC (grating-pair) GDD: no glass anywhere, the arrival-time tilt is pure diffraction geometry | spectrogram slope **82.8 fs/nm** (3-λ geometric ray oracle: 87.4); traced FWHM 100 → **791 fs** vs 786 fs from its own measured slope (0.6%) |

Deliberate deviations (all in the docstrings): the Treacy pair is
REFLECTIVE (aluminum, `mirror=1.0`) because the transmission plate's
truncated lamellar orders leak ~8% past the closure gate (future.md);
`erfiber_spm`/`treacy_compressor` set `coherent=False` (129 strata /
the diffraction-split arm cannot budget 1000 gather samples per key at
quick, and neither needs interference); the fs contrast pair pins
`--time-window` ±2 ps around the computed main-pulse group delay (the
auto window spans the lens double-bounce echoes → 2 ps bins);
`fs_lens_telescope` adds a 40 mm SF11 block because 10 mm of BK7 lens
glass alone broadens 100 fs by only 0.8% — invisible at any binning.
`prism_compressor` and `wideangle_retrofocus` from the round plan are
deferred (future.md) — their physics is engine-gated elsewhere. A
`pockels_switch` bench was built and then DROPPED: the o/e
recombination only exists in the coherent gather, and at any
gallery-scale bench the reconstruction's phase noise (phase-step
gate warnings) drowns the EO retardance — the voltage sweep moved
the detected power 0.6% where sin² predicts 61%. The transverse
Pockels physics is pinned instead by the engine oracle
(test_nlo_elements: sin²(πV/2V_π) at 1% on a beat-length cell).

## Rendered gallery

Representative detector renders (quick preset) for the eleven
design-usability demos live in [`gallery/`](gallery/), captured by the
Phase-6 verification run:
[aerosol_mie](gallery/aerosol_mie.png) ·
[diffuser_speckle](gallery/diffuser_speckle.png) ·
[airy_singleslit](gallery/airy_singleslit.png) ·
[imaging_analysis](gallery/imaging_analysis.png) (+ [MTF](gallery/imaging_analysis_mtf.png)) ·
[multiled_photometry](gallery/multiled_photometry.png) ·
[stokes_polarimeter](gallery/stokes_polarimeter.png) ·
[biaxial_conoscopy](gallery/biaxial_conoscopy.png) ·
[curved_focal_surface](gallery/curved_focal_surface.png) ·
[telephoto](gallery/telephoto.png) ·
[telephoto_zoom](gallery/telephoto_zoom.png) ·
[folded_periscope](gallery/folded_periscope.png) — plus the three
telephoto verification sweeps
[stop_d](gallery/telephoto_stopd_sweep.png),
[efl](gallery/telephoto_efl_sweep.png),
[zoom-z](gallery/telephoto_zoom_sweep.png).

## Feature-coverage matrix

The gallery + the 25 validation scenes as a *covering set* over every
shipped capability (absorbed from the retired `demosystems.md` working
document, 2026-07-11). ✅ = exercised by a runnable demo/scene today;
⏳ = the capability exists but its demo waits on the named backlog item;
🔶 = future-roadmap capability (acceptance-target demos specified in
`future.md`).

| Feature | Exercised by |
|--|--|
| Refraction / dispersion | beam_expander, prism_spectrometer, camera_triplet, lens_* scenes |
| Fresnel loss + energy audit | every demo (ledger < 1e-3 everywhere) |
| Spherical / conic+A4 asphere surfaces | lens_ball, lens_asphere (full-lens-corrected), schmidt corrector |
| Cylindrical / conical surfaces | lens_cyl, lens_rod, axicon |
| Fresnel-facet surface | lens_fresnel |
| Mesh / freeform fallback | mesh_freeform (no `--strict`) |
| Mirrors + beam folding | newtonian, dobsonian, schmidt_cassegrain, czerny_turner, **folded_periscope** |
| One-click fold operator + rigid refold | **folded_periscope** (bit-exact refold asserted) |
| Coherence / interference | michelson, doubleslit |
| Aperture diffraction (Airy) | **airy_singleslit** (disk scale ✅; ring nulls wash at quick budgets — honest limit) |
| Diffraction gratings | czerny_turner |
| Beamsplitter / dichroic TMM coatings | michelson, hot_mirror, pbs_cube (⚠ air-gap) |
| Spectral filters | filter_bandpass |
| Uniaxial birefringence + walk-off | calcite_displacer, wollaston, waveplate_quartz |
| Biaxial birefringence | ktp_walkoff, **biaxial_conoscopy** |
| Polarization (Jones, Malus, extinction) | pol_linear, pol_crossed, pol_circular (⚠ generator-only) |
| Full Stokes/DOP imaging | **stokes_polarimeter** |
| Mie particle scattering | **aerosol_mie** |
| Roughness / diffuser / speckle | **diffuser_speckle**, scatter_plate (ABg) |
| Ghost / stray-light analysis | ghost_doublet |
| Gaussian beams / apodization | gaussian_bench |
| TIR fiber guiding | fiber_coupler |
| Multi-element imaging | camera_triplet, microscope_objective, **telephoto** |
| Variable-driven prescriptions + expression placement | **telephoto** (efl/stop_d), **telephoto_zoom** (rational BFL(z) tracking) |
| Aperture stop / iris + equivalent f-number | camera_triplet, **telephoto** (paraxial f/# readout) |
| Named analysis products (PSF/MTF/EE/Zernike/spot/fans) | **imaging_analysis** |
| Per-source/detector/element charts + CSV | **multiled_photometry** |
| Photometric units (lux/lm) | **multiled_photometry** |
| Tabulated emission spectra (white LED) | **multiled_photometry** (CIE LED-B1) |
| Curved (Petzval) detector | curved_focal (cylinder), **curved_focal_surface** (sphere) |
| Stress / photoelastic birefringence | ⏳ `future.md` backlog (a): stress birefringence + the photoelastic_stress demo spec |
| Merit-function optimization | ✅ `scripts/optimize.py` (scipy + nevergrad/CMA); validation scene `auto_designed_lens` lives in `scripts/make_test_scenes.py`, not the demos/ gallery |
| Monte-Carlo tolerancing + sensitivity/compensators | ✅ `scripts/tolerance.py`; validation scene `tolerance_lens` lives in `scripts/make_test_scenes.py`, not the demos/ gallery |
| CAD import / illumination design | 🔶 `future.md` backlog (d) acceptance-target demos |

## The train workflow, in one demo

Open `michelson_folded.MieWB` and look at the **Optical Train** dock:
the beamsplitter's two arms hang off its `transmit`/`reflect` ports,
FoldA/FoldB carry fold checkboxes, and every distance is an expression
over the **Variables** dock (`arm1 - fold_in - fold_up`, …). Uncheck
both folds: the dogleg straightens into the plain michelson layout, the
fold mirrors ghost in the viewport and drop out of the simulation
(File → Export FCStd writes either configuration as a standalone,
plain-FreeCAD-editable file). Tick `arm2`'s Sweep box and Run: after
the run-count/estimate confirmation, the **Compare** dock fills with
power/visibility-vs-arm2 plots, a per-variant fringe gallery, and
difference maps. Efficiency A/B (quick preset, seed-matched): unfolded
1.088 mW → folded with ideal mirrors 1.084 mW (folding geometry is
free) → folded with bare-aluminum mirrors 0.857 mW (the four extra
reflections cost 21% — set `ideal_folds` to 1 to see the difference).

## Prescription sources

- **Cooke triplet**: MathWorks "Design a Cooke Triplet" published surface
  table, uniformly rescaled to 50 mm EFL (scale-invariant optics); crowns
  as BK7, flint as SF5 (nearest shipped glasses), sensor plane re-solved
  paraxially for those indices.
- **Schmidt-Cassegrain**: R. Suiter, "Design of the Schmidt-Cassegrain"
  (bay-astronomers.org), C8-class table; corrector uses the classic
  single-wavelength profile z = K[r⁴ − (3/2)a²r²], K = 1/(4(n−1)R_m³),
  neutral zone at 0.866a (Schroeder, *Astronomical Optics*).
- **Newtonian/Dobsonian diagonal sizing**: standard ATM formulas (Kriege &
  Berry, *The Dobsonian Telescope*); the demos use round flats — a
  circular mirror at 45° must be cone_diameter/cos 45° across.
- **Ball lens**: BFL = R(2−n)/(2(n−1)) (Edmund Optics ball-lens notes);
  at 0.6 mm beam diameter the focused cone stays inside the fiber's
  0.22 NA.
- **Fiber core index**: `fiber_core_na22` registry row — Sellmeier
  interpolation between SiO₂ (Malitson 1965) and GeO₂ (Fleming 1984,
  Appl. Opt. 23, 4486) pinned to NA 0.220 vs fused silica at 650 nm.
- **Michelson fringe math**: pitch = λ/(2θ) (Hecht, *Optics*); 0.158 mrad
  → 5 fringes across 10 mm at 633 nm.
- **Czerny-Turner angles**: grating equation d(sin θᵢ + sin θ_d) = mλ with
  d = 1/600 mm, θᵢ = −6.127°, solved so 400–700 nm spans 25 mm at
  f = 100 mm (C. Palmer, *Diffraction Grating Handbook*).
- **Prism**: minimum-deviation relation n = sin((A+D)/2)/sin(A/2) with
  the shipped SF5 Sellmeier row.
- **Lister objective**: reconstruction from Lister's 1830 two-doublet
  principle (Phil. Trans. R. Soc. 120) using the shipped BK7/SF5 achromat
  design scaled by `wizards.solve_achromat` — a teaching model, not a
  commercial prescription.
- **KTP biaxial indices** (`ktp_walkoff`): the shipped `ktp_nx/ny/nz`
  Sellmeier rows are an exact fold of Kato & Takaoka, "Sellmeier and
  thermo-optic dispersion formulas for KTP," Appl. Opt. **41**, 5040
  (2002) (the canonical biaxial oracle); the 45°-in-plane geometry is the
  maximum-walk-off cut, where the in-plane sheet behaves as a uniaxial
  e-wave with n_o=n_x, n_e=n_z and the out-of-plane sheet is n_y with zero
  walk-off (Yariv & Yeh, *Optical Waves in Crystals*, §4).
- **KTP optic-axis angle** (`biaxial_conoscopy`): KTP has n_x<n_y<n_z with
  n_y much nearer n_x (positive biaxial), so the acute bisectrix is the Z
  principal axis and the two optic axes lie in the X–Z principal plane at
  ±V_z from Z, where sin²V_z = (n_z²(n_y²−n_x²)) / (n_y²(n_z²−n_x²)); the
  shipped Kato & Takaoka (2002) Sellmeier rows give V_z ≈ 17–18° at 589 nm
  (2V_z, the acute optic-axial angle, ≈ 35°) — consistent with the
  literature KTP acute 2V (Bierlein & Vanherzeele, "Potassium titanyl
  phosphate," J. Opt. Soc. Am. B **6**, 622 (1989)). The demo places Z
  along the viewing axis (centred acute-bisectrix figure) and spins the
  crystal 45° so the optic-axial plane lies at 45° to the crossed
  polarizers (Bloss, *An Introduction to the Methods of Optical
  Crystallography*, ch. on interference figures).
- **CIE white-LED spectrum** (`multiled_photometry`): the `led_white_2733k`
  emission-registry row is the CIE standard illuminant **LED-B1** spectral
  power distribution (phosphor-converted blue LED, CCT ≈ 2733 K) from CIE
  015:2018, *Colorimetry, 4th ed.*, Table 12.1; the three monochromatic LED
  primitives are asymmetric-Gaussian datasheet fits (Cree XP-E2 / Lumileds
  LUXEON Z bins, per their `.meta.json`). The photometric block applies the
  CIE V(λ) photopic weighting to the same spectral cube (`--photometric`).
- **Gaussian beam** (`gaussian_bench`): standard TEM₀₀ propagation
  w(z)=w₀√(1+(z/z_R)²), z_R=πw₀²/λ (Siegman, *Lasers*, ch. 17); the M²
  factor scales the far-field divergence (ISO 11146). The waist sits at
  the emitting face; the incoherent beam-mode deposits directly (no
  coherent-gather dA subtlety).
- **Fresnel ghosts** (`ghost_doublet`): each uncoated air/glass surface
  reflects R=((n−1)/(n+1))² at normal incidence; a double bounce between
  any two surfaces returns a forward ghost of the direct beam × R² (Fest,
  *Stray Light Analysis and Control*, SPIE PM229; Peterson, "Analytic
  expressions for in-field scattered light," SPIE proc.). `--ghost-analysis`
  tags each detector ray with its reflection history and ranks the paths.
- **ABg scatter** (`scatter_plate`): the `polished_bk7_glass` registry row
  is a representative three-parameter ABg BSDF fit (Pfisterer,
  "Approximated Scatter Models for Stray Light Analysis," Optics &
  Photonics News 2011; Harvey et al., Opt. Eng. **51**, 013402, 2012); the
  tracer splits the reflected power into a specular ray plus a scattered
  lobe whose total integrated scatter is set by the ABg TIS.
- **Curved detector** (`curved_focal`): the cylindrical screen is a
  standard curved focal surface (as in a Rowland-circle spectrograph); the
  `CurvedDetectorGrid` accumulates per-pixel power on the analytic
  Sphere/Cylinder face with the exact area element (R·du·dv), so total
  detected power is curvature-invariant.
- **Mie aerosol** (`aerosol_mie`): exact Mie scattering (validated vs
  Wiscombe, MiePlot/Bohren & Huffman) off a log-normal water-droplet
  ensemble carried on the `--particles` continuum medium; extinction is
  Beer–Lambert with the ensemble-averaged cross-section, in-scattering
  samples the polarized differential cross-section (Bohren & Huffman,
  *Absorption and Scattering of Light by Small Particles*). Water n(λ) from
  the shipped registry; the geometry (LIDAR/nephelometer probe + forward +
  90° detectors) follows the classic single-scatter chamber.
- **Ground-glass diffuser** (`diffuser_speckle`): the `@dg_600` scatter is
  the deep-rough Beckmann/ground-glass limit (§5.4.1 of docs/RAYTRACER.md);
  the ~5° FWHM cone and grit→angle mapping follow standard ground-glass
  diffuser data (Edmund/Thorlabs diffuser notes). Coherent speckle and
  partial depolarization are the single-scatter model's honest outputs.
- **Airy diffraction** (`airy_singleslit`): circular-aperture Fraunhofer
  pattern, first dark ring at 1.22 λL/D (Airy 1835; Hecht, *Optics* §10.2);
  the pinhole primitive's air-filled plug implements the aperture contract
  (docs/RAYTRACER.md §5.10). Complements the existing `doubleslit` fringe
  bench.
- **Imaging analysis** (`imaging_analysis`): reuses the Cooke-triplet
  prescription (see camera_triplet above); the analysis products (PSF,
  FFT-MTF tangential/sagittal, Strehl via Maréchal, encircled energy,
  Standard/Fringe Zernike wavefront, spot + transverse/OPD ray fans) are
  the standard image-quality metrics (Born & Wolf; Mahajan, *Optical
  Imaging and Aberrations*). Products render from `--save-fields` (PSF/MTF/
  EE) and `--export-rays` (Strehl/Zernike/spot/fans).

## Physics caveats (honest limits)

- The fiber demo's core/cladding boundary uses the engine's 5 µm
  optical-contact air gap: rays inside the fiber NA guide exactly as they
  should; cladding-mode/leaky-ray power is not quantitative.
- The camera triplet's aberration correction is approximate (BK7/SF5
  substituted for the design glasses); the paraxial focus is exact.
- Detector quick-preset images are Monte-Carlo: expect noise, and expect
  zero-mean negative pixels in coherent scenes (clip only for display).
- `aerosol_mie` runs at a dense fog loading (phi=2e-2, ~24 g/m³ liquid
  water) so the single-scatter physics is visible in one 40 mm pass; a
  realistic atmospheric aerosol (phi~1e-6) is optically negligible over
  this path (τ~5e-5). The continuum medium is single-scatter (no multiple
  scattering between droplets), which is accurate at τ≈1 but would
  under-count the diffuse halo at much higher τ.
- `airy_singleslit` reproduces the Airy central-disk SCALE (~1.22 λL/D) but
  not the resolved ring nulls: the coherent gather yields a smooth
  azimuthal profile at the quick budget (adding rays only reduces speckle,
  not the null depth). Treat it as an aperture-diffraction / disk-scale
  demo, not a ring-metrology reference.
- `imaging_analysis` characterizes an ABERRATED lens (BK7/SF5 substituted
  for the Cooke design glasses → Strehl ~0, ~38 waves RMS at Ø5); it
  demonstrates that the full analysis pipeline runs and produces finite,
  self-consistent PSF/MTF/Strehl/Zernike/EE/spot/fan products, not a
  diffraction-limited wavefront.
