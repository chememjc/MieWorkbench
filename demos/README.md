# MieWorkbench demo gallery

Ten classic optical systems, each a self-contained `.MieWB` workbench
archive (double-click-open in the GUI, or run headlessly) plus the bare
`.FCStd` scene. All were assembled **through the GUI's own op path** by
`scripts/make_demos.py` — rebuild any of them with:

```bash
python3 scripts/make_demos.py --demo <name>        # or 'all'
python3 scripts/miewb_tool.py run demos/<name>.MieWB -o /tmp/<name>.MieSim
env/bin/python -m mieworkbench demos/<name>.MieWB  # open in the GUI
```

Every demo completes on the `quick` preset (1e5 rays, 512², 5λ) with the
energy ledger closing below 1e-3. `demos/UXNOTES.md` records the friction
found while building these through the interface (and the two real bugs
the exercise caught).

| Demo | System | What it shows | Detected (of 5 mW, quick preset) |
|---|---|---|---|
| `beam_expander` | 3× Keplerian expander: BK7 PCX f=50 + f=150, spacing f1+f2, convex sides out | Collimation preserved, 3× beam diameter; loss = the four uncoated Fresnel surfaces (0.96⁴ ≈ 0.85) | **4.23 mW** |
| `newtonian` | 150 mm f/6: parabolic primary (rfl 900), 45° round diagonal, folded focus, WHITE-LIGHT star (450–650 nm) | Exact parabolic focus, 90° fold, central-obstruction shadow; loss = two Al bounces + obstruction | **3.58 mW** |
| `dobsonian` | 200 mm f/5 Newtonian optics, white-light star (a Dobsonian is the same telescope on an alt-az mount) | Faster, larger variant of the above | **3.68 mW** |
| `michelson` | 25 mm 50:50 **plate** beamsplitter at 45°, 60 mm arms, one mirror tilted 0.158 mrad | Coherent two-beam interference: straight fringes (measured visibility 0.90) across the detector at 633 nm, pitch λ/2θ | **1.05 mW** at the fringe port |
| `prism_spectrometer` | 25 mm equilateral SF5 prism at minimum deviation (550 nm), f=100 camera lens | Chromatic dispersion: 450–650 nm spread ~2.3° → a ~4 mm spectrum (the honest prism-vs-grating tradeoff) | **0.60 mW** |
| `czerny_turner` | Crossed CT: divergent slit source, R=200 collimator, 600 g/mm reflective grating (`mirror=1.0`), R=200 camera mirror | Grating dispersion + off-axis mirror folding; 400–700 nm across ~25 mm, first order | **0.08 mW** (slit + overfill + order efficiency) |
| `camera_triplet` | Cooke triplet ~50 mm EFL (published design rescaled), iris stop ~f/5.6, 36×24 mm sensor, white-light scene (450–650 nm) | A real multi-element photographic objective; detected ≈ the f/5.6 pupil fraction of the 14 mm input beam | **1.26 mW** |
| `microscope_objective` | Lister-type: two air-spaced achromats (f=25 + f=50, 10 mm apart), finite conjugates, white-light point source | Aberration-corrected imaging of a point source | **2.79 mW** |
| `fiber_coupler` | 650 nm laser → 2 mm BK7 ball lens (BFL 0.47 mm) → 75 mm of 200 µm/0.22 NA fiber | TIR guiding down the fiber core (~60 bounces; `max_reflections` simparam) | **3.95 mW** at the exit face |
| `schmidt_cassegrain` | C8-class 203 mm f/10: quartic Schmidt corrector (hand-authored asphere), perforated spherical primary R 812.8, spherical secondary R 231.07, white-light star | Catadioptric folding: corrector → primary → secondary → focus through the primary's hole; loss ≈ two Al bounces + 11 % obstruction | **3.42 mW** |

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

## Physics caveats (honest limits)

- The fiber demo's core/cladding boundary uses the engine's 5 µm
  optical-contact air gap: rays inside the fiber NA guide exactly as they
  should; cladding-mode/leaky-ray power is not quantitative.
- The camera triplet's aberration correction is approximate (BK7/SF5
  substituted for the design glasses); the paraxial focus is exact.
- Detector quick-preset images are Monte-Carlo: expect noise, and expect
  zero-mean negative pixels in coherent scenes (clip only for display).
