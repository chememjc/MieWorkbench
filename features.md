# MieWorkbench — Competitive Feature Analysis

**Scope:** MieWorkbench (this repository) vs **Ansys Zemax OpticStudio (Premium)**,
**QUADOA Optical CAD**, and **3DOptix**.
**Method:** the MieWorkbench column is drawn from `docs/RAYTRACER.md`, `README.md`,
`future.md`, and the `mieworkbench/` + `scripts/raytracer/` source. Competitor columns
were built from live vendor documentation, manuals, SDK sources, and pricing pages
(July 2026); claims that could not be pinned to a primary source are marked *unverified*.
**Ratings:** ✅ full · 🟡 partial / shallower · ⚠️ workaround only · ❌ none.

> **Honesty note.** MieWorkbench is a physically-based **coherent Monte-Carlo,
> fully-vectorial non-sequential ray tracer**. It is a *simulation engine*, not a
> lens-*design* suite. Three of the four packages here (Zemax, QUADOA, and to a lesser
> extent 3DOptix) are design/analysis products with optimization, tolerancing, and a
> named imaging-analysis suite that MieWorkbench does not attempt. This document is
> deliberately unflattering about those gaps so the roadmap that follows is grounded in
> reality rather than advocacy.

---

## 1. Executive summary

**Overall winner, general feature set: Ansys Zemax OpticStudio (Premium).** It is the most
complete package on virtually every axis — the full sequential + non-sequential engine,
Physical Optics Propagation, the entire named analysis suite (MTF/PSF/Zernike/Strehl/
encircled energy), optimization, tolerancing, thermal, illumination/photometry, huge glass
catalogs, RCWA gratings, CAD import, and a deep scripting API. Nothing else here is close on
raw breadth.

**But breadth is not the whole story, and MieWorkbench is not trying to be Zemax.**
MieWorkbench wins a specific, defensible cluster of line items that matter enormously for
*physical-optics fidelity* work: a **coherent Huygens/Rayleigh–Sommerfeld field gather that
runs by default on every non-sequential scene and books absolute interferometric power**
(Zemax's non-sequential "coherent" mode is a geometric coherent ray-sum — phase referenced to
the pixel center, with no Huygens obliquity/spreading — so aperture diffraction needs a separate
scalar POP pass; 3DOptix's default is a gated ray-approximation; QUADOA's is a paid wave-optics
add-on); **exact Mie particle-cloud scattering** validated against Wiscombe MIEV0 (Zemax offers
Mie only through a bundled bulk-scatter DLL; QUADOA and 3DOptix omit it entirely); a **closed
9-bucket energy-audit ledger** with `<1e-3` closure that none of the commercial tools match;
**GPU/CUDA acceleration of the coherent gather** (Zemax's *ray tracing* is CPU-only — though it
GPU-accelerates Huygens PSF/MTF — and QUADOA is CPU-only); **native Stokes/DOP maps + validated
uniaxial birefringence with walk-off**; and **zero cost, full data locality, Linux-native,
headless-CLI, text-based (ZIP-container) formats.**

**Positioning of the field, one line each:**
- **Zemax OpticStudio Premium** — the complete professional standard; the yardstick.
- **QUADOA Optical CAD** — a modern *sequential imaging-design* powerhouse (optimization,
  tolerancing, Forbes/GRIN/freeform surfaces, full analysis suite) that deliberately has **no
  non-sequential engine** and no volume/Mie scattering — the philosophical inverse of MieWorkbench.
- **3DOptix** — a free/freemium, browser-based, GPU-cloud **non-sequential** tracer whose
  killer feature is a ~50,000-part real-vendor component catalog and near-zero onboarding
  friction; its wave-optics and polarization are real but shallow and tier-gated.
- **MieWorkbench** — a free, Linux-native, GPU-accelerated **coherent non-sequential physics
  engine** with best-in-class energy bookkeeping, Mie scattering, and polarization physics,
  but no optimization/tolerancing/design apparatus and a steep authoring workflow.

**Head-to-head verdicts (detail in §6):**
- **Zemax vs MieWorkbench → Zemax** (breadth is overwhelming; MieWorkbench wins ~10 physics/
  compute/cost line items).
- **QUADOA vs MieWorkbench → QUADOA overall**, but the two are more *complementary than
  competitive* — QUADOA owns design/optimization/analysis; MieWorkbench owns non-sequential
  physical transport, volume/Mie scattering, coherent-by-default, and the energy ledger.
- **3DOptix vs MieWorkbench → narrow, and it depends on the user.** 3DOptix wins the
  *usability/catalog/accessibility* axis decisively; MieWorkbench wins the *physics-depth*
  axis decisively. By raw line-item count 3DOptix edges ahead on breadth; for a physics-first
  optical engineer MieWorkbench is the more capable *engine*.

---

## 2. Package profiles

### 2.1 MieWorkbench (this repo)
A PySide6 + VTK desktop GUI wrapped around a coherent Monte-Carlo, fully-vectorial (Jones)
**non-sequential** ray tracer driven by annotated FreeCAD models. Physics: real interference/
diffraction via a Huygens/Rayleigh–Sommerfeld final gather; uniaxial birefringence with
walk-off; TMM coatings; four grating models (lamellar, Kogelnik VBG, Dammann, measured
table); Beckmann roughness + ground-glass diffusers; **exact Mie particle clouds** (validated
vs Wiscombe MIEV0). Detectors are planar, producing irradiance cubes, spectra, Stokes/DOP
maps, per-element power tables, and a **9-bucket energy-audit ledger** (closure gated at
`1e-3`). GPU-accelerated coherent gather (CUDA/torch). No optimization, no tolerancing, no
sequential design. Free, Linux, self-hosted; portable `.MieWB`/`.MieSim` ZIP formats + a
headless CLI. Design philosophy: *trace what a ray actually hits, and get the physics right,
with an auditable energy balance.* Target user: a physics-literate optical engineer/
researcher who needs interference, polarization, scattering, and stray-power fidelity, not a
merit-function lens designer.

### 2.2 Ansys Zemax OpticStudio (Premium)
The industry-standard optical design suite. Two paradigms in one product: **sequential** (Lens
Data Editor, surface-order tracing, the natural home of imaging design/optimization/
tolerancing) and **non-sequential** (NSC editor, physical hit-order, splitting/scatter,
illumination/stray-light), bridged by a mixed mode. Premium adds over Professional: **Physical
Optics Propagation** (gridded complex-field propagator with angular-spectrum/Fresnel
auto-switch), CAD part import, **RCWA** grating efficiency, TrueFreeForm, Radiant measured
sources, and advanced stray-light (Path Analysis / Critical Ray Tracer). *Note:* full
optimization (incl. Global/Hammer), tolerancing (incl. Monte-Carlo), and non-sequential mode
are **Professional-and-up**, not Premium-exclusive; **STAR (STOP/FEA import) is Enterprise-only
— not in base Premium.** Windows-only; ray tracing is CPU-only (Zemax deliberately rejected GPU
*for tracing*, but does GPU-accelerate Huygens PSF/MTF and single-mode fiber coupling);
deep ZOS-API (C#/Python/MATLAB, 4 modes) + ZPL macros. Commercial subscription (quote-only;
dated third-party band ≈ $4.9k–$14.9k/yr across editions). Target user: professional lens and
illumination designers.

### 2.3 QUADOA Optical CAD
A modern (Berlin, GmbH founded 2021) **sequential / multi-sequential** design suite built on an
object/assembly hierarchy (base surface + stacked parametric layers: form, aperture, phase,
polarization, coating, array) with the light path expressed as a separate reusable **Sequence**
object — its "Multi-Sequential Raytracing" USP. Strong where Zemax is strong: full
optimization (DLS + trust-region, glass substitution, multi-config), tolerancing (sensitivity
+ 9-distribution Monte-Carlo + chainable compensators), the full analysis suite (FFT/Huygens/
geometric PSF & MTF, Zernike, Strehl, encircled energy, ray fans, ghost analysis), Forbes
Q-type / biconic / freeform / **GRIN** surfaces, Schott+Ohara catalogs, STEP/IGES/STL import +
Zemax ZMX/SEQ interop, and a Python/MATLAB/C++ SDK. **Deliberately has no non-sequential
engine** ("under development"), **no volume/Mie/participating-media scattering**, and only
scalar (efficiency-free) phase gratings. Windows + Linux native. Commercial, quote-priced,
perpetual or subscription + à-la-carte toolboxes (the Wave-Optics toolbox is a paid add-on).
Target user: imaging/AR-VR/metrology/medical lens designers.

### 2.4 3DOptix
A free/freemium, **browser-based, GPU-cloud non-sequential** ray tracer. Its standout is a
**~50,000-component real-vendor catalog** (Thorlabs, Edmund, Semrock, Chroma, …) with correct
prescriptions and mountable cage/breadboard hardware — a genuine digital-twin-to-bench
workflow, plus its own BreadBox™ hardware line. Non-sequential geometric core is mature
(ray-splitting, MAX BOUNCES, 4 BRDF scatter models, manual stray-light workflow). It **does**
have a real scalar Huygens/Fresnel diffraction solver and coherent detection — but these are
advanced-tier, gated features layered on a default **ray-based interference approximation**
that the vendor itself describes as "physical optics using geometric optics," and its
MTF/PSF/wavefront/polarization products are internally contradicted on its own site (several
still marked "coming soon"). Polarization is shallow — component power (x/y) plus a spatial
circular-polarization map, but no exposed Jones/Mueller formalism or DOP product. No
optimization, no tolerancing, no thermal (all "coming soon" or
SciPy-only). Metered GPU-hours; ray ceilings and wave optics gated by tier (free tier is a
single 550 nm wavelength, no API). Python SDK (beta, paywalled). Cross-platform browser;
cloud-only (no offline, no data locality). Target user: students, educators, and lab
engineers laying out benchtop systems from catalog parts.

---

## 3. Detailed per-package feature write-ups

Organised by the shared taxonomy (A–R). Line-item ratings are consolidated in the master
table (§4); this section gives the qualitative shape of each package.

### 3.1 MieWorkbench
- **A Ray-tracing core:** non-sequential, stratified Monte-Carlo, coherent **and** incoherent
  simultaneously (per gather key), reflection-generation cap (default 6, adjustable to ~200),
  BVH for mesh faces, exact analytic intersection for canonical surfaces. No sequential mode.
- **B Physical optics:** the differentiator — a full coherent Huygens/Rayleigh–Sommerfeld
  gather with obliquity factor is the *default* engine; real interference + diffraction on
  every coherent run (double-slit pitch/visibility validated end-to-end). PSF is *implicit* in
  the focal image; **no named MTF/OTF, Zernike, Strehl, or encircled-energy product**; no
  ABCD/Gaussian-beam POP.
- **C Polarization:** full Jones tracing; Stokes + degree-of-polarization maps; **validated
  uniaxial birefringence with walk-off** (calcite 6.23°@45°/590 nm); TMM coatings; four grating
  models with **real diffraction efficiency** (Kogelnik VBG — transmission geometry only,
  reflection VBGs not modeled — Dammann, measured tables). No biaxial, no optical activity, no
  RCWA, no Mueller-matrix formalism.
- **D Surfaces:** spherical, even-asphere (conic + polynomial, extract-verified `<1 µm`),
  cylindrical, conical, toroidal — analytic. Freeform only via triangle **mesh (incoherent-
  power only)**. DOE via the grating models. **No GRIN. No CAD import** beyond FreeCAD-native.
- **E Sources:** collimated + divergent; mono/uniform/Gaussian spectra; coherent/incoherent;
  full polarization states. **No ray-aiming, no apodization, no Gaussian-beam source, no
  measured source files.**
- **F Detectors/analysis:** planar irradiance cube, spectra, Stokes/DOP, per-element power
  table, and the **9-bucket energy-audit ledger with `<1e-3` closure** — the strongest energy
  bookkeeping in the field. No curved detectors, no MTF/Strehl/encircled-energy/spot/ray-fan,
  no dedicated ghost tool.
- **G Materials:** ~24 materials (Sellmeier/Cauchy/constant/tabulated), 10 TMM coatings, small
  grating/filter/uniaxial libraries; every row requires a cited reference. **Tiny vs commercial
  catalogs.** No dn/dT.
- **H Scattering:** Beckmann roughness + diffusers + **exact Mie particle clouds** (explicit
  spheres up to a ~200k threshold, above which it switches to a continuum participating
  medium). **No measured BSDF import, no
  importance sampling, no dedicated stray-light tool.**
- **I/J/K/L/M Optimization / Tolerancing / Thermal / Photometry / Multi-config:** **none**
  (only CLI `--var` geometry sweeps and `--seeds` speckle averaging).
- **N GUI/UX:** VTK 3D view, 54-element parametric primitive library, ~20-level undo + macros,
  wizards (thick-lens solvers, waveplate solver), 1 s-debounced live ray preview, **tracer-bead
  animation** (photons at c/n), results galleries, validation/problems pane. Linux desktop,
  steep authoring curve.
- **O Coordinate system:** FreeCAD Placement (position + quaternion), absolute world pose +
  Euler, reference-point resolver (origin/CoM/optical-center/bbox/face-normal), **expression-
  bound placements routed through their driving alias**, **snap-to-optical-axis + drag-along-
  axis**, "Relative to" readout. Purely **spatial** — no sequential surface table, no
  assembly/pickup abstraction.
- **P Data:** `.MieWB`/`.MieSim`/`.FCStd` ZIP formats, headless CLI (`miewb_tool`), case
  locking; text-based inner members in a ZIP container. Scripting = external Python CLI only (no in-app API/macro).
- **Q Performance:** single-process vectorized numpy trace + **CUDA/torch coherent gather** (the
  only GPU engine of the four's design intent besides 3DOptix's cloud). No multi-process/
  multi-GPU (`--workers` unimplemented). Gather-dominated (hours at high resolution).
- **R Commercial:** free, Linux, self-hosted, full data locality; no formal support/community;
  steep learning curve; no vendor component catalog.

### 3.2 Zemax OpticStudio (Premium) — strengths that dominate
The complete analysis suite (B/F), optimization (I), tolerancing (J), thermal (K), illumination
+ photometry with IES/LDT and roadway design (L), the largest glass catalogs + 13 dispersion
models (G), RCWA gratings (C), measured BSDF + importance sampling + Path Analysis stray-light
(H), 20+ surface types + GRIN + diffractive phase surfaces + STEP/IGES/SAT/Parasolid import
(D), ~20 NSC source objects + ray-aiming + apodization (E), the 4-mode ZOS-API + ZPL + CSG
Booleans (P), and 160-config multi-config/zoom/scanning (M). Genuine *gaps* relative to
MieWorkbench: NSC "coherent" is a geometric coherent ray-sum with no Huygens spreading (rigorous diffraction is a
separate scalar POP pass, and POP is scalar, not vectorial); **ray tracing is CPU-only** (Zemax
GPU-accelerates only Huygens PSF/MTF and fiber coupling); **Mie only via a bundled bulk-scatter
DLL (MSP), not a first-class validated solver**; Stokes is ensemble-derived (no
Mueller); flux accounting is distributed, not a closed audit; **biaxial and optical activity
are absent** (as in MieWorkbench); Windows-only; expensive.

### 3.3 QUADOA — strengths and the paradigm inversion
Owns the entire *design* apparatus MieWorkbench lacks: optimization (I), tolerancing (J, with
9 MC distributions + chainable compensators), thermal (K), multi-config/zoom (M), the full
analysis suite (B/F), Forbes Q-type/biconic/freeform/**GRIN**/off-axis-asphere surfaces + CAD
& Zemax interop (D), real glass catalogs + 4 dispersion models (G), ray-aiming + apodization +
Gaussian-beam + imported ray-file sources (E), a **Stokes + Poincaré polarization suite**
(Mueller claimed in marketing but unconfirmed in the feature set) (C), and a Python/MATLAB/C++ SDK (P). Its coordinate model is a genuine *superset* of
MieWorkbench's (absolute placement **and** sequential chaining **and** nested assemblies **and**
parametric pickups, per-object switchable). *Gaps:* **no non-sequential engine at all**
(ghosts only, 1st/2nd-order-capped); **no volume/Mie/participating-media scattering**; gratings
are scalar and **efficiency-free** (no Kogelnik/Dammann/RCWA); bulk birefringence is claimed
but unverified in the manual; wave optics is a **paid add-on**; **CPU-only** (no GPU per its
own manual); no documented headless/CLI mode.

### 3.4 3DOptix — strengths and the shallow-physics caveat
Owns *accessibility*: browser, zero install, drag-and-drop from a **~50,000-part real-vendor
catalog** with mountable cage/breadboard hardware (O/G/N), structured tutorials + Academy,
cloud GPU with billion-ray ceilings on paid tiers (Q), real STEP CAD import (D), measured IES/
TM-25 source files (E), a genuine (if narrow) Huygens diffraction solver and coherent detection
(B), and PSF/MTF/spot/encircled-energy products MieWorkbench lacks (F). *Caveats, heavily
sourced:* wave optics and polarization are **shallow and tier-gated** — the default interference
mechanism is a ray-based approximation the vendor concedes is "not true wave physics";
polarization is shallow — component power plus a circular-polarization spatial map, but with
**no exposed Jones/Stokes/Mueller formalism**; coatings are 6
idealized presets (no TMM stack designer); **no optimization, tolerancing, thermal, or
photometric units**; MTF/PSF/wavefront/alignment features are variously "coming soon"; cloud-
only (no data locality); metered GPU-hours; small vendor with no independent review base.

---

## 4. Master comparison table

Legend: ✅ full · 🟡 partial/shallower · ⚠️ workaround · ❌ none. **MWB** = MieWorkbench.
"Best" names the strongest implementation; ties noted.

### A. Ray-tracing core
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|A1|Non-sequential tracing|✅|✅|❌|✅|**Zemax** — deepest NSC (splitting, budgets, mixed mode); MWB/3DOptix full; QUADOA none|
|A2|Sequential design tracing|❌|✅|✅|🟡|**Zemax** — canonical LDE; QUADOA strong; MWB none|
|A3|Monte-Carlo radiometric transport|✅|✅|❌|🟡|**Zemax/MWB** — true photon MC; QUADOA only MC *tolerancing*|
|A4|Coherent field tracing, absolute power|✅|🟡|🟡add-on|🟡gated|**MWB** — inline coherent gather books absolute power; others normalized/gated/add-on|
|A5|Ray splitting / bounce control|✅|✅|⚠️ghosts|✅|**Zemax** — full budgets; QUADOA emulates via ghost sequences|
|A6|Spatial acceleration (BVH)|🟡mesh|✅|❌|✅GPU|**3DOptix** — GPU-accelerated; Zemax & MWB both use spatial acceleration (MWB's mesh BVH is documented)|
|A7|Analytic vs mesh optical surfaces|✅+mesh|✅+CAD|✅analytic|✅+CAD|**Zemax** — analytic + true NURBS/ACIS import|

### B. Physical optics
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|B1|Real interference (default)|✅|✅POP|✅|🟡approx|**MWB/Zemax** — MWB inline-default; Zemax rigorous POP|
|B2|Free-space diffraction|✅|✅|✅|🟡|**Zemax** — Angular-spectrum/Fresnel POP; MWB Rayleigh–Sommerfeld gather full|
|B3|POP / beam propagation (gridded/ABCD)|⚠️|✅|✅add-on|🟡|**Zemax** — full gridded complex-field POP + ABCD|
|B4|PSF (named product)|⚠️implicit|✅FFT+Huygens|✅3 engines|🟡geo|**Zemax/QUADOA** — FFT+Huygens+geometric|
|B5|MTF / OTF|❌|✅|✅|🟡|**Zemax/QUADOA** — FFT/Huygens/geo MTF + through-focus|
|B6|Wavefront maps / Zernike|❌|✅3 bases|✅|❌|**Zemax** — Standard/Fringe/Annular Zernike|
|B7|Strehl ratio|❌|✅|✅|❌|**Zemax/QUADOA**|
|B8|Partial coherence modeling|❌|🟡Γ-model|🟡binary|❌|**Zemax** — parametric mutual-coherence|
|B9|Gaussian-beam (ABCD) analysis|❌|✅|✅|🟡|**Zemax** — paraxial + skew Gaussian|

### C. Polarization
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|C1|Jones-vector tracing|✅|✅|✅|🟡|**Zemax/QUADOA/MWB** — full Jones; 3DOptix shallow (component power + circ-pol map)|
|C2|Stokes / DOP maps|✅|🟡ensemble|✅|🟡map|**QUADOA/MWB** — first-class Stokes maps (QUADOA + Poincaré); 3DOptix circ-pol map only|
|C3|Mueller-matrix formalism|❌|⚠️|🟡claim|❌|**QUADOA** (unconfirmed) — Mueller claimed in marketing, not in feature list; Zemax via Jones-probe workaround|
|C4|Uniaxial birefringence + walk-off|✅|✅|🟡unverif|❌|**MWB/Zemax** — validated walk-off physics|
|C5|Biaxial birefringence|❌|❌|❌|❌|*none* — universal gap|
|C6|Optical activity / gyrotropy|❌|❌|❌|❌|*none* — universal gap|
|C7|TMM thin-film coatings|✅|✅|✅|⚠️presets|**Zemax** — largest catalog; MWB/QUADOA full TMM; 3DOptix 6 presets|
|C8|Polarizers / retarders / waveplates|✅|✅|✅|🟡|**Zemax/QUADOA** — ideal + real; MWB real via crystals|
|C9|Grating diffraction efficiency|✅models|✅RCWA|❌scalar|🟡|**Zemax** — rigorous RCWA; **MWB** best *closed-form* (Kogelnik/Dammann/table); QUADOA efficiency-free|

### D. Surfaces & geometry
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|D1|Spherical / conic|✅|✅|✅|✅|*tie*|
|D2|Even/odd polynomial asphere|✅|✅|✅|✅|**Zemax** — even/odd/extended; all support asphere|
|D3|Q-type (Forbes) asphere|❌|✅|✅|❌|**Zemax/QUADOA**|
|D4|Freeform (XY/Zernike/Chebyshev/grid)|⚠️mesh|✅20+|✅|⚠️claim|**Zemax** — TrueFreeForm + many analytic bases|
|D5|Diffractive / DOE / binary phase|✅grating|✅Binary1-3|✅phase|✅grating|**Zemax** — Binary 1/2/3 + grating|
|D6|Fresnel surfaces|✅facet|✅|✅|❌|**Zemax/QUADOA** — analytic Fresnel|
|D7|Toroidal / biconic|✅|✅|✅|🟡biconic|**Zemax/QUADOA/MWB**|
|D8|GRIN media|❌|✅|✅|❌|**Zemax/QUADOA** — analytic GRIN + DLL/Eikonal|
|D9|Mesh / CAD-imported optical surface|🟡incoh|✅|❌mech|🟡unstable|**Zemax** — true solid tracing|
|D10|CAD import (STEP/IGES/SAT/STL)|❌|✅|✅|🟡STEP|**Zemax** — widest kernel support|

### E. Sources
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|E1|Geometric source types|🟡2|✅20+|✅|✅|**Zemax** — ~20 NSC emitters|
|E2|Spectral definition|✅|✅|✅|🟡|**Zemax/MWB/QUADOA** — MWB adds Gaussian band|
|E3|Coherent source (inline)|✅|🟡POP|✅add-on|🟡gated|**MWB** — coherent inline by default|
|E4|Gaussian-beam source|❌|✅|✅|🟡noM²|**Zemax/QUADOA**|
|E5|Apodization|❌|✅|✅|❌|**Zemax/QUADOA**|
|E6|Ray-aiming to real pupil|❌|✅|✅|❌|**Zemax/QUADOA**|
|E7|Measured source files (IES/LDT/rayfile)|❌|✅|🟡rayfile|✅IES/TM25|**Zemax** — IES/LDT/rayfile/Radiant|
|E8|Full polarization state on source|✅|✅|✅|🟡|**Zemax/QUADOA/MWB**|

### F. Detectors & analysis
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|F1|Irradiance maps|✅|✅|✅|✅|*tie*|
|F2|Spectral / color detector|✅spectra|✅color|🟡|✅spectral|**Zemax** — CIE color + per-pixel spectrum|
|F3|Curved detectors|❌|✅|🟡|❌|**Zemax** — curved/annular detector objects|
|F4|Spot diagrams|❌|✅|✅|✅|**Zemax/QUADOA**|
|F5|Ray fans (OPD/transverse)|❌|✅|✅|❌|**Zemax/QUADOA**|
|F6|Encircled / ensquared energy|❌|✅|✅|🟡circ|**QUADOA** — encircle/ensquare/range-XY; Zemax full too|
|F7|Polarization / Stokes detector map|✅|🟡|✅|🟡xyz|**QUADOA/MWB**|
|F8|Energy-audit ledger + closure|✅`<1e-3`|🟡distrib|🟡scattered|⚠️manual|**MWB** — unique closed audit|
|F9|Ghost / stray-light analysis|❌|✅Path|✅ghost|🟡manual|**Zemax** — Path Analysis + Critical Ray Tracer|

### G. Materials & coatings
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|G1|Glass-catalog breadth|🟡~24|✅1000s|✅1000s|✅catalog|**Zemax** — most catalogs; QUADOA close; 3DOptix vendor parts|
|G2|Dispersion-model breadth|🟡4|✅13|✅4|🟡6|**Zemax** — 13 formulas|
|G3|dn/dT thermal index data|❌|✅|✅|❌|**Zemax/QUADOA**|
|G4|Coating model (TMM stack)|✅|✅|✅|⚠️presets|**Zemax** — largest; MWB/QUADOA full TMM|
|G5|Coating synthesis (needle)|❌|❌|🟡opt|❌|**QUADOA** — layer-thickness opt (no needle anywhere)|
|G6|Filters / transmission|✅|✅|⚠️|✅catalog|**Zemax** — internal-transmittance + interference filters|
|G7|Birefringent material library|✅3|✅|🟡unverif|❌|**Zemax/MWB**|
|G8|Component vendor catalog|❌|🟡|❌|✅~50k|**3DOptix** — ~50,000 real vendor parts|

### H. Scattering & stray light
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|H1|Surface scatter models|✅Beckmann|✅7|🟡3|✅4|**Zemax** — Lambertian/Gaussian/ABg/BSDF/DLL|
|H2|Measured BSDF import|❌|✅|❌|❌|**Zemax**|
|H3|Importance sampling|❌|✅|❌|❌|**Zemax**|
|H4|Volume / participating-media scatter|✅|🟡HG-DLL|❌|❌|**MWB** — continuum medium + HG beaten on rigor below|
|H5|Rigorous Mie particle scattering|✅Wiscombe|🟡DLL|❌|❌|**MWB** — first-class Mie validated vs Wiscombe MIEV0; Zemax offers Mie only via the MSP bulk-scatter DLL|
|H6|Dedicated stray-light workflow|❌|✅Path|🟡ghost|🟡manual|**Zemax** — Path Analysis / Critical Ray Tracer|

### I. Optimization
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|I1|Merit function / operands|❌|✅300+|✅named|⚠️scipy|**Zemax** — 300+ operands + wizard|
|I2|Local optimizer (DLS)|❌|✅|✅|⚠️|**Zemax/QUADOA**|
|I3|Global optimizer|❌|✅Hammer|🟡restart|❌|**Zemax** — Global Search + Hammer|
|I4|Glass substitution|❌|✅|✅|❌|**Zemax/QUADOA**|
|I5|Multi-config optimization|❌|✅|✅|❌|**Zemax/QUADOA**|

### J. Tolerancing
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|J1|Sensitivity analysis|❌|✅|✅|❌|**Zemax/QUADOA**|
|J2|Monte-Carlo tolerancing|❌|✅|✅9-dist|❌|**QUADOA** — 9 distributions; Zemax full too|
|J3|Compensators|❌|✅|✅chain|❌|**QUADOA/Zemax**|
|J4|Yield analysis|❌|✅|🟡|❌|**Zemax** — Yield/Quick/High-Yield|

### K. Thermal / STOP
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|K1|Thermal (bulk T, dn/dT, CTE)|❌|✅|✅|❌|**Zemax/QUADOA**|
|K2|STOP / FEA import|❌|❌*Ent*|⚠️GRIN-import|❌|**QUADOA** (thermal-lensing import); Zemax STAR is Enterprise-only, not Premium|

### L. Illumination / photometry
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|L1|Photometric units (lux/lumen)|❌|✅|❌|❌|**Zemax**|
|L2|Non-imaging / illumination design|⚠️|✅|⚠️|🟡|**Zemax** — NSC illumination + freeform opt|
|L3|IES/LDT export|❌|✅|❌|🟡|**Zemax**|

### M. Multi-configuration / zoom / scanning
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|M1|Multi-configuration editor|⚠️CLI|✅160|✅unlim|⚠️manual|**QUADOA/Zemax**|
|M2|Zoom systems|⚠️|✅|✅slider|⚠️|**QUADOA** — continuous slider; Zemax full|
|M3|Scanning systems|⚠️|✅|🟡|❌|**Zemax**|

### N. GUI / UX
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|N1|Interactive 3D view|✅|✅|✅|✅|*tie* — all live 3D|
|N2|Shaded / cross-section|🟡|✅|✅|✅|**Zemax/QUADOA/3DOptix**|
|N3|Undo/redo|✅20+cmd|🟡snapshot|✅|🟡|**MWB** — granular per-command + macros|
|N4|Wizards|🟡lens|✅|✅|🟡|**QUADOA/Zemax**|
|N5|Live update / auto-preview|✅1s|✅|✅1s|🟡run-on-demand|**MWB/QUADOA/Zemax**|
|N6|Ease of use / onboarding|❌steep|🟡|🟡|✅|**3DOptix** — zero-install, tutorials, Academy|
|N7|Cross-platform|🟡Linux|❌Win|✅Win/Linux|✅browser|**3DOptix/QUADOA**|
|N8|Ray animation / visualization|✅bead|🟡|🟡|🟡|**MWB** — physical c/n tracer beads|
|N9|Collaboration / sharing|❌|❌|❌|✅cloud|**3DOptix** — cloud share + Warehouse|

### O. Coordinate & positioning system
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|O1|Absolute spatial placement|✅|✅NSC|✅|✅|**tie** — all support absolute pose|
|O2|Sequential surface chaining|❌|✅|✅|❌|**Zemax/QUADOA** — thickness-to-next chain|
|O3|Relative / reference-object chaining|🟡readout|✅NSC ref|✅per-obj|✅LCS ref|**QUADOA** — per-object switchable reference modes|
|O4|Tilt/decenter + order control|✅Euler|✅CB-order|✅6-DoF+order|✅|**QUADOA** — explicit rot-order + pivot per object|
|O5|Coordinate break / pivot-about-point|✅|✅|✅pivot|🟡|**QUADOA/MWB/Zemax**|
|O6|Pickup / parametric constraints|✅expr|✅solves|✅lookup|❌|**Zemax** — rich solve set; MWB/QUADOA expression-bound|
|O7|Assemblies / grouping|🟡group|✅NSC ref|✅nested|✅cage|**QUADOA** — first-class nestable assemblies|
|O8|Snap-to-axis / auto-align|✅|⚠️|⚠️|🟡coming|**MWB** — shipped snap-to-optical-axis + drag-along-axis|
|O9|Optical-train / fold operations|⚠️manual|⚠️CB|⚠️seq|🟡mounts|**MWB** (closest, via snap) — but *no one* has a true 1-click fold operator|

### P. Data management
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|P1|Native project / archive format|✅ZIP|✅ZAR|✅optx|✅cloud|**Zemax/MWB** — bundle archives|
|P2|CAD import/export interop|❌FCStd|✅|✅|🟡STEP|**Zemax** — round-trip STEP/IGES/SAT + CSG|
|P3|In-app scripting API|❌|✅ZOS-API|✅SDK|🟡beta|**Zemax** — 4-mode C#/Py/MATLAB|
|P4|Macro language|❌|✅ZPL|🟡math|❌|**Zemax** — ZPL|
|P5|Headless / CLI batch|✅|✅API|❌|🟡SDK|**MWB/Zemax** — MWB CLI-first, Zemax via API|
|P6|Version-control-friendly formats|✅|✅ZMX|❌binary|❌cloud|**MWB/Zemax** — MWB stores text members in a ZIP container; Zemax ZMX is plain ASCII|

### Q. Performance & compute
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|Q1|CPU multithreading|🟡vectorized|✅|✅|n/a|**Zemax** — mature multi-core|
|Q2|GPU acceleration|✅gather|🟡Huygens|❌|✅cloud|**3DOptix/MWB** — 3DOptix cloud GPU, MWB local CUDA gather; Zemax GPU-accelerates Huygens PSF/MTF only, not tracing|
|Q3|Multi-process / distributed|❌|🟡instances|❌|✅cloud|**3DOptix** — cloud scale|
|Q4|Cloud compute|❌|🟡add-on|❌|✅|**3DOptix**|
|Q5|Ray / scale ceiling|🟡|✅|🟡|✅1e9|**3DOptix** — billion-ray tiers; Zemax high with caps|

### R. Commercial & practical
| # | Feature | MWB | Zemax | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|--|
|R1|Cost|✅free|❌$$$|🟡$$|✅freemium|**MWB** — free + no metering (3DOptix free tier is crippled)|
|R2|License / data locality|✅local|✅offline|✅offline|❌cloud|**MWB/Zemax/QUADOA** — offline; MWB full locality|
|R3|Platform / OS|🟡Linux|❌Win|✅Win/Linux|✅browser|**QUADOA/3DOptix**|
|R4|Ecosystem / community|❌|✅|🟡|🟡|**Zemax** — forum, KB, resellers|
|R5|Learning curve / onboarding|❌|🟡|🟡|✅|**3DOptix**|
|R6|Support / training|❌|✅|✅|🟡|**Zemax** — 24/5 + OpticsAcademy|
|R7|Data privacy / ITAR-suitable|✅|✅|✅|❌cloud|**MWB/Zemax/QUADOA** — 3DOptix cloud unsuitable for classified|

---

## 5. Overall winner (general feature set)

**Axis-by-axis tally** (who is strongest in each taxonomy block):

| Axis | Winner | Runner-up | MWB standing |
|--|--|--|--|
|A Ray-tracing core|Zemax|MWB / 3DOptix|Strong (wins coherent-absolute A4)|
|B Physical optics|Zemax|QUADOA|Weak on named products; unique on coherent-default|
|C Polarization|**Split: MWB/Zemax/QUADOA**|—|Co-leader (birefringence + grating efficiency)|
|D Surfaces & geometry|Zemax|QUADOA|Mid (no GRIN/CAD/Q-type/freeform-analytic)|
|E Sources|Zemax|QUADOA|Weak (no ray-aim/apod/Gaussian); wins coherent E3|
|F Detectors & analysis|Zemax|QUADOA|Wins energy-ledger F8 + Stokes F7|
|G Materials & coatings|Zemax|QUADOA / 3DOptix|Weak on breadth; strong on TMM/birefringence|
|H Scattering & stray light|**Split: Zemax (tools) / MWB (physics)**|—|Wins Mie H5 + volume H4|
|I Optimization|Zemax|QUADOA|None|
|J Tolerancing|QUADOA|Zemax|None|
|K Thermal/STOP|QUADOA/Zemax|—|None|
|L Illumination/photometry|Zemax|—|None|
|M Multi-config/zoom|QUADOA/Zemax|—|None (CLI sweep only)|
|N GUI/UX|3DOptix|QUADOA|Wins undo N3 + animation N8|
|O Coordinate system|QUADOA|Zemax|Wins snap-to-axis O8|
|P Data management|Zemax|MWB|Wins VC-friendliness P6 + CLI P5|
|Q Performance|3DOptix|MWB|Co-leader on GPU (Q2)|
|R Commercial|MWB|3DOptix / QUADOA|Wins cost R1 + locality R2/R7|

**Verdict.** Counting the 18 axes: **Zemax leads or co-leads ~11**, QUADOA ~5, 3DOptix ~2,
MieWorkbench ~2 (plus co-lead on C and H). **Ansys Zemax OpticStudio (Premium) is the overall
winner of the general feature set** by a decisive margin — it is the only package that is
excellent across design, physical optics, analysis, optimization, tolerancing, illumination,
materials, and interop simultaneously.

**The honest nuance the tally hides:** MieWorkbench is a *specialist engine*, not a suite. On
the specific axes it targets — **coherent non-sequential field fidelity, exact Mie/volume
scattering, closed energy accounting, polarization physics, GPU throughput, and cost/locality
— it wins or ties the market leader.** It "loses overall" the way a precision interferometry
bench "loses" to a full machine shop: fewer tools, but the ones it has are best-in-class.

---

## 6. Head-to-head tables (X vs MieWorkbench)

Winner per line item. "MWB" = MieWorkbench wins; "tie" = parity.

### 6.1 Zemax OpticStudio (Premium) vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|Zemax|deeper NSC; both full|
|A2 Sequential design|Zemax|MWB has none|
|A3 MC transport|tie|both true photon MC|
|A4 Coherent absolute-power field|**MWB**|Zemax NSC coherent is a geometric ray-sum (no Huygens spreading); MWB does a true RS gather inline|
|A5 Ray splitting|Zemax|richer budgets|
|A6 Spatial accel|tie|both use spatial acceleration (MWB's mesh BVH documented)|
|A7 Analytic/mesh|Zemax|+ CAD solids|
|B1–B9 Physical-optics products|Zemax (B1 tie)|MTF/Zernike/Strehl/POP/Gaussian all Zemax|
|C1 Jones|tie|
|C2 Stokes/DOP|**MWB**|Zemax ensemble-derived|
|C3 Mueller|Zemax|(workaround) vs MWB none → Zemax|
|C4 Birefringence+walk-off|tie|both validated uniaxial|
|C5/C6 Biaxial/activity|tie|both none|
|C7 TMM coatings|Zemax|larger catalog|
|C8 Polarizers|Zemax|
|C9 Grating efficiency|Zemax|RCWA rigorous > MWB closed-form|
|D1–D10 Surfaces/CAD|Zemax|GRIN, Q-type, freeform, CAD import|
|E1,E4–E7 Sources|Zemax|ray-aim/apod/Gaussian/IES|
|E3 Coherent source|**MWB**|inline vs POP-pass|
|E2,E8|tie|
|F1|tie|
|F2–F6,F9 Analysis/ghost|Zemax|
|F7 Stokes map|**MWB**|
|F8 Energy ledger|**MWB**|closed `<1e-3` audit|
|F3 Curved detector|Zemax|
|G1–G4,G6 Materials/coatings|Zemax|
|G7 Birefringent lib|tie|
|G8 Vendor catalog|Zemax(part)|neither strong; Zemax has some|
|H1–H3,H6 Scatter tools/BSDF/stray-light|Zemax|
|H4 Volume scatter|**MWB**|
|H5 Rigorous Mie|**MWB**|MWB first-class Wiscombe-validated Mie; Zemax Mie only via MSP DLL|
|I1–I5 Optimization|Zemax|MWB none|
|J1–J4 Tolerancing|Zemax|MWB none|
|K1 Thermal|Zemax|
|K2 STOP/FEA|tie|Zemax STAR is Enterprise, not Premium → both none at this tier|
|L1–L3 Photometry|Zemax|
|M1–M3 Multi-config|Zemax|
|N1,N5|tie|
|N3 Undo|**MWB**|granular per-command|
|N8 Animation|**MWB**|
|N2,N4,N6|Zemax|
|N7 Platform|tie|MWB Linux, Zemax Windows — an OS swap, neither is cross-platform|
|N9 Collaboration|tie|both none|
|O1|tie|
|O2 Sequential chaining|Zemax|
|O3,O6,O7 Ref/pickup/assembly|Zemax|
|O4,O5|tie|
|O8 Snap-to-axis|**MWB**|shipped auto-align|
|O9 Fold operator|**MWB**|closest (neither has 1-click)|
|P1|tie|
|P2–P4 CAD/API/macro|Zemax|
|P5 Headless CLI|tie|
|P6 VC-friendly|tie|MWB text members in ZIP; Zemax ZMX plain ASCII|
|Q1 CPU threads|Zemax|
|Q2 GPU|tie|MWB gather on CUDA; Zemax GPU-accelerates Huygens PSF/MTF (not tracing)|
|Q3–Q5 Distributed/scale|Zemax|
|R1 Cost|**MWB**|
|R2 Locality|tie|
|R3 Platform|tie|MWB Linux-only, Zemax Windows-only|
|R4,R6 Ecosystem/support|Zemax|
|R5 Learning curve|Zemax|
|R7 ITAR/privacy|tie|

**Zemax vs MieWorkbench overall winner: Zemax OpticStudio Premium.** MieWorkbench wins ~10
line items (coherent-absolute field, Stokes maps, the energy ledger, volume + first-class Mie
scatter, granular undo, ray animation, snap-to-optical-axis, the fold operator, cost, and
runs-on-Linux) — a meaningful, physics-and-cost-shaped cluster — but Zemax wins the large
majority on breadth, and several earlier-drafted MWB "wins" (BVH, GPU, VC-friendliness,
cross-platform) are more fairly scored as ties.

### 6.2 QUADOA Optical CAD vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|**MWB**|QUADOA has none|
|A2 Sequential|QUADOA|
|A3 MC transport|**MWB**|QUADOA MC is tolerancing-only|
|A4 Coherent absolute field|**MWB**|QUADOA wave optics is paid add-on|
|A5 Ray splitting|**MWB**|QUADOA ghosts only|
|A6 BVH|**MWB**|
|A7 Analytic/mesh|QUADOA|broader analytic + CAD import|
|B1 Interference|tie|
|B2 Diffraction|tie|
|B3–B9 POP/PSF/MTF/Zernike/Strehl/Gaussian|QUADOA|named suite MWB lacks|
|C1 Jones|tie|
|C2 Stokes/DOP|tie|both first-class (QUADOA adds Poincaré)|
|C3 Mueller|QUADOA|
|C4 Birefringence+walk-off|**MWB**|QUADOA unverified/no walk-off|
|C7 TMM coatings|tie|
|C8 Polarizers|QUADOA|
|C9 Grating efficiency|**MWB**|QUADOA scalar/efficiency-free|
|D2 Asphere|tie|
|D3,D4,D8,D9,D10 Q-type/freeform/GRIN/CAD|QUADOA|
|D5 DOE|tie|
|D6,D7 Fresnel/toroidal|tie|
|E1,E4,E5,E6,E7 Sources|QUADOA|ray-aim/apod/Gaussian/rayfile|
|E3 Coherent source|**MWB**|
|E2,E8|tie|
|F1|tie|
|F2 Spectral detector|**MWB**|QUADOA has none|
|F3 Curved detector|tie|both partial/none|
|F4–F6,F9 Spot/fan/EE/ghost|QUADOA|
|F7 Stokes map|tie|
|F8 Energy ledger|**MWB**|QUADOA scattered goals|
|G1,G2,G3 Materials/dispersion/thermal|QUADOA|
|G4 TMM coatings|tie|
|G7 Birefringent lib|**MWB**|
|G8 Vendor catalog|tie|neither|
|H1 Surface scatter|QUADOA|adds ABg|
|H4 Volume scatter|**MWB**|QUADOA none|
|H5 Rigorous Mie|**MWB**|QUADOA none|
|H6 Stray-light|QUADOA|ghost workflow|
|I1–I5 Optimization|QUADOA|MWB none|
|J1–J4 Tolerancing|QUADOA|MWB none|
|K1,K2 Thermal/STOP|QUADOA|MWB none|
|L1–L3 Photometry|tie|both essentially none|
|M1–M3 Multi-config|QUADOA|MWB none|
|N1,N5|tie|
|N2,N4 Shaded/wizards|QUADOA|
|N3 Undo|tie|both full undo; MWB more granular (per-command + macros)|
|N7 Cross-platform|QUADOA|Win+Linux vs Linux-only|
|N8 Animation|**MWB**|
|N9|tie|
|O2,O3,O4,O6,O7 Seq/ref/order/pickup/assembly|QUADOA|superset positioning model|
|O1,O5|tie|
|O8 Snap-to-axis|**MWB**|direct-manipulation vs lookup params|
|O9 Fold|**MWB**|
|P2,P3,P4 CAD/API/macro|QUADOA|SDK|
|P5 Headless CLI|**MWB**|QUADOA undocumented|
|P6 VC-friendly|**MWB**|QUADOA binary format|
|Q1 CPU|QUADOA|
|Q2 GPU|**MWB**|QUADOA CPU-only per manual|
|Q3–Q5|tie/QUADOA|neither distributed|
|R1 Cost|**MWB**|
|R2 Locality|tie|
|R3 Platform|QUADOA|
|R4,R6 Ecosystem/support|QUADOA|
|R5 Learning curve|QUADOA|(marketed; unverified)|
|R7 Privacy|tie|

**QUADOA vs MieWorkbench overall winner: QUADOA** — it wins the design/optimization/
tolerancing/analysis/surfaces/interop bulk. **But the split is unusually clean and
complementary:** MieWorkbench wins *every* non-sequential/physical-transport line
(non-sequential, MC transport, coherent-absolute, ray-splitting, BVH, volume + rigorous Mie
scatter, grating efficiency, energy ledger, birefringence walk-off, GPU, headless/VC/cost).
These are precisely QUADOA's stated non-goals. The two tools would be excellent *together*.

### 6.3 3DOptix vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|tie|both full NSC|
|A2 Sequential|3DOptix|claimed (MWB none)|
|A3 MC transport|**MWB**|3DOptix MC not first-class|
|A4 Coherent absolute field|**MWB**|3DOptix approx + gated|
|A5 Ray splitting|tie|
|A6 BVH/GPU accel|3DOptix|cloud GPU|
|A7 Analytic/mesh|3DOptix|+ STEP import|
|B1 Interference|**MWB**|3DOptix default is ray-approx|
|B2 Diffraction|**MWB**|MWB default; 3DOptix Huygens gated/narrow|
|B3 POP|3DOptix|Fresnel convolution (gated) vs MWB none-named|
|B4 PSF|3DOptix|named product|
|B5 MTF|3DOptix|(geometric; diffraction "coming soon")|
|B6 Wavefront/Zernike|tie|both effectively none|
|B7 Strehl|tie|both none|
|C1 Jones|**MWB**|3DOptix linear-tag only|
|C2 Stokes/DOP|**MWB**|
|C3 Mueller|tie|both none|
|C4 Birefringence|**MWB**|3DOptix none|
|C7 TMM coatings|**MWB**|3DOptix 6 presets|
|C8 Polarizers|**MWB**|
|C9 Grating efficiency|**MWB**|3DOptix single-order geometric|
|D1,D2 Sph/asphere|tie|
|D3 Q-type|tie|both none|
|D4 Freeform|tie|both weak|
|D5 DOE|tie|
|D6 Fresnel|**MWB**|3DOptix none found|
|D7 Toroidal|tie|
|D8 GRIN|tie|both none|
|D9,D10 Mesh/CAD import|3DOptix|STEP import|
|E1 Source types|3DOptix|
|E3 Coherent source|**MWB**|
|E4 Gaussian source|3DOptix|(no M²)|
|E5 Apodization|tie|both none|
|E6 Ray-aiming|tie|both none|
|E7 Measured source files|3DOptix|IES/TM-25|
|E2,E8 Spectral/pol source|**MWB**|3DOptix single-λ free tier, weak pol|
|F1 Irradiance|tie|
|F2 Spectral detector|tie|
|F3 Curved detector|tie|both none|
|F4 Spot diagram|3DOptix|
|F5 Ray fans|tie|both none|
|F6 Encircled energy|3DOptix|(circular)|
|F7 Stokes map|**MWB**|3DOptix xyz-split only|
|F8 Energy ledger|**MWB**|3DOptix manual diffing|
|F9 Ghost/stray-light|3DOptix|purpose-built manual workflow|
|G1,G8 Catalog/vendor parts|3DOptix|~50k parts decisive|
|G2 Dispersion|tie|
|G4 Coating TMM|**MWB**|3DOptix presets|
|G6 Filters|3DOptix|catalog|
|G7 Birefringent lib|**MWB**|
|H1 Surface scatter|tie|MWB Beckmann vs 3DOptix 4 BRDF|
|H4 Volume scatter|**MWB**|
|H5 Rigorous Mie|**MWB**|
|H6 Stray-light workflow|3DOptix|
|I,J,K optimization/tol/thermal|tie|both none (3DOptix SciPy-only)|
|L1 Photometry|tie|both radiometric-only|
|M multi-config|tie|both manual/CLI|
|N1 3D view|tie|
|N2 Shaded|3DOptix|
|N3 Undo|**MWB**|granular stack|
|N5 Live update|**MWB**|auto-retrace vs run-on-demand|
|N6 Ease of use|3DOptix|decisive|
|N7 Cross-platform|3DOptix|browser|
|N8 Animation|**MWB**|
|N9 Collaboration|3DOptix|cloud share|
|O1 Absolute placement|tie|
|O3 Reference chaining|3DOptix|LCS reference picker|
|O7 Assemblies/mounts|3DOptix|real cage/breadboard hardware|
|O8 Snap-to-axis|**MWB**|shipped vs 3DOptix "coming soon"|
|O9 Fold|**MWB**|
|P1 Format|tie|
|P2 CAD import|3DOptix|
|P3 Scripting API|3DOptix|(beta, paywalled) vs MWB CLI|
|P5 Headless|**MWB**|offline CLI|
|P6 VC-friendly|**MWB**|
|Q2 GPU|3DOptix|elastic cloud|
|Q3,Q4,Q5 Distributed/cloud/scale|3DOptix|
|R1 Cost|**MWB**|free tier full-feature vs 3DOptix crippled free tier|
|R2 Locality|**MWB**|offline vs cloud-only|
|R3 Platform|3DOptix|browser|
|R4 Ecosystem|3DOptix|
|R5 Onboarding|3DOptix|
|R7 Privacy/ITAR|**MWB**|cloud unsuitable for classified|

**3DOptix vs MieWorkbench overall winner: 3DOptix, narrowly, on breadth — but the axis
matters.** By raw line-item count 3DOptix edges ahead (catalog, accessibility, CAD import,
sources, cloud GPU, PSF/MTF/spot products, stray-light UI, collaboration). **MieWorkbench wins
the entire physics-depth cluster** (real wave optics by default, full Jones/Stokes
polarization, TMM coatings, birefringence, grating efficiency, rigorous Mie + volume
scattering, the energy ledger) **plus locality/cost/privacy/VC/animation/undo/live-update/
snap-to-axis.** For a **physics-first optical engineer** MieWorkbench is the more capable
*engine*; for a **lab engineer laying out catalog benchtop systems** 3DOptix is the better
*product*. Call it 3DOptix by breadth, MieWorkbench by depth.

### 6.4 Overall winner of the "X vs MieWorkbench" comparisons
Across the three pairings, **the commercial/established tool wins the overall verdict in every
case** (Zemax decisively, QUADOA on the design axis, 3DOptix narrowly on breadth). **The single
overall winner of the head-to-head set is Zemax OpticStudio Premium** — it beats MieWorkbench
on the most line items and is itself the general-feature-set winner. MieWorkbench's consistent,
*non-overlapping* wins across all three pairings — coherent-absolute non-sequential field,
rigorous Mie + volume scattering, the closed energy ledger, GPU throughput, and cost/locality/
privacy — define exactly the moat to defend and the gaps to close in §7.

---

## 7. Gap-closing roadmap — every line MieWorkbench does not win

For each gap: **① Ideal (best-in-class target)** and **② Pragmatic path** given this
architecture (FreeCAD worker + numpy/torch MC engine + PySide GUI), cross-referenced to
`future.md` seams and named modules. Effort tiers: **S** ≤1 wk · **M** ~1 mo · **L** ~1 quarter
· **XL** multi-quarter/research.

### 7.1 Named physical-optics analysis products (B4–B9, F4–F6) — *highest leverage*
MieWorkbench already computes the complex field; it just does not package the standard products.
- **① Ideal:** FFT-PSF + Huygens-PSF, FFT/geometric MTF (+ through-focus, vs-field), Strehl,
  encircled/ensquared energy, spot diagrams, transverse + OPD ray fans, and Zernike (Standard/
  Fringe/Annular) wavefront decomposition — as first-class result products with GUI panels.
- **② Pragmatic:** this splits into two tiers of difficulty — do not oversell it as pure
  post-processing.
  - **(a) Near-post-processing (cheapest):** PSF, MTF, and encircled/ensquared energy derive
    from the coherent detector field. Caveat: `--save-fields` currently writes complex `Ex/Ey`
    for **seed 0 only** (cross-seed field averaging is phase-meaningless) and is **off by
    default**, so a first cut is single-seed fidelity; MTF = FFT of the PSF; encircled energy =
    radial cumulative sum. This tier is a `post_process.py` module + an "Analysis" results tab.
  - **(b) Needs new engine plumbing:** **spot diagrams and ray fans** require persisting the
    **full per-ray hit/OPL population** — today only a ≤20k-ray *visualization* subset is written
    (`rays.npy`), and the gather samples are never exported. **Zernike and Strehl** additionally
    require an **exit-pupil / OPD-reference-sphere concept the engine does not currently have**
    (grep finds no `pupil`/`OPD`/`zernike`/`strehl`; "wavefront" in the code is only Igehy ray
    differentials). So these need a ray-export stage and a pupil/OPD-sampling stage first.
  - **Effort: PSF/MTF/EE M (seed-0 caveat); spot/ray-fan M (needs ray export); Zernike/Strehl L
    (needs pupil/OPD stage).** Even tier (a) closes the most conspicuous gap vs all three
    competitors.

### 7.2 Optimization (I1–I5) — *the biggest categorical gap*
- **① Ideal:** merit-function editor with named operands (EFL, RMS spot/wavefront, MTF@freq,
  boundary/edge constraints), local (DLS/derivative-free) + global (basin-hopping/CMA-ES)
  optimizers, glass substitution, multi-config awareness.
- **② Pragmatic:** MieWorkbench's variables are FreeCAD spreadsheet parameters + placements
  (already swept by `permute_model.py`/`--var`). Wrap that as an **optimization loop**:
  scipy.optimize (least_squares/differential_evolution) or `nevergrad`/CMA over a merit function
  built from the §7.1 analysis products (spot RMS, encircled energy, detected power). Start with
  a headless `scripts/optimize.py` driving the existing pipeline; add a GUI merit-function panel
  later. The real per-iteration cost is not only the trace but a full FreeCAD **rebuild →
  extract → trace** per variant (via `permute_model.py`), plus the expensive coherent gather —
  mitigate with geometry caching for unchanged bodies and a
  **geometric-only fast mode** (`coherent=false`, direct deposit) for the optimization inner
  loop, refining coherently at the end. **Effort: headless optimizer L; GUI M; global XL.**

### 7.3 Tolerancing (J1–J4)
- **① Ideal:** sensitivity + inverse-sensitivity, Monte-Carlo with multiple perturbation
  distributions, chainable compensators, yield reporting.
- **② Pragmatic:** MieWorkbench already has `--seeds` and `permute_model.py`; tolerancing is a
  structured sweep over perturbed placements/radii/index with statistics. Build
  `scripts/tolerance.py` that perturbs the FreeCAD model per a tolerance table, runs the
  (geometric-fast) pipeline N times, and aggregates a merit-metric distribution + sensitivity
  ranking. Compensators = a nested §7.2 optimize call per draw. **Effort: sensitivity M; MC
  tolerancing L; compensators L.**

### 7.4 Surfaces & geometry (D3, D4, D8, D9, D10)
- **① Ideal:** analytic Q-type (Forbes) + XY/Zernike/Chebyshev freeform *with coherent phase*;
  GRIN; native STEP/IGES import as traceable optical surfaces.
- **② Pragmatic:**
  - *Q-type/freeform analytic:* extend `surfaces.py`'s asphere machinery (already Newton-
    intersecting even polynomials, extract-verified `<1 µm`) to Forbes Qbfs/Qcon and XY/Zernike
    sag with analytic normals — reuses the existing bracket-guarded intersection. **M–L.**
  - *GRIN:* genuinely new — curved-ray integration inside the bulk (Runge–Kutta on the eikonal)
    replacing straight segments in `tracer.py`. Flagged in `future.md`. **XL.**
  - *CAD import:* FreeCAD *already* imports STEP/IGES; the missing piece is the GUI exposing
    "Import STEP as element" through the fc_server worker (`import_bodies`/`import_primitive`
    ops) and the extractor canonicalizing imported faces (falling back to the existing mesh-BVH
    path for non-canonical ones — already shipped, incoherent-only). **M for mesh-import; L for
    analytic-face recovery.**

### 7.5 Sources (E1, E4–E7)
- **① Ideal:** ~20 emitter types, Gaussian-beam source, apodization, ray-aiming to a stop,
  measured IES/TM-25/rayfile ingestion.
- **② Pragmatic:** `sources.py` already samples faces; add (a) **Gaussian/super-Gaussian
  apodization** as a per-sample amplitude weight (S–M), (b) a **Gaussian-beam source** (waist +
  M²) as a special divergent source with the right amplitude/phase profile — pairs naturally
  with the coherent gather (M), (c) **ray-aiming** by iterating emission direction to hit a named
  aperture body (M), (d) an **IES/TM-25/rayfile importer** producing weighted ray sets (M).
  **Effort: apodization S; Gaussian-beam M; ray-aiming M; source files M.**

### 7.6 Materials & coatings (G1–G3, G5, G6, G8)
- **① Ideal:** thousands of catalog glasses (Schott/Ohara/CDGM/…), 13 dispersion formulas,
  dn/dT, coating synthesis, a real vendor-component catalog.
- **② Pragmatic:** the loader architecture (`optprops.py`) already supports Sellmeier/Cauchy/
  constant/tabulated with mandatory citations. (a) **Import the public Schott/Ohara AGF glass
  catalogs** into `.miemat` rows (mostly a data-conversion script — the Sellmeier form is
  already supported) — instantly closes most of G1/G2 (**S–M**). (b) **Add dn/dT** columns +
  a temperature parameter applied in `materials.py` (**M**). (c) Coating *synthesis* (needle) is
  niche — defer. (d) A **vendor-component catalog** (like 3DOptix's) is a large data/business
  effort; a pragmatic subset: ship a handful of Thorlabs/Edmund parametric primitives with real
  prescriptions (**L**, ongoing). **Effort: glass catalog import S–M; dn/dT M; vendor catalog L+.**

### 7.7 Scattering & stray light (H1–H3, H6)
- **① Ideal:** measured BSDF/ABg import, importance sampling, a dedicated stray-light/ghost
  workflow with path ranking.
- **② Pragmatic:** (a) **ABg + tabulated BSDF import** as a new scatter-lobe sampler beside
  `roughness.beckmann_sample` (flagged in `future.md`) (**M**). (b) **Importance sampling**
  (scatter-to-target) in the scatter sampler (**M**). (c) A **ghost/stray-light analysis mode** —
  `future.md` notes the per-ray generation/medium-stack history already exists on `RayBatch`;
  a post-processor can rank reflection paths by detected power (**M**). MieWorkbench already
  *wins* the physics (rigorous Mie + volume); these close the tooling gap. **Effort: M each.**

### 7.8 Thermal / STOP (K1, K2)
- **① Ideal:** dn/dT + CTE thermal model; FEA (structural + thermal) deformation import onto
  surfaces.
- **② Pragmatic:** dn/dT is shared with §7.6. Full STOP is XL research (couples to an FEA tool);
  a pragmatic first step is **importing a deformed surface as a Grid-Sag/mesh** (reuses the mesh
  path) and a **thermal-lensing GRIN import** (shares §7.4 GRIN). **Effort: dn/dT M (via §7.6);
  deformation import L; coupled STOP XL.**

### 7.9 Illumination / photometry (L1–L3)
- **① Ideal:** photometric units (lux/lumen/candela via V(λ)), non-imaging design, IES/LDT export.
- **② Pragmatic:** **photometric units are a post-processing multiply** — apply the CIE V(λ)
  luminosity function to the existing spectral irradiance cube to emit lux/lumen/candela
  alongside W/m² (**S**). IES/LDT export from the far-field detector is a formatter (**M**).
  Non-imaging *design* (freeform tailoring) rides on §7.2 optimization (**L**). **Effort:
  photometric units S; IES/LDT export M.**

### 7.10 Multi-configuration / zoom (M1–M3)
- **① Ideal:** an in-GUI multi-config editor (per-config parameters) with zoom/scan support.
- **② Pragmatic:** the `--var` sweep + config-matrix already parameterize variants; wrap them as
  a **named-configuration table** in the GUI (a config = a set of spreadsheet/placement overrides)
  and let the run loop iterate configs, overlaying results via the existing `compare_runs.py`.
  **Effort: config-table GUI M; zoom/scan as config sequences M.**

### 7.11 CAD interop, scripting API, macros (P2–P4)
- **① Ideal:** round-trip STEP/IGES/SAT; an in-process scripting API; a macro language.
- **② Pragmatic:** (a) STEP/IGES import via §7.4; **export** of the FreeCAD model + traced rays
  to STEP/IGES is a fc_server op (**M**). (b) **In-app Python console** bound to the `Project`
  session object — MieWorkbench is already Python; expose the `core/project.py` API in a console
  pane (**M**). (c) A macro *language* is unnecessary given a Python console — skip. **Effort:
  CAD export M; Python console M.**

### 7.12 Compute & scale (A5 depth, A7, Q1, Q3–Q5)
- **① Ideal:** multi-core + multi-GPU + optional distributed/cloud scale-out.
- **② Pragmatic:** `future.md` already specifies the path — **shard primary rays via
  `SeedSequence.spawn` and merge ledgers/detector cubes** (all accumulators add linearly) for
  `--workers` multi-process (**M**) and multi-GPU (**L**). This directly attacks the
  gather-dominated wall-clock that is MieWorkbench's practical ceiling. Cloud scale-out is
  optional and against the data-locality value proposition — treat as opt-in (**L+**). **Effort:
  multi-process M; multi-GPU L.**

### 7.13 UX, coordinate system, ecosystem (D-region N/O/R lines)
- **Sequential-chain / relative optical-train positioning (O2, O3, O6, O7):** see the dedicated
  proposal in `UI_COORDINATE_PROPOSAL.md`. Pragmatic core: a **relative "downstream chain"**
  where an element's placement can be expressed as *distance-along-beam from the previous
  element* (reusing `transforms.py`'s `snap_to_axis_ops` + expression-bound placements), plus a
  **first-class assembly/group** object. **Effort: relative chaining M; assemblies M.**
- **Fold operator (O9) — explicitly requested:** a **one-click "insert fold mirror" + "fold the
  downstream train about this surface"** that auto-reflects the placements of all following
  elements. Builds directly on existing `align_rotation`/`snap_to_axis_ops`/`rotate_matrix`.
  **Effort: M.** (We found no true 1-click optical-train fold in the Zemax/QUADOA/3DOptix
  documentation — a genuine and defensible market gap; §UI proposal expands it.)
- **Ease of use / onboarding / cross-platform (N6, R5, R3):** ship a Windows/Mac build path
  (PySide6 + VTK are cross-platform; the blocker is the FreeCAD/optics-env/ParaView tooling —
  package as a bundled installer or container), plus a guided "new user" tutorial project and
  demo gallery (the `demos/` already exist). **Effort: cross-platform packaging L; onboarding M.**
- **Named products already close many N/F lines once §7.1 lands.**

### 7.14 Deliberate non-goals (document, don't chase)
- **Biaxial birefringence / optical activity (C5, C6):** *no competitor here has them either* —
  low competitive urgency, though `future.md` scopes both. Pursue only for specific research
  needs.
- **RCWA (C9 rigor):** Zemax-only; MieWorkbench's closed-form grating models cover most practical
  cases. XL research; defer unless sub-wavelength gratings become a target use case.
- **Coating needle-synthesis (G5):** nobody here has true needle synthesis (Zemax points users
  to Essential Macleod); not worth chasing.
- **Cloud compute (Q4):** conflicts with the data-locality value proposition; keep optional.

---

## 8. Priorities — MoSCoW (from the standpoint of the expert optical engineer)

Ranked by *impact on making MieWorkbench "best" per line item* × *leverage over existing code*.

### 8.1 Must Have (closes the largest, most-visible gaps cheaply; defines credibility)
1. **Named physical-optics analysis products** (§7.1: PSF, MTF, Strehl, encircled energy, spot
   diagrams, ray fans, Zernike). *Why:* the single most conspicuous gap vs **all three**
   competitors; its cheapest tier (PSF/MTF/encircled energy) is **post-processing over the
   coherent field the engine already computes** (spot/ray-fan/Zernike/Strehl need a modest
   ray-export + pupil/OPD stage first — see §7.1) — still the highest leverage on the board.
   Without MTF/PSF/Strehl the tool "looks like" it can't do imaging even though the physics is
   there.
2. **Glass-catalog import + dn/dT** (§7.6a–b). *Why:* ~24 materials is an immediate credibility
   gap; the loader already supports the Sellmeier form, so this is mostly a data-import script.
3. **Multi-process ray sharding** (`--workers`, §7.12). *Why:* the gather-dominated wall-clock
   (hours at high resolution) is the practical ceiling on real use; the seams are already
   specified in `future.md`.
4. **Photometric units** (§7.9, V(λ) post-multiply). *Why:* trivial (S) and removes a whole
   "❌" column cell; radiometric-only is a surprising gap for a tool with full spectra.
5. **Fold operator + relative optical-train chaining** (§7.13, O9/O2). *Why:* explicitly
   requested; a genuine differentiator (no competitor has 1-click fold); builds on shipped
   `align_rotation`/`snap_to_axis_ops`.

### 8.2 Should Have (high value, moderate effort; matches competitors on core design workflow)
6. **Headless optimization loop** (§7.2) with a geometric-fast inner loop. *Why:* the biggest
   *categorical* gap; even a scipy-based merit-function optimizer transforms the tool from
   "simulator" to "design-capable."
7. **Sensitivity + Monte-Carlo tolerancing** (§7.3). *Why:* pairs with optimization; reuses
   `--seeds`/`permute_model.py`; expected of any serious optical tool.
8. **Gaussian-beam source + apodization + ray-aiming** (§7.5). *Why:* laser/coherent workflows
   need Gaussian sources; apodization is nearly free; ray-aiming enables proper stop-defined
   pupils.
9. **CAD (STEP/IGES) import as elements** (§7.4 CAD). *Why:* FreeCAD already imports these;
   exposing it removes a hard "❌" and enables real mechanical/optomechanical scenes.
10. **In-app Python console** bound to the `Project` API (§7.11b). *Why:* closes the scripting-
    API gap cheaply since the app is already Python; unlocks power-user automation.
11. **BSDF/ABg import + ghost/stray-light analysis mode** (§7.7). *Why:* MieWorkbench already
    *wins* the scatter physics; this closes the tooling gap and leverages existing `RayBatch`
    history.
12. **Config-table multi-configuration** (§7.10). *Why:* zoom/thermal/scan workflows; wraps the
    existing sweep machinery.

### 8.3 Might Be Useful (worthwhile but narrower or higher-effort)
13. **Analytic Q-type / XY-Zernike freeform surfaces** with coherent phase (§7.4). *Why:*
    extends the existing asphere machinery; matters for freeform/AR-VR work but narrower audience.
14. **Multi-GPU gather** (§7.12). *Why:* pushes the scale ceiling further after multi-process.
15. **IES/LDT export + basic non-imaging** (§7.9). *Why:* only if illumination becomes a target
    domain.
16. **Vendor-component catalog subset** (§7.6d). *Why:* huge usability win (see 3DOptix) but a
    large, ongoing data/business effort; start with a curated Thorlabs/Edmund primitive set.
17. **Cross-platform packaging (Windows/Mac)** (§7.13). *Why:* broadens reach; blocker is the
    external tool stack, not the GUI.
18. **Surface-deformation (STOP-lite) import** (§7.8). *Why:* niche until thermal/structural
    coupling is a real requirement.

### 8.4 Not Really Important (defer or document as deliberate non-goals)
19. **RCWA gratings** (§7.14) — Zemax-only; closed-form models suffice for most cases; XL research.
20. **Biaxial birefringence / optical activity** (§7.14) — no competitor here has them; pursue
    only for a specific research need.
21. **Coating needle-synthesis** (§7.14) — nobody here has it; users pair with Essential Macleod.
22. **Native cloud compute** (§7.14) — conflicts with the data-locality/ITAR value proposition;
    keep strictly optional.
23. **A macro *language*** — redundant given an in-app Python console (§7.11).

---

## 9. Bottom line

MieWorkbench cannot and should not try to out-*breadth* Zemax OpticStudio. Its defensible,
already-winning moat is **physical-optics fidelity in a non-sequential engine**: coherent-by-
default field propagation with absolute power, rigorous Mie + volume scattering, full Jones/
Stokes polarization with validated birefringence, a closed and auditable energy ledger, GPU
throughput, and zero-cost full-locality operation. The **Must-Have** items (§8.1) are almost
all *packaging of physics the engine already computes* — PSF/MTF/Strehl/Zernike, photometric
units, glass catalogs — plus the multi-process speedup and the requested fold operator. Landing
those alone would flip roughly a dozen "❌/🟡" cells to "✅" and make MieWorkbench genuinely
best-in-class on every line item it targets, while the **Should-Have** optimization/tolerancing/
source work would, for the first time, let it credibly stand next to QUADOA and Zemax as a
*design* tool and not only a *simulation* engine.
