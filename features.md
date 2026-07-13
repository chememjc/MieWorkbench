# MieWorkbench — Competitive Feature Analysis

**Scope:** MieWorkbench (this repository) vs **Ansys Zemax OpticStudio (Premium)**,
**Keysight CODE V** (formerly Synopsys — see ownership note), **OSLO Premium**
(Lambda Research), **QUADOA Optical CAD**, and **3DOptix**.
**Method:** the MieWorkbench column is drawn from `docs/RAYTRACER.md`, `README.md`,
`future.md`, and the `mieworkbench/` + `scripts/raytracer/` source (refreshed 2026-07-12,
after the lowhanging / design-usability / c-engine / pulsed-optics rounds). Competitor
columns were built from live vendor documentation, manuals, SDK sources, and pricing
pages (July 2026); claims that could not be pinned to a primary source are marked
*unverified*. The OSLO column rests primarily on the Lambda Research *OSLO Optics
Reference* (©2011, read directly) cross-checked against current vendor pages; the
CODE V column rests on the ©2018 CODE V brochure + Keysight/Synopsys pages (bot-walled,
so several claims are search-indexed excerpts) — both flag dated/secondary claims.
**Ratings:** ✅ full · 🟡 partial / shallower · ⚠️ workaround only · ❌ none.

> **Ownership note (2025–2026 shake-up).** Two of the packages here changed hands in the
> Synopsys–Ansys merger and its antitrust remedy. **Zemax** (Ansys-owned since 2023) is now
> under **Synopsys** (which closed its Ansys acquisition in 2025). To satisfy a **US-FTC**
> divestiture condition (a 3-0 FTC consent order), Synopsys sold its *own* Optical Solutions Group — **CODE V**,
> LightTools, LucidShape, RSoft — to **Keysight Technologies** (closed 17 Oct 2025). So the
> two market-leading design suites, CODE V and Zemax, now sit on *opposite* sides of that
> deal (Keysight and Synopsys respectively). This document keeps the familiar "Zemax
> OpticStudio" and "CODE V" names; the corporate parents are noted where they matter (support,
> licensing, roadmap). *(Ownership dates: primary PRs; treat as of July 2026.)*

> **Honesty note.** MieWorkbench is a physically-based **coherent Monte-Carlo,
> fully-vectorial non-sequential ray tracer**. It is a *simulation engine*, not a
> lens-*design* suite. Four of the six packages here (Zemax, CODE V, OSLO, QUADOA) are
> design/analysis products with optimization, tolerancing, and a named imaging-analysis
> suite; the two new entrants (CODE V, OSLO) are *pure sequential-design* tools — the
> opposite pole from MieWorkbench. This document is deliberately unflattering about
> MieWorkbench's design/optimization gaps so the roadmap that follows is grounded in reality
> rather than advocacy — while also crediting the physics line-items MieWorkbench genuinely
> wins.

---

## 1. Executive summary

**Overall winner, general (single-product) feature set: Ansys Zemax OpticStudio (Premium),
narrowly over Keysight CODE V.** Zemax is the only package here that is excellent across
*both* paradigms in one product — the full sequential Lens-Data-Editor design engine **and** a
first-class non-sequential (illumination/stray-light/splitting) engine — plus the entire named
analysis suite (MTF/PSF/Zernike/Strehl/encircled energy), optimization, tolerancing, thermal,
photometry, huge glass catalogs, RCWA gratings, measured BSDF, CAD import, and a deep scripting
API. **CODE V matches or beats Zemax on the pure lens-design axes** — its **Global Synthesis**
global optimizer and its **Wavefront Differential** tolerancing (vendor-described as dramatically
faster than Monte Carlo — exact multiplier *unverified* — and runnable *inside* the optimization
loop for desensitization) are best-in-class — but CODE V
offloads non-sequential tracing, illumination, photometry, and volume/BSDF scatter to its sibling
product **LightTools**, so as a *single* product its breadth is narrower than Zemax's. **OSLO
Premium and QUADOA are strong second-tier design suites**; **3DOptix** owns accessibility and a
huge catalog. Nothing here approaches Zemax's raw single-product breadth.

**But breadth is not the whole story, and MieWorkbench is not trying to be any of them.**
MieWorkbench wins a specific, defensible cluster of line items that matter enormously for
*physical-optics fidelity* work: a **coherent Huygens/Rayleigh–Sommerfeld field gather that runs
by default on every non-sequential scene and books absolute interferometric power** (Zemax's
non-sequential "coherent" mode is a geometric coherent ray-sum; CODE V's BSP is an
imaging-framed beamlet propagator; OSLO's interferogram is a single-beam wavefront plot;
3DOptix's default is a gated ray-approximation; QUADOA's is a paid wave-optics add-on); **exact
Mie particle-cloud scattering** validated against Wiscombe MIEV0 (Zemax offers Mie only through a
bundled bulk-scatter DLL; CODE V/OSLO delegate all volume scatter to LightTools/TracePro; QUADOA
and 3DOptix omit it); a **closed 9-bucket energy-audit ledger** with `<1e-3` closure that *none*
of the five commercial tools match; **native Stokes/DOP maps**, **the only biaxial-crystal solver
in the field**, and validated uniaxial birefringence with walk-off; **GPU/CUDA acceleration**
(with a compiled OpenMP+CUDA C engine, ~8.3× wall-clock over the numpy reference); a **whole
ultrafast/time-domain/nonlinear axis** (pulsed sources, spectrogram/streak/cube, SHG/Kerr/SPM,
GDD budget) that every CW frequency-domain design suite here simply lacks; and **zero cost, full
data locality, Linux-native, headless-CLI, text-based (ZIP-container) formats.**

**Positioning of the field, one line each:**
- **Zemax OpticStudio Premium** — the complete professional standard; the broadest single product; the yardstick.
- **CODE V (Keysight)** — the lens-designer's design/optimization/tolerancing powerhouse (Global
  Synthesis, Wavefront Differential, BSP physical optics); deliberately narrow — non-sequential,
  illumination, and scatter live in sibling LightTools.
- **OSLO Premium (Lambda Research)** — a 50-year-mature sequential design tool with deep classical
  aberration theory, best-in-class Gaussian-beam/ABCD and partial-coherence tooling, and GRIN;
  non-sequential/illumination/stray-light delegated to sibling TracePro. Windows-only; a free EDU tier exists.
- **QUADOA Optical CAD** — a modern *sequential imaging-design* suite (optimization, tolerancing,
  Forbes/GRIN/freeform, full analysis) that deliberately has **no non-sequential engine** and no
  volume/Mie scattering — the philosophical inverse of MieWorkbench.
- **3DOptix** — a free/freemium, browser-based, GPU-cloud **non-sequential** tracer whose killer
  feature is a ~50,000-part real-vendor catalog and near-zero onboarding friction; its wave-optics
  and polarization are real but shallow and tier-gated.
- **MieWorkbench** — a free, Linux-native, GPU-accelerated **coherent non-sequential physics
  engine** with best-in-class energy bookkeeping, Mie scattering, polarization/birefringence
  physics, and a unique ultrafast/nonlinear layer — but no optimization/tolerancing/design
  apparatus and a steep authoring workflow.

**Head-to-head verdicts (detail in §6):**
- **Zemax vs MieWorkbench → Zemax** (breadth is overwhelming; MieWorkbench wins ~a dozen physics/
  compute/cost line items plus the entire time-domain axis).
- **CODE V vs MieWorkbench → CODE V overall**, but sharply complementary — CODE V owns
  design/optimization/tolerancing; MieWorkbench owns non-sequential physical transport, volume/Mie
  scattering, coherent-by-default, the energy ledger, and time-domain/nonlinear.
- **OSLO vs MieWorkbench → OSLO overall** on design breadth, but MieWorkbench wins every
  non-sequential/volume-scatter/coherent-default/time-domain line and cost/GPU.
- **QUADOA vs MieWorkbench → QUADOA overall**, complementary as above.
- **3DOptix vs MieWorkbench → depends on the user.** 3DOptix wins usability/catalog/accessibility
  decisively; MieWorkbench wins physics-depth decisively.

---

## 2. Package profiles

### 2.1 MieWorkbench (this repo)
A PySide6 + VTK desktop GUI wrapped around a coherent Monte-Carlo, fully-vectorial (Jones)
**non-sequential** ray tracer driven by annotated FreeCAD models. Physics: real interference/
diffraction via a Huygens/Rayleigh–Sommerfeld final gather; uniaxial **and biaxial** birefringence
with walk-off; TMM coatings; four grating models (lamellar, Kogelnik VBG, Dammann, measured
table); Beckmann roughness + ground-glass diffusers + measured ABg BSDF (BRDF-side); **exact Mie
particle clouds** (validated vs Wiscombe MIEV0); and a **pulsed/ultrafast/nonlinear layer** (fs/SC
sources, time-domain spectrogram/streak/cube, SHG/Pockels/Kerr/TPA/saturable, GDD budget).
Detectors are planar (curved incoherent), producing irradiance cubes, spectra, Stokes/DOP maps,
per-element power tables, a **9-bucket energy-audit ledger** (closure gated at `1e-3`), and — since
the lowhanging round — **named analysis products** (PSF, FFT-MTF, encircled/ensquared energy, spot
diagrams, ray/OPD fans, source-referenced Zernike + Maréchal Strehl, ghost/stray-light path
ranking, photometric lux/lm/cd, spectrometer λ-vs-x). GPU-accelerated coherent gather (CUDA/torch)
plus a compiled OpenMP+CUDA **C engine** (~8.3× wall-clock geomean). **No optimization, no
tolerancing, no sequential design.** Free, Linux, self-hosted; portable `.MieWB`/`.MieSim` ZIP
formats + a headless CLI. Design philosophy: *trace what a ray actually hits, get the physics
right, with an auditable energy balance.* Target user: a physics-literate optical engineer/
researcher who needs interference, polarization, scattering, ultrafast, and stray-power fidelity —
not a merit-function lens designer.

### 2.2 Ansys Zemax OpticStudio (Premium)
The industry-standard optical design suite (now Synopsys-owned; see ownership note). Two paradigms
in one product: **sequential** (Lens Data Editor, the natural home of imaging design/optimization/
tolerancing) and **non-sequential** (NSC editor, physical hit-order, splitting/scatter,
illumination/stray-light), bridged by a mixed mode. Premium adds over Professional: **Physical
Optics Propagation** (gridded complex-field propagator), CAD part import, **RCWA** grating
efficiency, TrueFreeForm, Radiant measured sources, advanced stray-light (Path Analysis / Critical
Ray Tracer). *Note:* full optimization (incl. Global/Hammer), tolerancing (incl. Monte-Carlo), and
non-sequential mode are **Professional-and-up**; **STAR (STOP/FEA import) is Enterprise-only — not
in base Premium.** Windows-only; ray tracing is CPU-only (Zemax GPU-accelerates only Huygens
PSF/MTF and single-mode fiber coupling); deep ZOS-API (C#/Python/MATLAB, 4 modes) + ZPL macros.
Commercial subscription (quote-only; dated third-party band ≈ $4.9k–$14.9k/yr). Target user:
professional lens and illumination designers.

### 2.3 CODE V (Keysight, formerly Synopsys/ORA)
The reference-grade **sequential, prescription-based lens-design and image-evaluation** program
(Lens Data Manager: an ordered surface list traced deterministically surface-to-surface — not a
spatial 3D/CAD non-sequential scene). Non-sequential surfaces exist only as a bounded exception
(prisms/fibers/corner cubes). Its optimization suite is best-in-class: **Global Synthesis** (a
directed global optimizer that returns many distinct local minima) + **Glass Expert** (automatic
glass substitution); its **Wavefront Differential** tolerancing is uniquely fast (much faster than
MC — exact multiplier *unverified* — with cross-terms, runnable inside the optimization loop for
tolerance desensitization). Deep,
mature classical analysis (PSF/MTF/Zernike/Strehl/spot/ray-fan/encircled energy), **Beam Synthesis
Propagation** (BSP — a beamlet physical-optics engine with NASA-TPF heritage, handling
GRIN/birefringent/segmented apertures), full freeform/Forbes-Q/DOE surfaces, **Image Simulation**
(convolve a scene through the modeled lens), MECo thermal/athermalization, and **SigFit** (third-
party) STOP integration. **Illumination, non-imaging, radiometric stray-light, BSDF/volume/Mie
scatter, IES/LDT, and photometric units are all delegated to sibling LightTools.** Polarization is
Jones ray tracing (uniaxial birefringence yes; no Stokes/DOP maps, no Mueller, no biaxial). No
native non-sequential MC engine, no energy-audit ledger, no GPU, Windows-only, no free tier, no
first-class native Python SDK — but it *is* scriptable via Macro-PLUS **and a documented COM
interface** (MATLAB/Excel automation; Python can drive it via COM). Target user: the professional lens designer (cameras,
zoom/cinema, lithography, space telescopes, meta-optics).

### 2.4 OSLO Premium (Lambda Research)
A **sequential, surface-based** optical design program with 50-year Rochester/Sinclair roots
(Lambda Research since 2001; also on the Altair Partner Alliance since Dec-2023). A chain of
local-coordinate analytic surfaces traced in prescribed order, with paraxial + real ray trace,
damped-least-squares + global (Adaptive Simulated Annealing) optimization, and ISO-10110
tolerancing (full Monte-Carlo with skew/kurtosis/yield). Genuine strengths: deep classical
aberration theory (Seidel through 7th order, eikonal tracing); **best-in-class Gaussian-beam/ABCD**
laser tooling (cavities, astigmatic beams, fiber coupling); an unusually complete **partial-
coherence "projector"** imaging module (Van Cittert–Zernike, Köhler, lithography σ); full Jones
polarization with uniaxial walk-off; **GRIN** (10 gradient forms + Gradium); a compiled internal
application language (**CCL**) plus interpreted SCP; and the full named analysis suite
(PSF/MTF/Zernike/Strehl/spot/fan/encircled energy). A **secondary, bolted-on non-sequential** mode
(surface "groups" for prisms/light-pipes/arrays) exists but is explicitly slower and surface-based;
real illumination/stray-light/BSDF scatter is delegated to sibling **TracePro**. Biaxial materials
are explicitly unsupported *(per the 2011 Optics Reference; unconfirmed against current Premium
docs)*; no energy ledger, no mesh/BVH, no GPU/multithread evidence, no undo/redo, Windows-only. Ships in tiers (**EDU free**, ~10-surface cap; Light; Standard; Premium).
Target user: classical lens/laser designers and educators (the free EDU tier is widely taught).

### 2.5 QUADOA Optical CAD
A modern (Berlin, GmbH founded 2021) **sequential / multi-sequential** design suite built on an
object/assembly hierarchy (base surface + stacked parametric layers) with the light path expressed
as a separate reusable **Sequence** object — its "Multi-Sequential Raytracing" USP. Strong where
Zemax is strong: full optimization (DLS + trust-region, glass substitution, multi-config),
tolerancing (sensitivity + 9-distribution Monte-Carlo + chainable compensators), the full analysis
suite (FFT/Huygens/geometric PSF & MTF, Zernike, Strehl, encircled energy, ray fans, ghost
analysis), Forbes Q-type / biconic / freeform / **GRIN** surfaces, Schott+Ohara catalogs,
STEP/IGES/STL import + Zemax ZMX/SEQ interop, a **Stokes + Poincaré** polarization suite, and a
Python/MATLAB/C++ SDK. Its coordinate model is a genuine superset (absolute + sequential chaining +
nested assemblies + parametric pickups, per-object switchable). **Deliberately has no
non-sequential engine** ("under development"), **no volume/Mie/participating-media scattering**, and
only scalar (efficiency-free) phase gratings. Wave optics is a **paid add-on**; CPU-only per its
own manual; no documented headless/CLI. Windows + Linux native. Target user: imaging/AR-VR/
metrology/medical lens designers.

### 2.6 3DOptix
A free/freemium, **browser-based, GPU-cloud non-sequential** ray tracer. Its standout is a
**~50,000-component real-vendor catalog** (Thorlabs, Edmund, Semrock, Chroma, …) with correct
prescriptions and mountable cage/breadboard hardware — a genuine digital-twin-to-bench workflow,
plus its own BreadBox™ hardware line. Non-sequential geometric core is mature (ray-splitting, MAX
BOUNCES, 4 BRDF scatter models, manual stray-light workflow). It **does** have a real scalar
Huygens/Fresnel diffraction solver and coherent detection — but these are advanced-tier, gated
features layered on a default **ray-based interference approximation** the vendor itself describes
as "physical optics using geometric optics"; MTF/PSF/wavefront/polarization products are variously
"coming soon." Polarization is shallow (component power + circular-polarization spatial map; no
exposed Jones/Mueller/DOP). Coatings are 6 idealized presets. No optimization, tolerancing,
thermal, or photometric units. Metered GPU-hours; ray ceilings and wave optics gated by tier (free
tier is a single 550 nm wavelength, no API). Python SDK (beta, paywalled). Cloud-only (no offline,
no data locality). Target user: students, educators, lab engineers laying out benchtop systems
from catalog parts.

---

## 3. Detailed per-package feature write-ups

Organised by the shared taxonomy (A–S). Line-item ratings are consolidated in the master table
(§4); this section gives the qualitative shape of each package.

### 3.1 MieWorkbench
- **A Ray-tracing core:** non-sequential, stratified Monte-Carlo, coherent **and** incoherent
  simultaneously (per gather key), reflection-generation cap (default 6, adjustable to ~200),
  scene-wide TLAS + mesh BLAS in the C engine (Python uses mesh BVH), exact analytic intersection
  for canonical surfaces. No sequential mode.
- **B Physical optics:** the differentiator — a full coherent Huygens/Rayleigh–Sommerfeld gather
  with obliquity factor is the *default* engine; real interference + diffraction on every coherent
  run (double-slit pitch/visibility validated end-to-end). **PSF, FFT-MTF (+ MTF50), and
  encircled/ensquared energy now ship as named products** (from `analysis_field.py`, `--save-fields`,
  seed-0/collimated-bench fidelity). **No gridded/ABCD POP or beamlet (BSP) propagator; no partial
  coherence; no measured-interferogram surface.**
- **C Polarization:** full Jones tracing; Stokes + degree-of-polarization maps; **validated
  uniaxial birefringence with walk-off** (calcite 6.23°@45°/590 nm) **and a validated biaxial
  two-sheet solver** (KTP/KTA/LBO/BiBO, `<1e-9`) — the only biaxial in this field; TMM coatings;
  four grating models with **real diffraction efficiency**. No conical refraction, no optical
  activity, no RCWA, no Mueller-matrix formalism.
- **D Surfaces:** spherical, even-asphere (extract-verified `<1 µm`), cylindrical, conical,
  toroidal — analytic. Freeform only via triangle **mesh (incoherent-power only)**. DOE via the
  grating models. **No GRIN, no Q-type/Forbes. CAD (STEP/IGES) enters only through FreeCAD →
  mesh-BVH (incoherent); no analytic-face recovery yet.**
- **E Sources:** collimated + divergent + **Gaussian-beam (waist/M²) + apodization**; mono/uniform/
  Gaussian-band/tabulated spectra; coherent/incoherent; full polarization states; **pulsed/fs/
  supercontinuum**. **No ray-aiming to a real pupil, no measured IES/rayfile ingestion.**
- **F Detectors/analysis:** planar irradiance cube, spectra, Stokes/DOP, per-element power table,
  the **9-bucket energy-audit ledger with `<1e-3` closure** (strongest in the field), plus spot
  diagrams, ray/OPD fans, encircled energy, **ghost/stray-light path ranking**, photometric lux
  maps, spectrometer profiles. Curved detectors are incoherent-only. No image-simulation product.
- **G Materials:** **168 materials** (41 Schott/Ohara Sellmeier glasses + metals/polymers/crystals),
  38 TMM coatings, 56 filters, 8 gratings, **17 birefringent crystals (13 uniaxial + 4 biaxial)**;
  every row requires a cited reference. **Smaller than commercial catalogs; no dn/dT engine hook
  (data staged), no vendor-component catalog.**
- **H Scattering:** Beckmann roughness + diffusers + measured **ABg BSDF (BRDF-side, v1)** +
  scatter-to-target importance aim + **exact Mie particle clouds** (explicit spheres to ~200k, then
  a continuum medium). **No BTDF, no dedicated Path-Analysis stray-light tool.**
- **I/J/K/L/M Optimization / Tolerancing / Thermal / Photometry / Multi-config:** optimization &
  tolerancing = **none** (CLI `--var` sweeps + `--seeds` speckle averaging only); **photometric
  units now shipped (L1)**; multi-config is CLI-sweep/Variables-dock only.
- **N GUI/UX:** VTK 3D view, ~54-element parametric primitive library, ~20-level undo + macros,
  wizards (thick-lens + waveplate solvers), 1 s-debounced live ray preview, **tracer-bead
  animation** (photons at c/n), results galleries, validation/problems pane. Linux desktop, steep
  authoring curve.
- **O Coordinate system:** FreeCAD Placement (position + quaternion), absolute world pose + Euler,
  reference-point resolver, expression-bound placements, **snap-to-optical-axis + drag-along-axis**,
  **an optical-train chain model (anchored/chained, ports) and a 1-click fold operator** (insert
  fold mirror + rigidly fold/unfold the downstream train). No sequential surface table.
- **P Data:** `.MieWB`/`.MieSim`/`.FCStd` ZIP formats, headless CLI (`miewb_tool`), case locking;
  text-based inner members. Scripting = external Python CLI only (**no in-app API/macro**).
- **Q Performance:** vectorized numpy trace + **CUDA/torch coherent gather** + a compiled
  **OpenMP+CUDA C engine** (~8.3× wall-clock geomean) + **multi-process `--workers` sharding**. No
  multi-node/cloud.
- **R Commercial:** free, Linux, self-hosted, full data locality; no formal support/community;
  steep learning curve; no vendor catalog.
- **S Ultrafast / time-domain / nonlinear (unique axis):** pulsed/fs/SC sources, time-domain
  profile/spectrogram/streak/cube products, χ² (SHG/Pockels) and χ³ (Kerr/TPA/saturable) events, a
  per-element GD/GDD/TOD dispersion budget, and source-side SPM/supercontinuum. **No competitor
  here models the time domain at all.** (Approximations: undepleted SHG with no walk-off, source-
  side-only SPM, material-only GDD — see §7/`future.md`.)

### 3.2 Zemax OpticStudio (Premium) — strengths that dominate
The complete analysis suite (B/F), optimization (I), tolerancing (J), thermal (K), illumination +
photometry with IES/LDT and roadway design (L), the largest glass catalogs + 13 dispersion models
(G), RCWA gratings (C), measured BSDF + importance sampling + Path Analysis stray-light (H), 20+
surface types + GRIN + diffractive phase surfaces + STEP/IGES/SAT/Parasolid import (D), ~20 NSC
source objects + ray-aiming + apodization (E), the 4-mode ZOS-API + ZPL + CSG Booleans (P), and
160-config multi-config/zoom/scanning (M). Genuine *gaps* relative to MieWorkbench: NSC "coherent"
is a geometric coherent ray-sum with no Huygens spreading (rigorous diffraction is a separate
scalar POP pass; POP is scalar, not vectorial); **ray tracing is CPU-only**; **Mie only via a
bundled bulk-scatter DLL (MSP)**; Stokes is ensemble-derived (no Mueller); flux accounting is
distributed, not a closed audit; **biaxial, optical activity, and any time-domain/pulsed/nonlinear
modeling are absent**; Windows-only; expensive.

### 3.3 CODE V (Keysight) — the design/optimization/tolerancing champion, deliberately narrow
Owns the *design purist's* apparatus: full optimization (I) anchored by **Global Synthesis**
(directed global optimizer surfacing many distinct minima) and **Glass Expert**; the fastest
tolerancing in the field (J) via **Wavefront Differential** (far faster than MC — multiplier *unverified* — cross-terms, in-loop
desensitization) plus full MC/yield; the mature classical analysis suite (B/F: PSF/MTF/Zernike/
Strehl/spot/fan/encircled energy); **Beam Synthesis Propagation** (B3/B10 — beamlet physical optics
handling GRIN/birefringent/segmented apertures); full Forbes-Q/Zernike/Chebyshev/XY freeform +
HOE/binary DOE + GRIN surfaces (D); **Image Simulation** (F10); MECo thermal + **SigFit** STOP
(K); 21-wavelength spectral + apodization + Gaussian/BSP coherent sources (E); and multi-config/
zoom (M). *Gaps vs MieWorkbench:* **no true non-sequential MC engine** (bounded prism/fiber
surfaces only — real NSC/illumination/BSDF/volume/Mie scatter all live in the *separate* LightTools
product); **no energy-audit ledger**; polarization is Jones-only with **no Stokes/DOP maps, no
Mueller, no biaxial, no optical activity**; grating efficiency is scalar (RCWA is in RSoft); **no
GPU, Windows-only, no cloud, no free tier, no first-class native Python SDK (Macro-PLUS + a documented COM
interface only)**; and **no
time-domain/pulsed/nonlinear modeling.**

### 3.4 OSLO Premium (Lambda Research) — mature classical design, delegated non-sequential
Owns 50 years of classical sequential design: deep aberration theory + eikonal tracing (A/B); the
full analysis suite (B/F); **best-in-class Gaussian-beam/ABCD** cavity/astigmatic/fiber-coupling
tooling (B9/E4) and a genuinely deep **partial-coherence projector** (B8); full DLS + **Adaptive
Simulated Annealing** global optimization (I) and rigorous ISO-10110 + Monte-Carlo tolerancing with
skew/kurtosis/yield (J); **GRIN** (10 forms + Gradium) and deep DOE (D); full Jones polarization
with uniaxial walk-off + 31-layer TMM coatings (C/G); thermal/athermalization (K); the **CCL**
compiled application language (P); 3000+ lens prescriptions + Schott/Ohara/Hoya catalogs (G). *Gaps
vs MieWorkbench:* **non-sequential is a slow, opt-in "surface groups" sub-mode** (real
NSC/illumination/BSDF scatter is delegated to sibling **TracePro**); **no automatic Fresnel R+T
splitting** (ghosts via manual multiconfiguration); **no mesh/BVH, no biaxial ("OSLO does not treat
biaxial materials" — 2011 manual, unconfirmed vs current Premium), no energy ledger, no Stokes map,
no Mueller**; **no GPU/multithread/cloud** and
**no undo/redo** anywhere in the 427-page reference; Windows-only; and **no time-domain/nonlinear
modeling.** Primary source is dated (2011 manual); later additions cross-checked where possible.

### 3.5 QUADOA — strengths and the paradigm inversion
Owns the entire *design* apparatus MieWorkbench lacks: optimization (I), tolerancing (J, with 9 MC
distributions + chainable compensators), thermal (K), multi-config/zoom (M), the full analysis
suite (B/F), Forbes Q-type/biconic/freeform/**GRIN**/off-axis-asphere surfaces + CAD & Zemax interop
(D), real glass catalogs + 4 dispersion models (G), ray-aiming + apodization + Gaussian-beam +
imported ray-file sources (E), a **Stokes + Poincaré** polarization suite (Mueller claimed in
marketing but unconfirmed) (C), and a Python/MATLAB/C++ SDK (P). Its coordinate model is a genuine
superset of MieWorkbench's. *Gaps:* **no non-sequential engine at all** (ghosts only); **no
volume/Mie/participating-media scattering**; gratings are scalar and **efficiency-free**; bulk
birefringence claimed but unverified; wave optics is a **paid add-on**; **CPU-only**; no documented
headless/CLI; and **no time-domain/nonlinear modeling.**

### 3.6 3DOptix — strengths and the shallow-physics caveat
Owns *accessibility*: browser, zero install, drag-and-drop from a **~50,000-part real-vendor
catalog** with mountable cage/breadboard hardware (O/G/N), structured tutorials + Academy, cloud
GPU with billion-ray ceilings on paid tiers (Q), real STEP CAD import (D), measured IES/TM-25 source
files (E), a genuine (if narrow) Huygens diffraction solver and coherent detection (B), and
PSF/MTF/spot/encircled-energy products (F). *Caveats, heavily sourced:* wave optics and polarization
are **shallow and tier-gated** — the default interference mechanism is a ray-based approximation the
vendor concedes is "not true wave physics"; polarization is component power + a circular-
polarization map with **no exposed Jones/Stokes/Mueller/DOP**; coatings are 6 idealized presets; **no
optimization, tolerancing, thermal, or photometric units**; several imaging products are "coming
soon"; cloud-only (no data locality); metered GPU-hours; small vendor with no independent review base.

---

## 4. Master comparison table

Legend: ✅ full · 🟡 partial/shallower · ⚠️ workaround · ❌ none. **MWB** = MieWorkbench.
Column order (physics engine → design suites → browser NSC): **MWB · Zemax · CODE V · OSLO · QUADOA
· 3DOptix**. "Best" names the strongest implementation; ties noted. Cells append a terse qualifier
directly to the emoji (e.g. `🟡mesh`). Rows appended after this analysis's field expansion:
**B10/B11, F10, I6, J5, and the whole new §S** — the pre-existing 118 IDs keep their numbers.

### A. Ray-tracing core
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|A1|Non-sequential tracing|✅|✅|⚠️bounded|🟡groups|❌|✅|**Zemax** — deepest NSC; MWB/3DOptix full; CODE V prism/fiber-only (→LightTools), OSLO slow groups (→TracePro), QUADOA none|
|A2|Sequential design tracing|❌|✅|✅|✅|✅|🟡|**Zemax/CODE V/OSLO/QUADOA** — canonical surface-order design; MWB none|
|A3|Monte-Carlo radiometric transport|✅|✅|⚠️|🟡scripted|❌|🟡|**Zemax/MWB** — true photon MC; CODE V/OSLO stochastic-utility only, QUADOA MC=tolerancing|
|A4|Coherent field tracing, absolute power|✅|🟡ray-sum|🟡BSP|🟡1-beam|🟡add-on|🟡gated|**MWB** — inline coherent gather books absolute power; others geometric/imaging-framed/gated/add-on|
|A5|Ray splitting / bounce control|✅|✅|✅ghost|⚠️manual|⚠️ghosts|✅|**Zemax** — full budgets; CODE V splits for ghosts; OSLO/QUADOA emulate via multiconfig/ghosts|
|A6|Spatial acceleration (BVH)|✅C-TLAS|✅|❌|❌|❌|✅GPU|**3DOptix** — GPU BVH; Zemax + MWB (C-engine TLAS) full; seq design tools don't need it|
|A7|Analytic vs mesh optical surfaces|✅+mesh|✅+CAD|⚠️grid-sag|🟡analytic|✅analytic|✅+CAD|**Zemax** — analytic + true NURBS/ACIS import|

### B. Physical optics
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|B1|Real interference (default)|✅|✅POP|🟡Hopkins|🟡1-beam|✅|🟡approx|**MWB/QUADOA** — inline default; Zemax rigorous POP; CODE V/OSLO single-beam|
|B2|Free-space diffraction|✅|✅|✅BSP|✅|✅|🟡|**Zemax/CODE V** — angular-spectrum POP / BSP; MWB Rayleigh–Sommerfeld gather full|
|B3|POP / beam propagation (gridded/ABCD/BSP)|⚠️|✅|✅BSP|🟡ABCD|✅add-on|🟡|**Zemax/CODE V** — full gridded POP / Beam Synthesis Propagation|
|B4|PSF (named product)|✅seed0|✅FFT+Huy|✅|✅|✅3 eng|🟡geo|**Zemax/QUADOA** — FFT+Huygens+geometric; CODE V/OSLO/MWB full|
|B5|MTF / OTF|✅FFT|✅|✅|✅|✅|🟡|**Zemax/CODE V/OSLO/QUADOA** — full diffraction MTF; MWB FFT-MTF now ships|
|B6|Wavefront maps / Zernike|🟡src-pupil|✅3 bases|✅|✅|✅|❌|**Zemax** — Standard/Fringe/Annular; CODE V/OSLO/QUADOA full; MWB source-referenced pupil|
|B7|Strehl ratio|🟡Maréchal|✅|✅|✅|✅|❌|**Zemax/CODE V/OSLO/QUADOA**; MWB Maréchal-only (no PSF-peak)|
|B8|Partial coherence modeling|❌|🟡Γ-model|✅IMS|✅projector|🟡binary|❌|**OSLO/CODE V** — Van Cittert–Zernike projector / Image-Sim partial coherence|
|B9|Gaussian-beam (ABCD) analysis|❌|✅|✅|✅cavity|✅|🟡noM²|**OSLO** — best-in-class cavity/astig/fiber; Zemax/CODE V/QUADOA full|
|B10|Beam-synthesis / beamlet physical optics|❌|🟡POP|✅BSP|🟡|🟡add-on|❌|**CODE V** — BSP beamlet engine (GRIN/birefringent/segmented apertures)|
|B11|Interferogram / measured-wavefront import|❌|✅|🟡|🟡|🟡|❌|**Zemax** — measured-data/interferogram surface; CODE V/OSLO/QUADOA partial|

### C. Polarization
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|C1|Jones-vector tracing|✅|✅|✅|✅|✅|🟡|**tie** (all but 3DOptix, which is component-power only)|
|C2|Stokes / DOP maps|✅|🟡ensemble|❌|🟡per-ray|✅|🟡map|**QUADOA/MWB** — first-class Stokes maps (QUADOA + Poincaré); CODE V none, OSLO per-ray only|
|C3|Mueller-matrix formalism|❌|⚠️|❌|❌|🟡claim|❌|**QUADOA** (unconfirmed) — Mueller claimed in marketing; Zemax via Jones-probe workaround|
|C4|Uniaxial birefringence + walk-off|✅|✅|✅|✅|🟡unverif|❌|**MWB/Zemax/CODE V/OSLO** — validated walk-off physics|
|C5|Biaxial birefringence|🟡biax|❌|❌|❌|❌|❌|**MWB** — the *only* biaxial two-sheet solver here (KTP/KTA/LBO/BiBO; no conical refraction). OSLO explicitly excludes biaxial|
|C6|Optical activity / gyrotropy|❌|❌|❌|❌|❌|❌|*none* — universal gap|
|C7|TMM thin-film coatings|✅|✅|✅|✅31-layer|✅|⚠️presets|**Zemax** — largest catalog; MWB/CODE V/OSLO/QUADOA full TMM; 3DOptix 6 presets|
|C8|Polarizers / retarders / waveplates|✅|✅|✅|✅|✅|🟡|**tie** (all but 3DOptix)|
|C9|Grating diffraction efficiency|✅models|✅RCWA|🟡scalar|🟡scalar|❌scalar|🟡|**Zemax** — rigorous RCWA; **MWB** best *closed-form* (Kogelnik/Dammann/table); CODE V/OSLO scalar (RCWA→RSoft)|

### D. Surfaces & geometry
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|D1|Spherical / conic|✅|✅|✅|✅|✅|✅|*tie*|
|D2|Even/odd polynomial asphere|✅|✅|✅|✅|✅|✅|**Zemax/CODE V** — even/odd/extended; all support asphere|
|D3|Q-type (Forbes) asphere|❌|✅|✅|🟡|✅|❌|**Zemax/CODE V/QUADOA** — native Forbes Q-type|
|D4|Freeform (XY/Zernike/Chebyshev/grid)|⚠️mesh|✅20+|✅|🟡|✅|⚠️claim|**Zemax** — TrueFreeForm; CODE V (Fringe-Zernike/Chebyshev/2D-Q/XY) close|
|D5|Diffractive / DOE / binary phase|✅grating|✅Binary1-3|✅HOE|✅|✅phase|✅grating|**Zemax/CODE V/OSLO** — Binary/HOE/Sweatt + grating|
|D6|Fresnel surfaces|✅facet|✅|✅|🟡approx|✅|❌|**Zemax/CODE V/QUADOA/MWB** — analytic Fresnel|
|D7|Toroidal / biconic|✅|✅|🟡AAS|✅|✅|🟡biconic|**Zemax/OSLO/QUADOA/MWB**|
|D8|GRIN media|❌|✅|✅|✅10-form|✅|❌|**OSLO** — 10 gradient forms + Gradium; Zemax/CODE V/QUADOA full|
|D9|Mesh / CAD-imported optical surface|🟡incoh|✅|❌mech|❌|❌mech|🟡unstable|**Zemax** — true solid tracing|
|D10|CAD import (STEP/IGES/SAT/STL)|🟡FC|✅|✅|🟡export|✅|🟡STEP|**Zemax** — widest kernel; CODE V/QUADOA STEP/IGES/SAT|

### E. Sources
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|E1|Geometric source types|🟡3|✅20+|🟡field|✅ext/array|✅|✅|**Zemax** — ~20 NSC emitters; OSLO extended/array/astig|
|E2|Spectral definition|✅|✅|✅21λ|✅|✅|🟡|**Zemax/CODE V/OSLO/MWB/QUADOA** — MWB adds Gaussian band + tabulated SPD|
|E3|Coherent source (inline)|✅|🟡POP|✅BSP|✅|✅add-on|🟡gated|**MWB/OSLO/CODE V** — coherent source inline|
|E4|Gaussian-beam source|✅M²|✅|✅|✅astig|✅|🟡noM²|**OSLO** — deepest astigmatic Gaussian; Zemax/CODE V/QUADOA/MWB full (MWB adds M²)|
|E5|Apodization|✅|✅|✅|✅|✅|❌|**tie** (all but 3DOptix)|
|E6|Ray-aiming to real pupil|❌|✅|🟡auto|✅5-mode|✅|❌|**Zemax/OSLO/QUADOA** — OSLO 5 ray-aim modes + telecentric|
|E7|Measured source files (IES/LDT/rayfile)|❌|✅|❌|🟡|🟡rayfile|✅IES/TM25|**Zemax** — IES/LDT/rayfile/Radiant; 3DOptix IES/TM-25 (CODE V→LightTools)|
|E8|Full polarization state on source|✅|✅|✅|✅|✅|🟡|**tie** (all but 3DOptix)|

### F. Detectors & analysis
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|F1|Irradiance maps|✅|✅|✅|🟡1-field|✅|✅|**tie** — OSLO single-field PSF, not scene irradiance|
|F2|Spectral / color detector|✅spectra|✅color|🟡|❌|🟡|✅spectral|**Zemax** — CIE color + per-pixel spectrum; MWB spectra + spectrometer mode|
|F3|Curved detectors|🟡incoh|✅|🟡conic|🟡|🟡|❌|**Zemax** — curved/annular detector objects; MWB incoherent-only|
|F4|Spot diagrams|✅|✅|✅|✅|✅|✅|**tie** — all six now ship spot diagrams (MWB via `--export-rays`)|
|F5|Ray fans (OPD/transverse)|✅|✅|✅|✅|✅|❌|**tie** (all but 3DOptix)|
|F6|Encircled / ensquared energy|✅|✅|✅|✅|✅|🟡circ|**tie** (all but 3DOptix partial)|
|F7|Polarization / Stokes detector map|✅|🟡|🟡pol-wt|🟡x/y|✅|🟡xyz|**QUADOA/MWB** — first-class Stokes maps|
|F8|Energy-audit ledger + closure|✅`<1e-3`|🟡distrib|❌|❌|🟡scattered|⚠️manual|**MWB** — unique closed `<1e-3` audit|
|F9|Ghost / stray-light analysis|✅rank|✅Path|🟡GhoView|🟡Narcissus|✅ghost|🟡manual|**Zemax** — Path Analysis + Critical Ray Tracer; MWB now ranks ghost paths; CODE V/OSLO specular-only (full→LightTools/TracePro)|
|F10|Image simulation (extended scene)|❌|✅|✅IMS|🟡|🟡|❌|**Zemax/CODE V** — convolve a scene through the modeled system|

### G. Materials & coatings
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|G1|Glass-catalog breadth|🟡168|✅1000s|✅catalogs|✅3000+|✅1000s|✅catalog|**Zemax/OSLO** — most catalogs; MWB 168 cited rows (small but growing)|
|G2|Dispersion-model breadth|🟡4|✅13|🟡~4|✅|✅4|🟡6|**Zemax** — 13 formulas|
|G3|dn/dT thermal index data|❌|✅|✅|✅|✅|❌|**Zemax/CODE V/OSLO/QUADOA** — MWB data staged, no engine hook|
|G4|Coating model (TMM stack)|✅|✅|✅|✅31|✅|⚠️presets|**Zemax** — largest; MWB/CODE V/OSLO/QUADOA full TMM (OSLO 31-layer)|
|G5|Coating synthesis (needle)|❌|❌|⚠️import|🟡opt|🟡opt|❌|**OSLO/QUADOA** — layer-thickness opt (no true needle anywhere)|
|G6|Filters / transmission|✅|✅|🟡|✅|⚠️|✅catalog|**Zemax/MWB/OSLO** — internal-transmittance + interference filters|
|G7|Birefringent material library|✅17|✅|🟡|✅5|🟡unverif|❌|**MWB/Zemax** — MWB 17 (13 uniaxial + 4 biaxial); OSLO 5|
|G8|Component vendor catalog|❌|🟡|🟡.seq|✅DB|❌|✅~50k|**3DOptix** — ~50,000 real vendor parts|

### H. Scattering & stray light
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|H1|Surface scatter models|✅Beckmann|✅7|❌|❌|🟡3|✅4|**Zemax** — Lambertian/Gaussian/ABg/BSDF/DLL; CODE V/OSLO delegate to LightTools/TracePro|
|H2|Measured BSDF import|🟡BRDF|✅|❌|❌|❌|❌|**Zemax** — full BSDF; MWB v1 BRDF-only|
|H3|Importance sampling|🟡aim|✅|⚠️LT|❌|❌|❌|**Zemax** — importance-target; MWB scatter-to-target aim|
|H4|Volume / participating-media scatter|✅|🟡HG-DLL|❌|❌|❌|❌|**MWB** — continuum medium + rigor below (CODE V→LightTools)|
|H5|Rigorous Mie particle scattering|✅Wiscombe|🟡DLL|❌|❌|❌|❌|**MWB** — first-class Mie validated vs Wiscombe MIEV0; Zemax only via MSP DLL; CODE V Mie→LightTools|
|H6|Dedicated stray-light workflow|🟡ghost|✅Path|⚠️LT|⚠️TracePro|🟡ghost|🟡manual|**Zemax** — Path Analysis / Critical Ray Tracer|

### I. Optimization
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|I1|Merit function / operands|❌|✅300+|✅60+|✅auto-EF|✅named|⚠️scipy|**Zemax** — 300+ operands + wizard; CODE V/OSLO full|
|I2|Local optimizer (DLS)|❌|✅|✅|✅DLS|✅|⚠️|**Zemax/CODE V/OSLO/QUADOA**|
|I3|Global optimizer|❌|✅Hammer|✅GS|✅ASA|🟡restart|❌|**Zemax/CODE V** — Global Search+Hammer / Global Synthesis; OSLO ASA|
|I4|Glass substitution|❌|✅|✅Expert|🟡|✅|❌|**Zemax/CODE V/QUADOA** — CODE V Glass Expert|
|I5|Multi-config optimization|❌|✅|✅|✅|✅|❌|**Zemax/CODE V/OSLO/QUADOA**|
|I6|Directed global synthesis (many minima)|❌|🟡Hammer|✅GS|🟡ASA|❌|❌|**CODE V** — Global Synthesis surfaces many distinct design forms in one run|

### J. Tolerancing
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|J1|Sensitivity analysis|❌|✅|✅|✅ISO|✅|❌|**Zemax/CODE V/OSLO/QUADOA**|
|J2|Monte-Carlo tolerancing|❌|✅|✅|✅stats|✅9-dist|❌|**QUADOA/OSLO** — 9 dist / full skew-kurtosis stats; Zemax/CODE V full|
|J3|Compensators|❌|✅|✅|✅|✅chain|❌|**all four design suites**|
|J4|Yield analysis|❌|✅|✅|✅|🟡|❌|**Zemax/CODE V/OSLO**|
|J5|Fast differential wavefront tolerancing|❌|🟡|✅|🟡analytic|❌|❌|**CODE V** — Wavefront Differential, far faster than MC (*unverified* multiplier), in-loop desensitization|

### K. Thermal / STOP
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|K1|Thermal (bulk T, dn/dT, CTE)|❌|✅|✅MECo|✅athermal|✅|❌|**Zemax/CODE V/OSLO/QUADOA**|
|K2|STOP / FEA import|❌|❌*Ent*|🟡SigFit|❌|⚠️GRIN-import|❌|**CODE V** — SigFit bridge (Zemax STAR is Enterprise-only, not Premium)|

### L. Illumination / photometry
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|L1|Photometric units (lux/lumen)|✅|✅|⚠️LT|🟡|❌|❌|**Zemax/MWB** — MWB CIE V(λ) lux/lm/cd; CODE V→LightTools|
|L2|Non-imaging / illumination design|⚠️|✅|🟡LT|⚠️TracePro|⚠️|🟡|**Zemax** — NSC illumination + freeform opt|
|L3|IES/LDT export|❌|✅|❌|❌|❌|🟡|**Zemax**|

### M. Multi-configuration / zoom / scanning
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|M1|Multi-configuration editor|⚠️CLI|✅160|✅12|✅|✅unlim|⚠️manual|**QUADOA/Zemax** — unlimited/160 configs; CODE V 12 ACONs|
|M2|Zoom systems|⚠️|✅|✅CAM|✅|✅slider|⚠️|**QUADOA/Zemax/CODE V/OSLO**|
|M3|Scanning systems|⚠️|✅|🟡|🟡|🟡|❌|**Zemax**|

### N. GUI / UX
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|N1|Interactive 3D view|✅|✅|✅|✅|✅|✅|*tie* — all live 3D|
|N2|Shaded / cross-section|🟡|✅|🟡|✅|✅|✅|**Zemax/OSLO/QUADOA/3DOptix**|
|N3|Undo/redo|✅20+cmd|🟡snapshot|❌|❌|✅|🟡|**MWB/QUADOA** — MWB granular per-command + macros; CODE V/OSLO none found|
|N4|Wizards|🟡lens|✅|✅Lens Wiz|🟡|✅|🟡|**Zemax/CODE V/QUADOA**|
|N5|Live update / auto-preview|✅1s|✅|⚠️plugin|✅slider|✅1s|🟡on-demand|**MWB/OSLO/QUADOA/Zemax** — OSLO slider-wheel; CODE V needs K2realm plugin|
|N6|Ease of use / onboarding|❌steep|🟡|🟡steep|🟡|🟡|✅|**3DOptix** — zero-install, tutorials, Academy|
|N7|Cross-platform|🟡Linux|❌Win|❌Win|❌Win|✅Win/Linux|✅browser|**QUADOA/3DOptix** — the three legacy suites are Windows-only|
|N8|Ray animation / visualization|✅bead|🟡|❌|🟡beam|🟡|🟡|**MWB** — physical c/n tracer beads|
|N9|Collaboration / sharing|❌|❌|❌|❌|❌|✅cloud|**3DOptix** — cloud share + Warehouse|

### O. Coordinate & positioning system
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|O1|Absolute spatial placement|✅|✅NSC|✅|🟡gc|✅|✅|**tie** — all support absolute pose|
|O2|Sequential surface chaining|🟡train|✅|✅|✅|✅|❌|**Zemax/CODE V/OSLO/QUADOA** — thickness-to-next chain; MWB element-level train|
|O3|Relative / reference-object chaining|✅chain|✅NSC ref|🟡|🟡rco|✅per-obj|✅LCS ref|**QUADOA/MWB/Zemax/3DOptix** — MWB train ports/distance-along-beam|
|O4|Tilt/decenter + order control|✅Euler|✅CB-order|✅D&B|✅tilt/dec|✅6-DoF+order|✅|**QUADOA** — explicit rot-order + pivot per object; CODE V/OSLO decenter-and-bend|
|O5|Coordinate break / pivot-about-point|✅|✅|✅D&B|✅pivot|✅pivot|🟡|**most suites**|
|O6|Pickup / parametric constraints|✅expr|✅solves|✅solves|✅pickups|✅lookup|❌|**Zemax/CODE V/OSLO** — rich solve set; MWB/QUADOA expression-bound|
|O7|Assemblies / grouping|🟡group|✅NSC ref|❌|🟡groups|✅nested|✅cage|**QUADOA** — first-class nestable assemblies|
|O8|Snap-to-axis / auto-align|✅|⚠️|❌|❌|⚠️|🟡coming|**MWB** — shipped snap-to-optical-axis + drag-along-axis|
|O9|Optical-train / fold operations|✅fold|⚠️CB|🟡D&B|🟡bend|⚠️seq|🟡mounts|**MWB** — 1-click insert-fold + fold/unfold the downstream train; seq tools fold implicitly via coordinate breaks|

### P. Data management
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|P1|Native project / archive format|✅ZIP|✅ZAR|🟡loose|🟡.len|✅optx|✅cloud|**Zemax/MWB/QUADOA** — bundle archives; CODE V/OSLO loose files|
|P2|CAD import/export interop|🟡FC|✅|🟡STEP|🟡export|✅|🟡STEP|**Zemax** — round-trip STEP/IGES/SAT + CSG|
|P3|In-app scripting API|❌|✅ZOS-API|🟡Macro+COM|✅CCL+DDE|✅SDK|🟡beta|**Zemax** — 4-mode C#/Py/MATLAB; OSLO CCL+DDE, QUADOA SDK; CODE V Macro-PLUS + COM automation (no first-class Python SDK)|
|P4|Macro language|❌|✅ZPL|✅Macro+|✅CCL|🟡math|❌|**OSLO/CODE V/Zemax** — CCL / Macro-PLUS / ZPL|
|P5|Headless / CLI batch|✅|✅API|✅IN|🟡DDE|❌|🟡SDK|**MWB/Zemax/CODE V** — MWB CLI-first, CODE V `IN` batch, Zemax via API|
|P6|Version-control-friendly formats|✅|✅ZMX|✅.seq|✅.len|❌binary|❌cloud|**MWB/Zemax/CODE V/OSLO** — ASCII sources; MWB text members in ZIP|

### Q. Performance & compute
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|Q1|CPU multithreading|✅|✅|✅|❌|✅|n/a|**Zemax/CODE V/MWB/QUADOA** — MWB C-engine OpenMP + `--workers`; OSLO no evidence|
|Q2|GPU acceleration|✅gather|🟡Huygens|❌|❌|❌|✅cloud|**3DOptix/MWB** — 3DOptix cloud GPU, MWB local CUDA gather + C engine|
|Q3|Multi-process / distributed|🟡workers|🟡instances|🟡1-machine|❌|❌|✅cloud|**3DOptix** — cloud scale; MWB `--workers` sharding|
|Q4|Cloud compute|❌|🟡add-on|❌|❌|❌|✅|**3DOptix**|
|Q5|Ray / scale ceiling|🟡|✅|🟡250-surf|🟡EDU-cap|🟡|✅1e9|**3DOptix** — billion-ray tiers; Zemax high with caps|

### R. Commercial & practical
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|R1|Cost|✅free|❌$$$|❌$$$$|🟡$$+EDU|🟡$$|✅freemium|**MWB** — free + no metering; OSLO has a free EDU tier; CODE V historically the most expensive|
|R2|License / data locality|✅local|✅offline|✅offline|✅offline|✅offline|❌cloud|**all desktop tools** — MWB full locality; 3DOptix cloud-only|
|R3|Platform / OS|🟡Linux|❌Win|❌Win|❌Win|✅Win/Linux|✅browser|**QUADOA/3DOptix** — Zemax/CODE V/OSLO Windows-only|
|R4|Ecosystem / community|❌|✅|🟡specialist|🟡EDU-base|🟡|🟡|**Zemax** — forum, KB, resellers; CODE V+Zemax ≈90% of the pro market|
|R5|Learning curve / onboarding|❌|🟡|🟡steep|🟡|🟡|✅|**3DOptix**|
|R6|Support / training|❌|✅|✅KeysightCare|✅+Altair|✅|🟡|**Zemax/CODE V/OSLO/QUADOA**|
|R7|Data privacy / ITAR-suitable|✅|✅|⚠️|🟡|✅|❌cloud|**MWB/Zemax/QUADOA** — desktop/local; 3DOptix cloud unsuitable for classified|

### S. Ultrafast / time-domain / nonlinear *(new axis — a MieWorkbench monopoly here)*
| # | Feature | MWB | Zemax | CODE V | OSLO | QUADOA | 3DOptix | Best · why |
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|--|
|S1|Pulsed / time-domain products (profile/spectrogram/streak/cube)|✅|❌|❌|❌|❌|❌|**MWB** — unique; every competitor here is a CW frequency-domain tool|
|S2|Nonlinear optics (χ² SHG/Pockels · χ³ Kerr/TPA/saturable)|🟡|❌|❌|❌|❌|❌|**MWB** — unique (undepleted/no-walk-off approximations, §7)|
|S3|Dispersion / GDD budget (GD/GDD/TOD per element)|🟡material|🟡chromatic|🟡chromatic|🟡chromatic|❌|❌|**MWB** — explicit per-element GDD budget; seq tools give material chromatic dispersion only|
|S4|Supercontinuum / SPM spectral broadening|🟡source|❌|❌|❌|❌|❌|**MWB** — unique (source-side SPM, exact FFT spectrum)|

---

## 5. Overall winner (general feature set)

**Axis-by-axis tally** (who is strongest in each taxonomy block, across all six):

| Axis | Winner | Runner-up | MWB standing |
|--|--|--|--|
|A Ray-tracing core|Zemax|MWB / 3DOptix|Strong (wins coherent-absolute A4)|
|B Physical optics|Zemax / CODE V|OSLO / QUADOA|Weak on POP/beamlet + partial-coherence; unique on coherent-default|
|C Polarization|**MWB**|Zemax / QUADOA|Leader — sole biaxial (C5) + Stokes maps (C2) + grating efficiency (C9)|
|D Surfaces & geometry|Zemax|CODE V|Mid (no GRIN/Q-type/CAD-optical/freeform-analytic)|
|E Sources|Zemax|OSLO|Improved (now Gaussian/apodization); weak on ray-aim/measured files|
|F Detectors & analysis|Zemax|CODE V / QUADOA|Wins energy-ledger F8 + Stokes F7; now full on PSF/MTF/EE/spot/fan|
|G Materials & coatings|Zemax|OSLO|Weak on breadth; strong on TMM/birefringence (17 crystals)|
|H Scattering & stray light|**Split: Zemax (tools) / MWB (physics)**|—|Wins Mie H5 + volume H4|
|I Optimization|CODE V / Zemax|OSLO / QUADOA|None|
|J Tolerancing|CODE V|Zemax / OSLO / QUADOA|None|
|K Thermal / STOP|CODE V|Zemax / OSLO|None|
|L Illumination / photometry|Zemax|—|Wins photometric L1 (tie w/ Zemax); no illumination design|
|M Multi-config / zoom|QUADOA / Zemax|CODE V / OSLO|None (CLI sweep only)|
|N GUI/UX|3DOptix|QUADOA|Wins undo N3 + animation N8|
|O Coordinate system|QUADOA|Zemax / MWB|Wins snap-to-axis O8 + fold O9|
|P Data management|Zemax|MWB|Wins VC-friendliness P6 + CLI P5|
|Q Performance|3DOptix|MWB|Co-leader on GPU (Q2)|
|R Commercial|MWB|3DOptix / OSLO|Wins cost R1 + locality R2/R7|
|S Ultrafast / time-domain / nonlinear|**MWB**|—|Sole implementer — a whole axis no competitor here touches|

**Verdict.** Across the 19 axes: **Zemax leads or co-leads ~10** (A, B, D, E, F, G, H-tools, L, M, P), **CODE V ~3–4** (I, J, K, co-B), **QUADOA ~2** (M, O), **3DOptix ~3** (N, Q, catalog G8), **OSLO 0 sole leads** (co-leads B/E/G/I/J), and **MieWorkbench ~4** (C, R, S, H-physics; co-leads F/N/O/Q). **The overall winner of the general single-product feature set is Ansys Zemax OpticStudio (Premium), now narrowly over Keysight CODE V** — Zemax is the only package excellent across *both* sequential design *and* non-sequential/illumination in one product, whereas CODE V (which matches or beats Zemax on pure optimization/tolerancing) offloads non-sequential, illumination, and scatter to the *separate* LightTools product. **CODE V + LightTools together would rival or exceed Zemax's breadth; as single products, Zemax wins.**

**The honest nuance the tally hides:** MieWorkbench is a *specialist engine*, not a suite. On the specific axes it targets — **coherent non-sequential field fidelity, exact Mie/volume scattering, closed energy accounting, polarization/biaxial physics, ultrafast/time-domain/nonlinear modeling, GPU throughput, and cost/locality — it wins or ties the entire field, market leaders included.** It "loses overall" the way a precision interferometry-and-ultrafast bench "loses" to a full machine shop: fewer tools, but the ones it has are best-in-class, and several (the closed ledger, first-class Mie, biaxial, the whole time-domain axis) exist nowhere else here.

---

## 6. Head-to-head tables (X vs MieWorkbench)

Winner per line item. "MWB" = MieWorkbench wins; "tie" = parity. Ranges collapsed where a whole
block goes one way. Reflects the refreshed MWB column (post lowhanging/pulsed rounds).

### 6.1 Zemax OpticStudio (Premium) vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|Zemax|deeper NSC; both full|
|A2 Sequential design|Zemax|MWB has none|
|A3 MC transport|tie|both true photon MC|
|A4 Coherent absolute-power field|**MWB**|Zemax NSC coherent is a geometric ray-sum; MWB does a true RS gather inline|
|A5–A7 Splitting/BVH/CAD|Zemax|richer budgets + CAD solids (A6 tie: both accelerated)|
|B1 Interference|tie|MWB inline-default vs Zemax POP|
|B2–B11 POP/beamlet/PSF/MTF/Zernike/Strehl/partial-coh/Gaussian/interferogram|Zemax|full named suite + gridded POP; MWB now full on PSF/MTF/EE but source-referenced Zernike/Strehl, no POP/partial-coherence|
|C1 Jones|tie|
|C2 Stokes/DOP|**MWB**|Zemax ensemble-derived|
|C3 Mueller|Zemax|(workaround) vs MWB none|
|C4 Birefringence+walk-off|tie|both validated uniaxial|
|C5 Biaxial|**MWB**|MWB has a biaxial solver; Zemax has none|
|C6 Optical activity|tie|both none|
|C7–C8 TMM/polarizers|Zemax|larger catalog|
|C9 Grating efficiency|Zemax|RCWA rigorous > MWB closed-form|
|D1–D10 Surfaces/CAD|Zemax|GRIN, Q-type, freeform, CAD-optical import|
|E1,E6,E7 Source types/ray-aim/measured files|Zemax|~20 emitters, ray-aiming, IES/rayfile|
|E3 Coherent source|**MWB**|inline vs POP-pass|
|E2,E4,E5,E8 Spectral/Gaussian/apod/pol|tie|MWB now has Gaussian-M²+apodization|
|F1,F4,F5,F6 Irradiance/spot/fan/EE|tie|MWB now ships these|
|F2,F3,F9,F10 Color/curved/stray-light/image-sim|Zemax|
|F7 Stokes map|**MWB**|
|F8 Energy ledger|**MWB**|closed `<1e-3` audit|
|G1–G6 Materials/coatings|Zemax|breadth + dn/dT|
|G7 Birefringent lib|**MWB**|17 crystals incl. biaxial|
|H1–H3,H6 Scatter tools/BSDF/stray-light|Zemax|
|H4 Volume scatter|**MWB**|
|H5 Rigorous Mie|**MWB**|Wiscombe-validated; Zemax MSP-DLL only|
|I1–I5 Optimization|Zemax|MWB none|
|J1–J5 Tolerancing|Zemax|MWB none|
|K1 Thermal|Zemax|
|K2 STOP/FEA|tie|Zemax STAR is Enterprise, not Premium → both none at this tier|
|L1 Photometry|tie|both lux/lm/cd|
|L2–L3 Illumination/IES|Zemax|
|M1–M3 Multi-config|Zemax|
|N3 Undo · N8 Animation|**MWB**|granular undo; c/n tracer beads|
|N1,N5|tie|N2,N4,N6 Zemax|
|N7 Platform|tie|MWB Linux, Zemax Windows — an OS swap|
|O2,O3,O6,O7 Seq/ref/pickup/assembly|Zemax|
|O8 Snap-to-axis · O9 Fold|**MWB**|shipped auto-align + 1-click fold|
|O1,O4,O5|tie|
|P2–P4 CAD/API/macro|Zemax|
|P1,P5,P6|tie|MWB VC-friendly + CLI|
|Q1 CPU|tie|MWB C-OpenMP + workers|
|Q2 GPU|tie|MWB gather on CUDA; Zemax GPU-accelerates Huygens PSF/MTF only|
|Q3–Q5 Distributed/scale|Zemax|
|R1 Cost|**MWB**|
|R2,R7 Locality/ITAR|tie|R4,R5,R6 Zemax|
|S1–S4 Time-domain/nonlinear|**MWB**|Zemax has no time-domain/pulsed/NLO modeling|

**Zemax vs MieWorkbench overall winner: Zemax OpticStudio Premium.** MieWorkbench wins ~15 line
items — coherent-absolute field, Stokes maps, biaxial, the energy ledger, volume + first-class Mie,
birefringent library, granular undo, ray animation, snap-to-axis, the fold operator, cost, and the
entire four-line time-domain axis — a meaningful physics/ultrafast/cost cluster. But Zemax wins the
large majority on breadth (design, optimization, tolerancing, illumination, materials, interop).

### 6.2 CODE V (Keysight) vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|**MWB**|CODE V bounded prism/fiber surfaces only (real NSC→LightTools)|
|A2 Sequential design|CODE V|MWB none|
|A3 MC transport|**MWB**|CODE V deterministic; MC→LightTools|
|A4 Coherent absolute field|**MWB**|CODE V BSP is imaging-framed beamlet, not absolute-power NSC|
|A5 Ray splitting|tie|CODE V splits for ghosts|
|A6 BVH|**MWB**|C-engine TLAS; CODE V has none (moot for sequential)|
|A7 Analytic/mesh|CODE V|+ grid-sag; MWB analytic + incoherent mesh|
|B1 Interference|**MWB**|MWB coherent-default vs CODE V Hopkins-autocorrelation|
|B2 Free-space diffraction|tie|BSP vs RS gather|
|B3,B10 POP / BSP beamlet|CODE V|Beam Synthesis Propagation is best-in-class|
|B4,B5 PSF/MTF|tie|both full|
|B6,B7 Zernike/Strehl|CODE V|true exit-pupil vs MWB source-referenced|
|B8 Partial coherence|CODE V|IMS partial coherence; MWB none|
|B9 Gaussian analysis|CODE V|ABCD; MWB has source but no ABCD analysis|
|B11 Interferogram import|CODE V|measured-wavefront surface|
|C1 Jones|tie|
|C2 Stokes/DOP|**MWB**|CODE V has no Stokes/DOP map|
|C3 Mueller|tie|both none|
|C4 Birefringence+walk-off|tie|both uniaxial (CODE V via BSP e-rays)|
|C5 Biaxial|**MWB**|CODE V none|
|C7 TMM coatings|tie|
|C8 Polarizers|tie|
|C9 Grating efficiency|**MWB**|CODE V scalar (RCWA→RSoft); MWB closed-form efficiency|
|D2 Asphere|tie|
|D3,D4,D5,D8 Q-type/freeform/DOE/GRIN|CODE V|full analytic set|
|D6 Fresnel|tie|
|D7 Toroidal/biconic|tie|CODE V biconic via AAS|
|D9 Mesh-optical|**MWB**|CODE V mesh is mechanical-only|
|D10 CAD import|CODE V|STEP/IGES/SAT export/import|
|E1 Source types|tie|CODE V field-defs, MWB 3 geometric|
|E3 Coherent source|tie|both inline (BSP / gather)|
|E4,E5 Gaussian/apod|tie|
|E6 Ray-aiming|CODE V|auto reference rays|
|E7 Measured source files|tie|both none native (CODE V→LightTools)|
|E8 Pol state|tie|
|F1 Irradiance|tie|
|F2 Spectral/color|tie|MWB spectra vs CODE V spectral|
|F3 Curved detector|tie|both partial|
|F4,F5,F6 Spot/fan/EE|tie|both full|
|F7 Stokes map|**MWB**|CODE V pol-weighted only|
|F8 Energy ledger|**MWB**|CODE V per-surface T only|
|F9 Ghost|tie|CODE V GhoView vs MWB path ranking|
|F10 Image simulation|CODE V|IMS; MWB none|
|G1,G2,G3 Materials/dispersion/thermal|CODE V|catalogs + dn/dT|
|G4 TMM coatings|tie|
|G7 Birefringent lib|**MWB**|17 crystals incl biaxial vs CODE V unspecified|
|H1–H3,H6 Scatter/BSDF/stray-light|CODE V(→LightTools)|neither native-strong; CODE V has the LightTools path|
|H4 Volume scatter|**MWB**|CODE V none native|
|H5 Rigorous Mie|**MWB**|CODE V none native (→LightTools)|
|I1–I6 Optimization|CODE V|Global Synthesis + Glass Expert; MWB none|
|J1–J5 Tolerancing|CODE V|Wavefront Differential; MWB none|
|K1 Thermal|CODE V|MECo|
|K2 STOP/FEA|CODE V|SigFit bridge|
|L1 Photometry|**MWB**|MWB native lux/lm/cd; CODE V→LightTools|
|L2,L3 Illumination/IES|CODE V(→LightTools)|
|M1–M3 Multi-config|CODE V|MWB CLI-only|
|N1|tie|N2,N4 CODE V|
|N3 Undo|**MWB**|CODE V none found|
|N5 Live update|**MWB**|CODE V non-live (K2realm plugin)|
|N7 Platform|**MWB**|MWB Linux vs CODE V Windows-only|
|N8 Animation|**MWB**|
|O2,O6 Seq/pickup|CODE V|O4,O5,O9 tie/CODE V decenter-and-bend|
|O3 Relative chaining|tie|
|O7 Assemblies|tie|both weak (CODE V flat list)|
|O8 Snap-to-axis|**MWB**|
|O9 Fold|tie|MWB explicit operator vs CODE V D&B|
|P1 Format|**MWB**|MWB ZIP bundle vs CODE V loose files|
|P3 Scripting API|CODE V|Macro-PLUS + COM automation vs MWB CLI-only — CODE V by breadth|
|P4 Macro|CODE V|Macro-PLUS|
|P5 Headless CLI|tie|both batchable|
|P6 VC-friendly|tie|both ASCII sources|
|Q1 CPU|tie|
|Q2 GPU|**MWB**|CODE V no GPU|
|Q3–Q5|tie/CODE V|neither cloud|
|R1 Cost|**MWB**|CODE V historically the most expensive|
|R2 Locality|tie|R3 Platform **MWB** (Linux vs Win-only)|
|R4,R6 Ecosystem/support|CODE V|R7 ITAR tie|
|S1–S4 Time-domain/nonlinear|**MWB**|CODE V has none|

**CODE V vs MieWorkbench overall winner: CODE V** — it wins the design/optimization/tolerancing/
thermal/analysis bulk decisively (and its Global Synthesis + Wavefront Differential are genuinely
best-in-class). **But the split is clean and complementary:** MieWorkbench wins *every*
non-sequential/physical-transport line — non-sequential, MC transport, coherent-absolute, volume +
rigorous Mie scatter, grating efficiency, energy ledger, Stokes maps, biaxial, GPU, Linux/cost, and
the entire time-domain/nonlinear axis. These are precisely CODE V's non-goals (it defers them to
LightTools). The two are almost non-overlapping.

### 6.3 OSLO Premium vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential|**MWB**|OSLO groups are slow/surface-based (real NSC→TracePro)|
|A2 Sequential design|OSLO|MWB none|
|A3 MC transport|**MWB**|OSLO stochastic is a scripted utility|
|A4 Coherent absolute field|**MWB**|OSLO interferogram is single-beam|
|A5 Ray splitting|**MWB**|OSLO manual multiconfig|
|A6 BVH|**MWB**|OSLO none|
|A7 Analytic/mesh|tie|OSLO analytic; MWB analytic + mesh|
|B1 Interference|**MWB**|coherent-default vs single-beam|
|B2 Diffraction|tie|
|B3 POP/ABCD|OSLO|deep ABCD; MWB none|
|B4,B5 PSF/MTF|tie|both full|
|B6,B7 Zernike/Strehl|OSLO|true pupil vs MWB source-referenced|
|B8 Partial coherence|OSLO|projector module is a genuine strength; MWB none|
|B9 Gaussian analysis|OSLO|best-in-class cavity/fiber; MWB source only|
|C1 Jones|tie|
|C2 Stokes/DOP|**MWB**|OSLO per-ray only, no map|
|C4 Birefringence+walk-off|tie|both validated uniaxial|
|C5 Biaxial|**MWB**|OSLO explicitly excludes biaxial|
|C7 TMM coatings|tie|OSLO 31-layer|
|C8 Polarizers|tie|
|C9 Grating efficiency|**MWB**|OSLO scalar/extended-scalar, single order|
|D2 Asphere|tie|
|D3 Q-type|OSLO|(claimed; unverified) vs MWB none|
|D4 Freeform|OSLO|(claimed) |
|D5 DOE|tie|
|D6 Fresnel|**MWB**|OSLO approx facet-less|
|D7 Toroidal|tie|
|D8 GRIN|OSLO|10 gradient forms; MWB none|
|D9 Mesh-optical|**MWB**|OSLO no mesh|
|D10 CAD import|tie|OSLO export-only; MWB FreeCAD-mesh|
|E1 Source types|OSLO|extended/array/astig|
|E3 Coherent source|tie|
|E4 Gaussian source|OSLO|deep astigmatic; MWB has M² Gaussian|
|E5 Apod|tie|
|E6 Ray-aiming|OSLO|5 modes + telecentric; MWB none|
|E7 Measured files|tie|both none|
|E8 Pol state|tie|
|F1 Irradiance|**MWB**|OSLO single-field PSF, not scene irradiance|
|F2 Spectral/color|**MWB**|OSLO no color/photometric detector|
|F3 Curved detector|tie|both partial|
|F4,F5,F6 Spot/fan/EE|tie|both full|
|F7 Stokes map|**MWB**|OSLO x/y components only|
|F8 Energy ledger|**MWB**|OSLO none|
|F9 Ghost|tie|OSLO Narcissus/ghost cited|
|F10 Image sim|OSLO|(partial) vs MWB none|
|G1,G2,G3 Materials/dispersion/thermal|OSLO|3000+ Rx + dn/dT|
|G4 TMM|tie|OSLO 31-layer|
|G7 Birefringent lib|**MWB**|17 crystals vs OSLO 5|
|H1–H6 Scatter/stray-light|OSLO/MWB split|OSLO delegates to TracePro (⚠️/❌); MWB wins volume H4 + Mie H5 outright|
|I1–I5 Optimization|OSLO|DLS + ASA global; MWB none|
|J1–J4 Tolerancing|OSLO|ISO-10110 + full MC; MWB none|
|K1 Thermal|OSLO|athermalization|
|K2 STOP|tie|both none|
|L1 Photometry|**MWB**|MWB native lux/lm/cd; OSLO radiometric-only (photometry→TracePro)|
|L2,L3 Illumination/IES|tie/OSLO|both weak (→TracePro)|
|M1–M3 Multi-config|OSLO|multiconfig + zoom; MWB CLI-only|
|N1|tie|N2 OSLO|
|N3 Undo|**MWB**|OSLO has no undo/redo|
|N5 Live update|tie|OSLO slider-wheel vs MWB 1s auto-preview|
|N7 Platform|**MWB**|MWB Linux vs OSLO Windows-only|
|N8 Animation|**MWB**|MWB c/n beads vs OSLO Gaussian-beam movie|
|O2 Seq chaining|OSLO|O4,O5,O6 tie/OSLO|
|O3 Relative chaining|tie|
|O7 Assemblies|tie|both weak|
|O8 Snap-to-axis|**MWB**|OSLO none|
|O9 Fold|**MWB**|MWB operator vs OSLO `bend`|
|P1 Format|**MWB**|MWB ZIP bundle vs OSLO flat .len|
|P3 Scripting API|OSLO|CCL+SCP+DDE vs MWB CLI-only|
|P4 Macro|OSLO|CCL/SCP|
|P5 Headless CLI|**MWB**|OSLO DDE only, no batch CLI|
|P6 VC-friendly|tie|both ASCII|
|Q1 CPU|**MWB**|OSLO no multithread evidence|
|Q2 GPU|**MWB**|OSLO no GPU|
|Q3–Q5|**MWB**|OSLO single-user desktop|
|R1 Cost|tie|both have a free tier (MWB fully free; OSLO EDU capped at 10 surfaces)|
|R2 Locality|tie|R3 Platform **MWB** (Linux vs Win-only)|
|R4,R6 Ecosystem/support|OSLO|50-yr base + Altair|
|S1–S4 Time-domain/nonlinear|**MWB**|OSLO has none|

**OSLO vs MieWorkbench overall winner: OSLO Premium** on classical design breadth (optimization,
tolerancing, GRIN, Gaussian-beam/ABCD, partial coherence, the analysis suite, 3000+ prescriptions).
**MieWorkbench wins every non-sequential/volume-scatter/coherent-default/Stokes/biaxial line, the
energy ledger, GPU/multithread, Linux/cost, and the whole time-domain axis** — again OSLO's
non-goals (it hands non-sequential and scatter to TracePro). Note the near-tie on cost (both have a
free tier, but OSLO's EDU is 10-surface-capped while MieWorkbench is fully free and unlimited).

### 6.4 QUADOA Optical CAD vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential · A3 MC · A4 Coherent · A5 Splitting · A6 BVH|**MWB**|QUADOA has none of these (no NSC engine)|
|A2 Sequential · A7 Analytic/CAD|QUADOA|broader analytic + CAD import|
|B1,B2 Interference/diffraction|tie|
|B3–B11 POP/PSF/MTF/Zernike/Strehl/Gaussian|QUADOA|named suite (MWB source-referenced Zernike/Strehl, no POP/partial-coherence)|
|B4,B5,F4,F5,F6 PSF/MTF/spot/fan/EE|tie|MWB now ships these|
|C2 Stokes/DOP|tie|both first-class (QUADOA adds Poincaré)|
|C3 Mueller|QUADOA|(claim)|
|C4 Birefringence+walk-off|**MWB**|QUADOA unverified/no walk-off|
|C5 Biaxial|**MWB**|QUADOA none|
|C9 Grating efficiency|**MWB**|QUADOA scalar/efficiency-free|
|D3,D4,D8,D9,D10 Q-type/freeform/GRIN/CAD|QUADOA|
|E1,E4–E7 Sources|QUADOA|ray-aim/rayfile (MWB now Gaussian/apod)|
|E3 Coherent source|**MWB**|
|F2 Spectral detector|**MWB**|QUADOA has none|
|F7 Stokes map|tie|F8 Energy ledger **MWB**|
|F9 Ghost|tie|both ghost-capable|
|G1,G2,G3 Materials/dispersion/thermal|QUADOA|G7 Birefringent lib **MWB**|
|H4 Volume · H5 Mie|**MWB**|QUADOA none|
|I1–I5 Optimization · J1–J5 Tolerancing · K Thermal · M Multi-config|QUADOA|MWB none|
|L1 Photometry|**MWB**|MWB now lux/lm/cd; QUADOA none|
|N3 Undo|tie|both full (MWB more granular)|N8 Animation **MWB**|
|N7 Cross-platform|QUADOA|Win+Linux vs Linux-only|
|O2,O3,O4,O6,O7 Seq/ref/order/pickup/assembly|QUADOA|superset positioning|
|O8 Snap · O9 Fold|**MWB**|
|P2,P3,P4 CAD/API/macro|QUADOA|SDK|P5 Headless **MWB**|P6 VC-friendly **MWB**|
|Q2 GPU|**MWB**|QUADOA CPU-only|
|R1 Cost|**MWB**|R3 Platform QUADOA|
|S1–S4 Time-domain/nonlinear|**MWB**|QUADOA none|

**QUADOA vs MieWorkbench overall winner: QUADOA** — it wins the design/optimization/tolerancing/
analysis/surfaces/interop bulk. **But the split is unusually clean and complementary:**
MieWorkbench wins every non-sequential/physical-transport line, volume + rigorous Mie, grating
efficiency, energy ledger, birefringence + biaxial, GPU, cost, and the time-domain axis — precisely
QUADOA's stated non-goals. The two would be excellent *together*.

### 6.5 3DOptix vs MieWorkbench
| Line | Winner | Note |
|--|--|--|
|A1 Non-sequential · A5 Splitting|tie|both full NSC|
|A2 Sequential|3DOptix|claimed (MWB none)|
|A3 MC · A4 Coherent absolute|**MWB**|3DOptix MC not first-class; coherent approx+gated|
|A6 BVH/GPU|3DOptix|cloud GPU|A7 3DOptix (+STEP)|
|B1,B2 Interference/diffraction|**MWB**|3DOptix default is ray-approx; Huygens gated/narrow|
|B3 POP|3DOptix|Fresnel convolution (gated)|
|B4,B5 PSF/MTF|tie|both have geo products; MWB adds FFT-MTF, 3DOptix diffraction "coming soon"|
|B6,B7 Wavefront/Strehl|**MWB**|3DOptix none|
|C1,C2,C4,C7,C8,C9 Polarization/coatings/gratings|**MWB**|3DOptix shallow (linear-tag, 6 preset coatings, single-order gratings)|
|C5 Biaxial|**MWB**|
|D1,D2,D5,D7 Sph/asphere/DOE/toroid|tie|
|D6 Fresnel|**MWB**|D9,D10 Mesh/CAD 3DOptix (STEP)|
|E1,E4,E7 Source types/Gaussian/measured|3DOptix|IES/TM-25|
|E2,E3,E8 Spectral/coherent/pol source|**MWB**|3DOptix single-λ free tier, weak pol|
|F1 Irradiance|tie|F2 Spectral tie|
|F4,F6 Spot/EE|3DOptix|(MWB also has these now → really tie/MWB)|
|F5 Ray fans|**MWB**|3DOptix none|
|F7 Stokes · F8 Ledger|**MWB**|
|F9 Ghost/stray-light|3DOptix|purpose-built manual workflow|
|G1,G8 Catalog/vendor parts|3DOptix|~50k parts decisive|
|G4 Coating TMM · G7 Birefringent|**MWB**|3DOptix presets/none|
|H4 Volume · H5 Mie|**MWB**|H1 tie|H6 3DOptix|
|I,J,K optimization/tol/thermal|tie|both none (3DOptix SciPy-only)|
|L1 Photometry|**MWB**|MWB now lux/lm/cd; 3DOptix radiometric-only|
|N2,N6,N7,N9 Shaded/ease/platform/collab|3DOptix|browser + catalog + cloud|
|N3 Undo · N5 Live · N8 Animation|**MWB**|
|O3,O7 Reference/assemblies|3DOptix|LCS + cage hardware|O8 Snap · O9 Fold **MWB**|
|P2,P3 CAD/API|3DOptix|(beta, paywalled)|P5 Headless · P6 VC **MWB**|
|Q2,Q3,Q4,Q5 GPU/cloud/scale|3DOptix|elastic cloud|
|R1 Cost|**MWB**|full-feature free vs 3DOptix crippled free tier|
|R2,R7 Locality/ITAR|**MWB**|offline vs cloud-only|R3,R4,R5 3DOptix|
|S1–S4 Time-domain/nonlinear|**MWB**|3DOptix none|

**3DOptix vs MieWorkbench overall winner: 3DOptix, narrowly, on breadth — but the axis matters.**
By raw line-item count 3DOptix edges ahead (catalog, accessibility, CAD import, sources, cloud GPU,
stray-light UI, collaboration). **MieWorkbench wins the entire physics-depth cluster** (real wave
optics by default, full Jones/Stokes/biaxial polarization, TMM coatings, birefringence, grating
efficiency, rigorous Mie + volume scattering, the energy ledger, the time-domain axis) plus
locality/cost/privacy/VC/animation/undo/snap. For a **physics-first optical engineer** MieWorkbench
is the more capable *engine*; for a **lab engineer laying out catalog benchtop systems** 3DOptix is
the better *product*.

### 6.6 Overall winner of the "X vs MieWorkbench" comparisons
Across the five pairings, **the commercial/established tool wins the overall verdict in every case**
(Zemax decisively; CODE V, OSLO, and QUADOA on the design axis; 3DOptix narrowly on breadth). **The
single overall winner of the head-to-head set is Zemax OpticStudio Premium** — it beats MieWorkbench
on the most line items and is itself the general-feature-set winner, with **CODE V a very close
second** on the design axes. MieWorkbench's consistent, *non-overlapping* wins across all five
pairings — coherent-absolute non-sequential field, rigorous Mie + volume scattering, the closed
energy ledger, Stokes/biaxial polarization, the whole ultrafast/time-domain axis, GPU throughput,
and cost/locality/privacy — define exactly the moat to defend and the (design-shaped) gaps to weigh
in §7.

---

## 7. Gap-closing roadmap — every line MieWorkbench does not win

For each remaining gap: **① Ideal (best-in-class target)** and **② Pragmatic path** given this
architecture (FreeCAD worker + numpy/torch MC engine + C engine + PySide GUI), cross-referenced to
`future.md` seams and named modules. Effort tiers: **S** ≤1 wk · **M** ~1 mo · **L** ~1 quarter ·
**XL** multi-quarter/research.

> **Already delivered since the 2026-07-09 draft (no longer gaps — moved out of this section):**
> named analysis products **PSF, FFT-MTF, encircled/ensquared energy, spot diagrams, ray/OPD fans,
> source-referenced Zernike + Maréchal Strehl** (lowhanging round); **ghost/stray-light path
> ranking**; **photometric lux/lm/cd** and the **spectrometer** mode; **Gaussian-beam (M²) sources +
> apodization**; **glass-catalog import (168 materials, 41 Schott/Ohara)**; **biaxial crystals**;
> **measured ABg BSDF (BRDF-side)**; **multi-process `--workers` sharding**; the **fold operator +
> relative optical-train chaining**; **curved detectors (incoherent)**; and the compiled **C
> engine**. See `future.md` "Delivered" for flags/modules. The list below is what genuinely remains.

### 7.1 Optimization (I1–I6) — *the biggest categorical gap*
Now the single largest gap vs Zemax, CODE V, OSLO, and QUADOA (all four win the whole I block).
- **① Ideal:** merit-function editor with named operands (EFL, RMS spot/wavefront, MTF@freq,
  boundary/edge constraints), local (DLS/derivative-free) + global optimizers, glass substitution,
  multi-config awareness — with a **CODE V-style directed global synthesis** (I6) surfacing many
  distinct design forms as the stretch target.
- **② Pragmatic:** MieWorkbench's variables are FreeCAD spreadsheet parameters + placements
  (already swept by `permute_model.py`/`--var`). Wrap that as an **optimization loop**:
  scipy.optimize (least_squares/differential_evolution) or `nevergrad`/CMA over a merit function
  built from the shipped analysis products (spot RMS, encircled energy, detected power). Start with
  a headless `scripts/optimize.py`; add a GUI merit-function panel later. Mitigate the per-iteration
  FreeCAD **rebuild → extract → trace** cost with geometry caching and a **geometric-fast inner
  loop** (`coherent=false`, direct deposit), refining coherently at the end. Acceptance target: the
  `auto_designed_lens` demo (`future.md` §d). **Effort: headless optimizer L; GUI M; global XL.**

### 7.2 Tolerancing (J1–J5)
- **① Ideal:** sensitivity + inverse-sensitivity, Monte-Carlo with multiple perturbation
  distributions, chainable compensators, yield reporting — and, as the stretch goal, a **CODE
  V-style fast differential wavefront tolerancing** (J5) cheap enough to run inside the optimizer.
- **② Pragmatic:** reuse `--seeds` + `permute_model.py`; tolerancing is a structured sweep over
  perturbed placements/radii/index with statistics. Build `scripts/tolerance.py` that perturbs the
  model per a tolerance table, runs the geometric-fast pipeline N times, and aggregates a
  merit-metric distribution + sensitivity ranking. Compensators = a nested §7.1 optimize call per
  draw. A finite-difference wavefront-differential mode (perturb, re-evaluate the Zernike vector)
  approximates J5. Acceptance target: `tolerance_yield` demo. **Effort: sensitivity M; MC
  tolerancing L; compensators L; differential J5 L.**

### 7.3 True exit-pupil imaging analysis (B6/B7 upgrade, B8, F10)
The Zernike/Strehl products shipped with a **source-referenced pupil** (exact for the collimated/
laser benches this tracer models) — not a true exit pupil for finite-conjugate, field-point imaging.
- **① Ideal:** an exit-pupil / chief-ray search (reference sphere on a field point's image), which
  also unlocks a **PSF-peak-ratio Strehl**, **partial-coherence imaging** (B8), and **image
  simulation** (F10, convolve a scene through the system).
- **② Pragmatic:** add a chief-ray / reference-sphere stage (`raytracer/analysis.py`) cross-checking
  `--save-fields` vs `--export-rays`; partial coherence and image-sim then ride on the field
  products. Flagged in `future.md` (a). **Effort: exit-pupil/PSF-peak Strehl L; partial coherence L;
  image-sim M once the field pipeline exists.**

### 7.4 Surfaces & geometry (D3, D4, D8, D9, D10)
- **① Ideal:** analytic Q-type (Forbes) + XY/Zernike/Chebyshev freeform *with coherent phase*; GRIN;
  native STEP/IGES import as traceable optical surfaces.
- **② Pragmatic:**
  - *Q-type/freeform analytic:* extend `surfaces.py`'s asphere machinery (Newton-intersecting even
    polynomials, extract-verified `<1 µm`) to Forbes Qbfs/Qcon + XY/Zernike sag with analytic
    normals. **M–L.**
  - *GRIN:* genuinely new — curved-ray integration inside the bulk (Runge–Kutta on the eikonal)
    replacing straight segments in `tracer.py`. Every design suite here except 3DOptix has GRIN.
    **XL.**
  - *CAD import:* FreeCAD *already* imports STEP/IGES; expose "Import STEP as element" through the
    fc_server worker and canonicalize imported faces (falling back to the shipped incoherent mesh-BVH
    path). Acceptance target: `cad_import_scene` demo. **M for mesh-import; L for analytic-face
    recovery.**

### 7.5 Sources (E1, E6, E7)
- **① Ideal:** ~20 emitter types, ray-aiming to a stop, measured IES/TM-25/rayfile ingestion.
  (Gaussian-beam + apodization already landed.)
- **② Pragmatic:** `sources.py` already samples faces; add (a) **ray-aiming** by iterating emission
  direction to hit a named aperture body (M), (b) an **IES/TM-25/rayfile importer** producing
  weighted ray sets (M), (c) more emitter geometries (S–M). **Effort: ray-aiming M; source files M.**

### 7.6 Materials & coatings (G1–G3, G5, G8)
- **① Ideal:** thousands of catalog glasses, 13 dispersion formulas, dn/dT, coating synthesis, a real
  vendor-component catalog.
- **② Pragmatic:** the loader (`optprops.py`) already supports Sellmeier/Cauchy/constant/tabulated
  with mandatory citations. (a) **Import more public Schott/Ohara/CDGM AGF catalogs** into `.miemat`
  rows (mostly a data script) — closes most of G1/G2 (**S–M**). (b) **Add dn/dT** columns + a
  temperature parameter in `materials.py` (data already compiled in `library.md`) — also unlocks
  thermal K1 (**M**). (c) Coating *synthesis* (needle) is niche — defer. (d) A **vendor-component
  catalog** subset (curated Thorlabs/Edmund parametric primitives with real prescriptions) is a
  large, ongoing data effort (**L+**). **Effort: catalog import S–M; dn/dT M; vendor catalog L+.**

### 7.7 Physical-optics beamlet / POP (B3, B10) & partial coherence (B8)
- **① Ideal:** a gridded complex-field / beamlet propagator (Zemax POP / CODE V BSP class) for
  general free-space beam propagation and segmented/low-f/# apertures.
- **② Pragmatic:** MieWorkbench's Rayleigh–Sommerfeld gather already *is* a physical-optics
  propagator to a detector; a **beamlet/POP mode** would propagate a gridded field surface-to-surface
  rather than only gathering at the final screen — genuinely new plumbing on top of the existing
  gather kernel. **Effort: gridded POP L–XL; beamlet BSP XL.** Lower priority — the coherent gather
  covers most MieWorkbench use cases.

### 7.8 Thermal / STOP (K1, K2)
- **① Ideal:** dn/dT + CTE thermal model; FEA deformation import (CODE V does this via SigFit).
- **② Pragmatic:** dn/dT is shared with §7.6. Full STOP is XL; a pragmatic first step is **importing a
  deformed surface as a Grid-Sag/mesh** (reuses the mesh path) and a **SigFit-style Zernike/grid
  deformation reader**. **Effort: dn/dT M (via §7.6); deformation import L; coupled STOP XL.**

### 7.9 Scattering & stray light (H2, H3, H6)
- **① Ideal:** measured BSDF/ABg import (both BRDF and BTDF), importance sampling, a dedicated
  stray-light workflow with path ranking. (Ghost path ranking + BRDF ABg + scatter-to-target aim
  already landed.)
- **② Pragmatic:** (a) **BTDF (transmitted-side) ABg** beside the shipped BRDF sampler (`future.md`
  a) (**M**). (b) Broaden importance sampling (**M**). (c) A fuller **Path-Analysis-style stray-light
  report** on top of the shipped ghost ranking (**M**). MieWorkbench already *wins* the scatter
  physics (H4/H5); these close the tooling gap. **Effort: M each.**

### 7.10 Multi-configuration / zoom (M1–M3)
- **② Pragmatic:** the `--var` sweep + Variables dock already parameterize variants; wrap them as a
  **named-configuration table** in the GUI and let the run loop iterate configs, overlaying via
  `compare_runs.py`. **Effort: config-table GUI M; zoom/scan as config sequences M.**

### 7.11 Scripting API, CAD interop (P2–P4)
- **② Pragmatic:** (a) STEP/IGES import via §7.4; **export** of the FreeCAD model + traced rays to
  STEP/IGES is a fc_server op (**M**). (b) An **in-app Python console** bound to the `Project` session
  object — MieWorkbench is already Python; expose `core/project.py` in a console pane (**M**), which
  also answers the "no scripting API" gap that Zemax/OSLO/QUADOA win. A macro *language* is
  unnecessary given a Python console. **Effort: CAD export M; Python console M.**

### 7.12 Compute & scale (Q3–Q5)
- **② Pragmatic:** `--workers` multi-process sharding already landed; extend to **multi-GPU** (merge
  detector cubes/ledgers, all accumulators add linearly) (**L**). Cloud scale-out conflicts with the
  data-locality value proposition — treat as opt-in (**L+**). **Effort: multi-GPU L.**

### 7.13 UX, coordinate system, ecosystem (N/O/R lines)
- **Assemblies / grouping (O7):** a first-class nestable assembly object (QUADOA-style) on top of the
  existing `miewb_group`. **Effort: M.**
- **Ease of use / onboarding / cross-platform (N6, R5, R3):** ship a Windows/Mac build path (PySide6
  + VTK are cross-platform; the blocker is the FreeCAD/optics-env/ParaView tooling — bundle as an
  installer or container), plus a guided tutorial project. **Effort: cross-platform packaging L;
  onboarding M.**
- **Ecosystem/support (R4, R6):** documentation, a demo gallery (exists), and a community channel —
  ongoing, not a code task.

### 7.14 Ultrafast / nonlinear fidelity (S2–S4) — *deepen the axis MieWorkbench uniquely owns*
No competitor here has *any* time-domain modeling, so these are moat-widening, not gap-closing:
- **Depleted-pump / walk-off SHG, split-step NLSE, mid-train SPM, cascaded harmonics** — the
  `future.md` "Pulsed-optics round follow-ups" list has the exact seams. **Effort: split-step NLSE
  L–XL; depleted SHG M; exact-uniaxial SHG L (needs §7.15 exact Fresnel).**

### 7.15 Deliberate non-goals (document, don't chase)
- **RCWA (C9 rigor):** Zemax-only among these; MieWorkbench's closed-form grating models + measured
  tables cover most practical cases. XL research; defer unless sub-wavelength gratings become a target.
- **Mueller-matrix formalism (C3):** only QUADOA claims it (unconfirmed); Jones + Stokes maps cover
  MieWorkbench's needs. Defer.
- **Optical activity / gyrotropy (C6):** *no package here has it.* Pursue only for a specific research need.
- **Coating needle-synthesis (G5):** nobody here has true needle synthesis; users pair with Essential
  Macleod. Not worth chasing.
- **Cloud compute (Q4):** conflicts with the data-locality/ITAR value proposition; keep optional.
- **Biaxial conical refraction:** biaxial itself is a MieWorkbench-unique *win*; the conical-refraction
  corner case is documented as an honest limit (`future.md` b).

---

## 8. Priorities — MoSCoW (from the standpoint of the expert optical engineer)

Ranked by *impact on making MieWorkbench "best" per line item* × *leverage over existing code*.
Re-ranked after the lowhanging/pulsed rounds landed the analysis products, photometry, glass
catalogs, Gaussian sources, biaxial, ghost analysis, and the fold operator — so the former "Must
Have" analysis-product item is now **done**, and optimization/tolerancing rise to the top.

### 8.1 Must Have (the largest remaining categorical gaps; defines design credibility)
1. **Headless optimization loop** (§7.1) with a geometric-fast inner loop. *Why:* now the single
   biggest gap — all four design suites (Zemax/CODE V/OSLO/QUADOA) win the entire I block; even a
   scipy-based merit-function optimizer transforms MieWorkbench from "simulator" to "design-capable."
2. **Sensitivity + Monte-Carlo tolerancing** (§7.2). *Why:* pairs with optimization; reuses
   `--seeds`/`permute_model.py`; expected of any serious optical tool; CODE V's dominance here (J5)
   sets the bar.
3. **dn/dT + more glass catalogs** (§7.6a–b). *Why:* cheap (loader already supports Sellmeier),
   removes the last "❌" on thermal G3/K1, and 168→thousands closes the credibility gap on G1/G2.
4. **True exit-pupil / PSF-peak Strehl** (§7.3). *Why:* upgrades the shipped-but-source-referenced
   Zernike/Strehl (B6/B7) to real finite-conjugate imaging analysis — the honest asterisk on an
   otherwise-complete analysis suite.
5. **In-app Python console** bound to the `Project` API (§7.11b). *Why:* closes the scripting-API gap
   (Zemax ZOS-API / OSLO CCL / QUADOA SDK all win it) cheaply, since the app is already Python.

### 8.2 Should Have (high value, moderate effort; matches competitors on core design workflow)
6. **CAD (STEP/IGES) import as elements** (§7.4 CAD). *Why:* FreeCAD already imports these; exposing
   it removes a hard "❌" and enables real optomechanical/stray-light scenes.
7. **GRIN media** (§7.4 GRIN). *Why:* every design suite here except 3DOptix has it; needed for
   gradient-index and thermal-lensing work. (XL, but categorically important.)
8. **Analytic Q-type / XY-Zernike freeform** with coherent phase (§7.4). *Why:* extends the existing
   asphere machinery; matters for freeform/AR-VR; CODE V/Zemax/QUADOA all win D3/D4.
9. **Ray-aiming + measured source files** (§7.5). *Why:* proper stop-defined pupils and IES/rayfile
   ingestion; the last source gaps vs Zemax/OSLO/QUADOA.
10. **Config-table multi-configuration** (§7.10). *Why:* zoom/thermal/scan workflows; wraps existing
    sweep machinery; all four suites win M.
11. **BTDF scatter + fuller stray-light report** (§7.9). *Why:* MieWorkbench already wins the scatter
    *physics*; this closes the remaining tooling gap.

### 8.3 Might Be Useful (worthwhile but narrower or higher-effort)
12. **Partial-coherence imaging + image simulation** (§7.3, §7.7). *Why:* OSLO/CODE V/Zemax win B8/
    F10; rides on the exit-pupil field pipeline once it exists.
13. **Gridded POP / beamlet propagator** (§7.7). *Why:* Zemax POP / CODE V BSP class; the coherent
    gather already covers most cases, so lower priority.
14. **Multi-GPU gather** (§7.12). *Why:* pushes the scale ceiling further after multi-process.
15. **Nestable assemblies + cross-platform packaging** (§7.13). *Why:* usability/reach; blockers are
    the external tool stack, not the GUI.
16. **Vendor-component catalog subset** (§7.6d). *Why:* huge usability win (see 3DOptix) but a large,
    ongoing data effort; start curated.
17. **Deepen ultrafast/NLO fidelity** (§7.14). *Why:* widens a moat no competitor here contests;
    depth over breadth.

### 8.4 Not Really Important (defer or document as deliberate non-goals)
18. **RCWA gratings** (§7.15) — Zemax-only; closed-form models suffice for most cases; XL research.
19. **Mueller-matrix formalism** (§7.15) — only QUADOA claims it; Jones + Stokes cover the need.
20. **Optical activity / conical refraction** (§7.15) — no package here has optical activity; conical
    refraction is a documented corner-case limit of a MieWorkbench-unique win.
21. **Coating needle-synthesis** (§7.15) — nobody here has it; users pair with Essential Macleod.
22. **Native cloud compute** (§7.15) — conflicts with the data-locality/ITAR value proposition.
23. **A macro *language*** — redundant given an in-app Python console (§7.11).

---

## 9. Bottom line

MieWorkbench cannot and should not try to out-*breadth* Zemax OpticStudio or out-*design* CODE V.
Its defensible, already-winning moat is **physical-optics fidelity in a non-sequential engine, plus
an ultrafast/time-domain axis nobody else here has**: coherent-by-default field propagation with
absolute power, rigorous Mie + volume scattering, full Jones/Stokes polarization with validated
uniaxial *and biaxial* birefringence, a closed and auditable energy ledger, pulsed/spectrogram/
SHG/Kerr/GDD modeling, GPU + C-engine throughput, and zero-cost full-locality operation. The
lowhanging and pulsed rounds already *landed* the previously-conspicuous analysis-product gap
(PSF/MTF/EE/spot/fans/Zernike/Strehl, photometry, glass catalogs, ghost analysis, Gaussian sources,
the fold operator) — so the remaining **Must-Have** work is now squarely the *design* apparatus:
**optimization and tolerancing** (the one categorical gap all four design suites win), plus dn/dT,
a true exit pupil, and an in-app scripting console. Landing those would, for the first time, let
MieWorkbench credibly stand next to CODE V, Zemax, OSLO, and QUADOA as a *design* tool and not only
a best-in-class *physics-and-ultrafast simulation* engine.
