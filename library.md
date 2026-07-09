# library.md — what should be in the library that isn't

**Untracked working document.** An exhaustive, engine-support-flagged inventory of materials,
coatings, optical properties, emission sources, detector types, and primitive elements that a
complete MieWorkbench library should contain but currently lacks — with **drop-in-ready, cited
values wherever a public authoritative source exists.** Companion to `features.md`,
`lowhanging.md`, `demosystems.md`.

## How to read this document

- **Flags:** each missing item is tagged **[data-only]** (a pure library/data addition that works
  with the current engine) or **[needs engine: X]** (requires new physics/plumbing — cross-
  referenced to `lowhanging.md`/`features.md`).
- **Citations are mandatory.** Every value carries a `reference`. Sources are prioritized: **NIST >
  peer-reviewed publication > refractiveindex.info (with the underlying publication named) >
  manufacturer datasheet.** Values without an authoritative public source are flagged **UNVERIFIED
  — do not ship without checking** rather than invented.
- **Drop-in format:** proposed rows follow the exact loader schema (below). Bulk numeric data
  (Sellmeier coefficients, n/k tables, R/T curves) is collected in the companion
  `library_data/*.csv` files (also untracked); this document is the annotated catalog + priorities.
- **Verification status:** numeric coefficients transcribed from sources are error-prone; a
  verification pass spot-checks a sample against the cited source. Per-item confidence is noted.

## Loader schema (target formats — from `scripts/raytracer/optprops.py` / `materials.py`)

Every registry hard-requires a non-empty `reference`. Tables never extrapolate (hard error out of
range). New file types (biaxial, scatter, emission spectra) are **proposed** schemas that need a
loader (flagged [needs engine]).

| File | Columns | Notes |
|--|--|--|
|`materials.miemat`|`name,class,model,p1..p6,nk_file,density_kg_m3,transmission_um_min,transmission_um_max,notes,reference`|class∈{gas,glass,liquid,polymer,metal,oxide,film,special}; model∈{sellmeier,cauchy,constant,tabulated}. Sellmeier p1-3=B1-3, p4-6=C1-3 (n²=1+ΣBᵢλ²/(λ²−Cᵢ), λ µm, Cᵢ>0). Cauchy p1-3=A,B,C. Constant p1=n,p2=k.|
|`nk/*.mienk`|`wavelength_nm,n,k`|≥2 strictly-increasing rows|
|`coating/coatings.miecoat`|`name,layers,table,aoi_deg,reference`|one of `layers` (`mat:nm`/`mat:qw@λ0`, `;`-sep) or `table`→`tables/<name>.mietab` (`wavelength_nm,Rs,Rp,Ts,Tp`)|
|`polarizer/polarizers.miepol`|`name,type,table_csv,retardance_waves,reference`|type∈{linear,circular_left,circular_right}; table `wavelength_nm,T_parallel,T_perpendicular`|
|`filter/filters.miefilt`|`name,table_csv,ref_thickness_mm,reference`|table `wavelength_nm,transmittance_internal`|
|`grating/gratings.miegrat`|`name,model,lines_per_mm,params,table_csv,reference`|model∈{lamellar,bragg_kogelnik,dammann,table}|
|`birefringence/uniaxial.miebrf`|`name,n_o_material,n_e_material,reference,notes`|o/e reference materials rows|
|`diffuser/diffusers.miedif`|`name,grit,slope_rms,reference`| |
|**proposed** `birefringence/biaxial.mibiax`|`name,n_x_material,n_y_material,n_z_material,reference,notes`|**[needs engine: biaxial solver]**|
|**proposed** `scatter/bsdf.miebsdf`|`name,model(abg/table),params,table_csv,reference`|**[needs engine: BSDF sampler]**|
|**proposed** `sources/emitters.miesrc` + `emission/*.miespec`|`name,kind,table_csv,reference` + `wavelength_nm,relative_power`|**[needs engine: spectral-emission source]**|

## Current inventory recap (what already exists — do not duplicate)

From the live library (exact counts): **168 materials** (41 optical glasses, 17 metals/semiconductors/IR,
35 polymers/liquids/gases, 5 film materials, 46 crystal-axis rows) · **18 n/k tables** · **38 coatings**
(TMM stacks: AR at 550/633/1064nm, V-coats, W-coats, HR dielectric; measured tables: protected mirrors,
dichroic/laser elements, standard 45°-AOI) · **17 polarizers** (Glan variants, Polaroid sheets, wire-grids,
circular types) · **56 filters** (Schott colored-glass series, interference bandpass) · **8 gratings**
(lamellar, Bragg/VPH, Dammann, echelle, ruled blazed) · **13 uniaxial crystals** (calcite, quartz, sapphire,
LiNbO3, LiTaO3, YVO4, BBO isomers, KDP, ADP, rutile, TeO2, MgF2) · **1 detector QE curve** (hamamatsu_s1223)
· **62 primitives** (8 LED monochromatic sources). Every entry is cited. The **`lamellar`** grating model now
has its first registry entry; the **`cauchy`** dispersion model is exercised by polymer/liquid/gas materials.

Existing materials (sample): vacuum, air, bk7, fused_silica, sapphire_o/e, water, glass, polystyrene, latex,
pmma, polycarbonate, tio2, aluminum, gold, silver, mgf2, sio2_film, detector, calcite_o/e, quartz_o/e, sf5,
fiber_core_na22. Existing crystals: calcite, quartz, sapphire (uniaxial). LED presets: deep_red_660, red_630,
amber_590, green_525, blue_470, royal_blue_450, uv_365, uv_385.

## Accuracy & verification note (read before using any value)

Refractive-index coefficients, n/k tables, and coating curves collected here are transcribed from
external sources. **Transcription of numeric constants is error-prone.** Therefore: (1) every value
cites its exact source so it is independently verifiable; (2) a verification pass re-checks a
sampled subset (especially Sellmeier coefficient signs/magnitudes and biaxial three-axis data)
against the cited source; (3) any value lacking an authoritative public source is labelled
**UNVERIFIED** and must be checked before production use — it is not invented. Prefer regenerating
critical tables directly from the cited dataset (e.g. refractiveindex.info YAML, NIST ASD) over
hand-transcription where possible.

---

# 1. Materials

The current library ships **25 materials**. Below are **144 new cited, drop-in rows** across five
groups. All are **[data-only]** (work with the current engine) except where noted. Full rows in
`library_data/materials_*.miemat` + `library_data/nk/*.mienk`. **Sellmeier order is block
(p1-3=B, p4-6=C)** — verified against `materials.py:393`.

**1.1 Optical glasses — 41 [data-only]** (`materials_glasses.miemat`). Schott crowns/flints
(N-SF11/6/10/2/57, N-BK10, N-K5, N-KF9, N-BAK1/4, N-SK16/4, N-SSK8, N-BAF10/52, N-LAK22/9,
N-LAF21, N-LASF9, N-FK51A/58, N-PK52A, F2/5, K7, LLF1, classic leaded SF1/2/6/10/11/57), Ohara
equivalents (S-BSL7, S-TIM5/25/35, S-NBM51, S-LAH64/66), N-BK7HT. Every Sellmeier set reproduces
catalog nd to **<3e-6**. Source: SCHOTT/OHARA Zemax catalogs via refractiveindex.info. dn/dT
captured in notes (**[needs engine: thermo-optic]**). *Caveats:* `S-BSL7` has a genuine negative
C2 (may trip the `Cᵢ>0` check); `BAK4`≈`N-BAK4` (leaded variant unsourceable).

**1.2 Metals, semiconductors, IR windows — 17** (`materials_metals_semiconductors_ir.miemat` +
`nk/`). Metals (Cu, Cr, Ni, Pt, Ti, W — tabulated n/k, Johnson&Christy/Rakić); semiconductors
(Si, Ge, GaAs, SiC — tabulated, Green/Aspnes/Skauli/Larruquert); IR windows (CaF2, BaF2, ZnSe,
ZnS-multispectral, KBr, NaCl, LiF — Sellmeier or tabulated, Malitson/Li/Debenham). All HIGH
except amorphous SiC (MED). Ge/GaAs interband edges left untabulated (no fabricated bridge).

**1.3 Polymers, liquids, gases, biological — 35** (`materials_polymers_liquids_gases_bio.miemat`).
Polymers (COC/Zeonex, PDMS, SU-8, PVA, NOA61, + spot-value cellulose/nylon/PET/PTFE/epoxy);
liquids (ethanol, methanol, glycerol, acetone, toluene, benzene, isopropanol, CS2, immersion oil,
Cargille 1.40/1.60, seawater, blood plasma); gases (N2, O2, CO2, He, Ar); tissue (soft tissue,
blood, epidermis, cytoplasm, collagen, intralipid — `class=liquid`, no bio class). **These
exercise the currently-unused `cauchy` model.** HIGH for peer-reviewed fits; MED/LOW spot values
and placeholder densities flagged per row. Gases + benzene are DERIVED Cauchy re-fits (residuals
stated).

**1.4 Coating film materials — 5** (`materials_films.miemat`). HfO2 (cauchy, HIGH), ZnS-film
(sellmeier, HIGH), Ta2O5/Nb2O5/Al2O3-film (constant, MED) — needed for the TMM stacks in §2.

**1.5 Crystals — uniaxial + biaxial** (`materials_crystals.miemat`). **Uniaxial [data-only]:**
LiNbO3, LiTaO3, YVO4, β-BBO, α-BBO, KDP, ADP, rutile TiO2, TeO2, MgF2-e (10 o/e pairs). **Biaxial
[needs engine: biaxial solver]:** KTP, KTA, LBO, BiBO (x/y/z axes, HIGH) + 5 mineral placeholders
(muscovite, aragonite, topaz, α-sulfur, borax — **INCOMPLETE/UNVERIFIED** handbook constants).
13 HIGH; LiTaO3 MED; α-BBO LOW. *Absorbing-crystal caveat:* rutile below 0.43 µm and α-sulfur in
blue need Im(n) **[needs engine: absorbing-crystal k]**. Nonlinear crystals carry linear index
only **[needs engine: χ²]**. Birefringence registry entries in §3.

# 2. Coatings

Current library ships **23** (5 TMM + 18 table). **15 new rows** in `library_data/coatings.miecoat`
(+ `coating_tables.csv`), all **[data-only]**:
- **AR (TMM `layers`):** MgF2_633, MgF2_1064 (single-layer QW); Vcoat_532/633/1064 (2-layer
  quarter-quarter — approximates a commercial V-coat, flagged); BBAR_QHQ_550 (3-layer QHQ/W-coat,
  Willey recipe).
- **HR dielectric stacks (TMM):** HR_TiO2SiO2_532/633/1064 (11-layer H(LH)⁵, R≈99.9%,
  self-verified via Macleod admittance); HR_Ta2O5SiO2_1064 (15-layer, conventional NIR laser HR).
- **Protected metal mirrors (tables):** protected_aluminum_0, protected_silver_0, protected_gold_0
  — engineering approximations anchored to Thorlabs band-average specs (not digitized curves).
- **Dichroic/laser (tables):** dichroic_805sp_45 (shortpass, Thorlabs DMSP805);
  laser_mirror_1064_45 (CVI Nd:YAG line mirror).

*Correctness note:* the `layers` field is **outer→substrate**; the existing `BBAR_MgF2_SiO2` row's
comment appears to state the reverse — verify. New film materials it depends on are in §1.4.
**Not attempted:** true non-QWOT-optimized V-coat thicknesses; needle/flip-flop coating
**synthesis** (no package here has it — `features.md` §G5).

# 3. Optical properties

**3.1 Birefringence — uniaxial: 10 new [data-only]** (`birefringence_uniaxial.miebrf`): linbo3,
litao3, yvo4, bbo, alpha_bbo, kdp, adp, rutile, teo2, mgf2 (adds the MgF2 e-ray so MgF2 waveplates/
Rochon prisms work). Current library has 3 (calcite/quartz/sapphire).

**3.2 Birefringence — biaxial: 9 [needs engine: biaxial solver]** (`birefringence_biaxial.mibiax`,
proposed `n_x/n_y/n_z` schema): ktp, kta, lbo, bibo (HIGH) + muscovite, aragonite, topaz,
alpha_sulfur, borax (UNVERIFIED placeholders). See `lowhanging.md` §4.1 for the solver's difficulty.

**3.3 Filters — 40 new [data-only]** (`filters.miefilt` + `filter_tables.csv`; current library 16):
the full **Schott colored-glass series** with real internal-transmittance tables (OG515/550/570,
RG610/630/645/665/695/715/780/830/850 longpass; BG3/7/18/39/40 bandpass; GG375–495 near-UV
longpass; KG1/2/5 heat-absorbing; UG1/5/11 UV-bandpass; NG4/9 neutral-density) — **31 verified
verbatim** against Schott datasheets — plus 6 interference bandpass (bp_405/450/532/633/650/780).

**3.4 Polarizers — 12 new [data-only]** (`polarizers.miepol` + `polarizer_tables.csv`; current 5):
Glan-Thompson, Glan-Taylor, Polaroid HN22/HN38 dichroic sheet, Moxtek visible + KRS-5 IR wire-grid,
and circular polarizers at 488/633/780 nm (L+R). Spec anchors real; wavelength dependence modeled.

**3.5 Gratings — 5 new [data-only]** (`gratings.miegrat` + `grating_tables.csv`; current 3),
including the first **`lamellar` registry row** (model was supported but had no entry): lamellar_1200,
echelle_79, transmission_iof_cubes (binary UV), vph_eso_574 (Kogelnik VPH), ruled_1200_500.

**3.6 Still needing engine support (data notes only, no drop-in rows yet):**
- **GRIN profiles** (radial/axial/Luneburg) **[needs engine: GRIN curved-ray integration]** —
  `features.md` §7.4; would need `n0`, gradient coeff, pitch per element.
- **Stress-optic coefficients** (photoelastic C, per material) **[needs engine: stress
  birefringence]** — `lowhanging.md` §4.2.
- **Measured BSDF / ABg scatter** **[needs engine: BSDF sampler]** — a proposed `scatter/bsdf.miebsdf`
  schema (`name,model(abg/table),params,reference`); populate from goniophotometer/ABg fits.
- **Absorbing-crystal Im(n)** for rutile-UV / α-sulfur-blue **[needs engine: complex uniaxial index]**.
- **χ² nonlinear coefficients** for LiNbO3/BBO/KTP/LBO/BiBO **[needs engine: χ² frequency conversion]**.

# 4. Emission sources

Sources today are geometric bodies with mono/uniform/Gaussian-band spectra. Richer sources need a
**proposed `emission/` category** (`emitters.miesrc` registry + `.mietab` spectra, kind ∈
{blackbody, continuous, line}) and a new branch in `sources.wavelength_strata()` — the existing
equal-probability CDF-quantile sampler generalizes cleanly. Data in `library_data/emission_*`.

- **Blackbody/Planck** (analytic) **[needs engine: blackbody source]**: blackbody_2700/3200/5778;
  exact CODATA constants (NIST).
- **Solar** **[needs engine: spectral source]**: solar_am1.5g (ASTM G173, HIGH), solar_am0 (E490
  proxy, UNVERIFIED).
- **CIE illuminants** **[needs engine]**: cie_d65, cie_a, cie_e (CIE official CSVs; ship full 1nm).
- **White LED** **[needs engine]**: led_white_2733k (CIE 015:2018 LED-B1, grep-verified).
- **Discharge lamps** **[needs engine: line source]**: hg/na/ne/ar/kr/xe/d2 (NIST ASD lines;
  intensities order-of-magnitude only; Xe flattened to continuous in v1).
- **Monochromatic LEDs — [data-only], already supported!** (`emission_led_monochromatic.csv`):
  deep-red 660 / red 630 / amber 590 / green 525 / blue 470 / royal-blue 450 / UV 365/385 (nominal
  labels; exact CWLs in the file) — map to the existing Gaussian source (`lambdac=CWL`, bounds at
  `CWL±FWHM/2.3548` — note **not** ±FWHM/2).
  Real published CWL/FWHM (Cree/Lumileds/Nichia PDFs).
- **Laser lines — [data-only], already supported:** HeNe 632.8, Nd:YAG 1064/532, Ar-ion 488/514.5,
  diodes 405/450/635/650/780/808/980, CO2 10600, Ti:Sapph 700–1000 — just primitive presets.

# 5. Detector types

Today: only `detector_plane` (planar, wavelength-independent). Missing types, ranked, with the
`DetectorGrid` (`detector.py`) work each needs:
1. **Photometric (lux/lm/cd)** **[SUPPORTED-adjacent, near-zero work]** — the CIE Ȳ=V(λ) table
   **already exists** in `detector.py` (`_CIE_Y`, used for sRGB); add `spectral_cube_to_lux()` +
   `--photometric`. Data: `library_data/detector_vlambda.csv` (CIE 018:2019). See `lowhanging.md` #2.
2. **QE-weighted (photocurrent/ADU)** **[PARTIAL: qe_curve property + 1 multiply]** — data:
   `library_data/detector_qe.csv` (Hamamatsu S1223 primary-sourced; CMOS peaks UNVERIFIED).
3. **Polarimeter/Stokes** **[SUPPORTED — already shipped]** — `render_stokes_maps` renders S0–S3/DOP
   from `--save-fields`; just needs packaging as a named detector type. See `lowhanging.md` #1.
4. **Power/energy meter** and **bolometer** **[SUPPORTED — zero work]** — `total_power_W` already
   computed; the existing `detector_plane` *is* an idealized flat-response bolometer.
5. **Spectrometer (λ-vs-position)** **[SUPPORTED-adjacent]** — grating dispersion + spectral cube
   already exist; needs a λ(x) rendering mode.
6. **Far-field/goniometric (cd, I(θ,φ))** **[PARTIAL: post-processing angular histogram of ray
   dirs]** — no new geometry needed.
7. **Curved/spherical detector** **[needs engine: non-planar UV grid]** — incoherent path moderate
   (reuse Sphere/Cylinder `to_uv`), coherent gather needs per-pixel obliquity. `lowhanging.md` §5.
8. **Photon-counting (shot noise)** **[PARTIAL basic / NONE dead-time]** — Poisson-sample the power
   cube ÷ photon energy; dead-time/afterpulsing out of scope.

# 6. Primitives / elements

Current: 54 catalog elements. Missing element types, by build difficulty:

**Tier A — buildable now (catalog/wizard work only):**
- **Toroidal lens** — `Torus` surface class already exists in `surfaces.py`; needs a builder.
- **Aspheric mirror (general conic)** — reuse `Asphere` + `surface_override` (generalizes the
  fixed-parabola `mirror_parabolic`).
- **Best-form singlet, cemented triplet, cylindrical achromat** — wizard math on existing lens
  builders (`core/wizards.py`).
- **Transmission/VPH/echelle grating primitives** — pre-wire the existing `grating` models onto
  `grating_plate` (data in §3.5).
- **Integrating sphere, beam dump/knife-edge, depolarizer, light pipe/homogenizing rod, CPC** —
  reuse `absorbance`/`mirror`/`diffuser`/TIR + geometry (CPC needs edge-ray wizard math).
- **Optical isolator** — polarizer halves exist; needs the Faraday rotator (Tier C).

**Tier B — one new analytic surface / moderate feature:**
- **Off-axis parabola (OAP)** — **the single most-requested blocked bench primitive**; needs the
  asphere-sag verifier extended to recover the vertex from the *parent* paraboloid (vertex is
  outside the clear aperture). **[needs engine: verifier fix]**.
- **Powell lens** — existing asphere machinery but needs an aggressive high-order fit to pass the
  1 µm sag gate.
- **Microlens/lenslet (Shack–Hartmann) array, retroreflector sheet** — tiling/perf question, not
  physics (untested at hundreds-of-faces scale).

**Tier C — genuinely new physics [needs engine]:**
- **GRIN rod** (curved-ray bulk integration), **EOM/Pockels** (field-dependent index), **AOM**
  (moving/time-varying grating), **Faraday rotator** (magneto-optic), **SLM** (arbitrary per-pixel
  phase/amplitude mask), **single-mode / PM / photonic-crystal fiber** (modal solver — the ray
  model only covers multimode), **holographic diffuser** (engineered BSDF).

# 7. `library_data/` CSV manifest

All drop-in data lives in `library_data/` (untracked), mirroring the loader schemas, every row
cited. See `library_data/README.md` for the full file list, the load procedure (append registry
rows; split `*_tables.csv` into per-item `.mietab`), and the critical correctness notes (block
Sellmeier order; the S-BSL7 negative-C2 and lbo_ny negative-B1 valid-but-check cases; coating
`layers` outer→substrate order; class taxonomy).

**Totals landed:** 144 materials (41 glass · 17 metal/semiconductor/IR · 35 polymer/liquid/gas/bio
· 5 film · 46 crystal-axis rows) · 13 n/k tables · 10 uniaxial birefringence · 15 coatings
· 40 filters · 12 polarizers · 5 gratings (incl. first `lamellar` registry entry) · 8 LED presets
· QE detector data (1 curve + coverage metrics + `report.json` keys).

**Staged (need engine work):** 9 biaxial crystals (§3.2, `needs engine: biaxial solver`), emission
sources (blackbody/solar/LED spectral forms, needs spectral-source engine), curved detectors
(§5.7, `needs engine: non-planar UV grid`).

**Net library growth (landed):** materials 25→168, coatings 23→38, filters 16→56, polarizers 5→17,
gratings 3→8 (first `lamellar` entry), uniaxial crystals 3→13, n/k tables 5→18, detector QE curves
(1 entry + infrastructure), LED presets (8), photometric lux mapping, spectrometer λ-profiles.

## Highest-priority additions (start here)
1. **Optical glasses (§1.1)** and **CIE V(λ) photometric data (§5.1)** — both essentially free
   (glasses are a data import; V(λ) already sits in `detector.py`) and each closes a conspicuous gap.
2. **Metals/semiconductors/IR + colored-glass filters (§1.2, §3.3)** — high-value, all verified.
3. **Uniaxial crystals + MgF2 e-ray (§3.1)** — unlocks more waveplate/polarizer scenes now.
4. **Biaxial crystals (§3.2)** — the data is ready; pairs with the biaxial solver in `lowhanging.md` §4.1.

---

*Every value in `library_data/` carries a `reference`; UNVERIFIED items are flagged, not invented.
Numeric coefficients should be spot-checked against their cited source before production use.*
