# Web spot-check running log (main assistant)

Source of truth: raw YAML from polyanskiy/refractiveindex.info-database (GitHub).
Glass specs path: database/data/specs/schott/optical/<NAME>.yml (formula 2 =
interleaved B1,C1,B2,C2,B3,C3 — our .miemat rows use block order p1-3=B, p4-6=C).

| Item | Checked against | Result |
|---|---|---|
| silicon.mienk (250/370/390/550/1000 nm) | main/Si/nk/Green-2008.yml | PASS exact (all 5 points verbatim) |
| caf2 Sellmeier | main/CaF2/nk/Malitson.yml (formula 1, λᵢ form) | PASS: λᵢ²→Cᵢ conversion exact (0.050263605²=0.00252643; 0.1003909²=0.01007833; 34.64904²=1200.556); range 0.23–9.7 ✓ |
| linbo3_o Sellmeier | main/LiNbO3/nk/Zelmon-o.yml (formula 2) | PASS coefficients exact. NOTE: RII fit range 0.4–5.0 µm; our row's transmission window 0.35–5.2 (advisory only) — consider tightening to 0.4–5.0 in fix pass |
| N-SF11 | specs/schott/optical/N-SF11.yml | PASS exact (coeffs, nd 1.78472, Vd 25.68, 0.37–2.5) |
| N-BAK4 | specs/schott/optical/N-BAK4.yml | PASS exact (coeffs, nd 1.56883, Vd 55.98, 0.334–2.5) |
| N-FK51A | specs/schott/optical/N-FK51A.yml | PASS exact (coeffs, nd 1.48656, Vd 84.47, 0.29–2.5) |
| germanium.mienk first row | main/Ge/nk/Aspnes.yml | PASS (206.6 nm: 1.023/2.774 exact). Our table subsamples Aspnes' eV grid; deeper rows unverified yet |
| S-BSL7 | REFIT in-repo (original Ohara C2<0) | nd err 3.4e-9, Vd 64.15 vs 64.14, |dn|<1e-5 over 0.37–2.4 µm, 3.9e-5 at 0.29 µm edge; all C>0 |

| copper.mienk (187.9/548.6/616.8/1937 nm) | main/Cu/nk/Johnson.yml | PASS exact (4 points verbatim J&C) |
| tungsten.mienk (8 points incl. ends) | main/W/nk/Rakic-BB.yml | PASS exact — NOTE: always ask fetches for exact gridpoint rows; a first fetch misread a neighbor row as a "mismatch" |
| yvo4_o Sellmeier refit | main/YVO4/nk/Shi-o-20C.yml (formula 4, −D·λ²) | PASS: max|dn|=1.5e-7 over published 0.48–1.34 µm range; n(633)=1.99335 as claimed. NOTE row's transmission window 0.4–5.0 extends past fit range (advisory field; same pattern as linbo3) |

| yvo4_e Sellmeier refit | main/YVO4/nk/Shi-e-20C.yml | PASS: max|dn|=1.8e-7; n_e(633)=2.21628; Δn(633)=+0.223 ✓ |
| znse Sellmeier | main/ZnSe/nk/Connolly.yml (formula 1) | PASS: λᵢ²→Cᵢ exact (0.200859853²=0.04034468, 0.391371166²=0.15317139, 47.1362108²=2221.822); range 0.54–18.2 ✓; n(10.6µm)=2.4028 standard |

| schott_og515 filter table | Schott OG515 datasheet PDF (Status 01.12.2014, via sydor.com) | PASS: all 10 points match at 2 sig figs (490/500 exact: 9.7e-4, 0.051; 510 0.32 vs 0.322; 600 0.99 vs 0.987); 400nm 1.3e-6 = documented stopband engineering fill. NIT: row cites "Status 2008", current sheet is 01.12.2014 — same values |

| chromium.mienk (3 pts) | main/Cr/nk/Rakic-LD.yml | PASS exact |
| platinum.mienk (3 pts) | main/Pt/nk/Rakic-BB.yml | PASS exact |
| nickel.mienk (first row) | main/Ni/nk/Rakic-LD.yml | PASS exact (301.8nm row verbatim; our grid subsamples so mid-band gridpoints differ by design) |
| titanium.mienk (188nm + 617nm) | main/Ti/nk/Johnson.yml (J&C 1974) | PASS exact |
| baf2 Sellmeier | Malitson 1964 canonical coefficients | PASS: λᵢ²→Cᵢ conversions exact (0.057789²/0.10968²/46.3864²) |
| lif Sellmeier | Li 1976 canonical coefficients | PASS: 0.07376²=0.0054405, 32.79²=1075.18 |

| LED CWLs green_525 (527nm) + royal_blue_450 (452nm) | Cree XLamp XP-E2 CLD-DS56 datasheet (downloads.cree-led.com) | PASS: green bins G2-G4 = 520-535nm, royal blue D3-D5 = 450-465nm; both CWLs in-bin, doc reference genuine |

| schott_kg1 filter table | Schott KG1 datasheet PDF (Status 01.12.2014, sydor.com, pdftotext-verified rows) | 7/8 points exact (290/350/550/650/850/950/1100); **750nm was a transcription slip: 0.439 (the datasheet's 760nm value) instead of 0.477 — FIXED in schott_kg1.mietab** |

## Data fixes applied by main assistant (post-WS-G)
- mgf2 note: wavelength label corrected 550nm→589nm (computed n(589)=1.377717 matches the quoted 1.3777; pre-existing library typo).
- lbo_nx/lbo_ny: transmission_um_min raised 0.16→0.20 (Sellmeier fit poles at 0.1616/0.1502 µm made the declared UV edge numerically invalid; physical edge noted in row).
- zns_film note: now states the refit value 2.38589 and its 4e-4 residual vs the Amotchkina table honestly.
- verify_library.py: per-anchor tolerance now sig-fig-aware + 5e-4 allowance for tabulated interpolation. Full run: PASS exit 0 (168 materials / 129 anchors / 13 uniaxial / all tables).

## Still to check (after WS-G emits SPOTCHECK.md)
- ZnSe (Connolly/Tatian refit row — DERIVED, needs residual sanity vs main/ZnSe)
- YVO4 Shi o/e (row says FITTED from -D·λ² form, max err ~1e-7 — verify n_o(633)=1.9934 / n_e(633)=2.2163 vs published)
- gold/copper Johnson&Christy points; tungsten/chromium Rakić
- 3 Schott filter internal-transmittance points (OG515, RG645, KG1) vs Schott datasheet
- Hamamatsu S1223 responsivity/QE points (after WS-C ships the curve)
- 2 LED CWLs vs datasheets (green_525→527 nm CWL, royal_blue_450→452)
- lbo/ktp biaxial constants (staged, UNVERIFIED placeholders — verify or leave flagged)
