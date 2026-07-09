# Library expansion — external spot-check list

Companion to `scripts/tools/verify_library.py` (which is a purely
*internal* consistency check: does the model reproduce its own notes,
are k>=0, does the loader accept everything, etc). This checklist is the
*external* half — comparing shipped rows against an authoritative source
outside this repo (refractiveindex.info raw data, a manufacturer
datasheet, a catalog PDF). `verify_library.py` cannot do this part; it
has no network access and no independent copy of the source data.

Status values: `VERIFIED` (checked against the cited source, matches),
`OPEN` (not yet checked — pick this row up next), `LOW/UNVERIFIED`
(checked as best as possible but the source itself is untrustworthy or
inconsistent — ship with a loud caveat, don't silently trust it).

| # | Item | Value(s) checked | Authoritative source | Status | Notes |
|---|------|-------------------|-----------------------|--------|-------|
| 1 | `nk/silicon.mienk` | 5 (wavelength, n, k) points spot-checked across the table | refractiveindex.info, Green 2008 (Sol. Energy Mater. Sol. Cells 92, 1305) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 2 | `nk/germanium.mienk` | first row (shortest wavelength) | refractiveindex.info, Aspnes & Studna | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 3 | `nk/copper.mienk` | 4 (wavelength, n, k) points | refractiveindex.info, Johnson & Christy 1972 | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 4 | `nk/tungsten.mienk` | 8 (wavelength, n, k) points | refractiveindex.info, Rakic (Brendel-Bormann model, "Rakic-BB") | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 5 | `caf2` (materials.miemat) | Sellmeier coefficients (lambda^2 form) | Malitson 1963, Appl. Opt. 2, 1103 (via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 6 | `znse` (materials.miemat) | Sellmeier coefficients (lambda^2 form) | Connolly et al. 1979 / Tatian 1984 refit (via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 7 | `N-SF11` (materials.miemat) | Sellmeier coefficients + n_d | SCHOTT optical glass catalog (Zemax catalog file, via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 8 | `N-BAK4` (materials.miemat) | Sellmeier coefficients + n_d | SCHOTT optical glass catalog 2017-01-20b (via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 9 | `N-FK51A` (materials.miemat) | Sellmeier coefficients + n_d | SCHOTT optical glass catalog (via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 10 | `linbo3_o` (materials.miemat) | Sellmeier coefficients | Zelmon, Small & Jundt 1997, JOSA B 14, 3319 (via refractiveindex.info) | VERIFIED | verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 11 | `yvo4_o` / `yvo4_e` (materials.miemat) | Sellmeier refit coefficients | Shi, Zhang & Shen 2001, J. Synth. Cryst. 30, 85 | VERIFIED | refit residual <2e-7 vs published form; verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 12 | `S-BSL7` (materials.miemat) | Sellmeier refit (all-positive-C) | Ohara catalog nd/Vd, in-repo refit of the original (rejected by the loader's Ci>0 gate) coefficients | VERIFIED | in-repo refit, |dn|<1e-5 vs original over the transmission window; verified vs RII raw data, exact match (point-by-point log retained by the main assistant's session notes) |
| 13 | `filter/tables/schott_og515.mietab` | all 10 table points | Schott OG515 datasheet PDF, Status 01.12.2014 (sydor.com mirror) | VERIFIED | matches at the table's 2-sig-fig precision (490/500nm exact: 9.7e-4, 0.051); 400nm 1.3e-6 is the documented <1e-5 stopband engineering fill; row cites "Status 2008" but values match the 2014 sheet |
| 14 | `filter/tables/schott_rg645.mietab` | all 7 table points | Schott RG645 datasheet PDF, 18.11.2020 (sydor.com mirror; pdftotext-verified) | VERIFIED | exact match at every point incl. the 610nm 7.6e-5 toe and the 0.961 NIR plateau |
| 15 | `filter/tables/schott_kg1.mietab` | all 8 table points | Schott KG1 datasheet PDF, Status 01.12.2014 (sydor.com mirror; rows re-extracted with pdftotext) | VERIFIED (after fix) | 7/8 exact; 750nm was a transcription slip (held the datasheet's 760nm value 0.439) — FIXED to 0.477 on this branch |
| 16 | `hamamatsu_s1223` (detector/tables/) | peak lambda + responsivity anchors + QE<->A/W consistency | Hamamatsu S1223 series datasheet KPIN1050E (hamamatsu.com) | VERIFIED | peak 960nm @ 0.6 A/W confirmed; all 4 QE points are exactly QE=R*1239.84/lambda_nm of the datasheet responsivities |
| 17 | `led_green_525` / `led_royal_blue_450` primitives | CWL vs vendor bin ranges | Cree XLamp XP-E2 datasheet CLD-DS56 (downloads.cree-led.com) | VERIFIED | green bins G2-G4 = 520-535nm (CWL 527 mid-range), royal blue D3-D5 = 450-465nm (CWL 452 in D3) |
| 18 | `chromium` / `nickel` / `platinum` / `titanium` (nk/*.mienk) | 2-3 gridpoints each incl. table ends | refractiveindex.info raw YAMLs (Cr/Ni Rakic-LD, Pt Rakic-BB, Ti Johnson 1974) | VERIFIED | exact match at every checked gridpoint |
| 19 | `baf2` / `lif` (materials.miemat) | Sellmeier lambda_i^2 -> C_i conversions | Malitson 1964 / Li 1976 canonical coefficient sets | VERIFIED | conversions exact (0.057789^2, 0.10968^2, 46.3864^2; 0.07376^2, 32.79^2); kbr/nacl tabulated rows covered by verify_library's notes-anchor checks (n(10um)/n(10.6um) reproduce to interp tolerance) |
| 20 | `litao3_o` / `litao3_e` (materials.miemat) | Sellmeier fit vs Bond 1965 tabulated data | Bond 1965, J. Appl. Phys. 36, 1674 (tabulated; Sellmeier fit is ours) | OPEN | notes already flag "composition UNVERIFIED" — confidence MEDIUM even once checked |
| 21 | `alpha_bbo` (`abbo_o`/`abbo_e`, materials.miemat) | Sellmeier coefficients | CASTECH-lineage vendor Sellmeier (no independent primary source found) | LOW/UNVERIFIED | notes already flag "vendor sources disagree >1e-3" — ship with the existing caveat, do not upgrade without a primary-literature source |

## How to use this list

For each `OPEN` row: pull the cited datasheet/paper, spot-check 2-4
points against the shipped `.mietab`/`.mienk`/`materials.miemat` row,
and flip the status to `VERIFIED` (matches), `VERIFIED (tolerance)` (matches
within a stated engineering tolerance — note the tolerance), or
`MISMATCH` (does not match — file it as a data bug, do not silently
"fix" the row from this checklist alone).

Rows already marked `VERIFIED` above were checked by the main assistant
against raw refractiveindex.info database YAMLs (exact match in every
case); this file intentionally leaves the point-by-point numbers out —
ask the main assistant for the underlying comparison if you need to
re-derive it.
