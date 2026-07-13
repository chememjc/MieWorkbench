# library_data/agf/ — Zemax AGF glass catalog sources

Raw Zemax/OpticStudio ".agf" ("ANSI Glass Format") glass-catalog files, consumed by
`scripts/tools/import_agf.py` to append Schott + Ohara optical glasses (including Schott
TIE-19 thermo-optic dn/dT data, where the catalog provides it) to
`opticalproperties/materials.miemat`. See that script's module docstring and
`--help` for the conversion contract, and `scripts/tools/verify_miemat_preserved.py`
for the guardrail that proves the import never altered a pre-existing row.

## Files and provenance

| File | Retrieved | Source | Notes |
|---|---|---|---|
| `schott.agf` | 2026-07-13 | [SCHOTT optical glass downloads](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads) — direct file: `https://media.schott.com/api/public/content/a79c07aa61da4c05a2c0bbab93d09a7f?v=3b65e351&download=true` ("Optical Glass – Overview Glass Types (ZEMAX format)", ZIP) | Fetched directly from SCHOTT's own download endpoint. Internal catalog header: `CC SCHOTT June 2025 preferred, inquiry, AR glasses`. 366 glass records, formula codes 1 (Schott) and 2 (Sellmeier 1) only. |
| `ohara.agf` | 2026-07-13 | [OHARA glass catalog page](https://oharacorp.com/glass-catalog/) (official). OHARA's site serves its Zemax catalog behind a Sucuri bot-firewall that blocked every automated fetch attempt from this environment (returns an HTML challenge page instead of the zip, even with a browser User-Agent + Referer). Retrieved instead from the actively-maintained community mirror **nzhagen/zemaxglass** (MIT-licensed compilation of vendor AGF files, "Reproduced here by permission of RadiantZemax"): `https://raw.githubusercontent.com/nzhagen/zemaxglass/master/src/ZemaxGlass/AGF_files/ohara.agf` | Internal catalog header: `CC Updated 18.10.19` (Ohara's own last-revision date inside the file, Oct 2019 — this mirror predates Schott's 2025 direct-download file above; a newer official Ohara AGF may exist but was not fetchable from here). 417 glass records, formula codes 1 (Schott) and 2 (Sellmeier 1) only. |

Both files are plain ASCII/CRLF text, licensed for redistribution by the respective
vendors to Zemax OpticStudio users (see each file's leading `CC`/`Re` comment line);
the community mirror explicitly carries "Reproduced here by permission of RadiantZemax
(www.radiantzemax.com)".

## Format

Each catalog is a sequence of records keyed by an `NM <name> <formula_code> <MIL#> <Nd>
<Vd> <exclude_sub> <status> [melt_freq]` line, followed by that glass's `GC` (comment),
`ED` (thermal expansion / **density** / dPgF), `CD` (dispersion coefficients), `TD`
(Schott TIE-19 thermo-optic coefficients, when the vendor supplies them), `OD`
(relative cost/environmental resistance codes), `LD` (transmission wavelength range,
um) and `IT` (internal transmittance table) lines. `scripts/tools/import_agf.py` parses
exactly the fields it needs (formula code + CD -> dispersion model params, ED -> density,
LD -> transmission range, TD -> thermo columns) and ignores the rest (GC/OD/IT/melt
codes are not part of the materials.miemat schema).

## Regenerating the import

```
python3 scripts/tools/import_agf.py library_data/agf/schott.agf library_data/agf/ohara.agf \
    --url schott.agf=https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads \
    --url ohara.agf=https://oharacorp.com/glass-catalog/ \
    --retrieved 2026-07-13 \
    --merge-into opticalproperties/materials.miemat
```

`--dry-run` prints the full parse/skip/merge report without writing anything.
`scripts/tools/verify_miemat_preserved.py` then proves every pre-existing row's
original 15 columns are byte-identical to `git show HEAD:opticalproperties/materials.miemat`.
