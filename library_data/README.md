# library_data/ — library additions, staged and merged

Companion to `library.md` (the full annotated catalog + citations). This directory used to hold
every drop-in row proposed there; the **[data-only]** rows have since been merged into the live
`opticalproperties/` registries by `scripts/tools/merge_library_data.py` (Workstream A of the
library-expansion round) and their source files were deleted from here. What's left in
`library_data/staged/` needs engine/loader work another workstream hasn't shipped yet, or is
in-flight source data for a workstream still in progress.

## What was merged (provenance)

Run once via `python3 scripts/tools/merge_library_data.py` (stdlib-only, system `python3`; safe to
re-run — it's idempotent and gracefully no-ops on the now-deleted sources). It appended:

| Registry | Rows appended | New total |
|--|--|--|
| `opticalproperties/materials.miemat` | 143 (of 144 proposed — `BAK4` dropped as a duplicate of `N-BAK4`, see below) | 168 |
| `opticalproperties/coating/coatings.miecoat` | 15 | 38 |
| `opticalproperties/filter/filters.miefilt` | 40 | 56 |
| `opticalproperties/polarizer/polarizers.miepol` | 12 | 17 |
| `opticalproperties/grating/gratings.miegrat` | 5 | 8 |
| `opticalproperties/birefringence/uniaxial.miebrf` | 10 | 13 |

Plus 13 `opticalproperties/nk/*.mienk` tabulated n/k files and 57 per-item `.mietab` tables split
out of the consolidated `*_tables.csv` sources. `BAK4` was dropped rather than kept (it duplicated
`N-BAK4`, the lead-free equivalent — the leaded variant's own Sellmeier fit wasn't sourceable);
`library.md` Sec.7's "169 materials" total didn't account for this drop, hence 168.
`scripts/raytracer/tests/test_opticalproperties.py::test_library_expansion_counts` pins these six
numbers against the shipped tree. `demos/library_tests/new_items.json` lists every name the merge
appended, per category, for downstream test-authoring.

## What's still staged (`library_data/staged/`) and why

| File | Proposed schema | Waits on |
|--|--|--|
| `birefringence_biaxial.mibiax` | `birefringence/biaxial.mibiax` (proposed) | **biaxial solver** — no loader or ray-tracing support for a third principal axis yet (`library.md` Sec.3.2, `lowhanging.md` Sec.4.1). The underlying materials (ktp/kta/lbo/bibo n_x/n_y/n_z + 5 mineral placeholders) are already live in `materials.miemat` as plain index rows — only the crystal-axis registry is blocked. |
| `emission_emitters.miesrc` + `emission_spectra.csv` | `emission/` (proposed: `emitters.miesrc` registry + `.miespec` tables) | **spectral-emission source** — sources today are geometric bodies with mono/uniform/Gaussian-band spectra; blackbody/solar/CIE-illuminant/discharge-lamp spectra need a new `sources.wavelength_strata()` branch and a loader (`library.md` Sec.4). |
| `detector_vlambda.csv` | proposed photometric detector property (CIE V(λ)) | **`--photometric` detector mode** — `detector.py` already has the CIE Ȳ table internally for sRGB; wiring a `spectral_cube_to_lux()` is unclaimed work (`library.md` Sec.5.1, `lowhanging.md` #2). |
| `detector_qe.csv` | proposed QE-weighted detector property (Hamamatsu S1223) | **QE-weighted detector mode** — another workstream ships the S1223 curve as part of that feature; kept here as reference data until it lands (`library.md` Sec.5.2). |

`emission_led_monochromatic.csv` (monochromatic LED CWL/FWHM presets — deep-red/red/amber/green/
blue/royal-blue/UV) stays at the top level of `library_data/`, **not** in `staged/`: unlike the
rest of `emission_*`, it needs no new engine support (it maps straight onto the existing Gaussian
source primitive) and is being consumed as source data by the in-flight LED-presets workstream.

## See also

`library.md` — the full annotated inventory (every group's citations, verification status,
taxonomy notes, and the complete Sec.7 manifest this directory used to mirror).
