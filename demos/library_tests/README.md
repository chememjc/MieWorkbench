# Library validation test templates

Nine parametric scene templates for end-to-end validation of newly-added optical
properties (materials, coatings, filters, polarizers, gratings, uniaxial crystals,
detector QE curves). Each template has a sensible default optical property and a
configurable test wavelength; the sweep runner (`scripts/run_library_tests.py`)
iterates every newly-added registry entry against its matching template, swapping
the property and retuning the source wavelength, then examines the run's energy
closure and per-detector power to gate success.

## Templates (what each swaps)

| Scene | File | Swaps | Test body | Source | Detectors |
|-------|------|-------|-----------|--------|-----------|
| Transmissive | `mat_transmissive` | `material` | Window | Laser | Detector |
| Reflective @ 45° | `mat_metal_45` | `material` | Mirror | Laser | Detector |
| Uniaxial crystal | `crystal_waveplate` | `material` | Waveplate | Laser | Detector |
| AR coating @ normal | `coated_plate_0` | `coating` | Window | Laser | Detector |
| Dichroic @ 45° | `coated_plate_45` | `coating` | Splitter | Laser | det_r, det_t |
| Colored glass | `filter_plate` | `filter` | Filter | 3-λ source | Detector |
| Polarizer | `polarizer_plate` | `polarizer` | Polarizer | Laser | Detector |
| Grating | `grating_plate` | `grating` | Plate | Laser | Detector |
| LED source | `led_source` | `lambdac`/`lambdamin`/`lambdamax` (source preset) | — | LED | Detector |

Each `.FCStd` + `.MieWB` pair is pre-configured; the `.MieWB` carries ultra-quick
simparams (1e4 rays, 512² pixels, 5 spectral bins for most, 3 for filters/broadband).

## Building the templates

Commands below assume a one-time `scripts/setup_env.sh` and, per shell,
`source scripts/miewb_env.sh` (loads `miewb.env`, exports `MIEWB_INST_DIR`).

```bash
"$MIEWB_FREECAD" -c scripts/make_library_tests.py -- \
    --out demos/library_tests
```

Rebuilds all 9 `.FCStd` files from source (currently hand-authored; the `dim`
spreadsheets carry tunable parameters like aperture diameter, coating layer
thicknesses, etc., but geometry is fixed). Requires `scripts/primitivelib.py`
for primitive builder access. Timestamps the `.FCStd` files; `.MieWB` packs are
regenerated on demand by the sweep runner.

## Running the sweep

```bash
python3 scripts/run_library_tests.py [OPTIONS]
```

Exercises every new item in `new_items.json` (indexed by category and name) with
the matching template, generates `results.csv` and `RESULTS.md`, and reports
exit status (0 = all items passed; nonzero = some items failed).

### Options

- `--category {materials,coatings,filters,polarizers,gratings,uniaxial}` — test one
  category only (default: all)
- `--items materials:copper,filters:bp_405` — test specific rows (category:name pairs),
  repeatable
- `--jobs N` — partition the sweep across N parallel subprocess workers (default: 1,
  sequential)
- `--force` — re-run all items even if already present in results.csv
- `--keep-failures` — preserve `.MieWB`/`.MieSim` for failed items (default: purge
  to save disk space)

Example:

```bash
python3 scripts/run_library_tests.py --category filters --jobs 4
```

## Output artifacts

- **`results.csv`** — one row per item tested; columns: category, name, template,
  test_wavelength, status (PASSED/FAILED/TIMEOUT), exit_code, closure_ok,
  per_detector_power (mW). Rerunnable/restartable; already-tested items skipped
  unless `--force`.

- **`RESULTS.md`** — human-readable summary table mirroring results.csv, plus a
  "failures" section listing any rows with status != PASSED.

- **`SPOTCHECK.md`** — read-only audit trail: per-item checks (wavelength range,
  source power, closure gate threshold, detector presence). Regenerated on every run.

- **`new_items.json`** — the source truth: list of (category, name) tuples to test,
  indexed from `opticalproperties/*/` registry files at sweep start.

- **`generated/`** — scratch directory (gitignored); contains per-item workspace
  subdirectories, `.MieWB` archives, and `.MieSim` results. Cleaned on successful
  completion unless `--keep-failures` is set.

## Troubleshooting

- **"GatherError: undersampled"**: the test wavelength is outside the item's
  tabulated range (rare; a detector's QE table outside its supported band) or the
  source's spectral bounds don't cover the effective wavelength. Check `SPOTCHECK.md`
  for the actual λ range and verify the item's CSV row.

- **"Energy closure gate failed"**: the tracer's energy budget check failed at the
  end-of-run audit (§6 in docs/RAYTRACER.md). Check `case.json["audit"]` for the
  per-body ledger; most often a sign of an incorrect material n/k table or a grating
  efficiency sum exceeding 1.

- **"Detector sees 0 mW"**: verify the detector face's auto-pick resolved correctly
  (scene layout / detector orientation — see docs/RAYTRACER.md §5.9). Manually
  pin `detector_face` in the `.MieWB`'s simparams if needed.

---

For the full validation catalog and physics details, see `docs/RAYTRACER.md` §10
(test scene catalog) and §11 (validation invariants).
