# Compare

`mieworkbench/panes/compare_pane.py` (`ComparePane`, dock).

## What it does

Wraps `scripts/compare_sweep.py` — the optics-env script that reads a
sweep manifest (or a plain list of case directories) and writes
`metrics.csv`, metric-vs-variable plots, a gallery, difference maps and
`summary.json` into an output directory. The pane never imports numpy/
h5py/matplotlib itself (the GUI venv ships numpy/scipy/h5py but no
matplotlib) — it only ever shells out to the optics-env python via
`QProcess` (mirroring `core/runner.py`/`core/raypreview.py`) and then
reads back `summary.json` + the PNGs `compare_sweep.py` already
rendered.

Metric columns tracked: `total_power_W`, `peak_irradiance_W_m2`,
`profile_visibility`, `centroid_x_mm`, `centroid_y_mm`,
`rms_spot_radius_mm`.

## How to use it

- After a sweep run, the GUI writes
  `results/<stem>/sweep-<case>.manifest.json`
  (`RunController.write_sweep_manifest`); load it here to compare every
  case in the sweep.
- **Add case…** appends an arbitrary finished case directory (dialog-free
  `add_case(case_dir)`) for ad-hoc comparisons outside a formal sweep —
  enables the "Compare N cases" button once at least one is added.
- **Compare** launches `compare_sweep.py`
  (`run_compare(manifest_path=None, case_dirs=None, ref=None,
  out_dir=None, python=None)`); results load automatically on finish.
- A slider scrubs through the sweep's cases; the gallery reuses
  `panes/results.py`'s `_Gallery` widget, so thumbnails support the same
  right-click export (Save image as… / Export CSV…) as the
  [Results](results.md) pane.

## Gotchas

- This pane requires a finished sweep or explicitly added finished cases
  — an empty/no-data state shows "Run a sweep to compare results, or add
  finished cases".
