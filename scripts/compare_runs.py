#!/usr/bin/env python
"""compare_runs.py — overlay the detector results of several finished
run_trace.py/post_process.py case directories and tabulate their headline
numbers.

MUST run under the optics env python (numpy + matplotlib + h5py):

    "$MIEWB_OPTICS_PYTHON" scripts/compare_runs.py \
        --cases results/example/quick results/example/normal \
        [--out results/comparisons/<name>]

For each case directory this reads report.json (headline numbers) and every
detectors/*.h5 (the spectral cube + grid metadata run_trace.py wrote), then
writes, per detector LABEL present in any of the cases:
  * profile_<label>.png   — horizontal irradiance profile through the peak
                            pixel, one curve per case
  * spectrum_<label>.png  — per-bin detected power spectrum, one curve per
                            case
plus a single compare.csv (case, detector, total_power_W, peak_irradiance,
profile_visibility) and a printed summary table.

A case missing a given detector label is simply skipped for that label's
plots (e.g. comparing runs against models with different detector sets).

Colours follow the Okabe-Ito CVD-safe palette, assigned by input position
(--cases order) so a case keeps the same colour across every plot; legend
labels are "<model>/<case>" (the case dir's parent + own name), which stays
unique even when two different models happen to share a case name.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import h5py                              # noqa: E402
import json                              # noqa: E402

# Okabe-Ito, fixed order by input position (never cycled within a plot).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000"]


def case_label(case_dir):
    """'<model>/<case>' display name — unique even across models that
    happen to share a case name."""
    return "%s/%s" % (case_dir.parent.name, case_dir.name)


def read_report(case_dir):
    path = case_dir / "report.json"
    if not path.exists():
        print("[warn] no report.json in %s (run post_process.py first)"
              % case_dir, file=sys.stderr)
        return {}
    return json.loads(path.read_text())


def detector_h5_paths(case_dir):
    """{label: Path} for every results/<case>/detectors/*.h5 file."""
    out = {}
    ddir = case_dir / "detectors"
    if not ddir.is_dir():
        return out
    for h5path in sorted(ddir.glob("*.h5")):
        with h5py.File(h5path, "r") as h:
            label = h.attrs["label"]
        out[label] = h5path
    return out


def load_profile(h5path):
    """(x_mm array, irradiance row through the peak pixel, peak row index)."""
    with h5py.File(h5path, "r") as h:
        cube = h["spectral_cube_mean"][...]
        pixel_m = float(h.attrs["pixel_m"])
    irr = cube.sum(axis=0) / (pixel_m ** 2)
    if not np.any(irr > 0):
        return None
    iy, ix = np.unravel_index(np.argmax(irr), irr.shape)
    xmm = (np.arange(irr.shape[1]) + 0.5) * pixel_m / 1e-3
    return xmm, irr[iy], iy


def load_spectrum(h5path):
    """(wavelength centers in nm, detected power per bin in mW)."""
    with h5py.File(h5path, "r") as h:
        cube = h["spectral_cube_mean"][...]
        lam_lo, lam_hi = float(h.attrs["lam_lo_m"]), float(h.attrs["lam_hi_m"])
    bins = cube.shape[0]
    lam_c_nm = (lam_lo + (np.arange(bins) + 0.5)
               * (lam_hi - lam_lo) / bins) / 1e-9
    pw_mw = cube.reshape(bins, -1).sum(axis=1) * 1e3
    return lam_c_nm, pw_mw


def plot_profiles(label, entries, out_dir):
    """entries: list of (case_label, color, h5path)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    any_plotted = False
    for name, color, h5path in entries:
        prof = load_profile(h5path)
        if prof is None:
            print("[warn] %s: detector %r has no signal, skipping profile"
                  % (name, label))
            continue
        xmm, row, iy = prof
        ax.plot(xmm, row, color=color, lw=1.6, label=name)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("irradiance [W/m$^2$]")
    ax.set_title("%s — horizontal profile through peak" % label)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    dest = out_dir / ("profile_%s.png" % label.replace(".", "_"))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_spectra(label, entries, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    any_plotted = False
    for name, color, h5path in entries:
        lam_nm, pw_mw = load_spectrum(h5path)
        ax.plot(lam_nm, pw_mw, color=color, lw=1.6, marker="o",
               markersize=3, label=name)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("detected power [mW]")
    ax.set_title("%s — detected spectrum" % label)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    dest = out_dir / ("spectrum_%s.png" % label.replace(".", "_"))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def write_compare_csv(cases, reports, out_dir):
    dest = out_dir / "compare.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "detector", "total_power_W", "peak_irradiance",
                    "profile_visibility"])
        for case_dir, report in zip(cases, reports):
            name = case_label(case_dir)
            dets = report.get("detectors", {})
            if not dets:
                w.writerow([name, "-", "", "", ""])
                continue
            for label, d in sorted(dets.items()):
                w.writerow([name, label, d.get("total_power_W", ""),
                           d.get("peak_irradiance_W_m2", ""),
                           d.get("profile_visibility", "")])
    return dest


def default_out(names):
    joined = "_vs_".join(n.replace("/", "_") for n in names)
    if len(joined) > 120:
        joined = "%s_vs_%d_more" % (names[0].replace("/", "_"), len(names) - 1)
    return common.RESULTS_DIR / "comparisons" / joined


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True, metavar="DIR",
                   help="results/<model>/<case> directories to overlay")
    p.add_argument("--out", default=None,
                   help="output directory (default: "
                        "results/comparisons/<case names>)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    cases = [Path(c).resolve() for c in args.cases]
    for c in cases:
        if not c.is_dir():
            raise SystemExit("compare_runs.py: case directory not found: %s"
                             % c)
    names = [case_label(c) for c in cases]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cases))]

    out_dir = Path(args.out).resolve() if args.out else default_out(names)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("compare_runs.py: comparing %d case(s): %s"
          % (len(cases), ", ".join(names)))

    reports = [read_report(c) for c in cases]
    det_paths = [detector_h5_paths(c) for c in cases]

    all_labels = sorted({label for dp in det_paths for label in dp})
    if not all_labels:
        print("[warn] no detectors/*.h5 found in any case — only "
              "compare.csv will be written")

    made = []
    for label in all_labels:
        entries = [(names[i], colors[i], det_paths[i][label])
                  for i in range(len(cases)) if label in det_paths[i]]
        p1 = plot_profiles(label, entries, out_dir)
        p2 = plot_spectra(label, entries, out_dir)
        made += [p for p in (p1, p2) if p is not None]

    csv_path = write_compare_csv(cases, reports, out_dir)

    print("\nwrote:")
    for m in made:
        print("  %s" % m)
    print("  %s" % csv_path)

    print("\nsummary:")
    header = ["case", "detector", "total_power_mW", "peak_irr_W/m2",
              "visibility"]
    rows = []
    for name, report in zip(names, reports):
        dets = report.get("detectors", {})
        if not dets:
            rows.append([name, "-", "-", "-", "-"])
            continue
        for label, d in sorted(dets.items()):
            p_mw = d.get("total_power_W")
            rows.append([
                name, label,
                "%.4g" % (p_mw * 1e3) if p_mw is not None else "-",
                "%.4g" % d["peak_irradiance_W_m2"]
                if d.get("peak_irradiance_W_m2") is not None else "-",
                "%.3g" % d["profile_visibility"]
                if d.get("profile_visibility") is not None else "-"])
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows
             else len(header[i]) for i in range(len(header))]

    def fmt(cols):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(header))
    for r in rows:
        print(fmt(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
