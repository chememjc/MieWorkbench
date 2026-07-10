#!/usr/bin/env python
"""compare_sweep.py — sweep-comparison companion to compare_runs.py.

MUST run under the optics env (numpy + matplotlib + h5py):

    /home3/optics/env/bin/python scripts/compare_sweep.py \
        --manifest results/<model>/sweep-<case>.manifest.json \
        [--out results/comparisons/sweep_<model>_<case>] [--ref <stem>]

or, for ad-hoc comparison of arbitrary finished cases (no variable axis):

    /home3/optics/env/bin/python scripts/compare_sweep.py \
        --cases results/example/quick results/example/normal \
        [--out results/comparisons/<name>] [--ref <case-label>]

Manifest mode reads results/<model_stem>/sweep-<case>.manifest.json (written
by the GUI's sweep runner):
    {"model": "<stem>", "case": "quick", "mode": "product"|"zip",
     "order": ["miewb_vars.gap", "miewb_vars.tilt"],
     "variants": [{"stem": "<variant model stem>",
                   "values": {"miewb_vars.gap": 20.0, ...},
                   "case_dir": "<abs results/<variant_stem>/<case>>"}, ...]}

For every case directory (manifest variant or --cases entry) this reads
report.json (total_power_W, peak_irradiance_W_m2, profile_visibility per
detector) and every detectors/*.h5, from which it derives, per detector,
the irradiance-weighted spot centroid (centroid_x_mm/centroid_y_mm, in the
detector's own local grid basis: x_lo/y_lo + pixel index * pixel_m — NEVER
assume a fixed grid convention, always read the h5 attrs) and RMS spot
radius about that centroid (rms_spot_radius_mm). Negative zero-mean MC
noise in the stored cube is clipped to zero for this weighting only (sums
in report.json stay honest).

Writes into --out:
  metrics.csv           one row per (variant, detector): every scalar
                         metric + (manifest mode) the swept variable values.
  plot_<metric>_<detector>_vs_<var>.png
                         metric-vs-variable plots (manifest mode only, one
                         per detector x metric x varying variable; in
                         product mode with 2+ varying variables, one line
                         per combination of the OTHER varying variables).
  gallery/<variant>_<detector>.png
                         mean-cube irradiance map per variant x detector,
                         viridis, SAME color scale across variants for a
                         given detector (global vmax) so they compare
                         visually.
  diff/<variant>_<detector>.png
                         signed (variant - ref) irradiance, RdBu_r,
                         symmetric limits; skipped for the ref itself and
                         for any variant whose grid shape doesn't match
                         the ref's (warns, doesn't fail).
  summary.json           everything the GUI ComparePane needs (see
                         build_summary() below); every image path is
                         RELATIVE to --out.

Colours follow the Okabe-Ito CVD-safe palette (same order/assignment
convention as compare_runs.py).

Exits nonzero with a clear message if NO case has a report.json.
"""

import argparse
import csv
import json
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

# Okabe-Ito, fixed order by input/group position (never cycled within a
# plot) -- identical palette to compare_runs.py.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000"]

METRIC_LABELS = {
    "total_power_W": "total power [W]",
    "peak_irradiance_W_m2": "peak irradiance [W/m$^2$]",
    "profile_visibility": "fringe visibility",
    "centroid_x_mm": "centroid x [mm]",
    "centroid_y_mm": "centroid y [mm]",
    "rms_spot_radius_mm": "RMS spot radius [mm]",
}
PLOT_METRICS = list(METRIC_LABELS)
REPORT_METRICS = ("total_power_W", "peak_irradiance_W_m2",
                  "profile_visibility")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def safe(s):
    """filesystem-safe token for a filename fragment."""
    return str(s).replace(".", "_").replace("/", "_").replace(" ", "_")


def short_var(var):
    """'miewb_vars.gap' -> 'gap' for axis labels/legends (filenames keep
    the full dotted name via safe())."""
    return str(var).rsplit(".", 1)[-1]


def case_label(case_dir):
    """'<model>/<case>' — same convention as compare_runs.py, used as the
    variant "stem" in --cases mode (no variable axis)."""
    return "%s/%s" % (case_dir.parent.name, case_dir.name)


def values_label(variant, order=None):
    """Human string for a variant's swept values, or its case label when
    there are none (--cases mode)."""
    values = variant.get("values") or {}
    if not values:
        return variant["stem"]
    keys = [k for k in (order or values.keys()) if k in values]
    return ", ".join("%s=%g" % (short_var(k), values[k]) for k in keys)


# ---------------------------------------------------------------------------
# report.json / detector h5 loading
# ---------------------------------------------------------------------------
def read_report(case_dir):
    path = case_dir / "report.json"
    if not path.exists():
        print("[warn] no report.json in %s (run post_process.py first)"
              % case_dir, file=sys.stderr)
        return {}
    return json.loads(path.read_text())


def detector_h5_paths(case_dir):
    out = {}
    ddir = case_dir / "detectors"
    if not ddir.is_dir():
        return out
    for h5path in sorted(ddir.glob("*.h5")):
        with h5py.File(h5path, "r") as h:
            label = h.attrs["label"]
        out[label] = h5path
    return out


def load_grid(h5path):
    """Load a detector cube into the LOCAL grid basis (x_lo/y_lo + pixel
    index * pixel_m — the h5 attrs are the only source of truth for this,
    never assume a fixed convention). Returns a dict with the per-pixel
    irradiance (noise-clipped for display/weighting; report.json's sums
    stay the honest unbiased source) plus enough metadata to render an
    image and to centroid/rms it."""
    with h5py.File(h5path, "r") as h:
        cube = h["spectral_cube_mean"][...]
        mask = h["mask"][...]
        pixel_m = float(h.attrs["pixel_m"])
        x_lo = float(h.attrs["x_lo"])
        y_lo = float(h.attrs["y_lo"])
        H = int(h.attrs["H"])
        W = int(h.attrs["W"])
    power = cube.sum(axis=0)                       # W per pixel (unbiased)
    irr = np.maximum(power, 0.0) / (pixel_m ** 2)   # W/m^2, clipped
    return {"irr": irr, "mask": mask, "pixel_m": pixel_m,
            "x_lo": x_lo, "y_lo": y_lo, "H": H, "W": W}


def grid_metrics(grid):
    """Irradiance-weighted centroid + RMS spot radius, in mm, in the
    detector's own local grid basis. None if the detector saw no signal."""
    irr = grid["irr"]
    total = float(irr.sum())
    if total <= 0:
        return None
    H, W = irr.shape
    xs = grid["x_lo"] + (np.arange(W) + 0.5) * grid["pixel_m"]
    ys = grid["y_lo"] + (np.arange(H) + 0.5) * grid["pixel_m"]
    Xg, Yg = np.meshgrid(xs, ys)
    cx = float((irr * Xg).sum() / total)
    cy = float((irr * Yg).sum() / total)
    r2 = float((irr * ((Xg - cx) ** 2 + (Yg - cy) ** 2)).sum() / total)
    return {"centroid_x_mm": cx / 1e-3, "centroid_y_mm": cy / 1e-3,
            "rms_spot_radius_mm": np.sqrt(max(r2, 0.0)) / 1e-3}


def build_variant_record(stem, values, case_dir):
    """Read one case dir into {"stem", "values", "case_dir", "detectors":
    {label: metrics}, "_grids": {label: grid}, "_has_report": bool} — the
    two underscore keys are working state, stripped before JSON output."""
    case_dir = Path(case_dir)
    report = read_report(case_dir)
    h5paths = detector_h5_paths(case_dir)
    labels = sorted(set(report.get("detectors", {})) | set(h5paths))
    detectors = {}
    grids = {}
    for label in labels:
        metrics = {k: v for k, v in
                  report.get("detectors", {}).get(label, {}).items()
                  if k in REPORT_METRICS}
        h5path = h5paths.get(label)
        if h5path is not None:
            grid = load_grid(h5path)
            grids[label] = grid
            gm = grid_metrics(grid)
            if gm:
                metrics.update(gm)
        detectors[label] = metrics
    return {"stem": stem, "values": dict(values or {}),
            "case_dir": str(case_dir), "detectors": detectors,
            "_grids": grids,
            "_has_report": (case_dir / "report.json").exists()}


# ---------------------------------------------------------------------------
# metrics.csv
# ---------------------------------------------------------------------------
def write_metrics_csv(variants, order, out_dir):
    dest = out_dir / "metrics.csv"
    var_cols = list(order or [])
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "case_dir"] + var_cols +
                  ["detector"] + list(PLOT_METRICS))
        for v in variants:
            varvals = [v["values"].get(c, "") for c in var_cols]
            if not v["detectors"]:
                w.writerow([v["stem"], v["case_dir"]] + varvals +
                          ["-"] + [""] * len(PLOT_METRICS))
                continue
            for label, d in sorted(v["detectors"].items()):
                w.writerow([v["stem"], v["case_dir"]] + varvals +
                          [label] + [d.get(m, "") for m in PLOT_METRICS])
    return dest


# ---------------------------------------------------------------------------
# metric-vs-variable plots (manifest mode)
# ---------------------------------------------------------------------------
def compute_varying(variants, order):
    varying = []
    for var in order:
        vals = {v["values"].get(var) for v in variants
                if var in v["values"]}
        if len(vals) > 1:
            varying.append(var)
    return varying


def plot_metric_vs_var(label, metric, var, variants, mode, variables_varying,
                       out_dir):
    other_vars = [ov for ov in variables_varying if ov != var]
    grouped = mode == "product" and len(other_vars) > 0

    points = []   # (group_key, x, y)
    for v in variants:
        if var not in v["values"]:
            continue
        y = v["detectors"].get(label, {}).get(metric)
        if y is None:
            continue
        x = v["values"][var]
        gkey = tuple((ov, v["values"].get(ov)) for ov in other_vars) \
            if grouped else None
        points.append((gkey, x, y))
    if not points:
        return None

    groups = {}
    for gkey, x, y in points:
        groups.setdefault(gkey, []).append((x, y))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, gkey in enumerate(sorted(groups, key=repr)):
        pts = sorted(groups[gkey])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = PALETTE[i % len(PALETTE)]
        glabel = None
        if gkey:
            glabel = ", ".join("%s=%g" % (short_var(k), val)
                               for k, val in gkey)
        ax.plot(xs, ys, "o-", color=color, lw=1.6, markersize=4,
               label=glabel)
    ax.set_xlabel(short_var(var))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title("%s — %s vs %s" % (label, METRIC_LABELS.get(metric, metric),
                                    short_var(var)))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grouped and len(groups) > 1:
        ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    dest = out_dir / ("plot_%s_%s_vs_%s.png"
                      % (safe(metric), safe(label), safe(var)))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def make_metric_plots(variants, order, mode, out_dir):
    variables_varying = compute_varying(variants, order) if order else []
    if not variables_varying:
        return variables_varying, []
    all_labels = sorted({label for v in variants for label in v["detectors"]})
    made = []
    for var in variables_varying:
        for label in all_labels:
            for metric in PLOT_METRICS:
                p = plot_metric_vs_var(label, metric, var, variants, mode,
                                       variables_varying, out_dir)
                if p is not None:
                    made.append(p)
    return variables_varying, made


# ---------------------------------------------------------------------------
# gallery + difference maps
# ---------------------------------------------------------------------------
def render_gallery(variants, order, out_dir):
    root = out_dir / "gallery"
    all_labels = sorted({label for v in variants for label in v["detectors"]})
    summary = {}
    for label in all_labels:
        entries = [v for v in variants if label in v["_grids"]]
        if not entries:
            continue
        vmax = max(float(e["_grids"][label]["irr"].max()) for e in entries)
        if vmax <= 0:
            vmax = 1.0
        root.mkdir(parents=True, exist_ok=True)
        recs = []
        for v in entries:
            irr = v["_grids"][label]["irr"]
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(irr, origin="lower", cmap="viridis",
                          vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, label="W/m$^2$")
            vlabel = values_label(v, order)
            ax.set_title("%s\n%s" % (label, vlabel))
            fname = "%s_%s.png" % (safe(v["stem"]), safe(label))
            dest = root / fname
            fig.savefig(dest, dpi=130, bbox_inches="tight")
            plt.close(fig)
            recs.append({"stem": v["stem"], "values_label": vlabel,
                        "image": str(Path("gallery") / fname)})
        summary[label] = recs
    return summary


def render_diffs(variants, order, ref_stem, out_dir):
    root = out_dir / "diff"
    ref = next((v for v in variants if v["stem"] == ref_stem), None)
    if ref is None:
        return {}
    all_labels = sorted({label for v in variants for label in v["detectors"]})
    summary = {}
    for label in all_labels:
        ref_grid = ref["_grids"].get(label)
        if ref_grid is None:
            continue
        recs = []
        for v in variants:
            if v["stem"] == ref_stem:
                continue
            grid = v["_grids"].get(label)
            if grid is None:
                continue
            if grid["irr"].shape != ref_grid["irr"].shape:
                print("[warn] detector %r: %s grid shape %s != ref %s "
                      "%s -- skipping diff" % (label, v["stem"],
                                               grid["irr"].shape, ref_stem,
                                               ref_grid["irr"].shape),
                      file=sys.stderr)
                continue
            root.mkdir(parents=True, exist_ok=True)
            diff = grid["irr"] - ref_grid["irr"]
            vmax = float(np.abs(diff).max())
            if vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(diff, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, label="$\\Delta$ W/m$^2$")
            vlabel = values_label(v, order)
            ax.set_title("%s — %s minus ref (%s)" % (label, vlabel, ref_stem))
            fname = "%s_%s.png" % (safe(v["stem"]), safe(label))
            dest = root / fname
            fig.savefig(dest, dpi=130, bbox_inches="tight")
            plt.close(fig)
            recs.append({"stem": v["stem"], "values_label": vlabel,
                        "image": str(Path("diff") / fname)})
        if recs:
            summary[label] = recs
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def default_out_manifest(model, case):
    return common.RESULTS_DIR / "comparisons" / ("sweep_%s_%s" % (model, case))


def default_out_cases(names):
    joined = "_vs_".join(n.replace("/", "_") for n in names)
    if len(joined) > 120:
        joined = "%s_vs_%d_more" % (names[0].replace("/", "_"), len(names) - 1)
    return common.RESULTS_DIR / "comparisons" / joined


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", metavar="PATH",
                  help="results/<model>/sweep-<case>.manifest.json")
    g.add_argument("--cases", nargs="+", metavar="DIR",
                  help="arbitrary results/<model>/<case> directories "
                       "(no variable axis; like compare_runs.py)")
    p.add_argument("--out", default=None,
                  help="output directory (default depends on mode)")
    p.add_argument("--ref", default=None,
                  help="reference variant stem / case label (default: "
                       "the first variant/case)")
    return p.parse_args(argv)


def load_manifest_variants(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    model = manifest["model"]
    case = manifest["case"]
    mode = manifest.get("mode", "product")
    order = list(manifest.get("order", []))
    variants = [
        build_variant_record(v["stem"], v.get("values", {}), v["case_dir"])
        for v in manifest["variants"]
    ]
    return model, case, mode, order, variants


def load_case_variants(case_dirs):
    variants = []
    for c in case_dirs:
        c = Path(c).resolve()
        if not c.is_dir():
            raise SystemExit("compare_sweep.py: case directory not found: "
                             "%s" % c)
        variants.append(build_variant_record(case_label(c), {}, c))
    return variants


def resolve_ref(variants, ref):
    if ref is None:
        return variants[0]["stem"]
    for v in variants:
        if v["stem"] == ref:
            return v["stem"]
    for v in variants:
        if Path(v["case_dir"]).name == ref:
            return v["stem"]
    raise SystemExit("compare_sweep.py: --ref %r matches no variant/case "
                     "(have: %s)" % (ref, ", ".join(v["stem"]
                                                    for v in variants)))


def main(argv=None):
    args = parse_args(argv)

    if args.manifest:
        model, case, mode, order, variants = load_manifest_variants(
            args.manifest)
        out_dir = Path(args.out).resolve() if args.out \
            else default_out_manifest(model, case)
    else:
        model = case = None
        mode = "cases"
        order = []
        variants = load_case_variants(args.cases)
        names = [v["stem"] for v in variants]
        out_dir = Path(args.out).resolve() if args.out \
            else default_out_cases(names)

    if not variants:
        raise SystemExit("compare_sweep.py: no variants/cases to compare")
    if not any(v["_has_report"] for v in variants):
        raise SystemExit(
            "compare_sweep.py: no case has report.json -- run "
            "post_process.py on at least one case first")

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_stem = resolve_ref(variants, args.ref)

    print("compare_sweep.py: comparing %d variant(s) [%s mode], ref=%s"
          % (len(variants), mode, ref_stem))

    csv_path = write_metrics_csv(variants, order, out_dir)
    variables_varying, plot_paths = make_metric_plots(
        variants, order, mode, out_dir)
    gallery = render_gallery(variants, order, out_dir)
    diffs = render_diffs(variants, order, ref_stem, out_dir)

    clean_variants = [
        {"stem": v["stem"], "values": v["values"], "case_dir": v["case_dir"],
        "detectors": v["detectors"]}
        for v in variants
    ]
    summary = {
        "mode": mode, "model": model, "case": case, "ref": ref_stem,
        "order": order, "variables_varying": variables_varying,
        "variants": clean_variants,
        "plots": [str(p.relative_to(out_dir)) for p in plot_paths],
        "gallery": gallery, "diffs": diffs,
    }
    common.write_json(out_dir / "summary.json", summary)

    print("\nwrote:")
    print("  %s" % csv_path)
    for p in plot_paths:
        print("  %s" % p)
    print("  %s" % (out_dir / "summary.json"))

    print("\nsummary:")
    header = ["variant", "detector", "power_mW", "peak_irr_W/m2", "vis",
              "cx_mm", "cy_mm", "rms_mm"]
    rows = []
    for v in variants:
        if not v["detectors"]:
            rows.append([v["stem"], "-", "-", "-", "-", "-", "-", "-"])
            continue
        for label, d in sorted(v["detectors"].items()):
            def g(k, fmt="%.4g"):
                val = d.get(k)
                return fmt % val if val is not None else "-"
            p_mw = d.get("total_power_W")
            rows.append([
                v["stem"], label,
                "%.4g" % (p_mw * 1e3) if p_mw is not None else "-",
                g("peak_irradiance_W_m2"), g("profile_visibility", "%.3g"),
                g("centroid_x_mm"), g("centroid_y_mm"),
                g("rms_spot_radius_mm"),
            ])
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
