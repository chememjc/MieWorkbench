#!/usr/bin/env python3
# =============================================================================
# verify_library.py -- computational self-check of the MERGED
# opticalproperties/ library (Workstream G). Run under the optics-env
# interpreter (numpy/scipy are hard dependencies of the raytracer package):
#
#   /home3/optics/env/bin/python scripts/tools/verify_library.py [--root DIR]
#                                                                 [--json OUT]
#                                                                 [--tol 2e-4]
#
# This is a READ-ONLY report: it never touches opticalproperties/ data files.
# It loads the library through the exact same code path run_trace uses
# (raytracer.optprops.load_optical_properties) and then re-derives numeric
# checks the loader itself does not (and is not supposed to) perform:
#
#   1. Full-tree load succeeds (one call; the loader's own hard-validation
#      already covers structural/schema errors -- this just proves it runs
#      clean against the CURRENT merged files).
#   2. Every material's n_complex(lambda) is sampled at 50 points across its
#      evaluation range: no NaN/inf, k >= 0, class-aware n bounds.
#   3. "Notes-anchored" index checks: materials.csv and uniaxial.miebrf notes
#      frequently embed a hand-verified index value ("nd=1.51633",
#      "n_o(633)=2.2864", ...) that this script parses out and re-derives
#      from the model, PASS/FAIL per anchor, tolerance --tol (default 2e-4).
#   4. Uniaxial registry: sign of (n_e - n_o) vs. "positive/negative
#      uniaxial" language in the notes.
#   5. Table sanity re-asserts (filter/polarizer/coating/grating) cheap to
#      recompute even though the loaders already hard-validate them.
#   6. Human-readable report to stdout; exit nonzero on any FAIL. --json
#      writes a machine-readable summary alongside.
#
# Supported notes-anchor regex forms (documented here, not just in code):
#   n_d(<lam>[nm|um]) calc=<val>   e.g. "n_d(587.6nm) calc=1.51673"
#   n_d=<val>                      e.g. "n_d=1.5168"           (lambda=587.6nm)
#   nd_calc=<val>                  e.g. "nd_calc=1.784720"     (lambda=587.6nm)
#   nd=<val>                       e.g. "nd=1.78472"           (lambda=587.6nm)
#   n_o(<lam>[nm|um])=<val>        e.g. "n_o(633)=2.2864"
#   n_e(<lam>[nm|um])=<val>        e.g. "n_e(633)=2.2022"
#   n(<lam>[nm|um])=<val>          e.g. "n(587.6nm)=1.4585", "n(10um)=2.20066"
# When no unit is given, <lam> <= 20 is assumed micrometres, otherwise nm
# (matches every bare-number anchor actually in the library: "n(0.6)"= 0.6um,
# "n(1064)" = 1064nm, etc). "nd"/"n_d" forms with no explicit wavelength are
# always the Fraunhofer d-line, 587.6nm, by definition.
# =============================================================================
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from raytracer import optprops  # noqa: E402
from raytracer.materials import MaterialError  # noqa: E402

DEFAULT_ROOT = REPO / "opticalproperties"
DEFAULT_TOL = 2e-4
N_SAMPLES = 50
DEFAULT_PROBE_UM = (0.4, 0.7)


# ---------------------------------------------------------------------------
# notes-anchor regex (documented in the module docstring above)
# ---------------------------------------------------------------------------
NUM = r"[\d.]+"
UNIT = r"(nm|um)"
NOTES_PATTERN = re.compile(r"""
    n_d\(\s*(?P<ndlam_v>%(NUM)s)\s*(?P<ndlam_u>%(UNIT)s)?\s*\)\s*calc\s*=\s*(?P<ndlam_val>%(NUM)s)
  | \bn_d\s*=\s*(?P<nd2_val>%(NUM)s)
  | \bnd_calc\s*=\s*(?P<ndcalc_val>%(NUM)s)
  | \bnd\s*=\s*(?P<ndbare_val>%(NUM)s)
  | n_o\(\s*(?P<nolam_v>%(NUM)s)\s*(?P<nolam_u>%(UNIT)s)?\s*\)\s*=\s*(?P<no_val>%(NUM)s)
  | n_e\(\s*(?P<nelam_v>%(NUM)s)\s*(?P<nelam_u>%(UNIT)s)?\s*\)\s*=\s*(?P<ne_val>%(NUM)s)
  | (?<![a-zA-Z_])n\(\s*(?P<nlam_v>%(NUM)s)\s*(?P<nlam_u>%(UNIT)s)?\s*\)\s*=\s*(?P<n_val>%(NUM)s)
""" % {"NUM": NUM, "UNIT": UNIT}, re.VERBOSE)

# used only to flag notes that LOOK like they carry an index anchor but were
# not captured by NOTES_PATTERN above (item 3's "list unparsed ... for human
# eyes" requirement)
SNIFF_PATTERN = re.compile(r"\bnd[_=]|\bn_d\b|\bn_[oe]\s*\(|\bn\s*\(\s*[\d.]+")

FRAUNHOFER_D_UM = 0.5876  # 587.6 nm


def _lam_um(value_str, unit):
    v = float(value_str)
    if unit == "nm":
        return v / 1000.0
    if unit == "um":
        return v
    # no explicit unit: heuristic documented in the module docstring
    return v if v <= 20.0 else v / 1000.0


def _decimals(value_str):
    """Number of digits after the decimal point in the quoted anchor value
    (a 4-sig-fig quote like '2.260' cannot be held to a 2e-4 gate)."""
    s = value_str.lower().split("e")[0]
    return len(s.split(".")[1]) if "." in s else 0


def parse_notes_anchors(notes):
    """-> list of (label, lam_um, target_n, quoted_decimals)."""
    out = []
    for m in NOTES_PATTERN.finditer(notes or ""):
        gd = m.groupdict()
        if gd["ndlam_val"] is not None:
            out.append(("n_d(lam)calc", _lam_um(gd["ndlam_v"], gd["ndlam_u"]),
                        float(gd["ndlam_val"]), _decimals(gd["ndlam_val"])))
        elif gd["nd2_val"] is not None:
            out.append(("n_d", FRAUNHOFER_D_UM, float(gd["nd2_val"]),
                        _decimals(gd["nd2_val"])))
        elif gd["ndcalc_val"] is not None:
            out.append(("nd_calc", FRAUNHOFER_D_UM, float(gd["ndcalc_val"]),
                        _decimals(gd["ndcalc_val"])))
        elif gd["ndbare_val"] is not None:
            out.append(("nd", FRAUNHOFER_D_UM, float(gd["ndbare_val"]),
                        _decimals(gd["ndbare_val"])))
        elif gd["no_val"] is not None:
            out.append(("n_o(lam)", _lam_um(gd["nolam_v"], gd["nolam_u"]),
                        float(gd["no_val"]), _decimals(gd["no_val"])))
        elif gd["ne_val"] is not None:
            out.append(("n_e(lam)", _lam_um(gd["nelam_v"], gd["nelam_u"]),
                        float(gd["ne_val"]), _decimals(gd["ne_val"])))
        elif gd["n_val"] is not None:
            out.append(("n(lam)", _lam_um(gd["nlam_v"], gd["nlam_u"]),
                        float(gd["n_val"]), _decimals(gd["n_val"])))
    return out


# ---------------------------------------------------------------------------
# report container
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.load_ok = False
        self.load_error = None
        self.material_count = 0
        self.material_range_fails = []      # (name, detail)
        self.material_range_warns = []      # (name, detail)
        self.anchor_checks = []             # (source, name, label, lam_um, target, computed, diff, status)
        self.unparsed_notes = []            # (source, name, notes)
        self.uniaxial_checks = []           # (name, n_o, n_e, dn, expected, actual, status)
        self.table_fails = []               # (category, name, detail)
        self.table_checked = {"filters": 0, "polarizers": 0, "coatings": 0,
                              "gratings": 0}

    @property
    def n_anchor_pass(self):
        return sum(1 for c in self.anchor_checks if c[-1] == "PASS")

    @property
    def n_anchor_fail(self):
        return sum(1 for c in self.anchor_checks if c[-1] == "FAIL")

    @property
    def any_fail(self):
        return (not self.load_ok) or self.material_range_fails \
            or self.n_anchor_fail or self.table_fails \
            or any(c[-1] == "FAIL" for c in self.uniaxial_checks)


# ---------------------------------------------------------------------------
# item 2: per-material range sampling
# ---------------------------------------------------------------------------
def eval_range_for(mat):
    """-> (lo_um, hi_um) the evaluation range for item 2, or None to skip
    (nothing to probe)."""
    if mat.model == "tabulated":
        if mat.nk_lambda_um is None or len(mat.nk_lambda_um) < 2:
            return None
        return float(mat.nk_lambda_um[0]), float(mat.nk_lambda_um[-1])
    if mat.trans_min is not None and mat.trans_max is not None:
        return float(mat.trans_min), float(mat.trans_max)
    return DEFAULT_PROBE_UM


def check_material_range(mat, report):
    rng = eval_range_for(mat)
    if rng is None:
        report.material_range_fails.append(
            (mat.name, "tabulated but no usable nk table range"))
        return
    lo, hi = rng
    if not (hi > lo):
        report.material_range_fails.append(
            (mat.name, "evaluation range is empty/inverted [%.6g, %.6g] um" % (lo, hi)))
        return
    # nudge the sample grid a hair inside [lo, hi]: n_complex round-trips
    # um -> m -> um internally (*1e-6 then *1e6), and evaluating EXACTLY at
    # a tabulated boundary can lose the last ULP and trip the loader's
    # strict "outside tabulated range" guard for no physical reason. 1e-9
    # relative margin is ~1e7x the double-precision round-trip error and
    # negligible next to the range itself.
    margin = max(abs(lo), abs(hi)) * 1e-9
    lam_um = np.linspace(lo + margin, hi - margin, N_SAMPLES)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            nk = mat.n_complex(lam_um * 1e-6)
        except MaterialError as exc:
            report.material_range_fails.append(
                (mat.name, "n_complex raised over its own eval range: %s" % exc))
            return
    n, k = nk.real, nk.imag
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(k)):
        report.material_range_fails.append(
            (mat.name, "NaN/inf encountered over [%.6g, %.6g] um" % (lo, hi)))
        return
    if np.any(k < -1e-9):
        bad = float(k[k < -1e-9].min())
        report.material_range_fails.append(
            (mat.name, "k < 0 (min %.6g) over [%.6g, %.6g] um" % (bad, lo, hi)))
        return

    is_metal_or_tab = (mat.cls == "metal") or (mat.model == "tabulated")
    hard_lo, hard_hi = (0.0, 7.0) if is_metal_or_tab else (0.0, 15.0)
    if np.any(n <= hard_lo) or np.any(n >= hard_hi if is_metal_or_tab else n > hard_hi):
        bad = float(n[(n <= hard_lo) | (n >= hard_hi)][0]) if np.any((n <= hard_lo) | (n >= hard_hi)) \
            else float(n.max())
        report.material_range_fails.append(
            (mat.name, "n=%.6g outside hard class bound (0, %.6g) over [%.6g, %.6g] um"
             % (bad, hard_hi, lo, hi)))
        return
    if not is_metal_or_tab and (np.any(n < 1.0) or np.any(n > 6.5)):
        bad = n[(n < 1.0) | (n > 6.5)]
        report.material_range_warns.append(
            (mat.name, "n in [%.6g, %.6g] strays outside the soft [1.0, 6.5] "
             "band over [%.6g, %.6g] um" % (bad.min(), bad.max(), lo, hi)))


# ---------------------------------------------------------------------------
# item 3: notes-anchored checks (materials.csv self-anchored + uniaxial.miebrf
# anchored to the o/e sub-materials)
# ---------------------------------------------------------------------------
def check_anchor(report, source, name, label, lam_um, target, mat, tol,
                 decimals):
    try:
        computed = complex(mat.n_complex(lam_um * 1e-6)).real
    except MaterialError as exc:
        report.anchor_checks.append(
            (source, name, label, lam_um, target, float("nan"), float("nan"),
             "FAIL(raised: %s)" % exc))
        return
    diff = abs(computed - target)
    # A quoted anchor can only be checked to its own precision (half an ULP
    # of the last quoted digit, +20% slack for the source's own rounding),
    # and tabulated materials interpolate linearly between grid points, which
    # legitimately strays a few 1e-4 from the closed-form value in the note.
    tol_eff = max(tol, 0.6 * 10.0 ** (-decimals))
    if getattr(mat, "model", None) == "tabulated":
        tol_eff = max(tol_eff, 5e-4)
    status = "PASS" if diff <= tol_eff else "FAIL"
    report.anchor_checks.append(
        (source, name, label, lam_um, target, computed, diff, status))


def check_materials_notes(matdb, report, tol):
    for name in matdb.used_names():
        mat = matdb.get(name)
        anchors = parse_notes_anchors(mat.notes)
        for label, lam_um, target, decimals in anchors:
            check_anchor(report, "materials.csv", name, label, lam_um, target,
                        mat, tol, decimals)
        if not anchors and SNIFF_PATTERN.search(mat.notes or ""):
            report.unparsed_notes.append(("materials.csv", name, mat.notes))


def check_uniaxial_notes(props, report, tol):
    for name, entry in props.uniaxial.items():
        notes = entry.get("notes", "") or ""
        anchors = parse_notes_anchors(notes)
        matched_any = False
        for label, lam_um, target, decimals in anchors:
            if label == "n_o(lam)":
                matched_any = True
                check_anchor(report, "uniaxial.miebrf", name, label, lam_um,
                            target, entry["o"], tol, decimals)
            elif label == "n_e(lam)":
                matched_any = True
                check_anchor(report, "uniaxial.miebrf", name, label, lam_um,
                            target, entry["e"], tol, decimals)
            # bare n(lam)/nd forms in birefringence notes aren't unambiguously
            # o- or e-ray -- left for the SNIFF/unparsed bucket below.
        if not matched_any and SNIFF_PATTERN.search(notes):
            report.unparsed_notes.append(("uniaxial.miebrf", name, notes))


# ---------------------------------------------------------------------------
# item 4: uniaxial sign check
# ---------------------------------------------------------------------------
SIGN_WORDS = re.compile(r"\b(positive|negative)\s+uniaxial\b", re.IGNORECASE)


def eval_range_for_sign(mat):
    rng = eval_range_for(mat)
    return rng if rng is not None else DEFAULT_PROBE_UM


def check_uniaxial_signs(props, report):
    for name, entry in props.uniaxial.items():
        mat_o, mat_e = entry["o"], entry["e"]
        lo_o, hi_o = eval_range_for_sign(mat_o)
        lo_e, hi_e = eval_range_for_sign(mat_e)
        lo, hi = max(lo_o, lo_e), min(hi_o, hi_e)
        if not (hi > lo):
            report.uniaxial_checks.append(
                (name, None, None, None, None, None,
                 "FAIL(o/e evaluation ranges do not overlap: o=[%.4g,%.4g] "
                 "e=[%.4g,%.4g] um)" % (lo_o, hi_o, lo_e, hi_e)))
            continue
        lam_um = 0.5 * (lo + hi)
        try:
            n_o = complex(mat_o.n_complex(lam_um * 1e-6)).real
            n_e = complex(mat_e.n_complex(lam_um * 1e-6)).real
        except MaterialError as exc:
            report.uniaxial_checks.append(
                (name, None, None, lam_um, None, None,
                 "FAIL(raised: %s)" % exc))
            continue
        dn = n_e - n_o
        m = SIGN_WORDS.search(entry.get("notes", "") or "")
        if m is None:
            status = "SKIP(no positive/negative uniaxial tag in notes)"
        else:
            expected = m.group(1).lower()
            actual = "positive" if dn > 0 else ("negative" if dn < 0 else "zero")
            status = "PASS" if actual == expected else \
                "FAIL(expected %s, dn=%.6g at %.4g um)" % (expected, dn, lam_um)
        report.uniaxial_checks.append((name, n_o, n_e, lam_um, dn,
                                       (m.group(1).lower() if m else None),
                                       status))


# ---------------------------------------------------------------------------
# item 5: table sanity re-asserts
# ---------------------------------------------------------------------------
def check_filters(props, report):
    for name, spec in props.filters.items():
        report.table_checked["filters"] += 1
        T = np.exp(-spec["alpha_per_m"] * spec["ref_thickness_m"])
        if np.any(T > 1.0 + 1e-9) or np.any(T < 1e-7 - 1e-12):
            bad_lo = float(T.min())
            bad_hi = float(T.max())
            report.table_fails.append(
                ("filters", name,
                 "recovered transmittance range [%.3g, %.3g] outside "
                 "[1e-7, 1]" % (bad_lo, bad_hi)))


def check_polarizers(props, report):
    for name, spec in props.polarizers.items():
        report.table_checked["polarizers"] += 1
        if np.any(spec["T_par"] <= spec["T_perp"]):
            report.table_fails.append(
                ("polarizers", name, "T_parallel <= T_perpendicular somewhere"))


def check_coatings(props, report):
    for name, spec in props.coatings.items():
        if spec["kind"] != "table":
            continue
        report.table_checked["coatings"] += 1
        for pol in ("s", "p"):
            R, T = spec["R%s" % pol], spec["T%s" % pol]
            if np.any((R < 0) | (R > 1)) or np.any((T < 0) | (T > 1)):
                report.table_fails.append(
                    ("coatings", name, "R%s/T%s outside [0,1]" % (pol, pol)))
                continue
            if np.any(R + T > 1.0 + 1e-9):
                report.table_fails.append(
                    ("coatings", name,
                     "R%s+T%s exceeds 1 (max %.6g)" % (pol, pol,
                                                        float((R + T).max()))))


def check_gratings(props, report):
    for name, spec in props.gratings.items():
        if spec["model"] != "table" or spec["table"] is None:
            continue
        report.table_checked["gratings"] += 1
        for order, tab in spec["table"].items():
            for pol in ("eta_s", "eta_p"):
                arr = tab[pol]
                if np.any((arr < 0) | (arr > 1)):
                    report.table_fails.append(
                        ("gratings", name,
                         "order %d %s outside [0,1]" % (order, pol)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(root, tol):
    report = Report()
    try:
        props = optprops.load_optical_properties(root=root)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash
        report.load_error = "%s: %s" % (type(exc).__name__, exc)
        return report, None
    report.load_ok = True

    with warnings.catch_warnings():
        # materials.py's _warn_outside_transmission fires legitimately
        # whenever item 3 evaluates an anchor outside trans_min/max, or
        # item 2 samples right up to the boundary; suppress for readability
        # per the task spec (still a real UserWarning if run interactively).
        warnings.simplefilter("ignore")

        report.material_count = len(props.matdb)
        for name in props.matdb.used_names():
            check_material_range(props.matdb.get(name), report)

        check_materials_notes(props.matdb, report, tol)
        check_uniaxial_notes(props, report, tol)
        check_uniaxial_signs(props, report)

        check_filters(props, report)
        check_polarizers(props, report)
        check_coatings(props, report)
        check_gratings(props, report)

    return report, props


def print_report(report, root, tol):
    print("=" * 78)
    print("verify_library.py -- opticalproperties/ self-check")
    print("  root: %s" % root)
    print("  notes-anchor tolerance: %.1e" % tol)
    print("=" * 78)

    if not report.load_ok:
        print("\n[FAIL] load_optical_properties() raised: %s" % report.load_error)
        return
    print("\n[PASS] full-tree load OK (%d materials)" % report.material_count)

    # --- item 2 ---
    n_fail = len(report.material_range_fails)
    n_warn = len(report.material_range_warns)
    n_ok = report.material_count - n_fail
    print("\n--- material range sampling (%d points each) ---" % N_SAMPLES)
    print("  %d/%d materials clean, %d FAIL, %d WARN"
          % (n_ok, report.material_count, n_fail, n_warn))
    for name, detail in report.material_range_fails:
        print("  FAIL  %-24s %s" % (name, detail))
    for name, detail in report.material_range_warns:
        print("  WARN  %-24s %s" % (name, detail))

    # --- item 3 ---
    print("\n--- notes-anchored index checks ---")
    print("  %d anchors parsed: %d PASS, %d FAIL"
          % (len(report.anchor_checks), report.n_anchor_pass, report.n_anchor_fail))
    for source, name, label, lam_um, target, computed, diff, status in report.anchor_checks:
        if status != "PASS":
            print("  FAIL  [%s] %-20s %-14s lam=%.5g um  target=%.6g  "
                  "computed=%.6g  diff=%.3g  (%s)"
                  % (source, name, label, lam_um, target, computed, diff, status))
    if report.n_anchor_fail == 0:
        print("  (all anchors within tolerance)")
    if report.unparsed_notes:
        print("\n  %d row(s) have notes that LOOK like they carry an index "
              "anchor but weren't parsed (human review):" % len(report.unparsed_notes))
        for source, name, notes in report.unparsed_notes:
            print("    [%s] %-20s %s" % (source, name, notes[:160]))

    # --- item 4 ---
    n_uni_fail = sum(1 for c in report.uniaxial_checks if c[-1].startswith("FAIL"))
    n_uni_skip = sum(1 for c in report.uniaxial_checks if c[-1].startswith("SKIP"))
    n_uni_pass = len(report.uniaxial_checks) - n_uni_fail - n_uni_skip
    print("\n--- uniaxial sign checks (%d crystals) ---" % len(report.uniaxial_checks))
    print("  %d PASS, %d FAIL, %d SKIP (no sign tag in notes)"
          % (n_uni_pass, n_uni_fail, n_uni_skip))
    for name, n_o, n_e, lam_um, dn, expected, status in report.uniaxial_checks:
        if status != "PASS":
            if n_o is not None:
                print("  %-6s %-14s n_o=%.6g n_e=%.6g dn=%.6g @%.4gum  %s"
                      % (status.split("(")[0], name, n_o, n_e, dn, lam_um, status))
            else:
                print("  %-6s %-14s %s" % (status.split("(")[0], name, status))

    # --- item 5 ---
    print("\n--- table sanity re-asserts ---")
    print("  checked: %s" % ", ".join("%s=%d" % kv for kv in report.table_checked.items()))
    if report.table_fails:
        for cat, name, detail in report.table_fails:
            print("  FAIL  [%s] %-24s %s" % (cat, name, detail))
    else:
        print("  (all clean)")

    print("\n" + "=" * 78)
    if report.any_fail:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")
    print("=" * 78)


def write_json(report, out_path, root):
    def anchor_row(c):
        source, name, label, lam_um, target, computed, diff, status = c
        return {"source": source, "name": name, "label": label,
                "lam_um": lam_um, "target": target, "computed": computed,
                "diff": diff, "status": status}

    def uni_row(c):
        name, n_o, n_e, lam_um, dn, expected, status = c
        return {"name": name, "n_o": n_o, "n_e": n_e, "lam_um": lam_um,
                "dn": dn, "expected_sign": expected, "status": status}

    data = {
        "root": str(root),
        "load_ok": report.load_ok,
        "load_error": report.load_error,
        "material_count": report.material_count,
        "material_range_fails": [{"name": n, "detail": d}
                                 for n, d in report.material_range_fails],
        "material_range_warns": [{"name": n, "detail": d}
                                 for n, d in report.material_range_warns],
        "anchor_checks": [anchor_row(c) for c in report.anchor_checks],
        "anchor_pass": report.n_anchor_pass,
        "anchor_fail": report.n_anchor_fail,
        "unparsed_notes": [{"source": s, "name": n, "notes": t}
                           for s, n, t in report.unparsed_notes],
        "uniaxial_checks": [uni_row(c) for c in report.uniaxial_checks],
        "table_fails": [{"category": c, "name": n, "detail": d}
                        for c, n, d in report.table_fails],
        "table_checked": report.table_checked,
        "result": "FAIL" if report.any_fail else "PASS",
    }
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="opticalproperties/ root (default: repo's)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="notes-anchor tolerance on n (default 2e-4)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write a machine-readable summary here")
    args = ap.parse_args()

    report, _props = run(args.root, args.tol)
    print_report(report, args.root, args.tol)
    if args.json:
        write_json(report, args.json, args.root)
        print("\nJSON summary written: %s" % args.json)

    sys.exit(1 if report.any_fail else 0)


if __name__ == "__main__":
    main()
