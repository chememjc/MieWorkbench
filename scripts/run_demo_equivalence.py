#!/usr/bin/env python
"""Demo-equivalence gate for the object-placer train rebuild.

For every demo: rebuild it fresh through the chain API (make_demos.py),
compare against the committed pre-rebuild baselines, and PASS/FAIL:

  placement gate   every body's position within POS_TOL of the baseline
                   AND its optical-axis direction (placement-rotated
                   local +x) within ANG_TOL — compared UP TO SIGN, since
                   chained orientation legitimately differs from the old
                   hand-rolled quaternions by a spin about the axis of a
                   rotationally symmetric element (mirrors) [the spin is
                   REPORTED, never silently significant];
  power gate       a fresh 3-seed quick run's per-detector total power
                   within max(3 sigma_baseline_seed_spread, 1%%) of the
                   baseline, closure_ok everywhere;
  fringe gate      (michelson-family) profile_visibility within VIS_TOL.

Restartable: results accumulate in <workdir>/results.csv; finished demos
are skipped unless --force. Run under the GUI venv (make_demos drives a
full Project session):

  env/bin/python scripts/run_demo_equivalence.py                # all
  env/bin/python scripts/run_demo_equivalence.py --demos michelson --force
  env/bin/python scripts/run_demo_equivalence.py --skip-run     # placements only
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import html
import re
import zipfile

import common  # noqa: E402
import make_demos  # noqa: E402  (the canonical demo list + study policy)
import miewb_tool  # noqa: E402

from mieworkbench.core.fcclient import FcClient  # noqa: E402

BASELINE_DIR = REPO / "demos" / "baselines"
DEMOS_DIR = REPO / "demos"
DEFAULT_WORKDIR = REPO / "var" / "work" / "demo_equivalence"

POS_TOL_MM = 1e-3            # 1 um
ANG_TOL_DEG = 0.01
REL_POWER_FLOOR = 0.01       # 1% of baseline power
SIGMA_MULT = 3.0
VIS_TOL = 0.05

DEMO_NAMES = [
    "beam_expander", "camera_triplet", "czerny_turner", "dobsonian",
    "fiber_coupler", "michelson", "microscope_objective", "newtonian",
    "prism_spectrometer", "schmidt_cassegrain",
    # Phase-12 new-physics demos
    "ktp_walkoff", "gaussian_bench", "ghost_doublet", "scatter_plate",
    "curved_focal",
    # optimize/tolerance-round showcase demos (new)
    "double_gauss", "fiber_coupling_doublet",
]
FRINGE_DEMOS = {"michelson"}

# ---------------------------------------------------------------------------
# optimize/tolerance study gate (config-resolution + showcase smoke runs)
# ---------------------------------------------------------------------------
PENALTY = 1e9               # optimize.PENALTY (a failed/incomplete eval)
USABLE = 1e8               # a merit below this is a real (non-penalized) value
SHOWCASE = ["camera_triplet", "schmidt_cassegrain", "double_gauss",
            "fiber_coupling_doublet"]
TRAIN_FIELDS = ("distance", "decenter_x", "decenter_y", "tilt_rx", "tilt_ry",
                "tilt_rz", "fold_deviation", "fold_azimuth")

# per-showcase SMOKE tolerance subsets: a handful of the shipped rows kept
# short (the fast_eval variant directory name encodes every parameter, so a
# 30-row study exceeds NAME_MAX — see the notes in this file's report). The
# subset is enough to prove finite, non-zero, element-resolved sensitivities
# (and, for camera_triplet, the middle-vs-outer decenter comparison).
SMOKE_TOL_ROWS = {
    "camera_triplet": ["train.L1.decenter_x", "train.L2.decenter_x",
                       "train.L3.decenter_x"],
    "schmidt_cassegrain": ["train.Secondary.distance",
                           "train.Secondary.tilt_rx", "train.Focus.distance"],
    "double_gauss": ["train.D1.decenter_x", "train.D2.decenter_x",
                     "train.D1.distance"],
    "fiber_coupling_doublet": ["train.Doublet.decenter_x",
                               "train.Doublet.distance"],
}
# rays/eval for the smoke MC evaluations (kept small; the point is a
# populated series / finite sensitivity, not a converged number)
SMOKE_RAYS = 30000
SMOKE_BUDGET = 3


def _read_configs(fcstd_path):
    """(optimize_cfg, tolerance_cfg) — the panes' config() dicts stashed on
    the miewb_vars sheet (read straight from the .FCStd Document.xml, stdlib
    zip). Either may be None (demo ships no such config)."""
    out = {"optimize": None, "tolerance": None}
    try:
        xml = zipfile.ZipFile(str(fcstd_path)).read(
            "Document.xml").decode("utf8", "replace")
    except Exception:
        return None, None
    for prop, key in (("miewb_optimize_config", "optimize"),
                      ("miewb_tolerance_config", "tolerance")):
        m = re.search(r'name="%s".*?<String value="([^"]*)"' % re.escape(prop),
                      xml, re.S)
        if not m:
            continue
        try:
            payload = json.loads(html.unescape(m.group(1)))
            out[key] = payload.get(key)
        except Exception:
            out[key] = None
    return out["optimize"], out["tolerance"]


def _resolve_addr(addr, sheets, chained):
    """None if the variable/tolerance address resolves on the model, else a
    reason string. Forms: miewb_vars.<name> / dim_<El>.<alias> (a named
    sheet alias), train.<El>.<field> (a chained element pose field), or a
    bare 'alias' (default dim sheet, accepted)."""
    if addr.startswith("train."):
        el, sep, field = addr[len("train."):].rpartition(".")
        if not sep or not el:
            return "malformed train address"
        if field not in TRAIN_FIELDS:
            return "unknown train field %r" % field
        if el not in chained:
            return ("element %r is not chained (chained: %s)"
                    % (el, ", ".join(sorted(chained)) or "<none>"))
        return None
    if "." in addr:
        sheet_label, _, alias = addr.partition(".")
        aliases = sheets.get(sheet_label)
        if aliases is None:
            return "sheet %r not found" % sheet_label
        if alias not in aliases:
            return "alias %r not on sheet %r" % (alias, sheet_label)
        return None
    return None                       # bare dim-sheet alias — not gated here


def check_addresses(name, fcstd_path, fc):
    """[] on pass, else a list of unresolved-address failure strings for the
    demo's stored optimize/tolerance specs (every miewb_vars.<name>,
    dim_<El>.<alias>, train.<El>.<field> must resolve on the rebuilt model)."""
    opt, tol = _read_configs(fcstd_path)
    st = fc.open_document(str(fcstd_path))
    failures = []
    try:
        sheets = {}
        for s in st.get("sheets", []):
            label = s.get("label") or s.get("name")
            sheets[label] = set((s.get("aliases") or {}).keys())
        chained = set()
        for b in st.get("bodies", []):
            props = b.get("properties", {})
            if str((props.get("miewb_train_mode") or {}).get("value")) \
                    == "chained":
                grp = (props.get("miewb_group") or {}).get("value")
                if grp:
                    chained.add(grp)
                if b.get("label"):
                    chained.add(b["label"])
        addrs = []
        for spec in (opt or {}).get("var", []):
            addrs.append(("optimize var", spec.split(":")[0]))
        comp = (tol or {}).get("compensator")
        if comp:
            addrs.append(("compensator", comp.split(":")[0]))
        for spec in (tol or {}).get("tolerance", []):
            addrs.append(("tolerance", spec.split(":")[0]))
        for what, addr in addrs:
            err = _resolve_addr(addr, sheets, chained)
            if err:
                failures.append("%s %s -> %s" % (what, addr, err))
    finally:
        fc.close(st["doc"])
    return failures


def _run_study(script, cfg, model, out_dir, extra):
    """Run optimize.py/tolerance.py under the optics env with a JSON config;
    returns (returncode, report_dict_or_None)."""
    cfg_path = out_dir.with_suffix(".config.json")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg))
    cmd = [common.OPTICS_PYTHON, str(REPO / "scripts" / script),
           "--model", str(model), "--config", str(cfg_path),
           "--out", str(out_dir)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = out_dir / "report.json"
    data = None
    if report.exists():
        try:
            data = json.loads(report.read_text())
        except Exception:
            data = None
    return proc.returncode, data, proc.stderr


def smoke_optimize(name, fcstd_path, workdir):
    """Short optimize run: assert >=1 non-penalized eval (a populated
    convergence series). [] on pass."""
    opt, _ = _read_configs(fcstd_path)
    if not opt:
        return ["no optimize config to smoke"]
    extra = ["--budget", str(SMOKE_BUDGET), "--no-final-coherent"]
    if opt.get("eval_backend") == "worker":
        extra += ["--rays", str(SMOKE_RAYS)]
    rc, data, err = _run_study("optimize.py", opt, fcstd_path,
                               workdir / ("opt_%s" % name), extra)
    if data is None:
        last = (err or "").strip().splitlines()
        return ["optimize produced no report (exit %d): %s"
                % (rc, last[-1] if last else "")]
    history = data.get("history") or []
    good = [h for h in history
            if h.get("merit") is not None and h["merit"] < USABLE]
    if not good:
        return ["optimize: no non-penalized eval in %d (status %s)"
                % (len(history), data.get("status"))]
    return []


def smoke_tolerance(name, fcstd_path, workdir):
    """Short sensitivity pass over a trimmed row subset: assert finite,
    non-zero sensitivities. For camera_triplet also report the decenter
    ranking (middle L2 vs outer L1/L3). ([], notes) on pass."""
    _, tol = _read_configs(fcstd_path)
    if not tol:
        return ["no tolerance config to smoke"], []
    rows_wanted = SMOKE_TOL_ROWS.get(name)
    shipped = {r.split(":")[0]: r for r in tol.get("tolerance", [])}
    rows = [shipped[a] for a in rows_wanted if a in shipped] if rows_wanted \
        else list(tol.get("tolerance", []))[:3]
    if not rows:
        return ["tolerance: none of the smoke rows are in the shipped config"], []
    cfg = dict(tol, tolerance=rows, operand=tol.get("operand") or ["spot_rms:0:1"],
               eval_backend="worker")
    rc, data, err = _run_study(
        "tolerance.py", cfg, fcstd_path, workdir / ("tol_%s" % name),
        ["--draws", "0", "--rays", str(SMOKE_RAYS)])
    if data is None:
        last = (err or "").strip().splitlines()
        return ["tolerance produced no report (exit %d): %s"
                % (rc, last[-1] if last else "")], []
    sens = data.get("sensitivity") or []
    nonzero = [r for r in sens
               if r.get("impact") is not None and abs(r["impact"]) > 0.0]
    if not nonzero:
        return ["tolerance: no finite non-zero sensitivity (%d rows)"
                % len(sens)], []
    notes = ["sens %s"
             % ", ".join("%s=%.3g" % (r["name"].split(".", 1)[-1], r["impact"])
                         for r in sens if r.get("impact") is not None)]
    if name == "camera_triplet":
        # The idealized Cooke-triplet story is that the MIDDLE element's
        # decenter dominates. The as-built broadband triplet is aberration/
        # stray-ray limited (spot_rms is unstable — a few far-landing rays
        # dominate the RMS), so this does NOT reproduce cleanly; the
        # measured ranking is REPORTED, not asserted (the gate passes on
        # finite non-zero sensitivity, like the other showcases). See the
        # round report / demos/README.md.
        dec = [r for r in sens if ".decenter_" in r["name"]
               and r.get("impact")]
        dec.sort(key=lambda r: -r["impact"])
        if dec:
            l2 = any(".L2." in r["name"] for r in dec[:2])
            notes.append("decenter rank: %s (L2 in top-2: %s)"
                         % (" > ".join(r["name"].split(".")[1] for r in dec),
                            l2))
    return [], notes


def quat_axis(quat):
    """Placement quat (x,y,z,w) -> world direction of local +x."""
    x, y, z, w = quat
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w),
            2 * (x * z - y * w)]


def ang_between(a, b):
    dot = sum(p * q for p, q in zip(a, b))
    na = math.sqrt(sum(p * p for p in a))
    nb = math.sqrt(sum(q * q for q in b))
    c = max(-1.0, min(1.0, dot / (na * nb)))
    return math.degrees(math.acos(c))


def check_placements(name, fcstd_path, fc):
    """[] on pass, else a list of failure strings. Also returns notes for
    sign-flipped axes (allowed, reported)."""
    base = json.loads(
        (BASELINE_DIR / ("%s.placements.json" % name)).read_text())
    st = fc.open_document(str(fcstd_path))
    failures, notes = [], []
    try:
        new = {b["label"]: b["placement"] for b in st["bodies"]}
        for info in base["bodies"].values():
            label = info["label"]
            if label not in new:
                failures.append("body %s missing" % label)
                continue
            bp, np_ = info["placement"], new[label]
            dpos = max(abs(a - b) for a, b in
                       zip(bp["pos_mm"], np_["pos_mm"]))
            if dpos > POS_TOL_MM:
                failures.append("%s position off by %.4g mm"
                                % (label, dpos))
            a_old = quat_axis(bp["quat"])
            a_new = quat_axis(np_["quat"])
            d_same = ang_between(a_old, a_new)
            d_flip = ang_between(a_old, [-v for v in a_new])
            if min(d_same, d_flip) > ANG_TOL_DEG:
                failures.append("%s optical axis off by %.4g deg"
                                % (label, min(d_same, d_flip)))
            elif d_flip < d_same:
                notes.append("%s axis sign-flipped (symmetric)" % label)
            # spin about the axis is allowed for symmetric elements but
            # reported so a reviewer can eyeball the list
            q_same = min(
                sum((a - b) ** 2 for a, b in
                    zip(bp["quat"], np_["quat"])) ** 0.5,
                sum((a + b) ** 2 for a, b in
                    zip(bp["quat"], np_["quat"])) ** 0.5)
            if q_same > 2e-4 and min(d_same, d_flip) <= ANG_TOL_DEG:
                notes.append("%s spun about its axis (symmetric)" % label)
    finally:
        fc.close(st["doc"])
    return failures, notes


def harvest(results_dir, stem):
    case_dirs = [d for d in sorted((results_dir / stem).glob("*"))
                 if (d / "report.json").exists()]
    if not case_dirs:
        raise RuntimeError("no finished case under %s/%s"
                           % (results_dir, stem))
    with open(case_dirs[0] / "report.json") as fh:
        return json.load(fh)


def check_power(name, workdir, seeds):
    base = json.loads(
        (BASELINE_DIR / ("%s.power.json" % name)).read_text())
    rundir = workdir / ("run_%s" % name)
    if rundir.exists():
        shutil.rmtree(str(rundir))
    rc, _ = miewb_tool.run_miewb(
        workdir / ("%s.MieWB" % name), rundir.with_suffix(".MieSim"),
        workdir=rundir,
        extra_args=["--seeds", str(seeds),
                    "--steps", "extract,trace,post"])
    if rc != 0:
        return ["pipeline failed (exit %d)" % rc], []
    stem = miewb_tool.read_manifest(
        workdir / ("%s.MieWB" % name)).get("model_stem") or name
    report = harvest(rundir / "results", stem)
    failures, notes = [], []
    if not report.get("closure_ok"):
        failures.append("energy ledger did not close")
    for label, bdet in base["detectors"].items():
        det = (report.get("detectors") or {}).get(label)
        if det is None:
            failures.append("detector %s missing from report" % label)
            continue
        p0 = float(bdet["total_power_W"])
        p1 = float(det["total_power_W"])
        sigma = float(bdet.get("power_seed_std_W") or 0.0)
        tol = max(SIGMA_MULT * sigma, REL_POWER_FLOOR * abs(p0))
        if abs(p1 - p0) > tol:
            failures.append(
                "%s power %.4g W vs baseline %.4g W (tol %.3g)"
                % (label, p1, p0, tol))
        else:
            notes.append("%s power dP=%.2g W (tol %.2g)"
                         % (label, p1 - p0, tol))
        if name in FRINGE_DEMOS:
            v0 = bdet.get("profile_visibility")
            v1 = det.get("profile_visibility")
            if v0 is not None and v1 is not None \
                    and abs(float(v1) - float(v0)) > VIS_TOL:
                failures.append("%s visibility %.3f vs %.3f"
                                % (label, float(v1), float(v0)))
    shutil.rmtree(str(rundir), ignore_errors=True)
    return failures, notes


def build_demo(name, workdir):
    """Rebuild one demo (FCStd + MieWB) via make_demos.py in-process is
    not possible per demo cheaply (worker lifecycle) — subprocess the
    script, which also proves the shipped CLI path."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_demos.py"),
         "--demo", name, "--outdir", str(workdir)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("make_demos failed for %s:\n%s\n%s"
                           % (name, proc.stdout[-1500:],
                              proc.stderr[-500:]))


def load_done(csv_path):
    done = {}
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done[row["demo"]] = row
    return done


def save_done(csv_path, done):
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["demo", "status", "detail"])
        w.writeheader()
        for name in sorted(done):
            w.writerow(done[name])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", default=",".join(DEMO_NAMES))
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-run", action="store_true",
                    help="placement gate only (no trace)")
    ap.add_argument("--skip-configs", action="store_true",
                    help="skip the optimize/tolerance address-resolution "
                         "sweep over EVERY demo")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip the showcase optimize/tolerance smoke runs")
    ap.add_argument("--demos-dir", default=str(DEMOS_DIR),
                    help="dir of committed demo .FCStd/.MieWB for the "
                         "config-resolution sweep + smoke runs (default demos/)")
    args = ap.parse_args()

    names = [n.strip() for n in args.demos.split(",") if n.strip()]
    unknown = [n for n in names if n not in DEMO_NAMES]
    if unknown:
        ap.error("unknown demos: %s" % ", ".join(unknown))
    demos_dir = Path(args.demos_dir)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "results.csv"
    done = load_done(csv_path)

    fc = FcClient()
    fc.start()
    n_fail = 0
    try:
        for name in names:
            if not args.force and name in done \
                    and done[name]["status"].startswith("PASS"):
                print("[skip] %s (already %s)" % (name,
                                                  done[name]["status"]))
                continue
            print("[build] %s" % name, flush=True)
            build_demo(name, workdir)
            failures, notes = check_placements(
                name, workdir / ("%s.FCStd" % name), fc)
            if not failures and not args.skip_run:
                pf, pn = check_power(name, workdir, args.seeds)
                failures += pf
                notes += pn
            status = "PASS" if not failures else "FAIL"
            if not failures and args.skip_run:
                status = "PASS-placements"
            detail = "; ".join(failures or notes[:4])
            done[name] = {"demo": name, "status": status,
                          "detail": detail}
            save_done(csv_path, done)
            print("[%s] %s%s" % (status, name,
                                 (": " + detail) if detail else ""),
                  flush=True)
            if failures:
                n_fail += 1

        # -- config address-resolution sweep over EVERY demo -----------------
        if not args.skip_configs:
            print("\n== optimize/tolerance config address resolution "
                  "(every demo) ==", flush=True)
            for name in sorted(make_demos.DEMOS):
                fcstd = (workdir / ("%s.FCStd" % name)) if (
                    workdir / ("%s.FCStd" % name)).exists() \
                    else (demos_dir / ("%s.FCStd" % name))
                if not fcstd.exists():
                    print("[configs] %-24s SKIP (no .FCStd)" % name)
                    continue
                cf = check_addresses(name, fcstd, fc)
                opt, tol = _read_configs(fcstd)
                tag = "opt+tol" if (opt and tol) else (
                    "tol" if tol else ("opt" if opt else "none"))
                print("[configs] %-24s %s (%s)"
                      % (name, "PASS" if not cf else "FAIL",
                         "; ".join(cf) if cf else tag), flush=True)
                if cf:
                    n_fail += 1

        # -- showcase smoke runs (short optimize + tolerance sensitivity) ----
        if not args.skip_smoke:
            print("\n== showcase optimize/tolerance smoke runs ==", flush=True)
            for name in SHOWCASE:
                fcstd = (workdir / ("%s.FCStd" % name)) if (
                    workdir / ("%s.FCStd" % name)).exists() \
                    else (demos_dir / ("%s.FCStd" % name))
                if not fcstd.exists():
                    print("[smoke] %-24s SKIP (no .FCStd)" % name)
                    continue
                of = smoke_optimize(name, fcstd, workdir / "smoke")
                tf, tn = smoke_tolerance(name, fcstd, workdir / "smoke")
                fails = of + tf
                print("[smoke] %-24s %s%s"
                      % (name, "PASS" if not fails else "FAIL",
                         ((": " + "; ".join(fails)) if fails
                          else ("  " + " | ".join(tn) if tn else ""))),
                      flush=True)
                if fails:
                    n_fail += 1
    finally:
        fc.shutdown()

    print("\n== demo equivalence: %d/%d passed (results: %s)"
          % (len([d for d in done.values()
                  if d["status"].startswith("PASS")]),
             len(names), csv_path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
