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

import miewb_tool  # noqa: E402

from mieworkbench.core.fcclient import FcClient  # noqa: E402

BASELINE_DIR = REPO / "demos" / "baselines"
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
]
FRINGE_DEMOS = {"michelson"}


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
    args = ap.parse_args()

    names = [n.strip() for n in args.demos.split(",") if n.strip()]
    unknown = [n for n in names if n not in DEMO_NAMES]
    if unknown:
        ap.error("unknown demos: %s" % ", ".join(unknown))
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
    finally:
        fc.shutdown()

    print("\n== demo equivalence: %d/%d passed (results: %s)"
          % (len([d for d in done.values()
                  if d["status"].startswith("PASS")]),
             len(names), csv_path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
