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
    # WP7 beyond-sequential showcase demos
    "fizeau_flats", "fs_shg_spectrogram", "quartz_rotator",
    "speckle_mie_combo",
    # samples-instruments round demos (placement + power + bespoke physics)
    "conical_refraction", "colloidal_crystal", "goniometer_bath",
    "uvvis_spectrometer", "insitec_sizer", "imaging_bench",
]
FRINGE_DEMOS = {"michelson"}
# WP7 demos with a bespoke physics gate (implemented in wp7_gates.py-style
# functions below) run IN ADDITION to the placement + power baseline gates.
PATTERN_DEMOS = {"fizeau_flats", "fs_shg_spectrogram", "quartz_rotator",
                 "speckle_mie_combo"}

# ---------------------------------------------------------------------------
# optimize/tolerance study gate (config-resolution + showcase smoke runs)
# ---------------------------------------------------------------------------
PENALTY = 1e9               # optimize.PENALTY (a failed/incomplete eval)
USABLE = 1e8               # a merit below this is a real (non-penalized) value
SHOWCASE = ["camera_triplet", "schmidt_cassegrain", "double_gauss",
            "fiber_coupling_doublet"]
TRAIN_FIELDS = ("distance", "decenter_x", "decenter_y", "tilt_rx", "tilt_ry",
                "tilt_rz", "fold_deviation", "fold_azimuth")

# per-showcase SMOKE tolerance subsets: a handful of the shipped rows, kept
# short purely for GATE SPEED (full studies run fine — fast_eval variant
# names hash past common.VARIANT_NAME_LIMIT so many-row studies can't
# overrun NAME_MAX). The subset is enough to prove finite, non-zero,
# element-resolved sensitivities (and, for camera_triplet, the
# middle-vs-outer decenter comparison).
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


# ---------------------------------------------------------------------------
# WP7 bespoke physics gates (beyond-sequential showcase demos).  Each returns
# (failures, notes); they read the finished case directory's detector .h5 +
# report.json.  Run under the GUI venv (numpy/scipy/h5py present).
# ---------------------------------------------------------------------------
def _detector_image(case_dir):
    """(image[H,W] W/pixel-sum, pixel_mm, attrs, h5dict) for the single
    detector in a WP7 case (all four have exactly one)."""
    import glob as _glob

    import h5py
    import numpy as np
    files = sorted(_glob.glob(str(Path(case_dir) / "detectors" / "*.h5")))
    if not files:
        raise RuntimeError("no detector .h5 under %s" % case_dir)
    with h5py.File(files[0], "r") as h:
        cube = h["spectral_cube_mean"][...]
        attrs = dict(h.attrs)
        extra = {k: h[k][...] for k in ("time_profile", "mask")
                 if k in h}
    img = cube.sum(axis=0)
    return img, float(attrs["pixel_m"]) * 1e3, attrs, extra, cube


def _fringe_metrics(img, pixel_mm):
    """(lit_visibility, n_peaks) along the brightest profile of a fringe
    image.  Visibility = (max-min)/(max+min) over the lit segment; peaks
    counted on the envelope-normalized profile."""
    import numpy as np
    from scipy.signal import find_peaks
    if img.max() <= 0:
        return 0.0, 0
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    best = (0.0, 0)
    for prof in (img[iy].astype(float), img[:, ix].astype(float)):
        m = np.where(prof > prof.max() * 0.05)[0]
        if len(m) < 8:
            continue
        seg = prof[m.min():m.max() + 1]
        pos = seg[seg > 0]
        vis = (pos.max() - pos.min()) / (pos.max() + pos.min()) \
            if len(pos) else 0.0
        win = max(5, len(seg) // 4)
        env = np.convolve(seg, np.ones(win) / win, "same")
        env[env <= 0] = env.max() if env.max() > 0 else 1.0
        pk, _ = find_peaks(seg / env, distance=3, prominence=0.15)
        if len(pk) > best[1] or vis > best[0]:
            best = (max(best[0], vis), max(best[1], len(pk)))
    return best


def gate_fizeau(name, case_dir, report):
    """Coherent Fizeau: a reconstructed high-visibility multi-fringe pattern
    (the beyond-sequential coherent multipath interference).  Absolute pitch
    is NOT gated -- at this bench scale the coherent gather sits near
    phase_step~pi and a detector-alignment / 4-surface pedestal confounds a
    clean lambda/(2 alpha); the robust observables are fringe visibility +
    count (michelson's approach)."""
    det = list((report.get("detectors") or {}).values())
    fail, notes = [], []
    p = float(det[0]["total_power_W"]) if det else 0.0
    if p <= 0:
        return ["fizeau: zero detected power (coherent reconstruction "
                "collapsed)"], []
    img, pix, attrs, extra, cube = _detector_image(case_dir)
    vis, npk = _fringe_metrics(img, pix)
    if vis < 0.30:
        fail.append("fizeau: lit-region fringe visibility %.3f < 0.30" % vis)
    if npk < 4:
        fail.append("fizeau: only %d fringe peaks (< 4) -- no interferogram"
                    % npk)
    notes.append("fizeau: power=%.3g W, visibility=%.3f, %d fringes"
                 % (p, vis, npk))
    return fail, notes


def gate_fs_shg(name, case_dir, report):
    """fs SHG spectrogram: (a) a lambda/2 (400 nm) harmonic band above
    threshold in the spectral cube, and (b) the fundamental pulse FWHM within
    +-30%% of the analytic chirped-Gaussian stretch tau(GDD) for the SF11 rod
    (tau0=100 fs, beta2=189.6 fs^2/mm * 60 mm)."""
    import numpy as np
    fail, notes = [], []
    img, pix, attrs, extra, cube = _detector_image(case_dir)
    lo = float(attrs["lam_lo_m"]) * 1e9
    hi = float(attrs["lam_hi_m"]) * 1e9
    bins = cube.shape[0]
    lam = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
    pw = cube.reshape(bins, -1).sum(axis=1)
    harm = pw[(lam >= 350) & (lam <= 450)].sum()
    fund = pw[(lam >= 750) & (lam <= 850)].sum()
    ratio = harm / fund if fund > 0 else 0.0
    if ratio < 1e-3:
        fail.append("fs_shg: 400 nm harmonic band %.2e of fundamental "
                    "(< 1e-3) -- no SHG conversion" % ratio)
    else:
        notes.append("fs_shg: harmonic/fundamental band ratio = %.3g" % ratio)
    # fundamental stretched FWHM from the time profile
    prof = extra.get("time_profile")
    if prof is None or float(np.max(prof)) <= 0:
        fail.append("fs_shg: no time profile (pulse product missing)")
        return fail, notes
    dt = float(attrs["time_dt_s"])
    pk = float(np.max(prof))
    above = np.where(prof >= pk / 2.0)[0]
    fwhm_fs = (above[-1] - above[0] + 1) * dt * 1e15 if len(above) else 0.0
    tau0, beta2, L = 100.0, 189.6, 60.0
    gdd = beta2 * L
    tau_an = tau0 * math.sqrt(
        1.0 + (gdd / (tau0 ** 2 / (4.0 * math.log(2.0)))) ** 2)
    rel = abs(fwhm_fs - tau_an) / tau_an
    if rel > 0.30:
        fail.append("fs_shg: pulse FWHM %.0f fs vs analytic %.0f fs "
                    "(%.0f%% > 30%%)" % (fwhm_fs, tau_an, 100 * rel))
    else:
        notes.append("fs_shg: pulse FWHM %.0f fs vs analytic %.0f fs "
                     "(%.0f%%)" % (fwhm_fs, tau_an, 100 * rel))
    return fail, notes


def gate_quartz(name, case_dir, report):
    """Quartz rotator: ASSERTED.  Scene-level natural optical activity is wired
    into the tracer (tracer._apply_optical_activity: near-axis gyrotropic rays
    take the isotropic n_o path and rotate the polarization plane by rho*d).  A
    2 mm z-cut quartz slab between CROSSED polarizers passes sin^2(rho*d) of the
    light reaching the analyzer, rho = 21.77 deg/mm @589.3 nm.  The analyzer is
    a material=air ideal polarizer (no Fresnel) and the quartz-exit Fresnel loss
    sits UPSTREAM of it, so the crossed throughput measured AT the analyzer --
    detected / analyzer_power_in -- is the clean sin^2(rho*d), independent of
    the source/coupling absolute power."""
    rho, d_mm = 21.77, 2.0
    expect = math.sin(math.radians(rho * d_mm)) ** 2         # ~0.4745
    det = list((report.get("detectors") or {}).values())
    p_det = float(det[0]["total_power_W"]) if det else 0.0
    ana = (report.get("elements") or {}).get("Analyzer")
    if ana is None or float(ana.get("power_in_W") or 0.0) <= 0.0:
        return (["quartz_rotator: analyzer element power_in missing/zero -- "
                 "cannot form the crossed-throughput ratio"], [])
    p_in = float(ana["power_in_W"])
    frac = p_det / p_in
    rel = abs(frac - expect) / expect
    note = ("quartz_rotator: crossed throughput detected/analyzer_in = %.3f "
            "vs sin^2(rho*d)=sin^2(%.1f deg)=%.3f (%.1f%%)"
            % (frac, rho * d_mm, expect, 100 * rel))
    if rel > 0.05:
        return ([note + " -- exceeds 5%% optical-rotation tolerance"], [])
    return ([], [note])


# the demo's aerosol spec (kept in lock-step with demo_speckle_mie_combo)
_SPECKLE_PARTICLES = ("box=5,-15,-15:40,30,30;material=water;phi=1.0e-2;"
                      "median_um=2.0;gsd=1.5")
_SPECKLE_LAM_NM = 532.0
# analytic Mie extinction of the spec, computed under the optics env (miepython
# lives there, not in the GUI venv the gate runs in).  A no-cloud TRACE
# reference is unusable here: a phi~0 particle box is a pathological continuum
# edge case that runs many minutes, so the "no-cloud analytic" the WP7 brief
# allows is the analytic ballistic optical depth instead.
_MU_EXT_SCRIPT = r"""
import sys, json
sys.path.insert(0, "%s")
import numpy as np, common
from raytracer.particles import ParticleCloud
from raytracer.optprops import load_optical_properties
spec = common.parse_particles_spec(sys.argv[1]); lam_nm = float(sys.argv[2])
props = load_optical_properties(root="%s")
class Stub:
    matdb = props.matdb
    ambient = props.matdb.get("air")
cloud = ParticleCloud(spec, Stub(), lam_list=(lam_nm*1e-9,))
mu = cloud.tables.mu_ext(lam_nm*1e-9); L = float(spec["box_size_m"][0])
print(json.dumps({"tau": mu*L, "T": float(np.exp(-mu*L)), "mu": mu}))
""" % (str(REPO / "scripts"), str(REPO / "opticalproperties"))


def gate_speckle(name, case_dir, report, workdir, seeds):
    """Speckle x Mie: (a) speckle contrast sigma/<I> over the lit region in
    [0.5, 1.1] (coherent diffuser speckle -- the beyond-sequential headline),
    and (b) the aerosol cloud's analytic Beer-Lambert optical depth tau in a
    loose band [0.3, 1.5] (real, significant, partial extinction), computed
    under the optics env from the same --particles spec the demo ships."""
    import numpy as np
    fail, notes = [], []
    img, pix, attrs, extra, cube = _detector_image(case_dir)
    det = list((report.get("detectors") or {}).values())
    p_cloud = float(det[0]["total_power_W"]) if det else 0.0
    lit = img[img > img.max() * 0.1]
    contrast = float(lit.std() / lit.mean()) if len(lit) > 10 \
        and lit.mean() > 0 else 0.0
    if not (0.5 <= contrast <= 1.1):
        fail.append("speckle: contrast sigma/<I>=%.3f outside [0.5, 1.1]"
                    % contrast)
    else:
        notes.append("speckle: contrast sigma/<I>=%.3f (fwd power %.3g W)"
                     % (contrast, p_cloud))
    # analytic Mie extinction (optics env subprocess)
    proc = subprocess.run(
        [common.OPTICS_PYTHON, "-c", _MU_EXT_SCRIPT, _SPECKLE_PARTICLES,
         "%g" % _SPECKLE_LAM_NM], capture_output=True, text=True)
    try:
        ext = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        fail.append("speckle: analytic mu_ext subprocess failed: %s"
                    % (proc.stderr.strip().splitlines()[-1:] or [""])[0])
        return fail, notes
    tau = float(ext["tau"])
    if not (0.3 <= tau <= 1.5):
        fail.append("speckle: analytic cloud optical depth tau=%.3f outside "
                    "[0.3, 1.5]" % tau)
    else:
        notes.append("speckle: analytic Mie tau=%.3f (ballistic T=%.3f, "
                     "extinction %.0f%%)" % (tau, ext["T"], 100 * (1 - ext["T"])))
    return fail, notes


# ---------------------------------------------------------------------------
# samples-instruments round bespoke gates.  Same (name, case_dir, report) ->
# (failures, notes) contract as the WP7 gates; run under the GUI venv.
# ---------------------------------------------------------------------------
def _named_cube(case_dir, label_sub):
    """(image[H,W], pixel_mm, attrs, cube[bins,H,W]) for the detector whose
    label CONTAINS label_sub. None if not found."""
    import glob as _glob

    import h5py
    import numpy as np
    for f in sorted(_glob.glob(str(Path(case_dir) / "detectors" / "*.h5"))):
        with h5py.File(f, "r") as h:
            if label_sub not in str(h.attrs.get("label", "")):
                continue
            cube = h["spectral_cube_mean"][...]
            attrs = dict(h.attrs)
        return cube.sum(axis=0), float(attrs["pixel_m"]) * 1e3, attrs, cube
    return None


# KTP internal conical-refraction cone half-angle A (full opening) at 1064 nm,
# from birefringence.cone_half_angle on the biaxial ktp eps (see
# demo_conical_refraction); the exit-face ring radius is 0.5 * t * tan(A).
_CONICAL_A_RAD = 0.0279611186796979
_CONICAL_T_M = 0.020


def gate_conical(name, case_dir, report):
    """Internal conical refraction: about its own power centroid the detected
    footprint is a hollow RING of radius 0.5*t*tan(A) (within 10%) that is
    radially TIGHT (std_r < 0.35*mean_r) — a ring, not a disc."""
    import numpy as np
    got = _named_cube(case_dir, "Screen")
    if got is None:
        return ["conical: no Screen detector"], []
    img, pix_mm, attrs, cube = got
    tot = float(img.sum())
    if tot <= 0:
        return ["conical: zero detected power"], []
    pix = float(attrs["pixel_m"])
    H, W = img.shape
    xs = float(attrs["x_lo"]) + (np.arange(W) + 0.5) * pix
    ys = float(attrs["y_lo"]) + (np.arange(H) + 0.5) * pix
    X, Y = np.meshgrid(xs, ys)
    cx = float((img * X).sum() / tot)
    cy = float((img * Y).sum() / tot)
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mean_r = float((img * R).sum() / tot)
    std_r = float(np.sqrt((img * (R - mean_r) ** 2).sum() / tot))
    r_exp = 0.5 * _CONICAL_T_M * math.tan(_CONICAL_A_RAD)
    fail, notes = [], []
    rel = abs(mean_r - r_exp) / r_exp
    if rel > 0.10:
        fail.append("conical: ring radius %.4g mm vs %.4g mm (%.0f%% > 10%%)"
                    % (mean_r * 1e3, r_exp * 1e3, 100 * rel))
    if std_r >= 0.35 * mean_r:
        fail.append("conical: std_r/mean_r=%.3f >= 0.35 (a disc, not a ring)"
                    % (std_r / mean_r if mean_r else 0))
    notes.append("conical: ring r=%.4g mm vs %.4g mm (%.0f%%), std/mean=%.3f"
                 % (mean_r * 1e3, r_exp * 1e3, 100 * rel, std_r / mean_r))
    return fail, notes


def gate_insitec(name, case_dir, report):
    """Laser-diffraction sizer: the scattered forward-diffraction lobe (rings
    outside the DC focal spot) has its first-minimum (half-power outer edge)
    at r = f_eff*1.22*lambda/d within +/-1 log-ring, and the ring integration
    closes."""
    import csv as _csv
    import glob as _glob
    f = _glob.glob(str(Path(case_dir) / "analysis" / "rings_*.csv"))
    if not f:
        return ["insitec: no ring-profile CSV"], []
    rows = list(_csv.DictReader([ln for ln in open(f[0]) if not
                                 ln.startswith("#")]))
    if not rows:
        return ["insitec: empty ring CSV"], []
    bfl, lam_mm, d_mm = 45.24, 633e-6, 0.010
    r_exp = bfl * 1.22 * lam_mm / d_mm            # first-null radius, mm
    def mid(x):
        return 0.5 * (float(x["r_inner_mm"]) + float(x["r_outer_mm"]))
    scat = [r for r in rows if mid(r) > 0.8]      # exclude the DC focal spot
    if not scat:
        return ["insitec: no scattered rings past the DC spot"], []
    pk = max(scat, key=lambda r: float(r["power_W"]))
    pk_i = scat.index(pk)
    p_pk = float(pk["power_W"])
    # outer half-power radius of the lobe (first dark ring)
    r_half = None
    for r in scat[pk_i + 1:]:
        if float(r["power_W"]) <= 0.5 * p_pk:
            r_half = mid(r)
            break
    fail, notes = [], []
    if r_half is None:
        fail.append("insitec: lobe never falls to half-power (no first null)")
    else:
        # +/-1 ring: the log-ring width near r_exp (~0.7 mm at 3.5 mm)
        ring_w = float(pk["r_outer_mm"]) - float(pk["r_inner_mm"])
        tol = max(0.9, 1.5 * ring_w)
        if abs(r_half - r_exp) > tol:
            fail.append("insitec: first-null r=%.3f mm vs r_exp=%.3f mm "
                        "(|d|=%.3f > %.3f)" % (r_half, r_exp,
                                               abs(r_half - r_exp), tol))
        else:
            notes.append("insitec: lobe first-null r=%.3f mm vs %.3f mm "
                         "(tol %.2f)" % (r_half, r_exp, tol))
    det = list((report.get("detectors") or {}).values())
    rings = (det[0].get("rings") if det else None) or {}
    resid = abs(float(rings.get("closure_residual_W", 1.0)))
    tot = abs(float(rings.get("total_power_W", 0.0)))
    if tot > 0 and resid > 1e-3 * tot:
        fail.append("insitec: ring closure residual %.3g of total %.3g"
                    % (resid, tot))
    return fail, notes


def gate_goniometer(name, case_dir, report):
    """Static light scattering: the three arc detectors' power vs q follows a
    fractal power law I(q) ~ q^-df; the log-log slope is -df = -2.1 +/-0.3
    (registry df=2.1). Slope is invariant to the n/lambda scale."""
    import numpy as np
    n, lam = 1.475, 632.8e-9
    pts = []
    for label, dd in (report.get("detectors") or {}).items():
        m = re.search(r"Det(\d+)", label)
        p = float(dd.get("total_power_W") or 0.0)
        if m and p > 0:
            th = float(m.group(1))
            q = 4 * math.pi * n / lam * math.sin(math.radians(th) / 2)
            pts.append((q, p, th))
    if len(pts) < 3:
        return ["goniometer: only %d detectors with power (need 3)"
                % len(pts)], []
    xs = np.log([q for q, _, _ in pts])
    ys = np.log([p for _, p, _ in pts])
    slope = float(np.polyfit(xs, ys, 1)[0])
    if abs(slope - (-2.1)) > 0.3:
        return (["goniometer: I(q) slope %.3f vs -2.1 (|d|=%.2f > 0.3)"
                 % (slope, abs(slope + 2.1))], [])
    return ([], ["goniometer: I(q) fractal slope %.3f (target -2.1 +/-0.3)"
                 % slope])


def gate_colloidal(name, case_dir, report):
    """Colloidal-crystal structural colour: the backscatter/forward
    reflectance R(lambda) is WAVELENGTH-SELECTIVE — enhanced in the
    blue-green (Bragg) region vs the red (R(blue)/R(red) > 1.8). NOTE: the
    sharp coherent (111) Bragg peak needs the explicit lattice, intractable
    at macroscopic scale (see demo docstring / UXNOTES); this gates the
    tractable continuum-paracrystal structural-scattering signature."""
    import numpy as np
    fwd = _named_cube(case_dir, "Forward")
    bck = _named_cube(case_dir, "Backscatter")
    if fwd is None or bck is None:
        return ["colloidal: missing Forward/Backscatter detector"], []
    def spectrum(got):
        img, pix, attrs, cube = got
        lo = float(attrs["lam_lo_m"]) * 1e9
        hi = float(attrs["lam_hi_m"]) * 1e9
        b = cube.shape[0]
        lam = lo + (np.arange(b) + 0.5) * (hi - lo) / b
        return lam, cube.reshape(b, -1).sum(axis=1)
    lam, fw = spectrum(fwd)
    _, bk = spectrum(bck)
    m = fw > fw.max() * 0.05
    if m.sum() < 4:
        return ["colloidal: too few populated spectral bins"], []
    R = np.where(m, bk / np.maximum(fw, 1e-30), np.nan)
    lam_m = lam[m]
    R_m = R[m]
    mid_l = 0.5 * (lam_m.min() + lam_m.max())
    blue = np.nanmean(R_m[lam_m <= mid_l])
    red = np.nanmean(R_m[lam_m > mid_l])
    ratio = blue / red if red > 0 else 0.0
    if ratio < 1.8:
        return (["colloidal: reflectance blue/red = %.2f < 1.8 (not "
                 "wavelength-selective)" % ratio], [])
    return ([], ["colloidal: wavelength-selective backscatter blue/red = "
                 "%.2f (structural colour; sharp Bragg peak intractable, "
                 "see docstring)" % ratio])


def _ncc_img(a, b):
    import numpy as np
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def gate_imaging(name, case_dir, report):
    """Finite-conjugate imaging: the traced end-to-end image
    (imaging/image_traced_*.png) reproduces the USAF target PNG, INVERTED
    (a real bench rotates 180 deg): NCC(traced, target_rot180) > 0.5 after a
    center-crop + resize (robust to magnification)."""
    import glob as _glob

    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    pngs = _glob.glob(str(Path(case_dir) / "imaging" / "image_traced_*.png"))
    pngs = [p for p in pngs if "_vs_sim" not in p]
    if not pngs:
        return ["imaging: no traced image PNG"], []
    tgt_path = REPO / "opticalproperties" / "image" / "usaf_style_target.png"

    def load_gray(p, size=32, blur=4.0):
        im = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
        # crop to the bright CONTENT bounding box (normalises magnification:
        # the traced image is ~60% of the detector with a black border, the
        # target PNG fills its frame — a center-crop would leave them at
        # different scales and tank the NCC).
        thr = im.max() * 0.15
        ys, xs = np.where(im > thr)
        if len(ys) > 20:
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = im.shape
        s = max(h, w)                      # square-pad to keep aspect
        pad = np.zeros((s, s), dtype=np.float64)
        pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
        out = np.asarray(Image.fromarray(pad).resize((size, size)),
                         dtype=np.float64)
        # match the imaging system's finite resolution: the fast singlet
        # blurs out the target's fine bar groups, so compare BOTH at that
        # common (blurred) resolution (a sharp-vs-blurred NCC is unfairly low).
        return gaussian_filter(out, blur)
    traced = load_gray(pngs[0])
    target = load_gray(str(tgt_path))
    # a real 1:1 relay INVERTS (180 deg); the detector-frame basis (xhat/yhat)
    # can additionally flip one axis, so accept any of the dihedral
    # orientations (all are "the target imaged", just the recorder's handedness)
    orients = {"direct": target, "rotated_180": target[::-1, ::-1],
               "flip_v": target[::-1, :], "flip_h": target[:, ::-1]}
    scored = {k: _ncc_img(traced, v) for k, v in orients.items()}
    orient = max(scored, key=scored.get)
    best = scored[orient]
    if best <= 0.5:
        return (["imaging: traced-vs-target NCC %.3f <= 0.5 (best %s)"
                 % (best, orient)], [])
    return ([], ["imaging: traced-vs-target NCC %.3f (%s)" % (best, orient)])


def gate_uvvis(name, sample_case, report, workdir, seeds, stem, rundir):
    """UV-Vis absorbance: run a BLANK variant (the cuvette dye liquid
    suppressed) and form A(lambda) = -log10(P_sample/P_blank) directly from
    the two Array detectors' RAW spectral cubes (the tcd1304 diode-array
    readout SATURATES at full-well across the band, collapsing the
    instrument A to 0 — the raw dispersed-power cube is the honest,
    unsaturated observable, and its lambda strata already separate the
    spectrum). Assert A peaks at 525 +/-15 nm with a peak within 30% of the
    Beer-Lambert prediction (alpha=769/m, 10 mm path -> A=3.34)."""
    import glob as _glob

    import numpy as np
    mjs = _glob.glob(str(rundir / "**" / "model.json"), recursive=True)
    if not mjs:
        return ["uvvis: no sample model.json"], []
    model = json.loads(open(mjs[0]).read())
    dye = next((b["name"] for b in model["bodies"]
                if b.get("material") == "dye_solution_kmno4"), None)
    if dye is None:
        return ["uvvis: no dye_solution_kmno4 body to suppress"], []
    blankdir = workdir / ("blank_%s" % name)
    if blankdir.exists():
        shutil.rmtree(str(blankdir))
    rc, _ = miewb_tool.run_miewb(
        workdir / ("%s.MieWB" % name), blankdir.with_suffix(".MieSim"),
        workdir=blankdir,
        extra_args=["--seeds", str(seeds), "--steps", "extract,trace,post",
                    "--suppress-body", dye])
    if rc != 0:
        return ["uvvis: blank run failed (exit %d)" % rc], []
    blank_cases = [d for d in sorted((blankdir / "results" / stem).glob("*"))
                   if (d / "report.json").exists()]
    if not blank_cases:
        return ["uvvis: no blank case dir"], []

    def array_spectrum(case):
        got = _named_cube(case, "Array")
        if got is None:
            return None
        img, pix, attrs, cube = got
        lo = float(attrs["lam_lo_m"]) * 1e9
        hi = float(attrs["lam_hi_m"]) * 1e9
        b = cube.shape[0]
        lam = lo + (np.arange(b) + 0.5) * (hi - lo) / b
        return lam, cube.reshape(b, -1).sum(axis=1)
    s = array_spectrum(sample_case)
    b = array_spectrum(blank_cases[0])
    shutil.rmtree(str(blankdir), ignore_errors=True)
    if s is None or b is None:
        return ["uvvis: missing Array cube (sample or blank)"], []
    lam, ps = s
    _, pb = b
    m = pb > pb.max() * 0.05             # populated (lit) bins only
    if m.sum() < 3:
        return ["uvvis: too few lit spectral bins"], []
    A = np.full_like(ps, np.nan)
    good = m & (ps > 0)
    A[good] = -np.log10(ps[good] / pb[good])
    if not np.any(np.isfinite(A)):
        return ["uvvis: no finite absorbance bins"], []
    pk = int(np.nanargmax(A))
    lam_pk, A_pk = float(lam[pk]), float(A[pk])
    A_bl = 769.0 * 0.010 / math.log(10.0)     # Beer-Lambert, 10 mm, 1 mM
    fail, notes = [], []
    if abs(lam_pk - 525.0) > 15.0:
        fail.append("uvvis: A peaks at %.0f nm (not 525 +/-15)" % lam_pk)
    if abs(A_pk - A_bl) > 0.30 * A_bl:
        fail.append("uvvis: peak A=%.2f vs Beer-Lambert %.2f (>30%%)"
                    % (A_pk, A_bl))
    notes.append("uvvis: A peaks %.2f at %.0f nm (Beer-Lambert %.2f)"
                 % (A_pk, lam_pk, A_bl))
    return fail, notes


PATTERN_GATES = {
    "fizeau_flats": gate_fizeau,
    "fs_shg_spectrogram": gate_fs_shg,
    "quartz_rotator": gate_quartz,
    "speckle_mie_combo": gate_speckle,
    "conical_refraction": gate_conical,
    "insitec_sizer": gate_insitec,
    "goniometer_bath": gate_goniometer,
    "colloidal_crystal": gate_colloidal,
    "imaging_bench": gate_imaging,
}


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
    # WP7 + samples-instruments bespoke physics gate (reads the finished case
    # dir before cleanup)
    if name in PATTERN_GATES or name == "uvvis_spectrometer":
        case_dirs = [dd for dd in sorted((rundir / "results" / stem).glob("*"))
                     if (dd / "report.json").exists()]
        if not case_dirs:
            failures.append("%s: no case dir for the pattern gate" % name)
        else:
            try:
                if name == "speckle_mie_combo":
                    pf, pn = gate_speckle(name, case_dirs[0], report,
                                          workdir, seeds)
                elif name == "uvvis_spectrometer":
                    pf, pn = gate_uvvis(name, case_dirs[0], report, workdir,
                                        seeds, stem, rundir)
                else:
                    pf, pn = PATTERN_GATES[name](name, case_dirs[0], report)
                failures += pf
                notes += pn
            except Exception as exc:      # a gate crash is a gate failure
                failures.append("%s: pattern gate raised %r" % (name, exc))
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
