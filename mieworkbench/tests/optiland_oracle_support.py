# =============================================================================
# optiland_oracle_support.py -- the P4a parity-ORACLE harness (engine3.md Sec.5
# / Sec.15 P4a). For each simple sequential scene it compares TWO independent
# physics truths on GEOMETRIC ground truths that both compute exactly:
#   (1) best-focus axial position of a collimated bundle;
#   (2) real ray LANDING radius on the detector across the full pupil fan
#       (subsumes the marginal + chief rays -- these scenes are on-axis, so the
#       "fan" is over pupil HEIGHT, see the note in the test docstring);
#   (3) spot RMS at the detector.
#
# The two truths:
#   * MieWorkbench C engine  -- the deterministic ray landings of a small
#     coherent=false Monte-Carlo run, EXPORTED via run_trace.py --export-rays
#     (rays_full.npz: per-ray birth_pos, detector pos, detector dir, all SI
#     world metres). Route chosen (per the task's guidance): the --viz-pattern
#     "fan" machinery is a GUI overlay that emits a fixed cardinal cross from
#     the source and is not cleanly reusable for a controllable field/height
#     fan headlessly; the --export-rays route gives the FULL deterministic
#     pupil->image map (each MC ray's landing is exact geometry given its birth
#     point -- the only stochastic thing is WHERE in the pupil each ray is
#     born), which is exactly what an aberration oracle needs.
#   * Optiland sequential   -- built by scripts/raytracer/optiland_bridge.py,
#     traced at the EXACT SAME pupil positions the MC run recorded, so the
#     focus fit and spot RMS use one identical pupil-sampling distribution and
#     the only residual is engine-vs-engine (an aberrated lens focuses
#     different pupil zones at different z, so a fair comparison MUST share the
#     sampling -- otherwise "best focus" differs by the sampling weight alone).
#
# Interpreter: env/bin/python (has Optiland + numpy). It shells out to the
# optics env (/home3/optics/env/bin/python, override MIEWB_OPTICS_PYTHON) to
# run the C engine; both are gated -- absent -> the caller skips.
# =============================================================================
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from raytracer import optiland_bridge as ob   # noqa: E402  (env has optiland)

OPTICS_PY = os.environ.get("MIEWB_OPTICS_PYTHON",
                           "/home3/optics/env/bin/python")
CENGINE = os.environ.get("MIEWB_CENGINE",
                         str(REPO / "cengine" / "build" / "miewb-trace"))


def mc_available():
    """True iff the optics-env python and the C-engine binary are present."""
    return Path(OPTICS_PY).exists() and Path(CENGINE).exists()


# --------------------------------------------------------------------------
# MieWorkbench C-engine side: run + export + load deterministic ray landings
# --------------------------------------------------------------------------
def run_cengine_export(stem, case_dir, rays=30000, seed=42):
    """Run the C engine on geometry/<stem> and export rays_full.npz. Small,
    monochromatic, coherent=false (the scenes' own sources). Returns the npz
    path. Raises on failure (never a silent skip mid-run)."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    model_json = REPO / "geometry" / stem / "model.json"
    cmd = [OPTICS_PY, str(SCRIPTS / "run_trace.py"),
           "--model-json", str(model_json), "--case-dir", str(case_dir),
           "--rays", str(int(rays)), "--nlambda", "1", "--resolution", "64",
           "--seeds", "1", "--seed0", str(int(seed)),
           "--engine", "c", "--export-rays"]
    env = dict(os.environ, MIEWB_CENGINE=CENGINE)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    npz = case_dir / "rays_full.npz"
    if r.returncode != 0 or not npz.exists():
        raise RuntimeError("C-engine export failed for %s (rc=%d):\n%s\n%s"
                           % (stem, r.returncode, r.stdout[-2000:],
                              r.stderr[-2000:]))
    return npz


def load_mc_direct_bundle(npz_path):
    """Load the DIRECT (generation-0, unscattered) transmitted ray bundle from
    the single detector: birth_pos, detector pos, detector dir (all (N,3) SI
    metres). These are the geometric image-forming rays -- Optiland's single
    deterministic path -- filtered away from Fresnel-ghost / scattered rays."""
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    dets = list(meta["detectors"].keys())
    if len(dets) != 1:
        raise RuntimeError("oracle expects exactly one detector, got %d"
                           % len(dets))
    p = dets[0]
    gen = d[f"{p}/generation"]
    scat = d[f"{p}/scattered"]
    keep = (gen == 0) & (scat == 0)
    return (d[f"{p}/birth_pos"][keep], d[f"{p}/pos"][keep],
            d[f"{p}/dir"][keep], meta["detectors"][p])


# --------------------------------------------------------------------------
# shared geometric-metric primitives (used identically on both bundles)
# --------------------------------------------------------------------------
def _transverse(pos_m, axis_x_m):
    """Transverse (y,z) offset from the axis, metres. Axis = world +x."""
    return pos_m[:, 1:3]                     # (y, z); axis is +x -> index 0


def best_focus_z_from_bundle(pos_m, dir_m):
    """Best-focus axial position (world x, metres) of a ray bundle given each
    ray's detector-plane position and direction. r(s) transverse = P + s*V,
    V = (dy/dx, dz/dx); minimize sum|P + sV|^2 over axial shift s, then
    z = x_det + s. Pure deterministic geometry -- identical algorithm for the
    MC export and the Optiland trace."""
    P = pos_m[:, 1:3]                        # (N,2) transverse metres
    Vx = dir_m[:, 0]
    V = dir_m[:, 1:3] / Vx[:, None]          # d(transverse)/dx
    denom = float(np.sum(V * V))
    s = -float(np.sum(P * V)) / denom
    x_det = float(np.mean(pos_m[:, 0]))
    return x_det + s


def spot_rms_m(pos_m):
    """RMS radial spot size (metres) about the transverse centroid."""
    P = pos_m[:, 1:3]
    P = P - P.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(P * P, axis=1))))


# --------------------------------------------------------------------------
# the oracle: build Optiland from the same model.json, trace the SAME pupil
# positions the MC run used, and adjudicate against the MC bundle.
# --------------------------------------------------------------------------
def run_oracle(stem, case_dir, rays=30000, seed=42):
    """Full oracle for one scene. Returns a dict of both-engine numbers +
    signed deltas (all lengths in metres unless the key says _mm).

    Optiland is traced at the EXACT SAME per-ray pupil positions the MC run
    recorded (birth_pos), so the two bundles share one input ray set and every
    metric is a clean engine-vs-engine comparison -- the shared functions
    best_focus_z_from_bundle / spot_rms_m run identically on both. (A collinear
    MC-sampling centroid offset ~pupil/sqrt(N) is reproduced by Optiland too,
    since each ray is deterministic, so it cancels in the per-ray residual and
    in both centroid-subtracted spot sizes.)"""
    npz = run_cengine_export(stem, case_dir, rays=rays, seed=seed)
    birth, pos, dirn, det_meta = load_mc_direct_bundle(npz)

    system = ob.load_sequential_system(REPO / "geometry" / stem)
    lam_um = system.wavelength_nm * 1e-3
    det_z_m = system.image_z_m
    rho_m = np.sqrt(birth[:, 1] ** 2 + birth[:, 2] ** 2)
    epd_mm = 2.0 * float(rho_m.max()) * ob.MM_PER_M * (1.0 + 1e-6)
    opt = ob.build_optic(system, epd_mm=epd_mm)
    opt_pos, opt_dir = ob.trace_pupil_world(
        opt, birth[:, 1], birth[:, 2], epd_mm, lam_um, det_z_m)

    # per-ray landing residual (2D world vector) -- the marginal/chief/full-fan
    # landing test, subsuming every field height in the pupil.
    land_resid_m = np.linalg.norm(opt_pos[:, 1:3] - pos[:, 1:3], axis=1)
    spot_mc_m = spot_rms_m(pos)
    spot_opt_m = spot_rms_m(opt_pos)

    # --- best focus ----------------------------------------------------------
    # MC: from its exported (pos, dir) -- its reported ray direction matches the
    #   independent analytic (adjudicated). Optiland: from POSITIONS at TWO
    #   image planes (dz apart), which is CONVENTION-FREE. The reason: Optiland's
    #   *reported* image-surface direction cosines carry an (n_air-1)-scale
    #   artifact in the transverse slope (a reduced/normalized-direction
    #   convention at the image surface), so a direction-derived Optiland focus
    #   is ~4 um long; but Optiland's ACTUAL geometry (its landing at ANY plane)
    #   is exact -- proven both by the machine-precision landing match above and
    #   by the two-plane focus agreeing with the C engine to <1 nm. See the test
    #   docstring's adjudication note. dz is arbitrary: the ray is straight in
    #   image space, so V = d(transverse)/dz is exact for any dz.
    focus_z_mc_m = best_focus_z_from_bundle(pos, dirn)
    dz_m = 1.0e-3
    sys2 = ob.load_sequential_system(REPO / "geometry" / stem)
    sys2.image_z_m = det_z_m + dz_m
    opt2 = ob.build_optic(sys2, epd_mm=epd_mm)
    p2, _ = ob.trace_pupil_world(opt2, birth[:, 1], birth[:, 2], epd_mm,
                                 lam_um, sys2.image_z_m)
    V = (p2[:, 1:3] - opt_pos[:, 1:3]) / dz_m
    P = opt_pos[:, 1:3]
    focus_z_opt_m = det_z_m - float(np.sum(P * V)) / float(np.sum(V * V))

    # afocal detection: near-zero output convergence (beam expander), from the
    # convention-free two-plane slopes.
    afocal = float(np.sqrt(np.mean(np.sum(V * V, axis=1)))) < 1e-6

    r_land_mc_m = np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)
    r_land_opt_m = np.sqrt(opt_pos[:, 1] ** 2 + opt_pos[:, 2] ** 2)
    return {
        "stem": stem, "n_rays": int(len(rho_m)),
        "wavelength_nm": system.wavelength_nm,
        "epd_mm": epd_mm, "det_z_m": det_z_m, "afocal": afocal,
        "paraxial_f2_mm": float(opt.paraxial.f2()),
        "focus_z_mc_m": focus_z_mc_m, "focus_z_opt_m": focus_z_opt_m,
        "focus_delta_m": focus_z_opt_m - focus_z_mc_m,
        "land_resid_max_m": float(land_resid_m.max()),
        "land_resid_rms_m": float(np.sqrt(np.mean(land_resid_m ** 2))),
        "spot_mc_m": spot_mc_m, "spot_opt_m": spot_opt_m,
        "spot_reldiff": abs(spot_opt_m - spot_mc_m) / max(spot_mc_m, 1e-15),
        "mag_mc": float(np.mean(r_land_mc_m / rho_m)),
        "mag_opt": float(np.mean(r_land_opt_m / rho_m)),
    }


SCENES = ["lens_pcx", "lens_dcx", "beam_expander", "camera_triplet"]


def _fmt(r):
    lines = [
        "=== %s  (n=%d direct rays, lambda=%.0f nm) ==="
        % (r["stem"], r["n_rays"], r["wavelength_nm"]),
        "  paraxial f2            : %10.4f mm" % r["paraxial_f2_mm"],
        "  detector plane z       : %10.4f mm" % (r["det_z_m"] * 1e3),
    ]
    if r["afocal"]:
        lines += [
            "  [AFOCAL] output collimated -- focus test skipped",
            "  beam magnification MC  : %10.5f" % r["mag_mc"],
            "  beam magnification Opt : %10.5f" % r["mag_opt"],
            "  magnification delta    : %10.3e" % abs(r["mag_opt"] - r["mag_mc"]),
        ]
    else:
        lines += [
            "  best focus z  MC       : %10.5f mm" % (r["focus_z_mc_m"] * 1e3),
            "  best focus z  Optiland : %10.5f mm" % (r["focus_z_opt_m"] * 1e3),
            "  focus delta            : %10.3e m  (tol 1e-6)"
            % r["focus_delta_m"],
        ]
    lines += [
        "  landing resid   max    : %10.3e m  (tol 1e-6)" % r["land_resid_max_m"],
        "  landing resid   rms    : %10.3e m" % r["land_resid_rms_m"],
        "  spot RMS  MC           : %10.4f um" % (r["spot_mc_m"] * 1e6),
        "  spot RMS  Optiland     : %10.4f um" % (r["spot_opt_m"] * 1e6),
        "  spot rel. diff         : %10.3e     (tol 1e-4)" % r["spot_reldiff"],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import tempfile
    if not mc_available():
        print("MC side unavailable (optics python or C engine missing) -- "
              "cannot run oracle.")
        sys.exit(1)
    rays = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    with tempfile.TemporaryDirectory(prefix="optiland_oracle_") as td:
        for stem in SCENES:
            r = run_oracle(stem, Path(td) / stem, rays=rays)
            print(_fmt(r))
            print()
