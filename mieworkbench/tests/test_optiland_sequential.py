# =============================================================================
# test_optiland_sequential.py -- P4b SEQUENTIAL-MODE bridge + evaluator tests
# (engine3.md Sec 5/8), the tier beyond the P4a on-axis oracle. Runs under
# env/bin/python (the GUI venv, which has Optiland + numpy):
#   QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
#       mieworkbench/tests/test_optiland_sequential.py -q
#
# Covers the P4b bridge extensions -- conics/aspheres, real aperture stops
# (entrance-pupil/vignetting), fold-free on-axis mirrors, Q-Forbes as an
# explicit BridgeUnsupported -- the deterministic geometric merit evaluator
# (spot RMS / best focus / geometric encircled energy), and the two headline
# gates: (b) sequential-vs-MC spot-RMS agreement and (c) the end-to-end DLS
# curvature -> focal-length recovery, run through the REAL
# optimize.OptimizationEngine (algorithm='dls') with the bridge as the merit.
# The MC-comparison cases shell out to the optics env's C engine (gated).
# =============================================================================
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optiland")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from raytracer import optiland_bridge as ob   # noqa: E402
import optimize                                # noqa: E402  (stdlib merit engine)
import common  # noqa: E402

OPTICS_PY = common.OPTICS_PYTHON
CENGINE = common.CENGINE_BINARY or str(REPO / "cengine" / "build" / "miewb-trace")


def _geo(stem):
    return REPO / "geometry" / stem


def _has(stem):
    return (_geo(stem) / "model.json").exists()


def _mc_available():
    return OPTICS_PY and Path(OPTICS_PY).exists() and Path(CENGINE).exists()


# --------------------------------------------------------------------------
# conics / aspheres
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has("lens_asphere"), reason="geometry/lens_asphere absent")
def test_asphere_maps_to_even_asphere_and_focuses():
    """The lens_asphere singlet (BK7 conic+A4 asphere front, plano back) maps
    to an Optiland even-asphere; its paraxial focal length matches the scene's
    design EFL (40 mm, make_test_scenes) to 0.5%."""
    s = ob.load_sequential_system(_geo("lens_asphere"))
    front = s.surfaces[0]
    assert front.surface_kind == "asphere"
    assert front.conic == pytest.approx(-1.0, abs=1e-9)     # parabola-ish
    assert len(front.coeffs_si) == 1                        # one A4 term
    f2 = float(ob.build_optic(s).paraxial.f2())
    assert abs(f2 - 40.0) / 40.0 < 5e-3, f2


def test_coeffs_si_to_mm_sag_matches_mieworkbench():
    """The SI->mm even-asphere coefficient conversion reproduces the
    MieWorkbench sag convention (conic + sum A_i r^(4+2i)) to <1 nm at the
    validity radius -- the [0.0] r^2 slot aligns A4,A6,... onto Optiland's
    r^4,r^6,... indexing."""
    from optiland.geometries import EvenAsphere
    from optiland.coordinate_system import CoordinateSystem
    R_m, k = 0.0206033, -1.0
    coeffs_si = [6586.562]            # A4, SI (m^-3), as lens_asphere carries
    r_max_m = 0.01
    # MieWorkbench analytic sag (metres)
    c = 1.0 / R_m
    for r_m in (0.002, 0.006, r_max_m):
        beta = (1.0 + k) * c * c
        zc = c * r_m * r_m / (1.0 + np.sqrt(1.0 - beta * r_m * r_m))
        poly = sum(a * r_m ** (4 + 2 * i) for i, a in enumerate(coeffs_si))
        sag_mw_mm = (zc + poly) * ob.MM_PER_M
        geo = EvenAsphere(CoordinateSystem(), radius=R_m * ob.MM_PER_M, conic=k,
                          coefficients=ob._coeffs_si_to_mm(coeffs_si))
        sag_opt_mm = float(geo.sag(0.0, r_m * ob.MM_PER_M))
        assert abs(sag_opt_mm - sag_mw_mm) < 1e-6, (r_m, sag_opt_mm, sag_mw_mm)


# --------------------------------------------------------------------------
# aperture stop: entrance pupil + vignetting (task item 1)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has("camera_triplet"),
                    reason="geometry/camera_triplet absent")
def test_stop_detected_and_dropped_paths():
    """model_stop=True detects the aluminium iris bore as a real stop (an
    Optiland float-by-stop aperture); model_stop=False keeps the P4a oracle
    behaviour (stop dropped, exactly the six glass surfaces)."""
    st = ob.load_sequential_system(_geo("camera_triplet"), model_stop=True)
    assert st.stop is not None
    assert st.stop["body_label"] == "Stop_iris"
    assert st.stop["semidiameter_m"] == pytest.approx(0.00347, abs=1e-5)
    s0 = ob.load_sequential_system(_geo("camera_triplet"), model_stop=False)
    assert s0.stop is None
    assert len(s0.surfaces) == 6          # oracle path unchanged


@pytest.mark.skipif(not _has("camera_triplet"),
                    reason="geometry/camera_triplet absent")
def test_stop_shrinks_entrance_pupil_and_spot():
    """The stop is load-bearing: WITH it the entrance pupil (EPD ~8.3 mm) and
    spot are far smaller than WITHOUT it (source-filled 14 mm) -- an optimizer
    that ignored the stop would design against the wrong (huge) pupil."""
    with_stop = ob.evaluate_geometry(
        ob.load_sequential_system(_geo("camera_triplet"), model_stop=True),
        n_rays=8000)
    no_stop = ob.evaluate_geometry(
        ob.load_sequential_system(_geo("camera_triplet"), model_stop=False),
        n_rays=8000)
    assert with_stop["epd_mm"] < 0.7 * no_stop["epd_mm"]
    assert with_stop["spot_rms_m"] < 0.5 * no_stop["spot_rms_m"]


@pytest.mark.slow
@pytest.mark.skipif(not (_has("camera_triplet") and _mc_available()),
                    reason="camera_triplet cache / C engine unavailable")
def test_stop_vignetting_matches_cengine():
    """Vignetting parity: the modelled stop's entrance-pupil RADIUS matches the
    MC engine's ACTUAL surviving-ray pupil radius (the iris clip) to <3%, and
    the geometric spot matches the MC direct-bundle spot to <10%."""
    birth, pos = _mc_direct_bundle("camera_triplet")
    rho_mm = np.sqrt(birth[:, 1] ** 2 + birth[:, 2] ** 2) * ob.MM_PER_M
    mc_pupil_radius_mm = float(rho_mm.max())
    P = pos[:, 1:3] - pos[:, 1:3].mean(0)
    mc_spot_um = float(np.sqrt(np.mean(np.sum(P * P, 1)))) * 1e6

    s = ob.load_sequential_system(_geo("camera_triplet"), model_stop=True)
    m = ob.evaluate_geometry(s, n_rays=20000)
    seq_pupil_radius_mm = m["epd_mm"] / 2.0
    seq_spot_um = m["spot_rms_m"] * 1e6
    assert abs(seq_pupil_radius_mm - mc_pupil_radius_mm) / mc_pupil_radius_mm \
        < 0.03, (seq_pupil_radius_mm, mc_pupil_radius_mm)
    assert abs(seq_spot_um - mc_spot_um) / mc_spot_um < 0.10, \
        (seq_spot_um, mc_spot_um)


# --------------------------------------------------------------------------
# mirrors (fold-free on-axis) + rejection of the unsupported cases
# --------------------------------------------------------------------------
def test_single_on_axis_mirror_builds():
    """A synthetic single concave on-axis mirror (R=-200 mm) builds a valid
    reflective Optiland system: |f2| ~ R/2 = 100 mm (the thickness sign flips
    after the reflection)."""
    surf = ob.OpticalSurface(
        vertex_x_m=0.1, radius_m=-0.2, is_flat=False, body_label="M",
        material_after="air", is_entry=False, surface_kind="sphere",
        is_mirror=True)
    sysm = ob.SequentialSystem([surf], image_z_m=0.0, wavelength_nm=633.0,
                               ambient="air", stem="mirror", det_label="D",
                               source_semidiameter_m=0.005, n_mirrors=1)
    opt = ob.build_optic(sysm)
    f2 = abs(float(opt.paraxial.f2()))
    assert 90.0 < f2 < 110.0, f2


@pytest.mark.skipif(not _has("schmidt_cassegrain"),
                    reason="geometry/schmidt_cassegrain absent")
def test_multi_mirror_double_pass_unsupported():
    with pytest.raises(ob.BridgeUnsupported):
        ob.load_sequential_system(_geo("schmidt_cassegrain"))


@pytest.mark.skipif(not _has("newtonian"), reason="geometry/newtonian absent")
def test_folded_mirror_unsupported():
    with pytest.raises(ob.BridgeUnsupported):
        ob.load_sequential_system(_geo("newtonian"))


# --------------------------------------------------------------------------
# Q-Forbes: explicit, documented BridgeUnsupported (not a silent wrong answer)
# --------------------------------------------------------------------------
def test_qforbes_face_unsupported():
    body = {"label": "QF", "bbox_m": {"min": [0, -1, -1], "max": [1, 1, 1]},
            "faces": [{"id": "F1", "surface": {"type": "qforbes",
                                               "kind": "qbfs"}}]}
    with pytest.raises(ob.BridgeUnsupported):
        ob._optical_faces(body)


# --------------------------------------------------------------------------
# the geometric merit block is finite and well-formed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stem", ["lens_pcx", "lens_dcx", "beam_expander"])
def test_evaluate_geometry_finite(stem):
    if not _has(stem):
        pytest.skip("geometry/%s absent" % stem)
    m = ob.evaluate_geometry(ob.load_sequential_system(_geo(stem)),
                             n_rays=2000)
    for k in ("spot_rms_m", "focus_shift_m", "ee_radius_m", "paraxial_f2_mm",
              "epd_mm"):
        assert np.isfinite(m[k]), (stem, k, m[k])
    assert m["spot_rms_m"] >= 0.0 and m["ee_radius_m"] >= 0.0


# --------------------------------------------------------------------------
# GATE (c): end-to-end DLS curvature -> focal-length recovery
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has("lens_pcx"), reason="geometry/lens_pcx absent")
def test_dls_recovers_curvature_and_focal_length():
    """Perturb the PCX front curvature by +5%, refocus with the DLS optimizer
    on the sequential backend, and recover the design focal length to <0.1% in
    well under 30 s. Drives the REAL optimize.OptimizationEngine(algorithm=
    'dls') -- the same driver + operand residuals the CLI uses -- with the
    Optiland bridge as the injected merit."""
    import time
    base = ob.load_sequential_system(_geo("lens_pcx"))
    design_R1_mm = base.surfaces[0].radius_m * ob.MM_PER_M
    efl_design = float(ob.build_optic(base).paraxial.f2())
    # place the detector at the design best focus so min-spot <=> in-focus
    focus_z = base.image_z_m + ob.evaluate_geometry(base, n_rays=4000)[
        "focus_shift_m"]

    def sys_with_R1(R1_mm):
        s = copy.deepcopy(base)
        s.surfaces[0].radius_m = R1_mm / ob.MM_PER_M
        s.image_z_m = focus_z
        return s

    def evaluate_fn(params):
        m = ob.evaluate_geometry(sys_with_R1(params["R_front"]), n_rays=4000)
        um = m["spot_rms_m"] * 1e6
        return {"backend_used": "sequential",
                "detectors": {base.det_label: {"spot": [
                    {"rms_radius_um": um, "rms_pw_radius_um": um,
                     "n_rays": m["n_rays"]}]}}}

    eng = optimize.OptimizationEngine(
        [{"name": "R_front", "start": design_R1_mm * 1.05,
          "lo": design_R1_mm * 0.9, "hi": design_R1_mm * 1.1}],
        [{"operand": "spot_rms", "detector": None, "target": 0.0,
          "weight": 1.0}],
        evaluate_fn, algorithm="dls", budget=40, tol=1e-8, progress=False)
    t0 = time.monotonic()
    best = eng.run()
    wall = time.monotonic() - t0

    R1_rec = best["params"]["R_front"]
    efl_rec = float(ob.build_optic(sys_with_R1(R1_rec)).paraxial.f2())
    rel = abs(efl_rec - efl_design) / efl_design
    print("\nDLS recovery: R1 %.5f->%.5f mm (design %.5f), EFL %.5f mm, "
          "err %.4f%%, %d evals, %.2fs"
          % (design_R1_mm * 1.05, R1_rec, design_R1_mm, efl_rec, rel * 100,
             eng.n_evals, wall))
    assert rel < 1e-3, (efl_rec, efl_design, rel)
    assert wall < 30.0, wall


# --------------------------------------------------------------------------
# GATE (b): sequential spot RMS matches the MC direct-bundle spot
# --------------------------------------------------------------------------
def _mc_direct_bundle(stem, rays=200000, seed=7):
    """Run the C engine and return (birth_pos, detector pos) of the DIRECT
    (gen-0, unscattered) geometric image-forming rays -- the SAME bundle the
    P4a oracle adjudicates. The report.json all-ray spot additionally folds in
    Fresnel ghosts + scatter, which the sequential trace deliberately omits."""
    cd = tempfile.mkdtemp(prefix="seq_mc_")
    subprocess.run(
        [OPTICS_PY, str(REPO / "scripts" / "run_trace.py"),
         "--model-json", str(_geo(stem) / "model.json"), "--case-dir", cd,
         "--rays", str(int(rays)), "--nlambda", "1", "--resolution", "128",
         "--seeds", "1", "--seed0", str(seed), "--engine", "c",
         "--export-rays"],
        capture_output=True, env=dict(os.environ, MIEWB_CENGINE=CENGINE),
        timeout=600)
    import json
    d = np.load(Path(cd) / "rays_full.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    p = list(meta["detectors"].keys())[0]
    keep = (d[f"{p}/generation"] == 0) & (d[f"{p}/scattered"] == 0)
    return d[f"{p}/birth_pos"][keep], d[f"{p}/pos"][keep]


@pytest.mark.slow
@pytest.mark.parametrize("stem", ["lens_pcx", "lens_dcx"])
def test_sequential_spot_matches_cengine(stem):
    """The sequential (Optiland) spot RMS at the detector matches the MC
    engine's direct-bundle spot RMS to <2% on the two clean singlets -- the
    arbiter in action (a second physics truth agreeing with the first)."""
    if not (_has(stem) and _mc_available()):
        pytest.skip("geometry/%s cache or C engine unavailable" % stem)
    _birth, pos = _mc_direct_bundle(stem)
    P = pos[:, 1:3] - pos[:, 1:3].mean(0)
    mc_um = float(np.sqrt(np.mean(np.sum(P * P, 1)))) * 1e6
    seq_um = ob.evaluate_geometry(
        ob.load_sequential_system(_geo(stem)), n_rays=20000)["spot_rms_m"] * 1e6
    rel = abs(seq_um - mc_um) / mc_um
    print("\n%s: sequential %.2f um vs MC-direct %.2f um (%.2f%%)"
          % (stem, seq_um, mc_um, rel * 100))
    assert rel < 0.02, (stem, seq_um, mc_um, rel)
