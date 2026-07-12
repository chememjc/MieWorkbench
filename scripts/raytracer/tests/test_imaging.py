# =============================================================================
# test_imaging.py — exit-pupil/chief-ray stage (raytracer/analysis_imaging.py)
# + the --imaging-products post renderers, validated against SYNTHETIC data
# (no tracing, no FreeCAD, matching test_post_analysis.py's approach):
#
#   * exit_pupil_center: chief lines synthesized through a known point (+
#     noise) -> recovered; single-source / parallel-line degeneracies ->
#     (None, reason).
#   * chief_ray: exact-through-E ray found + refined; telecentric bundles
#     (chief parallel to the detector normal) -> CRA ~ 0.
#   * grid_distortion oracle: landings h = f*tan(theta)*(1 + k*(t/tmax)^2)
#     -> the recovered distortion % matches the injected k (exact algebra
#     for the rows; the poly fit recovers k to ~1%).
#   * best_focus_scan: a synthetic astigmatic bundle (T and S sub-fans
#     aimed at two different focal planes) -> z_T, z_S recovered within 1%
#     (the scan is closed-form, so in practice exactly).
#   * strehl_psf_peak: zero OPD -> 1.0; Marechal agreement for small
#     random OPD (sigma ~ lam/50): |S_psf - S_marechal| < 0.01.
#   * render_imaging_products end-to-end on a hand-built rays_full.npz +
#     case/model dicts (perfect-imaging fan through a known pupil E):
#     PNGs, CSVs, report blocks, annotation-driven field angles, and the
#     SystemExit gate naming --export-rays when the npz is missing.
#   * Petzval end-to-end (slow): real trace of the extracted lens_pcx
#     singlet with a 3-bundle field fan laid out by core.wizards.
#     design_field_fan -> field-curvature sign + rough magnitude vs the
#     Petzval formula.
#
# Run: /home3/optics/env/bin/python -m pytest \
#          scripts/raytracer/tests/test_imaging.py -q
# =============================================================================
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))

from raytracer import analysis_imaging as ai               # noqa: E402


# ---------------------------------------------------------------------------
# exit_pupil_center
# ---------------------------------------------------------------------------
def test_exit_pupil_center_recovers_known_point():
    rng = np.random.default_rng(11)
    E_true = np.array([0.012, -0.004, 0.0035])
    rays = []
    for _ in range(6):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t = rng.uniform(0.02, 0.08)
        p = E_true + t * d + rng.normal(0.0, 1e-6, size=3)  # 1 um jitter
        rays.append((p, d))
    E, reason = ai.exit_pupil_center(rays)
    assert reason is None
    assert np.linalg.norm(E - E_true) < 1e-4


def test_exit_pupil_center_single_source_returns_none_with_reason():
    E, reason = ai.exit_pupil_center([(np.zeros(3),
                                       np.array([0.0, 0.0, 1.0]))])
    assert E is None
    assert "2 field bundles" in reason


def test_exit_pupil_center_parallel_lines_returns_none_with_reason():
    d = np.array([0.0, 0.0, 1.0])
    rays = [(np.array([x, 0.0, 0.0]), d) for x in (0.0, 0.001, 0.002)]
    E, reason = ai.exit_pupil_center(rays)
    assert E is None
    assert "parallel" in reason


# ---------------------------------------------------------------------------
# field bundles + chief rays
# ---------------------------------------------------------------------------
def _bundle_through(E, landing, n=200, spread=1e-3, seed=0, opl0=0.1,
                    land_jitter=None):
    """A synthetic converging bundle: rays from pupil points around E to
    landing points around `landing` (the exact E->landing ray included as
    index 0). land_jitter defaults to spread/20. Returns the group dict
    analysis_imaging consumes."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E, dtype=np.float64)
    landing = np.asarray(landing, dtype=np.float64)
    if land_jitter is None:
        land_jitter = spread / 20.0
    pup = E + np.concatenate([np.zeros((1, 3)),
                              rng.normal(0.0, spread, size=(n - 1, 3))])
    land = landing + np.concatenate(
        [np.zeros((1, 3)), rng.normal(0.0, land_jitter,
                                      size=(n - 1, 3))])
    d = land - pup
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return {"pos": land, "dir": d, "opl": np.full(n, opl0),
            "lam": np.full(n, 633e-9), "power": np.ones(n)}


def test_field_groups_and_centroid_ray():
    g0 = _bundle_through([0, 0, -0.05], [0, 0, 0], n=50, seed=1)
    g1 = _bundle_through([0, 0, -0.05], [0.005, 0, 0], n=70, seed=2)
    cols = {k: np.concatenate([g0[k], g1[k]]) for k in g0}
    cols["source_id"] = np.concatenate([np.zeros(50, dtype=np.int16),
                                        np.ones(70, dtype=np.int16)])
    groups = ai.field_groups(cols, min_rays=60)
    assert list(groups) == [1]                     # 50-ray group filtered
    groups = ai.field_groups(cols)
    assert sorted(groups) == [0, 1]
    c, d = ai.centroid_ray(groups[1])
    assert np.linalg.norm(c - [0.005, 0, 0]) < 1e-4
    exp = np.array([0.005, 0, 0.05])
    exp /= np.linalg.norm(exp)
    assert np.dot(d, exp) > 0.9999


def test_chief_ray_refines_toward_exact_center_ray():
    E = np.array([0.0, 0.0, -0.05])
    landing = np.array([0.004, -0.002, 0.0])
    g = _bundle_through(E, landing, n=400, seed=3)
    chief = ai.chief_ray(g, E, normal=[0.0, 0.0, 1.0])
    assert chief["method"] == "exit_pupil"
    assert np.linalg.norm(chief["landing"] - landing) < 1e-4
    exp = landing - E
    exp = exp / np.linalg.norm(exp)
    assert np.dot(chief["dir"], exp) > 0.99999
    exp_cra = math.degrees(math.acos(abs(exp[2])))
    assert abs(chief["cra_deg"] - exp_cra) < 0.05


def test_chief_ray_centroid_fallback_and_telecentric_cra_zero():
    # image-side telecentric: every incoming direction parallel to the
    # detector normal -> the pupil solve degenerates upstream (E=None)
    # and the centroid chief reports CRA ~ 0 exactly.
    n = 100
    rng = np.random.default_rng(4)
    g = {"pos": np.column_stack([rng.normal(0, 1e-3, n),
                                 rng.normal(0, 1e-3, n),
                                 np.zeros(n)]),
         "dir": np.tile([0.0, 0.0, 1.0], (n, 1)),
         "opl": np.zeros(n), "lam": np.full(n, 550e-9),
         "power": np.ones(n)}
    chief = ai.chief_ray(g, None, normal=[0.0, 0.0, 1.0])
    assert chief["method"] == "centroid"
    assert chief["cra_deg"] < 1e-9


# ---------------------------------------------------------------------------
# pupil plane / coordinates / OPD
# ---------------------------------------------------------------------------
def test_pupil_plane_and_coords_span_unit_disc():
    # collimated on-axis bundle along +z with a uniform disc footprint:
    # the plane radius estimate (RMS*sqrt(2)) recovers the disc radius and
    # the normalized coords span ~ the unit disc.
    r0 = 2e-3
    xs = np.linspace(-1, 1, 41)
    X, Y = np.meshgrid(xs, xs)
    keep = X ** 2 + Y ** 2 <= 1.0
    x, y = X[keep] * r0, Y[keep] * r0
    n = len(x)
    g = {"pos": np.column_stack([x, y, np.zeros(n)]),
         "dir": np.tile([0.0, 0.0, 1.0], (n, 1)),
         "opl": np.zeros(n), "lam": np.full(n, 633e-9),
         "power": np.ones(n)}
    E = np.array([0.0, 0.0, -0.04])
    plane = ai.pupil_plane(E, g)
    assert plane["radius_m"] == pytest.approx(r0, rel=0.03)
    xy, ok = ai.pupil_coords(g, plane)
    assert np.all(ok)
    rho = np.hypot(xy[:, 0], xy[:, 1])
    assert rho.max() == pytest.approx(r0 / plane["radius_m"], rel=1e-9)
    assert rho.max() < 1.1


def test_opd_exit_pupil_matches_r0_form():
    E = np.array([0.0, 0.0, -0.05])
    g = _bundle_through(E, [0.001, 0.0, 0.0], n=50, seed=5, opl0=0.2)
    chief = ai.chief_ray(g, E, normal=[0, 0, 1])
    opd = ai.opd_exit_pupil(g, chief, E=E)
    manual = (g["opl"]
              + np.linalg.norm(g["pos"] - chief["landing"], axis=-1)
              - chief["opl"])
    assert np.allclose(opd, manual)


# ---------------------------------------------------------------------------
# grid distortion oracle
# ---------------------------------------------------------------------------
def test_grid_distortion_recovers_injected_polynomial():
    f = 0.050
    k = 0.05                                 # +5% at the field edge
    thetas = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
    t = np.tan(np.radians(thetas))
    h = f * t * (1.0 + k * (t / t[-1]) ** 2)
    gd = ai.grid_distortion(thetas, h)

    # f_eff is calibrated on the innermost point (documented practice)
    x1 = t[0] / t[-1]
    assert gd["f_eff_m"] == pytest.approx(f * (1.0 + k * x1 ** 2),
                                          rel=1e-12)
    # each row's distortion % matches the exact algebra of the injected
    # model relative to that calibration
    for (th, hh, h_ref, pct), ti in zip(gd["rows"], t):
        x = ti / t[-1]
        expect = 100.0 * ((1.0 + k * x ** 2) / (1.0 + k * x1 ** 2) - 1.0)
        assert pct == pytest.approx(expect, abs=1e-9)
    # and the fitted quadratic coefficient recovers k (as a fraction)
    r_max, ks = gd["poly"]
    assert ks[0] == pytest.approx(k, rel=0.02, abs=1e-4)


def test_grid_distortion_rejects_bad_input():
    with pytest.raises(ValueError):
        ai.grid_distortion([], [])
    with pytest.raises(ValueError):
        ai.grid_distortion([0.0, 10.0], [0.0, 0.01])   # axis point included


# ---------------------------------------------------------------------------
# best_focus_scan — synthetic astigmatic bundle
# ---------------------------------------------------------------------------
def _astig_bundle(z_t, z_s, n_half=40, r=1.5e-3):
    """T rays (landing on the y axis) converge to (0,0,z_t); S rays
    (landing on the x axis) converge to (0,0,z_s). Detector plane z=0,
    normal +z. On-axis chief along +z."""
    y = np.concatenate([np.linspace(-r, -r / 10, n_half),
                        np.linspace(r / 10, r, n_half)])
    x = y.copy()
    pos_t = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])
    dir_t = np.column_stack([np.zeros_like(y), -y,
                             np.full_like(y, z_t)])
    pos_s = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    dir_s = np.column_stack([-x, np.zeros_like(x),
                             np.full_like(x, z_s)])
    pos = np.concatenate([pos_t, pos_s])
    d = np.concatenate([dir_t, dir_s])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = len(pos)
    return {"pos": pos, "dir": d, "opl": np.zeros(n),
            "lam": np.full(n, 550e-9), "power": np.ones(n)}


def test_best_focus_scan_recovers_tangential_and_sagittal_foci():
    z_t_true, z_s_true = 0.005, 0.008
    g = _astig_bundle(z_t_true, z_s_true)
    chief = {"landing": np.zeros(3), "dir": np.array([0.0, 0.0, 1.0]),
             "opl": 0.0}
    scan = ai.best_focus_scan(g, chief, normal=[0.0, 0.0, 1.0])
    assert scan["z_t_m"] == pytest.approx(z_t_true, rel=0.01)
    assert scan["z_s_m"] == pytest.approx(z_s_true, rel=0.01)
    assert scan["astig_m"] == pytest.approx(z_t_true - z_s_true, rel=0.02)
    assert scan["rms_t_m"] < 1e-9 and scan["rms_s_m"] < 1e-9
    assert scan["n_t"] == scan["n_s"] == 80


def test_best_focus_scan_negative_defocus_sign():
    # bundle that CONVERGED before the detector (focus at z < 0): rays
    # through F=(0,0,-z0) diverging onto the plane
    z0 = 0.004
    g = _astig_bundle(-z0, -z0)
    chief = {"landing": np.zeros(3), "dir": np.array([0.0, 0.0, 1.0]),
             "opl": 0.0}
    scan = ai.best_focus_scan(g, chief, normal=[0.0, 0.0, 1.0])
    assert scan["z_t_m"] == pytest.approx(-z0, rel=0.01)
    assert scan["z_s_m"] == pytest.approx(-z0, rel=0.01)


def test_best_focus_scan_z_range_clamps():
    g = _astig_bundle(0.005, 0.008)
    chief = {"landing": np.zeros(3), "dir": np.array([0.0, 0.0, 1.0]),
             "opl": 0.0}
    scan = ai.best_focus_scan(g, chief, normal=[0.0, 0.0, 1.0],
                              z_range=(-0.001, 0.001))
    assert scan["z_t_m"] == 0.001 and scan["z_s_m"] == 0.001


# ---------------------------------------------------------------------------
# strehl_psf_peak
# ---------------------------------------------------------------------------
def test_strehl_psf_peak_unity_at_zero_opd():
    n = 500
    opd = np.zeros(n)
    s = ai.strehl_psf_peak(None, opd, np.ones(n), 633e-9)
    assert s == pytest.approx(1.0, abs=1e-12)


def test_strehl_psf_peak_agrees_with_marechal_for_small_opd():
    rng = np.random.default_rng(7)
    lam = 633e-9
    opd = rng.normal(0.0, lam / 50.0, size=4000)
    opd -= opd.mean()                       # piston removal
    sigma = float(np.std(opd))
    s_psf = ai.strehl_psf_peak(None, opd, np.ones_like(opd), lam)
    s_mar = math.exp(-(2.0 * math.pi * sigma / lam) ** 2)
    assert abs(s_psf - s_mar) < 0.01


# ---------------------------------------------------------------------------
# render_imaging_products end-to-end on synthetic artifacts
# ---------------------------------------------------------------------------
def _synthetic_imaging_case(case_dir, thetas_deg=(0.0, 8.0, 16.0),
                            f=0.050, label="Det.Pad.Face1", n=300):
    """Perfect-imaging fan: pupil at E=(0,0,-f), bundle s lands exactly at
    h = f*tan(theta) on the +y axis of the z=0 detector plane. Writes
    rays_full.npz and returns (case, model, E)."""
    import post_process  # noqa: F401  (imported here to keep module load light)
    safe = label.replace(".", "_")
    E = np.array([0.0, 0.0, -f])
    payload = {}
    pos_all, dir_all, sid_all, opl_all = [], [], [], []
    for s, th in enumerate(thetas_deg):
        h = f * math.tan(math.radians(th))
        # zero landing jitter: a PERFECT imaging bundle (every ray of
        # bundle s lands exactly at its image point) — a jittered landing
        # would be a genuinely aberrated wave, contradicting the perfect-
        # imaging oracles below (E position, ~0 distortion, ~0 focus
        # shift, Strehl ~ 1)
        g = _bundle_through(E, [0.0, h, 0.0], n=n, seed=100 + s,
                            land_jitter=0.0)
        pos_all.append(g["pos"])
        dir_all.append(g["dir"])
        sid_all.append(np.full(n, s, dtype=np.int16))
        # ABERRATION-FREE wavefront by construction: every ray's total
        # OPL to the bundle's image point is the same constant (with the
        # landing AT that point, opl is simply constant), so the
        # exit-pupil OPD comes out identically zero
        opl_all.append(0.2 - np.linalg.norm(
            g["pos"] - np.array([0.0, h, 0.0])[None, :], axis=-1))
    ntot = n * len(thetas_deg)
    pos_cat = np.concatenate(pos_all)
    dir_cat = np.concatenate(dir_all)
    payload["%s/pos" % safe] = pos_cat
    payload["%s/dir" % safe] = dir_cat
    payload["%s/source_id" % safe] = np.concatenate(sid_all)
    payload["%s/opl" % safe] = np.concatenate(opl_all)
    # birth positions = the pupil points (finite everywhere)
    pup_pts = pos_cat - dir_cat * np.linalg.norm(
        pos_cat - E[None, :], axis=-1, keepdims=True)
    payload["%s/lam" % safe] = np.full(ntot, 550e-9)
    payload["%s/power" % safe] = np.ones(ntot)
    # the columns render_wavefront additionally needs (strata/coherent/
    # birth positions — birth = the pupil points, all finite)
    payload["%s/lam_stratum" % safe] = np.zeros(ntot, dtype=np.int16)
    payload["%s/pol_stratum" % safe] = np.zeros(ntot, dtype=np.int16)
    payload["%s/coherent" % safe] = np.ones(ntot, dtype=bool)
    payload["%s/birth_pos" % safe] = pup_pts
    meta = {"seed": 42, "model": "synthetic", "max_reflections": 6,
            "export_rays_max": 2000000,
            "detectors": {safe: {
                "label": label, "xhat": [1.0, 0.0, 0.0],
                "yhat": [0.0, 1.0, 0.0], "normal": [0.0, 0.0, 1.0],
                "x_lo": 0.0, "y_lo": 0.0, "n_total": ntot, "n_kept": ntot,
                "kept_fraction": 1.0}}}
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(case_dir / "rays_full.npz", **payload)

    names = ["Axis", "FieldP", "FieldM"][:len(thetas_deg)]
    powers = [1.0, 0.8, 0.6][:len(thetas_deg)]
    case = {
        "sources": names,
        "options": {"rays": float(n)},        # everything detected here
        "detected": {"seed42": {label: {
            "%d/0/0" % s: {"coherent_W": 0.0, "incoherent_W": powers[s],
                           "n_samples": n}
            for s in range(len(thetas_deg))}}},
    }
    # the annotation path: field_angle_deg present in the source dicts
    model = {"bodies": [
        {"label": nm, "name": nm, "role": "source",
         "source": {"power_mW": 1.0, "lambdac_nm": 550.0,
                    "emit_face": "%s.Pad.Face1" % nm,
                    "field_angle_deg": th},
         "faces": []}
        for nm, th in zip(names, thetas_deg)]}
    return case, model, E


def test_render_imaging_products_end_to_end(tmp_path):
    import post_process
    label = "Det.Pad.Face1"
    safe = label.replace(".", "_")
    thetas = (0.0, 8.0, 16.0)
    f = 0.050
    case, model, E = _synthetic_imaging_case(tmp_path, thetas, f=f,
                                             label=label)
    report = {"detectors": {}}
    csv_emitter = post_process.CsvEmitter(tmp_path / "data")
    post_process.render_imaging_products(
        tmp_path, case, model, report,
        ("distortion", "vignetting", "field_curves", "telecentricity"),
        csv_emitter)

    adir = tmp_path / "analysis"
    for prod in ("distortion", "vignetting", "field_curves",
                 "telecentricity"):
        assert (adir / ("imaging_%s_%s.png" % (prod, safe))).exists()
        assert (tmp_path / "data"
                / ("imaging_%s_%s.csv" % (prod, safe))).exists()

    block = report["detectors"][label]["imaging"]
    assert block["field_points"] == 3
    assert block["axis_source"] == "Axis"
    assert block["exit_pupil"]["ok"]
    # the centroid-ray direction is a nonlinear average over the 1 mm 3-D
    # pupil scatter, which biases the small-angle z triangulation by
    # ~0.3 mm on the 50 mm pupil distance — gate at 0.5 mm (1%)
    assert np.allclose(block["exit_pupil"]["center_mm"], E * 1e3,
                       atol=0.5)

    # distortion: perfect h = f*tan(theta) fan -> ~0 % everywhere and
    # f_eff ~ f
    dist = block["distortion"]
    assert dist["f_eff_mm"] == pytest.approx(f * 1e3, rel=0.01)
    for row in dist["rows"]:
        assert abs(row["distortion_pct"]) < 0.5

    # vignetting: normalized to the axis source's detected power
    vig = {r["source"]: r for r in block["vignetting"]["rows"]}
    assert vig["Axis"]["rel_illumination"] == pytest.approx(1.0)
    assert vig["FieldP"]["rel_illumination"] == pytest.approx(0.8)
    assert vig["FieldM"]["rel_illumination"] == pytest.approx(0.6)
    assert vig["Axis"]["ray_survival_frac"] == pytest.approx(1.0)

    # field curves: every bundle focuses ON the detector -> z ~ 0
    for row in block["field_curves"]["rows"]:
        assert abs(row["z_t_mm"]) < 0.1
        assert abs(row["z_s_mm"]) < 0.1

    # telecentricity: CRA equals the chief geometry angle
    tel = {r["source"]: r for r in block["telecentricity"]["rows"]}
    assert tel["Axis"]["cra_deg"] < 0.2
    assert tel["FieldP"]["cra_deg"] == pytest.approx(8.0, abs=0.3)
    assert tel["FieldM"]["cra_deg"] == pytest.approx(16.0, abs=0.3)


def test_render_wavefront_exit_pupil_mode(tmp_path):
    """pupil_mode='exit_pupil' runs the analysis_imaging pupil path:
    block flags the mode, reports strehl_psf_peak alongside
    strehl_marechal, and a near-perfect bundle scores ~1 on both."""
    import post_process
    label = "Det.Pad.Face1"
    case, model, _ = _synthetic_imaging_case(tmp_path, (0.0, 8.0, 16.0),
                                             label=label)
    report = {"detectors": {}}
    post_process.render_wavefront(tmp_path, report, None,
                                  pupil_mode="exit_pupil")
    block = report["detectors"][label]["wavefront"]
    assert block["pupil_mode"] == "exit_pupil"
    assert "pupil_note" not in block
    assert len(block["keys"]) == 3
    for row in block["keys"]:
        assert 0.0 <= row["strehl_psf_peak"] <= 1.0 + 1e-12
        assert 0.0 <= row["strehl_marechal"] <= 1.0 + 1e-12
        # the synthetic bundles carry only tiny jitter: near-perfect
        assert row["strehl_marechal"] > 0.9
        assert row["strehl_psf_peak"] > 0.9
    safe = label.replace(".", "_")
    assert (tmp_path / "analysis" / ("wavefront_%s.png" % safe)).exists()


def test_render_wavefront_exit_pupil_single_source_falls_back(tmp_path):
    import post_process
    label = "Det.Pad.Face1"
    case, model, _ = _synthetic_imaging_case(tmp_path, (0.0,),
                                             label=label)
    report = {"detectors": {}}
    post_process.render_wavefront(tmp_path, report, None,
                                  pupil_mode="exit_pupil")
    block = report["detectors"][label]["wavefront"]
    assert block["pupil_mode"] == "source"        # honest fallback
    assert "2 field bundles" in block["pupil_note"]
    assert "strehl_psf_peak" in block             # reported in both modes


def test_render_imaging_products_requires_export_rays(tmp_path):
    import post_process
    with pytest.raises(SystemExit) as exc:
        post_process.render_imaging_products(
            tmp_path, {"sources": []}, {"bodies": []},
            {"detectors": {}}, ("distortion",), None)
    assert "--export-rays" in str(exc.value)


def test_render_imaging_products_noop_without_products(tmp_path):
    import post_process
    post_process.render_imaging_products(
        tmp_path, {}, {"bodies": []}, {"detectors": {}}, (), None)
    assert not (tmp_path / "analysis").exists()


def test_render_imaging_products_single_source_fallback_note(tmp_path,
                                                             capsys):
    import post_process
    label = "Det.Pad.Face1"
    case, model, _ = _synthetic_imaging_case(tmp_path, (0.0,),
                                             label=label)
    report = {"detectors": {}}
    post_process.render_imaging_products(
        tmp_path, case, model, report, ("telecentricity",), None)
    block = report["detectors"][label]["imaging"]
    assert not block["exit_pupil"]["ok"]
    assert "fallback_reason" in block["exit_pupil"]
    assert "exit-pupil solve fell back" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Petzval end-to-end (slow): lens_pcx singlet + design_field_fan fan
# ---------------------------------------------------------------------------
GEO = REPO / "geometry"
requires_pcx = pytest.mark.skipif(
    not (GEO / "lens_pcx" / "model.json").exists(),
    reason="author+extract lens_pcx (make_test_scenes.py + "
           "extract_geometry.py)")


def _lens_pcx_scene(det_x):
    """geometry/lens_pcx with the Screen translated so its detecting
    plane sits at global x = det_x [m] (same helper as
    test_export_rays.py)."""
    import common
    from raytracer.scene import Scene
    from raytracer.optprops import load_optical_properties
    model = common.load_model(GEO / "lens_pcx" / "model.json")
    dx = None
    for b in model["bodies"]:
        if b.get("role") == "detector":
            xs = [f["surface"]["origin"][0] for f in b["faces"]]
            dx = det_x - min(xs)
            for f in b["faces"]:
                f["surface"]["origin"][0] += dx
                f["trim_polylines_xyz"] = [
                    [[c[0] + dx, c[1], c[2]] for c in loop]
                    for loop in f["trim_polylines_xyz"]]
    assert dx is not None
    common.validate_model(model)
    opt = load_optical_properties()
    return Scene(model, opt.matdb, opt.coatings, optprops=opt,
                 geometry_dir=str(GEO / "lens_pcx")), model


def _fan_bundle(center_mm, direction, source_id, rmax=1.5e-3, m=6000,
                seed=5, lam=633e-9):
    """Collimated disk bundle centered at center_mm [mm], propagating
    along `direction`, tagged source_id (the design_field_fan layout
    driven straight into the tracer)."""
    from raytracer.rays import RayBatch
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    # transverse basis
    t1 = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(t1) < 1e-9:
        t1 = np.cross(d, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d, t1)
    r = rmax * np.sqrt(rng.random(m))
    th = rng.uniform(0, 2 * np.pi, m)
    c = np.asarray(center_mm, dtype=np.float64) * 1e-3
    b = RayBatch(m)
    b.pos[:] = (c[None, :] + r[:, None] * np.cos(th)[:, None] * t1
                + r[:, None] * np.sin(th)[:, None] * t2)
    b.dir[:] = d
    b.s_hat[:] = t2
    b.Es[:] = 1.0
    b.Ep[:] = 1.0
    b.lam[:] = lam
    b.birth_power[:] = b.power
    b.coherent[:] = False
    b.source_id[:] = source_id
    b.birth_pos = b.pos.copy()
    return b


@pytest.mark.slow
@requires_pcx
def test_petzval_field_curvature_lens_pcx():
    """Field curvature of the extracted PCX singlet vs the Petzval
    formula 1/R_p = 1/(n f): the medial focal surface must sag TOWARD
    the lens off-axis (z < 0 along the chief) with |sag| in the right
    ballpark. LOOSE 30% tolerance, deliberately: MC ray statistics, the
    thick-lens/paraxial mismatch, third-order-only theory, and the
    astigmatism split (z_P = (3 z_S - z_T)/2 assumes exact Seidel
    relations with the stop at the thin lens) all stack up — the point
    is the SIGN and the magnitude class, not a precision oracle."""
    import math as _math
    from mieworkbench.core.wizards import design_field_fan
    from raytracer.tracer import Tracer, TraceConfig
    from raytracer.detector import DetectorGrid

    lam = 633e-9
    x_focus = 0.050236                 # paraxial focus (test_export_rays)
    scene, model = _lens_pcx_scene(x_focus)

    # lens prescription from the extracted geometry: front sphere radius
    # + BK7 index -> f = R/(n-1), R_p = n*f
    R_m = None
    for b in model["bodies"]:
        if b.get("role") == "optic":
            for f in b["faces"]:
                if f["surface"]["type"] == "sphere":
                    R_m = float(f["surface"]["radius"])
    assert R_m is not None
    n_idx = float(scene.matdb.get("bk7").n_complex(lam).real)
    f_m = R_m / (n_idx - 1.0)
    R_p = n_idx * f_m                  # |Petzval radius|

    # 3-angle field fan from the WIZARD (arc spacing, aimed at the front
    # vertex), driven straight into the tracer as collimated bundles
    theta = 6.0
    fan = design_field_fan([0.0, theta, -theta],
                           pivot_mm=(0.0, 0.0, 0.0), radius_mm=30.0,
                           aperture_mm=3.0)
    # one trace per field bundle (the scene itself has one source, so the
    # ledger is sized for source_id 0) — the records are re-tagged with
    # the fan index afterward, exactly what a real 3-source scene's
    # rays_full.npz would carry.
    per_key = {k: [] for k in ("pos", "dir", "opl", "lam", "power",
                               "source_id", "generation")}
    for sid, entry in enumerate(fan["sources"]):
        batch = _fan_bundle(entry["pos_mm"], entry["dir"], 0,
                            seed=50 + sid)
        grids = {fid: DetectorGrid(scene.faces[fid], 128, 4,
                                   (500e-9, 750e-9),
                                   label=scene.faces[fid].id)
                 for fid in scene.detector_faces}
        cfg = TraceConfig(rays=len(batch), n_lambda=1, seed=5,
                          power_floor=1e-9, export_rays=True)
        Tracer(scene, cfg, grids).run([batch])
        det = next(iter(grids.values()))
        recs = det.ray_records
        for k in per_key:
            if k == "source_id":
                n_hit = sum(len(r["pos"]) for r in recs)
                per_key[k].append(np.full(n_hit, sid, dtype=np.int16))
            else:
                per_key[k].append(np.concatenate([r[k] for r in recs]))
    normal = det.normal
    cols = {k: np.concatenate(v) for k, v in per_key.items()}
    # primary rays only (etalon ghosts would smear the focus metric)
    g0 = cols["generation"] == 0
    cols = {k: v[g0] for k, v in cols.items()}

    ai_groups = ai.field_groups(cols, min_rays=500)
    assert sorted(ai_groups) == [0, 1, 2]
    cents = [ai.centroid_ray(g) for g in ai_groups.values()]
    E, reason = ai.exit_pupil_center(cents)
    assert reason is None, reason
    chiefs = {s: ai.chief_ray(g, E, normal)
              for s, g in ai_groups.items()}
    c0 = chiefs[0]["landing"]

    sags = []
    for s in (1, 2):
        scan = ai.best_focus_scan(ai_groups[s], chiefs[s], normal)
        h = float(np.linalg.norm(chiefs[s]["landing"] - c0))
        # Petzval surface from the Seidel split z_P = (3 z_S - z_T)/2;
        # expected sag (toward the lens) = -h^2 / (2 R_p)
        z_p = (3.0 * scan["z_s_m"] - scan["z_t_m"]) / 2.0
        expect = -h ** 2 / (2.0 * R_p)
        sags.append((z_p, expect))
        assert z_p < 0.0, ("field must sag toward the lens", s, z_p)
    for z_p, expect in sags:
        assert abs(z_p - expect) < 0.30 * abs(expect), (z_p, expect)
