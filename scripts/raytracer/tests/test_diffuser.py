# =============================================================================
# test_diffuser.py — ground-glass diffusers: the deep-rough limit of the
# validated Beckmann microfacet path, parameterized by RMS slope (grit
# table / explicit slope / registry).
#
# Gates (user-mandated validation scope: the diffuser + what it touches):
#   t1  energy closure < 1e-3 with a diffuser in the scene
#   t2  transmitted angular width vs grit matches the calibration model
#   t3  smooth limit: a vanishing slope reproduces the plain window
#   t4  polarization: single-scatter depolarization leaks through a
#       crossed polarizer (monotonic in slope); unpolarized stays
#       symmetric. NOTE the model is SINGLE-scatter — depolarization is
#       real but weak (documented limit; real ground glass adds multiple
#       scattering).
#   t5  azimuthal symmetry at normal incidence
#   t6  a scene WITHOUT the property builds no scatter entries at all
#       (the property is purely additive; the full engine suite is the
#       bit-identity guard)
# =============================================================================
import numpy as np
import pytest

from raytracer.optprops import load_optical_properties
from raytracer.roughness import (
    GRIT_FWHM_DEG, diffuser_equivalent, slope_for_grit,
    slope_from_sigma_lcorr,
)
from raytracer.scene import Scene
from raytracer.materials import MaterialDB, load_coatings

from . import scenehelpers as sh

N_BK7 = 1.515          # d-line-ish index the grit calibration assumes


@pytest.fixture(scope="module")
def props():
    return load_optical_properties()


def diffuser_scene(value, rays=30000, det_half=0.05, det_x=0.1,
                   polarization=None, extra_bodies=(), optprops=None,
                   resolution=256):
    """source -> 2mm bk7 window with `value` on its EXIT face -> detector."""
    slab = sh.slab_body("Diff", "bk7", 0.0, 0.002, half=0.05)
    if value is not None:
        slab["diffuser_faces"] = {"Diff.Pad.Face2": value}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001,
                       polarization=polarization),
        slab,
        *extra_bodies,
        sh.detector_body(x=det_x, half=det_half),
    ])
    return sh.trace_scene(model, rays=rays, optprops=optprops,
                          resolution=resolution)


def _image_stats(det):
    """(total_W, per-axis stds in metres, centroid in metres) of the
    incoherent irradiance image."""
    img = det.inc.sum(axis=0)
    total = float(img.sum())
    ys, xs = np.mgrid[0:det.H, 0:det.W]
    pix = det.pixel if hasattr(det, "pixel") else det.pixel_m
    x = (xs + 0.5) * pix
    y = (ys + 0.5) * pix
    w = img / total
    cx, cy = float((w * x).sum()), float((w * y).sum())
    sx = float(np.sqrt((w * (x - cx) ** 2).sum()))
    sy = float(np.sqrt((w * (y - cy) ** 2).sum()))
    return total, (sx, sy), (cx, cy)


# ---------------------------------------------------------------------------
# unit level
# ---------------------------------------------------------------------------
def test_slope_for_grit_matches_calibration_table():
    for grit, fwhm_deg in GRIT_FWHM_DEG.items():
        m = slope_for_grit(grit)
        back = np.rad2deg(2 * np.sqrt(2 * np.log(2)) * (N_BK7 - 1) * m)
        assert back == pytest.approx(fwhm_deg, rel=1e-6)
    # log-log interpolation is monotonic between entries
    assert slope_for_grit(1500) < slope_for_grit(400) < slope_for_grit(120)


def test_diffuser_equivalent_round_trips_slope():
    sigma_nm, lcorr_um = diffuser_equivalent(0.1)
    assert slope_from_sigma_lcorr(sigma_nm * 1e-9, lcorr_um * 1e-6) \
        == pytest.approx(0.1, rel=1e-12)
    with pytest.raises(ValueError):
        diffuser_equivalent(0.0)


def test_registry_loads_dg_series(props):
    assert {"dg_120", "dg_220", "dg_600", "dg_1500"} <= set(props.diffusers)
    assert props.diffusers["dg_600"]["slope_rms"] == \
        pytest.approx(slope_for_grit(600))


def test_scene_resolves_all_three_spec_forms(props):
    for value in ("grit:600", "slope:%.12g" % slope_for_grit(600),
                  "@dg_600"):
        slab = sh.slab_body("Diff", "bk7", 0.0, 0.002, half=0.05)
        slab["diffuser_faces"] = {"Diff.Pad.Face2": value}
        model = sh.make_model([
            sh.source_body(power_mW=1.0, coherent=False),
            slab, sh.detector_body(x=0.1, half=0.05)])
        scene = Scene(model, props.matdb, props.coatings, optprops=props)
        entries = [r for r in scene.roughness.values()
                   if r.get("diffuser")]
        assert len(entries) == 1
        m = slope_from_sigma_lcorr(entries[0]["sigma_nm"] * 1e-9,
                                   entries[0]["lcorr_um"] * 1e-6)
        assert m == pytest.approx(slope_for_grit(600), rel=1e-9)


def test_diffuser_plus_roughness_same_face_is_contract_error(props):
    slab = sh.slab_body("Diff", "bk7", 0.0, 0.002, half=0.05)
    slab["diffuser_faces"] = {"Diff.Pad.Face2": "grit:600"}
    slab["roughness_faces"] = {"Diff.Pad.Face2": "50"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        slab, sh.detector_body(x=0.1, half=0.05)])
    import common
    with pytest.raises(common.ContractError):
        common.validate_model(model)


def test_t6_no_diffuser_builds_no_scatter_entries(props):
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False),
        sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.05),
        sh.detector_body(x=0.1, half=0.05)])
    scene = Scene(model, props.matdb, props.coatings, optprops=props)
    assert scene.roughness == {}


# ---------------------------------------------------------------------------
# end-to-end physics
# ---------------------------------------------------------------------------
def test_t1_closure_with_diffuser(props):
    result, grids, _ = diffuser_scene("grit:120", optprops=props)
    rep = result.ledger.report(["Src"])
    assert rep["closure_ok"], rep
    det = next(iter(grids.values()))
    total, _, _ = _image_stats(det)
    # forward scatter dominates: most of the milliwatt lands on the
    # (oversized) detector
    assert total > 0.5e-3


@pytest.mark.parametrize("grit", [120, 600])
def test_t2_transmitted_width_matches_grit_calibration(props, grit):
    """The detector-spot width reproduces the small-angle model the grit
    calibration is built on: per-axis sigma = L*(n-1)*m_rms/sqrt(2)."""
    result, grids, scene = diffuser_scene("grit:%d" % grit, rays=40000,
                                          optprops=props)
    det = next(iter(grids.values()))
    total, (sx, sy), _ = _image_stats(det)
    L = 0.1 - 0.002                      # diffuser exit face -> detector
    m = slope_for_grit(grit)
    expected = L * (N_BK7 - 1.0) * m / np.sqrt(2.0)
    # quadrature-subtract the source footprint (2mm square beam)
    beam = 0.002 / np.sqrt(12.0)
    for s in (sx, sy):
        s_corr = np.sqrt(max(s ** 2 - beam ** 2, 0.0))
        assert s_corr == pytest.approx(expected, rel=0.15)


def test_t3_smooth_limit_degenerates_to_plain_window(props):
    r_win, g_win, _ = diffuser_scene(None, rays=20000, optprops=props)
    r_dif, g_dif, _ = diffuser_scene("slope:0.001", rays=20000,
                                     optprops=props)
    det_w = next(iter(g_win.values()))
    det_d = next(iter(g_dif.values()))
    t_w, _, c_w = _image_stats(det_w)
    t_d, _, c_d = _image_stats(det_d)
    assert t_d == pytest.approx(t_w, rel=0.01)
    pix = det_w.pixel if hasattr(det_w, "pixel") else det_w.pixel_m
    assert abs(c_d[0] - c_w[0]) < 2 * pix
    assert abs(c_d[1] - c_w[1]) < 2 * pix


def test_t5_azimuthal_symmetry_at_normal_incidence(props):
    _, grids, _ = diffuser_scene("grit:220", rays=40000, optprops=props)
    det = next(iter(grids.values()))
    _, (sx, sy), _ = _image_stats(det)
    assert sx == pytest.approx(sy, rel=0.1)


def _crossed_polarizer_leak(props, diffuser_value, rays=30000):
    """linear:0 source -> (diffuser?) -> crossed linear polarizer ->
    detected fraction. Uses the shipped wire-grid polarizer registry."""
    pol = sh.slab_body("Pol", "pmma", 0.01, 0.012, half=0.05,
                       polarizer="edmund_wiregrid_vis",
                       polarizer_axis=[0.0, 1.0, 0.0])   # crossed vs z
    slab = sh.slab_body("Diff", "bk7", 0.0, 0.002, half=0.05)
    if diffuser_value is not None:
        slab["diffuser_faces"] = {"Diff.Pad.Face2": diffuser_value}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001,
                       polarization={"kind": "linear", "angle_deg": 0.0}),
        slab, pol,
        sh.detector_body(x=0.1, half=0.05),
    ])
    result, grids, _ = sh.trace_scene(model, rays=rays, optprops=props)
    rep = result.ledger.report(["Src"])
    assert rep["closure_ok"]
    det = next(iter(grids.values()))
    total, _, _ = _image_stats(det)
    return total / 1e-3                   # fraction of emitted


def test_t4_single_scatter_depolarization(props):
    """A crossed polarizer downstream of the diffuser leaks depolarized
    light on top of the polarizer's own extinction floor. The model is
    SINGLE-scatter, so the effect is small (honest limit — real ground
    glass depolarizes further via multiple scattering): measured with
    the shipped wire-grid polarizer (floor ~1.67e-4 of emitted), a
    slope-0.17 diffuser adds ~+3.9e-6 and slope-0.05 ~+2e-7, an
    ~(slope)^3 scaling between the alpha^2 and alpha^4 estimates. All
    runs share the deterministic seed, so the ordering is exact, not
    statistical."""
    floor = _crossed_polarizer_leak(props, None)
    weak = _crossed_polarizer_leak(props, "slope:0.05")
    strong = _crossed_polarizer_leak(props, "slope:0.17")
    assert strong > weak                    # monotonic in slope
    assert strong - floor > 2e-6            # real signal above the floor
    assert weak - floor < 0.5 * (strong - floor)


def test_t4_unpolarized_stays_symmetric(props):
    """Unpolarized light through the diffuser: a downstream polarizer at
    0 deg and at 90 deg transmits the same power (no polarization is
    CREATED by the diffuser at normal incidence, up to MC noise)."""
    def run(axis):
        pol = sh.slab_body("Pol", "pmma", 0.01, 0.012, half=0.05,
                           polarizer="edmund_wiregrid_vis",
                           polarizer_axis=list(axis))
        slab = sh.slab_body("Diff", "bk7", 0.0, 0.002, half=0.05)
        slab["diffuser_faces"] = {"Diff.Pad.Face2": "grit:220"}
        model = sh.make_model([
            sh.source_body(power_mW=1.0, coherent=False, half=0.001),
            slab, pol,
            sh.detector_body(x=0.1, half=0.05)])
        _, grids, _ = sh.trace_scene(model, rays=30000, optprops=props)
        det = next(iter(grids.values()))
        total, _, _ = _image_stats(det)
        return total
    p_z = run((0.0, 0.0, 1.0))
    p_y = run((0.0, 1.0, 0.0))
    assert p_z == pytest.approx(p_y, rel=0.05)
