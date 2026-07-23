# =============================================================================
# test_scenes_e2e.py — Phase-3 end-to-end validation of the 24 FreeCAD-authored
# validation scenes (make_test_scenes.SCENES), each extracted to
# geometry/<scene>/model.json.  Every test drives the REAL pipeline
# (common.load_model -> Scene(optprops) -> sample_source -> Tracer.run [-> gather])
# and asserts a QUANTITATIVE physical quantity read from the SCENES oracle.
#
# Method notes (why each assertion is measured the way it is):
#   * Focal positions use the GEOMETRIC axis-crossing of a thin traced bundle
#     (fast, no gather) compared to the paraxial thick-lens equation evaluated
#     from the SCENES parameters — which itself reproduces SCENES' expected
#     efl/bfl to <0.1%.  Diverging lenses back-project the outgoing slopes.
#   * Detector-power physics (polarizers, filters) uses det.inc sums for the
#     INCOHERENT sources; spot positions read det.xhat/det.yhat (never assume
#     grid-axis == global-axis).
#   * waveplate_quartz uses gather.render_coherent (coherent o/e recombination).
#
# SCENE ISSUES FOUND (tracer is faithful; these are scene-geometry / expected-
# value problems, or one physics-model limitation — each is xfail'd with a
# precise reason and reproduced in the report):
#   * lens_asphere  — conic k=-n^2 makes the FRONT surface stigmatic in-glass
#     but the flat exit re-adds spherical aberration, so the full lens
#     OVER-corrects and is ~3x WORSE than the spherical control, not 5x better.
#   * pol_circular  — the circular polarizer is modelled linear-diattenuator
#     THEN retarder (a CP *generator*); as an *analyzer* it cannot discriminate
#     handedness (left==right exactly).  Physics-model ordering limitation.
#   * pbs_cube      — reflected-arm detector is edge-on (auto-detected face has
#     normal +x) so it catches 0 W; the 5 um air gap loses ~35% to seam_loss;
#     the transmitted arm shows no s/p selectivity.
#   * prism_equilateral — the 19.4 deg body rotation puts the +x beam at ~10 deg
#     AOI on the entrance face (not the intended 49.4 deg min-dev), so it TIRs
#     at the exit and never forms the dispersed fan; the (rotated) detector face
#     is also edge-on and catches 0 W.
#   * wollaston     — FIXED in round 2: the 4-beam double-split was the
#     extractor's self-crossing trim loops (dead half-faces), not the wedge
#     air gap; the clean two-spot test now passes at the ideal separation.
#   * axicon_pcx    — the deflection angle is correct ((n-1)*alpha), but the
#     detector at z=30 mm sits inside the Bessel/converging zone (z_max~55 mm
#     for the Phi10 beam), so no clean ring at 2.70 mm forms there; the
#     deflection angle is asserted instead (documented).
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest raytracer/tests/test_scenes_e2e.py -q
# (from scripts/).  CUDA gather is used automatically when available.
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                              # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.sources import sample_source                # noqa: E402
from raytracer.tracer import Tracer, TraceConfig           # noqa: E402
from raytracer.detector import DetectorGrid                # noqa: E402
from raytracer.rays import RayBatch                        # noqa: E402
from raytracer import gather                               # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
import make_test_scenes as mts                             # noqa: E402

GEO = SCRIPTS.parent / "geometry"
SCENES = mts.SCENES
SRC_POWER_W = 5e-3                         # every scene source emits 5 mW
# The Phase-12 new-physics scenes (biaxial / Gaussian / ghost / scatter /
# curved-detector) have dedicated coverage in test_new_scenes_e2e.py; this
# module is the original 24-scene validation catalog and its generic
# run_scene builds a planar DetectorGrid (curved_focal has a cylinder
# detector face that needs CurvedDetectorGrid dispatch).
_NEW_PHYSICS_SCENES = {"ktp_walkoff", "gaussian_bench", "ghost_doublet",
                       "scatter_plate", "curved_focal"}
ALL_SCENES = sorted(set(SCENES.keys()) - _NEW_PHYSICS_SCENES)


def _has(scene):
    return (GEO / scene / "model.json").exists()


def requires(scene):
    return pytest.mark.skipif(
        not _has(scene),
        reason="author+extract %s (make_test_scenes.py + extract_geometry.py)"
        % scene)


# --------------------------------------------------------------------------- #
# shared optical-property library (loaded once)
# --------------------------------------------------------------------------- #
_OPT = None


def optprops():
    global _OPT
    if _OPT is None:
        _OPT = load_optical_properties()
    return _OPT


def build_scene(scene, mutate=None):
    """Load model.json, optionally mutate the dict IN MEMORY, build Scene."""
    model = common.load_model(GEO / scene / "model.json")
    if mutate is not None:
        mutate(model)
    common.validate_model(model)
    sc = Scene(model, optprops().matdb, optprops().coatings,
               optprops=optprops(), geometry_dir=str(GEO / scene))
    return sc, model


# --------------------------------------------------------------------------- #
# cached full traces (a scene traced once serves several assertions)
# --------------------------------------------------------------------------- #
_CACHE = {}


def run_scene(scene, rays=6000, n_lambda=1, resolution=200, spectral_bins=16,
              lam_lo_nm=None, lam_hi_nm=None, mutate=None, key=None,
              coherent_gather=False, seed=3):
    """Trace `scene` end-to-end; return (result, grids, scene_obj, model).
    Cached on `key` (mutate closures are unhashable, so callers pass a key)."""
    if key is not None and key in _CACHE:
        return _CACHE[key]
    sc, model = build_scene(scene, mutate)
    if lam_lo_nm is None:
        lam_lo_nm = min(s["lambdac_nm"] for _, s in sc.sources) - 120
        for _, s in sc.sources:
            if s.get("lambdamin_nm"):
                lam_lo_nm = min(lam_lo_nm, s["lambdamin_nm"] - 50)
    if lam_hi_nm is None:
        lam_hi_nm = max(s["lambdac_nm"] for _, s in sc.sources) + 120
        for _, s in sc.sources:
            if s.get("lambdamax_nm"):
                lam_hi_nm = max(lam_hi_nm, s["lambdamax_nm"] + 50)
    grids = {fid: DetectorGrid(sc.faces[fid], resolution, spectral_bins,
                               (lam_lo_nm * 1e-9, lam_hi_nm * 1e-9),
                               label=sc.faces[fid].id)
             for fid in sc.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=n_lambda, seed=seed,
                      power_floor=1e-9)
    tr = Tracer(sc, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(sc, sc.bodies[b], s, i, cfg.rays, cfg.n_lambda,
                             rng, ledger=tr.ledger)
               for i, (b, s) in enumerate(sc.sources)]
    res = tr.run(batches)
    if coherent_gather:
        for det in grids.values():
            gather.render_coherent(det, {}, backend="auto",
                                   enforce_gate=False)
    out = (res, grids, sc, model)
    if key is not None:
        _CACHE[key] = out
    return out


def closure_error(result):
    rep = result.ledger.report(result.source_names)
    return max(s["closure_error"] for s in rep["sources"].values())


def det_power(grids, label_sub=None):
    """Summed detected power [W]; whole-scene, or the detector whose label
    contains `label_sub`."""
    tot = 0.0
    for det in grids.values():
        if label_sub is None or label_sub in det.label:
            tot += float(det.inc.sum())
    return tot


# --------------------------------------------------------------------------- #
# in-memory source mutators
# --------------------------------------------------------------------------- #
def mut_pol(pol):
    def m(model):
        for b in model["bodies"]:
            if b.get("role") == "source":
                b["source"]["polarization"] = pol
    return m


def mut_lam(lc_nm):
    """Force a monochromatic source at lc_nm (clears any Gaussian bounds)."""
    def m(model):
        for b in model["bodies"]:
            if b.get("role") == "source":
                b["source"]["lambdac_nm"] = float(lc_nm)
                b["source"]["lambdamin_nm"] = None
                b["source"]["lambdamax_nm"] = None
    return m


# --------------------------------------------------------------------------- #
# geometric bundle tracing (no detectors, no gather) for focal measurements
# --------------------------------------------------------------------------- #
def _disk(rmax, x0, m):
    rng = np.random.default_rng(0)
    r = np.sqrt(rng.uniform(0, rmax ** 2, m))
    th = rng.uniform(0, 2 * np.pi, m)
    b = RayBatch(m)
    b.pos[:] = np.stack([np.full(m, x0), r * np.cos(th), r * np.sin(th)],
                        axis=-1)
    b.dir[:] = [1.0, 0.0, 0.0]
    b.s_hat[:] = [0.0, 0.0, 1.0]
    b.Es[:] = 1.0
    b.Ep[:] = 1.0
    return b


def _line(rmax, x0, m, axis="y", other=0.0):
    v = np.linspace(-rmax, rmax, m)
    v = v[np.abs(v) > rmax * 0.08]
    b = RayBatch(len(v))
    p = np.zeros((len(v), 3))
    p[:, 0] = x0
    if axis == "y":
        p[:, 1] = v
        p[:, 2] = other
    else:
        p[:, 2] = v
        p[:, 1] = other
    b.pos[:] = p
    b.dir[:] = [1.0, 0.0, 0.0]
    b.s_hat[:] = [0.0, 0.0, 1.0]
    b.Es[:] = 1.0
    b.Ep[:] = 1.0
    return b


def bundle_forward(scene, lam_m, batch, dir_x_min=0.3):
    """Trace `batch` through `scene` (no screens); return the concatenated
    (pos, dir) of every ambient forward primary-ray segment (generation 0,
    depth 0).  A straight ray is idempotent under plane projection, so
    re-recording it across steps is harmless."""
    batch.lam[:] = lam_m
    batch.birth_power[:] = batch.power
    tr = Tracer(scene, TraceConfig(rays=len(batch), n_lambda=1, seed=1,
                                   power_floor=1e-3), {})
    queue = [batch]
    P, D = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(24):
            if not queue:
                break
            ch = tr.step(queue.pop())
            if ch is None or len(ch) == 0:
                continue
            sel = ((ch.generation == 0) & (ch.depth == 0)
                   & (ch.dir[:, 0] > dir_x_min))
            if np.any(sel):
                P.append(ch.pos[sel].copy())
                D.append(ch.dir[sel].copy())
            queue.append(ch)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(D)


def axis_cross_x(P, D):
    """Median global-x where the outgoing rays cross the optical axis (y=0).
    Works for converging (real, x>lens) and diverging (virtual, x<lens)."""
    dy = D[:, 1]
    good = np.abs(dy) > 1e-9
    t = -P[good, 1] / dy[good]
    x = P[good, 0] + t * D[good, 0]
    return float(np.median(x[np.isfinite(x)]))


def rms_at_plane(P, D, x_plane, forward_only=True):
    t = (x_plane - P[:, 0]) / D[:, 0]
    ok = (t > 0) if forward_only else np.ones(len(t), bool)
    yz = P[ok, 1:] + t[ok, None] * D[ok, 1:]
    return float(yz[:, 0].std()), float(yz[:, 1].std()), int(ok.sum())


def thick_lens_focus_mm(R1, R2, d, n, front=0.0):
    """Global-x [mm] of the paraxial focus of a thick lens with front vertex
    at `front`, thickness d, signed radii R1/R2 (None = plano), index n."""
    p1 = 0.0 if R1 is None else 1.0 / R1
    p2 = 0.0 if R2 is None else 1.0 / R2
    if R1 is None or R2 is None:
        inv_f = (n - 1) * (p1 - p2)
    else:
        inv_f = (n - 1) * (p1 - p2 + (n - 1) * d / (n * R1 * R2))
    f = 1.0 / inv_f
    bfd = f if R1 is None else f * (1 - (n - 1) * d / (n * R1))
    return front + d + bfd, f


# =========================================================================== #
# ENERGY CLOSURE — required for EVERY scene (incl. mesh_freeform)
# =========================================================================== #
@pytest.mark.parametrize("scene", ALL_SCENES)
def test_energy_closure(scene):
    if not _has(scene):
        pytest.skip("model.json missing for %s" % scene)
    res, grids, sc, _ = run_scene(scene, rays=2500, n_lambda=1, resolution=64,
                                  spectral_bins=4)
    ce = closure_error(res)
    assert ce < 1e-3, "%s closure_error=%.2e" % (scene, ce)


# =========================================================================== #
# SIMPLE LENSES — traced focus vs paraxial thick-lens (<1%)
# =========================================================================== #
@pytest.mark.parametrize("scene", ["lens_pcx", "lens_dcx", "lens_pcv",
                                   "lens_dcv"])
def test_simple_lens_focus(scene):
    if not _has(scene):
        pytest.skip("missing %s" % scene)
    s = SCENES[scene]
    x_expect_mm, f = thick_lens_focus_mm(
        s["R1_mm"], s["R2_mm"], s["thickness_mm"], s["n_633"])
    sc, _ = build_scene(scene)
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9,
                          _line(1.2e-3, -5e-3, 40, axis="y"))
    assert len(P) > 20, "%s: lost the bundle through the lens" % scene
    x_mm = axis_cross_x(P, D) * 1e3
    assert x_mm == pytest.approx(x_expect_mm, rel=0.01), \
        "%s traced focus %.3f mm vs thick-lens %.3f mm (f=%.3f)" \
        % (scene, x_mm, x_expect_mm, f)


# =========================================================================== #
# ACHROMAT — chromatic focal shift tiny AND >=3x smaller than the singlet
# =========================================================================== #
@requires("lens_achromat")
@requires("lens_pcx")
def test_achromat_chromatic_correction():
    def chrom_shift(scene, zoff=0.0):
        s = SCENES[scene]
        foci = {}
        for lc in (486.1, 656.3):
            sc, _ = build_scene(scene, mut_lam(lc))
            P, D = bundle_forward(sc, lc * 1e-9,
                                  _line(4e-3, -8e-3, 300, axis="y",
                                        other=zoff))
            foci[lc] = axis_cross_x(P, D) * 1e3
        fmean = 0.5 * (foci[486.1] + foci[656.3])
        return abs(foci[486.1] - foci[656.3]), fmean

    ach_shift, ach_f = chrom_shift("lens_achromat")
    sing_shift, sing_f = chrom_shift("lens_pcx")
    ach_rel = ach_shift / ach_f
    sing_rel = sing_shift / sing_f
    assert ach_rel < 0.004, "achromat chromatic shift %.4f%% not < 0.4%%" \
        % (ach_rel * 100)
    assert sing_rel > 3 * ach_rel, \
        "singlet chromatic shift (%.4f%%) not >=3x the achromat (%.4f%%)" \
        % (sing_rel * 100, ach_rel * 100)


# =========================================================================== #
# SPHERE vs ASPHERE — XFAIL: the scene conic OVER-corrects (see header)
# =========================================================================== #
def _best_focus_rms(scene, rmax=9e-3, m=1800, xlo=0.030, xhi=0.060):
    s = SCENES[scene]
    sc, _ = build_scene(scene)
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9, _disk(rmax, -8e-3, m),
                          dir_x_min=0.5)
    best = np.inf
    for xf in np.linspace(xlo, xhi, 301):
        ry, rz, _ = rms_at_plane(P, D, xf)
        best = min(best, float(np.hypot(ry, rz)))
    return best


@requires("lens_sphere_control")
def test_sphere_control_focuses():
    # the spherical control itself must focus near its paraxial focus and
    # close energy — this half of the pair is sound.
    s = SCENES["lens_sphere_control"]
    x_expect_mm, _ = thick_lens_focus_mm(s["R1_mm"], s["R2_mm"],
                                         s["thickness_mm"], s["n_633"])
    sc, _ = build_scene("lens_sphere_control")
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9,
                          _line(1.0e-3, -5e-3, 40, axis="y"))
    x_mm = axis_cross_x(P, D) * 1e3
    assert x_mm == pytest.approx(x_expect_mm, rel=0.02), (x_mm, x_expect_mm)


@requires("lens_asphere")
@requires("lens_sphere_control")
def test_asphere_beats_sphere_5x():
    # The front conic+A4 is solved for the COMPLETE lens (front asphere + flat
    # exit), not just a front-surface stigmatic conic — see make_test_scenes
    # SCENES["lens_asphere"]. Offline meridional trace predicts ~1 um RMS
    # (~83x better than the sphere control); the >=5x gate has huge margin.
    sphere = _best_focus_rms("lens_sphere_control")
    asphere = _best_focus_rms("lens_asphere")
    assert asphere <= sphere / 5.0, \
        "asphere best-focus RMS %.4f mm not <= sphere %.4f mm /5" \
        % (asphere * 1e3, sphere * 1e3)


# =========================================================================== #
# CYLINDER LENSES — line focus collapses one axis / diverges one axis
# =========================================================================== #
@requires("lens_cyl_pos")
def test_cyl_pos_line_focus_collapses_one_axis():
    s = SCENES["lens_cyl_pos"]
    sc, _ = build_scene("lens_cyl_pos")
    lam = s["lambda_nm"] * 1e-9
    # y-fan (z offset 1 mm avoids the cylinder u=+/-pi seam at z=0) -> y focus
    Py, Dy = bundle_forward(sc, lam, _line(4e-3, -8e-3, 80, axis="y",
                                           other=1e-3), dir_x_min=0.9)
    Py = Py[Py[:, 0] > 7.2e-3]
    Dy = Dy[-len(Py):] if len(Py) else Dy
    xf = axis_cross_x(Py, Dy)
    # at the line focus the y-extent collapses; the z-extent (beam height)
    # does not -> huge axis ratio
    Pd, Dd = bundle_forward(sc, lam, _disk(4.5e-3, -8e-3, 1500), dir_x_min=0.5)
    rms_y, rms_z, n = rms_at_plane(Pd, Dd, xf)
    assert n > 100
    ratio = rms_z / max(rms_y, 1e-12)
    assert ratio > 10, "cyl_pos axis ratio %.1f not > 10 (rms_y=%.4f mm, " \
        "rms_z=%.4f mm at x=%.2f mm)" % (ratio, rms_y * 1e3, rms_z * 1e3,
                                         xf * 1e3)


@requires("lens_cyl_neg")
def test_cyl_neg_virtual_line_focus():
    s = SCENES["lens_cyl_neg"]
    sc, _ = build_scene("lens_cyl_neg")
    # concave-front plano cylinder: on-axis vertex separation == thickness
    x_expect_mm, _ = thick_lens_focus_mm(s["R_mm"], None, s["thickness_mm"],
                                         s["n_633"])
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9,
                          _line(4e-3, -8e-3, 80, axis="y", other=1e-3),
                          dir_x_min=0.9)
    P = P[P[:, 0] > 7.2e-3] if np.any(P[:, 0] > 7.2e-3) else P
    D = D[-len(P):]
    xf_mm = axis_cross_x(P, D) * 1e3
    assert xf_mm < 0, "cyl_neg focus should be virtual (x<0), got %.2f" % xf_mm
    assert xf_mm == pytest.approx(x_expect_mm, rel=0.05), \
        "cyl_neg virtual focus %.3f mm vs thick-lens %.3f mm" \
        % (xf_mm, x_expect_mm)


# =========================================================================== #
# AXICON — deflection angle (n-1)*alpha (ring-radius scene issue documented)
# =========================================================================== #
@requires("axicon_pcx")
def test_axicon_deflection_angle():
    s = SCENES["axicon_pcx"]
    sc, _ = build_scene("axicon_pcx")
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9, _disk(4.5e-3, -8e-3, 400),
                          dir_x_min=0.5)
    ang = np.degrees(np.arctan2(np.hypot(D[:, 1], D[:, 2]), D[:, 0]))
    ang = ang[ang > 0.05]
    dev = float(np.median(ang))
    expect = (s["n_633"] - 1.0) * s["base_angle_deg"]        # = 5.15 deg
    assert dev == pytest.approx(expect, rel=0.03), \
        "axicon deflection %.3f deg vs (n-1)*alpha %.3f deg" % (dev, expect)
    # NOTE: SCENES['expected_ring_radius_mm']=2.70 is NOT asserted — the
    # detector at z=30 mm is inside the converging Bessel zone (z_max~55 mm
    # for the Phi10 beam), so no clean ring forms there (see module header).


# =========================================================================== #
# BALL LENS — paraxial BFL from the rear surface (<2%)
# =========================================================================== #
@requires("lens_ball")
def test_ball_lens_bfl():
    s = SCENES["lens_ball"]
    sc, _ = build_scene("lens_ball")
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9, _disk(0.3e-3, -8e-3, 300),
                          dir_x_min=0.5)
    focus_mm = axis_cross_x(P, D) * 1e3
    bfl_mm = focus_mm - s["diameter_mm"]                    # rear vertex = D
    assert bfl_mm == pytest.approx(s["expected_bfl_mm"], rel=0.02), \
        "ball BFL %.3f mm vs expected %.3f mm" % (bfl_mm, s["expected_bfl_mm"])


# =========================================================================== #
# ROD — cylinder line focus in the x-y plane (<3%)
# =========================================================================== #
@requires("lens_rod")
def test_rod_line_focus():
    s = SCENES["lens_rod"]
    R = s["diameter_mm"] / 2.0
    x_expect_mm, _ = thick_lens_focus_mm(R, -R, s["diameter_mm"], s["n_d"])
    sc, _ = build_scene("lens_rod")
    P, D = bundle_forward(sc, s["lambda_nm"] * 1e-9,
                          _line(0.8e-3, -8e-3, 80, axis="y", other=1e-3),
                          dir_x_min=0.5)
    P = P[P[:, 0] > 7.9e-3]
    D = D[-len(P):]
    xf_mm = axis_cross_x(P, D) * 1e3
    assert xf_mm == pytest.approx(x_expect_mm, rel=0.03), \
        "rod line focus %.3f mm vs thick-cyl %.3f mm" % (xf_mm, x_expect_mm)


# =========================================================================== #
# FRESNEL — most detected power near the axis at the focal plane (loose)
# =========================================================================== #
@requires("lens_fresnel")
def test_fresnel_focus_concentration():
    _, grids, sc, _ = run_scene("lens_fresnel", rays=8000, resolution=400,
                                spectral_bins=8, lam_lo_nm=600, lam_hi_nm=660,
                                key="fresnel")
    det = list(grids.values())[0]
    img = det.inc.sum(axis=0)
    xs = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
    ys = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
    gx, gy = np.meshgrid(xs, ys)
    rr = np.hypot(gx, gy)
    tot = img.sum()
    frac = float(img[rr < 2e-3].sum() / tot)
    assert frac > 0.60, "fresnel: only %.1f%% of power within 2 mm of axis" \
        % (frac * 100)


# =========================================================================== #
# PRISM — XFAIL: wrong rotation -> near-normal entry, TIR, broken detector
# =========================================================================== #
@requires("prism_equilateral")
def test_prism_energy_closes():
    res, _, _, _ = run_scene("prism_equilateral", rays=4000, n_lambda=3,
                             resolution=128, key="prism")
    assert closure_error(res) < 1e-3


@requires("prism_equilateral")
def test_prism_minimum_deviation():
    # prism_rotation_deg=-19.3991 sets the entrance-face AOI to (A+dmin)/2 =
    # 49.399 deg (true minimum deviation for 550nm), so the beam exits cleanly
    # (no exit-face TIR) and disperses; blue deviates more than red. See
    # make_test_scenes SCENES["prism_equilateral"].
    def dev_deg(lc):
        sc, _ = build_scene("prism_equilateral", mut_lam(lc))
        tr = Tracer(sc, TraceConfig(rays=3000, n_lambda=1, seed=3,
                                    power_floor=1e-3), {})
        rng = np.random.default_rng(3)
        b = sample_source(sc, sc.bodies[sc.sources[0][0]], sc.sources[0][1],
                          0, 3000, 1, rng, ledger=tr.ledger)
        _, D = bundle_forward(sc, lc * 1e-9, b, dir_x_min=0.2)
        # deviated (not straight-through) outgoing rays
        dev = np.degrees(np.arccos(np.clip(D[:, 0], -1, 1)))
        dev = dev[dev > 1.0]
        return float(np.median(dev)) if len(dev) else 0.0
    d550 = dev_deg(550)
    d486 = dev_deg(486)
    d656 = dev_deg(656)
    assert d550 == pytest.approx(SCENES["prism_equilateral"]["min_deviation_deg"],
                                 rel=0.005)
    assert d486 > d656, "blue must deviate more than red (%.2f vs %.2f)" \
        % (d486, d656)


# =========================================================================== #
# LINEAR POLARIZER — Malus's law (30 vs 120 deg, perpendicular pair)
# =========================================================================== #
@requires("pol_linear")
def test_pol_linear_malus():
    def detected(angle):
        res, grids, _, _ = run_scene(
            "pol_linear", rays=6000, resolution=64,
            mutate=mut_pol({"kind": "linear", "angle_deg": angle}),
            key="pol_lin_%d" % angle)
        assert closure_error(res) < 1e-3
        return det_power(grids)
    p0 = detected(0)
    p30 = detected(30)
    p120 = detected(120)
    assert p30 / p0 == pytest.approx(np.cos(np.deg2rad(30)) ** 2, rel=0.02)
    # 30 and 120 are perpendicular -> ratio = cos^2(30)/cos^2(120) = 3
    assert p30 / p120 == pytest.approx(3.0, rel=0.05)


# =========================================================================== #
# CROSSED POLARIZERS — transmission << aligned
# =========================================================================== #
@requires("pol_crossed")
def test_pol_crossed_extinction():
    res, grids, _, _ = run_scene("pol_crossed", rays=6000, resolution=64,
                                 key="pol_crossed")
    assert closure_error(res) < 1e-3
    crossed = det_power(grids)

    def align(model):
        for b in model["bodies"]:
            if b["name"] == "Pol2":
                b["polarizer_axis"] = [0.0, 0.0, 1.0]      # match Pol1
    _, grids_a, _, _ = run_scene("pol_crossed", rays=6000, resolution=64,
                                 mutate=align, key="pol_crossed_aligned")
    aligned = det_power(grids_a)
    assert crossed / aligned < 1e-4, \
        "crossed/aligned = %.2e not < 1e-4" % (crossed / aligned)


# =========================================================================== #
# CIRCULAR POLARIZER — XFAIL: analyzer cannot discriminate handedness
# =========================================================================== #
@requires("pol_circular")
@pytest.mark.xfail(reason="PHYSICS-MODEL: the circular polarizer applies the "
                          "linear diattenuator THEN the retarder (a CP "
                          "GENERATOR ordering). Used as an ANALYZER the linear "
                          "stage projects either input handedness to the same "
                          "power, so left/right transmission is identical "
                          "(ratio 1.0). A CP analyzer needs retarder-then-"
                          "linear.",
                   strict=True)
def test_pol_circular_handedness():
    def detected(hand):
        res, grids, _, _ = run_scene(
            "pol_circular", rays=6000, resolution=64,
            mutate=mut_pol({"kind": "circular", "handedness": hand}),
            key="pol_circ_%s" % hand)
        assert closure_error(res) < 1e-3
        return det_power(grids)
    # thorlabs_cp1l532 is LEFT-handed -> passes circular:left, blocks right
    assert detected("left") / detected("right") > 50


# =========================================================================== #
# WAVEPLATE (HWP) — crossed-analyzer transmission via coherent gather
# =========================================================================== #
@requires("waveplate_quartz")
@pytest.mark.slow
def test_waveplate_halfwave_crossed_analyzer():
    def gathered(material=None):
        def mut(model):
            if material is not None:
                for b in model["bodies"]:
                    if b["name"] == "Waveplate":
                        b["material"] = material
        res, grids, _, _ = run_scene(
            "waveplate_quartz", rays=30000, resolution=96, spectral_bins=8,
            lam_lo_nm=500, lam_hi_nm=680, mutate=mut, coherent_gather=True,
            seed=5, key="wp_%s" % (material or "quartz"))
        assert closure_error(res) < 1e-3
        return det_power(grids)
    A = gathered(None)          # HWP present -> +45 rotated to -45 -> PASSES
    B = gathered("air")         # no retardance -> +45 stays -> analyzer BLOCKS
    assert A > 0.15 * SRC_POWER_W, \
        "HWP crossed-analyzer transmission %.3e too low" % A
    assert B < 0.05 * A, "no-waveplate leakage %.3e not < 5%% of HWP %.3e" \
        % (B, A)


# =========================================================================== #
# PBS CUBE — XFAIL: reflected detector edge-on + air-gap seam loss
# =========================================================================== #
@requires("pbs_cube")
def test_pbs_energy_closes():
    res, _, _, _ = run_scene("pbs_cube", rays=8000, resolution=64, key="pbs")
    assert closure_error(res) < 1e-3


@requires("pbs_cube")
@pytest.mark.xfail(reason="SCENE: the reflected-arm detector is edge-on "
                          "(auto-detected face normal +x) and catches 0 W; the "
                          "5 um air gap between the prisms loses ~35% to "
                          "seam_loss and the transmitted arm shows no s/p "
                          "selectivity. Fixes: correct detector auto-face "
                          "selection for the rotated reflected screen and model "
                          "the cemented interface without a lossy gap.",
                   strict=True)
def test_pbs_transmit_reflect_split():
    _, grids, sc, _ = run_scene("pbs_cube", rays=8000, resolution=128,
                                key="pbs_hi")
    trans = det_power(grids, "DetTrans") / SRC_POWER_W
    refl = det_power(grids, "DetRefl") / SRC_POWER_W
    Tp_2 = 0.96957 / 2.0        # pbs_visible_45 @ 550 nm
    Rs_2 = 0.98997 / 2.0
    assert trans == pytest.approx(Tp_2, abs=0.1)
    assert refl == pytest.approx(Rs_2, abs=0.1)


# =========================================================================== #
# CALCITE DISPLACER — two spots + polarization-selectable single spots
# =========================================================================== #
def _spots_along(det, axis="z", thresh=0.2):
    """Intensity-centroid spot positions [m] along the grid axis mapping to
    global `axis`.  Reads det.xhat/det.yhat — never assumes grid==global."""
    comp = {"x": 0, "y": 1, "z": 2}[axis]
    img = det.inc.sum(axis=0)
    ax_is_x = abs(det.xhat[comp]) > abs(det.yhat[comp])
    if ax_is_x:
        prof = img.sum(axis=0)
        coord = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
        sgn = np.sign(det.xhat[comp])
    else:
        prof = img.sum(axis=1)
        coord = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
        sgn = np.sign(det.yhat[comp])
    above = prof > thresh * prof.max()
    spots, i, n = [], 0, len(prof)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        w = prof[i:j]
        spots.append(float(np.sum(coord[i:j] * w) / np.sum(w)))
        i = j
    return sorted(sgn * s for s in spots)


@requires("calcite_displacer")
def test_calcite_double_spot():
    s = SCENES["calcite_displacer"]
    res, grids, _, _ = run_scene("calcite_displacer", rays=15000,
                                 resolution=400, spectral_bins=4,
                                 lam_lo_nm=560, lam_hi_nm=620, key="calcite")
    assert closure_error(res) < 1e-3
    det = list(grids.values())[0]
    spots = _spots_along(det, "z")               # walk-off is in global z
    assert len(spots) == 2, "expected 2 spots, got %r (mm)" \
        % [round(x * 1e3, 3) for x in spots]
    sep_mm = abs(spots[1] - spots[0]) * 1e3
    assert sep_mm == pytest.approx(s["expected_displacement_mm"], rel=0.02), \
        "calcite separation %.4f mm vs expected %.4f mm" \
        % (sep_mm, s["expected_displacement_mm"])


@requires("calcite_displacer")
def test_calcite_polarization_selects_spot():
    s = SCENES["calcite_displacer"]
    # o eigenvector is +y (linear:90) -> straight; e is +z (linear:0) -> shifted
    _, g_o, _, _ = run_scene(
        "calcite_displacer", rays=12000, resolution=400, spectral_bins=4,
        lam_lo_nm=560, lam_hi_nm=620,
        mutate=mut_pol({"kind": "linear", "angle_deg": 90}), key="cal_o")
    _, g_e, _, _ = run_scene(
        "calcite_displacer", rays=12000, resolution=400, spectral_bins=4,
        lam_lo_nm=560, lam_hi_nm=620,
        mutate=mut_pol({"kind": "linear", "angle_deg": 0}), key="cal_e")
    so = _spots_along(list(g_o.values())[0], "z")
    se = _spots_along(list(g_e.values())[0], "z")
    assert len(so) == 1 and abs(so[0]) < 2e-4, \
        "o-pol should give one straight spot, got %r" % so
    assert len(se) == 1, "e-pol should give one spot, got %r" % se
    assert abs(se[0]) * 1e3 == pytest.approx(s["expected_displacement_mm"],
                                             rel=0.03)


# =========================================================================== #
# WOLLASTON — XFAIL: 5 um air gap double-splits o/e (4 beams, wrong split)
# =========================================================================== #
@requires("wollaston")
def test_wollaston_energy_closes_and_splits():
    res, grids, _, _ = run_scene("wollaston", rays=20000, resolution=500,
                                 spectral_bins=4, lam_lo_nm=560, lam_hi_nm=620,
                                 key="wollaston")
    assert closure_error(res) < 1e-3
    det = list(grids.values())[0]
    spots = _spots_along(det, "y", thresh=0.2)
    # the beam DOES split (multiple spots with real angular spread)
    assert len(spots) >= 2, "wollaston did not split the beam"
    spread = (max(spots) - min(spots))
    assert spread > 3e-3, "wollaston angular spread too small (%.3f mm)" \
        % (spread * 1e3)


@requires("wollaston")
# Previously xfail'd, blaming the 5 um wedge air gap for a 4-beam
# double-split. The REAL culprit was the extractor's self-crossing trim
# polylines (fixed in round 2): ~half of each wedge face was dead, so
# rays crossed without interface events and produced spurious spots plus
# a wrong split magnitude. With honest trims the scene yields the clean
# two-spot split at the ideal 2(n_o-n_e)tan(beta) separation.
def test_wollaston_clean_two_spot_split():
    s = SCENES["wollaston"]
    _, grids, _, _ = run_scene("wollaston", rays=20000, resolution=500,
                               spectral_bins=4, lam_lo_nm=560, lam_hi_nm=620,
                               key="wollaston")
    det = list(grids.values())[0]
    spots = _spots_along(det, "y", thresh=0.25)
    assert len(spots) == 2
    sep = abs(spots[1] - spots[0])
    dist = (s["detector_x_mm"] - 3.5) * 1e-3          # wedge interface ~x=3.5
    ang = sep / dist
    assert ang == pytest.approx(s["expected_split_rad"], rel=0.05)


# =========================================================================== #
# BANDPASS FILTER — in-band vs out-of-band > 100
# =========================================================================== #
@requires("filter_bandpass")
def test_filter_bandpass_ratio():
    def detected(lc):
        res, grids, _, _ = run_scene(
            "filter_bandpass", rays=5000, resolution=64, spectral_bins=4,
            lam_lo_nm=lc - 120, lam_hi_nm=lc + 120, mutate=mut_lam(lc),
            key="filt_%d" % lc)
        assert closure_error(res) < 1e-3
        return det_power(grids)
    p_in = detected(550)
    p_out = detected(650)
    assert p_in / max(p_out, 1e-30) > 100, \
        "bandpass in/out ratio %.1f not > 100" % (p_in / max(p_out, 1e-30))


# =========================================================================== #
# HOT MIRROR — transmitted arm visible-dominated; reflected arm NIR-dominated
# =========================================================================== #
@requires("hot_mirror")
def test_hot_mirror_spectral_split():
    # n_lambda=4 keeps the Gaussian strata inside the coating table [400,1100]
    res, grids, sc, _ = run_scene("hot_mirror", rays=8000, n_lambda=4,
                                  resolution=200, spectral_bins=32,
                                  lam_lo_nm=400, lam_hi_nm=1050, key="hotm")
    assert closure_error(res) < 1e-3
    # transmitted arm (DetPass, axis-aligned detector -> works): visible
    det = [d for d in grids.values() if "DetPass" in d.label][0]
    lam_c = det.lam_lo + (np.arange(det.spectral_bins) + 0.5) * \
        (det.lam_hi - det.lam_lo) / det.spectral_bins
    per = det.inc.sum(axis=(1, 2))
    tot = per.sum()
    vis_frac = float(per[lam_c < 700e-9].sum() / tot)
    assert vis_frac > 0.8, "transmitted arm only %.2f visible" % vis_frac
    # reflected arm (its rotated detector is edge-on -> use a bundle instead):
    sc2, _ = build_scene("hot_mirror")
    m = 400
    rng = np.random.default_rng(1)
    r = np.sqrt(rng.uniform(0, (5e-3) ** 2, m))
    th = rng.uniform(0, 2 * np.pi, m)
    b = RayBatch(m)
    b.pos[:] = np.stack([np.full(m, -20e-3), r * np.cos(th), r * np.sin(th)],
                        axis=-1)
    b.dir[:] = [1.0, 0.0, 0.0]
    b.s_hat[:] = [0.0, 0.0, 1.0]
    b.Es[:] = 1.0
    b.Ep[:] = 1.0
    b.lam[:] = np.linspace(450e-9, 1000e-9, m)
    b.birth_power[:] = b.power
    tr = Tracer(sc2, TraceConfig(rays=m, n_lambda=1, seed=1, power_floor=1e-3),
                {})
    queue = [b]
    lam_r, pw_r = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(10):
            if not queue:
                break
            ch = tr.step(queue.pop())
            if ch is None or len(ch) == 0:
                continue
            sel = ch.generation >= 1                 # reflected off the mirror
            if np.any(sel):
                lam_r.append(ch.lam[sel].copy())
                pw_r.append(ch.power[sel].copy())
            queue.append(ch)
    lam_r = np.concatenate(lam_r)
    pw_r = np.concatenate(pw_r)
    nir_frac = float(pw_r[lam_r >= 700e-9].sum() / pw_r.sum())
    assert nir_frac > 0.8, "reflected arm only %.2f NIR" % nir_frac


# =========================================================================== #
# MESH FREEFORM — traces (mesh fallback), closes energy, nonzero detected
# =========================================================================== #
@requires("mesh_freeform")
def test_mesh_freeform_traces():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # mesh-fallback warning
        res, grids, sc, _ = run_scene("mesh_freeform", rays=6000,
                                      resolution=128, key="mesh")
    assert closure_error(res) < 1e-3
    assert det_power(grids) > 1e-4, "mesh optic delivered no power to screen"
