# =============================================================================
# test_new_scenes_e2e.py — Phase-12 end-to-end validation of the five new
# physics scenes authored in make_test_scenes.py and extracted to
# geometry/<scene>/model.json. Each test drives the REAL pipeline
# (common.load_model -> Scene(optprops) -> sample_source -> Tracer.run) and
# asserts a QUANTITATIVE physical quantity; EVERY scene closes energy <1e-3.
#
#   ktp_walkoff    biaxial two-spot walk-off: the in-plane sheet's transverse
#                  displacement matches the biaxial solver's own prediction
#                  (biaxial_ray_from_k, the same oracle as test_biaxial), the
#                  y-polarized sheet stays on axis.
#   gaussian_bench beam-mode second-moment width at 5 Rayleigh ranges matches
#                  w(z)=w0*sqrt(1+(z/zR)^2); the beam has visibly expanded.
#   ghost_doublet  the strongest generation-2 Fresnel ghost carries
#                  direct_power * R^2 (--ghost-analysis refl_hist path).
#   scatter_plate  the ABg scatter lobe reaches the reflected-arm screen
#                  (scattered=True ray records) with the reflected split
#                  conserving power (closure).
#   curved_focal   a concave cylindrical (CurvedDetectorGrid) screen hugging
#                  the PCX focus catches >90% of the focused power.
#
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest \
#          scripts/raytracer/tests/test_new_scenes_e2e.py -q
# =============================================================================
import sys
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
from raytracer.detector import (DetectorGrid,              # noqa: E402
                                CurvedDetectorGrid)
from raytracer.optprops import load_optical_properties     # noqa: E402
from raytracer import birefringence as bi                  # noqa: E402
import make_test_scenes as mts                             # noqa: E402

GEO = SCRIPTS.parent / "geometry"
SCENES = mts.SCENES
NEW_SCENES = ["ktp_walkoff", "gaussian_bench", "ghost_doublet",
              "scatter_plate", "curved_focal"]

_OPT = None


def optprops():
    global _OPT
    if _OPT is None:
        _OPT = load_optical_properties()
    return _OPT


def _has(scene):
    return (GEO / scene / "model.json").exists()


def requires(scene):
    return pytest.mark.skipif(
        not _has(scene),
        reason="author+extract %s (make_test_scenes.py + extract_geometry.py)"
        % scene)


def build_scene(scene):
    model = common.load_model(GEO / scene / "model.json")
    common.validate_model(model)
    sc = Scene(model, optprops().matdb, optprops().coatings,
               optprops=optprops(), geometry_dir=str(GEO / scene))
    return sc, model


def _grid_for(face, resolution, bins, lam_range):
    stype = face.surface.__class__.__name__
    cls = CurvedDetectorGrid if stype in ("Sphere", "Cylinder") \
        else DetectorGrid
    return cls(face, resolution, bins, lam_range, label=face.id)


def run_scene(scene, rays=8000, resolution=200, spectral_bins=8,
              lam_range_nm=(500.0, 760.0), seed=3, track_history=False,
              export_rays=False):
    sc, _ = build_scene(scene)
    lam_range = (lam_range_nm[0] * 1e-9, lam_range_nm[1] * 1e-9)
    grids = {fid: _grid_for(sc.faces[fid], resolution, spectral_bins,
                            lam_range)
             for fid in sc.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-12,
                      export_rays=export_rays, track_history=track_history)
    tr = Tracer(sc, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(sc, sc.bodies[b], s, i, cfg.rays, cfg.n_lambda,
                             rng, ledger=tr.ledger, export_rays=export_rays)
               for i, (b, s) in enumerate(sc.sources)]
    res = tr.run(batches)
    return res, grids, sc


def closure_error(result):
    rep = result.ledger.report(result.source_names)
    return max(s["closure_error"] for s in rep["sources"].values())


def _spots_along(det, axis="z", thresh=0.2):
    """Intensity-centroid spot positions [m] along the grid axis mapping to
    global `axis` (reads det.xhat/det.yhat, never assumes grid==global)."""
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


# =========================================================================== #
# ENERGY CLOSURE — required for EVERY new scene
# =========================================================================== #
@pytest.mark.parametrize("scene", NEW_SCENES)
def test_energy_closure(scene):
    if not _has(scene):
        pytest.skip("model.json missing for %s" % scene)
    res, _, _ = run_scene(scene, rays=3000, resolution=64, spectral_bins=4)
    ce = closure_error(res)
    assert ce < 1e-3, "%s closure_error=%.2e" % (scene, ce)


# =========================================================================== #
# KTP WALK-OFF — two spots; in-plane sheet displacement vs the solver oracle
# =========================================================================== #
def _ktp_walkoff_dz(t_m, lam_nm=633.0):
    """Solver-predicted in-plane-sheet transverse (global z) displacement for
    the ktp_walkoff geometry: X principal at 45deg in x-z, Y principal = y,
    beam along +x. Same oracle path as test_biaxial._expected_walkoff_dz."""
    mx, my, mz = optprops().matdb.get_biaxial("ktp")
    lam = lam_nm * 1e-9
    eps = np.array([[np.real(m.n_complex(lam)) ** 2 for m in (mx, my, mz)]])
    c = np.sqrt(0.5)
    x_ax = np.array([c, 0.0, c])
    y_ax = np.array([0.0, 1.0, 0.0])
    frame = np.stack([x_ax, y_ax, np.cross(x_ax, y_ax)])
    k = np.array([[1.0, 0.0, 0.0]])
    modes = bi.biaxial_modes_for_k(k, frame, eps)
    name = "slow" if abs(modes["D_slow"][0, 1]) < 0.5 else "fast"
    K = modes["n_%s" % name][:, None] * k
    s_ray, _, _ = bi.biaxial_ray_from_k(K, frame, eps)
    return t_m * s_ray[0, 2] / s_ray[0, 0]


@requires("ktp_walkoff")
@pytest.mark.slow
def test_ktp_walkoff_double_spot():
    s = SCENES["ktp_walkoff"]
    res, grids, _ = run_scene("ktp_walkoff", rays=15000, resolution=400,
                              spectral_bins=4, lam_range_nm=(560.0, 700.0))
    assert closure_error(res) < 1e-3
    det = list(grids.values())[0]
    spots = _spots_along(det, s["walkoff_axis"])
    assert len(spots) == 2, "expected 2 spots, got %r (mm)" \
        % [round(x * 1e3, 4) for x in spots]
    dz = _ktp_walkoff_dz(s["thickness_mm"] * 1e-3, s["lambda_nm"])
    walking = max(spots, key=abs)
    straight = min(spots, key=abs)
    assert abs(straight) < 1e-4, \
        "y-pol sheet should stay on axis, got %.4f mm" % (straight * 1e3)
    assert abs(walking - dz) < 0.05 * abs(dz), \
        "walk-off spot %.4f mm vs solver-predicted %.4f mm" \
        % (walking * 1e3, dz * 1e3)


# =========================================================================== #
# GAUSSIAN BENCH — second-moment beam width vs the propagation oracle
# =========================================================================== #
def _second_moment_width(det):
    xs = det.x_lo + (np.arange(det.W) + 0.5) * det.pixel_m
    ys = det.y_lo + (np.arange(det.H) + 0.5) * det.pixel_m
    gx, gy = np.meshgrid(xs, ys)
    irr = det.inc.sum(axis=0)
    tot = irr.sum()
    cx = (gx * irr).sum() / tot
    cy = (gy * irr).sum() / tot
    varx = ((gx - cx) ** 2 * irr).sum() / tot
    vary = ((gy - cy) ** 2 * irr).sum() / tot
    return float(np.sqrt(2.0 * (varx + vary)))


@requires("gaussian_bench")
@pytest.mark.slow
def test_gaussian_bench_beam_expansion():
    s = SCENES["gaussian_bench"]
    res, grids, _ = run_scene("gaussian_bench", rays=200000, resolution=400,
                              spectral_bins=4, lam_range_nm=(560.0, 700.0),
                              seed=1)
    assert closure_error(res) < 1e-3
    det = list(grids.values())[0]
    w_meas = _second_moment_width(det)
    w0 = s["beam_waist_mm"] * 1e-3
    lam = s["lambda_nm"] * 1e-9
    zR = np.pi * w0 ** 2 / lam
    z = (s["detector_x_mm"] - s["source_x_mm"]) * 1e-3
    w_exp = w0 * np.sqrt(1.0 + (z / zR) ** 2)
    assert abs(w_meas - w_exp) / w_exp < 0.08, \
        "beam width %.4f mm vs oracle %.4f mm (z=%.1f mm, zR=%.2f mm)" \
        % (w_meas * 1e3, w_exp * 1e3, z * 1e3, zR * 1e3)
    # the beam is genuinely expanded (5 Rayleigh ranges -> ~5x the waist)
    assert w_meas > 3.0 * w0


# =========================================================================== #
# GHOST DOUBLET — the dominant generation-2 ghost == direct * R^2
# =========================================================================== #
@requires("ghost_doublet")
def test_ghost_doublet_top_ghost_oracle():
    res, grids, sc = run_scene("ghost_doublet", rays=8000, resolution=128,
                               spectral_bins=8, lam_range_nm=(560.0, 700.0),
                               seed=11, track_history=True, export_rays=True)
    assert closure_error(res) < 1e-3
    det = next(iter(grids.values()))
    recs = det.ray_records
    assert recs, "no detector ray records"
    gen = np.concatenate([r["generation"] for r in recs]).astype(int)
    scat = np.concatenate([r["scattered"] for r in recs]).astype(bool)
    power = np.concatenate([r["power"] for r in recs])
    hist = np.concatenate([r["refl_hist"] for r in recs])

    g1 = next(b for b in sc.bodies if b.label == "Glass1")
    n_glass = float(np.real(sc.medium_index(g1.index,
                                            np.array([633e-9]))[0]))
    R = ((n_glass - 1.0) / (n_glass + 1.0)) ** 2
    direct_power = float(np.sum(power[gen == 0]))
    assert direct_power > 0
    expected_ghost = direct_power * R ** 2

    groups = {}
    cand = (gen >= 2) & (~scat)
    for i in np.where(cand)[0]:
        sig = tuple(int(x) for x in hist[i] if x >= 0)
        if len(sig) >= 2:
            groups.setdefault(sig, 0.0)
            groups[sig] += float(power[i])
    assert groups, "no generation-2 ghost paths recorded"
    top_val = max(groups.values())
    assert top_val == pytest.approx(expected_ghost, rel=0.05), \
        "top ghost %.3e vs direct*R^2 %.3e (R=%.5f)" \
        % (top_val, expected_ghost, R)


# =========================================================================== #
# SCATTER PLATE — the ABg scatter lobe reaches the reflected-arm screen
# =========================================================================== #
@requires("scatter_plate")
@pytest.mark.slow
def test_scatter_plate_lobe_present():
    res, grids, _ = run_scene("scatter_plate", rays=40000, resolution=128,
                              spectral_bins=4, lam_range_nm=(560.0, 700.0),
                              export_rays=True)
    assert closure_error(res) < 1e-3
    det = next(iter(grids.values()))
    recs = det.ray_records
    assert recs, "reflected-arm screen recorded no rays"
    scat = np.concatenate([r["scattered"] for r in recs]).astype(bool)
    power = np.concatenate([r["power"] for r in recs])
    total = float(power.sum())
    scattered = float(power[scat].sum())
    # the reflected arm catches real power, and a measurable diffuse
    # scatter lobe rides on the specular spot (scattered=True rays)
    assert total > 0.0
    assert int(scat.sum()) > 0, "no scattered rays reached the screen"
    assert scattered > 0.0, "scatter lobe carried no power"


# =========================================================================== #
# CURVED FOCAL — a concave cylindrical screen catches >90% of the focus
# =========================================================================== #
@requires("curved_focal")
def test_curved_focal_captures_focus():
    res, grids, _ = run_scene("curved_focal", rays=20000, resolution=200,
                              spectral_bins=4, lam_range_nm=(560.0, 700.0))
    assert closure_error(res) < 1e-3
    det = list(grids.values())[0]
    assert isinstance(det, CurvedDetectorGrid), \
        "curved detector face did not dispatch to CurvedDetectorGrid"
    emitted = float(res.ledger.emitted[0])
    captured = float(det.inc.sum())
    assert captured / emitted > 0.90, \
        "curved screen caught only %.1f%% of the focused power" \
        % (captured / emitted * 100.0)
