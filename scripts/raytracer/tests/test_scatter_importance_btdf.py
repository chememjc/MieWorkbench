# =============================================================================
# test_scatter_importance_btdf.py — P2 scatter upgrades (engine3.md §7.1):
#   (1) BTDF: the transmitted child at a scattering face is split into a
#       specular remainder + a scattered lobe about the refracted direction;
#   (2) importance sampling: scattered children aimed at the detectors'
#       solid angles (equal flux per ray, BSDF averaged over the cone), with
#       a full-sphere remainder so per-event energy closure stays exact.
#
# Gates:
#   a  BTDF energy split closure < 1e-3 on a transmissive slab scene, and the
#      loader round-trips the btdf columns (backward compatible).
#   b  importance-sampled detected power on an off-axis stray-light scene
#      agrees with a brute-force (importance-off, 10x rays) reference within
#      combined MC error, AND the variance is measurably lower at equal rays.
#   c  the scatter_btdf and scatter_importance tokens route to Python
#      (choose_engine reason / detect_features), and neither is in PORTED.
# =============================================================================
import types

import numpy as np
import pytest

import common
from raytracer import cengine
from raytracer.optprops import load_optical_properties
from raytracer.scene import Scene
from raytracer.sources import sample_source
from raytracer.tracer import Tracer, TraceConfig
from raytracer.detector import DetectorGrid

from . import scenehelpers as sh


@pytest.fixture(scope="module")
def props():
    return load_optical_properties()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rect_detector(name, centre, normal, up, half):
    """A single-face detector plane at `centre` (world), facing `normal`,
    with in-plane `up`; a `half`-half-width square. Lets us put a detector
    OFF the specular axis so it sees scattered light only."""
    c = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up = np.asarray(up, float)
    up = up - np.dot(up, n) * n
    up = up / np.linalg.norm(up)
    right = np.cross(up, n)
    corners = [c - half * right - half * up, c + half * right - half * up,
               c + half * right + half * up, c - half * right + half * up]
    face = {"id": "%s.Pad.Face1" % name,
            "surface": {"type": "plane", "origin": list(c), "normal": list(n)},
            "orientation_outward": True, "area_m2": float((2 * half) ** 2),
            "fingerprint": {}, "mesh_stl": "",
            "trim_polylines_xyz": [[list(p) for p in corners]]}
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": face["id"]}, "faces": [face]}


def _build(model, props, imp=False, rays=20000, seed=0,
           importance_limit=1.0):
    common.validate_model(model)
    scene = Scene(model, props.matdb, props.coatings, optprops=props)
    grids = {fid: DetectorGrid(scene.faces[fid], 96, 8, (500e-9, 760e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=rays, n_lambda=1, seed=seed, power_floor=1e-12,
                      importance_scatter=imp, importance_limit=importance_limit)
    tr = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(seed)
    batches = [sample_source(scene, scene.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tr.ledger)
               for i, (b, s) in enumerate(scene.sources)]
    res = tr.run(batches)
    return res, grids, scene


def _detected(grids, label):
    for g in grids.values():
        if g.label == label:
            return float(sum(g.detected_incoherent.values()))
    raise KeyError(label)


# ---------------------------------------------------------------------------
# (a) BTDF energy split closure
# ---------------------------------------------------------------------------
def test_btdf_loader_roundtrip(props):
    e = props.scatter["lightly_ground_glass_window"]
    assert e["btdf"] is not None
    assert e["btdf"]["A"] > 0 and e["btdf"]["B"] > 0 and e["btdf"]["g"] > 0
    assert e["btdf"]["tis_cap"] == pytest.approx(0.3)
    # backward compat: the reflected-only rows carry no btdf block
    assert props.scatter["polished_bk7_glass"]["btdf"] is None
    assert props.scatter["diamond_turned_aluminum"]["btdf"] is None


def test_btdf_scene_energy_closure(props):
    """A dielectric window whose exit face scatters on BOTH sides (BRDF+BTDF)
    conserves energy: reflected specular + reflected lobes + transmitted
    specular + transmitted lobes + absorbed == incident (< 1e-3)."""
    slab = sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.02)
    slab["scatter_faces"] = {"Win.Pad.Face2": "lightly_ground_glass_window"}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001),
        slab, sh.detector_body(x=0.1, half=0.05)])
    for imp in (False, True):
        res, _, _ = _build(model, props, imp=imp, rays=30000, seed=3)
        rep = res.ledger.report(["Src"])
        assert rep["closure_ok"], (imp, rep)
        assert rep["sources"]["Src"]["closure_error"] < 1e-3, (imp, rep)


def test_btdf_adds_transmitted_scatter(props):
    """Turning BTDF on peels power off the specular transmitted beam into a
    scattered transmitted lobe (vs the reflected-only row): an off-axis
    detector behind the window sees MORE light with the BTDF row."""
    def scene_for(entry):
        slab = sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.02)
        slab["scatter_faces"] = {"Win.Pad.Face2": entry}
        det = _rect_detector("Off", [0.10, 0.05, 0.0], [-1, 0, 0], [0, 1, 0],
                             0.02)
        return sh.make_model([
            sh.source_body(power_mW=1.0, coherent=False, half=0.001),
            slab, det])
    res_r, g_r, _ = _build(scene_for("polished_bk7_glass"), props,
                           rays=60000, seed=1)
    res_b, g_b, _ = _build(scene_for("lightly_ground_glass_window"), props,
                           rays=60000, seed=1)
    off_refl = _detected(g_r, "Off.Pad.Face1")
    off_btdf = _detected(g_b, "Off.Pad.Face1")
    assert off_btdf > 5.0 * max(off_refl, 1e-18)


# ---------------------------------------------------------------------------
# (b) importance sampling: unbiased + lower variance
# ---------------------------------------------------------------------------
def _straylight_model():
    """Pencil beam -> a reflective scatter mirror; a small FAR off-axis
    detector catches ONLY the back-scatter (the specular return misses it).
    The detector is small and distant so it subtends a small cone -- the
    regime where the equal-flux / cone-averaged BSDF estimator is unbiased
    (the residual bias is O(cone-radius^2)) and where brute-force sampling
    almost never hits it (so importance sampling wins hard on variance)."""
    slab = sh.slab_body("Mir", "bk7", 0.0, 0.002, half=0.03,
                        mirror=1.0, absorbance=0.0)
    slab["scatter_faces"] = {"Mir.Pad.Face1": "diamond_turned_aluminum"}
    # specular return travels back along -x at y ~ 0; put the detector far
    # off in +y (~24 deg off specular), facing +x (toward the mirror), so it
    # sees scatter only. Small + distant => a small solid angle brute-force
    # almost never samples, while area sampling always hits it.
    det = _rect_detector("Stray", [-0.275, 0.12, 0.0], [1, 0, 0], [0, 1, 0],
                         0.008)
    src = sh.source_body(power_mW=1.0, coherent=False, half=0.0008, x=-0.03)
    return sh.make_model([src, slab, det])


def test_importance_unbiased_and_lower_variance(props):
    model = _straylight_model()
    seeds = list(range(10))
    rays = 20000

    off = np.array([_detected(_build(model, props, imp=False, rays=rays,
                                     seed=s)[1], "Stray.Pad.Face1")
                    for s in seeds])
    on = np.array([_detected(_build(model, props, imp=True, rays=rays,
                                    seed=s)[1], "Stray.Pad.Face1")
                   for s in seeds])
    # brute-force reference: importance OFF at 10x rays
    brute = np.array([_detected(_build(model, props, imp=False, rays=10 * rays,
                                       seed=100 + s)[1], "Stray.Pad.Face1")
                      for s in seeds])

    mean_on, mean_brute = on.mean(), brute.mean()
    se = np.hypot(on.std(ddof=1) / np.sqrt(len(on)),
                  brute.std(ddof=1) / np.sqrt(len(brute)))
    assert mean_brute > 0
    # unbiased: importance-on mean agrees with the brute reference within 4 SE
    assert abs(mean_on - mean_brute) < 4.0 * se, \
        (mean_on, mean_brute, se)
    # variance reduction at EQUAL rays: importance-on estimator is tighter
    assert on.std(ddof=1) < 0.6 * off.std(ddof=1), \
        (on.std(ddof=1), off.std(ddof=1))


# ---------------------------------------------------------------------------
# (c) token discipline: both features route to Python
# ---------------------------------------------------------------------------
def _fake_args(**over):
    base = dict(rough_fresnel="micro", particles=None, particle_threshold=None,
                ray_differentials=False, export_rays=False,
                ghost_analysis=False, viz_pattern=None, save_fields=False,
                gdd_budget=False, importance_scatter=False, engine="auto")
    base.update(over)
    return types.SimpleNamespace(**base)


def _scatter_scene(props, entry="polished_bk7_glass"):
    slab = sh.slab_body("Win", "bk7", 0.0, 0.002, half=0.02)
    slab["scatter_faces"] = {"Win.Pad.Face2": entry}
    model = sh.make_model([
        sh.source_body(power_mW=1.0, coherent=False, half=0.001),
        slab, sh.detector_body(x=0.1, half=0.05)])
    common.validate_model(model)
    return Scene(model, props.matdb, props.coatings, optprops=props)


def test_btdf_token_emitted_and_unported(props):
    scene = _scatter_scene(props, "lightly_ground_glass_window")
    feats = cengine.detect_features(_fake_args(), scene)
    assert "scatter" in feats
    assert "scatter_btdf" in feats
    assert "scatter_btdf" not in cengine.PORTED
    # a reflected-only row does NOT emit the btdf token
    plain = _scatter_scene(props, "polished_bk7_glass")
    assert "scatter_btdf" not in cengine.detect_features(_fake_args(), plain)


def test_importance_token_emitted_and_unported(props):
    scene = _scatter_scene(props)
    off = cengine.detect_features(_fake_args(importance_scatter=False), scene)
    assert "scatter_importance" not in off
    on = cengine.detect_features(_fake_args(importance_scatter=True), scene)
    assert "scatter_importance" in on
    assert "scatter_importance" not in cengine.PORTED


def test_choose_engine_routes_python_on_new_features(props):
    """--engine auto must fall to Python for BTDF and for importance scatter
    (never silently run the C reflected-only full-lobe sampler)."""
    if cengine.binary_path() is None:
        pytest.skip("miewb-trace not built")
    btdf = _scatter_scene(props, "lightly_ground_glass_window")
    eng, reason = cengine.choose_engine(_fake_args(), btdf)
    assert eng == "python" and "scatter_btdf" in reason, reason
    imp = _scatter_scene(props)
    eng2, reason2 = cengine.choose_engine(
        _fake_args(importance_scatter=True), imp)
    assert eng2 == "python" and "scatter_importance" in reason2, reason2
