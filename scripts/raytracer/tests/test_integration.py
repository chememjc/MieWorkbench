# =============================================================================
# test_integration.py — P2 integration: Scene from the real extracted
# example contract, tracer loop, energy-ledger closure, and the traced
# thick-lens focal length vs the lensmaker equation.
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_integration.py -v
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))    # for scripts/common.py via name

import common                                            # noqa: E402
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import sample_source, wavelength_strata  # noqa: E402
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer.rays import RayBatch                      # noqa: E402

MODEL_JSON = SCRIPTS.parent / "geometry" / "example" / "model.json"

pytestmark = pytest.mark.skipif(
    not MODEL_JSON.exists(), reason="run extract_geometry.py first")


@pytest.fixture(scope="module")
def scene():
    model = common.load_model(MODEL_JSON)
    db = MaterialDB.load()
    coatings = load_coatings(db=db)
    return Scene(model, db, coatings)


def _detector_grids(scene, resolution=128):
    grids = {}
    for fid in scene.detector_faces:
        grids[fid] = DetectorGrid(scene.faces[fid], resolution,
                                  spectral_bins=8,
                                  lam_range=(430e-9, 680e-9),
                                  label=scene.faces[fid].id)
    return grids


def test_scene_builds(scene):
    assert len(scene.sources) == 2
    assert len(scene.detector_faces) == 3
    labels = {b.label for b in scene.bodies}
    assert {"Lens", "GlassSphere", "Target"} <= labels


def test_wavelength_strata(scene):
    # red laser: monochromatic
    red = [s for _, s in scene.sources if s["lambdac_nm"] == 633.0][0]
    lam = wavelength_strata(red, 5)
    assert len(lam) == 1 and abs(lam[0] - 633e-9) < 1e-15
    # green: asymmetric gaussian (532, -32/+32 here symmetric) — strata
    # bracket the center and stay within physical range
    green = [s for _, s in scene.sources if s["lambdac_nm"] == 532.0][0]
    lam = wavelength_strata(green, 9) / 1e-9
    assert len(lam) == 9
    assert np.all(np.diff(lam) > 0)
    assert 400 < lam.min() < 532 < lam.max() < 700


def test_full_trace_energy_closure(scene):
    grids = _detector_grids(scene)
    cfg = TraceConfig(rays=4000, n_lambda=5, seed=42, viz_rays=100)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(cfg.seed)
    batches = []
    for sid, (bidx, src) in enumerate(scene.sources):
        batches.append(sample_source(scene, scene.bodies[bidx], src, sid,
                                     cfg.rays, cfg.n_lambda, rng,
                                     ledger=tracer.ledger))
    result = tracer.run(batches)
    rep = result.ledger.report(result.source_names)
    for name, s in rep["sources"].items():
        assert s["emitted_W"] == pytest.approx(5e-3, rel=1e-9), name
        assert s["closure_error"] < 1e-3, (name, s)
    # light must actually arrive at the Target screen
    target = [g for g in grids.values() if "Pad001" in g.label][0]
    coh_power = sum(target.detected_geometric.values())
    inc_power = float(target.inc.sum())
    assert coh_power + inc_power > 1e-4      # >2% of the 5 mW sources
    # viz segments recorded
    assert len(result.viz.as_array()) > 100


def test_traced_focus_matches_lensmaker(scene):
    """Collimated paraxial 633 nm bundle through the extracted BK7 lens:
    the traced axis crossing must match the thick-lens equation."""
    # lens surface params from the scene itself
    lens = [b for b in scene.bodies if b.label == "Lens"][0]
    spheres = [scene.faces[f] for f in lens.face_ids
               if scene.faces[f].surface.__class__.__name__ == "Sphere"]
    assert len(spheres) == 2
    cs = sorted([(s.surface.c[0], s.surface.r) for s in spheres])
    # left surface: center +|c|, vertex at c - r (negative); right: mirror
    c_left, R = cs[1][0] - 0.0, cs[1][1]       # center at +0.0957
    v_left = c_left - R                        # ~ -0.0043
    c_right = cs[0][0]
    v_right = c_right + R                      # ~ +0.0043
    d = v_right - v_left
    n = float(np.real(scene.matdb.get("bk7").n_complex(
        np.array([633e-9]))[0]))
    R1, R2 = R, -R
    inv_f = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * d / (n * R1 * R2))
    f = 1 / inv_f
    bfd = f * (1 - (n - 1) * d / (n * R1))
    x_focus_expect = v_right + bfd

    # paraxial collimated bundle (avoid y=0 exactly; avoid the GlassSphere
    # which sits at y >= 0 upstream: use y < 0 rays only)
    m = 40
    y = np.linspace(-1.5e-3, -0.2e-3, m)
    batch = RayBatch(m)
    batch.pos[:] = np.stack([np.full(m, -0.03), y, np.zeros(m)], axis=-1)
    batch.dir[:] = [1.0, 0.0, 0.0]
    batch.s_hat[:] = [0.0, 0.0, 1.0]
    batch.Es[:] = 1.0
    batch.Ep[:] = 1.0
    batch.lam[:] = 633e-9
    batch.birth_power[:] = batch.power

    cfg = TraceConfig(rays=m, n_lambda=1, seed=1, power_floor=1e-3)
    tracer = Tracer(scene, cfg, {})            # no screens: pure geometry
    # drive the loop manually and harvest rays that exited the lens
    queue = [batch]
    focus_x = []
    for _ in range(12):
        if not queue:
            break
        children = tracer.step(queue.pop())
        if children is None or len(children) == 0:
            continue
        # post-lens forward bundle: no reflections, in ambient, past the
        # lens rear vertex, heading +x and converging (dy opposite sign
        # of y)
        c = children
        sel = ((c.generation == 0) & (c.depth == 0)
               & (c.pos[:, 0] > 0.004) & (c.dir[:, 0] > 0.9))
        if np.any(sel):
            p = c.pos[sel]
            dvec = c.dir[sel]
            tstar = -p[:, 1] / dvec[:, 1]
            focus_x.extend((p[:, 0] + tstar * dvec[:, 0]).tolist())
        queue.append(children)
    assert len(focus_x) >= m * 0.9, "lost rays through the lens"
    x_focus = float(np.median(focus_x))
    assert x_focus == pytest.approx(x_focus_expect, rel=5e-3), \
        (x_focus, x_focus_expect, f, bfd)


def test_energy_closure_with_roughness_and_grating(scene):
    """Regression: the roughness scattered-lobe amplitudes once conflated
    power FRACTIONS with absolute watts and amplified energy 1e9x. Full
    example scene with a rough sphere face + a transmission grating must
    still close the ledger below 1e-3."""
    import common as _c
    model = _c.load_model(MODEL_JSON)
    db = scene.matdb
    from raytracer.materials import load_coatings as _lc
    sc2 = Scene(model, db, _lc(db=db),
                rough_specs=[_c.parse_rough_spec(
                    "Body006.Revolution002.Face1:200:lcorr=5")],
                grating_specs=[_c.parse_grating_spec(
                    "Body003.Pad001.Face6:600:u")])
    cfg = TraceConfig(rays=1500, n_lambda=2, seed=5)
    tracer = Tracer(sc2, cfg, {})
    rng = np.random.default_rng(5)
    batches = [sample_source(sc2, sc2.bodies[b], s, i, cfg.rays,
                             cfg.n_lambda, rng, ledger=tracer.ledger)
               for i, (b, s) in enumerate(sc2.sources)]
    result = tracer.run(batches)
    rep = result.ledger.report(result.source_names)
    for name, s in rep["sources"].items():
        assert s["closure_error"] < 1e-3, (name, s["closure_error"])
    # roughness must actually have scattered something
    assert rep["by_body_W"] or any(
        s["absorbed_surface"] > 0 for s in rep["sources"].values())
