# =============================================================================
# test_viz_pattern.py — the --viz-pattern overlay must be PURELY visual.
#
# Gates:
#   1. sample_viz_pattern geometry: central ray + rings, deterministic,
#      inside the emit face, correct counts.
#   2. Physics invariance: tracing the SAME scene/seed with and without a
#      viz pattern produces bit-identical detector cubes and energy audit
#      (the pattern rays run in a separate viz-only pass).
#   3. The overlay rays actually land in the viz store (rays.npy content).
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_viz_pattern.py -v
# =============================================================================
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

import common                                            # noqa: E402
from raytracer.materials import MaterialDB, load_coatings  # noqa: E402
from raytracer.scene import Scene                        # noqa: E402
from raytracer.sources import (sample_source,            # noqa: E402
                               sample_viz_pattern)
from raytracer.tracer import Tracer, TraceConfig         # noqa: E402
from raytracer.detector import DetectorGrid              # noqa: E402
from raytracer import gather                             # noqa: E402

from .scenehelpers import make_model, source_body, detector_body  # noqa: E402


def build_scene():
    """10x10mm square emitter at x=-20mm -> detector plane at x=+20mm."""
    model = make_model([
        source_body("Src", x=-0.02, half=0.005, power_mW=5.0,
                    lambdac_nm=633.0, coherent=True),
        detector_body("Det", x=0.02, half=0.01),
    ])
    db = MaterialDB.load()
    return Scene(model, db, load_coatings(db=db))


PATTERN = {"kind": "rings", "dr_mm": 1.0, "nper": 8, "nrings": 3}


def test_pattern_geometry_and_determinism():
    scene = build_scene()
    bidx, src = scene.sources[0]
    b1 = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, PATTERN, 3)
    b2 = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, PATTERN, 3)
    # 1 central + 3 rings * 8 (emit face is 10x10 mm: rings at 1,2,3 mm
    # from center fit inside except ring-3 corners... all |p| <= 3mm fit
    # in a 5mm half-width square)
    assert len(b1.pos) == 1 + 3 * 8
    assert np.array_equal(b1.pos, b2.pos)          # deterministic
    assert np.array_equal(b1.lam, b2.lam)
    # central ray exactly at the face centroid ring radii as designed
    r = np.linalg.norm(b1.pos[1:] - b1.pos[0], axis=-1)
    expected = np.repeat([1e-3, 2e-3, 3e-3], 8)
    assert np.allclose(np.sort(r), np.sort(expected), atol=1e-9)
    # all rays share the emit direction (+x toward the detector)
    assert np.allclose(b1.dir, [1.0, 0.0, 0.0], atol=1e-12)


def test_rings_clipped_to_face():
    scene = build_scene()
    bidx, src = scene.sources[0]
    big = {"kind": "rings", "dr_mm": 4.0, "nper": 8, "nrings": None}
    b = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, big, 3)
    # emit face is 10x10 mm: ring at 4mm keeps only the points inside the
    # square (|y|,|z| <= 5mm - all 8 qualify), ring at ~7mm (max rim
    # radius ~7.07mm) keeps only the 4 on-diagonal... compute: at r=4mm
    # all 8 inside; the open-ended ring count reaches the rim radius
    assert len(b.pos) >= 1 + 8
    half = 0.005 + 1e-12
    assert np.all(np.abs(b.pos[:, 1]) <= half)
    assert np.all(np.abs(b.pos[:, 2]) <= half)


def run_trace_once(viz_pattern):
    scene = build_scene()
    grids = {fid: DetectorGrid(scene.faces[fid], 128, 4,
                               (600e-9, 660e-9),
                               label=scene.faces[fid].id)
             for fid in scene.detector_faces}
    cfg = TraceConfig(rays=20000, n_lambda=1, seed=11,
                      viz_rays=0 if viz_pattern else 200)
    tracer = Tracer(scene, cfg, grids)
    rng = np.random.default_rng(11)
    bidx, src = scene.sources[0]
    batch = sample_source(scene, scene.bodies[bidx], src, 0, cfg.rays, 1,
                          rng, ledger=tracer.ledger)
    result = tracer.run([batch])

    viz_arr = result.viz.as_array()
    if viz_pattern:
        viz_cfg = TraceConfig(rays=1, n_lambda=1, seed=11,
                              viz_rays=1 << 30)
        viz_tracer = Tracer(scene, viz_cfg, {})
        vb = sample_viz_pattern(scene, scene.bodies[bidx], src, 0,
                                viz_pattern, 1)
        viz_arr = viz_tracer.run([vb]).viz.as_array()

    det = list(grids.values())[0]
    area = scene.emit_faces[bidx].area_m2 / cfg.rays
    gather.render_coherent(det, {(0, 0, 0): area}, backend="numpy",
                           min_eff_samples=100)
    ledger_rep = tracer.ledger.report(["Src"])
    return det.inc.copy(), ledger_rep, viz_arr


def test_physics_bit_identical_with_and_without_pattern():
    cube_plain, audit_plain, viz_plain = run_trace_once(None)
    cube_pat, audit_pat, viz_pat = run_trace_once(PATTERN)
    # detector cube: BIT identical
    assert np.array_equal(cube_plain, cube_pat)
    # energy ledger identical
    for src_name, rep in audit_plain["sources"].items():
        for k, v in rep.items():
            assert audit_pat["sources"][src_name][k] == v, (src_name, k)
    # and the overlay actually produced deterministic pattern polylines:
    # 25 rays x 2 legs each (source -> detector screen, then the
    # transparent-screen pass-through -> scene escape)
    n_rays = 1 + 3 * 8
    assert len(viz_pat) == 2 * n_rays
    starts = viz_pat[:, 3:6]
    assert np.sum(np.isclose(starts[:, 0], -0.02)) == n_rays
    assert len(viz_plain) > 0                    # sanity on the default path
