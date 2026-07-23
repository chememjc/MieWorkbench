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
# Run: "$MIEWB_OPTICS_PYTHON" -m pytest scripts/raytracer/tests/test_viz_pattern.py -v
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


# =============================================================================
# --viz-pattern grammar (common.parse_viz_pattern_spec) — stdlib only.
# =============================================================================
def test_parse_fan_default():
    spec = common.parse_viz_pattern_spec("fan")
    assert spec == {"kind": "fan", "n": 5}


def test_parse_fan_with_n():
    spec = common.parse_viz_pattern_spec("fan:n=9")
    assert spec == {"kind": "fan", "n": 9}


def test_parse_fan_rejects_bad_grammar():
    import pytest
    for bad in ("fan:n=0", "fan:n=-1", "fan:n=x", "fan:bogus=1",
                "fan:n", "spiral"):
        with pytest.raises(ValueError):
            common.parse_viz_pattern_spec(bad)


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


# =============================================================================
# fan pattern: central ray + up to 4 cardinal rays + rim fillers.
# =============================================================================
FAN5 = {"kind": "fan", "n": 5}


def test_fan_geometry_and_determinism():
    scene = build_scene()
    bidx, src = scene.sources[0]
    b1 = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, FAN5, 3)
    b2 = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, FAN5, 3)
    assert len(b1.pos) == 5
    assert np.array_equal(b1.pos, b2.pos)          # deterministic

    # source face is a 10x10mm square centered on the beam axis (x=-20mm):
    # ray 0 is the centroid, rays 1..4 are the +y/-y/+x/-x cardinals at 95%
    # of the (square, so axis-aligned) half-width -> inside the aperture
    centroid = b1.pos[0]
    assert np.allclose(centroid, [-0.02, 0.0, 0.0], atol=1e-12)
    half = 0.005
    rel = b1.pos[1:] - centroid
    expected = 0.95 * half
    assert np.allclose(sorted(rel[:, 1].tolist(), reverse=True)[:1],
                       [expected], atol=1e-9)              # +y (top)
    assert np.allclose(sorted(rel[:, 1].tolist())[:1],
                       [-expected], atol=1e-9)             # -y (bottom)
    assert np.allclose(sorted(rel[:, 2].tolist(), reverse=True)[:1],
                       [expected], atol=1e-9)              # +x (right)
    assert np.allclose(sorted(rel[:, 2].tolist())[:1],
                       [-expected], atol=1e-9)             # -x (left)
    # every cardinal strictly inside the 10x10mm aperture
    assert np.all(np.abs(rel) <= half + 1e-9)
    # all rays share the emit direction (+x toward the detector)
    assert np.allclose(b1.dir, [1.0, 0.0, 0.0], atol=1e-12)


def test_fan_ray_count_and_rim_fillers_inside_aperture():
    scene = build_scene()
    bidx, src = scene.sources[0]
    for n in (1, 2, 4, 5, 9):
        pattern = {"kind": "fan", "n": n}
        b = sample_viz_pattern(scene, scene.bodies[bidx], src, 0, pattern, 3)
        assert len(b.pos) == n
        half = 0.005 + 1e-9
        assert np.all(np.abs(b.pos[:, 1]) <= half)
        assert np.all(np.abs(b.pos[:, 2]) <= half)
    # n=1 is exactly the centroid
    b1 = sample_viz_pattern(scene, scene.bodies[bidx], src, 0,
                            {"kind": "fan", "n": 1}, 3)
    assert np.allclose(b1.pos[0], [-0.02, 0.0, 0.0], atol=1e-12)


def test_fan_unknown_kind_rejected():
    scene = build_scene()
    bidx, src = scene.sources[0]
    import pytest
    with pytest.raises(ValueError):
        sample_viz_pattern(scene, scene.bodies[bidx], src, 0,
                           {"kind": "spiral"}, 3)


def test_physics_bit_identical_with_and_without_fan_pattern():
    cube_plain, audit_plain, viz_plain = run_trace_once(None)
    cube_fan, audit_fan, viz_fan = run_trace_once(FAN5)
    assert np.array_equal(cube_plain, cube_fan)
    for src_name, rep in audit_plain["sources"].items():
        for k, v in rep.items():
            assert audit_fan["sources"][src_name][k] == v, (src_name, k)
    n_rays = 5
    assert len(viz_fan) == 2 * n_rays
    starts = viz_fan[:, 3:6]
    assert np.sum(np.isclose(starts[:, 0], -0.02)) == n_rays


# =============================================================================
# Spherical emit face (divergent laser) — the fan must land on the cap
# with per-point (diverging) normal directions instead of being skipped.
# =============================================================================
def _spherical_source_body(name="DivSrc", x_apex=-0.02, roc=0.2,
                           aperture_r=0.001, power_mW=5.0):
    """Source whose emit face is a spherical cap of curvature radius
    `roc`, apex at (x_apex, 0, 0), bulging toward +x (the scene)."""
    centre = np.array([x_apex - roc, 0.0, 0.0])
    x_rim = centre[0] + np.sqrt(roc ** 2 - aperture_r ** 2)
    theta = np.linspace(0.0, 2 * np.pi, 65)[:-1]
    rim = np.stack([np.full_like(theta, x_rim),
                    aperture_r * np.cos(theta),
                    aperture_r * np.sin(theta)], axis=-1)
    face = {
        "id": "%s.Pad.Face1" % name,
        "surface": {"type": "sphere", "center": centre.tolist(),
                    "radius": roc},
        "orientation_outward": True,
        "area_m2": float(np.pi * aperture_r ** 2),
        "fingerprint": {},
        "mesh_stl": "",
        "trim_polylines_xyz": [[list(p) for p in rim]],
    }
    src = {"power_mW": power_mW, "lambdac_nm": 633.0,
           "emit_face": face["id"], "coherent": False}
    return {"name": name, "label": name, "role": "source",
            "source": src, "faces": [face]}


def test_fan_on_spherical_cap_diverges():
    model = make_model([
        _spherical_source_body(),
        detector_body("Det", x=0.02, half=0.01),
    ])
    db = MaterialDB.load()
    scene = Scene(model, db, load_coatings(db=db))
    (bidx, src), = scene.sources
    body = scene.bodies[bidx]

    batch = sample_viz_pattern(scene, body, src, 0,
                               {"kind": "fan", "n": 5}, 1)
    assert batch is not None and len(batch) == 5

    centre = np.array([-0.02 - 0.2, 0.0, 0.0])
    # every point lies ON the sphere
    r = np.linalg.norm(batch.pos - centre, axis=-1)
    assert np.allclose(r, 0.2, atol=1e-9)
    # ...within the aperture (rim radius 1mm around the x axis)
    assert np.all(np.linalg.norm(batch.pos[:, 1:], axis=-1) <= 1.001e-3)
    # central ray starts at the apex and heads +x
    assert np.linalg.norm(batch.pos[0][1:]) <= 1e-5
    assert batch.dir[0][0] > 0.999
    # the fan DIVERGES: outer directions differ from the axis
    outer = batch.dir[1:]
    assert np.all(outer[:, 0] > 0.9)          # still broadly +x
    axis_dots = outer @ np.array([1.0, 0.0, 0.0])
    assert np.all(axis_dots < 1.0 - 1e-8)     # but none exactly on-axis
    # deterministic
    batch2 = sample_viz_pattern(scene, body, src, 0,
                                {"kind": "fan", "n": 5}, 1)
    assert np.array_equal(batch.pos, batch2.pos)
    assert np.array_equal(batch.dir, batch2.dir)


def test_broadband_source_replicates_pattern_across_strata():
    """A broadband source's viz pattern emits one ray PER WAVELENGTH
    STRATUM from every pattern point (the red/green/blue bundle that
    makes chromatic behavior visible in the overlay); a monochromatic
    source keeps exactly one ray per point."""
    scene = build_scene()
    bidx, src = scene.sources[0]

    mono = sample_viz_pattern(scene, scene.bodies[bidx], src, 0,
                              PATTERN, 3)
    n_points = 1 + 3 * 8
    assert len(mono.pos) == n_points          # 1 stratum -> unchanged
    assert len(np.unique(mono.lam)) == 1

    wide = dict(src)
    wide["lambdamin_nm"] = 450.0
    wide["lambdamax_nm"] = 650.0
    broad = sample_viz_pattern(scene, scene.bodies[bidx], wide, 0,
                               PATTERN, 3)
    assert len(broad.pos) == 3 * n_points
    assert len(np.unique(broad.lam)) == 3
    # consecutive triplets share the SAME position with distinct lambdas
    trip = broad.pos.reshape(n_points, 3, 3)
    assert np.allclose(trip[:, 0], trip[:, 1])
    assert np.allclose(trip[:, 0], trip[:, 2])
    lam_trip = broad.lam.reshape(n_points, 3)
    assert (np.diff(np.sort(lam_trip[0])) > 0).all()
