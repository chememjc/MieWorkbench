"""Transform-engine unit tests: quaternion identities, rotation about a
point, composition/repeatability, and reference-point resolution."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.transforms import (  # noqa: E402
    BodyState, FaceInfo, Operation, Placement, ReferenceResolver,
    apply_world, axis_angle_quat, matrix_to_quat, quat_to_matrix,
    rotate_matrix, translate_matrix,
)


def approx(a, b, tol=1e-9):
    assert np.allclose(a, b, atol=tol), "%s != %s" % (a, b)


# -- quaternion / matrix round-trips ----------------------------------------
def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        R = quat_to_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        q2 = matrix_to_quat(R)
        # q and -q are the same rotation
        assert np.allclose(q, q2, atol=1e-9) or \
            np.allclose(q, -q2, atol=1e-9)


def test_axis_angle_basics():
    q = axis_angle_quat([0, 0, 1], 90.0)
    approx(quat_to_matrix(q) @ [1, 0, 0], [0, 1, 0])
    q = axis_angle_quat([0, 0, 2.0], 90.0)   # axis normalization
    approx(quat_to_matrix(q) @ [1, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError):
        axis_angle_quat([0, 0, 0], 10.0)


# -- rotation about a point ---------------------------------------------------
def test_rotate_about_point_moves_origin_correctly():
    # rotating +90deg about z at point (1,0,0): origin -> (1,-1,0)
    M = rotate_matrix([0, 0, 1], 90.0, about_mm=[1, 0, 0])
    approx((M @ [0, 0, 0, 1])[:3], [1, -1, 0])
    # the pivot itself is a fixed point
    approx((M @ [1, 0, 0, 1])[:3], [1, 0, 0])


def test_rotation_about_body_point_preserves_that_point():
    pl = Placement([10, 5, -3], axis_angle_quat([1, 1, 0], 30))
    pivot = np.array([2.0, -1.0, 4.0])
    M = rotate_matrix([0, 1, 0], 47.0, pivot)
    pl2 = apply_world(M, pl)
    # a body-local point that sat AT the pivot must not move
    local_at_pivot = pl.inverse_point(pivot)
    approx(pl2.transform_point(local_at_pivot), pivot)


def test_four_quarter_turns_are_identity():
    pl = Placement([1, 2, 3], axis_angle_quat([0, 1, 0], 20))
    M = rotate_matrix([0, 0, 1], 90.0, about_mm=[5, 5, 0])
    out = pl
    for _ in range(4):
        out = apply_world(M, out)
    approx(out.pos, pl.pos)
    assert np.allclose(out.quat, pl.quat, atol=1e-9) or \
        np.allclose(out.quat, -pl.quat, atol=1e-9)


def test_translate_repeat_accumulates():
    pl = Placement()
    M = translate_matrix([0, 0, 2.5])
    for _ in range(3):
        pl = apply_world(M, pl)
    approx(pl.pos, [0, 0, 7.5])


# -- placement dict round-trip ------------------------------------------------
def test_placement_dict_roundtrip():
    d = {"pos_mm": [1.0, -2.0, 3.0],
         "quat": list(axis_angle_quat([0, 0, 1], 45.0))}
    pl = Placement.from_dict(d)
    d2 = pl.to_dict()
    approx(d2["pos_mm"], d["pos_mm"])
    approx(d2["quat"], d["quat"])


# -- reference resolution -----------------------------------------------------
def make_lens_body(pos=(0, 0, 0)):
    """A fake 'lens': largest face is a disc at local x=+1 facing +x,
    bbox spans x in [-1, +1], com slightly off-center."""
    body_dict = {
        "name": "Lens", "label": "Lens",
        "placement": {"pos_mm": list(pos), "quat": [0, 0, 0, 1]},
        "center_of_mass_mm": [pos[0] + 0.2, pos[1], pos[2]],
        "bbox_mm": [pos[0] - 1, pos[1] - 5, pos[2] - 5,
                    pos[0] + 1, pos[1] + 5, pos[2] + 5],
    }
    # tessellation face metadata is BODY-LOCAL metres (placement stripped)
    faces = [
        {"id": "Lens.Rev.Face1", "centroid_m": [1 / 1000.0, 0.0, 0.0],
         "normal_hint": [1, 0, 0], "area_m2": 3.0e-4},
        {"id": "Lens.Rev.Face2", "centroid_m": [-1 / 1000.0, 0.0, 0.0],
         "normal_hint": [-1, 0, 0], "area_m2": 2.0e-4},
    ]
    return BodyState.from_worker(body_dict, faces)


def test_reference_points():
    lens = make_lens_body(pos=(10, 0, 0))
    rr = ReferenceResolver({"Lens": lens})
    approx(rr.resolve_point({"kind": "origin"}), [0, 0, 0])
    approx(rr.resolve_point({"kind": "fixed", "point_mm": [1, 2, 3]}),
           [1, 2, 3])
    approx(rr.resolve_point({"kind": "com", "body": "Lens"}),
           [10.2, 0, 0])
    approx(rr.resolve_point({"kind": "bbox_center", "body": "Lens"}),
           [10, 0, 0])
    # optical center: largest face (Face1 at x=11, normal +x); closest
    # point on that line to bbox center (10,0,0) is (10,0,0)
    approx(rr.resolve_point({"kind": "optical_center", "body": "Lens"}),
           [10, 0, 0])
    approx(rr.resolve_point({"kind": "face_point", "body": "Lens",
                             "face": "Lens.Rev.Face1", "t_mm": 5.0}),
           [16, 0, 0])
    # label lookup + unknown body
    approx(rr.resolve_point({"kind": "com", "body": "Lens"}),
           rr.resolve_point({"kind": "com", "body": "Lens"}))
    with pytest.raises(KeyError):
        rr.resolve_point({"kind": "com", "body": "Nope"})


def test_reference_points_follow_current_placement():
    lens = make_lens_body(pos=(10, 0, 0))
    rr = ReferenceResolver({"Lens": lens})
    # GUI-side move: +5mm in y without any FreeCAD round-trip
    lens.current = apply_world(translate_matrix([0, 5, 0]), lens.current)
    approx(rr.resolve_point({"kind": "com", "body": "Lens"}), [10.2, 5, 0])
    approx(rr.resolve_point({"kind": "optical_center", "body": "Lens"}),
           [10, 5, 0])


def test_axes():
    lens = make_lens_body()
    rr = ReferenceResolver({"Lens": lens})
    approx(rr.resolve_axis({"kind": "global", "axis": "y"}), [0, 1, 0])
    approx(rr.resolve_axis({"kind": "vector", "vector": [0, 0, 9]}),
           [0, 0, 1])
    approx(rr.resolve_axis({"kind": "face_normal", "body": "Lens",
                            "face": "Lens.Rev.Face1"}), [1, 0, 0])
    approx(rr.resolve_axis({"kind": "optical_axis", "body": "Lens"}),
           [1, 0, 0])
    approx(rr.resolve_axis({"kind": "two_points",
                            "a": {"kind": "origin"},
                            "b": {"kind": "fixed",
                                  "point_mm": [0, 3, 4]}}),
           [0, 0.6, 0.8])
    with pytest.raises(ValueError):
        rr.resolve_axis({"kind": "vector", "vector": [0, 0, 0]})


# -- operations ---------------------------------------------------------------
def test_operation_translate_vector_and_toward():
    lens = make_lens_body(pos=(0, 0, 0))
    screen = make_lens_body(pos=(50, 0, 0))
    screen.name = screen.label = "Screen"
    rr = ReferenceResolver({"Lens": lens, "Screen": screen})

    Operation("translate", {"vector_mm": [0, 0, 4]}).apply(rr, lens)
    approx(lens.current.pos, [0, 0, 4])

    # move lens 10mm toward the screen's center of mass
    Operation("translate", {
        "from": {"kind": "com", "body": "Lens"},
        "toward": {"kind": "com", "body": "Screen"},
        "distance_mm": 10.0}).apply(rr, lens)
    approx(lens.current.pos[0], 10.0 * (50.0 / np.hypot(50, 4)), tol=1e-6)


def test_operation_rotate_about_other_body():
    lens = make_lens_body(pos=(10, 0, 0))
    pivotb = make_lens_body(pos=(0, 0, 0))
    pivotb.name = pivotb.label = "Pivot"
    rr = ReferenceResolver({"Lens": lens, "Pivot": pivotb})
    op = Operation("rotate", {"axis": {"kind": "global", "axis": "z"},
                              "angle_deg": 90.0,
                              "about": {"kind": "bbox_center",
                                        "body": "Pivot"}})
    op.apply(rr, lens)
    approx(lens.com_world(), [0, 10.2, 0], tol=1e-9)
    # repeatable: three more quarter turns come back around
    for _ in range(3):
        op.apply(rr, lens)
    approx(lens.com_world(), [10.2, 0, 0], tol=1e-9)


def test_operation_matrix_resolved_at_apply_time():
    """'Apply again' must use live reference positions, not stale ones."""
    a = make_lens_body(pos=(0, 0, 0))
    b = make_lens_body(pos=(10, 0, 0))
    a.name = a.label = "A"
    b.name = b.label = "B"
    rr = ReferenceResolver({"A": a, "B": b})
    op = Operation("translate", {
        "from": {"kind": "bbox_center", "body": "A"},
        "toward": {"kind": "bbox_center", "body": "B"},
        "distance_mm": 4.0})
    op.apply(rr, a)          # A at x=4
    op.apply(rr, a)          # remaining gap 6mm, still toward B
    approx(a.current.pos, [8, 0, 0])
    op.apply(rr, a)          # gap 2mm < distance: overshoot allowed
    approx(a.current.pos, [12, 0, 0])
    # now A is PAST B: 'toward B' flips direction (live resolution)
    op.apply(rr, a)
    approx(a.current.pos, [8, 0, 0])
# ---- Round C: alignment / snap / projection / euler / set_placement -------
# (appended to mieworkbench/tests/test_transforms.py)

from mieworkbench.core.transforms import (  # noqa: E402
    align_rotation, euler_from_quat, project_point_on_axis, quat_from_euler,
    quat_to_axis_angle, snap_to_axis_ops,
)


def _axis_body(normal=(1, 0, 0), area=1.0, center=(0, 0, 0), pos=(0, 0, 0)):
    """A body whose single usable face gives it an optical axis == normal
    and an optical center at the bbox center (== `center` at identity)."""
    face = FaceInfo("F", np.array(center, float), np.array(normal, float),
                    float(area))
    return BodyState("b", "b", Placement(list(pos), [0, 0, 0, 1]),
                     com_local_mm=np.array(center, float),
                     bbox_center_local_mm=np.array(center, float),
                     faces=[face])


def test_align_rotation_maps_axis():
    for a, b in [([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [1, 0, 0]),
                 ([1, 2, 3], [-2, 1, 5])]:
        q = align_rotation(a, b)
        R = quat_to_matrix(q)
        an = np.array(a, float) / np.linalg.norm(a)
        bn = np.array(b, float) / np.linalg.norm(b)
        approx(R @ an, bn)


def test_align_rotation_hemisphere_preserving():
    # target points into the opposite hemisphere -> align to -target so the
    # element only tilts by the acute angle (never a ~180 deg flip)
    q = align_rotation([1, 0, 0], [-0.98, 0.2, 0.0], keep_hemisphere=True)
    _axis, ang = quat_to_axis_angle(q)
    assert ang < 90.0
    # without the guard it flips the full ~180 deg
    q2 = align_rotation([1, 0, 0], [-0.98, 0.2, 0.0], keep_hemisphere=False)
    _axis2, ang2 = quat_to_axis_angle(q2)
    assert ang2 > 90.0


def test_align_rotation_antiparallel():
    q = align_rotation([1, 0, 0], [-1, 0, 0], keep_hemisphere=False)
    approx(quat_to_matrix(q) @ [1, 0, 0], [-1, 0, 0])


def test_align_rotation_already_aligned_is_identity():
    q = align_rotation([0, 0, 1], [0, 0, 2])
    approx(q, [0, 0, 0, 1])


def test_project_point_on_axis():
    foot, t = project_point_on_axis([1, 5, 0], [0, 0, 0], [1, 0, 0])
    approx(foot, [1, 0, 0])
    assert np.isclose(t, 1.0)
    # non-unit direction is normalized
    foot2, t2 = project_point_on_axis([0, 0, 4], [0, 0, 1], [0, 0, 9])
    approx(foot2, [0, 0, 4])
    assert np.isclose(t2, 3.0)


def test_euler_quat_roundtrip():
    for rx, ry, rz in [(10, 20, 30), (-45, 15, 80), (5, -60, -120),
                       (0, 0, 0), (179, 10, -179)]:
        q = quat_from_euler(rx, ry, rz)
        e = euler_from_quat(q)
        approx(e, [rx, ry, rz], tol=1e-6)


def test_euler_quat_roundtrip_gimbal():
    # at ry=+/-90 the split is arbitrary; quat->euler->quat must still
    # reproduce the SAME rotation even though the angles differ
    for ry in (90.0, -90.0):
        q = quat_from_euler(25.0, ry, 40.0)
        q2 = quat_from_euler(*euler_from_quat(q))
        R, R2 = quat_to_matrix(q), quat_to_matrix(q2)
        assert np.allclose(R, R2, atol=1e-9)


def test_set_placement_operation_is_absolute():
    b = _axis_body(pos=(7, 8, 9))
    res = ReferenceResolver({"b": b})
    q = quat_from_euler(0, 90, 0).tolist()
    Operation("set_placement", {"pos_mm": [1, 2, 3], "quat": q}).apply(res, b)
    approx(b.current.pos, [1, 2, 3])
    approx(b.current.quat, q)


def test_snap_to_axis_composition():
    # body optical axis +x at center (2,0,0); snap onto the y-axis line
    # through (0,10,0). Expect: axis -> +/-y, center on the line, and the
    # along-axis coordinate of the projected center preserved.
    b = _axis_body(normal=(1, 0, 0), center=(0, 0, 0), pos=(2, 0, 0))
    res = ReferenceResolver({"b": b})
    c0 = b.optical_center_world().copy()
    tgt_pt = np.array([0.0, 10.0, 0.0])
    tgt_ax = np.array([0.0, 1.0, 0.0])
    for op in snap_to_axis_ops(b, tgt_pt, tgt_ax):
        op.apply(res, b)
    # aligned
    assert abs(abs(np.dot(b.optical_axis_world(), tgt_ax)) - 1.0) < 1e-9
    # centered on the target line (zero perpendicular distance)
    foot, t = project_point_on_axis(b.optical_center_world(), tgt_pt, tgt_ax)
    approx(b.optical_center_world(), foot)
    # along-axis coordinate == projection of the ORIGINAL center
    _foot0, t0 = project_point_on_axis(c0, tgt_pt, tgt_ax)
    assert np.isclose(t, t0)


def test_snap_to_axis_noop_when_already_on_axis():
    b = _axis_body(normal=(0, 1, 0), center=(0, 0, 0), pos=(0, 5, 0))
    # already on the y-axis line pointing +y -> no operations needed
    ops = snap_to_axis_ops(b, np.array([0.0, 0.0, 0.0]),
                           np.array([0.0, 1.0, 0.0]))
    assert ops == []
