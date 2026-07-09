"""Pure-math tests for scripts/train_solver.py (the shared stdlib chain
solver). Runs offscreen-free — no Qt, no FreeCAD; numpy appears only to
cross-check against mieworkbench.core.transforms (the parity that keeps
GUI and permute placements identical)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train_solver as ts  # noqa: E402

from mieworkbench.core import transforms as tr  # noqa: E402


# ---------------------------------------------------------------------------
# eval_expr / expr_names
# ---------------------------------------------------------------------------
def test_eval_plain_numbers():
    assert ts.eval_expr("3") == 3.0
    assert ts.eval_expr("2.5") == 2.5
    assert ts.eval_expr(4) == 4.0
    assert ts.eval_expr("-7.5") == -7.5
    assert ts.eval_expr("+3") == 3.0


def test_eval_arithmetic_precedence():
    assert ts.eval_expr("2+3*4") == 14.0
    assert ts.eval_expr("(2+3)*4") == 20.0
    assert ts.eval_expr("10/4") == 2.5
    assert ts.eval_expr("1 - 2 - 3") == -4.0
    assert ts.eval_expr("-(2+3)") == -5.0


def test_eval_variables():
    v = {"gap": 10.0, "f1": 25.0}
    assert ts.eval_expr("2*gap + 5", v) == 25.0
    assert ts.eval_expr("f1 - gap/2", v) == 20.0
    assert ts.eval_expr("gap", v) == 10.0


def test_eval_unknown_variable():
    with pytest.raises(ts.ExprError, match="unknown variable 'gap'"):
        ts.eval_expr("gap+1", {"other": 1.0})


def test_eval_division_by_zero():
    with pytest.raises(ts.ExprError, match="division by zero"):
        ts.eval_expr("1/0")
    with pytest.raises(ts.ExprError, match="division by zero"):
        ts.eval_expr("1/(a-a)", {"a": 3.0})


@pytest.mark.parametrize("bad", [
    "__import__('os')",
    "a.b",
    "f(1)",
    "2**8",
    "1 if 0 else 2",
    "[1,2]",
    "a % 2",
    "1 < 2",
    "lambda: 1",
    "True",
    "'text'",
    "",
    "   ",
    "2 +",
])
def test_eval_rejects_everything_else(bad):
    with pytest.raises(ts.ExprError):
        ts.eval_expr(bad, {"a": 1.0})


def test_expr_names():
    assert ts.expr_names("2*gap + f1/2") == {"gap", "f1"}
    assert ts.expr_names("3.5") == set()
    assert ts.expr_names(7) == set()


# ---------------------------------------------------------------------------
# resolve_variables + circular references
# ---------------------------------------------------------------------------
def test_resolve_variables_chain():
    out = ts.resolve_variables({"a": "1", "b": "a*2", "c": "b + a"})
    assert out == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_resolve_variables_order_independent():
    # definition order in the dict must not matter
    out = ts.resolve_variables({"z": "y + 1", "y": "x * 2", "x": "5"})
    assert out == {"x": 5.0, "y": 10.0, "z": 11.0}


def test_resolve_variables_self_cycle():
    with pytest.raises(ts.CycleError) as ei:
        ts.resolve_variables({"a": "a + 1"})
    assert "a -> a" in str(ei.value)


def test_resolve_variables_indirect_cycle_names_path():
    with pytest.raises(ts.CycleError) as ei:
        ts.resolve_variables({"a": "b+1", "b": "c*2", "c": "a-1", "d": "1"})
    msg = str(ei.value)
    assert "circular variable reference" in msg
    # the full loop appears, whatever its starting point
    assert msg.count("->") == 3


def test_resolve_variables_unknown_name():
    with pytest.raises(ts.ExprError, match="unknown variable"):
        ts.resolve_variables({"a": "ghost * 2"})


# ---------------------------------------------------------------------------
# Quaternion / matrix parity vs mieworkbench.core.transforms (numpy)
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(20260709)


def random_quat():
    q = RNG.normal(size=4)
    return (q / np.linalg.norm(q)).tolist()


@pytest.mark.parametrize("trial", range(20))
def test_quat_to_matrix_parity(trial):
    q = random_quat()
    ours = np.array(ts.quat_to_matrix3(q))
    ref = tr.quat_to_matrix(np.array(q))
    assert np.allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_matrix_to_quat_parity(trial):
    q = random_quat()
    R = tr.quat_to_matrix(np.array(q))
    ours = np.array(ts.matrix3_to_quat([list(r) for r in R]))
    ref = tr.matrix_to_quat(R)
    assert np.allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_axis_angle_parity(trial):
    axis = RNG.normal(size=3).tolist()
    angle = float(RNG.uniform(-180, 180))
    ours = np.array(ts.axis_angle_matrix3(axis, angle))
    ref = tr.quat_to_matrix(tr.axis_angle_quat(np.array(axis), angle))
    assert np.allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("trial", range(10))
def test_rotate_matrix_parity(trial):
    axis = RNG.normal(size=3).tolist()
    angle = float(RNG.uniform(-180, 180))
    about = RNG.uniform(-50, 50, size=3).tolist()
    ours = np.array(ts.rotate_matrix(axis, angle, about))
    ref = tr.rotate_matrix(np.array(axis), angle, np.array(about))
    assert np.allclose(ours, ref, atol=1e-10)


def test_euler_matrix_xyz_matches_transforms():
    rx, ry, rz = 21.0, -34.0, 55.0
    ours = np.array(ts.euler_matrix3("xyz", rx, ry, rz))
    ref = tr.quat_to_matrix(tr.quat_from_euler(rx, ry, rz))
    assert np.allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("order", ts.ROT_ORDERS)
@pytest.mark.parametrize("trial", range(5))
def test_euler_roundtrip_all_orders(order, trial):
    rx = float(RNG.uniform(-80, 80))
    ry = float(RNG.uniform(-80, 80))
    rz = float(RNG.uniform(-80, 80))
    R = ts.euler_matrix3(order, rx, ry, rz)
    gx, gy, gz = ts.euler_from_matrix3(R, order)
    R2 = ts.euler_matrix3(order, gx, gy, gz)
    assert np.allclose(np.array(R), np.array(R2), atol=1e-10)


@pytest.mark.parametrize("order", ts.ROT_ORDERS)
def test_euler_gimbal_pole_still_reproduces_matrix(order):
    # drive the middle rotation of each order to +/-90 deg
    mid = order[1]
    angles = {"x": 10.0, "y": 20.0, "z": 30.0}
    angles[mid] = 90.0
    R = ts.euler_matrix3(order, angles["x"], angles["y"], angles["z"])
    gx, gy, gz = ts.euler_from_matrix3(R, order)
    R2 = ts.euler_matrix3(order, gx, gy, gz)
    assert np.allclose(np.array(R), np.array(R2), atol=1e-9)


def test_placement_matrix_roundtrip():
    pl = {"pos_mm": [3.0, -2.0, 7.5], "quat": random_quat()}
    back = ts.matrix_placement(ts.placement_matrix(pl))
    assert np.allclose(back["pos_mm"], pl["pos_mm"], atol=1e-12)
    q1, q2 = np.array(back["quat"]), np.array(pl["quat"])
    assert min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)) < 1e-12


# ---------------------------------------------------------------------------
# Reflection / fold rotation
# ---------------------------------------------------------------------------
def test_reflect_matrix_reflects_points_and_dirs():
    M = ts.reflect_matrix([5.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    # point 2 mm before the plane lands 2 mm after it
    p = ts.mat4_mul(M, ts.translate_matrix([3.0, 1.0, 2.0]))
    assert np.allclose([p[0][3], p[1][3], p[2][3]], [7.0, 1.0, 2.0])


def test_reflect_matrix_is_improper_and_refused_for_placements():
    M = ts.reflect_matrix([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    with pytest.raises(ts.TrainError, match="improper"):
        ts.matrix_placement(M)


def test_fold_rotation_45deg_mirror():
    # mirror at origin, normal bisecting +x / +y: deviates +x into +y
    n = [-1.0, 1.0, 0.0]
    M, d_out = ts.fold_rotation([1.0, 0.0, 0.0], [10.0, 0.0, 0.0], n)
    assert np.allclose(d_out, [0.0, 1.0, 0.0], atol=1e-12)
    pl = ts.apply_to_placement(M, {"pos_mm": [30.0, 0.0, 0.0],
                                   "quat": [0, 0, 0, 1]})   # 20 mm past fold
    assert np.allclose(pl["pos_mm"], [10.0, 20.0, 0.0], atol=1e-9)
    # proper rotation: det == +1 (matrix_placement would raise otherwise)
    R = np.array([row[:3] for row in M[:3]])
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_fold_rotation_fixes_mirror_point():
    M, _ = ts.fold_rotation([1.0, 0.0, 0.0], [10.0, -3.0, 2.0],
                            [-1.0, 1.0, 0.0])
    pl = ts.apply_to_placement(M, {"pos_mm": [10.0, -3.0, 2.0],
                                   "quat": [0, 0, 0, 1]})
    assert np.allclose(pl["pos_mm"], [10.0, -3.0, 2.0], atol=1e-9)


def test_fold_rotation_normal_incidence():
    M, d_out = ts.fold_rotation([1.0, 0.0, 0.0], [5.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0])
    assert np.allclose(d_out, [-1.0, 0.0, 0.0], atol=1e-12)
    pl = ts.apply_to_placement(M, {"pos_mm": [8.0, 0.0, 0.0],
                                   "quat": [0, 0, 0, 1]})
    assert np.allclose(pl["pos_mm"], [2.0, 0.0, 0.0], atol=1e-9)


def test_fold_rotation_z_fold_composes_to_translation():
    # two parallel 45-deg mirrors (a periscope Z-fold): net effect on the
    # far arm is a pure translation, no net rotation. Points on the axis
    # shift onto the displaced axis AND back along it by the extra path.
    M1, d1 = ts.fold_rotation([1, 0, 0], [20.0, 0.0, 0.0], [-1, 1, 0])
    assert np.allclose(d1, [0.0, 1.0, 0.0], atol=1e-12)
    M2, d2 = ts.fold_rotation(d1, [20.0, 15.0, 0.0], [1, -1, 0])
    assert np.allclose(d2, [1.0, 0.0, 0.0], atol=1e-12)
    M = ts.mat4_mul(M2, M1)
    R = np.array([row[:3] for row in M[:3]])
    assert np.allclose(R, np.eye(3), atol=1e-12)
    # fold point (20,0,0): fixed by M1, then rotated about (20,15,0) by
    # -90 deg -> (5,15,0); net translation is (-15, +15, 0)
    assert np.allclose([M[0][3], M[1][3], M[2][3]], [-15.0, 15.0, 0.0],
                       atol=1e-9)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def test_stable_up():
    assert np.allclose(ts.stable_up([1, 0, 0]), [0, 0, 1])
    up = ts.stable_up([0, 0, 1])            # vertical beam falls back to +y
    assert np.allclose(up, [0, 1, 0])
    up = ts.stable_up([1, 0, 1])
    assert abs(ts.vdot(up, ts.vunit([1, 0, 1]))) < 1e-12


def test_frame_basis_right_handed():
    f = ts.make_frame([0, 0, 0], [1, 0, 0])
    u, v, d = ts.frame_basis(f)
    assert np.allclose(np.cross(u, v), d, atol=1e-12)
    assert np.allclose(v, [0, 0, 1])        # up = world z for an x beam
    assert np.allclose(u, [0, 1, 0])


# ---------------------------------------------------------------------------
# place_chained / derive_edge
# ---------------------------------------------------------------------------
LENS = {
    "label": "L1", "mode": "chained", "ref": "SRC", "port": "out",
    "distance": "10", "local": {
        "entry": [-2.0, 0.0, 0.0], "exit": [2.0, 0.0, 0.0],
        "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]},
}
X_FRAME = {"origin": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0],
           "up": [0.0, 0.0, 1.0]}


def test_place_chained_straight():
    pl = ts.place_chained(X_FRAME, LENS, {})
    # entry vertex (local -2,0,0) lands 10 mm down-beam
    assert np.allclose(pl["pos_mm"], [12.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(pl["quat"], [0, 0, 0, 1], atol=1e-12)


def test_place_chained_distance_expression():
    rec = dict(LENS, distance="2*gap + 1")
    pl = ts.place_chained(X_FRAME, rec, {"gap": 7.0})
    assert np.allclose(pl["pos_mm"], [17.0, 0.0, 0.0], atol=1e-12)


def test_place_chained_decenter_basis():
    # beam +x, up +z: u = up x dir = +y, v = +z
    rec = dict(LENS, decenter_x="1.5", decenter_y="-0.5")
    pl = ts.place_chained(X_FRAME, rec, {})
    assert np.allclose(pl["pos_mm"], [12.0, 1.5, -0.5], atol=1e-12)


def test_place_chained_tilt_about_entrance():
    rec = dict(LENS, tilt_rx="30")           # about u (= +y here)
    pl = ts.place_chained(X_FRAME, rec, {})
    entry_w = ts.transform_point(pl, [-2.0, 0.0, 0.0])
    assert np.allclose(entry_w, [10.0, 0.0, 0.0], atol=1e-12)  # pivot fixed
    axis_w = ts.transform_vector(pl, [1.0, 0.0, 0.0])
    assert np.isclose(ts.vdot(axis_w, [1, 0, 0]), math.cos(math.radians(30)),
                      atol=1e-12)


def test_place_chained_pivot_center_vs_entrance_differ():
    # tilt about a TRANSVERSE axis (rz would spin about the beam axis,
    # which moves nothing for an on-axis element)
    rec_e = dict(LENS, tilt_ry="20", pivot="entrance")
    rec_c = dict(LENS, tilt_ry="20", pivot="center")
    pe = ts.place_chained(X_FRAME, rec_e, {})
    pc = ts.place_chained(X_FRAME, rec_c, {})
    assert not np.allclose(pe["pos_mm"], pc["pos_mm"])
    # entrance pivot: entry vertex stays put
    assert np.allclose(ts.transform_point(pe, [-2.0, 0.0, 0.0]),
                       [10.0, 0.0, 0.0], atol=1e-12)
    # center pivot: element center stays on axis at entry+half-thickness
    c_w = ts.transform_point(pc, [0.0, 0.0, 0.0])
    assert np.allclose(c_w, [12.0, 0.0, 0.0], atol=1e-12)


def test_place_chained_rot_first_decenters_along_tilted_axes():
    # 90-deg tilt about u (= world +y for an +x beam): the tilted v axis
    # v' = R(+z) = +x, so a decenter_y of 2 shifts the ENTRY VERTEX 2 mm
    # along +x (rot_first) instead of 2 mm along +z (pos_first)
    rec_r = dict(LENS, tilt_rx="90", decenter_y="2",
                 pos_rot_order="rot_first")
    rec_p = dict(LENS, tilt_rx="90", decenter_y="2",
                 pos_rot_order="pos_first")
    entry_r = ts.transform_point(ts.place_chained(X_FRAME, rec_r, {}),
                                 [-2.0, 0.0, 0.0])
    entry_p = ts.transform_point(ts.place_chained(X_FRAME, rec_p, {}),
                                 [-2.0, 0.0, 0.0])
    assert np.allclose(entry_r, [12.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(entry_p, [10.0, 0.0, 2.0], atol=1e-9)


@pytest.mark.parametrize("trial", range(12))
def test_derive_edge_inverts_place_chained(trial):
    rec = dict(LENS,
               distance=str(float(RNG.uniform(1, 60))),
               decenter_x=str(float(RNG.uniform(-4, 4))),
               decenter_y=str(float(RNG.uniform(-4, 4))),
               tilt_rx=str(float(RNG.uniform(-40, 40))),
               tilt_ry=str(float(RNG.uniform(-40, 40))),
               tilt_rz=str(float(RNG.uniform(-40, 40))))
    frame = ts.make_frame(RNG.uniform(-20, 20, 3).tolist(),
                          RNG.normal(size=3).tolist())
    pl = ts.place_chained(frame, rec, {})
    edge = ts.derive_edge(frame, pl, rec)
    pl2 = ts.place_chained(frame, dict(rec, **{
        k: str(v) for k, v in edge.items()}), {})
    assert np.allclose(pl["pos_mm"], pl2["pos_mm"], atol=1e-9)
    q1, q2 = np.array(pl["quat"]), np.array(pl2["quat"])
    assert min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)) < 1e-9


# ---------------------------------------------------------------------------
# exit_frames
# ---------------------------------------------------------------------------
def test_exit_frames_passthrough_origin_on_exit_plane():
    pl = ts.place_chained(X_FRAME, LENS, {})
    frames = ts.exit_frames(LENS, pl, X_FRAME)
    assert np.allclose(frames["out"]["origin"], [14.0, 0.0, 0.0],
                       atol=1e-12)          # exit vertex plane
    assert np.allclose(frames["out"]["dir"], [1, 0, 0], atol=1e-12)


def test_exit_frames_decentered_element_keeps_beam_axis():
    rec = dict(LENS, decenter_x="2")
    pl = ts.place_chained(X_FRAME, rec, {})
    frames = ts.exit_frames(rec, pl, X_FRAME)
    # the train continues along the ORIGINAL axis, not the shifted lens
    assert np.allclose(frames["out"]["origin"][1:], [0.0, 0.0], atol=1e-12)


MIRROR = {
    "label": "FM", "mode": "chained", "ref": "SRC", "port": "out",
    "distance": "20", "fold": True, "folded": True,
    "tilt_rx": "0",
    "local": {
        "entry": [0.0, 0.0, 0.0], "exit": [0.0, 0.0, 0.0],
        # local axis faces INTO the incoming beam; mirror surface at the
        # origin, normal along the local axis
        "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
        "reflect_plane": {"point": [0.0, 0.0, 0.0],
                          "normal": [1.0, 0.0, 0.0]}},
}


def _mirror_at_45(**over):
    """A fold mirror 20 mm down-beam, tilted 45 deg about the vertical
    (v = up = +z for an +x beam -> tilt_ry). With tilt_ry=+45 the mirror
    normal becomes (cos45, sin45, 0) and the beam folds into -y;
    tilt_ry=-45 folds it into +y."""
    rec = dict(MIRROR, tilt_ry="45")
    rec.update(over)
    return rec


def test_exit_frames_reflect_45():
    rec = _mirror_at_45()
    pl = ts.place_chained(X_FRAME, rec, {})
    frames = ts.exit_frames(rec, pl, X_FRAME)
    d = frames["reflect"]["dir"]
    assert np.allclose(frames["reflect"]["origin"], [20.0, 0.0, 0.0],
                       atol=1e-9)
    # deviated by 90 degrees into the horizontal plane
    assert abs(ts.vdot(d, [1, 0, 0])) < 1e-9
    assert abs(d[2]) < 1e-9
    u, v, dd = ts.frame_basis(frames["reflect"])
    assert np.allclose(np.cross(u, v), dd, atol=1e-12)   # right-handed


def test_exit_frames_unfolded_passthrough_same_origin():
    rec = _mirror_at_45(folded=False)
    pl = ts.place_chained(X_FRAME, rec, {})
    frames = ts.exit_frames(rec, pl, X_FRAME)
    assert np.allclose(frames["reflect"]["origin"], [20.0, 0.0, 0.0],
                       atol=1e-9)
    assert np.allclose(frames["reflect"]["dir"], [1, 0, 0], atol=1e-12)
    assert np.allclose(frames["reflect"]["up"], X_FRAME["up"], atol=1e-12)


def test_exit_frames_deviate_port():
    rec = {"label": "P", "mode": "chained", "ref": "S", "fold": True,
           "folded": True, "fold_deviation": "dmin", "fold_azimuth": "0",
           "local": {"entry": [0, 0, 0], "exit": [0, 0, 0],
                     "axis": [1, 0, 0], "up": [0, 0, 1]}}
    pl = {"pos_mm": [30.0, 0.0, 0.0], "quat": [0, 0, 0, 1]}
    frames = ts.exit_frames(rec, pl, X_FRAME, {"dmin": 40.0})
    d = frames["deviate"]["dir"]
    # rotated 40 deg about u=+y: stays in the x-z ... u is +y, so the
    # beam tips out of the x-y plane? u = up x dir = z x x = y. Rotating
    # +x about +y by +40 deg: x cos40 - z sin40.
    assert np.isclose(ts.vdot(d, [1, 0, 0]), math.cos(math.radians(40)),
                      atol=1e-12)
    assert abs(d[1]) < 1e-12


# ---------------------------------------------------------------------------
# sort_chain / downstream_of / solve_chain
# ---------------------------------------------------------------------------
def _source(label="SRC"):
    return {"label": label, "mode": "anchored",
            "local": {"entry": [5.0, 0.0, 0.0], "exit": [5.0, 0.0, 0.0],
                      "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}}


def _plate(label, ref, dist, port="out", **over):
    rec = {"label": label, "mode": "chained", "ref": ref, "port": port,
           "distance": str(dist),
           "local": {"entry": [-1.0, 0.0, 0.0], "exit": [1.0, 0.0, 0.0],
                     "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}}
    rec.update(over)
    return rec


def test_sort_chain_orders_and_detects_cycles():
    recs = {"SRC": _source(), "A": _plate("A", "SRC", 10),
            "B": _plate("B", "A", 10)}
    order = ts.sort_chain(recs)
    assert order.index("SRC") < order.index("A") < order.index("B")
    recs["A"]["ref"] = "B"
    with pytest.raises(ts.CycleError) as ei:
        ts.sort_chain(recs)
    assert "->" in str(ei.value)


def test_sort_chain_dangling_reference():
    with pytest.raises(ts.TrainError, match="unknown element"):
        ts.sort_chain({"A": _plate("A", "GHOST", 5)})


def test_downstream_of():
    recs = {"SRC": _source(), "A": _plate("A", "SRC", 10),
            "B": _plate("B", "A", 10), "C": _plate("C", "A", 20),
            "X": dict(_source("X"))}
    assert ts.downstream_of(recs, "SRC") == ["A", "B", "C"]
    assert set(ts.downstream_of(recs, "A")) == {"B", "C"}
    assert ts.downstream_of(recs, "B") == []


def test_solve_chain_linear():
    recs = {"SRC": _source(), "A": _plate("A", "SRC", 10),
            "B": _plate("B", "A", 10)}
    anchors = {"SRC": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]}}
    out = ts.solve_chain(recs, anchors, {})
    # SRC exit vertex at x=5; A entry (local -1) at 15 -> pos 16;
    # A exit at 17; B entry at 27 -> pos 28
    assert np.allclose(out["placements"]["A"]["pos_mm"], [16, 0, 0],
                       atol=1e-12)
    assert np.allclose(out["placements"]["B"]["pos_mm"], [28, 0, 0],
                       atol=1e-12)


def test_solve_chain_variable_ripple():
    recs = {"SRC": _source(), "A": _plate("A", "SRC", "gap"),
            "B": _plate("B", "A", "gap*2")}
    anchors = {"SRC": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]}}
    p1 = ts.solve_chain(recs, anchors, {"gap": 10.0})["placements"]
    p2 = ts.solve_chain(recs, anchors, {"gap": 15.0})["placements"]
    # gap=10: A entry at 5+10=15 (pos 16), exit vertex 17; B a further
    # 2*gap=20 -> entry 37 -> pos 38. gap=15: 5+15+2 + 30 -> pos 53.
    assert np.allclose(p1["A"]["pos_mm"], [16, 0, 0], atol=1e-12)
    assert np.allclose(p1["B"]["pos_mm"], [38, 0, 0], atol=1e-12)
    assert np.allclose(p2["B"]["pos_mm"], [53, 0, 0], atol=1e-12)


def test_solve_chain_branching_beamsplitter():
    bs = _plate("BS", "SRC", 20)
    bs["local"]["reflect_plane"] = {"point": [0.0, 0.0, 0.0],
                                    "normal": [-1.0, 1.0, 0.0]}
    recs = {"SRC": _source(), "BS": bs,
            "T": _plate("T", "BS", 10, port="transmit"),
            "R": _plate("R", "BS", 10, port="reflect")}
    anchors = {"SRC": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]}}
    out = ts.solve_chain(recs, anchors, {})
    # BS entry at 25 -> pos 26, plane point at 26, transmit exit plane 27
    assert np.allclose(out["placements"]["T"]["pos_mm"], [38, 0, 0],
                       atol=1e-9)
    # reflect arm: plane hit at x=26, beam turns into +y (normal -1,1,0)
    r = out["placements"]["R"]["pos_mm"]
    assert np.allclose(r, [26.0, 11.0, 0.0], atol=1e-9)
    # the reflected element's own axis follows the new beam direction
    axis_w = ts.transform_vector(out["placements"]["R"], [1, 0, 0])
    assert np.allclose(axis_w, [0, 1, 0], atol=1e-9)


def test_solve_chain_periscope_two_folds_and_unfold():
    m1 = _mirror_at_45(tilt_ry="-45")            # fold +x into +y
    m1.update(label="M1", ref="SRC", distance="15")
    # second mirror folds the +y beam back into +x: incoming dir +y,
    # transported up stays +z; its normal must be (-1,1,0)/sqrt2, which
    # a +45 tilt of the (+y-facing) local axis about v=+z produces
    m2 = _mirror_at_45(tilt_ry="45")
    m2.update(label="M2", ref="M1", port="reflect", distance="12")
    det = _plate("DET", "M2", 8, port="reflect")
    recs = {"SRC": _source(), "M1": m1, "M2": m2, "DET": det}
    anchors = {"SRC": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]}}
    out = ts.solve_chain(recs, anchors, {})
    # M1 plane hit at x=20; M2 12 mm up the +y arm (20, 12, 0);
    # beam exits M2 along +x again; DET entry 8 mm further: (28+1, 12, 0)
    d = out["placements"]["DET"]["pos_mm"]
    assert np.allclose(d, [29.0, 12.0, 0.0], atol=1e-9)

    # unfold M1: everything downstream re-collinearizes onto +x
    recs_u = {k: dict(v) for k, v in recs.items()}
    recs_u["M1"] = dict(m1, folded=False)
    out_u = ts.solve_chain(recs_u, anchors, {})
    d_u = out_u["placements"]["DET"]["pos_mm"]
    # straight line: M1 hit at 20, M2 at 32 (still folding: +x -> ?),
    # M2's own orientation re-solves against the straight beam and its
    # reflect plane still deviates by 90 deg
    assert np.allclose(out_u["placements"]["M2"]["pos_mm"][0], 32.0,
                       atol=1e-9)
    assert abs(d_u[1]) > 1.0   # M2 still folds somewhere off-axis

    # unfold BOTH: fully straight
    recs_uu = {k: dict(v) for k, v in recs_u.items()}
    recs_uu["M2"] = dict(recs_u["M2"], folded=False)
    out_uu = ts.solve_chain(recs_uu, anchors, {})
    d_uu = out_uu["placements"]["DET"]["pos_mm"]
    assert np.allclose(d_uu, [41.0, 0.0, 0.0], atol=1e-9)

    # refold: EXACT original placements (pure re-solve determinism)
    out_r = ts.solve_chain(recs, anchors, {})
    for label in ("M1", "M2", "DET"):
        assert np.allclose(out_r["placements"][label]["pos_mm"],
                           out["placements"][label]["pos_mm"], atol=0)
        assert np.allclose(out_r["placements"][label]["quat"],
                           out["placements"][label]["quat"], atol=0)


def test_solve_chain_missing_anchor():
    with pytest.raises(ts.TrainError, match="no known placement"):
        ts.solve_chain({"SRC": _source()}, {}, {})


def test_solve_chain_bad_port_names_available():
    recs = {"SRC": _source(),
            "A": _plate("A", "SRC", 10, port="reflect")}
    anchors = {"SRC": {"pos_mm": [0, 0, 0], "quat": [0, 0, 0, 1]}}
    with pytest.raises(ts.TrainError, match="which has ports"):
        ts.solve_chain(recs, anchors, {})
