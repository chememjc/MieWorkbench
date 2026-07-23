"""TrainModel + Project chain-API tests against a scripted worker (no
FreeCAD, no Qt event loop). Element local ports are supplied through the
miewb_train_ports JSON property so these tests are independent of
primitivelib.port_frames formulas."""

import copy

import numpy as np
import pytest

from mieworkbench.core.project import ProjectError
from mieworkbench.core.train import TRAIN_GROUP, variables_from_sheets
from mieworkbench.core.transforms import Operation

from mieworkbench.tests.train_test_support import make_scene, pos_of as _pos_of


def _pos(project, name):
    return _pos_of(project, name)


# ---------------------------------------------------------------------------
# TrainModel snapshot behavior
# ---------------------------------------------------------------------------
def test_records_default_anchored():
    project, _ = make_scene()
    tm = project.train()
    assert tm.element_labels() == ["DET", "FM", "L1", "L2", "SRC"]
    assert all(tm.records()[el]["mode"] == "anchored"
               for el in tm.element_labels())
    assert not tm.has_train()


def test_local_ports_from_json_prop():
    project, _ = make_scene()
    tm = project.train()
    loc = tm.local_ports("L1")
    assert loc["entry"] == [-2, 0, 0]
    assert loc["exit"] == [2, 0, 0]


def test_variables_from_sheets_excludes_meta():
    project, _ = make_scene()
    assert project.train_variables() == {"gap": 25.0}
    assert variables_from_sheets([]) == {}


# ---------------------------------------------------------------------------
# Chaining + ripple
# ---------------------------------------------------------------------------
def test_set_chain_places_element():
    project, fake = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    # SRC exit vertex x=5; L1 entry (local -2) at 15 -> pos 17
    assert np.allclose(_pos(project, "L1"), [17, 0, 0])
    # props landed in the MieTrain group
    props = project.body("L1")["properties"]
    assert props["miewb_train_mode"]["group"] == TRAIN_GROUP
    assert props["miewb_train_distance"]["value"] == "10"


def test_set_chain_expression_uses_variables():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "gap*2"})
    assert np.allclose(_pos(project, "L1"), [57, 0, 0])   # 5+50 entry, +2


def test_chain_edit_ripples_downstream_one_undo():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    # L1 exit at 19; L2 entry at 39 -> pos 40
    assert np.allclose(_pos(project, "L2"), [40, 0, 0])
    before = {n: _pos(project, n) for n in ("L1", "L2")}

    project.set_chain("L1", {"ref": "SRC", "distance": "30"})
    assert np.allclose(_pos(project, "L1"), [37, 0, 0])
    assert np.allclose(_pos(project, "L2"), [60, 0, 0])   # followed rigidly

    project.undo()                     # ONE undo restores both
    assert np.allclose(_pos(project, "L1"), before["L1"])
    assert np.allclose(_pos(project, "L2"), before["L2"])


def test_set_chain_rejects_cycle_and_rolls_back():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    pos_before = {n: _pos(project, n) for n in ("L1", "L2")}
    with pytest.raises(Exception):
        project.set_chain("L1", {"ref": "L2", "distance": "5"})
    # macro aborted: chain props and poses unchanged
    assert project.body("L1")["properties"]["miewb_train_ref"]["value"] \
        == "SRC"
    for n in ("L1", "L2"):
        assert np.allclose(_pos(project, n), pos_before[n])


def test_set_anchored_freezes_pose():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    pos = _pos(project, "L1")
    project.set_anchored("L1")
    tm = project.train()
    assert not tm.is_chained("L1")
    assert np.allclose(_pos(project, "L1"), pos)


def test_move_element_syncs_chain_fields_and_ripples():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    project.move_element("L1", Operation("translate",
                                         {"vector_mm": [5.0, 0.0, 0.0]}))
    assert np.allclose(_pos(project, "L1"), [22, 0, 0])
    # edge field re-derived to a literal
    dist = project.body("L1")["properties"]["miewb_train_distance"]["value"]
    assert float(dist) == pytest.approx(15.0)
    # downstream followed
    assert np.allclose(_pos(project, "L2"), [45, 0, 0])
    # one undo restores everything
    project.undo()
    assert np.allclose(_pos(project, "L1"), [17, 0, 0])
    assert np.allclose(_pos(project, "L2"), [40, 0, 0])
    dist = project.body("L1")["properties"]["miewb_train_distance"]["value"]
    assert dist == "10"


def test_move_element_plain_for_untrained():
    project, _ = make_scene()
    project.move_element("DET", Operation("translate",
                                          {"vector_mm": [1.0, 0.0, 0.0]}))
    assert np.allclose(_pos(project, "DET"), [121, 0, 0])
    project.undo()
    assert np.allclose(_pos(project, "DET"), [120, 0, 0])


def test_move_anchored_parent_ripples_children():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.move_element("SRC", Operation("translate",
                                          {"vector_mm": [0.0, 3.0, 0.0]}))
    assert np.allclose(_pos(project, "SRC"), [0, 3, 0])
    assert np.allclose(_pos(project, "L1"), [17, 3, 0])
    project.undo()
    assert np.allclose(_pos(project, "L1"), [17, 0, 0])


# ---------------------------------------------------------------------------
# Folds through the chain API
# ---------------------------------------------------------------------------
def _fold_scene():
    project, fake = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                             "folded": True, "tilt_ry": "-45"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "15"})
    return project, fake


def test_fold_places_downstream_on_reflected_arm():
    project, _ = _fold_scene()
    # L1 exit x=19, FM mirror point at 39; -45deg tilt folds +x into +y
    assert np.allclose(_pos(project, "FM"), [39, 0, 0], atol=1e-9)
    assert np.allclose(_pos(project, "DET"), [39, 15, 0], atol=1e-9)


def test_unfold_refold_via_chain_edit_is_exact():
    project, _ = _fold_scene()
    folded = {n: copy.deepcopy(
        project.body_states[n].current.to_dict())
        for n in ("FM", "DET")}
    project.set_chain("FM", {"folded": False}, text="Unfold FM")
    # DET re-collinearizes onto +x, same along-beam distance
    assert np.allclose(_pos(project, "DET"), [54, 0, 0], atol=1e-9)
    # FM itself stays put (ghosted in place)
    assert np.allclose(_pos(project, "FM"), [39, 0, 0], atol=1e-9)
    project.set_chain("FM", {"folded": True}, text="Refold FM")
    for n in ("FM", "DET"):
        cur = project.body_states[n].current.to_dict()
        assert np.allclose(cur["pos_mm"], folded[n]["pos_mm"], atol=0)
        assert np.allclose(cur["quat"], folded[n]["quat"], atol=0)


def test_unfold_undo_restores_fold():
    project, _ = _fold_scene()
    det_folded = _pos(project, "DET")
    project.set_chain("FM", {"folded": False}, text="Unfold FM")
    assert not np.allclose(_pos(project, "DET"), det_folded)
    project.undo()
    assert np.allclose(_pos(project, "DET"), det_folded, atol=0)
    assert project.body("FM")["properties"]["miewb_train_folded"]["value"] \
        is True


# ---------------------------------------------------------------------------
# Validation / guards
# ---------------------------------------------------------------------------
def test_validate_reports_cycles():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    # forge a cycle behind the API's back (as a hand-edited file could)
    project.body("SRC")["properties"]["miewb_train_mode"] = {
        "type": "App::PropertyString", "group": TRAIN_GROUP,
        "value": "chained"}
    project.body("SRC")["properties"]["miewb_train_ref"] = {
        "type": "App::PropertyString", "group": TRAIN_GROUP, "value": "L1"}
    problems = project.train().validate()
    assert any("circular" in msg for sev, msg in problems)


def test_chain_refuses_expression_bound_placement():
    project, _ = make_scene()
    project.body("L1")["placement_bound"] = True
    with pytest.raises(ProjectError, match="expression"):
        project.set_chain("L1", {"ref": "SRC", "distance": "10"})


def test_validate_flags_bound_chained_element():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.body("L1")["placement_bound"] = True
    problems = project.train().validate()
    assert any("expression-bound" in msg for sev, msg in problems)


def test_derive_edge_roundtrip_through_project():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10",
                             "decenter_y": "1.5", "tilt_ry": "10"})
    edge = project.train().derive_edge("L1", project.train_variables())
    assert edge["distance"] == pytest.approx(10.0, abs=1e-9)
    assert edge["decenter_y"] == pytest.approx(1.5, abs=1e-9)
    assert edge["tilt_ry"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# candidate_edge / available_ports (the no-move conversion preview)
# ---------------------------------------------------------------------------
def test_available_ports():
    project, _ = make_scene()
    tm = project.train()
    assert tm.available_ports("L1") == ["out", "transmit"]
    assert "reflect" in tm.available_ports("FM")


def test_candidate_edge_matches_actual_chain():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    # L2 stays anchored at (60,0,0); its candidate edge vs L1 must equal
    # the distance that, when chained, reproduces its current pose:
    # L1 exit vertex at x=19, L2 entry (local -1) at 59 -> 40
    edge = project.train().candidate_edge("L2", "L1")
    assert edge["distance"] == pytest.approx(40.0, abs=1e-9)
    assert edge["decenter_x"] == pytest.approx(0.0, abs=1e-9)
    # converting with those floats must not move the element
    before = _pos(project, "L2")
    payload = {"ref": "L1"}
    payload.update({k: float(v) for k, v in edge.items()})
    project.set_chain("L2", payload)
    assert np.allclose(_pos(project, "L2"), before, atol=1e-9)


def test_candidate_edge_refuses_cycles_and_unknown_ports():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    tm = project.train()
    import train_solver
    with pytest.raises(train_solver.TrainError, match="cycle"):
        tm.candidate_edge("L1", "L2")
    with pytest.raises(train_solver.TrainError, match="cycle"):
        tm.candidate_edge("L1", "L1")
    with pytest.raises(train_solver.TrainError, match="no port"):
        tm.candidate_edge("DET", "L1", "reflect")


# ---------------------------------------------------------------------------
# Anchored pose expressions (Project API + ripple/undo/drag)
# ---------------------------------------------------------------------------
def _set_var(project, name, value_raw, row):
    """Write one miewb_vars value cell through the undoable Project path."""
    from mieworkbench.core.variables import cell_plan
    project.apply_variable_cells(
        cell_plan(name, row=row, value=value_raw), text="Set %s" % name)


def test_pose_expression_bakes_and_sweeps():
    project, _ = make_scene()
    # add R, theta variables (gap occupies row 1)
    _set_var(project, "R", "40", row=2)
    _set_var(project, "theta", "0", row=3)
    # DET starts anchored at (120,0,0); drive it onto a goniometer circle
    project.set_pose_expression("DET", "pos_x", "R*cos(theta)")
    project.set_pose_expression("DET", "pos_y", "R*sin(theta)")
    project.set_pose_expression("DET", "rot_rz", "theta")
    # theta=0 -> (40,0, z-kept)
    assert np.allclose(_pos(project, "DET")[:2], [40.0, 0.0], atol=1e-9)

    _set_var(project, "theta", "90", row=3)
    assert np.allclose(_pos(project, "DET")[:2], [0.0, 40.0], atol=1e-9)
    _set_var(project, "theta", "180", row=3)
    assert np.allclose(_pos(project, "DET")[:2], [-40.0, 0.0], atol=1e-9)

    # the expressions live as miewb_expr_* props on the primary body
    props = project.body("DET")["properties"]
    assert props["miewb_expr_pos_x"]["value"] == "R*cos(theta)"
    assert props["miewb_expr_pos_x"]["group"] == TRAIN_GROUP
    assert project.pose_expressions("DET") == {
        "pos_x": "R*cos(theta)", "pos_y": "R*sin(theta)", "rot_rz": "theta"}


def test_pose_expression_variable_edit_undo_restores():
    project, _ = make_scene()
    _set_var(project, "R", "40", row=2)
    _set_var(project, "theta", "0", row=3)
    project.set_pose_expression("DET", "pos_x", "R*cos(theta)")
    project.set_pose_expression("DET", "pos_y", "R*sin(theta)")
    at0 = _pos(project, "DET")
    _set_var(project, "theta", "90", row=3)
    assert np.allclose(_pos(project, "DET")[:2], [0.0, 40.0], atol=1e-9)
    project.undo()      # ONE undo (the variable-edit macro) restores pose
    assert np.allclose(_pos(project, "DET"), at0, atol=1e-9)


def test_set_pose_expression_undo_clears_prop_and_pose():
    project, _ = make_scene()
    _set_var(project, "R", "40", row=2)
    before = _pos(project, "DET")
    project.set_pose_expression("DET", "pos_x", "R")
    assert np.allclose(_pos(project, "DET")[0], 40.0, atol=1e-9)
    project.undo()
    assert "miewb_expr_pos_x" not in project.body("DET")["properties"]
    assert np.allclose(_pos(project, "DET"), before, atol=1e-9)


def test_clear_pose_expression_returns_to_literal():
    project, _ = make_scene()
    _set_var(project, "R", "40", row=2)
    project.set_pose_expression("DET", "pos_x", "R")
    project.clear_pose_expression("DET", "pos_x")
    assert project.pose_expressions("DET") == {}
    # editing the variable no longer moves it (pose is now literal)
    moved = _pos(project, "DET")
    _set_var(project, "R", "99", row=2)
    assert np.allclose(_pos(project, "DET"), moved, atol=1e-9)


def test_pose_expression_refused_on_chained_element():
    project, _ = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    with pytest.raises(ProjectError, match="chained"):
        project.set_pose_expression("L1", "pos_x", "5")


def test_pose_expression_bad_field_and_expression_named():
    project, _ = make_scene()
    with pytest.raises(ProjectError, match="unknown pose field"):
        project.set_pose_expression("DET", "pos_q", "5")
    with pytest.raises(ProjectError, match="pos_x"):
        project.set_pose_expression("DET", "pos_x", "nope*2")


def test_drag_clears_pose_expression_to_literal():
    project, _ = make_scene()
    _set_var(project, "R", "40", row=2)
    _set_var(project, "theta", "0", row=3)
    project.set_pose_expression("DET", "pos_x", "R*cos(theta)")
    project.set_pose_expression("DET", "pos_y", "R*sin(theta)")
    # a spatial drag makes the pose literal (expr -> literal, like a chained
    # drag re-derives its edge fields); the props are removed
    project.move_element("DET", Operation("translate", {"vector_mm": [0, 0, 5]}))
    assert project.pose_expressions("DET") == {}
    dragged = _pos(project, "DET")
    assert np.allclose(dragged, [40.0, 0.0, 5.0], atol=1e-9)
    # a later variable edit does NOT snap it back
    _set_var(project, "theta", "90", row=3)
    assert np.allclose(_pos(project, "DET"), dragged, atol=1e-9)


def test_pose_expression_validate_flags_chained():
    project, _ = make_scene()
    project.set_pose_expression("L1", "pos_x", "5")   # L1 anchored: fine
    # now chain it directly on the body props (bypassing set_chain's guard)
    # to exercise the validate() message
    project.set_property("L1", "miewb_train_mode", "chained",
                         ptype="string", group=TRAIN_GROUP)
    project.set_property("L1", "miewb_train_ref", "SRC",
                         ptype="string", group=TRAIN_GROUP)
    problems = project.train().validate()
    assert any("anchored-pose" in m for _s, m in problems)
