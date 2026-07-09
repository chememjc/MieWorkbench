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
