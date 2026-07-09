"""TrainModel + Project chain-API tests against a scripted worker (no
FreeCAD, no Qt event loop). Element local ports are supplied through the
miewb_train_ports JSON property so these tests are independent of
primitivelib.port_frames formulas."""

import copy
import json
import tempfile

import numpy as np
import pytest

from mieworkbench.core.geomcache import GeomCache
from mieworkbench.core.project import Project, ProjectError
from mieworkbench.core.train import (
    TRAIN_GROUP, TrainModel, variables_from_sheets,
)
from mieworkbench.core.transforms import BodyState, Operation


class TrainFakeWorker:
    """FcClient stand-in: group-aware set_property, group-rigid
    set_placement (fcops semantics: every miewb_group member gets the
    SAME placement)."""

    def __init__(self, structure):
        self.structure = structure
        self.ops = []

    def request(self, op, params=None, timeout=None):
        params = params or {}
        self.ops.append((op, copy.deepcopy(params)))
        if op == "get_structure":
            return copy.deepcopy(self.structure)
        if op == "set_property":
            body = self._body(params["body"])
            value = params["value"]
            ptype = params.get("ptype") or (
                "bool" if isinstance(value, bool)
                else "float" if isinstance(value, (int, float))
                else "string")
            fc_type = {"string": "App::PropertyString",
                       "float": "App::PropertyFloat",
                       "bool": "App::PropertyBool"}[ptype]
            body["properties"][params["name"]] = {
                "type": fc_type, "group": params.get("group", "Base"),
                "value": value}
            return self._mut()
        if op == "remove_property":
            self._body(params["body"]).pop(params["name"], None)
            self._body(params["body"])["properties"].pop(
                params["name"], None)
            return self._mut()
        if op == "set_placement":
            key = str(params["body"])
            members = [b for b in self.structure["bodies"]
                       if b["properties"].get("miewb_group", {})
                       .get("value") == key]
            if not members:
                members = [self._body(key)]
            pl = {"pos_mm": list(params["pos_mm"]),
                  "quat": list(params["quat"])}
            out = {}
            for b in members:
                b["placement"] = copy.deepcopy(pl)
                out[b["name"]] = copy.deepcopy(pl)
            return {"placements": out}
        if op == "tessellate":
            return {"bodies": {b: {"faces": [], "shape_key": "k1",
                                   "placement": self._body(b)["placement"]}
                               for b in (params.get("bodies") or [])}}
        raise AssertionError("unexpected op %r" % op)

    def _body(self, key):
        for b in self.structure["bodies"]:
            if b["name"] == key or b["label"] == key:
                return b
        raise KeyError(key)

    def _mut(self):
        return {"changed_bodies": [], "moved_bodies": [], "invalid": [],
                "placements": {}}


def _ports(entry, exit_, axis=(1, 0, 0), up=(0, 0, 1), reflect=None):
    d = {"entry": list(entry), "exit": list(exit_), "axis": list(axis),
         "up": list(up), "reflect_plane": reflect}
    return json.dumps(d)


def _body_dict(name, ports_json, pos=(0, 0, 0), extra_props=None):
    props = {
        "miewb_group": {"type": "App::PropertyString", "group": "Base",
                        "value": name},
        "miewb_train_ports": {"type": "App::PropertyString",
                              "group": TRAIN_GROUP, "value": ports_json},
    }
    props.update(extra_props or {})
    return {
        "name": name, "label": name, "tip": "%s_pad" % name,
        "face_count": 1, "solid_closed": True, "volume_mm3": 1.0,
        "center_of_mass_mm": list(pos), "bbox_mm": [0, 0, 0, 1, 1, 1],
        "placement": {"pos_mm": list(pos), "quat": [0.0, 0.0, 0.0, 1.0]},
        "placement_bound": False, "shape_key": "k_%s" % name,
        "properties": props,
    }


def make_scene():
    """SRC (anchored source) -> L1 -> L2 (lenses) + FM (fold mirror) +
    DET, all unchained initially."""
    structure = {
        "doc": "scene", "label": "scene", "file": "/nowhere/scene.FCStd",
        "bodies": [
            _body_dict("SRC", _ports([5, 0, 0], [5, 0, 0])),
            _body_dict("L1", _ports([-2, 0, 0], [2, 0, 0]), pos=(30, 0, 0)),
            _body_dict("L2", _ports([-1, 0, 0], [1, 0, 0]), pos=(60, 0, 0)),
            _body_dict("FM", _ports([0, 0, 0], [0, 0, 0], reflect={
                "point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}),
                pos=(90, 0, 0)),
            _body_dict("DET", _ports([0, 0, 0], [0, 0, 0]),
                       pos=(120, 0, 0)),
        ],
        "sheets": [{"name": "Spreadsheet", "label": "miewb_vars",
                    "aliases": {
                        "gap": {"cell": "B1", "raw": "=25", "value": 25.0,
                                "unit": ""},
                        "gap__min": {"cell": "C1", "raw": "=10",
                                     "value": 10.0, "unit": ""},
                        "gap__max": {"cell": "D1", "raw": "=40",
                                     "value": 40.0, "unit": ""},
                        "gap__n": {"cell": "E1", "raw": "=3", "value": 3.0,
                                   "unit": ""},
                        "gap__on": {"cell": "F1", "raw": "=1", "value": 1.0,
                                    "unit": ""},
                    }}],
    }
    project = Project()
    fake = TrainFakeWorker(structure)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_train_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    for b in project.structure["bodies"]:
        project.body_states[b["name"]] = BodyState.from_worker(b, [])
    fake.ops.clear()
    return project, fake


def _pos(project, name):
    return project.body_states[name].current.to_dict()["pos_mm"]


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
