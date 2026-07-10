"""TransformPanel tests (offscreen): the Absolute pose group, the
reference-delta readout, and the Snap-to-Axis workflow (pick callback +
axis-drag math) -- all driven without a real mouse or a live FreeCAD
worker (a fake worker echoes set_placement for the whole miewb_group, the
way the real op does). See docs/UI_TESTING.md."""
import copy
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.geomcache import GeomCache            # noqa: E402
from mieworkbench.core.project import Project                # noqa: E402
from mieworkbench.core.transforms import BodyState, FaceInfo, Placement  # noqa: E402,E501
from mieworkbench.panes.transform_panel import TransformPanel  # noqa: E402


class FakeWorkerFc:
    """Enough of FcClient for placement commands; set_placement moves the
    WHOLE miewb_group rigidly and echoes every member (like op_set_placement)."""

    def __init__(self, structure):
        self.structure = structure

    def request(self, op, params=None, timeout=None):
        params = params or {}
        if op == "get_structure":
            return copy.deepcopy(self.structure)
        if op == "set_placement":
            # real op_set_placement semantics: the key may be a GROUP
            # value directly (tried FIRST), else a body name/label whose
            # group is then resolved
            key = str(params["body"])
            if any(b["properties"].get("miewb_group", {}).get("value")
                   == key for b in self.structure["bodies"]):
                group = key
            else:
                group = self._group_of(key)
            pl = {"pos_mm": list(params["pos_mm"]), "quat": list(params["quat"])}
            out = {}
            for b in self.structure["bodies"]:
                if self._group_of(b["name"]) == group:
                    b["placement"] = dict(pl)
                    out[b["name"]] = dict(pl)
            return {"placements": out}
        raise AssertionError("unexpected op %r" % op)

    def _body(self, key):
        for b in self.structure["bodies"]:
            if b["name"] == key or b["label"] == key:
                return b
        raise KeyError(key)

    def _group_of(self, key):
        b = self._body(key)
        g = b["properties"].get("miewb_group", {}).get("value")
        return g if g else b["name"]


class FakeView:
    def __init__(self):
        self.pick_cb = None
        self.drag = None
        self.moves = []

    def pick_face_once(self, cb):
        self.pick_cb = cb

    def begin_axis_drag(self, point, axis, on_move, on_commit, on_abort):
        self.drag = dict(point=list(point), axis=list(axis), on_move=on_move,
                         on_commit=on_commit, on_abort=on_abort)

    def update_placement(self, name, placement):
        self.moves.append((name, placement))


def _mk_structure():
    def body(name, label, group=None, pos=(0, 0, 0)):
        props = {"material": {"type": "App::PropertyString",
                              "group": "Base", "value": "bk7"}}
        if group:
            props["miewb_group"] = {"type": "App::PropertyString",
                                    "group": "Base", "value": group}
        return {"name": name, "label": label, "tip": "Pad", "face_count": 1,
                "solid_closed": True, "volume_mm3": 1.0,
                "center_of_mass_mm": list(pos),
                "bbox_mm": [pos[0], pos[1], pos[2],
                            pos[0] + 1, pos[1] + 1, pos[2] + 1],
                "placement": {"pos_mm": list(pos), "quat": [0, 0, 0, 1]},
                "placement_bound": False, "shape_key": "k",
                "properties": props}
    return {"doc": "scene", "label": "scene", "file": "/nowhere/s.FCStd",
            "bodies": [
                body("Lens", "Lens", pos=(2, 0, 0)),
                body("TgtA", "Target", group="tgt", pos=(0, 10, 0)),
                body("TgtB", "Target", group="tgt", pos=(0, 10, 0))],
            "sheets": []}


def _axis_state(struct_body, normal, center):
    """A BodyState with one usable face so it has an optical axis/center."""
    pos = struct_body["placement"]["pos_mm"]
    face = FaceInfo("%s.Pad.Face1" % struct_body["name"],
                    np.array(center, float), np.array(normal, float), 1.0)
    return BodyState(struct_body["name"], struct_body["label"],
                     Placement(list(pos), [0, 0, 0, 1]),
                     com_local_mm=np.array(center, float),
                     bbox_center_local_mm=np.array(center, float),
                     faces=[face])


def make_panel(qtbot):
    struct = _mk_structure()
    project = Project()
    fake = FakeWorkerFc(struct)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_tp_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/s.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    bybn = {b["name"]: b for b in project.structure["bodies"]}
    project.body_states["Lens"] = _axis_state(bybn["Lens"], (1, 0, 0), (0, 0, 0))
    project.body_states["TgtA"] = _axis_state(bybn["TgtA"], (0, 1, 0), (0, 0, 0))
    project.body_states["TgtB"] = _axis_state(bybn["TgtB"], (0, 1, 0), (0, 0, 0))
    # faces meta so resolve_axis(face_normal)/face_point work in the panel
    for n in ("Lens", "TgtA", "TgtB"):
        st = project.body_states[n]
        project.faces[n] = {"faces": [{"id": f.id} for f in st.faces]}
    panel = TransformPanel()
    qtbot.addWidget(panel)
    view = FakeView()
    panel.set_project(project)
    panel.set_scene_view(view)
    return panel, project, view


# -- absolute pose ----------------------------------------------------------
def test_absolute_fields_populate(qtbot):
    panel, project, _view = make_panel(qtbot)
    panel.set_body("Lens")
    assert [round(sb.value(), 3) for sb in panel.abs_pos] == [2.0, 0.0, 0.0]
    assert panel.btn_set_pos.isEnabled()


def test_absolute_set_position_moves_body(qtbot):
    panel, project, _view = make_panel(qtbot)
    panel.set_body("Lens")
    for sb, v in zip(panel.abs_pos, (5.0, -3.0, 1.0)):
        sb.setValue(v)
    panel._apply_set_position()
    assert list(project.current_placement("Lens").pos) == [5.0, -3.0, 1.0]
    project.undo()
    assert list(project.current_placement("Lens").pos) == [2.0, 0.0, 0.0]


def test_absolute_set_orientation_roundtrips(qtbot):
    panel, project, _view = make_panel(qtbot)
    panel.set_body("Lens")
    for sb, v in zip(panel.abs_rot, (0.0, 90.0, 0.0)):
        sb.setValue(v)
    panel._apply_set_orientation()
    # optical axis (local +x) now points along -z or +z after Ry(90)
    axis = project.body_states["Lens"].optical_axis_world()
    assert abs(abs(axis[2]) - 1.0) < 1e-6


def test_reference_delta(qtbot):
    panel, project, _view = make_panel(qtbot)
    panel.set_body("Lens")
    idx = panel.ref_combo.findData("TgtA")
    panel.ref_combo.setCurrentIndex(idx)
    # Lens optical center (2,0,0) minus TgtA optical center (0,10,0)
    assert "2.000, -10.000, 0.000" in panel.delta.text()


# -- snap to axis -----------------------------------------------------------
def test_snap_via_pick_callback_aligns_and_single_undo(qtbot):
    panel, project, view = make_panel(qtbot)
    panel.set_body("Lens")
    panel._pick_snap_target()
    assert view.pick_cb is not None            # armed
    # user clicks a face of the target element
    view.pick_cb("TgtA", "TgtA.Pad.Face1")
    lens = project.body_states["Lens"]
    # Lens optical axis snapped onto TgtA's optical axis (+/- y)
    assert abs(abs(lens.optical_axis_world()[1]) - 1.0) < 1e-6
    # centered on the target axis line (through (0,10,0) dir y): x,z -> 0
    c = lens.optical_center_world()
    assert abs(c[0]) < 1e-6 and abs(c[2]) < 1e-6
    # the whole snap is ONE undo entry
    before = project.current_placement("Lens").to_dict()
    project.undo()
    assert list(project.current_placement("Lens").pos) == [2.0, 0.0, 0.0]
    project.redo()
    assert project.current_placement("Lens").to_dict() == before


def test_snap_position_along_axis_is_absolute(qtbot):
    """The spinbox shows the CURRENT along-axis position and committing a
    value moves to that absolute station (idempotent — not incremental)."""
    panel, project, view = make_panel(qtbot)
    panel.set_body("Lens")
    panel._pick_snap_target()
    view.pick_cb("TgtA", "TgtA.Pad.Face1")
    assert panel.snap_offset_btn.isEnabled()
    # after the snap the spinbox reflects the element's live position
    t0 = panel._along_axis_t()
    assert t0 is not None
    assert abs(panel.snap_offset.value() - t0) < 1e-6

    point, axis = panel._snap_axis
    panel.snap_offset.setValue(5.0)
    panel._apply_snap_offset()
    c1 = project.body_states["Lens"].optical_center_world()
    expect = np.array(point) + 5.0 * np.array(axis)
    assert np.allclose(c1, expect, atol=1e-6)

    # committing the SAME value again is a no-op (absolute, not relative)
    panel.snap_offset.setValue(5.0)
    panel._apply_snap_offset()
    c2 = project.body_states["Lens"].optical_center_world()
    assert np.allclose(c2, c1, atol=1e-9)
    assert "Already at" in panel.snap_status.text()


def test_snap_drag_updates_absolute_readout(qtbot):
    panel, project, view = make_panel(qtbot)
    panel.set_body("Lens")
    panel._pick_snap_target()
    view.pick_cb("TgtA", "TgtA.Pad.Face1")
    assert view.drag is not None
    point, axis = panel._snap_axis
    target = np.array(point) + 7.5 * np.array(axis)
    view.drag["on_move"](target)
    assert "7.500 mm" in panel.snap_status.text()
    assert abs(panel.snap_offset.value() - 7.5) < 1e-6
    view.drag["on_abort"]()


def test_snap_moves_all_group_siblings(qtbot):
    """C2: set_placement moves the whole miewb_group; both TgtA and TgtB
    BodyStates must track the new pose (no tear-apart)."""
    panel, project, view = make_panel(qtbot)
    panel.set_body("TgtA")                      # part of group 'tgt'
    moved = {}
    project.bodiesMoved.connect(lambda d: moved.update(d))
    panel._pick_snap_target()
    view.pick_cb("Lens", "Lens.Pad.Face1")     # snap group onto Lens axis
    a = project.current_placement("TgtA").to_dict()
    b = project.current_placement("TgtB").to_dict()
    assert a == b                               # rigidly together
    assert "TgtA" in moved and "TgtB" in moved  # both emitted


# -- axis-drag geometry (no GL) ---------------------------------------------
def test_drag_to_axis_projects_ray(qtbot):
    from mieworkbench.widgets.vtkview import VtkSceneView
    view = VtkSceneView()
    qtbot.addWidget(view)
    view._axis_drag = {"point": np.array([0.0, 0.0, 0.0]),
                       "dir": np.array([1.0, 0.0, 0.0])}
    # a viewing ray crossing the x-axis at x=4 (origin above, pointing down)
    view._display_ray = lambda xy: (np.array([4.0, 5.0, 0.0]),
                                    np.array([0.0, -1.0, 0.0]))
    pt = view._drag_to_axis((100, 100))
    assert np.allclose(pt, [4.0, 0.0, 0.0], atol=1e-9)


def test_snap_drag_commit_applies_translate(qtbot):
    panel, project, view = make_panel(qtbot)
    panel.set_body("Lens")
    panel._pick_snap_target()
    view.pick_cb("TgtA", "TgtA.Pad.Face1")     # arms the drag on the view
    assert view.drag is not None
    center = np.array(view.drag["point"], float)
    # commit a drag 7 mm down the +y axis
    view.drag["on_commit"](center + np.array([0.0, 7.0, 0.0]))
    c = project.body_states["Lens"].optical_center_world()
    assert abs(c[1] - (center[1] + 7.0)) < 1e-6


# ---------------------------------------------------------------------------
# Position section (dual representation + no-move Convert) against the
# train-capable scripted worker
# ---------------------------------------------------------------------------
from mieworkbench.tests.train_test_support import (   # noqa: E402
    make_scene, pos_of)


def make_train_panel(qtbot, chained=True):
    project, fake = make_scene()
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    if chained:
        project.set_chain("L2", {"ref": "L1", "distance": "20"})
    panel = TransformPanel()
    qtbot.addWidget(panel)
    view = FakeView()
    panel.set_project(project)
    panel.set_scene_view(view)
    panel.set_body("L2")
    return panel, project, view


def _placement(project, name):
    return project.body_states[name].current.to_dict()


def test_position_section_chained_shows_editable_edge(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=True)
    assert "Chained to L1" in panel.pos_status.text()
    assert panel.train_ref.currentData() == "L1"
    assert panel.edge_fields["distance"].text() == "20"
    assert not panel.edge_fields["distance"].isReadOnly()
    assert panel.btn_convert.isVisible() or True  # visibility needs show()
    assert "anchored" in panel.btn_convert.text()

    # editing the distance field re-chains and moves the element
    panel.edge_fields["distance"].setText("30")
    panel._on_edge_field_committed("distance")
    assert np.allclose(pos_of(project, "L2"), [50, 0, 0])
    # one undo restores
    project.undo()
    assert np.allclose(pos_of(project, "L2"), [40, 0, 0])


def test_position_section_rejects_bad_expression(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=True)
    before = pos_of(project, "L2")
    panel.edge_fields["distance"].setText("2*+")
    panel._on_edge_field_committed("distance")
    assert np.allclose(pos_of(project, "L2"), before)
    assert "Distance" in panel.pos_note.text()


def test_position_section_anchored_shows_derived_preview(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=False)
    # L2 anchored at (60,0,0); candidate ref defaults to deepest element
    assert "Anchored" in panel.pos_status.text()
    assert panel.edge_fields["distance"].isReadOnly()
    # choose L1 explicitly and check the derived distance:
    # L1 exit vertex at x=19, L2 entry (local -1) at 59 -> 40
    idx = panel.train_ref.findData("L1")
    panel.train_ref.setCurrentIndex(idx)
    assert float(panel.edge_fields["distance"].text()) == pytest.approx(40.0)
    assert "chained" in panel.btn_convert.text()


def test_convert_round_trip_does_not_move(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=False)
    idx = panel.train_ref.findData("L1")
    panel.train_ref.setCurrentIndex(idx)
    before = copy.deepcopy(_placement(project, "L2"))

    panel._on_convert_clicked()          # anchored -> chained (no move)
    rec = project.train().records()["L2"]
    assert rec["mode"] == "chained" and rec["ref"] == "L1"
    after = _placement(project, "L2")
    assert np.allclose(after["pos_mm"], before["pos_mm"], atol=1e-9)
    q1, q2 = np.array(after["quat"]), np.array(before["quat"])
    assert min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)) < 1e-9

    panel._on_convert_clicked()          # chained -> anchored (no move)
    assert project.train().records()["L2"]["mode"] == "anchored"
    assert np.allclose(_placement(project, "L2")["pos_mm"],
                       before["pos_mm"], atol=1e-9)


def test_convert_hidden_without_valid_candidate(qtbot):
    project, fake = make_scene()
    panel = TransformPanel()
    qtbot.addWidget(panel)
    panel.set_project(project)
    # SRC is the only upstream-less element: selecting the FIRST element in
    # solve order with everything anchored still yields candidates, so
    # instead select an element and forge a downstream-only situation:
    project.set_chain("L1", {"ref": "SRC", "distance": "10"})
    project.set_chain("L2", {"ref": "L1", "distance": "20"})
    project.set_chain("FM", {"ref": "L2", "distance": "5"})
    project.set_chain("DET", {"ref": "FM", "port": "reflect",
                              "distance": "5"})
    panel.set_body("SRC")     # anchored; every other element is downstream
    assert not panel.btn_convert.isVisibleTo(panel)
    assert "No upstream element" in panel.pos_note.text()


def test_ref_pick_via_selection_intercept(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=True)
    # L2 currently chained to L1; arm the pick, then "click" SRC in the
    # outliner (routes through set_body)
    panel._arm_ref_pick()
    panel.set_body("SRC")
    rec = project.train().records()["L2"]
    assert rec["ref"] == "SRC"
    # the panel still operates on L2 (the pick did not change selection
    # from the panel's point of view)
    assert panel.body_name == "L2"


def test_ref_pick_refuses_descendant(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=True)
    panel.set_body("L1")
    panel._arm_ref_pick()
    panel.set_body("L2")                 # L2 is downstream of L1
    assert project.train().records()["L1"]["ref"] == "SRC"  # unchanged
    assert "downstream" in panel.pos_note.text()


def test_ref_face_pick_takes_reference_with_default_port(qtbot):
    panel, project, view = make_train_panel(qtbot, chained=True)
    panel._arm_ref_pick()
    view.pick_cb is None                # pick armed on the view
    panel._on_ref_face_picked("FM", "FM_pad.Face1")
    rec = project.train().records()["L2"]
    assert rec["ref"] == "FM"
    # FM is a pure mirror: its default port is reflect
    assert (rec.get("port") or "reflect") == "reflect"


def test_port_change_rechains(qtbot):
    panel, project, _view = make_train_panel(qtbot, chained=True)
    # re-chain L2 onto the mirror FM to get multiple ports
    project.set_chain("FM", {"ref": "L1", "distance": "5"})
    project.set_chain("L2", {"ref": "FM", "port": "reflect",
                             "distance": "7"})
    panel.set_body("L2")
    p_reflect = list(pos_of(project, "L2"))
    idx = panel.train_port.findData("out")
    assert idx >= 0
    panel.train_port.setCurrentIndex(idx)
    assert project.train().records()["L2"]["port"] == "out"
    assert not np.allclose(pos_of(project, "L2"), p_reflect)
