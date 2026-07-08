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
            group = self._group_of(params["body"])
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


def test_snap_offset_translates_along_axis(qtbot):
    panel, project, view = make_panel(qtbot)
    panel.set_body("Lens")
    panel._pick_snap_target()
    view.pick_cb("TgtA", "TgtA.Pad.Face1")
    assert panel.snap_offset_btn.isEnabled()
    c0 = project.body_states["Lens"].optical_center_world().copy()
    panel.snap_offset.setValue(5.0)
    panel._apply_snap_offset()
    c1 = project.body_states["Lens"].optical_center_world()
    # moved +5 mm along the target axis (+y)
    assert np.allclose(c1 - c0, [0, 5, 0], atol=1e-6)


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
