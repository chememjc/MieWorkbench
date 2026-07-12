"""Undo/redo tests.

Three layers:
  1. UndoStack pure command mechanics (no Qt event loop needed beyond
     QObject signals; no FreeCAD).
  2. Project command capture against a scripted FakeWorkerFc (pre-image
     capture, inverse ordering of parameter+rebuild, dirty/clean sync).
  3. A real-FreeCAD end-to-end torture walk (marked 'freecad'):
     add -> move -> edit -> duplicate -> delete, undo to empty, redo to
     tip, comparing worker structures at both ends.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.project import Project  # noqa: E402
from mieworkbench.core.transforms import Operation  # noqa: E402
from mieworkbench.core.undostack import (  # noqa: E402
    Command, UndoStack,
)


# ---------------------------------------------------------------------------
# 1. UndoStack mechanics
# ---------------------------------------------------------------------------
def _cmd(log, name, cleanup=None):
    return Command(name,
                   lambda: log.append("do:" + name),
                   lambda: log.append("undo:" + name),
                   cleanup=cleanup)


def test_push_and_do_executes_then_records():
    log, stack = [], UndoStack()
    stack.push_and_do(_cmd(log, "a"))
    assert log == ["do:a"]
    assert stack.can_undo() and not stack.can_redo()
    assert stack.undo_text() == "a"


def test_undo_redo_walk():
    log, stack = [], UndoStack()
    for name in ("a", "b", "c"):
        stack.push_and_do(_cmd(log, name))
    assert stack.undo() and stack.undo()
    assert log[-2:] == ["undo:c", "undo:b"]
    assert stack.redo_text() == "b"
    assert stack.redo()
    assert log[-1] == "do:b"
    assert stack.undo_text() == "b"


def test_failed_first_execution_records_nothing():
    stack = UndoStack()

    def boom():
        raise RuntimeError("nope")
    with pytest.raises(RuntimeError):
        stack.push_and_do(Command("bad", boom, lambda: None))
    assert not stack.can_undo()


def test_new_command_disposes_redo_tail():
    log, disposed = [], []
    stack = UndoStack()
    stack.push_and_do(_cmd(log, "a"))
    stack.push_and_do(_cmd(log, "b", cleanup=lambda: disposed.append("b")))
    stack.undo()
    stack.push_and_do(_cmd(log, "c"))
    assert disposed == ["b"]
    assert not stack.can_redo()


def test_depth_eviction_disposes_oldest():
    disposed, log = [], []
    stack = UndoStack(depth=3)
    for name in "abcd":
        stack.push_and_do(
            _cmd(log, name, cleanup=lambda n=name: disposed.append(n)))
    assert disposed == ["a"]
    # only 3 undos are possible
    assert stack.undo() and stack.undo() and stack.undo()
    assert not stack.can_undo()


def test_mid_stack_failure_clears_and_reports():
    stack = UndoStack()
    errors = []
    stack.error.connect(errors.append)
    stack.push_and_do(Command("ok", lambda: None, lambda: None))

    def bad_undo():
        raise RuntimeError("worker gone")
    stack.push_and_do(Command("fragile", lambda: None, bad_undo))
    assert stack.undo() is False
    assert errors and "fragile" in errors[0]
    assert not stack.can_undo() and not stack.can_redo()


def test_macro_is_one_step_with_reverse_undo():
    log, stack = [], UndoStack()
    stack.begin_macro("combo")
    stack.push_and_do(_cmd(log, "x"))
    stack.push_and_do(_cmd(log, "y"))
    stack.end_macro()
    assert stack.undo_text() == "combo"
    stack.undo()
    assert log[-2:] == ["undo:y", "undo:x"]
    stack.redo()
    assert log[-2:] == ["do:x", "do:y"]


def test_abort_macro_rolls_back_children():
    log, stack = [], UndoStack()
    stack.begin_macro("broken")
    stack.push_and_do(_cmd(log, "x"))
    stack.abort_macro()
    assert log[-1] == "undo:x"
    assert not stack.can_undo()


def test_clean_index_tracks_saves():
    log, stack = [], UndoStack()
    stack.push_and_do(_cmd(log, "a"))
    stack.mark_clean()
    assert stack.is_clean()
    stack.push_and_do(_cmd(log, "b"))
    assert not stack.is_clean()
    stack.undo()
    assert stack.is_clean()          # back at the saved state
    stack.undo()
    assert not stack.is_clean()      # before the saved state
    stack.redo()
    assert stack.is_clean()


def test_clean_index_unreachable_after_divergence():
    log, stack = [], UndoStack()
    stack.push_and_do(_cmd(log, "a"))
    stack.mark_clean()
    stack.undo()
    stack.push_and_do(_cmd(log, "b"))   # overwrote the redo tail
    stack.redo()                        # no-op
    assert not stack.is_clean()


# ---------------------------------------------------------------------------
# 2. Project command capture (scripted worker; no FreeCAD, no tessellation)
# ---------------------------------------------------------------------------
class FakeWorkerFc:
    """Just enough of FcClient for property/sheet/placement commands."""

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
                "type": fc_type, "group": "Base", "value": value}
            return self._mut()
        if op == "remove_property":
            body = self._body(params["body"])
            body["properties"].pop(params["name"], None)
            return self._mut()
        if op == "set_spreadsheet":
            for s in self.structure["sheets"]:
                if s["label"] == params["sheet"] or \
                        s["name"] == params["sheet"]:
                    s["aliases"][params["alias"]]["raw"] = params["raw"]
            return self._mut()
        if op == "set_placement":
            body = self._body(params["body"])
            body["placement"] = {"pos_mm": list(params["pos_mm"]),
                                 "quat": list(params["quat"])}
            return {"placements": {params["body"]: body["placement"]}}
        if op == "rebuild_primitive":
            return self._mut(bodies=[copy.deepcopy(self.structure
                                                   ["bodies"][0])])
        if op == "tessellate":
            return {"bodies": {b: {"faces": [],
                                   "shape_key": "k1",
                                   "placement": self._body(b)["placement"]}
                               for b in (params.get("bodies")
                                         or ["Lens"])}}
        raise AssertionError("unexpected op %r" % op)

    def _body(self, key):
        for b in self.structure["bodies"]:
            if b["name"] == key or b["label"] == key:
                return b
        raise KeyError(key)

    def _mut(self, **extra):
        out = {"changed_bodies": [], "moved_bodies": [], "invalid": [],
               "placements": {}}
        out.update(extra)
        return out


def make_project():
    structure = {
        "doc": "scene", "label": "scene", "file": "/nowhere/scene.FCStd",
        "bodies": [{
            "name": "Lens", "label": "Lens", "tip": "Pad", "face_count": 1,
            "solid_closed": True, "volume_mm3": 1.0,
            "center_of_mass_mm": [0, 0, 0],
            "bbox_mm": [0, 0, 0, 1, 1, 1],
            "placement": {"pos_mm": [0.0, 0.0, 0.0],
                          "quat": [0.0, 0.0, 0.0, 1.0]},
            "placement_bound": False, "shape_key": "k1",
            "properties": {
                "material": {"type": "App::PropertyString",
                             "group": "Base", "value": "bk7"},
            },
        }],
        "sheets": [{"name": "Spreadsheet", "label": "dim", "aliases": {
            "lensth": {"cell": "B1", "raw": "=2 mm", "value": 2.0,
                       "unit": "mm"},
        }}],
    }
    project = Project()
    fake = FakeWorkerFc(structure)
    project._fc = fake
    from mieworkbench.core.geomcache import GeomCache
    import tempfile
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_undo_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    # minimal body state for placement commands
    from mieworkbench.core.transforms import BodyState
    project.body_states["Lens"] = BodyState.from_worker(
        project.structure["bodies"][0], [])
    return project, fake


def test_set_property_undo_restores_old_value():
    project, fake = make_project()
    project.set_property("Lens", "material", "sf5")
    assert project.body("Lens")["properties"]["material"]["value"] == "sf5"
    project.undo()
    assert project.body("Lens")["properties"]["material"]["value"] == "bk7"
    project.redo()
    assert project.body("Lens")["properties"]["material"]["value"] == "sf5"


def test_set_new_property_undo_removes_it():
    project, fake = make_project()
    project.set_property("Lens", "power", 5.0)
    assert "power" in project.body("Lens")["properties"]
    project.undo()
    assert "power" not in project.body("Lens")["properties"]


def test_remove_property_undo_restores_value_and_type():
    project, fake = make_project()
    project.set_property("Lens", "power", 5.0)
    project.remove_property("Lens", "power")
    assert "power" not in project.body("Lens")["properties"]
    project.undo()
    prop = project.body("Lens")["properties"]["power"]
    assert prop["value"] == 5.0
    assert prop["type"] == "App::PropertyFloat"


def test_sheet_edit_undo_restores_raw_and_rebuilds_after():
    project, fake = make_project()
    project.set_spreadsheet("dim", "lensth", "=3 mm", rebuild_group="grp")
    assert project._sheet_raw("dim", "lensth") == "=3 mm"
    fake.ops.clear()
    project.undo()
    assert project._sheet_raw("dim", "lensth") == "=2 mm"
    order = [op for op, _ in fake.ops
             if op in ("set_spreadsheet", "rebuild_primitive")]
    # the rebuild must come AFTER the value restore or geometry is stale
    assert order == ["set_spreadsheet", "rebuild_primitive"]


def test_apply_operation_undo_restores_placement():
    project, fake = make_project()
    project.apply_operation(
        "Lens", Operation("translate", {"vector_mm": [5, 0, 0]}))
    assert project.body_states["Lens"].current.to_dict()["pos_mm"][0] == \
        pytest.approx(5.0)
    project.undo()
    assert project.body_states["Lens"].current.to_dict()["pos_mm"][0] == \
        pytest.approx(0.0)
    project.redo()
    assert project.body_states["Lens"].current.to_dict()["pos_mm"][0] == \
        pytest.approx(5.0)


def test_undo_to_clean_index_clears_dirty():
    project, fake = make_project()
    project._set_dirty(False)
    project.undo_stack.mark_clean()
    project.set_property("Lens", "material", "sf5")
    assert project.is_dirty()
    project.undo()
    assert not project.is_dirty()


# ---------------------------------------------------------------------------
# 3. Real-FreeCAD torture walk
# ---------------------------------------------------------------------------
REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.freecad
def test_undo_torture_real_worker(tmp_path):
    project = Project()
    try:
        project.new_document(str(tmp_path / "scene.FCStd"))
        project.stash_root = str(tmp_path / "undo")
        empty = copy.deepcopy(project.structure)

        # add (wizard-style macro: import + param edit + rebuild)
        project.begin_macro("Add L1")
        project.import_primitive(
            os.path.join(REPO, "primitives", "lens_pcx.FCStd"), "L1")
        project.set_element_parameters(
            "dim_L1", {"ct": "=6 mm"}, rebuild_group="L1")
        project.end_macro()
        # move, place-about-point (polar), edit, duplicate, delete
        project.apply_operation(
            "L1", Operation("translate", {"vector_mm": [0, 10, 0]}))
        project.place_about_point(
            "L1", {"kind": "origin"}, {"kind": "global", "axis": "z"},
            "50", "30", aim_at_ref=True)
        project.set_property("L1", "roughness", "25")
        project.duplicate_element("L1", "L2")
        project.delete_element("L2")
        tip = copy.deepcopy(project.structure)

        while project.undo_stack.can_undo():
            assert project.undo()
        assert project.structure["bodies"] == empty["bodies"]
        assert {s["label"] for s in project.structure["sheets"]} == \
            {s["label"] for s in empty["sheets"]}

        while project.undo_stack.can_redo():
            assert project.redo()
        assert project.structure["bodies"] == tip["bodies"]
        assert {s["label"] for s in project.structure["sheets"]} == \
            {s["label"] for s in tip["sheets"]}
    finally:
        project.shutdown()
