"""OutlinerPane + SelectionModel + paste-offset helper tests (offscreen,
FakeProject-driven; no FreeCAD)."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.selection import SelectionModel  # noqa: E402
from mieworkbench.core.transforms import (  # noqa: E402
    BodyState, element_bounds,
)
from mieworkbench.mainwindow import _aabb_overlap  # noqa: E402
from mieworkbench.panes.outliner import (  # noqa: E402
    OutlinerPane, role_for_body,
)
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_two_body_scene,
)


# ---------------------------------------------------------------------------
# SelectionModel
# ---------------------------------------------------------------------------
def test_selection_model_emits_once_per_change(qtbot):
    model = SelectionModel()
    seen = []
    model.changed.connect(lambda b, f, o: seen.append((b, set(f), o)))
    model.select("Lens", {"f1"}, origin="scene3d")
    model.select("Lens", {"f1"}, origin="outliner")   # no-op duplicate
    model.select("Screen", (), origin="outliner")
    model.clear()
    assert seen == [("Lens", {"f1"}, "scene3d"),
                    ("Screen", set(), "outliner"),
                    ("", set(), "")]


# ---------------------------------------------------------------------------
# role classification (mirror of the 3D view's coloring)
# ---------------------------------------------------------------------------
def test_role_for_body():
    def body(props):
        return {"properties": {k: {"value": v} for k, v in props.items()}}
    assert role_for_body(body({"power": 5, "lambdac": 633})) == "source"
    assert role_for_body(body({"material": "detector"})) == "detector"
    assert role_for_body(body({"material": "bk7"})) == "optic"
    assert role_for_body(body({})) == "ignored"
    assert role_for_body(body({"material": "none"})) == "ignored"


# ---------------------------------------------------------------------------
# OutlinerPane
# ---------------------------------------------------------------------------
def _make_pane(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = OutlinerPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    return pane, project


def test_outliner_lists_elements(qtbot, tmp_path):
    pane, project = _make_pane(qtbot, tmp_path)
    labels = [pane.tree.topLevelItem(i).text(0)
              for i in range(pane.tree.topLevelItemCount())]
    # the canned scene has a primitive-built Lens (grouped as 'lensgrp')
    # and an ungrouped Screen
    assert "lensgrp" in labels
    assert "Screen" in labels


def test_outliner_click_emits_select(qtbot, tmp_path):
    pane, project = _make_pane(qtbot, tmp_path)
    seen = []
    pane.selectBodyRequested.connect(seen.append)
    for item in pane._walk():
        if item.text(0) == "Screen":
            pane.tree.setCurrentItem(item)
            break
    assert seen and seen[-1] == "Screen"


def test_outliner_programmatic_select_does_not_echo(qtbot, tmp_path):
    pane, project = _make_pane(qtbot, tmp_path)
    seen = []
    pane.selectBodyRequested.connect(seen.append)
    pane.set_selected_body("Screen")
    assert seen == []
    assert pane.selected_body() == "Screen"


def test_outliner_selected_element_is_group_for_primitive(qtbot, tmp_path):
    pane, project = _make_pane(qtbot, tmp_path)
    pane.set_selected_body("Lens")
    assert pane.selected_element() == "lensgrp"


def test_outliner_refresh_survives_scene_change(qtbot, tmp_path):
    pane, project = _make_pane(qtbot, tmp_path)
    pane.set_selected_body("Screen")
    project.structure["bodies"] = [
        b for b in project.structure["bodies"] if b["name"] != "Lens"]
    project.sceneLoaded.emit()
    labels = [pane.tree.topLevelItem(i).text(0)
              for i in range(pane.tree.topLevelItemCount())]
    assert "lensgrp" not in labels
    assert pane.selected_body() == "Screen"


# ---------------------------------------------------------------------------
# paste-offset helpers
# ---------------------------------------------------------------------------
def test_aabb_overlap():
    a = ([0, 0, 0], [10, 10, 10])
    b = ([5, 5, 5], [15, 15, 15])
    c = ([11, 0, 0], [20, 10, 10])
    assert _aabb_overlap(a, b)
    assert not _aabb_overlap(a, c)


def test_element_bounds_corrects_for_gui_side_moves():
    body = {
        "name": "Lens", "label": "Lens",
        "bbox_mm": [0.0, -5.0, -5.0, 4.0, 5.0, 5.0],
        "center_of_mass_mm": [2.0, 0.0, 0.0],
        "placement": {"pos_mm": [0.0, 0.0, 0.0],
                      "quat": [0.0, 0.0, 0.0, 1.0]},
        "face_count": 0, "solid_closed": True,
        "properties": {},
    }
    state = BodyState.from_worker(body, [])
    state.current = state.current.__class__.from_dict(
        {"pos_mm": [7.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]})
    lo, hi = element_bounds([body], {"Lens": state}, ["Lens"])
    assert lo[0] == 7.0 and hi[0] == 11.0
    assert element_bounds([body], {}, ["Missing"]) is None
