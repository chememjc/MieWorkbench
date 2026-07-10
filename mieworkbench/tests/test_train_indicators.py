"""Optical-train GUI indicator tests (offscreen, no GL/FreeCAD):

* VtkSceneView.set_excluded_bodies -- ghosting an excluded element's
  actors (dim opacity + grey tint, no specular pop), composing correctly
  with selection highlighting, and surviving a simulated bodiesReshaped
  rebuild (reload_bodies re-runs _build_body_actors, which must re-apply
  the live exclusion set -- see vtkview.py's set_excluded_bodies
  docstring).
* VtkSceneView.set_chain_links -- the dotted chain/fold linkage overlay:
  cell count, chain-vs-fold coloring, PickableOff (so FacePicker's
  vtkCellPicker never resolves a click onto a linkage line), and the
  line-stipple property this VTK build exposes.
* OutlinerPane.set_train_info -- status badges (chained glyph, excluded
  grey+italic, problem tooltip) applied idempotently against each row's
  stored pristine label.

Only actor/mapper/property *state* is asserted, never pixels (see
docs/UI_TESTING.md sec 2) -- nothing here calls Initialize()/Render().
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

from mieworkbench.panes.outliner import OutlinerPane  # noqa: E402
from mieworkbench.widgets.vtkview import (  # noqa: E402
    VtkSceneView, _CHAIN_LINK_COLOR, _FOLD_LINK_COLOR, _GHOST_GRAY,
    _GHOST_OPACITY,
)
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_two_body_scene,
)


# ---------------------------------------------------------------------------
# VtkSceneView.set_excluded_bodies
# ---------------------------------------------------------------------------
def _expected_ghost_color(base_color):
    return tuple(0.5 * c + 0.5 * g for c, g in zip(base_color, _GHOST_GRAY))


def test_set_excluded_bodies_ghosts_then_restores(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    lens_actor = view._body_actors["Lens"][0]
    screen_actor = view._body_actors["Screen"][0]
    base_color, base_opacity = view._actor_base_style[lens_actor]
    screen_opacity = view._actor_base_style[screen_actor][1]

    view.set_excluded_bodies({"Lens"})
    lens_prop = lens_actor.GetProperty()
    assert lens_prop.GetOpacity() == pytest.approx(_GHOST_OPACITY)
    assert lens_prop.GetColor() == pytest.approx(
        _expected_ghost_color(base_color))
    assert lens_prop.GetSpecular() == pytest.approx(0.0)
    # Screen was never excluded -- untouched
    assert screen_actor.GetProperty().GetOpacity() == pytest.approx(
        screen_opacity)

    view.set_excluded_bodies(set())
    assert lens_prop.GetOpacity() == pytest.approx(base_opacity)
    assert lens_prop.GetColor() == pytest.approx(base_color)


def test_exclusion_survives_simulated_retessellation(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    view.set_excluded_bodies({"Lens"})

    old_actor = view._body_actors["Lens"][0]
    # simulates the bodiesReshaped rebuild path (new STLs -> new actors)
    view.reload_bodies(faces, structure, only=["Lens"])
    new_actor = view._body_actors["Lens"][0]

    assert new_actor is not old_actor
    assert new_actor.GetProperty().GetOpacity() == pytest.approx(
        _GHOST_OPACITY)


def test_excluded_body_composes_with_selection(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    view.set_excluded_bodies({"Lens"})

    face_id = "Lens.Revolution.Face1"
    actor = view._face_actor_map[face_id]

    view.set_selection({face_id})
    prop = actor.GetProperty()
    assert prop.GetOpacity() == pytest.approx(_GHOST_OPACITY)  # still ghosted
    assert prop.GetEdgeVisibility()                             # but outlined

    view.clear_highlights()
    assert prop.GetOpacity() == pytest.approx(_GHOST_OPACITY)
    assert not prop.GetEdgeVisibility()


# ---------------------------------------------------------------------------
# VtkSceneView.set_chain_links
# ---------------------------------------------------------------------------
def test_set_chain_links_creates_expected_cells_and_clears(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    links = [
        {"from": [0.0, 0.0, 0.0], "to": [10.0, 0.0, 0.0], "kind": "chain"},
        {"from": [10.0, 0.0, 0.0], "to": [10.0, 5.0, 0.0], "kind": "fold"},
    ]
    view.set_chain_links(links)
    assert view._chain_links_actor is not None
    assert view._chain_links_polydata.GetNumberOfCells() == 2
    assert view._chain_links_actor.GetPickable() == 0

    view.set_chain_links([])
    assert view._chain_links_actor is None
    assert view._chain_links_polydata is None


def test_chain_links_world_coords_scaled_mm_to_m(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.set_chain_links(
        [{"from": [0.0, 0.0, 0.0], "to": [10.0, 0.0, 0.0], "kind": "chain"}])
    points = view._chain_links_polydata.GetPoints()
    assert points.GetPoint(0) == pytest.approx((0.0, 0.0, 0.0))
    assert points.GetPoint(1) == pytest.approx((0.01, 0.0, 0.0))


def test_chain_vs_fold_link_colors_differ(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    links = [
        {"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0], "kind": "chain"},
        {"from": [0.0, 0.0, 0.0], "to": [0.0, 1.0, 0.0], "kind": "fold"},
    ]
    view.set_chain_links(links)
    colors = view._chain_links_polydata.GetCellData().GetScalars()
    chain_rgb = colors.GetTuple3(0)
    fold_rgb = colors.GetTuple3(1)

    assert chain_rgb != fold_rgb
    assert chain_rgb == pytest.approx(
        tuple(round(255 * c) for c in _CHAIN_LINK_COLOR))
    assert fold_rgb == pytest.approx(
        tuple(round(255 * c) for c in _FOLD_LINK_COLOR))


def test_chain_links_use_line_stipple_pattern(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.set_chain_links(
        [{"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0], "kind": "chain"}])
    prop = view._chain_links_actor.GetProperty()
    # this VTK build's python bindings expose per-actor line stippling
    # (checked at authoring time: vtk 9.6.2) -- confirm it's actually wired
    # up rather than left at the "solid line" default (0xFFFF).
    assert prop.GetLineStipplePattern() != 0xFFFF
    assert prop.GetLineStippleRepeatFactor() >= 1


def test_chain_links_default_kind_is_chain(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.set_chain_links([{"from": [0.0, 0.0, 0.0], "to": [1.0, 0.0, 0.0]}])
    colors = view._chain_links_polydata.GetCellData().GetScalars()
    assert colors.GetTuple3(0) == pytest.approx(
        tuple(round(255 * c) for c in _CHAIN_LINK_COLOR))


# ---------------------------------------------------------------------------
# OutlinerPane.set_train_info
# ---------------------------------------------------------------------------
def _make_pane(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = OutlinerPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    return pane, project


def _item_for_element(pane, element):
    for item in pane._walk():
        if item.data(0, Qt.UserRole + 1) == element:
            return item
    return None


def test_set_train_info_chained_glyph_present(qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    pane.set_train_info({"lensgrp": {"chained": True}})
    item = _item_for_element(pane, "lensgrp")
    assert item is not None
    assert item.text(0).startswith("lensgrp")
    assert "\U0001F517" in item.text(0)


def test_set_train_info_fold_glyph_present(qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    pane.set_train_info({"Screen": {"fold": True, "folded": True}})
    item = _item_for_element(pane, "Screen")
    assert "⤵" in item.text(0)
    assert not item.font(0).italic()   # folded (deployed): not "unfolded"


def test_set_train_info_excluded_grey_italic(qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    pane.set_train_info({"Screen": {"excluded": True}})
    item = _item_for_element(pane, "Screen")
    assert "(excluded)" in item.text(0)
    assert item.font(0).italic()
    assert item.foreground(0).color() == QColor(140, 140, 140)


def test_set_train_info_problem_sets_red_and_tooltip(qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    pane.set_train_info({"Screen": {"problem": "dangling reference"}})
    item = _item_for_element(pane, "Screen")
    assert item.toolTip(0) == "dangling reference"
    assert item.foreground(0).color() == QColor(200, 40, 40)


def test_set_train_info_is_idempotent(qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    info = {"lensgrp": {"chained": True, "fold": True, "folded": True}}
    pane.set_train_info(info)
    item = _item_for_element(pane, "lensgrp")
    text_after_first = item.text(0)

    pane.set_train_info(info)
    assert item.text(0) == text_after_first
    assert item.text(0).count("\U0001F517") == 1
    assert item.text(0).count("⤵") == 1


def test_set_train_info_clears_stale_badges_when_info_drops_an_element(
        qtbot, tmp_path):
    pane, _project = _make_pane(qtbot, tmp_path)
    pane.set_train_info({"Screen": {"excluded": True}})
    item = _item_for_element(pane, "Screen")
    assert "(excluded)" in item.text(0)

    pane.set_train_info({})   # Screen no longer reported at all
    assert item.text(0) == "Screen"
    assert not item.font(0).italic()
