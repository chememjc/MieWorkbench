"""WP4 element-level selection tests (offscreen, FakeProject-driven).

Covers the SelectionModel's element/sub-selection split, the 3D view's
union highlight vs single-body re-highlight, the inspector/editor neutral
member-list state, the Clear-selection action, and selection-dependent
action gating — driven end-to-end through the MainWindow dispatcher.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.selection import SelectionModel  # noqa: E402
from mieworkbench.mainwindow import MainWindow  # noqa: E402
from mieworkbench.panes.element_editor import ElementEditorPane  # noqa: E402
from mieworkbench.panes.inspector3d import InspectorPane  # noqa: E402
from mieworkbench.panes.scene3d import Scene3DPane  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_two_member_group_scene,
)

CROWN_F = "Crown.Revolution.Face1"
FLINT_F = "Flint.Revolution.Face1"


# ---------------------------------------------------------------------------
# SelectionModel
# ---------------------------------------------------------------------------
def test_select_element_sets_element_bodies_and_primary():
    model = SelectionModel()
    seen = []
    model.changed.connect(lambda b, f, o: seen.append((b, set(f), o)))
    model.select_element("achro", ["Crown", "Flint"], origin="scene3d")
    assert model.element == "achro"
    assert model.bodies == ("Crown", "Flint")
    assert model.body == "Crown"          # primary defaults to bodies[0]
    assert model.is_element() is True
    assert seen == [("Crown", set(), "scene3d")]


def test_select_element_honours_explicit_primary():
    model = SelectionModel()
    model.select_element("achro", ["Crown", "Flint"], primary="Flint")
    assert model.body == "Flint"


def test_select_clears_element_and_is_sub_selection():
    model = SelectionModel()
    model.select_element("achro", ["Crown", "Flint"])
    model.select("Flint", origin="inspector_member")
    assert model.element is None
    assert model.bodies == ("Flint",)
    assert model.body == "Flint"
    assert model.is_element() is False


def test_select_element_dedupes():
    model = SelectionModel()
    seen = []
    model.changed.connect(lambda b, f, o: seen.append(b))
    model.select_element("achro", ["Crown", "Flint"])
    model.select_element("achro", ["Crown", "Flint"])   # no-op
    assert seen == ["Crown"]


def test_single_body_element_is_not_is_element():
    model = SelectionModel()
    model.select_element("Screen", ["Screen"])
    assert model.is_element() is False


def test_clear_from_element_state_emits_empty():
    model = SelectionModel()
    model.select_element("achro", ["Crown", "Flint"])
    seen = []
    model.changed.connect(lambda b, f, o: seen.append((b, o)))
    model.clear(origin="clear_action")
    assert model.element is None and model.bodies == () and model.body is None
    assert seen == [("", "clear_action")]


# ---------------------------------------------------------------------------
# Scene3DPane: union highlight vs single-body re-highlight
# ---------------------------------------------------------------------------
def test_scene3d_select_element_highlights_union(qtbot, tmp_path):
    structure, faces = make_two_member_group_scene(tmp_path)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(FakeProject(structure, faces))

    pane.select_element(["Crown", "Flint"])
    assert pane.view._selection == {CROWN_F, FLINT_F}


def test_scene3d_element_to_body_transition_rehighlights(qtbot, tmp_path):
    """Going from a whole-element union to a single member re-highlights
    even though the primary body is unchanged."""
    structure, faces = make_two_member_group_scene(tmp_path)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(FakeProject(structure, faces))

    pane.select_element(["Crown", "Flint"])
    assert pane.view._selection == {CROWN_F, FLINT_F}
    pane.select_body("Crown")            # same primary, smaller highlight
    assert pane.view._selection == {CROWN_F}


# ---------------------------------------------------------------------------
# InspectorPane / ElementEditorPane neutral member-list state
# ---------------------------------------------------------------------------
def test_inspector_set_element_shows_member_list(qtbot, tmp_path):
    structure, faces = make_two_member_group_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)

    pane.set_element(project, "achro", ["Crown", "Flint"])
    assert not pane.member_list.isHidden()
    assert pane.member_list.list.count() == 2
    assert pane._body_name is None

    chosen = []
    pane.memberSelected.connect(chosen.append)
    pane.member_list.list.setCurrentRow(1)
    pane.member_list._on_item_clicked(pane.member_list.list.item(1))
    assert chosen == ["Flint"]

    # a single body hides the list again
    pane.set_body(project, "Crown")
    assert pane.member_list.isHidden()
    assert pane._body_name == "Crown"


def test_inspector_empty_selection_hint(qtbot, tmp_path):
    structure, faces = make_two_member_group_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_element(project, None, [])
    assert not pane.member_list.isHidden()
    assert "No element" in pane.member_list.hint.text()


def test_editor_set_element_blanks_and_lists_members(qtbot, tmp_path):
    structure, faces = make_two_member_group_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    pane.set_element("achro", ["Crown", "Flint"])
    assert not pane.member_list.isHidden()
    assert pane.member_list.list.count() == 2
    assert pane._body_name is None
    assert pane.props_form.rowCount() == 0        # blanked


def test_editor_face_assignment_still_works_after_element(qtbot, tmp_path):
    """Regression: the inspector faceSelectionChanged -> editor face-
    assignment path must keep working after an element selection blanked
    the editor."""
    structure, faces = make_two_member_group_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    pane.set_element("achro", ["Crown", "Flint"])   # blank
    # sub-select a member as InspectorPane.faceSelectionChanged would
    pane.set_face_selection("Crown", {CROWN_F})
    assert pane.member_list.isHidden()
    assert pane._body_name == "Crown"

    pane.facemap_prop_combo.setCurrentText("coating")
    pane.facemap_value_combo.setCurrentText("MgF2")
    pane._on_assign_facemap()
    assert any(c[0] == "set_property" and c[1] == "Crown" and c[2] == "coating"
               for c in project.calls)


# ---------------------------------------------------------------------------
# MainWindow dispatcher: 3D pick expansion, member sub-selection, clear,
# action gating
# ---------------------------------------------------------------------------
def _window_on_fake(qtbot, tmp_path, scene=make_two_member_group_scene):
    structure, faces = scene(tmp_path)
    fake = FakeProject(structure, faces)
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = fake
    window.scene3d.set_project(fake)
    window.outliner.set_project(fake)
    window.element_editor.set_project(fake)
    window.transform_panel.set_project(fake)
    fake.sceneLoaded.emit()
    return window, fake


def test_dispatcher_expands_3d_pick_to_whole_element(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    # a 3D pick of one member (scene3d emits this after facePicked)
    window.selection.select("Crown", origin="scene3d")

    assert window.selection.element == "achro"
    assert window.selection.is_element() is True
    assert window.scene3d.view._selection == {CROWN_F, FLINT_F}
    assert not window.inspector.member_list.isHidden()
    assert window.inspector.member_list.list.count() == 2
    assert window.element_editor.member_list.list.count() == 2
    # selection-dependent actions enabled
    assert window.copy_action.isEnabled()
    assert window.delete_action.isEnabled()
    assert window.clear_selection_action.isEnabled()
    assert window.transform_panel.polar_apply_btn.isEnabled()


def test_dispatcher_member_list_click_sub_selects(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window.selection.select("Crown", origin="scene3d")   # whole element
    # inspector member-list click -> sub-selection of one body
    window.inspector.memberSelected.emit("Flint")

    assert window.selection.element is None
    assert window.selection.body == "Flint"
    assert window.scene3d.view._selection == {FLINT_F}
    assert window.inspector._body_name == "Flint"
    assert window.inspector.member_list.isHidden()


def test_dispatcher_outliner_child_row_sub_selects(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window.outliner.selectBodyRequested.emit("Flint")   # child row
    assert window.selection.element is None
    assert window.selection.body == "Flint"
    assert window.scene3d.view._selection == {FLINT_F}


def test_dispatcher_outliner_top_row_selects_element(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window.outliner.selectElementRequested.emit("achro", "Crown")
    assert window.selection.element == "achro"
    assert window.scene3d.view._selection == {CROWN_F, FLINT_F}


def test_clear_action_clears_and_disables(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window.selection.select("Crown", origin="scene3d")
    assert window.clear_selection_action.isEnabled()

    window.clear_selection_action.trigger()
    assert window.selection.body is None
    assert window.selection.element is None
    assert window.scene3d.view._selection == set()
    assert not window.copy_action.isEnabled()
    assert not window.delete_action.isEnabled()
    assert not window.clear_selection_action.isEnabled()
    assert not window.transform_panel.polar_apply_btn.isEnabled()


def test_clear_action_on_the_scene3d_toolbar(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    # the action was inserted as a QToolButton with setDefaultAction
    from PySide6.QtWidgets import QToolButton
    actions = [b.defaultAction() for b in
               window.scene3d.findChildren(QToolButton)]
    assert window.clear_selection_action in actions


def test_actions_disabled_when_nothing_selected(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window._update_selection_actions()
    assert not window.copy_action.isEnabled()
    assert not window.delete_action.isEnabled()
    assert not window.clear_selection_action.isEnabled()
    assert not window.transform_panel.polar_apply_btn.isEnabled()


def test_ungrouped_single_body_pick_is_plain_selection(qtbot, tmp_path):
    window, fake = _window_on_fake(qtbot, tmp_path)
    window.selection.select("Screen", origin="scene3d")
    assert window.selection.element is None
    assert window.selection.is_element() is False
    assert window.inspector._body_name == "Screen"
    assert window.inspector.member_list.isHidden()
