"""InspectorPane unit tests: set_body() scopes the view to a single body,
the pane auto-refreshes on that body's bodiesReshaped/bodiesMoved, face
picking (again via a direct facePicked emit -- no GL needed) plus the
Select-all/Clear buttons drive faceSelectionChanged.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.panes.inspector3d import InspectorPane  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_lens_two_faces_scene, make_two_body_scene,
)


def test_set_body_shows_only_that_body(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)

    pane.set_body(project, "Lens")

    assert set(pane.view._body_actors) == {"Lens"}
    assert pane._face_ids == ["Lens.Revolution.Face1"]


def test_switching_body_rebuilds_to_just_the_new_one(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)

    pane.set_body(project, "Lens")
    pane.set_body(project, "Screen")

    assert set(pane.view._body_actors) == {"Screen"}


def test_select_all_then_clear(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    received = []
    pane.faceSelectionChanged.connect(
        lambda body, fset: received.append((body, set(fset))))

    pane.select_all_button.click()
    assert received[-1] == (
        "Lens", {"Lens.Revolution.Face1", "Lens.Revolution.Face2"})

    pane.clear_button.click()
    assert received[-1] == ("Lens", set())


def test_face_picked_for_the_shown_body_updates_selection(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face1", False)
    assert pane.selection() == ("Lens", {"Lens.Revolution.Face1"})

    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face2", True)
    assert pane.selection() == (
        "Lens", {"Lens.Revolution.Face1", "Lens.Revolution.Face2"})


def test_face_picked_for_a_different_body_is_ignored(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    pane.view.facePicked.emit("Screen", "Screen.Pad.Face1", False)
    assert pane.selection() == ("Lens", set())


def test_bodies_reshaped_for_shown_body_rebuilds_view(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    actor_before = pane.view._body_actors["Lens"][0]
    project.bodiesReshaped.emit(["Lens"])
    assert pane.view._body_actors["Lens"][0] is not actor_before


def test_bodies_reshaped_for_other_body_is_ignored(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    actor_before = pane.view._body_actors["Lens"][0]
    project.bodiesReshaped.emit(["Screen"])
    assert pane.view._body_actors["Lens"][0] is actor_before


def test_bodies_moved_for_shown_body_updates_transform(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = InspectorPane()
    qtbot.addWidget(pane)
    pane.set_body(project, "Lens")

    project.bodiesMoved.emit(
        {"Lens": {"pos_mm": [5.0, 0.0, 0.0], "quat": [0, 0, 0, 1]}})
    m = pane.view._body_transforms["Lens"].GetMatrix()
    assert m.GetElement(0, 3) == 0.005
