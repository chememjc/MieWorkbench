"""Scene3DPane unit tests: Project-signal wiring (sceneLoaded/bodiesMoved/
bodiesReshaped), facePicked -> selectionChanged selection semantics, and
the rays overlay toolbar toggle -- all offscreen, driven by FakeProject
and the canned scenes in vtk_test_support.py. No real picking (that needs
a GL context, see the needs_gl-marked test elsewhere): selection is
exercised by emitting VtkSceneView.facePicked directly, exactly what a
real click would eventually emit.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.panes.scene3d import Scene3DPane  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_two_body_scene, write_simple_vtp,
)


def test_scene_loaded_builds_actors_for_every_body(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)

    pane.set_project(project)
    project.sceneLoaded.emit()

    assert set(pane.view._body_actors) == {"Lens", "Screen"}


def test_set_project_with_already_open_project_loads_immediately(
        qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)

    pane.set_project(project)   # structure already set -> no sceneLoaded
                                # needed
    assert set(pane.view._body_actors) == {"Lens", "Screen"}


def test_bodies_moved_updates_placement(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    project.bodiesMoved.emit(
        {"Lens": {"pos_mm": [10.0, 0.0, 0.0], "quat": [0, 0, 0, 1]}})

    m = pane.view._body_transforms["Lens"].GetMatrix()
    assert m.GetElement(0, 3) == 0.01


def test_bodies_reshaped_reloads_only_named_body(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    lens_before = pane.view._body_actors["Lens"][0]
    screen_before = pane.view._body_actors["Screen"][0]

    project.bodiesReshaped.emit(["Lens"])

    assert pane.view._body_actors["Lens"][0] is not lens_before
    assert pane.view._body_actors["Screen"][0] is screen_before


def test_face_picked_replaces_selection_across_bodies(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    received = []
    pane.selectionChanged.connect(
        lambda body, fset: received.append((body, set(fset))))

    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face1", False)
    assert received[-1] == ("Lens", {"Lens.Revolution.Face1"})
    assert pane.selection() == ("Lens", {"Lens.Revolution.Face1"})

    # picking a face on a DIFFERENT body always replaces the selection,
    # even with Ctrl held
    pane.view.facePicked.emit("Screen", "Screen.Pad.Face1", True)
    assert received[-1] == ("Screen", {"Screen.Pad.Face1"})


def test_face_picked_ctrl_click_toggles_within_same_body(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face1", False)
    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face1", True)
    assert pane.selection() == ("Lens", set())


def test_reshape_of_selected_body_clears_selection(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    pane.view.facePicked.emit("Lens", "Lens.Revolution.Face1", False)
    assert pane.selection()[0] == "Lens"

    project.bodiesReshaped.emit(["Lens"])
    assert pane.selection() == (None, set())


def test_rays_overlay_load_toggle_remove(qtbot, tmp_path):
    pane = Scene3DPane()
    qtbot.addWidget(pane)

    path = tmp_path / "rays.vtp"
    write_simple_vtp(path, with_rgb=True)
    pane.load_rays_vtp(path)
    assert pane.view._rays_actor is not None

    pane.rays_button.setChecked(False)
    assert pane.view._rays_actor.GetVisibility() == 0

    pane.rays_button.setChecked(True)
    assert pane.view._rays_actor.GetVisibility() == 1

    pane.remove_rays()
    assert pane.view._rays_actor is None
