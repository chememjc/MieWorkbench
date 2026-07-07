"""VtkSceneView unit tests: pure placement-transform math + role coloring
(no Qt/GL needed), then offscreen widget construction and scene-mutation
API (load_bodies/update_placement/reload_bodies/set_selection/camera/
overlay) using canned 1-triangle STL fixtures (see vtk_test_support.py).
Nothing here calls Initialize()/Render() -- VtkSceneView itself skips
those under QT_QPA_PLATFORM=offscreen, and the assertions below only
inspect actor/mapper/transform state, never pixels.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.core.transforms import axis_angle_quat  # noqa: E402
from mieworkbench.widgets.vtkview import (  # noqa: E402
    VtkSceneView, placement_to_vtk_transform, role_for_body,
)
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    make_two_body_scene, write_simple_vtp,
)


# ---------------------------------------------------------------------------
# placement_to_vtk_transform
# ---------------------------------------------------------------------------
def test_translation_scales_mm_to_m_identity_rotation():
    t = placement_to_vtk_transform(
        {"pos_mm": [1000.0, -500.0, 250.0], "quat": [0.0, 0.0, 0.0, 1.0]})
    m = t.GetMatrix()
    assert m.GetElement(0, 3) == pytest.approx(1.0)
    assert m.GetElement(1, 3) == pytest.approx(-0.5)
    assert m.GetElement(2, 3) == pytest.approx(0.25)
    for i in range(3):
        for j in range(3):
            assert m.GetElement(i, j) == pytest.approx(1.0 if i == j else 0.0)


def test_rotation_90deg_about_z_maps_x_axis_to_y_axis():
    quat = axis_angle_quat([0, 0, 1], 90.0)
    t = placement_to_vtk_transform({"pos_mm": [0, 0, 0], "quat": list(quat)})
    p = t.TransformPoint(1.0, 0.0, 0.0)
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[1] == pytest.approx(1.0, abs=1e-9)
    assert p[2] == pytest.approx(0.0, abs=1e-9)


def test_rotation_90deg_about_x_maps_y_axis_to_z_axis():
    quat = axis_angle_quat([1, 0, 0], 90.0)
    t = placement_to_vtk_transform({"pos_mm": [0, 0, 0], "quat": list(quat)})
    p = t.TransformPoint(0.0, 1.0, 0.0)
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-9)
    assert p[2] == pytest.approx(1.0, abs=1e-9)


def test_translation_and_rotation_compose_rotate_then_translate():
    quat = axis_angle_quat([0, 0, 1], 90.0)
    t = placement_to_vtk_transform(
        {"pos_mm": [1000.0, 0.0, 0.0], "quat": list(quat)})
    # local +x (1 m) rotates to world +y (1 m), then the 1 m x-translation
    # is added on top (rotation composes with translation in the matrix,
    # not the other way around).
    p = t.TransformPoint(1.0, 0.0, 0.0)
    assert p[0] == pytest.approx(1.0, abs=1e-9)
    assert p[1] == pytest.approx(1.0, abs=1e-9)
    assert p[2] == pytest.approx(0.0, abs=1e-9)


def test_default_placement_is_identity():
    t = placement_to_vtk_transform({})
    p = t.TransformPoint(1.0, 2.0, 3.0)
    assert p == pytest.approx((1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# role_for_body
# ---------------------------------------------------------------------------
def test_role_for_body_source_needs_power_and_lambdac():
    body = {"properties": {"power": {"value": 5.0},
                           "lambdac": {"value": 532.0}}}
    assert role_for_body(body) == "source"


def test_role_for_body_detector_by_material():
    body = {"properties": {"material": {"value": "detector"}}}
    assert role_for_body(body) == "detector"


def test_role_for_body_optic_default():
    body = {"properties": {"material": {"value": "BK7"}}}
    assert role_for_body(body) == "optic"


def test_role_for_body_no_properties_is_optic():
    assert role_for_body({}) == "optic"


# ---------------------------------------------------------------------------
# widget construction + scene API (offscreen)
# ---------------------------------------------------------------------------
def test_construct_offscreen(qtbot):
    view = VtkSceneView()
    qtbot.addWidget(view)
    assert view.renderer is not None


def test_load_bodies_builds_one_actor_per_face(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    assert len(view._actor_face_map) == 2
    assert set(view._body_actors) == {"Lens", "Screen"}
    ids = {fid for (_, fid) in view._actor_face_map.values()}
    assert ids == {"Lens.Revolution.Face1", "Screen.Pad.Face1"}
    # actors of one body share the exact same transform instance
    lens_actor = view._body_actors["Lens"][0]
    assert lens_actor.GetUserTransform() is view._body_transforms["Lens"]


def test_update_placement_mutates_shared_transform_in_place(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    transform_before = view._body_transforms["Screen"]
    view.update_placement(
        "Screen", {"pos_mm": [100.0, 0.0, 0.0], "quat": [0, 0, 0, 1]})

    assert view._body_transforms["Screen"] is transform_before
    m = transform_before.GetMatrix()
    assert m.GetElement(0, 3) == pytest.approx(0.1)


def test_update_placement_unknown_body_is_a_noop(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    view.update_placement("NoSuchBody", {"pos_mm": [1, 2, 3],
                                         "quat": [0, 0, 0, 1]})  # no raise


def test_reload_bodies_only_rebuilds_named_bodies(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    lens_actor_before = view._body_actors["Lens"][0]
    screen_actor_before = view._body_actors["Screen"][0]

    view.reload_bodies(faces, structure, only=["Lens"])

    assert view._body_actors["Lens"][0] is not lens_actor_before
    assert view._body_actors["Screen"][0] is screen_actor_before


def test_set_selection_highlights_then_clears(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    face_id = "Lens.Revolution.Face1"
    actor = view._face_actor_map[face_id]
    base_color = tuple(view._actor_base_style[actor][0])

    view.set_selection({face_id})
    assert actor.GetProperty().GetColor() == pytest.approx((1.0, 0.55, 0.0))
    assert actor.GetProperty().GetEdgeVisibility()

    view.clear_highlights()
    assert actor.GetProperty().GetColor() == pytest.approx(base_color)
    assert not actor.GetProperty().GetEdgeVisibility()


def test_fit_camera_and_view_along_do_not_crash_offscreen(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)

    view.fit_camera()
    for axis in ("+x", "-x", "+y", "-y", "+z", "-z"):
        view.view_along(axis)
    with pytest.raises(ValueError):
        view.view_along("bogus")


def test_vtp_overlay_uses_rgb_when_present_else_uniform_color(qtbot, tmp_path):
    view = VtkSceneView()
    qtbot.addWidget(view)

    plain_path = tmp_path / "rays_plain.vtp"
    write_simple_vtp(plain_path, with_rgb=False)
    actor = view.load_vtp_overlay(plain_path)
    assert actor.GetMapper().GetScalarVisibility() == 0
    view.remove_overlay()
    assert view._rays_actor is None

    rgb_path = tmp_path / "rays_rgb.vtp"
    write_simple_vtp(rgb_path, with_rgb=True)
    actor2 = view.load_vtp_overlay(rgb_path)
    assert actor2.GetMapper().GetScalarVisibility() == 1

    view.set_overlay_visible(False)
    assert actor2.GetVisibility() == 0
    view.set_overlay_visible(True)
    assert actor2.GetVisibility() == 1


def test_overlay_stale_greys_then_restores_rgb_coloring(qtbot, tmp_path):
    view = VtkSceneView()
    qtbot.addWidget(view)
    rgb_path = tmp_path / "rays_rgb.vtp"
    write_simple_vtp(rgb_path, with_rgb=True)
    actor = view.load_vtp_overlay(rgb_path)
    assert not view.overlay_is_stale()
    assert actor.GetMapper().GetScalarVisibility() == 1

    view.set_overlay_stale(True)
    assert view.overlay_is_stale()
    assert actor.GetMapper().GetScalarVisibility() == 0
    assert actor.GetProperty().GetColor() == pytest.approx((0.45,) * 3)
    assert actor.GetProperty().GetOpacity() == pytest.approx(0.35)

    view.set_overlay_stale(False)
    assert actor.GetMapper().GetScalarVisibility() == 1
    assert actor.GetProperty().GetOpacity() == pytest.approx(1.0)


def test_overlay_stale_survives_reload_and_empty_view(qtbot, tmp_path):
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.set_overlay_stale(True)     # no overlay yet: must not crash
    rgb_path = tmp_path / "rays_rgb.vtp"
    write_simple_vtp(rgb_path, with_rgb=True)
    actor = view.load_vtp_overlay(rgb_path)
    # a freshly loaded overlay is never stale
    assert not view.overlay_is_stale()
    assert actor.GetMapper().GetScalarVisibility() == 1
