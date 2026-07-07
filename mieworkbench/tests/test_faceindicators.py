"""Face-orientation indicator tests: pure classification/placement math
with plain floats, then offscreen actor bookkeeping through VtkSceneView
(actor construction is GPU-free; nothing here renders)."""

import math
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mieworkbench.widgets.faceindicators import (  # noqa: E402
    INDICATOR_BLUE, INDICATOR_GREEN, INDICATOR_RED, classify_indicators,
    glyph_points, indicator_radius, plus_x_face, rotation_to_normal,
)
from mieworkbench.widgets.vtkview import VtkSceneView, role_for_body  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    make_two_body_scene,
)


def _face(fid, centroid, normal=(0.0, 0.0, 1.0), area=1e-4):
    return {"id": fid, "centroid_m": list(centroid),
            "normal_hint": list(normal), "area_m2": area}


def _body(name="B", **props):
    properties = {}
    for k, v in props.items():
        properties[k] = {"value": v}
    return {"name": name, "properties": properties}


def _classify(body, faces):
    return classify_indicators(body, faces, role_for_body(body))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_source_gets_red_half_disc_on_closest_to_origin_face():
    body = _body(power=5.0, lambdac=633.0)
    faces = [_face("B.Pad.Face1", (0.0, 0.0, 0.05)),
             _face("B.Pad.Face2", (0.0, 0.0, 0.001))]
    (spec,) = _classify(body, faces)
    assert spec.face_id == "B.Pad.Face2"
    assert spec.glyph == "half_disc"
    assert spec.color == INDICATOR_RED


def test_spherical_source_single_face_is_skipped():
    body = _body(power=5.0, lambdac=633.0)
    faces = [_face("B.Pad.Face1", (0.0, 0.0, 0.0))]
    assert _classify(body, faces) == []


def test_detector_gets_red_half_disc_even_with_one_face():
    body = _body(material="detector")
    faces = [_face("B.Pad.Face1", (0.0, 0.0, 0.0))]
    (spec,) = _classify(body, faces)
    assert spec.glyph == "half_disc" and spec.color == INDICATOR_RED


def test_aperture_primitive_gets_green_disc_on_plus_x_face():
    body = _body(material="aluminum", miewb_primitive="iris")
    faces = [_face("B.Pad.Face1", (-0.001, 0.0, 0.0)),
             _face("B.Pad.Face2", (0.001, 0.0, 0.0))]
    (spec,) = _classify(body, faces)
    assert spec.face_id == "B.Pad.Face2"
    assert spec.glyph == "disc" and spec.color == INDICATOR_GREEN


def test_generic_optic_gets_blue_disc_hand_authored_included():
    body = _body(material="bk7")   # no miewb_primitive at all
    faces = [_face("B.Pad.Face1", (0.002, 0.0, 0.0)),
             _face("B.Pad.Face2", (-0.002, 0.0, 0.0))]
    (spec,) = _classify(body, faces)
    assert spec.face_id == "B.Pad.Face1"
    assert spec.color == INDICATOR_BLUE


def test_untagged_and_material_none_bodies_get_nothing():
    faces = [_face("B.Pad.Face1", (0.0, 0.0, 0.0))]
    assert _classify(_body(), faces) == []
    assert _classify(_body(material="none"), faces) == []
    assert _classify(_body(material="  "), faces) == []


# ---------------------------------------------------------------------------
# +x rule / radius / geometry
# ---------------------------------------------------------------------------
def test_plus_x_face_tie_broken_by_normal():
    faces = [_face("B.Pad.Face1", (0.001, 0.0, 0.0), normal=(-1, 0, 0)),
             _face("B.Pad.Face2", (0.001, 0.0, 0.0), normal=(1, 0, 0))]
    assert plus_x_face(faces) == "B.Pad.Face2"
    assert plus_x_face([]) is None
    assert plus_x_face([{"id": "x", "centroid_m": None}]) is None


def test_indicator_radius_scales_and_floors():
    assert indicator_radius(1e-4) == pytest.approx(0.18 * 0.01)
    assert indicator_radius(0.0) == pytest.approx(3e-4)   # floor
    assert indicator_radius(None) == pytest.approx(3e-4)


def test_rotation_to_normal_maps_z_and_is_orthonormal():
    for normal in ((0, 0, 1), (1, 0, 0), (0, 1, 0), (0.3, -0.4, 0.87)):
        rot = rotation_to_normal(normal)
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        assert np.allclose(rot @ [0, 0, 1], n, atol=1e-12)
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)
        assert np.linalg.det(rot) == pytest.approx(1.0)


def test_glyph_points_lie_on_lifted_plane_at_radius():
    centroid = (0.01, -0.02, 0.005)
    normal = (0.0, 1.0, 0.0)
    r = 0.002
    pts, tris = glyph_points(centroid, normal, r, "disc")
    center = pts[0]
    # lifted 10 um along the normal
    assert np.allclose(center, np.add(centroid, (0, 1e-5, 0)))
    for p in pts[1:]:
        assert np.linalg.norm(p - center) == pytest.approx(r)
        assert abs(float(np.dot(p - center, normal))) < 1e-12
    assert len(tris) == len(pts) - 2


def test_half_disc_spans_a_half_plane_only():
    pts, _tris = glyph_points((0, 0, 0), (0, 0, 1), 1.0, "half_disc")
    center = pts[0]
    # in the local frame the arc runs 0..pi: all y-components >= 0
    rot = rotation_to_normal((0, 0, 1))
    local = (pts[1:] - center) @ rot
    assert (local[:, 1] > -1e-12).all()
    # end points sit on the flat edge (local y == 0)
    assert local[0][1] == pytest.approx(0.0, abs=1e-12)
    assert local[-1][1] == pytest.approx(0.0, abs=1e-12)
    assert math.isclose(np.linalg.norm(local[0] - local[-1]), 2.0)


# ---------------------------------------------------------------------------
# VtkSceneView integration (offscreen)
# ---------------------------------------------------------------------------
def test_load_bodies_builds_indicator_actors(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    # Lens (optic) -> blue dot; Screen (detector) -> red half-disc
    assert view._indicators.actor_count() == 2
    for actors in view._indicators._body_actors.values():
        for actor in actors:
            assert not actor.GetPickable()
            assert actor.GetVisibility() == 1


def test_indicator_actors_share_the_body_transform(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    (lens_glyph,) = view._indicators._body_actors["Lens"]
    assert lens_glyph.GetUserTransform() is view._body_transforms["Lens"]
    # a placement move needs no glyph rebuild
    count_before = view._indicators.actor_count()
    view.update_placement("Lens", {"pos_mm": [9, 0, 0],
                                   "quat": [0, 0, 0, 1]})
    assert view._indicators.actor_count() == count_before


def test_reload_bodies_rebuilds_only_that_bodys_glyphs(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    (screen_before,) = view._indicators._body_actors["Screen"]
    (lens_before,) = view._indicators._body_actors["Lens"]
    view.reload_bodies(faces, structure, only=["Lens"])
    assert view._indicators._body_actors["Screen"][0] is screen_before
    assert view._indicators._body_actors["Lens"][0] is not lens_before


def test_visibility_toggle_applies_to_existing_and_new_glyphs(
        qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.load_bodies(faces, structure)
    view.set_face_indicators_visible(False)
    for actors in view._indicators._body_actors.values():
        for actor in actors:
            assert actor.GetVisibility() == 0
    # a reshape while hidden must produce hidden glyphs
    view.reload_bodies(faces, structure, only=["Lens"])
    (lens_glyph,) = view._indicators._body_actors["Lens"]
    assert lens_glyph.GetVisibility() == 0
    view.set_face_indicators_visible(True)
    assert lens_glyph.GetVisibility() == 1
