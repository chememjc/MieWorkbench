# =============================================================================
# test_detector_face_pin.py — the detector_face body-property pin.
#
# extract_geometry bakes a `detector_face` body property into the detector
# dict's "face" (autodetected=False), REPLACING the closest-to-origin
# auto-pick. Here we drive the Scene/DetectorGrid pipeline with a model dict
# whose detector.face names a NON-closest face and assert the grid is built on
# THAT face — plus a routing check that this path stays C-engine-routable
# while the additive CLI --detector-face (extra_detector_faces) does not.
# =============================================================================
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(SCRIPTS))

import common                                              # noqa: E402
from raytracer import cengine                              # noqa: E402
from raytracer.scene import Scene                          # noqa: E402
from raytracer.detector import DetectorGrid                # noqa: E402
from raytracer.optprops import load_optical_properties     # noqa: E402
from . import scenehelpers as sh                           # noqa: E402


def _detector_box(name="Det", x0=0.03, x1=0.05, half=0.01, face_idx=None):
    """A closed box detector with 6 analytic plane faces. `face_idx` (1..6)
    pins detector.face to that FaceN; None leaves it at the closest-to-origin
    cap (Face1, the -x near cap) to mirror the extract auto-pick."""
    faces = sh.box_faces(name, x0, x1, half)
    fid = faces[(face_idx or 1) - 1]["id"]
    return {"name": name, "label": name, "role": "detector",
            "detector": {"face": fid,
                         "autodetected": face_idx is None},
            "faces": faces}


def _fake_args(**over):
    base = dict(rough_fresnel="micro", particles=None, particle_threshold=None,
                ray_differentials=False, export_rays=False,
                ghost_analysis=False, viz_pattern=None, save_fields=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _build_scene(bodies, extra_detector_faces=()):
    model = sh.make_model(bodies)
    common.validate_model(model)
    props = load_optical_properties()
    return Scene(model, props.matdb, props.coatings, optprops=props,
                 extra_detector_faces=extra_detector_faces)


def test_pinned_non_closest_face_builds_grid_there():
    """detector.face = Face2 (the FAR +x cap, farther from origin than the
    near cap Face1) -> Scene records exactly Face2, and the grid is built on
    that face's plane (+x normal)."""
    det = _detector_box(face_idx=2)     # far cap, NOT closest to origin
    scene = _build_scene([sh.source_body(), det])

    far_id = "Det.Pad.Face2"
    near_id = "Det.Pad.Face1"
    # detector_faces is keyed by the integer face index into scene.faces
    recorded_ids = {scene.faces[fid].id for fid in scene.detector_faces}
    assert recorded_ids == {far_id}
    assert near_id not in recorded_ids

    fid = next(iter(scene.detector_faces))
    grid = DetectorGrid(scene.faces[fid], 64, 8, (500e-9, 800e-9),
                        label=scene.faces[fid].id)
    assert grid.face.id == far_id
    # the +x cap's canonical outward normal is +x
    assert np.allclose(grid.normal, [1, 0, 0])


def test_pinned_face_records_power():
    """A full trace through the pinned far cap records nonzero power there
    (proves the pinned face is a live detector, not just registered)."""
    det = _detector_box(x0=0.03, x1=0.031, half=0.01, face_idx=2)
    result, grids, scene = sh.trace_scene(
        sh.make_model([sh.source_body(power_mW=2.0), det]),
        rays=4000, resolution=64)
    fid = scene.face_by_name["Det.Pad.Face2"]
    grid = grids[fid]
    assert grid.face.id == "Det.Pad.Face2"
    assert float(np.sum(grid.inc)) > 0.0


def test_body_property_pin_stays_c_routable():
    """The pinned-primary-face path adds NO extra_detector_faces feature, so
    detect_features keeps it C-routable; the additive CLI path
    (extra_detector_faces) trips an unported feature."""
    args = _fake_args()

    pinned = _build_scene([sh.source_body(), _detector_box(face_idx=2)])
    feats_pinned = cengine.detect_features(args, pinned)
    assert "extra_detector_faces" not in feats_pinned

    extra = _build_scene([sh.source_body(), _detector_box(face_idx=1)],
                         extra_detector_faces=("Det.Pad.Face2",))
    feats_extra = cengine.detect_features(args, extra)
    assert "extra_detector_faces" in feats_extra
    assert "extra_detector_faces" not in cengine.PORTED


@pytest.mark.skipif(cengine.binary_path() is None,
                    reason="miewb-trace not built (cd cengine && ./build.sh)")
def test_engine_choice_pinned_vs_extra():
    """With the binary present: a body-property pin routes engine 'c' under
    auto, an extra_detector_faces screen routes 'python'."""
    args = _fake_args(engine="auto")

    pinned = _build_scene([sh.source_body(), _detector_box(face_idx=2)])
    eng, _reason = cengine.choose_engine(args, pinned)
    assert eng == "c"

    extra = _build_scene([sh.source_body(), _detector_box(face_idx=1)],
                         extra_detector_faces=("Det.Pad.Face2",))
    eng2, reason2 = cengine.choose_engine(args, extra)
    assert eng2 == "python", reason2
