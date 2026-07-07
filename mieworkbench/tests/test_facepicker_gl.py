"""GPU-touching FacePicker test: actually Initialize()s the render window,
frames a known scene, and verifies a synthetic center-of-view click
resolves (through vtkCellPicker) to a real face via VtkSceneView's
facePicked signal.

Needs a real OpenGL context to render anything to pick against, so it's
skipped whenever Qt is running the headless "offscreen" platform plugin
-- which this sandbox's test environment always sets
(QT_QPA_PLATFORM=offscreen). The conftest.py 'needs_gl' marker is
declared there but NOT auto-skipped (unlike 'freecad'), so the skip logic
lives here, at the point where it's actually true.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from mieworkbench.widgets.vtkview import VtkSceneView  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    make_two_body_scene,
)

pytestmark = pytest.mark.needs_gl

_OFFSCREEN = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"


@pytest.mark.skipif(_OFFSCREEN, reason="needs a real OpenGL context")
def test_center_click_picks_the_framed_face(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    view = VtkSceneView()
    qtbot.addWidget(view)
    view.resize(400, 400)
    view.show()
    view.load_bodies(faces, structure)

    # frame just the Lens face head-on so the center pixel is guaranteed
    # to land on it
    view.renderer.ResetCamera(
        view._body_actors["Lens"][0].GetBounds())
    view.interactor.Initialize()
    view.interactor.GetRenderWindow().Render()

    picked = []
    view.facePicked.connect(lambda b, f, a: picked.append((b, f, a)))

    interactor = view.interactor
    w, h = interactor.GetRenderWindow().GetSize()
    interactor.SetEventPosition(w // 2, h // 2)
    interactor.InvokeEvent("LeftButtonPressEvent")

    assert picked, "expected a face pick at the center of the framed view"
    body_name, face_id, additive = picked[0]
    assert body_name == "Lens"
    assert face_id == "Lens.Revolution.Face1"
    assert additive is False
