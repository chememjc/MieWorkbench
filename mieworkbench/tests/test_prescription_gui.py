"""ElementEditorPane 'Prescription' box (engine3 Sec 3, P5).

Offscreen widget tests (FakeProject, no FreeCAD): the read-only prescription
group appears for a covered primitive element and shows its analytic optical
surfaces, and stays hidden for a non-covered element. The displayed values
come from the SAME pure primitivelib.build_prescription_entry the extractor
cross-checks against (the single authoring path).
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.panes.element_editor import ElementEditorPane  # noqa: E402
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_two_body_scene,
)


def _pcx_scene(tmp_path):
    """make_two_body_scene, but the lens is a covered lens_pcx primitive with
    a dim_Lens1 sheet carrying its basic terms (mm)."""
    structure, faces = make_two_body_scene(tmp_path)
    lens = structure["bodies"][0]
    lens["label"] = "Lens1"
    lens["properties"]["miewb_primitive"]["value"] = "lens_pcx"
    lens["properties"]["miewb_group"]["value"] = "Lens1"
    structure["sheets"].append({
        "name": "dim_Lens1", "label": "dim_Lens1", "aliases": {
            "R_front": {"cell": "B1", "raw": "=25 mm", "value": 25.0,
                        "unit": "mm"},
            "ct": {"cell": "B2", "raw": "=5 mm", "value": 5.0, "unit": "mm"},
            "aperture": {"cell": "B3", "raw": "=20 mm", "value": 20.0,
                         "unit": "mm"},
        }})
    return structure, faces


def test_prescription_box_shows_for_covered_primitive(qtbot, tmp_path):
    structure, faces = _pcx_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())
    # set_face_selection schedules a deferred refresh (interval-0 QTimer);
    # wait for it (isVisibleTo is offscreen-safe -- the top-level is not shown)
    qtbot.waitUntil(
        lambda: pane.prescription_box.isVisibleTo(pane), timeout=2000)

    text = pane.prescription_label.text()
    assert "lens_pcx" in text
    assert "sphere" in text          # the front cap
    assert "cylinder" in text        # the edge rim
    # the front sphere radius (25 mm) is reported
    assert "25" in text


def test_prescription_box_hidden_for_non_primitive(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)   # miewb_primitive='lens'
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())
    # let the deferred refresh run, then confirm the box stays hidden for a
    # non-covered element (miewb_primitive='lens')
    qtbot.wait(50)
    assert not pane.prescription_box.isVisibleTo(pane)
