"""Real-FreeCAD end-to-end tests for the 3D view + element-editing panes.

Run with:
    MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
        -m pytest mieworkbench/tests/test_view3d_freecad.py -q

Opens example.FCStd through a REAL Project (starts the FreeCAD worker,
see core/project.py) and drives Scene3DPane/ElementEditorPane against the
live model: example.FCStd has 7 bodies (Lens/Laser/DivergentLaser/Target/
TargetLeft/TargetRight/GlassSphere) totalling 3+3+3+6+6+6+1 = 28 faces.
The Lens body's internal name is "Body" (label "Lens"), tip "Revolution",
3 faces, and already carries a whole-body coating='MgF2' + material='BK7'.
Its parameter sheet is 'dim', alias 'lensth' = "=2 mm".
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QFormLayout  # noqa: E402

from mieworkbench.core.project import Project  # noqa: E402
from mieworkbench.panes.element_editor import ElementEditorPane  # noqa: E402
from mieworkbench.panes.scene3d import Scene3DPane  # noqa: E402

pytestmark = pytest.mark.freecad

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(REPO, "example.FCStd")
LENS_BODY = "Body"
LENS_FEATURE = "Revolution"


@pytest.fixture
def project():
    p = Project()
    p.open_fcstd(EXAMPLE)
    try:
        yield p
    finally:
        p.shutdown()


def _field_widget(pane, prop_name):
    """The editor widget (combo/lineedit/checkbox) for a property row in
    ElementEditorPane's section (a) -- reaches through the row QWidget's
    HBoxLayout that pairs the editor with its Remove button."""
    for i in range(pane.props_form.rowCount()):
        label_item = pane.props_form.itemAt(
            i, QFormLayout.ItemRole.LabelRole)
        if label_item is not None and label_item.widget().text() == prop_name:
            field_item = pane.props_form.itemAt(
                i, QFormLayout.ItemRole.FieldRole)
            row_widget = field_item.widget()
            return row_widget.layout().itemAt(0).widget()
    raise AssertionError("no property row for %r" % prop_name)


def _find_sheet_row(pane, alias):
    for row in range(pane.sheet_table.rowCount()):
        if pane.sheet_table.item(row, 0).text() == alias:
            return row
    raise AssertionError("no sheet row for alias %r" % alias)


def test_scene3d_loads_28_face_actors(qtbot, project):
    pane = Scene3DPane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    assert len(pane.view._actor_face_map) == 28
    assert set(pane.view._body_actors) == set(project.body_names())


def test_edit_material_via_element_editor_updates_body_dict(qtbot, project):
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection(LENS_BODY, set())

    combo = _field_widget(pane, "material")
    combo.setCurrentText("N-BK7")
    combo.lineEdit().editingFinished.emit()

    assert (project.body(LENS_BODY)["properties"]["material"]["value"]
           == "N-BK7")


def test_face_assignment_on_two_lens_faces_matches_facemap_oracle(
        qtbot, project):
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection(
        LENS_BODY,
        {"%s.%s.Face1" % (LENS_BODY, LENS_FEATURE),
         "%s.%s.Face2" % (LENS_BODY, LENS_FEATURE)})

    pane.facemap_prop_combo.setCurrentText("coating")
    pane.facemap_value_combo.setCurrentText("SomeCoating")
    pane.facemap_assign_button.click()

    raw = project.body(LENS_BODY)["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(raw, body=LENS_BODY,
                                       feature=LENS_FEATURE)
    assert parsed == {
        "%s.%s.Face1" % (LENS_BODY, LENS_FEATURE): "SomeCoating",
        "%s.%s.Face2" % (LENS_BODY, LENS_FEATURE): "SomeCoating",
        "%s.%s.Face3" % (LENS_BODY, LENS_FEATURE): "MgF2",
    }


def test_sheet_edit_reshapes_and_reloads_actor(qtbot, project):
    scene_pane = Scene3DPane()
    qtbot.addWidget(scene_pane)
    scene_pane.set_project(project)

    editor_pane = ElementEditorPane()
    qtbot.addWidget(editor_pane)
    editor_pane.set_project(project)
    editor_pane.set_face_selection(LENS_BODY, set())

    lens_actor_before = scene_pane.view._body_actors[LENS_BODY][0]

    with qtbot.waitSignal(project.bodiesReshaped, timeout=60000) as blocker:
        row = _find_sheet_row(editor_pane, "lensth")
        edit = editor_pane.sheet_table.cellWidget(row, 1)
        edit.setText("3")
        edit.editingFinished.emit()

    assert LENS_BODY in blocker.args[0]
    assert scene_pane.view._body_actors[LENS_BODY][0] is not lens_actor_before

    # restore so a repeated run starts from the same baseline
    project.set_spreadsheet("dim", "lensth", "=2 mm")


def test_optics_changed_fires_for_contract_props_not_miewb(qtbot, project):
    """opticsChanged (the auto-preview trigger) must fire for contract
    property edits and geometry moves, but NOT for GUI-internal miewb_*
    bookkeeping writes."""
    hits = []
    project.opticsChanged.connect(lambda: hits.append(1))

    project.set_property(LENS_BODY, "absorbance", 0.25)
    assert len(hits) == 1
    project.remove_property(LENS_BODY, "absorbance")
    assert len(hits) == 2

    project.set_property(LENS_BODY, "miewb_scratch", "internal")
    assert len(hits) == 2          # filtered out
    project.remove_property(LENS_BODY, "miewb_scratch")
    assert len(hits) == 2

    # undo/redo replay the same _do_* paths and re-fire correctly
    project.undo()                 # re-adds miewb_scratch -> still quiet
    assert len(hits) == 2
    project.undo()                 # removes it again -> still quiet
    assert len(hits) == 2
    project.undo()                 # restores absorbance -> fires
    assert len(hits) == 3
    project.undo()                 # removes absorbance -> fires
    assert len(hits) == 4
