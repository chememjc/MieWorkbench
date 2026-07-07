"""ElementEditorPane tests.

Pure logic (no Qt): merge_facemap's recompose/collapse rules, checked
against scripts/common.py's parse_facemap_spec as the oracle; and
parse_sheet_raw/format_sheet_raw's unit-preserving number edit.

Widget tests: offscreen construction against FakeProject, property add/
edit/remove, per-face facemap assignment (set_face_selection wired the
way InspectorPane.faceSelectionChanged would drive it), and the parameter-
sheet table's commit -> set_spreadsheet (+ rebuild_primitive when the body
is primitive-built).
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common  # noqa: E402  (stdlib-only shared contract hub)
import pytest  # noqa: E402
from PySide6.QtWidgets import QFormLayout  # noqa: E402

from mieworkbench.panes.element_editor import (  # noqa: E402
    ElementEditorPane, format_sheet_raw, merge_facemap, parse_sheet_raw,
)
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_lens_two_faces_scene, make_two_body_scene,
)


# ---------------------------------------------------------------------------
# merge_facemap (oracle: common.parse_facemap_spec)
# ---------------------------------------------------------------------------
def test_merge_facemap_adds_a_face_alongside_an_existing_one():
    raw = merge_facemap("Face5=X", "Body", "Pad",
                        ["Body.Pad.Face3", "Body.Pad.Face5"],
                        {"Body.Pad.Face3"}, "MgF2")
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {"Body.Pad.Face3": "MgF2", "Body.Pad.Face5": "X"}


def test_merge_facemap_assigning_every_face_collapses_to_bare_value():
    raw = merge_facemap(None, "Body", "Pad",
                        ["Body.Pad.Face3", "Body.Pad.Face5"],
                        {"Body.Pad.Face3", "Body.Pad.Face5"}, "MgF2")
    assert raw == "MgF2"
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {common.FACEMAP_ALL: "MgF2"}


def test_merge_facemap_expands_existing_all_form_before_overriding():
    raw = merge_facemap("MgF2", "Body", "Pad",
                        ["Body.Pad.Face3", "Body.Pad.Face5"],
                        {"Body.Pad.Face3"}, "SiO2")
    parsed = common.parse_facemap_spec(raw, body="Body", feature="Pad")
    assert parsed == {"Body.Pad.Face3": "SiO2", "Body.Pad.Face5": "MgF2"}
    # not collapsed -- the two faces disagree
    assert common.FACEMAP_ALL not in parsed


def test_merge_facemap_reassigning_all_faces_to_same_value_recollapses():
    raw = merge_facemap("Face3=A;Face5=B", "Body", "Pad",
                        ["Body.Pad.Face3", "Body.Pad.Face5"],
                        {"Body.Pad.Face3", "Body.Pad.Face5"}, "Z")
    assert raw == "Z"


def test_merge_facemap_single_face_body_is_bare_all_form():
    # selecting the ONLY face of a body is selecting "every face" -- the
    # per-face table's oracle re-parse should agree it's the whole-body
    # shorthand, not a Face1=... entry.
    raw = merge_facemap(None, "Lens", "Revolution",
                        ["Lens.Revolution.Face1"],
                        {"Lens.Revolution.Face1"}, "SiO2")
    assert raw == "SiO2"


# ---------------------------------------------------------------------------
# parse_sheet_raw / format_sheet_raw
# ---------------------------------------------------------------------------
def test_sheet_raw_with_unit_round_trips_and_edits_only_the_number():
    parsed = parse_sheet_raw("=2 mm")
    assert parsed == {"has_eq": True, "number": 2.0, "suffix": " mm"}
    assert format_sheet_raw(parsed, 3) == "=3 mm"


def test_bare_sheet_raw_stays_bare_after_edit():
    parsed = parse_sheet_raw("633")
    assert parsed["has_eq"] is False
    assert parsed["suffix"] == ""
    assert format_sheet_raw(parsed, 633) == "633"


def test_sheet_raw_negative_float_with_unit():
    parsed = parse_sheet_raw("=-1.5 deg")
    assert parsed["number"] == pytest.approx(-1.5)
    assert format_sheet_raw(parsed, 2.25) == "=2.25 deg"


def test_sheet_raw_rejects_garbage():
    with pytest.raises(ValueError):
        parse_sheet_raw("not-a-number")


# ---------------------------------------------------------------------------
# widget construction + behavior (offscreen, FakeProject)
# ---------------------------------------------------------------------------
def _prop_labels(pane):
    labels = []
    for i in range(pane.props_form.rowCount()):
        item = pane.props_form.itemAt(i, QFormLayout.ItemRole.LabelRole)
        if item is not None:
            labels.append(item.widget().text())
    return labels


def test_construct_offscreen(qtbot):
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    assert pane.props_form.rowCount() == 0


def test_lists_properties_and_skips_internal(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    labels = _prop_labels(pane)
    assert "material" in labels
    assert "coating" in labels
    assert not any(l.startswith("miewb_") for l in labels)


def test_add_and_remove_property(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    pane.add_prop_combo.setCurrentText("power")
    pane.add_prop_button.click()
    # a real default, never the empty string
    assert project.body("Lens")["properties"]["power"]["value"] == 5.0
    # row rebuild is deferred to the next event-loop turn (crash fix)
    qtbot.waitUntil(lambda: "power [mW]" in _prop_labels(pane), timeout=2000)

    pane._on_remove_property("power")
    assert "power" not in project.body("Lens")["properties"]
    qtbot.waitUntil(lambda: "power [mW]" not in _prop_labels(pane),
                    timeout=2000)


def test_editing_a_numeric_property_commits_a_float(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    project.set_property("Lens", "power", 5.0)
    pane.set_face_selection("Lens", set())   # body unchanged, forces refresh
    # (property lists rebuild lazily on body change; force it directly)
    pane._refresh_properties()

    pane._commit_property("power", 12.5)
    assert project.body("Lens")["properties"]["power"]["value"] == 12.5
    assert isinstance(project.body("Lens")["properties"]["power"]["value"],
                      float)


def test_facemap_assign_partial_selection_keeps_other_face(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", {"Lens.Revolution.Face1"})

    pane.facemap_prop_combo.setCurrentText("coating")
    pane.facemap_value_edit.setText("SiO2")
    pane.facemap_assign_button.click()

    new_raw = project.body("Lens")["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(new_raw, body="Lens",
                                       feature="Revolution")
    # existing whole-body 'MgF2' expands, Face1 is overridden to SiO2,
    # Face2 keeps the old value
    assert parsed == {"Lens.Revolution.Face1": "SiO2",
                      "Lens.Revolution.Face2": "MgF2"}


def test_faces_table_lists_every_face_with_assignments(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    # one row per face; the canned Lens has a whole-body coating='MgF2'
    # which shows on every row (marked as whole-body) in bold
    assert pane.faces_table.rowCount() == 2
    for row in range(2):
        assert pane.faces_table.item(row, 0).text().startswith("Face")
        note = pane.faces_table.item(row, 1).text()
        assert "coating=MgF2" in note and "whole body" in note
        assert pane.faces_table.item(row, 0).font().bold()


def test_faces_table_selection_drives_face_selection(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    picked = []
    pane.facesPicked.connect(lambda b, f: picked.append((b, set(f))))
    pane.faces_table.selectRow(0)
    fid = pane.faces_table.item(0, 0).data(0x0100)
    assert picked and picked[-1] == ("Lens", {fid})
    assert pane._face_selection == {fid}


def test_faces_table_marks_active_face_for_sources(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "power", 5.0)
    project.set_property("Lens", "lambdac", 633.0)
    pane.set_face_selection("Lens", set())
    qtbot.waitUntil(
        lambda: any("(emit)" in pane.faces_table.item(r, 0).text()
                    for r in range(pane.faces_table.rowCount())),
        timeout=2000)


def test_typed_facemap_value_is_error_checked(qtbot, tmp_path):
    """Manually typing 'Face99=...' into a per-face property must be
    rejected with a visible message, not committed to the worker."""
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    calls_before = list(project.calls)
    pane._commit_property("coating", "Face99=MgF2")
    assert project.calls == calls_before          # nothing committed
    assert pane.face_warning.isVisibleTo(pane)
    assert "Face99" in pane.face_warning.text()

    # a valid typed value goes through and clears the warning
    pane._commit_property("coating", "Face1=SiO2")
    assert ("set_property", "Lens", "coating", "Face1=SiO2") \
        in project.calls
    assert not pane.face_warning.isVisibleTo(pane)


def test_validate_facemap_value_pure():
    from mieworkbench.panes.element_editor import validate_facemap_value
    assert validate_facemap_value("Face1=MgF2", "B", "Pad", 3) is None
    assert "Face9" in validate_facemap_value("Face9=MgF2", "B", "Pad", 3)
    assert validate_facemap_value("Face1=", "B", "Pad", 3) is not None


def _find_sheet_row(pane, alias):
    for row in range(pane.sheet_table.rowCount()):
        if pane.sheet_table.item(row, 0).text() == alias:
            return row
    raise AssertionError("no sheet row for alias %r" % alias)


def test_sheet_edit_preserves_unit_and_rebuilds_primitive(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    row = _find_sheet_row(pane, "lensth")
    editor = pane.sheet_table.cellWidget(row, 1)
    assert editor.text() == "2"
    editor.setText("3")
    editor.editingFinished.emit()

    assert ("set_spreadsheet", "dim", "lensth", "=3 mm") in project.calls
    assert any(c[0] == "rebuild_primitive" and c[1] == "lensgrp"
              for c in project.calls)


def test_bare_sheet_alias_edit_stays_bare(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    row = _find_sheet_row(pane, "wavelength")
    editor = pane.sheet_table.cellWidget(row, 1)
    assert editor.text() == "633"
    editor.setText("650")
    editor.editingFinished.emit()

    assert ("set_spreadsheet", "dim", "wavelength", "650") in project.calls


def test_properties_changed_for_other_body_does_not_refresh(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    labels_before = _prop_labels(pane)
    project.set_property("Screen", "absorbance", 0.1)
    qtbot.wait(20)   # let any (wrongly) scheduled refresh fire
    assert _prop_labels(pane) == labels_before


# ---------------------------------------------------------------------------
# crash regression: committing from a row widget used to delete that same
# widget synchronously (removeRow inside its own editingFinished/activated
# signal -> use-after-free). Both historical repro paths are exercised.
# ---------------------------------------------------------------------------
def test_commit_from_row_combo_does_not_destroy_sender(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane(optprops_root=str(tmp_path))  # no registries
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    # the canned Lens already has a material row: find its combo
    row_item = None
    for i in range(pane.props_form.rowCount()):
        from PySide6.QtWidgets import QFormLayout as _QF
        label = pane.props_form.itemAt(i, _QF.ItemRole.LabelRole)
        if label is not None and label.widget().text() == "material":
            row_item = pane.props_form.itemAt(i, _QF.ItemRole.FieldRole)
    assert row_item is not None
    combo = row_item.widget().layout().itemAt(0).widget()

    # crash path 2: pick/enter a value -> commit -> refresh must be deferred
    combo.lineEdit().setText("sf5")
    combo.lineEdit().editingFinished.emit()
    # the sender must still be alive right after the signal returns
    assert combo.lineEdit().text() == "sf5"
    assert project.body("Lens")["properties"]["material"]["value"] == "sf5"

    # crash path 1: click away with the same (now-empty-ish) text -- a
    # no-op commit must not mutate or schedule destruction of the sender
    calls_before = list(project.calls)
    combo.lineEdit().editingFinished.emit()
    assert project.calls == calls_before
    qtbot.wait(20)   # deferred refresh runs; pane must survive
    assert pane.props_form.rowCount() > 0


def test_add_property_does_not_clobber_existing_value(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane(optprops_root=str(tmp_path))
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    project.set_property("Lens", "power", 12.5)
    pane.add_prop_combo.setCurrentText("power")
    pane.add_prop_button.click()
    assert project.body("Lens")["properties"]["power"]["value"] == 12.5


def test_noop_commit_is_skipped(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane(optprops_root=str(tmp_path))
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    project.set_property("Lens", "power", 5.0)
    calls_before = list(project.calls)
    pane._commit_property("power", 5.0)
    assert project.calls == calls_before


def test_intermediate_numeric_text_does_not_raise(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane(optprops_root=str(tmp_path))
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    from PySide6.QtWidgets import QLineEdit
    w = QLineEdit("-")          # QDoubleValidator 'Intermediate' text
    calls_before = list(project.calls)
    pane._commit_numeric_property("power", w)    # must not raise
    w.setText("1e")
    pane._commit_numeric_property("power", w)    # must not raise
    assert project.calls == calls_before


def test_sheet_edit_with_unchanged_number_does_not_rebuild(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane(optprops_root=str(tmp_path))
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    row = _find_sheet_row(pane, "lensth")
    editor = pane.sheet_table.cellWidget(row, 1)
    assert editor.text() == "2"
    editor.editingFinished.emit()   # focus-out, value unchanged
    assert not any(c[0] == "rebuild_primitive" for c in project.calls)
    assert not any(c[0] == "set_spreadsheet" for c in project.calls)


def test_default_registry_value_prefers_known_entries():
    from mieworkbench.panes.element_editor import default_registry_value
    assert default_registry_value(
        "material", ["sf5", "bk7", "air"]) == "bk7"
    assert default_registry_value("material", ["sf5", "air"]) == "air"
    assert default_registry_value("polarizer", []) == ""
    assert default_registry_value(
        "grating", ["vbg_1800", "amp_600"]) == "Face1=@amp_600"
    assert default_registry_value("grating", []).startswith("Face1=")
