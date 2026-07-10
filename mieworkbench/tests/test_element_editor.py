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

from mieworkbench.core.units import label_with_unit  # noqa: E402
from mieworkbench.panes.element_editor import (  # noqa: E402
    ElementEditorPane, format_sheet_raw, merge_facemap, parse_sheet_raw,
)
from mieworkbench.tests.vtk_test_support import (  # noqa: E402
    FakeProject, make_lens_two_faces_scene, make_two_body_scene,
)


# ---------------------------------------------------------------------------
# merge_facemap now lives in core/facemaps.py (tests in test_facemaps.py);
# element_editor re-exports it, exercised via the widget tests below.
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


# ---------------------------------------------------------------------------
# new biaxial/apodization/scatter round properties
# ---------------------------------------------------------------------------
def test_beam_waist_and_m2_add_as_numeric_properties(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    for name in ("beam_waist", "m2"):
        pane.add_prop_combo.setCurrentText(name)
        pane.add_prop_button.click()
        value = project.body("Lens")["properties"][name]["value"]
        assert isinstance(value, float)
        qtbot.waitUntil(
            lambda n=name: label_with_unit(n) in _prop_labels(pane),
            timeout=2000)


def test_apodization_and_crystal_axis2_add_as_text_properties(qtbot, tmp_path):
    structure, faces = make_two_body_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    pane.add_prop_combo.setCurrentText("apodization")
    pane.add_prop_button.click()
    assert project.body("Lens")["properties"]["apodization"]["value"] \
        == "gaussian:w0=1"

    pane.add_prop_combo.setCurrentText("crystal_axis2")
    pane.add_prop_button.click()
    assert project.body("Lens")["properties"]["crystal_axis2"]["value"] \
        == "0,1,0"


def test_scatter_facemap_assign_and_menu(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", {"Lens.Revolution.Face1"})

    pane.facemap_prop_combo.setCurrentText("scatter")
    pane.facemap_value_combo.setCurrentText("polished_bk7_glass")
    pane.facemap_assign_button.click()

    raw = project.body("Lens")["properties"]["scatter"]["value"]
    parsed = common.parse_facemap_spec(raw, body="Lens",
                                       feature="Revolution")
    assert parsed == {"Lens.Revolution.Face1": "polished_bk7_glass"}

    # the table refresh is deferred (a QTimer singleShot, so a commit's own
    # signal handler never rebuilds rows out from under itself)
    qtbot.waitUntil(
        lambda: any(p == "scatter" for p, _v, _f in _assignment_rows(pane)),
        timeout=2000)

    menu = pane.build_active_properties_menu()
    assert "scatter" in menu.property_submenus


def test_scatter_value_options_read_the_real_registry(qtbot):
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    names = pane._facemap_value_options("scatter")
    assert "polished_bk7_glass" in names
    assert "polished_fused_silica" in names


def test_scatter_value_options_tolerate_missing_registry(qtbot, tmp_path):
    pane = ElementEditorPane(optprops_root=str(tmp_path / "no_such_root"))
    qtbot.addWidget(pane)
    assert pane._facemap_value_options("scatter") == []


def _assignment_rows(pane):
    """[(prop_text, value_text, faces_text), ...] from the Active
    Properties table (cell widgets included)."""
    from PySide6.QtWidgets import QComboBox as _QC
    rows = []
    for row in range(pane.assign_table.rowCount()):
        prop = pane.assign_table.item(row, 0).text()
        w = pane.assign_table.cellWidget(row, 1)
        value = w.currentText() if isinstance(w, _QC) else w.text()
        faces = pane.assign_table.cellWidget(row, 2).text()
        rows.append((prop, value, faces))
    return rows


def test_facemap_assign_partial_selection_keeps_other_face(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", {"Lens.Revolution.Face1"})

    pane.facemap_prop_combo.setCurrentText("coating")
    pane.facemap_value_combo.setCurrentText("SiO2")
    pane.facemap_assign_button.click()

    new_raw = project.body("Lens")["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(new_raw, body="Lens",
                                       feature="Revolution")
    # existing whole-body 'MgF2' expands, Face1 is overridden to SiO2,
    # Face2 keeps the old value
    assert parsed == {"Lens.Revolution.Face1": "SiO2",
                      "Lens.Revolution.Face2": "MgF2"}


def test_assign_with_no_selection_targets_the_whole_body(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    pane.facemap_prop_combo.setCurrentText("roughness")
    pane.facemap_value_combo.setCurrentText("50")
    pane.facemap_assign_button.click()
    # both faces -> collapses to the bare whole-body form
    assert project.body("Lens")["properties"]["roughness"]["value"] == "50"


def test_assignments_table_groups_by_property_and_value(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    # canned Lens: whole-body coating='MgF2'; add a per-face roughness
    project.set_property("Lens", "roughness", "Face1=50")
    pane.set_face_selection("Lens", set())

    rows = _assignment_rows(pane)
    assert ("coating", "MgF2", "whole body") in rows
    assert ("roughness", "50", "Face1") in rows
    assert len(rows) == 2


def test_assignments_table_filters_by_face_selection(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "coating", "Face1=SiO2;Face2=MgF2")
    project.set_property("Lens", "roughness", "Face1=50")

    pane.set_face_selection("Lens", {"Lens.Revolution.Face2"})
    rows = _assignment_rows(pane)
    # only Face2-touching assignments survive the filter
    assert rows == [("coating", "MgF2", "Face2")]
    assert "Face2" in pane.selection_label.text()

    pane.set_face_selection("Lens", set())
    assert len(_assignment_rows(pane)) == 3


def test_assignment_row_selection_picks_its_faces(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "roughness", "Face1=50")
    pane.set_face_selection("Lens", set())

    picked = []
    pane.facesPicked.connect(lambda b, f: picked.append((b, set(f))))
    row = [r for r, entry in enumerate(_assignment_rows(pane))
           if entry[0] == "roughness"][0]
    pane.assign_table.selectRow(row)
    assert picked and picked[-1] == ("Lens", {"Lens.Revolution.Face1"})
    assert pane._face_selection == {"Lens.Revolution.Face1"}
    # selecting a row must NOT collapse the table to the filtered view
    assert len(_assignment_rows(pane)) == 2


def test_assignment_value_edit_rewrites_those_faces_only(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "coating", "Face1=SiO2;Face2=MgF2")
    pane.set_face_selection("Lens", set())

    row = [r for r, entry in enumerate(_assignment_rows(pane))
           if entry[1] == "SiO2"][0]
    combo = pane.assign_table.cellWidget(row, 1)
    combo.setCurrentText("hard_gold")
    combo.lineEdit().editingFinished.emit()

    raw = project.body("Lens")["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(raw, body="Lens",
                                       feature="Revolution")
    assert parsed == {"Lens.Revolution.Face1": "hard_gold",
                      "Lens.Revolution.Face2": "MgF2"}


def test_assignment_face_toggle_add_and_remove(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "roughness", "Face1=50")
    pane.set_face_selection("Lens", set())

    (a,) = [x for x in pane._current_assignments()[0]
            if x.prop == "roughness"]
    # add Face2 to the assignment -> both faces -> collapses to bare form
    pane._on_assignment_face_toggled(a, "Lens.Revolution.Face2", True)
    assert project.body("Lens")["properties"]["roughness"]["value"] == "50"

    # remove both faces one by one; the property disappears with the last
    (a,) = [x for x in pane._current_assignments()[0]
            if x.prop == "roughness"]
    pane._on_assignment_face_toggled(a, "Lens.Revolution.Face1", False)
    (a,) = [x for x in pane._current_assignments()[0]
            if x.prop == "roughness"]
    pane._on_assignment_face_toggled(a, "Lens.Revolution.Face2", False)
    assert "roughness" not in project.body("Lens")["properties"]


def test_remove_assignment_button_removes_only_its_faces(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "coating", "Face1=SiO2;Face2=MgF2")
    pane.set_face_selection("Lens", set())

    (a,) = [x for x in pane._current_assignments()[0]
            if x.value == "SiO2"]
    pane._on_remove_assignment(a)
    raw = project.body("Lens")["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(raw, body="Lens",
                                       feature="Revolution")
    assert parsed == {"Lens.Revolution.Face2": "MgF2"}

    # removing a whole-body assignment removes the property
    (a,) = [x for x in pane._current_assignments()[0]
            if x.prop == "coating"]
    pane._on_remove_assignment(a)
    assert "coating" not in project.body("Lens")["properties"]


def test_invalid_raw_value_shows_as_invalid_row(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "roughness", "Face1=50;Face1=60")
    pane.set_face_selection("Lens", set())

    rows = _assignment_rows(pane)
    assert any(p == "roughness (invalid)" for p, _v, _f in rows)


def test_active_properties_menu_checkmarks_and_apply(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    # coating=MgF2 whole body (canned) -> checked for any selection
    pane.set_face_selection("Lens", {"Lens.Revolution.Face1"})

    menu = pane.build_active_properties_menu()
    assert menu is not None
    # NOTE: never retrieve submenus via QAction.menu() -- PySide6 hands
    # ownership of that wrapper to Python and the GC deletes the C++ menu
    submenus = menu.property_submenus
    assert set(submenus) == set(
        ("coating", "roughness", "diffuser", "scatter", "grating",
         "surface_override"))
    coating_items = {a.text(): a for a in submenus["coating"].actions()
                     if a.text() and a.text() != "Custom…"}
    assert coating_items["MgF2"].isChecked()

    # clicking the checked value removes it from the selected face
    coating_items["MgF2"].trigger()
    raw = project.body("Lens")["properties"]["coating"]["value"]
    parsed = common.parse_facemap_spec(raw, body="Lens",
                                       feature="Revolution")
    assert parsed == {"Lens.Revolution.Face2": "MgF2"}


def test_menu_grating_apply_never_collapses_to_bare_form(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())   # whole-body target

    pane._on_menu_value("grating", "600:v", was_fully_applied=False)
    raw = project.body("Lens")["properties"]["grating"]["value"]
    # the grating contract forbids the bare whole-body shorthand
    assert raw.startswith("Face")
    parsed = common.parse_facemap_spec(raw, body="Lens",
                                       feature="Revolution")
    assert set(parsed) == {"Lens.Revolution.Face1",
                           "Lens.Revolution.Face2"}


def test_menu_select_and_clear_face_selection(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    pane.set_face_selection("Lens", set())

    picked = []
    pane.facesPicked.connect(lambda b, f: picked.append((b, set(f))))
    pane._menu_select_all()
    assert picked[-1] == ("Lens", {"Lens.Revolution.Face1",
                                   "Lens.Revolution.Face2"})
    pane._menu_clear_selection()
    assert picked[-1] == ("Lens", set())


def test_emit_marker_appears_in_faces_display(qtbot, tmp_path):
    structure, faces = make_lens_two_faces_scene(tmp_path)
    project = FakeProject(structure, faces)
    pane = ElementEditorPane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    project.set_property("Lens", "power", 5.0)
    project.set_property("Lens", "lambdac", 633.0)
    # a per-face assignment names its faces, so the emit marker shows on
    # the working face (Face1: centroid closest to the origin)
    project.set_property("Lens", "roughness", "Face1=50")
    pane.set_face_selection("Lens", set())
    qtbot.waitUntil(
        lambda: any("Face1 (emit)" in f
                    for _p, _v, f in _assignment_rows(pane)),
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
