"""PropEditorPane tests (offscreen). Constructed against the real system
library read-only for tab/row-count checks; every edit-commit exercise
runs against a tmp COPY of the library so the real opticalproperties/
tree is never written to. The import-table dialog's mapping logic is
tested at the function level (apply_column_mapping / import_table_file),
never via a modal exec()."""
import csv
import os
import shutil
import sys

import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import libschema                        # noqa: E402
from mieworkbench.core.librarymgr import LibraryManager       # noqa: E402
from mieworkbench.core.proplib import LibraryWriteError        # noqa: E402
from mieworkbench.panes.prop_editor import (                   # noqa: E402
    INVALID_CELL_COLOR, MISSING_REFERENCE_COLOR, PropEditorError,
    PropEditorPane, apply_column_mapping,
)

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))
PRIMITIVES_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "primitives"))

CATEGORY_BY_LABEL = {
    "materials": "Materials", "coatings": "Coatings",
    "polarizers": "Polarizers", "filters": "Filters",
    "gratings": "Gratings", "uniaxial": "Birefringence",
    "biaxial": "Biaxial", "figures": "Figure Errors",
    "nonlinear": "Nonlinear", "scatter": "BSDF",
    "instruments": "Instruments", "samples": "Samples",
    "images": "Images",
}

# the categories landed alongside libschema.py's COLUMN_SCHEMA drift test
# (they were previously missing from CATEGORY_INFO/CATEGORY_TABS entirely
# -- not just undocumented) -- exercised individually below. samples/
# images (samples-instruments round) have no per-row spectral TABLE (see
# CATEGORY_TABS' comment) so they carry a None schema like the others here.
NEW_CATEGORIES = ("biaxial", "figures", "nonlinear", "scatter", "instruments",
                  "samples", "images")


def _tmp_system_manager(tmp_path, with_project=False):
    sys_copy = tmp_path / "opticalproperties"
    shutil.copytree(REPO_ROOT, sys_copy)
    kwargs = {}
    if with_project:
        kwargs["project_root"] = tmp_path / "project"
    return LibraryManager(sys_copy, PRIMITIVES_ROOT, **kwargs)


# ---------------------------------------------------------------------------
# construction against the real (read-only) system library
# ---------------------------------------------------------------------------
def test_tab_count_and_labels(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    assert pane.tabs.count() == 13
    labels = [pane.tabs.tabText(i) for i in range(pane.tabs.count())]
    assert labels == ["Materials", "Coatings", "Polarizers", "Filters",
                      "Gratings", "Birefringence", "Biaxial",
                      "Figure Errors", "Nonlinear", "BSDF", "Instruments",
                      "Samples", "Images"]


def test_row_counts_match_registries(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    # the editor's row count must match the real registry csv's row count
    # (not a hard-coded magic number, so this stays correct as rows are
    # added to the library) -- and every category must actually have rows.
    for category in CATEGORY_BY_LABEL:
        editor = pane.editor(category)
        expected = len(editor.current_lib().registry_rows(category))
        assert expected > 0, category
        assert editor.row_count() == expected, category


def test_reference_column_flagged_when_blank(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    ref_col = editor._fieldnames.index("reference")
    # every real row has a reference -- none should be flagged
    for r in range(editor.row_count()):
        item = editor.table.item(r, ref_col)
        assert item.text().strip()


def test_project_library_disabled_without_project_root(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    assert not pane.library_combo.model().item(1).isEnabled()


def test_project_library_enabled_with_project_root(qtbot, tmp_path):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT,
                         project_root=tmp_path / "project")
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    assert pane.library_combo.model().item(1).isEnabled()


def test_show_category_selects_tab_and_system_library(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    pane.show_category("coatings", "system")

    assert pane.tabs.currentWidget() is pane.editor("coatings")
    assert pane.library_combo.currentData() == "system"
    for editor in pane._editors.values():
        assert editor.which_library == "system"


def test_show_category_selects_tab_and_project_library(qtbot, tmp_path):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT,
                         project_root=tmp_path / "project")
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    pane.show_category("polarizers", "project")

    assert pane.tabs.currentWidget() is pane.editor("polarizers")
    assert pane.library_combo.currentData() == "project"
    for editor in pane._editors.values():
        assert editor.which_library == "project"


def test_show_category_then_different_category_keeps_library(qtbot, tmp_path):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT,
                         project_root=tmp_path / "project")
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    pane.show_category("materials", "project")
    pane.show_category("filters", "project")

    assert pane.tabs.currentWidget() is pane.editor("filters")
    assert pane.library_combo.currentData() == "project"


# ---------------------------------------------------------------------------
# edit-commit path (tmp copy only)
# ---------------------------------------------------------------------------
def test_commit_edit_writes_through_and_validates(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    editor = pane.editor("materials")
    editor.set_edit_mode(True)
    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("notes")
    editor.table.item(idx, col).setText("edited by test")

    assert editor.commit() is True

    reloaded = mgr.system_lib.registry_rows("materials")
    bk7 = next(r for r in reloaded if r["name"] == "bk7")
    assert bk7["notes"] == "edited by test"


def test_commit_blocked_on_blank_reference(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    editor = pane.editor("materials")
    editor.set_edit_mode(True)
    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    ref_col = editor._fieldnames.index("reference")
    editor.table.item(idx, ref_col).setText("")

    original_text = (mgr.system_lib.registry_path("materials")).read_text()
    with pytest.raises(PropEditorError):
        editor.commit()
    # nothing was written -- commit is blocked before any file touch
    assert mgr.system_lib.registry_path("materials").read_text() \
        == original_text


def test_commit_rolls_back_on_loader_rejection(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    editor = pane.editor("materials")
    editor.set_edit_mode(True)
    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("class")
    editor.table.item(idx, col).setText("not_a_real_class")

    before = mgr.system_lib.registry_path("materials").read_text()
    with pytest.raises(LibraryWriteError):
        editor.commit()
    after = mgr.system_lib.registry_path("materials").read_text()
    assert before == after   # rolled back


def test_add_and_delete_row(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("filters")
    start = editor.row_count()

    editor.add_row()
    assert editor.row_count() == start + 1

    editor.delete_row(start)   # remove the freshly-added blank row
    assert editor.row_count() == start


def test_selecting_a_row_with_a_table_plots_a_chart(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("coatings")

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "pbs_visible_45")
    editor.table.selectRow(idx)
    assert editor.chart_view.chart().series()   # at least one series plotted


# ---------------------------------------------------------------------------
# import-table mapping (pure function, no modal dialog)
# ---------------------------------------------------------------------------
def test_apply_column_mapping_success():
    src_headers = ["lam_nm", "n_val", "k_val"]
    src_rows = [
        {"lam_nm": "400", "n_val": "1.50", "k_val": "0.0"},
        {"lam_nm": "500", "n_val": "1.49", "k_val": "0.0"},
    ]
    mapping = {"wavelength_nm": "lam_nm", "n": "n_val", "k": "k_val"}
    headers, rows = apply_column_mapping(
        src_headers, src_rows, mapping, ("wavelength_nm", "n", "k"))
    assert headers == ["wavelength_nm", "n", "k"]
    assert rows == [(400.0, 1.50, 0.0), (500.0, 1.49, 0.0)]


def test_apply_column_mapping_missing_required_column():
    with pytest.raises(ValueError):
        apply_column_mapping(["lam_nm", "n_val"], [], {"n": "n_val"},
                             ("wavelength_nm", "n", "k"))


def test_apply_column_mapping_unknown_source_column():
    with pytest.raises(ValueError):
        apply_column_mapping(
            ["lam_nm"], [], {"wavelength_nm": "does_not_exist"},
            ("wavelength_nm",))


def test_apply_column_mapping_non_numeric_cell():
    with pytest.raises(ValueError):
        apply_column_mapping(
            ["lam_nm"], [{"lam_nm": "not-a-number"}],
            {"wavelength_nm": "lam_nm"}, ("wavelength_nm",))


def test_import_table_file_end_to_end(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    editor.set_edit_mode(True)

    editor.add_row()
    rows = editor.rows_from_table()
    new_idx = len(rows) - 1
    for col, value in (("name", "test_material"), ("class", "glass"),
                       ("model", "tabulated"),
                       ("density_kg_m3", "2500"),
                       ("reference", "unit test fixture")):
        editor.table.item(new_idx, editor._fieldnames.index(col)) \
            .setText(value)

    src_csv = tmp_path / "external_nk.csv"
    src_csv.write_text("lam_nm,n_val,k_val\n400,1.5,0.0\n500,1.49,0.0\n")

    written = editor.import_table_file(
        str(src_csv),
        {"wavelength_nm": "lam_nm", "n": "n_val", "k": "k_val"},
        "test_material")

    assert any(p.name == "test_material.mietab" for p in written)
    headers, table_rows = mgr.system_lib.table_data(
        "materials", "test_material.mietab")
    assert headers == ["wavelength_nm", "n", "k"]
    assert table_rows == [(400.0, 1.5, 0.0), (500.0, 1.49, 0.0)]

    rows_after = mgr.system_lib.registry_rows("materials")
    row = next(r for r in rows_after if r["name"] == "test_material")
    assert row["nk_file"] == "test_material.mietab"

    ok, _ = mgr.system_lib.validate()
    assert ok


def test_import_table_file_requires_existing_row(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")

    src_csv = tmp_path / "external_nk.csv"
    src_csv.write_text("lam_nm,n_val,k_val\n400,1.5,0.0\n")

    with pytest.raises(PropEditorError):
        editor.import_table_file(
            str(src_csv), {"wavelength_nm": "lam_nm", "n": "n_val",
                          "k": "k_val"}, "no_such_row")


# ---------------------------------------------------------------------------
# promote-to-system (non-modal path)
# ---------------------------------------------------------------------------
def test_promote_row_happy_path(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path, with_project=True)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    mgr.ensure_project_item("materials", "aluminum")
    # remove it from the system copy so promote adds it back cleanly
    rows = [r for r in mgr.system_lib.registry_rows("materials")
           if r["name"] != "aluminum"]
    fieldnames = mgr.system_lib.registry_fieldnames("materials")
    with open(mgr.system_lib.registry_path("materials"), "w",
             newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (mgr.system_lib.root / "nk" / "aluminum.mienk").unlink()
    mgr.system_lib.reload()

    editor = pane.editor("materials")
    editor.set_library("project")
    result = editor.promote_row("aluminum")
    assert not isinstance(result, dict)
    assert "aluminum" in mgr.system_lib.material_names()


def test_promote_row_conflict_returns_dict(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path, with_project=True)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    mgr.ensure_project_item("materials", "aluminum")
    rows = mgr.project_lib.registry_rows("materials")
    for r in rows:
        r["notes"] = r["notes"] + " changed"
    fieldnames = mgr.system_lib.registry_fieldnames("materials")
    with open(mgr.project_lib.registry_path("materials"), "w",
             newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    editor = pane.editor("materials")
    editor.set_library("project")
    result = editor.promote_row("aluminum")
    assert isinstance(result, dict)
    assert "system_row" in result and "project_row" in result


# ---------------------------------------------------------------------------
# libschema wiring: header tooltips, status line, advisory cell validation
# ---------------------------------------------------------------------------
def test_header_tooltips_set_and_nonempty(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")

    assert editor.table.columnCount() == len(editor._fieldnames)
    for c, col in enumerate(editor._fieldnames):
        header_item = editor.table.horizontalHeaderItem(c)
        assert header_item is not None, col
        tooltip = header_item.toolTip()
        assert tooltip.strip(), "no tooltip set for column %r" % col
        # every materials.miemat column is documented -- the tooltip must
        # come from the real schema, not the "no schema entry" fallback
        assert "no schema entry" not in tooltip, col


def test_header_tooltip_degrades_gracefully_for_unknown_column(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")

    assert libschema.tooltip_text("materials", "not_a_real_column") \
        == "not_a_real_column -- no schema entry (undocumented / " \
           "registry-added column)"


def test_status_label_updates_on_current_cell_changed(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    pane.show_category("materials", "system")
    editor = pane.editor("materials")

    assert pane.status_label.text() == ""
    col = editor._fieldnames.index("density_kg_m3")
    editor.table.setCurrentCell(0, col)
    assert pane.status_label.text().startswith("density_kg_m3")
    assert "kg/m^3" in pane.status_label.text()


def test_status_label_ignores_background_tab_cell_changes(qtbot):
    """Selecting a cell on a tab that ISN'T currently shown must not
    stomp the status label -- the pane only reflects the visible tab's
    editor (guarded via QTabWidget.currentWidget() in
    PropEditorPane._on_column_status_changed)."""
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    pane.show_category("materials", "system")

    background = pane.editor("coatings")
    col = background._fieldnames.index("aoi_deg")
    background.table.setCurrentCell(0, col)

    assert pane.status_label.text() == ""


def test_status_label_clears_on_tab_switch(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    pane.show_category("materials", "system")
    editor = pane.editor("materials")
    editor.table.setCurrentCell(0, editor._fieldnames.index("density_kg_m3"))
    assert pane.status_label.text() != ""

    pane.show_category("coatings", "system")
    assert pane.status_label.text() == ""


def test_column_status_helper_reports_no_schema_entry_for_unknown_column(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    assert editor._column_status(None) == ""
    assert editor._column_status(-1) == ""
    assert editor._column_status(len(editor._fieldnames) + 5) == ""


def test_validation_marks_bad_numeric_cell_and_clears_on_fix(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("density_kg_m3")
    item = editor.table.item(idx, col)

    item.setText("-500")
    assert item.background().color() == INVALID_CELL_COLOR
    assert item.toolTip()

    item.setText("2500")
    assert item.background().color() != INVALID_CELL_COLOR


def test_validation_flags_bad_enum_cell(qtbot, tmp_path):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("class")
    item = editor.table.item(idx, col)

    item.setText("not_a_real_class")
    assert item.background().color() == INVALID_CELL_COLOR

    item.setText("glass")
    assert item.background().color() != INVALID_CELL_COLOR


def test_validation_never_touches_reference_column_styling(qtbot, tmp_path):
    """The pre-existing blank-reference rule owns the reference column
    (applied at populate/reload time, checked here); the new advisory
    validator has no entry for `reference` and _on_item_changed explicitly
    skips the column, so a live edit must never paint it
    INVALID_CELL_COLOR (that would fight/shadow the existing rule)."""
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("reference")
    item = editor.table.item(idx, col)

    item.setText("")
    assert item.background().color() != INVALID_CELL_COLOR

    item.setText("some citation")
    assert item.background().color() != INVALID_CELL_COLOR

    # reload (populate-time path) still applies the pre-existing rule
    rows[idx]["reference"] = ""
    editor._populate_table(rows)
    reloaded_item = editor.table.item(idx, col)
    assert reloaded_item.background().color() == MISSING_REFERENCE_COLOR


# ---------------------------------------------------------------------------
# the five newly-wired categories: birefringence/biaxial, figure errors,
# nonlinear, BSDF scatter, instruments -- were previously missing from
# proplib.CATEGORY_INFO and this pane's CATEGORY_TABS entirely (not just
# undocumented in libschema), so the pane couldn't open them at all.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", NEW_CATEGORIES)
def test_new_category_rows_load_with_documented_header_tooltips(qtbot,
                                                                 category):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor(category)

    expected = len(editor.current_lib().registry_rows(category))
    assert expected > 0, category
    assert editor.row_count() == expected, category

    for c, col in enumerate(editor._fieldnames):
        header_item = editor.table.horizontalHeaderItem(c)
        assert header_item is not None, (category, col)
        tooltip = header_item.toolTip()
        assert tooltip.strip(), "no tooltip for %s.%s" % (category, col)
        assert "no schema entry" not in tooltip, (category, col)


# (category, row_name, validated_column, bad_value) -- one validated,
# non-reference column per category with a known-invalid value (enum
# mismatch or an out-of-range float per its libschema validator).
NEW_CATEGORY_BAD_CELL = (
    ("figures", "fig_lambda4_defocus_633", "r_norm_mm", "-5"),
    ("nonlinear", "linbo3_d", "kind", "not_a_real_kind"),
    ("scatter", "polished_fused_silica", "model", "not_abg"),
    ("instruments", "camera_generic", "class", "not_a_real_class"),
    ("samples", "latex_100nm_water", "dist", "not_a_real_dist"),
)


@pytest.mark.parametrize("category,row_name,column,bad_value",
                         NEW_CATEGORY_BAD_CELL)
def test_new_category_known_bad_cell_flags_amber_and_clears(
        qtbot, tmp_path, category, row_name, column, bad_value):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor(category)
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == row_name)
    col = editor._fieldnames.index(column)
    item = editor.table.item(idx, col)
    good_value = item.text()

    item.setText(bad_value)
    assert item.background().color() == INVALID_CELL_COLOR, \
        "%s.%s=%r should have been flagged" % (category, column, bad_value)

    item.setText(good_value)
    assert item.background().color() != INVALID_CELL_COLOR


@pytest.mark.parametrize("category,row_name", (
    ("biaxial", "ktp"),
    ("figures", "fig_lambda4_defocus_633"),
    ("nonlinear", "linbo3_d"),
    ("scatter", "polished_fused_silica"),
    ("instruments", "camera_generic"),
    ("samples", "latex_100nm_water"),
    ("images", "usaf_style_target"),
))
def test_new_category_edit_commit_round_trips(qtbot, tmp_path, category,
                                               row_name):
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor(category)
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == row_name)
    col = editor._fieldnames.index("notes")
    editor.table.item(idx, col).setText("edited by test")

    assert editor.commit() is True

    reloaded = mgr.system_lib.registry_rows(category)
    row = next(r for r in reloaded if r["name"] == row_name)
    assert row["notes"] == "edited by test"

    ok, msg = mgr.system_lib.validate()
    assert ok, msg


def test_nonlinear_commit_preserves_leading_comment_header(qtbot, tmp_path):
    """nonlinear.mienlo's packing-grammar documentation lives in a leading
    '#'-comment block ahead of the csv header -- a save through the pane
    must not silently drop it (core.proplib.CATEGORY_INFO["nonlinear"]
    ["comment_prefix"] + prop_editor._atomic_write_registry's re-prepend)."""
    mgr = _tmp_system_manager(tmp_path)
    path = mgr.system_lib.registry_path("nonlinear")
    original_header_lines = [
        line for line in path.read_text().splitlines(keepends=True)
        if line.lstrip().startswith("#")]
    assert original_header_lines   # sanity: the fixture really has comments

    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("nonlinear")
    editor.set_edit_mode(True)
    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "linbo3_d")
    col = editor._fieldnames.index("notes")
    editor.table.item(idx, col).setText("comment-preservation check")

    assert editor.commit() is True

    after_lines = path.read_text().splitlines(keepends=True)
    assert after_lines[:len(original_header_lines)] == original_header_lines
    # the rewritten header immediately follows the preserved comment block
    assert after_lines[len(original_header_lines)].startswith("kind,name,")


def test_validation_advisory_never_blocks_editing_or_row_gather(qtbot, tmp_path):
    """A cell flagged invalid by the advisory validator must remain
    editable and its (bad) text must still flow through
    rows_from_table() unchanged -- the GUI layer never rejects/strips it.
    The one real gate is the optprops loader, exercised separately by
    test_commit_rolls_back_on_loader_rejection."""
    mgr = _tmp_system_manager(tmp_path)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)
    editor = pane.editor("materials")
    editor.set_edit_mode(True)

    rows = editor.rows_from_table()
    idx = next(i for i, r in enumerate(rows) if r["name"] == "bk7")
    col = editor._fieldnames.index("density_kg_m3")
    item = editor.table.item(idx, col)

    item.setText("-500")
    assert item.background().color() == INVALID_CELL_COLOR
    assert item.flags() & Qt.ItemFlag.ItemIsEditable   # still editable
    gathered = editor.rows_from_table()
    assert gathered[idx]["density_kg_m3"] == "-500"    # not blocked/stripped
