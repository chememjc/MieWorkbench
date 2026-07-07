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

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.librarymgr import LibraryManager       # noqa: E402
from mieworkbench.core.proplib import LibraryWriteError        # noqa: E402
from mieworkbench.panes.prop_editor import (                   # noqa: E402
    PropEditorError, PropEditorPane, apply_column_mapping,
)

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))
PRIMITIVES_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "primitives"))

EXPECTED_COUNTS = {
    "materials": 24, "coatings": 10, "polarizers": 5, "filters": 3,
    "gratings": 3, "uniaxial": 3,
}


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

    assert pane.tabs.count() == 6
    labels = [pane.tabs.tabText(i) for i in range(pane.tabs.count())]
    assert labels == ["Materials", "Coatings", "Polarizers", "Filters",
                      "Gratings", "Birefringence"]


def test_row_counts_match_registries(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = PropEditorPane(mgr)
    qtbot.addWidget(pane)

    category_by_label = {
        "materials": "Materials", "coatings": "Coatings",
        "polarizers": "Polarizers", "filters": "Filters",
        "gratings": "Gratings", "uniaxial": "Birefringence",
    }
    for category, expected in EXPECTED_COUNTS.items():
        editor = pane.editor(category)
        assert editor.row_count() == expected, category
        assert category_by_label[category]  # sanity


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
