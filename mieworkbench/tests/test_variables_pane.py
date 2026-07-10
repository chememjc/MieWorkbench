"""VariablesPane tests (offscreen): table population, dialog-free
add/remove/commit API, expression display vs EditRole, invalid-input
rejection, sweep checkbox/spec/run-count, undo, cycle highlighting, and
the pane -> apply_variable_cells -> chain-ripple path. Driven with the
scripted TrainFakeWorker (no FreeCAD, no dialogs). See docs/UI_TESTING.md.

Run: QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
         mieworkbench/tests/test_variables_pane.py -q
"""

import os
import sys
import tempfile

import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.geomcache import GeomCache               # noqa: E402
from mieworkbench.core.project import Project                   # noqa: E402
from mieworkbench.panes.variables_pane import (                 # noqa: E402
    VariablesPane, COL_NAME, COL_VALUE,
)
from mieworkbench.tests.train_test_support import (             # noqa: E402
    TrainFakeWorker, make_scene, pos_of,
)


def make_pane(qtbot, project):
    pane = VariablesPane(project)
    qtbot.addWidget(pane)
    return pane


def _bare_project():
    """A Project with NO miewb_vars sheet at all (make_scene always seeds
    one with `gap`) - for the true empty-state test."""
    structure = {"doc": "scene", "label": "scene",
                "file": "/nowhere/scene.FCStd",
                "bodies": [], "sheets": []}
    project = Project()
    fake = TrainFakeWorker(structure)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_vars_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    fake.ops.clear()
    return project, fake


# ---------------------------------------------------------------------------
# Empty state / first add creates the sheet
# ---------------------------------------------------------------------------
def test_empty_state_table_empty_then_first_add_creates_sheet(qtbot):
    project, _ = _bare_project()
    pane = make_pane(qtbot, project)

    assert pane.table.rowCount() == 0
    assert project.variables_sheet() is None
    # everything else stays functional in the empty state
    assert pane.sweep_spec() == ([], [], [], [])
    assert pane.run_count() == 1
    assert pane.has_enabled_sweep() is False

    assert pane.add_variable("focus", value="50") is True
    assert project.variables_sheet() is not None
    assert pane.table.rowCount() == 1
    assert pane.item_for("focus", COL_NAME).text() == "focus"


# ---------------------------------------------------------------------------
# add_variable
# ---------------------------------------------------------------------------
def test_add_variable_creates_row_with_full_meta(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)

    ok = pane.add_variable("focus", value="12.5", vmin=0.0, vmax=25.0,
                           nstep=4, enabled=True, comment="note")
    assert ok is True

    aliases = project.variables_sheet()["aliases"]
    for suffix in ("", "__min", "__max", "__n", "__on"):
        assert ("focus" + suffix) in aliases

    assert pane._row_index("focus") is not None
    assert pane.item_for("focus", COL_NAME).text() == "focus"


def test_add_variable_rejects_duplicate_name(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    before = set(project.variables_sheet()["aliases"])

    assert pane.add_variable("gap", value="1") is False
    assert pane.status.text()
    assert set(project.variables_sheet()["aliases"]) == before


def test_add_variable_rejects_invalid_name(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    before = set(project.variables_sheet()["aliases"])

    assert pane.add_variable("R1", value="1") is False
    assert pane.status.text()
    assert set(project.variables_sheet()["aliases"]) == before


# ---------------------------------------------------------------------------
# Value editing: expressions, display vs EditRole, invalid rejection
# ---------------------------------------------------------------------------
def test_commit_value_expression_written_and_displayed(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    assert pane.add_variable("half", value="gap/2") is True

    item = pane.item_for("half", COL_VALUE)
    assert item.data(Qt.EditRole) == "gap/2"
    assert item.data(Qt.DisplayRole) == "gap/2  (= 12.5)"
    assert project.variables_sheet()["aliases"]["half"]["raw"] == "=gap/2"


def test_commit_value_invalid_expression_refused(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    before_raw = project.variables_sheet()["aliases"]["gap"]["raw"]

    assert pane.commit_field("gap", "value", "1/0") is False
    assert project.variables_sheet()["aliases"]["gap"]["raw"] == before_raw
    assert pane.status.text()
    item = pane.item_for("gap", COL_VALUE)
    assert item.foreground().color().name() == "#c0392b"


def test_commit_min_max_reject_non_numeric(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    before = dict(project.variables_sheet()["aliases"]["gap__min"])

    assert pane.commit_field("gap", "vmin", "abc") is False
    assert project.variables_sheet()["aliases"]["gap__min"] == before
    assert pane.status.text()


# ---------------------------------------------------------------------------
# Sweep checkbox / sweep_spec / run_count / mode
# ---------------------------------------------------------------------------
def test_sweep_enabled_toggle_and_spec(qtbot):
    project, _ = make_scene()      # gap ships enabled (gap__on = 1)
    pane = make_pane(qtbot, project)

    assert pane.has_enabled_sweep() is True
    varnames, mins, maxs, ns = pane.sweep_spec()
    assert varnames == ["miewb_vars.gap"]
    assert pane.sweep_mode == "product"
    assert pane.run_count() > 1

    assert pane.set_sweep_enabled("gap", False) is True
    assert project.variables_sheet()["aliases"]["gap__on"]["value"] == 0.0
    assert pane.has_enabled_sweep() is False
    assert pane.run_count() == 1

    pane.sweep_mode = "zip"
    assert pane.sweep_mode == "zip"


# ---------------------------------------------------------------------------
# remove_variable
# ---------------------------------------------------------------------------
def test_remove_variable_clears_row(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)

    assert pane.remove_variable("gap") is True
    aliases = project.variables_sheet()["aliases"]
    for suffix in ("", "__min", "__max", "__n", "__on"):
        assert ("gap" + suffix) not in aliases
    assert pane._row_index("gap") is None


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------
def test_undo_after_add_variable_restores_previous_state(qtbot):
    project, _ = make_scene()
    pane = make_pane(qtbot, project)
    before = set(project.variables_sheet()["aliases"])

    assert pane.add_variable("focus", value="100") is True
    assert "focus" in project.variables_sheet()["aliases"]

    project.undo()                  # ONE undo restores the pre-add state
    assert set(project.variables_sheet()["aliases"]) == before


# ---------------------------------------------------------------------------
# Chain ripple: editing a variable's value moves a chained element
# ---------------------------------------------------------------------------
def test_edit_variable_value_ripples_chained_element(qtbot):
    project, _ = make_scene()      # gap = 25
    project.set_chain("L1", {"ref": "SRC", "distance": "gap"})
    before = pos_of(project, "L1")

    pane = make_pane(qtbot, project)
    assert pane.commit_field("gap", "value", "40") is True

    after = pos_of(project, "L1")
    assert before != after
    assert after[0] - before[0] == pytest.approx(15.0)   # 40 - 25


# ---------------------------------------------------------------------------
# Cycle highlighting
# ---------------------------------------------------------------------------
def test_cyclic_rows_highlighted_red_with_tooltip(qtbot):
    project, _ = make_scene()
    sheet = project.variables_sheet()
    sheet["aliases"]["a"] = {"cell": "B2", "raw": "=b+1", "value": None,
                             "unit": ""}
    sheet["aliases"]["b"] = {"cell": "B3", "raw": "=a+1", "value": None,
                             "unit": ""}

    pane = make_pane(qtbot, project)
    pane.refresh()

    for name in ("a", "b"):
        item = pane.item_for(name, COL_NAME)
        assert item.foreground().color().name() == "#c0392b"
        assert "a" in item.toolTip() and "b" in item.toolTip()

    gap_item = pane.item_for("gap", COL_NAME)
    assert gap_item.foreground().color().name() != "#c0392b"
