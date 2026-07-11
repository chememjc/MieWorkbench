"""Offscreen GUI shakedown of the "no side-math" UI features against the
three shipped design-usability demo workspaces (telephoto, telephoto_zoom,
folded_periscope), driven through the REAL FreeCAD worker + a real
MainWindow / Project / panes stack.

This is the freecad-marked (env-gated) integration twin of the scripted
train_editor/element_editor/variables_pane unit tests: those pin the pane
logic against a fake worker; this proves the same features light up on the
actual shipped scenes.  One MainWindow (= one persistent worker) is reused
across all three demos to amortize the slow AppImage startup; each test
opens its demo through MainWindow.open_model (which unpacks the .MieWB into
a throwaway var/work/ workspace and NEVER mutates the shipped archive).

Run:
    MIEWB_RUN_FREECAD=1 QT_QPA_PLATFORM=offscreen env/bin/python \
        -m pytest mieworkbench/tests/test_design_usability_shakedown.py -q
"""

import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMOS = os.path.join(REPO, "demos")

from PySide6.QtCore import Qt                                    # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402
from mieworkbench.mainwindow import MainWindow                  # noqa: E402
from mieworkbench.panes.train_editor import (                   # noqa: E402
    COL_DIST, COL_FOLD, ROLE_ELEMENT,
)
from mieworkbench.core import opticalvalues                     # noqa: E402
from mieworkbench.panes.variables_pane import (                 # noqa: E402
    COL_NAME, COL_VALUE, COL_MIN, COL_MAX, COL_STEPS, COL_SWEEP,
)

pytestmark = pytest.mark.freecad


# ---------------------------------------------------------------------------
# one MainWindow (=one worker) for the whole module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mw(qapp):
    w = MainWindow()
    try:
        yield w
    finally:
        try:
            w.project.shutdown()
        except Exception:
            pass
        w.close()


def open_demo(mw, name):
    """Open demos/<name>.MieWB through the real worker; return the Project.
    The train editor rebuilds itself synchronously off sceneLoaded, but we
    pump the loop so the scheduled indicator/variable refreshes run too."""
    path = os.path.join(DEMOS, "%s.MieWB" % name)
    assert os.path.isfile(path), path
    mw.open_model(path)
    QApplication.processEvents()
    assert mw.project.is_open(), "failed to open %s" % name
    # make sure the panes reflect the freshly-opened scene
    mw.train_editor.rebuild()
    if mw.variables_pane is not None:
        mw.variables_pane.refresh()
    QApplication.processEvents()
    return mw.project


def pos_of(project, label):
    body = project.train().primary_body_name(label)
    return list(project.body_states[body].current.to_dict()["pos_mm"])


def summary_efl(text):
    m = re.search(r"EFL\s+([\d.eE+-]+)\s*mm", text)
    return float(m.group(1)) if m else None


def summary_fno(text):
    m = re.search(r"f/([\d.]+)", text)
    return float(m.group(1)) if m else None


def _dist_display(pane, element):
    return pane.item_for_element(element).data(COL_DIST, Qt.DisplayRole)


def _dist_editrole(pane, element):
    return pane.item_for_element(element).data(COL_DIST, Qt.EditRole)


# ---------------------------------------------------------------------------
# 1a. telephoto: system paraxial summary + dual-display expression cells
# ---------------------------------------------------------------------------
def test_telephoto_train_summary(mw, qapp):
    open_demo(mw, "telephoto")
    pane = mw.train_editor

    # the system paraxial summary is visible and reads EFL ~200, f/# ~4.0
    assert pane.summary.isVisibleTo(pane), \
        "telephoto train summary should be visible for a powered train"
    text = pane.summary.text()
    assert "paraxial" in text
    efl = summary_efl(text)
    assert efl is not None, "no EFL in summary: %r" % text
    assert abs(efl - 200.0) < 3.0, "telephoto system EFL %s (want ~200)" % efl
    fno = summary_fno(text)
    assert fno is not None, "no f/# in summary: %r" % text
    assert abs(fno - 4.0) < 0.6, "telephoto working f/# %s (want ~4.0)" % fno

    # expression-driven edge (FrontGroup distance = "efl * <coef>") renders
    # the "expr  (= value)" dual display; the EditRole keeps the raw expr
    disp = _dist_display(pane, "FrontGroup")
    edit = _dist_editrole(pane, "FrontGroup")
    assert "efl" in str(edit), "FrontGroup edit role should be the raw expr"
    assert "(=" in str(disp), \
        "FrontGroup distance should show dual 'expr (= value)': %r" % disp


def test_telephoto_element_paraxial(mw, qapp):
    project = open_demo(mw, "telephoto")
    ed = mw.element_editor

    # FrontGroup achromat: EFL ~122.5 mm, f/# and NA present
    front_body = project.train().primary_body_name("FrontGroup")
    ed.set_face_selection(front_body, set())
    qapp.processEvents()
    assert ed.paraxial_box.isVisibleTo(ed), \
        "FrontGroup should show a paraxial readout"
    ptext = ed.paraxial_label.text()
    efl = summary_efl(ptext)
    assert efl is not None, "no EFL in element paraxial: %r" % ptext
    assert abs(efl - 122.5) < 1.5, "FrontGroup EFL %s (want ~122.5)" % efl
    assert "f/" in ptext and "NA" in ptext

    # the iris ("Stop") is a passthrough element: paraxial box hidden
    stop_body = project.train().primary_body_name("Stop")
    ed.set_face_selection(stop_body, set())
    qapp.processEvents()
    assert not ed.paraxial_box.isVisibleTo(ed), \
        "iris should not show a paraxial readout"


def test_telephoto_variables_listed(mw, qapp):
    open_demo(mw, "telephoto")
    vp = mw.variables_pane
    assert vp is not None

    # both design variables are listed with values + sweep metadata
    for name in ("efl", "stop_d"):
        assert vp.item_for(name, COL_NAME) is not None, \
            "variable %s missing from the Variables pane" % name
        assert vp.item_for(name, COL_VALUE).data(Qt.EditRole)
        # sweep meta present (min/max/steps/checkbox all populated)
        assert vp.item_for(name, COL_MIN) is not None
        assert vp.item_for(name, COL_MAX) is not None
        assert vp.item_for(name, COL_STEPS) is not None
        assert vp.item_for(name, COL_SWEEP).flags() & Qt.ItemIsUserCheckable

    # the edit itself is accepted without raising a modal / error
    assert vp.commit_field("efl", "value", "250") is True
    assert not vp.status.text() or "error" not in vp.status.text().lower()


def test_zoom_variable_edit_resolves_efl(mw, qapp):
    """The 'edit a variable -> the system re-solves' feature on the zoom,
    where the zoom gap `z` moves ONLY chain distances (no primitive dim
    rebuild): the summary EFL tracks it and undo restores."""
    open_demo(mw, "telephoto_zoom")
    vp = mw.variables_pane
    assert vp.item_for("z", COL_NAME) is not None
    assert summary_efl(mw.train_editor.summary.text()) is not None

    assert vp.commit_field("z", "value", "84.0") is True
    mw.train_editor.rebuild()
    qapp.processEvents()
    efl = summary_efl(mw.train_editor.summary.text())
    assert efl is not None and efl > 230.0, \
        "zoom EFL should climb toward ~258 at z=84, got %s" % efl

    assert mw.project.undo()
    mw.variables_pane.refresh()
    mw.train_editor.rebuild()
    qapp.processEvents()
    efl_back = summary_efl(mw.train_editor.summary.text())
    assert efl_back is not None and abs(efl_back - 200.0) < 3.0, \
        "zoom EFL did not revert on undo: %s" % efl_back


@pytest.mark.xfail(strict=True, reason=(
    "PRODUCT BUG: editing a miewb_vars variable that drives a chained "
    "primitive's dim-sheet rebuild (telephoto efl/stop_d -> the achromats "
    "and iris rebuild) drops that element's miewb_train_* dynamic props, so "
    "the optical train falls apart (system EFL goes nonsense) and undo does "
    "not restore them. Contrast test_zoom_variable_edit_resolves_efl, where "
    "the edited variable moves only chain distances and re-solves fine. "
    "Remove this xfail when the rebuild preserves MieTrain props."))
def test_telephoto_efl_edit_resolves_system(mw, qapp):
    open_demo(mw, "telephoto")
    front_primary = mw.project.train().primary_body_name("FrontGroup")
    assert mw.variables_pane.commit_field("efl", "value", "250") is True
    mw.train_editor.rebuild()
    qapp.processEvents()

    # the chain wiring must survive a variable-driven rebuild ...
    props = mw.project.body(front_primary)["properties"]
    assert props.get("miewb_train_distance", {}).get("value"), \
        "FrontGroup lost its chain distance after the efl rebuild"
    # ... and the system EFL should track efl=250
    efl = summary_efl(mw.train_editor.summary.text())
    assert efl is not None and abs(efl - 250.0) < 4.0, \
        "after efl=250 system EFL %s" % efl


# ---------------------------------------------------------------------------
# 2. telephoto_zoom: summary at default z + insert-optical-value menu
# ---------------------------------------------------------------------------
def test_zoom_summary_and_sensor_dual_display(mw, qapp):
    open_demo(mw, "telephoto_zoom")
    pane = mw.train_editor

    assert pane.summary.isVisibleTo(pane)
    efl = summary_efl(pane.summary.text())
    assert efl is not None and abs(efl - 200.0) < 3.0, \
        "zoom system EFL %s at default z (want ~200)" % efl

    # the Sensor edge distance is the rational BFL(z) expression -> dual
    disp = _dist_display(pane, "Sensor")
    edit = _dist_editrole(pane, "Sensor")
    assert "z" in str(edit) and "(" in str(edit), \
        "Sensor distance should be the BFL(z) expression: %r" % edit
    assert "(=" in str(disp), \
        "Sensor distance should show dual 'expr (= value)': %r" % disp


def test_zoom_insert_optical_value_menu(mw, qapp):
    project = open_demo(mw, "telephoto_zoom")
    pane = mw.train_editor

    # the right-click menu on the Sensor distance cell offers a
    # "Previous element (RearGroup)" group of insertable optical values
    menu = pane._build_context_menu("Sensor", column=COL_DIST)
    groups = getattr(menu, "optical_value_groups", None)
    assert groups is not None, "no insert-optical-value submenu on Sensor dist"
    prev_keys = [k for k in groups if k.startswith("Previous element")]
    assert prev_keys, "expected a 'Previous element' group: %r" % list(groups)
    assert any("RearGroup" in k for k in prev_keys), \
        "expected 'Previous element (RearGroup)': %r" % prev_keys

    # and the underlying value model: the tracked focus distance (the
    # image-distance-after-RearGroup / system BFL) is ~27.6 mm -- the value
    # the demo actually tracks the sensor with
    entries = opticalvalues.value_menu_model(
        project.train(), "Sensor", "distance", variables=pane._variables())
    assert entries, "empty optical-value model for Sensor distance"
    # the well-known ~27.6 mm tracked-focus value appears somewhere
    values = [e["value"] for e in entries]
    assert any(abs(v - 27.6) < 1.5 for v in values), \
        "no ~27.6 mm tracked-focus value among %s" % (
            ["%.3f" % v for v in values])
    # sanity: the "BFL of RearGroup" label exists (its value is the negative
    # element BFL, not the system BFL -- see the findings notes)
    assert any(e["label"].startswith("BFL of RearGroup") for e in entries)


# ---------------------------------------------------------------------------
# 5. folded_periscope: fold checkboxes + unfold/refold round-trip in the GUI
# ---------------------------------------------------------------------------
def test_periscope_fold_checkboxes_and_roundtrip(mw, qapp):
    project = open_demo(mw, "periscope_open_guard" if False else
                        "folded_periscope")
    pane = mw.train_editor

    # both folds show as checked fold boxes in the train editor
    for fm in ("FM1", "FM2"):
        item = pane.item_for_element(fm)
        assert item is not None, "%s missing from train editor" % fm
        assert item.checkState(COL_FOLD) == Qt.Checked, \
            "%s fold box should be checked (folded)" % fm

    # capture a downstream element's folded pose
    exit_folded = pos_of(project, "Exit")
    l2_folded = pos_of(project, "L2")

    # Unfold All straightens the whole relay: the downstream arm moves and
    # the fold mirrors are excluded from the sim (unfold_all is a void action)
    pane.unfold_all()
    qapp.processEvents()
    exit_flat = pos_of(project, "Exit")
    assert max(abs(a - b) for a, b in zip(exit_flat, exit_folded)) > 1.0, \
        "Unfold All did not move the downstream Exit element"
    # unfolded fold rows render as excluded (greyed/italic)
    fm1 = pane.item_for_element("FM1")
    assert fm1.checkState(COL_FOLD) == Qt.Unchecked
    props = project.body(
        project.train().primary_body_name("FM1"))["properties"]
    assert props.get("miewb_exclude", {}).get("value") is True

    # Refold All restores the folded geometry bit-for-bit
    pane.refold_all()
    qapp.processEvents()
    exit_refold = pos_of(project, "Exit")
    l2_refold = pos_of(project, "L2")
    assert max(abs(a - b) for a, b in zip(exit_refold, exit_folded)) < 1e-6, \
        "Refold All did not restore Exit: %s vs %s" % (exit_refold, exit_folded)
    assert max(abs(a - b) for a, b in zip(l2_refold, l2_folded)) < 1e-6
    assert pane.item_for_element("FM1").checkState(COL_FOLD) == Qt.Checked


def test_periscope_variables_and_summary(mw, qapp):
    project = open_demo(mw, "folded_periscope")
    vp = mw.variables_pane
    assert vp is not None
    assert vp.item_for("arm", COL_NAME) is not None, \
        "the 'arm' variable should be listed"
    assert vp.item_for("arm", COL_VALUE).data(Qt.EditRole)

    # the afocal relay must NOT report a telephoto-like finite EFL: the
    # summary is either hidden, reads 'afocal', or shows a very large EFL
    # (near-afocal). Whatever it is gets recorded as a shakedown finding.
    pane = mw.train_editor
    text = pane.summary.text() if pane.summary.isVisibleTo(pane) else ""
    efl = summary_efl(text)
    afocal_ok = (not pane.summary.isVisibleTo(pane)) or ("afocal" in text) \
        or (efl is not None and abs(efl) > 1000.0)
    assert afocal_ok, \
        "periscope summary reads a small finite EFL %s (expected afocal-ish): %r" \
        % (efl, text)
