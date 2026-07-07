"""Integrated MainWindow tests (freecad-marked: real worker session)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

from mieworkbench.mainwindow import MainWindow  # noqa: E402

pytestmark = pytest.mark.freecad


@pytest.fixture()
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.project.shutdown()


def test_open_fcstd_populates_everything(window, qtbot):
    window.open_model(os.path.join(REPO, "example.FCStd"))
    assert window.project.is_open()
    assert window.save_action.isEnabled()
    assert len(window.project.body_names()) == 7
    # selection wiring: scene selection reaches inspector+transform panel
    window._on_scene_selection("Body", set())
    assert "Lens" in window.transform_panel.target.text()
    # validation on the real example: no blocking errors
    findings = window.problems.run_checks()
    from mieworkbench.core import validation
    assert not validation.has_errors(findings)


def test_open_miewb_workspace_flow(window, qtbot, tmp_path):
    import miewb_tool
    wb = tmp_path / "ex.MieWB"
    miewb_tool.pack_miewb(os.path.join(REPO, "example.FCStd"), wb,
                          simparams={"preset": "quick", "tag": "gui"})
    window.open_model(str(wb))
    assert window.workspace is not None
    assert window.miewb_path == str(wb)
    assert window.project.is_open()
    # simparams landed in the config matrix
    values = window.config_matrix.values()
    assert values.get("tag") == "gui"
    # runs would land inside the workspace
    case_dir = window._current_case_dir()
    assert case_dir.startswith(window.workspace)
    env = window._run_env()
    assert env["MIEWB_RESULTS_DIR"].startswith(window.workspace)


def test_add_element_via_wizard_headless(window, qtbot, tmp_path,
                                         monkeypatch):
    """Drive the add-element flow without the modal dialog: emulate the
    dialog result by calling the underlying project ops the handler
    performs."""
    scene = tmp_path / "fresh.FCStd"
    window.project.new_document(str(scene))
    window.model_path = str(scene)
    info = {"kind": "lens_pcx", "label": "Plano-convex lens",
            "path": os.path.join(REPO, "primitives", "lens_pcx.FCStd"),
            "params": {"R_front": {"default": 25.0, "unit": "mm"},
                       "ct": {"default": 5.0, "unit": "mm"},
                       "aperture": {"default": 20.0, "unit": "mm"}}}
    window.project.import_primitive(info["path"], "L1")
    window.project.set_spreadsheet("dim_L1", "R_front", "=30 mm")
    window.project.rebuild_primitive("L1")
    body = window.project.body("L1")
    assert body["properties"]["material"]["value"] == "bk7"


def test_new_miewb_from_scratch(window, qtbot, tmp_path):
    """File->New (.MieWB): fresh workspace + packed archive on disk,
    an empty scene ready for element adds, then copy/paste/delete round
    trip through the real worker."""
    import miewb_tool
    path = str(tmp_path / "fresh.MieWB")
    window._new_miewb(path)
    assert window.project.is_open()
    assert os.path.isfile(path)
    assert miewb_tool.sniff(path) == "MieWB"
    assert window.project.body_names() == []

    # add an element (bypassing the modal wizard), then copy/paste/delete
    window.project.import_primitive(
        os.path.join(REPO, "primitives", "lens_pcx.FCStd"), "L1")
    window.selection.select("L1", ())
    window._on_copy_element()
    window._on_paste_element()
    labels = {b["label"] for b in window.project.structure["bodies"]}
    assert labels == {"L1", "L1_copy"}
    # pasted copy must not sit on top of the original
    from mieworkbench.core.transforms import element_bounds
    b1 = element_bounds(window.project.structure["bodies"],
                        window.project.body_states,
                        window.project.element_bodies("L1"))
    b2 = element_bounds(window.project.structure["bodies"],
                        window.project.body_states,
                        window.project.element_bodies("L1_copy"))
    from mieworkbench.mainwindow import _aabb_overlap
    assert not _aabb_overlap(b1, b2)

    window._on_delete_element("L1_copy")
    labels = {b["label"] for b in window.project.structure["bodies"]}
    assert labels == {"L1"}
    # the whole trio is undoable
    assert window.project.undo()   # un-delete
    assert window.project.undo()   # un-paste
    assert window.project.undo()   # un-add
    assert window.project.body_names() == []


def test_new_fcstd_from_scratch(window, qtbot, tmp_path):
    path = str(tmp_path / "bare.FCStd")
    window._new_fcstd(path)
    assert window.project.is_open()
    assert os.path.isfile(path)
    assert window.workspace is None and window.miewb_path is None
