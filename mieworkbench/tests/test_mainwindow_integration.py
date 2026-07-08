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


# ---------------------------------------------------------------------------
# file lifecycle: Close / Revert to Saved / session reset on open
# ---------------------------------------------------------------------------
def _example_copy(tmp_path, name="ex.FCStd"):
    import shutil
    dst = tmp_path / name
    shutil.copy2(os.path.join(REPO, "example.FCStd"), dst)
    return str(dst)


def test_close_model_clears_session_and_rays(window, qtbot, tmp_path):
    from mieworkbench.tests.vtk_test_support import write_simple_vtp
    window.open_model(_example_copy(tmp_path))
    assert window.project.is_open()
    assert window.close_action.isEnabled()
    assert window.revert_action.isEnabled()

    rays = tmp_path / "rays.vtp"
    write_simple_vtp(rays, with_rgb=True)
    window.scene3d.load_rays_vtp(str(rays))
    assert window.scene3d.view._rays_actor is not None

    window._on_close_model()
    assert not window.project.is_open()
    assert window.model_path is None
    assert window.opened_path is None
    assert window.scene3d.view._rays_actor is None
    assert len(window.scene3d.view._body_actors) == 0
    assert not window.save_action.isEnabled()
    assert not window.close_action.isEnabled()
    assert not window.revert_action.isEnabled()


def test_revert_discards_unsaved_changes(window, qtbot, tmp_path):
    window.open_model(_example_copy(tmp_path))
    window.project.set_property("Body", "absorbance", 0.5)
    assert window.project.is_dirty()
    assert "absorbance" in window.project.body("Body")["properties"]

    window._on_revert()      # hidden window: no confirmation modal
    assert window.project.is_open()
    assert not window.project.is_dirty()
    assert "absorbance" not in window.project.body("Body")["properties"]
    assert len(window.project.body_names()) == 7


def test_open_second_model_clears_old_ray_overlay(window, qtbot, tmp_path):
    """The reported bug: rays from the previous session survived into a
    newly opened model and wrecked the render — and (round two) the
    Results pane, run indicators and run config leaked across the open."""
    from mieworkbench.tests.vtk_test_support import write_simple_vtp
    window.open_model(_example_copy(tmp_path, "first.FCStd"))
    rays = tmp_path / "rays.vtp"
    write_simple_vtp(rays, with_rgb=True)
    window.scene3d.load_rays_vtp(str(rays))
    window.inspector.set_body(window.project, "Body")
    window.inspector.load_rays_vtp(str(rays))
    window.results.case_dir = str(tmp_path)
    window.results.summary.setRowCount(2)
    window.results.pv_btn.setEnabled(True)
    window.stage_chips["trace"].setStyleSheet(
        window._chip_style("#22c55e"))
    window.config_matrix.widgets["seeds"].setValue(5)

    window.open_model(_example_copy(tmp_path, "second.FCStd"))
    assert window.project.is_open()
    assert window.scene3d.view._rays_actor is None
    assert window.inspector.view._rays_actor is None
    assert window.results.case_dir is None
    assert window.results.summary.rowCount() == 0
    assert not window.results.pv_btn.isEnabled()
    assert "#22c55e" not in window.stage_chips["trace"].styleSheet()
    assert window.config_matrix.values() == {}
    assert len(window.project.body_names()) == 7
    # the new scene renders its bodies (28 face actors, per example.FCStd)
    assert len(window.scene3d.view._actor_face_map) == 28


def test_unsaved_prompt_skipped_when_hidden(window, qtbot, tmp_path):
    window.open_model(_example_copy(tmp_path))
    window.project.set_property("Body", "absorbance", 0.25)
    assert window.project.is_dirty()
    # hidden windows never block on the modal: treated as discard
    assert window._maybe_save_changes("testing") is True
