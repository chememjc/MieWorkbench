"""Save/dirty-state flow tests (bugs B/C/D of the lowhanging round):

B - Save As must retarget the session (opened_path/miewb_path) so the
    title bar and later saves follow the NEW file.
C - opening a file must never write it; the open-time recompute
    divergence reported by the worker counts as unsaved changes.
D - running a simulation with unsaved changes prompts Save&Run/Cancel
    (dialog-free hook offscreen) instead of silently saving.
"""

import hashlib
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.project import Project  # noqa: E402
from mieworkbench.mainwindow import (  # noqa: E402
    MainWindow, QFileDialog, miewb_tool)

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LENS_DCX = os.path.join(REPO, "basemodels", "lens_dcx.FCStd")


def _stub_project(window, monkeypatch, dirty=False):
    """Replace the worker-backed persistence calls with recorders; the
    rest of the Project object stays real."""
    calls = []
    state = {"dirty": dirty}
    monkeypatch.setattr(window.project, "is_open", lambda: True)
    monkeypatch.setattr(window.project, "is_dirty",
                        lambda: state["dirty"])

    def save():
        calls.append("save")
        state["dirty"] = False

    def save_as(path):
        calls.append(("save_as", path))
        state["dirty"] = False

    monkeypatch.setattr(window.project, "save", save)
    monkeypatch.setattr(window.project, "save_as", save_as)
    return calls, state


# ---------------------------------------------------------------------------
# B: Save As retargets the session
# ---------------------------------------------------------------------------
def test_save_as_fcstd_retargets_session_and_title(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    calls, _ = _stub_project(window, monkeypatch)
    window.model_path = "/somewhere/old.FCStd"
    window.opened_path = "/somewhere/old.FCStd"
    window.miesim_out = "/somewhere/old.MieSim"

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: ("/elsewhere/new.FCStd",
                                      "FreeCAD model (*.FCStd)")))
    window._on_save_as()

    assert ("save_as", "/elsewhere/new.FCStd") in calls
    assert window.model_path == "/elsewhere/new.FCStd"
    assert window.opened_path == "/elsewhere/new.FCStd"
    assert window.miewb_path is None
    assert window.miesim_out is None
    assert "new.FCStd" in window.windowTitle()
    assert "old.FCStd" not in window.windowTitle()


def test_save_as_miewb_retargets_session_and_title(qtbot, monkeypatch,
                                                   tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    calls, _ = _stub_project(window, monkeypatch)
    ws_model = str(tmp_path / "model.FCStd")
    window.model_path = ws_model
    window.workspace = str(tmp_path)
    window.miewb_path = "/somewhere/old.MieWB"
    window.opened_path = "/somewhere/old.MieWB"
    window.miesim_out = "/somewhere/old.MieSim"

    packed = []
    monkeypatch.setattr(miewb_tool, "pack_miewb",
                        lambda model, path, **k: packed.append(path))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: ("/elsewhere/new.MieWB",
                                      "Workbench archive (*.MieWB)")))
    window._on_save_as()

    assert "save" in calls
    assert packed == ["/elsewhere/new.MieWB"]
    assert window.miewb_path == "/elsewhere/new.MieWB"
    assert window.opened_path == "/elsewhere/new.MieWB"
    assert window.miesim_out is None
    # workspace is the live unpacked session and must survive
    assert window.workspace == str(tmp_path)
    assert "new.MieWB" in window.windowTitle()


# ---------------------------------------------------------------------------
# C: recompute divergence is unsaved state (Project-side semantics)
# ---------------------------------------------------------------------------
def test_recompute_divergence_counts_as_dirty(qtbot):
    p = Project()
    emitted = []
    p.dirtyChanged.connect(emitted.append)

    assert not p.is_dirty()
    p._set_recompute_diverged(True)
    assert p.is_dirty()
    assert emitted == [True]

    # undoing back to a clean stack must NOT mask the divergence
    p._sync_dirty_from_stack()
    assert p.is_dirty()

    # edit-dirty on top changes nothing effective (no duplicate signal),
    # and clearing the edit flag still leaves the divergence dirty
    p._set_dirty(True)
    p._set_dirty(False)
    assert p.is_dirty()
    assert emitted == [True]

    p._set_recompute_diverged(False)
    assert not p.is_dirty()
    assert emitted == [True, False]


# ---------------------------------------------------------------------------
# D: Save & Run gate
# ---------------------------------------------------------------------------
def test_run_gate_cancel_blocks_without_saving(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "/somewhere/model.FCStd"
    calls, _ = _stub_project(window, monkeypatch, dirty=True)
    window._save_before_run_hook = False   # user picks Cancel

    assert window._confirm_save_before_run() is False
    assert calls == []
    # the whole preflight aborts before validation
    assert window._preflight() is None
    assert calls == []
    assert not window.runner.is_running()


def test_run_gate_saves_before_launch(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "/somewhere/model.FCStd"
    calls, _ = _stub_project(window, monkeypatch, dirty=True)
    monkeypatch.setattr(window.problems, "run_checks", lambda: [])

    args = window._preflight()
    assert calls == ["save"]         # saved exactly once, before launch
    assert args is not None          # then the run may proceed


def test_run_gate_noop_when_clean(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.model_path = "/somewhere/model.FCStd"
    calls, _ = _stub_project(window, monkeypatch, dirty=False)
    monkeypatch.setattr(window.problems, "run_checks", lambda: [])

    assert window._preflight() is not None
    assert calls == []               # nothing to save, nothing saved


# ---------------------------------------------------------------------------
# C (integration, real FreeCAD): opening never writes the file
# ---------------------------------------------------------------------------
@pytest.mark.freecad
def test_open_document_reports_divergence_and_never_writes(tmp_path):
    from mieworkbench.core.fcclient import FcClient

    model = str(tmp_path / "lens_dcx.FCStd")
    shutil.copy2(LENS_DCX, model)
    before = hashlib.sha256(open(model, "rb").read()).hexdigest()
    mtime = os.path.getmtime(model)

    with FcClient() as fc:
        st = fc.open_document(model)
        assert "recompute_changed" in st
        assert isinstance(st["recompute_changed"], list)
        fc.close(st["doc"])

    assert hashlib.sha256(open(model, "rb").read()).hexdigest() == before
    assert os.path.getmtime(model) == mtime
