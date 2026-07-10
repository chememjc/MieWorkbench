"""MainWindow optical-train wiring tests (offscreen, scripted worker).

Covers the object-placer round's mainwindow integration: the three train
docks, the train-indicator refresh (exclusion ghosting + chain links +
outliner badges), File -> Export FCStd, the sweep run flow (Variables pane
override, pre-sweep confirmation, manifest, Compare handoff).

These follow test_mainwindow.py's construct-and-drive idiom but inject the
train-aware TrainFakeWorker from train_test_support (assigning
window.project._fc and priming structure exactly like make_scene) so no
real FreeCAD worker is needed. The fake is subclassed HERE to add a
save_copy op for the export test (train_test_support is owned by another
agent this round and must not be edited).
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtWidgets import QDockWidget, QWidget  # noqa: E402

from mieworkbench.core.geomcache import GeomCache  # noqa: E402
from mieworkbench.mainwindow import MainWindow  # noqa: E402
from mieworkbench.tests.train_test_support import (  # noqa: E402
    TrainFakeWorker, make_scene)


class SaveCapableFake(TrainFakeWorker):
    """TrainFakeWorker + a save_copy op (Export FCStd path)."""

    def __init__(self, structure):
        super().__init__(structure)
        self.saved = []

    def request(self, op, params=None, timeout=None):
        if op == "save_copy":
            params = params or {}
            self.saved.append(dict(params))
            with open(params["path"], "w") as fh:
                fh.write("fake fcstd")
            return {"path": params["path"]}
        return super().request(op, params, timeout)


def _prime(window):
    """Make window.project a live train scene (SRC->L1->L2 + FM + DET),
    backed by a save-capable scripted worker. Mirrors make_scene's priming
    but on the MainWindow's own Project (so its dock/menu wiring stays)."""
    scene, _ = make_scene()                 # standalone Project + fake
    fake = SaveCapableFake(scene._fc.structure)
    p = window.project
    p._fc = fake
    p._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_mw_train_"))
    p.doc = scene.doc
    p.fcstd_path = scene.fcstd_path
    p.structure = scene.structure
    p.body_states = scene.body_states
    window.model_path = scene.fcstd_path
    fake.ops.clear()
    p.sceneLoaded.emit()
    return fake


# ---------------------------------------------------------------------------
# docks
# ---------------------------------------------------------------------------
def test_train_docks_exist_with_object_names_and_toggles(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    names = ["train_editor_dock", "compare_dock"]
    if window.variables_dock is not None:
        names.append("variables_dock")

    view_titles = set()
    for menu_action in window.menuBar().actions():
        if menu_action.text().replace("&", "") == "View":
            for act in menu_action.menu().actions():
                view_titles.add(act.text())

    for name in names:
        dock = window.findChild(QDockWidget, name)
        assert dock is not None, name
        toggle = dock.toggleViewAction()
        assert toggle is not None
        # every train dock's toggle is offered in the View menu
        assert toggle.text() in view_titles, dock.windowTitle()

    # host widgets carry their object names too
    assert window.findChild(QWidget, "train_editor_host") is not None
    assert window.findChild(QWidget, "compare_host") is not None


# ---------------------------------------------------------------------------
# train indicators: exclusion
# ---------------------------------------------------------------------------
def test_unfold_pushes_exclusion_to_view(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    _prime(window)

    p = window.project
    p.set_chain("L1", {"ref": "SRC", "distance": "10"})
    p.set_chain("FM", {"ref": "L1", "distance": "20", "fold": True,
                       "folded": True, "tilt_ry": "-45"})
    p.set_chain("DET", {"ref": "FM", "port": "reflect", "distance": "15"})

    window._refresh_train_indicators()
    assert "FM" not in window.scene3d.view._excluded_bodies

    p.set_fold_state("FM", False)          # unfold: ghost + sim-exclude FM
    window._refresh_train_indicators()
    assert "FM" in window.scene3d.view._excluded_bodies

    p.set_fold_state("FM", True)           # refold clears the exclusion
    window._refresh_train_indicators()
    assert "FM" not in window.scene3d.view._excluded_bodies


# ---------------------------------------------------------------------------
# train indicators: chain links
# ---------------------------------------------------------------------------
def test_chain_links_computed_for_chained_pair(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    _prime(window)

    p = window.project
    p.set_chain("L1", {"ref": "SRC", "distance": "10"})
    p.set_chain("L2", {"ref": "L1", "distance": "20"})

    seen = []
    window.scene3d.view.set_chain_links = seen.append
    window._refresh_train_indicators()

    assert seen, "set_chain_links was never called"
    links = seen[-1]
    assert len(links) >= 1
    for link in links:
        assert set(link) >= {"from", "to", "kind"}
        for coord in list(link["from"]) + list(link["to"]):
            assert math.isfinite(coord)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def test_export_fcstd_writes_a_file(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    fake = _prime(window)

    out = tmp_path / "exported.FCStd"
    window._export_fcstd(str(out))

    assert out.exists()
    assert fake.saved
    assert fake.saved[-1]["path"] == str(out)


# ---------------------------------------------------------------------------
# run flow with a sweep
# ---------------------------------------------------------------------------
def test_run_flow_uses_variables_sweep_and_writes_manifest(
        qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    if window.variables_pane is None:
        import pytest
        pytest.skip("VariablesPane not available in this build")
    _prime(window)

    # make the run land inside tmp (results_root = <workspace>/results)
    window.workspace = str(tmp_path)

    # the miewb_vars sheet from make_scene has gap enabled (min10/max40/n3)
    vp = window.variables_pane
    vp.refresh()
    vp.sweep_mode = "zip"        # non-default so it also reaches the argv
    assert vp.has_enabled_sweep()

    # bypass the save+validate gate (the scripted worker has no save op)
    monkeypatch.setattr(window, "_preflight", lambda: [])

    confirmed = []
    monkeypatch.setattr(window, "_confirm_sweep",
                        lambda summary: confirmed.append(summary) or True)

    launched = {}

    def fake_start(model_path, extra_args=None, steps=None, extra_env=None):
        launched["model"] = model_path
        launched["args"] = list(extra_args or [])
        launched["env"] = dict(extra_env or {})
        return True

    monkeypatch.setattr(window.runner, "start", fake_start)

    assert window._run_pipeline() is True

    # the pane's sweep superseded the config matrix var/min/max/n
    config = window._merged_run_config()
    assert config["var"] == ["miewb_vars.gap"]
    assert config["sweep_mode"] == "zip"

    args = launched["args"]
    assert "--var" in args and "miewb_vars.gap" in args
    assert "--min" in args and "--max" in args and "--n" in args
    assert "--sweep-mode" in args and "zip" in args

    # the pre-sweep confirmation was consulted with a run count
    assert confirmed and confirmed[-1]["runs"] == 4

    # the sweep manifest was written under the run's results root (tmp)
    manifest = window._pending_manifest
    assert manifest is not None
    assert os.path.exists(manifest)
    assert str(tmp_path) in manifest


def test_run_flow_from_config_matrix_var_fields(qtbot, tmp_path,
                                                monkeypatch):
    """Even without the Variables pane, raw config-matrix var/min/max/n
    fields trigger the same confirm-then-manifest multi-variant path."""
    window = MainWindow()
    qtbot.addWidget(window)
    _prime(window)
    window.workspace = str(tmp_path)

    # drive the sweep straight through the config matrix instead of the pane
    if window.variables_pane is not None:
        monkeypatch.setattr(window.variables_pane, "has_enabled_sweep",
                            lambda: False)
    window.config_matrix.set_values(
        {"var": ["ct"], "min": [1.0], "max": [3.0], "n": [3]})

    monkeypatch.setattr(window, "_preflight", lambda: [])
    confirmed = []
    monkeypatch.setattr(window, "_confirm_sweep",
                        lambda summary: confirmed.append(summary) or True)
    monkeypatch.setattr(window.runner, "start",
                        lambda *a, **k: True)

    assert window._run_pipeline() is True
    assert confirmed and confirmed[-1]["runs"] == 4
    assert window._pending_manifest and os.path.exists(
        window._pending_manifest)


# ---------------------------------------------------------------------------
# finished(0) -> compare
# ---------------------------------------------------------------------------
def test_finished_run_hands_manifest_to_compare(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)

    window._pending_manifest = "/tmp/some-sweep.manifest.json"
    captured = {}

    def fake_run_compare(manifest_path=None, **kw):
        captured["manifest"] = manifest_path
        return True

    monkeypatch.setattr(window.compare_pane, "run_compare", fake_run_compare)

    window._maybe_run_compare(0)

    assert captured.get("manifest") == "/tmp/some-sweep.manifest.json"
    # the manifest is consumed (no double compare on the next finish)
    assert window._pending_manifest is None


def test_failed_run_does_not_trigger_compare(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window._pending_manifest = "/tmp/some-sweep.manifest.json"
    calls = []
    monkeypatch.setattr(window.compare_pane, "run_compare",
                        lambda **k: calls.append(k) or True)
    window._maybe_run_compare(1)          # nonzero exit: no compare
    assert not calls
