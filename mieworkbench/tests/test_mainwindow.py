"""MainWindow smoke tests: dock/host layout, menu presence, and stage-chip
color reaction to a simulated progress event."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtWidgets import QDockWidget, QWidget  # noqa: E402

from mieworkbench.mainwindow import MainWindow  # noqa: E402

HOST_NAMES = (
    "scene3d_host", "outliner_host", "inspector_host",
    "element_editor_host", "transform_host", "library_host",
    "results_host", "problems_host",
)


def test_docks_and_hosts_exist(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    for name in HOST_NAMES:
        assert window.findChild(QWidget, name) is not None, name

    # outliner, inspector, element editor, transform, library, console,
    # results, problems
    docks = window.findChildren(QDockWidget)
    assert len(docks) == 8


def test_menu_actions_exist(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    menubar = window.menuBar()
    titles = [action.text() for action in menubar.actions()]
    assert any("File" in t for t in titles)
    assert any("Simulation" in t for t in titles)
    assert any("View" in t for t in titles)
    assert any("Help" in t for t in titles)

    assert window.run_action is not None
    assert window.estimate_action is not None
    assert window.dry_run_action is not None
    assert window.stop_action is not None


def test_progress_updates_stage_chip(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_progress(
        {"stage": "trace", "frac": 0.5, "msg": "seed 1/3",
         "status": "running"})
    assert "#3b82f6" in window.stage_chips["trace"].styleSheet()

    window._on_progress(
        {"stage": "trace", "frac": 1.0, "msg": "done",
         "status": "completed"})
    assert "#22c55e" in window.stage_chips["trace"].styleSheet()

    window._on_progress(
        {"stage": "post", "frac": None, "msg": "failed",
         "status": "failed"})
    assert "#ef4444" in window.stage_chips["post"].styleSheet()


def test_pipeline_stage_drives_overall_progress_bar(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_progress(
        {"stage": "pipeline", "frac": 0.5, "msg": "example/trace",
         "status": "running"})
    assert window.progress_bar.value() == 50


def test_run_requires_open_model(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.model_path is None

    warned = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))

    window._on_dry_run()
    assert warned
    assert not window.runner.is_running()


# ---------------------------------------------------------------------------
# file-lifecycle actions (offscreen, no document needed)
# ---------------------------------------------------------------------------
def test_close_and_revert_actions_exist_and_start_disabled(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert not window.close_action.isEnabled()
    assert not window.revert_action.isEnabled()
    # triggering them with nothing open must be a harmless no-op
    window._on_close_model()
    window._on_revert()
    assert not window.project.is_open()


def test_maybe_save_changes_true_when_nothing_open(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._maybe_save_changes("testing") is True


def test_reset_session_views_safe_with_no_project(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._reset_session_views()      # must not raise
    assert window.scene3d.view._rays_actor is None
    assert window.results.case_dir is None


def test_results_clear_case_resets_state(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.results.case_dir = str(tmp_path)
    window.results.title.setText("something")
    window.results.clear_case()
    assert window.results.case_dir is None
    assert window.results.title.text() == "No results loaded"

def test_ray_dimming_menu_exists_and_persists(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    saved_mode = window.settings._qs.value("ray_dimming_mode", None)
    saved_floor = window.settings._qs.value("ray_dimming_floor", None)
    try:
        # the View > Ray Dimming submenu with an exclusive mode group
        assert window.ray_dimming_menu is not None
        actions = (window.ray_dim_off_action,
                   window.ray_dim_linear_action,
                   window.ray_dim_sqrt_action)
        for act in actions:
            assert act.isCheckable()
        assert sum(act.isChecked() for act in actions) == 1
        assert window.ray_dim_floor_action is not None

        # selecting a mode fans out to BOTH 3D views and persists
        window.ray_dim_linear_action.trigger()
        assert window.scene3d.view._dim_mode == "linear"
        assert window.inspector.view._dim_mode == "linear"
        assert window.settings.get("ray_dimming_mode") == "linear"

        window.ray_dim_sqrt_action.trigger()
        assert window.scene3d.view._dim_mode == "sqrt"
        assert window.settings.get("ray_dimming_mode") == "sqrt"

        # floor via the dialog-free setter (modal dialogs hang offscreen)
        window._set_ray_dimming_floor(12.5)
        assert window.scene3d.view._dim_floor == 12.5
        assert window.inspector.view._dim_floor == 12.5
        assert float(window.settings.get("ray_dimming_floor")) == 12.5

        window.ray_dim_off_action.trigger()
        assert window.scene3d.view._dim_mode == "off"
    finally:
        for key, val in (("ray_dimming_mode", saved_mode),
                         ("ray_dimming_floor", saved_floor)):
            if val is None:
                window.settings._qs.remove(key)
            else:
                window.settings._qs.setValue(key, val)
        window.settings._qs.sync()
