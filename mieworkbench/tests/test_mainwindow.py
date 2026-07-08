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
    # dirty every per-session indicator the reset must cover
    window.stage_chips["trace"].setStyleSheet(
        window._chip_style("#22c55e"))
    window.progress_bar.setValue(80)
    window.console.append_line("old run output")
    window.config_matrix.widgets["seeds"].setValue(9)
    window._reset_session_views()      # must not raise
    assert window.scene3d.view._rays_actor is None
    assert window.results.case_dir is None
    assert "#22c55e" not in window.stage_chips["trace"].styleSheet()
    assert window.progress_bar.value() == 0
    assert window.config_matrix.values() == {}


def test_results_clear_case_resets_state(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    results = window.results
    results.case_dir = str(tmp_path)
    results.title.setText("something")
    # populate every widget clear_case must wipe (the leak: only the
    # pointer/title used to be reset; the tables/galleries/audit line
    # survived File > Open and showed the previous model's simulation)
    results.summary.setRowCount(2)
    results.power.setRowCount(3)
    results.audit.setText("energy closure: OK")
    results.pv_btn.setEnabled(True)
    img = tmp_path / "det.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")   # unloadable pixmap is fine
    results.galleries["images"].show_images([str(img)])
    assert results.galleries["images"]._grid.count() == 1

    results.clear_case()
    assert results.case_dir is None
    assert results.title.text() == "No results loaded"
    assert results.summary.rowCount() == 0
    assert results.power.rowCount() == 0
    assert results.audit.text() == ""
    assert not results.pv_btn.isEnabled()
    for gallery in results.galleries.values():
        assert gallery._grid.count() == 0
        assert gallery._paths == []

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

def test_extinction_combo_syncs_menu(qtbot):
    """Toolbar combo and View-menu radio group are two editors of the
    same extinction mode: changing either updates the other, with no
    signal recursion."""
    window = MainWindow()
    qtbot.addWidget(window)
    saved_mode = window.settings._qs.value("ray_dimming_mode", None)
    calls = []
    original = window._apply_ray_dimming
    window._apply_ray_dimming = lambda: calls.append(1) or original()
    try:
        # combo -> menu
        window.ray_dim_combo.setCurrentIndex(1)   # Linear
        assert window._ray_dim_mode == "linear"
        assert window.ray_dim_linear_action.isChecked()
        assert len(calls) == 1                    # exactly once, no loop

        # menu -> combo
        window.ray_dim_sqrt_action.trigger()
        assert window._ray_dim_mode == "sqrt"
        assert window.ray_dim_combo.currentIndex() == 2
        assert len(calls) == 2

        window.ray_dim_off_action.trigger()
        assert window.ray_dim_combo.currentIndex() == 0
    finally:
        if saved_mode is None:
            window.settings._qs.remove("ray_dimming_mode")
        else:
            window.settings._qs.setValue("ray_dimming_mode", saved_mode)
        window.settings._qs.sync()


def test_missing_relpower_hint_on_legacy_rays(qtbot, tmp_path):
    """Dimming on + a rays.vtp predating rel_power = silently inert
    coloring; the shell must say so in the status bar."""
    from mieworkbench.tests.vtk_test_support import write_simple_vtp
    window = MainWindow()
    qtbot.addWidget(window)
    saved_mode = window.settings._qs.value("ray_dimming_mode", None)
    try:
        legacy = tmp_path / "rays_legacy.vtp"
        write_simple_vtp(legacy, with_rgb=True)   # rgb but no rel_power
        window.scene3d.view.load_vtp_overlay(legacy)
        window._on_ray_dimming_mode("linear")
        assert "rel_power" in window.statusBar().currentMessage()
        assert window.scene3d.view.ray_dimming_data_missing()

        # with timing-capable rays there is no complaint
        window.statusBar().clearMessage()
        modern = tmp_path / "rays_modern.vtp"
        write_simple_vtp(modern, with_rgb=True, rel_power=[1.0, 0.5])
        window.scene3d.view.load_vtp_overlay(modern)
        window._on_ray_dimming_mode("sqrt")
        assert "rel_power" not in window.statusBar().currentMessage()
        assert not window.scene3d.view.ray_dimming_data_missing()
    finally:
        if saved_mode is None:
            window.settings._qs.remove("ray_dimming_mode")
        else:
            window.settings._qs.setValue("ray_dimming_mode", saved_mode)
        window.settings._qs.sync()
