"""MainWindow smoke tests: dock/host layout, menu presence, and stage-chip
color reaction to a simulated progress event."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtWidgets import QDockWidget, QWidget  # noqa: E402

from mieworkbench.core.settings import Settings  # noqa: E402
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
    assert len(docks) == 11   # +3: train editor, variables, compare


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
    results.analysis_metrics.setRowCount(4)
    results.sources.setRowCount(5)
    results.audit.setText("energy closure: OK")
    results.pv_btn.setEnabled(True)
    img = tmp_path / "det.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")   # unloadable pixmap is fine
    results.galleries["images"].show_images([str(img)])
    assert results.galleries["images"]._grid.count() == 1
    results.galleries["analysis"].show_images([str(img)])
    assert results.galleries["analysis"]._grid.count() == 1

    results.clear_case()
    assert results.case_dir is None
    assert results.title.text() == "No results loaded"
    assert results.summary.rowCount() == 0
    assert results.power.rowCount() == 0
    assert results.analysis_metrics.rowCount() == 0
    assert results.sources.rowCount() == 0
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

# ---------------------------------------------------------------------------
# tracer-bead animation shell (toolbar, overlay wiring, settings tab)
# ---------------------------------------------------------------------------
def _restore_key(window, key, saved):
    if saved is None:
        window.settings._qs.remove(key)
    else:
        window.settings._qs.setValue(key, saved)


# MainWindow.__init__ -> _init_animation reads these four persisted keys
# to seed the animation toolbar (speed/bead size/fps/enabled). A real
# session's saved values would otherwise leak into the factory-default
# assertions below, so they must be snapshotted and CLEARED before the
# window is even constructed, then restored once the window (and its own
# `settings` wrapper over the same QSettings store) exists.
_ANIM_SETTINGS_KEYS = ("anim_speed_mm_s", "anim_bead_size", "anim_fps",
                      "anim_enabled")


def _clear_anim_settings():
    """Snapshot + remove the four anim_* keys via a throwaway Settings()
    (same QSettings("CurtisAnalytical", "MieWorkbench") store the app
    uses) so a fresh MainWindow() sees factory defaults. Returns the
    snapshot for restoration after the window is built."""
    scratch = Settings()
    saved = {k: scratch._qs.value(k, None) for k in _ANIM_SETTINGS_KEYS}
    for k in _ANIM_SETTINGS_KEYS:
        scratch._qs.remove(k)
    scratch._qs.sync()
    return saved


def test_animation_toolbar_exists_and_gates_on_overlay(qtbot, tmp_path):
    from mieworkbench.tests.vtk_test_support import write_simple_vtp
    saved = _clear_anim_settings()
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        toolbars = [tb.objectName() for tb in window.findChildren(
            type(window.addToolBar("x")))]
        assert "animation_toolbar" in toolbars
        for name in ("anim_play_action", "anim_pause_action",
                     "anim_stop_action", "anim_step_action",
                     "anim_size_spin", "anim_speed_spin",
                     "anim_fps_combo", "anim_readout",
                     "anim_enable_action"):
            assert getattr(window, name) is not None, name

        # defaults per spec: 2 mm/s at 15 fps, transport gated off
        assert window.anim_speed_spin.value() == 2.0
        assert window.anim_fps_combo.currentText() == "15"
        assert not window.anim_play_action.isEnabled()

        window.anim_enable_action.setChecked(True)   # menu+toolbar action
        assert window.anim_controller.enabled
        assert not window.anim_play_action.isEnabled()   # still no rays

        # a timed overlay arms the transport...
        timed = tmp_path / "rays.vtp"
        write_simple_vtp(timed, with_rgb=True, rel_power=[1.0, 0.5],
                         opl=[(0.0, 0.02), (0.02, 0.05)])
        window.scene3d.load_rays_vtp(str(timed))
        assert window.anim_controller.has_segments()
        assert window.anim_play_action.isEnabled()
        assert "t = " in window.anim_readout.text()

        # ...a legacy overlay disarms it and says why
        legacy = tmp_path / "legacy.vtp"
        write_simple_vtp(legacy, with_rgb=True)
        window.scene3d.load_rays_vtp(str(legacy))
        assert not window.anim_controller.has_segments()
        assert not window.anim_play_action.isEnabled()
        assert "timing" in window.statusBar().currentMessage()

        # stale-grey also parks the animation
        window.scene3d.load_rays_vtp(str(timed))
        assert window.anim_play_action.isEnabled()
        window.scene3d.set_rays_stale(True)
        assert not window.anim_controller.has_segments()
    finally:
        for k in _ANIM_SETTINGS_KEYS:
            _restore_key(window, k, saved[k])
        window.settings._qs.sync()


def test_animation_step_updates_readout_and_beads(qtbot, tmp_path):
    from mieworkbench.tests.vtk_test_support import write_simple_vtp
    saved = _clear_anim_settings()
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        timed = tmp_path / "rays.vtp"
        write_simple_vtp(timed, with_rgb=True, rel_power=[1.0],
                         opl=[(0.0, 0.5)])
        window.scene3d.load_rays_vtp(str(timed))
        window.anim_enable_action.setChecked(True)
        assert window.scene3d.view.beads.actor.GetVisibility()
        # at t=0 the bead sits at the source point of the segment
        assert window.scene3d.view.beads.bead_count() == 1

        window.anim_step_action.trigger()
        assert "path = 0.13 mm" in window.anim_readout.text()  # 2/15 mm

        window.anim_stop_action.trigger()
        assert "path = 0.00 mm" in window.anim_readout.text()

        window.anim_enable_action.setChecked(False)
        assert not window.scene3d.view.beads.actor.GetVisibility()
    finally:
        for k in _ANIM_SETTINGS_KEYS:
            _restore_key(window, k, saved[k])
        window.settings._qs.sync()


def test_settings_defaults_tab_round_trip(qtbot):
    from mieworkbench.core.settings import SettingsDialog
    window = MainWindow()
    qtbot.addWidget(window)
    keys = ("ray_dimming_mode", "ray_dimming_floor", "anim_enabled",
            "anim_bead_size", "anim_speed_mm_s", "anim_fps",
            "anim_ray_cap")
    saved = {k: window.settings._qs.value(k, None) for k in keys}
    try:
        dialog = SettingsDialog(window.settings, window)
        qtbot.addWidget(dialog)
        dialog.dim_mode_combo.setCurrentIndex(1)      # Linear
        dialog.dim_floor_spin.setValue(7.5)
        dialog.anim_speed_spin.setValue(4.0)
        dialog.anim_fps_spin.setValue(30)
        dialog.anim_cap_spin.setValue(50)
        dialog._on_accept()                            # dialog-free path

        assert window.settings.get("ray_dimming_mode") == "linear"
        assert float(window.settings.get("ray_dimming_floor")) == 7.5
        assert float(window.settings.get("anim_speed_mm_s")) == 4.0
        # pushed live into the open session, both editors synced
        assert window._ray_dim_mode == "linear"
        assert window.ray_dim_combo.currentIndex() == 1
        assert window.anim_controller.speed_mm_s == 4.0
        assert window.anim_controller.fps == 30
        assert window.anim_controller.ray_cap == 50
        assert window.anim_speed_spin.value() == 4.0
        assert window.anim_fps_combo.currentText() == "30"
    finally:
        for k, v in saved.items():
            _restore_key(window, k, v)
        window.settings._qs.sync()
