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
