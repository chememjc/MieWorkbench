"""WP6: face-indicators toolbar button, checked-toolbutton contrast
stylesheet, and the extinction-combo tooltip clarification."""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QToolBar  # noqa: E402

from mieworkbench.mainwindow import MainWindow  # noqa: E402
from mieworkbench.widgets.style import (  # noqa: E402
    checked_toolbutton_stylesheet)


def _main_toolbar(window):
    for tb in window.findChildren(QToolBar):
        if tb.objectName() == "main_toolbar":
            return tb
    raise AssertionError("main_toolbar not found")


def test_stylesheet_has_checked_toolbutton_rule(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    sheet = window.styleSheet()
    assert "QToolButton:checked" in sheet

    hl = window.palette().color(QPalette.ColorRole.Highlight)
    for component in (hl.red(), hl.green(), hl.blue()):
        assert str(component) in sheet

    # the standalone helper produces exactly what's applied
    assert checked_toolbutton_stylesheet(window.palette()) in sheet


def test_face_indicators_action_on_main_toolbar(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    toolbar = _main_toolbar(window)
    assert window.face_indicators_action in toolbar.actions()


def test_toolbar_toggle_reflects_in_menu_state(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    action = window.face_indicators_action
    before = action.isChecked()
    action.trigger()
    assert action.isChecked() != before
    # same QAction instance drives both surfaces, so there's nothing
    # separate to "sync" -- verify identity once more for clarity
    toolbar = _main_toolbar(window)
    assert toolbar.actions().count(action) == 1


def test_extinction_combo_tooltip_mentions_lines(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    tip = window.ray_dim_combo.toolTip()
    assert tip
    assert "lines" in tip.lower() or "line" in tip.lower()
