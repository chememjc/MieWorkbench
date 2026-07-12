"""ZoomPairDialog (mieworkbench/panes/wizard_dialog.py) — offscreen GUI
wiring test for the zoom-pair calculator (future.md (a2)). The math
itself is oracle-tested in tests/test_wizards.py against the shipped
telephoto_zoom demo; this only checks the dialog reads its fields, calls
core.wizards.solve_zoom_pair, and populates the readouts/expression
lines without ever showing an unguarded modal (QT_QPA_PLATFORM=offscreen
safe: no .exec() call anywhere in this file)."""
import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core import wizards                       # noqa: E402
from mieworkbench.panes.wizard_dialog import ZoomPairDialog  # noqa: E402


def test_zoom_pair_dialog_computes_and_fills_readouts(qtbot):
    dlg = ZoomPairDialog()
    qtbot.addWidget(dlg)
    dlg.f1_edit.setText("100")
    dlg.f2_edit.setText("-50")
    dlg.z_edit.setText("30")
    dlg._compute()

    expected = wizards.solve_zoom_pair(100.0, -50.0, z_mm=30.0)
    assert dlg.result() is not None
    assert dlg.result()["bfl_mm"] == expected["bfl_mm"]
    assert dlg.bfl_expr_edit.text() == expected["bfl_expr"]
    assert dlg.efl_expr_edit.text() == expected["efl_expr"]
    assert "%.4g" % expected["bfl_mm"] in dlg.result_label.text()
    assert "%.4g" % expected["track_mm"] in dlg.result_label.text()


def test_zoom_pair_dialog_bad_input_shows_message_not_crash():
    dlg = ZoomPairDialog()
    dlg.f1_edit.setText("0")     # solve_zoom_pair rejects f1/f2 == 0
    dlg.f2_edit.setText("-50")
    dlg.z_edit.setText("30")
    dlg._compute()
    assert dlg.result() is None
    assert dlg.result_label.text()   # some explanatory text, not blank
    assert dlg.bfl_expr_edit.text() == ""   # untouched, no stale value
