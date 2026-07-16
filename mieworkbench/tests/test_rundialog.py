"""RunDialog tests (P1 per-run accuracy-vs-time dialog).

Two layers:
  - pure widget tests: construct RunDialog directly with a fake
    common.estimate()-shaped dict, check the expected fields render, and
    that accepting/rejecting drives its accepted/rejected signals (the
    "run callback") correctly -- WITHOUT ever calling .exec() (that would
    block offscreen test runs; see CLAUDE.md's "never show an unguarded
    modal in a code path tests exercise").
  - MainWindow wiring tests: _confirm_run_dialog is isVisible-guarded
    exactly like the existing _confirm_sweep (a hidden window always
    proceeds without touching the modal), honors the in-session
    "don't ask again" skip, and _run_pipeline aborts when it returns
    False (following test_mainwindow_train.py's _confirm_sweep-monkeypatch
    idiom for exercising the launch path itself).
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

import common                                              # noqa: E402
from PySide6.QtCore import Qt                              # noqa: E402
from PySide6.QtWidgets import QDialog, QDialogButtonBox    # noqa: E402

from mieworkbench.mainwindow import MainWindow             # noqa: E402
from mieworkbench.panes.rundialog import RunDialog         # noqa: E402

FAKE_ESTIMATE = {
    "trace_s": 12.0,
    "gather_s": 3.5,
    "total_s": 15.5,
    "accumulator_GB": 0.042,
    "fields_h5_GB": 0.0,
    "gather_pairs": 2.5e9,
    "spr": 1.7,
    "backend": "torch",
}

FAKE_PARAMS = {
    "preset": "quick",
    "rays": 100000,
    "resolution": 512,
    "nlambda": 5,
    "backend": "torch",
    "model_stem": "example",
    "n_coherent_sources": 1,
    "n_detectors": 1,
    "save_fields": False,
    "n_pol_strata": 1,
}


# ---------------------------------------------------------------------------
# pure widget tests
# ---------------------------------------------------------------------------
def _all_label_texts(dialog):
    from PySide6.QtWidgets import QLabel
    return [w.text() for w in dialog.findChildren(QLabel)]


def test_shows_resolved_run_parameters(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(dialog)
    texts = _all_label_texts(dialog)
    assert "quick" in texts
    assert "100000" in texts
    assert "512" in texts
    assert "example" in texts


def test_shows_predicted_wall_time_and_memory(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(dialog)
    texts = _all_label_texts(dialog)
    assert common.fmt_duration(FAKE_ESTIMATE["trace_s"]) in texts
    assert common.fmt_duration(FAKE_ESTIMATE["gather_s"]) in texts
    assert common.fmt_duration(FAKE_ESTIMATE["total_s"]) in texts
    assert "0.042 GB" in texts


def test_calibrated_vs_uncalibrated_label(qtbot):
    calibrated = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(calibrated)
    assert "calibrated" in calibrated._time_box_title(True)
    assert "fallback" not in calibrated._time_box_title(True)

    uncalibrated = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=False)
    qtbot.addWidget(uncalibrated)
    assert "uncalibrated fallback" in uncalibrated._time_box_title(False)

    unknown = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=None)
    qtbot.addWidget(unknown)
    title = unknown._time_box_title(None)
    assert "calibrated" not in title and "fallback" not in title


def test_coherent_projection_shown_labeled_as_projection(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(dialog)
    texts = " ".join(_all_label_texts(dialog))
    assert "2.5e+09" in texts
    assert "1.7" in texts
    assert "Projection only" in texts


def test_coherent_projection_hidden_when_no_coherent_sources(qtbot):
    params = dict(FAKE_PARAMS, n_coherent_sources=0)
    dialog = RunDialog(params, FAKE_ESTIMATE, calibrated=True)
    qtbot.addWidget(dialog)
    texts = " ".join(_all_label_texts(dialog))
    assert "Projection only" not in texts


def test_info_only_mode_has_no_skip_checkbox_or_run_cancel(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True,
                       info_only=True)
    qtbot.addWidget(dialog)
    assert dialog.skip_checkbox is None
    assert dialog.skip_requested() is False
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons.button(QDialogButtonBox.StandardButton.Close) is not None
    assert buttons.button(QDialogButtonBox.StandardButton.Ok) is None


def test_run_mode_accept_triggers_callback_cancel_does_not(qtbot):
    """The "accepting triggers the run callback while cancel does not"
    contract, exercised via QDialog's own accepted/rejected signals --
    the real mechanism MainWindow._confirm_run_dialog reads via
    dialog.exec()'s return code, without ever calling exec() here."""
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True,
                       info_only=False)
    qtbot.addWidget(dialog)
    assert dialog.skip_checkbox is not None

    ran = []
    cancelled = []
    dialog.accepted.connect(lambda: ran.append(True))
    dialog.rejected.connect(lambda: cancelled.append(True))

    buttons = dialog.findChild(QDialogButtonBox)
    ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn.text() == "Run"
    qtbot.mouseClick(ok_btn, Qt.LeftButton)
    assert ran == [True]
    assert cancelled == []
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_run_mode_cancel_does_not_accept(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True,
                       info_only=False)
    qtbot.addWidget(dialog)
    ran = []
    cancelled = []
    dialog.accepted.connect(lambda: ran.append(True))
    dialog.rejected.connect(lambda: cancelled.append(True))

    buttons = dialog.findChild(QDialogButtonBox)
    cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
    qtbot.mouseClick(cancel_btn, Qt.LeftButton)
    assert ran == []
    assert cancelled == [True]
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_skip_requested_reflects_checkbox(qtbot):
    dialog = RunDialog(FAKE_PARAMS, FAKE_ESTIMATE, calibrated=True,
                       info_only=False)
    qtbot.addWidget(dialog)
    assert dialog.skip_requested() is False
    dialog.skip_checkbox.setChecked(True)
    assert dialog.skip_requested() is True


# ---------------------------------------------------------------------------
# MainWindow wiring: isVisible guard, skip-session, _run_pipeline gating
# ---------------------------------------------------------------------------
def test_confirm_run_dialog_hidden_window_never_blocks(qtbot):
    """A constructed-but-not-shown MainWindow (every offscreen test) must
    proceed without ever building/exec'ing the modal."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.isVisible() is False
    assert window._confirm_run_dialog() is True


def test_confirm_run_dialog_skip_session_flag_short_circuits(qtbot,
                                                              monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "isVisible", lambda: True)
    window.settings.set_bool("run_dialog_skip_session", True)

    execs = []
    monkeypatch.setattr(RunDialog, "exec",
                        lambda self: execs.append(1))
    assert window._confirm_run_dialog() is True
    assert execs == []      # never even constructed/exec'd the modal


def test_confirm_run_dialog_accept_with_skip_checkbox_sets_flag(qtbot,
                                                                 monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "isVisible", lambda: True)
    assert window.settings.get_bool("run_dialog_skip_session", False) \
        is False

    def fake_exec(self):
        if self.skip_checkbox is not None:
            self.skip_checkbox.setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(RunDialog, "exec", fake_exec)

    assert window._confirm_run_dialog() is True
    assert window.settings.get_bool("run_dialog_skip_session", False) \
        is True


def test_confirm_run_dialog_cancel_returns_false_and_no_skip(qtbot,
                                                              monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "isVisible", lambda: True)
    monkeypatch.setattr(RunDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)

    assert window._confirm_run_dialog() is False
    assert window.settings.get_bool("run_dialog_skip_session", False) \
        is False


def test_run_pipeline_aborts_when_run_dialog_cancelled(qtbot, monkeypatch):
    """Mirrors test_mainwindow_train.py's _confirm_sweep-monkeypatch
    idiom: bypass the modal itself, drive _run_pipeline, and check the
    launch (runner.start) does/doesn't happen based on the confirmation
    result."""
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_preflight", lambda: [])
    monkeypatch.setattr(window, "_confirm_run_dialog", lambda: False)

    started = []
    monkeypatch.setattr(window.runner, "start",
                        lambda *a, **k: started.append(True) or True)

    assert window._run_pipeline() is False
    assert started == []


def test_run_pipeline_launches_when_run_dialog_confirmed(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_preflight", lambda: [])
    monkeypatch.setattr(window, "_confirm_run_dialog", lambda: True)
    monkeypatch.setattr(window, "_confirm_sweep", lambda summary: True)

    started = []
    monkeypatch.setattr(window.runner, "start",
                        lambda *a, **k: started.append(True) or True)

    assert window._run_pipeline() is True
    assert started == [True]


def test_run_pipeline_skips_run_dialog_on_dry_run(qtbot, monkeypatch):
    """--dry-run doesn't actually trace anything, so it shouldn't ask."""
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_preflight", lambda: [])

    called = []
    monkeypatch.setattr(window, "_confirm_run_dialog",
                        lambda: called.append(True) or True)
    monkeypatch.setattr(window.runner, "start", lambda *a, **k: True)

    assert window._run_pipeline(dry_run=True) is True
    assert called == []


# ---------------------------------------------------------------------------
# the existing "Estimate runtime" button opens the same dialog, info-only
# ---------------------------------------------------------------------------
def test_show_estimate_builds_info_only_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.isVisible() is False   # never blocks offscreen

    params = window.config_matrix.estimate_params()
    dialog = window._show_estimate(params)
    assert isinstance(dialog, RunDialog)
    assert dialog.info_only is True
    assert dialog.skip_checkbox is None
    assert window._last_estimate_dialog is dialog
