"""RunDialog - the P1 per-run accuracy-vs-time pre-run dialog.

Shown by MainWindow._run_pipeline() before every launch (owner
requirement: "always ask per run"), and reachable in an info-only mode
from the existing ConfigMatrix "Estimate runtime" button so the two
surfaces never show different numbers. Composes:
  - a read-only summary of the resolved run parameters (preset, rays,
    resolution, nlambda, backend, model);
  - the common.estimate() predicted trace/gather/total wall time, labeled
    "estimate (calibrated)" vs "estimate (uncalibrated fallback)" via the
    `calibrated` flag the caller resolves with
    common.estimate_is_calibrated();
  - accumulator/fields memory figures straight out of the estimate dict;
  - for coherent runs (n_coherent_sources >= 1): the projected gather
    pairs and spr (samples/ray) as an M_eff proxy, clearly labeled as a
    projection, not a measurement;
  - Run / Cancel buttons plus a "Don't ask again this session" checkbox
    (run mode), or a single Close button (info-only mode, no checkbox).

Guarded-modal contract (CLAUDE.md: "Never show an unguarded modal in a
pane code path" / "guard on isVisible()"): this module never calls
.exec() itself -- the caller (MainWindow) owns that, guarded on
isVisible() exactly like the existing _confirm_sweep. Tests construct a
RunDialog directly and drive its buttons/checkbox without ever invoking
exec(), so offscreen test runs can never block on it.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common     # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLabel,
    QVBoxLayout,
)

# run_params key -> display label, in display order.
_PARAM_LABELS = [
    ("preset", "Preset"),
    ("rays", "Rays"),
    ("resolution", "Resolution"),
    ("nlambda", "Wavelengths (nlambda)"),
    ("backend", "Backend"),
    ("model_stem", "Model"),
]


def _fmt_num(value):
    try:
        return "%g" % float(value)
    except (TypeError, ValueError):
        return str(value)


class RunDialog(QDialog):
    """Pre-run summary: run parameters + common.estimate() prediction.

    run_params: dict with any of the _PARAM_LABELS keys (missing keys are
        simply omitted from the summary) plus "n_coherent_sources" (used
        only to decide whether to show the coherent-gather projection
        box).
    estimate: a common.estimate(...) result dict.
    calibrated: common.estimate_is_calibrated(...)'s verdict, or None to
        omit the calibrated/fallback label entirely.
    info_only: True for the Estimate-button entry point (Close only, no
        Run/Cancel, no "don't ask again" checkbox).
    """

    def __init__(self, run_params, estimate, calibrated=None,
                info_only=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estimated Runtime" if info_only
                            else "Confirm Run")
        self.run_params = dict(run_params or {})
        self.estimate = dict(estimate or {})
        self.info_only = info_only
        self.skip_checkbox = None

        layout = QVBoxLayout(self)

        params_box = QGroupBox("Run parameters")
        params_form = QFormLayout()
        for key, label in _PARAM_LABELS:
            if key not in self.run_params or self.run_params[key] is None:
                continue
            params_form.addRow(label + ":",
                               QLabel(str(self.run_params[key])))
        params_box.setLayout(params_form)
        layout.addWidget(params_box)

        time_box = QGroupBox(self._time_box_title(calibrated))
        time_form = QFormLayout()
        time_form.addRow("Trace:", QLabel(
            common.fmt_duration(self.estimate.get("trace_s", 0.0))))
        time_form.addRow("Gather:", QLabel(
            common.fmt_duration(self.estimate.get("gather_s", 0.0))))
        total_label = QLabel(
            common.fmt_duration(self.estimate.get("total_s", 0.0)))
        total_label.setStyleSheet("font-weight: bold;")
        time_form.addRow("Total:", total_label)
        time_box.setLayout(time_form)
        layout.addWidget(time_box)

        mem_box = QGroupBox("Memory / disk")
        mem_form = QFormLayout()
        mem_form.addRow("Accumulator:", QLabel(
            "%.3f GB" % self.estimate.get("accumulator_GB", 0.0)))
        fields_gb = self.estimate.get("fields_h5_GB", 0.0) or 0.0
        if fields_gb:
            mem_form.addRow("Saved fields (.h5):",
                            QLabel("%.3f GB" % fields_gb))
        mem_box.setLayout(mem_form)
        layout.addWidget(mem_box)

        if (self.run_params.get("n_coherent_sources") or 0) >= 1:
            proj_box = QGroupBox(
                "Coherent gather projection (M_eff proxy)")
            proj_form = QFormLayout()
            proj_form.addRow("Projected pairs:", QLabel(
                _fmt_num(self.estimate.get("gather_pairs", 0.0))))
            proj_form.addRow("spr (samples/ray):", QLabel(
                _fmt_num(self.estimate.get("spr", 0.0))))
            note = QLabel(
                "Projection only, not a measured M_eff -- a diagnostic "
                "for how heavy the coherent gather will be.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            proj_form.addRow(note)
            proj_box.setLayout(proj_form)
            layout.addWidget(proj_box)

        if info_only:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close)
            # Close carries Qt's RejectRole; any click here should simply
            # dismiss the dialog with no further consequence to the
            # caller, so route every button to accept().
            buttons.clicked.connect(lambda _btn: self.accept())
        else:
            self.skip_checkbox = QCheckBox("Don't ask again this session")
            self.skip_checkbox.setChecked(False)
            self.skip_checkbox.setToolTip(
                "Skip this dialog for subsequent runs until MieWorkbench "
                "is restarted (never persisted across sessions)")
            layout.addWidget(self.skip_checkbox)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _time_box_title(self, calibrated):
        if calibrated is None:
            return "Predicted wall time"
        if calibrated:
            return "Predicted wall time — estimate (calibrated)"
        return "Predicted wall time — estimate (uncalibrated fallback)"

    def skip_requested(self):
        """True when the run-mode dialog was accepted with "don't ask
        again this session" checked. Always False in info-only mode."""
        return bool(self.skip_checkbox is not None
                    and self.skip_checkbox.isChecked())
