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

A third mode (`extend_ctx` given, P1 chunked-run contract's additive
extension) reuses this same shell for a COMPLETED C-engine case: instead
of the static estimate above, it shows the case's current rays / measured
M_eff proxy, a spin box (+ x2/x5/x10 presets) for the new total, and a
LIVE-recomputed projected ADDITIONAL wall time + pedestal-improvement
factor as the spin box changes (extend_target_rays() is what the caller
reads back on accept).

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
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
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
    estimate: a common.estimate(...) result dict (the BASE/current-rays
        estimate in extend mode -- see extend_ctx).
    calibrated: common.estimate_is_calibrated(...)'s verdict, or None to
        omit the calibrated/fallback label entirely.
    info_only: True for the Estimate-button entry point (Close only, no
        Run/Cancel, no "don't ask again" checkbox).
    extend_ctx: None for the normal pre-run dialog, or a dict activating
        extend mode (P1 chunked-run additive extension) with keys
        'current_rays', 'spr' (this case's OWN measured surviving-
        samples-per-ray, not a machine calibration), 'resolution',
        'nlambda', 'backend', 'n_coherent_sources', and optionally
        'n_detectors'/'save_fields'/'n_pol_strata'/'model_stem' (fed
        straight to common.estimate() on every spin-box change). Mutually
        exclusive with info_only (extend always asks Extend/Cancel).
    """

    def __init__(self, run_params, estimate, calibrated=None,
                info_only=False, extend_ctx=None, parent=None):
        super().__init__(parent)
        self.extend_ctx = dict(extend_ctx) if extend_ctx else None
        self.setWindowTitle(
            "Extend Run" if self.extend_ctx is not None
            else ("Estimated Runtime" if info_only else "Confirm Run"))
        self.run_params = dict(run_params or {})
        self.estimate = dict(estimate or {})
        self.info_only = info_only
        self.skip_checkbox = None
        self.rays_spin = None

        layout = QVBoxLayout(self)

        if self.extend_ctx is not None:
            banner = QLabel(
                "Extending a COMPLETED run: the additional rays trace and "
                "merge additively onto the existing checkpoint -- "
                "statistically equivalent to, not bit-identical with, a "
                "fresh run at the new total.")
            banner.setWordWrap(True)
            banner.setStyleSheet("font-weight: bold;")
            layout.addWidget(banner)

        params_box = QGroupBox("Run parameters")
        params_form = QFormLayout()
        for key, label in _PARAM_LABELS:
            if key not in self.run_params or self.run_params[key] is None:
                continue
            params_form.addRow(label + ":",
                               QLabel(str(self.run_params[key])))
        params_box.setLayout(params_form)
        layout.addWidget(params_box)

        if self.extend_ctx is not None:
            layout.addWidget(self._build_extend_box())

        self.trace_label = QLabel()
        self.gather_label = QLabel()
        self.total_label = QLabel()
        self.total_label.setStyleSheet("font-weight: bold;")
        time_box = QGroupBox(self._time_box_title(calibrated))
        time_form = QFormLayout()
        time_form.addRow("Trace:", self.trace_label)
        time_form.addRow("Gather:", self.gather_label)
        time_form.addRow(
            "Additional (projected):" if self.extend_ctx is not None
            else "Total:", self.total_label)
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

        if self.extend_ctx is None and (
                self.run_params.get("n_coherent_sources") or 0) >= 1:
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

        if self.extend_ctx is not None:
            self._update_extend_projection()
        else:
            self._set_time_labels(self.estimate)

        if info_only:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close)
            # Close carries Qt's RejectRole; any click here should simply
            # dismiss the dialog with no further consequence to the
            # caller, so route every button to accept().
            buttons.clicked.connect(lambda _btn: self.accept())
        elif self.extend_ctx is not None:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(
                QDialogButtonBox.StandardButton.Ok).setText("Extend")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
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

    # -- extend mode -----------------------------------------------------------
    def _build_extend_box(self):
        ctx = self.extend_ctx
        current = int(ctx.get("current_rays", 0))
        spr = float(ctx.get("spr", common.DEFAULT_SPR))

        box = QGroupBox("Extend to")
        form = QFormLayout()
        form.addRow("Current rays:", QLabel(_fmt_num(current)))
        form.addRow("Current M_eff proxy (spr x rays):",
                    QLabel(_fmt_num(spr * current)))

        self.rays_spin = QDoubleSpinBox()
        self.rays_spin.setDecimals(0)
        self.rays_spin.setRange(float(current + 1),
                                max(float(current) * 1000.0,
                                    float(current) + 1.0))
        self.rays_spin.setSingleStep(max(float(current), 1.0))
        self.rays_spin.setValue(float(current) * 2.0)
        self.rays_spin.valueChanged.connect(self._update_extend_projection)
        form.addRow("New total rays:", self.rays_spin)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        for mult in (2, 5, 10):
            btn = QPushButton("x%d" % mult)
            btn.clicked.connect(
                lambda _checked=False, m=mult:
                    self.rays_spin.setValue(float(current) * m))
            preset_row.addWidget(btn)
        preset_row.addStretch(1)
        preset_widget = QWidget()
        preset_widget.setLayout(preset_row)
        form.addRow("Presets:", preset_widget)

        self.meff_label = QLabel()
        form.addRow("Projected M_eff proxy:", self.meff_label)
        self.pedestal_label = QLabel()
        self.pedestal_label.setWordWrap(True)
        form.addRow("Projected pedestal improvement:", self.pedestal_label)
        note = QLabel(
            "Projection only (1/M_eff scaling from this case's own "
            "measured spr), not a re-measurement.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)

        box.setLayout(form)
        return box

    def _update_extend_projection(self, *_args):
        ctx = self.extend_ctx
        if ctx is None or self.rays_spin is None:
            return
        new_rays = self.rays_spin.value()
        current = max(int(ctx.get("current_rays", 0)), 1)
        spr = float(ctx.get("spr", common.DEFAULT_SPR))
        new_estimate = common.estimate(
            new_rays, ctx.get("resolution", 512), ctx.get("nlambda", 5),
            ctx.get("n_coherent_sources", 0), ctx.get("backend", "auto"),
            n_detectors=ctx.get("n_detectors", 1),
            save_fields=ctx.get("save_fields", False),
            n_pol_strata=ctx.get("n_pol_strata", 1),
            model_stem=ctx.get("model_stem"))
        delta_trace = max(
            new_estimate["trace_s"] - self.estimate.get("trace_s", 0.0), 0.0)
        delta_gather = max(
            new_estimate["gather_s"] - self.estimate.get("gather_s", 0.0),
            0.0)
        self._set_time_labels({"trace_s": delta_trace,
                               "gather_s": delta_gather,
                               "total_s": delta_trace + delta_gather})
        self.meff_label.setText(_fmt_num(spr * new_rays))
        factor = new_rays / current
        self.pedestal_label.setText(
            "~%.3gx lower speckle pedestal (1/M_eff scaling)" % factor)

    def extend_target_rays(self):
        """The spin box's current value, or None outside extend mode."""
        return (int(self.rays_spin.value())
                if self.rays_spin is not None else None)

    def _set_time_labels(self, est):
        self.trace_label.setText(
            common.fmt_duration(est.get("trace_s", 0.0)))
        self.gather_label.setText(
            common.fmt_duration(est.get("gather_s", 0.0)))
        self.total_label.setText(
            common.fmt_duration(est.get("total_s", 0.0)))

    def _time_box_title(self, calibrated):
        if calibrated is None:
            return "Predicted wall time"
        if calibrated:
            return "Predicted wall time — estimate (calibrated)"
        return "Predicted wall time — estimate (uncalibrated fallback)"

    def skip_requested(self):
        """True when the run-mode dialog was accepted with "don't ask
        again this session" checked. Always False in info-only/extend
        mode."""
        return bool(self.skip_checkbox is not None
                    and self.skip_checkbox.isChecked())
