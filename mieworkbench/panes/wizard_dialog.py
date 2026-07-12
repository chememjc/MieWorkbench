"""Add/customize-element wizard dialog.

Every primitive gets a geometry-parameter table prefilled with its
defaults (alias / value / unit, tooltips, `round_flag` as a "Circular
shape" checkbox) AND a device-properties form (source power/wavelength/
polarization, detector reflectivity, optic material/coating/filter...)
so the whole element is configured in one place. Lens primitives
additionally get a "design by focal length" section driving
core.wizards.design_lens: enter f + material (+ thickness), Compute
fills the parameter table with the solved radii and shows the exact
EFL/BFL cross-check.

The optional Preview button emits previewRequested — the main window
imports/rebuilds the element live in the 3D view while the dialog stays
open (Cancel rolls the previewed element back).

Re-customizing an existing element uses for_element(): same dialog,
prefilled from the element's parameter sheet and body properties, with
Apply semantics.
"""

import os
import sys

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ..core import wizards
from .element_wizard import (
    ParamTableWidget, PropertiesFormWidget, property_rows_for,
)

# primitive kind -> wizard form (reverse of LENS_FORMS[form]['primitive'];
# first form wins where two forms share a primitive)
_FORM_FOR_PRIMITIVE = {}
for _form, _spec in wizards.LENS_FORMS.items():
    _FORM_FOR_PRIMITIVE.setdefault(_spec["primitive"], _form)


class ElementWizardDialog(QDialog):
    """Collects (label, {alias: value}, {prop: value}) for one primitive
    instance — geometry parameters AND device properties."""

    previewRequested = Signal()

    def __init__(self, primitive_info, default_label, matdb=None,
                 registry_names=None, parent=None, customize=False,
                 show_preview=False):
        super().__init__(parent)
        self.info = primitive_info
        self.matdb = matdb
        self._customize = customize
        kind = primitive_info.get("kind", "?")
        title = ("Customize %s" if customize else "Add %s")
        self.setWindowTitle(title % primitive_info.get("label", kind))

        lay = QVBoxLayout(self)
        tip = QLabel(primitive_info.get("tooltip", ""))
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray;")
        lay.addWidget(tip)

        grid = QGridLayout()
        grid.addWidget(QLabel("Element label:"), 0, 0)
        self.label_edit = QLineEdit(default_label)
        self.label_edit.setToolTip("Name of the new element in the scene")
        if customize:
            self.label_edit.setEnabled(False)
        grid.addWidget(self.label_edit, 0, 1)
        lay.addLayout(grid)

        self._form = _FORM_FOR_PRIMITIVE.get(kind)
        if self._form is not None:
            lay.addWidget(self._build_designer())
        if kind == "waveplate":
            lay.addWidget(self._build_waveplate_designer())

        geom_box = QGroupBox("Geometry [mm/deg]")
        geom_lay = QVBoxLayout(geom_box)
        self.table = ParamTableWidget(primitive_info.get("params", {}))
        geom_lay.addWidget(self.table)
        lay.addWidget(geom_box, 2)

        rows = property_rows_for(primitive_info)
        self.props_form = None
        if rows:
            props_box = QGroupBox("Device properties")
            props_lay = QVBoxLayout(props_box)
            self.props_form = PropertiesFormWidget(
                rows, registry_names=registry_names)
            props_lay.addWidget(self.props_form)
            lay.addWidget(props_box, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(
            "Apply" if customize else "Add element")
        if show_preview:
            self.preview_button = QPushButton("Preview")
            self.preview_button.setToolTip(
                "Build/update the element in the 3D view now, keeping "
                "this dialog open (Cancel removes it again)")
            buttons.addButton(self.preview_button,
                              QDialogButtonBox.ActionRole)
            self.preview_button.clicked.connect(self.previewRequested.emit)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(460, 560)

    # -- prefill from an existing element ---------------------------------------
    @classmethod
    def for_element(cls, primitive_info, label, sheet_values=None,
                    prop_values=None, matdb=None, registry_names=None,
                    parent=None):
        """Customize-mode dialog prefilled from an existing element's
        sheet numbers ({alias: float}) and body properties
        ({name: value})."""
        dlg = cls(primitive_info, label, matdb=matdb,
                  registry_names=registry_names, parent=parent,
                  customize=True)
        for alias, value in (sheet_values or {}).items():
            dlg.table.set_value(alias, value)
        if dlg.props_form is not None:
            for name, value in (prop_values or {}).items():
                dlg.props_form.set_value(name, value)
        return dlg

    # -- focal-length designer -------------------------------------------------
    def _build_designer(self):
        box = QGroupBox("Design by focal length")
        g = QGridLayout(box)
        g.addWidget(QLabel("f [mm]:"), 0, 0)
        self.f_edit = QLineEdit("50")
        self.f_edit.setValidator(QDoubleValidator())
        self.f_edit.setToolTip("Target effective focal length (negative "
                               "for diverging forms)")
        g.addWidget(self.f_edit, 0, 1)
        g.addWidget(QLabel("Material:"), 0, 2)
        self.mat_combo = QComboBox()
        self.mat_combo.setEditable(True)
        mats = []
        if self.matdb is not None:
            try:
                mats = sorted(self.matdb)
            except Exception:
                mats = []
        self.mat_combo.addItems(mats or ["bk7"])
        idx = self.mat_combo.findText("bk7")
        if idx >= 0:
            self.mat_combo.setCurrentIndex(idx)
        self.mat_combo.setToolTip("Lens material (index at the d-line "
                                  "drives the design)")
        g.addWidget(self.mat_combo, 0, 3)
        btn = QPushButton("Compute radii")
        btn.setToolTip("Solve the thick-lens equation for this form and "
                       "fill the parameter table")
        btn.clicked.connect(self._compute)
        g.addWidget(btn, 1, 0, 1, 2)
        self.design_out = QLabel("")
        self.design_out.setStyleSheet("color: gray;")
        g.addWidget(self.design_out, 1, 2, 1, 2)
        return box

    # -- waveplate retardance designer -------------------------------------------
    def _build_waveplate_designer(self):
        box = QGroupBox("Design by retardance")
        g = QGridLayout(box)
        g.addWidget(QLabel("Type:"), 0, 0)
        self.wp_kind = QComboBox()
        self.wp_kind.addItem("half-wave (λ/2)", "half")
        self.wp_kind.addItem("quarter-wave (λ/4)", "quarter")
        self.wp_kind.setToolTip("Target retardance at the design "
                                "wavelength")
        g.addWidget(self.wp_kind, 0, 1)
        g.addWidget(QLabel("λ [nm]:"), 0, 2)
        self.wp_lambda = QLineEdit("633")
        self.wp_lambda.setValidator(QDoubleValidator())
        self.wp_lambda.setToolTip("Design wavelength")
        g.addWidget(self.wp_lambda, 0, 3)
        g.addWidget(QLabel("Order:"), 0, 4)
        self.wp_order = QLineEdit("0")
        self.wp_order.setToolTip("0 = true zero-order (thinnest); higher "
                                 "orders add whole waves of retardance "
                                 "(thicker, easier to make, more "
                                 "wavelength-sensitive)")
        g.addWidget(self.wp_order, 0, 5)
        btn = QPushButton("Compute thickness")
        btn.setToolTip("Solve the quartz plate thickness for this "
                       "retardance at λ and fill the parameter table")
        btn.clicked.connect(self._compute_waveplate)
        g.addWidget(btn, 1, 0, 1, 2)
        self.wp_out = QLabel("")
        self.wp_out.setStyleSheet("color: gray;")
        g.addWidget(self.wp_out, 1, 2, 1, 4)
        return box

    def _compute_waveplate(self):
        try:
            design = wizards.waveplate_thickness(
                self.wp_kind.currentData(), float(self.wp_lambda.text()),
                order=int(self.wp_order.text() or 0))
        except Exception as exc:
            self.wp_out.setText(str(exc))
            return
        self.table.set_value("thickness", design["thickness"])
        self.wp_out.setText(
            "thickness %.4f mm (%s, %.3g waves, Δn=%.5f)"
            % (design["thickness"], design["crystal"], design["waves"],
               design["delta_n"]))

    def _compute(self):
        try:
            f = float(self.f_edit.text())
            ct = self.params().get("ct")
            design = wizards.design_lens(
                self._form, f, matdb=self.matdb,
                material=self.mat_combo.currentText(), ct_mm=ct)
        except Exception as exc:
            self.design_out.setText(str(exc))
            return
        for alias, value in design["params"].items():
            self.table.set_value(alias, value)
        d = design["design"]
        parts = []
        if "efl" in d and d["efl"] is not None:
            parts.append("EFL %.3f mm" % d["efl"])
        if "bfl" in d and d["bfl"] is not None:
            parts.append("BFL %.3f mm" % d["bfl"])
        if "n" in d:
            parts.append("n=%.5f" % d["n"])
        self.design_out.setText("  ".join(parts))

    # -- results -----------------------------------------------------------------
    def element_label(self):
        return self.label_edit.text().strip()

    def params(self):
        return self.table.values()

    def changed_params(self):
        """Only the aliases whose value differs from the primitive default
        (these need sheet writes + a rebuild after import)."""
        return self.table.changed_values()

    def props(self):
        return self.props_form.values() if self.props_form else {}

    def changed_props(self):
        """Device properties that differ from the primitive's baked
        defaults (these need set_property calls after import)."""
        return self.props_form.changed_values() if self.props_form else {}


class ZoomPairDialog(QDialog):
    """Standalone two-group zoom-pair calculator (core.wizards.
    solve_zoom_pair) — future.md (a2) / UXNOTES_ROUND3 "nothing computes
    p,q,r,s for you". Not tied to any one primitive (unlike
    ElementWizardDialog): inputs f1/f2/z, outputs BFL/EFL/track readouts
    plus a copyable train-grammar expression string for BFL(z), following
    the same "Design by ..." section pattern as the lens/waveplate
    designers above (Compute button, gray result label)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoom-pair Calculator")
        lay = QVBoxLayout(self)

        note = QLabel(
            "Idealized thin-lens groups of focal length f1 (front) / f2 "
            "(rear); z is the gap between the groups' own principal "
            "planes, not the chain's vertex-to-vertex distance — for a "
            "real (thick) group, add pp1_rear - pp2_front from its own "
            "paraxial readout to the chain gap first.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)

        grid = QGridLayout()
        grid.addWidget(QLabel("f1 (front) [mm]:"), 0, 0)
        self.f1_edit = QLineEdit("100")
        self.f1_edit.setValidator(QDoubleValidator())
        grid.addWidget(self.f1_edit, 0, 1)
        grid.addWidget(QLabel("f2 (rear) [mm]:"), 0, 2)
        self.f2_edit = QLineEdit("-50")
        self.f2_edit.setValidator(QDoubleValidator())
        grid.addWidget(self.f2_edit, 0, 3)
        grid.addWidget(QLabel("z [mm]:"), 1, 0)
        self.z_edit = QLineEdit("30")
        self.z_edit.setValidator(QDoubleValidator())
        self.z_edit.setToolTip("Inter-group gap to evaluate BFL/EFL/track "
                               "at (principal-plane referenced)")
        grid.addWidget(self.z_edit, 1, 1)
        btn = QPushButton("Compute")
        btn.setToolTip("Solve BFL(z)/EFL(z)/track(z) and the general "
                       "rational expression for this f1/f2 pair")
        btn.clicked.connect(self._compute)
        grid.addWidget(btn, 1, 2, 1, 2)
        lay.addLayout(grid)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: gray;")
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)

        lay.addWidget(QLabel("BFL(z) expression (train grammar):"))
        self.bfl_expr_edit = QLineEdit()
        self.bfl_expr_edit.setReadOnly(True)
        lay.addWidget(self.bfl_expr_edit)

        lay.addWidget(QLabel("EFL(z) expression:"))
        self.efl_expr_edit = QLineEdit()
        self.efl_expr_edit.setReadOnly(True)
        lay.addWidget(self.efl_expr_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)
        self.resize(480, 320)
        self._last = None

    def _compute(self):
        try:
            f1 = float(self.f1_edit.text())
            f2 = float(self.f2_edit.text())
            z = float(self.z_edit.text())
            design = wizards.solve_zoom_pair(f1, f2, z_mm=z)
        except Exception as exc:
            self.result_label.setText(str(exc))
            self._last = None
            return
        self._last = design
        self.result_label.setText(
            "BFL %.4g mm   EFL %.4g mm   total track %.4g mm"
            % (design["bfl_mm"], design["efl_mm"], design["track_mm"]))
        self.bfl_expr_edit.setText(design["bfl_expr"])
        self.efl_expr_edit.setText(design["efl_expr"])

    def result(self):
        """The last computed solve_zoom_pair() dict, or None."""
        return self._last


class FieldFanDialog(QDialog):
    """Field-angle fan wizard page (core.wizards.design_field_fan): N
    field-point sources arc- or plane-spaced about a pivot, each aimed at
    it, each carrying its field_angle_deg property — the source layout
    the --imaging-products field renderers (distortion / vignetting /
    field curves / telecentricity) consume. Follows the ZoomPairDialog
    pattern (Compute button, gray result label, no unguarded modals);
    the CALLER (mainwindow) inserts the primitives from result() as one
    undo macro through the Project API."""

    def __init__(self, source_kinds=None, parent=None):
        """source_kinds: [(kind, display_label)] for the source-kind
        combo (defaults to laser_collimated only)."""
        super().__init__(parent)
        self.setWindowTitle("Field-angle Fan")
        lay = QVBoxLayout(self)

        note = QLabel(
            "Places N field-point sources aimed at a common pivot (the "
            "system's entrance aperture), one per field angle — the "
            "layout the imaging products (distortion, vignetting, field "
            "curves, telecentricity) analyze. Each source gets a "
            "field_angle_deg property recording its design angle.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)

        grid = QGridLayout()
        grid.addWidget(QLabel("N sources:"), 0, 0)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 25)
        self.n_spin.setValue(3)
        self.n_spin.setToolTip("Number of field points, evenly spanning "
                               "±θmax (ignored when an explicit angle "
                               "list is given)")
        grid.addWidget(self.n_spin, 0, 1)
        grid.addWidget(QLabel("±θmax [deg]:"), 0, 2)
        self.theta_edit = QLineEdit("15")
        self.theta_edit.setValidator(QDoubleValidator())
        self.theta_edit.setToolTip("Half-angle of the fan; N sources "
                                   "span [-θmax, +θmax]")
        grid.addWidget(self.theta_edit, 0, 3)

        grid.addWidget(QLabel("Angles [deg] (optional):"), 1, 0)
        self.angles_edit = QLineEdit("")
        self.angles_edit.setToolTip(
            "Explicit comma-separated field angles (e.g. '0, 8, 16'); "
            "when non-empty this overrides N/θmax")
        grid.addWidget(self.angles_edit, 1, 1, 1, 3)

        grid.addWidget(QLabel("Pivot x,y,z [mm]:"), 2, 0)
        self.pivot_edit = QLineEdit("0, 0, 0")
        self.pivot_edit.setToolTip("Common aim point (typically the "
                                   "first element's front vertex / the "
                                   "aperture stop)")
        grid.addWidget(self.pivot_edit, 2, 1, 1, 3)

        grid.addWidget(QLabel("Radius R [mm]:"), 3, 0)
        self.radius_edit = QLineEdit("100")
        self.radius_edit.setValidator(QDoubleValidator())
        self.radius_edit.setToolTip("Source stand-off from the pivot "
                                    "(auto-grown if the source bodies "
                                    "would overlap)")
        grid.addWidget(self.radius_edit, 3, 1)
        grid.addWidget(QLabel("Aperture [mm]:"), 3, 2)
        self.aperture_edit = QLineEdit("")
        self.aperture_edit.setValidator(QDoubleValidator())
        self.aperture_edit.setToolTip("Beam/emit diameter per source "
                                      "(blank = the primitive's default; "
                                      "also the overlap-guard bounding "
                                      "diameter)")
        grid.addWidget(self.aperture_edit, 3, 3)

        grid.addWidget(QLabel("Source kind:"), 4, 0)
        self.kind_combo = QComboBox()
        for kind, disp in (source_kinds
                           or [("laser_collimated", "Collimated laser")]):
            self.kind_combo.addItem(disp, kind)
        grid.addWidget(self.kind_combo, 4, 1)
        grid.addWidget(QLabel("Fan plane:"), 4, 2)
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("xy (about z)", "xy")
        self.plane_combo.addItem("xz (about y)", "xz")
        grid.addWidget(self.plane_combo, 4, 3)

        grid.addWidget(QLabel("Spacing:"), 5, 0)
        self.spacing_combo = QComboBox()
        self.spacing_combo.addItem("arc (constant distance to pivot)",
                                   "arc")
        self.spacing_combo.addItem("plane (constant axial distance)",
                                   "plane")
        grid.addWidget(self.spacing_combo, 5, 1)
        btn = QPushButton("Compute")
        btn.setToolTip("Solve the fan placements and preview them below")
        btn.clicked.connect(self._compute)
        grid.addWidget(btn, 5, 2, 1, 2)
        lay.addLayout(grid)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: gray;")
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add fan")
        buttons.accepted.connect(self._accept_with_compute)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(520, 420)
        self._last = None

    # -- parsing helpers ---------------------------------------------------
    def _angles(self):
        text = self.angles_edit.text().strip()
        if text:
            return [float(t) for t in text.replace(";", ",").split(",")
                    if t.strip()]
        return None

    def _pivot(self):
        parts = [t for t in self.pivot_edit.text().replace(";", ",")
                 .split(",") if t.strip()]
        if len(parts) != 3:
            raise ValueError("pivot needs three comma-separated "
                             "coordinates (x, y, z in mm)")
        return tuple(float(t) for t in parts)

    def _compute(self):
        try:
            angles = self._angles()
            n_or_angles = angles if angles is not None \
                else int(self.n_spin.value())
            aperture = self.aperture_edit.text().strip()
            design = wizards.design_field_fan(
                n_or_angles,
                theta_max_deg=float(self.theta_edit.text() or 0),
                pivot_mm=self._pivot(),
                radius_mm=float(self.radius_edit.text()),
                source_kind=self.kind_combo.currentData(),
                aperture_mm=float(aperture) if aperture else None,
                plane=self.plane_combo.currentData(),
                spacing=self.spacing_combo.currentData())
        except Exception as exc:
            self.result_label.setText(str(exc))
            self._last = None
            return False
        self._last = design
        lines = ["%d source(s) on R=%.4g mm (%s, %s plane):"
                 % (len(design["sources"]), design["radius_mm"],
                    design["spacing"], design["plane"])]
        for s in design["sources"]:
            lines.append("  θ=%+.4g°  at (%.4g, %.4g, %.4g) mm"
                         % ((s["angle_deg"],) + tuple(s["pos_mm"])))
        if design["note"]:
            lines.append("NOTE: %s" % design["note"])
        self.result_label.setText("\n".join(lines))
        return True

    def _accept_with_compute(self):
        # never an unguarded modal: a bad input just shows in the label
        if self._compute():
            self.accept()

    def result(self):
        """The last computed design_field_fan() dict, or None."""
        return self._last

    def aperture_mm(self):
        """Explicit per-source aperture [mm] or None (primitive default)."""
        text = self.aperture_edit.text().strip()
        return float(text) if text else None
