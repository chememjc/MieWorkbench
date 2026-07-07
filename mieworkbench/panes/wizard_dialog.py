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
    QLabel, QLineEdit, QPushButton, QVBoxLayout,
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
