"""Add-element wizard dialog.

Every primitive gets a parameter table prefilled with its defaults
(alias / value / unit, tooltips from the primitive metadata). Lens
primitives additionally get a "design by focal length" section driving
core.wizards.design_lens: enter f + material (+ thickness), Compute fills
the parameter table with the solved radii and shows the exact EFL/BFL
cross-check.
"""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ..core import wizards

# primitive kind -> wizard form (reverse of LENS_FORMS[form]['primitive'];
# first form wins where two forms share a primitive)
_FORM_FOR_PRIMITIVE = {}
for _form, _spec in wizards.LENS_FORMS.items():
    _FORM_FOR_PRIMITIVE.setdefault(_spec["primitive"], _form)


class ElementWizardDialog(QDialog):
    """Collects (label, {alias: value}) for one primitive instance."""

    def __init__(self, primitive_info, default_label, matdb=None,
                 parent=None):
        super().__init__(parent)
        self.info = primitive_info
        self.matdb = matdb
        kind = primitive_info.get("kind", "?")
        self.setWindowTitle("Add %s" % primitive_info.get("label", kind))

        lay = QVBoxLayout(self)
        tip = QLabel(primitive_info.get("tooltip", ""))
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray;")
        lay.addWidget(tip)

        grid = QGridLayout()
        grid.addWidget(QLabel("Element label:"), 0, 0)
        self.label_edit = QLineEdit(default_label)
        self.label_edit.setToolTip("Name of the new element in the scene")
        grid.addWidget(self.label_edit, 0, 1)
        lay.addLayout(grid)

        self._form = _FORM_FOR_PRIMITIVE.get(kind)
        if self._form is not None:
            lay.addWidget(self._build_designer())

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        params = primitive_info.get("params", {})
        self.table.setRowCount(len(params))
        self._aliases = []
        for row, (alias, spec) in enumerate(sorted(params.items())):
            self._aliases.append(alias)
            name_item = QTableWidgetItem(alias)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setToolTip(spec.get("help", ""))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(
                "%g" % spec.get("default", 0.0)))
            unit_item = QTableWidgetItem(spec.get("unit", ""))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, unit_item)
        lay.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add element")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(430, 420)

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
            if alias in self._aliases:
                row = self._aliases.index(alias)
                self.table.item(row, 1).setText("%.6g" % value)
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
        out = {}
        for row, alias in enumerate(self._aliases):
            item = self.table.item(row, 1)
            try:
                out[alias] = float(item.text())
            except (TypeError, ValueError):
                pass
        return out

    def changed_params(self):
        """Only the aliases whose value differs from the primitive default
        (these need sheet writes + a rebuild after import)."""
        defaults = {a: s.get("default")
                    for a, s in self.info.get("params", {}).items()}
        return {a: v for a, v in self.params().items()
                if defaults.get(a) is None
                or abs(v - float(defaults[a])) > 1e-12}
