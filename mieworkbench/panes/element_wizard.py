"""Element wizard building blocks.

ParamTableWidget       geometry parameters (Name | Value | Unit), with the
                       `round_flag` convention rendered as a "Circular
                       shape" checkbox instead of a bare 0/1 number.
PropertiesFormWidget   the DEVICE properties a primitive exposes (source
                       power/wavelength/polarization/Gaussian beam waist+
                       M^2+apodization, detector reflectivity, optic
                       material/coating/filter/OD...) so the wizard
                       configures the whole element, not just its
                       dimensions - full re-editing remains in the Element
                       Properties pane afterwards.
ApodizationEditor      composite row widget for a source's `apodization`
                       property (none/gaussian + w0 + order); a plain
                       string in every other row kind.
TypeChooserDialog      the type-first entry point ("what do you want to
                       add?"): role -> filtered primitive list; returns
                       the primitive info for the normal wizard.

property_rows_for() decides which properties a primitive exposes, from
its baked `props` in the .meta.json plus the role they imply. Units come
from core.units; registry-valued rows get combo boxes filled from the
loaded optical-property library.
"""

import os
import sys

_SCRIPTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QRadioButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.units import PROPERTY_UNITS, label_with_unit

ROUND_FLAG_ALIAS = "round_flag"

# property-name -> (kind, default, tooltip); combo choices are resolved
# at build time from the optical-property library
_SOURCE_PROPS = (
    ("power", "float", 5.0, "Source power"),
    ("lambdac", "float", 633.0, "Center wavelength"),
    ("coherent", "bool", False,
     "Coherent emission (interference/diffraction at the detector); "
     "geometric focus checks are cleaner with this off"),
    ("polarization", "choice", "unpolarized",
     "Emitted polarization state"),
)
_BROADBAND_PROPS = (
    ("lambdamin", "float", 450.0, "Spectral lower bound"),
    ("lambdamax", "float", 650.0, "Spectral upper bound"),
)
# Gaussian-beam source extras -- always offered on a source (unlike the
# broadband pair above, which only appears when the primitive already
# bakes lambdamin/lambdamax): beam_waist blank means "off" (plane/uniform
# emission), so its kind is the optional-float variant, not "float".
_BEAM_PROPS = (
    ("beam_waist", "float_optional", None,
     "Gaussian beam waist w0 in mm at the emitting face (blank = off, "
     "i.e. plane/uniform emission)"),
    ("m2", "float", 1.0,
     "Beam quality factor M² (>=1.0); only meaningful with a beam "
     "waist set"),
    ("apodization", "apod", None,
     "Transverse field-amplitude apodization across the emitting face"),
)
_DETECTOR_PROPS = (
    ("mirror", "float", 0.0,
     "Reflectivity: fraction of arriving light specularly reflected "
     "back off the detector surface (0 = perfectly transparent screen)"),
    ("absorbance", "float", 0.0,
     "Fraction of the non-reflected light absorbed at the surface"),
)
_POLARIZATION_CHOICES = ("unpolarized", "linear:0", "linear:45",
                         "linear:90", "circular:left", "circular:right")

_REGISTRY_KINDS = {"material": "materials", "coating": "coatings",
                   "polarizer": "polarizers", "filter": "filters",
                   "grating": "gratings"}


def property_rows_for(info):
    """[(name, kind, default, tooltip)] for the device-property form of
    one primitive, derived from its baked props + implied role."""
    props = dict(info.get("props") or {})
    rows = []
    if "power" in props and "lambdac" in props:
        for name, kind, default, tip in _SOURCE_PROPS:
            rows.append((name, kind, props.get(name, default), tip))
        if "lambdamin" in props or "lambdamax" in props:
            for name, kind, default, tip in _BROADBAND_PROPS:
                rows.append((name, kind, props.get(name, default), tip))
        for name, kind, default, tip in _BEAM_PROPS:
            rows.append((name, kind, props.get(name, default), tip))
    elif props.get("material") == "detector":
        for name, kind, default, tip in _DETECTOR_PROPS:
            rows.append((name, kind, props.get(name, default), tip))
    else:
        # optical element: expose the baked contract props for editing
        for name, value in sorted(props.items()):
            if name.startswith("miewb_"):
                continue
            if name == "material":
                rows.append((name, "registry", value,
                             "Bulk material (materials registry)"))
            elif name in ("coating", "polarizer", "filter", "grating"):
                rows.append((name, "registry", value,
                             "%s registry entry / per-face spec" % name))
            elif name in ("mirror", "absorbance"):
                rows.append((name, "float", value,
                             "Fraction in [0, 1]"))
            elif isinstance(value, bool):
                rows.append((name, "bool", value, name))
            else:
                rows.append((name, "text", value, name))
    return rows


class ParamTableWidget(QTableWidget):
    """Geometry parameters: Name | Value | Unit. round_flag renders as a
    'Circular shape' checkbox row."""

    def __init__(self, params, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self._params = dict(params or {})
        self._aliases = []
        self._round_box = None
        self.setRowCount(len(self._params))
        for row, (alias, spec) in enumerate(sorted(self._params.items())):
            self._aliases.append(alias)
            name_item = QTableWidgetItem(alias)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setToolTip(spec.get("help", ""))
            self.setItem(row, 0, name_item)
            if alias == ROUND_FLAG_ALIAS:
                name_item.setText("shape")
                box = QCheckBox("circular")
                box.setChecked(bool(spec.get("default", 1)))
                box.setToolTip(spec.get("help",
                                        "circular or rectangular outline"))
                self.setCellWidget(row, 1, box)
                self._round_box = box
            else:
                self.setItem(row, 1, QTableWidgetItem(
                    "%g" % spec.get("default", 0.0)))
            unit_item = QTableWidgetItem(spec.get("unit", ""))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, 2, unit_item)
        self.resizeColumnToContents(0)

    def set_value(self, alias, value):
        if alias not in self._aliases:
            return
        row = self._aliases.index(alias)
        if alias == ROUND_FLAG_ALIAS and self._round_box is not None:
            self._round_box.setChecked(bool(value))
        else:
            self.item(row, 1).setText("%.6g" % float(value))

    def values(self):
        out = {}
        for row, alias in enumerate(self._aliases):
            if alias == ROUND_FLAG_ALIAS and self._round_box is not None:
                out[alias] = 1.0 if self._round_box.isChecked() else 0.0
                continue
            item = self.item(row, 1)
            try:
                out[alias] = float(item.text())
            except (TypeError, ValueError):
                pass
        return out

    def changed_values(self):
        defaults = {a: s.get("default") for a, s in self._params.items()}
        return {a: v for a, v in self.values().items()
                if defaults.get(a) is None
                or abs(v - float(defaults[a])) > 1e-12}


class ApodizationEditor(QWidget):
    """Composite editor for a source's `apodization` property: a kind
    combo ('none'/'gaussian') plus w0 [mm] and order fields, shown/enabled
    only for 'gaussian'. Produces the canonical
    'gaussian:w0=<mm>[:order=<n>]' string (order omitted when 1), or None
    for 'off'.

    Mirrors scripts/common.py's parse_apodization_spec grammar loosely
    for live editing convenience only -- deliberately does NOT import
    scripts/common (this pane stays script-free); the pipeline's own
    parser at extraction time is the actual authority and will reject
    anything this widget can't already represent."""

    def __init__(self, spec=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["none", "gaussian"])
        lay.addWidget(self.kind_combo)
        lay.addWidget(QLabel("w0 [mm]:"))
        self.w0_edit = QLineEdit()
        self.w0_edit.setValidator(QDoubleValidator(0.0, 1e6, 6))
        self.w0_edit.setFixedWidth(70)
        lay.addWidget(self.w0_edit)
        lay.addWidget(QLabel("order:"))
        self.order_edit = QLineEdit("1")
        self.order_edit.setFixedWidth(40)
        lay.addWidget(self.order_edit)
        lay.addStretch(1)
        self.kind_combo.currentTextChanged.connect(self._update_enabled)
        self.set_spec(spec)

    def _update_enabled(self, _text=None):
        on = self.kind_combo.currentText() == "gaussian"
        self.w0_edit.setEnabled(on)
        self.order_edit.setEnabled(on)

    def set_spec(self, spec):
        text = str(spec).strip() if spec else ""
        if not text or text.lower() == "none":
            self.kind_combo.setCurrentText("none")
            self.w0_edit.clear()
            self.order_edit.setText("1")
            self._update_enabled()
            return
        parts = text.split(":")
        kind = parts[0].strip().lower()
        self.kind_combo.setCurrentText("gaussian" if kind == "gaussian"
                                       else "none")
        w0, order = "", "1"
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, _, val = part.partition("=")
            key = key.strip().lower()
            if key == "w0":
                w0 = val.strip()
            elif key == "order":
                order = val.strip()
        self.w0_edit.setText(w0)
        self.order_edit.setText(order or "1")
        self._update_enabled()

    def spec(self):
        """The canonical spec string, or None when 'off'/incomplete."""
        if self.kind_combo.currentText() != "gaussian":
            return None
        w0_text = self.w0_edit.text().strip()
        if not w0_text:
            return None
        try:
            w0 = float(w0_text)
        except ValueError:
            return None
        if w0 <= 0:
            return None
        order_text = self.order_edit.text().strip()
        try:
            order = int(order_text) if order_text else 1
        except ValueError:
            order = 1
        if order < 1:
            order = 1
        if order == 1:
            return "gaussian:w0=%g" % w0
        return "gaussian:w0=%g:order=%d" % (w0, order)


class PropertiesFormWidget(QTableWidget):
    """Device properties: Name [unit] | Value, one editor per row."""

    def __init__(self, rows, registry_names=None, parent=None):
        """rows: property_rows_for() output. registry_names: callable
        (registry_kind) -> [names] for combo population (or None)."""
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Property", "Value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self._rows = list(rows or ())
        self._editors = {}
        self.setRowCount(len(self._rows))
        for row, (name, kind, default, tip) in enumerate(self._rows):
            name_item = QTableWidgetItem(label_with_unit(name))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setToolTip(tip)
            self.setItem(row, 0, name_item)
            editor = self._make_editor(name, kind, default, registry_names)
            editor.setToolTip(tip)
            self.setCellWidget(row, 1, editor)
            self._editors[name] = (kind, editor, default)
        self.resizeColumnToContents(0)
        # beam_waist gates m2 (M^2 "only meaningful with a beam waist
        # set"): wire the cross-row enable state once both rows exist,
        # then sync to whatever the initial/prefilled values are.
        if "beam_waist" in self._editors and "m2" in self._editors:
            _bw_kind, bw_editor, _bw_default = self._editors["beam_waist"]
            _m2_kind, m2_editor, _m2_default = self._editors["m2"]
            def _sync_m2_enabled(_text=None, bw=bw_editor, m2=m2_editor):
                m2.setEnabled(bool(bw.text().strip()))
            bw_editor.textChanged.connect(_sync_m2_enabled)
            _sync_m2_enabled()

    def _make_editor(self, name, kind, default, registry_names):
        if kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(default))
            return box
        if kind == "choice":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(list(_POLARIZATION_CHOICES))
            combo.setCurrentText(str(default))
            return combo
        if kind == "registry":
            combo = QComboBox()
            combo.setEditable(True)
            names = []
            reg = _REGISTRY_KINDS.get(name)
            if registry_names is not None and reg:
                try:
                    names = sorted(registry_names(reg) or [])
                except Exception:
                    names = []
            combo.addItems(names)
            combo.setCurrentText(str(default))
            return combo
        if kind == "float_optional":
            edit = QLineEdit("" if default in (None, "")
                             else "%g" % float(default))
            edit.setValidator(QDoubleValidator())
            return edit
        if kind == "apod":
            return ApodizationEditor(default)
        edit = QLineEdit(str(default) if default is not None else "")
        if kind == "float":
            edit.setValidator(QDoubleValidator())
            edit.setText("%g" % float(default))
        return edit

    def set_value(self, name, value):
        entry = self._editors.get(name)
        if entry is None:
            return
        kind, editor, _default = entry
        if kind == "bool":
            editor.setChecked(bool(value))
        elif kind in ("choice", "registry"):
            editor.setCurrentText(str(value))
        elif kind == "float_optional":
            editor.setText("" if value in (None, "")
                           else "%g" % float(value))
        elif kind == "apod":
            editor.set_spec(value)
        else:
            editor.setText("%g" % float(value) if kind == "float"
                           else str(value))

    def values(self):
        out = {}
        for name, (kind, editor, _default) in self._editors.items():
            if kind == "bool":
                out[name] = editor.isChecked()
            elif kind in ("choice", "registry"):
                out[name] = editor.currentText().strip()
            elif kind == "float":
                try:
                    out[name] = float(editor.text())
                except (TypeError, ValueError):
                    pass
            elif kind == "float_optional":
                text = editor.text().strip()
                if text:
                    try:
                        out[name] = float(text)
                    except ValueError:
                        pass
                # blank -> "off": omit the key entirely (no value to write)
            elif kind == "apod":
                spec = editor.spec()
                if spec is not None:
                    out[name] = spec
                # 'none'/incomplete -> omit the key entirely
            else:
                out[name] = editor.text().strip()
        return out

    def changed_values(self):
        out = {}
        for name, value in self.values().items():
            _kind, _editor, default = self._editors[name]
            if isinstance(value, float) and isinstance(default, (int, float)):
                if abs(value - float(default)) > 1e-12:
                    out[name] = value
            elif value != default:
                out[name] = value
        return out


# roles offered by the type-first chooser -> primitive categories
ROLE_CATEGORIES = (
    ("Light source", "Lasers and broadband emitters (a body with power "
     "and a center wavelength; rays start here)", ("Sources",)),
    ("Detector", "A measurement plane that records irradiance images "
     "and spectra (transparent unless given reflectivity)",
     ("Detectors",)),
    ("Optical element", "Lenses, mirrors, prisms, beamsplitters, "
     "filters, polarizers... (anything rays interact with)", None),
    ("Generic solid", "A plain box/cylinder/sphere to customize with "
     "materials and per-face properties", ("Generic",)),
)
_NON_OPTIC_CATEGORIES = {"Sources", "Detectors", "Generic"}


class TypeChooserDialog(QDialog):
    """Step 1 of the add flow when nothing is selected in the Library:
    choose WHAT to add (with explanations), then WHICH primitive.
    Returns the primitive info via chosen_info()."""

    def __init__(self, primitives, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add element")
        self._primitives = list(primitives or ())
        self._info = None

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("What do you want to add?"))
        self._role_buttons = []
        for title, tip, _cats in ROLE_CATEGORIES:
            btn = QRadioButton(title)
            btn.setToolTip(tip)
            btn.toggled.connect(self._refill)
            lay.addWidget(btn)
            note = QLabel("    " + tip)
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            lay.addWidget(note)
            self._role_buttons.append(btn)

        lay.addWidget(QLabel("Element type:"))
        self.list = QListWidget()
        self.list.setToolTip("Primitives available for the chosen role")
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.list, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Configure…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._role_buttons[2].setChecked(True)   # optical element
        self.resize(460, 520)

    def _refill(self):
        idx = next((i for i, b in enumerate(self._role_buttons)
                    if b.isChecked()), 2)
        _title, _tip, cats = ROLE_CATEGORIES[idx]
        self.list.clear()
        for info in sorted(self._primitives,
                           key=lambda i: (i.get("category", ""),
                                          i.get("label", ""))):
            category = info.get("category", "")
            if cats is None:
                if category in _NON_OPTIC_CATEGORIES:
                    continue
            elif category not in cats:
                continue
            item = QListWidgetItem("%s — %s" % (info.get("label", "?"),
                                                category))
            item.setToolTip(info.get("tooltip", ""))
            item.setData(Qt.UserRole, info)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def accept(self):
        item = self.list.currentItem()
        if item is None:
            return
        self._info = item.data(Qt.UserRole)
        super().accept()

    def chosen_info(self):
        return self._info
