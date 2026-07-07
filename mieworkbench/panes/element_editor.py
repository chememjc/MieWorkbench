"""ElementEditorPane - edits the selected body's optical-contract
properties, per-face assignments, and parameter-sheet aliases through the
Project (this pane never talks to FreeCAD directly, same rule as every
other pane -- see core/project.py's module docstring).

Layout: three QGroupBoxes.
  a) Optical properties (Base tags) - one row per non-internal custom
     property currently on the body (miewb_* properties are internal
     primitive-rebuild bookkeeping and are skipped), plus an "Add
     property..." row. Edits commit on editingFinished/toggled/activated
     -> project.set_property (float for numeric contract props, bool for
     coherent, str otherwise); a per-row Remove button ->
     project.remove_property.
  b) Per-face assignments - assigns coating/roughness/grating/
     surface_override (the four per-face "facemap" properties, see
     scripts/common.py's parse_facemap_spec) onto the CURRENT face
     selection, fed by the slot set_face_selection(body, faces) (wire
     InspectorPane.faceSelectionChanged straight into it). Builds/updates
     the body's facemap string property by merging the new value onto the
     selected faces (see merge_facemap()) and shows a table of every
     current per-face entry across all four properties.
  c) Element parameters - the body's parameter-sheet aliases
     (project.sheet_for_body): each row's raw "=<num> <unit>" (or bare
     "<num>") string is parsed (parse_sheet_raw), the user edits just the
     number, and format_sheet_raw() recomposes the original prefix/unit
     verbatim on commit -> project.set_spreadsheet(...). If the body
     carries a miewb_primitive property (i.e. it was built by
     primitivelib.py), the edit is followed by
     project.rebuild_primitive(miewb_group's value).

project.propertiesChanged is connected once (set_project) and refreshes
whichever section(s) show data for the affected body.

Contract property semantics (cribbed from scripts/extract_geometry.py's
"Body tagging convention" header comment):
  material          materials.csv row name; 'detector' marks a detector
                    body; missing/'none' -> the body is ignored by the
                    tracer.
  power (mW), lambdac (nm)
                    presence of BOTH marks a light-source body.
  lambdamin/lambdamax (nm)
                    optional source spectral bounds.
  coherent          bool, default False (source only).
  coating           per-face registry name (coating/coatings.csv) or a
                    whole-body value; 'none' omits it.
  mirror, absorbance
                    float in [0,1] (extractor clamps + warns otherwise).
  roughness         float RMS nm (whole body) OR a per-face map string
                    'FaceN=sigma_nm[:lcorr=um];...'.
  polarization      source-only; common.parse_polarization_spec grammar
                    ('unpolarized' | 'linear:<deg>' | 'circular:left|
                    right' | 'elliptical:<psi>:<chi>').
  polarizer         registry name (polarizer/polarizers.csv);
                    polarizer_axis - body-local 'x,y,z' axis (default
                    local +z).
  crystal_axis      body-local 'x,y,z' axis (default local +x); any optic.
  filter            registry name (filter/filters.csv), bulk absorber.
  grating           per-face map 'FaceN=600:v;...' or '@registryname';
                    must name specific faces (no whole-body form).
  surface_override  per-face map declaring an analytic surface (e.g. an
                    asphere) in place of the tessellated mesh.
"""

import os
import re
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

try:
    from raytracer.optprops import load_optical_properties  # noqa: E402
except Exception:                                    # pragma: no cover
    load_optical_properties = None

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..core.units import label_with_unit

DEFAULT_OPTPROPS_ROOT = "/home3/raytracegui/opticalproperties"

CONTRACT_PROPERTIES = (
    "material", "power", "lambdac", "lambdamin", "lambdamax", "coherent",
    "polarization", "coating", "roughness", "filter", "polarizer",
    "polarizer_axis", "crystal_axis", "grating", "surface_override",
    "mirror", "absorbance",
)
FACEMAP_PROPERTIES = ("coating", "roughness", "grating", "surface_override")
REGISTRY_PROPERTIES = ("material", "polarizer", "filter", "coating", "grating")
NUMERIC_PROPERTIES = ("power", "lambdac", "lambdamin", "lambdamax", "mirror",
                      "absorbance")
BOOL_PROPERTIES = ("coherent",)

# Sensible starting values written when a property is first added, so a
# freshly added row is never the empty string (an empty `material` used to
# silently demote the body to "ignored" -- and the empty combo's
# focus-out crashed the pane, see _schedule_refresh()).
PROPERTY_DEFAULTS = {
    "power": 5.0,
    "lambdac": 633.0,
    "lambdamin": 450.0,
    "lambdamax": 650.0,
    "mirror": 1.0,
    "absorbance": 1.0,
    "roughness": "50",
    "polarization": "unpolarized",
    "polarizer_axis": "0,0,1",
    "crystal_axis": "1,0,0",
    "coherent": False,
    "surface_override": "",   # exotic, no universal default
}
# Registry-valued properties default to a well-known entry when the
# library has it, else the first name alphabetically.
_REGISTRY_PREFERRED = {
    "material": ("bk7",),
    "coating": ("MgF2", "mgf2"),
    "filter": ("bp_550_40",),
    "polarizer": ("ideal_linear",),
}


def default_registry_value(name, registry_names):
    """Pick the default for a registry-valued property from the loaded
    library names (pure logic -- unit-tested directly)."""
    names = sorted(registry_names or [])
    for preferred in _REGISTRY_PREFERRED.get(name, ()):
        if preferred in names:
            return preferred
    if name == "grating":
        # gratings are per-face maps; a bare registry name is invalid
        return "Face1=@%s" % names[0] if names else "Face1=600:v:orders=-1..1"
    return names[0] if names else ""

TOOLTIPS = {
    "material": "Row name in materials.csv; 'detector' marks a detector "
               "body; missing/'none' means the body is ignored.",
    "power": "Source power in mW (source bodies need power + lambdac).",
    "lambdac": "Source center wavelength in nm.",
    "lambdamin": "Source spectral lower bound in nm (optional).",
    "lambdamax": "Source spectral upper bound in nm (optional).",
    "coherent": "Whether this source's rays interfere coherently.",
    "polarization": "Source polarization: 'unpolarized' | 'linear:<deg>' | "
                    "'circular:left|right' | 'elliptical:<psi>:<chi>'.",
    "coating": "coating/coatings.csv registry name, or a per-face map "
              "'FaceN=name;...'.",
    "roughness": "RMS roughness in nm (whole body), or a per-face map "
                "'FaceN=sigma_nm[:lcorr=um];...'.",
    "filter": "filter/filters.csv registry name (bulk spectral absorber).",
    "polarizer": "polarizer/polarizers.csv registry name.",
    "polarizer_axis": "Body-local 'x,y,z' transmission axis (default "
                      "local +z).",
    "crystal_axis": "Body-local 'x,y,z' crystal axis (default local +x).",
    "grating": "Per-face map 'FaceN=lines_per_mm:groove;...' or "
              "'FaceN=@registryname'; must name specific faces.",
    "surface_override": "Per-face analytic surface override, e.g. "
                        "'FaceN=asphere:R=..;k=..;A4=..;r_max=..'.",
    "mirror": "Specular reflectance fraction in [0, 1].",
    "absorbance": "Absorbed fraction in [0, 1].",
}


# ---------------------------------------------------------------------------
# Pure logic (no Qt) - unit-tested directly.
# ---------------------------------------------------------------------------
def _bare_face_key(face_id):
    """'Lens.Tip.Face3' -> 'Face3'."""
    spec = common.parse_face_spec(face_id)
    return "Face%d" % spec["face_index"]


def _face_sort_key(face_id):
    return common.parse_face_spec(face_id)["face_index"]


def merge_facemap(existing_raw, body_name, feature, all_face_ids,
                  selected_face_ids, value):
    """Merge `value` onto `selected_face_ids` (full 'Body.Feature.FaceN'
    ids) within a per-face property whose current raw string is
    `existing_raw` (falsy if the property doesn't exist yet). Returns the
    new raw string in the bare 'FaceN=value;...' form, collapsed to the
    bare whole-value form when every face of the body ends up mapped to
    the same value (matches common.py's 'apply to every face' shorthand).
    Re-parses the result with common.parse_facemap_spec as an oracle and
    raises ValueError if it doesn't round-trip.
    """
    all_face_ids = set(all_face_ids or [])
    selected_face_ids = set(selected_face_ids or [])
    if existing_raw:
        current = common.parse_facemap_spec(str(existing_raw),
                                            body=body_name, feature=feature)
    else:
        current = {}

    if common.FACEMAP_ALL in current:
        expanded = {fid: current[common.FACEMAP_ALL] for fid in all_face_ids}
    else:
        expanded = dict(current)
    for fid in selected_face_ids:
        expanded[fid] = value

    values_on_all = ({expanded.get(fid) for fid in all_face_ids}
                     if all_face_ids else set())
    if (all_face_ids and len(expanded) == len(all_face_ids)
            and all_face_ids <= set(expanded) and len(values_on_all) == 1):
        new_raw = next(iter(values_on_all))
    else:
        parts = ["%s=%s" % (_bare_face_key(fid), expanded[fid])
                for fid in sorted(expanded, key=_face_sort_key)]
        new_raw = ";".join(parts)

    reparsed = common.parse_facemap_spec(new_raw, body=body_name,
                                         feature=feature)
    if common.FACEMAP_ALL in reparsed:
        check = {fid: reparsed[common.FACEMAP_ALL] for fid in all_face_ids}
    else:
        check = reparsed
    if check != expanded:
        raise ValueError("facemap merge failed to round-trip (%r != %r)"
                         % (check, expanded))
    return new_raw


_SHEET_NUM_RE = re.compile(
    r'^(?P<eq>=?)(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    r'(?P<suffix>.*)$')


def parse_sheet_raw(raw):
    """'=2 mm' -> {"has_eq": True, "number": 2.0, "suffix": " mm"};
    '633' -> {"has_eq": False, "number": 633.0, "suffix": ""}."""
    raw = str(raw)
    m = _SHEET_NUM_RE.match(raw.strip())
    if not m:
        raise ValueError("unrecognized spreadsheet raw value %r" % raw)
    return {"has_eq": m.group("eq") == "=",
           "number": float(m.group("num")),
           "suffix": m.group("suffix")}


def _fmt_num(value):
    value = float(value)
    if value == int(value):
        return str(int(value))
    return "%g" % value


def format_sheet_raw(parsed, new_number):
    """Recompose parse_sheet_raw()'s dict with a new number, preserving
    the original '=' prefix and unit suffix verbatim."""
    prefix = "=" if parsed["has_eq"] else ""
    return "%s%s%s" % (prefix, _fmt_num(new_number), parsed["suffix"])


def _fmt_property_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "%g" % value
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ElementEditorPane(QWidget):
    def __init__(self, parent=None, optprops_root=None):
        super().__init__(parent)
        self._project = None
        self._body_name = None
        self._face_selection = set()
        self._optprops_root = optprops_root or DEFAULT_OPTPROPS_ROOT
        self._optprops = None
        self._optprops_tried = False

        # Refreshes triggered by our own commits are deferred to the next
        # event-loop turn: rebuilding the rows synchronously would delete
        # the very widget whose editingFinished/activated signal is still
        # executing (use-after-free crash). A parented single-shot QTimer
        # both coalesces bursts and dies with the pane (teardown-safe).
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._deferred_refresh)

        self._build_properties_box()
        self._build_faces_box()
        self._build_sheet_box()

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.addWidget(self.props_box)
        body_layout.addWidget(self.faces_box)
        body_layout.addWidget(self.sheet_box)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_all()

    # -- section (a): optical properties --------------------------------------
    def _build_properties_box(self):
        self.props_box = QGroupBox("Optical properties (Base tags)")
        self.props_form = QFormLayout()

        self.add_prop_combo = QComboBox()
        self.add_prop_combo.setEditable(True)
        self.add_prop_combo.addItem("")
        self.add_prop_combo.addItems(list(CONTRACT_PROPERTIES))
        self.add_prop_combo.setCurrentText("")
        self.add_prop_button = QPushButton("Add property")
        self.add_prop_button.setToolTip(
            "Add a new custom property to the body (Add property…)")
        self.add_prop_button.clicked.connect(self._on_add_property)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Add property…"))
        add_row.addWidget(self.add_prop_combo, 1)
        add_row.addWidget(self.add_prop_button)

        layout = QVBoxLayout()
        layout.addLayout(self.props_form)
        layout.addLayout(add_row)
        self.props_box.setLayout(layout)

    def _on_add_property(self):
        name = self.add_prop_combo.currentText().strip()
        if not name or self._project is None or self._body_name is None:
            return
        body = self._project.body(self._body_name)
        if name in (body.get("properties", {}) or {}):
            self.add_prop_combo.setCurrentText("")
            return   # already present; don't clobber its value
        self._project.set_property(self._body_name, name,
                                   self._default_property_value(name))
        self.add_prop_combo.setCurrentText("")

    def _default_property_value(self, name):
        if name == "material":
            return default_registry_value(name, self._material_names())
        if name in REGISTRY_PROPERTIES:
            return default_registry_value(name, self._registry_names(name))
        return PROPERTY_DEFAULTS.get(name, "")

    def _on_remove_property(self, name):
        if self._project is None or self._body_name is None:
            return
        self._project.remove_property(self._body_name, name)

    def _commit_property(self, name, value):
        if self._project is None or self._body_name is None:
            return
        # No-op commits happen constantly (focus-out with an unchanged
        # value; a combo firing both activated and editingFinished for one
        # selection) -- skip them so the scene doesn't go dirty and the
        # rows aren't pointlessly rebuilt.
        body = self._project.body(self._body_name)
        props = body.get("properties", {}) or {}
        if name in props:
            current = props[name].get("value")
            if current == value or (current is not None
                                    and str(current) == str(value)):
                return
        self._project.set_property(self._body_name, name, value)

    def _commit_numeric_property(self, name, widget):
        text = widget.text().strip()
        if not text:
            return   # leave the stored value; refresh restores the display
        try:
            value = float(text)
        except ValueError:
            return   # validator-intermediate text like '-' or '1e'
        self._commit_property(name, value)

    def _material_names(self):
        props = self._get_optprops()
        return list(props.matdb) if props is not None else []

    def _registry_names(self, prop_name):
        props = self._get_optprops()
        if props is None:
            return []
        mapping = {"polarizer": props.polarizers, "filter": props.filters,
                  "coating": props.coatings, "grating": props.gratings}
        return list(mapping.get(prop_name, {}))

    def _get_optprops(self):
        if not self._optprops_tried:
            self._optprops_tried = True
            if load_optical_properties is not None:
                try:
                    self._optprops = load_optical_properties(
                        self._optprops_root)
                except Exception:
                    self._optprops = None
        return self._optprops

    def _make_registry_combo(self, name, value, names):
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(sorted(names))
        combo.setCurrentText("" if value is None else str(value))
        combo.lineEdit().editingFinished.connect(
            lambda n=name, c=combo: self._commit_property(n, c.currentText()))
        combo.activated.connect(
            lambda _idx, n=name, c=combo:
                self._commit_property(n, c.currentText()))
        return combo

    def _make_property_editor(self, name, value):
        if name == "material":
            return self._make_registry_combo(name, value,
                                             self._material_names())
        if name in REGISTRY_PROPERTIES:
            return self._make_registry_combo(name, value,
                                             self._registry_names(name))
        if name in BOOL_PROPERTIES:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(
                lambda checked, n=name: self._commit_property(n, checked))
            return widget
        if name in NUMERIC_PROPERTIES:
            widget = QLineEdit(_fmt_property_value(value))
            widget.setValidator(QDoubleValidator())
            widget.editingFinished.connect(
                lambda n=name, w=widget:
                    self._commit_numeric_property(n, w))
            return widget
        widget = QLineEdit(_fmt_property_value(value))
        widget.editingFinished.connect(
            lambda n=name, w=widget: self._commit_property(n, w.text()))
        return widget

    def _refresh_properties(self):
        while self.props_form.rowCount():
            self.props_form.removeRow(0)
        if self._project is None or self._body_name is None:
            return
        body = self._project.body(self._body_name)
        props = body.get("properties", {}) or {}
        for name in sorted(props):
            if name.startswith("miewb_"):
                continue
            editor = self._make_property_editor(name, props[name].get("value"))
            editor.setToolTip(TOOLTIPS.get(name, name))
            remove_btn = QPushButton("Remove")
            remove_btn.setToolTip("Remove property %r from the body" % name)
            remove_btn.clicked.connect(
                lambda _c=False, n=name: self._on_remove_property(n))
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor, 1)
            row_layout.addWidget(remove_btn)
            label = QLabel(label_with_unit(name))
            label.setToolTip(TOOLTIPS.get(name, name))
            self.props_form.addRow(label, row)

    # -- section (b): per-face assignments -----------------------------------
    def _build_faces_box(self):
        self.faces_box = QGroupBox("Per-face assignments")
        self.face_selection_label = QLabel("No faces selected")

        self.facemap_prop_combo = QComboBox()
        self.facemap_prop_combo.addItems(list(FACEMAP_PROPERTIES))
        self.facemap_value_edit = QLineEdit()
        self.facemap_value_edit.setPlaceholderText(
            "value, e.g. MgF2 / 50:lcorr=5 / 600:v")
        self.facemap_assign_button = QPushButton("Assign to selected faces")
        self.facemap_assign_button.setToolTip(
            "Assign the value to every currently selected face")
        self.facemap_assign_button.clicked.connect(self._on_assign_facemap)

        assign_row = QHBoxLayout()
        assign_row.addWidget(self.facemap_prop_combo)
        assign_row.addWidget(self.facemap_value_edit, 1)
        assign_row.addWidget(self.facemap_assign_button)

        self.facemap_table = QTableWidget(0, 3)
        self.facemap_table.setHorizontalHeaderLabels(
            ["Property", "Face", "Value"])
        self.facemap_table.verticalHeader().setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(self.face_selection_label)
        layout.addLayout(assign_row)
        layout.addWidget(self.facemap_table)
        self.faces_box.setLayout(layout)

    def set_face_selection(self, body_name, faces):
        """Slot: wire InspectorPane.faceSelectionChanged straight in.
        Also tracks which body is "current" for sections (a)/(c) -- a
        body change refreshes those too."""
        body_changed = body_name != self._body_name
        self._body_name = body_name
        self._face_selection = set(faces or [])
        if body_changed:
            self._refresh_properties()
            self._refresh_sheet_table()
        self._refresh_face_selection_label()
        self._refresh_facemap_table()

    def _refresh_face_selection_label(self):
        if not self._face_selection:
            self.face_selection_label.setText("No faces selected")
            return
        try:
            bare = sorted((_face_sort_key(f), _bare_face_key(f))
                         for f in self._face_selection)
            text = ", ".join(b for _, b in bare)
        except ValueError:
            text = ", ".join(sorted(self._face_selection))
        self.face_selection_label.setText("Selected: " + text)

    def _on_assign_facemap(self):
        if (self._project is None or self._body_name is None
                or not self._face_selection):
            return
        value = self.facemap_value_edit.text().strip()
        if not value:
            return
        prop_name = self.facemap_prop_combo.currentText()
        body = self._project.body(self._body_name)
        feature = body.get("tip")
        all_face_ids = [f["id"] for f in
                        self._project.faces.get(self._body_name, {})
                        .get("faces", [])]
        existing_raw = body.get("properties", {}).get(
            prop_name, {}).get("value")
        new_raw = merge_facemap(existing_raw, self._body_name, feature,
                                all_face_ids, self._face_selection, value)
        self._project.set_property(self._body_name, prop_name, new_raw)

    def _refresh_facemap_table(self):
        self.facemap_table.setRowCount(0)
        if self._project is None or self._body_name is None:
            return
        body = self._project.body(self._body_name)
        feature = body.get("tip")
        props = body.get("properties", {}) or {}
        rows = []
        for prop_name in FACEMAP_PROPERTIES:
            raw = props.get(prop_name, {}).get("value")
            if not raw:
                continue
            try:
                parsed = common.parse_facemap_spec(
                    str(raw), body=self._body_name, feature=feature)
            except ValueError:
                continue
            for face_key, value in parsed.items():
                if face_key == common.FACEMAP_ALL:
                    face_label = "ALL"
                else:
                    try:
                        face_label = _bare_face_key(face_key)
                    except ValueError:
                        face_label = face_key
                rows.append((prop_name, face_label, value))
        self.facemap_table.setRowCount(len(rows))
        for r, (prop_name, face_label, value) in enumerate(rows):
            self.facemap_table.setItem(r, 0, QTableWidgetItem(prop_name))
            self.facemap_table.setItem(r, 1, QTableWidgetItem(face_label))
            self.facemap_table.setItem(r, 2, QTableWidgetItem(str(value)))

    # -- section (c): element parameters -------------------------------------
    def _build_sheet_box(self):
        self.sheet_box = QGroupBox("Element parameters")
        self.sheet_table = QTableWidget(0, 3)
        self.sheet_table.setHorizontalHeaderLabels(["Alias", "Value", "Unit"])
        self.sheet_table.verticalHeader().setVisible(False)
        layout = QVBoxLayout()
        layout.addWidget(self.sheet_table)
        self.sheet_box.setLayout(layout)

    def _refresh_sheet_table(self):
        self.sheet_table.setRowCount(0)
        if self._project is None or self._body_name is None:
            return
        sheet = self._project.sheet_for_body(self._body_name)
        if sheet is None:
            return
        aliases = sheet.get("aliases", {}) or {}
        self.sheet_table.setRowCount(len(aliases))
        for row, alias in enumerate(sorted(aliases)):
            entry = aliases[alias]
            raw = entry.get("raw", "")
            try:
                parsed = parse_sheet_raw(raw)
            except ValueError:
                parsed = None
            number_edit = QLineEdit(
                _fmt_num(parsed["number"]) if parsed is not None else raw)
            if parsed is not None:
                number_edit.setValidator(QDoubleValidator())
                number_edit.editingFinished.connect(
                    lambda alias=alias, sheet_label=sheet["label"],
                    parsed=parsed, w=number_edit:
                        self._on_sheet_edit(sheet_label, alias, parsed,
                                           w.text()))
            unit_text = entry.get("unit") or (
                parsed["suffix"].strip() if parsed is not None else "")
            self.sheet_table.setItem(row, 0, QTableWidgetItem(alias))
            self.sheet_table.setCellWidget(row, 1, number_edit)
            self.sheet_table.setItem(row, 2, QTableWidgetItem(unit_text))

    def _on_sheet_edit(self, sheet_label, alias, parsed, new_number_text):
        if self._project is None or not new_number_text.strip():
            return
        try:
            new_number = float(new_number_text)
        except ValueError:
            return
        if new_number == parsed["number"]:
            return   # unchanged focus-out; don't rebuild the primitive
        new_raw = format_sheet_raw(parsed, new_number)
        rebuild_group = None
        if self._body_name is not None:
            body = self._project.body(self._body_name)
            props = body.get("properties", {}) or {}
            if "miewb_primitive" in props:
                rebuild_group = props.get("miewb_group", {}).get("value")
        # one undoable step: undo restores the old raw AND re-derives the
        # geometry (set_element_parameters handles the ordering)
        self._project.set_spreadsheet(sheet_label, alias, new_raw,
                                      rebuild_group=rebuild_group)

    # -- Project wiring -----------------------------------------------------
    def set_project(self, project):
        if self._project is not None:
            try:
                self._project.propertiesChanged.disconnect(
                    self._on_properties_changed)
            except (RuntimeError, TypeError):
                pass
        self._project = project
        if project is not None:
            project.propertiesChanged.connect(self._on_properties_changed)
        self._refresh_all()

    def _on_properties_changed(self, body_hint):
        if self._body_name is None:
            return
        if not body_hint or body_hint == self._body_name:
            # Deferred: this signal is emitted synchronously from Project
            # mutations, i.e. potentially from inside one of our own row
            # widgets' commit signals -- rebuilding rows right now would
            # delete the emitting widget mid-signal (crash).
            self._refresh_timer.start()

    def _deferred_refresh(self):
        if self._body_name is None:
            return
        self._refresh_properties()
        self._refresh_facemap_table()
        self._refresh_sheet_table()

    def _refresh_all(self):
        self._refresh_properties()
        self._refresh_face_selection_label()
        self._refresh_facemap_table()
        self._refresh_sheet_table()
