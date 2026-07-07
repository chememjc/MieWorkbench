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
  b) Active Properties - the per-face "facemap" properties (coating/
     roughness/diffuser/grating/surface_override, see scripts/common.py's
     parse_facemap_spec) shown ASSIGNMENT-centrically: one row per
     (property, value) pair with the face names it covers. The faces cell
     opens a checkable per-face menu (add/remove faces from the
     assignment); the value cell is a registry-fed dropdown (typed escape
     hatch included). With a face selection active (fed by
     set_face_selection(body, faces) from InspectorPane, or by selecting
     assignment rows here) the table filters to assignments touching the
     selection. A right-click (here or in the inspector's 3D view, via
     build_active_properties_menu) opens the property -> value apply/
     remove menu tree with checkmarks. All set-arithmetic lives in
     core/facemaps.py (pure, oracle-checked).
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

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMenu, QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core import facemaps
from ..core.facemaps import (   # noqa: F401  (re-exported for back-compat)
    FACEMAP_PROPERTIES, active_face_index, merge_facemap,
    validate_facemap_value,
)
from ..core.units import label_with_unit

DEFAULT_OPTPROPS_ROOT = "/home3/raytracegui/opticalproperties"

CONTRACT_PROPERTIES = (
    "material", "power", "lambdac", "lambdamin", "lambdamax", "coherent",
    "polarization", "coating", "roughness", "diffuser", "filter",
    "polarizer", "polarizer_axis", "crystal_axis", "grating",
    "surface_override", "mirror", "absorbance",
)
REGISTRY_PROPERTIES = ("material", "polarizer", "filter", "coating",
                       "grating", "diffuser")
NUMERIC_PROPERTIES = ("power", "lambdac", "lambdamin", "lambdamax", "mirror",
                      "absorbance")
BOOL_PROPERTIES = ("coherent",)

# offered in the Active Properties value dropdowns alongside registry rows
ROUGHNESS_PRESETS = ("10", "50", "200")
DIFFUSER_TEMPLATES = ("grit:120", "slope:0.08")

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
    "diffuser": "Ground-glass diffuser: 'grit:120' | 'slope:0.08' | "
                "'@dg_600' (whole body or per-face map 'FaceN=...'); "
                "mutually exclusive with roughness on the same face.",
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
# Pure logic (no Qt) - unit-tested directly. The per-face assignment
# arithmetic (merge_facemap and friends) lives in core/facemaps.py and is
# re-exported above.
# ---------------------------------------------------------------------------
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
        self._prop_library_provider = None

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
        # hand-typed per-face values ('Face3=MgF2;...') are error-checked
        # against the contract grammar AND the body's real face count
        # before anything reaches the worker
        if name in FACEMAP_PROPERTIES and value and \
                str(value).lstrip().startswith("Face"):
            faces_meta = (self._project.faces.get(self._body_name, {})
                          .get("faces", []))
            err = validate_facemap_value(value, self._body_name,
                                         body.get("tip"),
                                         len(faces_meta))
            if err:
                self._show_face_warning("Invalid %s value: %s"
                                        % (name, err))
                self._refresh_timer.start()   # restore the shown value
                return
            self._show_face_warning(None)
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

    def set_prop_library(self, provider):
        """`provider`: zero-arg callable returning the ACTIVE
        core.proplib.PropLibrary (the project library when a .MieWB is
        open, else the system one) -- the mainwindow injects this so every
        dropdown reflects the registries the trace will actually use. The
        module-level loader (hardcoded repo root) remains as a fallback
        when unset or failing."""
        self._prop_library_provider = provider

    def _library_categories(self):
        """{"materials": [...], "coatings": [...], ...} from the injected
        PropLibrary, falling back to the legacy direct loader."""
        if self._prop_library_provider is not None:
            try:
                return self._prop_library_provider().categories()
            except Exception:
                pass
        props = self._get_optprops()
        if props is None:
            return {}
        return {
            "materials": list(props.matdb),
            "coatings": list(props.coatings),
            "polarizers": list(props.polarizers),
            "filters": list(props.filters),
            "gratings": list(props.gratings),
            "diffusers": list(getattr(props, "diffusers", {}) or {}),
        }

    _REGISTRY_CATEGORY = {"polarizer": "polarizers", "filter": "filters",
                          "coating": "coatings", "grating": "gratings",
                          "diffuser": "diffusers"}

    def _material_names(self):
        return list(self._library_categories().get("materials", []))

    def _registry_names(self, prop_name):
        cats = self._library_categories()
        names = list(cats.get(self._REGISTRY_CATEGORY.get(prop_name, ""), []))
        if prop_name == "diffuser":
            # diffuser registry rows are referenced as '@name'; the two
            # template entries cover the direct grit/slope grammar
            return ["@%s" % n for n in names] + list(DIFFUSER_TEMPLATES)
        return names

    def _facemap_value_options(self, prop_name):
        """Dropdown options for a per-face property VALUE (stored form)."""
        if prop_name == "coating":
            return self._registry_names("coating")
        if prop_name == "grating":
            return ["@%s" % n for n in
                    self._library_categories().get("gratings", [])]
        if prop_name == "diffuser":
            return self._registry_names("diffuser")
        if prop_name == "roughness":
            return list(ROUGHNESS_PRESETS)
        return []   # surface_override: exotic, typed by design

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

    # -- section (b): Active Properties (per-face assignments) ----------------
    def _build_faces_box(self):
        self.faces_box = QGroupBox("Active Properties")
        # one row per (property, value) ASSIGNMENT with the faces it
        # covers -- not one row per face. With a face selection active
        # (3D picks in the inspector, or assignment rows here) the table
        # filters to assignments touching the selection.
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)

        self.assign_table = QTableWidget(0, 4)
        self.assign_table.setHorizontalHeaderLabels(
            ["Property", "Value", "Faces", ""])
        self.assign_table.verticalHeader().setVisible(False)
        self.assign_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.assign_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.assign_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.assign_table.itemSelectionChanged.connect(
            self._on_assignment_selection)
        self.assign_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.assign_table.customContextMenuRequested.connect(
            self._on_table_context_menu)
        self._assign_table_updating = False

        # add-assignment row: property dropdown + context-sensitive value
        # dropdown (registry rows / presets; the editable line edit is the
        # typed escape hatch) -> applies to the selected faces, or the
        # whole body when nothing is selected
        self.facemap_prop_combo = QComboBox()
        self.facemap_prop_combo.addItems(list(FACEMAP_PROPERTIES))
        self.facemap_prop_combo.currentTextChanged.connect(
            self._on_facemap_prop_changed)
        self.facemap_value_combo = QComboBox()
        self.facemap_value_combo.setEditable(True)
        self.facemap_assign_button = QPushButton("Assign")
        self.facemap_assign_button.setToolTip(
            "Assign the value to every selected face (pick faces in the "
            "Element Inspector's 3D view or select assignment rows "
            "above); with no selection it applies to the whole body")
        self.facemap_assign_button.clicked.connect(self._on_assign_facemap)

        assign_row = QHBoxLayout()
        assign_row.addWidget(self.facemap_prop_combo)
        assign_row.addWidget(self.facemap_value_combo, 1)
        assign_row.addWidget(self.facemap_assign_button)

        self.face_warning = QLabel("")
        self.face_warning.setStyleSheet("color: #b91c1c;")
        self.face_warning.setWordWrap(True)
        self.face_warning.hide()

        layout = QVBoxLayout()
        layout.addWidget(self.selection_label)
        layout.addWidget(self.assign_table)
        layout.addLayout(assign_row)
        layout.addWidget(self.face_warning)
        self.faces_box.setLayout(layout)
        self._on_facemap_prop_changed(self.facemap_prop_combo.currentText())

    facesPicked = Signal(str, set)   # body, faces chosen in the table

    def set_face_selection(self, body_name, faces):
        """Slot: wire InspectorPane.faceSelectionChanged straight in.
        Also tracks which body is "current" for sections (a)/(c) -- a
        body change refreshes those too. Any selection change re-filters
        the Active Properties table."""
        body_changed = body_name != self._body_name
        self._body_name = body_name
        self._face_selection = set(faces or [])
        self._show_face_warning(None)
        if body_changed:
            self._refresh_properties()
            self._refresh_sheet_table()
        self._refresh_assignments_table()

    # -- assignment data helpers ----------------------------------------------
    def _faces_meta(self):
        if self._project is None or self._body_name is None:
            return []
        return self._project.faces.get(self._body_name, {}).get("faces", [])

    def _all_face_ids(self):
        return [f["id"] for f in self._faces_meta()]

    def _current_assignments(self):
        """(assignments, all_face_ids) for the current body."""
        if self._project is None or self._body_name is None:
            return [], []
        body = self._project.body(self._body_name)
        all_ids = self._all_face_ids()
        assignments = facemaps.assignments_for_body(
            body.get("properties"), self._body_name, body.get("tip"),
            all_ids)
        return assignments, all_ids

    def _face_marker(self):
        """(active_face_index, ' (emit)'/' (detector)') for the current
        body, or (None, '')."""
        if self._project is None or self._body_name is None:
            return None, ""
        body = self._project.body(self._body_name)
        props = body.get("properties") or {}
        active = active_face_index(props, self._faces_meta())
        if active is None:
            return None, ""
        is_source = "power" in props and "lambdac" in props
        return active, (" (emit)" if is_source else " (detector)")

    def _face_display(self, face_ids, whole_body=False):
        if whole_body:
            return "whole body"
        if not face_ids:
            return "—"
        active, marker = self._face_marker()
        labels = []
        for fid in facemaps.sorted_face_ids(face_ids):
            label = facemaps.face_label(fid)
            if active is not None and \
                    common.parse_face_spec(fid)["face_index"] == active:
                label += marker
            labels.append(label)
        return ", ".join(labels)

    def _update_selection_label(self):
        if self._face_selection:
            self.selection_label.setText(
                "Showing properties on: %s — clear the face selection to "
                "see all" % self._face_display(self._face_selection))
        else:
            self.selection_label.setText(
                "All assignments (no faces selected — new assignments "
                "apply to the whole body)")

    # -- assignment table -------------------------------------------------------
    def _refresh_assignments_table(self):
        self._assign_table_updating = True
        try:
            self.assign_table.setRowCount(0)
            self._update_selection_label()
            if self._project is None or self._body_name is None:
                return
            assignments, _all_ids = self._current_assignments()
            shown = facemaps.filter_assignments(assignments,
                                                self._face_selection)
            self.assign_table.setRowCount(len(shown))
            for row, a in enumerate(shown):
                invalid = facemaps.assignment_is_invalid(a)
                prop_item = QTableWidgetItem(
                    a.prop + (" (invalid)" if invalid else ""))
                prop_item.setData(Qt.ItemDataRole.UserRole, a)
                prop_item.setToolTip(TOOLTIPS.get(a.prop, a.prop))
                if invalid:
                    prop_item.setForeground(Qt.GlobalColor.red)
                self.assign_table.setItem(row, 0, prop_item)
                self.assign_table.setCellWidget(
                    row, 1, self._make_assignment_value_editor(a))

                faces_btn = QPushButton(
                    self._face_display(a.face_ids, a.whole_body))
                faces_btn.setFlat(True)
                faces_btn.setStyleSheet("text-align: left;")
                faces_btn.setToolTip(
                    "The faces this %s applies to — click to add/remove "
                    "faces (checkboxes)" % a.prop)
                faces_btn.clicked.connect(
                    lambda _c=False, a=a, b=faces_btn:
                        self._on_faces_cell_clicked(a, b))
                if invalid:
                    faces_btn.setEnabled(False)
                self.assign_table.setCellWidget(row, 2, faces_btn)

                remove_btn = QPushButton("Remove")
                remove_btn.setToolTip(
                    "Remove this %s assignment entirely" % a.prop)
                remove_btn.clicked.connect(
                    lambda _c=False, a=a: self._on_remove_assignment(a))
                self.assign_table.setCellWidget(row, 3, remove_btn)
            self.assign_table.resizeColumnToContents(0)
            self.assign_table.resizeColumnToContents(3)
        finally:
            self._assign_table_updating = False

    def _make_assignment_value_editor(self, assignment):
        if facemaps.assignment_is_invalid(assignment):
            edit = QLineEdit(assignment.value)
            edit.setStyleSheet("color: #b91c1c;")
            edit.setToolTip(
                "This value string doesn't parse under the per-face "
                "grammar — fix it here or Remove the property")
            edit.editingFinished.connect(
                lambda p=assignment.prop, w=edit:
                    self._commit_property(p, w.text()))
            return edit
        options = self._facemap_value_options(assignment.prop)
        if options:
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(options)
            combo.setCurrentText(assignment.value)
            combo.lineEdit().editingFinished.connect(
                lambda a=assignment, c=combo:
                    self._on_assignment_value_edit(a, c.currentText()))
            combo.activated.connect(
                lambda _idx, a=assignment, c=combo:
                    self._on_assignment_value_edit(a, c.currentText()))
            return combo
        edit = QLineEdit(assignment.value)
        edit.editingFinished.connect(
            lambda a=assignment, w=edit:
                self._on_assignment_value_edit(a, w.text()))
        return edit

    def _on_assignment_selection(self):
        """Selecting assignment rows selects THEIR faces (union) -- the
        'show me where that coating is' gesture. Deliberately does not
        re-filter the table (rebuilding it mid-interaction would collapse
        the view under the user's pointer); 3D picks do the filtering."""
        if self._assign_table_updating or self._body_name is None:
            return
        faces = set()
        for item in self.assign_table.selectedItems():
            if item.column() == 0:
                a = item.data(Qt.ItemDataRole.UserRole)
                if a is not None:
                    faces |= set(a.face_ids)
        self._face_selection = faces
        self._update_selection_label()
        self.facesPicked.emit(self._body_name, set(faces))

    # -- commits ---------------------------------------------------------------
    def _facemap_raw(self, prop_name):
        body = self._project.body(self._body_name)
        return (body.get("properties", {}) or {}).get(
            prop_name, {}).get("value")

    def _commit_facemap_merge(self, prop_name, target_faces, value):
        """Merge `value` onto `target_faces` for `prop_name` and commit --
        one undoable set_property. Returns True on success."""
        body = self._project.body(self._body_name)
        try:
            new_raw = merge_facemap(
                existing_raw=self._facemap_raw(prop_name),
                body_name=self._body_name, feature=body.get("tip"),
                all_face_ids=self._all_face_ids(),
                selected_face_ids=target_faces, value=value,
                collapse=facemaps.facemap_collapse_allowed(prop_name))
        except ValueError as exc:
            self._show_face_warning("Invalid %s value: %s"
                                    % (prop_name, exc))
            return False
        self._show_face_warning(None)
        self._project.set_property(self._body_name, prop_name, new_raw)
        return True

    def _commit_facemap_removal(self, prop_name, target_faces):
        """Remove `target_faces` from `prop_name`'s map; removing the
        last face removes the property itself. One undoable step."""
        body = self._project.body(self._body_name)
        try:
            new_raw = facemaps.remove_faces(
                self._facemap_raw(prop_name), self._body_name,
                body.get("tip"), self._all_face_ids(), target_faces,
                collapse=facemaps.facemap_collapse_allowed(prop_name))
        except ValueError as exc:
            self._show_face_warning("Could not update %s: %s"
                                    % (prop_name, exc))
            return False
        self._show_face_warning(None)
        if new_raw is None:
            self._project.remove_property(self._body_name, prop_name)
        else:
            self._project.set_property(self._body_name, prop_name, new_raw)
        return True

    def _on_assignment_value_edit(self, assignment, new_value):
        if self._project is None or self._body_name is None:
            return
        new_value = str(new_value).strip()
        if not new_value or new_value == assignment.value:
            return
        if not self._commit_facemap_merge(assignment.prop,
                                          assignment.face_ids, new_value):
            self._refresh_timer.start()   # restore the shown value

    def _on_faces_cell_clicked(self, assignment, button):
        """Checkable per-face menu: toggle a face's membership of this
        assignment -- zero typing to re-scope it."""
        if self._project is None or self._body_name is None:
            return
        menu = QMenu(self)
        active, marker = self._face_marker()
        for f in self._faces_meta():
            fid = f["id"]
            idx = common.parse_face_spec(fid)["face_index"]
            label = "Face%d" % idx
            if active is not None and idx == active:
                label += marker
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(fid in assignment.face_ids)
            act.triggered.connect(
                lambda checked, a=assignment, fid=fid:
                    self._on_assignment_face_toggled(a, fid, checked))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _on_assignment_face_toggled(self, assignment, face_id, member):
        if member:
            self._commit_facemap_merge(assignment.prop, {face_id},
                                       assignment.value)
        else:
            self._commit_facemap_removal(assignment.prop, {face_id})

    def _on_remove_assignment(self, assignment):
        if self._project is None or self._body_name is None:
            return
        if facemaps.assignment_is_invalid(assignment) or \
                assignment.whole_body:
            self._project.remove_property(self._body_name, assignment.prop)
            return
        self._commit_facemap_removal(assignment.prop, assignment.face_ids)

    def _show_face_warning(self, message):
        if message:
            self.face_warning.setText(message)
            self.face_warning.show()
        else:
            self.face_warning.clear()
            self.face_warning.hide()

    def _on_facemap_prop_changed(self, prop_name):
        self.facemap_value_combo.clear()
        self.facemap_value_combo.addItems(
            self._facemap_value_options(prop_name))
        self.facemap_value_combo.setCurrentText("")
        self.facemap_value_combo.lineEdit().setPlaceholderText(
            TOOLTIPS.get(prop_name, ""))

    def _on_assign_facemap(self):
        if self._project is None or self._body_name is None:
            return
        value = self.facemap_value_combo.currentText().strip()
        if not value:
            return
        prop_name = self.facemap_prop_combo.currentText()
        target = set(self._face_selection) or set(self._all_face_ids())
        if not target:
            self._show_face_warning("The body has no faces to assign to")
            return
        self._commit_facemap_merge(prop_name, target, value)

    # -- "Active Properties" context menu ---------------------------------------
    def build_active_properties_menu(self, parent=None):
        """The property → value → apply/remove menu tree, shown on
        right-click over the assignment table AND (via the mainwindow)
        over the Element Inspector's 3D view. Checked = the value is on
        every selected face (click removes it there); italic = on some
        of them (click applies to all); unchecked = click applies. With
        no face selection the target is the whole body. Returns None
        when no body is current."""
        if self._project is None or self._body_name is None:
            return None
        assignments, all_ids = self._current_assignments()
        registry = {p: self._facemap_value_options(p)
                    for p in FACEMAP_PROPERTIES}
        model = facemaps.menu_model(assignments, self._face_selection,
                                    registry, all_ids)
        menu = QMenu(parent or self)
        scope = (self._face_display(self._face_selection)
                 if self._face_selection else "whole body")
        title = menu.addAction("Active Properties — %s" % scope)
        title.setEnabled(False)
        menu.addSeparator()
        # keep the submenu wrappers alive on the menu itself: PySide6's
        # QAction.menu() hands ownership of a NEW wrapper to Python, so
        # anyone (tests, tooling) retrieving a submenu that way would let
        # the GC delete the underlying C++ QMenu. property_submenus is
        # the supported access path.
        menu.property_submenus = {}
        for entry in model:
            sub = menu.addMenu(entry["prop"])
            menu.property_submenus[entry["prop"]] = sub
            sub.setToolTipsVisible(True)
            for item in entry["items"]:
                act = sub.addAction(item["value"])
                act.setCheckable(True)
                act.setChecked(item["checked"])
                if item["partial"]:
                    font = act.font()
                    font.setItalic(True)
                    act.setFont(font)
                    act.setToolTip("On some of the selected faces — "
                                   "click to apply to all of them")
                elif item["checked"]:
                    act.setToolTip("Click to remove from the selected "
                                   "faces")
                act.triggered.connect(
                    lambda _c=False, p=entry["prop"], v=item["value"],
                    was=item["checked"]: self._on_menu_value(p, v, was))
            if entry["items"]:
                sub.addSeparator()
            act = sub.addAction("Custom…")
            act.setToolTip("Type a value by hand (%s)"
                           % TOOLTIPS.get(entry["prop"], entry["prop"]))
            act.triggered.connect(
                lambda _c=False, p=entry["prop"]: self._on_menu_custom(p))
        menu.addSeparator()
        act = menu.addAction("Select all faces")
        act.triggered.connect(self._menu_select_all)
        act = menu.addAction("Clear face selection")
        act.triggered.connect(self._menu_clear_selection)
        return menu

    def _menu_target_faces(self):
        return set(self._face_selection) or set(self._all_face_ids())

    def _on_menu_value(self, prop_name, value, was_fully_applied):
        target = self._menu_target_faces()
        if not target:
            return
        if was_fully_applied:
            self._commit_facemap_removal(prop_name, target)
        else:
            self._commit_facemap_merge(prop_name, target, value)

    def _on_menu_custom(self, prop_name):
        value, ok = QInputDialog.getText(
            self, "Custom %s value" % prop_name,
            TOOLTIPS.get(prop_name, prop_name))
        value = value.strip() if ok and value else ""
        if not value:
            return
        target = self._menu_target_faces()
        if target:
            self._commit_facemap_merge(prop_name, target, value)

    def _menu_select_all(self):
        if self._body_name is None:
            return
        self._face_selection = set(self._all_face_ids())
        self.facesPicked.emit(self._body_name, set(self._face_selection))
        self._refresh_assignments_table()

    def _menu_clear_selection(self):
        if self._body_name is None:
            return
        self._face_selection = set()
        self.facesPicked.emit(self._body_name, set())
        self._refresh_assignments_table()

    def _on_table_context_menu(self, pos):
        menu = self.build_active_properties_menu(self)
        if menu is not None:
            menu.exec(self.assign_table.viewport().mapToGlobal(pos))

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
        self._refresh_assignments_table()
        self._refresh_sheet_table()

    def _refresh_all(self):
        self._refresh_properties()
        self._refresh_assignments_table()
        self._refresh_sheet_table()
