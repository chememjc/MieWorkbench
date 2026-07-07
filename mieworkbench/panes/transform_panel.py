"""Position / Orientation pane (top-right dock).

Translate and rotate the selected element with repeatable operations:
  - translate by a vector, or toward a reference point by a distance;
  - rotate about an axis (global / custom / face normal / optical axis)
    around a reference point (origin default, or any fixed/element point).
Reference points resolve LIVE at apply time (transforms.Operation), so
"Apply again" after other moves keeps meaning "toward the lens" etc.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from ..core.transforms import Operation

_REF_KINDS = [
    ("Origin", "origin"),
    ("Fixed point…", "fixed"),
    ("Element: optical center", "optical_center"),
    ("Element: center of mass", "com"),
    ("Element: bbox center", "bbox_center"),
    ("Element: point on face normal", "face_point"),
]


class ReferencePointPicker(QWidget):
    """Composes a transforms reference-spec dict; shows live coordinates."""

    changed = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(title), 0, 0)
        self.kind = QComboBox()
        for label, _ in _REF_KINDS:
            self.kind.addItem(label)
        self.kind.setToolTip("What the operation is anchored to")
        lay.addWidget(self.kind, 0, 1, 1, 2)

        self.body = QComboBox()
        self.body.setToolTip("Which element the reference point belongs to")
        lay.addWidget(self.body, 1, 1, 1, 2)

        self.face = QComboBox()
        self.face.setToolTip("Face whose centroid+normal define the line")
        lay.addWidget(self.face, 2, 1)
        self.t_mm = QDoubleSpinBox()
        self.t_mm.setRange(-10000.0, 10000.0)
        self.t_mm.setSuffix(" mm")
        self.t_mm.setToolTip("Distance along the face normal from the "
                             "face centroid")
        lay.addWidget(self.t_mm, 2, 2)

        self.point = QLineEdit("0, 0, 0")
        self.point.setToolTip("Fixed point x, y, z in mm")
        lay.addWidget(self.point, 3, 1, 1, 2)

        self.coords = QLabel("→ (0.000, 0.000, 0.000) mm")
        self.coords.setStyleSheet("color: gray;")
        lay.addWidget(self.coords, 4, 1, 1, 2)

        self.kind.currentIndexChanged.connect(self._update_enabled)
        for w in (self.kind, self.body, self.face):
            w.currentIndexChanged.connect(self._emit)
        self.t_mm.valueChanged.connect(self._emit)
        self.point.editingFinished.connect(self._emit)
        self._update_enabled()

    def set_project(self, project):
        self.project = project
        self.refresh_bodies()

    def refresh_bodies(self):
        current = self.body.currentText()
        self.body.blockSignals(True)
        self.body.clear()
        if self.project is not None and self.project.structure:
            for b in self.project.structure["bodies"]:
                self.body.addItem(b["label"], b["name"])
        idx = self.body.findText(current)
        if idx >= 0:
            self.body.setCurrentIndex(idx)
        self.body.blockSignals(False)
        self._refresh_faces()
        self._emit()

    def _refresh_faces(self):
        self.face.blockSignals(True)
        self.face.clear()
        name = self.body.currentData()
        if self.project is not None and name:
            for f in self.project.faces.get(name, {}).get("faces", []):
                self.face.addItem(f["id"].split(".")[-1], f["id"])
        self.face.blockSignals(False)

    def _update_enabled(self):
        kind = _REF_KINDS[self.kind.currentIndex()][1]
        self.body.setEnabled(kind in ("optical_center", "com",
                                      "bbox_center", "face_point"))
        self.face.setEnabled(kind == "face_point")
        self.t_mm.setEnabled(kind == "face_point")
        self.point.setEnabled(kind == "fixed")
        self._emit()

    def spec(self):
        kind = _REF_KINDS[self.kind.currentIndex()][1]
        if kind == "origin":
            return {"kind": "origin"}
        if kind == "fixed":
            try:
                xyz = [float(v) for v in self.point.text().split(",")]
                if len(xyz) != 3:
                    raise ValueError
            except ValueError:
                xyz = [0.0, 0.0, 0.0]
            return {"kind": "fixed", "point_mm": xyz}
        spec = {"kind": kind, "body": self.body.currentData()}
        if kind == "face_point":
            spec["face"] = self.face.currentData()
            spec["t_mm"] = self.t_mm.value()
        return spec

    def _emit(self):
        if self.project is not None:
            try:
                p = self.project.resolver().resolve_point(self.spec())
                self.coords.setText("→ (%.3f, %.3f, %.3f) mm"
                                    % (p[0], p[1], p[2]))
            except Exception:
                self.coords.setText("→ (unresolved)")
        self.changed.emit()

    # body combos also update after scene changes
    def notify_scene_changed(self):
        self.refresh_bodies()


class TransformPanel(QWidget):
    """The dock pane; operates on the externally-selected body."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.body_name = None
        self._last_op = None

        lay = QVBoxLayout(self)
        self.target = QLabel("No element selected")
        self.target.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.target)

        # -- translate ------------------------------------------------------
        tg = QGroupBox("Translate")
        tgl = QGridLayout(tg)
        self.d = []
        for i, axis in enumerate("XYZ"):
            tgl.addWidget(QLabel(axis), 0, 2 * i)
            e = QLineEdit("0")
            e.setValidator(QDoubleValidator())
            e.setMaximumWidth(70)
            e.setToolTip("Translation along global %s in mm" % axis)
            tgl.addWidget(e, 0, 2 * i + 1)
            self.d.append(e)
        btn_t = QPushButton("Apply translation")
        btn_t.setToolTip("Move the element by (dx, dy, dz) mm")
        btn_t.clicked.connect(self._apply_translate)
        tgl.addWidget(btn_t, 1, 0, 1, 6)

        self.toward = ReferencePointPicker("Toward:")
        tgl.addWidget(self.toward, 2, 0, 1, 6)
        row = QHBoxLayout()
        self.toward_dist = QDoubleSpinBox()
        self.toward_dist.setRange(-10000, 10000)
        self.toward_dist.setValue(10.0)
        self.toward_dist.setSuffix(" mm")
        self.toward_dist.setToolTip("Distance to move toward the reference "
                                    "(negative moves away)")
        row.addWidget(self.toward_dist)
        btn_tw = QPushButton("Move toward")
        btn_tw.setToolTip("Move the element the given distance toward the "
                          "reference point (from its own optical center)")
        btn_tw.clicked.connect(self._apply_toward)
        row.addWidget(btn_tw)
        tgl.addLayout(row, 3, 0, 1, 6)
        lay.addWidget(tg)

        # -- rotate ----------------------------------------------------------
        rg = QGroupBox("Rotate")
        rgl = QGridLayout(rg)
        rgl.addWidget(QLabel("Axis"), 0, 0)
        self.axis = QComboBox()
        for label in ("Global X", "Global Y", "Global Z",
                      "Custom vector…", "Selected element optical axis"):
            self.axis.addItem(label)
        self.axis.setToolTip("Rotation axis direction")
        rgl.addWidget(self.axis, 0, 1)
        self.axis_vec = QLineEdit("0, 0, 1")
        self.axis_vec.setToolTip("Custom axis vector x, y, z")
        self.axis_vec.setEnabled(False)
        rgl.addWidget(self.axis_vec, 0, 2)
        self.axis.currentIndexChanged.connect(
            lambda i: self.axis_vec.setEnabled(i == 3))
        rgl.addWidget(QLabel("Angle"), 1, 0)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-360.0, 360.0)
        self.angle.setValue(10.0)
        self.angle.setSuffix(" °")
        self.angle.setToolTip("Rotation angle in degrees "
                              "(right-hand rule about the axis)")
        rgl.addWidget(self.angle, 1, 1)
        self.about = ReferencePointPicker("About:")
        rgl.addWidget(self.about, 2, 0, 1, 3)
        btn_r = QPushButton("Apply rotation")
        btn_r.setToolTip("Rotate the element about the reference point")
        btn_r.clicked.connect(self._apply_rotate)
        rgl.addWidget(btn_r, 3, 0, 1, 3)
        lay.addWidget(rg)

        # -- repeat / history --------------------------------------------------
        row = QHBoxLayout()
        self.btn_again = QPushButton("Apply again")
        self.btn_again.setToolTip("Re-apply the last operation (references "
                                  "resolve at their live positions)")
        self.btn_again.setEnabled(False)
        self.btn_again.clicked.connect(self._apply_again)
        row.addWidget(self.btn_again)
        lay.addLayout(row)
        self.history = QListWidget()
        self.history.setToolTip("Operations applied this session")
        self.history.setMaximumHeight(110)
        lay.addWidget(self.history)
        lay.addStretch(1)

    # -- wiring ----------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.toward.set_project(project)
        self.about.set_project(project)
        project.sceneLoaded.connect(self.toward.notify_scene_changed)
        project.sceneLoaded.connect(self.about.notify_scene_changed)
        project.bodiesMoved.connect(
            lambda _d: (self.toward._emit(), self.about._emit()))

    def set_body(self, body_name):
        self.body_name = body_name
        if self.project is not None and body_name:
            b = self.project.body(body_name)
            note = ""
            if b.get("placement_bound"):
                note = "  (position driven by a spreadsheet expression)"
            self.target.setText("Element: %s%s" % (b["label"], note))
        else:
            self.target.setText("No element selected")

    # -- operations ---------------------------------------------------------------
    def _axis_spec(self):
        i = self.axis.currentIndex()
        if i < 3:
            return {"kind": "global", "axis": "xyz"[i]}
        if i == 3:
            vec = [float(v) for v in self.axis_vec.text().split(",")]
            return {"kind": "vector", "vector": vec}
        return {"kind": "optical_axis", "body": self.body_name}

    def _apply(self, op, label):
        if self.project is None or not self.body_name:
            QMessageBox.information(self, "MieWorkbench",
                                    "Select an element first.")
            return
        try:
            self.project.apply_operation(self.body_name, op)
        except Exception as exc:
            QMessageBox.warning(self, "MieWorkbench", str(exc))
            return
        self._last_op = (self.body_name, op, label)
        self.btn_again.setEnabled(True)
        self.history.addItem("%s: %s" % (self.body_name, label))
        self.history.scrollToBottom()

    def _apply_translate(self):
        try:
            v = [float(e.text() or 0) for e in self.d]
        except ValueError:
            QMessageBox.warning(self, "MieWorkbench",
                                "Translation components must be numbers.")
            return
        self._apply(Operation("translate", {"vector_mm": v}),
                    "translate (%g, %g, %g) mm" % tuple(v))

    def _apply_toward(self):
        spec = self.toward.spec()
        dist = self.toward_dist.value()
        op = Operation("translate", {
            "from": {"kind": "optical_center", "body": self.body_name},
            "toward": spec, "distance_mm": dist})
        self._apply(op, "move %g mm toward %s" % (dist, spec["kind"]))

    def _apply_rotate(self):
        try:
            axis = self._axis_spec()
        except ValueError:
            QMessageBox.warning(self, "MieWorkbench",
                                "Axis vector must be three numbers.")
            return
        op = Operation("rotate", {"axis": axis,
                                  "angle_deg": self.angle.value(),
                                  "about": self.about.spec()})
        self._apply(op, "rotate %g° about %s"
                    % (self.angle.value(), self.about.spec()["kind"]))

    def _apply_again(self):
        if self._last_op is None:
            return
        body, op, label = self._last_op
        if body != self.body_name:
            # repeating on a different element is a fresh decision
            self._last_op = (self.body_name, op, label)
            body = self.body_name
        self._apply(op, label + " (again)")
