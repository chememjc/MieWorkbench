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
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ..core.transforms import (
    Operation, euler_from_quat, quat_from_euler, snap_to_axis_ops,
)

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
        self._view = None            # 3D view, set by set_scene_view
        self._snap_axis = None       # (point, unit dir) from the last pick
        self._drag_base = None       # (base_center, base_placement_dict)

        lay = QVBoxLayout(self)
        self.target = QLabel("No element selected")
        self.target.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.target)

        # -- absolute pose + reference readout ------------------------------
        lay.addWidget(self._build_absolute_group())

        # -- snap to optical axis -------------------------------------------
        lay.addWidget(self._build_snap_group())

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

    # -- absolute pose ---------------------------------------------------------
    def _build_absolute_group(self):
        g = QGroupBox("Absolute (world frame)")
        gl = QGridLayout(g)
        self.abs_pos = []
        self.abs_rot = []
        gl.addWidget(QLabel("Pos"), 0, 0)
        for i, axis in enumerate("XYZ"):
            gl.addWidget(QLabel(axis), 0, 1 + 2 * i)
            sb = QDoubleSpinBox()
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(3)
            sb.setMaximumWidth(80)
            sb.setToolTip("World %s position of the element (mm)" % axis)
            gl.addWidget(sb, 0, 2 + 2 * i)
            self.abs_pos.append(sb)
        self.btn_set_pos = QPushButton("Set position")
        self.btn_set_pos.setToolTip("Move the element to the typed world "
                                    "position (keeps its orientation)")
        self.btn_set_pos.clicked.connect(self._apply_set_position)
        gl.addWidget(self.btn_set_pos, 1, 0, 1, 7)

        gl.addWidget(QLabel("Rot"), 2, 0)
        for i, axis in enumerate("XYZ"):
            gl.addWidget(QLabel("R" + axis.lower()), 2, 1 + 2 * i)
            sb = QDoubleSpinBox()
            sb.setRange(-360.0, 360.0)
            sb.setDecimals(2)
            sb.setSuffix("°")
            sb.setMaximumWidth(80)
            sb.setToolTip("Intrinsic X-Y-Z Euler angle R%s (degrees)"
                          % axis.lower())
            gl.addWidget(sb, 2, 2 + 2 * i)
            self.abs_rot.append(sb)
        self.btn_set_rot = QPushButton("Set orientation")
        self.btn_set_rot.setToolTip("Rotate the element to the typed "
                                    "intrinsic X-Y-Z angles (keeps position)")
        self.btn_set_rot.clicked.connect(self._apply_set_orientation)
        gl.addWidget(self.btn_set_rot, 3, 0, 1, 7)

        gl.addWidget(QLabel("Relative to:"), 4, 0, 1, 2)
        self.ref_combo = QComboBox()
        self.ref_combo.setToolTip("Element whose optical center the readout "
                                  "below is measured from (Origin by default)")
        self.ref_combo.currentIndexChanged.connect(self._update_delta)
        gl.addWidget(self.ref_combo, 4, 2, 1, 5)
        self.delta = QLabel("Δ → (0.000, 0.000, 0.000) mm")
        self.delta.setStyleSheet("color: gray;")
        gl.addWidget(self.delta, 5, 0, 1, 7)
        return g

    def _current_placement(self):
        if self.project is None or not self.body_name:
            return None
        return self.project.current_placement(self.body_name)

    def _placement_bound(self):
        if self.project is None or not self.body_name:
            return False
        return bool(self.project.body(self.body_name).get("placement_bound"))

    def _refresh_absolute(self):
        pl = self._current_placement()
        editable = pl is not None and not self._placement_bound()
        self.btn_set_pos.setEnabled(editable)
        self.btn_set_rot.setEnabled(editable)
        # don't clobber a value the user is mid-edit on
        focused = any(w.hasFocus() for w in self.abs_pos + self.abs_rot)
        if pl is not None and not focused:
            for sb, v in zip(self.abs_pos, pl.pos):
                sb.blockSignals(True)
                sb.setValue(float(v))
                sb.blockSignals(False)
            for sb, v in zip(self.abs_rot, euler_from_quat(pl.quat)):
                sb.blockSignals(True)
                sb.setValue(float(v))
                sb.blockSignals(False)
        self._update_delta()

    def _refresh_ref_combo(self):
        cur = self.ref_combo.currentData()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItem("Origin", None)
        if self.project is not None and self.project.structure:
            for b in self.project.structure["bodies"]:
                self.ref_combo.addItem(b["label"], b["name"])
        idx = self.ref_combo.findData(cur)
        self.ref_combo.setCurrentIndex(max(idx, 0))
        self.ref_combo.blockSignals(False)

    def _update_delta(self):
        if self.project is None or not self.body_name:
            self.delta.setText("Δ → (—) mm")
            return
        try:
            res = self.project.resolver()
            own = res.resolve_point(
                {"kind": "optical_center", "body": self.body_name})
            ref_name = self.ref_combo.currentData()
            ref = (res.resolve_point({"kind": "optical_center",
                                      "body": ref_name})
                   if ref_name else [0.0, 0.0, 0.0])
            d = [own[i] - ref[i] for i in range(3)]
            self.delta.setText("Δ → (%.3f, %.3f, %.3f) mm"
                               % (d[0], d[1], d[2]))
        except Exception:
            self.delta.setText("Δ → (unresolved) mm")

    def _apply_set_position(self):
        pl = self._current_placement()
        if pl is None:
            return
        pos = [sb.value() for sb in self.abs_pos]
        self._apply(Operation("set_placement",
                              {"pos_mm": pos, "quat": pl.quat.tolist()}),
                    "set position (%.3f, %.3f, %.3f) mm" % tuple(pos))

    def _apply_set_orientation(self):
        pl = self._current_placement()
        if pl is None:
            return
        rx, ry, rz = (sb.value() for sb in self.abs_rot)
        quat = quat_from_euler(rx, ry, rz).tolist()
        self._apply(Operation("set_placement",
                              {"pos_mm": pl.pos.tolist(), "quat": quat}),
                    "set orientation (%.2f, %.2f, %.2f)°" % (rx, ry, rz))

    # -- snap to axis ----------------------------------------------------------
    def _build_snap_group(self):
        g = QGroupBox("Snap to optical axis")
        gl = QGridLayout(g)
        self.snap_use_optical = QCheckBox(
            "Use the target element's optical axis")
        self.snap_use_optical.setChecked(True)
        self.snap_use_optical.setToolTip(
            "On: align to the target element's optical axis (its largest "
            "face normal). Off: align to the exact face you click.")
        gl.addWidget(self.snap_use_optical, 0, 0, 1, 3)
        self.snap_pick_btn = QPushButton("Pick target face…")
        self.snap_pick_btn.setToolTip(
            "Then click a face in the 3D view; the selected element rotates "
            "onto that axis and centers on the axis line (one undo step).")
        self.snap_pick_btn.clicked.connect(self._pick_snap_target)
        gl.addWidget(self.snap_pick_btn, 1, 0, 1, 3)
        gl.addWidget(QLabel("Along-axis offset"), 2, 0)
        self.snap_offset = QDoubleSpinBox()
        self.snap_offset.setRange(-1e5, 1e5)
        self.snap_offset.setDecimals(3)
        self.snap_offset.setSuffix(" mm")
        self.snap_offset.setToolTip("Distance to shift along the snapped "
                                    "axis (needs a target picked first)")
        gl.addWidget(self.snap_offset, 2, 1)
        self.snap_offset_btn = QPushButton("Apply offset")
        self.snap_offset_btn.setEnabled(False)
        self.snap_offset_btn.clicked.connect(self._apply_snap_offset)
        gl.addWidget(self.snap_offset_btn, 2, 2)
        self.snap_status = QLabel("")
        self.snap_status.setStyleSheet("color: gray;")
        gl.addWidget(self.snap_status, 3, 0, 1, 3)
        return g

    def set_scene_view(self, view):
        """Give the panel the 3D view it arms picks / drags on."""
        self._view = view

    def _pick_snap_target(self):
        if self.project is None or not self.body_name:
            QMessageBox.information(self, "MieWorkbench",
                                    "Select an element to snap first.")
            return
        if self._view is None:
            return
        self.snap_status.setText("Click a target face in the 3D view…")
        self.snap_pick_btn.setEnabled(False)
        self._view.pick_face_once(self._on_snap_target_picked)

    def _on_snap_target_picked(self, body_name, face_id):
        self.snap_pick_btn.setEnabled(True)
        if not body_name or not face_id:
            self.snap_status.setText("Pick cancelled.")
            return
        try:
            res = self.project.resolver()
            if self.snap_use_optical.isChecked():
                axis = res.resolve_axis({"kind": "optical_axis",
                                         "body": body_name})
                point = res.resolve_point({"kind": "optical_center",
                                           "body": body_name})
            else:
                axis = res.resolve_axis({"kind": "face_normal",
                                         "body": body_name, "face": face_id})
                point = res.resolve_point({"kind": "face_point",
                                           "body": body_name, "face": face_id})
            self._snap_axis = (list(point), list(axis))
            self.project.snap_to_axis(self.body_name, point, axis)
        except Exception as exc:
            self.snap_status.setText("")
            QMessageBox.warning(self, "MieWorkbench", str(exc))
            return
        self.snap_offset_btn.setEnabled(True)
        self.history.addItem("%s: snap to axis" % self.body_name)
        self.history.scrollToBottom()
        self.snap_status.setText(
            "Snapped. Drag along the axis in the 3D view (Esc cancels), "
            "or type an offset.")
        self._begin_snap_drag()

    def _apply_snap_offset(self):
        if self._snap_axis is None:
            return
        _point, axis = self._snap_axis
        d = self.snap_offset.value()
        v = [d * a for a in axis]
        self._apply(Operation("translate", {"vector_mm": v}),
                    "offset %g mm along axis" % d)

    def _begin_snap_drag(self):
        if self._view is None or self._snap_axis is None:
            return
        try:
            center = self.project.resolver().resolve_point(
                {"kind": "optical_center", "body": self.body_name})
            base_pl = self.project.current_placement(self.body_name)
        except Exception:
            return
        if base_pl is None:
            return
        self._drag_base = (list(center), base_pl.to_dict())
        _point, axis = self._snap_axis
        self._view.begin_axis_drag(
            center, axis, self._snap_drag_move, self._snap_drag_commit,
            self._snap_drag_abort)

    def _snap_drag_move(self, world_pt):
        if self._drag_base is None:
            return
        center, base_pl = self._drag_base
        delta = [world_pt[i] - center[i] for i in range(3)]
        preview = {"pos_mm": [base_pl["pos_mm"][i] + delta[i]
                              for i in range(3)],
                   "quat": base_pl["quat"]}
        self._view.update_placement(self.body_name, preview)

    def _snap_drag_commit(self, world_pt):
        base = self._drag_base
        self._drag_base = None
        if base is None:
            return
        center, base_pl = base
        delta = [world_pt[i] - center[i] for i in range(3)]
        if any(abs(c) > 1e-9 for c in delta):
            self._apply(Operation("translate", {"vector_mm": delta}),
                        "drag along axis")
        self.snap_status.setText("Placed on axis.")

    def _snap_drag_abort(self):
        base = self._drag_base
        self._drag_base = None
        if base is not None and self._view is not None:
            self._view.update_placement(self.body_name, base[1])
        self.snap_status.setText("Drag cancelled.")

    # -- wiring ----------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.toward.set_project(project)
        self.about.set_project(project)
        project.sceneLoaded.connect(self.toward.notify_scene_changed)
        project.sceneLoaded.connect(self.about.notify_scene_changed)
        project.sceneLoaded.connect(
            lambda: (self._refresh_ref_combo(), self._refresh_absolute()))
        project.bodiesMoved.connect(
            lambda _d: (self.toward._emit(), self.about._emit(),
                        self._refresh_absolute()))
        self._refresh_ref_combo()

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
        # a fresh selection invalidates any pending snap axis
        self._snap_axis = None
        self._drag_base = None
        self.snap_offset_btn.setEnabled(False)
        self.snap_status.setText("")
        self._refresh_absolute()

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
