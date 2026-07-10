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

from ..core import train as _trainmod  # noqa: F401  (puts scripts/ on sys.path)
from ..core.transforms import (
    Operation, euler_from_quat, project_point_on_axis, quat_from_euler,
    snap_to_axis_ops,
)

import train_solver  # noqa: E402

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

        # -- train positioning (compact; the Train Editor is the full UI) ---
        lay.addWidget(self._build_positioning_group())

        # -- absolute pose + reference readout ------------------------------
        lay.addWidget(self._build_absolute_group())

        # -- snap to optical axis -------------------------------------------
        lay.addWidget(self._build_snap_group())

        # -- translate (collapsed nudge tools; the Position section above
        # is the primary surface) --------------------------------------------
        tg = QGroupBox("Nudge (world frame)")
        tg.setToolTip("Incremental world-frame moves — for chained "
                      "elements the chain re-derives after every nudge")
        tg_body = QWidget()
        tgl = QGridLayout(tg_body)
        tgl.setContentsMargins(0, 0, 0, 0)
        _tg_lay = QVBoxLayout(tg)
        _tg_lay.addWidget(tg_body)
        tg.setCheckable(True)
        tg.toggled.connect(tg_body.setVisible)
        tg.setChecked(False)
        tg_body.setVisible(False)
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

        # -- rotate (collapsed) ------------------------------------------------
        rg = QGroupBox("Rotate (world frame)")
        rg.setToolTip("Rotate about an axis through a reference point — "
                      "chained elements re-derive their tilt fields")
        rg_body = QWidget()
        rgl = QGridLayout(rg_body)
        rgl.setContentsMargins(0, 0, 0, 0)
        _rg_lay = QVBoxLayout(rg)
        _rg_lay.addWidget(rg_body)
        rg.setCheckable(True)
        rg.toggled.connect(rg_body.setVisible)
        rg.setChecked(False)
        rg_body.setVisible(False)
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

    # -- train positioning -----------------------------------------------------
    _EDGE_FIELDS = ("distance", "decenter_x", "decenter_y")
    _EDGE_LABELS = {"distance": "Distance", "decenter_x": "Dec X",
                    "decenter_y": "Dec Y"}

    def _build_positioning_group(self):
        """The mode-aware Position section: BOTH representations of the
        selected element's pose. Its stored mode (anchored vs chained) is
        the editable one; the other is a live derived view, and Convert
        switches storage mode WITHOUT moving the element."""
        g = QGroupBox("Position")
        gl = QGridLayout(g)
        self.pos_status = QLabel("—")
        self.pos_status.setStyleSheet("font-weight: bold;")
        self.pos_status.setWordWrap(True)
        gl.addWidget(self.pos_status, 0, 0, 1, 4)

        gl.addWidget(QLabel("Reference"), 1, 0)
        self.train_ref = QComboBox()
        self.train_ref.setToolTip(
            "Upstream element this one is (or would be) chained to. For a "
            "chained element, changing it re-chains; for an anchored one "
            "it previews the would-be chain values.")
        self.train_ref.currentIndexChanged.connect(self._on_train_ref_changed)
        gl.addWidget(self.train_ref, 1, 1, 1, 2)
        self.btn_pick_ref = QPushButton("Pick…")
        self.btn_pick_ref.setToolTip(
            "Then click an element — or a specific face — in the 3D view "
            "or the outliner to use it as the reference (a face pick also "
            "selects the nearest exit port).")
        self.btn_pick_ref.clicked.connect(self._arm_ref_pick)
        gl.addWidget(self.btn_pick_ref, 1, 3)

        gl.addWidget(QLabel("Port"), 2, 0)
        self.train_port = QComboBox()
        self.train_port.setToolTip(
            "Which exit port of the reference the beam is taken from "
            "(transmit / reflect / deviate).")
        self.train_port.currentIndexChanged.connect(
            self._on_train_port_changed)
        gl.addWidget(self.train_port, 2, 1, 1, 2)

        self.edge_fields = {}
        for i, field in enumerate(self._EDGE_FIELDS):
            gl.addWidget(QLabel(self._EDGE_LABELS[field]), 3 + i, 0)
            e = QLineEdit()
            e.setToolTip({
                "distance": "Along-beam distance from the reference port "
                            "(exit vertex) to this element's entry vertex, "
                            "mm. Expressions over the global variables "
                            "work (e.g. arm1 - 5).",
                "decenter_x": "Transverse offset along the beam frame's "
                              "horizontal (u) axis, mm.",
                "decenter_y": "Transverse offset along the beam frame's "
                              "vertical (v = up) axis, mm.",
            }[field])
            e.editingFinished.connect(
                lambda f=field: self._on_edge_field_committed(f))
            gl.addWidget(e, 3 + i, 1, 1, 2)
            self.edge_fields[field] = e
        self.edge_eval = QLabel("")
        self.edge_eval.setStyleSheet("color: gray;")
        self.edge_eval.setWordWrap(True)
        gl.addWidget(self.edge_eval, 3, 3, 3, 1)

        self.btn_convert = QPushButton("Convert")
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        self.btn_convert.hide()
        gl.addWidget(self.btn_convert, 6, 0, 1, 4)
        self.pos_note = QLabel("")
        self.pos_note.setStyleSheet("color: gray;")
        self.pos_note.setWordWrap(True)
        gl.addWidget(self.pos_note, 7, 0, 1, 4)
        self._pos_updating = False
        self._ref_pick_armed = False
        self._candidate = {}   # element -> (ref, port) chosen while anchored
        return g

    # helpers ------------------------------------------------------------------
    def _element_and_rec(self):
        if self.project is None or not self.body_name:
            return None, None
        try:
            element = self.project.element_group(self.body_name)
            return element, self.project.train().records().get(element)
        except Exception:
            return None, None

    def _refresh_positioning(self):
        if not hasattr(self, "pos_status") or self._pos_updating:
            return
        self._pos_updating = True
        try:
            self._do_refresh_positioning()
        finally:
            self._pos_updating = False

    def _do_refresh_positioning(self):
        element, rec = self._element_and_rec()
        widgets = (self.train_ref, self.train_port, self.btn_pick_ref,
                   *self.edge_fields.values())
        if rec is None:
            self.pos_status.setText("—")
            self.pos_note.setText("")
            self.edge_eval.setText("")
            for w in widgets:
                w.setEnabled(False)
            self.btn_convert.hide()
            return
        for w in widgets:
            w.setEnabled(True)
        tm = self.project.train()
        chained = rec.get("mode") == "chained"

        # reference combo: everything except self + descendants
        try:
            order = train_solver.sort_chain(tm.records())
        except train_solver.TrainError:
            order = tm.element_labels()
        skip = set(tm.downstream_of(element)) | {element}
        candidates = [el for el in order if el not in skip]
        want_ref = (rec.get("ref") if chained
                    else self._candidate.get(element, (None, None))[0])
        if want_ref not in candidates:
            want_ref = candidates[-1] if candidates else None
        self.train_ref.blockSignals(True)
        self.train_ref.clear()
        for el in candidates:
            self.train_ref.addItem(el, el)
        if want_ref is not None:
            self.train_ref.setCurrentIndex(candidates.index(want_ref))
        self.train_ref.blockSignals(False)

        # port combo for the chosen reference
        ports = []
        if want_ref is not None:
            try:
                ports = tm.available_ports(want_ref)
            except Exception:
                ports = ["out"]
        want_port = (rec.get("port") if chained
                     else self._candidate.get(element, (None, None))[1])
        if want_port not in ports:
            want_port = (train_solver._default_port(tm.records()[want_ref])
                         if want_ref is not None else None)
        self.train_port.blockSignals(True)
        self.train_port.clear()
        for p in ports:
            self.train_port.addItem(p, p)
        if want_port in ports:
            self.train_port.setCurrentIndex(ports.index(want_port))
        self.train_port.blockSignals(False)

        variables = self.project.train_variables()
        if chained:
            self.pos_status.setText("🔗 Chained to %s (%s)"
                                    % (rec.get("ref", "?"),
                                       rec.get("port")
                                       or want_port or "out"))
            evals = []
            for field, w in self.edge_fields.items():
                raw = str(rec.get(field) or "0")
                if not w.hasFocus():
                    w.setText(raw)
                w.setReadOnly(False)
                try:
                    evals.append("%.4g" % train_solver.eval_expr(
                        raw, variables))
                except train_solver.TrainError:
                    evals.append("?")
            self.edge_eval.setText("= %s mm" % ", ".join(evals))
            self.btn_convert.setText("Convert to anchored (keeps position)")
            self.btn_convert.setToolTip(
                "Freeze this element at its current world pose (stops "
                "following the train). The element does not move.")
            self.btn_convert.show()
            self.pos_note.setText("")
        else:
            self.pos_status.setText("⚓ Anchored (world pose below is "
                                    "authoritative)")
            ok = False
            if want_ref is not None:
                try:
                    edge = tm.candidate_edge(element, want_ref, want_port,
                                             variables)
                    for field, w in self.edge_fields.items():
                        w.setText("%.4g" % edge[field])
                        w.setReadOnly(True)
                    self.edge_eval.setText("(derived preview)")
                    ok = True
                    self.pos_note.setText(
                        "Values show what the chain would be relative to "
                        "the chosen reference — Convert makes it real "
                        "without moving anything.")
                except train_solver.TrainError as exc:
                    for w in self.edge_fields.values():
                        w.setText("")
                        w.setReadOnly(True)
                    self.edge_eval.setText("")
                    self.pos_note.setText(str(exc))
            else:
                for w in self.edge_fields.values():
                    w.setText("")
                    w.setReadOnly(True)
                self.edge_eval.setText("")
                self.pos_note.setText("No upstream element to chain to.")
            self.btn_convert.setText("Convert to chained (keeps position)")
            self.btn_convert.setToolTip(
                "Store this element's position as a chain relative to the "
                "chosen reference/port. The element does not move — the "
                "shown values become its chain edge.")
            self.btn_convert.setVisible(ok)

    # combo / field commits ------------------------------------------------------
    def _on_train_ref_changed(self):
        if self._pos_updating:
            return
        element, rec = self._element_and_rec()
        if rec is None:
            return
        ref = self.train_ref.currentData()
        if not ref:
            return
        if rec.get("mode") == "chained":
            try:
                self.project.set_chain(element, {"ref": ref},
                                       text="Re-chain %s to %s"
                                       % (element, ref))
            except Exception as exc:
                self.pos_note.setText(str(exc))
        else:
            self._candidate[element] = (ref, None)
        self._refresh_positioning()

    def _on_train_port_changed(self):
        if self._pos_updating:
            return
        element, rec = self._element_and_rec()
        if rec is None:
            return
        port = self.train_port.currentData()
        if not port:
            return
        if rec.get("mode") == "chained":
            try:
                self.project.set_chain(element, {"port": port},
                                       text="Set %s port to %s"
                                       % (element, port))
            except Exception as exc:
                self.pos_note.setText(str(exc))
        else:
            ref = self._candidate.get(element, (None, None))[0] \
                or self.train_ref.currentData()
            self._candidate[element] = (ref, port)
        self._refresh_positioning()

    def _on_edge_field_committed(self, field):
        if self._pos_updating:
            return
        element, rec = self._element_and_rec()
        if rec is None or rec.get("mode") != "chained":
            return
        text = self.edge_fields[field].text().strip() or "0"
        if text == str(rec.get(field) or "0"):
            return
        try:
            train_solver.eval_expr(text, self.project.train_variables())
        except train_solver.TrainError as exc:
            self.pos_note.setText("%s: %s" % (self._EDGE_LABELS[field], exc))
            return
        try:
            self.project.set_chain(element, {field: text},
                                   text="Edit %s of %s" % (field, element))
        except Exception as exc:
            self.pos_note.setText(str(exc))
            return
        self.pos_note.setText("")
        self._refresh_positioning()

    # convert ---------------------------------------------------------------------
    def _on_convert_clicked(self):
        element, rec = self._element_and_rec()
        if rec is None:
            return
        try:
            if rec.get("mode") == "chained":
                self.project.set_anchored(element)
            else:
                ref = self.train_ref.currentData()
                port = self.train_port.currentData()
                tm = self.project.train()
                edge = tm.candidate_edge(element, ref, port,
                                         self.project.train_variables())
                payload = {"ref": ref, "port": port}
                payload.update({k: float(v) for k, v in edge.items()})
                self.project.set_chain(
                    element, payload,
                    text="Convert %s to chained" % element)
        except Exception as exc:
            self.pos_note.setText(str(exc))
            return
        self._refresh_positioning()

    # reference picking (3D face pick OR any selection while armed) ---------------
    def _arm_ref_pick(self):
        if self.project is None or not self.body_name:
            return
        self._ref_pick_armed = True
        self.pos_note.setText("Click an element (or a face) in the 3D view "
                              "or the outliner to use it as the reference…")
        if self._view is not None:
            self._view.pick_face_once(self._on_ref_face_picked)

    def _on_ref_face_picked(self, body_name, face_id):
        if not self._ref_pick_armed:
            return
        self._ref_pick_armed = False
        if not body_name:
            self.pos_note.setText("Pick cancelled.")
            return
        try:
            ref = self.project.element_group(body_name)
        except Exception:
            self.pos_note.setText("Could not resolve the picked element.")
            return
        port = self._infer_port(ref, body_name, face_id)
        self._take_reference(ref, port)

    def _infer_port(self, ref_element, body_name, face_id):
        """Face pick -> nearest exit port of the element (None = default)."""
        if not face_id:
            return None
        try:
            tm = self.project.train()
            loc = tm.local_ports(ref_element)
            state = self.project.body_states[
                tm.primary_body_name(ref_element)]
            face_state = self.project.body_states.get(body_name)
            fc = face_state.face_centroid_world(face_id)
            candidates = {}
            exit_w = state.current.transform_point(loc["exit"])
            candidates["out"] = exit_w
            candidates["transmit"] = exit_w
            rp = loc.get("reflect_plane")
            if rp:
                candidates["reflect"] = state.current.transform_point(
                    rp["point"])
            avail = set(tm.available_ports(ref_element))
            best, best_d = None, None
            for port, pt in candidates.items():
                if port not in avail:
                    continue
                d = sum((float(pt[i]) - float(fc[i])) ** 2 for i in range(3))
                if best_d is None or d < best_d:
                    best, best_d = port, d
            return best
        except Exception:
            return None

    def _take_reference(self, ref, port):
        element, rec = self._element_and_rec()
        if rec is None:
            return
        if ref == element or ref in self.project.train().downstream_of(
                element):
            self.pos_note.setText("%s cannot be the reference (it is "
                                  "downstream of %s)." % (ref, element))
            return
        if rec.get("mode") == "chained":
            payload = {"ref": ref}
            if port:
                payload["port"] = port
            try:
                self.project.set_chain(element, payload,
                                       text="Re-chain %s to %s"
                                       % (element, ref))
            except Exception as exc:
                self.pos_note.setText(str(exc))
                return
        else:
            self._candidate[element] = (ref, port)
        self.pos_note.setText("")
        self._refresh_positioning()

    def chain_to(self, ref_element):
        """Dialog-free: chain the selected element to `ref_element`."""
        if self.project is None or not self.body_name:
            return False
        element = self.project.element_group(self.body_name)
        try:
            self.project.set_chain(element, {"ref": ref_element},
                                   text="Chain %s to %s"
                                   % (element, ref_element))
        except Exception as exc:
            self.pos_note.setText(str(exc))
            return False
        self._refresh_positioning()
        return True

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
        gl.addWidget(QLabel("Position along axis"), 2, 0)
        self.snap_offset = QDoubleSpinBox()
        self.snap_offset.setRange(-1e5, 1e5)
        self.snap_offset.setDecimals(3)
        self.snap_offset.setSuffix(" mm")
        self.snap_offset.setToolTip(
            "The element's CURRENT signed distance from the snap reference "
            "point along the axis. Type a new value to move it to that "
            "absolute station (not an incremental offset).")
        gl.addWidget(self.snap_offset, 2, 1)
        self.snap_offset_btn = QPushButton("Move to position")
        self.snap_offset_btn.setToolTip(
            "Move the element so its optical center sits at the typed "
            "along-axis position")
        self.snap_offset_btn.setEnabled(False)
        self.snap_offset_btn.clicked.connect(self._apply_snap_offset)
        gl.addWidget(self.snap_offset_btn, 2, 2)
        self.snap_status = QLabel("")
        self.snap_status.setStyleSheet("color: gray;")
        gl.addWidget(self.snap_status, 3, 0, 1, 3)
        return g

    def _along_axis_t(self, point=None):
        """The selected element's optical-center coordinate along the
        armed snap axis (signed mm from the reference point), or None."""
        if self._snap_axis is None or self.project is None \
                or not self.body_name:
            return None
        try:
            axis_point, axis = self._snap_axis
            probe = point if point is not None else \
                self.project.resolver().resolve_point(
                    {"kind": "optical_center", "body": self.body_name})
            _foot, t = project_point_on_axis(probe, axis_point, axis)
            return float(t)
        except Exception:
            return None

    def _refresh_snap_position(self):
        t = self._along_axis_t()
        if t is None or self.snap_offset.hasFocus():
            return
        self.snap_offset.blockSignals(True)
        self.snap_offset.setValue(t)
        self.snap_offset.blockSignals(False)

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
            # isVisible() guard (UI_TESTING doctrine): a modal in this
            # path hangs offscreen test runs when the snap raises
            if self.isVisible():
                self.snap_status.setText("")
                QMessageBox.warning(self, "MieWorkbench", str(exc))
            else:
                self.snap_status.setText("Snap failed: %s" % exc)
            return
        self.snap_offset_btn.setEnabled(True)
        self.history.addItem("%s: snap to axis" % self.body_name)
        self.history.scrollToBottom()
        self._refresh_snap_position()
        self.snap_status.setText(
            "Snapped. Drag along the axis in the 3D view (Esc cancels), "
            "or type an absolute position.")
        self._begin_snap_drag()

    def _apply_snap_offset(self):
        """Move to the typed ABSOLUTE along-axis position (the spinbox
        shows the current position; committing the same value is a
        no-op)."""
        current = self._along_axis_t()
        if current is None:
            return
        _point, axis = self._snap_axis
        target = self.snap_offset.value()
        d = target - current
        if abs(d) < 1e-9:
            self.snap_status.setText("Already at %.3f mm along the axis."
                                     % target)
            return
        n = sum(a * a for a in axis) ** 0.5
        v = [d * a / n for a in axis]
        self._apply(Operation("translate", {"vector_mm": v}),
                    "position %g mm along axis" % target)
        self._refresh_snap_position()

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
        # live ABSOLUTE readout while dragging (world_pt is where the
        # optical center is heading, already on the axis line)
        t = self._along_axis_t(point=list(world_pt))
        if t is not None:
            self.snap_status.setText("at %.3f mm along the axis" % t)
            if not self.snap_offset.hasFocus():
                self.snap_offset.blockSignals(True)
                self.snap_offset.setValue(t)
                self.snap_offset.blockSignals(False)

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
            lambda: (self._refresh_ref_combo(), self._refresh_absolute(),
                     self._refresh_positioning()))
        project.propertiesChanged.connect(
            lambda _n: self._refresh_positioning())
        project.bodiesMoved.connect(
            lambda _d: (self.toward._emit(), self.about._emit(),
                        self._refresh_absolute(), self._refresh_positioning(),
                        self._refresh_snap_position()))
        self._refresh_ref_combo()
        self._refresh_positioning()

    def set_body(self, body_name):
        # armed reference pick: the NEXT selection (outliner or 3D click,
        # both route through here) becomes the chain reference instead of
        # changing what the panel operates on
        if getattr(self, "_ref_pick_armed", False) and body_name \
                and body_name != self.body_name:
            self._ref_pick_armed = False
            try:
                ref = self.project.element_group(body_name)
            except Exception:
                ref = None
            if ref:
                self._take_reference(ref, None)
            return
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
        self._refresh_positioning()

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
            # move_element keeps the optical train consistent (a chained
            # element re-derives its edge, downstream follows); it falls
            # back to the raw apply_operation for untrained elements.
            self.project.move_element(self.body_name, op)
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
