"""TrainEditorPane - an LDE-style editable view of the optical train.

The whole train is ONE indented tree (mirrors the outliner idiom): the
main train runs top-to-bottom, a multi-port element (beamsplitter / fold
mirror) sprouts indented per-port child branches, and a fold element
carries a checkable "folded" cell. Everything is visible at once.

Contract (see the task brief + docs/UI_TESTING.md):
  * every mutation goes through the Project train API (set_chain,
    set_anchored, set_fold_state, insert_fold_mirror) - never a raw
    property write;
  * numeric edge cells (distance / decenter / tilt) show the STORED
    expression verbatim, appending the evaluated value in parentheses for
    DISPLAY only ("gap*2  (= 50.0)"); the EditRole is the bare expression;
  * selection stays in sync with the shared SelectionModel, echo-broken by
    the origin string, exactly like the outliner;
  * errors from the Project API surface in a bottom status label, never a
    modal (offscreen tests must not block).

Everything mutating is also reachable through a dialog-free method so the
offscreen suite can drive it without a real editor widget or dialog:
commit_field, set_mode, toggle_fold, unfold_all/refold_all, insert_fold,
set_edge_details, chain_selected, anchor_selected, begin_pick_reference/
on_reference_picked -- plus, new this round (Phase G, see
demos/UXNOTES_ROUND2.md's "Round-2 resolutions" section for the full
fix/wontfix rationale per numbered item):

  * commit_port(element, port) -- the Port column's combo editor commits
    here; `port` must be one of the reference element's available exit
    ports (see _available_ports -- a cheap record-only approximation of
    train_solver.exit_frames' port set, no full solve needed: "out"/
    "transmit" always, "reflect" when the ref has a reflect_plane,
    "deviate" when the ref has an explicit fold_deviation or is a
    fold with no reflect_plane). Also reachable via each chained row's
    right-click "Chain onto port..." submenu (lists the SAME ref's
    ports -- the wishlist's "submenu keyed off the currently-selected
    prospective parent" variant was NOT built; documented as deferred).
  * mark_fold(element, is_fold) -- the fold IDENTITY bit alone (does NOT
    touch the folded open/closed state on unmark), reachable via the
    right-click "Make fold mirror (unfoldable)" / "Remove fold
    designation" entries. The Fold
    column's checkbox itself is now checkable for ANY chained element:
    checking a non-fold row is a shortcut for mark_fold(element, True)
    (which also sets folded=True); once an element IS a fold, the same
    checkbox reverts to meaning the folded/unfolded STATE (toggle_fold) --
    state and identity are deliberately two different controls.
  * commit_flip(element, flip) -- the new narrow "Flip" column.
  * set_edge_details(element, rot_order=None, pos_rot_order=None,
    pivot=None, fold_deviation=None, fold_azimuth=None) -- every
    parameter now defaults to None ("leave unchanged"), so a caller (the
    Edge-details dialog or a script) can commit a single field without
    restating the other four; the dialog also grew two expression-capable
    text fields for fold_deviation/fold_azimuth.
  * chain_selected() -- now chains the CURRENTLY selected element onto
    the PREVIOUSLY selected one (a small 2-deep distinct-element
    selection history tracked across BOTH tree clicks and external
    SelectionModel changes, e.g. an outliner or 3D-view pick), matching
    the natural "click the parent, click the child, then Chain to
    selected" gesture. Falls back to the last element in solve order
    (train_solver.sort_chain, NOT alphabetical tree position) only when
    there is no usable selection pair, and refuses with a status message
    (no silent guess) when even that is ambiguous/absent.

Public API the mainwindow wires:
  * TrainEditorPane(project, selection, parent=None)
  * begin_pick_reference(element) / on_reference_picked(body, face_id)
    - the mainwindow connects view.pick_face_once to on_reference_picked
    after begin_pick_reference emits pickReferenceRequested.
  * editAnchorRequested(str element) -- NEW signal, emitted by an
    anchored row's right-click "Set absolute pose..." context-menu entry
    (see UXNOTES_ROUND2.md #8). This pane does NOT wire it anywhere: the
    mainwindow is expected to connect it to whatever focuses/raises the
    Position/Orientation Absolute panel for `element`'s primary body.
"""

import collections
import math
import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QStyledItemDelegate,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..core import train as _trainmod  # noqa: F401  (puts scripts/ on sys.path)
from ..core import opticalvalues, paraxial
from ..core import variables as V
from ..core.project import ProjectError
import train_solver  # noqa: E402

TrainError = train_solver.TrainError

ORIGIN = "train_editor"

# columns
COL_ELEMENT = 0
COL_MODE = 1
COL_REF = 2
COL_PORT = 3
COL_DIST = 4
COL_DECX = 5
COL_DECY = 6
COL_TILTX = 7
COL_TILTY = 8
COL_TILTZ = 9
COL_FOLD = 10
COL_FLIP = 11
_HEADERS = ["Element", "Mode", "Reference", "Port", "Distance (mm)",
            "Dec X", "Dec Y", "Tilt X", "Tilt Y", "Tilt Z", "Fold", "Flip"]

# beam-frame axis convention (u = up x dir, v = up, d = dir -- see
# train_solver's module docstring); the header tooltips below are the
# only in-UI documentation of this (UXNOTES_ROUND2.md #10 kept the X/Y/Z
# names as-is -- they match the stored field names -- and tooltip'd them
# instead of renaming to Dec U/V).
_BEAM_FRAME_NOTE = (
    "beam-frame, not world/local: u = horizontal transverse (up x beam "
    "direction), v = up-ish transverse, tilts are about u/v/beam-direction "
    "in that order. Concretely, for a beam along +x with up +z: "
    "decenter/tilt X act along/about world +y, decenter/tilt Y along/about "
    "world +z. After one or more folds these diverge from world "
    "axes -- always the INCOMING beam's frame at this element, never the "
    "element's own local axes.")
_HEADER_TOOLTIPS = {
    COL_DIST: "Distance (mm) -- vertex-to-vertex along the beam, exit "
              "vertex to entry vertex.\n\n%s" % train_solver.EXPR_HELP,
    COL_DECX: "Decenter X (mm) -- offset along u, the %s\n\n%s"
              % (_BEAM_FRAME_NOTE, train_solver.EXPR_HELP),
    COL_DECY: "Decenter Y (mm) -- offset along v, the %s\n\n%s"
              % (_BEAM_FRAME_NOTE, train_solver.EXPR_HELP),
    COL_TILTX: "Tilt X (deg) -- rotation about u, the %s\n\n%s"
               % (_BEAM_FRAME_NOTE, train_solver.EXPR_HELP),
    COL_TILTY: "Tilt Y (deg) -- rotation about v, the %s\n\n%s"
               % (_BEAM_FRAME_NOTE, train_solver.EXPR_HELP),
    COL_TILTZ: "Tilt Z (deg) -- rotation about the beam direction d, the "
               "%s\n\n%s" % (_BEAM_FRAME_NOTE, train_solver.EXPR_HELP),
    COL_FOLD: "Fold state -- check to fold, uncheck to straighten the arm "
              "(mirror excluded). For a plain chained mirror, checking "
              "designates it a fold first.",
}

# shared by both context-menu wordings ("Make fold mirror (unfoldable)" /
# "Remove fold designation") -- same underlying toggle, same explanation.
_FOLD_MARK_TIP = (
    "Designates this chained mirror as a FOLD: it gains a folded/unfolded "
    "toggle -- unfolding straightens the downstream arm and excludes the "
    "mirror from simulation.")

# editable numeric edge columns -> solver record field
_COL_FIELD = {
    COL_DIST: "distance",
    COL_DECX: "decenter_x",
    COL_DECY: "decenter_y",
    COL_TILTX: "tilt_rx",
    COL_TILTY: "tilt_ry",
    COL_TILTZ: "tilt_rz",
}
_NUMERIC_COLS = set(_COL_FIELD)

ROLE_BODY = Qt.UserRole            # element primary body name (element rows)
ROLE_ELEMENT = Qt.UserRole + 1     # element label (None for port rows)

_MODE_ANCHORED = "⚓ anchored"      # anchor
_MODE_CHAINED = "\U0001f517 chained"    # link
_PORT_ICON = {"reflect": "↳", "deviate": "↳",
              "transmit": "↓", "out": "↓"}

_RED = QColor("#c0392b")
_GRAY = QColor("gray")


def _fmt(val):
    """Render an evaluated edge value: keep a trailing .0 on integers
    (matches the '(= 50.0)' display contract), else a compact %g."""
    if val == int(val):
        return "%.1f" % val
    return "%g" % val


class _ElementItem(QTreeWidgetItem):
    """A tree row that carries DISTINCT DisplayRole and EditRole strings per
    column (a plain QTreeWidgetItem collapses the two): the numeric edge
    cells show 'gap*2  (= 50.0)' but edit as the bare 'gap*2', and the mode
    cell shows an emoji glyph but edits as the canonical mode name."""

    def __init__(self):
        super().__init__()
        self._edit = {}      # column -> EditRole value
        self._disp = {}      # column -> DisplayRole value

    def set_dual(self, column, edit_val, disp_val):
        self._edit[column] = edit_val
        self._disp[column] = disp_val

    def data(self, column, role):
        if role == Qt.DisplayRole and column in self._disp:
            return self._disp[column]
        if role == Qt.EditRole and column in self._edit:
            return self._edit[column]
        return super().data(column, role)

    def setData(self, column, role, value):
        if role == Qt.EditRole and column in self._edit:
            self._edit[column] = value       # capture a committed edit
        super().setData(column, role, value)  # still emits itemChanged


class _TrainDelegate(QStyledItemDelegate):
    """Per-column editors: a mode combo, a port combo (choices computed
    from the row's reference element), plain text for the numeric edge
    columns, and NO editor (read-only) everywhere else."""

    def __init__(self, parent, pane):
        super().__init__(parent)
        self._pane = pane

    def createEditor(self, parent, option, index):
        col = index.column()
        if col == COL_MODE:
            cb = QComboBox(parent)
            cb.addItem(_MODE_ANCHORED, "anchored")
            cb.addItem(_MODE_CHAINED, "chained")
            return cb
        if col == COL_PORT:
            item = self._pane.tree.itemFromIndex(index)
            element = item.data(COL_ELEMENT, ROLE_ELEMENT) if item else None
            choices = self._pane._port_choices(element)
            if not choices:
                return None
            cb = QComboBox(parent)
            for p in choices:
                cb.addItem(p, p)
            return cb
        if col in _NUMERIC_COLS:
            return super().createEditor(parent, option, index)
        return None

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            i = editor.findData(index.data(Qt.EditRole))
            editor.setCurrentIndex(max(i, 0))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentData(), Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class TrainEditorPane(QWidget):
    """LDE-style optical-train editor dock. See the module docstring for
    the full public-API contract, including the Phase G additions
    (commit_port / mark_fold / commit_flip / editAnchorRequested)."""

    # emitted when the user arms "pick reference in 3D"; the mainwindow
    # connects this to view.pick_face_once(self.on_reference_picked)
    pickReferenceRequested = Signal(str)

    # emitted by an anchored row's "Set absolute pose..." context-menu
    # entry; NOT wired by this pane (see module docstring)
    editAnchorRequested = Signal(str)

    def __init__(self, project, selection, parent=None):
        super().__init__(parent)
        self._project = project
        self._selection = selection
        self._updating = False        # populating the tree (block echoes)
        self._committing = False       # inside a Project API call (defer rebuild)
        self._pending_pick = None      # element awaiting a 3D reference pick
        self._cur_element = None
        # last two DISTINCT selected elements (oldest first), updated on
        # every selection change regardless of origin (tree click, 3D
        # pick, outliner...) -- see chain_selected()
        self._select_history = collections.deque(maxlen=2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        lay.addLayout(self._build_toolbar())

        # whole-train paraxial readout (system EFL / f-number / image
        # distance / magnification) — refreshed by every _do_rebuild
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray;")
        self.summary.setVisible(False)
        lay.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(_HEADERS))
        self.tree.setHeaderLabels(_HEADERS)
        for col, tip in _HEADER_TOOLTIPS.items():
            self.tree.headerItem().setToolTip(col, tip)
        self.tree.setRootIsDecorated(True)
        self.tree.setItemDelegate(_TrainDelegate(self.tree, self))
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        self.tree.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.tree)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        if selection is not None:
            selection.changed.connect(self._on_selection_changed)
        self._connect_project()
        self._do_rebuild()

    # -- wiring ---------------------------------------------------------------
    def _connect_project(self):
        p = self._project
        if p is None:
            return
        p.sceneLoaded.connect(self._on_change)
        p.propertiesChanged.connect(self._on_change)
        p.bodiesMoved.connect(self._on_change)

    def _on_change(self, *_args):
        # bodiesMoved only needs a value refresh, but a full rebuild is cheap
        # and correct; guard against rebuild-during-edit reentrancy.
        if self._committing or self._updating:
            return
        self._do_rebuild()

    def rebuild(self):
        if not self._committing:
            self._do_rebuild()

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        def btn(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.clicked.connect(slot)
            row.addWidget(b)
            return b

        self.btn_chain_sel = btn(
            "Chain to selected…",
            "Click the intended parent, then the element to chain, then "
            "this button: chains the CURRENTLY selected element onto the "
            "PREVIOUSLY selected one",
            self.chain_selected)
        self.btn_anchor = btn(
            "Anchor", "Freeze the selected element at its current pose",
            self.anchor_selected)
        self.btn_insert_fold = btn(
            "Insert fold mirror…",
            "Insert a fold mirror down-beam and re-anchor the reflected arm",
            self._open_insert_fold_dialog)
        self.btn_unfold_all = btn(
            "Unfold all", "Straighten every fold in the train",
            self.unfold_all)
        self.btn_refold_all = btn(
            "Refold all", "Re-fold every fold in the train",
            self.refold_all)
        self.btn_pick_ref = btn(
            "Pick reference in 3D",
            "Then click an element in the 3D view to chain the selection to it",
            self._on_pick_ref_clicked)
        row.addStretch(1)
        return row

    # -- tree construction -----------------------------------------------------
    def _variables(self):
        try:
            return self._project.train_variables()
        except Exception:
            return {}

    def _do_rebuild(self):
        if self._project is None:
            return
        self._updating = True
        try:
            self.tree.clear()
            structure = getattr(self._project, "structure", None)
            if not structure:
                return
            tm = self._project.train()
            records = tm.records()
            labels = tm.element_labels()
            variables = self._variables()
            # solved world positions for the Distance-cell echo ("this
            # edge resolves to world (x,y,z)" -- removes the vertex-vs-
            # plane guesswork, UXNOTES_ROUND3 #8-11)
            try:
                self._solved_pos = {
                    el: p["pos_mm"]
                    for el, p in tm.solve(variables)["placements"].items()}
            except Exception:
                self._solved_pos = {}

            children = {}          # ref label -> [(child label, port)]
            for el in labels:
                rec = records[el]
                if rec.get("mode") == "chained" and rec.get("ref"):
                    ref = rec["ref"]
                    if ref in records:
                        port = rec.get("port") or train_solver._default_port(
                            records[ref])
                        children.setdefault(ref, []).append((el, port))

            def is_source(el):
                props = tm.primary_body(el).get("properties") or {}
                return "power" in props and "lambdac" in props

            roots = [el for el in labels
                     if records[el].get("mode") != "chained"]
            roots.sort(key=lambda el: (0 if is_source(el) else 1, el))

            seen = set()

            def add(parent, el):
                if el in seen:                     # cycle / re-entry guard
                    return
                seen.add(el)
                item = self._make_element_item(el, records[el], variables)
                if parent is None:
                    self.tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                kids = children.get(el, [])
                ports = {p for _, p in kids}
                if len(ports) > 1:
                    for port in sorted(ports):
                        pit = self._make_port_item(port)
                        item.addChild(pit)
                        for cel, cp in kids:
                            if cp == port:
                                add(pit, cel)
                else:
                    for cel, _cp in kids:
                        add(item, cel)

            for el in roots:
                add(None, el)
            for el in labels:                      # cycles / dangling refs
                if el not in seen:
                    add(None, el)

            self.tree.expandAll()
            self._apply_problem_highlights(tm, labels)
            for c in range(self.tree.columnCount()):
                self.tree.resizeColumnToContents(c)
            self._refresh_summary(tm, variables)
        finally:
            self._updating = False
        if self._selection is not None:
            self._select_body_row(self._selection.body)

    def _refresh_summary(self, tm, variables):
        """System paraxial line under the toolbar. Hidden when there is
        no resolvable powered train; failures degrade to hidden with the
        reason in the tooltip (never a modal)."""
        try:
            s = paraxial.system_summary(tm, variables)
        except Exception as exc:
            self.summary.setVisible(False)
            self.summary.setToolTip(str(exc))
            return
        if not s.get("n_optical_elements") or s.get("efl") is None:
            self.summary.setVisible(False)
            self.summary.setToolTip(
                "; ".join(s.get("warnings") or []) if s else "")
            return
        bits = []
        if s.get("afocal"):
            m_ang = s.get("angular_magnification")
            bits.append("afocal" if m_ang is None
                        else "afocal (angular magnification M=%.3g)" % m_ang)
        elif s["efl"] is not None and math.isfinite(s["efl"]):
            bits.append("EFL %.4g mm" % s["efl"])
        else:
            bits.append("afocal")
        if s.get("fno_working") and math.isfinite(s["fno_working"]):
            bits.append("f/%.3g" % s["fno_working"])
        if s.get("na"):
            bits.append("NA %.3g" % s["na"])
        img = s.get("image_distance_mm")
        if not s.get("afocal") and img is not None and math.isfinite(img):
            bits.append("image %.4g mm past %s"
                        % (img, s["path"][-1]["element"]
                           if s.get("path") else "last element"))
        if s.get("limiting_element"):
            bits.append("stop: %s" % s["limiting_element"])
        self.summary.setText(
            "System: %s   (paraxial, λ = %.0f nm)"
            % (" · ".join(bits), s.get("lambda_nm") or 0.0))
        tip = ("Paraxial ABCD over the chained train — tilts/decenters "
               "ignored, folds traversed straightened.")
        warn = s.get("warnings") or []
        if warn:
            tip += "\nWarnings: " + "; ".join(warn[:6])
        self.summary.setToolTip(tip)
        self.summary.setVisible(True)

    def _make_element_item(self, el, rec, variables):
        item = _ElementItem()
        chained = rec.get("mode") == "chained"
        item.setText(COL_ELEMENT, el)
        item.setData(COL_ELEMENT, ROLE_BODY,
                     self._project.train().primary_body_name(el))
        item.setData(COL_ELEMENT, ROLE_ELEMENT, el)

        item.set_dual(COL_MODE, "chained" if chained else "anchored",
                      _MODE_CHAINED if chained else _MODE_ANCHORED)
        item.setText(COL_REF, rec.get("ref", "") if chained else "")
        item.setText(COL_PORT, rec.get("port", "") if chained else "")

        for col, field in _COL_FIELD.items():
            stored = rec.get(field)
            stored = "" if stored in (None, "") else str(stored)
            item.set_dual(col, stored, self._display_value(stored, variables))
        # Distance-cell echo: where this edge actually puts the element
        # (solved world position of its primary body) -- kills the
        # vertex-vs-plane / thickness guesswork without opening the 3D view
        pos = getattr(self, "_solved_pos", {}).get(el)
        if chained and pos:
            item.setToolTip(
                COL_DIST,
                "Resolves to world (%.4g, %.4g, %.4g) mm\n\n%s"
                % (pos[0], pos[1], pos[2], _HEADER_TOOLTIPS[COL_DIST]))

        flags = item.flags() | Qt.ItemIsEditable
        if chained:
            # the Fold and Flip columns are checkable on ANY chained
            # element (see the module docstring: checking a non-fold row
            # is the "make foldable" shortcut, mark_fold(element, True))
            flags |= Qt.ItemIsUserCheckable
            item.setFlags(flags)
            if rec.get("fold"):
                folded = bool(rec.get("folded", True))
                item.setCheckState(COL_FOLD,
                                   Qt.Checked if folded else Qt.Unchecked)
                item.setToolTip(
                    COL_FOLD,
                    "Folded/unfolded -- unchecked straightens the "
                    "downstream arm and excludes this mirror from the "
                    "simulation")
                if not folded:
                    self._grey_italic(item)          # excluded (unfolded) row
            else:
                item.setCheckState(COL_FOLD, Qt.Unchecked)
                item.setToolTip(
                    COL_FOLD,
                    "Check to make this mirror a fold (unfoldable)")
            item.setCheckState(
                COL_FLIP, Qt.Checked if rec.get("flip") else Qt.Unchecked)
        else:
            item.setFlags(flags)
            item.setText(COL_FOLD, "")
            item.setText(COL_FLIP, "")
        return item

    def _make_port_item(self, port):
        icon = _PORT_ICON.get(port, "↓")
        item = QTreeWidgetItem()
        item.setText(COL_ELEMENT, "%s %s" % (port, icon))
        item.setFlags((item.flags() | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
        item.setData(COL_ELEMENT, ROLE_ELEMENT, None)
        item.setForeground(COL_ELEMENT, QBrush(_GRAY))
        return item

    def _display_value(self, stored, variables):
        if stored == "":
            return ""
        try:
            float(stored)
            return stored
        except (TypeError, ValueError):
            pass
        try:
            val = train_solver.eval_expr(stored, variables)
        except train_solver.TrainError:
            return stored
        return "%s  (= %s)" % (stored, _fmt(val))

    def _grey_italic(self, item):
        for c in range(self.tree.columnCount()):
            f = item.font(c)
            f.setItalic(True)
            item.setFont(c, f)
            item.setForeground(c, QBrush(_GRAY))

    def _apply_problem_highlights(self, tm, labels):
        try:
            problems = tm.validate()
        except Exception as exc:               # defensive; validate catches
            problems = [("error", str(exc))]
        prob = {}
        for _sev, msg in problems:
            tokens = set(re.split(r"\W+", msg))
            for el in labels:
                if el in tokens:
                    prob.setdefault(el, []).append(msg)
        if not prob:
            return
        for item in self._element_items():
            el = item.data(COL_ELEMENT, ROLE_ELEMENT)
            if el in prob:
                tip = "\n".join(prob[el])
                for c in range(self.tree.columnCount()):
                    item.setForeground(c, QBrush(_RED))
                    item.setToolTip(c, tip)

    # -- item / element lookup -------------------------------------------------
    def _walk(self, item=None):
        if item is None:
            for i in range(self.tree.topLevelItemCount()):
                yield from self._walk(self.tree.topLevelItem(i))
            return
        yield item
        for j in range(item.childCount()):
            yield from self._walk(item.child(j))

    def _element_items(self):
        for item in self._walk():
            if item.data(COL_ELEMENT, ROLE_ELEMENT) is not None:
                yield item

    def item_for_element(self, element):
        for item in self._element_items():
            if item.data(COL_ELEMENT, ROLE_ELEMENT) == element:
                return item
        return None

    # -- ports -------------------------------------------------------------------
    @staticmethod
    def _available_ports(ref_rec):
        """Cheap record-only approximation of the port set
        train_solver.exit_frames would compute for `ref_rec` (no full
        solve needed): "out"/"transmit" always; "reflect" when the local
        port geometry carries a reflect_plane; "deviate" when the record
        has an explicit fold_deviation, or is a fold with no reflect
        plane (a bare deviate-only fold, e.g. a grating with no mirror
        backing)."""
        loc = ref_rec.get("local") or {}
        ports = ["out", "transmit"]
        has_rp = bool(loc.get("reflect_plane"))
        if has_rp:
            ports.append("reflect")
        has_dev = ref_rec.get("fold_deviation") not in (None, "")
        if has_dev or (ref_rec.get("fold") and not has_rp):
            ports.append("deviate")
        return ports

    def _port_choices(self, element):
        """Available ports for `element`'s Port column combo: the ports
        of its OWN reference element, or [] when `element` isn't chained
        yet (the delegate then offers no editor -- read-only, same as an
        anchored row)."""
        if not element:
            return []
        records = self._project.train().records()
        rec = records.get(element)
        if rec is None or rec.get("mode") != "chained":
            return []
        ref_rec = records.get(rec.get("ref"))
        if ref_rec is None:
            return []
        return self._available_ports(ref_rec)

    # -- status / errors -------------------------------------------------------
    def _prefixed(self, msg, element):
        text = str(msg)
        if element and not text.startswith("%s: " % element):
            return "%s: %s" % (element, text)
        return text

    def _set_error(self, msg, element=None):
        self.status.setStyleSheet("color: #c0392b;")
        self.status.setText(self._prefixed(msg, element))
        if element:
            self._scroll_to_element(element)

    def _set_info(self, msg="", element=None):
        self.status.setStyleSheet("color: gray;")
        self.status.setText(self._prefixed(msg, element) if msg else "")

    def _scroll_to_element(self, element):
        item = self.item_for_element(element)
        if item is not None:
            self.tree.scrollToItem(item)

    def _mark_cell_error(self, element, field):
        item = self.item_for_element(element)
        if item is None:
            return
        col = next((c for c, f in _COL_FIELD.items() if f == field), None)
        if col is None:
            return
        # setForeground emits itemChanged; guard against a re-entrant commit
        self._updating = True
        try:
            item.setForeground(col, QBrush(_RED))
        finally:
            self._updating = False

    # -- the one place mutations run: suppress rebuilds, rebuild once after ----
    def _apply(self, fn):
        self._committing = True
        try:
            fn()
        finally:
            self._committing = False
            self._do_rebuild()

    # -- edits -----------------------------------------------------------------
    def commit_field(self, element, field, text):
        """Commit an edited numeric edge cell. Validates the expression
        against the train variables BEFORE touching the Project API; an
        invalid expression sets the error state and mutates nothing."""
        text = "" if text is None else str(text).strip()
        if text:
            try:
                train_solver.eval_expr(text, self._variables())
            except train_solver.TrainError as exc:
                self._set_error("Invalid value %r: %s" % (text, exc), element)
                self._mark_cell_error(element, field)
                return False
        try:
            self._apply(lambda: self._project.set_chain(
                element, {field: text},
                text="Edit %s of %s" % (field, element)))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info()
        return True

    def set_mode(self, element, mode):
        """Toggle an element between anchored and chained. Chaining needs a
        reference: the row's previously-stored ref, else the current
        selection-history pair, else the last element in solve order,
        else refuse (see _default_ref_for)."""
        mode = str(mode)
        if mode == "anchored":
            try:
                self._apply(lambda: self._project.set_anchored(element))
            except (ProjectError, TrainError) as exc:
                self._set_error(str(exc), element)
                return False
            self._set_info()
            return True
        if mode == "chained":
            ref = self._default_ref_for(element)
            if not ref:
                self._set_error(
                    "chaining needs a reference element", element)
                return False
            try:
                self._apply(lambda: self._project.set_chain(
                    element, {"ref": ref}, text="Chain %s" % element))
            except (ProjectError, TrainError) as exc:
                self._set_error(str(exc), element)
                return False
            self._set_info()
            return True
        self._set_error("Unknown mode %r" % mode, element)
        return False

    def _default_ref_for(self, element):
        rec = self._project.train().records().get(element, {})
        prev = rec.get("ref")
        if prev and prev in self._project.train().records() and prev != element:
            return prev
        ref = self._selected_pair_ref(element)
        if ref and ref != element:
            return ref
        return self._last_train_element(element)

    def toggle_fold(self, element, folded):
        try:
            self._apply(
                lambda: self._project.set_fold_state(element, bool(folded)))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info()
        return True

    def mark_fold(self, element, is_fold):
        """Context-menu 'Mark as fold mirror' / 'Unmark fold': sets or
        clears the fold IDENTITY bit alone, independent of the folded
        open/closed STATE (once an element IS a fold, the Fold column's
        checkbox reverts to meaning that state -- see toggle_fold).
        Marking also sets folded=True (a freshly-marked fold starts
        folded, matching insert_fold_mirror's new-fold default);
        unmarking leaves `folded` untouched (inert once fold=False)."""
        rec = self._project.train().records().get(element, {})
        if rec.get("mode") != "chained":
            self._set_error(
                "only a chained element can be marked as a fold", element)
            return False
        edge = {"fold": bool(is_fold)}
        if is_fold:
            edge["folded"] = True
        try:
            self._apply(lambda: self._project.set_chain(
                element, edge,
                text=("Mark %s as fold" % element) if is_fold
                else ("Unmark fold %s" % element)))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info()
        return True

    def commit_flip(self, element, flip):
        """Commit the Flip column's checkbox: set_chain {"flip": bool}
        (end-for-end mirroring of a lens -- its former exit surface
        faces the beam)."""
        try:
            self._apply(lambda: self._project.set_chain(
                element, {"flip": bool(flip)}, text="Flip %s" % element))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info()
        return True

    def commit_port(self, element, port):
        """Commit the Port column's combo edit (or the 'Chain onto
        port...' context-menu shortcut): writes {"port": port} via
        set_chain, after checking `port` is one of the reference's
        available exit ports (_available_ports)."""
        rec = self._project.train().records().get(element, {})
        if rec.get("mode") != "chained" or not rec.get("ref"):
            self._set_error(
                "pick a reference before choosing a port", element)
            return False
        ref = rec["ref"]
        ref_rec = self._project.train().records().get(ref, {})
        choices = self._available_ports(ref_rec)
        port = "" if port is None else str(port)
        if port not in choices:
            self._set_error(
                "port %r is not one of %s's ports (%s)"
                % (port, ref, ", ".join(choices) or "none"), element)
            return False
        try:
            self._apply(lambda: self._project.set_chain(
                element, {"port": port}, text="Set port of %s" % element))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info("chained to %s's %s port" % (ref, port), element)
        return True

    def unfold_all(self):
        self._apply(lambda: self._project.set_folds_all(False))

    def refold_all(self):
        self._apply(lambda: self._project.set_folds_all(True))

    def insert_fold(self, after, distance, deviation=90.0, azimuth=0.0):
        """Dialog-free core the Insert-fold dialog calls. Inserts a fold
        mirror `distance` mm down-beam of `after`'s exit port, re-anchoring
        that port's chained children onto the reflected arm. Returns the
        new element label."""
        label = [None]

        def go():
            label[0] = self._project.insert_fold_mirror(
                after, distance, deviation_deg=deviation,
                azimuth_deg=azimuth)
        try:
            self._apply(go)
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), after)
            return None
        self._set_info("Inserted fold mirror %s" % label[0])
        return label[0]

    def set_edge_details(self, element, rot_order=None, pos_rot_order=None,
                         pivot=None, fold_deviation=None, fold_azimuth=None):
        """Commit the Edge-details dialog's fields. Every parameter
        defaults to None, meaning "leave unchanged" -- a partial update
        writes only the fields actually supplied (a power user setting up
        several fold mirrors that only ever touch rot_order no longer
        needs to restate pos_rot_order/pivot/the deviate fields every
        time). fold_deviation/fold_azimuth are expression-capable
        strings, same as the numeric edge columns."""
        edge = {}
        if rot_order is not None:
            edge["rot_order"] = rot_order
        if pos_rot_order is not None:
            edge["pos_rot_order"] = pos_rot_order
        if pivot is not None:
            edge["pivot"] = pivot
        if fold_deviation is not None:
            edge["fold_deviation"] = fold_deviation
        if fold_azimuth is not None:
            edge["fold_azimuth"] = fold_azimuth
        if not edge:
            return True
        try:
            self._apply(lambda: self._project.set_chain(
                element, edge, text="Edge details of %s" % element))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return False
        self._set_info()
        return True

    # -- toolbar handlers ------------------------------------------------------
    def chain_selected(self):
        """Chain the CURRENTLY selected element onto the PREVIOUSLY
        selected one (see the selection-history deque maintained by
        _note_selected / _set_current_element). Falls back to the last
        element in solve order (never alphabetical tree position) only
        when there's no usable selection pair, and refuses -- rather than
        guessing -- when even that is ambiguous."""
        element = self._cur_element
        if not element:
            self._set_error("Select an element to chain first")
            return
        ref = self._selected_pair_ref(element)
        if not ref:
            ref = self._last_train_element(element)
        if not ref or ref == element:
            self._set_error(
                "no unambiguous upstream element to chain to -- select "
                "the intended parent, then this element, then Chain to "
                "selected", element)
            return
        try:
            self._apply(lambda: self._project.set_chain(
                element, {"ref": ref},
                text="Chain %s to %s" % (element, ref)))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return
        self._set_info("chained to %s" % ref, element)

    def _selected_pair_ref(self, element):
        """The element selected immediately BEFORE `element` in the
        2-deep selection history, or None if `element` isn't the most
        recent selection or there's no earlier one."""
        hist = list(self._select_history)
        if len(hist) >= 2 and hist[-1] == element:
            return hist[-2]
        return None

    def _last_train_element(self, element):
        tm = self._project.train()
        try:
            order = train_solver.sort_chain(tm.records())
        except train_solver.TrainError:
            order = tm.element_labels()
        skip = set(tm.downstream_of(element)) | {element}
        for el in reversed(order):
            if el not in skip:
                return el
        return None

    def anchor_selected(self):
        element = self._cur_element
        if not element:
            self._set_error("Select an element to anchor first")
            return
        try:
            self._apply(lambda: self._project.set_anchored(element))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return
        self._set_info()

    def _on_pick_ref_clicked(self):
        if not self._cur_element:
            self._set_error("Select an element to re-reference first")
            return
        self.begin_pick_reference(self._cur_element)

    # -- pick reference in 3D --------------------------------------------------
    def begin_pick_reference(self, element):
        """Arm a pending 3D reference pick for `element`; the mainwindow
        connects view.pick_face_once to on_reference_picked."""
        self._pending_pick = str(element)
        self._set_info("Click a reference element in the 3D view…")
        self.pickReferenceRequested.emit(str(element))

    def on_reference_picked(self, body_name, face_id=None):
        """The mainwindow calls this after its one-shot 3D pick."""
        element = self._pending_pick
        self._pending_pick = None
        if element is None or not body_name:
            return
        try:
            picked = self._project.element_group(body_name)
        except Exception:
            self._set_error("Unknown body %r" % body_name, element)
            return
        if picked == element:
            self._set_error("cannot chain to itself", element)
            return
        if picked in self._project.train().downstream_of(element):
            self._set_error(
                "cannot chain to its own descendant %s" % picked, element)
            return
        try:
            self._apply(lambda: self._project.set_chain(
                element, {"ref": picked},
                text="Chain %s to %s" % (element, picked)))
        except (ProjectError, TrainError) as exc:
            self._set_error(str(exc), element)
            return
        self._set_info("chained to %s" % picked, element)

    # -- item change routing ---------------------------------------------------
    def _defer(self, fn):
        """Run `fn` on the next event-loop turn, never synchronously
        inside the current call. _on_item_changed fires from INSIDE the
        edited QTreeWidgetItem's own setData/setCheckState call (Qt's
        item-view machinery is still unwinding on the C++ stack above
        us); _apply()'s rebuild does tree.clear(), which destroys that
        same item out from under its own still-running call and
        segfaults once control returns there. Deferring the whole
        mutate-then-rebuild to a fresh event-loop turn lets Qt's edit
        machinery finish first. (Dialog-free callers of commit_field /
        set_mode / toggle_fold / etc. that call the pane API directly,
        as every offscreen test does, never go through this path and are
        unaffected -- only real item edits/checkbox clicks route through
        _on_item_changed.)"""
        QTimer.singleShot(0, fn)

    def _on_item_changed(self, item, column):
        if self._updating or self._committing:
            return
        element = item.data(COL_ELEMENT, ROLE_ELEMENT)
        if element is None:
            return
        if column == COL_FOLD:
            checked = item.checkState(COL_FOLD) == Qt.Checked
            rec = self._project.train().records().get(element, {})
            if rec.get("fold"):
                self._defer(lambda: self.toggle_fold(element, checked))
            elif checked:
                self._defer(lambda: self.mark_fold(element, True))
        elif column == COL_FLIP:
            checked = item.checkState(COL_FLIP) == Qt.Checked
            self._defer(lambda: self.commit_flip(element, checked))
        elif column == COL_MODE:
            mode = item.data(COL_MODE, Qt.EditRole)
            self._defer(lambda: self.set_mode(element, mode))
        elif column == COL_PORT:
            port = item.data(COL_PORT, Qt.EditRole)
            self._defer(lambda: self.commit_port(element, port))
        elif column in _COL_FIELD:
            field = _COL_FIELD[column]
            text = item.data(column, Qt.EditRole)
            self._defer(lambda: self.commit_field(element, field, text))

    # -- selection sync --------------------------------------------------------
    def _note_selected(self, element):
        """Record `element` as the most recent distinct selection (see
        chain_selected / _selected_pair_ref)."""
        if not element:
            return
        if not self._select_history or self._select_history[-1] != element:
            self._select_history.append(element)

    def _set_current_element(self, element):
        if not element:
            return
        self._cur_element = element
        self._note_selected(element)

    def _on_current_item_changed(self, current, _previous):
        if self._updating or current is None:
            return
        element = current.data(COL_ELEMENT, ROLE_ELEMENT)
        if element is not None:
            self._set_current_element(element)
        body = current.data(COL_ELEMENT, ROLE_BODY)
        if body and self._selection is not None:
            self._selection.select(body, origin=ORIGIN)

    def _on_selection_changed(self, body_name, _faces, origin):
        if origin == ORIGIN:
            return
        self._select_body_row(body_name)

    def _select_body_row(self, body_name):
        self._updating = True
        try:
            target = None
            if body_name:
                try:
                    element = self._project.element_group(body_name)
                except Exception:
                    element = None
                for item in self._element_items():
                    if (item.data(COL_ELEMENT, ROLE_BODY) == body_name
                            or item.data(COL_ELEMENT, ROLE_ELEMENT)
                            == element):
                        target = item
                        break
            self.tree.setCurrentItem(target)
            if target is not None:
                self._set_current_element(
                    target.data(COL_ELEMENT, ROLE_ELEMENT))
        finally:
            self._updating = False

    # -- context menu / dialogs ------------------------------------------------
    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        element = item.data(COL_ELEMENT, ROLE_ELEMENT)
        if element is None:
            return
        column = self.tree.columnAt(pos.x())
        menu = self._build_context_menu(element, column=column)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _build_context_menu(self, element, column=None):
        """Build (but do not exec) the right-click menu for `element`'s
        row -- split out from _on_context_menu so tests can exercise the
        menu construction/actions without an event-loop-blocking exec()
        call. `column` (the clicked column) adds the cell-aware "Insert
        optical value" submenu on numeric edge columns."""
        rec = self._project.train().records().get(element, {})
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        act = menu.addAction("Edge details…")
        act.triggered.connect(lambda: self._open_edge_details_dialog(element))
        act = menu.addAction("Pick reference in 3D…")
        act.triggered.connect(lambda: self.begin_pick_reference(element))
        if rec.get("mode") == "chained":
            if rec.get("fold"):
                act = menu.addAction("Remove fold designation")
                act.triggered.connect(lambda: self.mark_fold(element, False))
            else:
                act = menu.addAction("Make fold mirror (unfoldable)")
                act.triggered.connect(lambda: self.mark_fold(element, True))
            act.setToolTip(_FOLD_MARK_TIP)
            act.setStatusTip(_FOLD_MARK_TIP)
            ref = rec.get("ref")
            ref_rec = (self._project.train().records().get(ref)
                      if ref else None)
            if ref_rec is not None:
                port_menu = menu.addMenu("Chain onto port…")
                for p in self._available_ports(ref_rec):
                    pact = port_menu.addAction(p)
                    pact.triggered.connect(
                        lambda checked=False, p=p:
                            self.commit_port(element, p))
        else:
            act = menu.addAction("Set absolute pose…")
            act.triggered.connect(
                lambda: self.editAnchorRequested.emit(element))
        self._add_optical_value_submenu(menu, element, column)
        return menu

    def _add_optical_value_submenu(self, menu, element, column):
        """Cell-aware 'Insert optical value' submenu on numeric edge
        columns. Submenu references are stored ON the menu object
        (menu.optical_value_submenu / .optical_value_groups) — never
        retrieve a submenu back through QAction.menu(), the PySide6
        ownership-transfer GC trap."""
        field = _COL_FIELD.get(column) if column is not None else None
        if not field:
            return
        try:
            entries = opticalvalues.value_menu_model(
                self._project.train(), element, field,
                variables=self._variables())
        except Exception:
            entries = []
        if not entries:
            return
        menu.addSeparator()
        sub = menu.addMenu("Insert optical value")
        sub.setToolTipsVisible(True)
        menu.optical_value_submenu = sub
        groups = {}
        menu.optical_value_groups = groups
        for e in entries:
            g = groups.get(e["group"])
            if g is None:
                g = sub.addMenu(e["group"])
                groups[e["group"]] = g
            act = g.addAction(e["label"])
            act.setToolTip("Insert the literal value %.6g" % e["value"])
            act.triggered.connect(
                lambda checked=False, e=e:
                    self._insert_optical_value(element, field, e, False))
            if e.get("suggest_var"):
                act2 = g.addAction("    … as variable %s = %.6g"
                                   % (e["suggest_var"], e["value"]))
                act2.setToolTip(
                    "Create/update miewb_vars '%s' with this value and "
                    "insert the variable name — the design stays "
                    "re-tunable from the Variables dock"
                    % e["suggest_var"])
                act2.triggered.connect(
                    lambda checked=False, e=e:
                        self._insert_optical_value(element, field, e, True))

    def _insert_optical_value(self, element, field, entry, as_variable):
        """Commit an opticalvalues entry into (element, field) — either
        the literal number or via a created/updated miewb_vars variable."""
        text = "%.6g" % entry["value"]
        if as_variable and entry.get("suggest_var"):
            name = entry["suggest_var"]
            try:
                self._project.ensure_variables_sheet()
                sheet = self._project.variables_sheet()
                rows = V.parse_sheet(sheet)
                if name in rows and rows[name].row:
                    plan = V.cell_plan(name, row=rows[name].row, value=text)
                else:
                    plan = V.cell_plan(
                        name, row=V.next_free_row(sheet), value=text,
                        vmin=entry["value"], vmax=entry["value"], nstep=0,
                        enabled=False,
                        comment="inserted: %s" % entry["label"])
                self._project.apply_variable_cells(
                    plan, text="Insert optical value %s" % name)
            except (ProjectError, TrainError, ValueError) as exc:
                self.status.setText("insert as variable failed: %s" % exc)
                return
            text = name
        self.commit_field(element, field, text)

    def _open_insert_fold_dialog(self):
        if self._project is None or not getattr(self._project, "structure",
                                                None):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Insert fold mirror")
        form = QFormLayout(dlg)
        after = QComboBox()
        for el in self._project.train().element_labels():
            after.addItem(el, el)
        if self._cur_element:
            i = after.findData(self._cur_element)
            if i >= 0:
                after.setCurrentIndex(i)
        form.addRow("After element", after)
        dist = QDoubleSpinBox()
        dist.setRange(-1e5, 1e5)
        dist.setDecimals(3)
        dist.setValue(10.0)
        dist.setSuffix(" mm")
        form.addRow("Distance", dist)
        dev = QDoubleSpinBox()
        dev.setRange(-360.0, 360.0)
        dev.setValue(90.0)
        dev.setSuffix(" °")
        form.addRow("Deviation", dev)
        az = QDoubleSpinBox()
        az.setRange(-360.0, 360.0)
        az.setValue(0.0)
        az.setSuffix(" °")
        form.addRow("Azimuth", az)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() == QDialog.Accepted:
            self.insert_fold(after.currentData(), dist.value(),
                             dev.value(), az.value())

    def _open_edge_details_dialog(self, element):
        rec = self._project.train().records().get(element, {})
        dlg = QDialog(self)
        dlg.setWindowTitle("Edge details — %s" % element)
        form = QFormLayout(dlg)
        rot = QComboBox()
        rot.addItems(list(train_solver.ROT_ORDERS))
        i = rot.findText(rec.get("rot_order") or "xyz")
        rot.setCurrentIndex(max(i, 0))
        form.addRow("Rotation order", rot)
        pos_rot = QComboBox()
        pos_rot.addItems(["pos_first", "rot_first"])
        i = pos_rot.findText(rec.get("pos_rot_order") or "pos_first")
        pos_rot.setCurrentIndex(max(i, 0))
        form.addRow("Position/rotation order", pos_rot)
        pivot = QComboBox()
        pivot.addItems(["entrance", "center", "exit"])
        i = pivot.findText(rec.get("pivot") or "entrance")
        pivot.setCurrentIndex(max(i, 0))
        form.addRow("Pivot", pivot)
        dev = QLineEdit(str(rec.get("fold_deviation") or ""))
        dev.setPlaceholderText("deg, e.g. 90 or an expression "
                               "(deviate-port angle)")
        form.addRow("Fold deviation", dev)
        az = QLineEdit(str(rec.get("fold_azimuth") or ""))
        az.setPlaceholderText("deg, e.g. 0 (deviate-port azimuth)")
        form.addRow("Fold azimuth", az)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() == QDialog.Accepted:
            dev_text = dev.text().strip() or None
            az_text = az.text().strip() or None
            self.set_edge_details(element, rot.currentText(),
                                  pos_rot.currentText(), pivot.currentText(),
                                  fold_deviation=dev_text,
                                  fold_azimuth=az_text)
