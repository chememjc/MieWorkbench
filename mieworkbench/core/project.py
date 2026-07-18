"""Project - the shared model-session object every pane talks to.

Owns the FreeCAD worker client, the tessellation cache, the current
document's structure, and per-body BodyState (live placements for the
transform engine). Panes NEVER call FcClient directly; they call Project
methods and react to its signals, so change routing stays in one place:

    set_spreadsheet/set_property/rebuild -> worker reports
        reshaped bodies -> selective re-tessellation -> bodiesReshaped
        moved bodies    -> placement refresh          -> bodiesMoved
    apply_operation (transform engine) -> instant BodyState update +
        bodiesMoved, then the placement is flushed to the worker
        synchronously (ms) - the 3D view never waits on tessellation
        for a pure move.

Worker calls are synchronous (the FreeCAD round-trip is milliseconds for
property/placement ops; tessellation of changed bodies is the only
seconds-scale call and only runs for reshaped bodies).
"""

import json
import os

from PySide6.QtCore import QObject, Signal

from .fcclient import FcClient
from .geomcache import GeomCache
from .train import (
    TRAIN_GROUP, EDGE_FIELDS, FIELD_PROPS, TrainModel, edge_props,
    variables_from_sheets,
)
from .transforms import (
    BodyState, Operation, Placement, ReferenceResolver, place_about_ops,
    snap_to_axis_ops,
)
from .undostack import Command, UndoStack


class ProjectError(RuntimeError):
    pass


_PTYPE_FROM_FC = {"App::PropertyFloat": "float", "App::PropertyBool": "bool",
                  "App::PropertyString": "string"}


def _num_or_expr(value):
    """Chain-field value -> stored string (floats canonicalized,
    expressions verbatim)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "%.17g" % float(value)
    return str(value)


def _subtract_expr(a, b):
    """a - b for chain-distance fields; stays symbolic when either side
    is a variable expression."""
    try:
        return "%.17g" % (float(a) - float(b))
    except (TypeError, ValueError):
        return "(%s) - (%s)" % (_num_or_expr(a), _num_or_expr(b))


class Project(QObject):
    sceneLoaded = Signal()                 # full (re)load: rebuild all views
    bodiesReshaped = Signal(list)          # [body_name]: re-mesh those actors
    bodiesMoved = Signal(dict)             # {body_name: placement_dict}
    propertiesChanged = Signal(str)        # body_name (or "" for sheet edits)
    dirtyChanged = Signal(bool)
    # "something that affects the traced optics changed": geometry
    # reshapes/moves, element add/delete/duplicate, and every property
    # edit EXCEPT GUI-internal miewb_* bookkeeping (this signal is the
    # only place that distinction can be made -- propertiesChanged's
    # payload is just a body name). Drives the auto ray-preview refresh;
    # undo/redo replay through the same _do_* paths, so they re-emit it
    # correctly for free.
    opticsChanged = Signal()

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._fc = None
        self._cache = None
        self.doc = None
        self.fcstd_path = None
        self.structure = None
        self.faces = {}          # body_name -> {"faces": [...], "placement"}
        self.body_states = {}    # body_name -> BodyState
        self._dirty = False
        self._recompute_diverged = False
        self.undo_stack = UndoStack(parent=self)
        self.undo_stack.indexChanged.connect(self._sync_dirty_from_stack)
        self.stash_root = None   # override for delete-undo stashes
        self._stash_seq = 0

    # -- lifecycle -----------------------------------------------------------
    @property
    def fc(self):
        if self._fc is None:
            appimage = None
            if self.settings is not None:
                appimage = self.settings.get("freecad") or None
            self._fc = FcClient(appimage=appimage)
            self._fc.start()
        return self._fc

    @property
    def cache(self):
        if self._cache is None:
            self._cache = GeomCache(self.fc)
        return self._cache

    def is_open(self):
        return self.doc is not None

    def open_fcstd(self, path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ProjectError("no such file: %s" % path)
        if self.doc is not None:
            self.close()
        self.structure = self.fc.open_document(path)
        self.doc = self.structure["doc"]
        self.fcstd_path = path
        self._refresh_geometry()
        self.undo_stack.clear()
        self._set_dirty(False)
        # the worker's open-time recompute may have changed the doc vs the
        # file (expression-bound placements); that IS unsaved state
        self._set_recompute_diverged(
            bool(self.structure.get("recompute_changed")))
        self.sceneLoaded.emit()

    def new_document(self, path):
        if self.doc is not None:
            self.close()
        self.structure = self.fc.new_document(path)
        self.doc = self.structure["doc"]
        self.fcstd_path = os.path.abspath(path)
        self.faces = {}
        self.body_states = {}
        self.undo_stack.clear()
        self._set_recompute_diverged(False)
        self._set_dirty(False)
        self.sceneLoaded.emit()

    def close(self):
        had_doc = self.doc is not None
        if had_doc:
            try:
                self.fc.close(self.doc)
            except Exception:
                pass
        self.doc = None
        self.fcstd_path = None
        self.structure = None
        self.faces = {}
        self.body_states = {}
        self.undo_stack.clear()
        self._set_recompute_diverged(False)
        self._set_dirty(False)
        if had_doc:
            # views rebuild against the now-empty structure (File -> Close);
            # internal close-before-reopen paths re-emit right after with
            # the new scene, so the extra clear is momentary and harmless
            self.sceneLoaded.emit()

    def revert(self):
        """Discard every unsaved change by re-opening the document from
        its last saved state on disk (the in-memory FreeCAD doc and the
        undo history are dropped wholesale)."""
        if self.doc is None:
            raise ProjectError("no document open to revert")
        path = self.fcstd_path
        self.close()
        self.open_fcstd(path)

    def shutdown(self):
        self.close()
        if self._fc is not None:
            self._fc.shutdown()
            self._fc = None

    # -- views over the structure ---------------------------------------------
    def body(self, name):
        for b in (self.structure or {}).get("bodies", []):
            if b["name"] == name or b["label"] == name:
                return b
        raise ProjectError("no body %r" % name)

    def body_names(self):
        return [b["name"] for b in (self.structure or {}).get("bodies", [])]

    def sheets(self):
        return (self.structure or {}).get("sheets", [])

    def sheet_for_body(self, name):
        """The element parameter sheet for a body: dim_<group> for
        primitive-built elements, else the primary 'dim' sheet if any."""
        b = self.body(name)
        group = b["properties"].get("miewb_group", {}).get("value")
        for sheet in self.sheets():
            if group and sheet["label"] == "dim_%s" % group:
                return sheet
        for sheet in self.sheets():
            if sheet["label"] == "dim":
                return sheet
        return None

    def resolver(self):
        return ReferenceResolver(self.body_states)

    def current_placement(self, name):
        """Live world Placement of a body (None if unknown)."""
        state = self.body_states.get(name)
        return state.current if state is not None else None

    # -- geometry -------------------------------------------------------------
    def _refresh_geometry(self, bodies=None):
        self.faces.update(self.cache.faces_for(
            self.doc, self.fcstd_path, structure=self.structure,
            bodies=bodies))
        for b in self.structure["bodies"]:
            name = b["name"]
            if bodies is not None and name not in bodies \
                    and name in self.body_states:
                # keep the live state; refresh only its placement source
                continue
            meta = self.faces.get(name, {}).get("faces", [])
            self.body_states[name] = BodyState.from_worker(b, meta)

    def _refetch_structure(self):
        self.structure = self.fc.request("get_structure", {"doc": self.doc})

    # -- mutations -------------------------------------------------------------
    def _route_mutation(self, result, body_hint=""):
        reshaped = result.get("changed_bodies", [])
        moved = result.get("moved_bodies", [])
        self._refetch_structure()
        if reshaped:
            self._refresh_geometry(bodies=reshaped)
            self.bodiesReshaped.emit(list(reshaped))
        if moved:
            placements = result.get("placements", {})
            for name in moved:
                if name in self.body_states and name in placements:
                    self.body_states[name].current = Placement.from_dict(
                        placements[name])
            self.bodiesMoved.emit({n: placements.get(n) for n in moved})
        self.propertiesChanged.emit(body_hint)
        if reshaped or moved:
            self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    # Mutations follow the command pattern: the PUBLIC method captures the
    # pre-image, builds a Command around the private _do_* body, and runs
    # it through the undo stack. Undo/redo invoke _do_* directly, so they
    # never re-record. Panes keep calling the public methods unchanged.

    def _do_set_property(self, body, name, value, ptype=None, group=None):
        params = {"doc": self.doc, "body": body, "name": name,
                  "value": value}
        if ptype:
            params["ptype"] = ptype
        if group:
            params["group"] = group
        result = self._route_mutation(
            self.fc.request("set_property", params), body)
        if not str(name).startswith("miewb_"):
            self.opticsChanged.emit()
        return result

    def _do_remove_property(self, body, name):
        result = self._route_mutation(
            self.fc.request("remove_property",
                            {"doc": self.doc, "body": body, "name": name}),
            body)
        if not str(name).startswith("miewb_"):
            self.opticsChanged.emit()
        return result

    def _prop_preimage(self, body, name):
        """(value, ptype) of an existing property, or None if absent."""
        try:
            entry = self.body(body).get("properties", {}).get(name)
        except ProjectError:
            entry = None
        if entry is None:
            return None
        return (entry.get("value"),
                _PTYPE_FROM_FC.get(entry.get("type"), "string"))

    def _body_label(self, body):
        try:
            return self.body(body)["label"]
        except ProjectError:
            return str(body)

    def set_property(self, body, name, value, ptype=None, group=None):
        old = self._prop_preimage(body, name)
        if old is None:
            undo = lambda: self._do_remove_property(body, name)
        else:
            undo = lambda: self._do_set_property(body, name, old[0], old[1],
                                                 group=group)
        self.undo_stack.push_and_do(Command(
            "Set %s on %s" % (name, self._body_label(body)),
            lambda: self._do_set_property(body, name, value, ptype,
                                          group=group),
            undo))

    def remove_property(self, body, name):
        old = self._prop_preimage(body, name)
        if old is None:
            raise ProjectError("%s has no property %r" % (body, name))
        self.undo_stack.push_and_do(Command(
            "Remove %s from %s" % (name, self._body_label(body)),
            lambda: self._do_remove_property(body, name),
            lambda: self._do_set_property(body, name, old[0], old[1])))

    def _do_set_spreadsheet(self, sheet, alias, raw):
        result = self.fc.request("set_spreadsheet",
                                 {"doc": self.doc, "sheet": sheet,
                                  "alias": alias, "raw": raw})
        return self._route_mutation(result)

    def _sheet_raw(self, sheet_key, alias):
        for s in self.sheets():
            if s["label"] == sheet_key or s["name"] == sheet_key:
                entry = s.get("aliases", {}).get(alias)
                return None if entry is None else entry.get("raw")
        return None

    def _do_rebuild_primitive(self, group):
        result = self.fc.request("rebuild_primitive",
                                 {"doc": self.doc, "group": group})
        # a rebuild replaces bodies wholesale: refresh everything for the
        # group members reported back
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.bodiesReshaped.emit(names)
        self.propertiesChanged.emit(names[0] if names else "")
        self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    def set_element_parameters(self, sheet, values, rebuild_group=None,
                               text=None):
        """Set one or more aliased cells and (optionally) rebuild the
        primitive group they drive -- ONE undoable step, so undo restores
        the old values AND re-derives the geometry in the right order."""
        old = {alias: self._sheet_raw(sheet, alias) for alias in values}

        def apply(vals):
            for alias, raw in vals.items():
                self._do_set_spreadsheet(sheet, alias, raw)
            if rebuild_group:
                self._do_rebuild_primitive(rebuild_group)
        self.undo_stack.push_and_do(Command(
            text or "Edit %s" % ", ".join(sorted(values)),
            lambda: apply(values), lambda: apply(old)))

    def set_spreadsheet(self, sheet, alias, raw, rebuild_group=None):
        self.set_element_parameters(sheet, {alias: raw},
                                    rebuild_group=rebuild_group)

    def rebuild_primitive(self, group):
        """Re-derive an element's geometry from its parameter sheet. Not
        an undoable step by itself (it changes no authored state); flows
        that edit the sheet should use set_element_parameters instead."""
        return self._do_rebuild_primitive(group)

    def _do_import_primitive(self, path, label):
        result = self.fc.request("import_primitive",
                                 {"doc": self.doc, "path": path,
                                  "label": label})
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.sceneLoaded.emit()      # new bodies + sheets: full view rebuild
        self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    def import_primitive(self, path, label):
        self.undo_stack.push_and_do(Command(
            "Add %s" % label,
            lambda: self._do_import_primitive(path, label),
            lambda: self._do_delete_element(label)))

    # -- element-level operations (delete / duplicate / restore) --------------
    def element_group(self, body_name):
        """The element identity of a body: its miewb_group value, or its
        own label for ungrouped single-body elements."""
        b = self.body(body_name)
        return (b["properties"].get("miewb_group", {}).get("value")
                or b["label"])

    def element_bodies(self, element):
        """Body names belonging to an element (group value or a member
        body's name/label)."""
        element = str(element)
        names = [b["name"] for b in self.structure.get("bodies", [])
                 if b["properties"].get("miewb_group", {}).get("value")
                 == element]
        if names:
            return names
        b = self.body(element)
        group = b["properties"].get("miewb_group", {}).get("value")
        if group:
            return [x["name"] for x in self.structure.get("bodies", [])
                    if x["properties"].get("miewb_group", {}).get("value")
                    == group]
        return [b["name"]]

    def _do_delete_element(self, element, stash_path=None):
        params = {"doc": self.doc, "element": str(element)}
        if stash_path:
            params["stash_path"] = str(stash_path)
        result = self.fc.request("delete_element", params)
        for name in result.get("deleted", []):
            self.faces.pop(name, None)
            self.body_states.pop(name, None)
        self._refetch_structure()
        self.sceneLoaded.emit()      # bodies + sheets gone: full rebuild
        self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    def _do_restore_from_stash(self, path):
        result = self.fc.request("import_bodies",
                                 {"doc": self.doc, "path": str(path)})
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.sceneLoaded.emit()
        self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    def _new_stash_path(self):
        root = self.stash_root or os.path.join(
            os.path.dirname(self.fcstd_path or "."), ".miewb_undo")
        os.makedirs(root, exist_ok=True)
        self._stash_seq += 1
        return os.path.join(root, "stash_%03d.FCStd" % self._stash_seq)

    def delete_element(self, element):
        """Delete an element (group of bodies + its dim sheet). Undoable:
        the worker stashes the element to a standalone .FCStd first (the
        pre-image restore_from_stash re-imports verbatim); the stash file
        is cleaned up when the command falls off the undo stack."""
        element = str(element)
        stash = self._new_stash_path()

        def cleanup():
            try:
                os.remove(stash)
            except OSError:
                pass
        self.undo_stack.push_and_do(Command(
            "Delete %s" % element,
            lambda: self._do_delete_element(element, stash_path=stash),
            lambda: self._do_restore_from_stash(stash),
            cleanup=cleanup))

    def _do_duplicate_element(self, element, new_label):
        result = self.fc.request("duplicate_element",
                                 {"doc": self.doc, "element": str(element),
                                  "new_label": str(new_label)})
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.sceneLoaded.emit()
        self.opticsChanged.emit()
        self._set_dirty(True)
        return result

    def duplicate_element(self, element, new_label):
        """Copy an element in-document under a new label/group (the paste
        half of copy/paste). Undoable (undo deletes the copy)."""
        self.undo_stack.push_and_do(Command(
            "Paste %s" % new_label,
            lambda: self._do_duplicate_element(element, new_label),
            lambda: self._do_delete_element(new_label)))

    # -- transforms --------------------------------------------------------------
    def apply_operation(self, body_name, operation):
        """Apply a transforms.Operation: instant view update via BodyState,
        then flush the placement to the worker. Expression-bound
        placements are refused by the worker with a message naming the
        driving alias - surface that as ProjectError."""
        state = self.body_states.get(body_name)
        if state is None:
            raise ProjectError("no body state for %r" % body_name)
        bdict = self.body(body_name)
        if bdict.get("placement_bound"):
            raise ProjectError(
                "%s's position is driven by a spreadsheet expression; "
                "edit the driving alias instead" % bdict["label"])
        pre = state.current.to_dict()
        operation.apply(self.resolver(), state)
        placement = state.current.to_dict()
        self._flush_placement(body_name, placement)
        # the move already happened (instant view feedback); record the
        # before/after placements so undo/redo replay them directly
        self.undo_stack.push_done(Command(
            "Move %s" % self._body_label(body_name),
            lambda: self._do_apply_placement(body_name, placement),
            lambda: self._do_apply_placement(body_name, pre)))
        return placement

    def snap_to_axis(self, body_name, target_point, target_axis,
                     offset_mm=0.0):
        """Align an element's optical axis to `target_axis` and center it
        on the axis line through `target_point`, then translate it
        `offset_mm` along that axis. The whole snap is one undo entry.
        Returns the list of applied Operations (may be empty if already
        on-axis and no offset). Refuses expression-bound placements."""
        state = self.body_states.get(body_name)
        if state is None:
            raise ProjectError("no body state for %r" % body_name)
        if self.body(body_name).get("placement_bound"):
            raise ProjectError(
                "%s's position is driven by a spreadsheet expression; "
                "edit the driving alias instead"
                % self.body(body_name)["label"])
        ops = snap_to_axis_ops(state, target_point, target_axis)
        if offset_mm:
            axis = self.resolver().resolve_axis(
                {"kind": "vector", "vector": list(target_axis)})
            ops.append(Operation("translate", {
                "vector_mm": [float(offset_mm) * float(a) for a in axis]}))
        if not ops:
            return []
        element = self.element_group(body_name)
        tm = self.train()
        self.begin_macro("Snap %s to axis" % self._body_label(body_name))
        try:
            for op in ops:
                self.apply_operation(body_name, op)
            # keep the optical train consistent, exactly like move_element
            if element in tm.records():
                if tm.is_chained(element):
                    edge = {k: float(v) for k, v in self.train().derive_edge(
                        element, self.train_variables()).items()
                            if k in EDGE_FIELDS}
                    body = tm.primary_body_name(element)
                    for name, value in sorted(edge_props(edge).items()):
                        self.set_property(body, name, value, ptype="string",
                                          group=TRAIN_GROUP)
                if tm.is_chained(element) or tm.downstream_of(element):
                    self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        return ops

    def place_about_point(self, body_name, ref_spec, axis_spec, r_expr,
                          theta_expr, aim_at_ref=True):
        """Polar/spherical place-about-point (future.md (a2): the "detector
        40 mm at 90 deg from the cloud center" nephelometer-ring /
        field-angle-fan affordance) — puts `body_name` on the circle of
        radius r_expr about the point `ref_spec` resolves to, in the plane
        perpendicular to `axis_spec`, at angle theta_expr. r_expr/
        theta_expr accept the train_solver expression grammar
        (train_solver.EXPR_HELP) evaluated against the live miewb_vars
        sheet — same grammar and vars mapping the chain edge fields use
        (mm / degrees, DEGREES-native trig). One undo entry; keeps the
        optical train consistent exactly like snap_to_axis (a chained
        downstream element ripples). Refuses expression-bound
        placements. Returns the applied Operation list."""
        import train_solver
        state = self.body_states.get(body_name)
        if state is None:
            raise ProjectError("no body state for %r" % body_name)
        if self.body(body_name).get("placement_bound"):
            raise ProjectError(
                "%s's position is driven by a spreadsheet expression; "
                "edit the driving alias instead"
                % self.body(body_name)["label"])
        variables = self.train_variables()
        try:
            r_mm = train_solver.eval_expr(r_expr, variables)
            theta_deg = train_solver.eval_expr(theta_expr, variables)
        except train_solver.TrainError as exc:
            raise ProjectError(str(exc))
        ops = place_about_ops(state, self.resolver(), ref_spec, axis_spec,
                              r_mm, theta_deg, aim_at_ref=aim_at_ref)
        element = self.element_group(body_name)
        tm = self.train()
        self.begin_macro("Place %s about point"
                         % self._body_label(body_name))
        try:
            for op in ops:
                self.apply_operation(body_name, op)
            # keep the optical train consistent, exactly like move_element
            if element in tm.records():
                if tm.is_chained(element):
                    edge = {k: float(v) for k, v in self.train().derive_edge(
                        element, self.train_variables()).items()
                            if k in EDGE_FIELDS}
                    body = tm.primary_body_name(element)
                    for name, value in sorted(edge_props(edge).items()):
                        self.set_property(body, name, value, ptype="string",
                                          group=TRAIN_GROUP)
                if tm.is_chained(element) or tm.downstream_of(element):
                    self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        return ops

    # -- optical train -----------------------------------------------------------
    def train(self):
        """A fresh TrainModel snapshot over the current structure (cheap;
        build one per interaction, never cache across mutations)."""
        return TrainModel(self.structure, self.body_states)

    def build_prescription(self):
        """The prescription-primary document (engine3 Sec 3, P5) for the
        current scene, or None if no element is a covered primitive. Keyed by
        miewb_group (== the TrainModel element label), computed from each
        element's kind (miewb_primitive) + dim params through the SAME pure
        primitivelib.build_prescription_entry the extractor cross-checks
        against (the single authoring path). Packed into the .MieWB so the
        extractor verifies + emits the optical surfaces from it."""
        try:
            import primitivelib
            from raytracer import prescription as prescription_mod
        except Exception:
            return None
        tm = self.train()
        entries = {}
        for element in tm.element_labels():
            primary = tm.primary_body(element)
            kind = (primary.get("properties", {}).get("miewb_primitive")
                    or {}).get("value")
            if not kind:
                continue
            params = tm._sheet_params(element)
            if not params:
                continue
            entry = primitivelib.build_prescription_entry(kind, params)
            if entry is not None:
                entries[element] = entry
        if not entries:
            return None
        return prescription_mod.new_document(entries)

    def train_variables(self):
        """{name: float} from the miewb_vars sheet (FreeCAD-evaluated)."""
        return variables_from_sheets(self.sheets())

    def _push_ripple_moves(self, text_prefix="Ripple"):
        """Re-solve the train and flush every chained element whose pose
        changed, pushing one done-command per move. MUST be called inside
        an open macro (the composite operators own the macro; nothing
        here opens one). Returns the number of elements moved."""
        tm = self.train()
        moves = tm.solve_moves(self.train_variables())
        for element, placement in moves.items():
            body = tm.primary_body_name(element)
            state = self.body_states.get(body)
            pre = state.current.to_dict() if state else dict(placement)
            self._do_apply_placement(body, placement)
            self.undo_stack.push_done(Command(
                "%s %s" % (text_prefix, element),
                lambda b=body, p=placement: self._do_apply_placement(b, p),
                lambda b=body, p=pre: self._do_apply_placement(b, p)))
        return len(moves)

    def _write_chain_props(self, element, edge):
        """Write chain-edge fields as MieTrain properties on the
        element's primary body — undoable children only; the CALLER owns
        the macro. `edge` is written verbatim (no mode defaulting)."""
        body = self.train().primary_body_name(element)
        for name, value in sorted(edge_props(edge).items()):
            ptype = "bool" if isinstance(value, bool) else "string"
            self.set_property(body, name, value, ptype=ptype,
                              group=TRAIN_GROUP)

    def set_chain(self, element, edge, text=None):
        """Chain an element / edit its chain edge, then ripple everything
        downstream — one undo step. `edge` maps solver record fields
        (mode, ref, port, distance, decenter_x/y, tilt_rx/ry/rz,
        rot_order, pos_rot_order, pivot, fold, folded, fold_deviation,
        fold_azimuth) to values; numeric fields accept variable
        expressions ("2*gap+5"); mode defaults to "chained"."""
        element = str(element)
        tm = self.train()
        body = tm.primary_body_name(element)
        if self.body(body).get("placement_bound"):
            raise ProjectError(
                "%s's position is driven by a spreadsheet expression; "
                "unbind it before chaining" % element)
        edge = dict(edge)
        edge.setdefault("mode", "chained")
        self.begin_macro(text or "Chain %s" % element)
        try:
            self._write_chain_props(element, edge)
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()

    def set_anchored(self, element):
        """Freeze an element at its current pose (chain edge fields stay
        behind, inert, for painless re-chaining). Undoable."""
        element = str(element)
        body = self.train().primary_body_name(element)
        self.set_property(body, FIELD_PROPS["mode"], "anchored",
                          ptype="string", group=TRAIN_GROUP)

    def sync_chain_from_pose(self, element, text=None):
        """After a spatial move of a CHAINED element: re-derive its
        distance/decenter/tilt from the new pose (expressions are
        replaced by literals) and ripple downstream. One undo step."""
        element = str(element)
        edge = self.train().derive_edge(element, self.train_variables())
        edge = {k: float(v) for k, v in edge.items() if k in EDGE_FIELDS}
        self.set_chain(element, edge,
                       text=text or "Reposition %s" % element)

    # -- global variables (miewb_vars sheet) ---------------------------------------
    def variables_sheet(self):
        """The miewb_vars sheet echo dict, or None."""
        for sheet in self.sheets():
            if sheet.get("label") == "miewb_vars" \
                    or sheet.get("name") == "miewb_vars":
                return sheet
        return None

    def ensure_variables_sheet(self):
        """Create the miewb_vars sheet if absent (idempotent worker op;
        an empty sheet is not undoable state worth tracking)."""
        if self.variables_sheet() is None:
            self.fc.request("create_sheet",
                            {"doc": self.doc, "label": "miewb_vars"})
            self._refetch_structure()
            self.propertiesChanged.emit("")
        return self.variables_sheet()

    def _do_set_cell(self, cell, raw, alias=None):
        params = {"doc": self.doc, "sheet": "miewb_vars",
                  "cell": cell, "raw": raw}
        if alias:
            params["alias"] = alias
        return self._route_mutation(self.fc.request("set_cell", params))

    def _cell_preimage(self, cell):
        """(raw, alias) currently in a miewb_vars cell ("" when empty —
        the echo only lists aliased cells, so un-aliased comment cells
        restore to empty on undo)."""
        sheet = self.variables_sheet() or {}
        for alias, entry in (sheet.get("aliases") or {}).items():
            if entry.get("cell") == cell:
                return (entry.get("raw") or "", alias)
        return ("", None)

    def _vars_referencing_groups(self):
        """Primitive groups whose dim_* sheet has any raw cell content
        mentioning miewb_vars — these must REBUILD when a variable
        changes (the GUI-side twin of permute_model's
        extend_touched_for_miewb_vars)."""
        groups = []
        for sheet in self.sheets():
            label = sheet.get("label") or ""
            if not label.startswith("dim_"):
                continue
            for entry in (sheet.get("aliases") or {}).values():
                if "miewb_vars" in str(entry.get("raw") or ""):
                    groups.append(label[len("dim_"):])
                    break
        return groups

    def apply_variable_cells(self, cells, text="Edit variables"):
        """Write miewb_vars cells (a core.variables.cell_plan list of
        {cell, raw, alias?}), rebuild any variable-referencing primitive
        groups, and ripple the train — ONE undo step."""
        self.ensure_variables_sheet()
        pre = {c["cell"]: self._cell_preimage(c["cell"]) for c in cells}
        self.begin_macro(text)
        try:
            for c in cells:
                old_raw, old_alias = pre[c["cell"]]
                cell, raw = c["cell"], c["raw"]
                alias = c.get("alias")
                self.undo_stack.push_and_do(Command(
                    "%s %s" % (text, cell),
                    lambda cl=cell, r=raw, a=alias:
                        self._do_set_cell(cl, r, a),
                    lambda cl=cell, r=old_raw, a=old_alias:
                        self._do_set_cell(cl, r, a)))
            for group in self._vars_referencing_groups():
                # rebuild is derived (not authored) state: replay on both
                # undo and redo so geometry re-derives either way
                self.undo_stack.push_and_do(Command(
                    "Rebuild %s" % group,
                    lambda g=group: self._do_rebuild_primitive(g),
                    lambda g=group: self._do_rebuild_primitive(g)))
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        self.opticsChanged.emit()

    # -- optimize/tolerance pane persistence (miewb_vars sheet) ------------------
    # Storage: JSON strings in two dynamic document properties on the
    # miewb_vars Spreadsheet object (created on first save, same as the
    # variables themselves) -- {"version": 1, "optimize"/"tolerance": cfg}
    # where cfg is exactly what OptimizePane.config()/TolerancePane.config()
    # produce (sheet-qualified var/tolerance/compensator specs). Travels
    # with the .FCStd automatically (miewb_tool packs it verbatim), so a
    # reopened scene's panes can be pre-populated via apply_config().
    _CONFIG_VERSION = 1
    OPTIMIZE_CONFIG_PROP = "miewb_optimize_config"
    TOLERANCE_CONFIG_PROP = "miewb_tolerance_config"

    def _config_raw(self, prop_name):
        """The raw JSON string currently stashed in `prop_name` on the
        miewb_vars sheet, or None (no sheet, or the sheet carries no such
        property yet)."""
        sheet = self.variables_sheet()
        if sheet is None:
            return None
        entry = (sheet.get("properties") or {}).get(prop_name)
        return None if entry is None else entry.get("value")

    def _get_config(self, prop_name, key):
        """{key: <pane config dict>} from the stashed JSON, or None (no
        sheet / no property / unparseable / version mismatch all degrade
        to "nothing stored" -- a fresh scene or a hand-edited document
        must never raise here)."""
        raw = self._config_raw(prop_name)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) \
                or payload.get("version") != self._CONFIG_VERSION:
            return None
        return payload.get(key)

    def get_optimize_config(self):
        """The last-saved OptimizePane.config() dict, or None."""
        return self._get_config(self.OPTIMIZE_CONFIG_PROP, "optimize")

    def get_tolerance_config(self):
        """The last-saved TolerancePane.config() dict, or None."""
        return self._get_config(self.TOLERANCE_CONFIG_PROP, "tolerance")

    def _do_set_sheet_property(self, sheet_name, name, raw):
        return self._route_mutation(
            self.fc.request("set_property",
                            {"doc": self.doc, "body": sheet_name,
                             "name": name, "value": raw, "ptype": "string",
                             "group": "Base"}),
            sheet_name)

    def _do_remove_sheet_property(self, sheet_name, name):
        return self._route_mutation(
            self.fc.request("remove_property",
                            {"doc": self.doc, "body": sheet_name,
                             "name": name}),
            sheet_name)

    def _set_config(self, prop_name, key, cfg, text):
        """Stash (cfg is not None) or clear (cfg is None) {"version": 1,
        key: cfg} in `prop_name` on the miewb_vars sheet -- one undoable
        Command with pre-image capture, no-op when the serialized config
        is unchanged. The sheet is created on first use (matching
        apply_variable_cells: the create itself is not tracked, only the
        property write is)."""
        old_raw = self._config_raw(prop_name)
        new_raw = (None if cfg is None else
                  json.dumps({"version": self._CONFIG_VERSION, key: cfg},
                            sort_keys=True))
        if new_raw == old_raw:
            return
        if new_raw is None:
            sheet = self.variables_sheet()
            if sheet is None:
                return
            sheet_name = sheet["name"]
            self.undo_stack.push_and_do(Command(
                text,
                lambda: self._do_remove_sheet_property(sheet_name, prop_name),
                lambda: self._do_set_sheet_property(sheet_name, prop_name,
                                                    old_raw)))
            return
        sheet = self.ensure_variables_sheet()
        sheet_name = sheet["name"]
        if old_raw is None:
            undo = lambda: self._do_remove_sheet_property(sheet_name,
                                                           prop_name)
        else:
            undo = lambda: self._do_set_sheet_property(sheet_name, prop_name,
                                                        old_raw)
        self.undo_stack.push_and_do(Command(
            text,
            lambda: self._do_set_sheet_property(sheet_name, prop_name,
                                                new_raw),
            undo))

    def set_optimize_config(self, cfg):
        """Persist (cfg dict) or clear (cfg=None) the OptimizePane's
        current config() on the miewb_vars sheet. Undoable; no-op when
        unchanged."""
        self._set_config(self.OPTIMIZE_CONFIG_PROP, "optimize", cfg,
                         "Save optimize configuration")

    def set_tolerance_config(self, cfg):
        """Persist (cfg dict) or clear (cfg=None) the TolerancePane's
        current config() on the miewb_vars sheet. Undoable; no-op when
        unchanged."""
        self._set_config(self.TOLERANCE_CONFIG_PROP, "tolerance", cfg,
                         "Save tolerance configuration")

    # -- folds -------------------------------------------------------------------
    def _fold_record(self, element):
        rec = self.train().records().get(str(element))
        if rec is None:
            raise ProjectError("unknown element %r" % element)
        if not rec.get("fold"):
            raise ProjectError("%s is not a fold element" % element)
        return rec

    def _set_fold_state_children(self, element, folded):
        """Macro children for one fold toggle (no macro management, no
        ripple — the caller batches those)."""
        import json as _json
        element = str(element)
        tm = self.train()
        primary = tm.primary_body_name(element)
        if not folded:
            # stash current poses of the fold + its downstream as the
            # exact-refold safety net (deterministic re-solve is primary)
            stash = {}
            for el in [element] + tm.downstream_of(element):
                state = self.body_states.get(tm.primary_body_name(el))
                if state is not None:
                    stash[el] = state.current.to_dict()
            self.set_property(primary, "miewb_train_unfold_stash",
                              _json.dumps(stash), ptype="string",
                              group=TRAIN_GROUP)
        self._write_chain_props(element, {"folded": bool(folded)})
        # sim exclusion rides on EVERY body of the fold mirror; the
        # extractor skips excluded bodies entirely
        for body in self.element_bodies(element):
            self.set_property(body, "miewb_exclude", not folded,
                              ptype="bool", group=TRAIN_GROUP)

    def set_fold_state(self, element, folded):
        """Unfold (straighten the downstream train, ghost + sim-exclude
        the mirror) or refold. Pure re-solve either way; refolding
        reproduces the folded placements exactly. One undo step."""
        element = str(element)
        rec = self._fold_record(element)
        if bool(rec.get("folded", True)) == bool(folded):
            return
        self.begin_macro(("Refold %s" if folded else "Unfold %s")
                         % element)
        try:
            self._set_fold_state_children(element, folded)
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        # exclusion changes the traced physics even though the props are
        # miewb_* (normally preview-silent) — refresh the ray preview
        self.opticsChanged.emit()

    def set_folds_all(self, folded):
        """Toggle every fold element at once (one undo step)."""
        tm = self.train()
        targets = [el for el in tm.folds()
                   if bool(tm.records()[el].get("folded", True))
                   != bool(folded)]
        if not targets:
            return []
        self.begin_macro("%s all folds"
                         % ("Refold" if folded else "Unfold"))
        try:
            for element in targets:
                self._set_fold_state_children(element, folded)
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        self.opticsChanged.emit()
        return targets

    def insert_fold_mirror(self, after_element, distance, label=None,
                           kind="mirror_flat", deviation_deg=90.0,
                           azimuth_deg=0.0, port=None, params=None):
        """Insert a fold mirror `distance` mm down-beam of
        `after_element`'s exit port and re-anchor that port's existing
        chained children onto the reflected beam (their along-beam
        distances re-measure from the mirror plane, so path lengths are
        preserved). deviation_deg is the beam deviation (90 = right
        angle); azimuth_deg spins the fold plane about the incoming beam
        (0 folds toward +u, 90 toward +v/up). One undo step; returns the
        new element's label."""
        import os.path as _osp
        after_element = str(after_element)
        tm = self.train()
        parent_rec = tm.records().get(after_element)
        if parent_rec is None:
            raise ProjectError("unknown element %r" % after_element)
        import train_solver
        port = port or train_solver._default_port(parent_rec)
        if label is None:
            base, n = "Fold", 1
            existing = set(tm.element_labels())
            while "%s%d" % (base, n) in existing:
                n += 1
            label = "%s%d" % (base, n)
        # children currently hanging off that port re-anchor to the mirror
        children = []
        for el, rec in tm.records().items():
            if rec.get("mode") == "chained" and rec.get("ref") \
                    == after_element:
                child_port = rec.get("port") or train_solver._default_port(
                    parent_rec)
                if child_port == port:
                    children.append((el, rec))
        tilt = -(180.0 - float(deviation_deg)) / 2.0
        prim_path = _osp.join(_osp.dirname(_osp.dirname(
            _osp.dirname(_osp.abspath(__file__)))), "primitives",
            "%s.FCStd" % kind)
        self.begin_macro("Insert fold %s after %s" % (label, after_element))
        try:
            self.import_primitive(prim_path, label)
            if params:
                sheet = "dim_%s" % label
                self.set_element_parameters(
                    sheet, {k: "=%.10g mm" % float(v)
                            for k, v in params.items()},
                    rebuild_group=label)
            self._write_chain_props(label, {
                "mode": "chained", "ref": after_element, "port": port,
                "distance": _num_or_expr(distance),
                "fold": True, "folded": True,
                "rot_order": "zyx",
                "tilt_rz": "%.10g" % float(azimuth_deg),
                "tilt_ry": "%.10g" % tilt,
            })
            for el, rec in children:
                new_dist = _subtract_expr(rec.get("distance", "0"),
                                          distance)
                self._write_chain_props(el, {
                    "ref": label, "port": "reflect",
                    "distance": new_dist})
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        return label

    def fold_about_surface(self, mirror_element, extra_elements=None):
        """Turn an EXISTING chained mirror into a fold: mark it, switch
        its chained children onto the reflect port, and (optionally)
        rigidly rotate explicitly-listed anchored elements about the
        mirror plane by the fold rotation. One undo step."""
        import train_solver
        mirror_element = str(mirror_element)
        tm = self.train()
        rec = tm.records().get(mirror_element)
        if rec is None:
            raise ProjectError("unknown element %r" % mirror_element)
        if rec.get("mode") != "chained":
            raise ProjectError(
                "%s must be chained before it can fold the train (its "
                "plane needs an incoming beam)" % mirror_element)
        loc = rec.get("local") or {}
        if not loc.get("reflect_plane"):
            raise ProjectError("%s has no reflective surface"
                               % mirror_element)
        solved = tm.solve(self.train_variables())
        frames = solved["frames"]
        parent_frame = tm.parent_frame(mirror_element,
                                       self.train_variables())
        primary = tm.primary_body_name(mirror_element)
        placement = self.body_states[primary].current.to_dict()
        rp = loc["reflect_plane"]
        pt_w = train_solver.transform_point(placement, rp["point"])
        n_w = train_solver.transform_vector(placement, rp["normal"])
        M, _ = train_solver.fold_rotation(parent_frame["dir"], pt_w, n_w)
        children = [el for el, r in tm.records().items()
                    if r.get("mode") == "chained"
                    and r.get("ref") == mirror_element]
        self.begin_macro("Fold train about %s" % mirror_element)
        try:
            self._write_chain_props(mirror_element,
                                    {"fold": True, "folded": True})
            for el in children:
                self._write_chain_props(el, {"port": "reflect"})
            for el in (extra_elements or []):
                body = tm.primary_body_name(str(el))
                cur = self.body_states[body].current.to_dict()
                new_pl = train_solver.apply_to_placement(M, cur)
                self.apply_operation(body, Operation("set_placement", {
                    "pos_mm": new_pl["pos_mm"], "quat": new_pl["quat"]}))
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        self.opticsChanged.emit()

    def move_element(self, body_name, operation):
        """Pane-facing move that keeps the optical train consistent: a
        chained element's edge fields re-derive from the new pose, and
        everything chained downstream follows rigidly. One undo step.
        (apply_operation stays the raw primitive — no ripple.)"""
        element = self.element_group(body_name)
        tm = self.train()
        in_train = element in tm.records() and (
            tm.is_chained(element) or tm.downstream_of(element))
        if not in_train:
            return self.apply_operation(body_name, operation)
        self.begin_macro("Move %s" % self._body_label(body_name))
        try:
            placement = self.apply_operation(body_name, operation)
            if tm.is_chained(element):
                edge = {k: float(v) for k, v in self.train().derive_edge(
                    element, self.train_variables()).items()
                        if k in EDGE_FIELDS}
                body = tm.primary_body_name(element)
                for name, value in sorted(edge_props(edge).items()):
                    self.set_property(body, name, value, ptype="string",
                                      group=TRAIN_GROUP)
            self._push_ripple_moves()
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        return placement

    def _flush_placement(self, body_name, placement):
        # address the worker by the element GROUP whenever the body has
        # one: op_set_placement's group-first match then moves EVERY
        # member rigidly. Passing a member's internal name would fall to
        # the single-body path and tear multi-body elements apart (found
        # by the achromat: no member carries the element label, so the
        # flint stayed behind while the crown moved).
        target = body_name
        try:
            b = self.body(body_name)
            group = (b.get("properties", {}).get("miewb_group") or {}) \
                .get("value")
            if group:
                target = group
        except ProjectError:
            pass
        result = self.fc.request(
            "set_placement",
            {"doc": self.doc, "body": target,
             "pos_mm": placement["pos_mm"], "quat": placement["quat"]})
        # set_placement moves the WHOLE miewb_group rigidly and returns
        # every member's new placement; consume them so sibling BodyStates
        # (and their scene actors) don't go stale (a multi-body element
        # would otherwise tear apart on a move - only the addressed body
        # tracked the new pose). Fall back to the single body if the
        # worker returned nothing (older workers / single bodies).
        placements = (result or {}).get("placements") or {body_name: placement}
        for name, pl in placements.items():
            state = self.body_states.get(name)
            if state is not None:
                state.current = Placement.from_dict(pl)
        self.bodiesMoved.emit(dict(placements))
        self.opticsChanged.emit()
        self._set_dirty(True)

    def _do_apply_placement(self, body_name, placement):
        state = self.body_states.get(body_name)
        if state is None:
            raise ProjectError("no body state for %r" % body_name)
        state.current = Placement.from_dict(placement)
        self._flush_placement(body_name, placement)

    # -- undo/redo facade ------------------------------------------------------
    def undo(self):
        return self.undo_stack.undo()

    def redo(self):
        return self.undo_stack.redo()

    def begin_macro(self, text):
        self.undo_stack.begin_macro(text)

    def end_macro(self):
        self.undo_stack.end_macro()

    def abort_macro(self):
        self.undo_stack.abort_macro()

    def _sync_dirty_from_stack(self):
        # undoing back to the last-saved index makes the doc clean again
        if self.undo_stack.is_clean() and self._dirty:
            self._set_dirty(False)

    # -- persistence ---------------------------------------------------------------
    def export_fcstd(self, path):
        """Write a standalone .FCStd copy of the CURRENT document state
        (fold states, placements and MieTrain metadata as they stand).
        The copy opens and edits in plain FreeCAD; the live document and
        dirty state are untouched."""
        path = os.path.abspath(path)
        self.fc.request("save_copy", {"doc": self.doc, "path": path})
        return path

    def save(self):
        self.fc.request("save", {"doc": self.doc})
        self.undo_stack.mark_clean()
        self._set_recompute_diverged(False)
        self._set_dirty(False)

    def save_as(self, path):
        self.fc.request("save_as", {"doc": self.doc, "path": path})
        self.fcstd_path = os.path.abspath(path)
        self.undo_stack.mark_clean()
        self._set_recompute_diverged(False)
        self._set_dirty(False)

    def is_dirty(self):
        # _recompute_diverged: the open-time recompute changed the doc vs
        # its file (expression-bound placements etc.) — genuinely unsaved
        # state even though the user made no edit. Kept separate from
        # _dirty so undoing to a clean stack can't mask it.
        return self._dirty or self._recompute_diverged

    def _set_dirty(self, flag):
        before = self._dirty or self._recompute_diverged
        self._dirty = flag
        after = self._dirty or self._recompute_diverged
        if after != before:
            self.dirtyChanged.emit(after)

    def _set_recompute_diverged(self, flag):
        before = self._dirty or self._recompute_diverged
        self._recompute_diverged = flag
        after = self._dirty or self._recompute_diverged
        if after != before:
            self.dirtyChanged.emit(after)
