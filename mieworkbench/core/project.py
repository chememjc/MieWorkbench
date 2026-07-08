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

import os

from PySide6.QtCore import QObject, Signal

from .fcclient import FcClient
from .geomcache import GeomCache
from .transforms import (
    BodyState, Operation, Placement, ReferenceResolver, snap_to_axis_ops,
)
from .undostack import Command, UndoStack


class ProjectError(RuntimeError):
    pass


_PTYPE_FROM_FC = {"App::PropertyFloat": "float", "App::PropertyBool": "bool",
                  "App::PropertyString": "string"}


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

    def _do_set_property(self, body, name, value, ptype=None):
        params = {"doc": self.doc, "body": body, "name": name,
                  "value": value}
        if ptype:
            params["ptype"] = ptype
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

    def set_property(self, body, name, value, ptype=None):
        old = self._prop_preimage(body, name)
        if old is None:
            undo = lambda: self._do_remove_property(body, name)
        else:
            undo = lambda: self._do_set_property(body, name, old[0], old[1])
        self.undo_stack.push_and_do(Command(
            "Set %s on %s" % (name, self._body_label(body)),
            lambda: self._do_set_property(body, name, value, ptype), undo))

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
        self.begin_macro("Snap %s to axis" % self._body_label(body_name))
        try:
            for op in ops:
                self.apply_operation(body_name, op)
        except Exception:
            self.abort_macro()
            raise
        self.end_macro()
        return ops

    def _flush_placement(self, body_name, placement):
        result = self.fc.request(
            "set_placement",
            {"doc": self.doc, "body": body_name,
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
    def save(self):
        self.fc.request("save", {"doc": self.doc})
        self.undo_stack.mark_clean()
        self._set_dirty(False)

    def save_as(self, path):
        self.fc.request("save_as", {"doc": self.doc, "path": path})
        self.fcstd_path = os.path.abspath(path)
        self.undo_stack.mark_clean()
        self._set_dirty(False)

    def is_dirty(self):
        return self._dirty

    def _set_dirty(self, flag):
        if flag != self._dirty:
            self._dirty = flag
            self.dirtyChanged.emit(flag)
