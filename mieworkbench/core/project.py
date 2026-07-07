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
from .transforms import BodyState, Placement, ReferenceResolver


class ProjectError(RuntimeError):
    pass


class Project(QObject):
    sceneLoaded = Signal()                 # full (re)load: rebuild all views
    bodiesReshaped = Signal(list)          # [body_name]: re-mesh those actors
    bodiesMoved = Signal(dict)             # {body_name: placement_dict}
    propertiesChanged = Signal(str)        # body_name (or "" for sheet edits)
    dirtyChanged = Signal(bool)

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
        self._set_dirty(False)
        self.sceneLoaded.emit()

    def new_document(self, path):
        if self.doc is not None:
            self.close()
        self.structure = self.fc.request("new_document", {"path": path})
        self.doc = self.structure["doc"]
        self.fcstd_path = os.path.abspath(path)
        self.faces = {}
        self.body_states = {}
        self._set_dirty(False)
        self.sceneLoaded.emit()

    def close(self):
        if self.doc is not None:
            try:
                self.fc.close(self.doc)
            except Exception:
                pass
        self.doc = None
        self.fcstd_path = None
        self.structure = None
        self.faces = {}
        self.body_states = {}
        self._set_dirty(False)

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
        self._set_dirty(True)
        return result

    def set_property(self, body, name, value, ptype=None):
        params = {"doc": self.doc, "body": body, "name": name,
                  "value": value}
        if ptype:
            params["ptype"] = ptype
        return self._route_mutation(
            self.fc.request("set_property", params), body)

    def remove_property(self, body, name):
        return self._route_mutation(
            self.fc.request("remove_property",
                            {"doc": self.doc, "body": body, "name": name}),
            body)

    def set_spreadsheet(self, sheet, alias, raw):
        result = self.fc.request("set_spreadsheet",
                                 {"doc": self.doc, "sheet": sheet,
                                  "alias": alias, "raw": raw})
        return self._route_mutation(result)

    def rebuild_primitive(self, group):
        result = self.fc.request("rebuild_primitive",
                                 {"doc": self.doc, "group": group})
        # a rebuild replaces bodies wholesale: refresh everything for the
        # group members reported back
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.bodiesReshaped.emit(names)
        self.propertiesChanged.emit(names[0] if names else "")
        self._set_dirty(True)
        return result

    def import_primitive(self, path, label):
        result = self.fc.request("import_primitive",
                                 {"doc": self.doc, "path": path,
                                  "label": label})
        names = [b["name"] for b in result.get("bodies", [])]
        self._refetch_structure()
        self._refresh_geometry(bodies=names or None)
        self.sceneLoaded.emit()      # new bodies + sheets: full view rebuild
        self._set_dirty(True)
        return result

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
        operation.apply(self.resolver(), state)
        placement = state.current.to_dict()
        self.fc.request("set_placement",
                        {"doc": self.doc, "body": body_name,
                         "pos_mm": placement["pos_mm"],
                         "quat": placement["quat"]})
        self.bodiesMoved.emit({body_name: placement})
        self._set_dirty(True)
        return placement

    # -- persistence ---------------------------------------------------------------
    def save(self):
        self.fc.request("save", {"doc": self.doc})
        self._set_dirty(False)

    def save_as(self, path):
        self.fc.request("save_as", {"doc": self.doc, "path": path})
        self.fcstd_path = os.path.abspath(path)
        self._set_dirty(False)

    def is_dirty(self):
        return self._dirty

    def _set_dirty(self, flag):
        if flag != self._dirty:
            self._dirty = flag
            self.dirtyChanged.emit(flag)
