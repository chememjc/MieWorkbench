"""Shared scripted-worker helpers for the optical-train test files
(not collected by pytest — imported by test_train_model.py /
test_fold.py the same way vtk_test_support.py is)."""

import copy
import json
import tempfile

from mieworkbench.core.geomcache import GeomCache
from mieworkbench.core.project import Project
from mieworkbench.core.train import TRAIN_GROUP
from mieworkbench.core.transforms import BodyState

import primitivelib  # noqa: E402  (scripts/ already on sys.path via the
                      # core.train import above; metadata only, no FreeCAD)
import train_solver   # noqa: E402

MIRROR_PORTS = {"entry": [0.0, 0.0, 0.0], "exit": [0.0, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
                "reflect_plane": {"point": [0.0, 0.0, 0.0],
                                  "normal": [-1.0, 0.0, 0.0]}}


def _dim_numeric(raw):
    """Best-effort float from a dim_<label> sheet's raw cell ("=3 mm" /
    "3" / an unparseable FreeCAD cross-sheet expression like
    "=<<miewb_vars>>.x * 1mm"). Unlike _recompute_variables (miewb_vars,
    where aliases legitimately reference each other and a resolve failure
    should be visible sheet-wide), dim-sheet params in every demo this
    supports are either plain numbers or a single miewb_vars reference,
    never inter-dependent -- so this resolves ONE alias at a time and
    quietly leaves unparseable ones as None. primitivelib.port_frames
    then falls back to that one alias's catalog default, which never
    affects a demo's chain-solved positions (the only param that hits
    this path -- camera_triplet's variable-driven iris hole_diameter --
    doesn't move the iris's port vertices at all)."""
    text = raw[1:] if isinstance(raw, str) and raw.startswith("=") else raw
    text = str(text).strip()
    if not text:
        return None
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        pass
    try:
        return train_solver.eval_expr(text, {})
    except train_solver.TrainError:
        return None


class TrainFakeWorker:
    """FcClient stand-in: group-aware set_property, group-rigid
    set_placement (fcops semantics: every miewb_group member gets the
    SAME placement), and a minimal import_primitive that synthesizes a
    mirror-like body carrying miewb_train_ports."""

    def __init__(self, structure):
        self.structure = structure
        self.ops = []

    def request(self, op, params=None, timeout=None):
        params = params or {}
        self.ops.append((op, copy.deepcopy(params)))
        if op == "get_structure":
            return copy.deepcopy(self.structure)
        if op == "set_property":
            body = self._body(params["body"])
            value = params["value"]
            ptype = params.get("ptype") or (
                "bool" if isinstance(value, bool)
                else "float" if isinstance(value, (int, float))
                else "string")
            fc_type = {"string": "App::PropertyString",
                       "float": "App::PropertyFloat",
                       "bool": "App::PropertyBool"}[ptype]
            body["properties"][params["name"]] = {
                "type": fc_type, "group": params.get("group", "Base"),
                "value": value}
            return self._mut()
        if op == "remove_property":
            self._body(params["body"])["properties"].pop(
                params["name"], None)
            return self._mut()
        if op == "set_placement":
            key = str(params["body"])
            members = [b for b in self.structure["bodies"]
                       if b["properties"].get("miewb_group", {})
                       .get("value") == key]
            if not members:
                members = [self._body(key)]
            pl = {"pos_mm": list(params["pos_mm"]),
                  "quat": list(params["quat"])}
            out = {}
            for b in members:
                b["placement"] = copy.deepcopy(pl)
                out[b["name"]] = copy.deepcopy(pl)
            return {"placements": out}
        if op == "import_primitive":
            label = str(params["label"])
            kind = params["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            spec = primitivelib.PRIMITIVES.get(kind)
            ports = MIRROR_PORTS
            if spec is not None:
                try:
                    ports = primitivelib.port_frames(kind, {})
                except KeyError:
                    pass          # kinds without a port formula: mirror ports
            body = body_dict(label, json.dumps(ports))
            body["properties"]["miewb_primitive"] = {
                "type": "App::PropertyString", "group": "Base",
                "value": kind}
            if spec is not None:
                # seed dim_<label> with the catalog defaults, same as the
                # real primitive's shipped 'dim' sheet -- TrainModel.
                # local_ports prefers this live sheet (via port_frames)
                # over the cached miewb_train_ports JSON above, so
                # subsequent set_spreadsheet edits (ct, R_front, ...)
                # actually move the element's ports.
                aliases = {}
                for alias, pspec in spec["params"].items():
                    raw = primitivelib.sheet_raw(pspec["default"],
                                                 pspec["unit"])
                    aliases[alias] = {"cell": alias, "raw": raw,
                                      "value": float(pspec["default"]),
                                      "unit": pspec["unit"]}
                dim_label = "dim_%s" % label
                self.structure.setdefault("sheets", []).append(
                    {"name": dim_label, "label": dim_label,
                     "aliases": aliases})
            self.structure["bodies"].append(body)
            return {"bodies": [copy.deepcopy(body)]}
        if op == "delete_element":
            element = str(params["element"])
            deleted = [b["name"] for b in self.structure["bodies"]
                       if (b["properties"].get("miewb_group", {})
                           .get("value") or b["label"]) == element]
            self.structure["bodies"] = [
                b for b in self.structure["bodies"]
                if b["name"] not in deleted]
            return {"deleted": deleted}
        if op == "tessellate":
            return {"bodies": {b: {"faces": [], "shape_key": "k1",
                                   "placement": self._body(b)["placement"]}
                               for b in (params.get("bodies") or [])}}
        if op == "create_sheet":
            label = str(params["label"])
            sheet = self._var_sheet(label)
            if sheet is None:
                sheet = {"name": label, "label": label, "aliases": {}}
                self.structure.setdefault("sheets", []).append(sheet)
            out = self._mut()
            out["sheet"] = copy.deepcopy(sheet)
            return out
        if op == "set_cell":
            sheet = self._var_sheet(str(params["sheet"]))
            if sheet is None:
                raise AssertionError(
                    "no spreadsheet %r (call create_sheet first)"
                    % params["sheet"])
            cell = str(params["cell"])
            raw = params.get("raw", "")
            alias = params.get("alias")
            aliases = sheet.setdefault("aliases", {})
            # setAlias semantics: one alias per cell -- drop whatever was
            # aliased to this cell before (re)assigning or clearing it
            for stale in [a for a, e in aliases.items()
                          if e.get("cell") == cell]:
                aliases.pop(stale, None)
            if raw != "" and alias:
                aliases[str(alias)] = {"cell": cell, "raw": str(raw),
                                       "value": None, "unit": ""}
            self._recompute_variables(sheet)
            return self._mut()
        if op == "set_spreadsheet":
            sheet_label = str(params["sheet"])
            alias = str(params["alias"])
            raw = params["raw"]
            sheet = self._var_sheet(sheet_label)
            if sheet is None:
                sheet = {"name": sheet_label, "label": sheet_label,
                         "aliases": {}}
                self.structure.setdefault("sheets", []).append(sheet)
            sheet.setdefault("aliases", {})[alias] = {
                "cell": alias, "raw": raw, "value": _dim_numeric(raw),
                "unit": ""}
            return self._mut()
        if op == "rebuild_primitive":
            # the real worker re-derives geometry (and can renumber faces);
            # for this scripted double the sheet values are already live
            # (set_spreadsheet updates them directly, and TrainModel reads
            # dim_<label> fresh every time), so this just reports back the
            # element's current bodies for _refresh_geometry/tessellate.
            group = str(params["group"])
            bodies = [b for b in self.structure["bodies"]
                     if (b["properties"].get("miewb_group", {})
                         .get("value") or b["label"]) == group]
            return {"bodies": [copy.deepcopy(b) for b in bodies]}
        raise AssertionError("unexpected op %r" % op)

    def _body(self, key):
        for b in self.structure["bodies"]:
            if b["name"] == key or b["label"] == key:
                return b
        raise KeyError(key)

    def _var_sheet(self, label):
        for s in self.structure.get("sheets", []):
            if s.get("label") == label or s.get("name") == label:
                return s
        return None

    def _recompute_variables(self, sheet):
        """Naive re-evaluation of every aliased cell in `sheet`, close
        enough to FreeCAD's live recompute for tests: resolves each raw
        "=<number>" or "=<expr over other aliases>" via
        train_solver.resolve_variables, so an edited cell's echoed
        `value` stays consistent with what the real worker would report
        (meta cells __min/__max/__n/__on are always plain numbers, so
        they round-trip through the same evaluator trivially). Leaves
        values untouched on a cycle/unknown-name failure -- same as a
        real spreadsheet showing a stale/error cell until it's fixed."""
        aliases = sheet.get("aliases") or {}
        raw_map = {}
        for alias, entry in aliases.items():
            r = entry.get("raw") or ""
            raw_map[alias] = r[1:] if r.startswith("=") else r
        try:
            values = train_solver.resolve_variables(raw_map)
        except train_solver.TrainError:
            return
        for alias, value in values.items():
            aliases[alias]["value"] = value

    def _mut(self):
        return {"changed_bodies": [], "moved_bodies": [], "invalid": [],
                "placements": {}}


def ports_json(entry, exit_, axis=(1, 0, 0), up=(0, 0, 1), reflect=None):
    return json.dumps({"entry": list(entry), "exit": list(exit_),
                       "axis": list(axis), "up": list(up),
                       "reflect_plane": reflect})


def body_dict(name, ports, pos=(0, 0, 0), extra_props=None):
    props = {
        "miewb_group": {"type": "App::PropertyString", "group": "Base",
                        "value": name},
        "miewb_train_ports": {"type": "App::PropertyString",
                              "group": TRAIN_GROUP, "value": ports},
    }
    props.update(extra_props or {})
    return {
        "name": name, "label": name, "tip": "%s_pad" % name,
        "face_count": 1, "solid_closed": True, "volume_mm3": 1.0,
        "center_of_mass_mm": list(pos), "bbox_mm": [0, 0, 0, 1, 1, 1],
        "placement": {"pos_mm": list(pos), "quat": [0.0, 0.0, 0.0, 1.0]},
        "placement_bound": False, "shape_key": "k_%s" % name,
        "properties": props,
    }


def make_scene(extra_sheets=None):
    """SRC (anchored source) -> L1 -> L2 (lenses) + FM (fold mirror) +
    DET, all unchained initially, plus a miewb_vars sheet with gap=25."""
    structure = {
        "doc": "scene", "label": "scene", "file": "/nowhere/scene.FCStd",
        "bodies": [
            body_dict("SRC", ports_json([5, 0, 0], [5, 0, 0])),
            body_dict("L1", ports_json([-2, 0, 0], [2, 0, 0]),
                      pos=(30, 0, 0)),
            body_dict("L2", ports_json([-1, 0, 0], [1, 0, 0]),
                      pos=(60, 0, 0)),
            body_dict("FM", ports_json([0, 0, 0], [0, 0, 0], reflect={
                "point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}),
                pos=(90, 0, 0)),
            body_dict("DET", ports_json([0, 0, 0], [0, 0, 0]),
                      pos=(120, 0, 0)),
        ],
        "sheets": [{"name": "Spreadsheet", "label": "miewb_vars",
                    "aliases": {
                        "gap": {"cell": "B1", "raw": "=25", "value": 25.0,
                                "unit": ""},
                        "gap__min": {"cell": "C1", "raw": "=10",
                                     "value": 10.0, "unit": ""},
                        "gap__max": {"cell": "D1", "raw": "=40",
                                     "value": 40.0, "unit": ""},
                        "gap__n": {"cell": "E1", "raw": "=3", "value": 3.0,
                                   "unit": ""},
                        "gap__on": {"cell": "F1", "raw": "=1", "value": 1.0,
                                    "unit": ""},
                    }}] + list(extra_sheets or []),
    }
    project = Project()
    fake = TrainFakeWorker(structure)
    project._fc = fake
    project._cache = GeomCache(fake, cache_root=tempfile.mkdtemp(
        prefix="miewb_train_test_"))
    project.doc = "scene"
    project.fcstd_path = "/nowhere/scene.FCStd"
    project.structure = fake.request("get_structure", {"doc": "scene"})
    for b in project.structure["bodies"]:
        project.body_states[b["name"]] = BodyState.from_worker(b, [])
    fake.ops.clear()
    return project, fake


def pos_of(project, name):
    return project.body_states[name].current.to_dict()["pos_mm"]
