"""TrainModel - the GUI-side view over the optical-train chain.

Qt-free (importable by make_demos and tests without a display). All
actual chain math lives in scripts/train_solver.py - the SAME module the
headless variant permuter uses - so GUI-baked and permute-rebaked
placements can never drift. This module's job is translation:

    Project.structure + BodyStates  ->  solver element records
    solver placements               <-  Project placement flushes

Storage contract (docs in scripts/train_solver.py):
  * per-element chain metadata = dynamic properties on the element's
    PRIMARY body (the one whose Label equals the element label), FreeCAD
    property group "MieTrain", names all miewb_train_* (the miewb_
    prefix keeps them invisible to the extractor and out of the
    opticsChanged ray-preview guard);
  * element-local port geometry comes from primitivelib.port_frames
    (exact, parameter-derived, rebuild-stable), overridable/cacheable via
    the miewb_train_ports JSON property for hand-authored bodies, with a
    thin-element optical-center fallback as the last resort.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train_solver  # noqa: E402
import primitivelib  # noqa: E402

TRAIN_GROUP = "MieTrain"

# property name <-> record field (all strings unless noted)
PROP_FIELDS = {
    "miewb_train_mode": "mode",
    "miewb_train_ref": "ref",
    "miewb_train_port": "port",
    "miewb_train_distance": "distance",
    "miewb_train_decenter_x": "decenter_x",
    "miewb_train_decenter_y": "decenter_y",
    "miewb_train_tilt_rx": "tilt_rx",
    "miewb_train_tilt_ry": "tilt_ry",
    "miewb_train_tilt_rz": "tilt_rz",
    "miewb_train_rot_order": "rot_order",
    "miewb_train_pos_rot_order": "pos_rot_order",
    "miewb_train_pivot": "pivot",
    "miewb_train_flip": "flip",              # bool
    "miewb_train_fold": "fold",              # bool
    "miewb_train_folded": "folded",          # bool
    "miewb_train_fold_deviation": "fold_deviation",
    "miewb_train_fold_azimuth": "fold_azimuth",
}
FIELD_PROPS = {v: k for k, v in PROP_FIELDS.items()}
BOOL_FIELDS = ("fold", "folded", "flip")

# edge fields the user edits / derive_edge returns
EDGE_FIELDS = ("distance", "decenter_x", "decenter_y",
               "tilt_rx", "tilt_ry", "tilt_rz")

# anchored-pose expression fields (train_solver.POSE_EXPR_FIELDS) <-> the
# miewb_expr_<field> primary-body property that stores each expression.
POSE_EXPR_FIELDS = train_solver.POSE_EXPR_FIELDS
EXPR_PREFIX = "miewb_expr_"
POSE_PROP_FIELDS = {EXPR_PREFIX + f: f for f in POSE_EXPR_FIELDS}
POSE_FIELD_PROPS = {v: k for k, v in POSE_PROP_FIELDS.items()}

PORTS_PROP = "miewb_train_ports"
EXCLUDE_PROP = "miewb_exclude"
ASSEMBLY_PROP = "miewb_assembly"

VARIABLES_SHEET = "miewb_vars"
_VAR_META_SUFFIXES = ("__min", "__max", "__n", "__on")

MOVE_TOL_MM = 1e-9
QUAT_TOL = 1e-12


def is_variable_meta(alias):
    return any(alias.endswith(s) for s in _VAR_META_SUFFIXES)


def variables_from_sheets(sheets):
    """{name: float} from the miewb_vars sheet echo (worker structure
    sheets list). FreeCAD already evaluated cell expressions, so the
    echoed `value` is authoritative; meta columns (__min etc.) are
    excluded."""
    for sheet in sheets or []:
        if sheet.get("label") == VARIABLES_SHEET \
                or sheet.get("name") == VARIABLES_SHEET:
            out = {}
            for alias, cell in (sheet.get("aliases") or {}).items():
                if is_variable_meta(alias):
                    continue
                try:
                    out[alias] = float(cell.get("value"))
                except (TypeError, ValueError):
                    continue
            return out
    return {}


def _prop_value(body_dict, name, default=None):
    entry = (body_dict.get("properties") or {}).get(name)
    return default if entry is None else entry.get("value", default)


class TrainModel:
    """A snapshot view: build one from the live Project state, ask it
    questions, throw it away. Nothing here mutates anything."""

    def __init__(self, structure, body_states):
        self.structure = structure or {"bodies": [], "sheets": []}
        self.body_states = body_states or {}
        self._elements = self._collect_elements()
        self._records = None

    # -- element enumeration -------------------------------------------------
    def _collect_elements(self):
        """{element_label: {"primary": body_dict, "bodies": [body_dict]}}
        An element = the bodies sharing a miewb_group value (or a lone
        body under its own label). The primary body is the one whose
        Label equals the element label (import_primitive's convention),
        falling back to the first member."""
        groups = {}
        for b in self.structure.get("bodies", []):
            element = _prop_value(b, "miewb_group") or b["label"]
            groups.setdefault(element, []).append(b)
        out = {}
        for element, bodies in groups.items():
            primary = next((b for b in bodies if b["label"] == element),
                           bodies[0])
            out[element] = {"primary": primary, "bodies": bodies}
        return out

    def elements(self):
        return dict(self._elements)

    def element_labels(self):
        return sorted(self._elements)

    def primary_body(self, element):
        try:
            return self._elements[element]["primary"]
        except KeyError:
            raise train_solver.TrainError("unknown element %r" % element)

    def primary_body_name(self, element):
        return self.primary_body(element)["name"]

    # -- local port geometry ---------------------------------------------------
    def local_ports(self, element):
        """Element-local port dict for the solver ("local" record field).
        Resolution order: exact primitive formulas -> the miewb_train_ports
        JSON property -> thin-element fallback at the optical center."""
        primary = self.primary_body(element)
        kind = _prop_value(primary, "miewb_primitive")
        if kind:
            params = self._sheet_params(element)
            if params is not None:
                try:
                    return primitivelib.port_frames(kind, params)
                except KeyError:
                    pass
        cached = _prop_value(primary, PORTS_PROP)
        if cached:
            try:
                return json.loads(cached)
            except (ValueError, TypeError):
                pass
        return self._fallback_ports(primary)

    def _sheet_params(self, element):
        label = "dim_%s" % element
        for sheet in self.structure.get("sheets", []):
            if sheet.get("label") == label:
                out = {}
                for alias, cell in (sheet.get("aliases") or {}).items():
                    try:
                        out[alias] = float(cell.get("value"))
                    except (TypeError, ValueError):
                        pass
                return out
        return None

    def _fallback_ports(self, primary):
        """Thin-element fallback for hand-authored bodies: entry = exit =
        the optical center (closest point on the largest-face normal line
        to the bbox center), axis = that face's normal - all in BODY-LOCAL
        coordinates. Distances to/from such an element are effectively
        measured center-to-center; exact vertex ports need port_frames or
        a miewb_train_ports property."""
        state = self.body_states.get(primary["name"])
        if state is None:
            return dict(train_solver._DEF_LOCAL)
        axis = None
        p0 = None
        f = state._axis_face()
        c = [float(v) for v in state.bbox_center_local_mm]
        if f is not None and f.normal_local is not None:
            n = [float(v) for v in f.normal_local]
            norm = math.sqrt(sum(v * v for v in n))
            if norm > 1e-12:
                axis = [v / norm for v in n]
                p0 = [float(v) for v in f.centroid_local_mm]
        if axis is None:
            return {"entry": c, "exit": c, "axis": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0], "reflect_plane": None}
        t = sum((c[i] - p0[i]) * axis[i] for i in range(3))
        oc = [p0[i] + t * axis[i] for i in range(3)]
        return {"entry": oc, "exit": oc, "axis": axis,
                "up": train_solver.stable_up(axis), "reflect_plane": None}

    # -- solver records -----------------------------------------------------------
    def record(self, element):
        primary = self.primary_body(element)
        rec = {"label": element, "mode": "anchored"}
        for prop, field in PROP_FIELDS.items():
            val = _prop_value(primary, prop)
            if val in (None, ""):
                continue
            rec[field] = bool(val) if field in BOOL_FIELDS else val
        pose = {}
        for prop, field in POSE_PROP_FIELDS.items():
            val = _prop_value(primary, prop)
            if val in (None, ""):
                continue
            pose[field] = val
        if pose:
            rec["pose_expr"] = pose
        rec["local"] = self.local_ports(element)
        return rec

    def pose_expressions(self, element):
        """{field: expr} anchored-pose expressions currently on `element`
        (empty when none). Fields are a subset of POSE_EXPR_FIELDS."""
        return dict((self.records().get(element) or {}).get("pose_expr")
                    or {})

    def has_pose_expr(self, element):
        rec = self.records().get(element)
        return bool(rec) and train_solver.has_pose_expr(rec)

    def records(self):
        if self._records is None:
            self._records = {el: self.record(el) for el in self._elements}
        return self._records

    def anchors(self):
        """Current placements of every element's primary body (the solver
        only reads the anchored ones, but supplying all lets callers diff
        solved vs current)."""
        out = {}
        for element, info in self._elements.items():
            state = self.body_states.get(info["primary"]["name"])
            if state is not None:
                out[element] = state.current.to_dict()
        return out

    def is_chained(self, element):
        return self.records()[element].get("mode") == "chained"

    def chained_elements(self):
        return [el for el in self.element_labels() if self.is_chained(el)]

    def has_train(self):
        return bool(self.chained_elements())

    def downstream_of(self, element):
        return train_solver.downstream_of(self.records(), element)

    def folds(self):
        return [el for el, rec in self.records().items()
                if rec.get("fold")]

    # -- solving -----------------------------------------------------------------
    def solve(self, variables=None):
        """Run the shared solver. Returns its dict ({"placements",
        "frames", "order"}); placements cover chained elements only."""
        return train_solver.solve_chain(self.records(), self.anchors(),
                                        variables or {})

    def solve_moves(self, variables=None):
        """{element: placement} for chained elements whose SOLVED pose
        differs from the current one (the flush list for a ripple)."""
        solved = self.solve(variables)["placements"]
        current = self.anchors()
        moves = {}
        for element, pl in solved.items():
            cur = current.get(element)
            if cur is None or _placements_differ(cur, pl):
                moves[element] = pl
        return moves

    def parent_frame(self, element, variables=None):
        """The exit port frame this chained element hangs from."""
        rec = self.records()[element]
        if rec.get("mode") != "chained":
            raise train_solver.TrainError("%r is anchored" % element)
        frames = self.solve(variables)["frames"]
        ref = rec["ref"]
        port = rec.get("port") or train_solver._default_port(
            self.records()[ref])
        return frames[ref][port]

    def derive_edge(self, element, variables=None):
        """Distance/decenter/tilt floats recovered from the element's
        CURRENT placement (after a spatial drag)."""
        rec = self.records()[element]
        frame = self.parent_frame(element, variables)
        state = self.body_states.get(self.primary_body_name(element))
        return train_solver.derive_edge(frame, state.current.to_dict(), rec)

    def available_ports(self, element):
        """Exit-port names an element offers (record-level approximation
        of train_solver.exit_frames): pass-through always; reflect when a
        reflect plane exists; deviate when an explicit deviation is set
        (or the element is a plane-less fold)."""
        rec = self.records()[element]
        loc = rec.get("local") or {}
        ports = ["out", "transmit"]
        if loc.get("reflect_plane"):
            ports.append("reflect")
        if rec.get("fold_deviation") not in (None, "") or (
                rec.get("fold") and not loc.get("reflect_plane")):
            ports.append("deviate")
        return ports

    def candidate_edge(self, element, ref, port=None, variables=None):
        """What `element`'s chain edge WOULD be against an arbitrary
        (ref, port), derived from its CURRENT placement — the no-move
        conversion preview (anchored -> chained). Returns the same float
        dict as derive_edge. Raises TrainError for unknown/cyclic
        references or unavailable ports."""
        element, ref = str(element), str(ref)
        recs = self.records()
        if element not in recs:
            raise train_solver.TrainError("unknown element %r" % element)
        if ref not in recs:
            raise train_solver.TrainError("unknown reference %r" % ref)
        if ref == element or ref in self.downstream_of(element):
            raise train_solver.TrainError(
                "chaining %s to %s would create a cycle" % (element, ref))
        port = port or train_solver._default_port(recs[ref])
        frames = self.solve(variables)["frames"]
        try:
            frame = frames[ref][port]
        except KeyError:
            raise train_solver.TrainError(
                "%s has no port %r (available: %s)"
                % (ref, port, ", ".join(sorted(frames.get(ref, {})))))
        state = self.body_states.get(self.primary_body_name(element))
        if state is None:
            raise train_solver.TrainError(
                "no live placement for %r" % element)
        return train_solver.derive_edge(frame, state.current.to_dict(),
                                        recs[element])

    # -- validation ---------------------------------------------------------------
    def validate(self, variables_raw=None):
        """Chain-level problems as [(severity, message)]. Checks: cycles
        (chain + variables), dangling refs, chained expression-bound
        placements, unknown ports, unresolvable expressions."""
        problems = []
        records = self.records()
        try:
            train_solver.sort_chain(records)
        except train_solver.TrainError as e:
            problems.append(("error", str(e)))
            return problems
        variables = {}
        if variables_raw is not None:
            try:
                variables = train_solver.resolve_variables(variables_raw)
            except train_solver.TrainError as e:
                problems.append(("error", str(e)))
        else:
            variables = variables_from_sheets(self.structure.get("sheets"))
        for element in self.chained_elements():
            primary = self.primary_body(element)
            if primary.get("placement_bound"):
                problems.append((
                    "error",
                    "%s is chained but its placement is expression-bound; "
                    "unchain it or remove the placement expression"
                    % element))
            if self.has_pose_expr(element):
                problems.append((
                    "error",
                    "%s is chained but carries anchored-pose expression(s); "
                    "pose expressions are valid on anchored elements only "
                    "(unchain it, or clear the pose expressions)"
                    % element))
        try:
            self.solve(variables)
        except train_solver.TrainError as e:
            problems.append(("error", str(e)))
        return problems


def _placements_differ(a, b):
    for i in range(3):
        if abs(a["pos_mm"][i] - b["pos_mm"][i]) > MOVE_TOL_MM:
            return True
    # quaternion double cover: q and -q are the same rotation
    same = all(abs(a["quat"][i] - b["quat"][i]) <= QUAT_TOL for i in range(4))
    flip = all(abs(a["quat"][i] + b["quat"][i]) <= QUAT_TOL for i in range(4))
    return not (same or flip)


def edge_props(edge):
    """Edge dict -> {property_name: string_value} ready for set_property
    (floats are stringified; expressions pass through verbatim)."""
    out = {}
    for field, value in edge.items():
        prop = FIELD_PROPS.get(field)
        if prop is None:
            raise train_solver.TrainError("unknown chain field %r" % field)
        if field in BOOL_FIELDS:
            out[prop] = bool(value)
        elif isinstance(value, float):
            out[prop] = "%.17g" % value
        else:
            out[prop] = str(value)
    return out
