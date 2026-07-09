"""FreeCAD-side optical-train application: read chain records/variables
from an open FreeCAD.Document, bake miewb_expr_* property expressions,
and write the solved chained placements.

Runs ONLY under the FreeCAD AppImage interpreter (imported by
permute_model.py per variant). All chain math is train_solver.py — the
same pure-stdlib module the GUI uses — so a variant re-bake reproduces
the GUI's placements bit-for-bit from identical inputs.

Idempotent by construction (the AppImage -c runner executes calling
scripts twice): applying the train to an already-solved document writes
the same placements again.
"""

import json

import FreeCAD

import primitivelib
import train_solver

VARIABLES_SHEET = "miewb_vars"
_VAR_META_SUFFIXES = ("__min", "__max", "__n", "__on")

TRAIN_PROPS = {
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
    "miewb_train_fold": "fold",
    "miewb_train_folded": "folded",
    "miewb_train_fold_deviation": "fold_deviation",
    "miewb_train_fold_azimuth": "fold_azimuth",
}
BOOL_FIELDS = ("fold", "folded")
PORTS_PROP = "miewb_train_ports"
EXPR_PREFIX = "miewb_expr_"


class TrainApplyError(RuntimeError):
    pass


def _bodies(doc):
    return [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]


def _sheet_by_label(doc, label):
    for o in doc.Objects:
        if o.TypeId == "Spreadsheet::Sheet" and o.Label == label:
            return o
    return None


def _cell_float(sheet, alias):
    v = sheet.get(alias)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(FreeCAD.Units.Quantity(v).Value)


def read_variables(doc):
    """{name: float} from the miewb_vars sheet (FreeCAD-evaluated cell
    values; sweep-meta __min/__max/__n/__on columns excluded)."""
    sheet = _sheet_by_label(doc, VARIABLES_SHEET)
    if sheet is None:
        return {}
    out = {}
    for cell in sheet.getUsedCells():
        try:
            alias = sheet.getAlias(cell)
        except Exception:
            alias = None
        if not alias or any(alias.endswith(s) for s in _VAR_META_SUFFIXES):
            continue
        try:
            out[alias] = _cell_float(sheet, alias)
        except Exception:
            continue
    return out


def _elements(doc):
    """{element_label: {"primary": body, "bodies": [body]}} - same
    grouping/primary convention as mieworkbench.core.train.TrainModel."""
    groups = {}
    for b in _bodies(doc):
        element = getattr(b, "miewb_group", None) or b.Label
        groups.setdefault(element, []).append(b)
    out = {}
    for element, bodies in groups.items():
        primary = next((b for b in bodies if b.Label == element), bodies[0])
        out[element] = {"primary": primary, "bodies": bodies}
    return out


def _local_ports(doc, element, primary):
    kind = getattr(primary, "miewb_primitive", None)
    if kind and kind in primitivelib.PRIMITIVES:
        sheet = _sheet_by_label(doc, "dim_%s" % element)
        if sheet is not None:
            try:
                params = primitivelib.read_params(sheet, kind)
                return primitivelib.port_frames(kind, params)
            except KeyError:
                pass
    cached = getattr(primary, PORTS_PROP, None)
    if cached:
        try:
            return json.loads(cached)
        except (ValueError, TypeError):
            pass
    return None            # solver default (thin element at local origin)


def _placement_dict(body):
    p = body.Placement
    return {"pos_mm": [p.Base.x, p.Base.y, p.Base.z],
            "quat": list(p.Rotation.Q)}


def _placement_expression(body):
    for path, expr in (body.ExpressionEngine or []):
        if str(path).startswith("Placement"):
            return "%s = %s" % (path, expr)
    return None


def read_records(doc):
    """(records, anchors) for train_solver.solve_chain."""
    records = {}
    anchors = {}
    for element, info in _elements(doc).items():
        primary = info["primary"]
        rec = {"label": element, "mode": "anchored"}
        for prop, field in TRAIN_PROPS.items():
            val = getattr(primary, prop, None)
            if val in (None, ""):
                continue
            rec[field] = bool(val) if field in BOOL_FIELDS else str(val)
        loc = _local_ports(doc, element, primary)
        if loc is not None:
            rec["local"] = loc
        records[element] = rec
        anchors[element] = _placement_dict(primary)
    return records, anchors


def bake_expressions(doc, variables):
    """Evaluate every miewb_expr_<prop> property against the variables
    and bake the float into <prop>. Returns the touched body names."""
    touched = []
    for body in _bodies(doc):
        for pname in list(body.PropertiesList):
            if not pname.startswith(EXPR_PREFIX):
                continue
            target = pname[len(EXPR_PREFIX):]
            expr = getattr(body, pname, None)
            if expr in (None, ""):
                continue
            if not hasattr(body, target):
                raise TrainApplyError(
                    "%s: %s drives missing property %r"
                    % (body.Label, pname, target))
            value = train_solver.eval_expr(expr, variables)
            setattr(body, target, float(value))
            touched.append(body.Name)
    return touched


def apply_train(doc, log=None):
    """Re-solve the optical train and write every chained element's
    placement (all group members get the shared placement — the fcops
    convention). Also bakes miewb_expr_* property expressions. Returns
    the number of elements whose placement was written."""
    say = log or (lambda msg: None)
    variables = read_variables(doc)
    bake_expressions(doc, variables)
    records, anchors = read_records(doc)
    if not any(r.get("mode") == "chained" for r in records.values()):
        return 0
    try:
        solved = train_solver.solve_chain(records, anchors, variables)
    except train_solver.TrainError as e:
        raise TrainApplyError("optical train solve failed: %s" % e)
    elements = _elements(doc)
    n = 0
    for element, pl in solved["placements"].items():
        for body in elements[element]["bodies"]:
            bound = _placement_expression(body)
            if bound is not None:
                raise TrainApplyError(
                    "chained element %s: body %s placement is "
                    "expression-bound (%s)" % (element, body.Name, bound))
            body.Placement = FreeCAD.Placement(
                FreeCAD.Vector(*[float(v) for v in pl["pos_mm"]]),
                FreeCAD.Rotation(*[float(v) for v in pl["quat"]]))
        say("train: placed %s at %s" % (element, pl["pos_mm"]))
        n += 1
    doc.recompute()
    return n
