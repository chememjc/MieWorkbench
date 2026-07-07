#!/usr/bin/env python3
# fcops.py - shared FreeCAD-side operation implementations for MieWorkbench.
#
# Runs ONLY under the FreeCAD AppImage's embedded Python (imported by
# fc_server.py / fc_batch.py). stdlib + FreeCAD modules only - no numpy.
#
# Every op is a function op_<name>(params_dict) -> JSON-serializable dict.
# Ops never print and never sys.exit; they raise OpError (or any exception)
# and the caller turns that into an error response.
#
# Conventions mirrored from extract_geometry.py (keep in sync):
#   - bodies are doc.Objects with TypeId == "PartDesign::Body"
#   - face ids are "<Body.Name>.<Tip.Name>.Face<idx>" (idx 1-based over
#     Shape.Faces)
#   - STLs are binary, in METRES (copy scaled by 0.001 before meshing),
#     LinearDeflection 0.03 mm, AngularDeflection 15 deg
#   - custom body tags live as dynamic App::Property* in group "Base"
#   - spreadsheet cells keep their raw "=<value> <unit>" form
#
# Difference from extract_geometry: tessellation here is BODY-LOCAL (the
# body Placement is stripped from the shape copy first) so the GUI can apply
# placements as cheap vtkTransforms without re-tessellating on every move.

import math
import os
import struct
import sys
import traceback

# parent scripts/ dir for primitivelib (rebuild op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: F401  (embedded)

MESH_LINEAR_DEFLECTION_MM = 0.03
MESH_ANGULAR_DEFLECTION_DEG = 15.0

FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)


class OpError(ValueError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _doc(name):
    doc = FreeCAD.getDocument(str(name))
    if doc is None:
        raise OpError("no open document named %r" % name)
    return doc


def _body(doc, name):
    obj = doc.getObject(str(name))
    if obj is None:
        # allow lookup by Label as a convenience
        matches = [o for o in doc.Objects
                   if o.TypeId == "PartDesign::Body" and o.Label == str(name)]
        if len(matches) == 1:
            return matches[0]
        raise OpError("no body %r in document %r" % (name, doc.Name))
    if obj.TypeId != "PartDesign::Body":
        raise OpError("object %r is %s, not a PartDesign::Body"
                      % (name, obj.TypeId))
    return obj


def _sheet(doc, name):
    obj = doc.getObject(str(name))
    if obj is not None and obj.TypeId == "Spreadsheet::Sheet":
        return obj
    matches = [o for o in doc.Objects
               if o.TypeId == "Spreadsheet::Sheet" and o.Label == str(name)]
    if len(matches) == 1:
        return matches[0]
    raise OpError("no spreadsheet %r in document %r" % (name, doc.Name))


def _vec3(v):
    return [float(v.x), float(v.y), float(v.z)]


_BASELINE_BODY_PROPS = None


def _baseline_body_props():
    """PropertiesList of a factory-fresh PartDesign::Body, cached.

    Anything a real body carries beyond this set is a user/tagging property
    (material, power, coating, ...). Robust against FreeCAD adding built-ins
    with group "Base" (Label itself reports group "Base").
    """
    global _BASELINE_BODY_PROPS
    if _BASELINE_BODY_PROPS is None:
        scratch = FreeCAD.newDocument("_fcops_baseline_probe")
        try:
            body = scratch.addObject("PartDesign::Body", "Body")
            _BASELINE_BODY_PROPS = set(body.PropertiesList)
        finally:
            FreeCAD.closeDocument(scratch.Name)
    return _BASELINE_BODY_PROPS


def _custom_props(obj):
    base = _baseline_body_props()
    out = {}
    for pname in obj.PropertiesList:
        if pname in base:
            continue
        try:
            ptype = obj.getTypeIdOfProperty(pname)
            group = obj.getGroupOfProperty(pname)
        except Exception:
            continue
        val = getattr(obj, pname)
        if not isinstance(val, (str, int, float, bool)):
            val = str(val)
        out[pname] = {"type": ptype, "group": group, "value": val}
    return out


def _placement_dict(obj):
    pl = obj.Placement
    q = pl.Rotation.Q  # (x, y, z, w)
    return {"pos_mm": _vec3(pl.Base),
            "quat": [float(q[0]), float(q[1]), float(q[2]), float(q[3])]}


def _shape_key(shape, placement=None):
    """Cheap geometric fingerprint of a body-local shape, for change
    detection and tessellation-cache keying.

    Placement-INdependent (a pure Placement move must not invalidate the
    tessellation cache) but sensitive to body-local translations: models in
    this project usually position bodies via sketch coordinates with an
    identity Placement, so the local center of mass is part of the key.
    """
    # Absolute quantization, not relative (%g) formatting: recompute /
    # placement-inverse float noise sits at ~1e-15 mm and must map to the
    # same key, while real edits move things by >= micrometres. 1e-4 mm
    # (0.1 um) is far below the 30 um mesh deflection, far above the noise.
    def q(x, nd):
        return round(float(x), nd) + 0.0   # +0.0 folds -0.0 into 0.0
    try:
        vol = q(shape.Volume, 3)
        area = q(shape.Area, 3)
        com = shape.CenterOfMass
        if placement is not None:
            com = placement.inverse().multVec(com)
        com_txt = "%.4f,%.4f,%.4f" % (q(com.x, 4), q(com.y, 4), q(com.z, 4))
    except Exception:
        vol, area, com_txt = -1.0, -1.0, "na"
    return "v%.3f_a%.3f_f%d_e%d_c%s" % (vol, area, len(shape.Faces),
                                        len(shape.Edges), com_txt)


def _body_dict(obj):
    shape = obj.Shape
    tip_name = obj.Tip.Name if obj.Tip else obj.Name
    try:
        closed = bool(shape.isClosed())
    except Exception:
        closed = False
    try:
        com = _vec3(shape.CenterOfMass)
        vol = float(shape.Volume)
    except Exception:
        com, vol = None, None
    bb = shape.BoundBox
    return {
        "name": obj.Name,
        "label": obj.Label,
        "tip": tip_name,
        "face_count": len(shape.Faces),
        "solid_closed": closed,
        "volume_mm3": vol,
        "center_of_mass_mm": com,
        "bbox_mm": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
        "placement": _placement_dict(obj),
        "placement_bound": _placement_expression(obj) is not None,
        "shape_key": _shape_key(shape, obj.Placement),
        "properties": _custom_props(obj),
    }


def _placement_expression(obj):
    """The expression bound to (any part of) Placement, or None.

    An expression-bound Placement re-asserts itself on every recompute, so
    direct set_placement writes would be silently undone; the GUI must
    route such moves through the driving spreadsheet alias instead."""
    try:
        for path, expr in (obj.ExpressionEngine or []):
            # paths look like ".Placement.Base.y" (leading dot) or
            # "Placement.Base.y" depending on origin - normalize
            if str(path).lstrip(".").startswith("Placement"):
                return {"path": str(path), "expression": str(expr)}
    except Exception:
        pass
    return None


def _sheet_dict(sheet):
    aliases = {}
    for cell in sheet.getUsedCells():
        alias = sheet.getAlias(cell)
        if not alias:
            continue
        raw = sheet.getContents(cell)
        try:
            qty = sheet.get(alias)
            # Base.Quantity -> internal-unit float (mm / deg / unitless)
            value = float(FreeCAD.Units.Quantity(qty).Value)
            unit = str(FreeCAD.Units.Quantity(qty).getUserPreferred()[2])
        except Exception:
            value, unit = None, None
        aliases[alias] = {"cell": cell, "raw": raw,
                          "value": value, "unit": unit}
    return {"name": sheet.Name, "label": sheet.Label, "aliases": aliases}


def _structure(doc):
    bodies = [_body_dict(o) for o in doc.Objects
              if o.TypeId == "PartDesign::Body"]
    sheets = [_sheet_dict(o) for o in doc.Objects
              if o.TypeId == "Spreadsheet::Sheet"]
    return {"doc": doc.Name, "label": doc.Label,
            "file": doc.FileName or None,
            "bodies": bodies, "sheets": sheets}


def _placement_tuple(obj):
    pl = obj.Placement
    q = pl.Rotation.Q
    return (round(pl.Base.x, 6), round(pl.Base.y, 6), round(pl.Base.z, 6),
            round(q[0], 9), round(q[1], 9), round(q[2], 9), round(q[3], 9))


def _fingerprint_bodies(doc):
    return {o.Name: (_shape_key(o.Shape, o.Placement), _placement_tuple(o))
            for o in doc.Objects if o.TypeId == "PartDesign::Body"}


def _recompute_and_diff(doc):
    """Recompute and report what a mutation touched.

    Spreadsheet aliases can drive body-local geometry (sketch dims) OR body
    Placements (expression bindings), so both are diffed:
      reshaped -> body-local shape changed, tessellation cache is stale
      moved    -> only the Placement changed, update the view transform
    """
    before = _fingerprint_bodies(doc)
    doc.recompute()
    after = _fingerprint_bodies(doc)
    reshaped = sorted(n for n in after
                      if n not in before or before[n][0] != after[n][0])
    reshaped += sorted(n for n in before if n not in after)
    moved = sorted(n for n in after
                   if n in before and before[n][0] == after[n][0]
                   and before[n][1] != after[n][1])
    invalid = [o.Name for o in doc.Objects
               if "Invalid" in o.State or "Error" in o.State]
    return reshaped, moved, invalid


# ---------------------------------------------------------------------------
# STL writing (mirrors extract_geometry.write_binary_stl)
# ---------------------------------------------------------------------------
def _write_binary_stl(path, mesh):
    facets = mesh.Facets
    header = b"mieworkbench fcops.py STL export".ljust(80, b"\x00")[:80]
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(facets)))
        for facet in facets:
            n = facet.Normal
            fh.write(struct.pack("<3f", float(n.x), float(n.y), float(n.z)))
            for p in facet.Points:
                fh.write(struct.pack("<3f", float(p[0]), float(p[1]),
                                     float(p[2])))
            fh.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

def _mutation_result(doc, extra=None):
    reshaped, moved, invalid = _recompute_and_diff(doc)
    placements = {}
    for name in set(reshaped) | set(moved):
        obj = doc.getObject(name)
        if obj is not None:
            placements[name] = _placement_dict(obj)
    out = {"changed_bodies": reshaped, "moved_bodies": moved,
           "invalid": invalid, "placements": placements}
    if extra:
        out.update(extra)
    return out

def op_ping(params):
    return {"pong": True, "pid": os.getpid(),
            "freecad": ".".join(FreeCAD.Version()[0:3])}


def op_open_document(params):
    path = params["path"]
    if not os.path.isfile(path):
        raise OpError("no such file: %s" % path)
    doc = FreeCAD.openDocument(path)
    doc.recompute()
    return _structure(doc)


def op_new_document(params):
    """Create a new document with an empty 'dim' spreadsheet, save to path."""
    path = params["path"]
    name = os.path.splitext(os.path.basename(path))[0]
    doc = FreeCAD.newDocument(name)
    sheet = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
    sheet.Label = "dim"
    doc.recompute()
    doc.saveAs(path)
    return _structure(doc)


def op_get_structure(params):
    return _structure(_doc(params["doc"]))


def op_tessellate(params):
    """Per-face body-local STLs in metres.

    params: doc, out_dir, bodies (optional list of names; default all),
            lin_defl_mm / ang_defl_deg optional.
    Face STL filename == "<face_id>.stl" under out_dir.
    """
    doc = _doc(params["doc"])
    out_dir = params["out_dir"]
    lin = float(params.get("lin_defl_mm", MESH_LINEAR_DEFLECTION_MM))
    ang = float(params.get("ang_defl_deg", MESH_ANGULAR_DEFLECTION_DEG))
    os.makedirs(out_dir, exist_ok=True)

    import MeshPart  # deferred: only needed here

    want = params.get("bodies")
    bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
              and (want is None or o.Name in want or o.Label in want)]
    result = {}
    for body in bodies:
        tip_name = body.Tip.Name if body.Tip else body.Name
        # body-local copy: strip the Placement, then scale mm -> m
        shp = body.Shape.copy()
        shp.Placement = FreeCAD.Placement()
        mat = FreeCAD.Matrix()
        mat.scale(0.001, 0.001, 0.001)
        shp = shp.transformGeometry(mat)
        faces = []
        for idx, face in enumerate(shp.Faces, start=1):
            face_id = "%s.%s.Face%d" % (body.Name, tip_name, idx)
            mesh = MeshPart.meshFromShape(
                Shape=face,
                LinearDeflection=lin / 1000.0,  # mm value, applied in metres
                AngularDeflection=math.radians(ang))
            stl_path = os.path.join(out_dir, face_id + ".stl")
            _write_binary_stl(stl_path, mesh)
            faces.append({"id": face_id, "stl": stl_path,
                          "area_m2": float(face.Area),
                          "centroid_m": _vec3(face.CenterOfMass),
                          "normal_hint": _face_normal_hint(face)})
        result[body.Name] = {"faces": faces,
                             "shape_key": _shape_key(body.Shape,
                                                     body.Placement),
                             "placement": _placement_dict(body)}
    return {"bodies": result}


def _face_normal_hint(face):
    try:
        u0, u1, v0, v1 = face.ParameterRange
        nrm = face.normalAt((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        if face.Orientation == "Reversed":
            nrm = nrm.multiply(-1.0)
        return _vec3(nrm)
    except Exception:
        return None


def op_set_property(params):
    """Set (creating if absent) a dynamic property in group 'Base'.

    params: doc, body, name, value, ptype optional
            ("string"|"float"|"bool" -> App::Property*; default from value).
    """
    doc = _doc(params["doc"])
    body = _body(doc, params["body"])
    name = str(params["name"])
    value = params["value"]
    ptype = params.get("ptype")
    if ptype is None:
        ptype = ("bool" if isinstance(value, bool)
                 else "float" if isinstance(value, (int, float))
                 else "string")
    fc_type = {"string": "App::PropertyString",
               "float": "App::PropertyFloat",
               "bool": "App::PropertyBool"}.get(ptype)
    if fc_type is None:
        raise OpError("unsupported ptype %r" % ptype)
    if name not in body.PropertiesList:
        body.addProperty(fc_type, name, "Base")
    setattr(body, name,
            value if ptype != "string" else str(value))
    return _mutation_result(doc)


def op_remove_property(params):
    doc = _doc(params["doc"])
    body = _body(doc, params["body"])
    name = str(params["name"])
    if name in _baseline_body_props():
        raise OpError("refusing to remove built-in property %r" % name)
    if name not in body.PropertiesList:
        raise OpError("body %s has no property %r" % (body.Name, name))
    body.removeProperty(name)
    return _mutation_result(doc)


def op_set_spreadsheet(params):
    """Set an aliased cell's raw content, preserving '=<num> <unit>' form.

    params: doc, sheet (name or label), alias, raw.
    Returns bodies whose geometry changed, for selective re-tessellation.
    """
    doc = _doc(params["doc"])
    sheet = _sheet(doc, params["sheet"])
    alias = str(params["alias"])
    cell = sheet.getCellFromAlias(alias)
    if not cell:
        raise OpError("sheet %s has no alias %r" % (sheet.Label, alias))
    sheet.set(cell, str(params["raw"]))
    return _mutation_result(doc, {"cell": cell})


def op_set_placement(params):
    """params: doc, body (name, label, or miewb_group of a multi-body
    element), pos_mm [x,y,z], quat [x,y,z,w].

    A miewb_group match applies the SAME placement to every member —
    rigid, because primitive builders bake inter-body offsets into local
    geometry and create members with identity placements."""
    doc = _doc(params["doc"])
    try:
        targets = [_body(doc, params["body"])]
    except OpError:
        targets = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
                   and getattr(o, "miewb_group", None) == str(params["body"])]
        if not targets:
            raise
    for body in targets:
        bound = _placement_expression(body)
        if bound is not None:
            raise OpError(
                "body %s Placement is expression-bound (%s = %s); edit the "
                "driving spreadsheet alias instead of setting the placement"
                % (body.Name, bound["path"], bound["expression"]))
    pos = params["pos_mm"]
    q = params["quat"]
    for body in targets:
        body.Placement = FreeCAD.Placement(
            FreeCAD.Vector(float(pos[0]), float(pos[1]), float(pos[2])),
            FreeCAD.Rotation(float(q[0]), float(q[1]), float(q[2]),
                             float(q[3])))
    # placement changes never alter body-local geometry; skip the diff
    doc.recompute()
    return {"placements": {b.Name: _placement_dict(b) for b in targets}}


def op_import_primitive(params):
    """Copy a primitive .FCStd's bodies + parameter spreadsheet into an
    open document.

    params: doc, path (primitive .FCStd), label (new element label).
    Single-body primitives get Label=label; multi-body ones (achromat,
    pbs_cube) get "<label>_<original label>". The primitive's 'dim' sheet
    is copied explicitly (its values are baked into the geometry, so it is
    NOT a dependency) and relabeled 'dim_<label>'; each body's
    miewb_group prop is rewritten to `label` so rebuilds stay grouped.
    """
    doc = _doc(params["doc"])
    path = params["path"]
    label = str(params["label"])
    if not os.path.isfile(path):
        raise OpError("no such primitive: %s" % path)
    for b in doc.Objects:
        if b.TypeId == "PartDesign::Body" and b.Label == label:
            raise OpError("label %r already used in document" % label)

    pre_names = {o.Name for o in doc.Objects}
    src = FreeCAD.openDocument(path)
    try:
        src_bodies = [o for o in src.Objects
                      if o.TypeId == "PartDesign::Body"]
        src_sheets = [o for o in src.Objects
                      if o.TypeId == "Spreadsheet::Sheet"]
        if not src_bodies:
            raise OpError("primitive %s has no PartDesign::Body" % path)
        src_labels = {o.Name: o.Label for o in src.Objects}
        doc.copyObject(src_bodies + src_sheets, True)
    finally:
        FreeCAD.closeDocument(src.Name)

    new_objs = [o for o in doc.Objects if o.Name not in pre_names]
    new_bodies = [o for o in new_objs if o.TypeId == "PartDesign::Body"]
    new_sheets = [o for o in new_objs if o.TypeId == "Spreadsheet::Sheet"]
    if not new_bodies:
        raise OpError("copyObject produced no bodies")
    if len(new_bodies) == 1:
        new_bodies[0].Label = label
    else:
        for b in new_bodies:
            b.Label = "%s_%s" % (label, b.Label)
    for b in new_bodies:
        if "miewb_group" in b.PropertiesList:
            b.miewb_group = label
    for sheet in new_sheets:
        sheet.Label = "dim_%s" % label
    return _mutation_result(doc, {
        "bodies": [_body_dict(b) for b in new_bodies],
        "sheets": [_sheet_dict(s) for s in new_sheets]})


def op_rebuild_primitive(params):
    """Rebuild a primitive-built element from its parameter sheet.

    params: doc, group (the element label / miewb_group value),
            sheet (name or label; default 'dim_<group>').
    Preserves Labels, Placements and user-added Base props. Only works on
    bodies carrying miewb_primitive/miewb_group tags (i.e. built by
    primitivelib); hand-authored elements use set_spreadsheet instead.
    """
    import primitivelib
    doc = _doc(params["doc"])
    group = str(params["group"])
    bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
              and getattr(o, "miewb_group", None) == group]
    if not bodies:
        raise OpError("no bodies with miewb_group %r" % group)
    kind = getattr(bodies[0], "miewb_primitive", None)
    if kind not in primitivelib.PRIMITIVES:
        raise OpError("unknown primitive kind %r on group %r"
                      % (kind, group))
    sheet = _sheet(doc, params.get("sheet") or ("dim_%s" % group))
    new_bodies = primitivelib.rebuild_element(doc, sheet, kind, group)
    return _mutation_result(doc, {
        "bodies": [_body_dict(b) for b in new_bodies]})


def _element_bodies(doc, key):
    """All bodies of the element identified by `key`, plus its group name.

    `key` is a miewb_group value, or a single body's Name/Label; a matched
    body that carries a miewb_group expands to the whole group (an element
    is always moved/deleted/duplicated as a unit, same rigid-group rule as
    op_set_placement). Returns (bodies, group_or_None)."""
    key = str(key)
    bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
              and getattr(o, "miewb_group", None) == key]
    if bodies:
        return bodies, key
    body = _body(doc, key)
    group = getattr(body, "miewb_group", None)
    if group:
        return ([o for o in doc.Objects if o.TypeId == "PartDesign::Body"
                 and getattr(o, "miewb_group", None) == group], str(group))
    return [body], None


def _element_sheets(doc, group):
    if not group:
        return []
    return [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"
            and o.Label == "dim_%s" % group]


def _remove_body(doc, body):
    """Remove a body and its owned features (same pattern as
    primitivelib.rebuild_element)."""
    feats = list(getattr(body, "Group", []) or [])
    doc.removeObject(body.Name)
    for f in feats:
        try:
            doc.removeObject(f.Name)
        except Exception:
            pass


def op_delete_element(params):
    """Delete an element (all bodies of a miewb_group, or one ungrouped
    body) and its dim_<group> sheet.

    params: doc, element (group value or body name/label),
            stash_path (optional) - before deleting, save a copy of the
            element into a standalone .FCStd there (labels/placements/
            props preserved verbatim), the crash-safe pre-image that
            op_import_bodies restores for undo.
    """
    doc = _doc(params["doc"])
    bodies, group = _element_bodies(doc, params["element"])
    sheets = _element_sheets(doc, group)
    stash_path = params.get("stash_path")
    if stash_path:
        stash = FreeCAD.newDocument("miewb_stash")
        try:
            labels = {o.Name: o.Label for o in bodies + sheets}
            pre = {o.Name for o in stash.Objects}
            stash.copyObject(bodies + sheets, True)
            new_objs = [o for o in stash.Objects if o.Name not in pre]
            # copyObject can uniquify labels; reassert the originals (a
            # fresh doc has no conflicts, so this always sticks)
            by_label_order = [o for o in new_objs if o.TypeId in
                              ("PartDesign::Body", "Spreadsheet::Sheet")]
            src_order = bodies + sheets
            if len(by_label_order) == len(src_order):
                for src, dst in zip(src_order, by_label_order):
                    dst.Label = labels[src.Name]
            stash.recompute()
            stash.saveAs(stash_path)
        finally:
            FreeCAD.closeDocument(stash.Name)
    deleted = [b.Name for b in bodies]
    for b in bodies:
        _remove_body(doc, b)
    for s in sheets:
        try:
            doc.removeObject(s.Name)
        except Exception:
            pass
    return _mutation_result(doc, {"deleted": deleted,
                                  "stash": stash_path or None})


def op_import_bodies(params):
    """Copy every body + sheet of a .FCStd into the open document with NO
    relabeling (labels/placements/props/miewb_group/sheet labels preserved
    verbatim) - the restore half of delete_element's stash, and the
    verbatim counterpart of op_import_primitive.

    params: doc, path.
    """
    doc = _doc(params["doc"])
    path = params["path"]
    if not os.path.isfile(path):
        raise OpError("no such file: %s" % path)
    pre_names = {o.Name for o in doc.Objects}
    src = FreeCAD.openDocument(path)
    try:
        src_bodies = [o for o in src.Objects
                      if o.TypeId == "PartDesign::Body"]
        src_sheets = [o for o in src.Objects
                      if o.TypeId == "Spreadsheet::Sheet"]
        if not src_bodies:
            raise OpError("%s has no PartDesign::Body" % path)
        src_labels = [o.Label for o in src_bodies + src_sheets]
        doc.copyObject(src_bodies + src_sheets, True)
    finally:
        FreeCAD.closeDocument(src.Name)
    new_objs = [o for o in doc.Objects if o.Name not in pre_names]
    new_bodies = [o for o in new_objs if o.TypeId == "PartDesign::Body"]
    new_sheets = [o for o in new_objs if o.TypeId == "Spreadsheet::Sheet"]
    if not new_bodies:
        raise OpError("copyObject produced no bodies")
    # reassert the source labels (copyObject may have uniquified them)
    for label, obj in zip(src_labels, new_bodies + new_sheets):
        obj.Label = label
    return _mutation_result(doc, {
        "bodies": [_body_dict(b) for b in new_bodies],
        "sheets": [_sheet_dict(s) for s in new_sheets]})


def op_duplicate_element(params):
    """Duplicate an element in-document under a new label/group.

    params: doc, element (group value or body name/label), new_label.
    Single-body elements get Label=new_label; multi-body members swap
    their '<group>_' label prefix for '<new_label>_'. miewb_group is
    rewritten and dim_<group> is copied to dim_<new_label>.
    """
    doc = _doc(params["doc"])
    new_label = str(params["new_label"])
    for b in doc.Objects:
        if b.TypeId == "PartDesign::Body" and b.Label == new_label:
            raise OpError("label %r already used in document" % new_label)
    bodies, group = _element_bodies(doc, params["element"])
    sheets = _element_sheets(doc, group)
    old_labels = [o.Label for o in bodies]

    pre_names = {o.Name for o in doc.Objects}
    doc.copyObject(bodies + sheets, True)
    new_objs = [o for o in doc.Objects if o.Name not in pre_names]
    new_bodies = [o for o in new_objs if o.TypeId == "PartDesign::Body"]
    new_sheets = [o for o in new_objs if o.TypeId == "Spreadsheet::Sheet"]
    if not new_bodies:
        raise OpError("copyObject produced no bodies")
    if len(new_bodies) == 1:
        new_bodies[0].Label = new_label
    else:
        for old_label, b in zip(old_labels, new_bodies):
            suffix = (old_label[len(group) + 1:]
                      if group and old_label.startswith(group + "_")
                      else old_label)
            b.Label = "%s_%s" % (new_label, suffix)
    for b in new_bodies:
        if "miewb_group" in b.PropertiesList:
            b.miewb_group = new_label
    for sheet in new_sheets:
        sheet.Label = "dim_%s" % new_label
    return _mutation_result(doc, {
        "bodies": [_body_dict(b) for b in new_bodies],
        "sheets": [_sheet_dict(s) for s in new_sheets]})


def op_save(params):
    doc = _doc(params["doc"])
    _, _, invalid = _recompute_and_diff(doc)
    if invalid:
        raise OpError("refusing to save with invalid objects: %s"
                      % ", ".join(invalid))
    if not doc.FileName:
        raise OpError("document has no file name; use save_as")
    doc.save()
    return {"file": doc.FileName}


def op_save_as(params):
    doc = _doc(params["doc"])
    _, _, invalid = _recompute_and_diff(doc)
    if invalid:
        raise OpError("refusing to save with invalid objects: %s"
                      % ", ".join(invalid))
    doc.saveAs(params["path"])
    return {"file": doc.FileName}


def op_save_copy(params):
    doc = _doc(params["doc"])
    _, _, invalid = _recompute_and_diff(doc)
    if invalid:
        raise OpError("refusing to save with invalid objects: %s"
                      % ", ".join(invalid))
    doc.saveCopy(params["path"])
    return {"file": params["path"]}


def op_close(params):
    doc = _doc(params["doc"])
    name = doc.Name
    FreeCAD.closeDocument(name)
    return {"closed": name}


def op_check(params):
    """Model sanity: recompute errors, open solids, pairwise overlaps.

    params: doc, overlaps (bool, default True; O(n^2) boolean commons).
    """
    doc = _doc(params["doc"])
    _, _, invalid = _recompute_and_diff(doc)
    bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
    open_solids = []
    for b in bodies:
        try:
            if not b.Shape.isClosed():
                open_solids.append(b.Name)
        except Exception:
            open_solids.append(b.Name)
    overlaps = []
    if params.get("overlaps", True):
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                try:
                    common = a.Shape.common(b.Shape)
                    if common.Volume > 1e-9:
                        overlaps.append({"a": a.Name, "b": b.Name,
                                         "volume_mm3": float(common.Volume)})
                except Exception:
                    pass
    return {"invalid": invalid, "open_solids": open_solids,
            "overlaps": overlaps}


def op_list_documents(params):
    return {"documents": [d for d in FreeCAD.listDocuments()]}


OPS = {name[3:]: fn for name, fn in list(globals().items())
       if name.startswith("op_") and callable(fn)}


def dispatch(op, params):
    """Run one op; returns (ok, result_or_error_dict)."""
    fn = OPS.get(op)
    if fn is None:
        return False, {"error": "unknown op %r" % op,
                       "known": sorted(OPS)}
    try:
        return True, fn(params or {})
    except Exception as exc:
        return False, {"error": "%s: %s" % (type(exc).__name__, exc),
                       "traceback": traceback.format_exc(limit=8)}
