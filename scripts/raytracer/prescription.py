#!/usr/bin/env python3
# =============================================================================
# prescription.py — the prescription-primary data model (engine3.md Sec 3, P5).
#
# THE PRINCIPLE: the optical prescription is the truth; the CAD is a view of
# it. A `prescription.json` records, per optical element, the BASIC TERMS the
# element was authored in (the dim-sheet parameters) and the exact analytic
# optical surfaces those terms imply -- in the SAME surface language
# model.json already uses (common._SURFACE_REQ). At extract time the surfaces
# are (a) verified against the tessellated FreeCAD faces to the 1 um gate and
# (b) emitted into model.json FROM THE PRESCRIPTION (exact params, never
# tessellation-derived canonicalization). The CAD may never silently drift
# from its prescription: a mismatch > 1 um is a hard error.
#
# CONTRACT: pure python standard library only -- like train_solver.py and
# common.py, this module is imported from EVERY interpreter stack (the GUI
# venv, the optics env, system python3, and FreeCAD's embedded python), so it
# must not import numpy / FreeCAD / torch. It only loads, validates, and
# saves the json; SURFACE GENERATION from primitive parameters lives in
# primitivelib.build_prescription_entry() (colocated with the geometry
# builders -- the single authoring path), and the FreeCAD-side verification
# lives in extract_geometry.py (it needs the tessellated shape).
#
# Storage:
#   .MieWB zip : a `prescription.json` member (miewb_tool round-trips it).
#   bare .FCStd: a sidecar `<stem>.prescription.json` next to the model.
#   .MieSim    : carried inside the embedded input.MieWB.
#
# Schema (schema_version 1):
#   {"schema_version": 1,
#    "elements": {
#       "<key>": {                       # key = the element's miewb_group
#          "kind":  "<primitive kind>" | "custom",
#          "params": {<basic terms, SI: metres / radians>},
#          "surfaces": [
#             {"role": "front"|"back"|"edge"|..., "material": "<glass>"?,
#              <the model.json surface dict, LOCAL body coords, SI metres:
#               type + geometry keys per common._SURFACE_REQ>},
#             ...
#          ]}}}
#
# Surface coordinates are BODY-LOCAL (the frame the builder builds in: the
# optical axis is local +x, the front vertex at the origin -- the primitive
# convention). extract_geometry transforms them to global through the body's
# FreeCAD Placement (exact, the same transform OCC applies to the geometry),
# so the stored prescription is placement-independent and survives the element
# being moved/chained/folded in the optical train.
#
# Self-check:  python3 scripts/raytracer/prescription.py
# =============================================================================
import json
import os
import sys
from pathlib import Path

# scripts/ is this file's grandparent; put it on the path so `import common`
# resolves whichever interpreter loads us (the raytracer package does the same
# with a bare `import common`).
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import common  # noqa: E402  (pure stdlib hub -- surface schema + validators)

SCHEMA_VERSION = 1

# The prescription reuses model.json's surface language verbatim (do NOT
# invent a second one). These are the two extra per-surface keys the
# prescription layer adds on top of a model.json surface dict.
_ROLE_KEY = "role"
_MATERIAL_KEY = "material"


class PrescriptionError(ValueError):
    """Raised on a malformed / self-inconsistent prescription.json."""


# ---------------------------------------------------------------------------
# Load / validate / save
# ---------------------------------------------------------------------------
def validate(doc, ctx="prescription"):
    """Validate a prescription dict in place; returns it. Raises
    PrescriptionError naming the offending element/surface on any problem.
    The surface geometry keys are checked with common's own
    _check_surface_params so the two schemas can never drift apart."""
    if not isinstance(doc, dict):
        raise PrescriptionError("%s: top level must be an object" % ctx)
    sv = doc.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise PrescriptionError(
            "%s: schema_version must be %d (got %r)"
            % (ctx, SCHEMA_VERSION, sv))
    elements = doc.get("elements")
    if not isinstance(elements, dict):
        raise PrescriptionError("%s: 'elements' must be an object" % ctx)
    for key, entry in elements.items():
        ectx = "%s element %r" % (ctx, key)
        if not isinstance(entry, dict):
            raise PrescriptionError("%s: must be an object" % ectx)
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            raise PrescriptionError("%s: 'kind' must be a non-empty string"
                                    % ectx)
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise PrescriptionError("%s: 'params' must be an object" % ectx)
        for pk, pv in params.items():
            if not isinstance(pv, (int, float)) or pv != pv:
                raise PrescriptionError(
                    "%s: param %r must be a number (got %r)"
                    % (ectx, pk, pv))
        surfaces = entry.get("surfaces")
        if not isinstance(surfaces, list) or not surfaces:
            raise PrescriptionError(
                "%s: 'surfaces' must be a non-empty list" % ectx)
        for i, surf in enumerate(surfaces):
            sctx = "%s surface %d" % (ectx, i)
            if not isinstance(surf, dict):
                raise PrescriptionError("%s: must be an object" % sctx)
            role = surf.get(_ROLE_KEY)
            if not isinstance(role, str) or not role:
                raise PrescriptionError(
                    "%s: 'role' must be a non-empty string" % sctx)
            mat = surf.get(_MATERIAL_KEY)
            if mat is not None and not isinstance(mat, str):
                raise PrescriptionError("%s: 'material' must be a string" % sctx)
            stype = surf.get("type")
            if stype not in common.SURFACE_TYPES or stype == "mesh":
                raise PrescriptionError(
                    "%s: surface 'type' must be one of %r (not 'mesh' -- a "
                    "prescription surface is always analytic)"
                    % (sctx, [t for t in common.SURFACE_TYPES if t != "mesh"]))
            try:
                common._check_surface_params(stype, surf, sctx)
            except common.ContractError as exc:
                raise PrescriptionError(str(exc))
    return doc


def load(path):
    """Load + validate a prescription.json. Returns the dict."""
    with open(path) as fh:
        doc = json.load(fh)
    return validate(doc, ctx=str(path))


def loads(text, ctx="prescription"):
    return validate(json.loads(text), ctx=ctx)


def save(path, doc):
    """Validate then atomically write a prescription.json (stable key order,
    so it round-trips byte-identically and diffs cleanly)."""
    validate(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(str(tmp), str(path))
    return path


def dumps(doc):
    validate(doc)
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Assembly / lookup helpers
# ---------------------------------------------------------------------------
def new_document(entries=None):
    """Build a prescription doc from {key: entry} (each entry as returned by
    primitivelib.build_prescription_entry). Validates before returning."""
    doc = {"schema_version": SCHEMA_VERSION, "elements": dict(entries or {})}
    return validate(doc)


def element_for(doc, key):
    """The entry for element `key`, or None. Never raises."""
    if not isinstance(doc, dict):
        return None
    return doc.get("elements", {}).get(key)


SIDECAR_SUFFIX = ".prescription.json"


def sidecar_path(fcstd_path):
    """The bare-FCStd sidecar prescription path: <stem>.prescription.json next
    to the model (document precedence: an explicit path/dict beats this)."""
    p = Path(fcstd_path)
    return p.with_name(p.stem + SIDECAR_SUFFIX)


# ---------------------------------------------------------------------------
def _self_check():
    # a minimal round-trip on a hand-built pcx-like entry
    entry = {
        "kind": "lens_pcx",
        "params": {"R_front": 0.025, "ct": 0.005, "aperture": 0.020},
        "surfaces": [
            {"role": "front", "material": "bk7", "type": "sphere",
             "center": [0.025, 0.0, 0.0], "radius": 0.025},
            {"role": "back", "material": "bk7", "type": "plane",
             "origin": [0.005, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
        ],
    }
    doc = new_document({"Lens1": entry})
    text = dumps(doc)
    back = loads(text)
    assert back == doc, "round-trip mismatch"
    assert element_for(back, "Lens1")["kind"] == "lens_pcx"
    # a deliberately bad surface must be rejected
    bad = json.loads(text)
    bad["elements"]["Lens1"]["surfaces"][0]["radius"] = -1.0
    try:
        validate(bad)
    except PrescriptionError:
        pass
    else:
        raise AssertionError("negative radius should have failed validation")
    print("prescription.py self-check OK")


if __name__ == "__main__":
    _self_check()
