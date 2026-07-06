#!/usr/bin/env python3
# =============================================================================
# common.py — stdlib-only shared hub for the optical ray tracer pipeline.
#
# Imported by every interpreter stack (FreeCAD embedded python, the optics
# mamba env, pvpython, and system python3). It must therefore import NOTHING
# beyond the python standard library — no numpy, no FreeCAD, no torch.
#
# Contents: pinned tool paths, presets, face-spec / physics-option parsing,
# model.json contract validation, runtime/memory estimators, variant naming.
#
# Self-check:  python3 scripts/common.py
# =============================================================================
import json
import math
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and pinned interpreters (SimulationsGuide.md §1/§2 conventions)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
BASEMODELS_DIR = PROJECT_DIR / "basemodels"
GEOMETRY_DIR = PROJECT_DIR / "geometry"
RESULTS_DIR = PROJECT_DIR / "results"
# Optical-properties library: materials.csv at the root, one subdirectory
# per category, per-item tables under <category>/tables/ (README §7).
OPTPROPS_DIR = PROJECT_DIR / "opticalproperties"
MATERIALS_CSV = OPTPROPS_DIR / "materials.csv"
NK_DATA_DIR = OPTPROPS_DIR / "nk"
COATINGS_CSV = OPTPROPS_DIR / "coating" / "coatings.csv"
BIREFRINGENCE_CSV = OPTPROPS_DIR / "birefringence" / "uniaxial.csv"
POLARIZERS_CSV = OPTPROPS_DIR / "polarizer" / "polarizers.csv"
FILTERS_CSV = OPTPROPS_DIR / "filter" / "filters.csv"
GRATINGS_CSV = OPTPROPS_DIR / "grating" / "gratings.csv"
CALIBRATION_JSON = RESULTS_DIR / ".calibration.json"

FREECAD_APPIMAGE = "/home3/freecad/FreeCAD.AppImage"
OPTICS_PYTHON = "/home3/optics/env/bin/python"
PVPYTHON = ("/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64"
            "/bin/pvpython")

# ---------------------------------------------------------------------------
# Physical constants / units
# ---------------------------------------------------------------------------
C_M_PER_S = 299_792_458.0
NM = 1e-9
UM = 1e-6
MM = 1e-3

def nm_to_m(v):
    return float(v) * NM

def mw_to_w(v):
    return float(v) * 1e-3

# ---------------------------------------------------------------------------
# Presets (fidelity ladder, antenna-project style)
# ---------------------------------------------------------------------------
PRESETS = {
    "quick":    {"rays": 1e5, "resolution": 512,  "nlambda": 5,
                 "spectral_bins": 16, "viz_rays": 500},
    "normal":   {"rays": 1e6, "resolution": 2048, "nlambda": 9,
                 "spectral_bins": 16, "viz_rays": 2000},
    "detailed": {"rays": 1e7, "resolution": 4096, "nlambda": 17,
                 "spectral_bins": 32, "viz_rays": 5000},
}

DEFAULTS = {
    "max_reflections": 6,
    "power_floor": 1e-4,
    # aligned with ExplicitRealization.MAX_BRUTE so the default can never
    # land in the explicit-selected-but-over-the-brute-cap dead zone
    "particle_threshold": 2e5,
    "seeds": 1,
    "backend": "auto",
    "ambient_material": "air",
}

# ---------------------------------------------------------------------------
# Face designation:  Body[001].[Feature].Face[N]   e.g. Body001.Pad.Face2
# ---------------------------------------------------------------------------
_FACE_RE = re.compile(r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\.Face(\d+)$")

def parse_face_spec(spec):
    """'Body001.Pad.Face2' -> dict(body, feature, face_index). Hard error."""
    m = _FACE_RE.match(spec.strip())
    if not m:
        raise ValueError(
            "bad face spec %r (expected Body.Feature.FaceN, e.g. "
            "Body001.Pad.Face2)" % spec)
    return {"body": m.group(1), "feature": m.group(2),
            "face_index": int(m.group(3)), "id": spec.strip()}

def parse_grating_value(value):
    """Grating options without the face part.

    Forms:  '600:v[:orders=a..b][:eff=e1,...]'   (lamellar, back-compat)
            '@name[:orders=a..b]'                (opticalproperties registry)
    Returns dict(model, lines_per_mm, groove, orders, efficiencies,
    registry).  Registry entries are resolved against grating/gratings.csv
    at trace time (model/lines_per_mm/params come from the registry row).
    """
    value = value.strip()
    out = {"model": "lamellar", "lines_per_mm": None, "groove": "v",
           "orders": (-2, 2), "efficiencies": None, "registry": None}
    if value.startswith("@"):
        parts = value[1:].split(":")
        if not parts[0]:
            raise ValueError("empty grating registry name in %r" % value)
        out["registry"] = parts[0]
        extras = parts[1:]
    else:
        parts = value.split(":")
        if len(parts) < 2:
            raise ValueError(
                "bad grating value %r (expected lines_per_mm:groove"
                "[:orders=a..b][:eff=e1,e2,...] or @registryname)" % value)
        out["lines_per_mm"] = float(parts[0])
        if out["lines_per_mm"] <= 0:
            raise ValueError("grating lines/mm must be > 0 in %r" % value)
        groove = parts[1]
        if groove not in ("u", "v") and not _is_vector3(groove):
            raise ValueError(
                "grating groove %r must be 'u', 'v', or 'x,y,z'" % groove)
        out["groove"] = groove
        extras = parts[2:]
    for extra in extras:
        if extra.startswith("orders="):
            lo, hi = extra[len("orders="):].split("..")
            out["orders"] = (int(lo), int(hi))
        elif extra.startswith("eff=") and out["registry"] is None:
            out["efficiencies"] = [float(x) for x in
                                   extra[len("eff="):].split(",")]
        else:
            raise ValueError("unknown grating option %r in %r"
                             % (extra, value))
    n_orders = out["orders"][1] - out["orders"][0] + 1
    if out["efficiencies"] is not None \
            and len(out["efficiencies"]) != n_orders:
        raise ValueError(
            "grating %r: %d efficiencies given for %d orders"
            % (value, len(out["efficiencies"]), n_orders))
    return out

def parse_grating_spec(spec):
    """'Body.Obj.FaceN:600:v[:orders=-2..2][:eff=...]' or
    'Body.Obj.FaceN:@name[:orders=a..b]'"""
    spec = spec.strip()
    head, sep, rest = spec.partition(":")
    if not sep:
        raise ValueError(
            "bad grating spec %r (expected Body.Obj.FaceN:<grating value>)"
            % spec)
    face = parse_face_spec(head)
    out = parse_grating_value(rest)
    out["face"] = face
    return out

def parse_rough_value(value):
    """'50[:lcorr=10]' -> dict(sigma_nm, lcorr_um)."""
    parts = str(value).strip().split(":")
    sigma_nm = float(parts[0])
    if sigma_nm < 0:
        raise ValueError("roughness sigma must be >= 0 in %r" % value)
    out = {"sigma_nm": sigma_nm, "lcorr_um": 10.0}
    for extra in parts[1:]:
        if extra.startswith("lcorr="):
            out["lcorr_um"] = float(extra[len("lcorr="):])
            if out["lcorr_um"] <= 0:
                raise ValueError("roughness lcorr must be > 0 in %r" % value)
        else:
            raise ValueError("unknown roughness option %r in %r"
                             % (extra, value))
    return out

def parse_rough_spec(spec):
    """'Body.Obj.FaceN:50[:lcorr=10]'  (sigma in nm, lcorr in um)"""
    spec = spec.strip()
    head, sep, rest = spec.partition(":")
    if not sep:
        raise ValueError(
            "bad roughness spec %r (expected Body.Obj.FaceN:sigma_nm"
            "[:lcorr=um])" % spec)
    face = parse_face_spec(head)
    out = parse_rough_value(rest)
    out["face"] = face
    return out

# ---------------------------------------------------------------------------
# Per-face property maps (body properties: coating / roughness / grating /
# surface_override).  Value forms, README §5:
#   'MgF2'                      -> {'__all__': 'MgF2'}        (every face)
#   'Face3=MgF2;Face5=x'        -> bare FaceN expanded with body+feature
#   'Body.Feat.Face3=MgF2'      -> full ids pass through
# ---------------------------------------------------------------------------
FACEMAP_ALL = "__all__"
_BARE_FACE_RE = re.compile(r"^Face(\d+)$")

def parse_facemap_spec(value, body=None, feature=None):
    """Parse a per-face property string into {face_id_or_FACEMAP_ALL: str}.

    Bare 'FaceN' keys require body+feature context (extract time); the
    trace stage only ever sees fully-qualified ids or FACEMAP_ALL.
    """
    value = str(value).strip()
    if not value:
        raise ValueError("empty per-face property value")
    out = {}
    entries = [e.strip() for e in value.split(";") if e.strip()]
    if not entries:
        raise ValueError("empty per-face property value %r" % value)
    for entry in entries:
        key, sep, val = entry.partition("=")
        key = key.strip()
        is_facekey = sep and (_BARE_FACE_RE.match(key)
                              or _FACE_RE.match(key))
        if not is_facekey:
            # whole entry is an every-face value ('MgF2', '200:lcorr=5')
            if len(entries) != 1:
                raise ValueError(
                    "per-face property %r mixes an all-faces value with "
                    "per-face entries" % value)
            out[FACEMAP_ALL] = entry
            return out
        if _BARE_FACE_RE.match(key):
            if not (body and feature):
                raise ValueError(
                    "bare face key %r needs Body.Feature context "
                    "(use the full Body.Feature.FaceN form)" % key)
            key = "%s.%s.%s" % (body, feature, key)
        parse_face_spec(key)   # syntax check
        if not val.strip():
            raise ValueError("empty value for face %r in %r" % (key, value))
        if key in out:
            raise ValueError("duplicate face %r in %r" % (key, value))
        out[key] = val.strip()
    return out

# ---------------------------------------------------------------------------
# Source polarization + crystal/polarizer axis specs (README §5.2/§5.9)
# ---------------------------------------------------------------------------
POLARIZATION_KINDS = ("unpolarized", "linear", "circular", "elliptical")

def parse_polarization_spec(spec):
    """'unpolarized' | 'linear:<deg>' | 'circular:left|right' |
    'elliptical:<psi_deg>:<chi_deg>'  ->  canonical dict.

    Angle reference: global +z projected into the transverse plane
    (fallback +y when the emission direction is parallel to z).
    """
    parts = [p.strip() for p in str(spec).strip().lower().split(":")]
    kind = parts[0]
    if kind == "unpolarized":
        if len(parts) != 1:
            raise ValueError("unpolarized takes no arguments: %r" % spec)
        return {"kind": "unpolarized"}
    if kind == "linear":
        if len(parts) != 2:
            raise ValueError("linear polarization needs an angle: %r" % spec)
        return {"kind": "linear", "angle_deg": float(parts[1])}
    if kind == "circular":
        if len(parts) != 2 or parts[1] not in ("left", "right"):
            raise ValueError(
                "circular polarization is 'circular:left' or "
                "'circular:right': %r" % spec)
        return {"kind": "circular", "handedness": parts[1]}
    if kind == "elliptical":
        if len(parts) != 3:
            raise ValueError(
                "elliptical polarization is 'elliptical:<psi_deg>:"
                "<chi_deg>': %r" % spec)
        chi = float(parts[2])
        if not (-45.0 <= chi <= 45.0):
            raise ValueError("ellipticity chi must be in [-45, 45] deg: %r"
                             % spec)
        return {"kind": "elliptical", "psi_deg": float(parts[1]),
                "chi_deg": chi}
    raise ValueError("unknown polarization kind %r (one of %s)"
                     % (kind, ", ".join(POLARIZATION_KINDS)))

def parse_axis_spec(spec):
    """'x,y,z' -> normalized [x, y, z].  Hard error on zero length."""
    try:
        v = [float(x) for x in str(spec).strip().split(",")]
    except ValueError:
        raise ValueError("bad axis spec %r (expected 'x,y,z')" % spec)
    if len(v) != 3:
        raise ValueError("axis spec %r needs exactly 3 components" % spec)
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        raise ValueError("axis spec %r has zero length" % spec)
    return [x / n for x in v]

def parse_particles_spec(spec):
    """'box=x0,y0,z0:dx,dy,dz;material=water;phi=1e-4;median_um=10;gsd=1.6'

    box is corner:size in mm (project CAD units).  Returns SI (m) box.
    """
    out = {"box_corner_m": None, "box_size_m": None, "material": None,
           "phi": None, "median_um": None, "gsd": 1.6}
    for kv in spec.strip().split(";"):
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError("bad particles field %r in %r" % (kv, spec))
        k, v = kv.split("=", 1)
        if k == "box":
            corner_s, size_s = v.split(":")
            corner = [float(x) * MM for x in corner_s.split(",")]
            size = [float(x) * MM for x in size_s.split(",")]
            if len(corner) != 3 or len(size) != 3:
                raise ValueError("particles box needs 3+3 numbers: %r" % v)
            if any(s <= 0 for s in size):
                raise ValueError("particles box size must be > 0: %r" % v)
            out["box_corner_m"], out["box_size_m"] = corner, size
        elif k == "material":
            out["material"] = v
        elif k == "phi":
            out["phi"] = float(v)
        elif k == "median_um":
            out["median_um"] = float(v)
        elif k == "gsd":
            out["gsd"] = float(v)
        elif k == "seed":
            out["seed"] = int(v)
        else:
            raise ValueError("unknown particles field %r in %r" % (k, spec))
    missing = [k for k in ("material", "phi", "median_um") if out[k] is None]
    if missing:
        raise ValueError("particles spec %r missing %s" % (spec, missing))
    if not (0 < out["phi"] < 1):
        raise ValueError("particles phi (mass fraction) must be in (0,1)")
    if out["gsd"] < 1.0:
        raise ValueError("particles gsd (geometric std dev) must be >= 1")
    # default box: 10x20x20 mm centered on the x-axis just before the origin
    if out["box_corner_m"] is None:
        out["box_corner_m"] = [-12.0 * MM, -10.0 * MM, -10.0 * MM]
        out["box_size_m"] = [10.0 * MM, 20.0 * MM, 20.0 * MM]
    return out

def _is_vector3(s):
    try:
        return len([float(x) for x in s.split(",")]) == 3
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# Sweep semantics (identical to antenna permute_model.py — project-wide law)
# ---------------------------------------------------------------------------
def sweep_values(vmin, vmax, n):
    """n=0 -> [min]; n=1 -> [min,max]; n>1 -> n+1 evenly spaced values."""
    if n < 0:
        raise ValueError("sweep n must be >= 0 (got %d)" % n)
    if n == 0:
        return [vmin]
    if n == 1:
        return [vmin, vmax]
    return [vmin + i * (vmax - vmin) / n for i in range(n + 1)]

def variant_name(stem, var, value):
    """simpledipole, dipolelen, 37.5 -> 'simpledipole-dipolelen37p5'
    (decimals '.'->'p', minus->'m' — matches the cfdsim convention)."""
    sval = ("%g" % value).replace(".", "p").replace("-", "m")
    return "%s-%s%s" % (stem, var, sval)

def case_name(preset, tag=None, seed=None):
    parts = [preset]
    if tag:
        parts.append(str(tag))
    if seed is not None:
        parts.append("seed%d" % seed)
    return "-".join(parts)

# ---------------------------------------------------------------------------
# model.json contract validation (hard gate before trace)
# ---------------------------------------------------------------------------
ROLES = ("optic", "source", "detector", "ignored")
SURFACE_TYPES = ("plane", "sphere", "cylinder", "cone", "torus", "asphere",
                 "mesh")
SCHEMA_VERSIONS = (1, 2)   # v2 adds polarization/polarizer/filter/
                           # crystal_axis/per-face maps/asphere

class ContractError(ValueError):
    pass

def _req(d, key, typ, ctx):
    if key not in d:
        raise ContractError("%s: missing required key %r" % (ctx, key))
    v = d[key]
    if typ is float:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ContractError("%s: key %r must be a number (got %r)"
                                % (ctx, key, type(v).__name__))
        return float(v)
    if not isinstance(v, typ):
        raise ContractError("%s: key %r must be %s (got %r)"
                            % (ctx, key, typ.__name__, type(v).__name__))
    return v

def _check_axis(v, key, ctx):
    if (not isinstance(v, list) or len(v) != 3
            or not all(isinstance(x, (int, float)) for x in v)):
        raise ContractError("%s: %r must be a list of 3 numbers" % (ctx, key))
    n = math.sqrt(sum(x * x for x in v))
    if abs(n - 1.0) > 1e-6:
        raise ContractError(
            "%s: %r must be a unit vector (extractor normalizes) — got "
            "|v|=%g" % (ctx, key, n))

def _check_facemap(v, key, ctx, value_check=None):
    """v2 per-face maps: {face_id_or_'__all__': value-string}."""
    if not isinstance(v, dict) or not v:
        raise ContractError("%s: %r must be a non-empty dict" % (ctx, key))
    for k, val in v.items():
        if k != FACEMAP_ALL:
            parse_face_spec(k)
        if not isinstance(val, str) or not val.strip():
            raise ContractError("%s: %s[%r] must be a non-empty string"
                                % (ctx, key, k))
        if value_check is not None:
            try:
                value_check(val)
            except ValueError as e:
                raise ContractError("%s: %s[%r]: %s" % (ctx, key, k, e))

def validate_model(model):
    """Validate a loaded model.json dict. Raises ContractError. Returns model."""
    ctx = "model.json"
    ver = _req(model, "schema_version", int, ctx)
    if ver not in SCHEMA_VERSIONS:
        raise ContractError("unsupported schema_version %r"
                            % model["schema_version"])
    _req(model, "source_fcstd", str, ctx)
    _req(model, "spreadsheet", dict, ctx)
    _req(model, "ambient_material", str, ctx)
    bodies = _req(model, "bodies", list, ctx)
    if not bodies:
        raise ContractError("model has no bodies")
    n_sources = n_detectors = 0
    seen_face_ids = set()
    for b in bodies:
        bctx = "body %r" % b.get("name", "<unnamed>")
        _req(b, "name", str, bctx)
        _req(b, "label", str, bctx)
        role = _req(b, "role", str, bctx)
        if role not in ROLES:
            raise ContractError("%s: bad role %r (must be one of %s)"
                                % (bctx, role, ", ".join(ROLES)))
        if role == "ignored":
            continue
        mirror = float(b.get("mirror", 0.0))
        absorb = float(b.get("absorbance", 0.0))
        if not (0.0 <= mirror <= 1.0) or not (0.0 <= absorb <= 1.0):
            raise ContractError(
                "%s: mirror/absorbance out of [0,1] after extract-time "
                "capping — extractor bug" % bctx)
        if role == "source":
            n_sources += 1
            src = _req(b, "source", dict, bctx)
            _req(src, "power_mW", float, bctx + ".source")
            _req(src, "lambdac_nm", float, bctx + ".source")
            _req(src, "emit_face", str, bctx + ".source")
            if not isinstance(src.get("coherent", False), bool):
                raise ContractError("%s: source.coherent must be bool" % bctx)
            pol = src.get("polarization")
            if pol is not None:
                if (not isinstance(pol, dict)
                        or pol.get("kind") not in POLARIZATION_KINDS):
                    raise ContractError(
                        "%s: source.polarization must be a dict with kind "
                        "in %s" % (bctx, ", ".join(POLARIZATION_KINDS)))
        if role == "detector":
            n_detectors += 1
            det = _req(b, "detector", dict, bctx)
            _req(det, "face", str, bctx + ".detector")
        if role == "optic":
            _req(b, "material", str, bctx)
        # ---- v2 optional per-body optical properties ----
        if b.get("polarizer") is not None:
            _req(b, "polarizer", str, bctx)
        if b.get("polarizer_axis") is not None:
            _check_axis(b["polarizer_axis"], "polarizer_axis", bctx)
        if b.get("filter") is not None:
            _req(b, "filter", str, bctx)
        if b.get("crystal_axis") is not None:
            _check_axis(b["crystal_axis"], "crystal_axis", bctx)
        if isinstance(b.get("coating"), dict):
            _check_facemap(b["coating"], "coating", bctx)
        if b.get("roughness_faces") is not None:
            _check_facemap(b["roughness_faces"], "roughness_faces", bctx,
                           value_check=parse_rough_value)
        if b.get("grating") is not None:
            _check_facemap(b["grating"], "grating", bctx,
                           value_check=parse_grating_value)
            if FACEMAP_ALL in b["grating"]:
                raise ContractError(
                    "%s: grating property must name specific faces "
                    "(FaceN=...), not apply to every face" % bctx)
        faces = _req(b, "faces", list, bctx)
        if role in ("optic", "detector") and not faces:
            raise ContractError("%s: role %s but no faces" % (bctx, role))
        for f in faces:
            fctx = "%s face %r" % (bctx, f.get("id", "<no id>"))
            fid = _req(f, "id", str, fctx)
            if fid in seen_face_ids:
                raise ContractError("%s: duplicate face id" % fctx)
            seen_face_ids.add(fid)
            parse_face_spec(fid)   # syntax check
            surf = _req(f, "surface", dict, fctx)
            stype = _req(surf, "type", str, fctx)
            if stype not in SURFACE_TYPES:
                raise ContractError("%s: bad surface type %r" % (fctx, stype))
            _req(f, "orientation_outward", bool, fctx)
            _req(f, "area_m2", float, fctx)
            _req(f, "fingerprint", dict, fctx)
            if stype != "mesh":
                _check_surface_params(stype, surf, fctx)
                _check_trim_polylines(f, fctx)
            _req(f, "mesh_stl", str, fctx)
    if n_sources == 0:
        raise ContractError("model has no light sources "
                            "(need power+lambdac properties on a body)")
    if n_detectors == 0:
        raise ContractError("model has no detectors (material=detector)")
    val = model.get("validation", {})
    if val.get("overlapping_solids"):
        raise ContractError("overlapping solids: %r"
                            % val["overlapping_solids"])
    return model

_SURFACE_REQ = {
    "plane":    (("origin", 3), ("normal", 3)),
    "sphere":   (("center", 3), ("radius", 0)),
    "cylinder": (("origin", 3), ("axis", 3), ("radius", 0)),
    "cone":     (("apex", 3), ("axis", 3), ("half_angle", 0)),
    "torus":    (("center", 3), ("axis", 3), ("major_r", 0), ("minor_r", 0)),
    # asphere: sag(r) = r^2/(R(1+sqrt(1-(1+k) r^2/R^2))) + sum coeffs[i]*r^(4+2i)
    # R is the SIGNED paraxial radius [m]; coeffs are even-order terms A4..
    # in SI (m^(1-order)); r_max trims the polynomial's validity disc.
    "asphere":  (("vertex", 3), ("axis", 3), ("R", 0), ("k", 0),
                 ("coeffs", -1), ("r_max", 0)),
}

def _check_trim_polylines(face, ctx):
    """Analytic faces must carry 3D trim wires (outer first) so the tracer
    can containment-test hits — a lens face is a trimmed spherical cap."""
    polys = face.get("trim_polylines_xyz")
    if not isinstance(polys, list) or not polys:
        raise ContractError(
            "%s: analytic face missing trim_polylines_xyz (list of 3D "
            "polylines, outer wire first)" % ctx)
    for i, poly in enumerate(polys):
        if not isinstance(poly, list) or len(poly) < 3:
            raise ContractError("%s: trim polyline %d needs >= 3 points"
                                % (ctx, i))
        for pt in poly:
            if (not isinstance(pt, list) or len(pt) != 3
                    or not all(isinstance(x, (int, float)) for x in pt)):
                raise ContractError("%s: trim polyline %d has a non-3D point"
                                    % (ctx, i))

def _check_surface_params(stype, surf, ctx):
    for key, dim in _SURFACE_REQ[stype]:
        if key not in surf:
            raise ContractError("%s: %s surface missing %r" % (ctx, stype, key))
        v = surf[key]
        if dim == 0:
            if not isinstance(v, (int, float)) or v != v:
                raise ContractError("%s: %r must be a number" % (ctx, key))
            if key in ("radius", "major_r", "minor_r", "r_max") and v <= 0:
                raise ContractError("%s: %r must be > 0" % (ctx, key))
            if key == "R" and v == 0:
                raise ContractError("%s: asphere R must be nonzero (signed)"
                                    % ctx)
        elif dim == -1:
            # variable-length list of numbers (asphere polynomial coeffs)
            if (not isinstance(v, list)
                    or not all(isinstance(x, (int, float)) for x in v)):
                raise ContractError("%s: %r must be a list of numbers"
                                    % (ctx, key))
        else:
            if (not isinstance(v, list) or len(v) != dim
                    or not all(isinstance(x, (int, float)) for x in v)):
                raise ContractError("%s: %r must be a list of %d numbers"
                                    % (ctx, key, dim))

def load_model(path):
    with open(path) as fh:
        return validate_model(json.load(fh))

# ---------------------------------------------------------------------------
# Runtime / memory estimation (calibrated like the antenna pipeline)
# ---------------------------------------------------------------------------
FALLBACK_TRACE_RAYS_PER_S = 2e5      # primary rays/s through the full loop
FALLBACK_GATHER_OPS_PER_S = {"torch": 2e10, "numpy": 1e9}

def calibrated_rate(kind, fallback):
    """Median of recorded rates for `kind` in .calibration.json, else fallback."""
    try:
        with open(CALIBRATION_JSON) as fh:
            entries = [e["rate"] for e in json.load(fh)
                       if e.get("kind") == kind and e.get("rate", 0) > 0]
        if entries:
            entries.sort()
            return entries[len(entries) // 2]
    except (OSError, ValueError, KeyError):
        pass
    return fallback

def record_calibration(kind, rate, meta=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CALIBRATION_JSON) as fh:
            entries = json.load(fh)
    except (OSError, ValueError):
        entries = []
    entries.append({"kind": kind, "rate": rate, "meta": meta or {}})
    tmp = str(CALIBRATION_JSON) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=1)
    os.replace(tmp, CALIBRATION_JSON)

def estimate(rays, resolution, nlambda, n_coherent_sources, backend,
             n_detectors=1, save_fields=False, n_pol_strata=1):
    """Return dict of runtime/memory/disk estimates for --dry-run output.

    n_pol_strata: 2 when any source is unpolarized (two mutually-incoherent
    polarization populations), 1 for explicitly polarized sources.
    """
    npix = resolution * resolution
    # gather samples ~ surviving rays; assume half the primaries survive
    nsamples = max(1.0, rays * 0.5)
    gather_ops = (nsamples * npix * max(1, n_coherent_sources) * nlambda
                  * max(1, n_pol_strata))
    backend_key = "torch" if backend in ("auto", "torch") else "numpy"
    trace_s = rays / calibrated_rate("trace", FALLBACK_TRACE_RAYS_PER_S)
    gather_s = gather_ops / calibrated_rate(
        "gather_" + backend_key, FALLBACK_GATHER_OPS_PER_S[backend_key])
    acc_bytes = (npix * 2 * 8 * max(1, n_coherent_sources) * nlambda
                 * max(1, n_pol_strata))
    field_bytes = acc_bytes * n_detectors if save_fields else 0
    return {
        "trace_s": trace_s,
        "gather_s": gather_s,
        "total_s": trace_s + gather_s,
        "accumulator_GB": acc_bytes / 1e9,
        "fields_h5_GB": field_bytes / 1e9,
        "gather_ops": gather_ops,
        "backend": backend_key,
    }

def fmt_duration(s):
    if s < 90:
        return "%.0f s" % s
    if s < 5400:
        return "%.1f min" % (s / 60)
    return "%.1f h" % (s / 3600)

# ---------------------------------------------------------------------------
# Case bookkeeping
# ---------------------------------------------------------------------------
def case_dir(model_stem, case):
    return RESULTS_DIR / model_stem / case

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)

def read_case_status(case_json_path):
    try:
        with open(case_json_path) as fh:
            return json.load(fh).get("status", "unknown")
    except (OSError, ValueError):
        return "missing"

# ---------------------------------------------------------------------------
# Self-check:  python3 scripts/common.py
# ---------------------------------------------------------------------------
def _selfcheck():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "ok" if cond else "FAIL"
        print("  [%s] %s %s" % (status, name, detail))
        ok = ok and bool(cond)

    print("common.py self-check")
    check("FreeCAD AppImage", os.path.exists(FREECAD_APPIMAGE),
          FREECAD_APPIMAGE)
    check("optics env python", os.path.exists(OPTICS_PYTHON), OPTICS_PYTHON)
    check("pvpython", os.path.exists(PVPYTHON), PVPYTHON)
    check("project dir", PROJECT_DIR.is_dir(), str(PROJECT_DIR))
    check("materials.csv", MATERIALS_CSV.exists(), str(MATERIALS_CSV))

    # pure-math invariants
    check("sweep n=0", sweep_values(1, 5, 0) == [1])
    check("sweep n=1", sweep_values(1, 5, 1) == [1, 5])
    check("sweep n=4", sweep_values(0, 4, 4) == [0, 1, 2, 3, 4])
    check("variant name", variant_name("m", "x", -2.5) == "m-xm2p5")
    f = parse_face_spec("Body001.Pad.Face2")
    check("face spec", f["body"] == "Body001" and f["face_index"] == 2)
    g = parse_grating_spec("Body.Obj.Face3:600:v:orders=-1..1:eff=0.1,0.8,0.1")
    check("grating spec", g["lines_per_mm"] == 600.0
          and g["orders"] == (-1, 1) and len(g["efficiencies"]) == 3)
    g2 = parse_grating_spec("Body.Obj.Face3:@vbg_1800:orders=0..1")
    check("grating registry spec", g2["registry"] == "vbg_1800"
          and g2["orders"] == (0, 1))
    r = parse_rough_spec("Body.Obj.Face1:50:lcorr=5")
    check("rough spec", r["sigma_nm"] == 50.0 and r["lcorr_um"] == 5.0)
    fm = parse_facemap_spec("Face3=MgF2;Body.Pad.Face5=pbs_visible_45",
                            body="Body", feature="Pad")
    check("facemap per-face", fm == {"Body.Pad.Face3": "MgF2",
                                     "Body.Pad.Face5": "pbs_visible_45"})
    fm2 = parse_facemap_spec("200:lcorr=5")
    check("facemap all-faces", fm2 == {FACEMAP_ALL: "200:lcorr=5"})
    for bad_fm in ("Face3=", "MgF2;Face3=x", "Face3=a;Face3=b"):
        try:
            parse_facemap_spec(bad_fm, body="B", feature="F")
            check("reject facemap %r" % bad_fm, False)
        except ValueError:
            check("reject facemap %r" % bad_fm, True)
    pol = parse_polarization_spec("linear:30")
    check("polarization linear", pol == {"kind": "linear",
                                         "angle_deg": 30.0})
    pol2 = parse_polarization_spec("circular:left")
    check("polarization circular", pol2["handedness"] == "left")
    pol3 = parse_polarization_spec("elliptical:30:15")
    check("polarization elliptical", pol3["psi_deg"] == 30.0
          and pol3["chi_deg"] == 15.0)
    for bad_pol in ("linear", "circular:up", "elliptical:0:60", "stokes"):
        try:
            parse_polarization_spec(bad_pol)
            check("reject polarization %r" % bad_pol, False)
        except ValueError:
            check("reject polarization %r" % bad_pol, True)
    ax = parse_axis_spec("0,0,2")
    check("axis spec", abs(ax[2] - 1.0) < 1e-12 and ax[0] == 0.0)
    check("optprops dirs", COATINGS_CSV.exists() and NK_DATA_DIR.is_dir(),
          str(OPTPROPS_DIR))
    p = parse_particles_spec("material=water;phi=1e-4;median_um=10")
    check("particles default box",
          abs(p["box_size_m"][0] - 0.010) < 1e-12
          and abs(p["box_size_m"][1] - 0.020) < 1e-12)
    for bad in ("Body.Face2", "Body..Face2", "Body.Pad.Face"):
        try:
            parse_face_spec(bad)
            check("reject %r" % bad, False)
        except ValueError:
            check("reject %r" % bad, True)
    est = estimate(1e5, 512, 5, 1, "auto")
    check("estimator sane", est["total_s"] > 0 and est["accumulator_GB"] > 0,
          "total=%s" % fmt_duration(est["total_s"]))
    print("SELF-CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(_selfcheck())
