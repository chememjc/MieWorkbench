#!/usr/bin/env python3
# =============================================================================
# extract_geometry.py — FreeCAD headless geometry extraction for the optical
# ray tracer pipeline.
#
# Reads  : an .FCStd model (bodies tagged per the "Base" group property
#          convention below) — default: PROJECT_DIR/*.FCStd + BASEMODELS_DIR/
#          *.FCStd, or an explicit --models list.
# Writes : geometry/<stem>/model.json  (validated against common.load_model())
#          geometry/<stem>/faces/<face_id>.stl  (binary STL, metres, per face)
#
# Run with the FreeCAD AppImage (its bundled Python has the FreeCAD modules):
#
#   /home3/freecad/FreeCAD.AppImage -c scripts/extract_geometry.py -- \
#       --models example.FCStd [--outdir geometry] [--strict] < /dev/null
#
# NOTE on sys.argv under the FreeCAD AppImage console (-c) mode: argv[0] is
# the FreeCAD launcher binary, argv[1:] is ["-c", "<this script's path>",
# "--", ...your args...]. We locate the bare "--" ourselves rather than
# assuming a fixed index (FreeCAD's own arg parser aborts if "--" is absent
# and our flags look like unknown FreeCAD options).
#
# NOTE on double execution: this AppImage build's -c mode runs the whole
# script TWICE per invocation (silent headless pass, then a GUI-spinup pass)
# before exiting 0. All work here is idempotent — no wall-clock timestamps
# anywhere in the JSON, deterministic tessellation for identical inputs — so
# reruns (and the doubled in-process run) produce byte-identical output.
# Doubled log lines and ~2x wall time are expected, not a bug.
#
# Body tagging convention (group "Base" custom properties):
#   material   (string) - row name in materials.csv; "detector" -> detector;
#                          missing/"none" -> ignored; else -> optic.
#   power (mW), lambdac (nm) (floats) - presence of BOTH marks a source body
#                          (sources typically carry no material property).
#   lambdamin, lambdamax (nm, optional), coherent (bool, default False).
#   coating (string, optional -> per-face map, "none" -> omitted),
#   mirror/absorbance (float, optional, clamped to [0,1] with a loud
#   warning), roughness (float RMS nm -> legacy whole-body, OR string
#   per-face map "FaceN=sigma_nm[:lcorr=um]") may appear on any non-ignored
#   body.
#
# Schema v2 (schema_version=2, always emitted) additionally supports, group
# "Base":
#   polarization (string, source bodies only) - common.parse_polarization_spec.
#   polarizer (string, registry name, optics only) + polarizer_axis (string
#   'x,y,z' BODY-LOCAL, rotated to global via Placement; default local +z).
#   crystal_axis (string 'x,y,z' BODY-LOCAL, rotated to global) - ALWAYS
#   emitted for every optic body (default local +x) since the tracer's own
#   local-+x default is ambiguous without the body's placement.
#   filter (string, registry name, optics only).
#   grating (string, per-face map "FaceN=600:v;...", optics only, must name
#   faces -- no "apply to every face" form).
#   surface_override (string, per-face map) - currently supports
#   'FaceN=asphere:R=<mm>;k=<float>;A4=<mm^-3>;...;r_max=<mm>' to declare an
#   analytic asphere on a revolved face in place of the mesh/canonicalized
#   fallback; verified against the actual FreeCAD geometry to 1 um before
#   being trusted (see build_asphere_surface()).
#
# Units: FreeCAD's native unit is mm; every length in model.json is SI
# metres. Wavelengths are nm, power is mW (matching the property units used
# in the FCStd itself), angles are radians.
#
# trim_polylines_xyz contract (per coordinator update, supersedes an earlier
# UV-space plan): OCC's per-face UV frame is not part of the analytic-surface
# contract (e.g. a plane's local x/y axes aren't recoverable from just
# {origin, normal}), so trim boundaries are exported as 3D polylines in
# metres instead of UV coordinates. One polyline per wire, outer wire first,
# discretized at 0.05 mm chord deflection, edges concatenated in wire order,
# closed curves represented without repeating the first point.
# =============================================================================
import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import FreeCAD
import Part  # noqa: F401  (surface type checks touch Part.* indirectly)
import MeshPart

# scripts/ dir on sys.path so "import common" works when run via the
# FreeCAD AppImage (which does not automatically add the script's own
# directory the way plain `python3 scripts/foo.py` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Overwriting output on every rerun is expected (idempotent writes); disable
# FreeCAD's "keep a .FCBak of the file I'm replacing" preference so batch
# runs don't accumulate backup clutter, and so opening the model read-only
# here never leaves stray backups next to the master.
FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
    "CreateBackupFiles", False)

FACE_PROBE_EPS_MM = 0.01        # outward/inward probe offset
FACE_PROBE_TOL_MM = 1e-6        # isInside tolerance
MESH_LINEAR_DEFLECTION_MM = 0.03    # SKILL.md "proven default", in mm
MESH_ANGULAR_DEFLECTION_DEG = 15.0
TRIM_DEFLECTION_MM = 0.05
AREA_TOL_REL = 0.01             # 1% analytic-vs-mesh area tripwire
CANON_SAMPLES = 12              # SurfaceOfRevolution canonicalization samples
CANON_REL_TOL = 1e-9            # "equidistant" tolerance for canonicalization
ASPHERE_VERIFY_GRID = 15        # ~15x14 (u,v) grid ~= 200 samples
ASPHERE_TOL_M = 1e-6            # 1 micron sag-declaration tolerance


def log(msg):
    # print() alone has been observed to buffer/drop under the AppImage
    # console in some invocations; use both PrintMessage and print.
    FreeCAD.Console.PrintMessage(msg + "\n")
    print(msg, flush=True)


def warn(msg, warnings_list=None):
    FreeCAD.Console.PrintWarning("WARNING: " + msg + "\n")
    print("WARNING: " + msg, flush=True)
    if warnings_list is not None:
        warnings_list.append(msg)


def die(msg):
    FreeCAD.Console.PrintError("ERROR: " + msg + "\n")
    print("ERROR: " + msg, flush=True)
    os._exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []

    p = argparse.ArgumentParser(
        prog="extract_geometry.py",
        description="Extract SI-metre geometry + per-face STLs from FreeCAD models.")
    p.add_argument("--models", nargs="+", default=None,
                    help="explicit .FCStd files (bare names resolved under "
                         "the project root, then basemodels/)")
    p.add_argument("--outdir", default=None,
                    help="output directory (default: %s)" % common.GEOMETRY_DIR)
    p.add_argument("--strict", action="store_true",
                    help="hard-fail (instead of warn) on any face that falls "
                         "back to tessellation-only ('mesh') representation")
    try:
        return p.parse_args(rest)
    except SystemExit:
        # argparse's own sys.exit() is swallowed under FreeCAD -c; force it.
        os._exit(2)


def resolve_model_path(name):
    p = Path(name)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cand = common.PROJECT_DIR / p.name
    if cand.exists():
        return cand
    cand2 = common.BASEMODELS_DIR / p.name
    if cand2.exists():
        return cand2
    # nothing matched; return the project-root candidate so the caller's
    # existence check produces an actionable "file not found" message.
    return cand


def collect_paths(args):
    if args.models:
        return [resolve_model_path(m) for m in args.models]
    paths = (sorted(common.PROJECT_DIR.glob("*.FCStd"))
             + sorted(common.BASEMODELS_DIR.glob("*.FCStd")))
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Small geometry helpers (mm FreeCAD values -> SI where noted)
# ---------------------------------------------------------------------------
def pt_m(v):
    """FreeCAD.Vector in mm -> [x,y,z] in metres."""
    return [v.x / 1000.0, v.y / 1000.0, v.z / 1000.0]


def unit_vec(v):
    length = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if length < 1e-15:
        return [0.0, 0.0, 1.0]
    return [v.x / length, v.y / length, v.z / length]


def dist_m(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) / 1000.0


def close3(a, b, tol=1e-9):
    return all(abs(a[i] - b[i]) < tol for i in range(3))


def rotated_local_axis(obj, local_xyz):
    """A body-LOCAL unit axis (list [x,y,z]) rotated into the GLOBAL frame
    by the body's placement rotation, re-normalized."""
    v = FreeCAD.Vector(*local_xyz)
    g = obj.Placement.Rotation.multVec(v)
    return unit_vec(g)


def str_prop_or_none(obj, name):
    """hasattr-probed string property -> stripped value, or None if the
    property is absent, blank, or the literal 'none' (case-insensitive) —
    the same convention used throughout this file (material/coating/...)."""
    if not hasattr(obj, name):
        return None
    v = getattr(obj, name)
    if v is None:
        return None
    v = str(v).strip()
    return None if (v == "" or v.lower() == "none") else v


# ---------------------------------------------------------------------------
# Per-face property maps (schema v2): common.parse_facemap_spec() splits its
# input on ';' assuming ';' ONLY ever separates distinct face entries (e.g.
# 'Face3=MgF2;Face5=x'). The asphere surface_override grammar legitimately
# embeds ';' INSIDE a single face's own value (e.g.
# 'Face7=asphere:R=25.0;k=-0.6;A4=1.2e-6;r_max=10'), which would otherwise
# be mis-split into bogus non-face "entries" ('k=-0.6', ...) and rejected.
# We cannot change common.py (shared across every interpreter stack), so we
# pre-protect ';' characters that do NOT introduce a new 'FaceN=' /
# 'Body.Feature.FaceN=' entry with a NUL placeholder, let
# common.parse_facemap_spec do its real ';'-split on the (now unambiguous)
# entry boundaries, then restore the placeholder in the resulting values.
# ---------------------------------------------------------------------------
_NEW_FACE_ENTRY_RE = re.compile(
    r"^\s*(Face\d+|[A-Za-z_]\w*\.[A-Za-z_]\w*\.Face\d+)\s*=")


def _protect_internal_semicolons(value):
    parts = value.split(";")
    out = [parts[0]]
    for part in parts[1:]:
        if _NEW_FACE_ENTRY_RE.match(part):
            out.append(";" + part)
        else:
            out.append("\x00" + part)
    return "".join(out)


def parse_facemap_value_safe(value, body, feature):
    """common.parse_facemap_spec(), safe for per-face values that embed
    ';' internally (see _protect_internal_semicolons above)."""
    protected = _protect_internal_semicolons(str(value).strip())
    result = common.parse_facemap_spec(protected, body=body, feature=feature)
    return {k: v.replace("\x00", ";") for k, v in result.items()}


# ---------------------------------------------------------------------------
# Asphere surface_override: 'asphere:R=<mm>;k=<float>;A4=<mm^-3>;A6=<mm^-5>;
# ...;r_max=<mm>' (r_max optional -> defaults from the actual face's radial
# extent). sag(r) = r^2/(R(1+sqrt(1-(1+k) r^2/R^2))) + sum coeffs[i]*r^(4+2i)
# — same convention documented in common.py's _SURFACE_REQ["asphere"].
#
# Units: FreeCAD property values are mm. R_m = R_mm*1e-3; r_max_m =
# r_max_mm*1e-3. Coefficient A_n (n=4,6,8,...) has FreeCAD-side units
# mm^(1-n) (dimensionally: sag_mm = A_n_mm * r_mm^n, mm = mm^(1-n) * mm^n).
# Converting to SI (sag_m = A_n_SI * r_m^n, both in metres):
#   sag_m = sag_mm * 1e-3 = A_n_mm * (r_m*1e3)^n * 1e-3
#         = A_n_mm * 1e^(3n-3) * r_m^n
#   =>  A_n_SI = A_n_mm * 10**(3*(n-1))
# Numeric check, n=4: A4_SI = A4_mm * 10**9. E.g. A4_mm=1.2e-6 mm^-3 ->
# A4_SI=1200 m^-3; sag at r=1mm=1e-3 m: A4 term = 1200*(1e-3)^4 = 1.2e-9 m,
# i.e. 1.2e-6 mm — matches A4_mm*r_mm^4 = 1.2e-6*1 = 1.2e-6 mm. Consistent.
# ---------------------------------------------------------------------------
_ASPHERE_COEFF_RE = re.compile(r"^A(\d+)$")


def asphere_sag_m(r_m, R_m, k, coeffs_si):
    """sag(r) in metres, or None if the conic term is imaginary (r beyond
    the surface's physical validity disc: 1-(1+k)c^2r^2 <= 0)."""
    c = 1.0 / R_m
    if r_m == 0.0:
        conic = 0.0
    else:
        disc = 1.0 - (1.0 + k) * c * c * r_m * r_m
        if disc <= 0.0:
            return None
        conic = c * r_m * r_m / (1.0 + math.sqrt(disc))
    poly = 0.0
    n = 4
    for a in coeffs_si:
        poly += a * (r_m ** n)
        n += 2
    return conic + poly


def parse_asphere_override_value(raw):
    """'asphere:R=25.0;k=-0.6;A4=1.2e-6;r_max=10' (mm units) -> dict with
    R_mm, k, coeffs_mm (contiguous A4,A6,... in order), r_max_mm (or None
    if not given)."""
    raw = raw.strip()
    prefix, sep, rest = raw.partition(":")
    if prefix.strip().lower() != "asphere" or not sep:
        raise ValueError(
            "surface_override value %r is not an 'asphere:...' spec" % raw)
    fields = {}
    for tok in rest.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        k, s2, v = tok.partition("=")
        if not s2:
            raise ValueError("bad asphere field %r in %r" % (tok, raw))
        fields[k.strip()] = v.strip()
    if "R" not in fields:
        raise ValueError("asphere spec %r missing R=" % raw)
    R_mm = float(fields.pop("R"))
    if R_mm == 0.0:
        raise ValueError("asphere R must be nonzero in %r" % raw)
    k_val = float(fields.pop("k", "0.0"))
    r_max_mm = float(fields.pop("r_max")) if "r_max" in fields else None
    coeff_items = []
    for key, v in fields.items():
        m = _ASPHERE_COEFF_RE.match(key)
        if not m:
            raise ValueError("unknown asphere field %r in %r" % (key, raw))
        order = int(m.group(1))
        if order < 4 or order % 2 != 0:
            raise ValueError(
                "asphere coeff order must be an even integer >= 4, got "
                "A%d in %r" % (order, raw))
        coeff_items.append((order, float(v)))
    coeff_items.sort()
    expected = 4
    for order, _ in coeff_items:
        if order != expected:
            raise ValueError(
                "asphere coeffs must be contiguous even orders starting at "
                "A4 (gap before A%d) in %r" % (order, raw))
        expected += 2
    return {"R_mm": R_mm, "k": k_val,
            "coeffs_mm": [v for _, v in coeff_items], "r_max_mm": r_max_mm}


def build_asphere_surface(face, face_id, override_val, warnings):
    """Verify + emit an analytic asphere surface dict for a face declared
    via surface_override. Dies (never silently corrupts phase) if the
    declaration doesn't match the actual FreeCAD geometry within 1 um."""
    spec = parse_asphere_override_value(override_val)
    surf = face.Surface
    tname = type(surf).__name__
    axis_pt = getattr(surf, "Location", None)
    if axis_pt is None:
        axis_pt = getattr(surf, "Position", None)
    axis_dir = getattr(surf, "Direction", None)
    if axis_dir is None:
        axis_dir = getattr(surf, "Axis", None)
    if axis_pt is None or axis_dir is None:
        die("%s: surface_override=asphere requires an axis-symmetric "
            "(revolved) face to recover a vertex/axis; got surface type %r "
            "with no recoverable axis attributes" % (face_id, tname))
    axis_dir = FreeCAD.Vector(axis_dir)
    L = axis_dir.Length
    if L < 1e-12:
        die("%s: surface_override=asphere: degenerate (zero-length) axis "
            "direction" % face_id)
    axis_dir = FreeCAD.Vector(axis_dir.x / L, axis_dir.y / L, axis_dir.z / L)

    # Locate the vertex: sample the meridian (fixed u, sweep v) and take the
    # sample nearest the axis (r ~ 0) as the r=0 point; project it onto the
    # axis to get a well-defined vertex point.
    u0, u1, v0, v1 = face.ParameterRange
    u_fixed = (u0 + u1) / 2.0
    n_scan = 200
    meridian = []   # (r_mm, axial_mm) relative to axis_pt
    for i in range(n_scan):
        v = v0 + (v1 - v0) * i / (n_scan - 1)
        try:
            p = face.Surface.value(u_fixed, v)
        except Exception:
            continue
        w = p - axis_pt
        axial = axis_dir.dot(w)
        foot = axis_pt + axis_dir * axial
        r_mm = (p - foot).Length
        meridian.append((r_mm, axial))
    if not meridian:
        die("%s: surface_override=asphere: could not sample the face's "
            "meridian" % face_id)
    best_r, best_axial = min(meridian, key=lambda s: s[0])
    if best_r > 0.05:   # mm; the profile is expected to touch the axis
        die("%s: surface_override=asphere: could not locate a vertex on "
            "the revolution axis (closest sample r=%.6g mm)"
            % (face_id, best_r))
    vertex_pt = axis_pt + axis_dir * best_axial

    # Sign convention: "sag opens along +axis" -> near-vertex samples must
    # have z = (p-vertex).axis >= 0; flip axis_dir if that's violated.
    near = sorted(meridian, key=lambda s: s[0])[:10]
    signed_z = [a - best_axial for r, a in near if r > 1e-6]
    if signed_z and (sum(signed_z) / len(signed_z)) < 0.0:
        axis_dir = axis_dir * -1.0

    R_m = spec["R_mm"] * 1e-3
    k = spec["k"]
    coeffs_si = [a * (10.0 ** (3 * (n - 1)))
                for n, a in zip(range(4, 4 + 2 * len(spec["coeffs_mm"]), 2),
                                spec["coeffs_mm"])]

    # ---- verify against a (u,v) grid of ~200 actual samples ----
    samples = []   # (r_m, z_m)
    nu = ASPHERE_VERIFY_GRID
    nv = ASPHERE_VERIFY_GRID - 1
    for iu in range(nu):
        uu = u0 + (u1 - u0) * (iu / (nu - 1))
        for iv in range(nv):
            vv = v0 + (v1 - v0) * (iv / (nv - 1))
            try:
                p = face.Surface.value(uu, vv)
            except Exception:
                continue
            w = p - vertex_pt
            z_mm = axis_dir.dot(w)
            foot = w - axis_dir * z_mm
            r_mm = foot.Length
            samples.append((r_mm / 1000.0, z_mm / 1000.0))
    if not samples:
        die("%s: surface_override=asphere: verification grid produced no "
            "samples" % face_id)

    r_max_m = (spec["r_max_mm"] * 1e-3 if spec["r_max_mm"] is not None
              else max(r for r, z in samples))
    if r_max_m <= 0.0:
        die("%s: surface_override=asphere: r_max must be > 0 (got %.6g m)"
            % (face_id, r_max_m))

    max_resid = -1.0
    first_bad = None
    for r_m, z_m in samples:
        if r_m > r_max_m * 1.001:
            continue
        sag_m = asphere_sag_m(r_m, R_m, k, coeffs_si)
        if sag_m is None:
            die("%s: surface_override=asphere declaration is unphysical at "
                "r=%.6g m (1-(1+k)*c^2*r^2 <= 0 for the declared R=%.6g m, "
                "k=%.6g)" % (face_id, r_m, R_m, k))
        resid = abs(z_m - sag_m)
        if resid > max_resid:
            max_resid = resid
        if resid >= ASPHERE_TOL_M and first_bad is None:
            first_bad = (r_m, z_m, sag_m, resid)
    if first_bad is not None:
        r_m, z_m, sag_m, resid = first_bad
        die("%s: surface_override=asphere declaration does not match the "
            "actual face geometry (max residual %.3g um over %d samples, "
            "tolerance %.1g um); first bad sample: r=%.6g m z_actual=%.6g m "
            "sag_declared=%.6g m residual=%.3g um"
            % (face_id, max_resid * 1e6, len(samples), ASPHERE_TOL_M * 1e6,
               r_m, z_m, sag_m, resid * 1e6))

    log("%s: surface_override=asphere verified OK (R=%.6g m k=%.6g "
        "coeffs=%r r_max=%.6g m, max residual %.3g um over %d samples)"
        % (face_id, R_m, k, coeffs_si, r_max_m, max_resid * 1e6,
           len(samples)))
    return {"type": "asphere", "vertex": pt_m(vertex_pt),
            "axis": unit_vec(axis_dir), "R": R_m, "k": k,
            "coeffs": coeffs_si, "r_max": r_max_m}


# ---------------------------------------------------------------------------
# Surface classification (incl. SurfaceOfRevolution canonicalization)
# ---------------------------------------------------------------------------
def canonicalize_revolution(face, surf, face_id, warnings):
    """Sample a SurfaceOfRevolution's meridian and try to recognize it as a
    native sphere or cylinder (OCC sometimes fails to recognize these
    natively, e.g. when the profile comes from a spline-approximated arc).
    Returns a surface dict or None (caller falls back to 'mesh')."""
    try:
        axis_pt = getattr(surf, "Location", None) or getattr(surf, "Position", None)
        axis_dir = getattr(surf, "Direction", None) or getattr(surf, "Axis", None)
        if axis_pt is None or axis_dir is None:
            warn("%s: cannot introspect SurfaceOfRevolution axis attributes"
                 % face_id, warnings)
            return None
        axis_dir = FreeCAD.Vector(axis_dir)
        L = axis_dir.Length
        if L < 1e-12:
            return None
        axis_dir = FreeCAD.Vector(axis_dir.x / L, axis_dir.y / L, axis_dir.z / L)

        u0, u1, v0, v1 = face.ParameterRange
        u_fixed = (u0 + u1) / 2.0
        n = CANON_SAMPLES
        pts = []
        for i in range(n):
            v = v0 + (v1 - v0) * i / (n - 1)
            try:
                pts.append(face.Surface.value(u_fixed, v))
            except Exception:
                continue
        if len(pts) < 3:
            return None

        def axial(p):
            w = p - axis_pt
            return axis_dir.dot(w)

        def perp_dist(p):
            w = p - axis_pt
            proj = axis_dir.dot(w)
            foot = axis_pt + axis_dir * proj
            return (p - foot).Length, foot

        perp = [perp_dist(p) for p in pts]
        dists = [d for d, _ in perp]
        davg = sum(dists) / len(dists)
        if davg > 1e-9 and all(abs(d - davg) <= CANON_REL_TOL * davg for d in dists):
            origin_pt = perp[0][1]
            return {"type": "cylinder", "origin": pt_m(origin_pt),
                    "axis": unit_vec(axis_dir), "radius": davg / 1000.0}

        # sphere candidate: solve for t (center = axis_pt + t*axis_dir) using
        # the two samples with the largest axial separation, then verify.
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                da = abs(axial(pts[i]) - axial(pts[j]))
                if best is None or da > best[0]:
                    best = (da, i, j)
        if best is None or best[0] < 1e-9:
            return None
        _, i, j = best
        pi, pj = pts[i], pts[j]
        wi, wj = pi - axis_pt, pj - axis_pt
        ai, aj = axis_dir.dot(wi), axis_dir.dot(wj)
        di2, dj2 = wi.Length ** 2, wj.Length ** 2
        denom = 2.0 * (aj - ai)
        if abs(denom) < 1e-9:
            return None
        t = (dj2 - di2) / denom
        center = axis_pt + axis_dir * t
        dists2 = [(p - center).Length for p in pts]
        ravg = sum(dists2) / len(dists2)
        if ravg > 1e-9 and all(abs(d - ravg) <= CANON_REL_TOL * ravg for d in dists2):
            return {"type": "sphere", "center": pt_m(center), "radius": ravg / 1000.0}
        return None
    except Exception as exc:
        warn("%s: SurfaceOfRevolution canonicalization raised %s" % (face_id, exc),
             warnings)
        return None


def classify_surface(face, face_id, warnings, strict):
    surf = face.Surface
    tname = type(surf).__name__

    if tname == "Plane":
        return {"type": "plane", "origin": pt_m(surf.Position),
                "normal": unit_vec(surf.Axis)}
    if tname == "Sphere":
        return {"type": "sphere", "center": pt_m(surf.Center),
                "radius": surf.Radius / 1000.0}
    if tname == "Cylinder":
        return {"type": "cylinder", "origin": pt_m(surf.Center),
                "axis": unit_vec(surf.Axis), "radius": surf.Radius / 1000.0}
    if tname == "Cone":
        return {"type": "cone", "apex": pt_m(surf.Apex),
                "axis": unit_vec(surf.Axis), "half_angle": float(surf.SemiAngle)}
    if tname in ("Toroid", "Torus"):
        return {"type": "torus", "center": pt_m(surf.Center),
                "axis": unit_vec(surf.Axis),
                "major_r": surf.MajorRadius / 1000.0,
                "minor_r": surf.MinorRadius / 1000.0}
    if tname == "SurfaceOfRevolution":
        canon = canonicalize_revolution(face, surf, face_id, warnings)
        if canon is not None:
            return canon
        msg = ("%s: SurfaceOfRevolution could not be canonicalized to a "
               "native sphere/cylinder; falling back to mesh representation "
               "(phase accuracy on this face is tessellation-limited)" % face_id)
        if strict:
            die(msg)
        warn(msg, warnings)
        return {"type": "mesh"}

    msg = ("%s: unsupported analytic surface type %r; falling back to mesh "
           "representation (phase accuracy on this face is "
           "tessellation-limited)" % (face_id, tname))
    if strict:
        die(msg)
    warn(msg, warnings)
    return {"type": "mesh"}


def canonical_normal_mm(surf, pt):
    """The normal the TRACER will compute from the contract's analytic
    params, evaluated at point pt (mm frame; surf params are metres).
    Mirrors scripts/raytracer/surfaces.py normal() conventions exactly:
    plane -> stored normal; sphere -> outward from center; cylinder/cone
    -> away from the axis (cone tilted by the half-angle); torus ->
    outward from the spine circle. Returns None for mesh faces."""
    import math
    t = surf["type"]
    p = FreeCAD.Vector(pt.x / 1000.0, pt.y / 1000.0, pt.z / 1000.0)
    if t == "plane":
        n = FreeCAD.Vector(*surf["normal"])
    elif t == "sphere":
        n = p - FreeCAD.Vector(*surf["center"])
    elif t == "cylinder":
        o = FreeCAD.Vector(*surf["origin"])
        a = FreeCAD.Vector(*surf["axis"])
        rel = p - o
        n = rel - a * rel.dot(a)
    elif t == "cone":
        apex = FreeCAD.Vector(*surf["apex"])
        a = FreeCAD.Vector(*surf["axis"])
        rel = p - apex
        h = rel.dot(a)
        radial = rel - a * h
        if radial.Length < 1e-15:
            return None
        rhat = radial.normalize()
        ha = surf["half_angle"]
        n = rhat * math.cos(ha) - a * math.sin(ha)
    elif t == "torus":
        c = FreeCAD.Vector(*surf["center"])
        a = FreeCAD.Vector(*surf["axis"])
        rel = p - c
        radial = rel - a * rel.dot(a)
        if radial.Length < 1e-15:
            return None
        ring = c + radial.normalize() * surf["major_r"]
        n = p - ring
    elif t == "asphere":
        # Implicit surface F(r,z) = z - sag(r) = 0 in the (radial, axial)
        # meridian plane (z measured from the vertex along +axis); its
        # gradient (pointing toward increasing F, i.e. the +axis side) is
        # (-sag'(r) in the radial direction, +1 along axis) — standard
        # optical-design asphere-normal convention. sag'(r) for the conic
        # term is c*r/sqrt(1-(1+k)c^2r^2) (closed form of the derivative of
        # r^2/(R(1+sqrt(1-(1+k)r^2/R^2)))); polynomial terms differentiate
        # termwise (d/dr[A_n r^n] = n A_n r^(n-1)).
        vtx = FreeCAD.Vector(*surf["vertex"])
        a = FreeCAD.Vector(*surf["axis"])
        rel = p - vtx
        z = a.dot(rel)
        radial = rel - a * z
        r = radial.Length
        if r < 1e-15:
            n = a
        else:
            rhat = radial.normalize()
            R = surf["R"]
            k = surf["k"]
            c = 1.0 / R
            disc = 1.0 - (1.0 + k) * c * c * r * r
            if disc <= 0.0:
                return None
            dsag = c * r / math.sqrt(disc)
            order = 4
            for coef in surf["coeffs"]:
                dsag += order * coef * (r ** (order - 1))
                order += 2
            n = a - rhat * dsag
    else:
        return None
    if n.Length < 1e-15:
        return None
    return n.normalize()


def orientation_probe(shape, face, face_id, surf, warnings):
    """Return (orientation_outward: bool, normal_hint: FreeCAD.Vector).

    CONTRACT SEMANTICS: orientation_outward states whether the CANONICAL
    normal (the one the tracer derives from the stored analytic params —
    NOT FreeCAD's orientation-corrected normalAt()) points out of the
    solid. Probing with normalAt() and storing raw geometric params is
    exactly the sign bug that killed every ray on authored pads (plane
    axis antiparallel to the oriented normal)."""
    u0, u1, v0, v1 = face.ParameterRange
    um, vm = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    try:
        pt = face.Surface.value(um, vm)
        nrm_hint = face.normalAt(um, vm)
    except Exception:
        com = face.CenterOfMass
        pt = com
        u, v = face.Surface.parameter(com)
        nrm_hint = face.normalAt(u, v)
    L = nrm_hint.Length
    if L > 1e-15:
        nrm_hint = FreeCAD.Vector(nrm_hint.x / L, nrm_hint.y / L,
                                  nrm_hint.z / L)

    n_probe = canonical_normal_mm(surf, pt)
    if n_probe is None:
        n_probe = nrm_hint       # mesh faces: probe the oriented normal

    eps = FACE_PROBE_EPS_MM
    p_out = pt + n_probe * eps
    p_in = pt - n_probe * eps
    tol = FACE_PROBE_TOL_MM
    out_is_inside = shape.isInside(p_out, tol, False)
    in_is_inside = shape.isInside(p_in, tol, False)

    if out_is_inside == in_is_inside:
        # thin solids / grazing probes: fall back to comparing the
        # canonical normal with the oriented normalAt (which FreeCAD
        # guarantees points out of a solid's outer shell)
        agree = n_probe.dot(nrm_hint) >= 0.0
        warn("%s: orientation probe ambiguous (both +/-eps isInside=%s); "
             "falling back to canonical-vs-normalAt comparison (-> %s)"
             % (face_id, out_is_inside, agree), warnings)
        probe_outward = agree
    else:
        probe_outward = (not out_is_inside)

    return probe_outward, nrm_hint


def trim_polylines_xyz(face, face_id, warnings):
    """One closed polyline per wire (outer wire first, then holes), points
    in SI metres, discretized at TRIM_DEFLECTION_MM chord deflection, edges
    concatenated in wire order, first point not repeated at the end."""
    wires = list(face.Wires)
    if not wires:
        return []
    try:
        outer = face.OuterWire
        ordered = [outer] + [w for w in wires if not w.isSame(outer)]
    except Exception:
        warn("%s: could not determine OuterWire; using Wires[] order as-is"
             % face_id, warnings)
        ordered = wires

    polylines = []
    for w in ordered:
        pts = []
        edges = w.OrderedEdges if hasattr(w, "OrderedEdges") else w.Edges
        chains = []
        for e in edges:
            try:
                epts = e.discretize(Deflection=TRIM_DEFLECTION_MM)
            except Exception:
                epts = [e.Vertexes[0].Point, e.Vertexes[-1].Point]
            chains.append([[q.x / 1000.0, q.y / 1000.0, q.z / 1000.0]
                           for q in epts])
        # Orient each edge chain head-to-tail before concatenating:
        # OrderedEdges orders the EDGES but does NOT flip the point
        # sequence of reversed edges, so multi-edge planar wires (every
        # pad rectangle/triangle) used to emit SELF-CROSSING loops -- the
        # even-odd trim containment test then rejected ~half of each such
        # face (dead half-faces: phantom transmission and seam-leak
        # kills; found via the BS-cube investigation, and the cause of
        # the wollaston scene's documented detected-power anomaly).
        for i, ch in enumerate(chains):
            if not pts:
                if len(chains) > 1:
                    n2 = chains[1]
                    if (close3(ch[0], n2[0]) or close3(ch[0], n2[-1])) and \
                       not (close3(ch[-1], n2[0])
                            or close3(ch[-1], n2[-1])):
                        ch = ch[::-1]
            else:
                if close3(pts[-1], ch[-1]) and not close3(pts[-1], ch[0]):
                    ch = ch[::-1]
            for pm in ch:
                if pts and close3(pts[-1], pm):
                    continue
                pts.append(pm)
        if len(pts) > 1 and close3(pts[0], pts[-1]):
            pts.pop()
        if len(pts) >= 3:
            polylines.append(pts)
        else:
            warn("%s: a trim wire discretized to only %d point(s); dropping"
                 % (face_id, len(pts)), warnings)
    return polylines


# ---------------------------------------------------------------------------
# STL export (hand-rolled binary writer -> fully deterministic, no
# FreeCAD-version-dependent header/timestamp noise, byte-identical reruns)
# ---------------------------------------------------------------------------
def write_binary_stl(path, mesh):
    facets = mesh.Facets
    header = b"opticalraytracer extract_geometry.py STL export".ljust(80, b"\x00")[:80]
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(facets)))
        for facet in facets:
            n = facet.Normal
            fh.write(struct.pack("<3f", float(n.x), float(n.y), float(n.z)))
            for p in facet.Points:
                fh.write(struct.pack("<3f", float(p[0]), float(p[1]), float(p[2])))
            fh.write(struct.pack("<H", 0))


def mesh_and_write_stl(face, out_path, face_id, analytic_area_m2, surf_type, warnings):
    """Scale a COPY of the face to metres, tessellate, write binary STL,
    return mesh_area_m2. Hard-fails (os._exit) if an analytic face's mesh
    area deviates >1% from its analytic area (tripwire)."""
    face_copy = face.copy()
    mat = FreeCAD.Matrix()
    mat.scale(0.001, 0.001, 0.001)
    face_copy = face_copy.transformGeometry(mat)

    # Small-radius faces (a fiber's 0.1 mm bore) are MARGINAL against the
    # area tripwire at the default deflections, and OCC's mesher is
    # chaotically sensitive — an epsilon placement change can flip a face
    # from 0.3% to 2.7% deficit. Retrying at progressively finer
    # deflection is honest (the tripwire still gates the FINAL mesh) and
    # removes the placement-noise lottery. Analytic faces trace
    # analytically regardless; the mesh feeds viz + the BVH fallback.
    mesh = None
    mesh_area_m2 = None
    first_err = None
    check = (surf_type != "mesh" and analytic_area_m2 is not None
             and analytic_area_m2 > 0)
    for attempt, scale in enumerate((1.0, 0.25, 0.0625)):
        mesh = MeshPart.meshFromShape(
            Shape=face_copy,
            LinearDeflection=MESH_LINEAR_DEFLECTION_MM * scale / 1000.0,
            AngularDeflection=math.radians(
                MESH_ANGULAR_DEFLECTION_DEG * max(scale, 0.25)))
        mesh_area_m2 = mesh.Area
        if not check:
            break
        rel_err = abs(mesh_area_m2 - analytic_area_m2) / analytic_area_m2
        if first_err is None:
            first_err = rel_err
        if rel_err < AREA_TOL_REL:
            if attempt:
                warn("%s: met the area tripwire only at %gx deflection "
                     "(%.2f%% at default, %.2f%% final)"
                     % (face_id, scale, first_err * 100.0,
                        rel_err * 100.0), warnings)
            break
    else:
        die("%s: mesh_area_m2=%.9g deviates %.2f%% from analytic "
            "area_m2=%.9g (tripwire is %.0f%%, even at 1/16 deflection)"
            % (face_id, mesh_area_m2,
               abs(mesh_area_m2 - analytic_area_m2) / analytic_area_m2
               * 100.0, analytic_area_m2, AREA_TOL_REL * 100.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(str(out_path), mesh)
    return mesh_area_m2


# ---------------------------------------------------------------------------
# Body classification / tagging
# ---------------------------------------------------------------------------
def classify_body(body):
    # miewb_exclude: set by the GUI on unfolded fold mirrors (and any
    # user-excluded element) — the body stays in the document (ghosted in
    # the GUI) but is invisible to the traced physics
    if getattr(body, "miewb_exclude", False):
        return "ignored"
    if hasattr(body, "power") and hasattr(body, "lambdac"):
        return "source"
    material = getattr(body, "material", None)
    if material is None or str(material).strip().lower() in ("", "none"):
        return "ignored"
    if str(material).strip().lower() == "detector":
        return "detector"
    return "optic"


def capped01(name, value, body_label, warnings):
    v = float(value)
    if v > 1.0:
        warn("%s: %s=%.6g > 1, capping to 1.0" % (body_label, name, v), warnings)
        v = 1.0
    elif v < 0.0:
        warn("%s: %s=%.6g < 0, capping to 0.0" % (body_label, name, v), warnings)
        v = 0.0
    return v


def coating_value(body):
    return str_prop_or_none(body, "coating")


# ---------------------------------------------------------------------------
# Spreadsheet echo
# ---------------------------------------------------------------------------
def sheet_echo(sheet, warnings):
    echo = {}
    for cell in sheet.getUsedCells():
        try:
            alias = sheet.getAlias(cell)
        except Exception:
            alias = None
        if not alias:
            continue
        raw = sheet.getContents(cell)
        v = sheet.get(alias)
        try:
            si = float(FreeCAD.Units.Quantity(v).getValueAs("m"))
        except Exception:
            try:
                si = float(v)
            except Exception as exc:
                warn("spreadsheet alias %r: could not coerce value %r to float (%s)"
                     % (alias, v, exc), warnings)
                continue
        echo[alias] = {"raw": raw, "si": si}
    return echo


# ---------------------------------------------------------------------------
# Per-body face walk
# ---------------------------------------------------------------------------
def extract_faces(body, shape, tip_name, out_dir, strict, warnings,
                  surface_override=None):
    faces_out = []
    closest_face_id = None
    closest_dist = None
    origin = FreeCAD.Vector(0, 0, 0)

    for idx, face in enumerate(shape.Faces, start=1):
        face_id = "%s.%s.Face%d" % (body.Name, tip_name, idx)

        override_val = None
        if surface_override:
            override_val = surface_override.get(
                face_id, surface_override.get(common.FACEMAP_ALL))
        if override_val is not None and override_val.strip().lower().startswith("asphere:"):
            surf = build_asphere_surface(face, face_id, override_val, warnings)
        else:
            if override_val is not None:
                warn("%s: surface_override value %r not recognized "
                     "(expected 'asphere:...'); ignoring, using "
                     "auto-detected surface type" % (face_id, override_val),
                     warnings)
            surf = classify_surface(face, face_id, warnings, strict)
        outward, nrm = orientation_probe(shape, face, face_id, surf,
                                         warnings)
        area_m2 = face.Area / 1e6
        centroid_m = pt_m(face.CenterOfMass)
        fingerprint = {"surface_type": surf["type"], "area_m2": area_m2,
                        "centroid": centroid_m, "normal_hint": [nrm.x, nrm.y, nrm.z]}
        u0, u1, v0, v1 = face.ParameterRange
        polylines = trim_polylines_xyz(face, face_id, warnings)

        stl_rel = "faces/%s.stl" % face_id
        stl_abs = out_dir / stl_rel
        analytic_area_for_check = area_m2 if surf["type"] != "mesh" else None
        mesh_area_m2 = mesh_and_write_stl(
            face, stl_abs, face_id, analytic_area_for_check, surf["type"], warnings)

        face_dict = {
            "id": face_id,
            "surface": surf,
            "orientation_outward": outward,
            "area_m2": area_m2,
            "fingerprint": fingerprint,
            "uv_bounds": [u0, u1, v0, v1],
            "trim_polylines_xyz": polylines,
            "mesh_stl": stl_rel,
            "mesh_area_m2": mesh_area_m2,
        }
        faces_out.append(face_dict)

        d = dist_m(face.CenterOfMass, origin)
        if closest_dist is None or d < closest_dist:
            closest_dist = d
            closest_face_id = face_id

    return faces_out, closest_face_id


# ---------------------------------------------------------------------------
# One FCStd -> one geometry/<stem>/model.json
# ---------------------------------------------------------------------------
def extract_one(fcstd_path, outdir, strict):
    fcstd_path = Path(fcstd_path).resolve()
    if not fcstd_path.exists():
        die("file not found: %s" % fcstd_path)

    stem = fcstd_path.stem
    out_dir = outdir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    doc = FreeCAD.openDocument(str(fcstd_path))
    try:
        doc.recompute()

        sheets = [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]
        # Primary sheet ('dim' by convention, else the first) echoes FLAT
        # (back-compat: permute_model / sweeps address bare aliases).
        # Every sheet ALSO echoes namespaced as '<sheet label>.<alias>' so
        # per-element parameter sheets (dim_<element>, from MieWorkbench
        # primitives) are addressable without collisions.
        spreadsheet = {}
        if sheets:
            primary = next((s for s in sheets if s.Label == "dim"),
                           sheets[0])
            spreadsheet.update(sheet_echo(primary, warnings))
            for sheet in sheets:
                for alias, rec in sheet_echo(sheet, warnings).items():
                    spreadsheet["%s.%s" % (sheet.Label, alias)] = rec
        else:
            warn("%s: no Spreadsheet::Sheet object found" % stem, warnings)

        bodies_solids = []   # (body, shape, role) for overlap checking
        bodies_out = []
        n_sources = n_detectors = 0

        for obj in doc.Objects:
            if obj.TypeId != "PartDesign::Body":
                continue
            role = classify_body(obj)
            if role == "ignored":
                log("%s is ignored" % obj.Label)
                continue

            shape = obj.Shape
            tip_name = obj.Tip.Name if obj.Tip else obj.Name

            # surface_override must be parsed before extract_faces() since
            # it steers per-face surface classification (asphere override).
            surf_override_raw = str_prop_or_none(obj, "surface_override")
            surf_override_map = {}
            if surf_override_raw is not None:
                try:
                    surf_override_map = parse_facemap_value_safe(
                        surf_override_raw, obj.Name, tip_name)
                except ValueError as e:
                    die("%s: bad surface_override spec %r: %s"
                        % (obj.Label, surf_override_raw, e))

            faces, closest_face_id = extract_faces(
                obj, shape, tip_name, out_dir, strict, warnings,
                surf_override_map)

            bbox = shape.BoundBox
            body_dict = {
                "name": obj.Name,
                "label": obj.Label,
                "role": role,
                "faces": faces,
                "volume_m3": shape.Volume / 1e9,
                "solid_closed": bool(shape.isClosed()),
                "bbox_m": {
                    "min": [bbox.XMin / 1000.0, bbox.YMin / 1000.0, bbox.ZMin / 1000.0],
                    "max": [bbox.XMax / 1000.0, bbox.YMax / 1000.0, bbox.ZMax / 1000.0],
                },
            }

            if hasattr(obj, "mirror"):
                body_dict["mirror"] = capped01("mirror", obj.mirror, obj.Label, warnings)
            if hasattr(obj, "absorbance"):
                body_dict["absorbance"] = capped01(
                    "absorbance", obj.absorbance, obj.Label, warnings)

            # roughness: legacy App::PropertyFloat (whole-body RMS nm) OR
            # (schema v2) App::PropertyString per-face map
            # 'Face1=200:lcorr=5;Face2=80'.
            if hasattr(obj, "roughness"):
                rv = obj.roughness
                if isinstance(rv, str):
                    raw = rv.strip()
                    if raw and raw.lower() != "none":
                        try:
                            rmap = parse_facemap_value_safe(
                                raw, obj.Name, tip_name)
                            for fk, fv in rmap.items():
                                common.parse_rough_value(fv)
                        except ValueError as e:
                            die("%s: bad roughness spec %r: %s"
                                % (obj.Label, raw, e))
                        body_dict["roughness_faces"] = rmap
                else:
                    body_dict["roughness_nm"] = float(rv)

            # diffuser: per-face map (or whole-body) of ground-glass specs
            # 'grit:120' | 'slope:0.08' | '@dg_600' (common.
            # parse_diffuser_value grammar; deep-rough scatter at trace).
            diffuser_raw = str_prop_or_none(obj, "diffuser")
            if diffuser_raw is not None:
                try:
                    dmap = parse_facemap_value_safe(
                        diffuser_raw, obj.Name, tip_name)
                    for fk, fv in dmap.items():
                        common.parse_diffuser_value(fv)
                except ValueError as e:
                    die("%s: bad diffuser spec %r: %s"
                        % (obj.Label, diffuser_raw, e))
                body_dict["diffuser_faces"] = dmap

            # scatter: per-face map (or whole-body) of ABg/BSDF registry
            # names 'name' | 'FaceN=polished_bk7_glass;...' (validated
            # against opticalproperties/scatter/ at scene build; measured
            # reflected-side scatter at trace). Names only, no value grammar
            # (like coating).
            scatter_raw = str_prop_or_none(obj, "scatter")
            if scatter_raw is not None:
                try:
                    smap = parse_facemap_value_safe(
                        scatter_raw, obj.Name, tip_name)
                except ValueError as e:
                    die("%s: bad scatter spec %r: %s"
                        % (obj.Label, scatter_raw, e))
                body_dict["scatter_faces"] = smap

            # coating (schema v2): per-face map, {'__all__': name} for the
            # legacy "whole body, one coating" form.
            coating_raw = coating_value(obj)
            if coating_raw is not None:
                try:
                    body_dict["coating"] = parse_facemap_value_safe(
                        coating_raw, obj.Name, tip_name)
                except ValueError as e:
                    die("%s: bad coating spec %r: %s"
                        % (obj.Label, coating_raw, e))

            # ---- schema v2 optics-only properties ----
            polarizer_raw = str_prop_or_none(obj, "polarizer")
            if polarizer_raw is not None:
                if role != "optic":
                    warn("%s: polarizer is only meaningful on optic bodies "
                         "(role=%s); ignoring" % (obj.Label, role), warnings)
                else:
                    body_dict["polarizer"] = polarizer_raw
                    axis_raw = str_prop_or_none(obj, "polarizer_axis")
                    try:
                        local = (common.parse_axis_spec(axis_raw)
                                if axis_raw is not None else [0.0, 0.0, 1.0])
                    except ValueError as e:
                        die("%s: bad polarizer_axis spec %r: %s"
                            % (obj.Label, axis_raw, e))
                    body_dict["polarizer_axis"] = rotated_local_axis(obj, local)

            filter_raw = str_prop_or_none(obj, "filter")
            if filter_raw is not None:
                if role != "optic":
                    warn("%s: filter is only meaningful on optic bodies "
                         "(role=%s); ignoring" % (obj.Label, role), warnings)
                else:
                    body_dict["filter"] = filter_raw

            crystal_axis_raw = str_prop_or_none(obj, "crystal_axis")
            if role == "optic":
                # ALWAYS emitted for optics (tracer default is local +x, but
                # local frame is unknown at trace time) so every optic gets
                # an unambiguous global crystal_axis.
                try:
                    local = (common.parse_axis_spec(crystal_axis_raw)
                            if crystal_axis_raw is not None else [1.0, 0.0, 0.0])
                except ValueError as e:
                    die("%s: bad crystal_axis spec %r: %s"
                        % (obj.Label, crystal_axis_raw, e))
                body_dict["crystal_axis"] = rotated_local_axis(obj, local)
            elif crystal_axis_raw is not None:
                warn("%s: crystal_axis is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)

            # biaxial crystals need a full principal frame: crystal_axis is
            # the X principal axis, crystal_axis2 the Y axis (Z = X x Y;
            # orthogonalization happens tracer-side). Emitted only when
            # authored — the scene loader errors if a biaxial material
            # lacks it.
            axis2_raw = str_prop_or_none(obj, "crystal_axis2")
            if axis2_raw is not None:
                if role != "optic":
                    warn("%s: crystal_axis2 is only meaningful on optic "
                         "bodies (role=%s); ignoring" % (obj.Label, role),
                         warnings)
                else:
                    try:
                        local2 = common.parse_axis_spec(axis2_raw)
                    except ValueError as e:
                        die("%s: bad crystal_axis2 spec %r: %s"
                            % (obj.Label, axis2_raw, e))
                    body_dict["crystal_axis2"] = rotated_local_axis(
                        obj, local2)

            grating_raw = str_prop_or_none(obj, "grating")
            if grating_raw is not None:
                if role != "optic":
                    warn("%s: grating is only meaningful on optic bodies "
                         "(role=%s); ignoring" % (obj.Label, role), warnings)
                else:
                    try:
                        gmap = parse_facemap_value_safe(
                            grating_raw, obj.Name, tip_name)
                        if common.FACEMAP_ALL in gmap:
                            raise ValueError(
                                "grating property must name specific faces "
                                "(FaceN=...), not apply to every face")
                        for fk, fv in gmap.items():
                            common.parse_grating_value(fv)
                    except ValueError as e:
                        die("%s: bad grating spec %r: %s"
                            % (obj.Label, grating_raw, e))
                    body_dict["grating"] = gmap

            if surf_override_raw is not None:
                body_dict["surface_override_raw"] = surf_override_raw

            if role == "source":
                n_sources += 1
                power_mw = float(obj.power)
                lambdac_nm = float(obj.lambdac)
                lambdamin = float(obj.lambdamin) if hasattr(obj, "lambdamin") else None
                lambdamax = float(obj.lambdamax) if hasattr(obj, "lambdamax") else None
                coherent = bool(obj.coherent) if hasattr(obj, "coherent") else False
                source_dict = {
                    "power_mW": power_mw,
                    "lambdac_nm": lambdac_nm,
                    "lambdamin_nm": lambdamin,
                    "lambdamax_nm": lambdamax,
                    "coherent": coherent,
                    "emit_face": closest_face_id,
                    "emit_face_autodetected": True,
                }
                pol_raw = str_prop_or_none(obj, "polarization")
                if pol_raw is not None:
                    try:
                        source_dict["polarization"] = \
                            common.parse_polarization_spec(pol_raw)
                    except ValueError as e:
                        die("%s: bad polarization spec %r: %s"
                            % (obj.Label, pol_raw, e))
                apod_raw = str_prop_or_none(obj, "apodization")
                if apod_raw is not None:
                    try:
                        source_dict["apodization"] = \
                            common.parse_apodization_spec(apod_raw)
                    except ValueError as e:
                        die("%s: bad apodization spec %r: %s"
                            % (obj.Label, apod_raw, e))
                if hasattr(obj, "beam_waist"):
                    waist_mm = float(obj.beam_waist)
                    if waist_mm <= 0:
                        die("%s: beam_waist must be > 0 mm (got %g)"
                            % (obj.Label, waist_mm))
                    m2 = float(obj.m2) if hasattr(obj, "m2") else 1.0
                    if m2 < 1.0:
                        die("%s: m2 must be >= 1.0 (got %g)"
                            % (obj.Label, m2))
                    source_dict["beam"] = {"waist_mm": waist_mm, "m2": m2}
                body_dict["source"] = source_dict
            elif role == "detector":
                n_detectors += 1
                det_dict = {
                    "face": closest_face_id,
                    "autodetected": True,
                }
                qe_curve = str_prop_or_none(obj, "qe_curve")
                if qe_curve is not None:
                    det_dict["qe_curve"] = qe_curve
                body_dict["detector"] = det_dict
            elif role == "optic":
                body_dict["material"] = str(obj.material)

            if role != "source" and str_prop_or_none(obj, "polarization") is not None:
                warn("%s: polarization property is only meaningful on "
                     "source bodies (role=%s); ignoring"
                     % (obj.Label, role), warnings)

            if role != "source" and str_prop_or_none(obj, "apodization") is not None:
                warn("%s: apodization property is only meaningful on "
                     "source bodies (role=%s); ignoring"
                     % (obj.Label, role), warnings)

            if role != "source" and hasattr(obj, "beam_waist"):
                warn("%s: beam_waist property is only meaningful on "
                     "source bodies (role=%s); ignoring"
                     % (obj.Label, role), warnings)

            bodies_out.append(body_dict)
            bodies_solids.append((obj.Name, shape))

        if not bodies_out:
            die("%s: no non-ignored bodies found" % stem)
        if n_sources == 0:
            die("%s: no light sources found (need power+lambdac properties "
                "on a body)" % stem)
        if n_detectors == 0:
            die("%s: no detectors found (material=detector)" % stem)

        overlaps = []
        nested = []
        for i in range(len(bodies_solids)):
            ni, si = bodies_solids[i]
            bi = si.BoundBox
            for j in range(i + 1, len(bodies_solids)):
                nj, sj = bodies_solids[j]
                bj = sj.BoundBox
                if (bi.XMax < bj.XMin or bi.XMin > bj.XMax
                        or bi.YMax < bj.YMin or bi.YMin > bj.YMax
                        or bi.ZMax < bj.ZMin or bi.ZMin > bj.ZMax):
                    continue   # bboxes disjoint -> solids can't overlap
                common_vol = si.common(sj).Volume
                if common_vol > 1e-12:
                    # PROPER NESTING (one solid strictly inside another) is
                    # supported by the tracer's LIFO medium stack and is how
                    # the beamsplitter cubes model their coated internal
                    # interface (a nested thin plate: glass-glass, no gap,
                    # no TIR); only PARTIAL overlap is non-manifold and
                    # rejected.
                    vi, vj = si.Volume, sj.Volume
                    inner = min(vi, vj)
                    if abs(common_vol - inner) <= 1e-6 * inner:
                        nested.append({"outer": ni if vi > vj else nj,
                                       "inner": nj if vi > vj else ni,
                                       "volume_mm3": common_vol})
                    else:
                        overlaps.append({"a": ni, "b": nj,
                                         "volume_mm3": common_vol})

        model = {
            "schema_version": 2,
            "source_fcstd": str(fcstd_path),
            "extracted_note": "generated by extract_geometry.py",
            "units_note": "all lengths SI metres; wavelengths nm; power mW; angles radians",
            "ambient_material": "air",
            "spreadsheet": spreadsheet,
            "bodies": bodies_out,
            "validation": {
                "overlapping_solids": overlaps,
                "nested_solids": nested,
                "warnings": warnings,
            },
        }
    finally:
        FreeCAD.closeDocument(doc.Name)

    out_json = out_dir / "model.json"
    common.write_json(out_json, model)

    # Hard gate: round-trip the file we just wrote through the shared
    # contract validator before declaring success.
    with open(out_json) as fh:
        loaded = json.load(fh)
    try:
        common.validate_model(loaded)
    except common.ContractError as exc:
        die("%s: contract validation FAILED on written model.json: %s"
            % (out_json, exc))

    log("%-24s bodies=%d sources=%d detectors=%d overlaps=%d warnings=%d -> %s"
        % (stem, len(bodies_out), n_sources, n_detectors, len(overlaps),
           len(warnings), out_json))
    return model


def main():
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else common.GEOMETRY_DIR
    paths = collect_paths(args)
    if not paths:
        die("no .FCStd files found (checked %s and %s; use --models to be explicit)"
            % (common.PROJECT_DIR, common.BASEMODELS_DIR))

    log("extract_geometry.py: processing %d model(s) -> %s%s"
        % (len(paths), outdir, "  [--strict]" if args.strict else ""))

    for i, p in enumerate(paths):
        common.progress_emit("extract", i / len(paths), p.stem)
        extract_one(p, outdir, args.strict)

    common.progress_emit("extract", 1.0,
                         "%d model(s) extracted" % len(paths),
                         status="completed")
    log("GEOMETRY EXTRACTION OK (%d model(s))" % len(paths))


# NOTE: no `if __name__ == "__main__"` guard — FreeCAD's console mode (-c)
# executes scripts with __name__ set to the module's basename, not
# "__main__", which would silently skip main() if guarded.
main()
sys.exit(0)
