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
#   power (mW), lambdac (nm) (floats) - presence of lambdac PLUS EITHER
#                          power or pulse_energy marks a source body
#                          (sources typically carry no material property).
#   lambdamin, lambdamax (nm, optional), coherent (bool, default False).
#   pulse_energy (uJ)/pulse_duration (ps FWHM)/rep_rate (Hz) (floats,
#   optional, source bodies only) - pulsed-optics Phase P3: an alternative
#   (pulse_energy) or supplement (pulse_duration/rep_rate) to plain power;
#   raytracer.scene derives whichever of {power, pulse_energy} is absent
#   from the other + rep_rate (power XOR pulse_energy is enforced there).
#   spectrum (string, source bodies only, optional) - emission-registry row
#   naming a tabulated emission spectrum; supersedes lambdamin/lambdamax.
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
#   detector_face (string, detector bodies only, optional) - bare "FaceN"
#   (this body) or a full "Body.Tip.FaceN" id pinning the detector's PRIMARY
#   sensing face, overriding the closest-to-world-origin auto-pick. Replaces
#   the primary face in place (no extra screen), so the scene stays
#   C-engine-routable (unlike the additive CLI --detector-face).
#   grating (string, per-face map "FaceN=600:v;...", optics only, must name
#   faces -- no "apply to every face" form).
#   surface_override (string, per-face map) - currently supports
#   'FaceN=asphere:R=<mm>;k=<float>;A4=<mm^-3>;...;r_max=<mm>' to declare an
#   analytic asphere on a revolved face in place of the mesh/canonicalized
#   fallback; verified against the actual FreeCAD geometry to 1 um before
#   being trusted (see build_asphere_surface()). Also supports
#   'FaceN=qbfs:R=<mm>;k=<float>;A0=<mm>;A1=<mm>;...;r_max=<mm>' (or
#   'qcon:...') for an ISO 10110-12 Forbes Q-type asphere (base conic + an
#   orthonormal-in-slope/amplitude Q-polynomial departure); same 1 um
#   verification gate (see build_qforbes_surface(), raytracer.surfaces.
#   QForbes).
#
#   Pulsed-optics Phase P8 (optics only, group "Base"):
#   nonlinear (string, registry row name in opticalproperties/nonlinear/
#   nonlinear.mienlo) - kind=pockels (needs a birefringent material +
#   crystal_axis; TRANSVERSE geometry only in this phase) or kind=chi2_*
#   (accepted + warned, the SHG/parametric event is a later phase).
#   pockels_voltage (float V, default 0) + pockels_gap (float mm, > 0,
#   the transverse E=V/d electrode gap) - Pockels cell operating point.
#   saturable (string) - nonlinear.mienlo kind=saturable registry row name,
#   OR inline 'sat:I_sat=<W/cm2>:T0=<0..1>' (common.parse_saturable_value).
#   tpa_beta (float cm/GW) - two-photon-absorption coefficient.
#   kerr_n2 (string) - nonlinear.mienlo kind=n2 registry row name, OR
#   inline 'n2:<m2/W>' (common.parse_kerr_n2_value); Kerr thin-lens phase,
#   coherent sources only.
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
import functools
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
from raytracer import prescription as prescription_mod  # noqa: E402

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


class ExtractError(RuntimeError):
    """A fatal extraction error (die()).

    Raised instead of exiting the process so this module is importable as a
    library (scripts/fcserver/fcops.py extracts an already-open document
    in-place for the fast evaluator, and the persistent worker must survive
    a failed extraction). The CLI entry point at the bottom of this file
    catches it and turns it into the historical exit(1)."""


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
    raise ExtractError(msg)


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
    p.add_argument("--prescription", default=None,
                    help="prescription.json driving the prescription-primary "
                         "cross-check (engine3 Sec 3). When omitted a "
                         "sibling <stem>.prescription.json sidecar is used if "
                         "present; otherwise every body extracts as before.")
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
# QForbes surface_override: 'qbfs:R=<mm>;k=<float>;A0=<mm>;A1=<mm>;...;
# r_max=<mm>' or 'qcon:...' (same field grammar, kind selects the basis --
# ISO 10110-12 Forbes Q-type asphere, engine3.md Sec 7.6). r_max is
# optional (defaults from the actual face's radial extent), same as
# asphere. sag(r) = conic_sag(r; R, k) + sec(theta_c(r)) * envelope(u) *
# Sum_n coeffs[n] * Q_n(...), u = r/r_max -- see raytracer.surfaces.QForbes
# for the exact math (base conic + orthonormal-in-slope/amplitude Forbes
# departure); same convention documented in common.py's
# _SURFACE_REQ["qforbes"].
#
# Units: FreeCAD property values are mm. R_m = R_mm*1e-3; r_max_m =
# r_max_mm*1e-3, same as asphere. Unlike asphere's power-series A_n (which
# need a per-order m^(1-n) unit conversion), a Forbes coefficient a_n has
# the SAME units as the sag itself -- the Q_n/P_n bases are dimensionless
# functions of u = r/r_max -- so A_n_SI = A_n_mm * 1e-3 for EVERY n (no
# exponent scaling, unlike asphere).
# ---------------------------------------------------------------------------
_QFORBES_COEFF_RE = re.compile(r"^A(\d+)$")
_QFORBES_INV_SQRT19 = 1.0 / math.sqrt(19.0)


# ---- pure-stdlib (no numpy -- this module runs under FreeCAD's embedded
# python) Forbes Qbfs/Qcon sag-only recurrence, mirroring
# raytracer.surfaces._qbfs_weighted_sum/_qcon_weighted_sum and their
# _qbfs_departure/_qcon_departure envelope wrappers exactly (value only;
# the extractor's <1 um verifier never needs the derivative). ----
@functools.lru_cache(maxsize=None)
def _f_qbfs_scalar(n):
    if n == 0:
        return 2.0
    if n == 1:
        return math.sqrt(19.0) / 2.0
    term1 = n * (n + 1) + 3
    term2 = _g_qbfs_scalar(n - 1) ** 2
    term3 = _h_qbfs_scalar(n - 2) ** 2
    return math.sqrt(term1 - term2 - term3)


@functools.lru_cache(maxsize=None)
def _g_qbfs_scalar(n_minus_1):
    if n_minus_1 == 0:
        return -0.5
    n_minus_2 = n_minus_1 - 1
    return -(1.0 + _g_qbfs_scalar(n_minus_2) * _h_qbfs_scalar(n_minus_2)) \
        / _f_qbfs_scalar(n_minus_1)


@functools.lru_cache(maxsize=None)
def _h_qbfs_scalar(n_minus_2):
    n = n_minus_2 + 2
    return -n * (n - 1) / (2.0 * _f_qbfs_scalar(n_minus_2))


def _qbfs_sag_scalar(coeffs, u):
    x = u * u
    M = len(coeffs) - 1
    if M < 0:
        return 0.0
    R = coeffs[0]
    if M > 0:
        Q1 = _QFORBES_INV_SQRT19 * (13.0 - 16.0 * x)
        R += coeffs[1] * Q1
        if M > 1:
            P_prev, P_curr = 2.0, 6.0 - 8.0 * x
            Q_prev, Q_curr = 1.0, Q1
            lin = 2.0 - 4.0 * x
            for n in range(2, M + 1):
                Pn = lin * P_curr - P_prev
                g = _g_qbfs_scalar(n - 1)
                h = _h_qbfs_scalar(n - 2)
                Qn = (Pn - g * Q_curr - h * Q_prev) / _f_qbfs_scalar(n)
                R += coeffs[n] * Qn
                P_prev, P_curr = P_curr, Pn
                Q_prev, Q_curr = Q_curr, Qn
    return x * (1.0 - x) * R


@functools.lru_cache(maxsize=None)
def _jacobi04_abc_scalar(n):
    a, b = 0.0, 4.0
    s = a + b
    A = (2 * n + s + 1) * (2 * n + s + 2) / (2.0 * (n + 1) * (n + s + 1))
    B = ((a * a - b * b) * (2 * n + s + 1)
        / (2.0 * (n + 1) * (n + s + 1) * (2 * n + s)))
    C = ((n + a) * (n + b) * (2 * n + s + 2)
        / ((n + 1) * (n + s + 1) * (2 * n + s)))
    return A, B, C


def _qcon_sag_scalar(coeffs, u):
    x = 2.0 * u * u - 1.0
    M = len(coeffs) - 1
    if M < 0:
        return 0.0
    R = coeffs[0]
    if M > 0:
        P1 = 3.0 * x - 2.0
        R += coeffs[1] * P1
        if M > 1:
            P_prev, P_curr = 1.0, P1
            for n in range(2, M + 1):
                A, B, C = _jacobi04_abc_scalar(n - 1)
                lin = A * x + B
                Pn = lin * P_curr - C * P_prev
                R += coeffs[n] * Pn
                P_prev, P_curr = P_curr, Pn
    return (u ** 4) * R


def qforbes_sag_m(r_m, R_m, k, kind, coeffs_si, r_max_m):
    """sag(r) in metres, or None if the conic term is imaginary (r beyond
    the surface's physical validity disc)."""
    c = 1.0 / R_m
    beta = (1.0 + k) * c * c
    disc = 1.0 - beta * r_m * r_m
    if disc <= 0.0:
        return None
    phi = math.sqrt(disc)
    zc = c * r_m * r_m / (1.0 + phi)
    zc1 = c * r_m / phi
    sigma_inv = math.sqrt(1.0 + zc1 * zc1)
    u = r_m / r_max_m if r_max_m else 0.0
    dep = (_qbfs_sag_scalar(coeffs_si, u) if kind == "qbfs"
          else _qcon_sag_scalar(coeffs_si, u))
    return zc + sigma_inv * dep


def parse_qforbes_override_value(raw):
    """'qbfs:R=25.0;k=-0.6;A0=1.2e-3;A1=-4e-4;r_max=10' or 'qcon:...' (mm
    units) -> dict with kind, R_mm, k, coeffs_mm (contiguous A0,A1,...
    dense from 0, in order), r_max_mm (or None if not given)."""
    raw = raw.strip()
    prefix, sep, rest = raw.partition(":")
    kind = prefix.strip().lower()
    if kind not in ("qbfs", "qcon") or not sep:
        raise ValueError(
            "surface_override value %r is not a 'qbfs:...' or 'qcon:...' "
            "spec" % raw)
    fields = {}
    for tok in rest.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        fkey, s2, v = tok.partition("=")
        if not s2:
            raise ValueError("bad %s field %r in %r" % (kind, tok, raw))
        fields[fkey.strip()] = v.strip()
    if "R" not in fields:
        raise ValueError("%s spec %r missing R=" % (kind, raw))
    R_mm = float(fields.pop("R"))
    if R_mm == 0.0:
        raise ValueError("%s R must be nonzero in %r" % (kind, raw))
    k_val = float(fields.pop("k", "0.0"))
    r_max_mm = float(fields.pop("r_max")) if "r_max" in fields else None
    coeff_items = []
    for key, v in fields.items():
        m = _QFORBES_COEFF_RE.match(key)
        if not m:
            raise ValueError("unknown %s field %r in %r" % (kind, key, raw))
        order = int(m.group(1))
        if order < 0:
            raise ValueError("%s coeff order must be >= 0, got A%d in %r"
                             % (kind, order, raw))
        coeff_items.append((order, float(v)))
    coeff_items.sort()
    expected = 0
    for order, _ in coeff_items:
        if order != expected:
            raise ValueError(
                "%s coeffs must be contiguous orders starting at A0 (gap "
                "before A%d) in %r" % (kind, order, raw))
        expected += 1
    return {"kind": kind, "R_mm": R_mm, "k": k_val,
            "coeffs_mm": [v for _, v in coeff_items], "r_max_mm": r_max_mm}


def build_qforbes_surface(face, face_id, override_val, warnings):
    """Verify + emit an analytic Forbes Q-type surface dict for a face
    declared via surface_override. Dies (never silently corrupts phase) if
    the declaration doesn't match the actual FreeCAD geometry within 1 um.
    Structurally identical to build_asphere_surface (same vertex/axis
    recovery off a revolved face, same verification-grid + tolerance
    gate); only the sag formula (qforbes_sag_m vs asphere_sag_m) and the
    coefficient unit conversion differ."""
    spec = parse_qforbes_override_value(override_val)
    kind = spec["kind"]
    surf = face.Surface
    tname = type(surf).__name__
    axis_pt = getattr(surf, "Location", None)
    if axis_pt is None:
        axis_pt = getattr(surf, "Position", None)
    axis_dir = getattr(surf, "Direction", None)
    if axis_dir is None:
        axis_dir = getattr(surf, "Axis", None)
    if axis_pt is None or axis_dir is None:
        die("%s: surface_override=%s requires an axis-symmetric (revolved) "
            "face to recover a vertex/axis; got surface type %r with no "
            "recoverable axis attributes" % (face_id, kind, tname))
    axis_dir = FreeCAD.Vector(axis_dir)
    L = axis_dir.Length
    if L < 1e-12:
        die("%s: surface_override=%s: degenerate (zero-length) axis "
            "direction" % (face_id, kind))
    axis_dir = FreeCAD.Vector(axis_dir.x / L, axis_dir.y / L, axis_dir.z / L)

    u0, u1, v0, v1 = face.ParameterRange
    u_fixed = (u0 + u1) / 2.0
    n_scan = 200
    meridian = []
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
        die("%s: surface_override=%s: could not sample the face's "
            "meridian" % (face_id, kind))
    best_r, best_axial = min(meridian, key=lambda s: s[0])
    if best_r > 0.05:   # mm; the profile is expected to touch the axis
        die("%s: surface_override=%s: could not locate a vertex on the "
            "revolution axis (closest sample r=%.6g mm)"
            % (face_id, kind, best_r))
    vertex_pt = axis_pt + axis_dir * best_axial

    near = sorted(meridian, key=lambda s: s[0])[:10]
    signed_z = [a - best_axial for r, a in near if r > 1e-6]
    if signed_z and (sum(signed_z) / len(signed_z)) < 0.0:
        axis_dir = axis_dir * -1.0

    R_m = spec["R_mm"] * 1e-3
    k = spec["k"]
    coeffs_si = [a * 1e-3 for a in spec["coeffs_mm"]]

    samples = []
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
        die("%s: surface_override=%s: verification grid produced no "
            "samples" % (face_id, kind))

    r_max_m = (spec["r_max_mm"] * 1e-3 if spec["r_max_mm"] is not None
              else max(r for r, z in samples))
    if r_max_m <= 0.0:
        die("%s: surface_override=%s: r_max must be > 0 (got %.6g m)"
            % (face_id, kind, r_max_m))

    max_resid = -1.0
    first_bad = None
    for r_m, z_m in samples:
        if r_m > r_max_m * 1.001:
            continue
        sag_m = qforbes_sag_m(r_m, R_m, k, kind, coeffs_si, r_max_m)
        if sag_m is None:
            die("%s: surface_override=%s declaration is unphysical at "
                "r=%.6g m (1-(1+k)*c^2*r^2 <= 0 for the declared R=%.6g m, "
                "k=%.6g)" % (face_id, kind, r_m, R_m, k))
        resid = abs(z_m - sag_m)
        if resid > max_resid:
            max_resid = resid
        if resid >= ASPHERE_TOL_M and first_bad is None:
            first_bad = (r_m, z_m, sag_m, resid)
    if first_bad is not None:
        r_m, z_m, sag_m, resid = first_bad
        die("%s: surface_override=%s declaration does not match the "
            "actual face geometry (max residual %.3g um over %d samples, "
            "tolerance %.1g um); first bad sample: r=%.6g m z_actual=%.6g "
            "m sag_declared=%.6g m residual=%.3g um"
            % (face_id, kind, max_resid * 1e6, len(samples),
               ASPHERE_TOL_M * 1e6, r_m, z_m, sag_m, resid * 1e6))

    log("%s: surface_override=%s verified OK (R=%.6g m k=%.6g coeffs=%r "
        "r_max=%.6g m, max residual %.3g um over %d samples)"
        % (face_id, kind, R_m, k, coeffs_si, r_max_m, max_resid * 1e6,
           len(samples)))
    return {"type": "qforbes", "kind": kind, "vertex": pt_m(vertex_pt),
            "axis": unit_vec(axis_dir), "R": R_m, "k": k,
            "coeffs": coeffs_si, "r_max": r_max_m}


# ---------------------------------------------------------------------------
# Prescription cross-check (engine3.md Sec 3, P5): when a prescription entry
# exists for a body, its analytic optical surfaces are the TRUTH. For each
# FreeCAD face that a prescription surface matches, we (a) VERIFY the
# tessellated geometry against the prescription to the same 1 um gate the
# asphere-override verifier uses, and (b) EMIT the model.json surface FROM THE
# PRESCRIPTION (exact params, transformed to global through the body
# Placement -- never FreeCAD's canonicalize_revolution sampling). A mismatch
# > 1 um is a HARD ERROR (the CAD drifted from its prescription; we never
# silently prefer either). Bodies WITHOUT a prescription extract exactly as
# before (full backward compatibility).
#
# Policy by surface type (matches primitivelib.build_prescription_entry):
#   sphere, asphere : emitted-from-prescription (verified). asphere reuses the
#                     existing build_asphere_surface machinery (vertex/axis
#                     recovered from the placed geometry; R/k/coeffs from the
#                     prescription).
#   cylinder        : VERIFIED against the prescription, kept in native OCC
#                     form (already exact; origin along the axis is free).
# ---------------------------------------------------------------------------
PRESCRIPTION_TOL_M = ASPHERE_TOL_M          # 1 um, the same gate
_PRESC_IDENTIFY_TOL_M = 1e-4                 # 100 um: loose "is this the face?"
_PRESC_VERIFY_GRID = ASPHERE_VERIFY_GRID    # reuse the asphere grid density


def presc_key_for_body(obj):
    """The prescription element key for a FreeCAD body: its miewb_group
    (stable across rebuilds / multi-body elements), falling back to Label."""
    grp = getattr(obj, "miewb_group", None)
    if isinstance(grp, str) and grp:
        return grp
    return obj.Label


def _presc_point_to_global(placement, p_local_m):
    """LOCAL SI-metre point -> GLOBAL SI-metre, via the body Placement (the
    exact transform OCC applies to the geometry). Params are stored in mm
    internally, so scale m->mm, transform, scale back."""
    v = FreeCAD.Vector(p_local_m[0] * 1000.0, p_local_m[1] * 1000.0,
                       p_local_m[2] * 1000.0)
    g = placement.multVec(v)
    return [g.x / 1000.0, g.y / 1000.0, g.z / 1000.0]


def _presc_dir_to_global(placement, d_local):
    g = placement.Rotation.multVec(FreeCAD.Vector(*d_local))
    L = g.Length or 1.0
    return [g.x / L, g.y / L, g.z / L]


def _sample_face_points_m(face, grid):
    """A (grid x grid-1) set of surface points, in SI metres, from the face's
    parametric (u,v) range (the same sampling the asphere verifier uses)."""
    u0, u1, v0, v1 = face.ParameterRange
    pts = []
    nu, nv = grid, grid - 1
    for iu in range(nu):
        uu = u0 + (u1 - u0) * (iu / (nu - 1))
        for iv in range(nv):
            vv = v0 + (v1 - v0) * (iv / (nv - 1))
            try:
                p = face.Surface.value(uu, vv)
            except Exception:
                continue
            pts.append((p.x / 1000.0, p.y / 1000.0, p.z / 1000.0))
    return pts


def _sphere_residuals(pts, center, radius):
    out = []
    cx, cy, cz = center
    for x, y, z in pts:
        d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        out.append(abs(d - radius))
    return out


def _cylinder_residuals(pts, origin, axis, radius):
    ox, oy, oz = origin
    ax, ay, az = axis
    an = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / an, ay / an, az / an
    out = []
    for x, y, z in pts:
        wx, wy, wz = x - ox, y - oy, z - oz
        t = wx * ax + wy * ay + wz * az          # projection onto the axis
        px, py, pz = wx - t * ax, wy - t * ay, wz - t * az
        out.append(abs(math.sqrt(px * px + py * py + pz * pz) - radius))
    return out


def _presc_asphere_to_override_str(surf):
    """A prescription 'asphere' surface (SI) -> the mm-unit 'asphere:...'
    string build_asphere_surface() parses, so the existing verify+emit path
    is reused verbatim (vertex/axis recovered from the placed geometry;
    R/k/coeffs taken from the prescription)."""
    R_mm = surf["R"] * 1000.0
    parts = ["R=%.12g" % R_mm, "k=%.12g" % surf["k"]]
    for i, a_si in enumerate(surf.get("coeffs", [])):
        order = 4 + 2 * i
        a_mm = a_si / (1000.0 ** (order - 1))
        parts.append("A%d=%.12g" % (order, a_mm))
    parts.append("r_max=%.12g" % (surf["r_max"] * 1000.0))
    return "asphere:" + ";".join(parts)


def try_prescription_face(face, face_id, placement, entry, claimed, warnings):
    """Match this FreeCAD face to an unclaimed prescription surface of `entry`
    and, on a match, verify (<1 um) + return the emitted model.json surface
    dict (sphere/asphere) or None for verify-only surfaces (cylinder, where
    the native OCC classification is kept). Records the claimed surface index
    in `claimed`. Returns (surf_dict_or_None, matched_bool). Dies on a
    verified surface whose geometry disagrees with the prescription by > 1 um
    (the CAD drifted from its prescription)."""
    surfaces = entry.get("surfaces", [])
    native_name = type(face.Surface).__name__
    is_revolution = (native_name == "SurfaceOfRevolution")
    # A prescription surface may only match a FreeCAD face of a COMPATIBLE
    # native OCC type (so a native Cylinder rim can never be mis-identified as
    # a grazing sphere cap, etc.). SurfaceOfRevolution is allowed everywhere
    # because OCC sometimes fails to natively recognize a spline-approximated
    # sphere/cylinder (the canonicalize_revolution case) -- and a single
    # element never carries both an asphere and a sphere/cylinder prescription
    # surface, so there is no ambiguity.
    _COMPAT = {
        "sphere": ("Sphere", "SurfaceOfRevolution"),
        "cylinder": ("Cylinder", "SurfaceOfRevolution"),
        "asphere": ("SurfaceOfRevolution",),
    }
    pts = None
    for idx, surf in enumerate(surfaces):
        if idx in claimed:
            continue
        stype = surf["type"]
        if native_name not in _COMPAT.get(stype, ()):
            continue

        if stype == "asphere":
            # match the single revolved (aspheric) face by type
            if not is_revolution:
                continue
            override_str = _presc_asphere_to_override_str(surf)
            out = build_asphere_surface(face, face_id, override_str, warnings)
            # carry the prescription role for downstream display
            out["role"] = surf.get("role")
            claimed.add(idx)
            log("%s: prescription surface (asphere, role=%s) verified + "
                "emitted from prescription" % (face_id, surf.get("role")))
            return out, True

        if stype == "sphere":
            center = _presc_point_to_global(placement, surf["center"])
            radius = surf["radius"]
            if pts is None:
                pts = _sample_face_points_m(face, _PRESC_VERIFY_GRID)
            if not pts:
                continue
            resids = _sphere_residuals(pts, center, radius)
            if min(resids) > _PRESC_IDENTIFY_TOL_M:
                continue                      # not this face
            max_r = max(resids)
            if max_r >= PRESCRIPTION_TOL_M:
                die("%s: CAD drifted from its prescription -- body %r "
                    "sphere surface (role=%s) residual %.3g um over %d "
                    "samples exceeds the %.1f um gate (prescription "
                    "center=%s radius=%.6g m)"
                    % (face_id, entry.get("kind"), surf.get("role"),
                       max_r * 1e6, len(pts), PRESCRIPTION_TOL_M * 1e6,
                       center, radius))
            claimed.add(idx)
            log("%s: prescription surface (sphere, role=%s) verified OK "
                "(max residual %.3g um) + emitted from prescription"
                % (face_id, surf.get("role"), max_r * 1e6))
            return {"type": "sphere", "center": center, "radius": radius,
                    "role": surf.get("role")}, True

        if stype == "cylinder":
            origin = _presc_point_to_global(placement, surf["origin"])
            axis = _presc_dir_to_global(placement, surf["axis"])
            radius = surf["radius"]
            if pts is None:
                pts = _sample_face_points_m(face, _PRESC_VERIFY_GRID)
            if not pts:
                continue
            resids = _cylinder_residuals(pts, origin, axis, radius)
            if min(resids) > _PRESC_IDENTIFY_TOL_M:
                continue
            max_r = max(resids)
            if max_r >= PRESCRIPTION_TOL_M:
                die("%s: CAD drifted from its prescription -- body %r "
                    "cylinder surface (role=%s) residual %.3g um over %d "
                    "samples exceeds the %.1f um gate (prescription "
                    "radius=%.6g m)"
                    % (face_id, entry.get("kind"), surf.get("role"),
                       max_r * 1e6, len(pts), PRESCRIPTION_TOL_M * 1e6,
                       radius))
            claimed.add(idx)
            log("%s: prescription surface (cylinder, role=%s) verified OK "
                "(max residual %.3g um; kept native OCC form)"
                % (face_id, surf.get("role"), max_r * 1e6))
            return None, True

    return None, False


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
    # a source needs lambdac PLUS one of {power, pulse_energy} (pulsed-
    # optics Phase P3: pulse_energy is a valid alternative to power, so a
    # pulse-only source authored without ever setting the 'power' property
    # still classifies correctly; scene.py enforces the power/pulse_energy
    # XOR that this OR-gate deliberately allows through to)
    if hasattr(body, "lambdac") and (hasattr(body, "power")
                                     or hasattr(body, "pulse_energy")):
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
                  surface_override=None, prescription_entry=None):
    faces_out = []
    closest_face_id = None
    closest_dist = None
    origin = FreeCAD.Vector(0, 0, 0)
    presc_claimed = set()
    placement = body.Placement

    for idx, face in enumerate(shape.Faces, start=1):
        face_id = "%s.%s.Face%d" % (body.Name, tip_name, idx)

        override_val = None
        if surface_override:
            override_val = surface_override.get(
                face_id, surface_override.get(common.FACEMAP_ALL))
        override_kind = (override_val.strip().lower().partition(":")[0]
                        if override_val is not None else None)

        # prescription-primary (engine3 Sec 3): if a prescription surface
        # matches this face, it is the truth -- verify + emit from it,
        # OVERRIDING both the surface_override property and native
        # classification for that face.
        presc_surf = None
        if prescription_entry is not None:
            presc_surf, _ = try_prescription_face(
                face, face_id, placement, prescription_entry,
                presc_claimed, warnings)

        if presc_surf is not None:
            surf = presc_surf
        elif override_kind == "asphere":
            surf = build_asphere_surface(face, face_id, override_val, warnings)
        elif override_kind in ("qbfs", "qcon"):
            surf = build_qforbes_surface(face, face_id, override_val, warnings)
        else:
            if override_val is not None:
                warn("%s: surface_override value %r not recognized "
                     "(expected 'asphere:...', 'qbfs:...', or 'qcon:...'); "
                     "ignoring, using auto-detected surface type"
                     % (face_id, override_val), warnings)
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

    # every prescription surface MUST have matched a face -- an unmatched one
    # means the CAD topology no longer realizes the prescription (a hard
    # error, same class as a > 1 um residual: the CAD drifted from its truth).
    if prescription_entry is not None:
        surfaces = prescription_entry.get("surfaces", [])
        missing = [s for i, s in enumerate(surfaces) if i not in presc_claimed]
        if missing:
            die("%s.%s: CAD does not realize its prescription -- %d "
                "prescription surface(s) matched no face: %s (kind=%r). "
                "The geometry drifted from the prescription."
                % (body.Name, tip_name, len(missing),
                   ", ".join("%s/%s" % (m.get("role"), m["type"])
                             for m in missing),
                   prescription_entry.get("kind")))

    return faces_out, closest_face_id


# ---------------------------------------------------------------------------
# One FCStd -> one geometry/<stem>/model.json
# ---------------------------------------------------------------------------
def load_prescription_for(fcstd_path, explicit=None):
    """Resolve the prescription doc for a model (document precedence): an
    explicit --prescription path wins; else a sibling
    <stem>.prescription.json sidecar; else None. Validated on load."""
    if explicit:
        return prescription_mod.load(explicit)
    sidecar = prescription_mod.sidecar_path(fcstd_path)
    if sidecar.exists():
        log("using prescription sidecar %s" % sidecar)
        return prescription_mod.load(sidecar)
    return None


def extract_one(fcstd_path, outdir, strict, prescription=None):
    """CLI wrapper: open the .FCStd, extract, close. The real work lives in
    extract_document() so scripts/fcserver/fcops.py can extract an
    ALREADY-OPEN document in place (the fast evaluator's persistent-worker
    path) and produce byte-identical output. `prescription` may be a resolved
    doc/path; when None a <stem>.prescription.json sidecar is auto-loaded."""
    fcstd_path = Path(fcstd_path).resolve()
    if not fcstd_path.exists():
        die("file not found: %s" % fcstd_path)
    stem = fcstd_path.stem
    if prescription is None or isinstance(prescription, (str, Path)):
        prescription = load_prescription_for(fcstd_path, prescription)
    doc = FreeCAD.openDocument(str(fcstd_path))
    try:
        return extract_document(doc, stem, outdir / stem, strict,
                                fcstd_path, prescription=prescription)
    finally:
        FreeCAD.closeDocument(doc.Name)


def extract_document(doc, stem, out_dir, strict, source_fcstd,
                     face_cache=None, prescription=None):
    """Build, write and contract-validate <out_dir>/model.json (+ per-face
    faces/*.stl) from an already-open FreeCAD document.

    This is the WHOLE extraction contract: extract_one() above is a thin
    open/close wrapper around it, and fcops.op_extract_model() calls it on
    the fast evaluator's persistent in-memory document. `stem` is used only
    for log/warning text (must match the variant stem for byte-identical
    warnings); `source_fcstd` is echoed verbatim into the model's
    provenance field.

    face_cache (optional) skips re-tessellation/re-classification of bodies
    whose geometry did not change between calls. Protocol (implemented by
    fcops._ExtractFaceCache; keyed on shape fingerprint + placement +
    surface_override + strict):
        payload_or_None = face_cache.lookup(body, tip_name, override_raw)
        face_cache.store(body, tip_name, override_raw, payload)
    where payload = {"faces": [...face dicts...], "closest_face_id": str,
    "warnings": [warning strings emitted while walking this body's faces]}
    and lookup() has already placed every referenced faces/*.stl file in
    out_dir when it returns a payload. Raises ExtractError on any fatal
    problem (never exits the process).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    doc.recompute()

    # prescription-primary (engine3 Sec 3, P5): {element key -> entry}. When a
    # body carries a matching entry the extractor verifies its optical faces
    # against the prescription (<1 um gate) and emits them FROM THE
    # PRESCRIPTION. `prescription` is the full doc (schema_version/elements),
    # a bare {key: entry} map, or None (every existing scene extracts
    # unchanged).
    presc_elements = {}
    if prescription:
        presc_elements = prescription.get("elements", prescription) \
            if isinstance(prescription, dict) else {}
        if presc_elements:
            log("%s: prescription-primary cross-check active for %d "
                "element(s)" % (stem, len(presc_elements)))

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

        presc_entry = presc_elements.get(presc_key_for_body(obj)) \
            if presc_elements else None

        # face cache (fast evaluator, see docstring): replay an unchanged
        # body's faces + warnings verbatim instead of re-tessellating. A body
        # with a prescription bypasses the cache -- the cache key does not
        # cover the prescription, so the verify+emit must run fresh (this is
        # the correctness-first path; the fast evaluator does not drive
        # prescriptions).
        cached = (face_cache.lookup(obj, tip_name, surf_override_raw)
                  if (face_cache is not None and presc_entry is None)
                  else None)
        if cached is not None:
            faces = cached["faces"]
            closest_face_id = cached["closest_face_id"]
            for w in cached.get("warnings", []):
                warn(w, warnings)
        else:
            n_warn_mark = len(warnings)
            faces, closest_face_id = extract_faces(
                obj, shape, tip_name, out_dir, strict, warnings,
                surf_override_map, prescription_entry=presc_entry)
            if face_cache is not None and presc_entry is None:
                face_cache.store(obj, tip_name, surf_override_raw, {
                    "faces": faces,
                    "closest_face_id": closest_face_id,
                    "warnings": warnings[n_warn_mark:]})

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

        # edge_blackened (engine3 Sec 11 / P8): a blackened lens EDGE/barrel
        # for ghost & stray-light suppression. When the bool prop is set on an
        # optic, every CYLINDRICAL face (the lens/rod barrel -- identified by
        # analytic surface TYPE, so it is immune to the FaceN renumbering that
        # a rebuild causes) is marked fully absorbing via a per-face absorbance
        # map. The refracting surfaces (sphere/asphere/plane caps) stay clear.
        if getattr(obj, "edge_blackened", False):
            if role != "optic":
                warn("%s: edge_blackened is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                edge_map = {f["id"]: 1.0 for f in faces
                            if f["surface"]["type"] == "cylinder"}
                if edge_map:
                    body_dict["absorbance_faces"] = edge_map
                else:
                    warn("%s: edge_blackened set but no cylindrical edge/barrel "
                         "face found; nothing blackened" % obj.Label, warnings)

        # optional per-body operating temperature (deg C); shifts glasses
        # carrying a thermo-optic model. Blank/none -> scene-global temp.
        if hasattr(obj, "temperature"):
            tv = obj.temperature
            if not (isinstance(tv, str) and tv.strip().lower() in ("", "none")):
                try:
                    body_dict["temperature"] = float(tv)
                except (TypeError, ValueError):
                    die("%s: temperature %r is not a number" % (obj.Label, tv))

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
            # optional explicit importance-scatter targets: comma-separated
            # detector labels this body's scatter aims at (--importance-scatter
            # only; absent => every detector). Names only, resolved at trace.
            targets_raw = str_prop_or_none(obj, "scatter_targets")
            if targets_raw is not None and targets_raw.strip():
                body_dict["scatter_targets"] = targets_raw.strip()

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

        # figure_error (Zernike surface figure error, engine3 Sec 11 / P8):
        # per-face map of figures-registry names ('name' | 'FaceN=name;...'),
        # resolved to a PerturbedSurface at scene build. Names only, no value
        # grammar (like coating/scatter). Geometry-only here: the CAD is the
        # UNPERTURBED shape by design, so no <1 um surface-verify gate applies.
        figure_raw = str_prop_or_none(obj, "figure_error")
        if figure_raw is not None:
            if role != "optic":
                warn("%s: figure_error is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                try:
                    body_dict["figure_error"] = parse_facemap_value_safe(
                        figure_raw, obj.Name, tip_name)
                except ValueError as e:
                    die("%s: bad figure_error spec %r: %s"
                        % (obj.Label, figure_raw, e))

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

        # ---- pulsed-optics Phase P8: Pockels / saturable / TPA / -----
        # ---- Kerr n2 --------------------------------------------------
        nonlinear_raw = str_prop_or_none(obj, "nonlinear")
        if nonlinear_raw is not None:
            if role != "optic":
                warn("%s: nonlinear is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                body_dict["nonlinear"] = nonlinear_raw

        if hasattr(obj, "pockels_voltage"):
            if role != "optic":
                warn("%s: pockels_voltage is only meaningful on optic "
                     "bodies (role=%s); ignoring" % (obj.Label, role),
                     warnings)
            else:
                body_dict["pockels_voltage"] = float(obj.pockels_voltage)

        if hasattr(obj, "pockels_gap"):
            if role != "optic":
                warn("%s: pockels_gap is only meaningful on optic "
                     "bodies (role=%s); ignoring" % (obj.Label, role),
                     warnings)
            else:
                gap = float(obj.pockels_gap)
                if gap <= 0:
                    die("%s: pockels_gap must be > 0 mm (got %g)"
                        % (obj.Label, gap))
                body_dict["pockels_gap_mm"] = gap

        saturable_raw = str_prop_or_none(obj, "saturable")
        if saturable_raw is not None:
            if role != "optic":
                warn("%s: saturable is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                try:
                    common.parse_saturable_value(saturable_raw)
                except ValueError as e:
                    die("%s: bad saturable spec %r: %s"
                        % (obj.Label, saturable_raw, e))
                body_dict["saturable"] = saturable_raw

        if hasattr(obj, "tpa_beta"):
            if role != "optic":
                warn("%s: tpa_beta is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                body_dict["tpa_beta"] = float(obj.tpa_beta)

        kerr_n2_raw = str_prop_or_none(obj, "kerr_n2")
        if kerr_n2_raw is not None:
            if role != "optic":
                warn("%s: kerr_n2 is only meaningful on optic bodies "
                     "(role=%s); ignoring" % (obj.Label, role), warnings)
            else:
                try:
                    common.parse_kerr_n2_value(kerr_n2_raw)
                except ValueError as e:
                    die("%s: bad kerr_n2 spec %r: %s"
                        % (obj.Label, kerr_n2_raw, e))
                body_dict["kerr_n2"] = kerr_n2_raw

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
            # power may be absent on a pulse_energy-only source (see
            # classify_body's OR-gate). 0.0 (not None) is the "unset"
            # sentinel here -- same convention as mirror/absorbance/
            # roughness_nm elsewhere in this contract -- because
            # common.validate_model requires source.power_mW to
            # already be a real float (_req(src, "power_mW", float,
            # ...), a pre-existing schema gate this phase doesn't
            # touch). scene.py's power/pulse_energy XOR + derivation
            # treats power_mW == 0.0 as "not authored" and overwrites
            # it with the real derived average power.
            power_mw = float(obj.power) if hasattr(obj, "power") else 0.0
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
            # design-angle annotation (core.wizards.design_field_fan);
            # optional, consumed by
            # analysis_imaging.field_angle_annotations_from_model.
            if hasattr(obj, "field_angle_deg"):
                source_dict["field_angle_deg"] = float(obj.field_angle_deg)
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
            spectrum_raw = str_prop_or_none(obj, "spectrum")
            if spectrum_raw is not None:
                source_dict["spectrum"] = spectrum_raw
                # a tabulated emission spectrum defines the full lambda
                # distribution; a lambdamin/lambdamax Gaussian would
                # contradict it -- the table wins, drop the bounds.
                if lambdamin is not None or lambdamax is not None:
                    warn("%s: spectrum %r takes precedence over "
                         "lambdamin/lambdamax (dropping the Gaussian "
                         "bounds)" % (obj.Label, spectrum_raw), warnings)
                    source_dict["lambdamin_nm"] = None
                    source_dict["lambdamax_nm"] = None
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
            # pulsed-source properties (Phase P3): optional and
            # independently omitted when absent, same as
            # polarization/apodization/spectrum/beam above. scene.py
            # enforces the power/pulse_energy XOR, requires rep_rate
            # alongside pulse_energy, and derives whichever of
            # {power_mW, pulse_energy_uJ} is missing.
            if hasattr(obj, "pulse_energy"):
                source_dict["pulse_energy_uJ"] = float(obj.pulse_energy)
            if hasattr(obj, "pulse_duration"):
                source_dict["pulse_duration_ps"] = float(obj.pulse_duration)
            if hasattr(obj, "rep_rate"):
                source_dict["rep_rate_hz"] = float(obj.rep_rate)
            # spm (string, Phase P6): source-side self-phase-modulation
            # spec ('phimax:<rad>' or 'gamma:<W^-1km^-1>:length:<m>');
            # string passthrough — raytracer.sources.install_spm
            # parses/validates (it needs the derived pulse block,
            # which only exists engine-side)
            spm_raw = str_prop_or_none(obj, "spm")
            if spm_raw is not None:
                source_dict["spm"] = spm_raw
            body_dict["source"] = source_dict
        elif role == "detector":
            n_detectors += 1
            det_dict = {
                "face": closest_face_id,
                "autodetected": True,
            }
            # detector_face (string): pin the detector's PRIMARY face,
            # overriding the closest-to-world-origin auto-pick (which
            # lands on a thin edge face on rotated/off-axis detectors and
            # silently detects 0 mW). Accepts a bare 'FaceN' (resolved
            # against THIS body's extracted faces, same id form as
            # extract_faces: Body.Tip.FaceN) or an already-full face id.
            # Unlike the CLI --detector-face this replaces the primary
            # face in place (no extra transparent screen), so the scene
            # stays C-engine-routable.
            det_face_raw = str_prop_or_none(obj, "detector_face")
            if det_face_raw is not None:
                face_ids = [f["id"] for f in faces]
                df = det_face_raw.strip()
                if re.match(r"^Face\d+$", df):
                    resolved = "%s.%s.%s" % (obj.Name, tip_name, df)
                else:
                    resolved = df
                if resolved not in face_ids:
                    die("%s: detector_face %r resolves to %r which is "
                        "not one of this body's faces: %s"
                        % (obj.Label, det_face_raw, resolved,
                           ", ".join(face_ids)))
                det_dict["face"] = resolved
                det_dict["autodetected"] = False
            qe_curve = str_prop_or_none(obj, "qe_curve")
            if qe_curve is not None:
                det_dict["qe_curve"] = qe_curve
            # instrument (string): opt this detector body into the virtual
            # instrument layer (engine3.md P2.5 §9) -- a
            # opticalproperties/instrument/instruments.mieinst row name,
            # optionally 'row:mode' with mode in {ideal, full} (default
            # full, mirrors qe_curve's passthrough -- post_process.py
            # parses/validates the row and mode, this stage is a pure
            # string carry).
            instrument_raw = str_prop_or_none(obj, "instrument")
            if instrument_raw is not None:
                det_dict["instrument"] = instrument_raw
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

        if role != "source" and str_prop_or_none(obj, "spectrum") is not None:
            warn("%s: spectrum property is only meaningful on "
                 "source bodies (role=%s); ignoring"
                 % (obj.Label, role), warnings)

        if role != "detector" and \
                str_prop_or_none(obj, "detector_face") is not None:
            warn("%s: detector_face property is only meaningful on "
                 "detector bodies (role=%s); ignoring"
                 % (obj.Label, role), warnings)

        if role != "detector" and \
                str_prop_or_none(obj, "instrument") is not None:
            warn("%s: instrument property is only meaningful on "
                 "detector bodies (role=%s); ignoring"
                 % (obj.Label, role), warnings)

        bodies_out.append(body_dict)
        bodies_solids.append((obj.Name, shape))

    if not bodies_out:
        die("%s: no non-ignored bodies found" % stem)
    if n_sources == 0:
        die("%s: no light sources found (need lambdac plus power or "
            "pulse_energy properties on a body)" % stem)
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
        "source_fcstd": str(source_fcstd),
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

    # An explicit --prescription applies to a single-model run; with multiple
    # models each still auto-discovers its own <stem>.prescription.json.
    explicit_presc = args.prescription
    if explicit_presc and len(paths) > 1:
        die("--prescription names one file but %d models were given; use "
            "per-model <stem>.prescription.json sidecars instead" % len(paths))

    for i, p in enumerate(paths):
        common.progress_emit("extract", i / len(paths), p.stem)
        extract_one(p, outdir, args.strict, prescription=explicit_presc)

    common.progress_emit("extract", 1.0,
                         "%d model(s) extracted" % len(paths),
                         status="completed")
    log("GEOMETRY EXTRACTION OK (%d model(s))" % len(paths))


# NOTE: no `if __name__ == "__main__"` guard — FreeCAD's console mode (-c)
# executes scripts with __name__ set to the module's basename, not
# "__main__", which would silently skip main() if guarded. Instead, run
# main() only when THIS file is the script FreeCAD (or plain python) was
# asked to execute: under `AppImage -c scripts/extract_geometry.py -- ...`
# sys.argv contains this file's path, while a library import (fcops'
# in-place extraction op inside fc_server.py) has fc_server.py there
# instead — so importing this module never triggers a batch extraction.
def _run_as_script():
    base = os.path.basename(__file__)
    return any(os.path.basename(str(a)) == base for a in sys.argv)


if _run_as_script():
    try:
        main()
    except ExtractError:
        # die() already printed the ERROR line; preserve the historical
        # hard exit(1) (sys.exit is swallowed under FreeCAD -c).
        os._exit(1)
    sys.exit(0)
