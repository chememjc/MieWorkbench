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
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and pinned interpreters (SimulationsGuide.md §1/§2 conventions)
# ---------------------------------------------------------------------------
def _env_path(var, default):
    """Env-overridable path. Defaults preserve historical behavior; the
    MieWorkbench GUI (and remote/headless runs) override via MIEWB_* so a
    project can run against a workspace directory or relocated tools."""
    return Path(os.environ.get(var, "")) if os.environ.get(var) else default


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
BASEMODELS_DIR = PROJECT_DIR / "basemodels"
GEOMETRY_DIR = _env_path("MIEWB_GEOMETRY_DIR", PROJECT_DIR / "geometry")
RESULTS_DIR = _env_path("MIEWB_RESULTS_DIR", PROJECT_DIR / "results")
# Optical-properties library: materials.miemat at the root, one subdirectory
# per category, per-item tables under <category>/tables/ (README §7).
#
# Library files use self-describing extensions (.miemat/.mienk/.miecoat/
# .miepol/.miefilt/.miegrat/.miebrf/.mietab); content is still plain CSV.
# resolve_prop_file() keeps old-style all-.csv libraries (e.g. a user's
# --optical-properties DIR that predates this migration) working: it
# prefers the new name but falls back to the legacy same-stem .csv file.
OPTPROPS_DIR = _env_path("MIEWB_OPTPROPS_DIR",
                         PROJECT_DIR / "opticalproperties")


def resolve_prop_file(preferred_path):
    """preferred_path: the new self-describing filename (e.g.
    .../materials.miemat). Returns preferred_path if it exists; else the
    legacy same-stem .csv sibling if THAT exists (printing a one-line
    NOTE to stderr); else preferred_path unchanged, so a later "not
    found" error names the new extension."""
    preferred_path = Path(preferred_path)
    legacy_path = preferred_path.with_suffix(".csv")
    if preferred_path.exists():
        return preferred_path
    if legacy_path.exists():
        print("NOTE: using legacy %s; rename to %s"
              % (legacy_path, preferred_path), file=sys.stderr)
        return legacy_path
    return preferred_path


MATERIALS_CSV = resolve_prop_file(OPTPROPS_DIR / "materials.miemat")
NK_DATA_DIR = OPTPROPS_DIR / "nk"
COATINGS_CSV = resolve_prop_file(OPTPROPS_DIR / "coating" / "coatings.miecoat")
BIREFRINGENCE_CSV = resolve_prop_file(
    OPTPROPS_DIR / "birefringence" / "uniaxial.miebrf")
POLARIZERS_CSV = resolve_prop_file(
    OPTPROPS_DIR / "polarizer" / "polarizers.miepol")
FILTERS_CSV = resolve_prop_file(OPTPROPS_DIR / "filter" / "filters.miefilt")
GRATINGS_CSV = resolve_prop_file(OPTPROPS_DIR / "grating" / "gratings.miegrat")
CALIBRATION_JSON = RESULTS_DIR / ".calibration.json"

FREECAD_APPIMAGE = os.environ.get(
    "MIEWB_FREECAD", "/home3/freecad/FreeCAD.AppImage")
OPTICS_PYTHON = os.environ.get(
    "MIEWB_OPTICS_PYTHON", "/home3/optics/env/bin/python")
PVPYTHON = os.environ.get(
    "MIEWB_PVPYTHON",
    "/home3/paraview/ParaView-6.1.1-MPI-Linux-Python3.12-x86_64"
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

def parse_diffuser_value(value):
    """Ground-glass diffuser spec value grammar:
        'grit:120'    catalog grit number (mapped to an RMS microfacet
                      slope by raytracer.roughness.slope_for_grit)
        'slope:0.08'  RMS microfacet slope directly (dimensionless)
        '@dg_600'     opticalproperties diffuser-registry entry
    -> {"grit": int} | {"slope": float} | {"registry": str}."""
    value = str(value).strip()
    if value.startswith("@"):
        name = value[1:]
        if not name:
            raise ValueError("empty diffuser registry reference %r" % value)
        return {"registry": name}
    head, sep, rest = value.partition(":")
    if not sep:
        raise ValueError("bad diffuser value %r (expected grit:<G>, "
                         "slope:<m>, or @registryname)" % value)
    if head == "grit":
        grit = int(rest)
        if grit <= 0:
            raise ValueError("diffuser grit must be > 0 in %r" % value)
        return {"grit": grit}
    if head == "slope":
        slope = float(rest)
        if not 0.0 < slope < 1.0:
            raise ValueError("diffuser RMS slope must be in (0, 1) in %r"
                             % value)
        return {"slope": slope}
    raise ValueError("unknown diffuser option %r in %r" % (head, value))


def parse_saturable_value(value):
    """Saturable-absorber body property value grammar (pulsed-optics
    Phase P8):
        'sat:I_sat=<W/cm2>:T0=<0..1>'   inline spec (no registry needed)
        '@name' | 'name'                nonlinear.mienlo kind=saturable
                                         registry row (resolved against
                                         opticalproperties/nonlinear/
                                         nonlinear.mienlo at Scene build)
    -> {"inline": True, "I_sat_W_cm2": float, "T0": float}
       | {"registry": str}
    """
    value = str(value).strip()
    if value.startswith("sat:"):
        out = {"inline": True}
        for part in value[len("sat:"):].split(":"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "bad saturable inline spec %r (expected "
                    "sat:I_sat=<W/cm2>:T0=<0..1>)" % value)
            k, v = part.split("=", 1)
            k = k.strip()
            if k == "I_sat":
                out["I_sat_W_cm2"] = float(v)
            elif k == "T0":
                out["T0"] = float(v)
            else:
                raise ValueError("unknown saturable inline option %r in %r"
                                 % (k, value))
        if out.get("I_sat_W_cm2") is None or out["I_sat_W_cm2"] <= 0:
            raise ValueError(
                "saturable inline spec needs I_sat > 0 (W/cm^2): %r"
                % value)
        if out.get("T0") is None or not 0.0 < out["T0"] <= 1.0:
            raise ValueError(
                "saturable inline spec needs T0 in (0, 1]: %r" % value)
        return out
    name = value[1:] if value.startswith("@") else value
    if not name:
        raise ValueError("empty saturable registry reference %r" % value)
    return {"registry": name}


def parse_kerr_n2_value(value):
    """Kerr (chi3 n2) body property value grammar (pulsed-optics Phase P8):
        'n2:<m2/W>'      inline nonlinear-index value (no registry needed;
                         may be negative, e.g. some semiconductors)
        '@name' | 'name'  nonlinear.mienlo kind=n2 registry row (resolved
                         against opticalproperties/nonlinear/
                         nonlinear.mienlo at Scene build)
    -> {"inline": True, "n2_m2_W": float} | {"registry": str}
    """
    value = str(value).strip()
    if value.startswith("n2:"):
        raw = value[len("n2:"):]
        try:
            n2 = float(raw)
        except ValueError:
            raise ValueError("bad kerr_n2 inline value %r (expected "
                             "n2:<m2/W>)" % value)
        if n2 == 0.0:
            raise ValueError("kerr_n2 inline value must be non-zero: %r"
                             % value)
        return {"inline": True, "n2_m2_W": n2}
    name = value[1:] if value.startswith("@") else value
    if not name:
        raise ValueError("empty kerr_n2 registry reference %r" % value)
    return {"registry": name}


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

def parse_apodization_spec(spec):
    """'gaussian:w0=<mm>[:order=<n>]' -> canonical dict.

    Transverse FIELD amplitude apodization across a source's emitting face
    (source-only, any source): intensity ~ exp(-2 (r/w0)^(2*order)), r the
    distance from the face center. order=1 is a plain Gaussian; order>1 is
    a super-Gaussian (flatter core, steeper skirt)."""
    parts = [p.strip() for p in str(spec).strip().split(":")]
    kind = parts[0].lower()
    if kind != "gaussian":
        raise ValueError("unknown apodization kind %r (only 'gaussian' is "
                         "supported): %r" % (kind, spec))
    out = {"kind": "gaussian", "w0_mm": None, "order": 1}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError("bad apodization field %r (expected k=v) in %r"
                             % (part, spec))
        key, _, val = part.partition("=")
        key = key.strip().lower()
        if key == "w0":
            out["w0_mm"] = float(val)
        elif key == "order":
            out["order"] = int(val)
        else:
            raise ValueError("unknown apodization field %r in %r"
                             % (key, spec))
    if out["w0_mm"] is None or out["w0_mm"] <= 0:
        raise ValueError("apodization needs w0=<mm> > 0: %r" % spec)
    if out["order"] < 1:
        raise ValueError("apodization order must be >= 1: %r" % spec)
    return out


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

    'phi' (mass fraction) and 'tau' (target Beer-Lambert optical depth
    along the box's along-beam length, box_size dx) are mutually
    exclusive — supply one. 'tau' is only PARSED here (this module is
    pure stdlib by contract); it is resolved to a phi by the Mie ensemble
    in raytracer.particles.ParticleCloud (needs the material's Qext,
    which needs numpy/scipy), and the resolved value + the tau it came
    from are echoed back via ParticleCloud.diagnostics()['tau_resolved']
    (hence into case.json).
    """
    out = {"box_corner_m": None, "box_size_m": None, "material": None,
           "phi": None, "tau": None, "median_um": None, "gsd": 1.6}
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
        elif k == "tau":
            out["tau"] = float(v)
        elif k == "median_um":
            out["median_um"] = float(v)
        elif k == "gsd":
            out["gsd"] = float(v)
        elif k == "seed":
            out["seed"] = int(v)
        else:
            raise ValueError("unknown particles field %r in %r" % (k, spec))
    missing = [k for k in ("material", "median_um") if out[k] is None]
    if missing:
        raise ValueError("particles spec %r missing %s" % (spec, missing))
    if out["phi"] is None and out["tau"] is None:
        raise ValueError("particles spec %r missing ['phi (or tau)']" % spec)
    if out["phi"] is not None and out["tau"] is not None:
        raise ValueError(
            "particles spec %r: 'phi' and 'tau' are mutually exclusive "
            "(phi=%r, tau=%r) — pick one: an explicit mass fraction, or a "
            "target optical depth to solve phi for"
            % (spec, out["phi"], out["tau"]))
    if out["phi"] is not None and not (0 < out["phi"] < 1):
        raise ValueError("particles phi (mass fraction) must be in (0,1)")
    if out["tau"] is not None and not out["tau"] > 0:
        raise ValueError("particles tau (target optical depth) must be > 0")
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
    (decimals '.'->'p', minus->'m' — matches the cfdsim convention).
    Sheet-qualified vars ('dim_Lens1.ct') sanitize the '.' to '_' so the
    variant stem stays a clean filename."""
    sval = ("%g" % value).replace(".", "p").replace("-", "m")
    return "%s-%s%s" % (stem, str(var).replace(".", "_"), sval)

def sweep_combos(value_lists, mode="product"):
    """Combinations for a multi-variable sweep — the ONE place combination
    order is defined (permute_model.py and run_pipeline.py's
    variant_output_names() both call this so they can never drift apart).

    value_lists: one list of swept values per variable (each from
    sweep_values()), in --var order.

    mode "product": cartesian (itertools.product order — the historical
    behavior, one variant per combination of every variable's values).
    mode "zip": variables advance together, one variant per index; every
    list must have equal length, except length-1 lists which broadcast
    (so a single-value var can ride along a longer sweep unchanged).

    Returns a list of tuples. Raises ValueError on an unknown mode or a
    zip length mismatch (names the offending lengths)."""
    if mode == "product":
        return list(itertools.product(*value_lists))
    if mode != "zip":
        raise ValueError(
            "unknown sweep mode %r (must be 'product' or 'zip')" % mode)
    if not value_lists:
        return [()]
    lengths = [len(vl) for vl in value_lists]
    non_broadcast = [n for n in lengths if n != 1]
    target = non_broadcast[0] if non_broadcast else 1
    if any(n != target for n in non_broadcast):
        raise ValueError(
            "sweep-mode zip requires every value list to have equal "
            "length (or length 1 to broadcast); got lengths %s" % lengths)
    return [tuple(vl[i] if len(vl) > 1 else vl[0] for vl in value_lists)
            for i in range(target)]

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
            apod = src.get("apodization")
            if apod is not None:
                if (not isinstance(apod, dict) or apod.get("kind") != "gaussian"
                        or not isinstance(apod.get("w0_mm"), (int, float))
                        or apod["w0_mm"] <= 0
                        or not isinstance(apod.get("order"), int)
                        or apod["order"] < 1):
                    raise ContractError(
                        "%s: source.apodization must be a dict "
                        "{kind:'gaussian', w0_mm>0, order>=1}" % bctx)
            beam = src.get("beam")
            if beam is not None:
                if (not isinstance(beam, dict)
                        or not isinstance(beam.get("waist_mm"), (int, float))
                        or beam["waist_mm"] <= 0
                        or not isinstance(beam.get("m2"), (int, float))
                        or beam["m2"] < 1.0):
                    raise ContractError(
                        "%s: source.beam must be a dict "
                        "{waist_mm>0, m2>=1.0}" % bctx)
        if role == "detector":
            n_detectors += 1
            det = _req(b, "detector", dict, bctx)
            _req(det, "face", str, bctx + ".detector")
            if det.get("qe_curve") is not None:
                _req(det, "qe_curve", str, bctx + ".detector")
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
        # ---- pulsed-optics Phase P8 (Pockels / saturable / TPA / Kerr) ---
        if b.get("nonlinear") is not None:
            _req(b, "nonlinear", str, bctx)
        if b.get("pockels_voltage") is not None:
            _req(b, "pockels_voltage", float, bctx)
        if b.get("pockels_gap_mm") is not None:
            gap = _req(b, "pockels_gap_mm", float, bctx)
            if gap <= 0:
                raise ContractError(
                    "%s: pockels_gap_mm must be > 0 (got %g)" % (bctx, gap))
        if b.get("saturable") is not None:
            _req(b, "saturable", str, bctx)
        if b.get("tpa_beta") is not None:
            _req(b, "tpa_beta", float, bctx)
        if b.get("kerr_n2") is not None:
            _req(b, "kerr_n2", str, bctx)
        if isinstance(b.get("coating"), dict):
            _check_facemap(b["coating"], "coating", bctx)
        if b.get("roughness_faces") is not None:
            _check_facemap(b["roughness_faces"], "roughness_faces", bctx,
                           value_check=parse_rough_value)
        if b.get("diffuser_faces") is not None:
            _check_facemap(b["diffuser_faces"], "diffuser_faces", bctx,
                           value_check=parse_diffuser_value)
            if b.get("roughness_faces") is not None:
                overlap = set(b["diffuser_faces"]) \
                    & set(b["roughness_faces"])
                if overlap or FACEMAP_ALL in b["diffuser_faces"] \
                        or FACEMAP_ALL in b["roughness_faces"]:
                    raise ContractError(
                        "%s: diffuser and roughness declared on the same "
                        "face(s) — they are alternative models of one "
                        "surface, pick one per face" % bctx)
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

    n_detectors: for fields_h5_GB only — the number of detectors that will
    ACTUALLY receive --save-fields Ex/Ey field maps (every detector by
    default, or the smaller --save-fields-detectors subset; the caller
    resolves that count and passes it here, e.g. run_trace.py's
    n_field_dets). save_fields: whether --save-fields is set at all;
    fields_h5_GB is exactly 0 when it is not, regardless of n_detectors.
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
    # fields_h5_GB formula (accurate to ~2x; a disk-budget aid, not a
    # precise prediction — see docstring):
    #   n_keys       = n_coherent_sources * nlambda * n_pol_strata
    #                  (one gather.render_coherent key per (source,
    #                  lambda-stratum, pol-stratum); --save-fields only
    #                  applies to the coherent gather)
    #   bytes/key/det = npix * 2 (Ex, Ey) * 16 B (complex128/element)
    #   fields_h5_GB  = bytes/key/det * n_keys * n_detectors / 1e9
    #                  , or 0 outright when save_fields is False.
    n_keys = max(1, n_coherent_sources) * nlambda * max(1, n_pol_strata)
    field_bytes = (npix * 2 * 16 * n_keys * max(0, n_detectors)
                   if save_fields else 0)
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
# Progress events + heartbeat (MieWorkbench integration; CLI-neutral)
#
# Two channels, both optional and both harmless to plain CLI use:
#   1. stdout event lines "@MIEWB {json}" - only when MIEWB_PROGRESS=1 in
#      the environment (the GUI sets it on every pipeline QProcess).
#   2. <case_dir>/progress.json - atomically rewritten on every emit when a
#      case_dir is given; doubles as the liveness heartbeat for the case
#      lock. Written unconditionally: it is a tiny sidecar next to files
#      the stage is already writing.
# ---------------------------------------------------------------------------
PROGRESS_PREFIX = "@MIEWB "


def progress_emit(stage, frac, msg="", case_dir=None, status="running",
                  extra=None):
    """Report stage progress. frac in [0,1] or None when indeterminate."""
    event = {"ev": "progress", "stage": str(stage),
             "frac": None if frac is None else max(0.0, min(1.0, float(frac))),
             "msg": str(msg), "status": status,
             "pid": os.getpid(), "t": _now_s()}
    if extra:
        event.update(extra)
    if os.environ.get("MIEWB_PROGRESS") == "1":
        print(PROGRESS_PREFIX + json.dumps(event, separators=(",", ":")),
              flush=True)
    if case_dir:
        try:
            write_json(Path(case_dir) / "progress.json", event)
        except OSError:
            pass  # progress must never take a stage down


def parse_progress_line(line):
    """GUI/console side: '@MIEWB {...}' -> dict, else None."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        return json.loads(line[len(PROGRESS_PREFIX):])
    except ValueError:
        return None


def _now_s():
    import time
    return time.time()


# ---------------------------------------------------------------------------
# Case locks: one writer per case dir.
#
# Lock file <case_dir>/.lock.json is created O_EXCL. A lock is STALE (safe
# to steal) when its heartbeat (progress.json mtime, falling back to the
# lock file's own mtime) is older than LOCK_STALE_S *and* the recorded pid
# is dead (when the pid is checkable, i.e. same host).
# ---------------------------------------------------------------------------
LOCK_STALE_S = 120.0


class CaseLocked(RuntimeError):
    def __init__(self, info):
        super().__init__("case is locked by pid %s on %s since %s"
                         % (info.get("pid"), info.get("host"),
                            info.get("started")))
        self.info = info


def _lock_path(case_dir):
    return Path(case_dir) / ".lock.json"


def lock_info(case_dir):
    """Return the lock dict, or None if the case is unlocked."""
    try:
        with open(_lock_path(case_dir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def lock_is_stale(case_dir, info=None):
    info = info if info is not None else lock_info(case_dir)
    if info is None:
        return False
    heartbeat = None
    for name in ("progress.json", ".lock.json"):
        try:
            mtime = os.path.getmtime(Path(case_dir) / name)
            heartbeat = mtime if heartbeat is None else max(heartbeat, mtime)
        except OSError:
            pass
    if heartbeat is not None and (_now_s() - heartbeat) < LOCK_STALE_S:
        return False
    import socket
    if info.get("host") == socket.gethostname() and info.get("pid"):
        try:
            os.kill(int(info["pid"]), 0)
            return False        # heartbeat old but process is alive
        except (OSError, ValueError):
            return True         # pid is gone -> stale
    # other host / unknown pid: trust the heartbeat age alone
    return True


def acquire_case_lock(case_dir, force=False):
    """Create <case_dir>/.lock.json; raises CaseLocked if held and fresh.

    force=True steals the lock regardless (caller confirmed with the user).
    Returns the lock dict that was written.
    """
    import socket
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(case_dir)
    existing = lock_info(case_dir)
    if existing is not None:
        if not force and not lock_is_stale(case_dir, existing):
            raise CaseLocked(existing)
        try:
            os.unlink(path)
        except OSError:
            pass
    info = {"pid": os.getpid(), "host": socket.gethostname(),
            "started": _now_s(),
            "cmdline": " ".join(sys.argv[:6])}
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, json.dumps(info, indent=1).encode())
    finally:
        os.close(fd)
    return info


def release_case_lock(case_dir):
    """Remove the lock if THIS process owns it."""
    info = lock_info(case_dir)
    if info is not None and info.get("pid") == os.getpid():
        try:
            os.unlink(_lock_path(case_dir))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Viz-ray pattern specs (visual ray overlay only - never affects physics)
#
#   rings:dr=<mm>:nper=<N>[:nrings=<K>]
#     one central ray plus concentric rings every dr mm with N rays per
#     ring step, out to the emit face's rim (or K rings if given).
#
#   fan[:n=<K>]  (K default 5)
#     one central ray plus up to 4 cardinal rays (top/bottom/right/left of
#     the emit face) and, beyond that, evenly-spaced rim-filler rays.
# ---------------------------------------------------------------------------
def parse_viz_pattern_spec(spec):
    """Parse a --viz-pattern value; returns a dict or raises ValueError."""
    parts = str(spec).strip().split(":")
    kind = parts[0].strip().lower()
    if kind == "rings":
        out = {"kind": "rings", "dr_mm": None, "nper": None, "nrings": None}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError("bad viz-pattern field %r (expected k=v)"
                                 % part)
            key, _, val = part.partition("=")
            key = key.strip().lower()
            try:
                if key == "dr":
                    out["dr_mm"] = float(val)
                elif key == "nper":
                    out["nper"] = int(val)
                elif key == "nrings":
                    out["nrings"] = int(val)
                else:
                    raise ValueError("unknown viz-pattern key %r" % key)
            except (TypeError, ValueError) as exc:
                if "viz-pattern" in str(exc):
                    raise
                raise ValueError("bad viz-pattern value %r for key %r"
                                 % (val, key))
        if out["dr_mm"] is None or out["dr_mm"] <= 0:
            raise ValueError("viz pattern needs dr=<mm> > 0")
        if out["nper"] is None or out["nper"] < 1:
            raise ValueError("viz pattern needs nper=<int> >= 1")
        if out["nrings"] is not None and out["nrings"] < 0:
            raise ValueError("viz pattern nrings must be >= 0")
        return out
    elif kind == "fan":
        out = {"kind": "fan", "n": 5}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError("bad viz-pattern field %r (expected k=v)"
                                 % part)
            key, _, val = part.partition("=")
            key = key.strip().lower()
            if key != "n":
                raise ValueError("unknown viz-pattern key %r" % key)
            try:
                out["n"] = int(val)
            except (TypeError, ValueError):
                raise ValueError("bad viz-pattern value %r for key %r"
                                 % (val, key))
        if out["n"] < 1:
            raise ValueError("viz pattern needs n=<int> >= 1")
        return out
    else:
        raise ValueError("unknown viz pattern %r (expected 'rings:...' or "
                         "'fan[:n=...]')" % kind)


# ---------------------------------------------------------------------------
# Self-check:  python3 scripts/common.py
# ---------------------------------------------------------------------------
def element_power_table(audit, name_to_label=None):
    """Per-element power table from an audit.json dict, averaged across
    seeds: {label: {power_in_W, power_out_W, absorbed_W, detected_W}}.

    in/out come from the tracer's boundary-flux tallies
    (element_flux_W); absorbed sums the body-tagged losses (by_body_W:
    bulk/particle/polarizer/seam) plus surface absorption (by_surface_W
    minus the detected share it historically also holds); detected keys
    are detector FACE ids ('Body.Feature.FaceN'), grouped here under the
    body name -- pass name_to_label (model.json name -> label) so those
    rows merge with the flux rows, which use body LABELS. Older audits
    without element_flux_W yield only the absorbed/detected columns.
    Stdlib-only: shared by post_process (the report.json writer) and the
    GUI results pane (old-case fallback)."""
    name_to_label = name_to_label or {}
    seeds = audit.get("per_seed", [])
    if not seeds:
        return {}
    table = {}

    def row(label):
        return table.setdefault(label, {"power_in_W": 0.0,
                                        "power_out_W": 0.0,
                                        "absorbed_W": 0.0,
                                        "detected_W": 0.0})

    n = float(len(seeds))
    for rep in seeds:
        detected = rep.get("detected_W", {})
        for label, fx in rep.get("element_flux_W", {}).items():
            r = row(label)
            r["power_in_W"] += fx.get("in_W", 0.0) / n
            r["power_out_W"] += fx.get("out_W", 0.0) / n
        for label, w in rep.get("by_body_W", {}).items():
            row(label)["absorbed_W"] += w / n
        for label, w in rep.get("by_surface_W", {}).items():
            absorbed = w - detected.get(label, 0.0)
            if absorbed > 0:
                row(label)["absorbed_W"] += absorbed / n
        for face_id, w in detected.items():
            body = face_id.split(".", 1)[0]
            row(name_to_label.get(body, body))["detected_W"] += w / n
    return {k: table[k] for k in sorted(table)}


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
    check("materials.miemat", MATERIALS_CSV.exists(), str(MATERIALS_CSV))

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
    apod = parse_apodization_spec("gaussian:w0=2")
    check("apodization default order", apod["w0_mm"] == 2.0
          and apod["order"] == 1)
    apod2 = parse_apodization_spec("gaussian:w0=1.5:order=3")
    check("apodization super-gaussian", apod2["order"] == 3)
    for bad_apod in ("flattop:w0=1", "gaussian", "gaussian:w0=0",
                     "gaussian:w0=-1", "gaussian:w0=1:order=0",
                     "gaussian:w0=1:bogus=2"):
        try:
            parse_apodization_spec(bad_apod)
            check("reject apodization %r" % bad_apod, False)
        except ValueError:
            check("reject apodization %r" % bad_apod, True)
    ax = parse_axis_spec("0,0,2")
    check("axis spec", abs(ax[2] - 1.0) < 1e-12 and ax[0] == 0.0)
    check("optprops dirs", COATINGS_CSV.exists() and NK_DATA_DIR.is_dir(),
          str(OPTPROPS_DIR))
    p = parse_particles_spec("material=water;phi=1e-4;median_um=10")
    check("particles default box",
          abs(p["box_size_m"][0] - 0.010) < 1e-12
          and abs(p["box_size_m"][1] - 0.020) < 1e-12)
    pt = parse_particles_spec("material=water;tau=1.0;median_um=10")
    check("particles tau parses, phi left unresolved",
          pt["tau"] == 1.0 and pt["phi"] is None)
    try:
        parse_particles_spec(
            "material=water;phi=1e-4;tau=1.0;median_um=10")
        check("reject phi+tau together", False)
    except ValueError:
        check("reject phi+tau together", True)
    try:
        parse_particles_spec("material=water;median_um=10")
        check("reject neither phi nor tau", False)
    except ValueError:
        check("reject neither phi nor tau", True)
    for bad in ("Body.Face2", "Body..Face2", "Body.Pad.Face"):
        try:
            parse_face_spec(bad)
            check("reject %r" % bad, False)
        except ValueError:
            check("reject %r" % bad, True)
    est = estimate(1e5, 512, 5, 1, "auto")
    check("estimator sane", est["total_s"] > 0 and est["accumulator_GB"] > 0,
          "total=%s" % fmt_duration(est["total_s"]))

    # progress / lock / viz-pattern helpers (MieWorkbench integration)
    ev = parse_progress_line(
        PROGRESS_PREFIX + '{"ev":"progress","stage":"trace","frac":0.5}')
    check("progress line parse", ev is not None and ev["frac"] == 0.5)
    check("progress noise ignored",
          parse_progress_line("[trace] seed 1/3") is None)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        progress_emit("trace", 0.25, "seed 1/4", case_dir=td)
        with open(Path(td) / "progress.json") as fh:
            check("progress.json heartbeat",
                  json.load(fh)["frac"] == 0.25)
        acquire_case_lock(td)
        try:
            acquire_case_lock(td)
            check("second lock rejected", False)
        except CaseLocked:
            check("second lock rejected", True)
        check("own lock not stale", not lock_is_stale(td))
        release_case_lock(td)
        check("lock released", lock_info(td) is None)
    vp = parse_viz_pattern_spec("rings:dr=0.5:nper=8:nrings=4")
    check("viz pattern", vp["dr_mm"] == 0.5 and vp["nper"] == 8
          and vp["nrings"] == 4)
    vp2 = parse_viz_pattern_spec("rings:dr=1:nper=12")
    check("viz pattern open rings", vp2["nrings"] is None)
    vp3 = parse_viz_pattern_spec("fan")
    check("viz pattern fan default", vp3["kind"] == "fan" and vp3["n"] == 5)
    vp4 = parse_viz_pattern_spec("fan:n=9")
    check("viz pattern fan n", vp4["n"] == 9)
    for bad_vp in ("spiral:dr=1", "rings:dr=0:nper=4", "rings:nper=4",
                   "rings:dr=1:nper=0", "rings:dr=x:nper=4",
                   "fan:n=0", "fan:n=x", "fan:bogus=1"):
        try:
            parse_viz_pattern_spec(bad_vp)
            check("reject viz %r" % bad_vp, False)
        except ValueError:
            check("reject viz %r" % bad_vp, True)
    print("SELF-CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(_selfcheck())
