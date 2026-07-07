#!/usr/bin/env python3
# =============================================================================
# primitivelib.py — parametric single-element primitive builders for the
# MieWorkbench library ("add element" wizards / primitives/*.FCStd).
#
# Interpreter: the FreeCAD AppImage's embedded python for BUILDING; the
# module-level PRIMITIVES metadata dict is importable WITHOUT FreeCAD
# (guarded imports, same pattern as make_test_scenes.py) so the GUI can
# list primitives, parameter specs and defaults under plain python.
#
# Design: each primitive .FCStd contains
#   - one 'dim'-labeled Spreadsheet with one aliased cell per GEOMETRY
#     parameter (raw "=<val> mm" / "=<val> deg" / bare number for counts);
#   - one (or, for achromat/pbs_cube, two) PartDesign::Body built FROM
#     those parameter values, tagged with the usual Base contract props
#     (material/power/lambdac/... per README §5) plus:
#       miewb_primitive : str  — the PRIMITIVES kind that built it
#       miewb_group     : str  — shared by all bodies of one element
#     so the GUI's element editor knows how to rebuild it.
#
# WHY REBUILD-ON-EDIT (not constraint expressions): parameter changes can
# change TOPOLOGY (R -> flat surface, facet counts, sign flips pcx<->pcv),
# which no constraint expression can do. So the spreadsheet is the single
# source of truth and fcserver's 'rebuild_primitive' op re-runs the builder
# with the current alias values, preserving Label/props/Placement. Hand-
# authored user primitives with real cell expressions keep working through
# the ordinary set_spreadsheet -> recompute path instead.
#
# Reuses make_test_scenes.py's proven geometry helpers (lens_meridian,
# revolve_body, pad_body, new_body_pad, ...).
# =============================================================================
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

# make_test_scenes runs its main() at module scope under FreeCAD (the -c
# no-__main__-guard convention); this flag tells it we only want its
# geometry helpers, not a scene build.
os.environ.setdefault("MIEWB_MTS_LIBRARY_ONLY", "1")

try:
    import FreeCAD as App
    import Part
    import make_test_scenes as mts
    _HAVE_FREECAD = True
except Exception:               # metadata-only import path (plain python)
    App = None
    Part = None
    mts = None
    _HAVE_FREECAD = False


def P(default, unit, help_text):
    return {"default": default, "unit": unit, "help": help_text}


# ---------------------------------------------------------------------------
# Primitive registry (importable without FreeCAD)
# ---------------------------------------------------------------------------
PRIMITIVES = {
    # -- sources ------------------------------------------------------------
    "laser_collimated": {
        "category": "Sources", "label": "Collimated laser",
        "tooltip": "Cylindrical housing emitting a collimated beam from "
                   "its flat +x end cap.",
        "params": {"radius": P(5.0, "mm", "beam (emit face) radius"),
                   "length": P(10.0, "mm", "housing length")},
        "props": {"power": 5.0, "lambdac": 633.0, "coherent": True},
    },
    "laser_divergent": {
        "category": "Sources", "label": "Divergent laser",
        "tooltip": "Laser with a convex spherical emit cap: emitted rays "
                   "diverge from a virtual point (radius of curvature = "
                   "roc).",
        "params": {"radius": P(5.0, "mm", "emit aperture radius"),
                   "roc": P(200.0, "mm", "emit cap radius of curvature "
                                         "(divergence = radius/roc)"),
                   "length": P(10.0, "mm", "housing length")},
        "props": {"power": 5.0, "lambdac": 633.0, "coherent": True},
    },
    "source_broadband": {
        "category": "Sources", "label": "Broadband source",
        "tooltip": "Incoherent broadband disc emitter (set lambdamin / "
                   "lambdamax in the properties).",
        "params": {"radius": P(5.0, "mm", "emit face radius"),
                   "length": P(10.0, "mm", "housing length")},
        "props": {"power": 5.0, "lambdac": 550.0,
                  "lambdamin": 450.0, "lambdamax": 650.0,
                  "coherent": False},
    },
    # -- detectors ----------------------------------------------------------
    "detector_plane": {
        "category": "Detectors", "label": "Detector plane",
        "tooltip": "Square thin-screen detector; its -x face records "
                   "irradiance. Transparent to the beam.",
        "params": {"half": P(15.0, "mm", "half-width of the square screen"),
                   "thickness": P(1.0, "mm", "screen thickness")},
        "props": {"material": "detector"},
    },
    # -- spherical lenses (revolved meridians) -------------------------------
    "lens_pcx": {
        "category": "Lenses", "label": "Plano-convex lens",
        "tooltip": "Convex R1 toward -x, flat back.",
        "params": {"R_front": P(25.0, "mm", "front radius of curvature (>0)"),
                   "ct": P(5.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], None),
    },
    "lens_dcx": {
        "category": "Lenses", "label": "Biconvex lens",
        "tooltip": "Convex both sides (R1 front, -R2 back).",
        "params": {"R_front": P(40.0, "mm", "front radius (>0)"),
                   "R_back": P(40.0, "mm", "back radius magnitude (>0)"),
                   "ct": P(6.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], -p["R_back"]),
    },
    "lens_pcv": {
        "category": "Lenses", "label": "Plano-concave lens",
        "tooltip": "Flat front, concave back — diverging.",
        "params": {"R_back": P(25.0, "mm", "back radius of curvature (>0)"),
                   "ct": P(3.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (None, p["R_back"]),
    },
    "lens_dcv": {
        "category": "Lenses", "label": "Biconcave lens",
        "tooltip": "Concave both sides — diverging.",
        "params": {"R_front": P(40.0, "mm", "front radius magnitude (>0)"),
                   "R_back": P(40.0, "mm", "back radius (>0)"),
                   "ct": P(3.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (-p["R_front"], p["R_back"]),
    },
    "lens_meniscus": {
        "category": "Lenses", "label": "Meniscus lens",
        "tooltip": "Both surfaces curve the same way (R1, R2 same sign).",
        "params": {"R_front": P(20.0, "mm", "front radius (signed)"),
                   "R_back": P(40.0, "mm", "back radius (signed)"),
                   "ct": P(4.0, "mm", "center thickness"),
                   "aperture": P(18.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
        "meridian": lambda p: (p["R_front"], p["R_back"]),
    },
    "lens_ball": {
        "category": "Lenses", "label": "Ball lens",
        "tooltip": "Full sphere.",
        "params": {"diameter": P(8.0, "mm", "sphere diameter")},
        "props": {"material": "bk7"},
    },
    "lens_rod": {
        "category": "Lenses", "label": "Rod lens",
        "tooltip": "Cylinder rod (axis along z) — cylinder lens in x-y.",
        "params": {"diameter": P(8.0, "mm", "rod diameter"),
                   "length": P(20.0, "mm", "rod length along z")},
        "props": {"material": "bk7"},
    },
    "lens_cyl": {
        "category": "Lenses", "label": "Cylindrical lens",
        "tooltip": "Plano-convex (R>0) or plano-concave (R<0) cylinder "
                   "lens, cylinder axis along z: line focus.",
        "params": {"R": P(25.0, "mm", "front radius (signed: <0 concave)"),
                   "ct": P(5.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "aperture in y (diameter)"),
                   "height": P(20.0, "mm", "extent along z")},
        "props": {"material": "bk7"},
    },
    "lens_asphere": {
        "category": "Lenses", "label": "Aspheric lens",
        "tooltip": "Plano-convex conic asphere (revolved exact-sag "
                   "BSpline + surface_override; extractor verifies "
                   "<1 um).",
        "params": {"R": P(20.6033, "mm", "vertex radius of curvature"),
                   "k": P(-2.29547, "", "conic constant (k=-n^2 kills "
                                        "spherical aberration)"),
                   "ct": P(6.0, "mm", "center thickness"),
                   "aperture": P(20.0, "mm", "clear aperture diameter")},
        "props": {"material": "bk7"},
    },
    "lens_fresnel": {
        "category": "Lenses", "label": "Fresnel lens",
        "tooltip": "Collapsed plano-convex lens: annular conical facets "
                   "matching the ideal thin-lens local slope.",
        "params": {"aperture": P(24.0, "mm", "clear aperture diameter"),
                   "f_design": P(50.0, "mm", "design focal length"),
                   "n_design": P(1.51508, "", "design refractive index"),
                   "n_facets": P(12.0, "", "number of annular facets"),
                   "back": P(2.0, "mm", "substrate thickness behind the "
                                        "deepest facet")},
        "props": {"material": "bk7"},
    },
    "lens_achromat": {
        "category": "Lenses", "label": "Achromatic doublet",
        "tooltip": "Cemented crown+flint doublet (modeled with a 5 um "
                   "air gap at the interface). Two bodies.",
        "params": {"R_front": P(31.0, "mm", "crown front radius"),
                   "R_iface": P(-21.956, "mm", "cemented interface radius "
                                               "(signed)"),
                   "R_back": P(-64.497, "mm", "flint back radius (signed)"),
                   "ct_crown": P(6.0, "mm", "crown center thickness"),
                   "ct_flint": P(3.0, "mm", "flint center thickness"),
                   "gap": P(0.005, "mm", "interface air gap"),
                   "aperture": P(18.0, "mm", "clear aperture diameter")},
        "props": {},   # per-body materials set by the builder
    },
    # -- other refractives ----------------------------------------------------
    "axicon": {
        "category": "Lenses", "label": "Axicon",
        "tooltip": "Conical front (apex toward -x), flat base: turns a "
                   "beam into a ring / Bessel zone.",
        "params": {"base_angle": P(10.0, "deg", "cone base angle"),
                   "aperture": P(22.0, "mm", "base diameter")},
        "props": {"material": "bk7"},
    },
    "prism": {
        "category": "Prisms & Mirrors", "label": "Equilateral prism",
        "tooltip": "Equilateral dispersing prism, apex up (+y), length "
                   "along z.",
        "params": {"side": P(25.0, "mm", "triangle side length"),
                   "height": P(25.0, "mm", "extent along z"),
                   "rotation": P(0.0, "deg", "rotation about z")},
        "props": {"material": "bk7"},
    },
    "mirror_flat": {
        "category": "Prisms & Mirrors", "label": "Flat mirror",
        "tooltip": "Aluminum plate; combine with a 'mirror' or coating "
                   "property for partial reflectors.",
        "params": {"half": P(12.5, "mm", "half-width of the square face"),
                   "thickness": P(3.0, "mm", "plate thickness")},
        "props": {"material": "aluminum"},
    },
    "window": {
        "category": "Plates & Filters", "label": "Optical window",
        "tooltip": "Plane-parallel plate.",
        "params": {"half": P(12.5, "mm", "half-width"),
                   "thickness": P(3.0, "mm", "plate thickness")},
        "props": {"material": "bk7"},
    },
    "polarizer_plate": {
        "category": "Polarization", "label": "Polarizer",
        "tooltip": "Linear polarizer plate (registry item + body-local "
                   "transmission axis).",
        "params": {"half": P(10.0, "mm", "half-width"),
                   "thickness": P(1.0, "mm", "plate thickness")},
        "props": {"material": "bk7", "polarizer": "ideal_linear",
                  "polarizer_axis": "0,0,1"},
    },
    "waveplate": {
        "category": "Polarization", "label": "Waveplate (quartz)",
        "tooltip": "Uniaxial quartz retarder; retardance set by thickness "
                   "and crystal_axis.",
        "params": {"half": P(8.0, "mm", "half-width"),
                   "thickness": P(0.0298, "mm", "plate thickness (sets "
                                                "retardance)")},
        "props": {"material": "quartz", "crystal_axis": "0,0,1"},
    },
    "filter_plate": {
        "category": "Plates & Filters", "label": "Spectral filter",
        "tooltip": "Bulk (Beer-Lambert) spectral filter plate; pick the "
                   "filter registry item in the properties.",
        "params": {"half": P(12.5, "mm", "half-width"),
                   "thickness": P(3.0, "mm", "plate thickness")},
        "props": {"material": "bk7", "filter": "bp_550_40"},
    },
    "grating_plate": {
        "category": "Plates & Filters", "label": "Diffraction grating",
        "tooltip": "Plate whose front (-x) face carries a grating spec "
                   "(default 600 l/mm vertical grooves; edit the "
                   "'grating' property or use the wizard).",
        "params": {"half": P(12.5, "mm", "half-width"),
                   "thickness": P(3.0, "mm", "plate thickness")},
        "props": {"material": "bk7", "grating": "Face1=600:v"},
    },
    "pbs_cube": {
        "category": "Polarization", "label": "PBS cube",
        "tooltip": "Polarizing beamsplitter: two 45-deg prisms, coated "
                   "hypotenuse, 5 um gap. Two bodies.",
        "params": {"cube": P(20.0, "mm", "cube edge length"),
                   "height": P(20.0, "mm", "extent along z"),
                   "gap": P(0.005, "mm", "hypotenuse air gap")},
        "props": {},   # per-body props set by the builder
    },
}


# ---------------------------------------------------------------------------
# Builders (FreeCAD only). Each: fn(doc, group, p) -> [bodies]
# `group` is the element label stem; single-body builders name the body
# `group` itself, multi-body builders append a suffix.
# ---------------------------------------------------------------------------
def safe_set_props(body, props):
    """Like make_test_scenes.set_props but tolerates already-existing
    properties (needed on the rebuild path, where props are re-applied to
    freshly built bodies that may already carry some of them)."""
    for k, v in (props or {}).items():
        if k not in body.PropertiesList:
            if isinstance(v, bool):
                body.addProperty("App::PropertyBool", k, "Base")
            elif isinstance(v, (int, float)):
                body.addProperty("App::PropertyFloat", k, "Base")
            else:
                body.addProperty("App::PropertyString", k, "Base")
        setattr(body, k, v)


def _tag(bodies, kind, group):
    for b in bodies:
        safe_set_props(b, {"miewb_primitive": kind, "miewb_group": group})
    return bodies


def _simple_lens(doc, group, p, meridian):
    R1, R2 = meridian
    edges, _ = mts.lens_meridian(R1, R2, p["ct"], p["aperture"] / 2.0, 0.0)
    return [mts.revolve_body(doc, group, edges)]


def _build_laser_collimated(doc, group, p):
    return [mts.new_body_pad(doc, group, group,
                             circle=(0.0, 0.0, p["radius"]),
                             x_start=-p["length"], length=p["length"])]


def _build_laser_divergent(doc, group, p):
    # rod with a convex (+x-bulging) spherical emit cap at x=0:
    # lens_meridian back surface with R=-roc bulges toward +x
    edges, _ = mts.lens_meridian(None, -p["roc"], p["length"],
                                 p["radius"], -p["length"])
    return [mts.revolve_body(doc, group, edges)]


def _build_detector_plane(doc, group, p):
    h = p["half"]
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-h, -h, 2 * h, 2 * h)],
                             x_start=0.0, length=p["thickness"])]


def _build_plate(doc, group, p):
    h = p["half"]
    return [mts.new_body_pad(doc, group, group,
                             rects=[(-h, -h, 2 * h, 2 * h)],
                             x_start=0.0, length=p["thickness"])]


def _build_lens_ball(doc, group, p):
    R = p["diameter"] / 2.0
    edges = [mts._arc3(0.0, 0.0, R, R, 2 * R, 0.0),
             mts._line(2 * R, 0.0, 0.0, 0.0)]
    return [mts.revolve_body(doc, group, edges)]


def _build_lens_rod(doc, group, p):
    R = p["diameter"] / 2.0
    circ = [Part.Circle(App.Vector(R, 0.0, 0.0), App.Vector(0, 0, 1), R)]
    return [mts.pad_body(doc, group, circ, plane="XY",
                         offset=-p["length"] / 2.0, length=p["length"])]


def _build_lens_cyl(doc, group, p):
    edges, _ = mts._cyl_lens_profile(p["R"], p["ct"], p["aperture"] / 2.0)
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-p["height"] / 2.0, length=p["height"])]


def _build_lens_asphere(doc, group, p):
    sa = p["aperture"] / 2.0
    R, k, ct = p["R"], p["k"], p["ct"]
    n_samp = 41
    pts = [App.Vector(mts._asphere_sag(sa * i / (n_samp - 1), R, k),
                      sa * i / (n_samp - 1), 0)
           for i in range(n_samp)]
    bs = Part.BSplineCurve()
    bs.interpolate(pts)
    xfr = pts[-1].x
    edges = [bs,
             mts._line(xfr, sa, ct, sa),
             mts._line(ct, sa, ct, 0.0),
             mts._line(ct, 0.0, 0.0, 0.0)]
    body = mts.revolve_body(doc, group, edges)
    mts.set_props(body, {"surface_override":
                         "Face1=asphere:R=%.6f;k=%.6f;r_max=%.4f"
                         % (R, k, sa)})
    return [body]


def _build_lens_fresnel(doc, group, p):
    sa = p["aperture"] / 2.0
    n = int(round(p["n_facets"]))
    f, nglass = p["f_design"], p["n_design"]
    pts = [(0.0, 0.0)]
    for i in range(n):
        v0 = sa * i / n
        v1 = sa * (i + 1) / n
        slope = 0.5 * (v0 + v1) / ((nglass - 1.0) * f)
        dx = slope * (v1 - v0)
        pts.append((0.0, v0))
        pts.append((dx, v1))
    deepest = max(x for x, _ in pts)
    back = deepest + p["back"]
    pts.append((back, sa))
    pts.append((back, 0.0))
    poly = []
    for q in pts:
        if not poly or (abs(poly[-1][0] - q[0]) > 1e-9
                        or abs(poly[-1][1] - q[1]) > 1e-9):
            poly.append(q)
    edges = [mts._line(a[0], a[1], b[0], b[1])
             for a, b in zip(poly, poly[1:])]
    edges.append(mts._line(poly[-1][0], poly[-1][1],
                           poly[0][0], poly[0][1]))
    return [mts.revolve_body(doc, group, edges)]


def _build_lens_achromat(doc, group, p):
    sa = p["aperture"] / 2.0
    crown_edges, _ = mts.lens_meridian(p["R_front"], p["R_iface"],
                                       p["ct_crown"], sa, 0.0)
    crown = mts.revolve_body(doc, group + "_crown", crown_edges,
                             props={"material": "bk7"})
    flint_edges, _ = mts.lens_meridian(p["R_iface"], p["R_back"],
                                       p["ct_flint"], sa,
                                       p["ct_crown"] + p["gap"])
    flint = mts.revolve_body(doc, group + "_flint", flint_edges,
                             props={"material": "sf5"})
    return [crown, flint]


def _build_axicon(doc, group, p):
    sa = p["aperture"] / 2.0
    axial = sa * math.tan(math.radians(p["base_angle"]))
    edges = [mts._line(0.0, 0.0, axial, sa),
             mts._line(axial, sa, axial, 0.0),
             mts._line(axial, 0.0, 0.0, 0.0)]
    return [mts.revolve_body(doc, group, edges)]


def _build_prism(doc, group, p):
    L, H = p["side"], p["height"]
    R = L / math.sqrt(3.0)
    verts = [(R * math.cos(math.radians(a)), R * math.sin(math.radians(a)))
             for a in (90.0, 210.0, 330.0)]
    edges = [mts._line(verts[i][0], verts[i][1],
                       verts[(i + 1) % 3][0], verts[(i + 1) % 3][1])
             for i in range(3)]
    pl = App.Placement(App.Vector(0, 0, 0),
                       App.Rotation(App.Vector(0, 0, 1), p["rotation"]))
    return [mts.pad_body(doc, group, edges, plane="XY",
                         offset=-H / 2.0, length=H, placement=pl)]


def _build_pbs_cube(doc, group, p):
    """Two 45-deg prisms split along the D-B diagonal, hypotenuse of the
    entrance prism coated, exit prism shifted +gap along the hypotenuse
    normal (1,1)/sqrt2 — geometry ported from make_test_scenes.make_pbs_cube."""
    c, H, gap = p["cube"], p["height"], p["gap"]
    half = c / 2.0
    A, B, C, D = (0.0, -half), (c, -half), (c, half), (0.0, half)
    p1 = [A, D, B]
    p1_edges = [mts._line(*p1[i], *p1[(i + 1) % 3]) for i in range(3)]
    b1 = mts.pad_body(doc, group + "_in", p1_edges, plane="XY",
                      offset=-H / 2.0, length=H,
                      props={"material": "bk7"})
    p2 = [D, B, C]
    p2_edges = [mts._line(*p2[i], *p2[(i + 1) % 3]) for i in range(3)]
    shift = gap / math.sqrt(2.0)
    pl2 = App.Placement(App.Vector(shift, shift, 0.0), App.Rotation())
    b2 = mts.pad_body(doc, group + "_out", p2_edges, plane="XY",
                      offset=-H / 2.0, length=H,
                      props={"material": "bk7"}, placement=pl2)
    doc.recompute()
    hyp = mts._find_face_by_normal(b1, (1.0, 1.0, 0.0))
    if hyp is None:
        raise ValueError("pbs_cube: hypotenuse face not found on %s"
                         % b1.Name)
    safe_set_props(b1, {"coating": "Face%d=pbs_visible_45" % hyp})
    return [b1, b2]


def _lens_builder(kind):
    meridian = PRIMITIVES[kind]["meridian"]
    return lambda doc, group, p: _simple_lens(doc, group, p, meridian(p))


_BUILDERS = None


def builders():
    global _BUILDERS
    if _BUILDERS is None:
        if not _HAVE_FREECAD:
            raise RuntimeError("primitivelib builders need FreeCAD")
        _BUILDERS = {
            "laser_collimated": _build_laser_collimated,
            "laser_divergent": _build_laser_divergent,
            "source_broadband": _build_laser_collimated,
            "detector_plane": _build_detector_plane,
            "lens_ball": _build_lens_ball,
            "lens_rod": _build_lens_rod,
            "lens_cyl": _build_lens_cyl,
            "lens_asphere": _build_lens_asphere,
            "lens_fresnel": _build_lens_fresnel,
            "lens_achromat": _build_lens_achromat,
            "axicon": _build_axicon,
            "prism": _build_prism,
            "mirror_flat": _build_plate,
            "window": _build_plate,
            "polarizer_plate": _build_plate,
            "waveplate": _build_plate,
            "filter_plate": _build_plate,
            "grating_plate": _build_plate,
            "pbs_cube": _build_pbs_cube,
        }
        for kind, spec in PRIMITIVES.items():
            if "meridian" in spec:
                _BUILDERS[kind] = _lens_builder(kind)
    return _BUILDERS


# ---------------------------------------------------------------------------
# Build / rebuild entry points (FreeCAD only)
# ---------------------------------------------------------------------------
def sheet_raw(value, unit):
    if unit:
        return "=%.10g %s" % (value, unit)
    return "%.10g" % value


def make_sheet(doc, kind, label="dim"):
    """Create the parameter spreadsheet for `kind` with default values."""
    sheet = doc.addObject("Spreadsheet::Sheet", "Spreadsheet")
    sheet.Label = label
    row = 1
    for alias, spec in PRIMITIVES[kind]["params"].items():
        cell_lbl = "A%d" % row
        cell_val = "B%d" % row
        sheet.set(cell_lbl, alias)
        sheet.set(cell_val, sheet_raw(spec["default"], spec["unit"]))
        sheet.setAlias(cell_val, alias)
        row += 1
    return sheet


def read_params(sheet, kind):
    """Alias values (floats, FreeCAD internal units: mm / deg) for `kind`."""
    import FreeCAD
    out = {}
    for alias, spec in PRIMITIVES[kind]["params"].items():
        cell = sheet.getCellFromAlias(alias)
        if not cell:
            raise ValueError("sheet %s: missing alias %r for primitive %r"
                             % (sheet.Label, alias, kind))
        qty = sheet.get(alias)
        out[alias] = float(FreeCAD.Units.Quantity(qty).Value)
    return out


def build_primitive(doc, kind, group=None, params=None):
    """Build `kind` into doc: bodies + contract props + tagging. Returns
    the list of bodies. `params` defaults to the spec defaults."""
    spec = PRIMITIVES[kind]
    group = group or kind
    if params is None:
        params = {a: s["default"] for a, s in spec["params"].items()}
    bodies = builders()[kind](doc, group, params)
    for b in bodies:
        safe_set_props(b, spec["props"])
    _tag(bodies, kind, group)
    doc.recompute()
    return bodies


def rebuild_element(doc, sheet, kind, group):
    """Rebuild all bodies of `group` from the sheet's current parameter
    values, preserving each body's Label, Placement and any extra custom
    props the user added since. Returns the new bodies."""
    params = read_params(sheet, kind)
    old = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"
           and getattr(o, "miewb_group", None) == group]
    if not old:
        raise ValueError("no bodies with miewb_group %r" % group)
    keep = []
    baseline = {"miewb_primitive", "miewb_group"}
    for b in old:
        extra = {}
        for pname in b.PropertiesList:
            if pname in baseline:
                continue
            try:
                if b.getGroupOfProperty(pname) != "Base":
                    continue
                ptype = b.getTypeIdOfProperty(pname)
            except Exception:
                continue
            if ptype in ("App::PropertyString", "App::PropertyFloat",
                         "App::PropertyBool") \
                    and pname not in ("Label",):
                extra[pname] = getattr(b, pname)
        keep.append({"label": b.Label, "placement": b.Placement,
                     "extra": extra})
        # remove the body and its owned features
        feats = list(getattr(b, "Group", []) or [])
        doc.removeObject(b.Name)
        for f in feats:
            try:
                doc.removeObject(f.Name)
            except Exception:
                pass
    doc.recompute()
    bodies = builders()[kind](doc, group, params)
    _tag(bodies, kind, group)
    for b, k in zip(bodies, keep):
        b.Label = k["label"]
        b.Placement = k["placement"]
        safe_set_props(b, k["extra"])
    doc.recompute()
    return bodies
