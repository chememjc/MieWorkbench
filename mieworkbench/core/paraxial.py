"""Paraxial (ABCD) engine — pure functions, no Qt, plain floats.

Computes per-element cardinal values (EFL / BFL / FFL / principal
planes) and whole-train system properties (system EFL, image distance,
magnification, working f-number, NA, limiting aperture) from the same
data the train solver uses: primitive dim sheets + the material library
+ chain-edge distances. This is the calculation backend for the GUI's
"no side math" features (element paraxial readout, train system summary,
right-click insert-optical-value menus).

Conventions
-----------
* Light travels along the local optical axis; distances in mm; angles
  are REDUCED paraxial angles (omega = n * u), so a translation of d in
  index n is [[1, d/n], [0, 1]] and a refracting surface of power
  P = (n2 - n1)/R is [[1, 0], [-P, 1]].  Matrices are (A, B, C, D)
  tuples acting on column (y, omega).
* Sign convention matches wizards.py / primitivelib meridians: R > 0 is
  convex toward the source (-x), R < 0 convex toward the image, None is
  flat.  EFL = -1/C of the element matrix (entry vertex -> exit
  vertex); BFL = -A/C from the exit vertex; FFL = D/C from the entry
  vertex (negative = focus upstream of the element, the usual case);
  principal planes are reported as offsets from the entry (pp1) / exit
  (pp2) vertices along the beam.
* Mirrors are handled UNFOLDED: a reflecting surface of focal length f
  contributes power 1/f and propagation simply continues along the
  (reflected) beam — exactly how the chain solver's `distance` fields
  measure, so chained distances feed straight in.

Honest limits (also surfaced via the `warnings` list in results)
----------------------------------------------------------------
* Paraxial only: no aberrations, small-angle.
* Tilts and decenters on chain edges are IGNORED (a warning is emitted
  when any are nonzero); folds are traversed as the equivalent
  straightened path.
* Cylindrical elements (lens_cyl, lens_rod) are evaluated in their
  POWER meridian only and flagged `cylindrical`.
* lens_fresnel uses its ideal thin-lens design values (flagged
  `approximate`).
* Branching trains follow the default-port arm; other branches are
  reported in warnings.
* Elements this module doesn't recognize optically (prisms, gratings,
  hand-authored bodies, ...) pass through as identity.

The exact thick-lens formulas in wizards.py (`thick_lens_efl/bfl`) are
the oracle this module's tests pin against — the two must always agree.
"""

import math

from . import wizards

# primitivelib comes from scripts/ (already importable once core.train
# has been imported anywhere in the process; make it self-sufficient the
# same way train.py does).
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import primitivelib  # noqa: E402
import train_solver  # noqa: E402

DESIGN_LAMBDA_FALLBACK_NM = wizards.D_LINE_NM

IDENT = (1.0, 0.0, 0.0, 1.0)

# dim-sheet aliases that name an element's clear aperture, in preference
# order (primitivelib conventions: lenses use `aperture`, plates/sources
# `width`/`diameter`, iris/pinhole/annular the `hole_diameter` OPENING —
# which must win over their outer/disc diameters).
_APERTURE_ALIASES = ("hole_diameter", "aperture", "diameter", "width")

# plane-parallel-slab kinds: flat both sides, real glass — modeled as a
# slab (captures the t*(1-1/n) image shift); everything else unknown is
# a pure identity.
_SLAB_KINDS = {
    "window", "polarizer_plate", "waveplate", "diffuser_plate",
    "filter_longpass", "filter_shortpass", "filter_bandpass", "nd_filter",
    "filter_plate", "bs_plate", "pellicle", "hot_mirror_plate",
}

# achromat builder's fixed glass pair (primitivelib._build_lens_achromat;
# wizards.solve_achromat scales the same BK7/SF5 shipped design).
_ACHROMAT_GLASSES = ("bk7", "sf5")


def mmul(m2, m1):
    """Matrix product m2 @ m1 for (A, B, C, D) tuples."""
    a2, b2, c2, d2 = m2
    a1, b1, c1, d1 = m1
    return (a2 * a1 + b2 * c1, a2 * b1 + b2 * d1,
            c2 * a1 + d2 * c1, c2 * b1 + d2 * d1)


def surface(n1, n2, R):
    """Refracting spherical surface n1 -> n2, radius R (None = flat)."""
    if R is None:
        return IDENT
    return (1.0, 0.0, -(n2 - n1) / R, 1.0)


def translate(d, n=1.0):
    """Axial translation d mm inside index n (reduced-angle convention)."""
    return (1.0, d / n, 0.0, 1.0)


def thin(power):
    """Thin element of power 1/f (mirrors: 1/f = 2/R spherical)."""
    return (1.0, 0.0, -power, 1.0)


# |EFL| beyond this is reported as afocal (angular magnification) rather
# than a meaninglessly huge focal length (UXNOTES_ROUND3 #29 / future.md
# (a2) "near-afocal EFL suppression") — 100 m is an arbitrary but
# defensible bound: no cataloged element/system in this library has a
# legitimate design EFL anywhere near it, while a train that is merely
# very slightly off collimation (typical of a beam-expander pair a hair
# off its exact spacing) easily produces C ~ 1e-6 /mm and a formally
# "finite" EFL of kilometers.
NEAR_AFOCAL_EFL_MM = 1.0e5


def cardinals_from_matrix(M, thickness_mm):
    """EFL/BFL/FFL/principal planes from an entry->exit element matrix.

    When the system power |C| is at or below NEAR_AFOCAL_EFL_MM's
    reciprocal (|EFL| > NEAR_AFOCAL_EFL_MM), `afocal` is set True and
    `angular_magnification` is populated with D (the ABCD matrix's D
    element): for an EXACTLY afocal system (C == 0 identically) the
    output reduced angle omega' = D * omega for every ray regardless of
    height, so D IS the (height-independent) angular magnification; near
    collimation it is reported as an approximation of the same quantity.
    """
    a, b, c, d = M
    if abs(c) < 1e-15:
        return {"efl": math.inf, "bfl": math.inf, "ffl": math.inf,
                "pp1_mm": 0.0, "pp2_mm": 0.0, "afocal": True,
                "angular_magnification": d, "thickness_mm": thickness_mm}
    efl = -1.0 / c
    bfl = -a / c
    ffl = d / c
    afocal = abs(efl) > NEAR_AFOCAL_EFL_MM
    out = {"efl": efl, "bfl": bfl, "ffl": ffl,
           # principal-plane offsets: H1 from entry vertex (+ = downstream),
           # H2 from exit vertex (+ = downstream)
           "pp1_mm": ffl + efl, "pp2_mm": bfl - efl,
           "afocal": afocal, "thickness_mm": thickness_mm}
    if afocal:
        out["angular_magnification"] = d
    return out


# ---------------------------------------------------------------------------
# per-kind element matrices
# ---------------------------------------------------------------------------
def element_matrix(kind, params, index_fn, lam_nm, material=None,
                   n_ambient=1.0):
    """(M, thickness_mm, meta) for one catalog element.

    index_fn(material_name, lam_nm) -> real index. `material` is the
    primary body's material property (falls back to the catalog default
    when None). meta flags: passthrough / mirror / cylindrical /
    approximate.
    """
    meta = {}
    prim = primitivelib.PRIMITIVES.get(kind) or {}

    def n_of(mat, fallback="bk7"):
        return float(index_fn(mat or fallback, lam_nm))

    # --- refractive lenses with a meridian lambda (signed radii) ---------
    if "meridian" in prim and kind != "lens_cyl":
        R1, R2 = prim["meridian"](params)
        n = n_of(material or (prim.get("props") or {}).get("material"))
        ct = float(params["ct"])
        M = mmul(surface(n, n_ambient, R2),
                 mmul(translate(ct, n), surface(n_ambient, n, R1)))
        return M, ct, meta

    if kind == "lens_asphere":
        n = n_of(material or "bk7")
        ct = float(params["ct"])
        M = mmul(surface(n, n_ambient, None),
                 mmul(translate(ct, n),
                      surface(n_ambient, n, float(params["R"]))))
        # conic/A4/A6 terms don't change the PARAXIAL matrix (vertex R only)
        return M, ct, meta

    if kind == "lens_ball" or kind == "lens_rod":
        n = n_of(material or "bk7")
        D = float(params["diameter"])
        M = mmul(surface(n, n_ambient, -D / 2.0),
                 mmul(translate(D, n), surface(n_ambient, n, D / 2.0)))
        if kind == "lens_rod":
            meta["cylindrical"] = True
        return M, D, meta

    if kind == "lens_cyl":
        n = n_of(material or "bk7")
        ct = float(params["ct"])
        M = mmul(surface(n, n_ambient, None),
                 mmul(translate(ct, n),
                      surface(n_ambient, n, float(params["R"]))))
        meta["cylindrical"] = True
        return M, ct, meta

    if kind == "lens_fresnel":
        f = float(params["f_design"])
        meta["approximate"] = True
        return thin(1.0 / f), 0.0, meta

    if kind == "lens_achromat":
        n_crown = n_of(_ACHROMAT_GLASSES[0])
        n_flint = n_of(_ACHROMAT_GLASSES[1])
        R1 = float(params["R_front"])
        Ri = float(params["R_iface"])
        R3 = float(params["R_back"])
        ct1 = float(params["ct_crown"])
        ct2 = float(params["ct_flint"])
        gap = float(params.get("gap", 0.005))
        M = IDENT
        for step in (surface(n_ambient, n_crown, R1),
                     translate(ct1, n_crown),
                     surface(n_crown, n_ambient, Ri),
                     translate(gap, n_ambient),
                     surface(n_ambient, n_flint, Ri),
                     translate(ct2, n_flint),
                     surface(n_flint, n_ambient, R3)):
            M = mmul(step, M)
        return M, ct1 + gap + ct2, meta

    # --- mirrors (unfolded: pure power, zero thickness) -------------------
    if kind in ("mirror_concave", "mirror_annular"):
        meta["mirror"] = True
        return thin(2.0 / float(params["R"])), 0.0, meta
    if kind == "mirror_convex":
        meta["mirror"] = True
        return thin(-2.0 / float(params["R"])), 0.0, meta
    if kind == "mirror_parabolic":
        meta["mirror"] = True
        return thin(1.0 / float(params["rfl"])), 0.0, meta
    if kind in ("mirror_flat", "mirror_d_shaped"):
        meta["mirror"] = True
        return IDENT, 0.0, meta

    # --- plane-parallel slabs ---------------------------------------------
    if kind in _SLAB_KINDS and "thickness" in params:
        mat = material or (prim.get("props") or {}).get("material")
        if mat and mat not in ("detector", "aluminum", "none"):
            try:
                n = n_of(mat)
                t = float(params["thickness"])
                M = translate(t, n)
                return M, t, meta
            except Exception:
                pass  # unknown material -> identity passthrough

    meta["passthrough"] = True
    return IDENT, 0.0, meta


def element_cardinals(kind, params, index_fn, lam_nm, material=None):
    """Cardinal values dict for one element (see cardinals_from_matrix),
    plus the element_matrix meta flags merged in."""
    M, t, meta = element_matrix(kind, params, index_fn, lam_nm,
                                material=material)
    out = cardinals_from_matrix(M, t)
    out.update(meta)
    return out


# ---------------------------------------------------------------------------
# train integration
# ---------------------------------------------------------------------------
def _prop(body, name):
    from .train import _prop_value
    return _prop_value(body, name)


def _is_source(body):
    return (_prop(body, "power") not in (None, "")
            and _prop(body, "lambdac") not in (None, ""))


def design_wavelength_nm(train_model):
    """First source's lambdac (chain solve order), else the d-line."""
    try:
        order = train_solver.sort_chain(train_model.records())
    except train_solver.TrainError:
        order = train_model.element_labels()
    for el in order:
        try:
            primary = train_model.primary_body(el)
        except train_solver.TrainError:
            continue
        lam = _prop(primary, "lambdac")
        pwr = _prop(primary, "power")
        if lam not in (None, "") and pwr not in (None, ""):
            try:
                return float(lam)
            except (TypeError, ValueError):
                continue
    return DESIGN_LAMBDA_FALLBACK_NM


def element_aperture_mm(train_model, element):
    """Clear-aperture diameter from the element's dim sheet, or None."""
    params = train_model._sheet_params(element)
    if not params:
        return None
    for key in _APERTURE_ALIASES:
        if key in params and params[key] and params[key] > 0:
            return float(params[key])
    return None


def chain_path(train_model, variables=None):
    """Ordered single-arm optical path from the chain root.

    Returns (path, warnings). Each path entry:
      {element, kind, params, material, spacing_mm, aperture_mm,
       is_source, is_detector}
    spacing_mm = chain `distance` (vertex-to-vertex along the beam) from
    the previous path element. Follows the default-port arm on branches.
    Anchored elements other than the root are skipped with a warning
    (their spacing is not derivable without a geometric solve).
    """
    warnings = []
    records = train_model.records()
    variables = variables or {}
    try:
        order = train_solver.sort_chain(records)
    except train_solver.TrainError as e:
        return [], ["chain not solvable: %s" % e]

    # children keyed by (ref, effective port)
    children = {}
    for el in order:
        rec = records[el]
        if rec.get("mode") != "chained":
            continue
        ref = rec.get("ref")
        if not ref:
            continue
        port = rec.get("port") or train_solver._default_port(
            records.get(ref, {}))
        children.setdefault((ref, port), []).append(el)

    # root: first chained element's transitive anchored ancestor
    root = None
    for el in order:
        rec = records[el]
        if rec.get("mode") == "chained":
            ref = rec.get("ref")
            while ref and records.get(ref, {}).get("mode") == "chained":
                ref = records[ref].get("ref")
            root = ref or el
            break
    if root is None:
        return [], ["no chained elements"]

    path = []
    cur = root
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        rec = records.get(cur) or {}
        primary = train_model.primary_body(cur)
        kind = _prop(primary, "miewb_primitive")
        params = train_model._sheet_params(cur) or {}
        spacing = None
        if rec.get("mode") == "chained":
            spacing = train_solver._edge_value(rec, "distance", variables)
            for f in ("decenter_x", "decenter_y",
                      "tilt_rx", "tilt_ry", "tilt_rz"):
                try:
                    if abs(train_solver._edge_value(rec, f, variables)) \
                            > 1e-9:
                        warnings.append(
                            "%s: nonzero %s ignored (paraxial)" % (cur, f))
                except train_solver.TrainError:
                    pass
        path.append({
            "element": cur,
            "kind": kind,
            "params": params,
            "material": _prop(primary, "material"),
            "spacing_mm": spacing,
            "aperture_mm": element_aperture_mm(train_model, cur),
            "is_source": _is_source(primary),
            "is_detector": (_prop(primary, "material") == "detector"),
        })
        port = train_solver._default_port(rec) if rec else "out"
        kids = children.get((cur, port), [])
        # also accept explicit transmit-port children when default differs
        if not kids:
            for p in train_solver.TRANSMIT_PORTS + ("reflect", "deviate"):
                kids = children.get((cur, p), [])
                if kids:
                    port = p
                    break
        if len(kids) > 1:
            warnings.append("branch at %s port %s: following %s "
                            "(ignoring %s)" % (cur, port, kids[0],
                                               ", ".join(kids[1:])))
        cur = kids[0] if kids else None
    return path, warnings


def system_summary(train_model, variables=None, index_fn=None, lam_nm=None,
                   object_distance_mm=math.inf, through_element=None):
    """Whole-train paraxial summary over the chain path.

    Returns {efl, bfl, image_distance_mm, magnification, fno_working, na,
    limiting_element, lambda_nm, n_optical_elements, warnings, path}.
    image_distance_mm is measured from the LAST optical element's exit
    vertex; object_distance_mm from the FIRST optical element's entry
    vertex (inf = collimated input). `through_element` truncates the
    system after that element (the "image distance from element X"
    building block for the insert-value menus).
    """
    if index_fn is None:
        matdb = wizards._default_matdb()

        def index_fn(mat, lam):
            return wizards.index_at(matdb, mat, lam)

    if lam_nm is None:
        lam_nm = design_wavelength_nm(train_model)

    path, warnings = chain_path(train_model, variables)
    out = {"efl": None, "bfl": None, "image_distance_mm": None,
           "magnification": None, "fno_working": None, "na": None,
           "limiting_element": None, "lambda_nm": lam_nm,
           "n_optical_elements": 0, "warnings": warnings, "path": path,
           "afocal": False, "angular_magnification": None}
    if not path:
        return out

    # build the optical sequence: (element matrices) interleaved with
    # air translations; track ray heights at each element entry for the
    # aperture search. Skip sources; stop at the first detector (it is
    # the screen, not an optic) but remember its position.
    seq = []            # (kind_tag, matrix, element_or_None)
    stations = []       # (element, aperture_mm) at entry planes
    pending_gap = 0.0
    n_opt = 0
    detector_gap = None
    for i, entry in enumerate(path):
        if entry["spacing_mm"] is not None and i > 0:
            pending_gap += float(entry["spacing_mm"])
        if entry["is_source"]:
            continue
        if entry["is_detector"]:
            detector_gap = pending_gap
            break
        if entry["kind"] is None and not entry["params"]:
            # hand-authored body: passthrough, but keep its aperture stop
            M, t, meta = IDENT, 0.0, {"passthrough": True}
        else:
            try:
                M, t, meta = element_matrix(
                    entry["kind"], entry["params"], index_fn, lam_nm,
                    material=entry["material"])
            except (KeyError, TypeError, ValueError) as e:
                warnings.append("%s: not modeled (%s), passthrough"
                                % (entry["element"], e))
                M, t, meta = IDENT, 0.0, {"passthrough": True}
        seq.append(("gap", translate(pending_gap), None, pending_gap))
        stations.append((entry["element"], entry["aperture_mm"]))
        seq.append(("elem", M, entry["element"], t))
        if not meta.get("passthrough"):
            n_opt += 1
        if meta.get("cylindrical"):
            warnings.append("%s: cylindrical — power meridian only"
                            % entry["element"])
        if meta.get("approximate"):
            warnings.append("%s: thin-lens approximation"
                            % entry["element"])
        pending_gap = 0.0
        if through_element is not None \
                and entry["element"] == through_element:
            break

    out["n_optical_elements"] = n_opt
    if n_opt == 0:
        return out

    # strip the leading gap before the first element for the SYSTEM
    # matrix (entry-vertex referenced), keep it as the object offset.
    first_gap = seq[0][3] if seq and seq[0][0] == "gap" else 0.0
    M_sys = IDENT
    for tag, M, el, t in seq[1:]:
        M_sys = mmul(M, M_sys)
    card = cardinals_from_matrix(M_sys, None)
    out["efl"] = card["efl"]
    out["bfl"] = card["bfl"]
    out["afocal"] = card["afocal"]
    out["angular_magnification"] = card.get("angular_magnification")

    # ---- image distance + magnification for the given object ------------
    # object_distance_mm is measured to the FIRST element's entry vertex;
    # M_sys is entry-vertex referenced, so the object translation is
    # exactly object_distance_mm (first_gap is only the source's actual
    # chain spacing, reported for callers that want it).
    if math.isinf(object_distance_mm):
        out["image_distance_mm"] = card["bfl"]
        out["magnification"] = 0.0 if not card["afocal"] else None
    else:
        Mo = mmul(M_sys, translate(float(object_distance_mm)))
        a, b, c, d = Mo
        if abs(d) > 1e-15:
            # rays from the axial object point enter as (y=0, w):
            # y(z) = (b + z*d) * w — the image plane is where it vanishes
            z = -b / d
            out["image_distance_mm"] = z
            out["magnification"] = a + z * c
        else:
            out["image_distance_mm"] = math.inf
            out["magnification"] = None
    out["first_gap_mm"] = first_gap

    # ---- marginal-ray aperture search ------------------------------------
    # trial ray: collimated (y=1, w=0) for infinite conjugate; from the
    # axial object point (y=0, w=1) otherwise.
    if math.isinf(object_distance_mm):
        y, w = 1.0, 0.0
    else:
        # from the axial object point: height at first-element entry
        y, w = float(object_distance_mm), 1.0
    tightest = None
    for tag, M, el, t in seq[1:]:
        if tag == "elem":
            # station height BEFORE applying the element
            idx = [s for s in stations if s[0] == el]
            ap = idx[0][1] if idx else None
            if ap and abs(y) > 1e-12:
                ratio = (ap / 2.0) / abs(y)
                if tightest is None or ratio < tightest[1]:
                    tightest = (el, ratio)
        a, b, c, d = M
        y, w = a * y + b * w, c * y + d * w

    if tightest is not None:
        el, scale = tightest
        out["limiting_element"] = el
        # scale trial ray so the limiting aperture is exactly filled;
        # final reduced angle w (air: n=1) IS the image-space NA (paraxial)
        na = abs(w) * scale
        out["na"] = na
        out["fno_working"] = (1.0 / (2.0 * na)) if na > 1e-15 else math.inf

    if detector_gap is not None and out.get("image_distance_mm") is not None:
        out["detector_gap_mm"] = detector_gap

    return out
