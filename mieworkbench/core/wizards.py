"""Lens/element wizard math - pure functions, no Qt.

Sign convention matches the project's meridian builder
(scripts/make_test_scenes.lens_meridian and primitivelib): R_front > 0 is
convex-toward-source (-x), R_back < 0 is convex-toward-detector, None is
flat. That is the standard lensmaker convention, so:

    1/f = (n-1) [ 1/R1 - 1/R2 + (n-1) d / (n R1 R2) ]        (thick lens)
    BFL = f (1 - (n-1) d / (n R1))

The wizard solves the inverse problem (target focal length -> radii) per
lens FORM, then reports the exact resulting EFL/BFL from the thick-lens
formula as a cross-check the UI displays. Oracles: the expected_efl_mm
constants in make_test_scenes.SCENES (pinned against the lensmaker
equation in the original project's test suite).
"""

import math
import sys
from pathlib import Path

D_LINE_NM = 587.6


def index_at(matdb, material, lam_nm=D_LINE_NM):
    """Real refractive index of a library material at lam_nm."""
    return float(matdb.get(material).n_complex(lam_nm * 1e-9).real)


def thick_lens_efl(R1, R2, n, d):
    """Exact thick-lens EFL. R1/R2 in mm (None = flat), d = center
    thickness mm. Returns EFL in mm (inf for an afocal window)."""
    c1 = 0.0 if R1 is None else 1.0 / R1
    c2 = 0.0 if R2 is None else 1.0 / R2
    inv_f = (n - 1.0) * (c1 - c2 + (n - 1.0) * d * c1 * c2 / n)
    if abs(inv_f) < 1e-15:
        return math.inf
    return 1.0 / inv_f


def thick_lens_bfl(R1, R2, n, d):
    """Back focal length (from the rear vertex)."""
    f = thick_lens_efl(R1, R2, n, d)
    if math.isinf(f):
        return math.inf
    c1 = 0.0 if R1 is None else 1.0 / R1
    return f * (1.0 - (n - 1.0) * d * c1 / n)


def _result(R1, R2, n, d, extra=None):
    out = {"R_front": R1, "R_back": R2, "ct": d, "n": n,
           "efl": thick_lens_efl(R1, R2, n, d),
           "bfl": thick_lens_bfl(R1, R2, n, d)}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# per-form solvers: focal length + material index (+ thickness) -> radii
# ---------------------------------------------------------------------------
def solve_pcx(f, n, d):
    """Plano-convex, curved side toward the source. f > 0."""
    if f <= 0:
        raise ValueError("plano-convex needs f > 0")
    return _result((n - 1.0) * f, None, n, d)


def solve_pcv(f, n, d):
    """Plano-concave (flat front, concave back). f < 0."""
    if f >= 0:
        raise ValueError("plano-concave needs f < 0")
    # 1/f = (n-1)(-1/R2)  ->  R2 = -(n-1) f  (> 0: concave back)
    return _result(None, -(n - 1.0) * f, n, d)


def solve_equiconvex(f, n, d):
    """Symmetric biconvex R1 = -R2 = R. f > 0. Exact thick-lens root:
    R = f (n-1) [1 + sqrt(1 - d / (n f))]."""
    if f <= 0:
        raise ValueError("biconvex needs f > 0")
    disc = 1.0 - d / (n * f)
    if disc < 0:
        raise ValueError("no biconvex solution: thickness %.3g mm is too "
                         "large for f = %.3g mm" % (d, f))
    R = f * (n - 1.0) * (1.0 + math.sqrt(disc))
    return _result(R, -R, n, d)


def solve_equiconcave(f, n, d):
    """Symmetric biconcave R1 = -R, R2 = +R. f < 0."""
    if f >= 0:
        raise ValueError("biconcave needs f < 0")
    disc = 1.0 - d / (n * f)      # f < 0 -> disc > 1, always fine
    R = -f * (n - 1.0) * (1.0 + math.sqrt(disc))
    return _result(-R, R, n, d)


def solve_best_form(f, n, d):
    """Minimum-spherical-aberration singlet for an object at infinity:
    thin-lens shape factor q* = 2 (n^2 - 1) / (n + 2), then
    R1 = 2 f (n-1)/(q+1), R2 = 2 f (n-1)/(q-1). Exact EFL reported for
    the chosen radii (slightly off f for thick lenses - shown to the
    user, who can tweak ct)."""
    if f <= 0:
        raise ValueError("best-form solver expects f > 0")
    q = 2.0 * (n * n - 1.0) / (n + 2.0)
    R1 = 2.0 * f * (n - 1.0) / (q + 1.0)
    R2 = 2.0 * f * (n - 1.0) / (q - 1.0)
    return _result(R1, R2, n, d, {"shape_factor": q})


def solve_meniscus(f, n, d, R_front):
    """Meniscus with a chosen front radius; solves R_back exactly from the
    thick-lens formula."""
    c1 = 1.0 / R_front
    # 1/f = (n-1) [c1 - c2 + (n-1) d c1 c2 / n]  -> linear in c2
    k = (n - 1.0) * d * c1 / n
    denom = (n - 1.0) * (k - 1.0)
    if abs(denom) < 1e-15:
        raise ValueError("degenerate meniscus (front surface power "
                         "cancels the thickness term)")
    c2 = (1.0 / f - (n - 1.0) * c1) / denom
    if abs(c2) < 1e-12:
        return _result(R_front, None, n, d)
    return _result(R_front, 1.0 / c2, n, d)


def solve_ball(f, n):
    """Ball lens: EFL = n D / (4 (n-1)) -> D = 4 f (n-1) / n; BFL = EFL -
    D/2 (from the rear surface)."""
    if f <= 0:
        raise ValueError("ball lens needs f > 0")
    D = 4.0 * f * (n - 1.0) / n
    return {"diameter": D, "n": n, "efl": f, "bfl": f - D / 2.0}


# the SCENES lens_asphere reference design (f=40, BK7@633, ct=6): (k, A4)
# solved for the COMPLETE lens (front asphere + flat exit) by exact
# meridional ray trace — k=-n^2 corrected only the front surface and the
# flat exit re-added spherical aberration (see make_test_scenes.py).
_ASPHERE_REF = {"f": 40.0, "k": -1.0, "A4_mm3": 6.586562e-06}


def solve_asphere(f, n, d):
    """Plano-convex asphere: vertex radius from the PCX solution, plus
    the full-lens-corrected front profile (k, A4) scale-transferred from
    the solved f=40 BK7 reference (aspheric sag is scale-invariant, so
    A4 scales as (f_ref/f)^3; exact for BK7-like n with proportionally
    scaled thickness, a good starting point otherwise)."""
    out = solve_pcx(f, n, d)
    out["k"] = _ASPHERE_REF["k"]
    out["A4_mm3"] = _ASPHERE_REF["A4_mm3"] * (_ASPHERE_REF["f"] / f) ** 3
    return out


def solve_cyl(f, n, d):
    """Plano-convex/concave cylinder lens (line focus): same R math as
    the spherical PCX/PCV in the power meridian."""
    if f > 0:
        return {"R": (n - 1.0) * f, "n": n,
                "efl": thick_lens_efl((n - 1.0) * f, None, n, d)}
    return {"R": (n - 1.0) * f, "n": n,     # negative R = concave
            "efl": thick_lens_efl((n - 1.0) * f, None, n, d)}


# shipped achromat reference design (primitivelib lens_achromat defaults):
_ACHROMAT_REF = {"f": 50.0, "R_front": 31.0, "R_iface": -21.956,
                 "R_back": -64.497, "ct_crown": 6.0, "ct_flint": 3.0}

# negative achromat (f < 0): no fixed shipped design to scale (the shipped
# lens_achromat prescription is a positive doublet), so the cemented
# crown/flint doublet is built directly from the V-number achromatic
# power split, cementing thicknesses fixed at the telephoto demo's
# validated rear-group values (make_demos._telephoto_solve, "cemented
# rear-group design study" — an air-spaced dcv+dcx pair failed extraction
# there: the concave rim sag overlapped the next element; cemented is the
# real negative-doublet convention for exactly that reason).
_NEG_ACHROMAT_CT_CROWN = 2.5
_NEG_ACHROMAT_CT_FLINT = 3.5
_NEG_ACHROMAT_GAP = 0.005


def _abbe_number(matdb, material):
    """V = (n_d - 1) / (n_F - n_C) on the Fraunhofer d/F/C lines (587.6 /
    486.1 / 656.3 nm) — the standard dispersion figure the two-thin-lens-
    in-contact achromatic condition is built from."""
    n_d = index_at(matdb, material, D_LINE_NM)
    n_F = index_at(matdb, material, 486.1)
    n_C = index_at(matdb, material, 656.3)
    return (n_d - 1.0) / (n_F - n_C)


def solve_achromat(f, matdb=None):
    """f > 0: scale the shipped BK7/SF5 f=50mm achromat design to a new
    focal length (radii and thicknesses scale linearly with f, preserving
    the achromatic correction, which depends only on the glass pair).

    f < 0: no positive shipped design to scale — a negative (diverging)
    cemented BK7/SF5 doublet is solved directly from the standard
    2-thin-lens-in-contact achromatic condition
        phi_crown = phi_total * V_crown / (V_crown - V_flint)
        phi_flint = phi_total - phi_crown
    (V = Abbe number; this power split is an identity for EITHER sign of
    phi_total = 1/f — for f < 0 it comes out crown-negative, flint-
    positive: a "crown-negative" doublet, matching the telephoto demo's
    validated rear-group construction, make_demos._telephoto_solve).
    The crown is built EQUICONCAVE (thin-lens closed form R = 2(n-1)/
    |phi_crown|, R_front=-R, R_iface=+R); the flint's back radius is then
    solved algebraically off the SAME cemented interface so its own
    thin-lens power exactly completes the split — both radii come from
    closed-form algebra, not a numeric search, so the returned
    prescription's thin-lens EFL matches the requested f to floating-
    point precision (the small thick-lens correction from real center
    thicknesses is the same order as the positive path's nominal-vs-exact
    residual, see test_achromat_close_to_design_focal_length)."""
    if f == 0:
        raise ValueError("achromat solver needs f != 0")
    if f > 0:
        s = f / _ACHROMAT_REF["f"]
        return {"R_front": _ACHROMAT_REF["R_front"] * s,
                "R_iface": _ACHROMAT_REF["R_iface"] * s,
                "R_back": _ACHROMAT_REF["R_back"] * s,
                "ct_crown": _ACHROMAT_REF["ct_crown"] * s,
                "ct_flint": _ACHROMAT_REF["ct_flint"] * s,
                "efl": f}
    db = matdb if matdb is not None else _default_matdb()
    n_crown = index_at(db, "bk7", D_LINE_NM)
    n_flint = index_at(db, "sf5", D_LINE_NM)
    v_crown = _abbe_number(db, "bk7")
    v_flint = _abbe_number(db, "sf5")
    phi_total = 1.0 / f
    phi_crown = phi_total * v_crown / (v_crown - v_flint)
    phi_flint = phi_total - phi_crown
    if abs(phi_crown) < 1e-12:
        raise ValueError("negative achromat solve degenerate at f=%.3g mm "
                         "(zero crown power)" % f)
    a_eq = 2.0 * (n_crown - 1.0) / abs(phi_crown)
    R_front = -a_eq
    R_iface = a_eq
    inv_back = 1.0 / R_iface - phi_flint / (n_flint - 1.0)
    if abs(inv_back) < 1e-12:
        raise ValueError("negative achromat solve degenerate at f=%.3g mm "
                         "(flint back surface would be flat)" % f)
    R_back = 1.0 / inv_back
    return {"R_front": R_front, "R_iface": R_iface, "R_back": R_back,
            "ct_crown": _NEG_ACHROMAT_CT_CROWN,
            "ct_flint": _NEG_ACHROMAT_CT_FLINT, "efl": f}


# ---------------------------------------------------------------------------
# Two-group zoom-pair calculator (future.md (a2), UXNOTES_ROUND3 "no
# magic numbers": "nothing computes p,q,r,s for you"). Idealized thin-lens
# groups of focal length f1_mm (front) / f2_mm (rear); z_mm is the
# THIN-LENS-EQUIVALENT gap -- the separation between the front group's
# own rear principal plane (H2) and the rear group's own front principal
# plane (H1), NOT the chain's vertex-to-vertex distance. For a real
# THICK group (an achromat doublet, say) that's a KNOWN, constant offset
# from the vertex-to-vertex chain gap: z_mm = gap_vertex + pp1_rear -
# pp2_front (both from core.paraxial.element_cardinals), and the group's
# vertex-referenced BFL is bfl_vertex = bfl(z_mm) + pp2_rear -- exact,
# because two principal planes act as a unit-magnification reference pair
# (a standard cascaded-system identity: any thick group looks, from
# outside, exactly like an ideal thin lens of its own EFL sitting at its
# own H2 -- see core/paraxial.py's cardinals_from_matrix header for the
# pp1/pp2 sign convention this relies on). For thin/idealized groups
# (this function's own inputs) that offset is simply zero.
# ---------------------------------------------------------------------------
def _solve_quadratic(a, b, c):
    """Real roots of a*z^2 + b*z + c = 0 (falls back to linear/empty for
    degenerate a); returns a (possibly empty) list, ascending order."""
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    r1, r2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
    return sorted([r1, r2])


def solve_zoom_pair(f1_mm, f2_mm, z_mm=None, track_mm=None):
    """Two-thin-lens-group zoom relationship (collimated input): BFL(z),
    EFL(z), and total track z + BFL(z) as the inter-group gap z varies,
    for idealized thin groups of focal length f1_mm (front) / f2_mm
    (rear) -- see the module-level note above for the exact vertex-gap
    correction when applying this to real thick groups (achromat pairs).

    BFL(z) = -(pA + qA*z) / (pC + qC*z),  EFL(z) = -1/(pC + qC*z)
    (both affine-over-affine; the coefficients come straight out of the
    two-thin-lens ABCD product -- see core/paraxial.py's mmul/thin/
    translate for the same machinery). `bfl_expr`/`efl_expr` are ready
    train-grammar expression strings (the telephoto_zoom demo's own
    `distance` field convention, make_demos.demo_telephoto_zoom).

    z_mm (optional): evaluate bfl_mm/efl_mm/track_mm at this specific gap.
    track_mm (optional): solve the gap(s) z_for_track (0, 1, or 2 real
    roots, ascending) whose total track z + BFL(z) equals track_mm --
    track(z) = T rearranges to the quadratic
        qC*z^2 + (pC - qA - T*qC)*z - (pA + T*pC) = 0.
    """
    if f1_mm == 0 or f2_mm == 0:
        raise ValueError("zoom pair needs both focal lengths nonzero")
    pA, qA = 1.0, -1.0 / f1_mm
    pC, qC = -1.0 / f1_mm - 1.0 / f2_mm, 1.0 / (f1_mm * f2_mm)

    def bfl(z):
        denom = pC + qC * z
        return math.inf if abs(denom) < 1e-15 else -(pA + qA * z) / denom

    def efl(z):
        denom = pC + qC * z
        return math.inf if abs(denom) < 1e-15 else -1.0 / denom

    def track(z):
        b = bfl(z)
        return math.inf if math.isinf(b) else z + b

    out = {"f1_mm": f1_mm, "f2_mm": f2_mm,
           "pA": pA, "qA": qA, "pC": pC, "qC": qC,
           "bfl_expr": "-(%.10g + %.10g*z)/(%.10g + %.10g*z)"
                      % (pA, qA, pC, qC),
           "efl_expr": "-1/(%.10g + %.10g*z)" % (pC, qC),
           "bfl": bfl, "efl": efl, "track": track}
    if z_mm is not None:
        out["z_mm"] = z_mm
        out["bfl_mm"] = bfl(z_mm)
        out["efl_mm"] = efl(z_mm)
        out["track_mm"] = track(z_mm)
    if track_mm is not None:
        a2 = qC
        a1 = pC - qA - track_mm * qC
        a0 = -(pA + track_mm * pC)
        out["z_for_track"] = _solve_quadratic(a2, a1, a0)
    return out


def solve_fresnel(f, n, aperture, n_facets=12):
    """Fresnel lens: the design focal length and index feed the facet
    slopes directly (primitivelib lens_fresnel params)."""
    if f <= 0:
        raise ValueError("fresnel needs f > 0")
    return {"f_design": f, "n_design": n, "aperture": aperture,
            "n_facets": n_facets, "efl": f}


# form registry the wizard UI iterates: name -> (label, solver kwargs)
LENS_FORMS = {
    "pcx": {"label": "Plano-convex", "solver": solve_pcx,
            "primitive": "lens_pcx",
            "map": lambda r: {"R_front": r["R_front"], "ct": r["ct"]}},
    "pcv": {"label": "Plano-concave", "solver": solve_pcv,
            "primitive": "lens_pcv",
            "map": lambda r: {"R_back": r["R_back"], "ct": r["ct"]}},
    "dcx": {"label": "Biconvex (symmetric)", "solver": solve_equiconvex,
            "primitive": "lens_dcx",
            "map": lambda r: {"R_front": r["R_front"],
                              "R_back": -r["R_back"], "ct": r["ct"]}},
    "dcv": {"label": "Biconcave (symmetric)", "solver": solve_equiconcave,
            "primitive": "lens_dcv",
            "map": lambda r: {"R_front": -r["R_front"],
                              "R_back": r["R_back"], "ct": r["ct"]}},
    "best": {"label": "Best-form singlet", "solver": solve_best_form,
             "primitive": "lens_meniscus",
             "map": lambda r: {"R_front": r["R_front"],
                               "R_back": r["R_back"], "ct": r["ct"]}},
    "asphere": {"label": "Aspheric (conic + A4)", "solver": solve_asphere,
                "primitive": "lens_asphere",
                "map": lambda r: {"R": r["R_front"], "k": r["k"],
                                  "A4_mm3": r["A4_mm3"], "ct": r["ct"]}},
    "ball": {"label": "Ball lens", "solver": solve_ball,
             "primitive": "lens_ball",
             "map": lambda r: {"diameter": r["diameter"]}},
    "achromat": {"label": "Achromatic doublet", "solver": solve_achromat,
                 "primitive": "lens_achromat",
                 "map": lambda r: {k: r[k] for k in
                                   ("R_front", "R_iface", "R_back",
                                    "ct_crown", "ct_flint")}},
    "fresnel": {"label": "Fresnel lens", "solver": solve_fresnel,
                "primitive": "lens_fresnel",
                "map": lambda r: {"f_design": r["f_design"],
                                  "n_design": r["n_design"],
                                  "aperture": r["aperture"],
                                  "n_facets": r["n_facets"]}},
    "cyl": {"label": "Cylindrical", "solver": solve_cyl,
            "primitive": "lens_cyl",
            "map": lambda r: {"R": r["R"]}},
}


def design_lens(form, f_mm, matdb=None, material="bk7", lam_nm=D_LINE_NM,
                ct_mm=None, **kw):
    """One-call wizard backend: form + focal length (+ material/thickness)
    -> {"primitive": kind, "params": {alias: value}, "design": raw solver
    output}. The params dict maps straight onto the primitive's dim-sheet
    aliases."""
    if form not in LENS_FORMS:
        raise ValueError("unknown lens form %r (know: %s)"
                         % (form, ", ".join(sorted(LENS_FORMS))))
    spec = LENS_FORMS[form]
    solver = spec["solver"]
    n = index_at(matdb, material, lam_nm) if matdb is not None \
        else kw.pop("n", 1.51680)
    if form == "achromat":
        design = solver(f_mm)
    elif form == "ball":
        design = solver(f_mm, n)
    elif form == "fresnel":
        design = solver(f_mm, n, kw.get("aperture", 24.0),
                        kw.get("n_facets", 12))
    elif form == "best":
        design = solver(f_mm, n, ct_mm if ct_mm is not None else 4.0)
    elif form == "cyl":
        design = solver(f_mm, n, ct_mm if ct_mm is not None else 5.0)
        design["ct"] = ct_mm if ct_mm is not None else 5.0
    else:
        d = ct_mm if ct_mm is not None else (5.0 if f_mm > 0 else 3.0)
        design = solver(f_mm, n, d)
    params = spec["map"](design)
    return {"primitive": spec["primitive"], "params": params,
            "design": design}


# ---------------------------------------------------------------------------
# Waveplate thickness solver (primitivelib 'waveplate' kind, quartz by
# default). Pure math like the rest of this module -- the only non-pure bit
# is lazily loading the real opticalproperties/ birefringence registry
# (raytracer.optprops.load_optical_properties) so the o/e indices used are
# the SAME ones the ray tracer itself uses, not a re-typed constant. Follows
# proplib.py's sys.path-insert-then-import pattern (same scripts/ layout).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")


def _default_matdb():
    """Lazily load the shipped opticalproperties/ library's MaterialDB (with
    uniaxial birefringence attached) the first time a caller needs one and
    doesn't supply their own -- cached for the process lifetime."""
    global _DEFAULT_MATDB
    try:
        return _DEFAULT_MATDB
    except NameError:
        pass
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    from raytracer.optprops import load_optical_properties
    _DEFAULT_MATDB = load_optical_properties().matdb
    return _DEFAULT_MATDB


def birefringence_at(matdb, crystal, lam_nm):
    """(n_o, n_e): real refractive indices of a registered uniaxial crystal
    at lam_nm, from the birefringence registry (matdb.get_uniaxial, set up
    by raytracer.optprops.load_uniaxial/attach_uniaxial)."""
    mo, me = matdb.get_uniaxial(crystal)
    lam_m = lam_nm * 1e-9
    return (float(mo.n_complex(lam_m).real), float(me.n_complex(lam_m).real))


WAVEPLATE_RETARDANCE_WAVES = {"half": 0.5, "quarter": 0.25}


def waveplate_thickness(kind, lambda_nm, order=0, matdb=None,
                        crystal="quartz"):
    """Retarder thickness (primitivelib 'waveplate' kind's 'thickness' dim,
    which sets its retardance) for a zero- or multi-order uniaxial
    waveplate.

    kind: 'half' (0.5 wave retardance) or 'quarter' (0.25 wave).
    order: non-negative integer; total retardance = order + the fractional
    waves above, e.g. a first-order half-wave plate (order=1) retards
    1.5 waves.
    matdb: a MaterialDB with uniaxial birefringence attached (as returned by
    raytracer.optprops.load_optical_properties(...).matdb); defaults to a
    lazily-loaded copy of the shipped opticalproperties/ library so this is
    usable standalone (waveplate_thickness("half", 633.0)).
    crystal: birefringence-registry crystal name (default 'quartz', matching
    the primitivelib 'waveplate' kind's default material).

    thickness_mm = (order + retardance_waves) * lambda_nm * 1e-6 / |n_e-n_o|
    using |n_e - n_o| evaluated AT lambda_nm (not a fixed d-line constant),
    so multi-wavelength designs stay exact despite quartz's (weak)
    birefringence dispersion.
    """
    retardance = WAVEPLATE_RETARDANCE_WAVES.get(kind)
    if retardance is None:
        raise ValueError("waveplate kind must be one of %s, got %r"
                         % (sorted(WAVEPLATE_RETARDANCE_WAVES), kind))
    if order < 0 or int(order) != order:
        raise ValueError("order must be a non-negative integer, got %r"
                         % (order,))
    db = matdb if matdb is not None else _default_matdb()
    n_o, n_e = birefringence_at(db, crystal, lambda_nm)
    delta_n = abs(n_e - n_o)
    if delta_n <= 0.0:
        raise ValueError("%s has zero birefringence at %.1f nm"
                         % (crystal, lambda_nm))
    waves = order + retardance
    thickness_mm = waves * lambda_nm * 1e-6 / delta_n
    return {"thickness": thickness_mm, "waves": waves, "n_o": n_o,
            "n_e": n_e, "delta_n": delta_n, "kind": kind,
            "order": order, "lambda_nm": lambda_nm, "crystal": crystal}
