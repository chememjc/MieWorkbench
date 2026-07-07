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


def solve_asphere(f, n, d):
    """Plano-convex conic asphere: vertex radius from the PCX solution,
    conic k = -n^2 (kills on-axis spherical aberration for a collimated
    beam; the original project's lens_asphere convention)."""
    out = solve_pcx(f, n, d)
    out["k"] = -(n * n)
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


def solve_achromat(f):
    """Scale the shipped BK7/SF5 f=50mm achromat design to a new focal
    length (radii and thicknesses scale linearly with f, preserving the
    achromatic correction, which depends only on the glass pair)."""
    if f <= 0:
        raise ValueError("achromat solver needs f > 0")
    s = f / _ACHROMAT_REF["f"]
    return {"R_front": _ACHROMAT_REF["R_front"] * s,
            "R_iface": _ACHROMAT_REF["R_iface"] * s,
            "R_back": _ACHROMAT_REF["R_back"] * s,
            "ct_crown": _ACHROMAT_REF["ct_crown"] * s,
            "ct_flint": _ACHROMAT_REF["ct_flint"] * s,
            "efl": f}


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
    "asphere": {"label": "Aspheric (conic)", "solver": solve_asphere,
                "primitive": "lens_asphere",
                "map": lambda r: {"R": r["R_front"], "k": r["k"],
                                  "ct": r["ct"]}},
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
