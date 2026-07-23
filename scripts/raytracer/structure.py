# =============================================================================
# structure.py — inter-particle structure factor S(q) for the scattering-
# sample round (registry: sample/samples.miesamp, loader: optprops.py
# load_samples). Numpy-only, vectorized over q arrays.
#
# UNITS, LOUDLY: q is in INVERSE MICROMETRES (1/um) EVERYWHERE in this file.
# Radii/lengths (r_hs_um, xi_um, r0_um, a_um) are in MICROMETRES. This
# matches the engine-internal length unit (particles.py, mie.py) so that
# q = 4*pi*n_host/lambda_um * sin(theta/2) is directly in 1/um at trace
# time with no extra conversion. Passing q in 1/nm or 1/Angstrom (the usual
# SAXS/SANS convention) will silently give nonsense peak positions.
#
# S(q) multiplies the independent-scatterer (Rayleigh-Gans / Mie ensemble)
# cross section: I(q) = N * P(q) * S(q). model='none' -> S(q) == 1
# (uncorrelated scatterers, the pre-existing particles.py behaviour).
#
# Models implemented (each a pure, vectorized function of q):
#   sq_percus_yevick  monodisperse hard-sphere PY closure (Wertheim 1963 /
#                     Thiele 1963, closed form via Ashcroft & Lekner 1966 /
#                     Kinning & Thomas 1984 Macromolecules 17, 1712).
#   sq_baxter         Baxter (1968) J. Chem. Phys. 49, 2770 sticky hard
#                     sphere, PY closure — see the JUDGMENT CALL note below.
#   sq_fractal        Teixeira (1988) J. Appl. Cryst. 21, 781 fractal
#                     aggregate / Ornstein-Zernike gel.
#   sq_paracrystal    powder-averaged ideal paracrystal, Hosemann disorder
#                     (Hosemann & Bagchi 1962) — see the JUDGMENT CALL note.
#   sq_table          tabulated S(q), linear interp, clamped outside range.
#   sq_evaluate       one dispatcher over the samples-registry (model,
#                     params) pair, used by the ensemble-tables integration.
#
# JUDGMENT CALL — sq_baxter: Baxter's original 1968 solution factorizes the
# OZ equation through an auxiliary function Q(r) and a boundary-value
# problem at contact; reconstructing that Q(r) algebra exactly (rather than
# from a primary-source copy) is error-prone. What IS reproduced exactly
# here, independently re-derived and cross-checked two ways (the tau -> inf
# limit AND the low-density second-virial expansion B2 = B2_HS*(1-1/(4tau))
# both match to machine precision — see test_structure.py and the module
# comment above sq_baxter): the stickiness quadratic for Baxter's lambda,
# and the resulting mu = phi*lambda enters S(q) through a contact-
# scattering ("sticky delta-shell") term added to the exact PY hard-sphere
# background, amplitude Gamma = 2*mu/(1-phi)^2 chosen to match the EXACT
# low-density (leading-order-in-phi) limit of Baxter's second-virial
# relation B2 = B2_HS*(1-1/(4*tau)) (independently re-derived from the
# standard compressibility virial expansion S(0) ~ 1-2*B2*n; see the
# module tests). Baxter's own S(0) formula has further phi-dependent terms
# beyond this leading order that were NOT reproduced here: an early
# attempt at the fuller nonlinear compressibility expression gave
# NEGATIVE S(0) at ordinary literature test points (phi=0.1, tau=0.15),
# i.e. it was wrong, and rather than ship an unverified nonlinear formula
# this implementation deliberately stays at the term it COULD verify two
# independent ways. A small Gaussian envelope keeps the
# shell's oscillatory tail from ever driving S(q) negative at the q values
# where the plain sinc tail would overshoot (checked on an internal dense
# probe grid — sq_baxter raises ValueError if no envelope choice keeps
# S(q) > 0, which is the practical extension of "unphysical tau/phi combo"
# beyond the bare discriminant). This is NOT a claim to reproduce Baxter's
# exact Q(r) algebra at every q; it is a controlled, positivity-guaranteed,
# S(0)-exact approximation good enough for a Monte-Carlo ray tracer's
# ensemble structure factor. Root selection: Baxter's lambda quadratic has
# two roots; the SMALLER one is physical (it is the branch that -> 0 as
# tau -> infinity, i.e. no residual stickiness when the well vanishes; the
# larger root diverges in that limit and is spurious).
#
# JUDGMENT CALL — sq_paracrystal: this is a PRAGMATIC powder paracrystal.
# Peak POSITIONS and selection rules are exact (conventional-cell cubic
# reciprocal lattice, standard fcc/bcc/sc systematic absences, generated
# and grouped by |q| shell so accidentally-degenerate (hkl) triples merge
# correctly for a powder average). The LINESHAPE is a pragmatic
# pseudo-Voigt-free choice: each allowed shell contributes a normalized
# Lorentzian whose width grows as g^2 * q_hkl^2 (Hosemann's paracrystalline
# rule that higher-order reflections broaden as the ORDER SQUARED) and
# whose intensity is Debye-Waller-damped as exp(-g^2 * n_hkl^2) (n_hkl =
# q_hkl / (2*pi/a), the shell's order index). This is not a derivation of
# Hosemann's full paracrystal line-shape theory (which is a proper convolu-
# tion, not a bare sum of Lorentzians); it is chosen because it is smooth,
# always positive (a sum of positive Lorentzians plus 1 can never go
# negative), reduces correctly to sharp Bragg peaks at exact literature
# positions as g -> 0, and washes out to a smooth liquid-like curve as g
# grows — exactly the qualitative behaviour the registry contract asks for.
# At small g the many-shell sum can still be visibly elevated far out along
# a densely-populated lattice's higher orders (most noticeable for sc,
# which has no systematic absences) before eventually decaying past
# n_hkl ~ 1/g; this only matters deep in the tail and does not affect the
# first few peaks tested here.
# =============================================================================
import numpy as np
from scipy.special import gamma as _gamma_fn

__all__ = ["sq_percus_yevick", "sq_baxter", "sq_fractal", "sq_paracrystal",
           "sq_table", "sq_evaluate"]

# A = q*sigma below this uses the Taylor series (the closed form's
# sin/cos - polynomial cancellation loses all precision well before this;
# see the module tests for the calibration).
_A_SMALL = 1e-2
# q*xi below this uses the analytic fractal S(0) limit.
_FRACTAL_X_SMALL = 1e-3
# dense internal probe grid sq_baxter uses to guarantee S(q) > 0 for ALL q,
# not just the caller's query points (a physically-required property of
# any structure factor, not merely a courtesy at q=0).
_BAXTER_PROBE = np.linspace(1e-3, 60.0, 4000)


# ---------------------------------------------------------------------------
# Percus-Yevick monodisperse hard sphere
# ---------------------------------------------------------------------------
def _py_direct_correlation(A, eta):
    """n*c(q) for PY hard spheres, A = q*sigma (sigma = hard-sphere
    diameter). Closed form (Ashcroft & Lekner 1966 / Kinning & Thomas
    1984): c(r) = -(alpha + beta*(r/sigma) + gamma*(r/sigma)^3) for
    r<sigma, its Fourier transform reduces to

      C(A) = -24*eta*[ alpha/A^3 (sinA - A cosA)
                      + beta/A^4 (2A sinA + (2-A^2) cosA - 2)
                      + gamma/A^6 ((4A^3-24A) sinA
                                   - (A^4-12A^2+24) cosA + 24) ]

    with alpha = (1+2eta)^2/(1-eta)^4, beta = -6eta(1+eta/2)^2/(1-eta)^4,
    gamma = eta*alpha/2. The bracketed sin/cos combination is a removable
    0/0 at A=0 in exact arithmetic but a catastrophic-cancellation trap in
    float64 well before A actually reaches 0 (verified against the exact
    compressibility limit in test_structure.py); below _A_SMALL this uses
    the Taylor series (matched to O(A^2), decent to O(A^4) on two of the
    three terms) instead, giving an exactly finite, NaN-free A=0 value.
    """
    A = np.asarray(A, dtype=np.float64)
    alpha = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    beta = -6.0 * eta * (1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4
    gamma = eta * alpha / 2.0

    small = A < _A_SMALL
    Asafe = np.where(small, 1.0, A)   # dummy value where small (unused)
    sinA, cosA = np.sin(Asafe), np.cos(Asafe)
    f1 = (sinA - Asafe * cosA) / Asafe ** 3
    f2 = (2.0 * Asafe * sinA + (2.0 - Asafe ** 2) * cosA - 2.0) / Asafe ** 4
    f3 = ((4.0 * Asafe ** 3 - 24.0 * Asafe) * sinA
          - (Asafe ** 4 - 12.0 * Asafe ** 2 + 24.0) * cosA + 24.0) \
        / Asafe ** 6
    C_exact = -24.0 * eta * (alpha * f1 + beta * f2 + gamma * f3)

    A2 = A * A
    f1s = 1.0 / 3.0 - A2 / 30.0 + A2 * A2 / 840.0
    f2s = 1.0 / 4.0 - A2 / 36.0 + A2 * A2 / 960.0
    f3s = 1.0 / 6.0 - A2 / 48.0
    C_series = -24.0 * eta * (alpha * f1s + beta * f2s + gamma * f3s)

    return np.where(small, C_series, C_exact)


def sq_percus_yevick(q, r_hs_um, phi_hs):
    """Monodisperse hard-sphere Percus-Yevick structure factor.

    q: array-like, 1/um. r_hs_um: hard-sphere radius, um (sigma = 2*r).
    phi_hs: volume (packing) fraction, (0, ~0.74).

    S(0) equals the exact compressibility limit (1-phi)^4/(1+2phi)^2 (to
    the series' truncation order at A < _A_SMALL, effectively machine
    precision — pinned by test_structure.py). S(q) > 0 everywhere; at
    phi=0.49 the first peak is ~2.5-3.2 near q*sigma ~ 6.5-7.3 (Kinning &
    Thomas / Hansen & McDonald "Theory of Simple Liquids" ballpark).
    """
    q = np.asarray(q, dtype=np.float64)
    sigma = 2.0 * r_hs_um
    A = q * sigma
    Cq = _py_direct_correlation(A, phi_hs)
    return 1.0 / (1.0 - Cq)


# ---------------------------------------------------------------------------
# Baxter sticky hard sphere (PY closure) — see the module-header judgment
# call note for exactly what is and is not reproduced from Baxter (1968).
# ---------------------------------------------------------------------------
def _baxter_lambda(phi_hs, tau_stick):
    """Solve Baxter's stickiness quadratic for lambda (his own symbol,
    unrelated to optical wavelength):

        (phi/12)*lambda^2 - [tau + phi/(1-phi)]*lambda
                           + (1+phi/2)/(1-phi)^2 = 0

    Returns the SMALLER (physical) root — the one that -> 0 as
    tau -> infinity, recovering the plain PY hard sphere exactly; the
    larger root diverges in that limit and is unphysical. Raises
    ValueError if the discriminant is negative (no real lambda: this
    tau/phi combination has no sticky-sphere PY solution)."""
    if tau_stick <= 0:
        raise ValueError("sq_baxter: tau_stick must be > 0, got %g"
                          % tau_stick)
    Aq = phi_hs / 12.0
    Bq = tau_stick + phi_hs / (1.0 - phi_hs)
    Cq = (1.0 + phi_hs / 2.0) / (1.0 - phi_hs) ** 2
    disc = Bq * Bq - 4.0 * Aq * Cq
    if disc < 0.0:
        raise ValueError(
            "sq_baxter: no physical solution for tau_stick=%g, phi_hs=%g "
            "(stickiness quadratic discriminant < 0 -- too sticky for "
            "this packing fraction under the PY closure)"
            % (tau_stick, phi_hs))
    return (Bq - np.sqrt(disc)) / (2.0 * Aq)


def sq_baxter(q, r_hs_um, phi_hs, tau_stick):
    """Baxter (1968) sticky hard sphere, PY closure.

    q: 1/um. r_hs_um: hard-sphere radius, um. phi_hs: packing fraction.
    tau_stick: Baxter's stickiness (> 0; smaller = stickier; -> infinity
    recovers sq_percus_yevick exactly, pinned to 1e-6 by test_structure.py).

    Raises ValueError for an unphysical (tau_stick, phi_hs) combination:
    either the stickiness quadratic has no real root, or (a further,
    practical check -- see the module header) no contact-shell envelope
    keeps S(q) > 0 everywhere.
    """
    q = np.asarray(q, dtype=np.float64)
    sigma = 2.0 * r_hs_um
    lam = _baxter_lambda(phi_hs, tau_stick)
    mu = phi_hs * lam
    # Amplitude of the contact (sticky-shell) term -- see the module
    # header for the derivation/verification of this leading-order form.
    Gamma = 2.0 * mu / (1.0 - phi_hs) ** 2

    # Gaussian envelope width (dimensionless in A=q*sigma): small enough
    # to leave S(0) exact (envelope(0)=1) and to leave tau->inf -> Gamma=0
    # untouched, large enough that the shell's sinc(A) tail cannot drive
    # 1-C(A) negative at any A for the (phi,Gamma) combinations the probe
    # grid below actually accepts.
    damp = 0.3

    def _Cq(Aarr):
        shell = np.sinc(Aarr / np.pi) * np.exp(-(damp * Aarr) ** 2)
        return _py_direct_correlation(Aarr, phi_hs) + Gamma * shell

    probe_S = 1.0 / (1.0 - _Cq(_BAXTER_PROBE))
    if not np.all(np.isfinite(probe_S)) or np.any(probe_S <= 0.0):
        raise ValueError(
            "sq_baxter: tau_stick=%g, phi_hs=%g gives a non-positive S(q) "
            "somewhere -- unphysical combo" % (tau_stick, phi_hs))

    A = q * sigma
    return 1.0 / (1.0 - _Cq(A))


# ---------------------------------------------------------------------------
# Teixeira (1988) fractal aggregate / Ornstein-Zernike gel
# ---------------------------------------------------------------------------
def sq_fractal(q, xi_um, df, r0_um):
    """Teixeira (1988) J. Appl. Cryst. 21, 781 mass-fractal aggregate
    structure factor:

      S(q) = 1 + [df*Gamma(df-1)/(q*r0)^df]
                 * sin[(df-1)*arctan(q*xi)] / [1+1/(q*xi)^2]^((df-1)/2)

    q, 1/um. xi_um: fractal (cutoff) correlation length, um. df: fractal
    dimension, (1, 3]. r0_um: primary-particle radius, um.

    The bracketed ratio is a removable 0/0 as q -> 0 (both arctan and the
    denominator power blow up/vanish together); below q*xi < _FRACTAL_X_
    SMALL this returns the analytic small-q limit instead, derived by
    Taylor-expanding arctan and the power term to leading order:

      S(0) = 1 + df*Gamma(df)*(xi/r0)^df

    (using (df-1)*Gamma(df-1) = Gamma(df); verified against the raw
    formula numerically down to q*xi ~ 1e-6 in test_structure.py). In the
    fractal window 1/xi << q << 1/r0, arctan(q*xi) -> pi/2 and the power
    term -> 1, so S(q)-1 ~ q^(-df): a straight line of slope -df on a
    log-log plot, the fractal dimension's usual scattering signature.
    """
    q = np.asarray(q, dtype=np.float64)
    x = q * xi_um
    limit = 1.0 + df * _gamma_fn(df) * (xi_um / r0_um) ** df

    small = np.abs(x) < _FRACTAL_X_SMALL
    xsafe = np.where(small, 1.0, x)
    qsafe = np.where(small, 1.0 / xi_um, q)   # keeps (q*r0)^df finite too
    ratio = np.sin((df - 1.0) * np.arctan(xsafe)) \
        / (1.0 + 1.0 / xsafe ** 2) ** ((df - 1.0) / 2.0)
    S_exact = 1.0 + df * _gamma_fn(df - 1.0) / (qsafe * r0_um) ** df * ratio

    return np.where(small, limit, S_exact)


# ---------------------------------------------------------------------------
# Powder-averaged ideal paracrystal, Hosemann disorder
# ---------------------------------------------------------------------------
_LATTICE_EXTRA_SHELLS = 4     # margin of shells beyond the requested q max
_PARA_SIGMA_FLOOR = 1e-6      # 1/um; keeps a shell's Lorentzian finite-width


def _lattice_allowed(lattice, h, k, l):
    if lattice == "sc":
        return True
    if lattice == "bcc":
        return (h + k + l) % 2 == 0
    if lattice == "fcc":
        return (h % 2) == (k % 2) == (l % 2)
    raise ValueError("sq_paracrystal: lattice must be fcc/bcc/sc, got %r"
                     % (lattice,))


def _lattice_shells(lattice, n_shells):
    """-> sorted [(h^2+k^2+l^2, multiplicity), ...] for the conventional
    cubic cell, h,k,l in [-n_shells, n_shells] (excluding 0,0,0), grouped
    by |q|-shell (accidentally-degenerate (hkl) triples at the same
    h^2+k^2+l^2 but different Miller indices merge automatically, which
    is exactly what a POWDER average needs — only |q| matters)."""
    shells = {}
    rng = range(-n_shells, n_shells + 1)
    for h in rng:
        for k in rng:
            for l in rng:
                if h == 0 and k == 0 and l == 0:
                    continue
                if not _lattice_allowed(lattice, h, k, l):
                    continue
                s2 = h * h + k * k + l * l
                shells[s2] = shells.get(s2, 0) + 1
    return sorted(shells.items())


def sq_paracrystal(q, lattice, a_um, g):
    """Powder-averaged ideal paracrystal with Hosemann disorder — see the
    module-header judgment-call note for the honest scope of this model
    (exact peak positions/selection rules, pragmatic Lorentzian lineshape).

    q: 1/um. lattice: 'fcc'|'bcc'|'sc' (conventional cubic cell,
    standard selection rules: fcc h,k,l same parity; bcc h+k+l even;
    sc unrestricted). a_um: lattice constant, um. g: Hosemann disorder
    parameter, (0,1) -- small g = sharp Bragg peaks, g -> 1 = smooth
    liquid-like.

    Each allowed shell at q_hkl = 2*pi*sqrt(h^2+k^2+l^2)/a contributes a
    unit-area Lorentzian of half-width sigma_hkl = g^2*q_hkl^2/q1
    (q1 = 2*pi/a; Hosemann's rule that higher orders broaden as the
    order SQUARED), weighted by multiplicity and a Debye-Waller-like
    exp(-g^2*n_hkl^2) order damping (n_hkl = q_hkl/q1). S(q) = 1 + (sum
    of these, all positive) is therefore ALWAYS > 0 by construction, and
    -> 1 away from every peak (each Lorentzian vanishes off-shell).
    """
    q = np.asarray(q, dtype=np.float64)
    q1 = 2.0 * np.pi / a_um
    qmax = float(np.max(q)) if q.size else 0.0
    n_max = qmax / q1
    n_shells = int(np.ceil(n_max)) + _LATTICE_EXTRA_SHELLS

    S = np.ones_like(q)
    for s2, mult in _lattice_shells(lattice, n_shells):
        n_hkl = np.sqrt(float(s2))
        q_hkl = q1 * n_hkl
        sigma = max(g * g * q_hkl * q_hkl / q1, _PARA_SIGMA_FLOOR)
        weight = mult * np.exp(-g * g * n_hkl * n_hkl)
        lorentzian = (sigma / np.pi) / ((q - q_hkl) ** 2 + sigma * sigma)
        S = S + weight * lorentzian
    return S


# ---------------------------------------------------------------------------
# tabulated S(q)
# ---------------------------------------------------------------------------
def sq_table(q, q_tab_per_um, s_tab):
    """Linear interpolation of a tabulated S(q). BEYOND the tabulated
    range this CLAMPS to the nearest endpoint value (never extrapolates,
    so a poorly-chosen table can never go negative) -- authors should set
    s_tab[-1] = 1 so the high-q clamp is physically sensible."""
    q = np.asarray(q, dtype=np.float64)
    q_tab_per_um = np.asarray(q_tab_per_um, dtype=np.float64)
    s_tab = np.asarray(s_tab, dtype=np.float64)
    return np.interp(q, q_tab_per_um, s_tab,
                      left=s_tab[0], right=s_tab[-1])


# ---------------------------------------------------------------------------
# dispatcher — model/params exactly as the samples registry stores them
# (optprops.load_samples' "sq_model"/"sq_params"); context supplies
# trace-time defaults for params the registry allows to be omitted.
# ---------------------------------------------------------------------------
def sq_evaluate(model, params, q, context=None):
    """S(q) for one samples-registry (sq_model, sq_params) pair.

    params: the validated sq_params dict from optprops.load_samples
    (model 'table' carries the resolved arrays under
    params["table_data"] = {"q_per_um", "s"}).
    context: optional {"phi_v": trace-time volume fraction,
    "r_mean_um": trace-time volume-weighted mean particle radius} used to
    fill in phi_hs/r_hs_um/r0_um when the registry row omitted them.

    Raises ValueError for an unknown model or missing required params.
    """
    context = context or {}
    q = np.asarray(q, dtype=np.float64)

    if model == "none":
        return np.ones_like(q)

    if model in ("py", "baxter"):
        phi_hs = params.get("phi_hs")
        if phi_hs is None:
            phi_hs = context.get("phi_v")
        r_hs_um = params.get("r_hs_um")
        if r_hs_um is None:
            r_hs_um = context.get("r_mean_um")
        if phi_hs is None or r_hs_um is None:
            raise ValueError(
                "sq_evaluate: model=%s needs phi_hs/r_hs_um (from "
                "sq_params or context phi_v/r_mean_um)" % model)
        if model == "py":
            return sq_percus_yevick(q, r_hs_um, phi_hs)
        return sq_baxter(q, r_hs_um, phi_hs, params["tau_stick"])

    if model == "fractal":
        r0_um = params.get("r0_um")
        if r0_um is None:
            r0_um = context.get("r_mean_um")
        if r0_um is None:
            raise ValueError(
                "sq_evaluate: model=fractal needs r0_um (from sq_params "
                "or context r_mean_um)")
        return sq_fractal(q, params["xi_um"], params["df"], r0_um)

    if model == "paracrystal":
        return sq_paracrystal(q, params["lattice"], params["a_um"],
                              params["g"])

    if model == "table":
        td = params["table_data"]
        return sq_table(q, td["q_per_um"], td["s"])

    raise ValueError("sq_evaluate: unknown sq_model %r" % (model,))
