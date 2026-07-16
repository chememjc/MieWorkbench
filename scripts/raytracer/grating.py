# =============================================================================
# grating.py — vector diffraction-grating equation and idealized lamellar
# efficiency model. Pure functions only: no tracer/scene imports. The tracer
# wires these in against the parsed dicts from common.parse_grating_spec.
#
# Conventions (must match fresnel.py exactly, since order m=0 is required to
# reduce EXACTLY to Snell refraction / specular reflection):
#   * d      : unit incident ray direction
#   * n_hat  : unit surface normal pointing INTO the incident medium
#              (cos_i = -d . n_hat >= 0)
#   * g_hat  : unit groove-PERIODICITY vector, lying in the local tangent
#              plane (perpendicular to n_hat), pointing along the direction
#              the groove pattern repeats in (perpendicular to the groove
#              lines themselves)
#   * n1, n2 : real refractive indices of the incident / far media
#
# Vector grating equation (all quantities normalized by the vacuum
# wavenumber k0 = 2*pi/lam so we work with dimensionless "optical direction
# cosines" n*dir instead of raw k-vectors — avoids carrying huge k0 factors
# through the arithmetic):
#
#   T = n1 * d_tang + m * (lam / d_period) * g_hat        (tangential part)
#
# where d_tang = d - (d.n_hat) n_hat is the tangential projection of d and
# d_period = 1e-3 / lines_per_mm [m] is the groove period. The order m is
# propagating on the transmitted side iff n2^2 - |T|^2 >= 0, and on the
# reflected side iff n1^2 - |T|^2 >= 0 (grating orders can be evanescent on
# either side independently of the other).
#
#   dirs_t = (T - sqrt(max(n2^2-|T|^2,0)) * n_hat) / n2
#   dirs_r = (T + sqrt(max(n1^2-|T|^2,0)) * n_hat) / n1
#
# At m=0, T = n1*d_tang and these reduce algebraically (see test suite) to
# refract_dir / reflect_dir exactly:
#   dirs_t|_{m=0} = (n1/n2)*d_tang - cos_t*n_hat  == refract_dir(...)
#   dirs_r|_{m=0} = d_tang + cos_i*n_hat           == reflect_dir(...)
# =============================================================================
#
# Polarization- and wavelength-resolved efficiency models (README §1.9):
#   order_efficiencies(spec, lam, cos_i, orders) dispatches on spec["model"]:
#     "lamellar"       -> analytic binary-phase / explicit list, eta_s == eta_p
#     "bragg_kogelnik" -> Kogelnik 1969 coupled-wave, TRANSMISSION geometry
#     "dammann"        -> exact Fourier orders of a binary +-pi phase profile
#     "table"          -> per-order interpolated efficiency/amplitude:
#         legacy (v1) tables -> real eta_s/eta_p at the ray lambda (cos_i /
#             azimuth ignored)
#         v2 RCWA tables (schema=="v2") -> COMPLEX per-order amplitudes with
#             phase, interpolated multilinearly over (lambda, theta, phi);
#             |amp|^2 = order efficiency. Use order_amplitudes() for phase.
#   Missing "model"/"params"/"table" keys default to legacy lamellar behavior
#   so CLI-parsed specs (common.parse_grating_value) still trace unchanged.
# =============================================================================
import numpy as np

from .optprops import interp_hard      # sibling util (not tracer/scene)


def _unit_rows(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def order_directions(d, n_hat, g_hat, lines_per_mm, lam, orders, n1, n2):
    """Vector grating equation for a set of diffraction orders.

    d, n_hat, g_hat : (N,3) unit vectors (g_hat tangential to n_hat)
    lines_per_mm    : scalar or (N,) grating frequency [1/mm]
    lam             : (N,) vacuum wavelength [m]
    orders          : iterable of int diffraction orders
    n1, n2          : (N,) real refractive indices (incident / far side)

    Returns {m: (dirs_t (N,3), prop_t (N,) bool, dirs_r (N,3), prop_r (N,) bool)}
    """
    d = np.asarray(d, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    g_hat = np.asarray(g_hat, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    n1 = np.asarray(n1, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)

    N = d.shape[0]
    n1 = np.broadcast_to(n1, (N,)).astype(np.float64)
    n2 = np.broadcast_to(n2, (N,)).astype(np.float64)
    lam = np.broadcast_to(lam, (N,)).astype(np.float64)

    cos_i = -np.sum(d * n_hat, axis=-1)
    d_tang = d + cos_i[:, None] * n_hat        # tangential projection of d

    d_period = 1e-3 / np.asarray(lines_per_mm, dtype=np.float64)  # [m]
    lam_over_d = lam / d_period                # scalar or (N,)
    lam_over_d = np.broadcast_to(lam_over_d, (N,)).astype(np.float64)

    out = {}
    for m in orders:
        T = n1[:, None] * d_tang + (m * lam_over_d)[:, None] * g_hat
        Tmag2 = np.sum(T * T, axis=-1)

        rad_t = n2 ** 2 - Tmag2
        prop_t = rad_t >= 0.0
        nz_t = np.sqrt(np.maximum(rad_t, 0.0))
        dirs_t = (T - nz_t[:, None] * n_hat) / n2[:, None]
        dirs_t = _unit_rows(dirs_t)

        rad_r = n1 ** 2 - Tmag2
        prop_r = rad_r >= 0.0
        nz_r = np.sqrt(np.maximum(rad_r, 0.0))
        dirs_r = (T + nz_r[:, None] * n_hat) / n1[:, None]
        dirs_r = _unit_rows(dirs_r)

        out[m] = (dirs_t, prop_t, dirs_r, prop_r)
    return out


def lamellar_efficiencies(orders, duty=0.5, efficiencies=None):
    """Idealized scalar-model power efficiencies of a binary (lamellar)
    phase grating with phase depth pi, duty cycle `duty` (fraction of the
    period at phase 0; the remaining (1-duty) fraction is phase pi).

    Exact Fourier-series result (Parseval-conserving: eta_0 = c_0^2 with
    c_0 = 2*duty-1, and eta_m = (2 sin(pi*m*duty)/(pi*m))^2 for m != 0; sum
    over ALL integers m equals 1 exactly). At duty=0.5 this reduces to the
    textbook result: eta_0 = 0, eta_{odd} = (2/(pi*m))^2 (eta_{+-1}=4/pi^2),
    eta_{even!=0} = 0 (sin(pi*m*0.5) vanishes for even m).

    If `efficiencies` is given (explicit per-order list matching `orders`),
    it is used as-is (pass-through) after validating sum <= 1.

    Returns (effs: {m: eta}, total: float) — NOT renormalized to 1; the
    remainder (1 - total) is absorbed into orders/paths not requested here.
    """
    orders = list(orders)
    if efficiencies is not None:
        effs = [float(e) for e in efficiencies]
        if len(effs) != len(orders):
            raise ValueError(
                "lamellar_efficiencies: %d efficiencies given for %d orders"
                % (len(effs), len(orders)))
        total = float(sum(effs))
        if total > 1.0 + 1e-9:
            raise ValueError(
                "lamellar_efficiencies: explicit efficiencies sum to %.6g "
                "> 1" % total)
        return dict(zip(orders, effs)), total

    eta = {}
    for m in orders:
        if m == 0:
            eta[m] = (2.0 * duty - 1.0) ** 2
        else:
            eta[m] = (2.0 * np.sin(np.pi * m * duty) / (np.pi * m)) ** 2
    total = float(sum(eta.values()))
    return eta, total


# ---------------------------------------------------------------------------
# bragg_kogelnik — Kogelnik (1969) coupled-wave theory, TRANSMISSION geometry
# ---------------------------------------------------------------------------
# Reference: H. Kogelnik, "Coupled Wave Theory for Thick Hologram Gratings",
# Bell Syst. Tech. J. 48, 2909 (1969). For a lossless dielectric transmission
# phase grating the first-order (Bragg) diffraction efficiency is
#
#     eta_1 = sin^2( sqrt(nu^2 + xi^2) ) * nu^2 / (nu^2 + xi^2)      (Kogelnik)
#
# with the coupling strength and dephasing (his transmission-case results)
#
#     nu_s = pi * dn * d / ( lam * sqrt(c_R c_S) )        (s / TE polarization)
#     nu_p = nu_s * | cos( 2 (theta - slant) ) |          (p / TM: field-vector
#                                                          projection between
#                                                          the two beams)
#     xi   = vartheta * d / ( 2 c_S )                      (Bragg mismatch)
#     vartheta = K cos(phi_g - theta) - K^2 / (2 beta)     (dephasing measure)
#
# where d = film thickness, dn = index-modulation amplitude, K = 2 pi / Lambda
# is the grating vector (Lambda = 1/lines_per_mm period), beta = 2 pi / lam the
# probe wavenumber, theta the LOCAL incidence angle, phi_g = pi/2 - slant the
# grating-vector angle from the surface normal (slant_deg=0 => unslanted, K in
# the surface plane => c_R = c_S = cos theta), and c_R = cos theta,
# c_S = cos theta - (K/beta) cos phi_g the reference / signal obliquity factors.
# The Bragg condition vartheta = 0 gives cos(phi_g - theta) = lam/(2 Lambda).
#
# THIN-HOLOGRAM APPROXIMATION (documented limitation): the tracer already
# handles the substrate interfaces; refraction of the probe INTO the hologram's
# average index is NOT modeled here. The hologram is treated as a thin boundary
# layer evaluated at the LOCAL incidence angle (beta uses the incident-medium
# wavenumber, n~1). This is the standard "thin boundary" idealization; for a
# quantitative VBG one would Snell-refract theta and use beta = 2 pi n / lam.
# Only orders 0 and +1 carry power (eta_0 = 1 - eta_1, lossless dielectric);
# all other requested orders get zero. REFLECTION-geometry VBGs are NOT modeled
# (they need the reflection coupled-wave solution with tanh/sinh, not sin).
def bragg_kogelnik_eta(nu, xi):
    """First-order (Bragg) efficiency of a lossless dielectric transmission
    volume phase grating: eta_1 = sin^2(sqrt(nu^2+xi^2)) * nu^2/(nu^2+xi^2).
    nu = coupling strength (>=0), xi = Bragg dephasing. nu=xi=0 -> 0. Clamped
    to [0, 1]. Vectorized over nu, xi."""
    nu = np.asarray(nu, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    arg = nu * nu + xi * xi
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(arg > 0.0, nu * nu / np.where(arg > 0.0, arg, 1.0), 0.0)
        eta = np.sin(np.sqrt(arg)) ** 2 * frac
    return np.clip(eta, 0.0, 1.0)


def _kogelnik_nu_xi(spec, lam, cos_i):
    """Geometry -> (nu_s, nu_p, xi, propagating) per ray for the transmission
    Kogelnik model. `propagating` masks rays whose signal-wave obliquity
    c_S <= 0 (diffracted order evanescent -> no coupling)."""
    p = spec.get("params", {})
    d = float(p["thickness_um"]) * 1e-6                 # film thickness [m]
    dn = float(p["dn"])                                 # index modulation
    slant = np.deg2rad(float(p.get("slant_deg", 0.0)))  # fringe slant [rad]
    period = 1e-3 / float(spec["lines_per_mm"])         # grating period [m]

    lam = np.asarray(lam, dtype=np.float64)
    theta = np.arccos(np.clip(np.asarray(cos_i, dtype=np.float64), -1.0, 1.0))
    K = 2.0 * np.pi / period
    beta = 2.0 * np.pi / lam                            # thin-boundary: n ~ 1
    phi_g = 0.5 * np.pi - slant                         # grating-vec vs normal

    c_R = np.cos(theta)
    c_S = np.cos(theta) - (K / beta) * np.cos(phi_g)
    vartheta = K * np.cos(phi_g - theta) - K * K / (2.0 * beta)

    propagating = c_S > 1e-12
    cs = np.where(propagating, c_S, 1.0)               # guard sqrt/divide
    nu_s = np.pi * dn * d / (lam * np.sqrt(c_R * cs))
    nu_p = nu_s * np.abs(np.cos(2.0 * (theta - slant)))
    xi = vartheta * d / (2.0 * cs)
    return nu_s, nu_p, xi, propagating


# ---------------------------------------------------------------------------
# dammann — exact Fourier orders of a binary +-pi phase profile
# ---------------------------------------------------------------------------
# Profile phi(x) over one normalized period [0,1) is +-pi such that
# exp(i phi(x)) = s_j (=+1/-1) on segment [x_j, x_{j+1}); the sign ALTERNATES
# starting +1 at x=0, with transition points x_1..x_K strictly increasing in
# (0,1) (x_0=0, x_{K+1}=1). The exact Fourier coefficient of exp(i phi) is
#     c_m = sum_j s_j (exp(-2 pi i m x_{j+1}) - exp(-2 pi i m x_j)) / (-2 pi i m)
#     c_0 = sum_j s_j (x_{j+1} - x_j)
# and eta_m = |c_m|^2 (SCALAR: eta_s == eta_p for a binary-phase grating). The
# profile is real so |c_m| = |c_{-m}| exactly. Parseval: sum_all_m |c_m|^2 = 1.
def dammann_coefficients(transitions, orders):
    """Complex Fourier coefficients {m: c_m} of exp(i*phi(x)) for the binary
    +-pi profile defined by `transitions` (list in (0,1), strictly
    increasing)."""
    tr = ([float(transitions)] if np.isscalar(transitions)
          else [float(x) for x in transitions])
    xs = [0.0] + tr + [1.0]
    signs = [(-1) ** j for j in range(len(xs) - 1)]     # +1,-1,+1,...
    out = {}
    for m in orders:
        if m == 0:
            out[m] = complex(sum(s * (xs[j + 1] - xs[j])
                                 for j, s in enumerate(signs)))
        else:
            c = 0.0 + 0.0j
            for j, s in enumerate(signs):
                a, b = xs[j], xs[j + 1]
                c += s * (np.exp(-2j * np.pi * m * b)
                          - np.exp(-2j * np.pi * m * a)) / (-2j * np.pi * m)
            out[m] = c
    return out


def dammann_efficiencies(transitions, orders):
    """{m: |c_m|^2} power efficiencies for the binary +-pi Dammann profile."""
    return {m: float(np.abs(c) ** 2)
            for m, c in dammann_coefficients(transitions, orders).items()}


# ---------------------------------------------------------------------------
# v2 RCWA tables — complex per-order amplitude, interpolated multilinearly on
# (lambda, theta, phi). Amplitudes are stored so that |amp|^2 equals the
# co-polarized order efficiency and arg(amp) is the diffracted-order phase
# (engine3 §7.5; generated by scripts/tools/gen_rcwa_table.py via meent RCWA).
# Interpolating the COMPLEX components directly (re/im each), NOT magnitude +
# unwrapped phase, is the Zemax/Lumerical approach and sidesteps 2*pi phase-
# unwrapping ambiguities.
# ---------------------------------------------------------------------------
def _interp3_complex(axes, grid, q):
    """Multilinear interpolation of a complex 3-D grid.

    axes : (ax0, ax1, ax2) each 1-D strictly-increasing node coordinates
    grid : complex ndarray (n0, n1, n2)
    q    : (N, 3) query points (columns aligned with axes)

    Query points are CLAMPED to each axis' node range (edge-hold; no
    extrapolation past the sampled directions). Axes of length 1 contribute
    a constant (zero weight). Returns (N,) complex128.
    """
    q = np.asarray(q, dtype=np.float64)
    N = q.shape[0]
    i0 = []
    i1 = []
    tw = []
    for k, a in enumerate(axes):
        a = np.asarray(a, dtype=np.float64)
        n = a.size
        qc = np.clip(q[:, k], a[0], a[-1])
        if n == 1:
            lo = np.zeros(N, dtype=np.intp)
            hi = lo
            t = np.zeros(N, dtype=np.float64)
        else:
            lo = np.clip(np.searchsorted(a, qc, side="right") - 1, 0, n - 2)
            hi = lo + 1
            denom = a[hi] - a[lo]
            t = np.where(denom > 0, (qc - a[lo]) / np.where(denom > 0,
                                                            denom, 1.0), 0.0)
        i0.append(lo)
        i1.append(hi)
        tw.append(t)
    out = np.zeros(N, dtype=np.complex128)
    for cx in (0, 1):
        wx = tw[0] if cx else (1.0 - tw[0])
        ix = i1[0] if cx else i0[0]
        for cy in (0, 1):
            wy = tw[1] if cy else (1.0 - tw[1])
            iy = i1[1] if cy else i0[1]
            for cz in (0, 1):
                wz = tw[2] if cz else (1.0 - tw[2])
                iz = i1[2] if cz else i0[2]
                out += (wx * wy * wz) * grid[ix, iy, iz]
    return out


def _v2_amplitudes(table, lam, cos_i, azimuth, orders):
    """Interpolate a v2 RCWA table -> complex (amp_s, amp_p) each (N, no).
    lam [m], cos_i [-], azimuth [deg] are per-ray; orders lists the columns.
    Missing table orders yield zero amplitude (no diffracted child)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    cos_i = np.broadcast_to(np.atleast_1d(np.asarray(cos_i, np.float64)),
                            lam.shape)
    if azimuth is None:
        azimuth = np.zeros_like(lam)
    azimuth = np.broadcast_to(np.atleast_1d(np.asarray(azimuth, np.float64)),
                              lam.shape)
    lam_um = lam * 1e6
    theta_deg = np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))
    # the grating is symmetric under phi -> -phi for a co-polarized diagonal
    # response; fold the ray azimuth into the tabulated (phi >= 0) range.
    phi_deg = np.abs(azimuth)
    q = np.column_stack([lam_um, theta_deg, phi_deg])
    axes = (table["lam_um"], table["theta_deg"], table["phi_deg"])
    N, no = lam.shape[0], len(orders)
    amp_s = np.zeros((N, no), dtype=np.complex128)
    amp_p = np.zeros((N, no), dtype=np.complex128)
    for j, m in enumerate(orders):
        gs = table["amp_s"].get(m)
        if gs is None:
            continue
        amp_s[:, j] = _interp3_complex(axes, gs, q)
        amp_p[:, j] = _interp3_complex(axes, table["amp_p"][m], q)
    return amp_s, amp_p


def _is_v2_table(spec):
    t = spec.get("table")
    return isinstance(t, dict) and t.get("schema") == "v2"


def order_amplitudes(spec, lam, cos_i, orders, azimuth=None):
    """Per-ray, per-order COMPLEX diffracted amplitudes (amp_s, amp_p), each
    (n_rays, n_orders), with |amp|^2 = the order efficiency and arg(amp) the
    coherent diffracted-order phase.

    For v2 RCWA tables the amplitudes carry real phase, interpolated over
    (lambda, theta, phi). For every other model (lamellar / bragg_kogelnik /
    dammann / legacy real table) the amplitude is sqrt(efficiency) with zero
    relative phase — BIT-IDENTICAL to the historical sqrt(eta) behavior, so
    the coherent machinery is unchanged for those specs."""
    orders = list(orders)
    if _is_v2_table(spec):
        return _v2_amplitudes(spec["table"], lam, cos_i, azimuth, orders)
    eta_s, eta_p = order_efficiencies(spec, lam, cos_i, orders)
    return (np.sqrt(eta_s).astype(np.complex128),
            np.sqrt(eta_p).astype(np.complex128))


# ---------------------------------------------------------------------------
# unified efficiency dispatch (polarization- and wavelength-resolved)
# ---------------------------------------------------------------------------
def order_efficiencies(spec, lam, cos_i, orders, azimuth=None):
    """Per-ray, per-order power efficiencies (eta_s, eta_p), each shape
    (n_rays, n_orders) aligned column-for-column with `orders`.

    spec keys: "model" in {lamellar, bragg_kogelnik, dammann, table} (missing
    => lamellar); plus model-specific data ("efficiencies", "params",
    "table"). lam [m], cos_i [-] are per-ray (scalars broadcast). For scalar
    (lamellar/dammann) models eta_s == eta_p exactly.

    `azimuth` [deg] is used only by v2 RCWA tables (conical-mount direction);
    every other model ignores it. A v2 table returns |amp|^2 (its co-polarized
    order efficiency); use order_amplitudes() to get the phase-carrying
    complex amplitudes.
    """
    orders = list(orders)
    if _is_v2_table(spec):
        amp_s, amp_p = _v2_amplitudes(spec["table"], lam, cos_i, azimuth,
                                      orders)
        return np.abs(amp_s) ** 2, np.abs(amp_p) ** 2
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    cos_i = np.broadcast_to(np.atleast_1d(np.asarray(cos_i, dtype=np.float64)),
                            lam.shape)
    N, no = lam.shape[0], len(orders)
    col = {m: j for j, m in enumerate(orders)}
    eta_s = np.zeros((N, no), dtype=np.float64)
    eta_p = np.zeros((N, no), dtype=np.float64)

    model = spec.get("model") or "lamellar"

    if model == "lamellar":
        effs, _ = lamellar_efficiencies(orders,
                                        efficiencies=spec.get("efficiencies"))
        for m, j in col.items():
            eta_s[:, j] = effs.get(m, 0.0)
        eta_p[...] = eta_s

    elif model == "dammann":
        tr = spec.get("params", {}).get("transitions")
        if tr is None:
            raise ValueError("dammann grating spec missing "
                             "params['transitions']")
        effs = dammann_efficiencies(tr, orders)
        for m, j in col.items():
            eta_s[:, j] = effs[m]
        eta_p[...] = eta_s

    elif model == "bragg_kogelnik":
        nu_s, nu_p, xi, prop = _kogelnik_nu_xi(spec, lam, cos_i)
        e1s = np.where(prop, bragg_kogelnik_eta(nu_s, xi), 0.0)
        e1p = np.where(prop, bragg_kogelnik_eta(nu_p, xi), 0.0)
        if 0 in col:
            eta_s[:, col[0]] = 1.0 - e1s
            eta_p[:, col[0]] = 1.0 - e1p
        if 1 in col:
            eta_s[:, col[1]] = e1s
            eta_p[:, col[1]] = e1p

    elif model == "table":
        table = spec.get("table")
        if not table:
            raise ValueError("table grating spec missing per-order 'table'")
        lam_um = lam * 1e6
        ctx = "grating table"
        for m, j in col.items():
            t = table.get(m)
            if t is None:
                continue                    # spec order absent from table -> 0
            eta_s[:, j] = interp_hard(lam_um, t["lam_um"], t["eta_s"], ctx)
            eta_p[:, j] = interp_hard(lam_um, t["lam_um"], t["eta_p"], ctx)

    else:
        raise ValueError("unknown grating model %r" % model)

    return eta_s, eta_p


def groove_vector(face_surface, groove_spec, n_hat_sample):
    """Resolve a CLI groove spec ('u' | 'v' | 'x,y,z') into a per-sample unit
    periodicity vector field lying in the local tangent plane.

    face_surface  : analytic surface object (Plane/Sphere/...) exposing
                    constant t1/t2 tangent-frame vectors (see surfaces.py)
    groove_spec   : 'u', 'v', or 'x,y,z' (explicit direction, any frame)
    n_hat_sample  : (N,3) unit normals (per-sample; may vary over a curved
                    face) used to project the base vector into the local
                    tangent plane.
    """
    n_hat_sample = np.asarray(n_hat_sample, dtype=np.float64)
    N = n_hat_sample.shape[0]

    if groove_spec == "u":
        base = np.broadcast_to(face_surface.t1, (N, 3)).copy()
    elif groove_spec == "v":
        base = np.broadcast_to(face_surface.t2, (N, 3)).copy()
    else:
        try:
            vec = np.array([float(x) for x in groove_spec.split(",")],
                            dtype=np.float64)
        except ValueError:
            raise ValueError("groove spec must be 'u', 'v', or 'x,y,z': %r"
                             % groove_spec)
        if vec.shape != (3,):
            raise ValueError("groove vector spec must have 3 components: %r"
                             % groove_spec)
        base = np.broadcast_to(vec, (N, 3)).copy()

    comp = np.sum(base * n_hat_sample, axis=-1, keepdims=True)
    tang = base - comp * n_hat_sample
    norm = np.linalg.norm(tang, axis=-1, keepdims=True)
    if np.any(norm[:, 0] < 1e-8):
        raise ValueError(
            "groove vector %r is nearly parallel to the surface normal"
            % groove_spec)
    return tang / norm


# ---------------------------------------------------------------------------
# Tracer-facing adapter (wired by tracer.step for faces named in --grating)
# ---------------------------------------------------------------------------
def apply_to_batch(tracer, fid, grp):
    """Diffract a ray group off a grating face. Spawns one child per
    propagating order. Reflective if the body's mirror >= 0.5, else
    transmissive. Order efficiencies come from order_efficiencies(spec, ...):
    polarization- (eta_s, eta_p) and wavelength-resolved per the grating model
    (lamellar / bragg_kogelnik / dammann / table).

    The incident Jones vector (Es, Ep) is rotated into THIS interface's (s,p)
    basis before the per-polarization amplitudes sqrt(eta_s), sqrt(eta_p) are
    applied, and each child's s_hat is set to that basis (exactly like the
    Fresnel path in tracer._optic_children). Limitations (documented): order
    rays inherit opl continuously (no inter-order grating phase offsets beyond
    OPL); the thin-hologram Kogelnik / scalar Dammann caveats live in the model
    docstrings above."""
    import numpy as np
    from . import fresnel as fr
    from . import poltransport as pt
    from .rays import RayBatch

    scene = tracer.scene
    spec = scene.gratings[fid]
    face = scene.faces[fid]
    body = scene.body_of_face(fid)
    m_rays = len(grp)

    n_out = face.normal_out_of_solid(grp.pos)
    sgn = -np.sign(np.sum(n_out * grp.dir, axis=-1))
    n_hat = n_out * sgn[:, None]
    entering = sgn > 0
    cos_i = -np.sum(grp.dir * n_hat, axis=-1)

    cur = grp.current_medium()
    n1 = np.empty(m_rays)
    for mm in np.unique(cur):
        s = cur == mm
        n1[s] = np.real(scene.medium_index(int(mm), grp.lam[s]))
    if body.material not in (None, "detector"):
        n2 = np.real(scene.matdb.get(body.material).n_complex(grp.lam))
    else:
        n2 = np.ones(m_rays)
    # exiting rays swap sides
    n1s = np.where(entering, n1, n2)
    n2s = np.where(entering, n2, n1)

    g_hat = groove_vector(face.surface, spec["groove"], n_hat)
    if g_hat.ndim == 1:
        g_hat = np.broadcast_to(g_hat, (m_rays, 3))

    # azimuth of the plane of incidence measured from the groove-periodicity
    # vector g_hat (phi=0 = classical mount): phi = atan2(d_t . t_perp,
    # d_t . g_hat) with t_perp = n_hat x g_hat the in-plane axis. Used only by
    # v2 RCWA tables (conical mount); every other model ignores it.
    t_perp = np.cross(n_hat, g_hat)
    d_tang = grp.dir + cos_i[:, None] * n_hat
    azimuth = np.degrees(np.arctan2(np.sum(d_tang * t_perp, axis=-1),
                                    np.sum(d_tang * g_hat, axis=-1)))

    lo, hi = spec["orders"]
    orders = list(range(lo, hi + 1))
    # complex per-order amplitudes (|amp|^2 = efficiency; arg = diffracted
    # phase for v2 RCWA tables, zero for the analytic/legacy models). eta_*
    # (real power efficiencies) drive the energy ledger below.
    amp_s, amp_p = order_amplitudes(spec, grp.lam, cos_i, orders, azimuth)
    eta_s = np.abs(amp_s) ** 2
    eta_p = np.abs(amp_p) ** 2
    dirs = order_directions(grp.dir, n_hat, g_hat, spec["lines_per_mm"],
                            grp.lam, orders, n1s, n2s)

    # rotate the incident Jones vector into this interface's (s,p) basis
    s_new, p_new = fr.pol_basis(grp.dir, n_hat)
    p_old = np.cross(grp.dir, grp.s_hat)
    Es, Ep = fr.rotate_jones(grp.Es, grp.Ep, grp.s_hat, p_old, s_new, p_new)
    Is = np.abs(Es) ** 2                      # per-ray s / p incident power
    Ip = np.abs(Ep) ** 2
    p_in = Is + Ip

    reflective = body.mirror >= 0.5
    order_power = np.zeros(m_rays)             # power sent into propagating ords
    children = []
    for j, m_ord in enumerate(orders):
        dirs_t, prop_t, dirs_r, prop_r = dirs[m_ord]
        d_new, prop = (dirs_r, prop_r) if reflective else (dirs_t, prop_t)
        contrib = Is * eta_s[:, j] + Ip * eta_p[:, j]
        order_power += np.where(prop, contrib, 0.0)
        # spawn a child only where the order both propagates and carries power
        keep = prop & (contrib > 0.0)
        if not np.any(keep):
            continue
        prop = keep
        child = grp.select(prop)
        child.dir = d_new[prop]
        # s_hat must be perpendicular to the CHILD's own direction. For
        # planar diffraction every order shares the TE direction n x d
        # (along the grooves), which equals the incident s basis up to
        # sign at non-degenerate incidence -- but at exactly NORMAL
        # incidence pol_basis's fallback s is arbitrary-transverse, and
        # inheriting it verbatim leaves a diffracted child whose s_hat
        # is NOT perpendicular to its direction. The next polarization-
        # aware interface (e.g. the second grating of a Treacy pair)
        # then rotate_jones's through a skewed basis and silently loses
        # cos^2(theta) of the power (9.3% closure leak in the
        # treacy_compressor demo). Rebuild per child; the specular order
        # at normal incidence (n x d degenerate) keeps the incident
        # basis, which IS perpendicular there.
        cr = np.cross(n_hat[prop], child.dir)
        nrm = np.linalg.norm(cr, axis=-1)
        ok_frame = nrm > 1e-9
        s_child = np.where(
            ok_frame[:, None],
            cr / np.maximum(nrm, 1e-300)[:, None], s_new[prop])
        # keep the sign aligned with the incident s so the (Es, Ep)
        # amplitudes keep their meaning (a flipped s is a pi phase jump
        # on the s component = a different polarization state)
        flip = np.sum(s_child * s_new[prop], axis=-1) < 0.0
        s_child = np.where((ok_frame & flip)[:, None], -s_child, s_child)
        child.s_hat = s_child
        # complex per-order amplitudes: |amp|^2 = efficiency and arg(amp) =
        # diffracted-order phase. For the analytic/legacy models amp is real
        # (sqrt(eta), zero phase) so child.Es/Ep are BIT-IDENTICAL to before;
        # v2 RCWA tables inject the real diffracted phase here, composing with
        # the coherent OPL the child already carries.
        amp_s_ord = amp_s[prop, j]
        amp_p_ord = amp_p[prop, j]
        child.Es = Es[prop] * amp_s_ord
        child.Ep = Ep[prop] * amp_p_ord
        if child.Jmat is not None:
            # local step is diag(amp_s, amp_p) in the interface (s,p) basis: a
            # diattenuator (analytic/legacy models, real amp) or a
            # diattenuator+retarder (v2 tables, complex amp). Retardance/spin
            # the child later reports is this step plus geometric Q plus
            # whatever upstream/downstream optics add.
            R1 = pt.basis_rot2(grp.s_hat[prop], p_old[prop],
                               s_new[prop], p_new[prop])
            j_step = np.einsum('nij,njk->nik',
                               pt.diag2(amp_s_ord, amp_p_ord), R1)
            pt.update(child, grp.s_hat[prop], grp.dir[prop],
                     child.s_hat, child.dir, j_step)
        if reflective:
            child.generation += 1
            gen_ok = child.generation <= tracer.cfg.max_reflections
            if not np.all(gen_ok):
                tracer.ledger.credit("truncated_generation",
                                     child.source_id[~gen_ok],
                                     child.power[~gen_ok])
                child = child.select(gen_ok)
        else:
            # transmitted orders change medium like refraction
            ent = entering[prop]
            body_idx = np.full(int(np.sum(prop)), body.index)
            child.push_medium(ent, body_idx)
            child.pop_medium(~ent, body_idx)
        if len(child):
            children.append(child)
    # energy bookkeeping: per-ray, per-polarization absorbed remainder
    #   absorbed = p_in - sum_m (|Es|^2 eta_s[m] + |Ep|^2 eta_p[m])
    # (efficiency losses + evanescent orders). Truncated-generation power was
    # part of order_power and is credited to its own bucket above, so the
    # partition p_in = absorbed + surviving_children + truncated stays exact.
    p_absorbed = np.clip(p_in - order_power, 0.0, None)
    if np.any(p_absorbed > 0):
        tracer.ledger.credit("absorbed_surface", grp.source_id, p_absorbed,
                             where=body.label + ":grating")
    return RayBatch.concatenate(children) if children else None
