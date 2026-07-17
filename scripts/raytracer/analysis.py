# =============================================================================
# analysis.py — exit-wavefront analysis: Zernike decomposition + Strehl.
# Numpy-only, float64, vectorized over ray samples.
#
# Pupil model (v1, documented limitation): the SOURCE-REFERENCED pupil — each
# exported ray's normalized transverse birth position on the emitting face is
# its pupil coordinate. Exact for the collimated/laser benches this tracer
# models; NOT a true exit pupil for finite-conjugate, field-point imaging.
# The exit-pupil/chief-ray search stage lives in analysis_imaging.py (a
# least-squares pupil center from >= 2 field bundles' centroid rays, chief
# rays, reference-sphere OPD, PSF-peak Strehl); post_process.render_wavefront
# selects between the two via its pupil_mode parameter ("source" — this
# module's model, the default — | "exit_pupil"), falling back to "source"
# with a report note when the exit-pupil solve degenerates (single field
# point / telecentric image side).
#
# OPD definition: for a ray with accumulated optical path `opl` (metres,
# already n-weighted along its path) landing at `hit`, the wavefront error
# to a reference point r0 in ambient index n_amb is
#
#     W_i = opl_i + n_amb * |hit_i - r0|  -  (chief value)
#
# i.e. every ray is propagated to the common reference point and referenced
# to the chief ray (the sample closest to the pupil centre). Piston, tip and
# tilt are additionally removed by the Zernike fit itself (they are fitted
# coefficients — report them, subtract them from the residual RMS).
#
# Zernike convention: Noll indexing (j = 1 piston, 2/3 tilt, 4 defocus,
# 5/6 astigmatism, 7/8 coma, 11 spherical), Noll-normalized so each term has
# unit RMS over the unit disc — a coefficient IS its RMS contribution. The
# Fringe index of each term is provided for interop (fringe_index()).
#
# Strehl: Maréchal approximation exp(-(2*pi*sigma/lam)^2) from the
# piston/tip/tilt-removed residual RMS sigma (metres). The PSF-peak-ratio
# Strehl (|sum a e^{ikW}|^2 / (sum a)^2 over the pupil samples) lives in
# analysis_imaging.strehl_psf_peak and is reported ALONGSIDE strehl_marechal
# by post_process.render_wavefront (never replacing it).
# =============================================================================
import math

import numpy as np


# ---------------------------------------------------------------------------
# Noll <-> (n, m) bookkeeping
# ---------------------------------------------------------------------------
def noll_to_nm(j):
    """Radial order n and SIGNED azimuthal order m for Noll index j >= 1.
    Noll's rule: within an n, |m| ascends; even j -> cos (+m), odd j -> sin
    (-m)."""
    if j < 1:
        raise ValueError("Noll index starts at 1 (got %r)" % j)
    n = 0
    j_rem = j - 1
    while j_rem >= n + 1:
        n += 1
        j_rem -= n
    # j_rem in [0, n]: position within the order
    m_abs = n - 2 * ((n - j_rem) // 2)
    if m_abs == 0:
        return n, 0
    m = m_abs if (j % 2 == 0) else -m_abs
    return n, m


def fringe_index(n, m):
    """Fringe (University of Arizona) index of the (n, m) Zernike term."""
    return int((1 + (n + abs(m)) / 2) ** 2 - 2 * abs(m) + (1 - np.sign(m)) / 2)


_NOLL_NAMES = {
    1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus",
    5: "astig 45", 6: "astig 0", 7: "coma y", 8: "coma x",
    9: "trefoil y", 10: "trefoil x", 11: "spherical",
}


def noll_name(j):
    return _NOLL_NAMES.get(j, "Z%d" % j)


# ---------------------------------------------------------------------------
# Zernike evaluation (Noll-normalized: unit RMS over the unit disc)
# ---------------------------------------------------------------------------
def _radial(n, m_abs, rho):
    """R_n^|m|(rho) by the explicit factorial sum (n <= ~20 is exact in
    float64; wavefront fits never need more)."""
    r = np.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        c = ((-1.0) ** k * math.factorial(n - k)
             / (math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)))
        r += c * rho ** (n - 2 * k)
    return r


def zernike(j, rho, theta):
    """Noll-normalized Zernike Z_j sampled at polar pupil coords (rho in
    [0,1], theta radians). Vectorized over samples."""
    n, m = noll_to_nm(j)
    rad = _radial(n, abs(m), np.asarray(rho, dtype=np.float64))
    if m == 0:
        return np.sqrt(n + 1.0) * rad
    norm = np.sqrt(2.0 * (n + 1.0))
    if m > 0:
        return norm * rad * np.cos(m * theta)
    return norm * rad * np.sin(-m * theta)


def zernike_basis(jmax, rho, theta):
    """(N, jmax) design matrix of Z_1..Z_jmax at the sample points."""
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack([zernike(j, rho, theta) for j in range(1, jmax + 1)],
                    axis=1)


def _radial_prime(n, m_abs, rho):
    """dR_n^|m|/drho by term-wise differentiation of the factorial sum."""
    r = np.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        p = n - 2 * k
        if p == 0:
            continue
        c = ((-1.0) ** k * math.factorial(n - k)
             / (math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)))
        r += c * p * rho ** (p - 1)
    return r


def zernike_cart_grad(j, u, v):
    """Cartesian gradient (dZ_j/du, dZ_j/dv) of the Noll-normalized Zernike
    Z_j at UNIT-DISC coordinates (u, v) (rho = hypot(u,v), theta = atan2(v,u)).

    Analytic — the polar chain rule with the seam handled at the origin
    (rho -> 0): only tilt (n=1) has a finite gradient there; every higher
    term's Cartesian gradient -> 0 as rho -> 0, so a zero fill is exact in
    the limit. Vectorized over samples; returns (gu, gv) float64 arrays.
    Validated against finite differences in test_figure_error.py."""
    n, m = noll_to_nm(j)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    safe = rho > 1e-12
    rho_s = np.where(safe, rho, 1.0)
    ma = abs(m)
    rad = _radial(n, ma, rho)
    radp = _radial_prime(n, ma, rho)
    # drho/du = u/rho, drho/dv = v/rho ; dtheta/du = -v/rho^2, dtheta/dv = u/rho^2
    drho_du = np.where(safe, u / rho_s, 0.0)
    drho_dv = np.where(safe, v / rho_s, 0.0)
    dth_du = np.where(safe, -v / (rho_s * rho_s), 0.0)
    dth_dv = np.where(safe, u / (rho_s * rho_s), 0.0)
    if m == 0:
        c = math.sqrt(n + 1.0)
        dZ_drho = c * radp
        gu = dZ_drho * drho_du
        gv = dZ_drho * drho_dv
    else:
        norm = math.sqrt(2.0 * (n + 1.0))
        if m > 0:
            ang = np.cos(m * theta)
            dang = -m * np.sin(m * theta)
        else:
            ang = np.sin(ma * theta)
            dang = ma * np.cos(ma * theta)
        dZ_drho = norm * radp * ang
        dZ_dth = norm * rad * dang
        gu = dZ_drho * drho_du + dZ_dth * dth_du
        gv = dZ_drho * drho_dv + dZ_dth * dth_dv
    # tilt (n=1) is the only term with a nonzero gradient at the origin;
    # its radial part is linear so radp is constant and the zero-fill above
    # would wrongly kill it -> restore the exact constant-slope value.
    if n == 1:
        norm = math.sqrt(2.0 * (n + 1.0))
        if m > 0:      # Z2 = 2 u  -> (2, 0)
            gu = np.where(safe, gu, norm)
            gv = np.where(safe, gv, 0.0)
        else:          # Z3 = 2 v  -> (0, 2)
            gu = np.where(safe, gu, 0.0)
            gv = np.where(safe, gv, norm)
    return gu, gv


def fit_zernike(rho, theta, opd, jmax=15, weights=None):
    """Weighted least-squares Zernike fit of an OPD sample set.

    Returns dict:
      coeffs (jmax,)   Noll coefficients, same units as opd (a coefficient
                       is its RMS contribution over the unit disc)
      residual_rms     RMS of (opd - fit) over the samples
      rms_wavefront    RMS of the fit EXCLUDING piston/tip/tilt (j>=4),
                       i.e. sqrt(sum coeffs[3:]^2) — the sigma for Strehl
      pv               peak-to-valley of the piston/tip/tilt-removed fit
                       evaluated at the samples
    """
    opd = np.asarray(opd, dtype=np.float64)
    A = zernike_basis(jmax, rho, theta)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        coeffs, *_ = np.linalg.lstsq(A * w[:, None], opd * w, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(A, opd, rcond=None)
    fit = A @ coeffs
    lowest = min(3, jmax)
    shape_fit = fit - A[:, :lowest] @ coeffs[:lowest]
    return {
        "coeffs": coeffs,
        "residual_rms": float(np.sqrt(np.mean((opd - fit) ** 2))),
        "rms_wavefront": float(np.sqrt(np.sum(coeffs[3:] ** 2)))
        if jmax > 3 else 0.0,
        "pv": float(shape_fit.max() - shape_fit.min()) if len(fit) else 0.0,
    }


# ---------------------------------------------------------------------------
# OPD from exported rays + Strehl
# ---------------------------------------------------------------------------
def opd_from_rays(pupil_xy, hit_xyz, opl, ref_point, n_ambient=1.0):
    """Chief-referenced OPD (metres) per ray.

    pupil_xy : (N,2) NORMALIZED pupil coordinates (unit disc).
    hit_xyz  : (N,3) landing points (metres, global).
    opl      : (N,) accumulated optical path length at the landing point.
    ref_point: (3,) wavefront reference (image) point.
    Chief ray = sample with the smallest pupil radius.
    Returns (opd (N,), rho (N,), theta (N,))."""
    pupil_xy = np.asarray(pupil_xy, dtype=np.float64)
    hit_xyz = np.asarray(hit_xyz, dtype=np.float64)
    opl = np.asarray(opl, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    total = opl + n_ambient * np.linalg.norm(hit_xyz - ref, axis=-1)
    rho = np.linalg.norm(pupil_xy, axis=-1)
    theta = np.arctan2(pupil_xy[:, 1], pupil_xy[:, 0])
    chief = int(np.argmin(rho))
    return total - total[chief], rho, theta


def strehl_marechal(sigma_opd_m, lam_m):
    """Maréchal Strehl estimate from the piston/tip/tilt-removed RMS
    wavefront error sigma (metres) at wavelength lam (metres)."""
    x = 2.0 * np.pi * sigma_opd_m / lam_m
    return float(np.exp(-x * x))
