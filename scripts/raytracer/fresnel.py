# =============================================================================
# fresnel.py — complex-index Fresnel coefficients, Snell refraction, TIR.
#
# Conventions (used consistently across the tracer):
#   * d      : unit ray direction (pointing INTO the interface)
#   * n_hat  : unit surface normal pointing INTO the incident medium
#              (i.e. against the ray: cos_i = -d . n_hat >= 0)
#   * n1     : refractive index of the incident medium (assumed lossless or
#              weakly absorbing — real part used for direction geometry;
#              bulk loss is handled separately via Beer-Lambert)
#   * n2     : complex refractive index n + ik of the far medium
#   * s-polarization basis s_hat = normalize(d x n_hat); p_hat = d x s_hat
#
# Amplitude coefficients follow Born & Wolf / Hecht sign conventions:
#   rs = (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)
#   rp = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)
#   ts = 2 n1 cos_i / (n1 cos_i + n2 cos_t)
#   tp = 2 n1 cos_i / (n2 cos_i + n1 cos_t)
# with the complex branch Im(n2 cos_t) >= 0 (decaying transmitted wave), which
# reproduces |r|=1 with the correct analytic phase under TIR and handles
# metals without special cases.
#
# Power coefficients: R = |r|^2;  T = Re(n2 cos_t) / (n1 cos_i) * |t|^2.
# For lossless dielectrics R + T = 1 to machine precision (tested).
# =============================================================================
import numpy as np


def cos_theta_t(cos_i, n1, n2):
    """Complex cosine of the transmitted angle with the physical branch.

    cos_i : (N,) real, >= 0
    n1    : (N,) real (or complex with tiny Im; real part governs geometry)
    n2    : (N,) complex
    """
    n1 = np.asarray(n1, dtype=np.complex128)
    n2 = np.asarray(n2, dtype=np.complex128)
    sin_i2 = 1.0 - np.asarray(cos_i, dtype=np.float64) ** 2
    # Snell invariant: (n1/n2)^2 sin_i^2
    s2 = (n1 / n2) ** 2 * sin_i2
    ct = np.sqrt(1.0 - s2 + 0j)
    # Branch selection. The decay condition Im(n2*ct) >= 0 governs a
    # genuinely EVANESCENT/absorbed root; for an effectively PROPAGATING
    # one (|Im| at numerical-dust scale relative to |q|) the radiation
    # condition Re(n2*ct) >= 0 governs instead. The old unconditional
    # Im-rule flipped a perfectly propagating root whenever the INCIDENT
    # medium was weakly absorbing (water, k~1e-8) and n2 exactly lossless:
    # 1-s2 picks up a ~-1e-13 imaginary from n1's absorption, the
    # principal sqrt then has Im(ct) < 0 by dust, and the flip turned
    # ct = +0.99999 into -0.99999 — the near-cancelling Fresnel
    # denominator (n1 ci + n2 ct ~ n1 - n2) made |rs| ~ 15, and a
    # nested-cylinder chord loop amplified every bounce by |rs|^2 ~ 230
    # (the samples-instruments vial_cylindrical closure explosion, 1 mW
    # in -> 1e8 W booked; only water->lossless-glass pairs triggered it:
    # decalin (k=0) leaves Im exactly 0, bk7 (k>0) keeps Im(q) > 0).
    q = n2 * ct
    im, re = np.imag(q), np.real(q)
    propagating = np.abs(im) <= 1e-9 * np.abs(q)
    flip = np.where(propagating, re < 0.0, im < 0.0)
    ct = np.where(flip, -ct, ct)
    return ct


def fresnel_coeffs(cos_i, n1, n2):
    """Complex amplitude coefficients (rs, rp, ts, tp), vectorized."""
    n1 = np.asarray(n1, dtype=np.complex128)
    n2 = np.asarray(n2, dtype=np.complex128)
    ci = np.asarray(cos_i, dtype=np.float64)
    ct = cos_theta_t(ci, n1, n2)

    a1, a2 = n1 * ci, n2 * ct          # s-pol admittance-like terms
    b1, b2 = n2 * ci, n1 * ct          # p-pol
    rs = (a1 - a2) / (a1 + a2)
    rp = (b1 - b2) / (b1 + b2)
    ts = 2.0 * n1 * ci / (a1 + a2)
    tp = 2.0 * n1 * ci / (b1 + b2)
    return rs, rp, ts, tp, ct


def power_coeffs(rs, rp, ts, tp, cos_i, ct, n1, n2):
    """Power reflectance/transmittance (Rs, Rp, Ts, Tp).

    T uses the projected-Poynting factor Re(n2 ct)/(n1 cos_i); n1 must be
    (effectively) real — the tracer guarantees rays propagate in transparent
    or weakly absorbing media between surfaces.
    """
    n1r = np.real(np.asarray(n1))
    fac = np.real(np.asarray(n2) * ct) / (n1r * np.asarray(cos_i))
    Rs = np.abs(rs) ** 2
    Rp = np.abs(rp) ** 2
    Ts = fac * np.abs(ts) ** 2
    Tp = fac * np.abs(tp) ** 2
    return Rs, Rp, Ts, Tp


def is_tir(cos_i, n1, n2, tol=0.0):
    """True where total internal reflection occurs (real-index test)."""
    n1r = np.real(np.asarray(n1))
    n2r = np.real(np.asarray(n2))
    sin_i2 = 1.0 - np.asarray(cos_i) ** 2
    return (n1r / n2r) ** 2 * sin_i2 > 1.0 + tol


def reflect_dir(d, n_hat):
    """Specular reflection of direction(s) d about normal(s) n_hat."""
    d = np.asarray(d, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    return d - 2.0 * np.sum(d * n_hat, axis=-1, keepdims=True) * n_hat


def refract_dir(d, n_hat, cos_i, n1, n2):
    """Refracted unit direction via the vector form of Snell's law.

    Uses REAL parts of the indices (geometry); the amplitude/absorption
    physics lives in the Fresnel coefficients and Beer-Lambert. Where TIR
    holds the returned direction is meaningless — mask with is_tir().
    """
    d = np.asarray(d, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    eta = (np.real(np.asarray(n1)) / np.real(np.asarray(n2)))[..., None]
    ci = np.asarray(cos_i, dtype=np.float64)[..., None]
    ct2 = 1.0 - eta ** 2 * (1.0 - ci ** 2)
    ct = np.sqrt(np.maximum(ct2, 0.0))
    t = eta * d + (eta * ci - ct) * n_hat
    # normalize defensively (eta*... is unit-norm analytically)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    return t


def pol_basis(d, n_hat):
    """s_hat perpendicular to the plane of incidence; p_hat completes it.

    Degenerate (normal incidence) rays get an arbitrary but deterministic
    s_hat perpendicular to d.
    """
    d = np.asarray(d, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    s = np.cross(d, n_hat)
    norm = np.linalg.norm(s, axis=-1, keepdims=True)
    degenerate = (norm[..., 0] < 1e-12)
    if np.any(degenerate):
        # pick the global axis most orthogonal to d, project, normalize
        dd = d[degenerate]
        ax = np.zeros_like(dd)
        ax[np.arange(len(dd)), np.argmin(np.abs(dd), axis=-1)] = 1.0
        sfix = np.cross(dd, ax)
        sfix /= np.linalg.norm(sfix, axis=-1, keepdims=True)
        s[degenerate] = sfix
        norm = np.linalg.norm(s, axis=-1, keepdims=True)
    s = s / norm
    p = np.cross(d, s)
    return s, p


def basis_rotation_matrix(s_old, p_old, s_new, p_new):
    """The 2x2 re-expression coefficients (css, csp, cps, cpp) that
    rotate_jones applies to the field, factored out so P2's cumulative
    Jones-matrix tracking (poltransport.py) uses EXACTLY this transform
    and cannot drift from what the field itself undergoes. Both bases
    must be orthonormal pairs perpendicular to the SAME ray direction."""
    css = np.sum(s_new * s_old, axis=-1)
    csp = np.sum(s_new * p_old, axis=-1)
    cps = np.sum(p_new * s_old, axis=-1)
    cpp = np.sum(p_new * p_old, axis=-1)
    return css, csp, cps, cpp


def rotate_jones(Es, Ep, s_old, p_old, s_new, p_new):
    """Re-express Jones amplitudes in a new (s,p) basis.

    Both bases must be orthonormal pairs perpendicular to the SAME ray
    direction. The transport is the 2x2 rotation
      [Es'; Ep'] = [[s_new.s_old, s_new.p_old], [p_new.s_old, p_new.p_old]] .
    Unitary => |Es'|^2+|Ep'|^2 == |Es|^2+|Ep|^2 (tested).
    """
    css, csp, cps, cpp = basis_rotation_matrix(s_old, p_old, s_new, p_new)
    Es2 = css * Es + csp * Ep
    Ep2 = cps * Es + cpp * Ep
    return Es2, Ep2
