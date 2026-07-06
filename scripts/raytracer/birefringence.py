# =============================================================================
# birefringence.py — uniaxial-crystal double refraction (calcite/quartz/
# sapphire) for plane waves. Numpy-only, float64, vectorized over ray batches.
#
# Model: a uniaxial crystal has an ordinary index n_o and an extraordinary
# index n_e, and an optic axis unit vector c. The ORDINARY wave sees an
# isotropic sphere |k| = n_o*k0 (plain Snell). The EXTRAORDINARY wave sees an
# ellipsoidal normal surface; its phase index depends on the angle theta
# between the wavevector k and c:
#
#     1/n(theta)^2 = cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2        (cos = k.c)
#
# so n(0) = n_o (k || c) and n(90deg) = n_e (k _|_ c). The e-wave normal
# surface (in k0 = omega/c units, so indices ARE wavevector magnitudes) is
#
#     (K.c)^2 / n_o^2 + |K - (K.c)c|^2 / n_e^2 = 1
#
# DERIVED/VERIFIED here (Born & Wolf / Yariv "Optical Waves in Crystals");
# note the n_o/n_e placement — plugging K = n(theta)*k_hat back in reproduces
# the index formula above (pinned by test_normal_surface_residual). The
# extraordinary RAY (Poynting / group velocity) is NOT parallel to K: it points
# along grad_K of the dispersion relation,
#
#     s_e ~ K/n_e^2 + (K.c)*(1/n_o^2 - 1/n_e^2)*c
#
# and makes the walk-off angle rho with k. Signed walk-off convention here:
#     rho = angle_from_c(s_e) - angle_from_c(k)
# i.e. rho > 0 when the e-ray is deflected to a LARGER polar angle from the
# supplied c than the wavevector (the sign of a NEGATIVE uniaxial crystal such
# as calcite for the usual acute-angle geometry). POSITIVE uniaxials (quartz,
# n_e > n_o) give the opposite sign for the same geometry.
#
# Interface conventions match fresnel.py exactly:
#   * d / k_hat point INTO the interface (along propagation).
#   * n_hat is the unit surface normal pointing INTO the incident medium, i.e.
#     AGAINST the incident ray: cos_i = -d . n_hat >= 0.
# Tangential-wavevector continuity (k_t conserved) is used at every crossing;
# for the ordinary wave (and the isotropic limit n_e == n_o) everything reduces
# EXACTLY to fresnel.refract_dir (pinned by tests to 1e-12).
#
# Basis (eigenbasis): D-field basis orthonormal WITH k_hat, for the tracer's
# Jones decomposition. e_o_hat _|_ (k,c) plane (D of the ordinary wave); e_e_hat
# = k_hat x e_o_hat (in the (k,c) plane, _|_ k). Both are true D directions
# (D _|_ k always); the e-wave's E is slightly non-transverse but its D is not.
#
# Out of scope: absorbing crystals — any Im(n_o)/Im(n_e) is ignored (geometry
# uses real indices only, exactly as fresnel.refract_dir does); dispersion is
# supported by passing per-ray n_o/n_e. Biaxial crystals, optical activity, and
# gyrotropy are not modelled.
# =============================================================================
import numpy as np

from . import fresnel  # noqa: E402  (ordinary wave delegates to Snell here)

_DEGEN = 1e-9   # |k x c| below this: k || c, o/e degenerate (n_e(0) = n_o)


# ---------------------------------------------------------------------------
# small vectorization helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _bcast_vec(v, n):
    """Broadcast a (3,) or (n,3) direction to (n,3) float64 (not normalized)."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = np.broadcast_to(v, (n, 3))
    return np.ascontiguousarray(v)


def _bcast_scalar(x, n):
    """Broadcast a scalar or (n,) index to (n,) float64."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = np.broadcast_to(x, (n,))
    return np.ascontiguousarray(x)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


# ---------------------------------------------------------------------------
# phase index and eigenbasis
# ---------------------------------------------------------------------------
def n_e_theta(cos_kc, n_o, n_e):
    """Extraordinary PHASE index for a wavevector at angle theta to the optic
    axis, cos_kc = k_hat . c_hat.  1/n^2 = cos^2/n_o^2 + sin^2/n_e^2.

    Vectorized; n_o / n_e may be per-ray arrays (dispersion). Returns n_o at
    cos_kc = +-1 and n_e at cos_kc = 0 exactly.
    """
    c2 = np.asarray(cos_kc, dtype=np.float64) ** 2
    inv = c2 / np.asarray(n_o, dtype=np.float64) ** 2 \
        + (1.0 - c2) / np.asarray(n_e, dtype=np.float64) ** 2
    return 1.0 / np.sqrt(inv)


def eigenbasis(k_hat, c_axis):
    """Orthonormal D-field eigenbasis (e_o_hat, e_e_hat) transverse to k_hat.

    e_o_hat ~ k_hat x c_axis  (ordinary D, _|_ the (k,c) plane)
    e_e_hat = k_hat x e_o_hat (extraordinary D, in the (k,c) plane, _|_ k)

    At k_hat || c_axis (|k x c| < 1e-9) the o/e split is degenerate; a
    deterministic transverse pair is returned (no NaNs), via the global axis
    most orthogonal to k_hat (same trick as fresnel.pol_basis).
    """
    k = _unit(k_hat)
    n = k.shape[0]
    c = _unit(_bcast_vec(c_axis, n))
    eo = np.cross(k, c)
    norm = np.linalg.norm(eo, axis=-1)
    degen = norm < _DEGEN
    if np.any(degen):
        kk = k[degen]
        ax = np.zeros_like(kk)
        ax[np.arange(len(kk)), np.argmin(np.abs(kk), axis=-1)] = 1.0
        fix = np.cross(kk, ax)
        eo[degen] = fix
        norm = np.linalg.norm(eo, axis=-1)
    eo = eo / norm[:, None]
    ee = np.cross(k, eo)                     # unit (k _|_ eo, both unit)
    return eo, ee


def walkoff_angle(cos_kc, n_o, n_e):
    """Signed extraordinary walk-off angle rho (radians), cos_kc = k_hat.c_hat.

    rho = angle_from_c(s_e) - theta, with the e-ray polar direction from c
    given by (sin(theta)/n_e^2, cos(theta)/n_o^2). Equivalent to
    arctan((n_o^2/n_e^2) tan theta) - theta for theta in (0, 90deg). rho > 0
    for negative uniaxials (calcite) in the acute-angle geometry; opposite for
    positive uniaxials (quartz). Zero at theta = 0 and theta = 90deg.
    """
    cos_kc = np.clip(np.asarray(cos_kc, dtype=np.float64), -1.0, 1.0)
    n_o = np.asarray(n_o, dtype=np.float64)
    n_e = np.asarray(n_e, dtype=np.float64)
    theta = np.arccos(cos_kc)
    sin_t = np.sqrt(1.0 - cos_kc ** 2)
    angle_s = np.arctan2(sin_t / n_e ** 2, cos_kc / n_o ** 2)
    return angle_s - theta


# ---------------------------------------------------------------------------
# refraction into / out of the crystal
# ---------------------------------------------------------------------------
def refract_in(d_in, n_hat, c_axis, n1, n_o, n_e):
    """Refract from an isotropic medium (real index n1) INTO a uniaxial
    crystal, splitting into ordinary and extraordinary waves.

    d_in, n_hat : (n,3) unit vectors (fresnel convention: n_hat against the
                  ray, cos_i = -d.n_hat >= 0).
    c_axis      : (3,) or (n,3) optic axis (need not be unit).
    n1, n_o, n_e: scalars or (n,) (dispersion allowed).

    Returns dict of per-ray arrays:
      k_o (n,3)      ordinary unit wavevector (ordinary Snell with n_o)
      k_e (n,3)      extraordinary unit wavevector (tangential continuity +
                     e-wave normal surface, inward branch)
      s_e (n,3)      extraordinary RAY (Poynting) unit direction — walk-off
      n_phase_o (n,) = n_o
      n_phase_e (n,) = |K_e| = n(theta_e), the e phase index
      n_ray_e (n,)   = n_phase_e * cos(rho): OPL per metre ALONG the e-ray path
                       (phase advances along k_e; length measured along s_e)
      tir_o, tir_e (n,) bool: no propagating solution (evanescent)
    """
    d = _unit(d_in)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    c = _unit(_bcast_vec(c_axis, n))
    n1 = _bcast_scalar(n1, n)
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)

    cos_i = -_dot(d, nh)

    # --- ordinary wave: plain Snell with n_o -------------------------------
    k_o = fresnel.refract_dir(d, nh, cos_i, n1, n_o)
    tir_o = fresnel.is_tir(cos_i, n1, n_o)

    # --- extraordinary wave ------------------------------------------------
    # tangential incident wavevector (conserved), in k0 units:
    t_vec = n1[:, None] * (d - _dot(d, nh)[:, None] * nh)   # _|_ n_hat
    A = _dot(t_vec, t_vec)                                  # |k_t|^2
    p = _dot(t_vec, c)
    q = _dot(nh, c)
    kappa = n_e ** 2 / n_o ** 2 - 1.0

    # solve (K.c)^2/n_o^2 + |Kperp|^2/n_e^2 = 1 for K = t_vec + s*n_hat.
    # After substitution this is a s^2 + b s + cc = 0:
    a = 1.0 + kappa * q ** 2                                # > 0 (n_e,n_o > 0)
    b = 2.0 * kappa * p * q
    cc = A + kappa * p ** 2 - n_e ** 2
    disc = b ** 2 - 4.0 * a * cc
    tir_e = disc < 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    s = (-b - sq) / (2.0 * a)                               # inward branch
    K_e = t_vec + s[:, None] * nh
    n_phase_e = np.linalg.norm(K_e, axis=-1)
    k_e = K_e / n_phase_e[:, None]

    # extraordinary ray direction = grad_K of the dispersion relation
    Kc = _dot(K_e, c)
    grad = K_e / (n_e ** 2)[:, None] \
        + (Kc * (1.0 / n_o ** 2 - 1.0 / n_e ** 2))[:, None] * c
    s_e = _unit(grad)
    cos_rho = _dot(k_e, s_e)                                # > 0 always
    n_ray_e = n_phase_e * cos_rho

    return {
        "k_o": k_o, "k_e": k_e, "s_e": s_e,
        "n_phase_o": n_o.copy(), "n_phase_e": n_phase_e,
        "n_ray_e": n_ray_e, "tir_o": tir_o, "tir_e": tir_e,
    }


def ray_from_k(k_hat, c_axis, n_o, n_e):
    """e-wave: (s_ray, n_phase, n_ray) for an internal wavevector direction.

    s_ray is the Poynting/ray unit direction (grad_K of the dispersion
    relation), n_phase = n(theta) the phase index, n_ray = n_phase*cos(rho)
    the OPL-per-metre along the ray path. Used by the tracer for internal
    e-mode reflections (reflect k, recompute the ray)."""
    kh = _unit(k_hat)
    n = kh.shape[0]
    c = _unit(_bcast_vec(c_axis, n))
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)
    cos_kc = _dot(kh, c)
    n_phase = n_e_theta(cos_kc, n_o, n_e)
    K = n_phase[:, None] * kh
    grad = K / (n_e ** 2)[:, None] \
        + (_dot(K, c) * (1.0 / n_o ** 2 - 1.0 / n_e ** 2))[:, None] * c
    s_ray = _unit(grad)
    n_ray = n_phase * _dot(kh, s_ray)
    return s_ray, n_phase, n_ray


def k_from_ray(s_ray, c_axis, n_o, n_e):
    """Invert the e-wave ray<->wavevector map: unit k_hat whose Poynting
    direction is s_ray. The map s ~ M K with
    M = I/n_e^2 + (1/n_o^2 - 1/n_e^2) c c^T is linear and positive, so
    K ~ M^-1 s = n_e^2 s + (n_o^2 - n_e^2)(s.c) c (then normalized).
    Round-trip with ray_from_k is exact (pinned by test)."""
    s = _unit(s_ray)
    n = s.shape[0]
    c = _unit(_bcast_vec(c_axis, n))
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)
    K = (n_e ** 2)[:, None] * s \
        + ((n_o ** 2 - n_e ** 2) * _dot(s, c))[:, None] * c
    return _unit(K)


def refract_out(k_hat_int, mode_is_e, n_hat, c_axis, n_o, n_e, n2):
    """Refract an internal crystal wave OUT into an isotropic medium (real n2).

    Uses WAVEVECTOR tangential continuity: |k_int| = n_phase(mode, theta),
    k_t conserved, |k_out| = n2. n_hat is against the internal ray (into the
    crystal), cos_i = -k_hat_int.n_hat >= 0.

    k_hat_int : (n,3) internal unit wavevector direction
    mode_is_e : (n,) bool — True for extraordinary, False for ordinary
    Returns (d_out (n,3) unit, tir (n,) bool). For mode o this is EXACTLY
    fresnel.refract_dir with n1 = n_o (pinned by test).
    """
    kh = _unit(k_hat_int)
    n = kh.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    c = _unit(_bcast_vec(c_axis, n))
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)
    n2 = _bcast_scalar(n2, n)
    mode_is_e = np.asarray(mode_is_e, dtype=bool)

    cos_kc = _dot(kh, c)
    n_phase = np.where(mode_is_e, n_e_theta(cos_kc, n_o, n_e), n_o)

    K = n_phase[:, None] * kh
    K_t = K - _dot(K, nh)[:, None] * nh          # conserved tangential
    Kt2 = _dot(K_t, K_t)
    s2 = n2 ** 2 - Kt2
    tir = s2 < 0.0
    s = -np.sqrt(np.maximum(s2, 0.0))            # outgoing (along -n_hat side)
    K_out = K_t + s[:, None] * nh
    d_out = _unit(K_out)
    return d_out, tir
