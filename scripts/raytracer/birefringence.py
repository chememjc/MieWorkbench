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
# supported by passing per-ray n_o/n_e. Optical activity and gyrotropy are not
# modelled.
#
# BIAXIAL extension (n_x != n_y != n_z: KTP, LBO, ...) lives at the bottom of
# this module. The o/e split becomes a slow/fast two-sheet split of the
# quartic wave-normal surface
#
#     H(K) = u*P - Q + eps_x*eps_y*eps_z = 0,   u = |K|^2 (k0 units),
#     P = sum_i eps_i K_i^2,   Q = sum_i eps_i (eps_j + eps_k) K_i^2
#
# (K components in the CRYSTAL principal frame, eps_i = n_i^2; derived from
# the Fresnel equation of wave normals sum_i eps_i K_i^2/(eps_i - u) = 0 by
# clearing denominators and dividing out the trivial factor u — pinned by
# test_biaxial_normal_surface_residual). Interface refraction substitutes
# K = t_vec + s*n_hat (t_vec the conserved tangential wavevector, t.n = 0)
# giving a QUARTIC in s, solved batched via companion-matrix eigenvalues;
# the <= 2 real inward roots are the slow (larger n_phase) and fast sheets.
# D eigenvectors come from the 2x2 symmetric eigenproblem of the transverse
# projection of the inverse permittivity (robust everywhere, including
# principal planes where the k_i/(1/n^2 - 1/eps_i) form is 0/0), and the
# ray/Poynting direction from grad_K H ~ K_i*(P + eps_i*u - w_i).
#
# Biaxial honest limits: near an optic axis the two sheets meet (conical
# refraction) — eigenvectors degenerate there and the returned pair is an
# arbitrary orthonormal transverse basis; internal reflection is treated
# mode-preserving (cross-sheet coupling at internal reflections is ignored,
# the same tier of approximation as the effective-index interface Fresnel).
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


def degeneracy_mask(k_hat, c_axis):
    """Boolean (n,) mask: True where k_hat lies inside the conical-point
    degeneracy cone of the optic axis (|k x c| < _DEGEN, i.e. k_hat ~ ||
    c_axis). Split out from eigenbasis() so callers (the tracer's
    conical-point runtime guard, engine3.md Sec 7.2) can count/report
    degenerate rays without duplicating the threshold test; eigenbasis()
    itself uses this exact test internally, so the two never disagree."""
    k = _unit(k_hat)
    n = k.shape[0]
    c = _unit(_bcast_vec(c_axis, n))
    norm = np.linalg.norm(np.cross(k, c), axis=-1)
    return norm < _DEGEN


def eigenbasis(k_hat, c_axis):
    """Orthonormal D-field eigenbasis (e_o_hat, e_e_hat) transverse to k_hat.

    e_o_hat ~ k_hat x c_axis  (ordinary D, _|_ the (k,c) plane)
    e_e_hat = k_hat x e_o_hat (extraordinary D, in the (k,c) plane, _|_ k)

    At k_hat || c_axis (|k x c| < 1e-9, see degeneracy_mask) the o/e split
    is degenerate; a deterministic transverse pair is returned (no NaNs),
    via the global axis most orthogonal to k_hat (same trick as
    fresnel.pol_basis). No physics change here from the pre-degeneracy_mask
    version -- same construction, just sharing the threshold test.
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


def n_group_e_theta(cos_kc, mat_o, mat_e, lam_m, rel_step=1e-3):
    """Directional GROUP index for the extraordinary wave at FIXED angle:
    n_g,e(theta) = n_e(theta, lam) - lam * d n_e(theta, lam)/d lam, with the
    lambda-derivative taken at constant cos_kc (central difference over the
    materials' own dispersion).

    First-cut limitation (documented in RAYTRACER.md): the angular-dispersion
    term (d theta/d lam along a refracted ray) and the walk-off ray-vs-
    wavevector path-length distinction are neglected -- both are second-order
    for the mm-scale crystals in scope. mat_o / mat_e are Material objects
    (MaterialDB.get_uniaxial pair); vectorized over per-ray cos_kc/lam_m."""
    lam_m = np.asarray(lam_m, dtype=np.float64)
    h = lam_m * rel_step

    def ne(lm):
        return n_e_theta(cos_kc,
                         np.real(mat_o.n_complex(lm)),
                         np.real(mat_e.n_complex(lm)))

    d1 = (ne(lam_m + h) - ne(lam_m - h)) / (2.0 * h)
    return ne(lam_m) - lam_m * d1


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


# ===========================================================================
# EXACT uniaxial interface amplitudes  (Lekner 1991 boundary-value problem)
# ===========================================================================
# WHAT THIS IS.  refract_in/refract_out above give the GEOMETRY (o/e
# wavevectors, walk-off ray directions, phase indices).  The interface
# AMPLITUDES — how much of the incident field couples into each reflected
# and transmitted mode — were historically computed in the tracer with an
# "isotropic effective-index" approximation: the incident field is split
# into an o-channel (Fresnel with n_o) and an e-channel (Fresnel with the
# e phase index n(theta)), and the s<->p / o<->e CROSS terms are dropped.
# That approximation is worst exactly where the cross terms r_sp / r_ps are
# largest — optic axis at ~45 deg azimuth (they are ODD in azimuth and
# vanish at 0/90 deg; Lekner, J. Opt. Soc. Am. A 40, 722 (2023)).
#
# The functions below replace it with the EXACT amplitudes for a
# non-absorbing isotropic/uniaxial interface.  They solve the same
# boundary-value problem Lekner factors analytically in J. Phys.: Condens.
# Matter 3 (1991) 6121: continuity of tangential E and tangential H across
# the interface, four scalar conditions in four unknown mode amplitudes.
# We assemble and solve that 4x4 linear system per ray (vectorized,
# ~Fresnel cost) rather than transcribing Lekner's explicit closed forms —
# the linear solve IS the closed-form solution of the same equations, is
# far less error-prone, and reduces to fresnel.fresnel_coeffs by
# construction in the isotropic limit (oracle-pinned to 1e-12).
#
# CONVENTIONS (pinned; DO NOT introduce a second basis).
#   * d, n_hat as everywhere else: n_hat points INTO the incident medium,
#     cos_i = -d . n_hat >= 0.  zhat = -n_hat points INTO the transmission
#     medium.
#   * s/p basis is EXACTLY fresnel.pol_basis: s_hat = normalize(d x n_hat)
#     (perpendicular to the plane of incidence), p_hat = d x s_hat.  The
#     reflection Jones (r_ss, r_sp, r_ps, r_pp) returned here is in this
#     basis for BOTH the incident and the reflected wave (the reflected
#     p-direction is pol_basis(reflect_dir(d), n_hat).p), so it drops
#     straight into the tracer's rotate_jones bookkeeping and equals
#     fresnel rs/rp in the isotropic limit with NO extra sign flips.
#   * Crystal modes are returned in the o/e EIGENMODE basis: the ordinary
#     E-field is along e_o_hat = normalize(k x c) (D=E for the o wave);
#     the extraordinary E-field is E_e = eps^{-1} D_e with D_e = e_e_hat
#     (the in-plane transverse D of eigenbasis()), eps = n_o^2 I +
#     (n_e^2 - n_o^2) c c^T.  E_e is NOT transverse to k_e (walk-off).
#   * Every field is a plane wave exp(i(k.r - w t)); in k0 = w/c units
#     (|k| = refractive index) Faraday gives H = k x E for every mode,
#     isotropic or anisotropic.
#
# ENERGY / FLUX NORMALIZATION (engine3.md Sec 7.4).  Amplitudes are E-field
# ratios, but energy must be counted with the NORMAL time-averaged Poynting
# flux S_z = (1/2) Re(E x H*) . zhat — NOT |amp|^2 * k_z, because the
# e-wave's walk-off makes S_z differ from the k_z-weighting.  We therefore
# return every transmitted/reflected amplitude FLUX-NORMALIZED: multiplied
# by sqrt(|S_z(mode)| / S_z(incident)).  With that normalization the 4x4
# scattering matrix is UNITARY for a lossless interface, so the modulus-
# squared of each amplitude is directly its power fraction and
# |r_ss|^2 + |r_ps|^2 + T_o + T_e == 1 (checked to 1e-10 on a dense
# (theta, alpha, phi) grid for calcite and quartz).
#
# BRANCH CHOICE (documented per the P6 contract).  The transmitted e-wave
# normal component is the root of the ellipsoid quadratic that carries
# energy INTO the transmission medium.  refract_in() selects the inward
# WAVEVECTOR root s = (-b - sqrt(disc))/(2a); for a non-absorbing uniaxial
# this coincides with the positive-Poynting-flux (causal) root, which the
# flux-closure oracle confirms by returning S_z(e) > 0 across the whole
# grid (a wrong branch would give S_z(e) < 0 and break closure).  For the
# EXIT problem the two crystal modes share the tangential wavevector K_t;
# the reflected (back-into-crystal) o-root is +sqrt(n_o^2 - |K_t|^2) and
# the reflected e-root is s_refl = -b/a - s_incident (Vieta on the same
# quadratic — the second intersection of the K_t line with the ellipsoid),
# which is exact and needs no separate sign hunt.  The transmitted-into-
# lower-index mode is allowed to go evanescent (complex k_z under TIR):
# the arithmetic is complex128 throughout, so its normal flux Re(...) -> 0
# automatically and all the energy returns to the reflected channels.
# ---------------------------------------------------------------------------
def _eps_inv_apply(v, c, n_o, n_e):
    """eps^{-1} v for the uniaxial tensor eps = n_o^2 I + (n_e^2-n_o^2)cc^T:
    v/n_o^2 + (1/n_e^2 - 1/n_o^2)(c.v) c.  v may be complex, c/n_o/n_e real,
    all (n,3)/(n,)."""
    inv_o2 = 1.0 / n_o ** 2
    return v * inv_o2[:, None] \
        + ((1.0 / n_e ** 2 - inv_o2) * _dot(c, v))[:, None] * c


def _pz(K, E, zhat):
    """Normal time-averaged Poynting flux (x2, the 1/2 cancels in ratios)
    of a unit-amplitude plane wave: Re( (E x conj(H)) . zhat ), H = K x E.
    K may be complex (evanescent); E complex; zhat real unit (n,3)."""
    H = np.cross(K, E)
    return np.real(_dot(np.cross(E, np.conj(H)), zhat))


def _solve_bvp(cols, rhs):
    """Solve the interface 4x4 for the mode amplitudes.

    cols : sequence of four (n,4) complex arrays — the columns of the
           system matrix (one per unknown mode amplitude), each already
           containing [.s, .t, H.s, H.t] tangential projections.
    rhs  : (n,4,K) complex right-hand side(s).
    Returns x (n,4,K) with the four amplitudes stacked on axis 1.
    """
    A = np.stack(cols, axis=2)                       # (n,4,4)
    return np.linalg.solve(A, rhs)


def _tang_cols(vE, vH, s_hat, t_hat):
    """[vE.s, vE.t, vH.s, vH.t] stacked to (n,4) — a single BVP column."""
    return np.stack([_dot(vE, s_hat), _dot(vE, t_hat),
                     _dot(vH, s_hat), _dot(vH, t_hat)], axis=1)


def uniaxial_interface_in(d_in, n_hat, c_axis, n1, n_o, n_e, res=None):
    """EXACT amplitudes for an isotropic (n1) -> uniaxial interface.

    Returns a dict of per-ray arrays:
      rss, rsp, rps, rpp  : complex reflection Jones in the pol_basis(d)
                            s/p basis (reflected field = R . incident,
                            with R = [[rss, rsp], [rps, rpp]]); already
                            flux-normalized (reflection is same-medium).
      tos, top            : complex flux-normalized coupling of the
                            incident s / p field into the ORDINARY mode
                            (A_o = tos*Es_i + top*Ep_i; |A_o|^2 = o power).
      tes, tep            : same for the EXTRAORDINARY mode.
      s_new, p_new        : the incident s/p basis vectors (n,3).
      eo_o                : ordinary child D/E direction (unit, = child s_hat).
      eo_e                : extraordinary child s_hat (= eigenbasis eo of k_e).
    Geometry (k_o, k_e, s_e, n_phase_e, n_ray_e, tir_o, tir_e) is unchanged
    from refract_in(); pass its result as `res` to avoid recomputing it."""
    d = _unit(d_in)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    c = _unit(_bcast_vec(c_axis, n))
    n1 = _bcast_scalar(n1, n)
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)
    if res is None:
        res = refract_in(d, nh, c, n1, n_o, n_e)

    zhat = -nh
    s_hat, p_i = fresnel.pol_basis(d, nh)            # incident s, p
    t_hat = np.cross(zhat, s_hat)                    # 2nd tangential dir

    Ki = (n1[:, None] * d).astype(np.complex128)
    d_r = fresnel.reflect_dir(d, nh)
    Kr = (n1[:, None] * d_r).astype(np.complex128)
    p_r = np.cross(d_r, s_hat)                        # reflected p-basis

    ko_dir = res["k_o"]
    Ko = (n_o[:, None] * ko_dir).astype(np.complex128)
    eo_o, _ = eigenbasis(ko_dir, c)                  # ordinary E (unit)
    ke_dir = res["k_e"]
    Ke = (res["n_phase_e"][:, None] * ke_dir).astype(np.complex128)
    eo_e, ee_e = eigenbasis(ke_dir, c)               # D_e = ee_e
    E_e = _eps_inv_apply(ee_e.astype(np.complex128), c, n_o, n_e)

    s_c = s_hat.astype(np.complex128)
    p_r_c = p_r.astype(np.complex128)
    eo_o_c = eo_o.astype(np.complex128)
    # H = K x E for every mode
    H_rs = np.cross(Kr, s_c)
    H_rp = np.cross(Kr, p_r_c)
    H_o = np.cross(Ko, eo_o_c)
    H_e = np.cross(Ke, E_e)
    H_is = np.cross(Ki, s_c)
    H_ip = np.cross(Ki, p_i.astype(np.complex128))

    # unknowns [r_s, r_p, t_o, t_e]; BVP  cols . x = -[E_inc; H_inc]
    cols = [
        _tang_cols(s_c, H_rs, s_hat, t_hat),
        _tang_cols(p_r_c, H_rp, s_hat, t_hat),
        _tang_cols(-eo_o_c, -H_o, s_hat, t_hat),
        _tang_cols(-E_e, -H_e, s_hat, t_hat),
    ]
    rhs = -np.stack([
        _tang_cols(s_c, H_is, s_hat, t_hat),                 # incident s
        _tang_cols(p_i.astype(np.complex128), H_ip, s_hat, t_hat),  # inc p
    ], axis=2)                                       # (n,4,2)
    x = _solve_bvp(cols, rhs)                        # (n,4,2)

    # flux normalization: fraction of incident normal Poynting flux
    pz_inc = np.maximum(_pz(Ki, s_c, zhat), 1e-300)  # = n1 cos_i (both pols)
    pz_o = _pz(Ko, eo_o_c, zhat)
    pz_e = _pz(Ke, E_e, zhat)
    fo = np.sqrt(np.clip(pz_o, 0.0, None) / pz_inc)
    fe = np.sqrt(np.clip(pz_e, 0.0, None) / pz_inc)

    rss, rps, t_o_s, t_e_s = x[:, 0, 0], x[:, 1, 0], x[:, 2, 0], x[:, 3, 0]
    rsp, rpp, t_o_p, t_e_p = x[:, 0, 1], x[:, 1, 1], x[:, 2, 1], x[:, 3, 1]
    return {
        "rss": rss, "rsp": rsp, "rps": rps, "rpp": rpp,
        "tos": t_o_s * fo, "top": t_o_p * fo,
        "tes": t_e_s * fe, "tep": t_e_p * fe,
        "s_new": s_hat, "p_new": p_i,
        "eo_o": eo_o, "eo_e": eo_e,
    }


def uniaxial_interface_out(k_hat_int, mode_is_e, n_hat, c_axis,
                           n_o, n_e, n2, d_out=None):
    """EXACT amplitudes for a uniaxial -> isotropic (n2) interface.

    The incident is a single crystal eigenmode (mode_is_e selects o vs e)
    with internal wavevector DIRECTION k_hat_int (unit).  Returns a dict:
      t_s, t_p   : complex flux-normalized coupling of the incident mode
                   into the transmitted isotropic s / p field
                   (Es_out = t_s*A_in, Ep_out = t_p*A_in; |t_s|^2+|t_p|^2 =
                   T_total).  Basis = pol_basis(k_hat_int, n_hat); the
                   transmitted p-direction is pol_basis(d_out, n_hat).p.
                   These come from the FULL 4x4 BVP (both reflected crystal
                   modes present), so they are exact.
      T_total    : real transmitted power fraction |t_s|^2+|t_p|^2.  The
                   transmission side is isotropic, so this is clean and
                   unambiguous; reflectance is R = 1 - T_total by energy
                   conservation (the crystal-side incident/reflected fields
                   interfere, so the reflected mode fluxes do NOT simply add
                   to R — full-field S_z continuity across the interface is
                   exact to machine precision, but the SELF-flux split is
                   not, unlike the isotropic-reflected ENTRY problem).
      refl_phase : arg of the same-mode reflection amplitude (a sensible
                   phase for the tracer's mode-preserving internal-reflection
                   child, which carries the exact power R = 1 - T_total).
      s_new, p_new : incident s/p basis (n,3).
      tir        : bool, transmitted wave evanescent (no real refraction).
    """
    kh = _unit(k_hat_int)
    n = kh.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    c = _unit(_bcast_vec(c_axis, n))
    n_o = _bcast_scalar(n_o, n)
    n_e = _bcast_scalar(n_e, n)
    n2 = _bcast_scalar(n2, n)
    is_e = np.asarray(mode_is_e, dtype=bool)
    zhat = -nh

    cos_kc = _dot(kh, c)
    n_phase = np.where(is_e, n_e_theta(cos_kc, n_o, n_e), n_o)
    K_in = n_phase[:, None] * kh
    t_vec = K_in - _dot(K_in, nh)[:, None] * nh      # conserved tangential
    Kt2 = _dot(t_vec, t_vec)

    # incident-mode E-vector (o: unit e_o; e: eps^{-1} D_e)
    eo_i, ee_i = eigenbasis(kh, c)
    E_in = np.where(is_e[:, None],
                    _eps_inv_apply(ee_i.astype(np.complex128), c, n_o, n_e),
                    eo_i.astype(np.complex128))
    K_in_c = K_in.astype(np.complex128)

    # reflected ORDINARY (sphere): back into the crystal, K.nh > 0
    s_o = np.sqrt(np.clip(n_o ** 2 - Kt2, 0.0, None))
    K_ro = t_vec + s_o[:, None] * nh
    eo_ro, _ = eigenbasis(_unit(K_ro), c)
    K_ro_c = K_ro.astype(np.complex128)
    E_ro = eo_ro.astype(np.complex128)

    # reflected EXTRAORDINARY: second root of the ellipsoid quadratic for
    # this K_t (Vieta: s1+s2 = -b/a; incident root = K_in.nh).
    p = _dot(t_vec, c)
    q = _dot(nh, c)
    kappa = n_e ** 2 / n_o ** 2 - 1.0
    a = 1.0 + kappa * q ** 2
    b = 2.0 * kappa * p * q
    s_in = _dot(K_in, nh)
    s_re = -b / a - s_in
    K_re = t_vec + s_re[:, None] * nh
    ke_dir_r = _unit(K_re)
    _, ee_re = eigenbasis(ke_dir_r, c)
    K_re_c = K_re.astype(np.complex128)
    E_re = _eps_inv_apply(ee_re.astype(np.complex128), c, n_o, n_e)

    # transmitted isotropic s/p (n2); complex k_z handles TIR
    s_hat, _ = fresnel.pol_basis(kh, nh)
    t_hat = np.cross(zhat, s_hat)
    if d_out is None:
        d_out, _ = refract_out(kh, is_e, nh, c, n_o, n_e, np.real(n2))
    s2 = (n2 ** 2 - Kt2).astype(np.complex128)
    tir = np.real(s2) < 0.0
    qz = np.sqrt(s2)                                 # complex under TIR
    # transmitted wavevector = K_t + qz * zhat  (zhat into medium 2)
    Kt2v = t_vec.astype(np.complex128)
    K_ts = Kt2v + qz[:, None] * zhat
    E_ts = s_hat.astype(np.complex128)               # transmitted s
    p_out = np.cross(d_out, s_hat)                    # transmitted p dir
    E_tp = p_out.astype(np.complex128)
    # under TIR both transmitted modes are evanescent with the same
    # complex qz; when propagating |K_ts| = n2.  Both share K_t + qz*zhat.
    K_tp = K_ts

    H_ro = np.cross(K_ro_c, E_ro)
    H_re = np.cross(K_re_c, E_re)
    H_ts = np.cross(K_ts, E_ts)
    H_tp = np.cross(K_tp, E_tp)
    H_in = np.cross(K_in_c, E_in)

    cols = [
        _tang_cols(E_ro, H_ro, s_hat, t_hat),        # r_o
        _tang_cols(E_re, H_re, s_hat, t_hat),        # r_e
        _tang_cols(-E_ts, -H_ts, s_hat, t_hat),      # t_s
        _tang_cols(-E_tp, -H_tp, s_hat, t_hat),      # t_p
    ]
    rhs = -_tang_cols(E_in, H_in, s_hat, t_hat)[:, :, None]   # (n,4,1)
    x = _solve_bvp(cols, rhs)[:, :, 0]               # (n,4)

    pz_inc = np.maximum(np.abs(_pz(K_in_c, E_in, zhat)), 1e-300)
    f_ro = np.sqrt(np.abs(_pz(K_ro_c, E_ro, zhat)) / pz_inc)
    f_re = np.sqrt(np.abs(_pz(K_re_c, E_re, zhat)) / pz_inc)
    f_ts = np.sqrt(np.clip(_pz(K_ts, E_ts, zhat), 0.0, None) / pz_inc)
    f_tp = np.sqrt(np.clip(_pz(K_tp, E_tp, zhat), 0.0, None) / pz_inc)

    r_o, r_e, t_s, t_p = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    ts_n = t_s * f_ts
    tp_n = t_p * f_tp
    T_total = np.abs(ts_n) ** 2 + np.abs(tp_n) ** 2
    r_same = np.where(is_e, r_e * f_re, r_o * f_ro)
    return {
        "t_s": ts_n, "t_p": tp_n, "T_total": T_total,
        "refl_phase": np.angle(r_same),
        "s_new": s_hat, "p_new": np.cross(kh, s_hat),
        "tir": tir,
    }


# ===========================================================================
# BIAXIAL crystals (slow/fast two-sheet split) — see module header
# ===========================================================================
def _bcast_frame(frame, n):
    """Broadcast a (3,3) or (n,3,3) crystal frame to (n,3,3) float64.
    Rows are the principal axes expressed in GLOBAL coordinates, so
    v_crystal = frame @ v_global."""
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim == 2:
        f = np.broadcast_to(f, (n, 3, 3))
    return np.ascontiguousarray(f)


def _bcast_eps(eps, n):
    """Broadcast (3,) or (n,3) principal permittivities (n_i^2) to (n,3)."""
    e = np.asarray(eps, dtype=np.float64)
    if e.ndim == 1:
        e = np.broadcast_to(e, (n, 3))
    return np.ascontiguousarray(e)


def _to_crystal(v, frame):
    return np.einsum("nij,nj->ni", frame, v)


def _to_global(v, frame):
    return np.einsum("nji,nj->ni", frame, v)


def biaxial_modes_for_k(k_hat, frame, eps):
    """Per-sheet phase indices + D eigenvectors for an internal wavevector
    DIRECTION k_hat (unit, global coords).

    Solves the 2x2 symmetric eigenproblem of the inverse permittivity
    projected onto the plane transverse to k: eigenvalues are 1/n^2 of the
    two sheets, eigenvectors the D directions. Robust everywhere the
    k_i/(1/n^2 - 1/eps_i) closed form is 0/0 (principal planes); at an
    optic axis the sheets meet and the returned basis is an arbitrary
    orthonormal transverse pair (conical-refraction limitation).

    Returns dict: n_slow >= n_fast (n,), D_slow, D_fast (n,3) unit, global.
    """
    k = _unit(k_hat)
    n = k.shape[0]
    fr = _bcast_frame(frame, n)
    eta = 1.0 / _bcast_eps(eps, n)               # (n,3) inverse permittivity

    # deterministic transverse basis (a, b) _|_ k (same trick as eigenbasis)
    ax = np.zeros_like(k)
    ax[np.arange(n), np.argmin(np.abs(k), axis=-1)] = 1.0
    a = _unit(np.cross(k, ax))
    b = np.cross(k, a)

    ac = _to_crystal(a, fr)
    bc = _to_crystal(b, fr)
    Baa = _dot(ac * eta, ac)
    Bbb = _dot(bc * eta, bc)
    Bab = _dot(ac * eta, bc)

    # closed-form symmetric 2x2 eigen: lam = mean +- r
    mean = 0.5 * (Baa + Bbb)
    diff = 0.5 * (Baa - Bbb)
    r = np.sqrt(diff ** 2 + Bab ** 2)
    lam_slow = mean - r                          # smaller 1/n^2 -> larger n
    lam_fast = mean + r
    # eigenvector for lam_slow in the (a,b) plane; guard the degenerate
    # (optic-axis / isotropic) case r ~ 0 with the (1,0) fallback
    va = np.where(np.abs(Bab) > _DEGEN * np.maximum(1.0, np.abs(mean)),
                  Bab, np.where(diff <= 0.0, 1.0, 0.0))
    vb = np.where(np.abs(Bab) > _DEGEN * np.maximum(1.0, np.abs(mean)),
                  lam_slow - Baa, np.where(diff <= 0.0, 0.0, 1.0))
    norm = np.sqrt(va ** 2 + vb ** 2)
    norm = np.where(norm < _DEGEN, 1.0, norm)
    va, vb = va / norm, vb / norm
    D_slow = va[:, None] * a + vb[:, None] * b
    D_fast = -vb[:, None] * a + va[:, None] * b  # orthogonal partner
    return {
        "n_slow": 1.0 / np.sqrt(lam_slow),
        "n_fast": 1.0 / np.sqrt(lam_fast),
        "D_slow": D_slow, "D_fast": D_fast,
    }


def biaxial_ray_from_k(K, frame, eps):
    """Ray (Poynting) unit direction + n_ray for wavevectors K on the
    normal surface (K in k0 units, global coords, |K| = n_phase).

    grad_K H ~ K_i * (P + eps_i*u - w_i) componentwise in the crystal
    frame (w_i = eps_i*(eps_j + eps_k)). Where the gradient vanishes
    (isotropic degeneracy / conical point) the wave direction is returned.
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    fr = _bcast_frame(frame, n)
    ep = _bcast_eps(eps, n)
    Kc = _to_crystal(K, fr)
    u = _dot(Kc, Kc)
    P = _dot(ep * Kc, Kc)
    w = ep * (np.sum(ep, axis=-1, keepdims=True) - ep)   # eps_i*(eps_j+eps_k)
    grad_c = Kc * (P[:, None] + ep * u[:, None] - w)
    gnorm = np.linalg.norm(grad_c, axis=-1)
    k_hat = _unit(K)
    bad = gnorm < _DEGEN * np.maximum(1.0, u)
    grad_c[bad] = _to_crystal(k_hat[bad], fr[bad])
    s_ray = _unit(_to_global(grad_c, fr))
    # orient along propagation (grad points outward on the surface; make
    # s.k >= 0 which is the physical energy-flow side)
    flip = _dot(s_ray, k_hat) < 0.0
    s_ray[flip] = -s_ray[flip]
    n_phase = np.sqrt(u)
    n_ray = n_phase * _dot(k_hat, s_ray)
    return s_ray, n_phase, n_ray


def _biaxial_quartic_roots(t_vec, n_hat, frame, eps):
    """Real roots s of H(t_vec + s*n_hat) = 0, batched via companion-matrix
    eigenvalues. Returns (roots (n,4) float64, real_mask (n,4) bool)."""
    n = t_vec.shape[0]
    fr = _bcast_frame(frame, n)
    ep = _bcast_eps(eps, n)
    tc = _to_crystal(t_vec, fr)
    nc = _to_crystal(n_hat, fr)
    A = _dot(tc, tc)
    w = ep * (np.sum(ep, axis=-1, keepdims=True) - ep)
    det = np.prod(ep, axis=-1)

    P0 = _dot(ep * tc, tc)
    P1 = 2.0 * _dot(ep * tc, nc)
    P2 = _dot(ep * nc, nc)                       # > 0 always
    Q0 = _dot(w * tc, tc)
    Q1 = 2.0 * _dot(w * tc, nc)
    Q2 = _dot(w * nc, nc)

    # H(s) = (A + s^2)(P0 + P1 s + P2 s^2) - (Q0 + Q1 s + Q2 s^2) + det
    c4 = P2
    c3 = P1
    c2 = P0 + A * P2 - Q2
    c1 = A * P1 - Q1
    c0 = A * P0 - Q0 + det

    comp = np.zeros((n, 4, 4))
    comp[:, 1, 0] = comp[:, 2, 1] = comp[:, 3, 2] = 1.0
    comp[:, 0, 3] = -c0 / c4
    comp[:, 1, 3] = -c1 / c4
    comp[:, 2, 3] = -c2 / c4
    comp[:, 3, 3] = -c3 / c4
    roots = np.linalg.eigvals(comp)              # (n,4) complex

    # Double real roots (isotropic limit, conical points) come back from
    # the eigensolver as a complex pair with |Im| ~ sqrt(machine eps), so
    # a strict imag cut misclassifies them. Newton-polish the real parts
    # against the real quartic, then classify by residual: true real
    # roots polish to ~1e-14, genuinely complex (evanescent/TIR) pairs
    # leave an O(Im^2) residual.
    s = roots.real.copy()
    C = [c0[:, None], c1[:, None], c2[:, None], c3[:, None], c4[:, None]]

    def _h(x):
        return C[0] + x * (C[1] + x * (C[2] + x * (C[3] + x * C[4])))

    def _dh(x):
        return C[1] + x * (2 * C[2] + x * (3 * C[3] + x * 4 * C[4]))

    # safeguarded Newton: at a double root H' ~ 0 and a raw h/dh step is
    # float-noise/float-noise (diverges); only accept improving steps —
    # eigenvalue-accurate double roots already satisfy the residual gate
    for _ in range(3):
        h = _h(s)
        dh = _dh(s)
        step = h / np.where(np.abs(dh) < 1e-300, 1.0, dh)
        cand = s - np.where(np.abs(dh) < 1e-300, 0.0, step)
        s = np.where(np.abs(_h(cand)) < np.abs(h), cand, s)
    scale = (np.abs(C[0]) + np.abs(s) * (np.abs(C[1]) + np.abs(s) * (
        np.abs(C[2]) + np.abs(s) * (np.abs(C[3]) + np.abs(s) * np.abs(C[4])))))
    real = (np.abs(_h(s)) <= 1e-9 * np.maximum(scale, 1e-30)) \
        & (np.abs(roots.imag) < 1e-2 * (1.0 + np.abs(roots.real)))
    return s, real


def refract_in_biaxial(d_in, n_hat, frame, n1, eps):
    """Refract from an isotropic medium (real index n1) INTO a biaxial
    crystal, splitting into slow and fast sheet waves.

    d_in, n_hat : (n,3) unit vectors, fresnel convention (n_hat against the
                  incident ray, cos_i = -d.n_hat >= 0).
    frame       : (3,3) or (n,3,3), rows = principal axes in global coords.
    n1          : scalar or (n,).
    eps         : (3,) or (n,3) principal permittivities (n_x^2,n_y^2,n_z^2),
                  per-ray for dispersion.

    Returns dict of per-ray arrays, per sheet m in {slow, fast} (slow =
    larger n_phase; when only one inward root survives it is assigned to
    the slow sheet — the higher-index sheet supports the larger tangential
    wavevector, so the fast sheet is the one that goes evanescent first):
      k_<m> (n,3) unit wavevector, s_<m> (n,3) unit ray direction,
      n_phase_<m>, n_ray_<m> (n,), D_<m> (n,3) unit D eigenvector,
      tir_<m> (n,) bool.
    """
    d = _unit(d_in)
    n = d.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    n1 = _bcast_scalar(n1, n)
    fr = _bcast_frame(frame, n)
    ep = _bcast_eps(eps, n)

    t_vec = n1[:, None] * (d - _dot(d, nh)[:, None] * nh)   # _|_ n_hat
    roots, real = _biaxial_quartic_roots(t_vec, nh, fr, ep)

    # inward branch: K.n_hat = s < 0 (n_hat points against the incident
    # ray). Keep real inward roots, sorted by n_phase DESC (slow first).
    inward = real & (roots < 1e-12)
    A = _dot(t_vec, t_vec)
    n_ph = np.sqrt(A[:, None] + roots ** 2)
    n_ph_sort = np.where(inward, n_ph, -np.inf)
    order = np.argsort(-n_ph_sort, axis=1)
    idx = np.arange(n)
    out = {}
    for j, name in enumerate(("slow", "fast")):
        pick = order[:, j]
        ok = inward[idx, pick]
        s = np.where(ok, roots[idx, pick], -1.0)   # placeholder when TIR
        K = t_vec + s[:, None] * nh
        k_hat = _unit(K)
        s_ray, n_phase, n_ray = biaxial_ray_from_k(K, fr, ep)
        modes = biaxial_modes_for_k(k_hat, fr, ep)
        # the eigenvector belonging to THIS sheet: match by phase index
        d_slow = np.abs(modes["n_slow"] - n_phase)
        d_fast = np.abs(modes["n_fast"] - n_phase)
        use_slow = (d_slow <= d_fast)[:, None]
        D = np.where(use_slow, modes["D_slow"], modes["D_fast"])
        out["k_%s" % name] = k_hat
        out["s_%s" % name] = s_ray
        out["n_phase_%s" % name] = n_phase
        out["n_ray_%s" % name] = n_ray
        out["D_%s" % name] = D
        out["tir_%s" % name] = ~ok
    return out


def reflect_internal_biaxial(k_in, n_hat, frame, eps):
    """Mode-preserving internal reflection of a biaxial wave: the
    tangential wavevector is conserved and the reflected wave returns on
    the SAME sheet (cross-sheet coupling ignored — see module header).

    k_in  : (n,3) incident wavevector in k0 units (|k_in| = n_phase_in).
    n_hat : (n,3) unit normal AGAINST the incident wave (cos_i > 0).
    Returns (K_refl (n,3) k0 units, ok (n,) bool). ok=False marks rays
    with no returning real root on the incident sheet (grazing/conical
    corner cases); callers should kill those rays into the seam bucket.
    """
    k_in = np.asarray(k_in, dtype=np.float64)
    n = k_in.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    fr = _bcast_frame(frame, n)
    ep = _bcast_eps(eps, n)

    t_vec = k_in - _dot(k_in, nh)[:, None] * nh
    roots, real = _biaxial_quartic_roots(t_vec, nh, fr, ep)
    # the reflected wave travels back INTO the crystal: K.n_hat = s > 0
    # (n_hat is against the incident wave). Sheet identity is NOT "nearest
    # n_phase" (a large direction change moves both sheet indices): a root
    # is on the slow/fast sheet iff its |K| matches that sheet's mode
    # index for its own direction. Pick the returning root on the SAME
    # sheet as the incident wave.
    back = real & (roots > 1e-12)
    n_in = np.linalg.norm(k_in, axis=-1)
    in_modes = biaxial_modes_for_k(_unit(k_in), fr, ep)
    in_is_slow = (np.abs(in_modes["n_slow"] - n_in)
                  <= np.abs(in_modes["n_fast"] - n_in))
    score = np.full(roots.shape, np.inf)
    for j in range(4):
        K_j = t_vec + roots[:, j][:, None] * nh
        n_j = np.linalg.norm(K_j, axis=-1)
        m_j = biaxial_modes_for_k(_unit(K_j), fr, ep)
        n_same = np.where(in_is_slow, m_j["n_slow"], m_j["n_fast"])
        score[:, j] = np.where(back[:, j], np.abs(n_j - n_same), np.inf)
    pick = np.argmin(score, axis=1)
    idx = np.arange(n)
    best = score[idx, pick]
    # a root genuinely on the sheet matches its own mode index to solver
    # precision; anything worse means no same-sheet returning wave exists
    ok = best < 1e-6 * np.maximum(1.0, n_in)
    s = np.where(ok, roots[idx, pick], 1.0)
    K_refl = t_vec + s[:, None] * nh
    return K_refl, ok


def refract_out_biaxial(K_int, n_hat, n2):
    """Refract an internal biaxial wave OUT into an isotropic medium.
    Pure tangential-wavevector continuity — needs only the internal K in
    k0 units (|K_int| = n_phase), not the crystal tensor.

    n_hat is against the internal wave (into the crystal), cos_i > 0.
    Returns (d_out (n,3) unit, tir (n,) bool)."""
    K = np.asarray(K_int, dtype=np.float64)
    n = K.shape[0]
    nh = _unit(_bcast_vec(n_hat, n))
    n2 = _bcast_scalar(n2, n)
    K_t = K - _dot(K, nh)[:, None] * nh
    Kt2 = _dot(K_t, K_t)
    s2 = n2 ** 2 - Kt2
    tir = s2 < 0.0
    s = -np.sqrt(np.maximum(s2, 0.0))
    return _unit(K_t + s[:, None] * nh), tir


# ===========================================================================
# BIAXIAL conical refraction — optic axes, cone angle, per-ray dispatch, and
# the perturbed two-sheet fan.
#
# The two sheets of the biaxial wave-normal surface touch at four points, the
# two optic axes (binormals) +-c1, +-c2. There the eigenproblem in
# biaxial_modes_for_k is doubly degenerate and biaxial_ray_from_k's gradient
# has a conical (tangent-cone) singularity: a single wave normal along an
# optic axis fans out into a whole CONE of ray directions (Hamilton's
# internal conical refraction). Rather than special-case a closed-form ring
# law, the design reproduces the cone from the SAME machinery used
# everywhere else: perturb the wave normal by a tiny angle toward a set of
# transverse azimuths and solve both sheets at each perturbed k. As the
# perturbation shrinks the perturbed rays converge onto the exact Hamilton
# cone, so the fan is correct by construction and shares every convention
# (frame rows = principal axes in global coords; eps = n_i^2; slow = larger
# n_phase) with the rest of the biaxial block.
# ===========================================================================
def biaxial_optic_axes(eps):
    """The two optic axes (binormals) of a biaxial crystal, as (2,3) unit
    vectors in the CRYSTAL PRINCIPAL frame (components along the principal
    axes, i.e. the frame in which eps is diagonal).

    For principal indices n1 < n2 < n3 (eps_i = n_i^2 sorted ascending) the
    axes lie in the plane spanned by the extreme axes (n1 and n3) and are
    symmetric about the n3 (largest-index) axis, tilted toward the n1 axis
    by the polar angle theta measured FROM the n3 axis:

        tan^2(theta) = (1/n1^2 - 1/n2^2) / (1/n2^2 - 1/n3^2)

    (Born & Wolf, Principles of Optics 7th ed. Sec. 15.3.3, Eq. (16);
    Landau & Lifshitz, Electrodynamics of Continuous Media Sec. 97. Same
    formula the KTP principal-plane oracle in test_biaxial.py already uses.)
    Verified against the existing solver: biaxial_modes_for_k's two sheet
    indices coincide on these directions to solver precision.

    eps : (3,) principal permittivities (n_x^2, n_y^2, n_z^2), any order.
    Returns (2,3): [c1, c2] with c1, c2 symmetric about the extreme axis
    (c1 tilted +theta, c2 tilted -theta). Degenerate uniaxial limits are
    handled gracefully: eps of the two smallest principal values equal ->
    theta = 0 -> both axes collapse onto the largest-index axis; the two
    largest equal -> theta = 90deg -> both collapse onto the smallest-index
    axis (the returned pair is then the SAME unit vector twice).
    """
    e = np.asarray(eps, dtype=np.float64).reshape(3)
    order = np.argsort(e)                    # [i_small, i_mid, i_large]
    ia, ib, ic = int(order[0]), int(order[1]), int(order[2])
    inv = 1.0 / e
    num = max(inv[ia] - inv[ib], 0.0)        # 1/n1^2 - 1/n2^2 >= 0
    den = max(inv[ib] - inv[ic], 0.0)        # 1/n2^2 - 1/n3^2 >= 0
    # arctan2 gives theta = 0 when num == 0 (two small eps equal) and
    # theta = pi/2 when den == 0 (two large eps equal), so both uniaxial
    # limits fall out without a branch.
    theta = np.arctan2(np.sqrt(num), np.sqrt(den))
    e_small = np.zeros(3); e_small[ia] = 1.0    # n1 (smallest-index) axis
    e_large = np.zeros(3); e_large[ic] = 1.0    # n3 (largest-index) axis
    ct, st = np.cos(theta), np.sin(theta)
    c1 = ct * e_large + st * e_small
    c2 = ct * e_large - st * e_small
    # uniaxial limit: the axes collapse onto one principal axis (c1 || c2);
    # return that merged axis twice, sign-canonicalized to be identical.
    if abs(float(np.dot(c1, c2))) > 1.0 - 1e-12:
        c2 = c1
    return np.stack([c1, c2])


def cone_half_angle(eps):
    """Opening angle A (radians) of Hamilton's INTERNAL conical-refraction
    cone: the angle between the two extreme (diametrically-opposite) ray
    generators produced when a single wave normal points exactly along an
    optic axis. Equivalently the angle between the two limiting s_ray
    directions reached by approaching the optic axis with in-plane-of-the-
    axes perturbations from the two opposite sides.

        tan A = sqrt((n2^2 - n1^2)(n3^2 - n2^2)) / (n1 n3)

    (Born & Wolf, Principles of Optics 7th ed. Sec. 15.3.3; Berry & Jeffrey,
    "Conical diffraction", Prog. Optics 50 (2007) Sec. 2. One optic-axis
    generator of this cone is the wave normal itself, so the cone is skewed
    -- it TOUCHES the optic-axis direction -- and A is the full opening, not
    a half-cone semi-angle; the conventional semi-angle is A/2.) Cross-checked
    numerically against biaxial_ray_from_k at epsilon-perturbed wave normals
    (test_biaxial.py) to <1e-8 relative.

    eps : (3,) principal permittivities (n_x^2, n_y^2, n_z^2), any order.
    Returns a Python float (radians).
    """
    e = np.asarray(eps, dtype=np.float64).reshape(3)
    n1s, n2s, n3s = np.sort(e)               # n1^2 <= n2^2 <= n3^2
    num = np.sqrt(max(n2s - n1s, 0.0) * max(n3s - n2s, 0.0))
    den = np.sqrt(n1s * n3s)
    return float(np.arctan2(num, den))


def axis_proximity(k_hat, eps, frame):
    """Per-ray angle (radians) from each wave normal to its NEAREST optic
    axis -- the tracer's conical-refraction dispatch criterion. Zero exactly
    on an axis; pi/2 farthest away.

    k_hat : (N,3) unit wave-normal directions in GLOBAL coords.
    eps   : (3,) principal permittivities (n_x^2, n_y^2, n_z^2).
    frame : (3,3) or (N,3,3), rows = principal axes in global coords (the
            SAME convention as biaxial_modes_for_k: v_crystal = frame @
            v_global).
    Returns (N,) angles. Optic axes are treated as undirected lines, so the
    result is in [0, pi/2] (proximity to +-axis alike).
    """
    k = _unit(k_hat)
    n = k.shape[0]
    fr = _bcast_frame(frame, n)
    axes_c = biaxial_optic_axes(eps)             # (2,3) crystal frame
    # crystal -> global for each ray: v_global = frame^T @ v_crystal
    axes_g = np.einsum("nji,kj->nki", fr, axes_c)   # (n,2,3)
    cosang = np.abs(np.einsum("nki,ni->nk", axes_g, k))   # (n,2), line sense
    cosang = np.clip(cosang, 0.0, 1.0)
    return np.min(np.arccos(cosang), axis=1)


def conical_fan(k_hat, frame, eps, n_fan, eps_angle):
    """Perturbed two-sheet fan approximating Hamilton's internal cone for a
    SINGLE degenerate wave normal k_hat (unit, GLOBAL coords) pointing at or
    near an optic axis of the crystal. Perturbs k_hat by the small angle
    eps_angle toward n_fan equally-spaced transverse azimuths and solves
    BOTH sheets at each perturbed wave normal with the existing
    biaxial_modes_for_k / biaxial_ray_from_k -- the 2*n_fan children trace
    out the cone (max angle between any two s_ray -> cone_half_angle(eps) as
    eps_angle -> 0, and the polarization D rotates by phi/2 around the ring,
    the classic half-turn law).

    k_hat     : (3,) unit wave normal, global coords (already inside the
                crystal, at/near the optic axis).
    frame     : (3,3) or (n_fan,3,3), rows = principal axes in global coords.
    eps       : (3,) or (n_fan,3) principal permittivities (n_x^2..n_z^2).
    n_fan     : number of azimuths; the returned arrays have 2*n_fan rows.
    eps_angle : perturbation half-cone angle (radians); small (~1e-3).

    Returns a dict of GLOBAL-frame arrays, slow-sheet block first then fast
    (row j and row n_fan+j share azimuth phi_j):
      k       (2N,3) unit perturbed wave normals,
      s_ray   (2N,3) unit ray / Poynting directions,
      D_hat   (2N,3) unit D-field eigenvectors (transverse to k),
      n_phase (2N,)  phase indices,
      n_ray   (2N,)  ray indices (n_phase * cos walk-off),
      sheet   (2N,)  0 = slow, 1 = fast,
      azimuth (2N,)  perturbation azimuth phi in [0, 2pi).
    """
    k0 = _unit(np.asarray(k_hat, dtype=np.float64).reshape(3))
    # deterministic transverse basis (e1, e2) _|_ k0 (same trick as the
    # biaxial eigen-solve): e1 from the global axis most orthogonal to k0.
    ax = np.zeros(3)
    ax[int(np.argmin(np.abs(k0)))] = 1.0
    e1 = _unit(np.cross(k0, ax))
    e2 = np.cross(k0, e1)
    phi = 2.0 * np.pi * np.arange(n_fan, dtype=np.float64) / n_fan
    that = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2   # (n_fan,3)
    dirs = _unit(np.cos(eps_angle) * k0 + np.sin(eps_angle) * that)

    modes = biaxial_modes_for_k(dirs, frame, eps)
    K_slow = modes["n_slow"][:, None] * dirs
    K_fast = modes["n_fast"][:, None] * dirs
    s_slow, np_slow, nr_slow = biaxial_ray_from_k(K_slow, frame, eps)
    s_fast, np_fast, nr_fast = biaxial_ray_from_k(K_fast, frame, eps)

    cat = np.concatenate
    return {
        "k": cat([dirs, dirs]),
        "s_ray": cat([s_slow, s_fast]),
        "D_hat": cat([modes["D_slow"], modes["D_fast"]]),
        "n_phase": cat([np_slow, np_fast]),
        "n_ray": cat([nr_slow, nr_fast]),
        "sheet": cat([np.zeros(n_fan), np.ones(n_fan)]),
        "azimuth": cat([phi, phi]),
    }
