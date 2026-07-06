# =============================================================================
# differentials.py — Igehy-1999 ray differentials, vectorized over ray batches.
#
# A ray carries FOUR (N, 3) derivative arrays w.r.t. two abstract wavefront
# parameters x, y:
#     dPdx, dDdx   position/direction derivative along the x parameter
#     dPdy, dDdy   position/direction derivative along the y parameter
# and we track the transverse wavefront-patch area
#     dA = |dPdx_perp x dPdy_perp|
# where _perp projects out the (unit) ray-direction component. dA is the local
# ray-differential estimate of the physical area a wavelet patch subtends; the
# gather uses it to size the coherent dA and to fall back per-sample when it is
# unavailable (NaN).
#
# ---------------------------------------------------------------------------
# SIGN CONVENTIONS (must match fresnel.py exactly — read that header):
#   * D      : unit ray direction, pointing INTO the interface.
#   * n_hat  : unit surface normal pointing INTO the incident medium, i.e.
#              AGAINST the ray, so cos_i = -D . n_hat >= 0  and  D . n_hat <= 0.
#   * eta    : n1 / n2 (REAL parts) — the geometric refraction ratio.
#   * S      : the (N, 3, 3) shape operator dn_hat/dp, the per-point spatial
#              derivative of the SAME n_hat that is passed in. surfaces.py's
#              `normal_derivative` returns d(CANONICAL normal)/dp; the caller
#              MUST sign-correct it to the tracer's n_hat before calling
#              reflect()/refract() here (see below).
#
# ---------------------------------------------------------------------------
# HOW THE CALLER SIGN-CORRECTS surface.normal_derivative (IMPORTANT):
#   surface.normal(p)            -> n_can (canonical: sphere outward, plane
#                                   stored n, asphere +axis, ... )
#   surface.normal_derivative(p) -> dn_can/dp  (canonical shape operator)
#   The tracer's working normal is  n_hat = s * n_can  with the scalar/per-ray
#   flip  s = -sign(D . n_can)  chosen so n_hat opposes the ray (equivalently
#   s = face.outward_sign when the ray strikes the solid from outside, and the
#   opposite when it strikes an internal face from the far side). Because the
#   flip is a constant ±1 over the local patch,
#       S = s[:, None, None] * surface.normal_derivative(p)
#   is exactly dn_hat/dp. Pass THAT S. For init_curved the emitted direction is
#   sign * n_can, so pass the RAW canonical normal_derivative there and let the
#   `sign` argument carry the flip.
#   (reflect() is even in n_hat, so its result is insensitive to the flip;
#   refract() is NOT — get the flip right for transmission.)
#
# ---------------------------------------------------------------------------
# DERIVED DIRECTION-DERIVATIVE FORMULAS (exact, verified by finite differences
# in tests/test_ray_differentials.py):
#
#   Let N = n_hat, dN = S @ dP_hit  (differential of the normal field along the
#   surface, evaluated at the differential hit-point displacement dP_hit).
#
#   Reflection   D' = D - 2 (D.N) N :
#       dD' = dD - 2 [ (dD.N + D.dN) N + (D.N) dN ]
#
#   Refraction   D' = eta D - mu N ,  mu = eta (D.N) - (D'.N)
#                (matches fresnel.refract_dir: D' = eta D + (eta ci - ct) N with
#                 ci = -D.N, ct = sqrt(1 - eta^2(1 - (D.N)^2)), so D'.N = -ct):
#       dmu = [ eta - eta^2 (D.N) / (D'.N) ] (dD.N + D.dN)
#       dD' = eta dD - dmu N - mu dN
#   Grazing/TIR (|D'.N| ~ 0) -> dD' rows are NaN (the tracer drops the
#   differential there and the gather falls back per-sample).
# =============================================================================
import numpy as np

_EPS = 1e-12


def _apply_S(S, dP):
    """(N,3,3) shape operator applied to (N,3) displacement -> (N,3)."""
    return np.einsum("nij,nj->ni", S, dP)


def _hcol(h):
    """Broadcast a scalar or (N,) source-radius into a column for (N,3) mul."""
    h = np.asarray(h, dtype=np.float64)
    return h if h.ndim == 0 else h[:, None]


# ---------------------------------------------------------------------------
# Source initialization
# ---------------------------------------------------------------------------
def init_flat(dirs, e1, e2, h):
    """Collimated (flat) emitting face.

    The wavefront is planar, so the direction derivatives vanish and the
    position derivatives span the emit-face tangent basis scaled by the
    footprint radius h.

    dirs : (N,3) unit ray directions (unused geometrically here; kept for a
           uniform source API).
    e1,e2: (N,3) orthonormal in-face tangent vectors (the x,y parameter axes).
    h    : scalar or (N,) footprint scale per parameter step.
    returns (dPdx, dDdx, dPdy, dDdy), each (N,3) float64.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    hc = _hcol(h)
    dPdx = hc * e1
    dPdy = hc * e2
    z = np.zeros_like(dPdx)
    return dPdx, z.copy(), dPdy, z.copy()


def init_curved(dirs, e1, e2, h, S, sign):
    """Curved emitting face with canonical shape operator S.

    The emitted direction equals sign * n_can, so its spatial derivative is
    sign * (dn_can/dp) = sign * S applied to the position derivative.

    dirs : (N,3) unit ray directions = sign[:,None] * canonical_normal.
    e1,e2: (N,3) in-face tangent basis.
    h    : scalar or (N,) footprint scale.
    S    : (N,3,3) RAW canonical normal_derivative(p) (NOT sign-corrected;
           the `sign` argument applies the flip).
    sign : (N,) per-ray +/-1 with dirs = sign * canonical_normal.
    returns (dPdx, dDdx, dPdy, dDdy), each (N,3) float64.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    sgn = np.asarray(sign, dtype=np.float64)[:, None]
    hc = _hcol(h)
    dPdx = hc * e1
    dPdy = hc * e2
    dDdx = sgn * _apply_S(S, dPdx)
    dDdy = sgn * _apply_S(S, dPdy)
    return dPdx, dDdx, dPdy, dDdy


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def transfer(dP, dD, D, t):
    """Free-space propagation of the POSITION derivative by distance t.

    The direction derivative is unchanged in free space; only dP evolves as
        dP <- dP + t dD.
    D is accepted for API symmetry (unused). t : scalar or (N,).
    """
    dP = np.asarray(dP, dtype=np.float64)
    dD = np.asarray(dD, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    tc = t if t.ndim == 0 else t[:, None]
    return dP + tc * dD


def transfer_to_surface(dP, dD, D, t, n_hat):
    """Position derivative propagated to the ACTUAL surface hit-point.

    Beyond plain free transfer this adds the dt term that keeps the offset ray
    landing ON the surface (so the shape operator applies at reflect/refract):

        dt      = -[(dP + t dD) . n_hat] / (D . n_hat)
        dP_hit  = dP + t dD + dt D

    Where |D . n_hat| < 1e-12 (ray grazing the tangent plane) the dt term is
    dropped and plain transfer is returned. t : scalar or (N,).
    """
    dP = np.asarray(dP, dtype=np.float64)
    dD = np.asarray(dD, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    tc = t if t.ndim == 0 else t[:, None]

    base = dP + tc * dD
    Dn = np.sum(D * n_hat, axis=-1)
    num = np.sum(base * n_hat, axis=-1)
    safe = np.abs(Dn) >= _EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(safe, -num / np.where(safe, Dn, 1.0), 0.0)
    return base + dt[:, None] * D


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------
def reflect(dP_hit, dD, D, n_hat, S):
    """Differential of specular reflection D' = D - 2(D.N)N.

    dP_hit : (N,3) position derivative AT the surface (from
             transfer_to_surface); returned unchanged (reflection does not move
             the hit point differential).
    dD     : (N,3) incoming direction derivative.
    D      : (N,3) incoming unit direction.
    n_hat  : (N,3) surface normal into the incident medium.
    S      : (N,3,3) SIGN-CORRECTED shape operator dn_hat/dp (see module head).
    returns (dP_hit, dD_reflected).
    """
    dP_hit = np.asarray(dP_hit, dtype=np.float64)
    dD = np.asarray(dD, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    N = np.asarray(n_hat, dtype=np.float64)
    dN = _apply_S(np.asarray(S, dtype=np.float64), dP_hit)

    DN = np.sum(D * N, axis=-1)                       # (N,)
    dDN = np.sum(dD * N, axis=-1) + np.sum(D * dN, axis=-1)   # d(D.N)
    dD_ref = dD - 2.0 * (dDN[:, None] * N + DN[:, None] * dN)
    return dP_hit, dD_ref


def refract(dP_hit, dD, D, n_hat, S, eta, D_t):
    """Differential of Snell refraction, matching fresnel.refract_dir exactly.

    D' = eta D - mu N,  mu = eta (D.N) - (D'.N),  with D_t the already-computed
    refracted unit direction (its projection D_t.N supplies -ct directly and
    avoids recomputing the branch).

        dmu = [eta - eta^2 (D.N)/(D_t.N)] (dD.N + D.dN)
        dD' = eta dD - dmu N - mu dN

    Grazing/TIR rows (|D_t.N| < 1e-12) return dD' = NaN — the tracer drops the
    differential there.

    eta : scalar or (N,) real n1/n2.  D_t : (N,3) refracted unit direction.
    S   : (N,3,3) SIGN-CORRECTED dn_hat/dp.
    returns (dP_hit, dD_refracted).
    """
    dP_hit = np.asarray(dP_hit, dtype=np.float64)
    dD = np.asarray(dD, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    N = np.asarray(n_hat, dtype=np.float64)
    D_t = np.asarray(D_t, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    etac = eta if eta.ndim == 0 else eta[:, None]
    eta1 = eta                                        # for (N,) scalar products
    dN = _apply_S(np.asarray(S, dtype=np.float64), dP_hit)

    DN = np.sum(D * N, axis=-1)                       # (N,)
    DtN = np.sum(D_t * N, axis=-1)                    # = -ct (N,)
    dDN = np.sum(dD * N, axis=-1) + np.sum(D * dN, axis=-1)   # d(D.N)

    good = np.abs(DtN) >= _EPS
    DtN_safe = np.where(good, DtN, 1.0)
    mu = eta1 * DN - DtN                              # (N,)
    dmu = (eta1 - eta1 ** 2 * DN / DtN_safe) * dDN    # (N,)

    dD_ref = etac * dD - dmu[:, None] * N - mu[:, None] * dN
    dD_ref = np.where(good[:, None], dD_ref, np.nan)
    return dP_hit, dD_ref


# ---------------------------------------------------------------------------
# Wavefront patch area
# ---------------------------------------------------------------------------
def patch_area(dPdx, dPdy, D):
    """Transverse wavefront-patch area dA (N,).

    Projects both position derivatives onto the plane perpendicular to the unit
    ray direction D and returns |dPdx_perp x dPdy_perp|. NaN in -> NaN out (the
    gather falls back per-sample on those rays).
    """
    dPdx = np.asarray(dPdx, dtype=np.float64)
    dPdy = np.asarray(dPdy, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    Dhat = D / np.linalg.norm(D, axis=-1, keepdims=True)
    px = dPdx - np.sum(dPdx * Dhat, axis=-1, keepdims=True) * Dhat
    py = dPdy - np.sum(dPdy * Dhat, axis=-1, keepdims=True) * Dhat
    return np.linalg.norm(np.cross(px, py), axis=-1)
