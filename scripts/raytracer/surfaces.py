# =============================================================================
# surfaces.py — analytic surface primitives: exact ray intersection, canonical
# normals, canonical UV parameterization, and trimmed-face containment.
#
# Why analytic: coherent interference needs optical-path accuracy << lambda/10
# (~50 nm). Tessellated surfaces inject micron-scale sag error (~10 rad of
# phase noise) and quantized normals; ray-quadric intersection in float64 is
# exact to ~nm over 0.1 m paths. Meshes are a fallback for genuinely
# non-analytic faces only (handled in geometry.py, not here).
#
# Conventions:
#   * All geometry in SI metres, float64.
#   * intersect(o, d) returns (t, valid): t shape (N, K) candidate ray
#     parameters (K = max root count of the primitive), valid marks real,
#     finite roots. Positivity/epsilon/trim filtering happens in
#     AnalyticFace.intersect.
#   * normal(p) returns the CANONICAL geometric normal (sphere: outward from
#     center; cylinder/cone/torus: outward from axis/tube; plane: stored n).
#     The face flips it by orientation_outward to get the out-of-solid normal.
#   * to_uv(p) maps surface points to a canonical 2D parameterization used
#     ONLY for containment tests (trim wires are mapped through the same
#     function, so any consistent parameterization works). Azimuthal
#     coordinates are radians in (-pi, pi]; containment handles the seam by
#     testing u, u±2pi.
# =============================================================================
import math
from functools import lru_cache

import numpy as np

TWO_PI = 2.0 * np.pi


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _plane_frame(normal):
    """Deterministic orthonormal in-plane frame (t1, t2) for a unit normal."""
    n = _unit(normal)
    a = np.zeros(3)
    a[int(np.argmin(np.abs(n)))] = 1.0
    t1 = np.cross(a, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


class Plane:
    K = 1

    def __init__(self, origin, normal):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.n = _unit(normal)
        self.t1, self.t2 = _plane_frame(self.n)

    def intersect(self, o, d):
        dn = d @ self.n
        num = (self.origin - o) @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / dn
        valid = np.abs(dn) > 1e-15
        return t[:, None], valid[:, None]

    def normal(self, p):
        return np.broadcast_to(self.n, p.shape).copy()

    def to_uv(self, p):
        rel = p - self.origin
        return np.stack([rel @ self.t1, rel @ self.t2], axis=-1)

    def normal_derivative(self, p):
        # Flat: the canonical normal field is constant, dn/dp == 0.
        p = np.asarray(p, dtype=np.float64)
        return np.zeros((p.shape[0], 3, 3), dtype=np.float64)

    periodic_u = False


class Sphere:
    K = 2

    def __init__(self, center, radius):
        self.c = np.asarray(center, dtype=np.float64)
        self.r = float(radius)
        # deterministic pole axis for UV; poles are containment-fragile only
        # if the trim wire passes exactly through them (rare; tests cover caps)
        self.axis = np.array([0.0, 0.0, 1.0])
        self.t1, self.t2 = _plane_frame(self.axis)

    def intersect(self, o, d):
        oc = o - self.c
        b = np.sum(oc * d, axis=-1)
        c = np.sum(oc * oc, axis=-1) - self.r ** 2
        disc = b * b - c
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t = np.stack([-b - sq, -b + sq], axis=-1)
        valid = np.stack([ok, ok], axis=-1)
        return t, valid

    def normal(self, p):
        n = p - self.c
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.c
        x = rel @ self.t1
        y = rel @ self.t2
        z = rel @ self.axis
        u = np.arctan2(y, x)                      # azimuth (-pi, pi]
        v = np.arctan2(z, np.hypot(x, y))         # latitude (-pi/2, pi/2)
        return np.stack([u, v], axis=-1)

    def uv_to_xyz(self, u, v):
        """Inverse of to_uv: (azimuth u, latitude v) -> world points on the
        sphere. Vectorized; u, v broadcast to a common (N,) shape. The exact
        inverse of to_uv on the valid range (used by CurvedDetectorGrid to
        place pixel centers and by area weighting)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        return (self.c
                + self.r * (cv * cu)[:, None] * self.t1
                + self.r * (cv * su)[:, None] * self.t2
                + self.r * sv[:, None] * self.axis)

    def normal_derivative(self, p):
        # nhat = (p-c)/r; implicit F = |p-c|^2 - r^2 gives H = 2I, |grad F| =
        # 2r, so dnhat/dp = (I - nhat nhat^T)/r. Positive radius, canonical
        # outward normal -> positive shape operator (convex).
        n = self.normal(p)
        I3 = np.eye(3)
        nn = n[:, :, None] * n[:, None, :]
        return (I3[None] - nn) / self.r

    periodic_u = True


class Cylinder:
    K = 2

    def __init__(self, origin, axis, radius):
        self.o = np.asarray(origin, dtype=np.float64)
        self.a = _unit(axis)
        self.r = float(radius)
        self.t1, self.t2 = _plane_frame(self.a)

    def intersect(self, o, d):
        oc = o - self.o
        d_perp = d - np.outer(d @ self.a, self.a)
        oc_perp = oc - np.outer(oc @ self.a, self.a)
        A = np.sum(d_perp * d_perp, axis=-1)
        B = np.sum(d_perp * oc_perp, axis=-1)
        C = np.sum(oc_perp * oc_perp, axis=-1) - self.r ** 2
        disc = B * B - A * C
        ok = (disc >= 0.0) & (A > 1e-30)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        Asafe = np.where(A > 1e-30, A, 1.0)
        t = np.stack([(-B - sq) / Asafe, (-B + sq) / Asafe], axis=-1)
        valid = np.stack([ok, ok], axis=-1)
        return t, valid

    def normal(self, p):
        rel = p - self.o
        n = rel - np.outer(rel @ self.a, self.a)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.o
        x = rel @ self.t1
        y = rel @ self.t2
        u = np.arctan2(y, x)
        v = rel @ self.a
        return np.stack([u, v], axis=-1)

    def uv_to_xyz(self, u, v):
        """Inverse of to_uv: (azimuth u, axial v [m]) -> world points on the
        cylinder. Vectorized; u, v broadcast to a common (N,) shape."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cu, su = np.cos(u), np.sin(u)
        return (self.o
                + self.r * cu[:, None] * self.t1
                + self.r * su[:, None] * self.t2
                + v[:, None] * self.a)

    def normal_derivative(self, p):
        # Radial distance surface: implicit F = |p_perp|^2 - r^2 with
        # p_perp = rel - (rel.a)a gives H = 2(I - a a^T), |grad F| = 2r.
        # Using nhat . a = 0: dnhat/dp = (I - a a^T - nhat nhat^T)/r.
        n = self.normal(p)
        I3 = np.eye(3)
        aa = np.outer(self.a, self.a)
        nn = n[:, :, None] * n[:, None, :]
        return (I3[None] - aa[None] - nn) / self.r

    periodic_u = True


class Cone:
    """Infinite cone: apex, unit axis (opening direction), half-angle."""
    K = 2

    def __init__(self, apex, axis, half_angle):
        self.apex = np.asarray(apex, dtype=np.float64)
        self.a = _unit(axis)
        self.ha = float(half_angle)
        self.cos2 = np.cos(self.ha) ** 2
        self.t1, self.t2 = _plane_frame(self.a)

    def intersect(self, o, d):
        co = o - self.apex
        dv = d @ self.a
        cv = co @ self.a
        A = dv * dv - self.cos2 * np.sum(d * d, axis=-1)
        B = dv * cv - self.cos2 * np.sum(d * co, axis=-1)
        C = cv * cv - self.cos2 * np.sum(co * co, axis=-1)
        disc = B * B - A * C
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        lin = np.abs(A) < 1e-30
        Asafe = np.where(lin, 1.0, A)
        t1 = np.where(lin, -C / np.where(np.abs(B) > 1e-30, 2.0 * B, 1.0),
                      (-B - sq) / Asafe)
        t2 = np.where(lin, np.inf, (-B + sq) / Asafe)
        t = np.stack([t1, t2], axis=-1)
        valid = np.stack([ok | lin, ok & ~lin], axis=-1)
        # reject the mirror cone (points behind the apex along the axis)
        p1 = o + t[:, 0:1] * d
        p2 = o + t[:, 1:2] * d
        h1 = (p1 - self.apex) @ self.a
        h2 = (p2 - self.apex) @ self.a
        valid[:, 0] &= h1 >= 0.0
        valid[:, 1] &= h2 >= 0.0
        return t, valid

    def normal(self, p):
        rel = p - self.apex
        h = rel @ self.a
        radial = rel - np.outer(h, self.a)
        rlen = np.linalg.norm(radial, axis=-1, keepdims=True)
        rhat = np.where(rlen > 1e-300, radial / rlen, 0.0)
        n = rhat * np.cos(self.ha) - self.a * np.sin(self.ha)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.apex
        x = rel @ self.t1
        y = rel @ self.t2
        u = np.arctan2(y, x)
        v = rel @ self.a
        return np.stack([u, v], axis=-1)

    def normal_derivative(self, p):
        # Implicit F = rho*cos(ha) - h*sin(ha) (rho = dist from axis,
        # h = axial) is already unit-gradient (|grad F| = 1) with
        # grad F = nhat. Hessian H = cos(ha)*(I - a a^T - what what^T)/rho
        # (what = radial unit). dnhat/dp = (I - nhat nhat^T) H.
        p = np.asarray(p, dtype=np.float64)
        rel = p - self.apex
        h = rel @ self.a
        w = rel - h[:, None] * self.a
        rho = np.linalg.norm(w, axis=-1)
        safe = rho > 1e-300
        what = np.where(safe[:, None], w / np.where(safe[:, None], rho[:, None],
                                                    1.0), 0.0)
        n = self.normal(p)
        I3 = np.eye(3)
        aa = np.outer(self.a, self.a)
        ww = what[:, :, None] * what[:, None, :]
        nn = n[:, :, None] * n[:, None, :]
        rho_s = np.where(safe, rho, 1.0)
        H = (np.cos(self.ha) * (I3[None] - aa[None] - ww)
             / rho_s[:, None, None])
        J = (I3[None] - nn) @ H
        # apex is a curvature singularity; report zero there rather than inf
        J[~safe] = 0.0
        return J

    periodic_u = True


class Torus:
    """Torus: center, unit axis, major radius R, minor radius r."""
    K = 4

    def __init__(self, center, axis, major_r, minor_r):
        self.c = np.asarray(center, dtype=np.float64)
        self.a = _unit(axis)
        self.R = float(major_r)
        self.r = float(minor_r)
        self.t1, self.t2 = _plane_frame(self.a)

    def intersect(self, o, d):
        # Work in the torus local frame (axis = z). Quartic:
        # (|p|^2 + R^2 - r^2)^2 = 4 R^2 (px^2 + py^2),  p = o' + t d'
        rel = o - self.c
        M = np.stack([self.t1, self.t2, self.a], axis=0)   # world->local
        ol = rel @ M.T
        dl = d @ M.T
        dd = np.sum(dl * dl, axis=-1)
        od = np.sum(ol * dl, axis=-1)
        oo = np.sum(ol * ol, axis=-1)
        R2 = self.R ** 2
        r2 = self.r ** 2
        k = oo - r2 - R2
        # quartic coefficients (monic after division by dd^2; dd == 1 for
        # unit directions but keep general)
        c4 = dd * dd
        c3 = 4.0 * dd * od
        c2 = 2.0 * dd * k + 4.0 * od * od + 4.0 * R2 * dl[:, 2] ** 2
        c1 = 4.0 * od * k + 8.0 * R2 * ol[:, 2] * dl[:, 2]
        c0 = k * k + 4.0 * R2 * (ol[:, 2] ** 2 - r2)
        # Derivation: with G = oo + R2 - r2 the quartic of
        # (t^2 dd + 2 t od + G)^2 = 4 R2 |p_xy(t)|^2 rewritten via
        # k = G - 2 R2 gives exactly c4..c0 above (verified against
        # np.roots in tests). Solve via batched companion eigenvalues.
        n = len(dd)
        comp = np.zeros((n, 4, 4), dtype=np.float64)
        comp[:, 1, 0] = 1.0
        comp[:, 2, 1] = 1.0
        comp[:, 3, 2] = 1.0
        c4s = np.where(np.abs(c4) > 1e-300, c4, 1.0)
        comp[:, 0, 3] = -c0 / c4s
        comp[:, 1, 3] = -c1 / c4s
        comp[:, 2, 3] = -c2 / c4s
        comp[:, 3, 3] = -c3 / c4s
        roots = np.linalg.eigvals(comp)                    # (n, 4) complex
        realish = np.abs(roots.imag) < 1e-9 * np.maximum(1.0,
                                                         np.abs(roots.real))
        t = np.where(realish, roots.real, np.inf)
        # polish real roots with two Newton steps on the quartic
        with np.errstate(invalid="ignore", over="ignore"):
            for _ in range(2):
                f = (((c4[:, None] * t + c3[:, None]) * t + c2[:, None]) * t
                     + c1[:, None]) * t + c0[:, None]
                fp = ((4.0 * c4[:, None] * t + 3.0 * c3[:, None]) * t
                      + 2.0 * c2[:, None]) * t + c1[:, None]
                step = np.where(np.isfinite(t) & (np.abs(fp) > 1e-300),
                                f / fp, 0.0)
                step = np.where(np.isfinite(step), step, 0.0)
                t = t - step
        valid = np.isfinite(t) & realish
        return t, valid

    def normal(self, p):
        rel = p - self.c
        h = rel @ self.a
        radial = rel - np.outer(h, self.a)
        rlen = np.linalg.norm(radial, axis=-1, keepdims=True)
        rhat = np.where(rlen > 1e-300, radial / rlen, 0.0)
        ring = self.c + self.R * rhat        # nearest point on the spine circle
        n = p - ring
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.c
        x = rel @ self.t1
        y = rel @ self.t2
        u = np.arctan2(y, x)                 # angle around the main axis
        h = rel @ self.a
        rho = np.hypot(x, y) - self.R
        v = np.arctan2(h, rho)               # angle around the tube
        return np.stack([u, v], axis=-1)

    def normal_derivative(self, p):
        # Implicit (local frame, axis=z): F = (s-R)^2 + z^2 - r^2,
        # s = hypot(qx, qy). grad F = 2(p - ring) = 2 r nhat, |grad F| = 2 r.
        # Local Hessian is analytic; transform to world by the orthonormal
        # frame M = [t1; t2; a] (H_world = M^T H_local M) and apply the
        # normalized-gradient rule dnhat/dp = (I - nhat nhat^T) H / |grad F|.
        p = np.asarray(p, dtype=np.float64)
        M = np.stack([self.t1, self.t2, self.a], axis=0)   # world->local rows
        q = (p - self.c) @ M.T
        u, v, w = q[:, 0], q[:, 1], q[:, 2]
        s = np.hypot(u, v)
        s = np.where(s > 1e-300, s, 1e-300)
        N = len(p)
        Hl = np.zeros((N, 3, 3), dtype=np.float64)
        Hl[:, 0, 0] = 2.0 * (u * u / s ** 2 + (s - self.R) * v * v / s ** 3)
        Hl[:, 1, 1] = 2.0 * (v * v / s ** 2 + (s - self.R) * u * u / s ** 3)
        Hl[:, 0, 1] = Hl[:, 1, 0] = 2.0 * self.R * u * v / s ** 3
        Hl[:, 2, 2] = 2.0
        Hw = np.einsum("ij,njk,kl->nil", M.T, Hl, M)
        gnorm = 2.0 * np.sqrt((s - self.R) ** 2 + w ** 2)
        n = self.normal(p)
        I3 = np.eye(3)
        nn = n[:, :, None] * n[:, None, :]
        return (I3[None] - nn) @ Hw / gnorm[:, None, None]

    periodic_u = True


class Asphere:
    """Axis-symmetric optical asphere (conic + even polynomial), a graph over
    the disc perpendicular to the axis at the vertex:

        z(r) = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))
                 + sum_i coeffs[i] * r^(4 + 2 i),      c = 1/R

    A point p is on the surface when (p - vertex).axis == z(r), with
    r = |radial part of (p - vertex)|, valid only for r <= r_max (the
    polynomial diverges past the validity disc; the face trim wires carry the
    true optical boundary, r_max is a hard cutoff). Signed R: the canonical
    normal always has a positive axial component (points into the +axis
    half-space), so orientation_outward is defined against that vector exactly
    as for the quadrics.

    to_uv is a PLANAR projection onto the axis frame (u = w.t1, v = w.t2),
    NOT (azimuth, r): a full cap's rim is a single circle around the axis, and
    the periodic band machinery in TrimPolygon cannot represent an
    axis-centred disc for a non-sphere (its v-extremes collapse to the rim
    radius). The planar projection maps the rim to an actual circle so the
    generic even-odd polygon test contains the interior and handles holes /
    non-circular trims. periodic_u is therefore False.
    """
    K = 4                       # up to this many candidate roots per ray

    def __init__(self, vertex, axis, R, k, coeffs, r_max):
        self.v = np.asarray(vertex, dtype=np.float64)
        self.a = _unit(axis)
        self.R = float(R)
        self.c_curv = 1.0 / float(R)
        self.k = float(k)
        self.coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        self.r_max = float(r_max)
        self.t1, self.t2 = _plane_frame(self.a)
        # axial extent of the cap over [0, r_max], sampled to bound the ray
        # search slab robustly even for wiggly polynomial terms
        rr = np.linspace(0.0, self.r_max, 64)
        zz, _ = self._sag(rr)
        zz = zz[np.isfinite(zz)]
        self._z_lo = float(min(0.0, zz.min())) if zz.size else 0.0
        self._z_hi = float(max(0.0, zz.max())) if zz.size else 0.0

    # ---- sag and its radial derivatives (arrays of r) -------------------
    def _sag(self, r):
        """Return (z, valid). valid marks r within the conic/validity disc."""
        beta = (1.0 + self.k) * self.c_curv ** 2
        arg = 1.0 - beta * r * r
        valid = (arg > 0.0) & (r <= self.r_max + 1e-12)
        root = np.sqrt(np.where(arg > 0.0, arg, 1.0))
        conic = self.c_curv * r * r / (1.0 + root)
        poly = np.zeros_like(r)
        for i, A in enumerate(self.coeffs):
            poly = poly + A * r ** (4 + 2 * i)
        z = np.where(valid, conic + poly, np.nan)
        return z, valid

    def _sag_p(self, r):
        """dz/dr (conic slope c r / sqrt(1-(1+k)c^2 r^2) + poly derivative)."""
        beta = (1.0 + self.k) * self.c_curv ** 2
        arg = 1.0 - beta * r * r
        root = np.sqrt(np.where(arg > 0.0, arg, np.nan))
        sp = self.c_curv * r / root
        for i, A in enumerate(self.coeffs):
            m = 4 + 2 * i
            sp = sp + A * m * r ** (m - 1)
        return sp

    def _sag_pp(self, r):
        """d2z/dr2 (conic term c/(1-(1+k)c^2 r^2)^{3/2} + poly)."""
        beta = (1.0 + self.k) * self.c_curv ** 2
        arg = 1.0 - beta * r * r
        base = np.where(arg > 0.0, arg, np.nan)
        spp = self.c_curv / base ** 1.5
        for i, A in enumerate(self.coeffs):
            m = 4 + 2 * i
            spp = spp + A * m * (m - 1) * r ** (m - 2)
        return spp

    # ---- intersection ---------------------------------------------------
    def intersect(self, o, d):
        o = np.asarray(o, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        N = len(o)
        rel0 = o - self.v
        h0 = rel0 @ self.a                       # axial offset of origin
        da = d @ self.a                          # axial ray slope
        w0 = rel0 - h0[:, None] * self.a         # radial part of origin
        dp = d - da[:, None] * self.a            # radial part of direction
        A = np.sum(dp * dp, axis=-1)             # |dp|^2
        w0dp = np.sum(w0 * dp, axis=-1)
        w0w0 = np.sum(w0 * w0, axis=-1)

        # r(t)^2 = w0w0 + 2 t w0dp + t^2 A  <= r_max^2  ->  bounding cylinder
        Cc = w0w0 - self.r_max ** 2
        t_lo = np.full(N, -np.inf)
        t_hi = np.full(N, np.inf)
        curved = A > 1e-300
        with np.errstate(invalid="ignore", divide="ignore"):
            disc = w0dp ** 2 - A * Cc
            sq = np.sqrt(np.where(disc > 0.0, disc, 0.0))
            cyl_lo = (-w0dp - sq) / np.where(curved, A, 1.0)
            cyl_hi = (-w0dp + sq) / np.where(curved, A, 1.0)
        # curved rays that never enter the disc -> empty interval
        miss_cyl = curved & (disc <= 0.0)
        t_lo = np.where(curved, cyl_lo, t_lo)
        t_hi = np.where(curved, cyl_hi, t_hi)
        # rays parallel to the axis (A~0): inside disc iff w0w0 <= r_max^2
        par = ~curved
        miss_par = par & (Cc > 0.0)

        # axial slab z in [z_lo, z_hi] (+ margin) constrains t via h(t)=h0+t da
        margin = 1e-7 + 0.01 * (self._z_hi - self._z_lo)
        z_lo = self._z_lo - margin
        z_hi = self._z_hi + margin
        axial = np.abs(da) > 1e-300
        with np.errstate(invalid="ignore", divide="ignore"):
            ta = (z_lo - h0) / np.where(axial, da, 1.0)
            tb = (z_hi - h0) / np.where(axial, da, 1.0)
        slab_lo = np.minimum(ta, tb)
        slab_hi = np.maximum(ta, tb)
        t_lo = np.where(axial, np.maximum(t_lo, slab_lo), t_lo)
        t_hi = np.where(axial, np.minimum(t_hi, slab_hi), t_hi)
        # ray perpendicular to axis and outside the slab never meets the cap
        miss_flat = (~axial) & ((h0 < z_lo) | (h0 > z_hi))

        t_start = np.maximum(t_lo, 0.0)          # forward hits only
        t_end = t_hi
        bad = miss_cyl | miss_par | miss_flat | ~(t_end > t_start)
        t_start = np.where(bad, np.nan, t_start)
        t_end = np.where(bad, np.nan, t_end)

        # cache per-ray coefficients for the f/f' closures
        def r_of(t):
            r2 = w0w0 + 2.0 * t * w0dp + t * t * A
            return np.sqrt(np.maximum(r2, 0.0))

        def f_of(t):
            r = r_of(t)
            z, ok = self._sag(r)
            return np.where(ok, (h0 + t * da) - z, np.nan)

        def fp_of(t):
            r = r_of(t)
            rs = np.where(r > 1e-300, r, 1.0)
            drdt = (w0dp + t * A) / rs
            sp = np.where(r > 1e-300, self._sag_p(r), 0.0)
            return da - sp * drdt

        # sample f over each ray's valid interval and bracket sign changes
        S = 32
        u = np.linspace(0.0, 1.0, S + 1)
        span = (t_end - t_start)
        tsamp = t_start[:, None] + span[:, None] * u[None, :]   # (N, S+1)
        r2s = (w0w0[:, None] + 2.0 * tsamp * w0dp[:, None]
               + tsamp ** 2 * A[:, None])
        rs = np.sqrt(np.maximum(r2s, 0.0))
        beta = (1.0 + self.k) * self.c_curv ** 2
        arg = 1.0 - beta * rs * rs
        oks = (arg > 0.0) & (rs <= self.r_max + 1e-12)
        root = np.sqrt(np.where(arg > 0.0, arg, 1.0))
        zs = self.c_curv * rs * rs / (1.0 + root)
        for i, Acoef in enumerate(self.coeffs):
            zs = zs + Acoef * rs ** (4 + 2 * i)
        fsamp = np.where(oks, (h0[:, None] + tsamp * da[:, None]) - zs, np.nan)

        roots = np.full((N, self.K), np.inf)
        nfound = np.zeros(N, dtype=np.intp)
        for j in range(S):
            fa = fsamp[:, j]
            fb = fsamp[:, j + 1]
            sc = (np.isfinite(fa) & np.isfinite(fb)
                  & (np.sign(fa) != np.sign(fb)) & (fa != 0.0))
            active = sc & (nfound < self.K)
            if not np.any(active):
                continue
            a = tsamp[:, j].copy()
            b = tsamp[:, j + 1].copy()
            fa_ = fa.copy()
            for _ in range(8):                    # bisection
                m = 0.5 * (a + b)
                fm = f_of(m)
                left = np.sign(fm) == np.sign(fa_)
                a = np.where(active & left, m, a)
                fa_ = np.where(active & left, fm, fa_)
                b = np.where(active & ~left, m, b)
            t = 0.5 * (a + b)
            for _ in range(12):                   # Newton, bracket-guarded
                ft = f_of(t)
                fpt = fp_of(t)
                step = np.where(np.abs(fpt) > 1e-300, ft / fpt, 0.0)
                tn = t - step
                out = (tn < a) | (tn > b) | ~np.isfinite(tn)
                tn = np.where(out, 0.5 * (a + b), tn)
                ftn = f_of(tn)
                left = np.sign(ftn) == np.sign(fa_)
                a = np.where(active & left, tn, a)
                fa_ = np.where(active & left, ftn, fa_)
                b = np.where(active & ~left, tn, b)
                t = tn
            conv = active & (np.abs(f_of(t)) < 1e-13)
            idx = np.where(conv)[0]
            if idx.size:
                roots[idx, nfound[idx]] = t[idx]
                nfound[idx] += 1
        valid = np.isfinite(roots)
        return roots, valid

    # ---- geometry -------------------------------------------------------
    def _radial(self, p):
        rel = p - self.v
        h = rel @ self.a
        w = rel - h[:, None] * self.a
        r = np.linalg.norm(w, axis=-1)
        return rel, h, w, r

    def normal(self, p):
        # F = h - z(r); grad F = a - z'(r) what  ->  canonical normal (axial
        # component always +1 before normalization, i.e. faces +axis).
        p = np.asarray(p, dtype=np.float64)
        _, _, w, r = self._radial(p)
        safe = r > 1e-300
        rs = np.where(safe, r, 1.0)
        what = np.where(safe[:, None], w / rs[:, None], 0.0)
        sp = np.where(safe, self._sag_p(r), 0.0)
        g = self.a[None] - sp[:, None] * what
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.v
        return np.stack([rel @ self.t1, rel @ self.t2], axis=-1)

    def normal_derivative(self, p):
        # Implicit F = (p-v).a - z(r). grad F = a - z'(r) what,
        # |grad F| = sqrt(1 + z'^2). Hessian
        #   H = -[ z''(r) what what^T + (z'(r)/r)(I - a a^T - what what^T) ].
        # At r->0 both z'' and z'/r -> curvature c, giving H = -c (I - a a^T),
        # independent of the (undefined) radial direction.
        p = np.asarray(p, dtype=np.float64)
        _, _, w, r = self._radial(p)
        safe = r > 1e-12
        rs = np.where(safe, r, 1.0)
        what = np.where(safe[:, None], w / rs[:, None], self.t1[None])
        sp = np.where(safe, self._sag_p(r), 0.0)
        spp = self._sag_pp(r)
        sp_over_r = np.where(safe, sp / rs, spp)      # limit c at r->0
        n = self.normal(p)
        I3 = np.eye(3)
        aa = np.outer(self.a, self.a)
        ww = what[:, :, None] * what[:, None, :]
        H = -(spp[:, None, None] * ww
              + sp_over_r[:, None, None] * (I3[None] - aa[None] - ww))
        gnorm = np.sqrt(1.0 + sp ** 2)
        nn = n[:, :, None] * n[:, None, :]
        return (I3[None] - nn) @ H / gnorm[:, None, None]

    periodic_u = False


# =============================================================================
# QForbes -- ISO 10110-12 Forbes Q-type asphere: base conic + an orthonormal
# Q-polynomial departure (Forbes 2007/2010; see docs/../engine3.md Sec 7.6).
# Two kinds, selected by `kind`:
#
#   'qbfs' -- departure from the best-fit sphere/conic, ORTHONORMAL IN SLOPE
#             (Forbes, "Robust, efficient computational methods for axially
#             symmetric optical aspheres", Opt. Express 18(19) 19700, 2010,
#             Sec 3 / Appendix A). Basis Q_n(u^2), envelope u^2(1-u^2),
#             u = r/r_max.
#   'qcon' -- departure from an arbitrary conic, AMPLITUDE-orthonormal
#             (Forbes, "Robust and fast computation of the sag of a conic +
#             normal departure", Opt. Express 18(13) 13851, 2010, Sec 5).
#             Basis = Jacobi polynomials P_n^(0,4)(2u^2-1) (~ Chebyshev-3rd-
#             kind-like family scaled by u^4 in Forbes' Sec 5 derivation),
#             envelope u^4.
#
# Both bases are built via the SAME numerically stable recurrence Forbes
# gives in Appendix A of the Qbfs paper (Qbfs) / the standard 3-term Jacobi
# recurrence (Qcon) -- deliberately never forming the raw r^4, r^6, r^8, ...
# monomial basis, which is ill-conditioned (near-collinear) over a finite
# aperture (engine3.md Sec 7.6). prysm (pinned git SHA -- see
# mieworkbench/tests/test_qforbes_prysm_oracle.py) is the reference
# implementation this was checked against to 1e-12 (sag, first radial
# derivative); this module never imports prysm (test-only oracle dependency,
# not shipped with the engine).
#
# The departure is normalized by 1/sigma = sec(theta_c(r)), the EXACT secant
# of the base conic's own local surface-normal tilt (Forbes' "cos factor",
# oe-18-19700 Sec 3): sigma_inv(r) = sqrt(1 + zc'(r)^2) where zc'(r) is the
# base conic's OWN first derivative. This is an algebraic identity (not an
# approximation) -- expand the base conic's unit normal
#   n_hat = (-zc'(r) rhat, 1) / sqrt(1+zc'(r)^2)
# and sigma_inv is exactly the reciprocal of n_hat's axial component, i.e.
# the secant of the local surface tilt. It composes with the departure via
# a plain product rule; the SECOND derivative additionally needs the base
# conic's THIRD derivative zc'''(r) (this class's own extension beyond what
# prysm's public raytracing API computes -- prysm stops at z'). All of the
# sigma_inv/zc'''/departure-2nd-derivative algebra is validated against
# finite differences in scripts/raytracer/tests/test_qforbes.py (never
# against prysm, which does not expose it).
# =============================================================================
_QBFS_INV_SQRT19 = 1.0 / math.sqrt(19.0)


@lru_cache(maxsize=None)
def _f_qbfs(n):
    """Forbes (2010) oe-18-19700 Eq. (A.16): f(m)."""
    if n == 0:
        return 2.0
    if n == 1:
        return math.sqrt(19.0) / 2.0
    term1 = n * (n + 1) + 3
    term2 = _g_qbfs(n - 1) ** 2
    term3 = _h_qbfs(n - 2) ** 2
    return math.sqrt(term1 - term2 - term3)


@lru_cache(maxsize=None)
def _g_qbfs(n_minus_1):
    """Forbes (2010) oe-18-19700 Eq. (A.15): g(m-1)."""
    if n_minus_1 == 0:
        return -0.5
    n_minus_2 = n_minus_1 - 1
    return -(1.0 + _g_qbfs(n_minus_2) * _h_qbfs(n_minus_2)) \
        / _f_qbfs(n_minus_1)


@lru_cache(maxsize=None)
def _h_qbfs(n_minus_2):
    """Forbes (2010) oe-18-19700 Eq. (A.14): h(m-2)."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2.0 * _f_qbfs(n_minus_2))


def _qbfs_weighted_sum(coeffs, x):
    """Sum_n coeffs[n] * Q_n(x) and its 1st/2nd derivatives wrt x = u^2,
    via the Forbes Appendix-A auxiliary recurrence (the UNENVELOPED Q_n --
    i.e. Qbfs_n(u) = u^2(1-u^2) * Q_n(u^2)). Q_n itself comes from a shifted
    Chebyshev-3rd-kind auxiliary polynomial P_n (P0=2, P1=6-4x*2,
    Pn=(2-4x)P_{n-1}-P_{n-2}) converted via f/g/h so that Qbfs_n are
    orthonormal-in-slope on [0,1]. coeffs may be shorter than any degree
    used elsewhere; empty -> identically zero. Returns (R, dR/dx, d2R/dx2),
    each broadcast to x's shape."""
    x = np.asarray(x, dtype=np.float64)
    M = len(coeffs) - 1
    if M < 0:
        z = np.zeros_like(x)
        return z, z.copy(), z.copy()
    R = np.full_like(x, coeffs[0])          # Q0 = 1, dQ0 = d2Q0 = 0
    dR = np.zeros_like(x)
    d2R = np.zeros_like(x)
    if M == 0:
        return R, dR, d2R
    Q1 = _QBFS_INV_SQRT19 * (13.0 - 16.0 * x)
    dQ1 = np.full_like(x, -16.0 * _QBFS_INV_SQRT19)
    R = R + coeffs[1] * Q1
    dR = dR + coeffs[1] * dQ1
    # d2Q1 = 0 (Q1 is linear in x)
    if M == 1:
        return R, dR, d2R

    P_prev = np.full_like(x, 2.0)
    P_curr = 6.0 - 8.0 * x
    dP_prev = np.zeros_like(x)
    dP_curr = np.full_like(x, -8.0)
    d2P_prev = np.zeros_like(x)
    d2P_curr = np.zeros_like(x)
    Q_prev, Q_curr = np.ones_like(x), Q1
    dQ_prev, dQ_curr = np.zeros_like(x), dQ1
    d2Q_prev, d2Q_curr = np.zeros_like(x), np.zeros_like(x)
    lin = 2.0 - 4.0 * x
    for n in range(2, M + 1):
        Pn = lin * P_curr - P_prev
        dPn = -4.0 * P_curr + lin * dP_curr - dP_prev
        d2Pn = -8.0 * dP_curr + lin * d2P_curr - d2P_prev
        g = _g_qbfs(n - 1)
        h = _h_qbfs(n - 2)
        inv_f = 1.0 / _f_qbfs(n)
        Qn = (Pn - g * Q_curr - h * Q_prev) * inv_f
        dQn = (dPn - g * dQ_curr - h * dQ_prev) * inv_f
        d2Qn = (d2Pn - g * d2Q_curr - h * d2Q_prev) * inv_f
        c = coeffs[n]
        if c:
            R = R + c * Qn
            dR = dR + c * dQn
            d2R = d2R + c * d2Qn
        P_prev, P_curr = P_curr, Pn
        dP_prev, dP_curr = dP_curr, dPn
        d2P_prev, d2P_curr = d2P_curr, d2Pn
        Q_prev, Q_curr = Q_curr, Qn
        dQ_prev, dQ_curr = dQ_curr, dQn
        d2Q_prev, d2Q_curr = d2Q_curr, d2Qn
    return R, dR, d2R


@lru_cache(maxsize=None)
def _jacobi04_abc(n):
    """Standard 3-term Jacobi recurrence coefficients (A&S / DLMF 18.9),
    alpha=0, beta=4: P_n = (A x + B) P_{n-1} - C P_{n-2}, called with
    n-1 to produce P_n (matches the convention used below)."""
    a, b = 0.0, 4.0
    s = a + b
    Anum = (2 * n + s + 1) * (2 * n + s + 2)
    Aden = 2.0 * (n + 1) * (n + s + 1)
    A = Anum / Aden
    Bnum = (a * a - b * b) * (2 * n + s + 1)
    Bden = 2.0 * (n + 1) * (n + s + 1) * (2 * n + s)
    B = Bnum / Bden
    Cnum = (n + a) * (n + b) * (2 * n + s + 2)
    Cden = (n + 1) * (n + s + 1) * (2 * n + s)
    C = Cnum / Cden
    return A, B, C


def _qcon_weighted_sum(coeffs, x):
    """Sum_n coeffs[n] * P_n^(0,4)(x) and its 1st/2nd derivatives wrt
    x = 2u^2-1, via the standard 3-term Jacobi recurrence (Forbes 2010
    oe-18-13-13851 Sec 5's "Qcon" amplitude-orthonormal basis). coeffs:
    same convention as _qbfs_weighted_sum. Returns (R, dR/dx, d2R/dx2)."""
    x = np.asarray(x, dtype=np.float64)
    M = len(coeffs) - 1
    if M < 0:
        z = np.zeros_like(x)
        return z, z.copy(), z.copy()
    R = np.full_like(x, coeffs[0])          # P0 = 1, dP0 = d2P0 = 0
    dR = np.zeros_like(x)
    d2R = np.zeros_like(x)
    if M == 0:
        return R, dR, d2R
    # alpha=0, beta=4: P1 = (alpha+1) + (alpha+beta+2)(x-1)/2 = 3x - 2
    P1 = 3.0 * x - 2.0
    dP1 = np.full_like(x, 3.0)
    R = R + coeffs[1] * P1
    dR = dR + coeffs[1] * dP1
    # d2P1 = 0 (P1 is linear in x)
    if M == 1:
        return R, dR, d2R

    P_prev, P_curr = np.ones_like(x), P1
    dP_prev, dP_curr = np.zeros_like(x), dP1
    d2P_prev, d2P_curr = np.zeros_like(x), np.zeros_like(x)
    for n in range(2, M + 1):
        A, B, C = _jacobi04_abc(n - 1)
        lin = A * x + B
        Pn = lin * P_curr - C * P_prev
        dPn = A * P_curr + lin * dP_curr - C * dP_prev
        d2Pn = 2.0 * A * dP_curr + lin * d2P_curr - C * d2P_prev
        c = coeffs[n]
        if c:
            R = R + c * Pn
            dR = dR + c * dPn
            d2R = d2R + c * d2Pn
        P_prev, P_curr = P_curr, Pn
        dP_prev, dP_curr = dP_curr, dPn
        d2P_prev, d2P_curr = d2P_curr, d2Pn
    return R, dR, d2R


def _qbfs_departure(coeffs, u):
    """z_dep(u), dz_dep/du, d2z_dep/du2 for the Qbfs envelope
    u^2(1-u^2)."""
    x = u * u
    R, dRdx, d2Rdx2 = _qbfs_weighted_sum(coeffs, x)
    dSdu = 2.0 * u * dRdx
    d2Sdu2 = 2.0 * dRdx + 4.0 * u * u * d2Rdx2
    E = x * (1.0 - x)
    dEdu = 2.0 * u - 4.0 * u ** 3
    d2Edu2 = 2.0 - 12.0 * u * u
    z = E * R
    dz = dEdu * R + E * dSdu
    d2z = d2Edu2 * R + 2.0 * dEdu * dSdu + E * d2Sdu2
    return z, dz, d2z


def _qcon_departure(coeffs, u):
    """z_dep(u), dz_dep/du, d2z_dep/du2 for the Qcon envelope u^4."""
    x = 2.0 * u * u - 1.0
    R, dRdx, d2Rdx2 = _qcon_weighted_sum(coeffs, x)
    dSdu = 4.0 * u * dRdx
    d2Sdu2 = 4.0 * dRdx + 16.0 * u * u * d2Rdx2
    E = u ** 4
    dEdu = 4.0 * u ** 3
    d2Edu2 = 12.0 * u * u
    z = E * R
    dz = dEdu * R + E * dSdu
    d2z = d2Edu2 * R + 2.0 * dEdu * dSdu + E * d2Sdu2
    return z, dz, d2z


class QForbes:
    """Axis-symmetric Forbes Q-type asphere (kind='qbfs' or 'qcon'; see the
    module-level comment above for the math and the oracle test). Same
    graph-over-a-disc contract as Asphere: z(r) valid for r <= r_max, a
    point p is on the surface when (p-vertex).axis == z(r). to_uv is the
    same planar (u,v) = (w.t1, w.t2) projection as Asphere, for the same
    reason (a full cap's rim must map to an actual circle for the
    even-odd trim-polygon test); periodic_u is False.

    coeffs[n] is the amplitude of the n-th orthonormal Q term, in the
    SAME length units as the sag itself (metres) -- unlike Asphere's
    r^(4+2i) power-series coefficients, no per-order unit scaling is
    needed (the Q_n/P_n bases are already dimensionless functions of
    u = r/r_max).
    """
    K = 4                       # up to this many candidate roots per ray

    def __init__(self, vertex, axis, R, k, coeffs, r_max, kind="qbfs"):
        if kind not in ("qbfs", "qcon"):
            raise ValueError(
                "QForbes kind must be 'qbfs' or 'qcon', got %r" % (kind,))
        self.kind = kind
        self.v = np.asarray(vertex, dtype=np.float64)
        self.a = _unit(axis)
        self.R = float(R)
        self.c_curv = 1.0 / float(R)
        self.k = float(k)
        self.coeffs = [float(c) for c in coeffs]
        self.r_max = float(r_max)
        self.t1, self.t2 = _plane_frame(self.a)
        rr = np.linspace(0.0, self.r_max, 64)
        zz, _ = self._sag(rr)
        zz = zz[np.isfinite(zz)]
        self._z_lo = float(min(0.0, zz.min())) if zz.size else 0.0
        self._z_hi = float(max(0.0, zz.max())) if zz.size else 0.0

    # ---- base conic: sag + 1st/2nd/3rd radial derivative -----------------
    def _conic(self, r):
        """(zc, zc', zc'', zc''', valid) of the base conic ONLY (no
        departure). zc''' feeds sigma_inv'' (see _sag_pp); Asphere doesn't
        need a 3rd derivative since its poly departure has no cos-factor
        normalization to differentiate through."""
        beta = (1.0 + self.k) * self.c_curv ** 2
        arg = 1.0 - beta * r * r
        valid = arg > 0.0
        phi = np.sqrt(np.where(valid, arg, 1.0))
        c = self.c_curv
        zc = c * r * r / (1.0 + phi)
        zc1 = c * r / phi
        zc2 = c / phi ** 3
        zc3 = 3.0 * c * beta * r / phi ** 5
        return zc, zc1, zc2, zc3, valid

    def _departure(self, u):
        fn = _qbfs_departure if self.kind == "qbfs" else _qcon_departure
        return fn(self.coeffs, u)

    # ---- sag and its radial derivatives (arrays of r) --------------------
    def _sag(self, r):
        """Return (z, valid); valid marks r within the conic disc AND
        r <= r_max (same hard-cutoff contract as Asphere._sag)."""
        r = np.asarray(r, dtype=np.float64)
        zc, zc1, _, _, valid = self._conic(r)
        valid = valid & (r <= self.r_max + 1e-12)
        zdep, _, _ = self._departure(r / self.r_max)
        sigma_inv = np.sqrt(1.0 + zc1 * zc1)
        z = np.where(valid, zc + sigma_inv * zdep, np.nan)
        return z, valid

    def _sag_p(self, r):
        """dz/dr = zc' + d(sigma_inv * z_dep)/dr (product rule); sigma_inv
        = sqrt(1+zc'^2) is EXACTLY sec(theta_c(r)) -- see the module
        docstring -- so sigma_inv' = zc' zc'' / sigma_inv."""
        r = np.asarray(r, dtype=np.float64)
        zc, zc1, zc2, _, _ = self._conic(r)
        zdep, dzdep_du, _ = self._departure(r / self.r_max)
        sigma_inv = np.sqrt(1.0 + zc1 * zc1)
        sigma_inv1 = zc1 * zc2 / sigma_inv
        dzdep_dr = dzdep_du / self.r_max
        return zc1 + sigma_inv1 * zdep + sigma_inv * dzdep_dr

    def _sag_pp(self, r):
        """d2z/dr2. sigma_inv'' comes from differentiating sigma_inv' =
        zc'zc''/sigma_inv again: writing g=zc', sigma_inv=sqrt(1+g^2)
        satisfies sigma_inv*sigma_inv'=g*g' (differentiate sigma_inv^2 =
        1+g^2), and differentiating THAT gives
        sigma_inv'^2 + sigma_inv*sigma_inv'' = g'^2 + g*g''
        => sigma_inv'' = (g'^2 + g*g'' - sigma_inv'^2) / sigma_inv,
        needing g'' = zc''' (the base conic's third derivative, _conic's
        4th return value)."""
        r = np.asarray(r, dtype=np.float64)
        zc, zc1, zc2, zc3, _ = self._conic(r)
        zdep, dzdep_du, d2zdep_du2 = self._departure(r / self.r_max)
        sigma_inv = np.sqrt(1.0 + zc1 * zc1)
        sigma_inv1 = zc1 * zc2 / sigma_inv
        sigma_inv2 = (zc2 * zc2 + zc1 * zc3 - sigma_inv1 * sigma_inv1) \
            / sigma_inv
        dzdep_dr = dzdep_du / self.r_max
        d2zdep_dr2 = d2zdep_du2 / (self.r_max * self.r_max)
        return (zc2 + sigma_inv2 * zdep + 2.0 * sigma_inv1 * dzdep_dr
                + sigma_inv * d2zdep_dr2)

    # ---- intersection (identical slab-bracketed Newton pattern to
    # Asphere.intersect, calling self._sag/_sag_p instead of duplicating
    # the sag formula inline in the sample scan) ---------------------------
    def intersect(self, o, d):
        o = np.asarray(o, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        N = len(o)
        rel0 = o - self.v
        h0 = rel0 @ self.a
        da = d @ self.a
        w0 = rel0 - h0[:, None] * self.a
        dp = d - da[:, None] * self.a
        A = np.sum(dp * dp, axis=-1)
        w0dp = np.sum(w0 * dp, axis=-1)
        w0w0 = np.sum(w0 * w0, axis=-1)

        Cc = w0w0 - self.r_max ** 2
        t_lo = np.full(N, -np.inf)
        t_hi = np.full(N, np.inf)
        curved = A > 1e-300
        with np.errstate(invalid="ignore", divide="ignore"):
            disc = w0dp ** 2 - A * Cc
            sq = np.sqrt(np.where(disc > 0.0, disc, 0.0))
            cyl_lo = (-w0dp - sq) / np.where(curved, A, 1.0)
            cyl_hi = (-w0dp + sq) / np.where(curved, A, 1.0)
        miss_cyl = curved & (disc <= 0.0)
        t_lo = np.where(curved, cyl_lo, t_lo)
        t_hi = np.where(curved, cyl_hi, t_hi)
        par = ~curved
        miss_par = par & (Cc > 0.0)

        margin = 1e-7 + 0.01 * (self._z_hi - self._z_lo)
        z_lo = self._z_lo - margin
        z_hi = self._z_hi + margin
        axial = np.abs(da) > 1e-300
        with np.errstate(invalid="ignore", divide="ignore"):
            ta = (z_lo - h0) / np.where(axial, da, 1.0)
            tb = (z_hi - h0) / np.where(axial, da, 1.0)
        slab_lo = np.minimum(ta, tb)
        slab_hi = np.maximum(ta, tb)
        t_lo = np.where(axial, np.maximum(t_lo, slab_lo), t_lo)
        t_hi = np.where(axial, np.minimum(t_hi, slab_hi), t_hi)
        miss_flat = (~axial) & ((h0 < z_lo) | (h0 > z_hi))

        t_start = np.maximum(t_lo, 0.0)
        t_end = t_hi
        bad = miss_cyl | miss_par | miss_flat | ~(t_end > t_start)
        t_start = np.where(bad, np.nan, t_start)
        t_end = np.where(bad, np.nan, t_end)

        def r_of(t):
            r2 = w0w0 + 2.0 * t * w0dp + t * t * A
            return np.sqrt(np.maximum(r2, 0.0))

        def f_of(t):
            r = r_of(t)
            z, ok = self._sag(r)
            return np.where(ok, (h0 + t * da) - z, np.nan)

        def fp_of(t):
            r = r_of(t)
            rs = np.where(r > 1e-300, r, 1.0)
            drdt = (w0dp + t * A) / rs
            sp = np.where(r > 1e-300, self._sag_p(r), 0.0)
            return da - sp * drdt

        S = 32
        u = np.linspace(0.0, 1.0, S + 1)
        span = (t_end - t_start)
        tsamp = t_start[:, None] + span[:, None] * u[None, :]   # (N, S+1)
        # r_of/f_of above are written for 1D t (bisection/Newton, below);
        # the 2D sample scan needs its own explicitly-broadcast r^2 (same
        # shape trick Asphere.intersect uses), then reuses self._sag.
        r2s = (w0w0[:, None] + 2.0 * tsamp * w0dp[:, None]
              + tsamp ** 2 * A[:, None])
        rs = np.sqrt(np.maximum(r2s, 0.0))
        zs, oks = self._sag(rs)
        fsamp = np.where(oks, (h0[:, None] + tsamp * da[:, None]) - zs, np.nan)

        roots = np.full((N, self.K), np.inf)
        nfound = np.zeros(N, dtype=np.intp)
        for j in range(S):
            fa = fsamp[:, j]
            fb = fsamp[:, j + 1]
            sc = (np.isfinite(fa) & np.isfinite(fb)
                  & (np.sign(fa) != np.sign(fb)) & (fa != 0.0))
            active = sc & (nfound < self.K)
            if not np.any(active):
                continue
            a = tsamp[:, j].copy()
            b = tsamp[:, j + 1].copy()
            fa_ = fa.copy()
            for _ in range(8):                    # bisection
                m = 0.5 * (a + b)
                fm = f_of(m)
                left = np.sign(fm) == np.sign(fa_)
                a = np.where(active & left, m, a)
                fa_ = np.where(active & left, fm, fa_)
                b = np.where(active & ~left, m, b)
            t = 0.5 * (a + b)
            for _ in range(12):                   # Newton, bracket-guarded
                ft = f_of(t)
                fpt = fp_of(t)
                step = np.where(np.abs(fpt) > 1e-300, ft / fpt, 0.0)
                tn = t - step
                out = (tn < a) | (tn > b) | ~np.isfinite(tn)
                tn = np.where(out, 0.5 * (a + b), tn)
                ftn = f_of(tn)
                left = np.sign(ftn) == np.sign(fa_)
                a = np.where(active & left, tn, a)
                fa_ = np.where(active & left, ftn, fa_)
                b = np.where(active & ~left, tn, b)
                t = tn
            conv = active & (np.abs(f_of(t)) < 1e-13)
            idx = np.where(conv)[0]
            if idx.size:
                roots[idx, nfound[idx]] = t[idx]
                nfound[idx] += 1
        valid = np.isfinite(roots)
        return roots, valid

    # ---- geometry (identical pattern to Asphere.normal/to_uv/
    # normal_derivative) ----------------------------------------------------
    def _radial(self, p):
        rel = p - self.v
        h = rel @ self.a
        w = rel - h[:, None] * self.a
        r = np.linalg.norm(w, axis=-1)
        return rel, h, w, r

    def normal(self, p):
        p = np.asarray(p, dtype=np.float64)
        _, _, w, r = self._radial(p)
        safe = r > 1e-300
        rs = np.where(safe, r, 1.0)
        what = np.where(safe[:, None], w / rs[:, None], 0.0)
        sp = np.where(safe, self._sag_p(r), 0.0)
        g = self.a[None] - sp[:, None] * what
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def to_uv(self, p):
        rel = p - self.v
        return np.stack([rel @ self.t1, rel @ self.t2], axis=-1)

    def normal_derivative(self, p):
        p = np.asarray(p, dtype=np.float64)
        _, _, w, r = self._radial(p)
        safe = r > 1e-12
        rs = np.where(safe, r, 1.0)
        what = np.where(safe[:, None], w / rs[:, None], self.t1[None])
        sp = np.where(safe, self._sag_p(r), 0.0)
        spp = self._sag_pp(r)
        sp_over_r = np.where(safe, sp / rs, spp)
        n = self.normal(p)
        I3 = np.eye(3)
        aa = np.outer(self.a, self.a)
        ww = what[:, :, None] * what[:, None, :]
        H = -(spp[:, None, None] * ww
              + sp_over_r[:, None, None] * (I3[None] - aa[None] - ww))
        gnorm = np.sqrt(1.0 + sp ** 2)
        nn = n[:, :, None] * n[:, None, :]
        return (I3[None] - nn) @ H / gnorm[:, None, None]

    periodic_u = False


def make_surface(spec):
    """Build a primitive from a model.json 'surface' dict."""
    t = spec["type"]
    if t == "plane":
        return Plane(spec["origin"], spec["normal"])
    if t == "sphere":
        return Sphere(spec["center"], spec["radius"])
    if t == "cylinder":
        return Cylinder(spec["origin"], spec["axis"], spec["radius"])
    if t == "cone":
        return Cone(spec["apex"], spec["axis"], spec["half_angle"])
    if t == "torus":
        return Torus(spec["center"], spec["axis"], spec["major_r"],
                     spec["minor_r"])
    if t == "asphere":
        return Asphere(spec["vertex"], spec["axis"], spec["R"], spec["k"],
                       spec["coeffs"], spec["r_max"])
    if t == "qforbes":
        return QForbes(spec["vertex"], spec["axis"], spec["R"], spec["k"],
                       spec["coeffs"], spec["r_max"], kind=spec["kind"])
    raise ValueError("not an analytic surface type: %r" % t)


# ---------------------------------------------------------------------------
# Trimmed faces
# ---------------------------------------------------------------------------
class TrimPolygon:
    """Containment test in canonical UV, seam- and pole-aware.

    Built from 3D trim polylines (outer wire first, holes after) mapped
    through the SAME surface.to_uv used for query points — any consistent
    parameterization therefore works.

    Three regimes, chosen at construction:

    1. UNTRIMMED: the face area matches the primitive's full finite area
       (complete sphere / torus — their trim wires are degenerate seams
       that map to lines in UV). contains() is True everywhere.
    2. BAND: at least one wire has nonzero winding number around the
       periodic u axis (it encircles the axis: sphere zones/polar caps,
       full-revolution cylinder/cone/torus bands). Containment is a v-range
       test. For spheres the wire v-extremes alone cannot distinguish a
       zone from a polar cap, so the band is chosen by matching the face
       area (zone / cap-to-north / cap-to-south) when area is provided.
    3. POLYGON: generic even-odd crossing test on the unwrapped-u polygon,
       with queries tested at u + k*2pi, k in {-1, 0, 1}.
    """

    def __init__(self, surface, polylines_xyz, face_area=None):
        self.surface = surface
        self.periodic = surface.periodic_u
        self.mode = "polygon"
        self.loops = []
        windings = []
        for poly in polylines_xyz:
            uv = surface.to_uv(np.asarray(poly, dtype=np.float64))
            u = uv[:, 0].copy()
            w = 0
            if self.periodic:
                du = np.diff(np.append(u, u[0]))
                du = (du + np.pi) % TWO_PI - np.pi     # principal deltas
                w = int(np.round(np.sum(du) / TWO_PI))
                u = np.concatenate([[u[0]], u[0] + np.cumsum(du[:-1])])
            windings.append(w)
            self.loops.append(np.stack([u, uv[:, 1]], axis=-1))

        # Regime 1: untrimmed closed primitive (full sphere / torus).
        full = self._full_primitive_area()
        if (face_area is not None and full is not None
                and abs(face_area - full) <= 0.01 * full):
            self.mode = "untrimmed"
            return

        # Regime 2: some wire encircles the periodic axis.
        if any(w != 0 for w in windings):
            wv = np.concatenate([lp[:, 1] for lp, w in
                                 zip(self.loops, windings) if w != 0])
            v_lo, v_hi = float(wv.min()), float(wv.max())
            self.v_band = self._choose_band(v_lo, v_hi, face_area)
            # non-winding loops become holes tested as polygons
            self.hole_loops = [lp for lp, w in zip(self.loops, windings)
                               if w == 0]
            self.mode = "band"
            return

    def _full_primitive_area(self):
        s = self.surface
        if isinstance(s, Sphere):
            return 4.0 * np.pi * s.r ** 2
        if isinstance(s, Torus):
            return 4.0 * np.pi ** 2 * s.R * s.r
        return None

    def _choose_band(self, v_lo, v_hi, face_area):
        """Pick the v-interval; for spheres disambiguate zone vs polar cap
        by area matching (area of zone v1..v2 is 2 pi R^2 (sin v2 - sin v1))."""
        s = self.surface
        if not isinstance(s, Sphere) or face_area is None:
            return (v_lo, v_hi)
        R2 = s.r ** 2
        candidates = [
            ((v_lo, v_hi),
             2 * np.pi * R2 * (np.sin(v_hi) - np.sin(v_lo))),
            ((v_lo, np.pi / 2),
             2 * np.pi * R2 * (1.0 - np.sin(v_lo))),            # north cap
            ((-np.pi / 2, v_hi),
             2 * np.pi * R2 * (np.sin(v_hi) + 1.0)),            # south cap
        ]
        best = min(candidates, key=lambda c: abs(c[1] - face_area))
        return best[0]

    @staticmethod
    def _in_loop(u, v, loop):
        """Vectorized even-odd crossing test for one closed loop."""
        lu = loop[:, 0]
        lv = loop[:, 1]
        lu2 = np.roll(lu, -1)
        lv2 = np.roll(lv, -1)
        inside = np.zeros(u.shape, dtype=bool)
        for x1, y1, x2, y2 in zip(lu, lv, lu2, lv2):
            crosses = ((y1 > v) != (y2 > v))
            if not np.any(crosses):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1 + (v - y1) / (y2 - y1) * (x2 - x1)
            inside ^= crosses & (u < xint)
        return inside

    def _in_polygon(self, u, v, loops):
        shifts = (0.0,) if not self.periodic else (0.0, TWO_PI, -TWO_PI)
        inside = np.zeros(u.shape, dtype=bool)
        for k in shifts:
            acc = np.zeros(u.shape, dtype=bool)
            for loop in loops:
                acc ^= self._in_loop(u + k, v, loop)
            inside |= acc
        return inside

    def contains(self, uv):
        u = uv[:, 0]
        v = uv[:, 1]
        if self.mode == "untrimmed":
            return np.ones(u.shape, dtype=bool)
        if self.mode == "band":
            lo, hi = self.v_band
            ok = (v >= lo - 1e-9) & (v <= hi + 1e-9)
            if self.hole_loops:
                ok &= ~self._in_polygon(u, v, self.hole_loops)
            return ok
        return self._in_polygon(u, v, self.loops)


class AnalyticFace:
    """A trimmed analytic face: primitive + trim + solid orientation."""

    def __init__(self, face_id, surface, trim_polylines_xyz,
                 orientation_outward, body_index, face_index,
                 area_m2=None):
        self.id = face_id
        self.surface = surface
        self.trim = TrimPolygon(surface, trim_polylines_xyz,
                                face_area=area_m2)
        self.outward_sign = 1.0 if orientation_outward else -1.0
        self.body_index = body_index
        self.face_index = face_index
        self.area_m2 = area_m2

    # t_eps: 100 nm — far above float64 root-cancellation error (~1e-9 m
    # worst case at 0.1 m scale), far below any legitimate same-face re-hit
    # (e.g. an internal reflection crossing a sphere is >= um). This is the
    # self-intersection guard; whole-face exclusion would wrongly suppress
    # legitimate re-hits of the same curved face.
    def intersect(self, o, d, t_eps=1e-7, exclude_mask=None):
        """Smallest positive contained hit per ray.

        Returns (t, hit_mask); t = inf where no hit. exclude_mask (N,) True
        suppresses this face entirely for those rays (used by tests and
        special-case callers; the tracer relies on t_eps instead).
        """
        t_cand, valid = self.surface.intersect(o, d)
        t_cand = np.where(valid & (t_cand > t_eps), t_cand, np.inf)
        order = np.argsort(t_cand, axis=1)
        t_sorted = np.take_along_axis(t_cand, order, axis=1)
        t_best = np.full(len(o), np.inf)
        remaining = np.isfinite(t_sorted[:, 0])
        if exclude_mask is not None:
            remaining &= ~exclude_mask
        for k in range(t_sorted.shape[1]):
            active = remaining & ~np.isfinite(t_best) \
                & np.isfinite(t_sorted[:, k])
            if not np.any(active):
                continue
            pts = o[active] + t_sorted[active, k, None] * d[active]
            ok = self.trim.contains(self.surface.to_uv(pts))
            idx = np.where(active)[0][ok]
            t_best[idx] = t_sorted[idx, k]
        return t_best, np.isfinite(t_best)

    def normal_out_of_solid(self, p):
        """Unit normal pointing OUT of the owning solid at points p."""
        return self.outward_sign * self.surface.normal(p)
