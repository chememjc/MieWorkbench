/* ===========================================================================
 * diffk.h — Igehy-1999 ray differentials + per-surface shape operators.
 *
 * EXACT port of scripts/raytracer/differentials.py and the
 * `normal_derivative` methods of scripts/raytracer/surfaces.py (see those
 * files' headers for the sign conventions). Every function here must match
 * the Python engine to ~1e-12 relative on the same inputs; the per-surface
 * shape operators are additionally validated against central finite
 * differences of surf_normal() (tests/test_diffk_normal_deriv.c, 1e-5 rel),
 * and the Igehy transfer kernels against finite differences of full rays
 * (tests/test_diffk_transfer.c) — the same oracles Python uses in
 * scripts/raytracer/tests/test_ray_differentials.py.
 *
 * A ray carries FOUR kvec3 derivatives w.r.t. two abstract wavefront
 * parameters x, y: (dPdx, dDdx) and (dPdy, dDdy); the transverse wavefront-
 * patch area dA = |dPdx_perp x dPdy_perp| sizes the coherent gather's dA.
 * NaN rows mean "differential lost" (grating orders, scatter lobes,
 * birefringent splits, TIR/grazing) — the gather falls back per-sample.
 *
 * SIGN CORRECTION (differentials.py head): surf_normal_derivative() returns
 * d(CANONICAL normal)/dp; the CALLER multiplies by the per-ray flip
 * s = -sign(D . n_can) so the shape operator matches the tracer's working
 * n_hat = s * n_can before calling diff_reflect()/diff_refract(). For
 * diff_init_curved the RAW canonical operator is passed and the `sign`
 * argument carries the flip.
 * =========================================================================== */
#ifndef MIEWB_DIFFK_H
#define MIEWB_DIFFK_H

#include "kmath.h"
#include "surf.h"

#define DIFF_EPS 1e-12

/* ------------------------------------------------------------------- 3x3 */
typedef struct { double m[3][3]; } km3;

KFN km3 m3_zero(void) {
    km3 A;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A.m[i][j] = 0.0;
    return A;
}
KFN km3 m3_ident(void) {
    km3 A = m3_zero();
    A.m[0][0] = A.m[1][1] = A.m[2][2] = 1.0;
    return A;
}
/* outer product a_i b_j */
KFN km3 m3_outer(kvec3 a, kvec3 b) {
    km3 A;
    const double av[3] = { a.x, a.y, a.z };
    const double bv[3] = { b.x, b.y, b.z };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A.m[i][j] = av[i] * bv[j];
    return A;
}
KFN km3 m3_add(km3 A, km3 B) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A.m[i][j] += B.m[i][j];
    return A;
}
KFN km3 m3_sub(km3 A, km3 B) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A.m[i][j] -= B.m[i][j];
    return A;
}
KFN km3 m3_scale(km3 A, double s) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A.m[i][j] *= s;
    return A;
}
/* matrix product A @ B */
KFN km3 m3_mul(km3 A, km3 B) {
    km3 C = m3_zero();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += A.m[i][k] * B.m[k][j];
            C.m[i][j] = s;
        }
    return C;
}
KFN km3 m3_transpose(km3 A) {
    km3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) C.m[i][j] = A.m[j][i];
    return C;
}
/* matrix-vector A @ v */
KFN kvec3 m3_apply(km3 A, kvec3 v) {
    return v3(A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
              A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
              A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z);
}
/* the projector (I - n n^T) applied on the LEFT of a matrix H: (I - nn) @ H */
KFN km3 m3_proj_lhs(kvec3 n, km3 H) {
    km3 P = m3_sub(m3_ident(), m3_outer(n, n));
    return m3_mul(P, H);
}

/* asphere second sag derivative d2z/dr2 (surfaces.py Asphere._sag_pp) — the
 * conic term c/(1-(1+k)c^2 r^2)^{3/2} plus the even-polynomial second
 * derivative. asp_sag_p lives in surf.h. */
KFN double asp_sag_pp(const SurfC *s, double r) {
    double beta = (1.0 + s->u.asp.k) * s->u.asp.c_curv * s->u.asp.c_curv;
    double arg = 1.0 - beta * r * r;
    double base = arg > 0.0 ? arg : NAN;
    double spp = s->u.asp.c_curv / (base * sqrt(base));
    for (int i = 0; i < s->u.asp.n_coeffs; i++) {
        int mm = 4 + 2 * i;
        spp += s->u.asp.coeffs[i] * mm * (mm - 1) * pow(r, (double)(mm - 2));
    }
    return spp;
}

/* --------------------------------------------------------------------------
 * Per-surface CANONICAL shape operator dn_can/dp (surfaces.py normal_derivative)
 * QForbes is Python-only (never a SurfC kind); SURF_MESH has NO analytic
 * operator (the caller detects that via surf_has_shape_op and kills the
 * differential, mirroring Python's `hasattr(surf,'normal_derivative')`).
 * ------------------------------------------------------------------------ */
KFN int surf_has_shape_op(const SurfC *s) {
    return s->kind != SURF_MESH;
}

KFN km3 surf_normal_derivative(const SurfC *s, kvec3 p) {
    switch (s->kind) {
    case SURF_PLANE:
        /* flat: constant canonical normal field, dn/dp == 0 */
        return m3_zero();
    case SURF_SPHERE: {
        /* (I - nhat nhat^T)/r (surfaces.py:130-137) */
        kvec3 n = v3_unit(v3_sub(p, s->u.sphere.c));
        km3 A = m3_sub(m3_ident(), m3_outer(n, n));
        return m3_scale(A, 1.0 / s->u.sphere.r);
    }
    case SURF_CYLINDER: {
        /* (I - a a^T - nhat nhat^T)/r (surfaces.py:190-198) */
        kvec3 rel = v3_sub(p, s->u.cyl.o);
        kvec3 nn = v3_sub(rel, v3_scale(s->u.cyl.a, v3_dot(rel, s->u.cyl.a)));
        kvec3 n = v3_unit(nn);
        km3 A = m3_sub(m3_sub(m3_ident(), m3_outer(s->u.cyl.a, s->u.cyl.a)),
                       m3_outer(n, n));
        return m3_scale(A, 1.0 / s->u.cyl.r);
    }
    case SURF_CONE: {
        /* H = cos(ha)(I - a a^T - what what^T)/rho;
         * J = (I - nhat nhat^T) H; apex -> 0 (surfaces.py:257-281) */
        kvec3 rel = v3_sub(p, s->u.cone.apex);
        double h = v3_dot(rel, s->u.cone.a);
        kvec3 w = v3_sub(rel, v3_scale(s->u.cone.a, h));
        double rho = v3_norm(w);
        int safe = rho > 1e-300;
        if (!safe) return m3_zero();
        kvec3 what = v3_scale(w, 1.0 / rho);
        kvec3 n = surf_normal(s, p);
        km3 base = m3_sub(m3_sub(m3_ident(),
                                 m3_outer(s->u.cone.a, s->u.cone.a)),
                          m3_outer(what, what));
        km3 H = m3_scale(base, cos(s->u.cone.ha) / rho);
        return m3_proj_lhs(n, H);
    }
    case SURF_TORUS: {
        /* local analytic Hessian -> world by M=[t1;t2;a];
         * dnhat/dp = (I - nn) Hw / |grad F| (surfaces.py:369-392) */
        kvec3 rel = v3_sub(p, s->u.tor.c);
        km3 M;
        M.m[0][0] = s->u.tor.t1.x; M.m[0][1] = s->u.tor.t1.y; M.m[0][2] = s->u.tor.t1.z;
        M.m[1][0] = s->u.tor.t2.x; M.m[1][1] = s->u.tor.t2.y; M.m[1][2] = s->u.tor.t2.z;
        M.m[2][0] = s->u.tor.a.x;  M.m[2][1] = s->u.tor.a.y;  M.m[2][2] = s->u.tor.a.z;
        kvec3 q = m3_apply(M, rel);
        double u = q.x, v = q.y, w = q.z;
        double sig = hypot(u, v);
        if (sig <= 1e-300) sig = 1e-300;
        double R = s->u.tor.R;
        km3 Hl = m3_zero();
        Hl.m[0][0] = 2.0 * (u * u / (sig * sig)
                            + (sig - R) * v * v / (sig * sig * sig));
        Hl.m[1][1] = 2.0 * (v * v / (sig * sig)
                            + (sig - R) * u * u / (sig * sig * sig));
        Hl.m[0][1] = Hl.m[1][0] = 2.0 * R * u * v / (sig * sig * sig);
        Hl.m[2][2] = 2.0;
        km3 Hw = m3_mul(m3_mul(m3_transpose(M), Hl), M);
        double gnorm = 2.0 * sqrt((sig - R) * (sig - R) + w * w);
        kvec3 n = surf_normal(s, p);
        return m3_scale(m3_proj_lhs(n, Hw), 1.0 / gnorm);
    }
    case SURF_ASPHERE: {
        /* H = -[z'' ww + (z'/r)(I - aa - ww)];
         * dnhat/dp = (I - nn) H / sqrt(1 + z'^2) (surfaces.py:628-650) */
        kvec3 rel = v3_sub(p, s->u.asp.v);
        double h = v3_dot(rel, s->u.asp.a);
        kvec3 w = v3_sub(rel, v3_scale(s->u.asp.a, h));
        double r = v3_norm(w);
        int safe = r > 1e-12;
        double rs = safe ? r : 1.0;
        kvec3 what = safe ? v3_scale(w, 1.0 / rs) : s->u.asp.t1;
        double sp = safe ? asp_sag_p(s, r) : 0.0;
        double spp = asp_sag_pp(s, r);
        double sp_over_r = safe ? sp / rs : spp;    /* limit c at r->0 */
        kvec3 n = surf_normal(s, p);
        km3 aa = m3_outer(s->u.asp.a, s->u.asp.a);
        km3 ww = m3_outer(what, what);
        km3 iso = m3_sub(m3_sub(m3_ident(), aa), ww);
        km3 H = m3_scale(m3_add(m3_scale(ww, spp),
                                m3_scale(iso, sp_over_r)), -1.0);
        double gnorm = sqrt(1.0 + sp * sp);
        return m3_scale(m3_proj_lhs(n, H), 1.0 / gnorm);
    }
    default:
        return m3_zero();
    }
}

/* Sign-corrected shape operator dn_hat/dp: multiply the canonical operator
 * by the per-ray flip s = -sign(D . n_can), the SAME flip the tracer uses to
 * make n_hat oppose the ray (differentials.py head). Used by the transfer
 * oracle, which starts from the ray direction D. */
KFN km3 surf_shape_operator(const SurfC *s, kvec3 p, kvec3 D) {
    kvec3 n_can = surf_normal(s, p);
    double dn = v3_dot(D, n_can);
    double flip = -((dn > 0.0) - (dn < 0.0));
    if (flip == 0.0) flip = 1.0;
    return m3_scale(surf_normal_derivative(s, p), flip);
}

/* Same operator addressed by the tracer's already-computed working normal
 * n_hat (the interface path, tracer.py:1255-1257):
 *   fsign = sign(n_hat . n_can);  S = fsign * dn_can/dp
 * Bit-for-bit the Python interface computation, including fsign == 0 (exact
 * grazing) -> zero operator. Equivalent to surf_shape_operator's -sign(D.n_can)
 * whenever n_hat opposes the ray, which it always does at an interface. */
KFN km3 surf_shape_operator_nhat(const SurfC *s, kvec3 p, kvec3 n_hat) {
    kvec3 n_can = surf_normal(s, p);
    double dn = v3_dot(n_hat, n_can);
    double fsign = (dn > 0.0) - (dn < 0.0);
    return m3_scale(surf_normal_derivative(s, p), fsign);
}

/* --------------------------------------------------------------------------
 * Igehy transfer kernels (differentials.py)
 * ------------------------------------------------------------------------ */

/* free-space propagation of the POSITION derivative: dP <- dP + t dD */
KFN kvec3 diff_transfer(kvec3 dP, kvec3 dD, double t) {
    return v3_fma(dP, t, dD);
}

/* dt-correct an ALREADY free-transferred position derivative onto the actual
 * surface hit point (the tracer's interface `_to_surf`, tracer.py:1261-1263):
 *   dt = -(dP . n_hat) / (D . n_hat);  dP_hit = dP + dt D
 * |D.n_hat| < 1e-12 -> unchanged (Python routes denom to inf -> dt=0). */
KFN kvec3 diff_to_surface(kvec3 dP, kvec3 D, kvec3 n_hat) {
    double denom = v3_dot(D, n_hat);
    if (fabs(denom) < DIFF_EPS) return dP;
    double dt = -v3_dot(dP, n_hat) / denom;
    return v3_fma(dP, dt, D);
}

/* the module's combined transfer_to_surface (differentials.py:146-171),
 * = diff_to_surface(diff_transfer(dP,dD,t), D, n_hat). Used by the unit
 * oracle; the tracer splits the two steps (segment advance + interface). */
KFN kvec3 diff_transfer_to_surface(kvec3 dP, kvec3 dD, kvec3 D, double t,
                                   kvec3 n_hat) {
    return diff_to_surface(diff_transfer(dP, dD, t), D, n_hat);
}

/* reflection direction differential (differentials.py:177-198):
 *   dN = S @ dP_hit;  dD' = dD - 2[(dD.N + D.dN) N + (D.N) dN] */
KFN kvec3 diff_reflect(kvec3 dP_hit, kvec3 dD, kvec3 D, kvec3 N, km3 S) {
    kvec3 dN = m3_apply(S, dP_hit);
    double DN = v3_dot(D, N);
    double dDN = v3_dot(dD, N) + v3_dot(D, dN);
    kvec3 term = v3_add(v3_scale(N, dDN), v3_scale(dN, DN));
    return v3_sub(dD, v3_scale(term, 2.0));
}

/* refraction direction differential (differentials.py:201-239):
 *   dN = S @ dP_hit;  dmu = [eta - eta^2 (D.N)/(D_t.N)](dD.N + D.dN)
 *   dD' = eta dD - dmu N - mu dN,  mu = eta(D.N) - (D_t.N)
 * |D_t.N| < 1e-12 (grazing/TIR) -> NaN (the caller drops the differential). */
KFN kvec3 diff_refract(kvec3 dP_hit, kvec3 dD, kvec3 D, kvec3 N, km3 S,
                       double eta, kvec3 D_t) {
    kvec3 dN = m3_apply(S, dP_hit);
    double DN = v3_dot(D, N);
    double DtN = v3_dot(D_t, N);
    double dDN = v3_dot(dD, N) + v3_dot(D, dN);
    if (fabs(DtN) < DIFF_EPS)
        return v3(NAN, NAN, NAN);
    double mu = eta * DN - DtN;
    double dmu = (eta - eta * eta * DN / DtN) * dDN;
    return v3_sub(v3_sub(v3_scale(dD, eta), v3_scale(N, dmu)),
                  v3_scale(dN, mu));
}

/* transverse wavefront-patch area dA (differentials.py:245-258):
 * project both position derivatives perpendicular to D, |px x py|.
 * NaN in -> NaN out (natural under IEEE). */
KFN double diff_patch_area(kvec3 dPdx, kvec3 dPdy, kvec3 D) {
    kvec3 Dhat = v3_unit(D);
    kvec3 px = v3_sub(dPdx, v3_scale(Dhat, v3_dot(dPdx, Dhat)));
    kvec3 py = v3_sub(dPdy, v3_scale(Dhat, v3_dot(dPdy, Dhat)));
    return v3_norm(v3_cross(px, py));
}

#endif /* MIEWB_DIFFK_H */
