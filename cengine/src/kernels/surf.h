/* ===========================================================================
 * surf.h — analytic surface primitives: per-ray intersection candidates,
 * canonical normals, canonical UV.
 *
 * EXACT port of scripts/raytracer/surfaces.py (see that file's header for
 * the conventions). Every function here must match the Python engine to
 * ~1e-12 relative on the same inputs — pinned by cengine unit goldens and
 * scripts/raytracer/tests/test_cengine_parity.py.
 *
 * Conventions (surfaces.py:11-24):
 *   - SI metres, float64.
 *   - surf_roots() returns K candidate ray parameters, ascending NOT
 *     guaranteed; invalid roots are +INF. Positivity/t_eps/trim filtering
 *     happens in face_intersect (scene.h).
 *   - surf_normal() is the CANONICAL geometric normal (sphere: outward from
 *     center; plane: stored n). The face flips it by orientation_outward.
 *   - surf_to_uv() maps to the canonical 2D parameterization used ONLY for
 *     trim containment; azimuthal u in (-pi, pi].
 *
 * Phase B: all six analytic kinds (plane/sphere/cylinder/cone/torus/
 * asphere). Mesh faces arrive with the phase-C BLAS.
 * =========================================================================== */
#ifndef MIEWB_SURF_H
#define MIEWB_SURF_H

#include "kmath.h"
#include "quartic.h"

/* max even-asphere polynomial coefficients (A4..; extractor emits <= 8) */
#define ASPHERE_MAX_COEFFS 12

enum {
    SURF_PLANE = 0,
    SURF_SPHERE = 1,
    SURF_CYLINDER = 2,
    SURF_CONE = 3,
    SURF_TORUS = 4,
    SURF_ASPHERE = 5,
    SURF_MESH = 6,
};

/* Maximum root count of any primitive (torus/asphere have 4). */
#define SURF_K_MAX 4

typedef struct {
    uint8_t kind;
    uint8_t periodic_u;     /* sphere/cyl/cone/torus: azimuthal wrap in UV */
    union {
        struct {            /* surfaces.py Plane */
            kvec3 origin, n, t1, t2;
        } plane;
        struct {            /* surfaces.py Sphere; axis fixed +z */
            kvec3 c;
            double r;
            kvec3 axis, t1, t2;
        } sphere;
        struct {            /* surfaces.py Cylinder */
            kvec3 o, a, t1, t2;
            double r;
        } cyl;
        struct {            /* surfaces.py Cone (infinite, apex + axis) */
            kvec3 apex, a, t1, t2;
            double ha, cos2;    /* half-angle, cos^2(ha) */
        } cone;
        struct {            /* surfaces.py Torus */
            kvec3 c, a, t1, t2;
            double R, r;
        } tor;
        struct {            /* surfaces.py Asphere (conic + even poly cap) */
            kvec3 v, a, t1, t2;
            double R, c_curv, k, r_max;
            double coeffs[ASPHERE_MAX_COEFFS];
            int32_t n_coeffs;
            double z_lo, z_hi;  /* sampled axial extent (constructor) */
        } asp;
    } u;
} SurfC;

/* ---- constructors (host-side; fill the derived frames once) ---- */
KFN SurfC surf_make_plane(kvec3 origin, kvec3 normal) {
    SurfC s = {0};
    s.kind = SURF_PLANE;
    s.periodic_u = 0;
    s.u.plane.origin = origin;
    s.u.plane.n = v3_unit(normal);
    k_plane_frame(s.u.plane.n, &s.u.plane.t1, &s.u.plane.t2);
    return s;
}

KFN SurfC surf_make_sphere(kvec3 center, double radius) {
    SurfC s = {0};
    s.kind = SURF_SPHERE;
    s.periodic_u = 1;
    s.u.sphere.c = center;
    s.u.sphere.r = radius;
    /* deterministic pole axis for UV (surfaces.py:86) */
    s.u.sphere.axis = v3(0.0, 0.0, 1.0);
    k_plane_frame(s.u.sphere.axis, &s.u.sphere.t1, &s.u.sphere.t2);
    return s;
}

KFN SurfC surf_make_cylinder(kvec3 origin, kvec3 axis, double radius) {
    SurfC s = {0};
    s.kind = SURF_CYLINDER;
    s.periodic_u = 1;
    s.u.cyl.o = origin;
    s.u.cyl.a = v3_unit(axis);
    s.u.cyl.r = radius;
    k_plane_frame(s.u.cyl.a, &s.u.cyl.t1, &s.u.cyl.t2);
    return s;
}

KFN SurfC surf_make_cone(kvec3 apex, kvec3 axis, double half_angle) {
    SurfC s = {0};
    s.kind = SURF_CONE;
    s.periodic_u = 1;
    s.u.cone.apex = apex;
    s.u.cone.a = v3_unit(axis);
    s.u.cone.ha = half_angle;
    s.u.cone.cos2 = cos(half_angle) * cos(half_angle);
    k_plane_frame(s.u.cone.a, &s.u.cone.t1, &s.u.cone.t2);
    return s;
}

KFN SurfC surf_make_torus(kvec3 center, kvec3 axis, double major_r,
                          double minor_r) {
    SurfC s = {0};
    s.kind = SURF_TORUS;
    s.periodic_u = 1;
    s.u.tor.c = center;
    s.u.tor.a = v3_unit(axis);
    s.u.tor.R = major_r;
    s.u.tor.r = minor_r;
    k_plane_frame(s.u.tor.a, &s.u.tor.t1, &s.u.tor.t2);
    return s;
}

/* asphere sag z(r) and dz/dr — port of Asphere._sag/_sag_p
 * (surfaces.py:437-459). Returns 0 (invalid) outside the conic/validity
 * disc, with *z untouched. */
KFN int asp_sag(const SurfC *s, double r, double *z) {
    double beta = (1.0 + s->u.asp.k) * s->u.asp.c_curv * s->u.asp.c_curv;
    double arg = 1.0 - beta * r * r;
    if (!(arg > 0.0) || r > s->u.asp.r_max + 1e-12) return 0;
    double zz = s->u.asp.c_curv * r * r / (1.0 + sqrt(arg));
    for (int i = 0; i < s->u.asp.n_coeffs; i++)
        zz += s->u.asp.coeffs[i] * pow(r, (double)(4 + 2 * i));
    *z = zz;
    return 1;
}

KFN double asp_sag_p(const SurfC *s, double r) {
    double beta = (1.0 + s->u.asp.k) * s->u.asp.c_curv * s->u.asp.c_curv;
    double arg = 1.0 - beta * r * r;
    double sp = s->u.asp.c_curv * r / sqrt(arg > 0.0 ? arg : NAN);
    for (int i = 0; i < s->u.asp.n_coeffs; i++) {
        int m = 4 + 2 * i;
        sp += s->u.asp.coeffs[i] * m * pow(r, (double)(m - 1));
    }
    return sp;
}

KFN SurfC surf_make_asphere(kvec3 vertex, kvec3 axis, double R, double k,
                            const double *coeffs, int n_coeffs,
                            double r_max) {
    SurfC s = {0};
    s.kind = SURF_ASPHERE;
    s.periodic_u = 0;   /* planar-projection UV (surfaces.py:409-415) */
    s.u.asp.v = vertex;
    s.u.asp.a = v3_unit(axis);
    s.u.asp.R = R;
    s.u.asp.c_curv = 1.0 / R;
    s.u.asp.k = k;
    s.u.asp.r_max = r_max;
    s.u.asp.n_coeffs = n_coeffs > ASPHERE_MAX_COEFFS ? ASPHERE_MAX_COEFFS
                                                     : n_coeffs;
    for (int i = 0; i < s.u.asp.n_coeffs; i++)
        s.u.asp.coeffs[i] = coeffs[i];
    k_plane_frame(s.u.asp.a, &s.u.asp.t1, &s.u.asp.t2);
    /* sampled axial extent over [0, r_max] (surfaces.py:428-434) */
    double z_lo = 0.0, z_hi = 0.0;
    for (int i = 0; i < 64; i++) {
        double rr = s.u.asp.r_max * (double)i / 63.0;
        double zz;
        if (asp_sag(&s, rr, &zz)) {
            if (zz < z_lo) z_lo = zz;
            if (zz > z_hi) z_hi = zz;
        }
    }
    s.u.asp.z_lo = z_lo;
    s.u.asp.z_hi = z_hi;
    return s;
}

/* ---- intersection candidates ----
 * Fills t[0..K-1] with candidate ray parameters (+INF where invalid) and
 * returns K (the primitive's root count). Assumes |d| == 1. */
KFN int surf_roots(const SurfC *s, kvec3 o, kvec3 d, double t[SURF_K_MAX]) {
    switch (s->kind) {
    case SURF_PLANE: {
        /* surfaces.py:55-61: t = ((origin - o).n) / (d.n), valid |dn|>1e-15 */
        double dn = v3_dot(d, s->u.plane.n);
        double num = v3_dot(v3_sub(s->u.plane.origin, o), s->u.plane.n);
        t[0] = (fabs(dn) > 1e-15) ? num / dn : INFINITY;
        return 1;
    }
    case SURF_SPHERE: {
        /* surfaces.py:89-98: b = oc.d, c = |oc|^2 - r^2, disc = b^2 - c */
        kvec3 oc = v3_sub(o, s->u.sphere.c);
        double b = v3_dot(oc, d);
        double c = v3_dot(oc, oc) - s->u.sphere.r * s->u.sphere.r;
        double disc = b * b - c;
        if (disc >= 0.0) {
            double sq = sqrt(disc);
            t[0] = -b - sq;
            t[1] = -b + sq;
        } else {
            t[0] = INFINITY;
            t[1] = INFINITY;
        }
        return 2;
    }
    case SURF_CYLINDER: {
        /* surfaces.py:148-161: quadratic on the axis-perpendicular parts */
        kvec3 oc = v3_sub(o, s->u.cyl.o);
        double da = v3_dot(d, s->u.cyl.a);
        double oa = v3_dot(oc, s->u.cyl.a);
        kvec3 dp = v3_sub(d, v3_scale(s->u.cyl.a, da));
        kvec3 op = v3_sub(oc, v3_scale(s->u.cyl.a, oa));
        double A = v3_dot(dp, dp);
        double B = v3_dot(dp, op);
        double C = v3_dot(op, op) - s->u.cyl.r * s->u.cyl.r;
        double disc = B * B - A * C;
        if (disc >= 0.0 && A > 1e-30) {
            double sq = sqrt(disc);
            t[0] = (-B - sq) / A;
            t[1] = (-B + sq) / A;
        } else {
            t[0] = INFINITY;
            t[1] = INFINITY;
        }
        return 2;
    }
    case SURF_CONE: {
        /* surfaces.py:211-235 incl. the linear (A~0) branch and the
         * mirror-cone rejection (points behind the apex) */
        kvec3 co = v3_sub(o, s->u.cone.apex);
        double dv = v3_dot(d, s->u.cone.a);
        double cv = v3_dot(co, s->u.cone.a);
        double A = dv * dv - s->u.cone.cos2 * v3_dot(d, d);
        double B = dv * cv - s->u.cone.cos2 * v3_dot(d, co);
        double C = cv * cv - s->u.cone.cos2 * v3_dot(co, co);
        double disc = B * B - A * C;
        int lin = fabs(A) < 1e-30;
        if (lin) {
            t[0] = (fabs(B) > 1e-30) ? -C / (2.0 * B) : INFINITY;
            t[1] = INFINITY;
        } else if (disc >= 0.0) {
            double sq = sqrt(disc);
            t[0] = (-B - sq) / A;
            t[1] = (-B + sq) / A;
        } else {
            t[0] = INFINITY;
            t[1] = INFINITY;
        }
        for (int i = 0; i < 2; i++) {
            if (!isfinite(t[i])) continue;
            kvec3 p = v3_fma(o, t[i], d);
            if (v3_dot(v3_sub(p, s->u.cone.apex), s->u.cone.a) < 0.0)
                t[i] = INFINITY;
        }
        return 2;
    }
    case SURF_TORUS: {
        /* surfaces.py:294-344: quartic in the torus local frame; the
         * Python engine solves via companion eigenvalues + 2 Newton
         * polish steps — here a closed-form quartic + the SAME polish */
        kvec3 rel = v3_sub(o, s->u.tor.c);
        kvec3 ol = v3(v3_dot(rel, s->u.tor.t1), v3_dot(rel, s->u.tor.t2),
                      v3_dot(rel, s->u.tor.a));
        kvec3 dl = v3(v3_dot(d, s->u.tor.t1), v3_dot(d, s->u.tor.t2),
                      v3_dot(d, s->u.tor.a));
        double dd = v3_dot(dl, dl);
        double od = v3_dot(ol, dl);
        double oo = v3_dot(ol, ol);
        double R2 = s->u.tor.R * s->u.tor.R;
        double r2 = s->u.tor.r * s->u.tor.r;
        double k = oo - r2 - R2;
        double c4 = dd * dd;
        double c3 = 4.0 * dd * od;
        double c2 = 2.0 * dd * k + 4.0 * od * od + 4.0 * R2 * dl.z * dl.z;
        double c1 = 4.0 * od * k + 8.0 * R2 * ol.z * dl.z;
        double c0 = k * k + 4.0 * R2 * (ol.z * ol.z - r2);
        double roots[4];
        int n = k_solve_quartic(c4, c3, c2, c1, c0, roots);
        for (int i = 0; i < 4; i++) t[i] = INFINITY;
        for (int i = 0; i < n && i < 4; i++) {
            double x = roots[i];
            /* two Newton polish steps on the quartic (surfaces.py:333-342) */
            for (int it = 0; it < 2; it++) {
                double f = (((c4 * x + c3) * x + c2) * x + c1) * x + c0;
                double fp = ((4.0 * c4 * x + 3.0 * c3) * x + 2.0 * c2) * x
                            + c1;
                if (fabs(fp) > 1e-300 && isfinite(f)) {
                    double step = f / fp;
                    if (isfinite(step)) x -= step;
                }
            }
            t[i] = x;
        }
        return 4;
    }
    case SURF_ASPHERE: {
        /* surfaces.py:473-599: bounding cylinder + axial slab -> valid t
         * interval; sample f(t) = h(t) - z(r(t)) at 32 points to bracket
         * sign changes; 8 bisection + 12 bracket-guarded Newton steps.
         * Constants are VERBATIM — they define the parity contract. */
        for (int i = 0; i < 4; i++) t[i] = INFINITY;
        kvec3 rel0 = v3_sub(o, s->u.asp.v);
        double h0 = v3_dot(rel0, s->u.asp.a);
        double da = v3_dot(d, s->u.asp.a);
        kvec3 w0 = v3_sub(rel0, v3_scale(s->u.asp.a, h0));
        kvec3 dp = v3_sub(d, v3_scale(s->u.asp.a, da));
        double A = v3_dot(dp, dp);
        double w0dp = v3_dot(w0, dp);
        double w0w0 = v3_dot(w0, w0);

        double t_lo = -INFINITY, t_hi = INFINITY;
        double Cc = w0w0 - s->u.asp.r_max * s->u.asp.r_max;
        if (A > 1e-300) {                       /* bounding cylinder */
            double disc = w0dp * w0dp - A * Cc;
            if (disc <= 0.0) return 4;          /* never enters the disc */
            double sq = sqrt(disc);
            t_lo = (-w0dp - sq) / A;
            t_hi = (-w0dp + sq) / A;
        } else if (Cc > 0.0) {
            return 4;                           /* parallel, outside disc */
        }
        double margin = 1e-7 + 0.01 * (s->u.asp.z_hi - s->u.asp.z_lo);
        double z_lo = s->u.asp.z_lo - margin;
        double z_hi = s->u.asp.z_hi + margin;
        if (fabs(da) > 1e-300) {                /* axial slab */
            double ta = (z_lo - h0) / da;
            double tb = (z_hi - h0) / da;
            double slo = ta < tb ? ta : tb;
            double shi = ta < tb ? tb : ta;
            if (slo > t_lo) t_lo = slo;
            if (shi < t_hi) t_hi = shi;
        } else if (h0 < z_lo || h0 > z_hi) {
            return 4;                           /* perpendicular, misses */
        }
        double t_start = t_lo > 0.0 ? t_lo : 0.0;   /* forward only */
        double t_end = t_hi;
        if (!(t_end > t_start)) return 4;

        /* f(t) = h(t) - z(r(t)); NAN outside the sag validity disc */
        #define ASP_R_OF(tt) \
            sqrt(fmax(w0w0 + 2.0 * (tt) * w0dp + (tt) * (tt) * A, 0.0))
        #define ASP_F_OF(tt, fout) do { \
            double _r = ASP_R_OF(tt); double _z; \
            (fout) = asp_sag(s, _r, &_z) ? (h0 + (tt) * da) - _z : NAN; \
        } while (0)

        const int S = 32;
        double span = t_end - t_start;
        int nfound = 0;
        double fprev;
        ASP_F_OF(t_start, fprev);
        double tprev = t_start;
        for (int j = 1; j <= S && nfound < 4; j++) {
            double tj = t_start + span * (double)j / (double)S;
            double fj;
            ASP_F_OF(tj, fj);
            int bracket = isfinite(fprev) && isfinite(fj)
                          && ((fprev > 0.0) != (fj > 0.0)) && fprev != 0.0;
            if (bracket) {
                double a = tprev, b = tj, fa = fprev;
                for (int it = 0; it < 8; it++) {        /* bisection */
                    double m = 0.5 * (a + b);
                    double fm;
                    ASP_F_OF(m, fm);
                    if ((fm > 0.0) == (fa > 0.0)) { a = m; fa = fm; }
                    else b = m;
                }
                double x = 0.5 * (a + b);
                for (int it = 0; it < 12; it++) {       /* guarded Newton */
                    double fx;
                    ASP_F_OF(x, fx);
                    double r = ASP_R_OF(x);
                    double rs = r > 1e-300 ? r : 1.0;
                    double drdt = (w0dp + x * A) / rs;
                    double sp = r > 1e-300 ? asp_sag_p(s, r) : 0.0;
                    double fpx = da - sp * drdt;
                    double xn = (fabs(fpx) > 1e-300) ? x - fx / fpx : x;
                    if (xn < a || xn > b || !isfinite(xn))
                        xn = 0.5 * (a + b);
                    double fxn;
                    ASP_F_OF(xn, fxn);
                    if ((fxn > 0.0) == (fa > 0.0)) { a = xn; fa = fxn; }
                    else b = xn;
                    x = xn;
                }
                double fx;
                ASP_F_OF(x, fx);
                if (isfinite(fx) && fabs(fx) < 1e-13)
                    t[nfound++] = x;
            }
            fprev = fj;
            tprev = tj;
        }
        #undef ASP_F_OF
        #undef ASP_R_OF
        return 4;
    }
    default:
        /* unreachable: scene validation rejects unported kinds up front */
        t[0] = INFINITY;
        return 1;
    }
}

/* ---- canonical normal at a surface point ---- */
KFN kvec3 surf_normal(const SurfC *s, kvec3 p) {
    switch (s->kind) {
    case SURF_PLANE:
        return s->u.plane.n;
    case SURF_SPHERE:
        return v3_unit(v3_sub(p, s->u.sphere.c));
    case SURF_CYLINDER: {
        kvec3 rel = v3_sub(p, s->u.cyl.o);
        kvec3 n = v3_sub(rel, v3_scale(s->u.cyl.a,
                                       v3_dot(rel, s->u.cyl.a)));
        return v3_unit(n);
    }
    case SURF_CONE: {
        /* surfaces.py:237-244: n = rhat cos(ha) - a sin(ha) */
        kvec3 rel = v3_sub(p, s->u.cone.apex);
        double h = v3_dot(rel, s->u.cone.a);
        kvec3 rad = v3_sub(rel, v3_scale(s->u.cone.a, h));
        double rl = v3_norm(rad);
        kvec3 rhat = rl > 1e-300 ? v3_scale(rad, 1.0 / rl)
                                 : v3(0.0, 0.0, 0.0);
        kvec3 n = v3_sub(v3_scale(rhat, cos(s->u.cone.ha)),
                         v3_scale(s->u.cone.a, sin(s->u.cone.ha)));
        return v3_unit(n);
    }
    case SURF_TORUS: {
        /* surfaces.py:346-354: outward from the nearest spine point */
        kvec3 rel = v3_sub(p, s->u.tor.c);
        double h = v3_dot(rel, s->u.tor.a);
        kvec3 rad = v3_sub(rel, v3_scale(s->u.tor.a, h));
        double rl = v3_norm(rad);
        kvec3 rhat = rl > 1e-300 ? v3_scale(rad, 1.0 / rl)
                                 : v3(0.0, 0.0, 0.0);
        kvec3 ring = v3_fma(s->u.tor.c, s->u.tor.R, rhat);
        return v3_unit(v3_sub(p, ring));
    }
    case SURF_ASPHERE: {
        /* surfaces.py:609-619: grad(h - z(r)) = a - z'(r) what */
        kvec3 rel = v3_sub(p, s->u.asp.v);
        double h = v3_dot(rel, s->u.asp.a);
        kvec3 w = v3_sub(rel, v3_scale(s->u.asp.a, h));
        double r = v3_norm(w);
        kvec3 what = r > 1e-300 ? v3_scale(w, 1.0 / r)
                                : v3(0.0, 0.0, 0.0);
        double sp = r > 1e-300 ? asp_sag_p(s, r) : 0.0;
        kvec3 g = v3_sub(s->u.asp.a, v3_scale(what, sp));
        return v3_unit(g);
    }
    default:
        return v3(0.0, 0.0, 1.0);
    }
}

/* ---- canonical UV (for trim containment only) ---- */
KFN void surf_to_uv(const SurfC *s, kvec3 p, double *uu, double *vv) {
    switch (s->kind) {
    case SURF_PLANE: {
        kvec3 rel = v3_sub(p, s->u.plane.origin);
        *uu = v3_dot(rel, s->u.plane.t1);
        *vv = v3_dot(rel, s->u.plane.t2);
        return;
    }
    case SURF_SPHERE: {
        kvec3 rel = v3_sub(p, s->u.sphere.c);
        double x = v3_dot(rel, s->u.sphere.t1);
        double y = v3_dot(rel, s->u.sphere.t2);
        double z = v3_dot(rel, s->u.sphere.axis);
        *uu = atan2(y, x);                 /* azimuth (-pi, pi] */
        *vv = atan2(z, hypot(x, y));       /* latitude (-pi/2, pi/2) */
        return;
    }
    case SURF_CYLINDER: {
        /* surfaces.py:168-174: (azimuth, axial distance) */
        kvec3 rel = v3_sub(p, s->u.cyl.o);
        *uu = atan2(v3_dot(rel, s->u.cyl.t2), v3_dot(rel, s->u.cyl.t1));
        *vv = v3_dot(rel, s->u.cyl.a);
        return;
    }
    case SURF_CONE: {
        /* surfaces.py:246-252 */
        kvec3 rel = v3_sub(p, s->u.cone.apex);
        *uu = atan2(v3_dot(rel, s->u.cone.t2), v3_dot(rel, s->u.cone.t1));
        *vv = v3_dot(rel, s->u.cone.a);
        return;
    }
    case SURF_TORUS: {
        /* surfaces.py:356-364: (main-axis angle, tube angle) */
        kvec3 rel = v3_sub(p, s->u.tor.c);
        double x = v3_dot(rel, s->u.tor.t1);
        double y = v3_dot(rel, s->u.tor.t2);
        *uu = atan2(y, x);
        double h = v3_dot(rel, s->u.tor.a);
        double rho = hypot(x, y) - s->u.tor.R;
        *vv = atan2(h, rho);
        return;
    }
    case SURF_ASPHERE: {
        /* surfaces.py:621-623: PLANAR projection (not azimuthal — see the
         * Asphere docstring for why) */
        kvec3 rel = v3_sub(p, s->u.asp.v);
        *uu = v3_dot(rel, s->u.asp.t1);
        *vv = v3_dot(rel, s->u.asp.t2);
        return;
    }
    default:
        *uu = 0.0;
        *vv = 0.0;
        return;
    }
}

/* ---- inverse UV (source area sampling; sources.py _uv_to_xyz) ---- */
KFN kvec3 surf_uv_to_xyz(const SurfC *s, double u, double v) {
    switch (s->kind) {
    case SURF_PLANE:
        return v3_add(s->u.plane.origin,
                      v3_add(v3_scale(s->u.plane.t1, u),
                             v3_scale(s->u.plane.t2, v)));
    case SURF_SPHERE: {
        double cu = cos(u), su = sin(u), cv = cos(v), sv = sin(v);
        kvec3 r = s->u.sphere.c;
        r = v3_fma(r, s->u.sphere.r * cv * cu, s->u.sphere.t1);
        r = v3_fma(r, s->u.sphere.r * cv * su, s->u.sphere.t2);
        r = v3_fma(r, s->u.sphere.r * sv, s->u.sphere.axis);
        return r;
    }
    case SURF_CYLINDER: {
        /* sources._uv_to_xyz cylinder branch */
        kvec3 r = s->u.cyl.o;
        r = v3_fma(r, s->u.cyl.r * cos(u), s->u.cyl.t1);
        r = v3_fma(r, s->u.cyl.r * sin(u), s->u.cyl.t2);
        r = v3_fma(r, v, s->u.cyl.a);
        return r;
    }
    default:
        /* emitting faces are plane/sphere/cylinder in practice; other
         * kinds would need their own inverse — reject at request load */
        return v3(0.0, 0.0, 0.0);
    }
}

#endif /* MIEWB_SURF_H */
