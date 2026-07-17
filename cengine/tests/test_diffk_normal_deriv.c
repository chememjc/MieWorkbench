/* test_diffk_normal_deriv.c — per-surface shape-operator oracle (WORK PLAN
 * step 1). For each analytic surface kind, compares the analytic
 * surf_normal_derivative() (kernels/diffk.h, a formula-for-formula port of
 * surfaces.py's normal_derivative) against a CENTRAL FINITE DIFFERENCE of the
 * C canonical normal surf_normal(), over randomized orientations and on-
 * surface points. Bar: 1e-5 relative (Frobenius), matching the Python
 * finite-difference bar in test_ray_differentials.py.
 *
 * The canonical normal field n(p) of every one of these primitives equals
 * grad(F)/|grad(F)| of the surface's implicit F, so its spatial Jacobian at
 * an ON-SURFACE point is exactly the analytic shape operator — the FD of
 * surf_normal is the ground truth. QForbes is Python-only (not a SurfC kind)
 * and Plane's operator is identically zero (checked directly). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels/diffk.h"

static int failures = 0;

/* tiny reproducible LCG in [0,1) */
static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static double urand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}
static double urange(double lo, double hi) { return lo + (hi - lo) * urand(); }
static kvec3 rand_unit(void) {
    for (;;) {
        kvec3 v = v3(urange(-1, 1), urange(-1, 1), urange(-1, 1));
        double n = v3_norm(v);
        if (n > 0.2 && n <= 1.0) return v3_scale(v, 1.0 / n);
    }
}

/* central FD Jacobian of surf_normal at p: column j = dn/dp_j */
static km3 fd_normal_jac(const SurfC *s, kvec3 p, double h) {
    km3 J;
    for (int j = 0; j < 3; j++) {
        kvec3 ep = p, em = p;
        double *pp = (j == 0) ? &ep.x : (j == 1) ? &ep.y : &ep.z;
        double *pm = (j == 0) ? &em.x : (j == 1) ? &em.y : &em.z;
        *pp += h; *pm -= h;
        kvec3 np = surf_normal(s, ep);
        kvec3 nm = surf_normal(s, em);
        kvec3 col = v3_scale(v3_sub(np, nm), 0.5 / h);
        J.m[0][j] = col.x; J.m[1][j] = col.y; J.m[2][j] = col.z;
    }
    return J;
}

static double m3_frob(km3 A) {
    double s = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) s += A.m[i][j] * A.m[i][j];
    return sqrt(s);
}

static double worst[8];   /* worst rel err per kind index, for reporting */

static void check_point(const char *name, int kidx, const SurfC *s, kvec3 p) {
    km3 ana = surf_normal_derivative(s, p);
    km3 fd = fd_normal_jac(s, p, 1e-7);
    double err = m3_frob(m3_sub(ana, fd));
    double den = m3_frob(fd);
    double rel = err / (den > 1e-12 ? den : 1e-12);
    if (rel > worst[kidx]) worst[kidx] = rel;
    if (!(rel < 1e-5)) {
        fprintf(stderr, "FAIL %s: rel=%.3e  |ana|=%.4g |fd|=%.4g\n",
                name, rel, m3_frob(ana), den);
        failures++;
    }
}

int main(void) {
    for (int i = 0; i < 8; i++) worst[i] = 0.0;
    const int NCFG = 24, NPT = 12;

    /* ---- Plane: operator must be exactly zero ---- */
    for (int c = 0; c < NCFG; c++) {
        SurfC s = surf_make_plane(v3(urange(-1, 1), urange(-1, 1),
                                     urange(-1, 1)), rand_unit());
        for (int k = 0; k < NPT; k++) {
            kvec3 p = v3(urange(-1, 1), urange(-1, 1), urange(-1, 1));
            km3 ana = surf_normal_derivative(&s, p);
            if (m3_frob(ana) != 0.0) {
                fprintf(stderr, "FAIL plane: nonzero operator\n");
                failures++;
            }
        }
    }

    /* ---- Sphere: p = c + r * unit ---- */
    for (int c = 0; c < NCFG; c++) {
        kvec3 cen = v3(urange(-0.2, 0.2), urange(-0.2, 0.2), urange(-0.2, 0.2));
        double r = urange(0.01, 0.2);
        SurfC s = surf_make_sphere(cen, r);
        for (int k = 0; k < NPT; k++) {
            kvec3 p = v3_fma(cen, r, rand_unit());
            check_point("sphere", 1, &s, p);
        }
    }

    /* ---- Cylinder: p = o + r*rhat + axial*a ---- */
    for (int c = 0; c < NCFG; c++) {
        kvec3 o = v3(urange(-0.2, 0.2), urange(-0.2, 0.2), urange(-0.2, 0.2));
        kvec3 a = rand_unit();
        double r = urange(0.01, 0.15);
        SurfC s = surf_make_cylinder(o, a, r);
        for (int k = 0; k < NPT; k++) {
            double ang = urange(0, 6.283185307), ax = urange(-0.3, 0.3);
            kvec3 rhat = v3_add(v3_scale(s.u.cyl.t1, cos(ang)),
                                v3_scale(s.u.cyl.t2, sin(ang)));
            kvec3 p = v3_add(v3_add(o, v3_scale(rhat, r)), v3_scale(a, ax));
            check_point("cylinder", 2, &s, p);
        }
    }

    /* ---- Cone: p = apex + Hax*a + rho*rhat, rho = Hax*tan(ha) ---- */
    for (int c = 0; c < NCFG; c++) {
        kvec3 apex = v3(urange(-0.2, 0.2), urange(-0.2, 0.2), urange(-0.2, 0.2));
        kvec3 a = rand_unit();
        double ha = urange(0.15, 1.2);
        SurfC s = surf_make_cone(apex, a, ha);
        for (int k = 0; k < NPT; k++) {
            double ang = urange(0, 6.283185307), Hax = urange(0.03, 0.3);
            double rho = Hax * tan(ha);
            kvec3 rhat = v3_add(v3_scale(s.u.cone.t1, cos(ang)),
                                v3_scale(s.u.cone.t2, sin(ang)));
            kvec3 p = v3_add(v3_add(apex, v3_scale(a, Hax)),
                             v3_scale(rhat, rho));
            check_point("cone", 3, &s, p);
        }
    }

    /* ---- Torus: p = c + (R + r cos v) rhat + r sin v * a ---- */
    for (int c = 0; c < NCFG; c++) {
        kvec3 cen = v3(urange(-0.2, 0.2), urange(-0.2, 0.2), urange(-0.2, 0.2));
        kvec3 a = rand_unit();
        double R = urange(0.05, 0.2), r = urange(0.005, 0.03);
        SurfC s = surf_make_torus(cen, a, R, r);
        for (int k = 0; k < NPT; k++) {
            double u = urange(0, 6.283185307), v = urange(0, 6.283185307);
            kvec3 rhat = v3_add(v3_scale(s.u.tor.t1, cos(u)),
                                v3_scale(s.u.tor.t2, sin(u)));
            kvec3 p = v3_add(v3_add(cen, v3_scale(rhat, R + r * cos(v))),
                             v3_scale(a, r * sin(v)));
            check_point("torus", 4, &s, p);
        }
    }

    /* ---- Asphere: p = vertex + rr*what + sag(rr)*a, rr well inside r_max
     * (varied conic k + one A4 term; the FD probes the full conic+poly). */
    for (int c = 0; c < NCFG; c++) {
        kvec3 vtx = v3(urange(-0.1, 0.1), urange(-0.1, 0.1), urange(-0.1, 0.1));
        kvec3 a = rand_unit();
        double R = urange(0.02, 0.1) * (urand() < 0.5 ? 1.0 : -1.0);
        double kk = urange(-1.5, 0.5);
        double A4 = urange(-300.0, 300.0);
        double r_max = 0.02;
        SurfC s = surf_make_asphere(vtx, a, R, kk, &A4, 1, r_max);
        for (int k = 0; k < NPT; k++) {
            double ang = urange(0, 6.283185307);
            double rr = urange(0.002, 0.8 * r_max);
            /* guard the conic domain 1-(1+k)c^2 r^2 > 0 */
            double beta = (1.0 + kk) / (R * R);
            if (1.0 - beta * rr * rr <= 0.05) continue;
            kvec3 what = v3_add(v3_scale(s.u.asp.t1, cos(ang)),
                                v3_scale(s.u.asp.t2, sin(ang)));
            double z;
            if (!asp_sag(&s, rr, &z)) continue;
            kvec3 p = v3_add(v3_add(vtx, v3_scale(what, rr)),
                             v3_scale(a, z));
            check_point("asphere", 5, &s, p);
        }
    }

    printf("worst rel err: sphere=%.2e cyl=%.2e cone=%.2e torus=%.2e "
           "asphere=%.2e\n", worst[1], worst[2], worst[3], worst[4],
           worst[5]);
    if (failures) {
        fprintf(stderr, "%d shape-operator oracle failures\n", failures);
        return 1;
    }
    printf("diffk normal_derivative oracle: all surfaces PASS (< 1e-5)\n");
    return 0;
}
