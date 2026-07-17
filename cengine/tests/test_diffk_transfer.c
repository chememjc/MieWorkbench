/* test_diffk_transfer.c — Igehy transfer-kernel oracle (WORK PLAN step 2).
 * Ports scripts/raytracer/tests/test_ray_differentials.py::_fd_case: trace a
 * central ray to an analytic surface, propagate (dPdx,dDdx,dPdy,dDdy) through
 * transfer_to_surface + reflect/refract, and compare dP_hit / dD_out against a
 * CENTRAL FINITE DIFFERENCE of four offset rays traced through the SAME path.
 * Bar 1e-5 relative. Also checks the free-space r^2 spherical-wave law, the
 * collimated area invariant, and TIR/grazing NaN propagation. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kernels/diffk.h"
#include "kernels/fresnel.h"

static int failures = 0;
static double worst = 0.0;

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}
static kvec3 randn3(void) {
    /* Box-Muller-ish: just uniform in [-1,1] is fine for a conditioning seed */
    return v3(2 * urand() - 1, 2 * urand() - 1, 2 * urand() - 1);
}

/* smallest positive root > 1e-9 (nearest hit) */
static int nearest_hit(const SurfC *s, kvec3 o, kvec3 d, double *t_out,
                       kvec3 *p_out) {
    double t[SURF_K_MAX];
    int K = surf_roots(s, o, d, t);
    double best = INFINITY;
    for (int i = 0; i < K; i++)
        if (isfinite(t[i]) && t[i] > 1e-9 && t[i] < best) best = t[i];
    if (!isfinite(best)) return 0;
    *t_out = best;
    *p_out = v3_fma(o, best, d);
    return 1;
}

/* trace_interaction: returns t, p, sflip, n_hat, d_out */
static int trace_interaction(const SurfC *s, kvec3 o, kvec3 d, int refract,
                             double n1, double n2, double *t_out, kvec3 *p_out,
                             kvec3 *nhat_out, kvec3 *dout_out) {
    d = v3_unit(d);
    double t;
    kvec3 p;
    if (!nearest_hit(s, o, d, &t, &p)) return 0;
    kvec3 n_can = surf_normal(s, p);
    double dn = v3_dot(d, n_can);
    double sflip = -((dn > 0) - (dn < 0));
    if (sflip == 0.0) sflip = 1.0;
    kvec3 n_hat = v3_scale(n_can, sflip);
    double cos_i = -v3_dot(d, n_hat);
    kvec3 d_out = refract ? fresnel_refract_dir(d, n_hat, cos_i, n1, n2)
                          : fresnel_reflect_dir(d, n_hat);
    *t_out = t; *p_out = p; *nhat_out = n_hat; *dout_out = d_out;
    return 1;
}

static double rel_err(kvec3 a, kvec3 b) {
    double nb = v3_norm(b);
    return v3_norm(v3_sub(a, b)) / (nb > 1e-30 ? nb : 1e-30);
}

static void fd_case(const char *name, const SurfC *s, kvec3 O0, kvec3 D0,
                    int refract, double n1, double n2) {
    D0 = v3_unit(D0);
    kvec3 dPdx = randn3(), dPdy = randn3(), dDdx = randn3(), dDdy = randn3();
    /* dD perpendicular to D0 (matches the Python fixture) */
    dDdx = v3_sub(dDdx, v3_scale(D0, v3_dot(dDdx, D0)));
    dDdy = v3_sub(dDdy, v3_scale(D0, v3_dot(dDdy, D0)));

    double t0; kvec3 p0, n_hat0, dout0;
    if (!trace_interaction(s, O0, D0, refract, n1, n2, &t0, &p0, &n_hat0,
                           &dout0)) {
        fprintf(stderr, "FAIL %s: central ray missed\n", name);
        failures++;
        return;
    }
    km3 S0 = surf_shape_operator(s, p0, D0);
    kvec3 dPx = diff_transfer_to_surface(dPdx, dDdx, D0, t0, n_hat0);
    kvec3 dPy = diff_transfer_to_surface(dPdy, dDdy, D0, t0, n_hat0);
    kvec3 dDx, dDy;
    if (refract) {
        double eta = n1 / n2;
        dDx = diff_refract(dPx, dDdx, D0, n_hat0, S0, eta, dout0);
        dDy = diff_refract(dPy, dDdy, D0, n_hat0, S0, eta, dout0);
    } else {
        dDx = diff_reflect(dPx, dDdx, D0, n_hat0, S0);
        dDy = diff_reflect(dPy, dDdy, D0, n_hat0, S0);
    }

    /* FD oracle: 4 offset rays */
    double h = 1e-7;
    kvec3 pxp, pxm, pyp, pym, dxp, dxm, dyp, dym, np, dp2;
    double tt;
    #define OFFSET(dP, dD, step, pv, dv)                                      \
        do {                                                                  \
            kvec3 Oo = v3_fma(O0, (step), (dP));                              \
            kvec3 Do = v3_unit(v3_fma(D0, (step), (dD)));                     \
            if (!trace_interaction(s, Oo, Do, refract, n1, n2, &tt, &np,      \
                                   &dp2, &(dv))) {                            \
                fprintf(stderr, "FAIL %s: offset ray missed\n", name);       \
                failures++; return;                                          \
            }                                                                 \
            (pv) = np;                                                        \
        } while (0)
    OFFSET(dPdx, dDdx, +h, pxp, dxp);
    OFFSET(dPdx, dDdx, -h, pxm, dxm);
    OFFSET(dPdy, dDdy, +h, pyp, dyp);
    OFFSET(dPdy, dDdy, -h, pym, dym);
    #undef OFFSET
    kvec3 dPx_fd = v3_scale(v3_sub(pxp, pxm), 0.5 / h);
    kvec3 dPy_fd = v3_scale(v3_sub(pyp, pym), 0.5 / h);
    kvec3 dDx_fd = v3_scale(v3_sub(dxp, dxm), 0.5 / h);
    kvec3 dDy_fd = v3_scale(v3_sub(dyp, dym), 0.5 / h);

    double e[4] = { rel_err(dPx, dPx_fd), rel_err(dPy, dPy_fd),
                    rel_err(dDx, dDx_fd), rel_err(dDy, dDy_fd) };
    const char *lbl[4] = { "dP_hit x", "dP_hit y", "dD out x", "dD out y" };
    for (int i = 0; i < 4; i++) {
        if (e[i] > worst) worst = e[i];
        if (!(e[i] < 1e-5)) {
            fprintf(stderr, "FAIL %s [%s]: rel=%.3e\n", name, lbl[i], e[i]);
            failures++;
        }
    }
}

int main(void) {
    /* ---- free-space spherical wave: dP=0, dD spans transverse; area ~ r^2 */
    {
        kvec3 D = v3_unit(v3(0.3, -0.7, 0.5));
        kvec3 a = v3(0, 0, 0);
        double ax = fabs(D.x), ay = fabs(D.y), az = fabs(D.z);
        if (ax <= ay && ax <= az) a.x = 1; else if (ay <= az) a.y = 1; else a.z = 1;
        kvec3 e1 = v3_unit(v3_cross(a, D));
        kvec3 e2 = v3_cross(D, e1);
        kvec3 z = v3(0, 0, 0);
        double radii[5] = { 0.5, 1.0, 2.0, 4.0, 8.0 };
        double ratio0 = 0.0;
        int ok = 1;
        for (int i = 0; i < 5; i++) {
            kvec3 dPx = diff_transfer(z, e1, radii[i]);
            kvec3 dPy = diff_transfer(z, e2, radii[i]);
            double A = diff_patch_area(dPx, dPy, D);
            double ratio = A / (radii[i] * radii[i]);
            if (i == 0) ratio0 = ratio;
            else if (fabs(ratio / ratio0 - 1.0) > 1e-10) ok = 0;
        }
        if (!ok) { fprintf(stderr, "FAIL spherical-wave r^2\n"); failures++; }
    }
    /* ---- collimated area invariant: dD=0 -> constant patch area */
    {
        kvec3 D = v3_unit(v3(0.1, 0.2, 1.0));
        kvec3 a = v3(0, 0, 0);
        double ax = fabs(D.x), ay = fabs(D.y), az = fabs(D.z);
        if (ax <= ay && ax <= az) a.x = 1; else if (ay <= az) a.y = 1; else a.z = 1;
        kvec3 e1 = v3_unit(v3_cross(a, D));
        kvec3 e2 = v3_cross(D, e1);
        double h = 0.003;
        kvec3 dPx0 = v3_scale(e1, h), dPy0 = v3_scale(e2, h), z = v3(0, 0, 0);
        double a0 = diff_patch_area(dPx0, dPy0, D);
        double ts[4] = { 0.0, 0.05, 0.5, 5.0 };
        for (int i = 0; i < 4; i++) {
            kvec3 dPx = diff_transfer(dPx0, z, ts[i]);
            kvec3 dPy = diff_transfer(dPy0, z, ts[i]);
            if (fabs(diff_patch_area(dPx, dPy, D) - a0) > 1e-14) {
                fprintf(stderr, "FAIL collimated invariant\n"); failures++;
            }
        }
    }

    /* ---- FD reflect/refract on sphere, cylinder, asphere (seeded points
     * mirror the Python fixtures) ---- */
    {
        SurfC sph = surf_make_sphere(v3(0.01, -0.02, 0.03), 0.05);
        for (int rr = 0; rr < 2; rr++)
            fd_case(rr ? "sphere refract" : "sphere reflect", &sph,
                    v3(0.30, 0.12, 0.22), v3(-1.0, -0.45, -0.6), rr, 1.0, 1.5);

        SurfC cyl = surf_make_cylinder(v3(0, 0, 0), v3(0.15, 0.25, 1.0), 0.04);
        for (int rr = 0; rr < 2; rr++)
            fd_case(rr ? "cylinder refract" : "cylinder reflect", &cyl,
                    v3(0.25, 0.10, 0.05), v3(-1.0, -0.35, 0.1), rr, 1.0, 1.5);

        double A4 = 5.0e2;
        SurfC asp = surf_make_asphere(v3(0, 0, 0), v3(0, 0, 1), 0.03, -0.5,
                                      &A4, 1, 0.02);
        for (int rr = 0; rr < 2; rr++)
            fd_case(rr ? "asphere refract" : "asphere reflect", &asp,
                    v3(0.006, -0.004, 0.20), v3(0.02, -0.015, -1.0), rr,
                    1.0, 1.5);
    }

    /* ---- TIR / grazing produces NaN, no crash (Plane, 60deg past critical) */
    {
        SurfC pl = surf_make_plane(v3(0, 0, 0), v3(0, 0, 1));
        double n1 = 1.5, n2 = 1.0;
        kvec3 n_hat = v3(0, 0, 1);
        double th = 60.0 * K_PI / 180.0;
        kvec3 D = v3(sin(th), 0.0, -cos(th));
        double cos_i = -v3_dot(D, n_hat);
        kvec3 d_out = fresnel_refract_dir(D, n_hat, cos_i, n1, n2);
        km3 S = surf_normal_derivative(&pl, v3(0, 0, 0));   /* zero */
        kvec3 dP_hit = v3(0.001, 0.002, 0.0), dD = v3(0.0, 0.001, 0.0);
        kvec3 dD_ref = diff_refract(dP_hit, dD, D, n_hat, S, n1 / n2, d_out);
        if (!(isnan(dD_ref.x) && isnan(dD_ref.y) && isnan(dD_ref.z))) {
            fprintf(stderr, "FAIL TIR: dD not NaN\n"); failures++;
        }
        double dA = diff_patch_area(dP_hit, dD_ref, D);
        if (!isnan(dA)) { fprintf(stderr, "FAIL TIR: dA not NaN\n"); failures++; }
    }

    printf("worst transfer rel err: %.2e\n", worst);
    if (failures) {
        fprintf(stderr, "%d transfer-kernel oracle failures\n", failures);
        return 1;
    }
    printf("diffk transfer oracle: reflect/refract/free-space PASS (< 1e-5)\n");
    return 0;
}
