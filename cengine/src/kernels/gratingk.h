/* ===========================================================================
 * gratingk.h — vector grating equation + Kogelnik efficiency (per-ray).
 *
 * Port of grating.py order_directions / bragg_kogelnik_eta /
 * _kogelnik_nu_xi. The lambda-only efficiency models (lamellar, dammann,
 * table) are pre-resolved by the Python glue through the SAME
 * order_efficiencies code into per-(order, lam_idx) tables — only the
 * angle-dependent Kogelnik model is evaluated here.
 * =========================================================================== */
#ifndef MIEWB_GRATINGK_H
#define MIEWB_GRATINGK_H

#include "kmath.h"

/* One order's directions (grating.py:57-105, scalar). Returns propagation
 * flags for the transmitted/reflected sides. */
KFN void grating_order_dirs(kvec3 d, kvec3 n_hat, kvec3 g_hat,
                            double lines_per_mm, double lam, int m,
                            double n1, double n2,
                            kvec3 *dir_t, int *prop_t,
                            kvec3 *dir_r, int *prop_r) {
    double cos_i = -v3_dot(d, n_hat);
    kvec3 d_tang = v3_fma(d, cos_i, n_hat);
    double d_period = 1e-3 / lines_per_mm;
    double lam_over_d = lam / d_period;
    kvec3 T = v3_add(v3_scale(d_tang, n1),
                     v3_scale(g_hat, (double)m * lam_over_d));
    double Tmag2 = v3_dot(T, T);

    double rad_t = n2 * n2 - Tmag2;
    *prop_t = rad_t >= 0.0;
    double nz_t = sqrt(rad_t > 0.0 ? rad_t : 0.0);
    *dir_t = v3_unit(v3_scale(v3_sub(T, v3_scale(n_hat, nz_t)), 1.0 / n2));

    double rad_r = n1 * n1 - Tmag2;
    *prop_r = rad_r >= 0.0;
    double nz_r = sqrt(rad_r > 0.0 ? rad_r : 0.0);
    *dir_r = v3_unit(v3_scale(v3_add(T, v3_scale(n_hat, nz_r)), 1.0 / n1));
}

/* Kogelnik transmission-VBG first-order efficiency
 * (grating.py:184-223): returns (eta_s, eta_p) for order +1; order 0
 * carries 1 - eta (lossless dielectric). thin-boundary n ~ 1. */
KFN void kogelnik_eta(double thickness_m, double dn, double slant_rad,
                      double lines_per_mm, double lam, double cos_i,
                      double *eta_s, double *eta_p) {
    double period = 1e-3 / lines_per_mm;
    double ci = cos_i;
    if (ci > 1.0) ci = 1.0;
    if (ci < -1.0) ci = -1.0;
    double theta = acos(ci);
    double Kv = K_TWO_PI / period;
    double beta = K_TWO_PI / lam;
    double phi_g = 0.5 * K_PI - slant_rad;

    double c_R = cos(theta);
    double c_S = cos(theta) - (Kv / beta) * cos(phi_g);
    double vartheta = Kv * cos(phi_g - theta) - Kv * Kv / (2.0 * beta);

    if (c_S <= 1e-12) {         /* diffracted order evanescent */
        *eta_s = 0.0;
        *eta_p = 0.0;
        return;
    }
    double nu_s = K_PI * dn * thickness_m / (lam * sqrt(c_R * c_S));
    double nu_p = nu_s * fabs(cos(2.0 * (theta - slant_rad)));
    double xi = vartheta * thickness_m / (2.0 * c_S);

    /* bragg_kogelnik_eta: sin^2(sqrt(nu^2+xi^2)) nu^2/(nu^2+xi^2) */
    double as = nu_s * nu_s + xi * xi;
    double es = as > 0.0 ? sin(sqrt(as)) * sin(sqrt(as)) * nu_s * nu_s / as
                         : 0.0;
    double ap = nu_p * nu_p + xi * xi;
    double ep = ap > 0.0 ? sin(sqrt(ap)) * sin(sqrt(ap)) * nu_p * nu_p / ap
                         : 0.0;
    if (es < 0.0) es = 0.0;
    if (es > 1.0) es = 1.0;
    if (ep < 0.0) ep = 0.0;
    if (ep > 1.0) ep = 1.0;
    *eta_s = es;
    *eta_p = ep;
}

#endif /* MIEWB_GRATINGK_H */
