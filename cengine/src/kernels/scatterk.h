/* ===========================================================================
 * scatterk.h — stochastic surface-scatter samplers: Beckmann microfacets
 * (roughness.py) and the ABg / Harvey-Shack reflected lobe (scatter.py).
 * Per-ray scalar, all randomness through the lineage-keyed Philox RNG
 * (rng.h) so results stay thread-count invariant.
 * =========================================================================== */
#ifndef MIEWB_SCATTERK_H
#define MIEWB_SCATTERK_H

#include "kmath.h"
#include "../rng.h"

/* Davies / Bennett-Porteus specular retention (roughness.py:12-29):
 * A = exp(-(4 pi sigma cos_i / lam)^2) */
KFN double rough_specular_factor(double sigma_m, double cos_i, double lam) {
    double x = 4.0 * K_PI * sigma_m * cos_i / lam;
    return exp(-(x * x));
}

/* Deterministic tangent frame (roughness._tangent_frame — same argmin
 * convention as k_plane_frame but ax x n order). */
KFN void rough_tangent_frame(kvec3 n, kvec3 *t1, kvec3 *t2) {
    double axc = fabs(n.x), ayc = fabs(n.y), azc = fabs(n.z);
    kvec3 a = v3(0.0, 0.0, 0.0);
    if (axc <= ayc && axc <= azc)      a.x = 1.0;
    else if (ayc <= azc)               a.y = 1.0;
    else                               a.z = 1.0;
    *t1 = v3_unit(v3_cross(a, n));
    *t2 = v3_cross(n, *t1);
}

/* One Beckmann microfacet normal (roughness.beckmann_sample, scalar):
 * tan^2(theta) = -slope^2 ln(U) (Exponential inversion), tilts > 89 deg
 * rejected and redrawn; phi uniform. `draw_base` separates the draw
 * streams of multiple lobes at one interaction. */
KFN kvec3 beckmann_facet(kvec3 n_hat, double slope, uint64_t ray_key,
                         uint32_t event, uint32_t draw_base) {
    const double tan2_max = 3.28051445350672486e3;  /* tan^2(89 deg) */
    double tan2 = 0.0;
    for (uint32_t a = 0; a < 60; a++) {
        double u = rng_uniform(ray_key, event, draw_base + a);
        if (u < 1e-300) u = 1e-300;
        tan2 = -(slope * slope) * log(u);
        if (tan2 <= tan2_max) break;
        if (a == 59) tan2 = tan2_max;   /* negligible-probability tail */
    }
    double phi = K_TWO_PI * rng_uniform(ray_key, event, draw_base + 63);
    double theta = atan(sqrt(tan2));
    double st = sin(theta), ct = cos(theta);
    kvec3 t1, t2;
    rough_tangent_frame(n_hat, &t1, &t2);
    kvec3 m = v3_scale(n_hat, ct);
    m = v3_fma(m, st * cos(phi), t1);
    m = v3_fma(m, st * sin(phi), t2);
    return v3_unit(m);
}

/* ABg total integrated scatter, g == 2 closed form (scatter.abg_tis /
 * _radial_tis_umax): TIS = pi A ln(1 + umax^2/B), umax = 1 - sin(theta_i).
 * g != 2 is feature-routed to the Python engine. */
KFN double abg_tis_g2(double A, double B, double cos_i) {
    double s2 = 1.0 - cos_i * cos_i;
    if (s2 < 0.0) s2 = 0.0;
    if (s2 > 1.0) s2 = 1.0;
    double beta0 = sqrt(s2);
    double umax = 1.0 - beta0;
    if (umax <= 0.0) return 0.0;
    return K_PI * A * log1p(umax * umax / B);
}

/* One ABg scatter direction (scatter.sample_abg, scalar, g == 2):
 * radial inverse CDF u = sqrt(B ((1 + umax^2/B)^r - 1)), uniform azimuth,
 * horizon fold-back. d_spec: specular reflected direction. */
KFN kvec3 abg_sample_g2(double A, double B, kvec3 d_spec, kvec3 n_hat,
                        uint64_t ray_key, uint32_t event,
                        uint32_t draw_base) {
    (void)A;    /* the normalized radial CDF is A-independent */
    double dn = v3_dot(d_spec, n_hat);
    kvec3 t_spec = v3_sub(d_spec, v3_scale(n_hat, dn));
    double beta0 = v3_norm(t_spec);
    double umax = 1.0 - beta0;
    if (umax < 0.0) umax = 0.0;
    double u = 0.0;
    if (umax > 0.0) {
        double r = rng_uniform(ray_key, event, draw_base);
        u = sqrt(B * (pow(1.0 + umax * umax / B, r) - 1.0));
    }
    double psi = K_TWO_PI * rng_uniform(ray_key, event, draw_base + 1);
    kvec3 t1, t2;
    rough_tangent_frame(n_hat, &t1, &t2);
    kvec3 beta_vec = t_spec;
    beta_vec = v3_fma(beta_vec, u * cos(psi), t1);
    beta_vec = v3_fma(beta_vec, u * sin(psi), t2);
    double beta2 = v3_dot(beta_vec, beta_vec);
    if (beta2 > 1.0) {          /* fold float-edge overshoot back */
        beta_vec = v3_scale(beta_vec, sqrt(1.0 - 1e-15) / sqrt(beta2));
        beta2 = 1.0 - 1e-15;
    }
    double w = sqrt(1.0 - beta2 > 0.0 ? 1.0 - beta2 : 0.0);
    return v3_unit(v3_fma(beta_vec, w, n_hat));
}

#endif /* MIEWB_SCATTERK_H */
