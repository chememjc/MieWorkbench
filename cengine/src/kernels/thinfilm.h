/* ===========================================================================
 * thinfilm.h — multilayer thin-film coatings, characteristic-matrix (TMM)
 * method, Macleod formulation with tilted admittances.
 *
 * EXACT port of scripts/raytracer/thinfilm.py (see its header for the
 * formulas). Per-ray scalar form; layer indices n_j arrive PRE-RESOLVED at
 * the ray's stratum wavelength (plan D1) as a flat array. Zero layers
 * degenerates exactly to the bare Fresnel interface (pinned by
 * test_kernels.py, mirrored in tests/test_tmm_gold.c at 1e-12).
 * =========================================================================== */
#ifndef MIEWB_THINFILM_H
#define MIEWB_THINFILM_H

#include "kmath.h"

typedef struct {
    kcplx rs, rp, ts, tp;
    double Rs, Rp, Ts, Tp;
} TmmC;

/* n_j cos(theta_j) = sqrt(n_j^2 - beta^2), branch Im >= 0
 * (thinfilm._n_cos, thinfilm.py:26-29) */
KFN kcplx tmm_n_cos(kcplx n, double beta) {
    kcplx v = kc_sqrt(kc_sub(kc_mul(n, n), kc(beta * beta, 0.0)));
    if (v.im < 0.0) v = kc(-v.re, -v.im);
    return v;
}

/* One polarization's [B, C] recursion + r/t (thinfilm.py:60-83).
 * layer_n: m complex indices at THIS ray's wavelength, incident side
 * first; layer_d: m thicknesses [m]. pol_p selects the p admittances. */
KFN void tmm_one_pol(double lam, double beta, kcplx eta0, kcplx eta_sub,
                     const kcplx *layer_n, const double *layer_d, int m,
                     int pol_p, kcplx *r_out, kcplx *t_out,
                     kcplx *eta_sub_out) {
    kcplx B = kc(1.0, 0.0);
    kcplx C = eta_sub;
    /* multiply from the substrate side outward (reverse layer order) —
     * identical to the Python loop over reversed(layer_n) */
    for (int j = m - 1; j >= 0; j--) {
        kcplx nc = tmm_n_cos(layer_n[j], beta);
        kcplx eta_j = pol_p
            ? kc_div(kc_mul(layer_n[j], layer_n[j]), nc)
            : nc;
        /* delta = 2 pi d (n cos) / lam — complex for absorbing layers */
        kcplx delta = kc_scale(nc, K_TWO_PI * layer_d[j] / lam);
        /* cos/sin of a complex angle: cos(a+bi), sin(a+bi) */
        double ca = cos(delta.re), sa = sin(delta.re);
        double ch = cosh(delta.im), sh = sinh(delta.im);
        kcplx cd = kc(ca * ch, -sa * sh);
        kcplx sd = kc(sa * ch, ca * sh);
        kcplx i_sd = kc(-sd.im, sd.re);               /* i * sin(delta) */
        kcplx B2 = kc_add(kc_mul(cd, B), kc_mul(kc_div(i_sd, eta_j), C));
        kcplx C2 = kc_add(kc_mul(kc_mul(i_sd, eta_j), B), kc_mul(cd, C));
        B = B2;
        C = C2;
    }
    kcplx denom = kc_add(kc_mul(eta0, B), C);
    *r_out = kc_div(kc_sub(kc_mul(eta0, B), C), denom);
    *t_out = kc_div(kc_scale(eta0, 2.0), denom);
    *eta_sub_out = eta_sub;
}

/* Full tmm_coeffs + tmm_power (thinfilm.py:32-102) for one ray. */
KFN TmmC tmm_eval(double lam, double cos_i, kcplx n_in, kcplx n_sub,
                  const kcplx *layer_n, const double *layer_d, int m) {
    double n0 = n_in.re;                    /* real part governs geometry */
    double s2 = 1.0 - cos_i * cos_i;
    double beta = n0 * sqrt(s2 > 0.0 ? s2 : 0.0);

    kcplx eta0_s = kc(n0 * cos_i, 0.0);
    double nci = n0 * cos_i;
    kcplx eta0_p = kc(n0 * n0 / (nci > 1e-300 ? nci : 1e-300), 0.0);

    kcplx nc_sub = tmm_n_cos(n_sub, beta);
    kcplx eta_sub_s = nc_sub;
    kcplx eta_sub_p = kc_div(kc_mul(n_sub, n_sub), nc_sub);

    TmmC out;
    kcplx es, ep;
    tmm_one_pol(lam, beta, eta0_s, eta_sub_s, layer_n, layer_d, m, 0,
                &out.rs, &out.ts, &es);
    tmm_one_pol(lam, beta, eta0_p, eta_sub_p, layer_n, layer_d, m, 1,
                &out.rp, &out.tp, &ep);
    /* Macleod rp sign is opposite the Born & Wolf convention fresnel.py
     * uses — flip so zero layers reproduce fresnel_coeffs exactly,
     * phases included (thinfilm.py:86-90) */
    out.rp = kc(-out.rp.re, -out.rp.im);
    out.Rs = kc_abs2(out.rs);
    out.Rp = kc_abs2(out.rp);
    out.Ts = es.re / eta0_s.re * kc_abs2(out.ts);
    out.Tp = ep.re / eta0_p.re * kc_abs2(out.tp);
    return out;
}

#endif /* MIEWB_THINFILM_H */
