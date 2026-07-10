/* ===========================================================================
 * birefk.h — uniaxial-crystal double refraction, per-ray scalar port of
 * scripts/raytracer/birefringence.py (see its header for the model and the
 * o/e conventions; all functions here mirror the numpy versions name-for-name
 * and are pinned by the calcite/quartz parity tests).
 *
 * Biaxial (two-sheet quartic) is NOT ported — scenes using biaxial
 * materials are feature-routed to the Python engine.
 * =========================================================================== */
#ifndef MIEWB_BIREFK_H
#define MIEWB_BIREFK_H

#include "kmath.h"
#include "fresnel.h"

#define BIREF_DEGEN 1e-9    /* |k x c| below this: o/e degenerate */

/* extraordinary phase index at cos(k,c) (birefringence.n_e_theta) */
KFN double biref_n_e_theta(double cos_kc, double n_o, double n_e) {
    double c2 = cos_kc * cos_kc;
    double inv = c2 / (n_o * n_o) + (1.0 - c2) / (n_e * n_e);
    return 1.0 / sqrt(inv);
}

/* D-field eigenbasis transverse to k (birefringence.eigenbasis):
 * e_o ~ k x c (ordinary D), e_e = k x e_o. Deterministic fallback at
 * k || c via the global axis most orthogonal to k. */
KFN void biref_eigenbasis(kvec3 k, kvec3 c, kvec3 *eo, kvec3 *ee) {
    kvec3 v = v3_cross(k, c);
    double nrm = v3_norm(v);
    if (nrm < BIREF_DEGEN) {
        double ax = fabs(k.x), ay = fabs(k.y), az = fabs(k.z);
        kvec3 a = v3(0.0, 0.0, 0.0);
        if (ax <= ay && ax <= az)      a.x = 1.0;
        else if (ay <= az)             a.y = 1.0;
        else                           a.z = 1.0;
        v = v3_cross(k, a);
        nrm = v3_norm(v);
    }
    *eo = v3_scale(v, 1.0 / nrm);
    *ee = v3_cross(k, *eo);
}

/* refract_in (birefringence.py:176-243), scalar. Outputs the ordinary
 * and extraordinary wavevectors, the e-ray (Poynting) direction, phase
 * indices and TIR flags. */
typedef struct {
    kvec3 k_o, k_e, s_e;
    double n_phase_e, n_ray_e;
    int tir_o, tir_e;
} BirefIn;

KFN BirefIn biref_refract_in(kvec3 d, kvec3 nh, kvec3 c, double n1,
                             double n_o, double n_e) {
    BirefIn r;
    double cos_i = -v3_dot(d, nh);
    /* ordinary: plain Snell with n_o */
    r.k_o = fresnel_refract_dir(d, nh, cos_i, n1, n_o);
    r.tir_o = fresnel_is_tir(cos_i, kc(n1, 0.0), kc(n_o, 0.0));
    /* extraordinary: tangential continuity + e normal surface */
    kvec3 t_vec = v3_scale(v3_sub(d, v3_scale(nh, v3_dot(d, nh))), n1);
    double A = v3_dot(t_vec, t_vec);
    double p = v3_dot(t_vec, c);
    double q = v3_dot(nh, c);
    double kappa = n_e * n_e / (n_o * n_o) - 1.0;
    double a = 1.0 + kappa * q * q;
    double b = 2.0 * kappa * p * q;
    double cc = A + kappa * p * p - n_e * n_e;
    double disc = b * b - 4.0 * a * cc;
    r.tir_e = disc < 0.0;
    double sq = sqrt(disc > 0.0 ? disc : 0.0);
    double sroot = (-b - sq) / (2.0 * a);          /* inward branch */
    kvec3 K_e = v3_fma(t_vec, sroot, nh);
    r.n_phase_e = v3_norm(K_e);
    r.k_e = v3_scale(K_e, 1.0 / r.n_phase_e);
    /* e-ray direction = grad_K of the dispersion relation */
    double Kc = v3_dot(K_e, c);
    kvec3 grad = v3_add(v3_scale(K_e, 1.0 / (n_e * n_e)),
                        v3_scale(c, Kc * (1.0 / (n_o * n_o)
                                          - 1.0 / (n_e * n_e))));
    r.s_e = v3_unit(grad);
    double cos_rho = v3_dot(r.k_e, r.s_e);
    r.n_ray_e = r.n_phase_e * cos_rho;
    return r;
}

/* ray_from_k (birefringence.py:246-265): (s_ray, n_phase, n_ray) for an
 * internal e-wavevector. */
KFN void biref_ray_from_k(kvec3 kh, kvec3 c, double n_o, double n_e,
                          kvec3 *s_ray, double *n_phase, double *n_ray) {
    double cos_kc = v3_dot(kh, c);
    double np_ = biref_n_e_theta(cos_kc, n_o, n_e);
    kvec3 K = v3_scale(kh, np_);
    kvec3 grad = v3_add(v3_scale(K, 1.0 / (n_e * n_e)),
                        v3_scale(c, v3_dot(K, c)
                                 * (1.0 / (n_o * n_o)
                                    - 1.0 / (n_e * n_e))));
    *s_ray = v3_unit(grad);
    *n_phase = np_;
    *n_ray = np_ * v3_dot(kh, *s_ray);
}

/* k_from_ray (birefringence.py:268-281): invert the e Poynting map. */
KFN kvec3 biref_k_from_ray(kvec3 s, kvec3 c, double n_o, double n_e) {
    kvec3 K = v3_add(v3_scale(s, n_e * n_e),
                     v3_scale(c, (n_o * n_o - n_e * n_e) * v3_dot(s, c)));
    return v3_unit(K);
}

/* refract_out (birefringence.py:284-316): internal wave -> isotropic n2
 * via wavevector tangential continuity. */
KFN kvec3 biref_refract_out(kvec3 kh, int mode_is_e, kvec3 nh, kvec3 c,
                            double n_o, double n_e, double n2, int *tir) {
    double cos_kc = v3_dot(kh, c);
    double n_phase = mode_is_e ? biref_n_e_theta(cos_kc, n_o, n_e) : n_o;
    kvec3 K = v3_scale(kh, n_phase);
    kvec3 K_t = v3_sub(K, v3_scale(nh, v3_dot(K, nh)));
    double Kt2 = v3_dot(K_t, K_t);
    double s2 = n2 * n2 - Kt2;
    *tir = s2 < 0.0;
    double sroot = -sqrt(s2 > 0.0 ? s2 : 0.0);   /* outgoing side */
    return v3_unit(v3_fma(K_t, sroot, nh));
}

#endif /* MIEWB_BIREFK_H */
