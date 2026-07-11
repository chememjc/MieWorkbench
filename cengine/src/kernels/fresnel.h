/* ===========================================================================
 * fresnel.h — complex-index Fresnel coefficients, Snell refraction, TIR.
 *
 * EXACT port of scripts/raytracer/fresnel.py (per-ray scalar form).
 * Conventions (fresnel.py header):
 *   d      : unit ray direction (INTO the interface)
 *   n_hat  : unit surface normal INTO the incident medium
 *            (cos_i = -d . n_hat >= 0)
 *   n1     : incident-medium index (real part governs geometry)
 *   n2     : complex far-medium index n + ik
 *   Born & Wolf / Hecht sign conventions:
 *     rs = (n1 ci - n2 ct) / (n1 ci + n2 ct)
 *     rp = (n2 ci - n1 ct) / (n2 ci + n1 ct)
 *     ts = 2 n1 ci / (n1 ci + n2 ct);  tp = 2 n1 ci / (n2 ci + n1 ct)
 *   with the complex branch Im(n2 ct) >= 0 (decaying transmitted wave):
 *   reproduces |r|=1 with the correct analytic TIR phase and handles
 *   metals without special cases.
 *   Power: R = |r|^2;  T = Re(n2 ct) / (n1 ci) * |t|^2.
 *   Lossless dielectrics: R + T = 1 to machine precision (pinned by
 *   test_kernels.py at 1e-12 — this port must meet the same gate).
 * =========================================================================== */
#ifndef MIEWB_FRESNEL_H
#define MIEWB_FRESNEL_H

#include "kmath.h"

typedef struct {
    kcplx rs, rp, ts, tp;   /* amplitude coefficients */
    kcplx ct;               /* complex cos(theta_t), physical branch */
    double Rs, Rp, Ts, Tp;  /* power coefficients */
} FresnelC;

/* fresnel.cos_theta_t (fresnel.py:29-45): physical-branch complex cosine. */
KFN kcplx fresnel_cos_theta_t(double cos_i, kcplx n1, kcplx n2) {
    double sin_i2 = 1.0 - cos_i * cos_i;
    kcplx ratio = kc_div(n1, n2);
    kcplx s2 = kc_scale(kc_mul(ratio, ratio), sin_i2);
    kcplx ct = kc_sqrt(kc_sub(kc(1.0, 0.0), s2));
    /* branch: Im(n2 * ct) >= 0 => transmitted wave decays into medium 2 */
    kcplx n2ct = kc_mul(n2, ct);
    if (n2ct.im < 0.0) ct = kc(-ct.re, -ct.im);
    return ct;
}

/* fresnel.fresnel_coeffs + power_coeffs in one call (they always travel
 * together in the tracer). */
KFN FresnelC fresnel_eval(double cos_i, kcplx n1, kcplx n2) {
    FresnelC f;
    f.ct = fresnel_cos_theta_t(cos_i, n1, n2);
    kcplx a1 = kc_scale(n1, cos_i);          /* n1 ci (s-pol) */
    kcplx a2 = kc_mul(n2, f.ct);             /* n2 ct */
    kcplx b1 = kc_scale(n2, cos_i);          /* n2 ci (p-pol) */
    kcplx b2 = kc_mul(n1, f.ct);             /* n1 ct */
    kcplx two_n1_ci = kc_scale(n1, 2.0 * cos_i);
    f.rs = kc_div(kc_sub(a1, a2), kc_add(a1, a2));
    f.rp = kc_div(kc_sub(b1, b2), kc_add(b1, b2));
    f.ts = kc_div(two_n1_ci, kc_add(a1, a2));
    f.tp = kc_div(two_n1_ci, kc_add(b1, b2));
    /* power_coeffs (fresnel.py:64-77): T carries the projected-Poynting
     * factor Re(n2 ct) / (n1_real ci) */
    double fac = kc_mul(n2, f.ct).re / (n1.re * cos_i);
    f.Rs = kc_abs2(f.rs);
    f.Rp = kc_abs2(f.rp);
    f.Ts = fac * kc_abs2(f.ts);
    f.Tp = fac * kc_abs2(f.tp);
    return f;
}

/* fresnel.is_tir (fresnel.py:80-85): real-index TIR test. */
KFN int fresnel_is_tir(double cos_i, kcplx n1, kcplx n2) {
    double sin_i2 = 1.0 - cos_i * cos_i;
    double ratio = n1.re / n2.re;
    return ratio * ratio * sin_i2 > 1.0;
}

/* fresnel.reflect_dir (fresnel.py:88-92). */
KFN kvec3 fresnel_reflect_dir(kvec3 d, kvec3 n_hat) {
    return v3_sub(d, v3_scale(n_hat, 2.0 * v3_dot(d, n_hat)));
}

/* fresnel.refract_dir (fresnel.py:95-111): vector Snell with REAL indices;
 * meaningless under TIR — caller masks with fresnel_is_tir / Ts+Tp. */
KFN kvec3 fresnel_refract_dir(kvec3 d, kvec3 n_hat, double cos_i,
                              double n1_re, double n2_re) {
    double eta = n1_re / n2_re;
    double ct2 = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    double ct = sqrt(ct2 > 0.0 ? ct2 : 0.0);
    kvec3 t = v3_add(v3_scale(d, eta),
                     v3_scale(n_hat, eta * cos_i - ct));
    return v3_unit(t);   /* defensive normalize, matches Python */
}

/* fresnel.pol_basis (fresnel.py:114-136): s_hat perpendicular to the plane
 * of incidence, deterministic fallback at normal incidence. */
KFN void fresnel_pol_basis(kvec3 d, kvec3 n_hat, kvec3 *s, kvec3 *p) {
    kvec3 sv = v3_cross(d, n_hat);
    double nrm = v3_norm(sv);
    if (nrm < 1e-12) {
        /* degenerate: global axis most orthogonal to d (numpy argmin |d|,
         * FIRST minimum), projected via the cross product */
        double ax = fabs(d.x), ay = fabs(d.y), az = fabs(d.z);
        kvec3 a = v3(0.0, 0.0, 0.0);
        if (ax <= ay && ax <= az)      a.x = 1.0;
        else if (ay <= az)             a.y = 1.0;
        else                           a.z = 1.0;
        sv = v3_cross(d, a);
        nrm = v3_norm(sv);
    }
    *s = v3_scale(sv, 1.0 / nrm);
    *p = v3_cross(d, *s);
}

/* fresnel.rotate_jones (fresnel.py:139-153): re-express (Es, Ep) in a new
 * orthonormal (s, p) basis about the SAME direction. Unitary. */
KFN void fresnel_rotate_jones(kcplx Es, kcplx Ep,
                              kvec3 s_old, kvec3 p_old,
                              kvec3 s_new, kvec3 p_new,
                              kcplx *Es2, kcplx *Ep2) {
    double css = v3_dot(s_new, s_old);
    double csp = v3_dot(s_new, p_old);
    double cps = v3_dot(p_new, s_old);
    double cpp = v3_dot(p_new, p_old);
    *Es2 = kc_add(kc_scale(Es, css), kc_scale(Ep, csp));
    *Ep2 = kc_add(kc_scale(Es, cps), kc_scale(Ep, cpp));
}

#endif /* MIEWB_FRESNEL_H */
