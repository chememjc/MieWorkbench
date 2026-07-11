/* test_fresnel_gold.c — Fresnel kernel invariants, mirroring
 * scripts/raytracer/tests/test_kernels.py at the same tolerances:
 *   - lossless dielectric R + T = 1 to 1e-12 across incidence angles
 *   - normal incidence R = ((n1-n2)/(n1+n2))^2
 *   - Brewster: Rp < 1e-20 at theta_B = atan(n2/n1)
 *   - TIR: |rs| = |rp| = 1 beyond the critical angle, with the analytic
 *     Fresnel-rhomb phase difference
 *   - refract_dir obeys Snell (n1 sin_i = n2 sin_t) to 1e-12
 */
#include <stdio.h>
#include <stdlib.h>
#include "kernels/fresnel.h"

static int failures = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void check_close(double a, double b, double tol, const char *what) {
    if (!(fabs(a - b) <= tol)) {
        fprintf(stderr, "FAIL: %s — %.17g vs %.17g (tol %g)\n", what, a, b,
                tol);
        failures++;
    }
}

int main(void) {
    /* ---- R + T = 1, air -> BK7-ish glass, sweep of angles ---- */
    kcplx n1 = kc(1.0, 0.0), n2 = kc(1.5168, 0.0);
    for (int i = 0; i < 89; i++) {
        double th = (i + 0.5) * K_PI / 180.0;
        FresnelC f = fresnel_eval(cos(th), n1, n2);
        check_close(f.Rs + f.Ts, 1.0, 1e-12, "Rs+Ts=1");
        check_close(f.Rp + f.Tp, 1.0, 1e-12, "Rp+Tp=1");
    }

    /* ---- normal incidence ---- */
    {
        FresnelC f = fresnel_eval(1.0, n1, n2);
        double r0 = (1.0 - 1.5168) / (1.0 + 1.5168);
        check_close(f.Rs, r0 * r0, 1e-14, "normal-incidence Rs");
        check_close(f.Rp, r0 * r0, 1e-14, "normal-incidence Rp");
    }

    /* ---- Brewster ---- */
    {
        double thB = atan(1.5168);
        FresnelC f = fresnel_eval(cos(thB), n1, n2);
        check(f.Rp < 1e-20, "Brewster Rp ~ 0");
        check(f.Rs > 0.1, "Brewster Rs finite");
    }

    /* ---- TIR magnitude + Fresnel-rhomb phase (glass -> air) ----
     * Analytic phase difference (Hecht):
     *   tan(delta/2) = cos_i sqrt(sin_i^2 - n^2) / sin_i^2, n = n2/n1.
     * The engine's rs/rp phases must reproduce delta = phi_s - phi_p
     * (sign fixed by the B&W convention used in fresnel.py; the Python
     * test pins the magnitude). */
    {
        kcplx g = kc(1.51, 0.0), air = kc(1.0, 0.0);
        double th = 54.0 * K_PI / 180.0;   /* past critical (41.5 deg) */
        FresnelC f = fresnel_eval(cos(th), g, air);
        check_close(kc_abs(f.rs), 1.0, 1e-12, "TIR |rs| = 1");
        check_close(kc_abs(f.rp), 1.0, 1e-12, "TIR |rp| = 1");
        double n = 1.0 / 1.51;
        double si = sin(th), ci = cos(th);
        double delta = 2.0 * atan(ci * sqrt(si * si - n * n) / (si * si));
        double dphi = kc_arg(f.rs) - kc_arg(f.rp);
        /* wrap to (-pi, pi]; with the B&W convention of fresnel.py the
         * s-p phase difference IS the rhomb delta directly */
        while (dphi > K_PI) dphi -= K_TWO_PI;
        while (dphi <= -K_PI) dphi += K_TWO_PI;
        check_close(fabs(dphi), delta, 1e-10, "Fresnel rhomb TIR phase");
    }

    /* ---- vector Snell ---- */
    {
        kvec3 n_hat = v3(0.0, 0.0, 1.0);   /* toward incident medium */
        double th = 33.0 * K_PI / 180.0;
        kvec3 d = v3(sin(th), 0.0, -cos(th));
        kvec3 t = fresnel_refract_dir(d, n_hat, cos(th), 1.0, 1.5168);
        double sin_t = sqrt(t.x * t.x + t.y * t.y);
        check_close(1.0 * sin(th), 1.5168 * sin_t, 1e-12, "Snell invariant");
        check_close(v3_norm(t), 1.0, 1e-12, "refract_dir unit norm");
    }

    /* ---- absorbing metal branch sanity: R < 1, T accounts the rest via
     * the projected-Poynting factor (energy conservation is NOT R+T=1 for
     * absorbing n2 — just bounds) ---- */
    {
        kcplx au = kc(0.18, 3.0);          /* gold-ish at visible */
        FresnelC f = fresnel_eval(cos(0.3), n1, au);
        check(f.Rs > 0.8 && f.Rs < 1.0, "metal Rs plausible");
        check(f.Rp > 0.8 && f.Rp < 1.0, "metal Rp plausible");
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("fresnel_gold: all checks passed\n");
    return 0;
}
