/* ===========================================================================
 * quartic.h — real roots of cubic/quartic polynomials.
 *
 * Used by the torus intersection (surf.h), replacing the Python engine's
 * batched companion-matrix eigenvalue solve (surfaces.py:318-331). The
 * Python engine polishes its eigenvalue roots with 2 Newton steps; the
 * same polish is applied by the caller here, so the closed-form solver
 * only needs bracket-quality accuracy. Parity is gated by the C-vs-Python
 * torus root test in test_cengine_parity.py / tests/test_quartic.c.
 *
 * Algorithm: Graphics Gems I, Jochen Schwarze, "Cubic and Quartic Roots"
 * (depressed quartic -> resolvent cubic -> two quadratics), with cbrt-
 * based trigonometric/Cardano cubic handling.
 * =========================================================================== */
#ifndef MIEWB_QUARTIC_H
#define MIEWB_QUARTIC_H

#include "kmath.h"

#define QUARTIC_EPS 1e-30

/* x^2 + p x + q = 0 -> up to 2 real roots; returns count */
KFN int k_solve_quadratic(double p, double q, double roots[2]) {
    double D = 0.25 * p * p - q;
    if (D < 0.0) return 0;
    double sq = sqrt(D);
    roots[0] = -0.5 * p - sq;
    roots[1] = -0.5 * p + sq;
    return 2;
}

/* x^3 + A x^2 + B x + C = 0 -> 1..3 real roots; returns count */
KFN int k_solve_cubic(double A, double B, double C, double roots[3]) {
    /* depressed: x = y - A/3 -> y^3 + 3py + 2q = 0 */
    double sq_A = A * A;
    double p = (1.0 / 3.0) * (-(1.0 / 3.0) * sq_A + B);
    double q = 0.5 * ((2.0 / 27.0) * A * sq_A - (1.0 / 3.0) * A * B + C);
    double cb_p = p * p * p;
    double D = q * q + cb_p;
    int n;
    if (fabs(D) < QUARTIC_EPS * QUARTIC_EPS) {
        if (fabs(q) < QUARTIC_EPS) {        /* triple root 0 */
            roots[0] = 0.0;
            n = 1;
        } else {                             /* one single + one double */
            double u = cbrt(-q);
            roots[0] = 2.0 * u;
            roots[1] = -u;
            n = 2;
        }
    } else if (D < 0.0) {                    /* three real (trig form) */
        double phi = (1.0 / 3.0) * acos(-q / sqrt(-cb_p));
        double t = 2.0 * sqrt(-p);
        roots[0] = t * cos(phi);
        roots[1] = -t * cos(phi + K_PI / 3.0);
        roots[2] = -t * cos(phi - K_PI / 3.0);
        n = 3;
    } else {                                 /* one real (Cardano) */
        double sqrt_D = sqrt(D);
        double u = cbrt(sqrt_D - q);
        double v = -cbrt(sqrt_D + q);
        roots[0] = u + v;
        n = 1;
    }
    double sub = (1.0 / 3.0) * A;
    for (int i = 0; i < n; i++) roots[i] -= sub;
    return n;
}

/* c4 x^4 + c3 x^3 + c2 x^2 + c1 x + c0 = 0 -> up to 4 real roots;
 * returns count. Degenerate c4 falls back to the cubic. */
KFN int k_solve_quartic(double c4, double c3, double c2, double c1,
                        double c0, double roots[4]) {
    if (fabs(c4) < QUARTIC_EPS) {
        if (fabs(c3) < QUARTIC_EPS) {
            if (fabs(c2) < QUARTIC_EPS) {
                if (fabs(c1) < QUARTIC_EPS) return 0;
                roots[0] = -c0 / c1;
                return 1;
            }
            return k_solve_quadratic(c1 / c2, c0 / c2, roots);
        }
        return k_solve_cubic(c2 / c3, c1 / c3, c0 / c3, roots);
    }
    double A = c3 / c4, B = c2 / c4, C = c1 / c4, D = c0 / c4;
    /* depressed: x = y - A/4 -> y^4 + p y^2 + q y + r = 0 */
    double sq_A = A * A;
    double p = -3.0 / 8.0 * sq_A + B;
    double q = 0.125 * sq_A * A - 0.5 * A * B + C;
    double r = -3.0 / 256.0 * sq_A * sq_A + 1.0 / 16.0 * sq_A * B
               - 0.25 * A * C + D;
    int n = 0;
    if (fabs(r) < QUARTIC_EPS) {
        /* no absolute term: y (y^3 + p y + q) = 0 */
        n = k_solve_cubic(0.0, p, q, roots);
        roots[n++] = 0.0;
    } else {
        /* resolvent cubic: z^3 - p/2 z^2 - r z + (r p / 2 - q^2 / 8) = 0.
         * A real quartic ALWAYS factors into two real quadratics, but only
         * through a cubic root z with u = z^2 - r >= 0 AND v = 2z - p >= 0
         * — picking an arbitrary real root silently drops every root of
         * ~1/3 of torus rays (found by the torus parity test). Choose the
         * real root maximizing min(u, v), clamping scale-relative rounding
         * negatives to zero. */
        double zr[3];
        int nz = k_solve_cubic(-0.5 * p, -r, 0.5 * r * p - 0.125 * q * q,
                               zr);
        double z = zr[0], u = 0.0, v = 0.0, best = -INFINITY;
        for (int i = 0; i < nz; i++) {
            double ui = zr[i] * zr[i] - r;
            double vi = 2.0 * zr[i] - p;
            double m = ui < vi ? ui : vi;
            if (m > best) { best = m; z = zr[i]; u = ui; v = vi; }
        }
        double scale = fabs(z * z) + fabs(r) + fabs(p) + QUARTIC_EPS;
        if (u < 0.0) {
            if (u > -1e-9 * scale) u = 0.0; else return 0;
        }
        if (v < 0.0) {
            if (v > -1e-9 * scale) v = 0.0; else return 0;
        }
        u = sqrt(u);
        v = sqrt(v);
        /* y^2 + (q<0 ? -v : v) y + (z - u) and the sign-flipped partner */
        double sv = (q < 0.0) ? -v : v;
        n = k_solve_quadratic(sv, z - u, roots);
        n += k_solve_quadratic(-sv, z + u, roots + n);
    }
    double sub = 0.25 * A;
    for (int i = 0; i < n; i++) roots[i] -= sub;
    return n;
}

#endif /* MIEWB_QUARTIC_H */
