/* ===========================================================================
 * trim.h — trimmed-face containment queries in canonical UV.
 *
 * EXACT port of surfaces.TrimPolygon (scripts/raytracer/surfaces.py:675-798).
 * Three regimes, chosen at CONSTRUCTION time (host side, scene.c —
 * trim_build there ports __init__/_full_primitive_area/_choose_band):
 *
 *   TRIM_UNTRIMMED — full closed primitive (complete sphere/torus): always
 *                    contained.
 *   TRIM_BAND      — a wire encircles the periodic u axis: containment is
 *                    a v-range test (+ optional polygon holes).
 *   TRIM_POLYGON   — generic even-odd crossing test on the unwrapped-u
 *                    polygon; periodic surfaces test u + k*2pi, k in
 *                    {-1, 0, +1}.
 *
 * The QUERY side lives here (KFN — needed on the GPU by the phase-D
 * occlusion kernel); loop data is stored flattened (SoA offsets) so the
 * same struct uploads to the device untouched.
 * =========================================================================== */
#ifndef MIEWB_TRIM_H
#define MIEWB_TRIM_H

#include "kmath.h"

enum {
    TRIM_UNTRIMMED = 0,
    TRIM_BAND = 1,
    TRIM_POLYGON = 2,
};

typedef struct {
    uint8_t mode;
    uint8_t periodic;       /* surface.periodic_u (query shifts +-2pi) */
    double v_lo, v_hi;      /* BAND regime bounds */
    /* Flattened loops (unwrapped u, v). In POLYGON mode these are ALL
     * loops; in BAND mode only the non-winding hole loops. */
    int32_t n_loops;
    const int32_t *loop_off;    /* n_loops + 1 offsets into pts_u/pts_v */
    const double *pts_u;
    const double *pts_v;
} TrimC;

/* Even-odd crossing test for one closed loop — port of
 * TrimPolygon._in_loop (surfaces.py:760-775). Edge list wraps last->first
 * exactly like np.roll(-1). */
KFN int trim_in_loop(const TrimC *t, int32_t loop, double u, double v) {
    int32_t lo = t->loop_off[loop];
    int32_t hi = t->loop_off[loop + 1];
    int inside = 0;
    for (int32_t i = lo; i < hi; i++) {
        int32_t j = (i + 1 < hi) ? i + 1 : lo;   /* wrap = np.roll(-1) */
        double y1 = t->pts_v[i], y2 = t->pts_v[j];
        if ((y1 > v) != (y2 > v)) {
            double x1 = t->pts_u[i], x2 = t->pts_u[j];
            double xint = x1 + (v - y1) / (y2 - y1) * (x2 - x1);
            if (u < xint) inside = !inside;
        }
    }
    return inside;
}

/* Port of _in_polygon (surfaces.py:777-785): XOR across loops per shift,
 * OR across the periodic shifts. */
KFN int trim_in_polygon(const TrimC *t, double u, double v) {
    double shifts[3];
    int n_shift = 1;
    shifts[0] = 0.0;
    if (t->periodic) {
        shifts[1] = K_TWO_PI;
        shifts[2] = -K_TWO_PI;
        n_shift = 3;
    }
    for (int k = 0; k < n_shift; k++) {
        int acc = 0;
        for (int32_t l = 0; l < t->n_loops; l++)
            acc ^= trim_in_loop(t, l, u + shifts[k], v);
        if (acc) return 1;
    }
    return 0;
}

/* Port of TrimPolygon.contains (surfaces.py:787-798). */
KFN int trim_contains(const TrimC *t, double u, double v) {
    if (t->mode == TRIM_UNTRIMMED) return 1;
    if (t->mode == TRIM_BAND) {
        if (v < t->v_lo - 1e-9 || v > t->v_hi + 1e-9) return 0;
        if (t->n_loops > 0 && trim_in_polygon(t, u, v)) return 0; /* holes */
        return 1;
    }
    return trim_in_polygon(t, u, v);
}

#endif /* MIEWB_TRIM_H */
