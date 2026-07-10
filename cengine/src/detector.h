/* ===========================================================================
 * detector.h — planar detector grids: trim mask, incoherent bilinear splat,
 * per-key detected tallies.
 *
 * Port of detector.py DetectorGrid + _IncoherentAccumMixin. The grid
 * GEOMETRY (xhat/yhat/x_lo/pixel_m/W/H) arrives pre-computed from the
 * Python glue (bit-identical basis); the mask and splat are computed here.
 *
 * Threading model: the parallel trace records DetHit entries per thread;
 * det_apply_hits replays them into the float64 cube SERIALLY (in thread
 * order) after each batch — deterministic accumulation without per-thread
 * cube copies (a 2048^2 x 16-bin cube is 512 MB; 32 copies would not fit).
 * =========================================================================== */
#ifndef MIEWB_DETECTOR_H
#define MIEWB_DETECTOR_H

#include "scene.h"

/* one incoherent detector arrival (grid coords precomputed in parallel) */
typedef struct {
    int32_t det;
    float fx, fy;           /* fractional pixel coords from det_to_grid */
    int32_t bin;
    double power;
    int16_t source_id, lam_stratum, pol_stratum;
} DetHit;

typedef struct {
    DetHit *v;
    int64_t n, cap;
} DetHitVec;

void dethits_init(DetHitVec *h);
void dethits_free(DetHitVec *h);
void dethits_push(DetHitVec *h, const DetHit *hit);
void dethits_clear(DetHitVec *h);

/* detector.py to_grid (detector.py:167-171) */
static inline void det_to_grid(const DetC *d, kvec3 p, double *fx,
                               double *fy) {
    *fx = (v3_dot(p, d->xhat) - d->x_lo) / d->pixel_m;
    *fy = (v3_dot(p, d->yhat) - d->y_lo) / d->pixel_m;
}

/* detector.py lam_bin (detector.py:39-42) */
static inline int det_lam_bin(const DetC *d, double lam) {
    double span = d->lam_hi - d->lam_lo;
    if (span < 1e-30) span = 1e-30;
    int b = (int)((lam - d->lam_lo) / span * d->spectral_bins);
    if (b < 0) b = 0;
    if (b >= d->spectral_bins) b = d->spectral_bins - 1;
    return b;
}

/* Compute the trim mask over pixel centers — port of the mask block in
 * DetectorGrid.__init__ (detector.py:147-164). */
void det_compute_mask(DetC *d, const SceneC *s);

/* Serially replay recorded hits: bilinear splat into d->inc (the exact
 * math of deposit_incoherent, detector.py:44-69) plus the per-key
 * detected_incoherent tallies (detector.py:70-85). */
void det_apply_hits(SceneC *s, const DetHitVec *hits);

/* Write det_<i>_inc.npy / det_<i>_mask.npy into out_dir. */
void det_write_outputs(const SceneC *s);

#endif /* MIEWB_DETECTOR_H */
