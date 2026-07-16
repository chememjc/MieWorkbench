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

/* one coherent detector arrival: the full Huygens sample record
 * (tracer._detector_event -> add_gather_samples, segment-START state) */
typedef struct {
    int32_t det;
    int16_t source_id, lam_stratum, pol_stratum;
    kvec3 pos, dir, s_hat;
    kcplx Es, Ep;
    double lam, opl, power;
    uint8_t scattered;
    uint64_t ray_key;
    uint32_t event_ctr;         /* P1 canonical-sort tiebreaker */
} GatherHit;

typedef struct {
    GatherHit *v;
    int64_t n, cap;
} GatherHitVec;

void gathhits_init(GatherHitVec *h);
void gathhits_free(GatherHitVec *h);
void gathhits_push(GatherHitVec *h, const GatherHit *hit);
void gathhits_clear(GatherHitVec *h);

/* Serially file recorded coherent hits into the per-(source,stratum,pol)
 * GKey sample sets on their detectors + the detected_geometric tallies. */
void det_apply_gather_hits(SceneC *s, const GatherHitVec *hits);
void det_free_gkeys(DetC *d);

/* P1 chunked-run contract (gather_skip trace-only mode): serialize every
 * detector's per-key coherent sample sets to <out_dir>/gk_*.npy plus a
 * gkeys.json manifest, so the Python driver can accumulate samples across
 * chunks for the single final gather. */
void det_dump_gkeys(const SceneC *s);

/* P1 gather_only mode: load the driver's MERGED sample dump + accumulator
 * snapshots (same layout + acc_<i>_*.npy) from s->gather_input into the
 * DetC/GKey structures, so the normal in-binary gather_run (tiled kernel)
 * runs over them. */
void det_load_gather_state(SceneC *s);

/* --export-rays: one per-detector-event landing record (the ray state AT
 * the hit — pos/opl already advanced; tracer._export_records) */
typedef struct {
    int32_t det;
    kvec3 pos, dir, birth_pos;
    double opl, lam, power;
    int16_t source_id, lam_stratum, pol_stratum, generation;
    int8_t pol_mode;
    uint8_t scattered, coherent;
    int32_t refl_hist[HIST_DEPTH];
} ExportRec;

typedef struct {
    ExportRec *v;
    int64_t n, cap;
} ExportVec;

void exportvec_init(ExportVec *e);
void exportvec_free(ExportVec *e);
void exportvec_push(ExportVec *e, const ExportRec *r);
void exportvec_clear(ExportVec *e);

static inline void export_fill(ExportRec *er, int32_t det, const Ray *r) {
    er->det = det;
    er->pos = r->pos;
    er->dir = r->dir;
    er->birth_pos = r->birth_pos;
    er->opl = r->opl;
    er->lam = r->lam;
    er->power = ray_power(r);
    er->source_id = r->source_id;
    er->lam_stratum = r->lam_stratum;
    er->pol_stratum = r->pol_stratum;
    er->generation = r->generation;
    er->pol_mode = r->pol_mode;
    er->scattered = r->scattered;
    er->coherent = r->coherent;
    for (int i = 0; i < HIST_DEPTH; i++)
        er->refl_hist[i] = r->refl_hist[i];
}

/* merge thread export buffers into per-detector arrays and write
 * exp_<i>_*.npy outputs */
void det_collect_exports(SceneC *s, const ExportVec *e);
void det_write_exports(const SceneC *s);
void det_free_exports(SceneC *s);

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
