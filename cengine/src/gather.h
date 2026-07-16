/* ===========================================================================
 * gather.h — coherent Huygens-Fresnel final gather (host orchestration).
 *
 * Port of gather.render_coherent (gather.py:443-613): per (source,
 * lam_stratum, pol_stratum) sample set — E3 projection, M_eff gate,
 * smooth/speckle population split, G=4 cross-estimator, hot-pixel
 * sub-grid refinement, per-population power renormalization, optional
 * occlusion, optional complex field maps. Adds intensities into the
 * detector's spectral cube.
 *
 * Kernel backends: CUDA (cuda/gather.cu, when built with CUDA) and a CPU
 * OpenMP twin — both evaluate the SAME per-pair math (kernels/gatherk.h).
 *
 * Documented deviation from the Python engine: the cross-estimator groups
 * samples by (ray_key & 3) instead of numpy's default_rng(0) draw — the
 * estimator is unbiased under ANY independent grouping, so only the noise
 * realization differs (statistical parity; physics metrics like fringe
 * pitch/visibility are grouping-independent).
 * =========================================================================== */
#ifndef MIEWB_GATHER_H
#define MIEWB_GATHER_H

#include "scene.h"

/* Run the gather over every detector's sample sets; accumulate into
 * det->inc; write fields_<...>.npy when save_fields; append per-key
 * diagnostics JSON to <out_dir>/gather.json. Returns total kernel
 * (sample x point) pair count for calibration. */
int64_t gather_run(SceneC *s);

/* Field evaluation at arbitrary points, one population.
 * points: (Q,3); Exs/Eys: per-sample complex64 amplitudes (detector-frame
 * projected, sqrt(dA) folded); group: per-sample 0..G-1 (cross-estimator);
 * occ_mask: (n_tiles x M) 0/1 visibility or NULL; tile_of_point: (Q,) or
 * NULL. Outputs per-group complex64 sums Ex/Ey[(g * Q + q)]. G <= 4. */
typedef struct {
    int64_t M, Q;
    int G;
    const double *pos, *dir, *opl;      /* M*3, M*3, M */
    const float *Exs, *Eys;             /* M*2 interleaved re,im */
    const uint8_t *group;               /* M */
    const uint8_t *occ_mask;            /* n_tiles*M or NULL */
    const int32_t *tile_of_point;       /* Q or NULL (occlusion tiles) */
    const double *points;               /* Q*3 */
    kvec3 nrm;
    double k;                           /* 2 pi / lambda */
    float *Ex, *Ey;                     /* out: G*Q*2 interleaved */
    /* ---- tile factorization (P1; NULL/0 => plain exact kernel) ----
     * Points are permuted tile-major: point_order[tile_start[t] ..
     * tile_start[t+1]) lists the q indices of factorization tile t,
     * whose fp64 centre is tile_centers[t*3..] and whose max |P - p0|
     * is tile_dpmax[t]. Tiles hold at most GATHER_TILE_CAP points.
     * near_exact_pairs (out) counts (tile, sample) pairs whose R fell
     * inside GATHER_NEAR_FACTOR*dpmax and were routed through the exact
     * fp64 gather_pair instead of the fp32 tile path. */
    int use_tiled;
    int64_t n_ptiles;
    const double *tile_centers;         /* n_ptiles*3 */
    const int64_t *tile_start;          /* n_ptiles+1 */
    const int64_t *point_order;         /* Q */
    const float *tile_dpmax;            /* n_ptiles */
    int64_t near_exact_pairs;           /* out */
} GatherJob;

#define GATHER_TILE_CAP 128             /* == CUDA block size */

/* non-const: the kernels write job->near_exact_pairs (tiled path) */
void gather_points_cpu(GatherJob *job);
#ifdef MIEWB_HAS_CUDA
/* returns 0 on success, nonzero if CUDA unavailable at runtime */
int gather_points_cuda(GatherJob *job);
int gather_cuda_available(void);
#endif

#endif /* MIEWB_GATHER_H */
