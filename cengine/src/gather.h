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
    /* ---- P1 NUFFT angular-spectrum route (use_nufft => gather_points_nufft
     * runs instead of the exact/tiled kernels; auto-gated per key by
     * gather.c). The gate fills the k-grid + detector-frame parameters;
     * occlusion keys are never routed here. See gather_nufft.c. */
    int use_nufft;
    kvec3 xhat, yhat;                   /* detector in-plane basis */
    double nufft_tol;                   /* NUFFT eps (default 1e-9) */
} GatherJob;

/* Gate parameters shared by the runtime gate (gather.c) and the route
 * (gather_nufft.c) so both agree on the exact k-grid. Computed from the
 * job's samples + target points; pure geometry, O(M+Q). */
typedef struct {
    int ok;                 /* separating plane AND obliquity-separable */
    int separating;         /* all samples strictly on one side of plane */
    int64_t N;              /* square k-grid: N x N modes */
    double dk;              /* physical k-spacing per mode [1/m] */
    double kprime;          /* k * n_amb */
    double sin_max;         /* max sin(theta) over samples->corners */
    double half_extent;     /* padded detector-frame half-extent [m] */
    double sep_margin_lam;  /* min |sample - plane| distance, in wavelengths */
    double obliq_var;       /* max variation of cos(theta_prop) across det */
    double zref, zdet;      /* oriented reference / detector z [m] (az<zdet) */
    kvec3 nrm_eff;          /* detector normal oriented toward the samples */
} NufftParams;

/* Compute (and gate) the NUFFT k-grid from a job. lambda = source vacuum
 * wavelength [m]. Returns via *out; out->ok is the separability+plane gate.
 * Pure host geometry, no CUDA — safe to call even without cuFINUFFT. */
int nufft_compute_params(const GatherJob *j, double lambda, NufftParams *out);

#define GATHER_TILE_CAP 128             /* == CUDA block size */

/* non-const: the kernels write job->near_exact_pairs (tiled path) */
void gather_points_cpu(GatherJob *job);
#ifdef MIEWB_HAS_CUDA
/* returns 0 on success, nonzero if CUDA unavailable at runtime */
int gather_points_cuda(GatherJob *job);
int gather_cuda_available(void);
/* P3 persistent worker: warm the primary context at serve start; release the
 * reusable device-buffer pool at worker exit. Both are no-ops when no CUDA
 * device is present. */
void gather_cuda_worker_init(void);
void gather_cuda_pool_free(void);
#endif

/* P1 NUFFT angular-spectrum route (gather_nufft.c). Fills job->Ex/Ey
 * (G*Q*2 interleaved) exactly as gather_points_cpu does, via
 * type-1 cuFINUFFT -> RS-I propagator on the k-grid -> type-2 cuFINUFFT.
 * Returns 1 on success, 0 if unavailable (binary built without cuFINUFFT,
 * or a runtime CUDA/plan failure) so the caller falls back to tiled/exact. */
int gather_points_nufft(GatherJob *job);

/* 1 if this binary was built with cuFINUFFT (NUFFT route compiled in). */
int nufft_available(void);
/* Free device VRAM in bytes (cudaMemGetInfo), or 0 if unavailable. */
int64_t nufft_free_vram_bytes(void);

#endif /* MIEWB_GATHER_H */
