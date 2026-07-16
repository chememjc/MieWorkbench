/* ===========================================================================
 * gather_nufft.c — P1 NUFFT angular-spectrum coherent-gather fast path.
 *
 * A THIRD gather route beside the exact fp64 kernel (gather_points_cpu) and
 * the tile-factorized kernel (gather_points_cpu_tiled), chosen PER COHERENCE
 * KEY by the runtime gate in gather.c. It computes the SAME field the exact
 * kernel accumulates,
 *
 *   E(P) = sum_i a_i K_i exp(i k (opl_i + n_amb r_ip)) / r_ip,
 *   K_i  = 0.5 (cos_prop_i + cos_det_i),  zeroed for back-radiation,
 *
 * via the Weyl plane-wave (angular-spectrum) decomposition:
 *
 *   type-1 cuFINUFFT  : scattered samples s_i -> uniform (k_x,k_y) grid,
 *                       weights a_i exp(i k opl_i) * axial z-fold
 *   RS-I propagator   : multiply on the grid by the exact free-space
 *                       transfer function to the detector plane
 *   type-2 cuFINUFFT  : k-grid -> detector pixel positions
 *
 * exact to NUFFT tolerance (1e-9) + two GATED approximations (gate in
 * gather.c, nufft_compute_params below):
 *
 *   (1) OBLIQUITY SEPARABILITY. cos_det is EXACT in the Weyl route (the
 *       0.5 cos_det half maps to the (k_z/k') factor on the propagator —
 *       the RS-I dipole term). cos_prop_i = r_hat_ip . dir_i depends on the
 *       EVALUATION point and is NOT separable; we fold cos_prop_i evaluated
 *       at the detector CENTRE into a_i and gate on its variation across the
 *       detector staying < 1e-3 (spec's separability analysis). Keys that
 *       fail route to the tiled/exact kernel.
 *   (2) AXIAL z-FOLD. Samples are not coplanar; the per-sample residual
 *       exp(-i k_z (az_i - zref)) is split paraxially: the k-independent
 *       part exp(-i k' delta_i) folds into a_i exactly, the residual
 *       defocus is bounded by the separability gate (small angular extent).
 *
 * The two obliquity halves are two type-1 transforms sharing one plan
 * (weights differ by cos_prop_i); the two Jones components (Ex, Ey) and the
 * G<=4 cross-estimator groups are handled as extra strength vectors on the
 * same non-uniform points (cuFINUFFT ntransf batching). Occlusion keys are
 * never routed here (per (tile,sample) visibility does not separate) — the
 * caller gates them out.
 *
 * Absolute Weyl normalization is irrelevant: gather.c renormalizes each
 * population's cube to its geometric power, so only the RELATIVE spatial
 * pattern + the A/B obliquity ratio + the propagator phase must be right.
 * =========================================================================== */
#include "gather.h"
#include "log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ambient index for the free flight — the exact kernel's GATHER_C_AMBIENT_N
 * (kernels/gatherk.h); phase uses k' = k * n_amb, amplitude uses 1/r. */
#define NUFFT_AMBIENT_N 1.000272
/* spatial padding so samples+pixels sit inside one NUFFT period (avoid
 * wraparound at the field edges) */
#define NUFFT_PAD 2.0
#define NUFFT_BAND_OVER 4.0    /* k-band over-coverage vs geometric angle */
/* gates */
#define NUFFT_SEP_MARGIN_LAM 10.0    /* min sample->plane distance [lambda] */
#define NUFFT_OBLIQ_TOL 1.0e-3       /* max cos_prop variation across det */
#define NUFFT_N_MIN 16
#define NUFFT_N_MAX 65536            /* sanity cap (VRAM gate is the real one) */

/* ------------------------------------------------------ gate / grid params */
int nufft_compute_params(const GatherJob *j, double lambda, NufftParams *out) {
    memset(out, 0, sizeof *out);
    int64_t M = j->M, Q = j->Q;
    if (M < 1 || Q < 1 || lambda <= 0.0) return 0;
    kvec3 nrm = j->nrm, xh = j->xhat, yh = j->yhat;

    /* detector plane offset (pixels are coplanar) + detector-frame centre
     * + the 4 bounding-box corner points (extremes of gx+gy and gx-gy);
     * these bound the true max sample->pixel angle without needing W/H. */
    double zdet = v3_dot(v3(j->points[0], j->points[1], j->points[2]), nrm);
    double cx = 0.0, cy = 0.0, det_half_x = 0.0, det_half_y = 0.0;
    kvec3 pcen = v3(0, 0, 0);
    for (int64_t q = 0; q < Q; q++) {
        kvec3 P = v3(j->points[q * 3], j->points[q * 3 + 1],
                     j->points[q * 3 + 2]);
        cx += v3_dot(P, xh);
        cy += v3_dot(P, yh);
        pcen = v3_add(pcen, P);
    }
    cx /= (double)Q; cy /= (double)Q;
    pcen = v3_scale(pcen, 1.0 / (double)Q);
    kvec3 corner_pt[4];
    double best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int c = 0; c < 4; c++) corner_pt[c] = pcen;
    for (int64_t q = 0; q < Q; q++) {
        kvec3 P = v3(j->points[q * 3], j->points[q * 3 + 1],
                     j->points[q * 3 + 2]);
        double gx = v3_dot(P, xh) - cx, gy = v3_dot(P, yh) - cy;
        if (fabs(gx) > det_half_x) det_half_x = fabs(gx);
        if (fabs(gy) > det_half_y) det_half_y = fabs(gy);
        double score[4] = {gx + gy, -(gx + gy), gx - gy, -(gx - gy)};
        for (int c = 0; c < 4; c++)
            if (score[c] > best[c]) { best[c] = score[c]; corner_pt[c] = P; }
    }

    /* signed sample->plane distances; orient normal so samples sit BEHIND
     * the detector (az < zdet). All must share one sign (separating plane). */
    double dsum = 0.0, dmin_abs = INFINITY, zref = 0.0;
    int any_pos = 0, any_neg = 0;
    for (int64_t i = 0; i < M; i++) {
        double az = v3_dot(v3(j->pos[i * 3], j->pos[i * 3 + 1],
                              j->pos[i * 3 + 2]), nrm);
        double d = zdet - az;
        if (d > 0) any_pos = 1;
        else if (d < 0) any_neg = 1;
        if (fabs(d) < dmin_abs) dmin_abs = fabs(d);
        dsum += d;
        zref += az;
    }
    zref /= (double)M;
    double sign = (dsum >= 0.0) ? 1.0 : -1.0;
    kvec3 nrm_eff = v3_scale(nrm, sign);
    /* re-express z on the oriented normal */
    double zdet_e = sign * zdet, zref_e = sign * zref;

    out->sep_margin_lam = dmin_abs / lambda;
    int separating = !(any_pos && any_neg);   /* strictly one side */

    /* per-sample: source half-extent, max sample->corner angle (sin_max),
     * and the obliquity-fold variation (cos_prop at corners vs centre). The
     * corners are the true detector bbox extremes found above. */
    double src_half_x = 0.0, src_half_y = 0.0, sin_max = 0.0, obliq_var = 0.0;
    for (int64_t i = 0; i < M; i++) {
        kvec3 pos = v3(j->pos[i * 3], j->pos[i * 3 + 1], j->pos[i * 3 + 2]);
        kvec3 dir = v3(j->dir[i * 3], j->dir[i * 3 + 1], j->dir[i * 3 + 2]);
        double ax = v3_dot(pos, xh) - cx, ay = v3_dot(pos, yh) - cy;
        if (fabs(ax) > src_half_x) src_half_x = fabs(ax);
        if (fabs(ay) > src_half_y) src_half_y = fabs(ay);
        kvec3 uc = v3_unit(v3_sub(pcen, pos));
        double cp_c = v3_dot(uc, dir);        /* cos_prop at det centre */
        for (int c = 0; c < 4; c++) {
            kvec3 u = v3_unit(v3_sub(corner_pt[c], pos));
            double cang = v3_dot(u, dir);
            if (cang > 1.0) cang = 1.0;
            if (cang < -1.0) cang = -1.0;
            double s = sqrt(1.0 - cang * cang);
            if (s > sin_max) sin_max = s;
            double dv = fabs(cang - cp_c);
            if (dv > obliq_var) obliq_var = dv;
        }
    }
    if (sin_max < 1e-9) sin_max = 1e-9;
    /* Each Huygens sample is an ideal point emitter (a FLAT k-spectrum);
     * truncating the k-band at the bare geometric detector angle rings.
     * Over-cover the band so the reconstructed (band-limited) spherical wave
     * matches the exact kernel at the detector to tolerance. */
    double band_sin = NUFFT_BAND_OVER * sin_max;
    if (band_sin > 0.999) band_sin = 0.999;
    /* Real-space FOV (half-span, per the larger axis) must contain: the
     * source aperture, the detector aperture, AND the propagation footprint
     * |Dz|*band_sin — else a source's periodic ghost propagates back into the
     * detector region and aliases. Use the max-axis extent for a square grid. */
    double src_half = src_half_x > src_half_y ? src_half_x : src_half_y;
    double det_half = det_half_x > det_half_y ? det_half_x : det_half_y;
    double prop_spread = fabs(zdet_e - zref_e) * band_sin;
    double half_ext = src_half + det_half + prop_spread;
    if (half_ext < 1e-12) half_ext = 1e-12;
    double kprime = (K_TWO_PI / lambda) * NUFFT_AMBIENT_N;
    double Lpad = NUFFT_PAD * half_ext;
    double dk = (0.5 * K_TWO_PI) / Lpad;                /* pi / Lpad */
    int64_t N = (int64_t)ceil(2.0 * kprime * band_sin / dk);
    if (N < NUFFT_N_MIN) N = NUFFT_N_MIN;
    if (N % 2) N += 1;                                  /* even: centre at N/2 */

    out->N = N;
    out->dk = dk;
    out->kprime = kprime;
    out->sin_max = sin_max;
    out->half_extent = half_ext;
    out->obliq_var = obliq_var;
    out->zref = zref_e;
    out->zdet = zdet_e;
    out->nrm_eff = nrm_eff;
    out->separating = separating;
    out->ok = separating && (out->sep_margin_lam > NUFFT_SEP_MARGIN_LAM)
              && (obliq_var < NUFFT_OBLIQ_TOL) && (N <= NUFFT_N_MAX);
    return out->ok;
}

#ifndef MIEWB_HAS_CUFINUFFT
/* Built without cuFINUFFT: the route is unavailable — caller falls back. */
int gather_points_nufft(GatherJob *j) {
    (void)j;
    return 0;
}
int nufft_available(void) { return 0; }
int64_t nufft_free_vram_bytes(void) { return 0; }
#else
/* ======================================================================== */
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufinufft.h>

int nufft_available(void) { return 1; }
int64_t nufft_free_vram_bytes(void) {
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) != cudaSuccess) return 0;
    return (int64_t)freeb;
}

#define CK(call, label) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        LOGW("gather nufft: CUDA error at %s: %s", label, \
             cudaGetErrorString(_e)); \
        goto fail; \
    } } while (0)

/* small host complex helper */
typedef struct { double re, im; } cx_t;
static inline cx_t cx_mul(cx_t a, cx_t b) {
    cx_t r = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    return r;
}

int gather_points_nufft(GatherJob *j) {
    double lambda = K_TWO_PI / j->k;
    NufftParams p;
    nufft_compute_params(j, lambda, &p);
    /* the caller's gate owns the accuracy/cost decision; the route only
     * REQUIRES a separating plane (correctness) — without it the Weyl
     * decomposition is invalid. (MIEWB_NUFFT_FORCE exercises this path on
     * keys the accuracy gate would otherwise reject.) */
    if (!p.separating) return 0;

    int64_t M = j->M, Q = j->Q, N = p.N;
    int G = j->G < 1 ? 1 : j->G;
    double dk = p.dk, kprime = p.kprime;
    double k = j->k;                     /* 2pi/lambda (opl phase coeff) */
    double Dz = p.zdet - p.zref;         /* > 0 with the oriented normal */
    kvec3 xh = j->xhat, yh = j->yhat;
    kvec3 nrm_e = p.nrm_eff;             /* oriented toward the samples */

    /* detector-frame centre (same as compute_params) */
    double cx = 0.0, cy = 0.0;
    for (int64_t q = 0; q < Q; q++) {
        kvec3 P = v3(j->points[q * 3], j->points[q * 3 + 1],
                     j->points[q * 3 + 2]);
        cx += v3_dot(P, xh); cy += v3_dot(P, yh);
    }
    cx /= (double)Q; cy /= (double)Q;
    kvec3 pcen = v3(0, 0, 0);
    for (int64_t q = 0; q < Q; q++)
        pcen = v3_add(pcen, v3(j->points[q * 3], j->points[q * 3 + 1],
                               j->points[q * 3 + 2]));
    pcen = v3_scale(pcen, 1.0 / (double)Q);

    /* ---- host buffers -------------------------------------------------- */
    double *sx = (double *)malloc((size_t)M * sizeof(double));
    double *sy = (double *)malloc((size_t)M * sizeof(double));
    double *tx = (double *)malloc((size_t)Q * sizeof(double));
    double *ty = (double *)malloc((size_t)Q * sizeof(double));
    cx_t *wbase = (cx_t *)malloc((size_t)M * sizeof(cx_t));   /* per-sample */
    double *cprop = (double *)malloc((size_t)M * sizeof(double));
    cuDoubleComplex *hc = (cuDoubleComplex *)malloc(
        (size_t)G * M * sizeof(cuDoubleComplex));            /* strengths */
    cuDoubleComplex *hfkA = (cuDoubleComplex *)malloc(
        (size_t)G * N * N * sizeof(cuDoubleComplex));
    cuDoubleComplex *hfkB = (cuDoubleComplex *)malloc(
        (size_t)G * N * N * sizeof(cuDoubleComplex));
    cuDoubleComplex *hout = (cuDoubleComplex *)malloc(
        (size_t)G * Q * sizeof(cuDoubleComplex));
    /* device */
    double *d_sx = NULL, *d_sy = NULL, *d_tx = NULL, *d_ty = NULL;
    cuDoubleComplex *d_c = NULL, *d_fkA = NULL, *d_fkB = NULL, *d_out = NULL;
    cufinufft_plan plan1 = NULL, plan2 = NULL;
    int ret = 0;
    if (!sx || !sy || !tx || !ty || !wbase || !cprop || !hc || !hfkA
            || !hfkB || !hout) {
        LOGW("gather nufft: host allocation failed"); goto fail;
    }

    /* scaled non-uniform coordinates (centred, so type1/type2 cancel) */
    for (int64_t i = 0; i < M; i++) {
        kvec3 pos = v3(j->pos[i * 3], j->pos[i * 3 + 1], j->pos[i * 3 + 2]);
        kvec3 dir = v3(j->dir[i * 3], j->dir[i * 3 + 1], j->dir[i * 3 + 2]);
        double ax = v3_dot(pos, xh) - cx, ay = v3_dot(pos, yh) - cy;
        sx[i] = dk * ax; sy[i] = dk * ay;
        /* per-sample base weight: exp(i k opl) * axial z-fold, cos_prop */
        double az = v3_dot(pos, nrm_e);
        double delta = az - p.zref;
        double ph = k * j->opl[i];
        double zf = -kprime * delta;              /* exp(i(ph)) exp(i zf) */
        cx_t rot = {cos(ph), sin(ph)};
        cx_t zfld = {cos(zf), sin(zf)};
        wbase[i] = cx_mul(rot, zfld);
        double cp = v3_dot(v3_unit(v3_sub(pcen, pos)), dir);
        cprop[i] = cp;                            /* cos_prop at det centre */
    }
    for (int64_t q = 0; q < Q; q++) {
        kvec3 P = v3(j->points[q * 3], j->points[q * 3 + 1],
                     j->points[q * 3 + 2]);
        tx[q] = dk * (v3_dot(P, xh) - cx);
        ty[q] = dk * (v3_dot(P, yh) - cy);
    }

    /* ---- device coords ------------------------------------------------- */
    CK(cudaMalloc(&d_sx, (size_t)M * sizeof(double)), "malloc sx");
    CK(cudaMalloc(&d_sy, (size_t)M * sizeof(double)), "malloc sy");
    CK(cudaMalloc(&d_tx, (size_t)Q * sizeof(double)), "malloc tx");
    CK(cudaMalloc(&d_ty, (size_t)Q * sizeof(double)), "malloc ty");
    CK(cudaMalloc(&d_c, (size_t)G * M * sizeof(cuDoubleComplex)), "malloc c");
    CK(cudaMalloc(&d_fkA, (size_t)G * N * N * sizeof(cuDoubleComplex)),
       "malloc fkA");
    CK(cudaMalloc(&d_fkB, (size_t)G * N * N * sizeof(cuDoubleComplex)),
       "malloc fkB");
    CK(cudaMalloc(&d_out, (size_t)G * Q * sizeof(cuDoubleComplex)),
       "malloc out");
    CK(cudaMemcpy(d_sx, sx, (size_t)M * sizeof(double),
                  cudaMemcpyHostToDevice), "cp sx");
    CK(cudaMemcpy(d_sy, sy, (size_t)M * sizeof(double),
                  cudaMemcpyHostToDevice), "cp sy");
    CK(cudaMemcpy(d_tx, tx, (size_t)Q * sizeof(double),
                  cudaMemcpyHostToDevice), "cp tx");
    CK(cudaMemcpy(d_ty, ty, (size_t)Q * sizeof(double),
                  cudaMemcpyHostToDevice), "cp ty");

    /* ---- plans (double precision for tol 1e-9) ------------------------- */
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.modeord = 0;                    /* CMCL increasing: index i -> i-N/2 */
    opts.gpu_device_id = 0;
    int64_t nmodes[3] = {N, N, 1};
    double tol = j->nufft_tol > 0.0 ? j->nufft_tol : 1e-9;
    if (cufinufft_makeplan(1, 2, nmodes, -1, G, tol, &plan1, &opts)) {
        LOGW("gather nufft: type-1 makeplan failed"); goto fail;
    }
    if (cufinufft_setpts(plan1, M, d_sx, d_sy, NULL, 0, NULL, NULL, NULL)) {
        LOGW("gather nufft: type-1 setpts failed"); goto fail;
    }
    if (cufinufft_makeplan(2, 2, nmodes, +1, G, tol, &plan2, &opts)) {
        LOGW("gather nufft: type-2 makeplan failed"); goto fail;
    }
    if (cufinufft_setpts(plan2, Q, d_tx, d_ty, NULL, 0, NULL, NULL, NULL)) {
        LOGW("gather nufft: type-2 setpts failed"); goto fail;
    }

    /* propagator prefactor (i/2pi) dk^2 — constant, washed by renorm but
     * kept so the A/B ratio and phase are exact. */
    double pref = dk * dk / K_TWO_PI;    /* times i */

    for (int comp = 0; comp < 2; comp++) {
        const float *amp_src = (comp == 0) ? j->Exs : j->Eys;
        float *out_field = (comp == 0) ? j->Ex : j->Ey;

        /* --- term A: strengths 0.5 cos_prop * base * amp (per group) --- */
        memset(hc, 0, (size_t)G * M * sizeof(cuDoubleComplex));
        for (int64_t i = 0; i < M; i++) {
            if (cprop[i] <= 0.0) continue;         /* back-radiation: K=0 */
            int g = j->group ? j->group[i] : 0;
            if (g >= G) g = G - 1;
            cx_t a = {(double)amp_src[i * 2], (double)amp_src[i * 2 + 1]};
            cx_t w = cx_mul(a, wbase[i]);
            double sc = 0.5 * cprop[i];
            hc[(size_t)g * M + i].x = sc * w.re;
            hc[(size_t)g * M + i].y = sc * w.im;
        }
        CK(cudaMemcpy(d_c, hc, (size_t)G * M * sizeof(cuDoubleComplex),
                      cudaMemcpyHostToDevice), "cp cA");
        if (cufinufft_execute(plan1, d_c, d_fkA)) {
            LOGW("gather nufft: type-1 execute A failed"); goto fail;
        }
        /* --- term B: strengths 0.5 * base * amp (cos_det via propagator) */
        memset(hc, 0, (size_t)G * M * sizeof(cuDoubleComplex));
        for (int64_t i = 0; i < M; i++) {
            if (cprop[i] <= 0.0) continue;
            int g = j->group ? j->group[i] : 0;
            if (g >= G) g = G - 1;
            cx_t a = {(double)amp_src[i * 2], (double)amp_src[i * 2 + 1]};
            cx_t w = cx_mul(a, wbase[i]);
            hc[(size_t)g * M + i].x = 0.5 * w.re;
            hc[(size_t)g * M + i].y = 0.5 * w.im;
        }
        CK(cudaMemcpy(d_c, hc, (size_t)G * M * sizeof(cuDoubleComplex),
                      cudaMemcpyHostToDevice), "cp cB");
        if (cufinufft_execute(plan1, d_c, d_fkB)) {
            LOGW("gather nufft: type-1 execute B failed"); goto fail;
        }

        /* --- k-grid: H = fkA * P_A + fkB * P_B, propagate to detector --- */
        CK(cudaMemcpy(hfkA, d_fkA,
                      (size_t)G * N * N * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToHost), "cp fkA->h");
        CK(cudaMemcpy(hfkB, d_fkB,
                      (size_t)G * N * N * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToHost), "cp fkB->h");
        double kp2 = kprime * kprime;
        for (int g = 0; g < G; g++) {
            cuDoubleComplex *FA = hfkA + (size_t)g * N * N;
            cuDoubleComplex *FB = hfkB + (size_t)g * N * N;
            for (int64_t i2 = 0; i2 < N; i2++) {
                double k2 = (double)(i2 - N / 2) * dk;
                for (int64_t i1 = 0; i1 < N; i1++) {
                    double k1 = (double)(i1 - N / 2) * dk;
                    size_t idx = (size_t)i1 + (size_t)N * i2;
                    double kr2 = k1 * k1 + k2 * k2;
                    if (kr2 >= kp2) {          /* evanescent: drop */
                        FA[idx].x = 0.0; FA[idx].y = 0.0;
                        continue;
                    }
                    double kz = sqrt(kp2 - kr2);
                    double phz = kz * Dz;       /* exp(i kz Dz) */
                    cx_t eph = {cos(phz), sin(phz)};
                    /* P_A = (i pref / kz) eph ; P_B = (i pref / kprime) eph */
                    cx_t iprefA = {0.0, pref / kz};
                    cx_t iprefB = {0.0, pref / kprime};
                    cx_t PA = cx_mul(iprefA, eph);
                    cx_t PB = cx_mul(iprefB, eph);
                    cx_t fa = {FA[idx].x, FA[idx].y};
                    cx_t fb = {FB[idx].x, FB[idx].y};
                    cx_t H = cx_mul(fa, PA);
                    cx_t hb = cx_mul(fb, PB);
                    H.re += hb.re; H.im += hb.im;
                    FA[idx].x = H.re; FA[idx].y = H.im;   /* reuse hfkA=H */
                }
            }
        }
        CK(cudaMemcpy(d_fkA, hfkA,
                      (size_t)G * N * N * sizeof(cuDoubleComplex),
                      cudaMemcpyHostToDevice), "cp H->d");
        if (cufinufft_execute(plan2, d_out, d_fkA)) {
            LOGW("gather nufft: type-2 execute failed"); goto fail;
        }
        CK(cudaMemcpy(hout, d_out, (size_t)G * Q * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToHost), "cp out->h");
        for (int g = 0; g < G; g++)
            for (int64_t q = 0; q < Q; q++) {
                size_t o = ((size_t)g * Q + q) * 2;
                out_field[o] = (float)hout[(size_t)g * Q + q].x;
                out_field[o + 1] = (float)hout[(size_t)g * Q + q].y;
            }
    }
    ret = 1;

fail:
    if (plan1) cufinufft_destroy(plan1);
    if (plan2) cufinufft_destroy(plan2);
    cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_tx); cudaFree(d_ty);
    cudaFree(d_c); cudaFree(d_fkA); cudaFree(d_fkB); cudaFree(d_out);
    free(sx); free(sy); free(tx); free(ty); free(wbase); free(cprop);
    free(hc); free(hfkA); free(hfkB); free(hout);
    return ret;
}
#endif /* MIEWB_HAS_CUFINUFFT */
