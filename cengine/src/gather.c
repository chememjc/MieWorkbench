/* ===========================================================================
 * gather.c — see gather.h. Line references are to scripts/raytracer/
 * gather.py, the reference implementation this must match.
 * =========================================================================== */
#include "gather.h"
#include "detector.h"
#include "kernels/gatherk.h"
#include "kernels/quartic.h"
#include "log.h"
#include "npyio.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CROSS_GROUPS 4               /* gather.py:616 */
#define HOT_FRAC 5e-4                  /* _hot_pixels frac */
#define HOT_CAP 768                    /* _hot_pixels cap */
#define SUBGRID_MAX 24                 /* _subgrid_factor s_max */

/* ------------------------------------------------------------ CPU kernel */
void gather_points_cpu(GatherJob *j) {
    memset(j->Ex, 0, (size_t)j->G * j->Q * 2 * sizeof(float));
    memset(j->Ey, 0, (size_t)j->G * j->Q * 2 * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < j->Q; q++) {
        kvec3 P = v3(j->points[q * 3], j->points[q * 3 + 1],
                     j->points[q * 3 + 2]);
        kcplx32 ex[N_CROSS_GROUPS] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        kcplx32 ey[N_CROSS_GROUPS] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        const uint8_t *occ_col = NULL;
        if (j->occ_mask && j->tile_of_point)
            occ_col = j->occ_mask + (size_t)j->tile_of_point[q] * j->M;
        for (int64_t i = 0; i < j->M; i++) {
            kvec3 pos = v3(j->pos[i * 3], j->pos[i * 3 + 1],
                           j->pos[i * 3 + 2]);
            kvec3 dir = v3(j->dir[i * 3], j->dir[i * 3 + 1],
                           j->dir[i * 3 + 2]);
            kcplx32 Exs = {j->Exs[i * 2], j->Exs[i * 2 + 1]};
            kcplx32 Eys = {j->Eys[i * 2], j->Eys[i * 2 + 1]};
            float vis = occ_col ? (float)occ_col[i] : 1.0f;
            int g = j->group ? j->group[i] : 0;
            gather_pair(P, pos, dir, j->opl[i], Exs, Eys, j->k, j->nrm,
                        vis, &ex[g], &ey[g]);
        }
        for (int g = 0; g < j->G; g++) {
            j->Ex[((size_t)g * j->Q + q) * 2] = ex[g].re;
            j->Ex[((size_t)g * j->Q + q) * 2 + 1] = ex[g].im;
            j->Ey[((size_t)g * j->Q + q) * 2] = ey[g].re;
            j->Ey[((size_t)g * j->Q + q) * 2 + 1] = ey[g].im;
        }
    }
}

/* ------------------------------------------------ tile factorization */
/* Host-side point tiling for the P1 tile-factorized kernel: a permutation
 * of the Q points into tiles of <= GATHER_TILE_CAP, each with an fp64
 * centre (member mean) and max member distance dpmax. One code path
 * serves the structured detector grid (TH x TW pixel tiles) and the
 * arbitrary hot-pixel subgrid point lists (contiguous chunks). */
typedef struct {
    int64_t n_ptiles;
    double *centers;                    /* n*3 */
    int64_t *start;                     /* n+1 */
    int64_t *order;                     /* Q */
    float *dpmax;                       /* n */
    float dpmax_max;
} PTiles;

static void ptiles_finalize(PTiles *pt, const double *pts) {
    pt->dpmax_max = 0.0f;
    for (int64_t t = 0; t < pt->n_ptiles; t++) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        int64_t q0 = pt->start[t], q1 = pt->start[t + 1];
        for (int64_t l = q0; l < q1; l++) {
            const double *p = pts + pt->order[l] * 3;
            cx += p[0]; cy += p[1]; cz += p[2];
        }
        double inv = q1 > q0 ? 1.0 / (double)(q1 - q0) : 0.0;
        cx *= inv; cy *= inv; cz *= inv;
        pt->centers[t * 3] = cx;
        pt->centers[t * 3 + 1] = cy;
        pt->centers[t * 3 + 2] = cz;
        double d2max = 0.0;
        for (int64_t l = q0; l < q1; l++) {
            const double *p = pts + pt->order[l] * 3;
            double dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > d2max) d2max = d2;
        }
        pt->dpmax[t] = (float)sqrt(d2max);
        if (pt->dpmax[t] > pt->dpmax_max) pt->dpmax_max = pt->dpmax[t];
    }
}

static void ptiles_alloc(PTiles *pt, int64_t n_ptiles, int64_t Q) {
    pt->n_ptiles = n_ptiles;
    pt->centers = (double *)malloc((size_t)n_ptiles * 3 * sizeof(double));
    pt->start = (int64_t *)malloc(((size_t)n_ptiles + 1) * sizeof(int64_t));
    pt->order = (int64_t *)malloc((size_t)Q * sizeof(int64_t));
    pt->dpmax = (float *)malloc((size_t)n_ptiles * sizeof(float));
    if (!pt->centers || !pt->start || !pt->order || !pt->dpmax)
        die(EXIT_PHYSICS, "gather: point-tile allocation failed");
}

static void ptiles_free(PTiles *pt) {
    free(pt->centers); free(pt->start); free(pt->order); free(pt->dpmax);
    memset(pt, 0, sizeof *pt);
}

/* fp32 phase-error budget for the tiled kernel (rad); the tile size is
 * chosen so the gatherk.h bound 5e-7*k*dpmax stays inside it */
#define GATHER_PHASE_BUDGET_RAD 1e-3

/* detector grid: TH x TW pixel tiles, starting at 16x8 (== the CUDA
 * block) and halving until the tile half-diagonal fits the phase budget
 * at k_max. Smaller tiles amortize less fp64 but never break the
 * budget; 1x1 degenerates to the exact per-point staging (dp = 0). */
static void ptiles_build_grid(int32_t H, int32_t W, const double *pts,
                              double pixel_m, double k_max, PTiles *pt) {
    int32_t TH = 8, TW = 16;
    double dp_allowed = GATHER_PHASE_BUDGET_RAD / (5e-7 * k_max);
    while (TH * TW > 1
           && 0.5 * pixel_m * sqrt((double)TW * TW + (double)TH * TH)
              > dp_allowed) {
        if (TW >= TH) TW = (TW + 1) / 2;
        else TH = (TH + 1) / 2;
    }
    int32_t Th = (H + TH - 1) / TH, Tw = (W + TW - 1) / TW;
    ptiles_alloc(pt, (int64_t)Th * Tw, (int64_t)H * W);
    int64_t at = 0;
    for (int32_t tr = 0; tr < Th; tr++)
        for (int32_t tc = 0; tc < Tw; tc++) {
            pt->start[(int64_t)tr * Tw + tc] = at;
            int32_t y1 = (tr + 1) * TH < H ? (tr + 1) * TH : H;
            int32_t x1 = (tc + 1) * TW < W ? (tc + 1) * TW : W;
            for (int32_t iy = tr * TH; iy < y1; iy++)
                for (int32_t ix = tc * TW; ix < x1; ix++)
                    pt->order[at++] = (int64_t)iy * W + ix;
        }
    pt->start[pt->n_ptiles] = at;
    ptiles_finalize(pt, pts);
}

/* arbitrary point list (hot-pixel subgrid): contiguous chunks */
static void ptiles_build_chunks(int64_t Q, const double *pts, PTiles *pt) {
    int64_t n = (Q + GATHER_TILE_CAP - 1) / GATHER_TILE_CAP;
    ptiles_alloc(pt, n, Q);
    for (int64_t q = 0; q < Q; q++) pt->order[q] = q;
    for (int64_t t = 0; t <= n; t++) {
        int64_t s = t * GATHER_TILE_CAP;
        pt->start[t] = s < Q ? s : Q;
    }
    ptiles_finalize(pt, pts);
}

/* staged per-(tile, sample) record, shared layout with the CUDA kernel */
typedef struct {
    float ux, uy, uz, R, ph0;
    float dirx, diry, dirz;
    float exr, exi, eyr, eyi;
    uint8_t group, near;
} StagedS;

#define STAGE_CHUNK 256

/* CPU tile-factorized kernel: OpenMP over tiles, samples staged in
 * chunks against the tile centre, fp32 inner loop; near-field samples
 * routed through the exact fp64 gather_pair. */
static void gather_points_cpu_tiled(GatherJob *j) {
    memset(j->Ex, 0, (size_t)j->G * j->Q * 2 * sizeof(float));
    memset(j->Ey, 0, (size_t)j->G * j->Q * 2 * sizeof(float));
    float nxf = (float)j->nrm.x, nyf = (float)j->nrm.y,
          nzf = (float)j->nrm.z;
    float k_f = (float)j->k;
    int64_t near_total = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:near_total)
    for (int64_t t = 0; t < j->n_ptiles; t++) {
        int64_t q0 = j->tile_start[t], q1 = j->tile_start[t + 1];
        int np = (int)(q1 - q0);
        if (np <= 0) continue;
        kvec3 p0 = v3(j->tile_centers[t * 3], j->tile_centers[t * 3 + 1],
                      j->tile_centers[t * 3 + 2]);
        double near_R = GATHER_NEAR_FACTOR * (double)j->tile_dpmax[t];
        kvec3 P[GATHER_TILE_CAP];
        float dpx[GATHER_TILE_CAP], dpy[GATHER_TILE_CAP],
              dpz[GATHER_TILE_CAP], dp2[GATHER_TILE_CAP];
        const uint8_t *occ_of_p[GATHER_TILE_CAP];
        for (int l = 0; l < np; l++) {
            int64_t q = j->point_order[q0 + l];
            P[l] = v3(j->points[q * 3], j->points[q * 3 + 1],
                      j->points[q * 3 + 2]);
            dpx[l] = (float)(P[l].x - p0.x);
            dpy[l] = (float)(P[l].y - p0.y);
            dpz[l] = (float)(P[l].z - p0.z);
            dp2[l] = dpx[l] * dpx[l] + dpy[l] * dpy[l] + dpz[l] * dpz[l];
            occ_of_p[l] = (j->occ_mask && j->tile_of_point)
                ? j->occ_mask + (size_t)j->tile_of_point[q] * j->M : NULL;
        }
        kcplx32 ex[GATHER_TILE_CAP][4], ey[GATHER_TILE_CAP][4];
        memset(ex, 0, sizeof ex);
        memset(ey, 0, sizeof ey);
        StagedS sh[STAGE_CHUNK];
        for (int64_t base = 0; base < j->M; base += STAGE_CHUNK) {
            int chunk = (int)((j->M - base < STAGE_CHUNK)
                              ? (j->M - base) : STAGE_CHUNK);
            int n_near = 0;
            for (int i = 0; i < chunk; i++) {
                int64_t s = base + i;
                kvec3 pos = v3(j->pos[s * 3], j->pos[s * 3 + 1],
                               j->pos[s * 3 + 2]);
                double R = gather_stage_tile(p0, pos, j->opl[s], j->k,
                                             &sh[i].ux, &sh[i].uy,
                                             &sh[i].uz, &sh[i].R,
                                             &sh[i].ph0);
                sh[i].dirx = (float)j->dir[s * 3];
                sh[i].diry = (float)j->dir[s * 3 + 1];
                sh[i].dirz = (float)j->dir[s * 3 + 2];
                sh[i].exr = j->Exs[s * 2];
                sh[i].exi = j->Exs[s * 2 + 1];
                sh[i].eyr = j->Eys[s * 2];
                sh[i].eyi = j->Eys[s * 2 + 1];
                sh[i].group = j->group ? j->group[s] : 0;
                sh[i].near = R < near_R;
                if (sh[i].near) n_near++;
            }
            near_total += (int64_t)n_near * np;
            for (int l = 0; l < np; l++) {
                for (int i = 0; i < chunk; i++) {
                    float vis = occ_of_p[l]
                        ? (float)occ_of_p[l][base + i] : 1.0f;
                    int g = sh[i].group;
                    if (sh[i].near) {
                        int64_t s = base + i;
                        kcplx32 Exs = {sh[i].exr, sh[i].exi};
                        kcplx32 Eys = {sh[i].eyr, sh[i].eyi};
                        gather_pair(P[l],
                                    v3(j->pos[s * 3], j->pos[s * 3 + 1],
                                       j->pos[s * 3 + 2]),
                                    v3(j->dir[s * 3], j->dir[s * 3 + 1],
                                       j->dir[s * 3 + 2]),
                                    j->opl[s], Exs, Eys, j->k, j->nrm,
                                    vis, &ex[l][g], &ey[l][g]);
                    } else {
                        kcplx32 Exs = {sh[i].exr, sh[i].exi};
                        kcplx32 Eys = {sh[i].eyr, sh[i].eyi};
                        gather_pair_tile(dpx[l], dpy[l], dpz[l], dp2[l],
                                         sh[i].ux, sh[i].uy, sh[i].uz,
                                         sh[i].R, sh[i].ph0, sh[i].dirx,
                                         sh[i].diry, sh[i].dirz,
                                         nxf, nyf, nzf, k_f, Exs, Eys,
                                         vis, &ex[l][g], &ey[l][g]);
                    }
                }
            }
        }
        for (int l = 0; l < np; l++) {
            int64_t q = j->point_order[q0 + l];
            for (int g = 0; g < j->G; g++) {
                j->Ex[((size_t)g * j->Q + q) * 2] = ex[l][g].re;
                j->Ex[((size_t)g * j->Q + q) * 2 + 1] = ex[l][g].im;
                j->Ey[((size_t)g * j->Q + q) * 2] = ey[l][g].re;
                j->Ey[((size_t)g * j->Q + q) * 2 + 1] = ey[l][g].im;
            }
        }
    }
    j->near_exact_pairs = near_total;
}

/* ------------------------------------------------------------- helpers */
static int dbl_cmp(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static double np_percentile(double *a, int64_t n, double pct) {
    if (n == 0) return 0.0;
    qsort(a, (size_t)n, sizeof(double), dbl_cmp);
    double idx = pct / 100.0 * (double)(n - 1);
    int64_t lo = (int64_t)idx;
    int64_t hi = lo + 1 < n ? lo + 1 : n - 1;
    double frac = idx - (double)lo;
    return a[lo] * (1.0 - frac) + a[hi] * frac;
}

/* symmetric 3x3 eigenvalues (ascending) via the characteristic cubic */
static void sym3_eigvals(const double m[6], double ev[3]) {
    /* m: xx, yy, zz, xy, xz, yz */
    double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5];
    /* det(M - tI) = -t^3 + tr t^2 - s t + det */
    double tr = a + b + c;
    double s = a * b + a * c + b * c - d * d - e * e - f * f;
    double det = a * (b * c - f * f) - d * (d * c - f * e)
                 + e * (d * f - b * e);
    /* t^3 - tr t^2 + s t - det = 0 */
    double r[3];
    int n = k_solve_cubic(-tr, s, -det, r);
    if (n == 1) { r[1] = r[0]; r[2] = r[0]; }
    else if (n == 2) { r[2] = r[1]; }
    /* ascending */
    for (int i = 0; i < 2; i++)
        for (int jj = 0; jj < 2 - i; jj++)
            if (r[jj] > r[jj + 1]) {
                double t = r[jj];
                r[jj] = r[jj + 1];
                r[jj + 1] = t;
            }
    ev[0] = r[0]; ev[1] = r[1]; ev[2] = r[2];
}

/* effective_samples (gather.py:52-59): M_eff = (sum|a|)^2 / sum|a|^2 with
 * a = |E3| per sample */
static double effective_samples(const double *amp, int64_t m) {
    double s1 = 0.0, s2 = 0.0;
    for (int64_t i = 0; i < m; i++) {
        s1 += amp[i];
        s2 += amp[i] * amp[i];
    }
    return s2 > 0.0 ? s1 * s1 / s2 : 0.0;
}

/* check_sampling diagnostic (gather.py:62-95) */
static double check_sampling(const GKey *g, const double *grid_pts,
                             int32_t H, int32_t W) {
    int64_t m = g->n;
    if (m < 4)
        die(EXIT_PHYSICS, "gather needs >= 4 samples, got %lld",
            (long long)m);
    double mean[3] = {0, 0, 0};
    for (int64_t i = 0; i < m; i++)
        for (int c = 0; c < 3; c++) mean[c] += g->pos[i * 3 + c];
    for (int c = 0; c < 3; c++) mean[c] /= (double)m;
    double cov[6] = {0, 0, 0, 0, 0, 0};   /* xx yy zz xy xz yz */
    for (int64_t i = 0; i < m; i++) {
        double x = g->pos[i * 3] - mean[0];
        double y = g->pos[i * 3 + 1] - mean[1];
        double z = g->pos[i * 3 + 2] - mean[2];
        cov[0] += x * x; cov[1] += y * y; cov[2] += z * z;
        cov[3] += x * y; cov[4] += x * z; cov[5] += y * z;
    }
    for (int c = 0; c < 6; c++) cov[c] /= (double)m;
    double ev[3];
    sym3_eigvals(cov, ev);
    double e0 = ev[2] > 1e-30 ? ev[2] : 1e-30;   /* largest */
    double e1 = ev[1] > 1e-30 ? ev[1] : 1e-30;
    double a = 4.0 * sqrt(e0);
    double b = 4.0 * sqrt(e1);
    double delta = (b < 1e-12) ? a / (double)m
                               : sqrt(a * b / (double)m);
    /* theta_max over the 4 extreme pixel centers */
    const double *corners[4] = {
        grid_pts, grid_pts + (size_t)(W - 1) * 3,
        grid_pts + (size_t)(H - 1) * W * 3,
        grid_pts + ((size_t)H * W - 1) * 3};
    double sin_max = 0.0;
    double lam_min = INFINITY;
    for (int64_t i = 0; i < m; i++)
        if (g->lam[i] < lam_min) lam_min = g->lam[i];
    for (int ci = 0; ci < 4; ci++) {
        for (int64_t i = 0; i < m; i++) {
            kvec3 v = v3(corners[ci][0] - g->pos[i * 3],
                         corners[ci][1] - g->pos[i * 3 + 1],
                         corners[ci][2] - g->pos[i * 3 + 2]);
            v = v3_unit(v);
            double cosang = v.x * g->dir[i * 3] + v.y * g->dir[i * 3 + 1]
                            + v.z * g->dir[i * 3 + 2];
            if (cosang > 1.0) cosang = 1.0;
            if (cosang < -1.0) cosang = -1.0;
            double s = sqrt(1.0 - cosang * cosang);
            if (s > sin_max) sin_max = s;
        }
    }
    double k = K_TWO_PI / lam_min;
    return k * delta * (sin_max > 1e-9 ? sin_max : 1e-9);
}

/* pixel-center grid points (detector.py:148-162 mapping, the same math
 * det_compute_mask uses) */
static double *det_grid_points(const SceneC *s, const DetC *d) {
    const FaceC *face = &s->faces[d->face_id];
    kvec3 origin = face->surf.u.plane.origin;
    kvec3 n_comp = origin;
    n_comp = v3_sub(n_comp, v3_scale(d->xhat, v3_dot(origin, d->xhat)));
    n_comp = v3_sub(n_comp, v3_scale(d->yhat, v3_dot(origin, d->yhat)));
    double *pts = (double *)malloc((size_t)d->H * d->W * 3
                                   * sizeof(double));
    if (!pts) die(EXIT_PHYSICS, "gather: grid point allocation failed");
    for (int32_t iy = 0; iy < d->H; iy++) {
        double gy = d->y_lo + (iy + 0.5) * d->pixel_m;
        for (int32_t ix = 0; ix < d->W; ix++) {
            double gx = d->x_lo + (ix + 0.5) * d->pixel_m;
            kvec3 p = n_comp;
            p = v3_fma(p, gx, d->xhat);
            p = v3_fma(p, gy, d->yhat);
            size_t at = ((size_t)iy * d->W + ix) * 3;
            pts[at] = p.x;
            pts[at + 1] = p.y;
            pts[at + 2] = p.z;
        }
    }
    return pts;
}

/* ------------------------------------------------------- occlusion mask */
/* Any-hit shadow test over the scene TLAS, skipping ALL detector-screen
 * faces (run_trace passes occ_faces = every face not in grids). Returns 1
 * if some face blocks the segment (t in (eps, dist-eps)). */
static int shadow_blocked(const SceneC *s, kvec3 o, kvec3 d, double dist) {
    const double eps = 1e-7;                    /* _OCC_T_EPS */
    if (!s->tlas.nodes) {
        for (int32_t fid = 0; fid < s->n_faces; fid++) {
            if (s->faces[fid].detector >= 0) continue;
            double t = face_intersect(&s->faces[fid], o, d);
            if (t > eps && t < dist - eps) return 1;
        }
        return 0;
    }
    kvec3 inv = v3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    int32_t stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode *nd = &s->tlas.nodes[stack[--sp]];
        if (!bvh_ray_box(o, inv, dist, nd->bbmin, nd->bbmax, eps))
            continue;
        if (nd->left < 0) {
            for (int32_t i = 0; i < nd->count; i++) {
                int32_t fid = s->tlas.order[nd->start + i];
                if (s->faces[fid].detector >= 0) continue;
                double t = face_intersect(&s->faces[fid], o, d);
                if (t > eps && t < dist - eps) return 1;
            }
        } else {
            if (sp + 2 > BVH_STACK_MAX) return 0;   /* give up: visible */
            stack[sp++] = nd->left;
            stack[sp++] = nd->right;
        }
    }
    return 0;
}

/* _build_occlusion (gather.py:376-413): (n_tiles x M) visibility mask via
 * one shadow ray per (sample, tile-center) pair, OpenMP over pairs. */
static uint8_t *build_occlusion(const SceneC *s, const DetC *d,
                                const GKey *g, const double *grid_pts,
                                int tile, int32_t *tile_of_point_out,
                                int64_t *n_tiles_out, double *frac_out) {
    int32_t Th = (d->H + tile - 1) / tile;
    int32_t Tw = (d->W + tile - 1) / tile;
    int64_t n_tiles = (int64_t)Th * Tw;
    /* tile centers = pixel centers at clamped (row, col) mid-tiles */
    double *centers = (double *)malloc((size_t)n_tiles * 3
                                       * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc((size_t)n_tiles * g->n);
    if (!centers || !mask)
        die(EXIT_PHYSICS, "gather: occlusion mask allocation failed "
            "(%lld tiles x %lld samples)", (long long)n_tiles,
            (long long)g->n);
    for (int32_t tr = 0; tr < Th; tr++) {
        int32_t rc = tr * tile + tile / 2;
        if (rc > d->H - 1) rc = d->H - 1;
        for (int32_t tc = 0; tc < Tw; tc++) {
            int32_t cc = tc * tile + tile / 2;
            if (cc > d->W - 1) cc = d->W - 1;
            size_t src = ((size_t)rc * d->W + cc) * 3;
            size_t dst = ((size_t)tr * Tw + tc) * 3;
            memcpy(centers + dst, grid_pts + src, 3 * sizeof(double));
        }
    }
    /* per-grid-point tile ids */
    for (int32_t iy = 0; iy < d->H; iy++)
        for (int32_t ix = 0; ix < d->W; ix++)
            tile_of_point_out[(size_t)iy * d->W + ix] =
                (iy / tile) * Tw + (ix / tile);

    int64_t blocked_pairs = 0;
    #pragma omp parallel for schedule(static) reduction(+:blocked_pairs)
    for (int64_t pair = 0; pair < n_tiles * g->n; pair++) {
        int64_t ti = pair / g->n;
        int64_t si = pair % g->n;
        kvec3 o = v3(g->pos[si * 3], g->pos[si * 3 + 1],
                     g->pos[si * 3 + 2]);
        kvec3 c = v3(centers[ti * 3], centers[ti * 3 + 1],
                     centers[ti * 3 + 2]);
        kvec3 seg = v3_sub(c, o);
        double dist = v3_norm(seg);
        uint8_t vis = 1;
        if (dist > 1e-7) {
            kvec3 dir = v3_scale(seg, 1.0 / dist);
            if (shadow_blocked(s, o, dir, dist)) vis = 0;
        }
        mask[(size_t)ti * g->n + si] = vis;
        if (!vis) blocked_pairs++;
    }
    free(centers);
    *n_tiles_out = n_tiles;
    *frac_out = n_tiles * g->n > 0
        ? (double)blocked_pairs / (double)(n_tiles * g->n) : 0.0;
    return mask;
}

/* ------------------------------------------------ kernel dispatch layer */
static void run_job(GatherJob *j, uint8_t backend) {
#ifdef MIEWB_HAS_CUDA
    if (backend != 2) {
        if (gather_points_cuda(j) == 0) return;
        if (backend == 1)
            die(EXIT_CUDA, "gather backend=cuda requested but the CUDA "
                "kernel failed / no device");
        LOGW("gather: CUDA unavailable — falling back to the CPU kernel");
    }
#else
    if (backend == 1)
        die(EXIT_CUDA, "gather backend=cuda requested but this binary was "
            "built without CUDA");
#endif
    if (j->use_tiled) gather_points_cpu_tiled(j);
    else gather_points_cpu(j);
}

/* per-population intensity at points via the G-group cross-estimator
 * (_cross_intensity, gather.py:619-657). Returns intensity (Q,) in
 * `inten` and optionally the plain complex field sums (fields_ex/ey,
 * complex64 pairs) for --save-fields. inv_ilam = -i/lambda factor. */
static void cross_intensity(const SceneC *s, GatherJob *j, int unbiased,
                            double lam0, double *inten,
                            float *field_ex, float *field_ey) {
    int64_t Q = j->Q;
    int G = (unbiased && j->M >= 4 * N_CROSS_GROUPS) ? N_CROSS_GROUPS : 1;
    j->G = G;
    if (G == 1) j->group = NULL;       /* single group: ignore ids */
    run_job(j, s->gather_backend);
    /* combine groups: tot = |sum_g E|^2 (+ fields), unbiased subtracts
     * sum_g |E_g|^2 and divides by (1 - 1/G) */
    double inv_lam = 1.0 / lam0;       /* |(-i/lam) z|^2 = |z|^2 / lam^2 */
    #pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < Q; q++) {
        float exr = 0, exi = 0, eyr = 0, eyi = 0;
        double sum_sq = 0.0;
        for (int g = 0; g < G; g++) {
            float ar = j->Ex[((size_t)g * Q + q) * 2];
            float ai = j->Ex[((size_t)g * Q + q) * 2 + 1];
            float br = j->Ey[((size_t)g * Q + q) * 2];
            float bi = j->Ey[((size_t)g * Q + q) * 2 + 1];
            exr += ar; exi += ai; eyr += br; eyi += bi;
            sum_sq += (double)ar * ar + (double)ai * ai
                      + (double)br * br + (double)bi * bi;
        }
        double tot = (double)exr * exr + (double)exi * exi
                     + (double)eyr * eyr + (double)eyi * eyi;
        double ii = (G > 1) ? (tot - sum_sq) / (1.0 - 1.0 / (double)G)
                            : tot;
        inten[q] = ii * inv_lam * inv_lam;
        if (field_ex) {
            /* apply 1/(i lam) = -i/lam once per point (gather.py:164) */
            field_ex[q * 2] += (float)(exi * inv_lam);
            field_ex[q * 2 + 1] += (float)(-exr * inv_lam);
            field_ey[q * 2] += (float)(eyi * inv_lam);
            field_ey[q * 2 + 1] += (float)(-eyr * inv_lam);
        }
    }
}

/* hot-pixel detection (_hot_pixels, gather.py:707-736) */
static int64_t hot_pixels(const DetC *d, const double *grid_pts,
                          const GKey *g, const int64_t *sel_idx,
                          int64_t n_sel, int32_t *hy, int32_t *hx) {
    kvec3 nrm = d->normal;
    double plane_off = grid_pts[0] * nrm.x + grid_pts[1] * nrm.y
                       + grid_pts[2] * nrm.z;
    double *pmap = (double *)calloc((size_t)d->H * d->W, sizeof(double));
    if (!pmap) die(EXIT_PHYSICS, "gather: hot-pixel map allocation failed");
    double total = 0.0;
    for (int64_t k = 0; k < n_sel; k++) {
        int64_t i = sel_idx[k];
        kvec3 dir = v3(g->dir[i * 3], g->dir[i * 3 + 1],
                       g->dir[i * 3 + 2]);
        double denom = v3_dot(dir, nrm);
        if (fabs(denom) <= 1e-9) continue;
        kvec3 pos = v3(g->pos[i * 3], g->pos[i * 3 + 1],
                       g->pos[i * 3 + 2]);
        double t = (plane_off - v3_dot(pos, nrm)) / denom;
        if (t <= 0.0) continue;
        kvec3 land = v3_fma(pos, t, dir);
        double fx = (v3_dot(land, d->xhat) - d->x_lo) / d->pixel_m;
        double fy = (v3_dot(land, d->yhat) - d->y_lo) / d->pixel_m;
        int32_t xi = (int32_t)floor(fx);
        int32_t yi = (int32_t)floor(fy);
        if (xi < 0 || xi >= d->W || yi < 0 || yi >= d->H) continue;
        pmap[(size_t)yi * d->W + xi] += g->power[i];
        total += g->power[i];
    }
    int64_t n_hot = 0;
    if (total > 0.0) {
        for (int32_t iy = 0; iy < d->H && n_hot < HOT_CAP; iy++)
            for (int32_t ix = 0; ix < d->W && n_hot < HOT_CAP; ix++)
                if (pmap[(size_t)iy * d->W + ix] > HOT_FRAC * total) {
                    hy[n_hot] = iy;
                    hx[n_hot] = ix;
                    n_hot++;
                }
        /* Python keeps the TOP `cap` by power; the cap is rarely reached
         * — when it is, our first-HOT_CAP-in-scan-order subset differs
         * (documented; refinement choice, not physics) */
    }
    free(pmap);
    return n_hot;
}

/* _subgrid_factor (gather.py:739-748) */
static int subgrid_factor(const DetC *d, const GKey *g,
                          const int64_t *sel_idx, int64_t n_sel,
                          double lam0) {
    double *sins = (double *)malloc((size_t)n_sel * sizeof(double));
    if (!sins) die(EXIT_PHYSICS, "gather: subgrid allocation failed");
    for (int64_t k = 0; k < n_sel; k++) {
        int64_t i = sel_idx[k];
        double c = fabs(g->dir[i * 3] * d->normal.x
                        + g->dir[i * 3 + 1] * d->normal.y
                        + g->dir[i * 3 + 2] * d->normal.z);
        double s2 = 1.0 - c * c;
        if (s2 < 0.0) s2 = 0.0;
        if (s2 > 1.0) s2 = 1.0;
        sins[k] = sqrt(s2);
    }
    double sin95 = np_percentile(sins, n_sel, 95.0);
    free(sins);
    if (sin95 < 1e-6) return 1;
    double delta = lam0 / (4.0 * sin95);
    double sf = ceil(d->pixel_m / delta);
    if (sf < 1.0) sf = 1.0;
    if (sf > SUBGRID_MAX) sf = SUBGRID_MAX;
    return (int)sf;
}

/* ------------------------------------------------------------- top level */
int64_t gather_run(SceneC *s) {
    char path[1200];
    snprintf(path, sizeof path, "%s/gather.json", s->out_dir);
    FILE *jf = fopen(path, "w");
    if (!jf) die(EXIT_PHYSICS, "gather: cannot write %s", path);
    fprintf(jf, "{");
    int first_det = 1;
    int64_t total_pairs = 0;
    const char *backend_name =
#ifdef MIEWB_HAS_CUDA
        (s->gather_backend != 2 && gather_cuda_available()) ? "cuda" :
#endif
        "cpu";

    for (int di = 0; di < s->n_dets; di++) {
        DetC *d = &s->dets[di];
        if (d->n_gkeys == 0) continue;
        double *grid_pts = det_grid_points(s, d);
        int64_t Q = (int64_t)d->H * d->W;
        int tiled = !s->gather_exact;
        PTiles gridtiles;
        memset(&gridtiles, 0, sizeof gridtiles);
        if (tiled) {
            /* size tiles for the shortest wavelength this detector will
             * gather (largest k) so every key meets the phase budget */
            double lam_min = INFINITY;
            for (int32_t ki = 0; ki < d->n_gkeys; ki++)
                if (d->gkeys[ki].lam[0] < lam_min)
                    lam_min = d->gkeys[ki].lam[0];
            ptiles_build_grid(d->H, d->W, grid_pts, d->pixel_m,
                              K_TWO_PI / lam_min, &gridtiles);
        }
        fprintf(jf, "%s\n  \"%s\": {", first_det ? "" : ",", d->label);
        first_det = 0;
        int first_key = 1;

        for (int32_t ki = 0; ki < d->n_gkeys; ki++) {
            GKey *g = &d->gkeys[ki];
            int64_t M = g->n;
            int64_t near_pairs_key = 0;
            float dpmax_key = tiled ? gridtiles.dpmax_max : 0.0f;
            log_progress("trace", 0.96, "gather %s key %d/%d (%lld "
                         "samples, %s)", d->label, ki + 1, d->n_gkeys,
                         (long long)M, backend_name);
            LOGI("gather %s key %d/%d: %lld samples (%s)", d->label,
                 ki + 1, d->n_gkeys, (long long)M, backend_name);
            /* ---- E3 projection + amplitudes (gather.py:485-499) ---- */
            const SourceC *src = &s->sources[g->source_id];
            double dA = 1.0;
            if (src->sample_area)
                dA = src->sample_area[g->lam_stratum * src->n_pol
                                      + g->pol_stratum];
            double sqrt_dA = sqrt(dA);
            float *Exs = (float *)malloc((size_t)M * 2 * sizeof(float));
            float *Eys = (float *)malloc((size_t)M * 2 * sizeof(float));
            double *amp = (double *)malloc((size_t)M * sizeof(double));
            uint8_t *group = (uint8_t *)malloc((size_t)M);
            if (!Exs || !Eys || !amp || !group)
                die(EXIT_PHYSICS, "gather: projection allocation failed");
            for (int64_t i = 0; i < M; i++) {
                kvec3 sh = v3(g->s_hat[i * 3], g->s_hat[i * 3 + 1],
                              g->s_hat[i * 3 + 2]);
                kvec3 dr = v3(g->dir[i * 3], g->dir[i * 3 + 1],
                              g->dir[i * 3 + 2]);
                kvec3 ph = v3_cross(dr, sh);
                /* E3 = (Es s_hat + Ep p_hat) sqrt(dA); project on
                 * xhat/yhat (complex128 -> complex64, gather.py:130-131,
                 * 203-204, 498-499) */
                double sx = v3_dot(sh, s->dets[di].xhat);
                double px = v3_dot(ph, s->dets[di].xhat);
                double sy = v3_dot(sh, s->dets[di].yhat);
                double py = v3_dot(ph, s->dets[di].yhat);
                kcplx ex = kc_scale(kc_add(kc_scale(g->Es[i], sx),
                                           kc_scale(g->Ep[i], px)),
                                    sqrt_dA);
                kcplx ey = kc_scale(kc_add(kc_scale(g->Es[i], sy),
                                           kc_scale(g->Ep[i], py)),
                                    sqrt_dA);
                Exs[i * 2] = (float)ex.re;
                Exs[i * 2 + 1] = (float)ex.im;
                Eys[i * 2] = (float)ey.re;
                Eys[i * 2 + 1] = (float)ey.im;
                /* |E3|: E3's squared norm equals |Es|^2 + |Ep|^2 (s,p
                 * orthonormal) times dA */
                amp[i] = sqrt((kc_abs2(g->Es[i]) + kc_abs2(g->Ep[i]))
                              * dA);
                group[i] = (uint8_t)(g->ray_key[i] & 3);
            }
            double m_eff = effective_samples(amp, M);
            double step = check_sampling(g, grid_pts, d->H, d->W);
            if (s->enforce_gate && m_eff < s->min_eff_samples)
                die(EXIT_PHYSICS,
                    "gather undersampled on %s for source/stratum "
                    "(%d, %d, %d): effective samples M_eff=%.0f < %.0f "
                    "(speckle pedestal %.2e of peak). Increase --rays by "
                    "~%.0fx.", d->label, g->source_id, g->lam_stratum,
                    g->pol_stratum, m_eff, s->min_eff_samples,
                    1.0 / (m_eff > 1e-9 ? m_eff : 1e-9),
                    s->min_eff_samples / (m_eff > 1e-9 ? m_eff : 1e-9));

            /* ---- occlusion mask (once per key) ---- */
            uint8_t *occ_mask = NULL;
            int32_t *tile_of_point = NULL;
            int64_t n_tiles = 0;
            double occ_frac = 0.0;
            if (s->occlusion) {
                tile_of_point = (int32_t *)malloc((size_t)Q
                                                  * sizeof(int32_t));
                if (!tile_of_point)
                    die(EXIT_PHYSICS, "gather: tile map allocation "
                        "failed");
                occ_mask = build_occlusion(s, d, g, grid_pts, s->occ_tile,
                                           tile_of_point, &n_tiles,
                                           &occ_frac);
            }

            /* ---- populations: smooth (unbiased) + speckle ---- */
            double *inten = (double *)calloc((size_t)Q, sizeof(double));
            double *pop_inten = (double *)malloc((size_t)Q
                                                 * sizeof(double));
            float *field_ex = NULL, *field_ey = NULL;
            if (s->save_fields) {
                field_ex = (float *)calloc((size_t)Q * 2, sizeof(float));
                field_ey = (float *)calloc((size_t)Q * 2, sizeof(float));
            }
            if (!inten || !pop_inten)
                die(EXIT_PHYSICS, "gather: intensity allocation failed");
            char popdiag[1024] = "";
            size_t pd_at = 0;

            for (int pop = 0; pop < 2; pop++) {
                int want_scattered = (pop == 1);
                int unbiased = !want_scattered;
                /* build the population's sample subset */
                int64_t n_sel = 0;
                for (int64_t i = 0; i < M; i++)
                    if ((g->scattered[i] != 0) == want_scattered) n_sel++;
                if (n_sel == 0) continue;
                double *ppos = (double *)malloc((size_t)n_sel * 3
                                                * sizeof(double));
                double *pdir = (double *)malloc((size_t)n_sel * 3
                                                * sizeof(double));
                double *popl = (double *)malloc((size_t)n_sel
                                                * sizeof(double));
                float *pExs = (float *)malloc((size_t)n_sel * 2
                                              * sizeof(float));
                float *pEys = (float *)malloc((size_t)n_sel * 2
                                              * sizeof(float));
                uint8_t *pgroup = (uint8_t *)malloc((size_t)n_sel);
                uint8_t *pocc = NULL;
                int64_t *sel_idx = (int64_t *)malloc(
                    (size_t)n_sel * sizeof(int64_t));
                if (!ppos || !pdir || !popl || !pExs || !pEys || !pgroup
                        || !sel_idx)
                    die(EXIT_PHYSICS, "gather: population allocation "
                        "failed");
                if (occ_mask) {
                    pocc = (uint8_t *)malloc((size_t)n_tiles * n_sel);
                    if (!pocc)
                        die(EXIT_PHYSICS, "gather: population occlusion "
                            "allocation failed");
                }
                double p_pop = 0.0;
                int64_t at = 0;
                for (int64_t i = 0; i < M; i++) {
                    if ((g->scattered[i] != 0) != want_scattered)
                        continue;
                    memcpy(ppos + at * 3, g->pos + i * 3,
                           3 * sizeof(double));
                    memcpy(pdir + at * 3, g->dir + i * 3,
                           3 * sizeof(double));
                    popl[at] = g->opl[i];
                    pExs[at * 2] = Exs[i * 2];
                    pExs[at * 2 + 1] = Exs[i * 2 + 1];
                    pEys[at * 2] = Eys[i * 2];
                    pEys[at * 2 + 1] = Eys[i * 2 + 1];
                    pgroup[at] = group[i];
                    sel_idx[at] = i;
                    p_pop += g->power[i];
                    if (pocc)
                        for (int64_t t = 0; t < n_tiles; t++)
                            pocc[(size_t)t * n_sel + at] =
                                occ_mask[(size_t)t * M + i];
                    at++;
                }

                GatherJob job;
                memset(&job, 0, sizeof job);
                job.M = n_sel;
                job.Q = Q;
                job.pos = ppos;
                job.dir = pdir;
                job.opl = popl;
                job.Exs = pExs;
                job.Eys = pEys;
                job.group = pgroup;
                job.occ_mask = pocc;
                job.tile_of_point = tile_of_point;
                job.points = grid_pts;
                job.nrm = d->normal;
                job.k = K_TWO_PI / g->lam[0];
                if (tiled) {
                    job.use_tiled = 1;
                    job.n_ptiles = gridtiles.n_ptiles;
                    job.tile_centers = gridtiles.centers;
                    job.tile_start = gridtiles.start;
                    job.point_order = gridtiles.order;
                    job.tile_dpmax = gridtiles.dpmax;
                }
                job.Ex = (float *)malloc((size_t)N_CROSS_GROUPS * Q * 2
                                         * sizeof(float));
                job.Ey = (float *)malloc((size_t)N_CROSS_GROUPS * Q * 2
                                         * sizeof(float));
                if (!job.Ex || !job.Ey)
                    die(EXIT_PHYSICS, "gather: accumulator allocation "
                        "failed (%lld points)", (long long)Q);
                cross_intensity(s, &job, unbiased, g->lam[0], pop_inten,
                                field_ex, field_ey);
                total_pairs += n_sel * Q;
                near_pairs_key += job.near_exact_pairs;

                /* ---- hot-pixel sub-grid refinement ---- */
                int32_t hy[HOT_CAP], hx[HOT_CAP];
                int64_t n_hot = hot_pixels(d, grid_pts, g, sel_idx, n_sel,
                                           hy, hx);
                int s_sub = n_hot
                    ? subgrid_factor(d, g, sel_idx, n_sel, g->lam[0]) : 1;
                if (n_hot > 0 && s_sub > 1) {
                    int64_t n_pts = n_hot * s_sub * s_sub;
                    double *spts = (double *)malloc((size_t)n_pts * 3
                                                    * sizeof(double));
                    int32_t *stile = NULL;
                    double *sinten = (double *)malloc(
                        (size_t)n_pts * sizeof(double));
                    if (!spts || !sinten)
                        die(EXIT_PHYSICS, "gather: subgrid allocation "
                            "failed");
                    if (occ_mask) {
                        stile = (int32_t *)malloc((size_t)n_pts
                                                  * sizeof(int32_t));
                        if (!stile)
                            die(EXIT_PHYSICS, "gather: subgrid tile "
                                "allocation failed");
                    }
                    /* _subpixel_points (gather.py:751-764) */
                    for (int64_t h = 0; h < n_hot; h++) {
                        size_t base_at = ((size_t)hy[h] * d->W + hx[h])
                                         * 3;
                        kvec3 base = v3(grid_pts[base_at],
                                        grid_pts[base_at + 1],
                                        grid_pts[base_at + 2]);
                        kvec3 corner = v3_sub(
                            base, v3_add(
                                v3_scale(d->xhat, 0.5 * d->pixel_m),
                                v3_scale(d->yhat, 0.5 * d->pixel_m)));
                        int32_t tid = tile_of_point
                            ? tile_of_point[(size_t)hy[h] * d->W + hx[h]]
                            : 0;
                        for (int sy = 0; sy < s_sub; sy++)
                            for (int sx = 0; sx < s_sub; sx++) {
                                double oy = (sy + 0.5) / s_sub
                                            * d->pixel_m;
                                double ox = (sx + 0.5) / s_sub
                                            * d->pixel_m;
                                kvec3 p = corner;
                                p = v3_fma(p, ox, d->xhat);
                                p = v3_fma(p, oy, d->yhat);
                                int64_t at2 = (h * s_sub + sy) * s_sub
                                              + sx;
                                spts[at2 * 3] = p.x;
                                spts[at2 * 3 + 1] = p.y;
                                spts[at2 * 3 + 2] = p.z;
                                if (stile) stile[at2] = tid;
                            }
                    }
                    GatherJob sj = job;
                    sj.Q = n_pts;
                    sj.points = spts;
                    sj.tile_of_point = stile;
                    PTiles subtiles;
                    memset(&subtiles, 0, sizeof subtiles);
                    if (tiled) {
                        ptiles_build_chunks(n_pts, spts, &subtiles);
                        sj.n_ptiles = subtiles.n_ptiles;
                        sj.tile_centers = subtiles.centers;
                        sj.tile_start = subtiles.start;
                        sj.point_order = subtiles.order;
                        sj.tile_dpmax = subtiles.dpmax;
                    }
                    sj.Ex = (float *)malloc((size_t)N_CROSS_GROUPS
                                            * n_pts * 2 * sizeof(float));
                    sj.Ey = (float *)malloc((size_t)N_CROSS_GROUPS
                                            * n_pts * 2 * sizeof(float));
                    if (!sj.Ex || !sj.Ey)
                        die(EXIT_PHYSICS, "gather: subgrid accumulator "
                            "allocation failed");
                    cross_intensity(s, &sj, unbiased, g->lam[0], sinten,
                                    NULL, NULL);
                    total_pairs += n_sel * n_pts;
                    near_pairs_key += sj.near_exact_pairs;
                    if (tiled && subtiles.dpmax_max > dpmax_key)
                        dpmax_key = subtiles.dpmax_max;
                    if (tiled) ptiles_free(&subtiles);
                    for (int64_t h = 0; h < n_hot; h++) {
                        double mean = 0.0;
                        for (int c = 0; c < s_sub * s_sub; c++)
                            mean += sinten[h * s_sub * s_sub + c];
                        mean /= (double)(s_sub * s_sub);
                        pop_inten[(size_t)hy[h] * d->W + hx[h]] = mean;
                    }
                    free(sj.Ex);
                    free(sj.Ey);
                    free(spts);
                    free(sinten);
                    free(stile);
                }

                /* ---- normalize to the population's geometric power
                 * (gather.py:553-560) ---- */
                double raw = 0.0;
                for (int64_t q = 0; q < Q; q++) raw += pop_inten[q];
                double factor = raw > 0.0 ? p_pop / raw : 0.0;
                for (int64_t q = 0; q < Q; q++)
                    inten[q] += pop_inten[q] * factor;
                if (field_ex) {
                    /* fields scale by sqrt(factor) (gather.py:561-565);
                     * per-population scaling applied to THIS pop's
                     * contribution — approximated by scaling the running
                     * sum only when a single population exists (the
                     * usual case); mixed-population field maps carry the
                     * smooth pop's factor (documented) */
                    double sf = sqrt(factor > 0.0 ? factor : 0.0);
                    for (int64_t q = 0; q < Q * 2; q++) {
                        field_ex[q] = (float)(field_ex[q] * sf);
                        field_ey[q] = (float)(field_ey[q] * sf);
                    }
                }
                pd_at += (size_t)snprintf(
                    popdiag + pd_at, sizeof popdiag - pd_at,
                    "%s\"%s\": {\"power_W\": %.17g, \"raw_integral\": "
                    "%.17g, \"norm_factor_applied\": %.17g, "
                    "\"norm_factor_dimensionless\": %.17g, "
                    "\"n_samples\": %lld, \"refined_pixels\": %lld, "
                    "\"subgrid\": %d}",
                    pd_at ? ", " : "",
                    want_scattered ? "speckle" : "smooth", p_pop, raw,
                    factor, factor / (d->pixel_m * d->pixel_m),
                    (long long)n_sel, (long long)n_hot, s_sub);

                free(job.Ex);
                free(job.Ey);
                free(ppos);
                free(pdir);
                free(popl);
                free(pExs);
                free(pEys);
                free(pgroup);
                free(pocc);
                free(sel_idx);
            }

            /* mask + noise floor + cube accumulation
             * (gather.py:585-591) */
            for (int64_t q = 0; q < Q; q++)
                if (!d->mask[q]) inten[q] = 0.0;
            double neg_sq = 0.0;
            int64_t n_neg = 0;
            for (int64_t q = 0; q < Q; q++)
                if (inten[q] < 0.0) {
                    neg_sq += inten[q] * inten[q];
                    n_neg++;
                }
            double noise_floor = n_neg
                ? sqrt(neg_sq / (double)n_neg) : 0.0;
            int b = det_lam_bin(d, g->lam[0]);
            double *plane = d->inc + (size_t)b * Q;
            for (int64_t q = 0; q < Q; q++) plane[q] += inten[q];

            if (s->save_fields) {
                /* complex64 -> complex128 maps in the .h5 layout the
                 * Python glue packs (fields/<s>_<l>_<p>/{Ex,Ey}) */
                double *cx = (double *)malloc((size_t)Q * 2
                                              * sizeof(double));
                for (int64_t q = 0; q < Q * 2; q++)
                    cx[q] = (double)field_ex[q];
                snprintf(path, sizeof path,
                         "%s/det_%d_field_%d_%d_%d_Ex.npy", s->out_dir,
                         di, g->source_id, g->lam_stratum,
                         g->pol_stratum);
                npy_write(path, cx, "<c16", 2,
                          (size_t[]){(size_t)d->H, (size_t)d->W});
                for (int64_t q = 0; q < Q * 2; q++)
                    cx[q] = (double)field_ey[q];
                snprintf(path, sizeof path,
                         "%s/det_%d_field_%d_%d_%d_Ey.npy", s->out_dir,
                         di, g->source_id, g->lam_stratum,
                         g->pol_stratum);
                npy_write(path, cx, "<c16", 2,
                          (size_t[]){(size_t)d->H, (size_t)d->W});
                free(cx);
            }

            size_t key = ((size_t)g->source_id * s->max_strata
                          + g->lam_stratum) * s->max_pol
                         + g->pol_stratum;
            /* fp32 roundoff bound for the tile-factorized path (see
             * gatherk.h error budget): ~5e-7*k*dpmax + phase0-cast floor */
            double kk = K_TWO_PI / g->lam[0];
            double phase_bound = tiled
                ? 5e-7 * kk * (double)dpmax_key + 4e-7 : 0.0;
            fprintf(jf,
                    "%s\n    \"%d/%d/%d\": {\"n_samples\": %lld, "
                    "\"effective_samples\": %.17g, \"lambda_nm\": %.17g, "
                    "\"phase_step_rad\": %.17g, "
                    "\"detected_geometric_W\": %.17g, "
                    "\"noise_floor_W_per_px\": %.17g, "
                    "\"n_differential_dA\": 0, "
                    "\"backend\": \"%s\", "
                    "\"gather_mode\": \"%s\", "
                    "\"phase_err_bound_rad\": %.6g, "
                    "\"near_exact_pairs\": %lld, "
                    "\"occlusion_frac_blocked\": %.6g, "
                    "\"populations\": {%s}}",
                    first_key ? "" : ",", g->source_id, g->lam_stratum,
                    g->pol_stratum, (long long)M, m_eff,
                    g->lam[0] / 1e-9, step, d->det_geom_W[key],
                    noise_floor, backend_name,
                    tiled ? "tiled" : "exact", phase_bound,
                    (long long)near_pairs_key, occ_frac, popdiag);
            first_key = 0;
            fflush(jf);     /* per-key flush: partial gather.json stays
                             * readable for progress introspection */

            free(inten);
            free(pop_inten);
            free(field_ex);
            free(field_ey);
            free(Exs);
            free(Eys);
            free(amp);
            free(group);
            free(occ_mask);
            free(tile_of_point);
        }
        fprintf(jf, "\n  }");
        if (tiled) ptiles_free(&gridtiles);
        free(grid_pts);
    }
    fprintf(jf, "%s}\n", first_det ? "" : "\n");
    if (fclose(jf) != 0)
        die(EXIT_PHYSICS, "gather: short write to gather.json");
    return total_pairs;
}
