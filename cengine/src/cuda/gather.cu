/* ===========================================================================
 * gather.cu — CUDA backend for the coherent Huygens gather.
 *
 * Same per-(sample, point) math as the CPU kernel — both include
 * kernels/gatherk.h, so the two backends cannot drift (the KFN macro
 * compiles gather_pair as __host__ __device__ here).
 *
 * Kernel shape: one thread per evaluation point, samples streamed through
 * shared memory in chunks; the G<=4 cross-estimator groups accumulate in
 * registers. Unlike the torch backend (which materializes ~6 fp64
 * (pixel_chunk x sample_chunk) tensors in GLOBAL memory per tile,
 * gather.py:220-266), everything here lives in registers/smem — the
 * kernel is fp64-ALU-bound, not bandwidth-bound.
 *
 * Precision: identical contract to gather.py:27-30 — r and phase in
 * float64, reduced mod 2pi before float32 trig (__sincosf), complex64
 * accumulation. Matches the torch backend's precision class (the
 * torch-vs-numpy 5e-3 gate precedent).
 *
 * Error discipline: every CUDA call goes through CUDA_CHECK -> die(4)
 * with file:line and the failing call; gather_points_cuda returns nonzero
 * (instead of dying) only for "no device", so the host can fall back to
 * the CPU kernel under backend=auto.
 * =========================================================================== */
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
#include "../log.h"
}
#include "../kernels/gatherk.h"

extern "C" {
#include "../gather.h"
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess)                                             \
            die(EXIT_CUDA, "CUDA error at %s:%d in %s: %s", __FILE__,      \
                __LINE__, #call, cudaGetErrorString(_e));                  \
    } while (0)

#define G_MAX 4
#define SAMPLE_CHUNK 256        /* smem: 256 * 76 B ~ 19 KB */
#define BLOCK 128

/* ---------------------------------------------------------------------------
 * Persistent device-buffer pool (P3 worker mode).
 *
 * The one-shot binary cudaMalloc'd + cudaFree'd every gather buffer per call.
 * In --serve mode the process (hence the CUDA primary context) outlives many
 * gathers, so those allocations are pure per-request overhead. Each buffer
 * grows monotonically to the largest request seen and is REUSED across
 * requests; nothing is freed until gather_cuda_pool_free() at worker exit.
 * The pool is a process-global singleton — the gather host path is single-
 * threaded (no omp region around gather_points_cuda), so no locking is
 * needed. In one-shot mode the pool is still used (one grow, freed by
 * main() before exit / reclaimed by context teardown), so the code path is
 * identical and the parity tests cover both.
 * ------------------------------------------------------------------------- */
typedef struct { void *ptr; size_t cap; } DevBuf;
static DevBuf g_pos, g_dir, g_opl, g_pts, g_Exs, g_Eys, g_Ex, g_Ey,
              g_group, g_occ, g_tile, g_centers, g_tstart, g_order,
              g_dpmax, g_near;

/* Reserve >= need bytes in b, reusing the existing allocation when it is
 * already large enough. Grows by freeing + reallocating (contents are
 * overwritten by the caller's H2D copy every request, so no data is kept).
 * need==0 returns the current pointer untouched (optional-buffer case). */
static void *dev_reserve(DevBuf *b, size_t need) {
    if (need == 0) return b->ptr;
    if (b->cap < need) {
        if (b->ptr) cudaFree(b->ptr);
        b->ptr = NULL;
        b->cap = 0;
        CUDA_CHECK(cudaMalloc(&b->ptr, need));
        b->cap = need;
    }
    return b->ptr;
}

extern "C" void gather_cuda_pool_free(void) {
    DevBuf *all[] = {&g_pos, &g_dir, &g_opl, &g_pts, &g_Exs, &g_Eys, &g_Ex,
                     &g_Ey, &g_group, &g_occ, &g_tile, &g_centers, &g_tstart,
                     &g_order, &g_dpmax, &g_near};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        if (all[i]->ptr) cudaFree(all[i]->ptr);
        all[i]->ptr = NULL;
        all[i]->cap = 0;
    }
}

/* Force primary-context creation up front (serve start) so the first served
 * request does not pay it. Best-effort: a co-tenanted / absent GPU just
 * means the gather falls back to the CPU kernel at run time, so failure here
 * is ignored rather than fatal. */
extern "C" void gather_cuda_worker_init(void) {
    if (gather_cuda_available()) (void)cudaFree(0);
}

/* staged sample record in shared memory */
struct SampleS {
    double px, py, pz;
    double dx, dy, dz;
    double opl;
    float exr, exi, eyr, eyi;
    unsigned char group;
};

__global__ void gather_kernel(int64_t M, int64_t Q, int G,
                              const double *__restrict__ pos,
                              const double *__restrict__ dir,
                              const double *__restrict__ opl,
                              const float *__restrict__ Exs,
                              const float *__restrict__ Eys,
                              const unsigned char *__restrict__ group,
                              const unsigned char *__restrict__ occ_mask,
                              const int *__restrict__ tile_of_point,
                              const double *__restrict__ points,
                              double nx, double ny, double nz, double k,
                              float *__restrict__ Ex,
                              float *__restrict__ Ey) {
    __shared__ SampleS sh[SAMPLE_CHUNK];
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    /* every thread participates in the staging loads, so keep threads
     * with q >= Q alive through the loop and mask their accumulation */
    int active = q < Q;
    kvec3 P = v3(0.0, 0.0, 0.0);
    const unsigned char *occ_col = NULL;
    if (active) {
        P = v3(points[q * 3], points[q * 3 + 1], points[q * 3 + 2]);
        if (occ_mask && tile_of_point)
            occ_col = occ_mask + (size_t)tile_of_point[q] * M;
    }
    kvec3 nrm = v3(nx, ny, nz);
    kcplx32 ex[G_MAX] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    kcplx32 ey[G_MAX] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};

    for (int64_t base = 0; base < M; base += SAMPLE_CHUNK) {
        int chunk = (int)((M - base < SAMPLE_CHUNK) ? (M - base)
                                                    : SAMPLE_CHUNK);
        __syncthreads();
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            int64_t s = base + i;
            sh[i].px = pos[s * 3];
            sh[i].py = pos[s * 3 + 1];
            sh[i].pz = pos[s * 3 + 2];
            sh[i].dx = dir[s * 3];
            sh[i].dy = dir[s * 3 + 1];
            sh[i].dz = dir[s * 3 + 2];
            sh[i].opl = opl[s];
            sh[i].exr = Exs[s * 2];
            sh[i].exi = Exs[s * 2 + 1];
            sh[i].eyr = Eys[s * 2];
            sh[i].eyi = Eys[s * 2 + 1];
            sh[i].group = group ? group[s] : 0;
        }
        __syncthreads();
        if (active) {
            for (int i = 0; i < chunk; i++) {
                float vis = occ_col
                    ? (float)occ_col[base + i] : 1.0f;
                kcplx32 exs = {sh[i].exr, sh[i].exi};
                kcplx32 eys = {sh[i].eyr, sh[i].eyi};
                int g = sh[i].group;
                gather_pair(P, v3(sh[i].px, sh[i].py, sh[i].pz),
                            v3(sh[i].dx, sh[i].dy, sh[i].dz), sh[i].opl,
                            exs, eys, k, nrm, vis, &ex[g], &ey[g]);
            }
        }
    }
    if (active) {
        for (int g = 0; g < G; g++) {
            Ex[((size_t)g * Q + q) * 2] = ex[g].re;
            Ex[((size_t)g * Q + q) * 2 + 1] = ex[g].im;
            Ey[((size_t)g * Q + q) * 2] = ey[g].re;
            Ey[((size_t)g * Q + q) * 2 + 1] = ey[g].im;
        }
    }
}

/* ---- P1 tile-factorized kernel: one block per point tile ----
 * fp64 staging against the tile centre is shared by the whole block
 * (BLOCK-x amortization of the fp64 MUFU chains); the per-(point,sample)
 * inner loop is fp32 via the exact-identity residual (gatherk.h). Near
 * samples (R < GATHER_NEAR_FACTOR*dpmax) take the exact fp64 gather_pair
 * from global memory; they are counted for the diagnostics block. */
struct StagedTS {
    float ux, uy, uz, R, ph0;
    float dirx, diry, dirz;
    float exr, exi, eyr, eyi;
    unsigned char group, near;
};

__global__ void gather_kernel_tiled(
        int64_t M, int64_t Q, int G,
        const double *__restrict__ pos,
        const double *__restrict__ dir,
        const double *__restrict__ opl,
        const float *__restrict__ Exs,
        const float *__restrict__ Eys,
        const unsigned char *__restrict__ group,
        const unsigned char *__restrict__ occ_mask,
        const int *__restrict__ tile_of_point,
        const double *__restrict__ points,
        const double *__restrict__ tile_centers,
        const int64_t *__restrict__ tile_start,
        const int64_t *__restrict__ point_order,
        const float *__restrict__ tile_dpmax,
        double nx, double ny, double nz, double k,
        float *__restrict__ Ex, float *__restrict__ Ey,
        unsigned long long *__restrict__ near_count) {
    __shared__ StagedTS sh[SAMPLE_CHUNK];
    __shared__ unsigned int near_chunk;
    int64_t t = blockIdx.x;
    int64_t q0 = tile_start[t], q1 = tile_start[t + 1];
    int np = (int)(q1 - q0);
    int l = threadIdx.x;
    int active = l < np;
    kvec3 p0 = v3(tile_centers[t * 3], tile_centers[t * 3 + 1],
                  tile_centers[t * 3 + 2]);
    double near_R = GATHER_NEAR_FACTOR * (double)tile_dpmax[t];
    kvec3 P = v3(0.0, 0.0, 0.0);
    float dpx = 0.f, dpy = 0.f, dpz = 0.f, dp2 = 0.f;
    int64_t q = 0;
    const unsigned char *occ_col = NULL;
    if (active) {
        q = point_order[q0 + l];
        P = v3(points[q * 3], points[q * 3 + 1], points[q * 3 + 2]);
        dpx = (float)(P.x - p0.x);
        dpy = (float)(P.y - p0.y);
        dpz = (float)(P.z - p0.z);
        dp2 = dpx * dpx + dpy * dpy + dpz * dpz;
        if (occ_mask && tile_of_point)
            occ_col = occ_mask + (size_t)tile_of_point[q] * M;
    }
    kvec3 nrm = v3(nx, ny, nz);
    float nxf = (float)nx, nyf = (float)ny, nzf = (float)nz;
    float k_f = (float)k;
    kcplx32 ex[G_MAX] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    kcplx32 ey[G_MAX] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    unsigned long long near_local = 0;

    for (int64_t base = 0; base < M; base += SAMPLE_CHUNK) {
        int chunk = (int)((M - base < SAMPLE_CHUNK) ? (M - base)
                                                    : SAMPLE_CHUNK);
        __syncthreads();
        if (threadIdx.x == 0) near_chunk = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            int64_t si = base + i;
            kvec3 posv = v3(pos[si * 3], pos[si * 3 + 1],
                            pos[si * 3 + 2]);
            double R = gather_stage_tile(p0, posv, opl[si], k,
                                         &sh[i].ux, &sh[i].uy, &sh[i].uz,
                                         &sh[i].R, &sh[i].ph0);
            sh[i].dirx = (float)dir[si * 3];
            sh[i].diry = (float)dir[si * 3 + 1];
            sh[i].dirz = (float)dir[si * 3 + 2];
            sh[i].exr = Exs[si * 2];
            sh[i].exi = Exs[si * 2 + 1];
            sh[i].eyr = Eys[si * 2];
            sh[i].eyi = Eys[si * 2 + 1];
            sh[i].group = group ? group[si] : 0;
            unsigned char nr = R < near_R;
            sh[i].near = nr;
            if (nr) atomicAdd(&near_chunk, 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0 && near_chunk)
            near_local += (unsigned long long)near_chunk
                          * (unsigned long long)np;
        if (active) {
            for (int i = 0; i < chunk; i++) {
                float vis = occ_col ? (float)occ_col[base + i] : 1.0f;
                int g = sh[i].group;
                kcplx32 exs = {sh[i].exr, sh[i].exi};
                kcplx32 eys = {sh[i].eyr, sh[i].eyi};
                if (sh[i].near) {
                    int64_t si = base + i;
                    gather_pair(P,
                                v3(pos[si * 3], pos[si * 3 + 1],
                                   pos[si * 3 + 2]),
                                v3(dir[si * 3], dir[si * 3 + 1],
                                   dir[si * 3 + 2]),
                                opl[si], exs, eys, k, nrm, vis,
                                &ex[g], &ey[g]);
                } else {
                    gather_pair_tile(dpx, dpy, dpz, dp2,
                                     sh[i].ux, sh[i].uy, sh[i].uz,
                                     sh[i].R, sh[i].ph0,
                                     sh[i].dirx, sh[i].diry, sh[i].dirz,
                                     nxf, nyf, nzf, k_f, exs, eys, vis,
                                     &ex[g], &ey[g]);
                }
            }
        }
    }
    if (active) {
        for (int g = 0; g < G; g++) {
            Ex[((size_t)g * Q + q) * 2] = ex[g].re;
            Ex[((size_t)g * Q + q) * 2 + 1] = ex[g].im;
            Ey[((size_t)g * Q + q) * 2] = ey[g].re;
            Ey[((size_t)g * Q + q) * 2 + 1] = ey[g].im;
        }
    }
    if (threadIdx.x == 0 && near_local && near_count)
        atomicAdd(near_count, near_local);
}

extern "C" int gather_cuda_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaSuccess && n > 0;
}

extern "C" int gather_points_cuda(GatherJob *j) {
    if (!gather_cuda_available()) return 1;

    unsigned char *d_group = NULL, *d_occ = NULL;
    int *d_tile = NULL;
    size_t M = (size_t)j->M, Q = (size_t)j->Q;

    /* pooled: reused + grown across served requests (never freed here) */
    double *d_pos = (double *)dev_reserve(&g_pos, M * 3 * sizeof(double));
    double *d_dir = (double *)dev_reserve(&g_dir, M * 3 * sizeof(double));
    double *d_opl = (double *)dev_reserve(&g_opl, M * sizeof(double));
    float *d_Exs = (float *)dev_reserve(&g_Exs, M * 2 * sizeof(float));
    float *d_Eys = (float *)dev_reserve(&g_Eys, M * 2 * sizeof(float));
    double *d_pts = (double *)dev_reserve(&g_pts, Q * 3 * sizeof(double));
    float *d_Ex = (float *)dev_reserve(
        &g_Ex, (size_t)j->G * Q * 2 * sizeof(float));
    float *d_Ey = (float *)dev_reserve(
        &g_Ey, (size_t)j->G * Q * 2 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_pos, j->pos, M * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dir, j->dir, M * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opl, j->opl, M * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Exs, j->Exs, M * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Eys, j->Eys, M * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts, j->points, Q * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
    if (j->group && j->G > 1) {
        d_group = (unsigned char *)dev_reserve(&g_group, M);
        CUDA_CHECK(cudaMemcpy(d_group, j->group, M,
                              cudaMemcpyHostToDevice));
    }
    if (j->occ_mask && j->tile_of_point) {
        /* mask rows = tiles; total size inferred from the max tile id */
        int32_t max_tile = 0;
        for (int64_t q = 0; q < j->Q; q++)
            if (j->tile_of_point[q] > max_tile)
                max_tile = j->tile_of_point[q];
        size_t mask_bytes = ((size_t)max_tile + 1) * M;
        d_occ = (unsigned char *)dev_reserve(&g_occ, mask_bytes);
        CUDA_CHECK(cudaMemcpy(d_occ, j->occ_mask, mask_bytes,
                              cudaMemcpyHostToDevice));
        d_tile = (int *)dev_reserve(&g_tile, Q * sizeof(int));
        CUDA_CHECK(cudaMemcpy(d_tile, j->tile_of_point, Q * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    double *d_centers = NULL;
    int64_t *d_tstart = NULL, *d_order = NULL;
    float *d_dpmax = NULL;
    unsigned long long *d_near = NULL;
    if (j->use_tiled) {
        /* one block per point tile; GATHER_TILE_CAP must equal BLOCK */
        static_assert(GATHER_TILE_CAP == BLOCK,
                      "tile cap and CUDA block size must match");
        size_t nt = (size_t)j->n_ptiles;
        d_centers = (double *)dev_reserve(&g_centers, nt * 3 * sizeof(double));
        d_tstart = (int64_t *)dev_reserve(&g_tstart,
                                          (nt + 1) * sizeof(int64_t));
        d_order = (int64_t *)dev_reserve(&g_order, Q * sizeof(int64_t));
        d_dpmax = (float *)dev_reserve(&g_dpmax, nt * sizeof(float));
        d_near = (unsigned long long *)dev_reserve(
            &g_near, sizeof(unsigned long long));
        CUDA_CHECK(cudaMemcpy(d_centers, j->tile_centers,
                              nt * 3 * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tstart, j->tile_start,
                              (nt + 1) * sizeof(int64_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_order, j->point_order,
                              Q * sizeof(int64_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpmax, j->tile_dpmax, nt * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_near, 0, sizeof(unsigned long long)));
        gather_kernel_tiled<<<(unsigned)j->n_ptiles, BLOCK>>>(
            j->M, j->Q, j->G, d_pos, d_dir, d_opl, d_Exs, d_Eys, d_group,
            d_occ, d_tile, d_pts, d_centers, d_tstart, d_order, d_dpmax,
            j->nrm.x, j->nrm.y, j->nrm.z, j->k, d_Ex, d_Ey, d_near);
    } else {
        int64_t blocks = ((int64_t)Q + BLOCK - 1) / BLOCK;
        gather_kernel<<<(unsigned)blocks, BLOCK>>>(
            j->M, j->Q, j->G, d_pos, d_dir, d_opl, d_Exs, d_Eys, d_group,
            d_occ, d_tile, d_pts, j->nrm.x, j->nrm.y, j->nrm.z, j->k,
            d_Ex, d_Ey);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (j->use_tiled) {
        unsigned long long near_pairs = 0;
        CUDA_CHECK(cudaMemcpy(&near_pairs, d_near,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        j->near_exact_pairs = (int64_t)near_pairs;
    }

    CUDA_CHECK(cudaMemcpy(j->Ex, d_Ex,
                          (size_t)j->G * Q * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(j->Ey, d_Ey,
                          (size_t)j->G * Q * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Buffers stay in the pool (dev_reserve) for reuse by the next served
     * request; gather_cuda_pool_free() releases them at worker exit. */
    return 0;
}
