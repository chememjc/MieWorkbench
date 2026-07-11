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

extern "C" int gather_cuda_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaSuccess && n > 0;
}

extern "C" int gather_points_cuda(const GatherJob *j) {
    if (!gather_cuda_available()) return 1;

    double *d_pos = NULL, *d_dir = NULL, *d_opl = NULL, *d_pts = NULL;
    float *d_Exs = NULL, *d_Eys = NULL, *d_Ex = NULL, *d_Ey = NULL;
    unsigned char *d_group = NULL, *d_occ = NULL;
    int *d_tile = NULL;
    size_t M = (size_t)j->M, Q = (size_t)j->Q;

    CUDA_CHECK(cudaMalloc(&d_pos, M * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dir, M * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_opl, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Exs, M * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Eys, M * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pts, Q * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ex, (size_t)j->G * Q * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ey, (size_t)j->G * Q * 2 * sizeof(float)));
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
        CUDA_CHECK(cudaMalloc(&d_group, M));
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
        CUDA_CHECK(cudaMalloc(&d_occ, mask_bytes));
        CUDA_CHECK(cudaMemcpy(d_occ, j->occ_mask, mask_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_tile, Q * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_tile, j->tile_of_point, Q * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    int64_t blocks = ((int64_t)Q + BLOCK - 1) / BLOCK;
    gather_kernel<<<(unsigned)blocks, BLOCK>>>(
        j->M, j->Q, j->G, d_pos, d_dir, d_opl, d_Exs, d_Eys, d_group,
        d_occ, d_tile, d_pts, j->nrm.x, j->nrm.y, j->nrm.z, j->k,
        d_Ex, d_Ey);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(j->Ex, d_Ex,
                          (size_t)j->G * Q * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(j->Ey, d_Ey,
                          (size_t)j->G * Q * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_pos);
    cudaFree(d_dir);
    cudaFree(d_opl);
    cudaFree(d_Exs);
    cudaFree(d_Eys);
    cudaFree(d_pts);
    cudaFree(d_Ex);
    cudaFree(d_Ey);
    cudaFree(d_group);
    cudaFree(d_occ);
    cudaFree(d_tile);
    return 0;
}
