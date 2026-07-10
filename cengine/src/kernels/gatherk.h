/* ===========================================================================
 * gatherk.h — the per-(sample, point) math of the coherent Huygens gather,
 * shared verbatim by the CPU (OpenMP) and CUDA kernels so the two backends
 * cannot drift.
 *
 * EXACT port of the inner loop of gather.points_numpy/points_torch
 * (gather.py:111-273):
 *
 *   E(p) = sum_i E_i sqrt(dA_i) K_i exp(i k (opl_i + n_amb r_ip)) / r_ip
 *          * (1 / (i lambda))
 *   K = clamp(0.5 (cos theta_prop + cos theta_det), 0, 1), zero for
 *       back-radiation (cos theta_prop <= 0)
 *
 * PRECISION CONTRACT (gather.py:27-30): r and the total phase in float64,
 * reduced mod 2pi BEFORE float32 trig; accumulation in complex64. Path
 * lengths are 1e5-1e6 waves — float32 phase would destroy fringes.
 * =========================================================================== */
#ifndef MIEWB_GATHERK_H
#define MIEWB_GATHERK_H

#include "kmath.h"

/* default ambient index for the free flight (gather.py:45) */
#define GATHER_C_AMBIENT_N 1.000272

/* float32 complex accumulator (matches numpy/torch complex64) */
typedef struct { float re, im; } kcplx32;

/* One (sample, point) contribution. Inputs:
 *   P        : evaluation point
 *   pos, dir : sample origin + geometric ray direction
 *   opl      : accumulated optical path at the sample [m]
 *   Exs, Eys : sample Jones amplitudes projected on the detector frame,
 *              sqrt(dA) folded in (complex64, matching gather.py:130-131)
 *   k        : 2 pi / lambda;  nrm : detector normal
 *   occ_vis  : 0 blocks the pair (gather occlusion), 1 visible
 * Accumulates into (Ex, Ey). The final (1 / i lambda) factor is applied
 * once per point by the caller (gather.py:134,164-165). */
KFN void gather_pair(kvec3 P, kvec3 pos, kvec3 dir, double opl,
                     kcplx32 Exs, kcplx32 Eys, double k, kvec3 nrm,
                     float occ_vis, kcplx32 *Ex, kcplx32 *Ey) {
    /* r in f64 (the phase depends on it at the 1e-9-relative level) */
    double dx = P.x - pos.x, dy = P.y - pos.y, dz = P.z - pos.z;
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < 1e-18) r2 = 1e-18;                 /* gather.py:147 clamp */
    double r = sqrt(r2);
    double inv_r = 1.0 / r;
    double rhat_dot_dir = (dx * dir.x + dy * dir.y + dz * dir.z) * inv_r;
    double cos_det = fabs(dx * nrm.x + dy * nrm.y + dz * nrm.z) * inv_r;
    double K = 0.5 * (rhat_dot_dir + cos_det);
    if (K < 0.0) K = 0.0;
    if (K > 1.0) K = 1.0;
    if (rhat_dot_dir <= 0.0) K = 0.0;           /* no back-radiation */
    K *= occ_vis;
    if (K == 0.0) return;
    /* phase in f64, reduced mod 2pi BEFORE f32 trig (gather.py:156-159) */
    double phase = fmod(k * (opl + GATHER_C_AMBIENT_N * r), K_TWO_PI);
    float w = (float)(K * inv_r);
#ifdef __CUDA_ARCH__
    float sp, cp;
    __sincosf((float)phase, &sp, &cp);
#else
    float sp = (float)sin(phase);
    float cp = (float)cos(phase);
#endif
    /* (cp + i sp) * w * (Exs, Eys), complex64 accumulation */
    float wr = w * cp, wi = w * sp;
    Ex->re += wr * Exs.re - wi * Exs.im;
    Ex->im += wr * Exs.im + wi * Exs.re;
    Ey->re += wr * Eys.re - wi * Eys.im;
    Ey->im += wr * Eys.im + wi * Eys.re;
}

#endif /* MIEWB_GATHERK_H */
