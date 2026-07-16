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

#define K_TWO_PI_F 6.28318530717958647692f
#define K_INV_TWO_PI_F 0.15915494309189533577f

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
    /* one k_rsqrt chain instead of sqrt + divide (two MUFU Newton chains on
     * the device). r = r2*inv_r differs from sqrt(r2) by <=2 ulp; the phase
     * impact k*|dr| is O(1e-9 rad), far below both the fp32 trig cast and
     * the MC speckle pedestal. */
    double inv_r = k_rsqrt(r2);
    double r = r2 * inv_r;
    double rhat_dot_dir = (dx * dir.x + dy * dir.y + dz * dir.z) * inv_r;
    double cos_det = fabs(dx * nrm.x + dy * nrm.y + dz * nrm.z) * inv_r;
    double K = 0.5 * (rhat_dot_dir + cos_det);
    if (K < 0.0) K = 0.0;
    if (K > 1.0) K = 1.0;
    if (rhat_dot_dir <= 0.0) K = 0.0;           /* no back-radiation */
    K *= occ_vis;
    if (K == 0.0) return;
    /* phase in f64, reduced mod 2pi BEFORE f32 trig (gather.py:156-159).
     * x - 2pi*trunc(x*inv2pi) replaces IEEE-exact fmod, whose RCP64H Newton
     * chains were ~27% of the kernel. x >= 0 here (opl, r > 0) so trunc is
     * floor; the reduced phase differs from fmod by O(n_waves*eps*2pi)
     * < ~5e-9 rad at 1e6 waves — below the fp32 cast resolution of the
     * trig argument. The f64-before-f32-trig contract is preserved. */
    double ph_x = k * (opl + GATHER_C_AMBIENT_N * r);
    double phase = ph_x - K_TWO_PI * trunc(ph_x * K_INV_TWO_PI);
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

/* ===========================================================================
 * Tile-factorized path (P1 gather-precision round).
 *
 * The fp64 work above is per (sample, point). Factorize per detector TILE:
 * for tile centre p0, stage per (tile, sample) ONCE in fp64
 *
 *   R      = |p0 - pos_i|
 *   u      = (p0 - pos_i)/R
 *   phase0 = (k (opl_i + n_amb R)) mod 2pi
 *
 * then every point P = p0 + dp in the tile needs only fp32:
 *
 *   r^2 - R^2 = 2 R (u . dp) + |dp|^2        -- EXACT algebraic identity
 *                                               for the stored (R, u),
 *                                               NOT a series expansion
 *   r - R     = (2 R (u . dp) + |dp|^2) / (r + R)
 *   phase     = phase0 + k (r - R)           -- fp32, re-reduced mod 2pi
 *
 * Error budget (all fp32 roundoff, no truncation term; dp_max = max |dp|
 * over the tile, and R >= GATHER_NEAR_FACTOR*dp_max enforced by the near
 * guard): u stored fp32 (~1e-7 non-unit) contributes k*1e-7*dp_max; the
 * 2R(u.dp) product with fp32 R contributes ~k*eps32*dp_max; k_f32*resid
 * and the final re-reduction each ~|k*resid|*eps32 <= k*2*dp_max*eps32.
 * Total < ~5e-7 * k * dp_max + 4e-7 rad — reported per key as
 * phase_err_bound_rad. At dp_max = 45 um, 633 nm: ~2.5e-4 rad, far below
 * both the 1e-3 budget and the MC speckle pedestal.
 *
 * Samples with R < GATHER_NEAR_FACTOR*dp_max are flagged at staging and
 * accumulated through the EXACT fp64 gather_pair above instead (counted:
 * near_exact_pairs). A tile of one point (dp = 0) never takes this path
 * at all — the dispatcher routes tile_px=1 / --gather-exact to the plain
 * kernel, which is the bit-exactness anchor.
 * =========================================================================== */

#define GATHER_NEAR_FACTOR 8.0

/* fp64 per-(tile, sample) staging. Returns R (fp64) so the caller can
 * apply the near-field guard; writes the fp32 staged fields. */
KFN double gather_stage_tile(kvec3 p0, kvec3 pos, double opl, double k,
                             float *ux, float *uy, float *uz,
                             float *R_f, float *ph0) {
    double dx = p0.x - pos.x, dy = p0.y - pos.y, dz = p0.z - pos.z;
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < 1e-18) r2 = 1e-18;
    double inv_r = k_rsqrt(r2);
    double R = r2 * inv_r;
    *ux = (float)(dx * inv_r);
    *uy = (float)(dy * inv_r);
    *uz = (float)(dz * inv_r);
    *R_f = (float)R;
    /* same fp64 trunc-reduction as gather_pair — at dp = 0 the staged
     * phase0 is bit-identical to the plain kernel's trig argument */
    double x = k * (opl + GATHER_C_AMBIENT_N * R);
    *ph0 = (float)(x - K_TWO_PI * trunc(x * K_INV_TWO_PI));
    return R;
}

/* fp32 per-(point, sample) contribution against a staged sample.
 * dp = P - p0 (fp32), dp2 = |dp|^2 precomputed per point. dirx/y/z is the
 * sample's geometric direction (fp32), n the detector normal (fp32). */
KFN void gather_pair_tile(float dpx, float dpy, float dpz, float dp2,
                          float ux, float uy, float uz, float R,
                          float ph0, float dirx, float diry, float dirz,
                          float nx, float ny, float nz, float k_f,
                          kcplx32 Exs, kcplx32 Eys, float occ_vis,
                          kcplx32 *Ex, kcplx32 *Ey) {
    /* v = P - pos = R u + dp; amplitude-grade fp32 (needs ~1e-3) */
    float vx = fmaf(R, ux, dpx);
    float vy = fmaf(R, uy, dpy);
    float vz = fmaf(R, uz, dpz);
    float r2f = vx * vx + vy * vy + vz * vz;
    if (r2f < 1e-18f) r2f = 1e-18f;
#ifdef __CUDA_ARCH__
    float inv_rf = rsqrtf(r2f);
#else
    float inv_rf = 1.0f / sqrtf(r2f);
#endif
    float rf = r2f * inv_rf;
    float rhat_dot_dir = (vx * dirx + vy * diry + vz * dirz) * inv_rf;
    float cos_det = fabsf(vx * nx + vy * ny + vz * nz) * inv_rf;
    float K = 0.5f * (rhat_dot_dir + cos_det);
    if (K < 0.0f) K = 0.0f;
    if (K > 1.0f) K = 1.0f;
    if (rhat_dot_dir <= 0.0f) K = 0.0f;
    K *= occ_vis;
    if (K == 0.0f) return;
    /* phase-grade: the exact-identity residual, never |v|^2 - R^2 (that
     * form reintroduces the R^2 cancellation the identity avoids) */
    float udp = ux * dpx + uy * dpy + uz * dpz;
    float num = fmaf(2.0f * R, udp, dp2);
    float resid = num / (rf + R);
    float phase = fmaf(k_f, resid, ph0);
    phase -= K_TWO_PI_F * truncf(phase * K_INV_TWO_PI_F);
    float sp, cp;
#ifdef __CUDA_ARCH__
    __sincosf(phase, &sp, &cp);
#else
    sp = sinf(phase);
    cp = cosf(phase);
#endif
    float w = K * inv_rf;
    float wr = w * cp, wi = w * sp;
    Ex->re += wr * Exs.re - wi * Exs.im;
    Ex->im += wr * Exs.im + wi * Exs.re;
    Ey->re += wr * Eys.re - wi * Eys.im;
    Ey->im += wr * Eys.im + wi * Eys.re;
}

#endif /* MIEWB_GATHERK_H */
