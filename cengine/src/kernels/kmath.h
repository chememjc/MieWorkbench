/* ===========================================================================
 * kmath.h — shared CPU/GPU math primitives for the MieWorkbench C engine.
 *
 * This header is included by BOTH the C11 host code (gcc) and CUDA device
 * code (nvcc, C++). It is therefore restricted to the common subset:
 *   - no C99 <complex.h> (not CUDA-compatible): complex numbers are the
 *     explicit `kcplx` struct with free functions, mirroring numpy
 *     complex128 semantics where it matters (sqrt branch, angle).
 *   - no designated initializers in expressions, no VLAs, no C++ features.
 *
 * KFN expands to `__host__ __device__ static inline` under nvcc so the same
 * function bodies compile for CPU and GPU — one physics source, no drift.
 *
 * All geometry is SI metres, float64 (double), matching the Python engine
 * (scripts/raytracer/rays.py header: phase needs float64 — paths are
 * 1e5-1e6 waves).
 * =========================================================================== */
#ifndef MIEWB_KMATH_H
#define MIEWB_KMATH_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define KFN __host__ __device__ static inline
#else
#define KFN static inline
#endif

#define K_PI 3.14159265358979323846
#define K_TWO_PI 6.28318530717958647692
#define K_INV_TWO_PI 0.15915494309189533577

/* ------------------------------------------------------------------ vec3 */
typedef struct { double x, y, z; } kvec3;

KFN kvec3 v3(double x, double y, double z) {
    kvec3 r; r.x = x; r.y = y; r.z = z; return r;
}
KFN kvec3 v3_add(kvec3 a, kvec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
KFN kvec3 v3_sub(kvec3 a, kvec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
KFN kvec3 v3_scale(kvec3 a, double s) { return v3(a.x*s, a.y*s, a.z*s); }
KFN double v3_dot(kvec3 a, kvec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
KFN kvec3 v3_cross(kvec3 a, kvec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
KFN double v3_norm(kvec3 a) { return sqrt(v3_dot(a, a)); }
KFN kvec3 v3_unit(kvec3 a) {
    double n = v3_norm(a);
    return v3_scale(a, 1.0 / n);
}
/* a + t*b — the ubiquitous ray-advance operation */
KFN kvec3 v3_fma(kvec3 a, double t, kvec3 b) {
    return v3(a.x + t*b.x, a.y + t*b.y, a.z + t*b.z);
}

/* 1/sqrt(x). On the device this is a single MUFU-seeded Newton chain
 * (<=2 ulp) instead of the sqrt + divide pair (two chains); the host form
 * is the exact pair. Callers that derive both r and 1/r from one k_rsqrt
 * accept an ulp-level difference in r (see gatherk.h phase note). */
KFN double k_rsqrt(double x) {
#ifdef __CUDA_ARCH__
    return rsqrt(x);
#else
    return 1.0 / sqrt(x);
#endif
}

/* Deterministic orthonormal in-plane frame for a unit normal — EXACT port
 * of surfaces._plane_frame (surfaces.py:36-44): pick the global axis with
 * the smallest |component| of n, t1 = unit(cross(axis, n)), t2 = n x t1.
 * The tie-breaking (argmin picks the FIRST minimum, numpy convention)
 * must match or every plane's UV frame — and thus every trim polygon —
 * silently rotates. */
KFN void k_plane_frame(kvec3 n, kvec3 *t1, kvec3 *t2) {
    double ax = fabs(n.x), ay = fabs(n.y), az = fabs(n.z);
    kvec3 a = v3(0.0, 0.0, 0.0);
    /* numpy argmin: first index of the minimum value */
    if (ax <= ay && ax <= az)      a.x = 1.0;
    else if (ay <= az)             a.y = 1.0;
    else                           a.z = 1.0;
    kvec3 c = v3_cross(a, n);
    *t1 = v3_unit(c);
    *t2 = v3_cross(n, *t1);
}

/* --------------------------------------------------------------- complex */
/* Explicit complex double, numpy-complex128-compatible semantics. */
typedef struct { double re, im; } kcplx;

KFN kcplx kc(double re, double im) { kcplx r; r.re = re; r.im = im; return r; }
KFN kcplx kc_add(kcplx a, kcplx b) { return kc(a.re+b.re, a.im+b.im); }
KFN kcplx kc_sub(kcplx a, kcplx b) { return kc(a.re-b.re, a.im-b.im); }
KFN kcplx kc_mul(kcplx a, kcplx b) {
    return kc(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
}
KFN kcplx kc_scale(kcplx a, double s) { return kc(a.re*s, a.im*s); }
KFN kcplx kc_div(kcplx a, kcplx b) {
    /* Smith's algorithm not needed at optical magnitudes; plain form
     * matches numpy for the well-conditioned values seen here. */
    double d = b.re*b.re + b.im*b.im;
    return kc((a.re*b.re + a.im*b.im) / d, (a.im*b.re - a.re*b.im) / d);
}
KFN double kc_abs2(kcplx a) { return a.re*a.re + a.im*a.im; }
KFN double kc_abs(kcplx a) { return hypot(a.re, a.im); }
KFN double kc_arg(kcplx a) { return atan2(a.im, a.re); }
KFN kcplx kc_conj(kcplx a) { return kc(a.re, -a.im); }

/* Principal square root, numpy branch: result has Re >= 0; for negative
 * real inputs the result is +i*sqrt(|x|) (numpy: sqrt(-1+0j) = +1j). */
KFN kcplx kc_sqrt(kcplx a) {
    double m = kc_abs(a);
    double re = sqrt(0.5 * (m + a.re));
    double im = sqrt(0.5 * (m - a.re));
    if (a.im < 0.0) im = -im;
    /* a.im == +0.0 with a.re < 0 gives +i*sqrt (numpy convention) */
    return kc(re, im);
}

/* exp(i*phi) */
KFN kcplx kc_cis(double phi) { return kc(cos(phi), sin(phi)); }

#endif /* MIEWB_KMATH_H */
