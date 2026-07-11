/* ===========================================================================
 * rng.h — counter-based Philox4x32-10 RNG with ray-lineage keying.
 *
 * DESIGN CONTRACT (plan decision D2 — expensive to reverse):
 *   Every random draw in the engine is a PURE FUNCTION of
 *       (global_seed, ray_key, event_ctr, draw index)
 *   so results are bit-identical regardless of thread count, scheduling,
 *   or batch splitting. This is what makes the OpenMP trace reproducible
 *   and single-ray debugging (--trace-ray) possible.
 *
 *   - Primaries get  ray_key = mix(seed, source_id, primary_index).
 *   - Children get   ray_key = mix(parent_key, parent_event_ctr, child_slot)
 *     where child_slot enumerates the split branch (see CHILD_SLOT_*).
 *   - event_ctr increments once per surface interaction; draws within one
 *     interaction use consecutive `draw` indices.
 *
 *   It intentionally does NOT reproduce numpy's PCG64 streams — the agreed
 *   parity bar is deterministic (1e-12) for non-random physics and
 *   statistical (3-seed +-max(3sigma,1%)) for Monte-Carlo aggregates.
 *
 * Philox4x32-10 reference: Salmon et al., "Parallel random numbers: as easy
 * as 1, 2, 3" (SC'11). Constants below are the canonical ones (identical to
 * numpy.random.Philox / CUDA cuRAND Philox4_32_10).
 *
 * Header-only, C11 and CUDA compatible (KFN from kmath.h).
 * =========================================================================== */
#ifndef MIEWB_RNG_H
#define MIEWB_RNG_H

#include <stdint.h>
#include "kernels/kmath.h"

/* Child-slot enumeration: stable IDs for every way an interaction can spawn
 * a child ray. Used to derive the child's ray_key; adding new physics means
 * appending new slots (NEVER renumbering — that changes every stream). */
enum {
    CHILD_SLOT_TRANSMIT   = 0,
    CHILD_SLOT_REFLECT    = 1,
    CHILD_SLOT_ROUGH_R0   = 2,   /* roughness scattered reflection lobe j */
    CHILD_SLOT_ROUGH_R1   = 3,
    CHILD_SLOT_ROUGH_T0   = 4,   /* roughness scattered transmission lobe j */
    CHILD_SLOT_ROUGH_T1   = 5,
    CHILD_SLOT_ABG_0      = 6,   /* ABg scatter lobes */
    CHILD_SLOT_ABG_1      = 7,
    CHILD_SLOT_ORDINARY   = 8,   /* birefringent o-ray */
    CHILD_SLOT_EXTRAORD   = 9,   /* birefringent e-ray */
    CHILD_SLOT_GRATING0   = 16,  /* grating order m -> slot GRATING0 + (m+8) */
    CHILD_SLOT_PARTICLE   = 40,  /* particle-scatter continuation */
};

/* ------------------------------------------------------ Philox4x32-10 core */
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u   /* golden-ratio Weyl increments */
#define PHILOX_W1 0xBB67AE85u

typedef struct { uint32_t v[4]; } philox4x32_ctr;
typedef struct { uint32_t v[2]; } philox4x32_key;

KFN void philox_round(philox4x32_ctr *c, const philox4x32_key *k) {
    uint64_t p0 = (uint64_t)PHILOX_M0 * c->v[0];
    uint64_t p1 = (uint64_t)PHILOX_M1 * c->v[2];
    uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    philox4x32_ctr out;
    out.v[0] = hi1 ^ c->v[1] ^ k->v[0];
    out.v[1] = lo1;
    out.v[2] = hi0 ^ c->v[3] ^ k->v[1];
    out.v[3] = lo0;
    *c = out;
}

KFN philox4x32_ctr philox4x32_10(philox4x32_ctr c, philox4x32_key k) {
    /* 10 rounds, key bumped by the Weyl constants between rounds */
    for (int i = 0; i < 10; i++) {
        philox_round(&c, &k);
        k.v[0] += PHILOX_W0;
        k.v[1] += PHILOX_W1;
    }
    return c;
}

/* ----------------------------------------------------------- lineage keys */
/* splitmix64 finalizer — a well-tested 64-bit mixer (Vigna). Used to derive
 * ray keys; Philox itself provides the stream quality, the mixer only needs
 * to decorrelate the key space. */
KFN uint64_t k_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Primary ray key: pure function of (seed, source, primary sample index). */
KFN uint64_t rng_primary_key(uint64_t seed, uint32_t source_id,
                             uint64_t primary_index) {
    return k_mix64(seed ^ k_mix64(((uint64_t)source_id << 40)
                                  ^ primary_index));
}

/* Child ray key: pure function of the parent's key, the interaction counter
 * at the split, and the branch slot. */
KFN uint64_t rng_child_key(uint64_t parent_key, uint32_t event_ctr,
                           uint32_t child_slot) {
    return k_mix64(parent_key ^ (((uint64_t)event_ctr << 32)
                                 | (uint64_t)child_slot)
                   ^ 0xA5A5A5A5DEADBEEFull);
}

/* ------------------------------------------------------------ draw helpers */
/* One Philox block keyed by (ray_key) with counter (event, draw_block, 0, 0).
 * Each block yields 4x uint32; helpers below carve doubles out of it. */
KFN philox4x32_ctr rng_block(uint64_t ray_key, uint32_t event_ctr,
                             uint32_t draw_block) {
    philox4x32_key k;
    k.v[0] = (uint32_t)(ray_key & 0xFFFFFFFFu);
    k.v[1] = (uint32_t)(ray_key >> 32);
    philox4x32_ctr c;
    c.v[0] = event_ctr;
    c.v[1] = draw_block;
    c.v[2] = 0x4D494557u;   /* "MIEW" domain tag */
    c.v[3] = 0x42454E47u;   /* "BENG" */
    return philox4x32_10(c, k);
}

/* uint64 from two lanes */
KFN uint64_t rng_u64_from(philox4x32_ctr b, int lane01) {
    return ((uint64_t)b.v[lane01 * 2] << 32) | b.v[lane01 * 2 + 1];
}

/* Uniform double in [0, 1) with 53-bit resolution (numpy-style: take the
 * top 53 bits of a uint64). */
KFN double rng_u01_from(philox4x32_ctr b, int lane01) {
    return (double)(rng_u64_from(b, lane01) >> 11) * (1.0 / 9007199254740992.0);
}

/* Convenience: the n-th uniform double for (ray_key, event). Draw indices
 * n = 0, 1, 2, ... map to (block = n/2, lane = n%2) — cheap and stateless. */
KFN double rng_uniform(uint64_t ray_key, uint32_t event_ctr, uint32_t n) {
    philox4x32_ctr b = rng_block(ray_key, event_ctr, n >> 1);
    return rng_u01_from(b, (int)(n & 1));
}

/* Standard normal via Box-Muller (draws 2 uniforms = one block). `n` selects
 * an independent pair index. Marsaglia-polar would need rejection (stateful);
 * Box-Muller keeps the draw count fixed per index, which the counter-based
 * design requires. */
KFN void rng_normal2(uint64_t ray_key, uint32_t event_ctr, uint32_t n,
                     double *z0, double *z1) {
    philox4x32_ctr b = rng_block(ray_key, event_ctr, 0x8000000u + n);
    double u1 = rng_u01_from(b, 0);
    double u2 = rng_u01_from(b, 1);
    /* u1 = 0 would give log(0); the 53-bit grid's smallest nonzero step */
    if (u1 < 1.11022302462515654e-16) u1 = 1.11022302462515654e-16;
    double r = sqrt(-2.0 * log(u1));
    *z0 = r * cos(K_TWO_PI * u2);
    *z1 = r * sin(K_TWO_PI * u2);
}

#endif /* MIEWB_RNG_H */
